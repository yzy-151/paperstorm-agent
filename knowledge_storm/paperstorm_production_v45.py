import hashlib
import json
import sqlite3
import threading
import time
import uuid
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional


class _ClosingSQLiteConnection(sqlite3.Connection):
    """Commit or roll back like sqlite3, then always release the file handle."""

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


class ProductionControlPlaneV45:
    """SQLite WAL control plane for the local production-governance baseline."""

    def __init__(self, database_path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._schema_lock = threading.Lock()
        self._ensure_schema()

    def register_resource(
        self,
        tenant_id: str,
        resource_type: str,
        resource_id: str,
        owner_user_id: str,
        allowed_user_ids=None,
        metadata: Optional[Dict] = None,
        version: int = 1,
    ):
        payload = _resource_payload(
            tenant_id,
            resource_type,
            resource_id,
            owner_user_id,
            allowed_user_ids,
            metadata,
            version,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO resources
                    (resource_type, resource_id, tenant_id, owner_user_id,
                     allowed_user_ids, metadata, version, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(resource_type, resource_id) DO UPDATE SET
                    tenant_id=excluded.tenant_id,
                    owner_user_id=excluded.owner_user_id,
                    allowed_user_ids=excluded.allowed_user_ids,
                    metadata=excluded.metadata,
                    version=excluded.version,
                    updated_at=excluded.updated_at
                """,
                (
                    payload["resource_type"],
                    payload["resource_id"],
                    payload["tenant_id"],
                    payload["owner_user_id"],
                    _json(payload["allowed_user_ids"]),
                    _json(payload["metadata"]),
                    payload["version"],
                    _now(),
                ),
            )
        return payload

    def get_resource(self, resource_type: str, resource_id: str):
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM resources WHERE resource_type=? AND resource_id=?",
                (resource_type, resource_id),
            ).fetchone()
        if row is None:
            return None
        return _resource_row(row)

    def list_accessible_resources(
        self, tenant_id: str, user_id: str, resource_type: str
    ):
        """Return resource metadata only when the tenant/user policy matches."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM resources WHERE resource_type=? AND tenant_id=?",
                (resource_type, tenant_id),
            ).fetchall()
        resources = []
        for row in rows:
            resource = _resource_row(row)
            if (
                resource["owner_user_id"] == user_id
                or user_id in resource["allowed_user_ids"]
            ):
                resources.append(resource)
        return resources

    def authorize(
        self,
        tenant_id: str,
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str = "read",
    ):
        resource = self.get_resource(resource_type, resource_id)
        allowed = bool(
            resource
            and resource["tenant_id"] == tenant_id
            and (
                resource["owner_user_id"] == user_id
                or user_id in resource["allowed_user_ids"]
            )
        )
        reason = "resource policy matched" if allowed else "tenant or user policy denied"
        event = {
            "event_id": uuid.uuid4().hex,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "action": action,
            "decision": "allow" if allowed else "deny",
            "reason": reason,
            "created_at": _now(),
        }
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO audit_events
                    (event_id, tenant_id, user_id, resource_type, resource_id,
                     action, decision, reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                tuple(event.values()),
            )
        if not allowed:
            raise PermissionError(
                "Access denied for {0}/{1} in tenant {2}".format(
                    resource_type, resource_id, tenant_id
                )
            )
        return dict(event, allowed=True)

    def list_audit_events(self, limit: int = 100):
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM audit_events ORDER BY rowid ASC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def execute_idempotent(
        self,
        scope: str,
        key: str,
        payload: Dict,
        operation,
        wait_timeout_seconds: float = 10.0,
    ):
        fingerprint = _digest(_json(payload))
        owner_token = uuid.uuid4().hex
        deadline = time.monotonic() + max(0.1, float(wait_timeout_seconds))
        owns_claim = False
        while time.monotonic() < deadline:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT * FROM idempotency WHERE scope=? AND request_key=?",
                    (scope, key),
                ).fetchone()
                if row is None:
                    connection.execute(
                        """
                        INSERT INTO idempotency
                            (scope, request_key, fingerprint, status, owner_token,
                             result, error, created_at, updated_at)
                        VALUES (?, ?, ?, 'running', ?, '', '', ?, ?)
                        """,
                        (scope, key, fingerprint, owner_token, _now(), _now()),
                    )
                    connection.commit()
                    owns_claim = True
                    break
                if row["fingerprint"] != fingerprint:
                    connection.rollback()
                    raise ValueError("Idempotency key was reused with a different payload.")
                if row["status"] == "succeeded":
                    connection.commit()
                    return {
                        "result": json.loads(row["result"]),
                        "idempotent_replay": True,
                    }
                if row["status"] == "failed":
                    connection.execute(
                        """
                        UPDATE idempotency
                        SET status='running', owner_token=?, error='', updated_at=?
                        WHERE scope=? AND request_key=?
                        """,
                        (owner_token, _now(), scope, key),
                    )
                    connection.commit()
                    owns_claim = True
                    break
                connection.commit()
            finally:
                connection.close()
            time.sleep(0.01)
        if not owns_claim:
            raise TimeoutError("Timed out waiting for idempotent request result.")
        try:
            result = operation()
        except Exception as error:
            with self._connect() as connection:
                connection.execute(
                    """
                    UPDATE idempotency SET status='failed', error=?, updated_at=?
                    WHERE scope=? AND request_key=? AND owner_token=?
                    """,
                    (repr(error), _now(), scope, key, owner_token),
                )
            raise
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE idempotency SET status='succeeded', result=?, updated_at=?
                WHERE scope=? AND request_key=? AND owner_token=?
                """,
                (_json(result), _now(), scope, key, owner_token),
            )
        return {"result": result, "idempotent_replay": False}

    def set_cache(
        self,
        namespace: str,
        key: str,
        value,
        ttl_seconds: float = 300,
        tags=None,
    ):
        expires_at = time.time() + max(0.0, float(ttl_seconds))
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO cache_entries
                    (namespace, cache_key, value, tags, expires_at, created_at, last_access_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace, cache_key) DO UPDATE SET
                    value=excluded.value, tags=excluded.tags,
                    expires_at=excluded.expires_at, created_at=excluded.created_at,
                    last_access_at=excluded.last_access_at
                """,
                (namespace, key, _json(value), _json(tags or []), expires_at, _now(), _now()),
            )
        return {"namespace": namespace, "key": key, "expires_at": expires_at}

    def get_cache(self, namespace: str, key: str):
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM cache_entries WHERE namespace=? AND cache_key=?",
                (namespace, key),
            ).fetchone()
            hit = bool(row and float(row["expires_at"]) > time.time())
            if row and not hit:
                connection.execute(
                    "DELETE FROM cache_entries WHERE namespace=? AND cache_key=?",
                    (namespace, key),
                )
            if hit:
                connection.execute(
                    """
                    UPDATE cache_entries SET hit_count=hit_count+1, last_access_at=?
                    WHERE namespace=? AND cache_key=?
                    """,
                    (_now(), namespace, key),
                )
            self._increment_cache_stat(connection, "hits" if hit else "misses", 1)
        return {
            "hit": hit,
            "value": json.loads(row["value"]) if hit else None,
            "namespace": namespace,
            "key": key,
        }

    def invalidate_cache(self, tag: Optional[str] = None, namespace: Optional[str] = None):
        with self._connect() as connection:
            rows = connection.execute("SELECT namespace, cache_key, tags FROM cache_entries").fetchall()
            selected = []
            for row in rows:
                if namespace and row["namespace"] != namespace:
                    continue
                if tag and tag not in json.loads(row["tags"] or "[]"):
                    continue
                selected.append((row["namespace"], row["cache_key"]))
            connection.executemany(
                "DELETE FROM cache_entries WHERE namespace=? AND cache_key=?", selected
            )
            self._increment_cache_stat(connection, "invalidations", len(selected))
        return len(selected)

    def cache_metrics(self):
        with self._connect() as connection:
            rows = connection.execute("SELECT metric, value FROM cache_stats").fetchall()
        stats = {row["metric"]: int(row["value"]) for row in rows}
        total = stats.get("hits", 0) + stats.get("misses", 0)
        return {
            **stats,
            "hit_rate": round(stats.get("hits", 0) / max(1, total), 4),
        }

    def enqueue_job(
        self,
        tenant_id: str,
        job_type: str,
        payload: Dict,
        idempotency_key: str,
        max_attempts: int = 3,
    ):
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM jobs
                WHERE tenant_id=? AND job_type=? AND idempotency_key=?
                """,
                (tenant_id, job_type, idempotency_key),
            ).fetchone()
            if row is None:
                job_id = uuid.uuid4().hex
                connection.execute(
                    """
                    INSERT INTO jobs
                        (job_id, tenant_id, job_type, payload, idempotency_key,
                         status, attempts, max_attempts, result, error,
                         available_at, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, 'queued', 0, ?, '', '', ?, ?, ?)
                    """,
                    (
                        job_id,
                        tenant_id,
                        job_type,
                        _json(payload),
                        idempotency_key,
                        max(1, int(max_attempts)),
                        time.time(),
                        _now(),
                        _now(),
                    ),
                )
                row = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return _job_row(row)

    def run_worker_tick(self, handlers: Dict):
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM jobs
                WHERE status IN ('queued', 'retrying') AND available_at <= ?
                ORDER BY created_at ASC LIMIT 1
                """,
                (time.time(),),
            ).fetchone()
            if row is None:
                connection.commit()
                return {"status": "idle"}
            attempts = int(row["attempts"]) + 1
            connection.execute(
                "UPDATE jobs SET status='running', attempts=?, updated_at=? WHERE job_id=?",
                (attempts, _now(), row["job_id"]),
            )
            connection.commit()
        finally:
            connection.close()
        handler = handlers.get(row["job_type"])
        if handler is None:
            error = KeyError("No handler for job type {0}".format(row["job_type"]))
            return self._fail_job(row, attempts, error)
        try:
            result = handler(json.loads(row["payload"]))
        except Exception as error:
            return self._fail_job(row, attempts, error)
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE jobs SET status='succeeded', result=?, error='', updated_at=?
                WHERE job_id=?
                """,
                (_json(result), _now(), row["job_id"]),
            )
            completed = connection.execute(
                "SELECT * FROM jobs WHERE job_id=?", (row["job_id"],)
            ).fetchone()
        return _job_row(completed)

    def execute_resilient(
        self,
        operation_name: str,
        operation,
        fallback=None,
        max_attempts: int = 2,
        failure_threshold: int = 3,
        cooldown_seconds: float = 30,
    ):
        circuit = self._get_circuit(operation_name)
        if circuit["state"] == "open":
            elapsed = time.time() - float(circuit.get("opened_at") or 0)
            if elapsed < float(cooldown_seconds):
                error = RuntimeError("Circuit is open for {0}".format(operation_name))
                return _degraded_result(fallback, error, "open", 0)
        last_error = None
        attempts = 0
        for attempts in range(1, max(1, int(max_attempts)) + 1):
            try:
                result = operation()
                self._set_circuit(operation_name, "closed", 0, 0)
                return {
                    "result": result,
                    "degraded": False,
                    "circuit_state": "closed",
                    "attempts": attempts,
                }
            except (ConnectionError, TimeoutError) as error:
                last_error = error
        failures = int(circuit.get("failure_count") or 0) + attempts
        state = "open" if failures >= max(1, int(failure_threshold)) else "closed"
        self._set_circuit(
            operation_name, state, failures, time.time() if state == "open" else 0
        )
        return _degraded_result(fallback, last_error, state, attempts)

    def trace_span(
        self,
        trace_id: str,
        component: str,
        operation: str,
        parent_span_id: str = "",
        attributes: Optional[Dict] = None,
    ):
        return _TraceSpanV45(
            self, trace_id, component, operation, parent_span_id, attributes
        )

    def record_span(self, payload: Dict):
        span = {
            "span_id": payload.get("span_id") or uuid.uuid4().hex,
            "trace_id": payload.get("trace_id") or uuid.uuid4().hex,
            "parent_span_id": payload.get("parent_span_id") or "",
            "component": payload.get("component") or "agent",
            "operation": payload.get("operation") or "unknown",
            "status": payload.get("status") or "success",
            "started_at": payload.get("started_at") or payload.get("timestamp") or _now(),
            "ended_at": payload.get("ended_at") or payload.get("timestamp") or _now(),
            "duration_ms": float(payload.get("duration_ms") or 0),
            "token_count": int(payload.get("token_count") or 0),
            "cost_usd": float(payload.get("cost_usd") or 0),
            "attributes": payload.get("attributes") or payload.get("details") or {},
        }
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO spans
                    (span_id, trace_id, parent_span_id, component, operation,
                     status, started_at, ended_at, duration_ms, token_count,
                     cost_usd, attributes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    span["span_id"], span["trace_id"], span["parent_span_id"],
                    span["component"], span["operation"], span["status"],
                    span["started_at"], span["ended_at"], span["duration_ms"],
                    span["token_count"], span["cost_usd"], _json(span["attributes"]),
                ),
            )
        return span

    def list_spans(self, trace_id: str):
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM spans WHERE trace_id=? ORDER BY started_at, rowid",
                (trace_id,),
            ).fetchall()
        return [dict(row, attributes=json.loads(row["attributes"] or "{}")) for row in rows]

    def status(self):
        with self._connect() as connection:
            counts = {}
            for table in ["resources", "audit_events", "idempotency", "cache_entries", "jobs", "spans"]:
                counts[table] = connection.execute(
                    "SELECT COUNT(*) AS count FROM {0}".format(table)
                ).fetchone()["count"]
        return {
            "version": "v4.5",
            "backend": "sqlite-wal",
            "database_path": str(self.database_path),
            "counts": counts,
            "cache": self.cache_metrics(),
        }

    def _fail_job(self, row, attempts, error):
        status = "retrying" if attempts < int(row["max_attempts"]) else "failed"
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE jobs SET status=?, error=?, available_at=?, updated_at=?
                WHERE job_id=?
                """,
                (status, repr(error), time.time(), _now(), row["job_id"]),
            )
            updated = connection.execute(
                "SELECT * FROM jobs WHERE job_id=?", (row["job_id"],)
            ).fetchone()
        return _job_row(updated)

    def _increment_cache_stat(self, connection, metric, value):
        connection.execute(
            """
            INSERT INTO cache_stats(metric, value) VALUES (?, ?)
            ON CONFLICT(metric) DO UPDATE SET value=value+excluded.value
            """,
            (metric, int(value)),
        )

    def _get_circuit(self, name):
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM circuit_breakers WHERE operation_name=?", (name,)
            ).fetchone()
        return dict(row) if row else {
            "operation_name": name,
            "state": "closed",
            "failure_count": 0,
            "opened_at": 0,
        }

    def _set_circuit(self, name, state, failure_count, opened_at):
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO circuit_breakers
                    (operation_name, state, failure_count, opened_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(operation_name) DO UPDATE SET
                    state=excluded.state, failure_count=excluded.failure_count,
                    opened_at=excluded.opened_at, updated_at=excluded.updated_at
                """,
                (name, state, int(failure_count), float(opened_at), _now()),
            )

    def _connect(self):
        connection = sqlite3.connect(
            str(self.database_path),
            timeout=10,
            check_same_thread=False,
            factory=_ClosingSQLiteConnection,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=10000")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _ensure_schema(self):
        with self._schema_lock:
            with self._connect() as connection:
                connection.executescript(_SCHEMA)


class PaperStormProductionRuntimeV45:
    runtime_name = "paperstorm-production-v4.5"

    def __init__(self, root_dir, task_service, control_plane=None):
        from .paperstorm_langgraph_v44 import PaperStormLangGraphRuntime

        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.task_service = task_service
        self.control = control_plane or ProductionControlPlaneV45(
            self.root_dir / "production_v45.sqlite"
        )
        self.graph_runtime_class = PaperStormLangGraphRuntime

    def invoke(self, tenant_id: str = "local", **payload):
        tenant_id = str(tenant_id or "local")
        thread_id = str(payload["thread_id"])
        request_id = str(payload["request_id"])
        user_id = str(payload.get("user_id") or "local-user")
        resource = self.control.get_resource("conversation_thread", thread_id)
        if resource is None:
            self.control.register_resource(
                tenant_id=tenant_id,
                resource_type="conversation_thread",
                resource_id=thread_id,
                owner_user_id=user_id,
            )
        self.control.authorize(
            tenant_id, user_id, "conversation_thread", thread_id, "invoke"
        )
        idempotency_payload = dict(payload, tenant_id=tenant_id)

        def run_graph():
            trace_id = uuid.uuid4().hex
            self.control.register_resource(
                tenant_id=tenant_id,
                resource_type="trace",
                resource_id=trace_id,
                owner_user_id=user_id,
            )
            with self.control.trace_span(
                trace_id, "agent_runtime", "conversation_graph"
            ):
                runtime = self.graph_runtime_class(
                    self.root_dir / "langgraph_v44",
                    self.task_service,
                )
                graph_result = runtime.invoke(**payload)
            for event in graph_result.get("node_events") or []:
                self.control.record_span(
                    {
                        "span_id": event.get("span_id"),
                        "trace_id": trace_id,
                        "component": _component_for_node(event.get("node", "")),
                        "operation": event.get("node", ""),
                        "status": event.get("status", "success"),
                        "timestamp": event.get("timestamp"),
                        "duration_ms": event.get("duration_ms", 0),
                        "attributes": event.get("details") or {},
                    }
                )
            return dict(
                graph_result,
                runtime=self.runtime_name,
                graph_runtime=graph_result.get("runtime", "langgraph-v4.4"),
                trace_id=trace_id,
            )

        outcome = self.control.execute_idempotent(
            scope="{0}/{1}".format(tenant_id, thread_id),
            key=request_id,
            payload=idempotency_payload,
            operation=run_graph,
        )
        result = dict(outcome["result"])
        result["governance"] = {
            "tenant_id": tenant_id,
            "idempotent_replay": bool(outcome["idempotent_replay"]),
            "control_plane": "sqlite-wal-v4.5",
        }
        return result

    def get_thread_state(self, thread_id: str):
        runtime = self.graph_runtime_class(
            self.root_dir / "langgraph_v44", self.task_service
        )
        return runtime.get_thread_state(thread_id)

    def get_thread_history(self, thread_id: str, limit: int = 50):
        runtime = self.graph_runtime_class(
            self.root_dir / "langgraph_v44", self.task_service
        )
        return runtime.get_thread_history(thread_id, limit=limit)

    def get_graph_spec(self):
        runtime = self.graph_runtime_class(
            self.root_dir / "langgraph_v44", self.task_service
        )
        return dict(
            runtime.get_graph_spec(),
            runtime=self.runtime_name,
            governance={
                "acl": "tenant/resource/user pre-access policy",
                "idempotency": "SQLite unique(scope, request_key)",
                "telemetry": "SQLite span store",
                "cache": "TTL + tag invalidation",
            },
        )


class _TraceSpanV45(AbstractContextManager):
    def __init__(self, control, trace_id, component, operation, parent_span_id, attributes):
        self.control = control
        self.trace_id = trace_id
        self.component = component
        self.operation = operation
        self.parent_span_id = parent_span_id
        self.attributes = attributes or {}
        self.span_id = uuid.uuid4().hex
        self.started_at = _now()
        self.started = time.perf_counter()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.control.record_span(
            {
                "span_id": self.span_id,
                "trace_id": self.trace_id,
                "parent_span_id": self.parent_span_id,
                "component": self.component,
                "operation": self.operation,
                "status": "error" if exc else "success",
                "started_at": self.started_at,
                "ended_at": _now(),
                "duration_ms": (time.perf_counter() - self.started) * 1000,
                "attributes": dict(
                    self.attributes, error=repr(exc) if exc else ""
                ),
            }
        )
        return False


def _resource_payload(tenant_id, resource_type, resource_id, owner, allowed, metadata, version):
    return {
        "tenant_id": str(tenant_id or "local"),
        "resource_type": str(resource_type),
        "resource_id": str(resource_id),
        "owner_user_id": str(owner or "local-user"),
        "allowed_user_ids": sorted(set(str(item) for item in (allowed or []))),
        "metadata": metadata or {},
        "version": max(1, int(version)),
    }


def _resource_row(row):
    return {
        "tenant_id": row["tenant_id"],
        "resource_type": row["resource_type"],
        "resource_id": row["resource_id"],
        "owner_user_id": row["owner_user_id"],
        "allowed_user_ids": json.loads(row["allowed_user_ids"] or "[]"),
        "metadata": json.loads(row["metadata"] or "{}"),
        "version": int(row["version"]),
    }


def _job_row(row):
    result = dict(row)
    result["payload"] = json.loads(result.get("payload") or "{}")
    result["result"] = json.loads(result.get("result") or "{}")
    return result


def _degraded_result(fallback, error, state, attempts):
    if fallback is None:
        raise error
    return {
        "result": fallback(error),
        "degraded": True,
        "circuit_state": state,
        "attempts": attempts,
        "error": repr(error),
    }


def _component_for_node(node):
    if "memory" in node:
        return "memory"
    if node in {"knowledge_retrieval", "evidence_grade"}:
        return "retrieval"
    if node == "deep_research":
        return "storm_tool"
    if node == "classify":
        return "router"
    return "agent_runtime"


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS resources (
    resource_type TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    owner_user_id TEXT NOT NULL,
    allowed_user_ids TEXT NOT NULL,
    metadata TEXT NOT NULL,
    version INTEGER NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(resource_type, resource_id)
);
CREATE TABLE IF NOT EXISTS audit_events (
    event_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    resource_type TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    action TEXT NOT NULL,
    decision TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS idempotency (
    scope TEXT NOT NULL,
    request_key TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    status TEXT NOT NULL,
    owner_token TEXT NOT NULL,
    result TEXT NOT NULL,
    error TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(scope, request_key)
);
CREATE TABLE IF NOT EXISTS cache_entries (
    namespace TEXT NOT NULL,
    cache_key TEXT NOT NULL,
    value TEXT NOT NULL,
    tags TEXT NOT NULL,
    expires_at REAL NOT NULL,
    created_at TEXT NOT NULL,
    last_access_at TEXT NOT NULL,
    hit_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(namespace, cache_key)
);
CREATE TABLE IF NOT EXISTS cache_stats (
    metric TEXT PRIMARY KEY,
    value INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    job_type TEXT NOT NULL,
    payload TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL,
    max_attempts INTEGER NOT NULL,
    result TEXT NOT NULL,
    error TEXT NOT NULL,
    available_at REAL NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(tenant_id, job_type, idempotency_key)
);
CREATE TABLE IF NOT EXISTS circuit_breakers (
    operation_name TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    failure_count INTEGER NOT NULL,
    opened_at REAL NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS spans (
    span_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    parent_span_id TEXT NOT NULL,
    component TEXT NOT NULL,
    operation TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT NOT NULL,
    duration_ms REAL NOT NULL,
    token_count INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    attributes TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_tenant ON audit_events(tenant_id, created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, available_at);
CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id, started_at);
"""
