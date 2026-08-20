"""Optional Langfuse observability with a fail-open local event mirror."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


REDACTED = "***REDACTED***"
_SECRET_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "password",
    "secret",
    "token",
    "access_token",
    "refresh_token",
}
_IDENTITY_KEYS = ("user_id", "email", "username")


def sanitize_payload(value: Any, max_string_length: int = 4000) -> Any:
    """Return JSON-safe telemetry data with credentials and identities protected."""
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            normalized = str(key).lower()
            if _is_secret_key(normalized):
                output[str(key)] = REDACTED
            elif normalized in _IDENTITY_KEYS and item:
                output[str(key)] = _pseudonymize(item)
            else:
                output[str(key)] = sanitize_payload(item, max_string_length)
        return output
    if isinstance(value, (list, tuple, set)):
        return [sanitize_payload(item, max_string_length) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        return value if len(value) <= max_string_length else value[:max_string_length] + "...[truncated]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return sanitize_payload(str(value), max_string_length)


def _pseudonymize(value: Any) -> str:
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]
    return "user_{0}".format(digest)


def _is_secret_key(key: str) -> bool:
    normalized = key.replace("-", "_").replace(" ", "_")
    return normalized in _SECRET_KEYS or normalized.endswith(
        ("_api_key", "_access_token", "_refresh_token", "_password", "_secret")
    )


def numeric_scores(payload: Any, prefix: str = "") -> Dict[str, float]:
    """Flatten numeric leaves so existing benchmark reports become trace scores."""
    scores = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            name = "{0}.{1}".format(prefix, key) if prefix else str(key)
            scores.update(numeric_scores(value, name))
    elif isinstance(payload, (int, float)) and not isinstance(payload, bool) and prefix:
        scores[prefix] = float(payload)
    return scores


class PaperStormObservability:
    """Record local telemetry and optionally export the same model to Langfuse."""

    def __init__(self, root_dir, enabled=False, langfuse_client=None):
        self.root_dir = Path(root_dir) / "observability"
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.root_dir / "events.jsonl"
        self.enabled = bool(enabled)
        self.client = langfuse_client
        self._lock = threading.Lock()
        self._export_failures = 0
        self._local_write_failures = 0
        self._last_error = ""

    def trace(self, name, input=None, metadata=None, session_id="", user_id="", tags=None):
        return TraceHandle(
            self,
            name=name,
            input=input or {},
            metadata=metadata or {},
            session_id=session_id,
            user_id=user_id,
            tags=tags or [],
        )

    def status(self):
        if self._local_write_failures or self._export_failures:
            status = "degraded"
        elif not self.enabled:
            status = "local-only"
        elif self.client is None:
            status = "unavailable"
        else:
            # Langfuse exports asynchronously. Client construction confirms the
            # configuration, not end-to-end collector reachability.
            status = "configured"
        return {
            "provider": "langfuse",
            "status": status,
            "remote_enabled": self.enabled,
            "sdk_available": self.client is not None,
            "export_failures": self._export_failures,
            "local_write_failures": self._local_write_failures,
            "last_error": self._last_error,
            "local_events_path": str(self.events_path),
            "environment": os.getenv("LANGFUSE_TRACING_ENVIRONMENT", "development"),
        }

    def flush(self):
        if self.client is not None:
            self._remote(lambda: self.client.flush())

    def _record(self, event, **payload):
        row = {"event": event, "timestamp": _now(), **sanitize_payload(payload)}
        try:
            with self._lock:
                with self.events_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception as error:  # Telemetry storage is outside the agent contract.
            self._local_write_failures += 1
            self._last_error = sanitize_payload(str(error), 500)

    def _remote(self, operation):
        if not self.enabled or self.client is None:
            return None
        try:
            return operation()
        except Exception as error:  # Observability must never break the agent path.
            self._export_failures += 1
            self._last_error = sanitize_payload(str(error), 500)
            return None


class TraceHandle(AbstractContextManager):
    def __init__(self, owner, name, input, metadata, session_id, user_id, tags):
        self.owner = owner
        self.name = str(name)
        self.trace_id = uuid.uuid4().hex
        self.input = sanitize_payload(input)
        self.metadata = sanitize_payload(metadata)
        self.session_id = str(session_id or "")
        self.user_id = _pseudonymize(user_id) if user_id else ""
        self.tags = [str(tag) for tag in tags]
        self.remote = None
        self.closed = False

    def __enter__(self):
        self.owner._record(
            "trace.start",
            trace_id=self.trace_id,
            name=self.name,
            input=self.input,
            metadata=self.metadata,
            session_id=self.session_id,
            user_id=self.user_id,
            tags=self.tags,
        )
        self.remote = self.owner._remote(
            lambda: self.owner.client.start_observation(
                name=self.name,
                as_type="agent",
                input=self.input,
                metadata={
                    **self.metadata,
                    "paperstorm_trace_id": self.trace_id,
                    "langfuse_session_id": self.session_id,
                    "langfuse_user_id": self.user_id,
                    "langfuse_tags": self.tags,
                },
            )
        )
        return self

    def span(self, name, input=None, metadata=None, as_type="span"):
        return SpanHandle(self, name, input or {}, metadata or {}, as_type)

    def score(self, name, value, comment=""):
        payload = {"name": str(name), "value": value, "comment": str(comment or "")}
        self.owner._record("score", trace_id=self.trace_id, **payload)
        if self.remote is not None:
            self.owner._remote(lambda: self.remote.score_trace(**payload))

    def end(self, output=None, error=None, metadata=None):
        if self.closed:
            return
        payload = {
            "output": sanitize_payload(output or {}),
            "metadata": sanitize_payload(metadata or {}),
            "error": sanitize_payload(str(error), 1000) if error else "",
        }
        self.owner._record("trace.end", trace_id=self.trace_id, name=self.name, **payload)
        if self.remote is not None:
            self.owner._remote(lambda: self.remote.update(**_remote_update(payload)))
            self.owner._remote(lambda: self.remote.end())
        self.closed = True

    def __exit__(self, exc_type, exc, _traceback):
        if not self.closed:
            self.end(error=exc if exc is not None else None)
        return False


class SpanHandle(AbstractContextManager):
    def __init__(self, trace, name, input, metadata, as_type):
        self.trace = trace
        self.owner = trace.owner
        self.name = str(name)
        self.span_id = uuid.uuid4().hex
        self.input = sanitize_payload(input)
        self.metadata = sanitize_payload(metadata)
        self.as_type = str(as_type or "span")
        self.remote = None
        self.closed = False

    def __enter__(self):
        self.owner._record(
            "span.start",
            trace_id=self.trace.trace_id,
            span_id=self.span_id,
            parent_id=self.trace.trace_id,
            name=self.name,
            as_type=self.as_type,
            input=self.input,
            metadata=self.metadata,
        )
        if self.trace.remote is not None:
            self.remote = self.owner._remote(
                lambda: self.trace.remote.start_observation(
                    name=self.name,
                    as_type=self.as_type,
                    input=self.input,
                    metadata={**self.metadata, "paperstorm_span_id": self.span_id},
                )
            )
        return self

    def end(self, output=None, error=None, metadata=None):
        if self.closed:
            return
        payload = {
            "output": sanitize_payload(output or {}),
            "metadata": sanitize_payload(metadata or {}),
            "error": sanitize_payload(str(error), 1000) if error else "",
        }
        self.owner._record(
            "span.end",
            trace_id=self.trace.trace_id,
            span_id=self.span_id,
            parent_id=self.trace.trace_id,
            name=self.name,
            **payload,
        )
        if self.remote is not None:
            self.owner._remote(lambda: self.remote.update(**_remote_update(payload)))
            self.owner._remote(lambda: self.remote.end())
        self.closed = True

    def __exit__(self, exc_type, exc, _traceback):
        if not self.closed:
            self.end(error=exc if exc is not None else None)
        return False


def build_observability(root_dir, langfuse_client=None):
    offline = str(os.getenv("PAPERSTORM_TEST_OFFLINE", "0")).strip().lower() in {
        "1", "true", "yes", "on"
    }
    enabled = (
        not offline
        and os.getenv("PAPERSTORM_OBSERVABILITY", "").lower() == "langfuse"
    )
    credentials = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))
    client = None if offline else langfuse_client
    if enabled and credentials and client is None:
        try:
            from langfuse import get_client

            client = get_client()
        except Exception:
            client = None
    return PaperStormObservability(
        root_dir,
        enabled=enabled and credentials,
        langfuse_client=client,
    )


def _remote_update(payload):
    update = {
        "output": payload.get("output") or {},
        "metadata": payload.get("metadata") or {},
    }
    if payload.get("error"):
        update["level"] = "ERROR"
        update["status_message"] = payload["error"]
    return update


def _now():
    return datetime.now(timezone.utc).isoformat()
