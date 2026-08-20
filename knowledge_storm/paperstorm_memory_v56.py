"""Production-oriented local memory substrate for PaperStorm v5.6.

The module deliberately keeps storage local and dependency-light.  It borrows
the episode/fact/provenance model from temporal context graphs without forcing
users to operate a graph database.
"""

import hashlib
import functools
import json
import math
import os
import re
import sqlite3
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from .paperstorm_memory_v43 import MemoryCandidateV43, MemoryWritePolicy, _model_dump
class LongTermMemoryServiceV56:
    """SQLite-backed episodic and long-term memory with temporal retrieval."""

    def __init__(
        self,
        root_dir,
        embedding_provider=None,
        candidate_extractor=None,
        retrieval_mode=None,
    ):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root_dir / "memory_v56.sqlite3"
        explicit_mode = retrieval_mode is not None
        self.retrieval_mode = str(
            retrieval_mode or ("semantic" if embedding_provider is not None else "lexical")
        ).strip().lower()
        if self.retrieval_mode not in {"lexical", "semantic"}:
            raise ValueError("memory retrieval_mode must be lexical or semantic")
        if self.retrieval_mode == "lexical":
            self.embedding_provider = None
        else:
            self.embedding_provider = embedding_provider or build_memory_embedding_provider()
            if explicit_mode and _is_hash_provider(self.embedding_provider):
                raise ValueError(
                    "semantic memory requires a real semantic embedding model; "
                    "hash embeddings are allowed only in explicit offline benchmark fixtures"
                )
        self.candidate_extractor = candidate_extractor
        self._initialize()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(str(self.db_path), timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self):
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS memory_episodes (
                    episode_id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(namespace, source_id, content_hash)
                );
                CREATE INDEX IF NOT EXISTS idx_episodes_namespace_time
                    ON memory_episodes(namespace, occurred_at);

                CREATE TABLE IF NOT EXISTS memory_facts (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    content TEXT NOT NULL,
                    canonical_key TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    importance REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    valid_from TEXT NOT NULL,
                    valid_to TEXT,
                    expires_at TEXT,
                    status TEXT NOT NULL,
                    supersedes_id TEXT,
                    metadata_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_facts_namespace_key
                    ON memory_facts(namespace, canonical_key);
                CREATE INDEX IF NOT EXISTS idx_facts_namespace_validity
                    ON memory_facts(namespace, valid_from, valid_to, status);

                CREATE TABLE IF NOT EXISTS memory_fact_sources (
                    fact_id TEXT NOT NULL,
                    episode_id TEXT,
                    source_id TEXT NOT NULL,
                    PRIMARY KEY(fact_id, source_id),
                    FOREIGN KEY(fact_id) REFERENCES memory_facts(id)
                );
                CREATE TABLE IF NOT EXISTS memory_entities (
                    entity_id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    canonical_name TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    UNIQUE(namespace, canonical_name)
                );
                CREATE TABLE IF NOT EXISTS memory_fact_entities (
                    fact_id TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    PRIMARY KEY(fact_id, entity_id),
                    FOREIGN KEY(fact_id) REFERENCES memory_facts(id),
                    FOREIGN KEY(entity_id) REFERENCES memory_entities(entity_id)
                );
                CREATE TABLE IF NOT EXISTS memory_events (
                    event_id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memory_settings (
                    namespace TEXT PRIMARY KEY,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memory_candidates (
                    candidate_id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memory_fact_vectors (
                    fact_id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    dim INTEGER NOT NULL,
                    vector_blob BLOB NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_fact_vectors_backend
                    ON memory_fact_vectors(backend, namespace);
                """
            )

    def storage_info(self):
        with self._connect() as connection:
            journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        return {"backend": "sqlite", "path": str(self.db_path), "journal_mode": journal_mode}

    def ingest_episode(
        self,
        namespace: str,
        content: str,
        source_id: str = "",
        role: str = "user",
        occurred_at: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        namespace = _validate_namespace(namespace)
        content = str(content or "").strip()
        if not content:
            raise ValueError("episode content is required")
        source_id = str(source_id or uuid.uuid4().hex)
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        occurred_at = occurred_at or _now()
        with self._connect() as connection:
            row = connection.execute(
                """SELECT episode_id FROM memory_episodes
                   WHERE namespace=? AND source_id=? AND content_hash=?""",
                (namespace, source_id, content_hash),
            ).fetchone()
            if row:
                return {"episode_id": row["episode_id"], "deduplicated": True}
            episode_id = uuid.uuid4().hex
            connection.execute(
                """INSERT INTO memory_episodes
                   (episode_id, namespace, source_id, role, content, content_hash,
                    occurred_at, metadata_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    episode_id,
                    namespace,
                    source_id,
                    str(role or "user"),
                    content,
                    content_hash,
                    occurred_at,
                    _json(metadata or {}),
                    _now(),
                ),
            )
            self._event(connection, namespace, "episode_ingested", {"episode_id": episode_id, "source_id": source_id})
        return {"episode_id": episode_id, "deduplicated": False}

    def list_episodes(self, namespace: str):
        namespace = _validate_namespace(namespace)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM memory_episodes WHERE namespace=? ORDER BY occurred_at, created_at",
                (namespace,),
            ).fetchall()
        return [_episode_dict(row) for row in rows]

    def ingest_message(self, namespace: str, message: str, source_message_id: str = "", subject: str = "user"):
        episode = self.ingest_episode(namespace, message, source_id=source_message_id, role=subject)
        if _blocks_memory_write(message):
            return {"status": "skipped", "reason": "user explicitly blocked durable memory", "episode": episode}
        candidate = self._extract_candidate(message, source_message_id, subject)
        if candidate is None:
            return {"status": "skipped", "reason": "no durable memory signal", "episode": episode}
        payload = _model_dump(candidate)
        payload.setdefault("metadata", {})["episode_id"] = episode["episode_id"]
        if float(candidate.confidence) < 0.85:
            queued = self._queue_candidate(namespace, payload)
            return {"status": "queued", "candidate": queued, "episode": episode}
        return {"status": "persisted", "memory": self.upsert(namespace=namespace, **payload), "episode": episode}

    def _extract_candidate(self, message, source_message_id, subject):
        if self.candidate_extractor is None:
            return MemoryWritePolicy.extract(
                message, source_message_id=source_message_id, subject=subject
            )
        prompt = build_memory_candidate_prompt(message, source_message_id, subject)
        try:
            raw = self.candidate_extractor(prompt)
            if isinstance(raw, str):
                match = re.search(r"\{.*\}", raw, flags=re.S)
                if not match:
                    return None
                raw = json.loads(match.group(0))
            if not isinstance(raw, dict) or raw.get("should_write") is False:
                return None
            payload = dict(raw)
            payload.pop("should_write", None)
            payload.setdefault("subject", subject)
            payload.setdefault("source_message_ids", [source_message_id] if source_message_id else [])
            payload.setdefault("importance", 0.7)
            payload.setdefault("confidence", 0.7)
            payload.setdefault("metadata", {})
            payload["metadata"] = dict(payload["metadata"], extractor="llm_structured")
            return MemoryCandidateV43(**payload)
        except Exception:
            # Extraction failure must not turn an ordinary chat turn into a
            # durable write. Explicit rule signals remain a safe fallback.
            return MemoryWritePolicy.extract(
                message, source_message_id=source_message_id, subject=subject
            )

    def upsert(
        self,
        namespace: str,
        memory_type: str,
        subject: str,
        content: str,
        canonical_key: str,
        source_message_ids: Optional[List[str]] = None,
        confidence: float = 0.9,
        importance: float = 0.7,
        valid_from: Optional[str] = None,
        valid_to: Optional[str] = None,
        expires_at: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        namespace = _validate_namespace(namespace)
        content = str(content or "").strip()
        canonical_key = str(canonical_key or "").strip().lower()
        if not content or not canonical_key:
            raise ValueError("memory content and canonical_key are required")
        valid_from = valid_from or _now()
        metadata = dict(metadata or {})
        source_message_ids = [str(item) for item in (source_message_ids or []) if item]
        with self._connect() as connection:
            active = connection.execute(
                """SELECT * FROM memory_facts
                   WHERE namespace=? AND canonical_key=? AND status='active'
                   ORDER BY valid_from DESC LIMIT 1""",
                (namespace, canonical_key),
            ).fetchone()
            if active and _normalize(active["content"]) == _normalize(content):
                result = self._fact_dict(connection, active)
                result["deduplicated"] = True
                self._event(connection, namespace, "memory_deduplicated", {"memory_id": active["id"]})
                return result

            supersedes_id = active["id"] if active else None
            if active:
                connection.execute(
                    "UPDATE memory_facts SET status='superseded', valid_to=?, updated_at=? WHERE id=?",
                    (valid_from, _now(), active["id"]),
                )
            memory_id = uuid.uuid4().hex
            now = _now()
            connection.execute(
                """INSERT INTO memory_facts
                   (id, namespace, memory_type, subject, content, canonical_key,
                    confidence, importance, created_at, updated_at, valid_from,
                    valid_to, expires_at, status, supersedes_id, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)""",
                (
                    memory_id,
                    namespace,
                    memory_type,
                    str(subject or "unknown"),
                    content,
                    canonical_key,
                    float(confidence),
                    float(importance),
                    now,
                    now,
                    valid_from,
                    valid_to,
                    expires_at,
                    supersedes_id,
                    _json(metadata),
                ),
            )
            self._attach_sources(connection, memory_id, namespace, source_message_ids, metadata.get("episode_id"))
            self._attach_entities(connection, memory_id, namespace, _entities(content, metadata))
            self._event(connection, namespace, "memory_upserted", {"memory_id": memory_id, "supersedes_id": supersedes_id})
            row = connection.execute("SELECT * FROM memory_facts WHERE id=?", (memory_id,)).fetchone()
            result = self._fact_dict(connection, row)
            if self.embedding_provider is not None:
                self._store_vector(
                    connection,
                    memory_id,
                    namespace,
                    self._embed_text(_search_text(result)),
                )
        result["deduplicated"] = False
        return result

    def list_memories(self, namespace: str, include_inactive: bool = False, memory_type: Optional[str] = None):
        namespace = _validate_namespace(namespace)
        query = "SELECT * FROM memory_facts WHERE namespace=?"
        parameters: List = [namespace]
        if not include_inactive:
            query += " AND status='active' AND valid_from<=? AND (valid_to IS NULL OR valid_to>?) AND (expires_at IS NULL OR expires_at>?)"
            now = _now()
            parameters.extend([now, now, now])
        if memory_type:
            query += " AND memory_type=?"
            parameters.append(memory_type)
        query += " ORDER BY updated_at DESC"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
            return [self._fact_dict(connection, row) for row in rows]

    def get_memory(self, namespace: str, memory_id: str, include_inactive: bool = False):
        return next((item for item in self.list_memories(namespace, include_inactive) if item["id"] == memory_id), None)

    def search(
        self,
        namespace: str,
        query: str,
        top_k: int = 5,
        memory_types: Optional[List[str]] = None,
        as_of: Optional[str] = None,
    ):
        started = time.perf_counter()
        namespace = _validate_namespace(namespace)
        if not self.is_enabled(namespace):
            return {"status": "disabled", "namespace": namespace, "query": query, "results": [], "latency_ms": _elapsed(started)}
        at = as_of or _now()
        sql = """SELECT * FROM memory_facts WHERE namespace=? AND status!='deleted'
                 AND valid_from<=? AND (valid_to IS NULL OR valid_to>?)
                 AND (expires_at IS NULL OR expires_at>?)"""
        parameters: List = [namespace, at, at, at]
        if memory_types:
            sql += " AND memory_type IN ({})".format(",".join("?" for _ in memory_types))
            parameters.extend(memory_types)
        with self._connect() as connection:
            rows = connection.execute(sql, parameters).fetchall()
            records = [self._fact_dict(connection, row) for row in rows]
        ranked = self._rank(records, query, at)
        return {
            "status": "ok",
            "namespace": namespace,
            "query": query,
            "as_of": at,
            "top_k": max(1, int(top_k or 5)),
            "candidate_count": len(records),
            "results": _mmr_select(ranked, max(1, int(top_k or 5))),
            "retrieval_mode": self.retrieval_mode,
            "embedding_backend": str(
                getattr(self.embedding_provider, "name", "disabled")
                if self.embedding_provider is not None
                else "disabled"
            ),
            "embedding_kind": (
                "disabled"
                if self.embedding_provider is None
                else "test_hash"
                if _is_hash_provider(self.embedding_provider)
                else "semantic_model"
            ),
            "retrieval": (
                "namespace+time filter -> BM25+{0}+entity -> RRF -> importance+recency -> MMR".format(
                    "test-hash-dense"
                    if _is_hash_provider(self.embedding_provider)
                    else "real-dense"
                )
                if self.embedding_provider is not None
                else "namespace+time filter -> BM25+entity -> RRF -> importance+recency -> MMR"
            ),
            "latency_ms": _elapsed(started),
        }

    def edit(self, namespace: str, memory_id: str, content: str, **updates):
        current = self.get_memory(namespace, memory_id, include_inactive=True)
        if not current:
            raise KeyError("Unknown memory_id: {0}".format(memory_id))
        return self.upsert(
            namespace=namespace,
            memory_type=updates.get("memory_type", current["memory_type"]),
            subject=updates.get("subject", current["subject"]),
            content=content,
            canonical_key=updates.get("canonical_key", current["canonical_key"]),
            source_message_ids=updates.get("source_message_ids", current.get("source_message_ids", [])),
            confidence=updates.get("confidence", current["confidence"]),
            importance=updates.get("importance", current["importance"]),
            expires_at=updates.get("expires_at", current.get("expires_at")),
            metadata=dict(current.get("metadata") or {}, edited_from=memory_id),
        )

    def delete(self, namespace: str, memory_id: str, reason: str = "user_request"):
        namespace = _validate_namespace(namespace)
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM memory_facts WHERE namespace=? AND id=?", (namespace, memory_id)).fetchone()
            if not row:
                raise KeyError("Unknown memory_id: {0}".format(memory_id))
            connection.execute("UPDATE memory_facts SET status='deleted', updated_at=? WHERE id=?", (_now(), memory_id))
            self._event(connection, namespace, "memory_deleted", {"memory_id": memory_id, "reason": reason})
            result = self._fact_dict(connection, connection.execute("SELECT * FROM memory_facts WHERE id=?", (memory_id,)).fetchone())
        return result

    def set_enabled(self, namespace: str, enabled: bool):
        namespace = _validate_namespace(namespace)
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO memory_settings(namespace, enabled, updated_at) VALUES (?, ?, ?)
                   ON CONFLICT(namespace) DO UPDATE SET enabled=excluded.enabled, updated_at=excluded.updated_at""",
                (namespace, int(bool(enabled)), _now()),
            )
        return {"namespace": namespace, "enabled": bool(enabled)}

    def is_enabled(self, namespace: str):
        with self._connect() as connection:
            row = connection.execute("SELECT enabled FROM memory_settings WHERE namespace=?", (_validate_namespace(namespace),)).fetchone()
        return True if row is None else bool(row["enabled"])

    def audit_events(self, namespace: Optional[str] = None):
        with self._connect() as connection:
            if namespace:
                rows = connection.execute("SELECT * FROM memory_events WHERE namespace=? ORDER BY created_at", (_validate_namespace(namespace),)).fetchall()
            else:
                rows = connection.execute("SELECT * FROM memory_events ORDER BY created_at").fetchall()
        return [{"event_id": row["event_id"], "namespace": row["namespace"], "event": row["event_type"], "timestamp": row["created_at"], "payload": _loads(row["payload_json"])} for row in rows]

    def export_namespace(self, namespace: str):
        return {"namespace": namespace, "enabled": self.is_enabled(namespace), "exported_at": _now(), "episodes": self.list_episodes(namespace), "memories": self.list_memories(namespace, True), "events": self.audit_events(namespace)}

    def consolidate_pending(self, minimum_confidence: float = 0.65):
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM memory_candidates WHERE status='pending'").fetchall()
        persisted, skipped = [], []
        for row in rows:
            payload = _loads(row["payload_json"])
            if float(payload.get("confidence", 0)) >= minimum_confidence:
                persisted.append(self.upsert(namespace=row["namespace"], **payload))
                status = "persisted"
            else:
                skipped.append(row["candidate_id"])
                status = "skipped"
            with self._connect() as connection:
                connection.execute("UPDATE memory_candidates SET status=? WHERE candidate_id=?", (status, row["candidate_id"]))
        return {"persisted": persisted, "skipped": skipped, "processed": len(persisted) + len(skipped)}

    def _rank(self, records, query, as_of):
        if not records:
            return []
        query_tokens = _tokenize(query)
        corpus = [_tokenize(_search_text(item)) for item in records]
        lexical = _bm25_scores(corpus, query_tokens)
        dense = None
        vectors = [[] for _ in records]
        if self.embedding_provider is not None:
            query_vector = self.embedding_provider.embed_query(query)
            vectors = self._load_vectors(records)
            dense = [float(_cosine(query_vector, vector)) for vector in vectors]
        query_entities = {item.lower() for item in _entities(query, {})}
        entity = [len(query_entities.intersection({item.lower() for item in record.get("entities", [])})) / max(1, len(query_entities)) for record in records]
        signals = [lexical, entity]
        if dense is not None:
            signals.insert(1, dense)
        ranks = [_ranks(signal) for signal in signals]
        reference = _parse_datetime(as_of)
        output = []
        for index, record in enumerate(records):
            rrf = sum(1.0 / (60 + rank[index]) for rank in ranks)
            valid_from = _parse_datetime(record["valid_from"])
            temporal = 1.0 if valid_from <= reference else 0.0
            age_days = max(0.0, (reference - valid_from).total_seconds() / 86400.0)
            recency = 1.0 / (1.0 + age_days / 30.0)
            final = rrf + 0.02 * float(record["importance"]) + 0.015 * entity[index] + 0.01 * temporal + 0.005 * recency
            item = dict(record)
            item["scores"] = {"lexical": round(lexical[index], 6), "entity": round(entity[index], 6), "temporal": temporal, "rrf": round(rrf, 6), "importance": record["importance"], "recency": round(recency, 6), "final": round(final, 6)}
            reasons = [("lexical", lexical[index]), ("entity", entity[index]), ("temporal", temporal)]
            if dense is not None:
                item["scores"]["dense"] = round(dense[index], 6)
                reasons.insert(1, ("dense", dense[index]))
                item["_vector"] = vectors[index].tolist()
            item["retrieval_reasons"] = [name for name, score in reasons if score > 0]
            output.append(item)
        output.sort(key=lambda item: item["scores"]["final"], reverse=True)
        return output

    def _provider_fingerprint(self):
        provider = self.embedding_provider
        name = str(getattr(provider, "name", None) or type(provider).__name__)
        model = str(getattr(provider, "model_name", None) or "")
        dim = int(getattr(provider, "dim", 0) or 0)
        return "{0}|{1}|{2}".format(name, model, dim)

    def _embed_text(self, text):
        vector = self.embedding_provider.embed([str(text or "")])[0]
        return np.asarray(vector, dtype=np.float32).reshape(-1)

    def _store_vector(self, connection, fact_id, namespace, vector):
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        connection.execute(
            """INSERT OR REPLACE INTO memory_fact_vectors
               (fact_id, namespace, backend, dim, vector_blob, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (fact_id, namespace, self._provider_fingerprint(), int(vector.size), vector.tobytes(), _now()),
        )

    def _load_vectors(self, records):
        if not records:
            return []
        backend = self._provider_fingerprint()
        record_ids = [str(record["id"]) for record in records]
        placeholders = ",".join("?" for _ in record_ids)
        vectors_by_id = {}
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT fact_id, backend, dim, vector_blob FROM memory_fact_vectors
                   WHERE fact_id IN ({0})""".format(placeholders),
                record_ids,
            ).fetchall()
            for row in rows:
                if str(row["backend"]) != backend:
                    continue
                vectors_by_id[str(row["fact_id"])] = np.frombuffer(
                    bytes(row["vector_blob"]), dtype=np.float32
                ).reshape(int(row["dim"]))
        missing = [record for record in records if str(record["id"]) not in vectors_by_id]
        if missing:
            encoded = self.embedding_provider.embed([_search_text(item) for item in missing])
            with self._connect() as connection:
                for record, vector in zip(missing, encoded):
                    self._store_vector(
                        connection, str(record["id"]), str(record["namespace"]), vector
                    )
            for record, vector in zip(missing, encoded):
                vectors_by_id[str(record["id"])] = np.asarray(vector, dtype=np.float32).reshape(-1)
        return [vectors_by_id[str(record["id"])] for record in records]
    def _attach_sources(self, connection, fact_id, namespace, source_ids, episode_id):
        for source_id in source_ids:
            linked = episode_id
            if not linked:
                row = connection.execute("SELECT episode_id FROM memory_episodes WHERE namespace=? AND source_id=? ORDER BY created_at DESC LIMIT 1", (namespace, source_id)).fetchone()
                linked = row["episode_id"] if row else None
            connection.execute("INSERT OR IGNORE INTO memory_fact_sources(fact_id, episode_id, source_id) VALUES (?, ?, ?)", (fact_id, linked, source_id))

    def _attach_entities(self, connection, fact_id, namespace, entities):
        for display_name in entities:
            canonical = display_name.lower().strip()
            entity_id = hashlib.sha256((namespace + "\0" + canonical).encode("utf-8")).hexdigest()[:32]
            connection.execute("INSERT OR IGNORE INTO memory_entities(entity_id, namespace, canonical_name, display_name) VALUES (?, ?, ?, ?)", (entity_id, namespace, canonical, display_name))
            connection.execute("INSERT OR IGNORE INTO memory_fact_entities(fact_id, entity_id) VALUES (?, ?)", (fact_id, entity_id))

    def _fact_dict(self, connection, row):
        sources = connection.execute("SELECT source_id, episode_id FROM memory_fact_sources WHERE fact_id=? ORDER BY source_id", (row["id"],)).fetchall()
        entities = connection.execute("""SELECT e.display_name FROM memory_entities e JOIN memory_fact_entities fe ON fe.entity_id=e.entity_id WHERE fe.fact_id=? ORDER BY e.display_name""", (row["id"],)).fetchall()
        result = dict(row)
        result["metadata"] = _loads(result.pop("metadata_json"))
        result["provenance"] = [dict(item) for item in sources]
        result["source_message_ids"] = [item["source_id"] for item in sources]
        result["entities"] = [item["display_name"] for item in entities]
        return result

    def _event(self, connection, namespace, event_type, payload):
        connection.execute("INSERT INTO memory_events(event_id, namespace, event_type, payload_json, created_at) VALUES (?, ?, ?, ?, ?)", (uuid.uuid4().hex, namespace, event_type, _json(payload), _now()))

    def _queue_candidate(self, namespace, payload):
        candidate_id = uuid.uuid4().hex
        with self._connect() as connection:
            connection.execute("INSERT INTO memory_candidates(candidate_id, namespace, payload_json, status, created_at) VALUES (?, ?, ?, 'pending', ?)", (candidate_id, namespace, _json(payload), _now()))
        return dict(payload, candidate_id=candidate_id, namespace=namespace, status="pending")


@functools.lru_cache(maxsize=4)
def build_memory_embedding_provider(model_name=None, cache_folder=None):
    """Load the real local semantic model used by opt-in memory retrieval."""
    from .paperstorm_retrieval_v41 import SentenceTransformerProvider

    model = (
        model_name
        or os.getenv("PAPERSTORM_MEMORY_EMBEDDING_MODEL")
        or os.getenv("PAPERSTORM_EMBEDDING_MODEL")
        or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    cache = cache_folder or os.getenv("PAPERSTORM_MODEL_CACHE") or os.getenv("HF_HOME")
    return SentenceTransformerProvider(model_name=model, cache_folder=cache)


def _is_hash_provider(provider):
    name = str(getattr(provider, "name", "") or type(provider).__name__).lower()
    return "hash" in name


def _mmr_select(ranked, top_k, diversity=0.2):
    selected = []
    remaining = list(ranked)
    while remaining and len(selected) < top_k:
        def score(item):
            relevance = item["scores"]["final"]
            redundancy = max((_cosine(item.get("_vector", []), chosen.get("_vector", [])) for chosen in selected), default=0.0)
            return relevance - diversity * max(0.0, redundancy)
        best = max(remaining, key=score)
        remaining.remove(best)
        best = dict(best)
        best.pop("_vector", None)
        selected.append(best)
    return selected


def _bm25_scores(documents: List[List[str]], query_tokens: List[str]):
    average = sum(map(len, documents)) / max(1, len(documents))
    frequency = Counter()
    for document in documents:
        frequency.update(set(document))
    output = []
    for document in documents:
        counts, score = Counter(document), 0.0
        for token in query_tokens:
            count = counts.get(token, 0)
            if count:
                df = frequency[token]
                idf = math.log(1 + (len(documents) - df + 0.5) / (df + 0.5))
                score += idf * (count * 2.5) / (count + 1.5 * (0.25 + 0.75 * len(document) / max(1.0, average)))
        output.append(score)
    return output


def _ranks(scores):
    output = [0] * len(scores)
    for rank, index in enumerate(sorted(range(len(scores)), key=lambda i: scores[i], reverse=True), 1):
        output[index] = rank
    return output


def _cosine(left: Iterable[float], right: Iterable[float]):
    left, right = list(left), list(right)
    numerator = sum(a * b for a, b in zip(left, right))
    denominator = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return numerator / denominator if denominator else 0.0


def _tokenize(text):
    lowered = str(text or "").lower()
    tokens = re.findall(r"[a-z0-9_\-]+", lowered)
    for sequence in re.findall(r"[\u4e00-\u9fff]+", lowered):
        tokens.extend(sequence)
        tokens.extend(sequence[index:index + 2] for index in range(len(sequence) - 1))
    return tokens


def _entities(text, metadata):
    explicit = [str(item).strip() for item in metadata.get("entities", []) if str(item).strip()]
    inferred = re.findall(r"\b[A-Z][A-Z0-9_-]{1,15}\b", str(text or ""))
    return list(dict.fromkeys(explicit + inferred))


def _search_text(record):
    return " ".join([str(record.get("subject", "")), str(record.get("content", "")), str(record.get("canonical_key", "")), " ".join(record.get("entities", []))])


def _episode_dict(row):
    result = dict(row)
    result["metadata"] = _loads(result.pop("metadata_json"))
    return result


def _validate_namespace(namespace):
    value = str(namespace or "").strip()
    if not value or len(value) > 256 or any(character in value for character in "\r\n\0"):
        raise ValueError("valid namespace is required")
    return value


def _parse_datetime(value):
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _normalize(text):
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _blocks_memory_write(message):
    lowered = str(message or "").lower()
    return any(
        marker in lowered
        for marker in ("不要记住", "别记住", "不要保存", "do not remember", "forget this")
    )


def build_memory_candidate_prompt(message, source_message_id="", subject="user"):
    schema = {
        "should_write": False,
        "memory_type": "semantic | episodic | procedural | preference",
        "subject": subject,
        "content": "one durable, context-independent fact",
        "canonical_key": "stable snake_case key",
        "confidence": 0.0,
        "importance": 0.0,
        "source_message_ids": [source_message_id] if source_message_id else [],
        "expires_at": None,
        "metadata": {"reason": ""},
    }
    return (
        "你是长期记忆候选提取器，只输出一个 JSON 对象。\n"
        "只有稳定偏好、用户明确事实、长期决策或可复用流程才应写入。\n"
        "闲聊、临时任务、论文正文、未经证实的推测和敏感凭据不得写入。\n"
        "如果用户拒绝记忆，should_write 必须为 false。内容必须脱离当前轮次仍可理解，"
        "不得把外部论文结论伪装成用户事实。\n"
        "Schema：{0}\n用户消息：{1}"
    ).format(_json(schema), str(message or ""))


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads(value):
    return json.loads(value or "{}")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _elapsed(started):
    return round((time.perf_counter() - started) * 1000, 3)


# Explicit compatibility name for callers that want to switch implementations
# without changing their service construction code.
LongTermMemoryService = LongTermMemoryServiceV56
