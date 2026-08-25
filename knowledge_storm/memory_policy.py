import hashlib
import json
import math
import re
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional

from pydantic import BaseModel, Field


MemoryType = Literal["semantic", "episodic", "procedural", "preference"]
MemoryStatus = Literal["active", "superseded", "deleted"]


class MemoryRecord(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    memory_type: MemoryType
    namespace: str
    subject: str
    content: str
    canonical_key: str
    source_message_ids: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    importance: float = Field(default=0.7, ge=0.0, le=1.0)
    created_at: str = Field(default_factory=lambda: _now())
    updated_at: str = Field(default_factory=lambda: _now())
    valid_from: str = Field(default_factory=lambda: _now())
    valid_to: Optional[str] = None
    expires_at: Optional[str] = None
    status: MemoryStatus = "active"
    supersedes_id: Optional[str] = None
    metadata: Dict = Field(default_factory=dict)


class MemoryCandidate(BaseModel):
    memory_type: MemoryType
    subject: str
    content: str
    canonical_key: str
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    importance: float = Field(default=0.7, ge=0.0, le=1.0)
    source_message_ids: List[str] = Field(default_factory=list)
    expires_at: Optional[str] = None
    metadata: Dict = Field(default_factory=dict)


class LongTermMemoryService:
    """Auditable cross-session memory, isolated from thread context and document RAG."""

    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.event_path = self.root_dir / "memory_events.jsonl"
        self.pending_path = self.root_dir / "memory_candidates.jsonl"

    def ingest_message(
        self,
        namespace: str,
        message: str,
        source_message_id: str = "",
        subject: str = "user",
    ):
        namespace = _validate_namespace(namespace)
        candidate = MemoryWritePolicy.extract(
            message=message,
            source_message_id=source_message_id,
            subject=subject,
        )
        if candidate is None:
            return {"status": "skipped", "reason": "no durable memory signal"}
        payload = _model_dump(candidate)
        if candidate.confidence < 0.85:
            queued = dict(
                payload,
                candidate_id=uuid.uuid4().hex,
                namespace=namespace,
                queued_at=_now(),
            )
            _append_jsonl(self.pending_path, queued)
            return {"status": "queued", "candidate": queued}
        memory = self.upsert(namespace=namespace, **payload)
        return {"status": "persisted", "memory": memory}

    def upsert(
        self,
        namespace: str,
        memory_type: MemoryType,
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
        if not content:
            raise ValueError("memory content is required")
        if not canonical_key:
            raise ValueError("canonical_key is required")

        active = self.list_memories(namespace)
        same_key = [item for item in active if item["canonical_key"] == canonical_key]
        normalized = _normalize(content)
        duplicate = next(
            (item for item in same_key if _normalize(item["content"]) == normalized),
            None,
        )
        if duplicate:
            self._append_event(
                "memory_deduplicated",
                namespace,
                {"memory_id": duplicate["id"], "canonical_key": canonical_key},
            )
            return dict(duplicate, deduplicated=True)

        superseded = same_key[0] if same_key else None
        if superseded:
            self._append_event(
                "memory_status_changed",
                namespace,
                {
                    "memory_id": superseded["id"],
                    "status": "superseded",
                    "reason": "conflicting_active_value",
                    "updated_at": _now(),
                },
            )
        record = MemoryRecord(
            memory_type=memory_type,
            namespace=namespace,
            subject=str(subject or "unknown"),
            content=content,
            canonical_key=canonical_key,
            source_message_ids=[str(item) for item in (source_message_ids or []) if item],
            confidence=confidence,
            importance=importance,
            valid_from=valid_from or _now(),
            valid_to=valid_to,
            expires_at=expires_at,
            supersedes_id=superseded["id"] if superseded else None,
            metadata=dict(metadata or {}),
        )
        payload = _model_dump(record)
        self._append_event("memory_upserted", namespace, {"memory": payload})
        return dict(payload, deduplicated=False)

    def edit(self, namespace: str, memory_id: str, content: str, **updates):
        namespace = _validate_namespace(namespace)
        current = self.get_memory(namespace, memory_id, include_inactive=True)
        if not current:
            raise KeyError("Unknown memory_id: {0}".format(memory_id))
        return self.upsert(
            namespace=namespace,
            memory_type=updates.get("memory_type", current["memory_type"]),
            subject=updates.get("subject", current["subject"]),
            content=content,
            canonical_key=updates.get("canonical_key", current["canonical_key"]),
            source_message_ids=updates.get(
                "source_message_ids", current.get("source_message_ids") or []
            ),
            confidence=updates.get("confidence", current["confidence"]),
            importance=updates.get("importance", current["importance"]),
            expires_at=updates.get("expires_at", current.get("expires_at")),
            metadata=dict(current.get("metadata") or {}, edited_from=memory_id),
        )

    def delete(self, namespace: str, memory_id: str, reason: str = "user_request"):
        namespace = _validate_namespace(namespace)
        current = self.get_memory(namespace, memory_id, include_inactive=True)
        if not current:
            raise KeyError("Unknown memory_id: {0}".format(memory_id))
        self._append_event(
            "memory_status_changed",
            namespace,
            {
                "memory_id": memory_id,
                "status": "deleted",
                "reason": reason,
                "updated_at": _now(),
            },
        )
        return dict(current, status="deleted", deletion_reason=reason)

    def list_memories(
        self,
        namespace: str,
        include_inactive: bool = False,
        memory_type: Optional[str] = None,
    ):
        namespace = _validate_namespace(namespace)
        state = self._state()
        records = [
            item
            for item in state["memories"].values()
            if item.get("namespace") == namespace
        ]
        if memory_type:
            records = [item for item in records if item.get("memory_type") == memory_type]
        if not include_inactive:
            records = [item for item in records if self._is_active(item)]
        records.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        return records

    def get_memory(self, namespace: str, memory_id: str, include_inactive: bool = False):
        records = self.list_memories(namespace, include_inactive=include_inactive)
        return next((item for item in records if item["id"] == memory_id), None)

    def search(
        self,
        namespace: str,
        query: str,
        top_k: int = 5,
        memory_types: Optional[List[str]] = None,
    ):
        started = time.perf_counter()
        namespace = _validate_namespace(namespace)
        if not self.is_enabled(namespace):
            return {
                "status": "disabled",
                "namespace": namespace,
                "query": query,
                "results": [],
                "latency_ms": _elapsed_ms(started),
            }
        records = self.list_memories(namespace)
        if memory_types:
            allowed = set(memory_types)
            records = [item for item in records if item["memory_type"] in allowed]
        ranked = _hybrid_rank(records, query)
        return {
            "status": "ok",
            "namespace": namespace,
            "query": query,
            "top_k": max(1, int(top_k or 5)),
            "results": ranked[: max(1, int(top_k or 5))],
            "candidate_count": len(records),
            "latency_ms": _elapsed_ms(started),
            "retrieval": "namespace_filter -> lexical+dense -> RRF -> importance+recency",
        }

    def set_enabled(self, namespace: str, enabled: bool):
        namespace = _validate_namespace(namespace)
        self._append_event(
            "memory_setting_changed",
            namespace,
            {"enabled": bool(enabled)},
        )
        return {"namespace": namespace, "enabled": bool(enabled)}

    def is_enabled(self, namespace: str):
        namespace = _validate_namespace(namespace)
        return self._state()["settings"].get(namespace, {}).get("enabled", True)

    def export_namespace(self, namespace: str):
        namespace = _validate_namespace(namespace)
        return {
            "namespace": namespace,
            "enabled": self.is_enabled(namespace),
            "exported_at": _now(),
            "memories": self.list_memories(namespace, include_inactive=True),
            "events": self.audit_events(namespace),
        }

    def audit_events(self, namespace: Optional[str] = None):
        events = _read_jsonl(self.event_path)
        if namespace:
            namespace = _validate_namespace(namespace)
            events = [item for item in events if item.get("namespace") == namespace]
        return events

    def consolidate_pending(self, minimum_confidence: float = 0.65):
        candidates = _read_jsonl(self.pending_path)
        completed = {
            (item.get("payload") or {}).get("candidate_id")
            for item in self.audit_events()
            if item.get("event") == "memory_candidate_consolidated"
        }
        persisted = []
        skipped = []
        for item in candidates:
            candidate_id = item.get("candidate_id")
            if candidate_id in completed:
                continue
            if float(item.get("confidence", 0)) < minimum_confidence:
                skipped.append(candidate_id)
                self._append_event(
                    "memory_candidate_consolidated",
                    item["namespace"],
                    {"candidate_id": candidate_id, "status": "skipped"},
                )
                continue
            payload = {key: value for key, value in item.items() if key in _candidate_fields()}
            persisted.append(self.upsert(namespace=item["namespace"], **payload))
            self._append_event(
                "memory_candidate_consolidated",
                item["namespace"],
                {"candidate_id": candidate_id, "status": "persisted"},
            )
        return {"persisted": persisted, "skipped": skipped, "processed": len(persisted) + len(skipped)}

    def _append_event(self, event_type: str, namespace: str, payload: Dict):
        event = {
            "event_id": uuid.uuid4().hex,
            "event": event_type,
            "namespace": namespace,
            "timestamp": _now(),
            "payload": payload,
        }
        _append_jsonl(self.event_path, event)
        return event

    def _state(self):
        memories = {}
        settings = {}
        for event in _read_jsonl(self.event_path):
            payload = event.get("payload") or {}
            if event.get("event") == "memory_upserted":
                record = dict(payload.get("memory") or {})
                if record.get("id"):
                    memories[record["id"]] = record
            elif event.get("event") == "memory_status_changed":
                record = memories.get(payload.get("memory_id"))
                if record:
                    record["status"] = payload.get("status", record.get("status"))
                    record["updated_at"] = payload.get("updated_at", _now())
                    if payload.get("reason"):
                        record.setdefault("metadata", {})["status_reason"] = payload["reason"]
            elif event.get("event") == "memory_setting_changed":
                settings[event.get("namespace", "")] = {
                    "enabled": bool(payload.get("enabled", True))
                }
        return {"memories": memories, "settings": settings}

    @staticmethod
    def _is_active(record: Dict):
        if record.get("status") != "active":
            return False
        now = datetime.now(timezone.utc)
        for field in ["valid_to", "expires_at"]:
            value = record.get(field)
            if value and _parse_datetime(value) <= now:
                return False
        valid_from = record.get("valid_from")
        return not valid_from or _parse_datetime(valid_from) <= now


class MemoryWritePolicy:
    """Deterministic baseline; production can replace extraction with structured LLM output."""

    @staticmethod
    def extract(message: str, source_message_id: str = "", subject: str = "user"):
        text = str(message or "").strip()
        lowered = text.lower()
        if not text or re.search(r"不要记住|别记住|无需记住|forget this", lowered):
            return None
        tentative = bool(re.search(r"可能要记住|也许要记住|might remember", lowered))
        explicit = bool(re.search(r"请记住|记住：|记住:|remember that", lowered))
        preference = bool(
            re.search(r"偏好|喜欢|习惯|以后.{0,12}(?:回答|输出)|(?:回答|输出).{0,8}(?:中文|英文)", text)
        )
        procedural = bool(re.search(r"操作规范|规则是|流程是|以后.{0,20}(?:先|必须|不要)", text))
        stable_fact = bool(re.search(r"\bPIM\b.{0,12}(?:指|表示|means)|我的.{0,12}是|项目.{0,12}(?:是|使用)", text, re.I))
        if not any([explicit, preference, procedural, stable_fact]):
            return None

        content = re.sub(r"^(?:请)?记住[：:]?\s*", "", text, flags=re.I).strip()
        memory_type = "preference" if preference else "procedural" if procedural else "semantic"
        canonical_key = _canonical_key(content, memory_type)
        confidence = 0.7 if tentative else 0.96 if explicit else 0.88
        return MemoryCandidate(
            memory_type=memory_type,
            subject=subject,
            content=content,
            canonical_key=canonical_key,
            confidence=confidence,
            importance=0.85 if preference or procedural else 0.75,
            source_message_ids=[source_message_id] if source_message_id else [],
                metadata={"extractor": "deterministic_memory_policy", "explicit": explicit},
        )


def _hybrid_rank(records: List[Dict], query: str):
    if not records:
        return []
    query_tokens = _tokenize(query)
    corpus_tokens = [_tokenize(_search_text(item)) for item in records]
    lexical = _bm25_scores(corpus_tokens, query_tokens)
    query_vector = _hash_embedding(query_tokens)
    dense = [_cosine(query_vector, _hash_embedding(tokens)) for tokens in corpus_tokens]
    lexical_rank = _ranks(lexical)
    dense_rank = _ranks(dense)
    now = datetime.now(timezone.utc)
    ranked = []
    for index, record in enumerate(records):
        if lexical[index] <= 0 and dense[index] < 0.15:
            continue
        rrf = (1.0 / (60 + lexical_rank[index])) + (1.0 / (60 + dense_rank[index]))
        updated = _parse_datetime(record.get("updated_at") or record.get("created_at") or _now())
        age_days = max(0.0, (now - updated).total_seconds() / 86400.0)
        recency = 1.0 / (1.0 + age_days / 30.0)
        importance = float(record.get("importance", 0.7))
        final = rrf + 0.02 * importance + 0.01 * recency
        item = dict(record)
        item["scores"] = {
            "lexical": round(lexical[index], 6),
            "dense": round(dense[index], 6),
            "rrf": round(rrf, 6),
            "importance": round(importance, 4),
            "recency": round(recency, 4),
            "final": round(final, 6),
        }
        ranked.append(item)
    ranked.sort(key=lambda item: item["scores"]["final"], reverse=True)
    return ranked


def _bm25_scores(documents: List[List[str]], query_tokens: List[str]):
    if not documents:
        return []
    average_length = sum(len(item) for item in documents) / max(1, len(documents))
    document_frequency = Counter()
    for document in documents:
        document_frequency.update(set(document))
    scores = []
    for document in documents:
        counts = Counter(document)
        score = 0.0
        for token in query_tokens:
            frequency = counts.get(token, 0)
            if not frequency:
                continue
            df = document_frequency.get(token, 0)
            idf = math.log(1 + (len(documents) - df + 0.5) / (df + 0.5))
            denominator = frequency + 1.5 * (
                1 - 0.75 + 0.75 * len(document) / max(1.0, average_length)
            )
            score += idf * (frequency * 2.5) / denominator
        scores.append(score)
    return scores


def _hash_embedding(tokens: Iterable[str], dimensions: int = 64):
    vector = [0.0] * dimensions
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _cosine(left: List[float], right: List[float]):
    return sum(a * b for a, b in zip(left, right))


def _ranks(scores: List[float]):
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    ranks = [0] * len(scores)
    for rank, index in enumerate(order, start=1):
        ranks[index] = rank
    return ranks


def _tokenize(text: str):
    lowered = str(text or "").lower()
    tokens = re.findall(r"[a-z0-9_\-]+", lowered)
    for sequence in re.findall(r"[\u4e00-\u9fff]+", lowered):
        tokens.extend(list(sequence))
        tokens.extend(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return tokens


def _search_text(record: Dict):
    return " ".join(
        [
            str(record.get("subject") or ""),
            str(record.get("content") or ""),
            str(record.get("canonical_key") or ""),
            json.dumps(record.get("metadata") or {}, ensure_ascii=False),
        ]
    )


def _canonical_key(content: str, memory_type: str):
    lowered = content.lower()
    if re.search(r"(?:回答|输出).{0,8}中文|中文.{0,8}(?:回答|输出)", content):
        return "response_language"
    if re.search(r"(?:回答|输出).{0,8}英文|英文.{0,8}(?:回答|输出)", content):
        return "response_language"
    if "pim" in lowered and re.search(r"指|表示|means", lowered):
        return "term:pim"
    if re.search(r"我的.{0,8}(?:名字|姓名).{0,4}是", content):
        return "user:name"
    digest = hashlib.sha256(_normalize(content).encode("utf-8")).hexdigest()[:16]
    return "{0}:{1}".format(memory_type, digest)


def _validate_namespace(namespace: str):
    value = str(namespace or "").strip().lower()
    if not re.fullmatch(r"(?:user|team|org)/[a-z0-9][a-z0-9._-]{0,127}", value):
        raise ValueError("namespace must look like user/<id>, team/<id>, or org/<id>")
    return value


def _parse_datetime(value: str):
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalize(text: str):
    return re.sub(r"\s+", "", str(text or "").strip().lower())


def _candidate_fields():
    return {
        "memory_type",
        "subject",
        "content",
        "canonical_key",
        "confidence",
        "importance",
        "source_message_ids",
        "expires_at",
        "metadata",
    }


def _append_jsonl(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _model_dump(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _elapsed_ms(started: float):
    return round((time.perf_counter() - started) * 1000, 4)


def _now():
    return datetime.now(timezone.utc).isoformat()
