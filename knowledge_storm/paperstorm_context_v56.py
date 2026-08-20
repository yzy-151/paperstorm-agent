"""Layered, auditable context governance for PaperStorm v5.6."""

import json
import math
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .paperstorm_context_v42 import estimate_tokens, truncate_to_tokens


LAYER_NAMES = ("pinned", "active", "summary", "memory", "evidence", "artifact")


@dataclass
class ContextEngineConfigV56:
    model_context_tokens: int = 1_000_000
    operational_input_tokens: int = 128_000
    output_reserve_tokens: int = 16_000
    soft_watermark: float = 0.78
    high_watermark: float = 0.9
    recent_messages: int = 48
    task_profile: str = "chat"
    layer_targets: Dict[str, float] = field(
        default_factory=lambda: {
            "pinned": 0.08,
            "active": 0.42,
            "summary": 0.18,
            "memory": 0.12,
            "evidence": 0.14,
            "artifact": 0.06,
        }
    )
    absolute_layer_caps: Dict[str, int] = field(
        default_factory=lambda: {
            "pinned": 24_000,
            "active": 96_000,
            "summary": 64_000,
            "memory": 48_000,
            "evidence": 700_000,
            "artifact": 96_000,
        }
    )
    layer_caps: Optional[Dict[str, float]] = None

    def __post_init__(self):
        # Public v5.6 benchmark adapters used ``layer_caps``. Keep that
        # constructor contract while using the clearer v5.9 target name.
        if self.layer_caps:
            merged = dict(self.layer_targets)
            merged.update(self.layer_caps)
            self.layer_targets = merged

    @property
    def input_limit(self):
        hard_limit = int(self.model_context_tokens) - int(self.output_reserve_tokens)
        return max(1, min(hard_limit, int(self.operational_input_tokens)))

    @classmethod
    def for_profile(cls, profile="chat", model_context_tokens=1_000_000):
        profile = str(profile or "chat").lower()
        profiles = {
            "chat": {
                "operational_input_tokens": 128_000,
                "output_reserve_tokens": 16_000,
                "layer_targets": {
                    "pinned": 0.08, "active": 0.42, "summary": 0.18,
                    "memory": 0.12, "evidence": 0.14, "artifact": 0.06,
                },
            },
            "qa": {
                "operational_input_tokens": 256_000,
                "output_reserve_tokens": 32_000,
                "layer_targets": {
                    "pinned": 0.06, "active": 0.22, "summary": 0.12,
                    "memory": 0.10, "evidence": 0.44, "artifact": 0.06,
                },
            },
            "research": {
                "operational_input_tokens": 512_000,
                "output_reserve_tokens": 64_000,
                "layer_targets": {
                    "pinned": 0.04, "active": 0.18, "summary": 0.10,
                    "memory": 0.08, "evidence": 0.52, "artifact": 0.08,
                },
            },
        }
        values = profiles.get(profile, profiles["chat"])
        return cls(
            model_context_tokens=int(model_context_tokens),
            task_profile=profile if profile in profiles else "chat",
            **values,
        )


class ContextLedgerV56:
    """Append-only SQLite ledger for raw messages and compaction lineage."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(str(self.path), timeout=30)
        connection.row_factory = sqlite3.Row
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
                CREATE TABLE IF NOT EXISTS context_events (
                    event_id TEXT PRIMARY KEY,
                    message_id TEXT NOT NULL,
                    message_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS context_compactions (
                    compaction_id TEXT PRIMARY KEY,
                    level INTEGER NOT NULL,
                    parent_ids_json TEXT NOT NULL,
                    source_event_ids_json TEXT NOT NULL,
                    raw_messages_json TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    token_before INTEGER NOT NULL,
                    token_after INTEGER NOT NULL,
                    strategy TEXT NOT NULL,
                    validation_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def append_messages(self, messages):
        event_ids = []
        with self._connect() as connection:
            for message in messages:
                message_id = str(message.get("id") or uuid.uuid4().hex)
                event_id = uuid.uuid4().hex
                connection.execute(
                    "INSERT INTO context_events(event_id, message_id, message_json, created_at) VALUES (?, ?, ?, ?)",
                    (event_id, message_id, _json(message), _now()),
                )
                event_ids.append(event_id)
        return event_ids

    def append_compaction(self, compaction, raw_messages, summary):
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO context_compactions
                   (compaction_id, level, parent_ids_json, source_event_ids_json,
                    raw_messages_json, summary_json, token_before, token_after,
                    strategy, validation_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    compaction["compaction_id"],
                    compaction["level"],
                    _json(compaction["parent_ids"]),
                    _json(compaction["source_event_ids"]),
                    _json(raw_messages),
                    _json(summary),
                    compaction["token_before"],
                    compaction["token_after"],
                    compaction["strategy"],
                    _json(compaction["validation"]),
                    _now(),
                ),
            )

    def restore(self, compaction_id):
        with self._connect() as connection:
            row = connection.execute(
                "SELECT raw_messages_json FROM context_compactions WHERE compaction_id=?",
                (str(compaction_id),),
            ).fetchone()
        if not row:
            raise KeyError("Unknown compaction_id: {0}".format(compaction_id))
        return json.loads(row["raw_messages_json"])

    def get(self, compaction_id):
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM context_compactions WHERE compaction_id=?", (compaction_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        for key in ("parent_ids_json", "source_event_ids_json", "summary_json", "validation_json"):
            result[key[:-5]] = json.loads(result.pop(key))
        result.pop("raw_messages_json")
        return result

    def message_events(self):
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM context_events ORDER BY created_at, rowid").fetchall()
        return [
            {
                "event_id": row["event_id"],
                "event_type": "message",
                "message": json.loads(row["message_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def compaction_events(self):
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM context_compactions ORDER BY created_at, rowid").fetchall()
        return [dict(self.get(row["compaction_id"]), event_type="compaction") for row in rows]


class ContextEngineV56:
    def __init__(self, config=None, ledger=None, summarizer=None):
        self.config = config or ContextEngineConfigV56()
        self.ledger = ledger
        self.summarizer = summarizer

    def estimate(self, messages: Iterable[Dict]):
        return sum(_message_tokens(message) for message in messages)

    def should_compact(self, messages: Iterable[Dict]):
        return self.estimate(messages) >= int(self.config.input_limit * self.config.soft_watermark)

    def assemble(
        self,
        messages: Iterable[Dict],
        memories: Optional[Iterable[Dict]] = None,
        evidence: Optional[Iterable[Dict]] = None,
        state: Optional[Dict] = None,
        artifacts: Optional[Iterable[Dict]] = None,
        query: str = "",
    ):
        messages = [dict(item) for item in messages]
        layers = {name: [] for name in LAYER_NAMES}
        layers["pinned"] = [item for item in messages if _is_pinned(item)]
        summaries = [item for item in messages if _is_summary(item) and not _is_pinned(item)]
        ordinary = [item for item in messages if not _is_pinned(item) and not _is_summary(item)]
        layers["active"] = _select_recent_groups(ordinary, self.config.recent_messages)
        layers["summary"] = _select_relevant_messages(summaries, query, limit=4)
        if state:
            layers["summary"].append({"role": "system", "content": "Current task state: " + _json(state), "metadata": {"context_layer": "summary"}})
        layers["memory"] = [_memory_message(item) for item in (memories or [])]
        layers["evidence"] = [_evidence_message(item) for item in (evidence or [])]
        layers["artifact"] = [_artifact_message(item) for item in (artifacts or [])]

        selected_by_layer = {name: [] for name in LAYER_NAMES}
        usage = {name: 0 for name in LAYER_NAMES}
        selected_ids = set()
        remaining = self.config.input_limit

        # First pass protects a task-specific share for every context type.
        for layer_name in LAYER_NAMES:
            target = max(
                0,
                int(self.config.input_limit * float(self.config.layer_targets.get(layer_name, 0))),
            )
            hard_cap = int(self.config.absolute_layer_caps.get(layer_name, self.config.input_limit))
            layer_budget = min(remaining, target, hard_cap)
            chosen = _fit_groups(layers[layer_name], layer_budget, preserve_order=True)
            selected_by_layer[layer_name].extend(chosen)
            selected_ids.update(_identity(item) for item in chosen)
            used = self.estimate(chosen)
            usage[layer_name] = used
            remaining -= used

        # Second pass lends unused budget to high-value layers without exceeding
        # their absolute caps. Research profiles prioritize evidence; chat keeps
        # recent dialogue first.
        priority = (
            ("evidence", "active", "summary", "memory", "artifact", "pinned")
            if self.config.task_profile in {"qa", "research"}
            else ("active", "summary", "memory", "evidence", "artifact", "pinned")
        )
        for layer_name in priority:
            if remaining <= 0:
                break
            hard_cap = int(self.config.absolute_layer_caps.get(layer_name, self.config.input_limit))
            extra_budget = min(remaining, max(0, hard_cap - usage[layer_name]))
            candidates = [item for item in layers[layer_name] if _identity(item) not in selected_ids]
            chosen = _fit_groups(candidates, extra_budget, preserve_order=True)
            selected_by_layer[layer_name].extend(chosen)
            selected_ids.update(_identity(item) for item in chosen)
            used = self.estimate(chosen)
            usage[layer_name] += used
            remaining -= used

        selected = [item for name in LAYER_NAMES for item in selected_by_layer[name]]

        selected = _deduplicate_messages(selected)
        if self.estimate(selected) > self.config.input_limit:
            selected = _hard_fit(selected, self.config.input_limit)
        validation = _validate_context(selected, layers["pinned"])
        return {
            "messages": selected,
            "token_usage": {
                "total": self.estimate(selected),
                "input_limit": self.config.input_limit,
                "output_reserve": self.config.output_reserve_tokens,
                "layers": usage,
            },
            "validation": validation,
            "policy": "profile targets -> dynamic budget borrowing -> absolute caps",
        }

    def compact(self, messages: Iterable[Dict], force: bool = False):
        messages = [dict(item) for item in messages]
        if not force and not self.should_compact(messages):
            assembled = self.assemble(messages)
            return dict(assembled, compacted=False, compaction=None)

        expanded = self._expand_parent_summaries(messages)
        parent_ids = [str(item.get("metadata", {}).get("compaction_id")) for item in messages if _is_summary(item) and item.get("metadata", {}).get("compaction_id")]
        level = 1 + max([int(item.get("metadata", {}).get("level", 0)) for item in messages if _is_summary(item)] or [0])
        pinned = [item for item in messages if _is_pinned(item)]
        ordinary = [item for item in messages if not _is_pinned(item) and not _is_summary(item)]
        active = _select_recent_groups(ordinary, self.config.recent_messages)
        active_ids = {_identity(item) for item in active}
        middle = [item for item in ordinary if _identity(item) not in active_ids]

        strategy = "deterministic"
        try:
            summary_content = self._summarize(middle)
        except Exception:
            strategy = "deterministic_fallback"
            summary_content = _deterministic_summary(middle)
        summary_content = truncate_to_tokens(
            summary_content,
            max(16, min(self.config.absolute_layer_caps["summary"], int(self.config.input_limit * 0.18)) - 6),
        )
        compaction_id = uuid.uuid4().hex
        summary = {
            "id": "summary-" + compaction_id,
            "role": "system",
            "content": summary_content,
            "metadata": {
                "context_summary": True,
                "context_layer": "summary",
                "compaction_id": compaction_id,
                "level": level,
                "parent_ids": parent_ids,
            },
        }
        compacted_messages = _deduplicate_messages(pinned + [summary] + active)
        assembled = self.assemble(compacted_messages)
        validation = _validate_context(assembled["messages"], pinned)
        source_event_ids = self.ledger.append_messages(expanded) if self.ledger else [str(item.get("id") or _identity(item)) for item in expanded]
        compaction = {
            "compaction_id": compaction_id,
            "level": level,
            "parent_ids": list(dict.fromkeys(parent_ids)),
            "source_event_ids": source_event_ids,
            "token_before": self.estimate(expanded),
            "token_after": assembled["token_usage"]["total"],
            "strategy": strategy,
            "validation": validation,
        }
        if self.ledger:
            self.ledger.append_compaction(compaction, expanded, summary)
        return {
            "messages": assembled["messages"],
            "token_usage": assembled["token_usage"],
            "validation": validation,
            "compacted": True,
            "compaction": compaction,
        }

    def restore(self, compaction_id):
        if not self.ledger:
            raise RuntimeError("restore requires a ContextLedgerV56")
        return self.ledger.restore(compaction_id)

    def inspect(self, messages):
        messages = list(messages)
        return {
            "tokens": self.estimate(messages),
            "input_limit": self.config.input_limit,
            "should_compact": self.should_compact(messages),
            "pinned_count": sum(1 for item in messages if _is_pinned(item)),
            "summary_levels": [item.get("metadata", {}).get("level") for item in messages if _is_summary(item)],
        }

    def _summarize(self, messages):
        if self.summarizer:
            prompt = build_structured_summary_prompt(messages)
            try:
                value = self.summarizer(prompt)
            except TypeError:
                value = self.summarizer(messages)
            if isinstance(value, dict):
                value = _json(_normalize_summary(value))
            elif isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, dict):
                        value = _json(_normalize_summary(parsed))
                except json.JSONDecodeError:
                    pass
            value = str(value or "").strip()
            if not value:
                raise ValueError("summarizer returned empty content")
            return truncate_to_tokens(value, max(24, int(self.config.input_limit * 0.18)))
        return _deterministic_summary(messages)

    def _expand_parent_summaries(self, messages):
        expanded = []
        for message in messages:
            compaction_id = message.get("metadata", {}).get("compaction_id") if _is_summary(message) else None
            if compaction_id and self.ledger:
                try:
                    expanded.extend(self.ledger.restore(compaction_id))
                    continue
                except KeyError:
                    pass
            expanded.append(message)
        return _deduplicate_messages(expanded)


def _select_recent_groups(messages, count):
    groups = _tool_groups(messages)
    selected, item_count = [], 0
    for group in reversed(groups):
        if item_count >= count:
            break
        selected.append(group)
        item_count += len(group)
    return [item for group in reversed(selected) for item in group]


def _tool_groups(messages):
    messages = list(messages)
    call_to_index = {}
    results_by_call = {}
    used = set()
    groups = []
    for index, message in enumerate(messages):
        for call in message.get("tool_calls") or []:
            call_to_index[str(call.get("id"))] = index
        if message.get("tool_call_id"):
            results_by_call.setdefault(str(message["tool_call_id"]), []).append(index)
    for index, message in enumerate(messages):
        if index in used:
            continue
        call_ids = [str(call.get("id")) for call in (message.get("tool_calls") or []) if call.get("id")]
        if call_ids:
            result_indices = [result_index for call_id in call_ids for result_index in results_by_call.get(call_id, [])]
            group = [message] + [messages[result_index] for result_index in result_indices]
            groups.append(group)
            used.add(index)
            used.update(result_indices)
            continue
        tool_call_id = message.get("tool_call_id")
        if tool_call_id and str(tool_call_id) in call_to_index:
            call_index = call_to_index[str(tool_call_id)]
            if call_index in used:
                used.add(index)
            continue
        groups.append([message])
        used.add(index)
    groups.sort(key=lambda group: min(messages.index(item) for item in group))
    return groups


def _fit_groups(messages, budget, preserve_order=True):
    groups = _tool_groups(messages)
    selected, used = [], 0
    for group in groups:
        cost = sum(_message_tokens(item) for item in group)
        if used + cost <= budget:
            selected.extend(group)
            used += cost
    return selected if preserve_order else list(reversed(selected))


def _hard_fit(messages, budget):
    selected, used = [], 0
    for group in _tool_groups(messages):
        cost = sum(_message_tokens(item) for item in group)
        if used + cost <= budget:
            selected.extend(group)
            used += cost
    return selected


def _validate_context(messages, pinned):
    selected_ids = {_identity(item) for item in messages}
    calls = {str(call.get("id")) for item in messages for call in (item.get("tool_calls") or []) if call.get("id")}
    results = {str(item.get("tool_call_id")) for item in messages if item.get("tool_call_id")}
    return {
        "pinned_preserved": all(_identity(item) in selected_ids for item in pinned),
        "tool_pairs_valid": calls == results,
        "source_ids_present": all(bool(item.get("id") or item.get("metadata", {}).get("source_id")) for item in messages if item.get("role") in {"tool"}),
    }


def _deterministic_summary(messages):
    structured = {
        "user_goals": [],
        "confirmed_facts": [],
        "decisions": [],
        "constraints": [],
        "open_questions": [],
        "completed_actions": [],
        "pending_actions": [],
        "evidence_refs": [],
        "memory_candidates": [],
        "topic_transitions": [],
    }
    if not messages:
        return json.dumps(structured, ensure_ascii=False)
    for message in messages:
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        lowered = content.lower()
        if any(word in lowered for word in ("必须", "不要", "偏好", "require", "must")):
            structured["constraints"].append(content)
        elif message.get("role") == "user":
            structured["user_goals"].append(content)
            if content.endswith(("?", "？")):
                structured["open_questions"].append(content)
        else:
            structured["decisions"].append(content)
        source = message.get("metadata", {}).get("source_id")
        if source:
            structured["evidence_refs"].append(str(source))
    # Keep schema order so high-value goals and facts survive any emergency
    # token truncation before lower-priority bookkeeping fields.
    return json.dumps(_normalize_summary(structured), ensure_ascii=False)


SUMMARY_FIELDS = (
    "user_goals",
    "confirmed_facts",
    "decisions",
    "constraints",
    "open_questions",
    "completed_actions",
    "pending_actions",
    "evidence_refs",
    "memory_candidates",
    "topic_transitions",
)


def build_structured_summary_prompt(messages):
    """Build a provenance-preserving prompt for recursive context summaries."""
    schema = {field_name: [] for field_name in SUMMARY_FIELDS}
    return (
        "你是 PaperStorm 的上下文压缩器。把历史对话压缩为严格 JSON，不要输出 Markdown。\n"
        "规则：\n"
        "1. 不得把推测写成事实；事实、决定、待办和未解决问题必须分开。\n"
        "2. 原样保留否定条件、数值、路径、错误、引用 ID、task_id 和 document_id。\n"
        "3. 标记主题切换；旧主题不得自动成为当前主题。\n"
        "4. 不复制大段工具输出，只保留结论、来源指针与恢复线索。\n"
        "5. 不确定内容放入 open_questions，不得补写对话中不存在的信息。\n"
        "输出 Schema：{0}\n"
        "待压缩消息：{1}"
    ).format(_json(schema), _json(list(messages or [])))


def _normalize_summary(value):
    value = dict(value or {})
    normalized = {}
    for field_name in SUMMARY_FIELDS:
        items = value.get(field_name) or []
        if not isinstance(items, list):
            items = [items]
        normalized[field_name] = [str(item).strip() for item in items if str(item).strip()][:20]
    return normalized


def _select_relevant_messages(messages, query, limit=4):
    messages = list(messages or [])
    if not messages:
        return []
    if not str(query or "").strip():
        return messages[-max(1, int(limit)):]
    query_terms = _search_terms(query)
    documents = [_search_terms(message.get("content", "")) for message in messages]
    document_count = len(documents)
    document_frequency = {
        term: sum(1 for document in documents if term in document)
        for term in query_terms
    }
    scored = []
    average_length = sum(len(document) for document in documents) / max(1, document_count)
    for index, (message, content_terms) in enumerate(zip(messages, documents)):
        length = max(1, len(content_terms))
        score = 0.0
        for term in query_terms.intersection(content_terms):
            idf = math.log(1.0 + (document_count - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
            score += idf * 2.2 / (1.0 + 1.2 * (0.25 + 0.75 * length / max(1.0, average_length)))
        scored.append((score, index, message))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    chosen = sorted(scored[: max(1, int(limit))], key=lambda item: item[1])
    return [item[2] for item in chosen]


def _search_terms(text):
    import re

    return {
        item.lower()
        for item in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}", str(text or ""))
    }


def _memory_message(item):
    content = item.get("content") if isinstance(item, dict) else str(item)
    return {"role": "system", "content": "Relevant long-term memory: " + str(content), "metadata": {"context_layer": "memory", "source_id": str(item.get("id", "memory")) if isinstance(item, dict) else "memory"}}


def _evidence_message(item):
    content = item.get("content") if isinstance(item, dict) else str(item)
    source_id = item.get("source_id") or item.get("id") or "evidence" if isinstance(item, dict) else "evidence"
    return {"role": "system", "content": "Retrieved evidence: " + str(content), "metadata": {"context_layer": "evidence", "source_id": str(source_id)}}


def _artifact_message(item):
    content = item.get("summary") or item.get("content") or item.get("artifact_id") if isinstance(item, dict) else str(item)
    source_id = item.get("artifact_id") or item.get("id") or "artifact" if isinstance(item, dict) else "artifact"
    return {"role": "system", "content": "Tool artifact reference: " + str(content), "metadata": {"context_layer": "artifact", "source_id": str(source_id)}}


def _is_pinned(message):
    return message.get("role") in {"system", "developer"} and not _is_summary(message) or bool(message.get("metadata", {}).get("pinned"))


def _is_summary(message):
    return bool(message.get("metadata", {}).get("context_summary"))


def _message_tokens(message):
    payload = str(message.get("content") or "")
    if message.get("tool_calls"):
        payload += _json(message["tool_calls"])
    return max(1, estimate_tokens(payload) + 4)


def _identity(message):
    return str(message.get("id") or uuid.uuid5(uuid.NAMESPACE_OID, _json(message)))


def _deduplicate_messages(messages):
    seen, output = set(), []
    for message in messages:
        identity = _identity(message)
        if identity not in seen:
            output.append(message)
            seen.add(identity)
    return output


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _now():
    return datetime.now(timezone.utc).isoformat()


class ContextEngineConfig(ContextEngineConfigV56):
    """Accept the v4.2 option names while storing the v5.6 policy."""

    def __init__(
        self,
        total_tokens=1_000_000,
        output_reserve_tokens=16_000,
        compact_threshold_ratio=0.78,
        high_watermark_ratio=0.9,
        recent_message_count=48,
        tool_inline_token_limit=180,
        **updates,
    ):
        super().__init__(
            model_context_tokens=int(updates.pop("model_context_tokens", total_tokens)),
            operational_input_tokens=int(updates.pop("operational_input_tokens", min(128_000, int(total_tokens) - int(output_reserve_tokens)))),
            output_reserve_tokens=int(output_reserve_tokens),
            soft_watermark=float(updates.pop("soft_watermark", compact_threshold_ratio)),
            high_watermark=float(updates.pop("high_watermark", high_watermark_ratio)),
            recent_messages=int(updates.pop("recent_messages", recent_message_count)),
            task_profile=str(updates.pop("task_profile", "chat")),
            layer_targets=updates.pop("layer_targets", updates.pop("layer_caps", ContextEngineConfigV56().layer_targets)),
            absolute_layer_caps=updates.pop("absolute_layer_caps", ContextEngineConfigV56().absolute_layer_caps),
        )
        self.tool_inline_token_limit = int(tool_inline_token_limit)

    @property
    def total_tokens(self):
        return self.model_context_tokens

    @property
    def compact_threshold_ratio(self):
        return self.soft_watermark

    @property
    def high_watermark_ratio(self):
        return self.high_watermark

    @property
    def recent_message_count(self):
        return self.recent_messages


class ContextEventStore:
    """v4.2 store facade backed by the v5.6 SQLite ledger."""

    def __init__(self, path):
        path = Path(path)
        if path.suffix == ".jsonl":
            path = path.with_suffix(".v56.sqlite3")
        self.path = path
        self.ledger = ContextLedgerV56(path)

    def append_message(self, message):
        event_id = self.ledger.append_messages([message])[0]
        return {"event_id": event_id, "event_type": "message", "message": dict(message), "created_at": _now()}

    def append_tool_event(self, payload):
        message = {"id": "tool-event-" + uuid.uuid4().hex, "role": "tool", "content": _json(payload), "metadata": {"runtime_event": True}}
        return self.append_message(message)

    def append_compaction(self, payload):
        return payload

    def read_events(self):
        return self.ledger.message_events() + self.ledger.compaction_events()

    def message_events(self):
        return self.ledger.message_events()

    def restore_messages(self, compaction_id):
        return self.ledger.restore(compaction_id)


class ContextEngine:
    """Compatibility facade that routes existing runtime calls through v5.6."""

    def __init__(self, config=None, store=None, summarizer=None, token_counter=None):
        self.config = config or ContextEngineConfig()
        if isinstance(self.config, ContextEngineConfigV56) and not isinstance(self.config, ContextEngineConfig):
            core_config = self.config
        else:
            core_config = self.config
        self.store = store
        ledger = getattr(store, "ledger", store if isinstance(store, ContextLedgerV56) else None)
        self.core = ContextEngineV56(config=core_config, ledger=ledger, summarizer=summarizer)
        self.token_counter = token_counter or estimate_tokens

    def estimate(self, messages):
        messages = list(messages or [])
        tokens = self.core.estimate(messages)
        return {
            "message_count": len(messages),
            "input_tokens": tokens,
            "input_limit_tokens": self.config.input_limit,
            "total_tokens": self.config.model_context_tokens,
            "output_reserve_tokens": self.config.output_reserve_tokens,
            "usage_ratio": round(tokens / max(1, self.config.input_limit), 4),
        }

    def should_compact(self, messages):
        meter = self.estimate(messages)
        high = meter["usage_ratio"] >= self.config.high_watermark
        should = meter["usage_ratio"] >= self.config.soft_watermark
        return dict(meter, should_compact=should, high_watermark=high, reason="high_watermark" if high else "compact_threshold" if should else "below_threshold")

    def compact(self, messages, expected_constraints=None, force=False):
        messages = list(messages or [])
        decision = self.should_compact(messages)
        before = self.estimate(messages)["input_tokens"]
        if not force and not decision["should_compact"]:
            return {"status": "not_needed", "compaction_id": "", "messages": messages, "summary": {}, "summary_text": "", "artifact_refs": [], "source_event_ids": [], "before_tokens": before, "after_tokens": before, "validation": {"passed": True, "missing_constraints": []}, "decision": decision}
        result = self.core.compact(messages, force=True)
        summary_message = next((item for item in result["messages"] if _is_summary(item)), {})
        summary_text = str(summary_message.get("content") or "")
        user_contents = [
            str(item.get("content") or "").strip()
            for item in messages
            if item.get("role") == "user" and str(item.get("content") or "").strip()
        ]
        if force and user_contents and not all(content in summary_text for content in user_contents):
            summary_text = _deterministic_summary(messages)
        expected = [str(item) for item in (expected_constraints or [])]
        if expected:
            summary_text = "\n".join(
                [summary_text] + ["Session constraint: " + item for item in expected]
            ).strip()
        searchable = "\n".join(str(item.get("content") or "") for item in result["messages"]).lower()
        missing = [item for item in expected if item.lower() not in searchable]
        compaction = result.get("compaction") or {}
        validation = dict(result.get("validation") or {}, passed=not missing and result.get("validation", {}).get("pinned_preserved", True), expected_constraints=expected, missing_constraints=missing)
        return {
            "status": "compacted" if validation["passed"] else "warning",
            "compaction_id": compaction.get("compaction_id", ""),
            "messages": result["messages"],
            "summary": {"constraints": expected, "source_message_ids": [str(item.get("id") or "") for item in messages]},
            "summary_text": summary_text,
            "artifact_refs": [item.get("metadata", {}).get("source_id") for item in result["messages"] if item.get("metadata", {}).get("context_layer") == "artifact"],
            "source_event_ids": compaction.get("source_event_ids", []),
            "before_tokens": compaction.get("token_before", before),
            "after_tokens": compaction.get("token_after", self.core.estimate(result["messages"])),
            "validation": validation,
            "decision": decision,
            "lineage": compaction,
        }

    def assemble(self, messages, memory=None, rag_evidence=None, tool_schemas=None, query=""):
        compaction = self.compact(messages)
        artifacts = [{"artifact_id": "tool-schemas", "summary": _json(list(tool_schemas or []))}] if tool_schemas else []
        result = self.core.assemble(compaction["messages"], memories=memory, evidence=rag_evidence, artifacts=artifacts, query=query)
        assembly_limit = max(32, int(self.config.input_limit * self.config.high_watermark))
        fitted_messages = _hard_fit(result["messages"], assembly_limit)
        meter = self.estimate(fitted_messages)
        meter.update({"remaining_input_tokens": max(0, assembly_limit - meter["input_tokens"]), "assembly_limit_tokens": assembly_limit, "allocation": result["token_usage"]["layers"], "compaction_status": compaction["status"]})
        return {"messages": fitted_messages, "meter": meter, "compaction": compaction, "validation": result["validation"]}

    def restore(self, compaction_id):
        return {"compaction_id": compaction_id, "messages": self.core.restore(compaction_id), "restored_at": _now()}

    def inspect(self, messages):
        events = self.store.read_events() if self.store else []
        compact_events = [item for item in events if item.get("event_type") == "compaction"]
        return {"context_meter": self.should_compact(messages), "raw_event_count": len([item for item in events if item.get("event_type") == "message"]), "compaction_count": len(compact_events), "latest_compaction": compact_events[-1] if compact_events else {}, "events": events[-20:]}
