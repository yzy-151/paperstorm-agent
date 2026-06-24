import hashlib
import json
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional


SUMMARY_FIELDS = [
    "goal",
    "constraints",
    "completed",
    "in_progress",
    "decisions",
    "entities",
    "sources",
    "errors",
    "todos",
    "source_message_ids",
]


@dataclass
class ContextEngineConfig:
    total_tokens: int = 4096
    output_reserve_tokens: int = 768
    compact_threshold_ratio: float = 0.72
    high_watermark_ratio: float = 0.9
    recent_message_count: int = 6
    tool_inline_token_limit: int = 180

    @property
    def input_limit(self):
        return max(64, self.total_tokens - self.output_reserve_tokens)


class ContextEventStore:
    """Append-only raw context events. Compaction never rewrites source messages."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append_message(self, message: Dict):
        return self._append("message", {"message": dict(message)})

    def append_tool_event(self, payload: Dict):
        return self._append("tool", {"payload": dict(payload)})

    def append_compaction(self, payload: Dict):
        payload = dict(payload)
        payload.setdefault("compaction_id", uuid.uuid4().hex)
        return self._append("compaction", payload)

    def read_events(self):
        if not self.path.exists():
            return []
        events = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))
        return events

    def message_events(self):
        return [item for item in self.read_events() if item.get("event_type") == "message"]

    def restore_messages(self, compaction_id: str):
        events = self.read_events()
        compaction = next(
            (
                item
                for item in reversed(events)
                if item.get("event_type") == "compaction"
                and item.get("compaction_id") == compaction_id
            ),
            None,
        )
        if compaction is None:
            raise KeyError("Unknown compaction_id: {0}".format(compaction_id))
        source_ids = set(compaction.get("source_event_ids") or [])
        return [
            dict(item["message"])
            for item in events
            if item.get("event_type") == "message" and item.get("event_id") in source_ids
        ]

    def _append(self, event_type: str, payload: Dict):
        events = self.read_events()
        record = {
            "event_id": uuid.uuid4().hex,
            "sequence": len(events) + 1,
            "event_type": event_type,
            "created_at": _now(),
        }
        record.update(payload)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record


class ContextEngine:
    """Token-driven, recoverable context assembler and compactor."""

    def __init__(
        self,
        config: Optional[ContextEngineConfig] = None,
        store: Optional[ContextEventStore] = None,
        summarizer: Optional[Callable[[List[Dict]], Dict]] = None,
        token_counter: Optional[Callable[[str], int]] = None,
    ):
        self.config = config or ContextEngineConfig()
        self.store = store
        self.summarizer = summarizer
        self.token_counter = token_counter or estimate_tokens

    def estimate(self, messages: Iterable[Dict]):
        messages = list(messages or [])
        content_tokens = sum(
            self.token_counter(str(item.get("content") or "")) + 4 for item in messages
        )
        return {
            "message_count": len(messages),
            "input_tokens": content_tokens,
            "input_limit_tokens": self.config.input_limit,
            "total_tokens": self.config.total_tokens,
            "output_reserve_tokens": self.config.output_reserve_tokens,
            "usage_ratio": round(content_tokens / max(1, self.config.input_limit), 4),
        }

    def should_compact(self, messages: Iterable[Dict]):
        meter = self.estimate(messages)
        ratio = meter["usage_ratio"]
        high_watermark = ratio >= self.config.high_watermark_ratio
        should = ratio >= self.config.compact_threshold_ratio
        return dict(
            meter,
            should_compact=should,
            high_watermark=high_watermark,
            reason=(
                "high_watermark"
                if high_watermark
                else "compact_threshold"
                if should
                else "below_threshold"
            ),
        )

    def compact(
        self,
        messages: Iterable[Dict],
        expected_constraints: Optional[List[str]] = None,
        force: bool = False,
    ):
        messages = [dict(item) for item in (messages or [])]
        before = self.estimate(messages)
        decision = self.should_compact(messages)
        if not force and not decision["should_compact"]:
            return {
                "status": "not_needed",
                "compaction_id": "",
                "messages": messages,
                "summary": _empty_summary(messages),
                "summary_text": "",
                "artifact_refs": [],
                "before_tokens": before["input_tokens"],
                "after_tokens": before["input_tokens"],
                "validation": {"passed": True, "missing_constraints": []},
                "decision": decision,
            }
        try:
            preserved, middle = self._partition(messages)
            middle_view, artifact_refs = self._artifactize_tools(middle)
            summary = (
                self.summarizer(middle_view)
                if self.summarizer is not None
                else _structured_summary(messages, middle_view, artifact_refs)
            )
            summary = _normalize_summary(summary, messages)
            expected_constraints = expected_constraints or []
            for constraint in expected_constraints:
                rendered = "Session constraint: {0}".format(constraint)
                if rendered not in summary["constraints"]:
                    summary["constraints"].append(rendered)
            summary_text = _summary_text(summary)
            summary_message = {
                "id": "context-summary-{0}".format(uuid.uuid4().hex[:12]),
                "role": "system",
                "content": summary_text,
                "metadata": {"derived": True, "kind": "context_handoff"},
            }
            output_messages = self._merge_compacted(preserved, summary_message)
            searchable = "\n".join(
                [summary_text]
                + [str(item.get("content") or "") for item in output_messages]
            ).lower()
            missing = [
                item for item in expected_constraints if str(item).lower() not in searchable
            ]
            validation = {
                "passed": not missing,
                "expected_constraints": expected_constraints,
                "missing_constraints": missing,
            }
            compaction_id = uuid.uuid4().hex
            source_event_ids = self._source_event_ids(messages)
            result = {
                "status": "compacted" if validation["passed"] else "warning",
                "compaction_id": compaction_id,
                "messages": output_messages,
                "summary": summary,
                "summary_text": summary_text,
                "artifact_refs": artifact_refs,
                "source_event_ids": source_event_ids,
                "before_tokens": before["input_tokens"],
                "after_tokens": self.estimate(output_messages)["input_tokens"],
                "validation": validation,
                "decision": decision,
            }
            if self.store is not None:
                self.store.append_compaction(
                    {
                        key: value
                        for key, value in result.items()
                        if key not in {"messages"}
                    }
                )
            return result
        except Exception as error:
            return {
                "status": "fallback_original",
                "compaction_id": "",
                "messages": messages,
                "summary": _empty_summary(messages),
                "summary_text": "",
                "artifact_refs": [],
                "before_tokens": before["input_tokens"],
                "after_tokens": before["input_tokens"],
                "validation": {"passed": False, "missing_constraints": []},
                "decision": decision,
                "error": str(error),
            }

    def assemble(
        self,
        messages: Iterable[Dict],
        memory: Optional[Iterable[Dict]] = None,
        rag_evidence: Optional[Iterable[Dict]] = None,
        tool_schemas: Optional[Iterable[Dict]] = None,
    ):
        raw_messages = [dict(item) for item in (messages or [])]
        compaction = self.compact(raw_messages)
        core = compaction["messages"]
        sections = {
            "core": core,
            "memory": [dict(item) for item in (memory or [])],
            "rag": [dict(item) for item in (rag_evidence or [])],
            "tools": (
                [
                    {
                        "role": "system",
                        "content": "Available tool schemas: {0}".format(
                            json.dumps(list(tool_schemas or []), ensure_ascii=False)
                        ),
                    }
                ]
                if tool_schemas
                else []
            ),
        }
        accepted = []
        allocation = {}
        assembly_limit = max(
            32, int(self.config.input_limit * self.config.high_watermark_ratio)
        )
        remaining = assembly_limit
        for section_name in ["core", "memory", "rag", "tools"]:
            used = 0
            section_messages = sections[section_name]
            for index, message in enumerate(section_messages):
                message_budget = remaining
                if section_name == "core":
                    remaining_items = len(section_messages) - index
                    message_budget = max(5, remaining // max(1, remaining_items))
                fitted = self._fit_message(message, message_budget)
                if fitted is None:
                    continue
                cost = self.estimate([fitted])["input_tokens"]
                accepted.append(fitted)
                used += cost
                remaining -= cost
                if remaining <= 4:
                    break
            allocation[section_name] = used
            if remaining <= 4:
                break
        meter = self.estimate(accepted)
        meter.update(
            {
                "remaining_input_tokens": max(0, self.config.input_limit - meter["input_tokens"]),
                "assembly_limit_tokens": assembly_limit,
                "allocation": allocation,
                "compaction_status": compaction["status"],
            }
        )
        return {"messages": accepted, "meter": meter, "compaction": compaction}

    def restore(self, compaction_id: str):
        if self.store is None:
            raise RuntimeError("restore requires a ContextEventStore")
        return {
            "compaction_id": compaction_id,
            "messages": self.store.restore_messages(compaction_id),
            "restored_at": _now(),
        }

    def inspect(self, messages: Iterable[Dict]):
        messages = list(messages or [])
        events = self.store.read_events() if self.store else []
        compact_events = [item for item in events if item.get("event_type") == "compaction"]
        latest = compact_events[-1] if compact_events else {}
        return {
            "context_meter": self.should_compact(messages),
            "raw_event_count": len([item for item in events if item.get("event_type") == "message"]),
            "compaction_count": len(compact_events),
            "latest_compaction": latest,
            "events": events[-20:],
        }

    def _partition(self, messages):
        system = [item for item in messages if item.get("role") == "system"]
        first_goal = next((item for item in messages if item.get("role") == "user"), None)
        recent = messages[-max(1, self.config.recent_message_count) :]
        preserved_ids = {id(item) for item in system + recent}
        if first_goal is not None:
            preserved_ids.add(id(first_goal))
        preserved = [item for item in messages if id(item) in preserved_ids]
        middle = [item for item in messages if id(item) not in preserved_ids]
        return preserved, middle

    def _artifactize_tools(self, messages):
        output = []
        refs = []
        for message in messages:
            content = str(message.get("content") or "")
            if message.get("role") == "tool" and self.token_counter(content) > self.config.tool_inline_token_limit:
                message_id = str(message.get("id") or uuid.uuid4().hex)
                digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
                uri = "context://message/{0}#{1}".format(message_id, digest[:12])
                refs.append(
                    {
                        "message_id": message_id,
                        "uri": uri,
                        "sha256": digest,
                        "original_tokens": self.token_counter(content),
                    }
                )
                replacement = dict(message)
                replacement["content"] = "[tool artifact: {0}]".format(uri)
                replacement["metadata"] = dict(
                    replacement.get("metadata") or {}, artifact_uri=uri, original_sha256=digest
                )
                output.append(replacement)
            else:
                output.append(dict(message))
        return output, refs

    def _merge_compacted(self, preserved, summary_message):
        system = [item for item in preserved if item.get("role") == "system"]
        non_system = [item for item in preserved if item.get("role") != "system"]
        return system + [summary_message] + non_system

    def _source_event_ids(self, messages):
        if self.store is not None:
            return [item["event_id"] for item in self.store.message_events()]
        return [str(item.get("id") or index) for index, item in enumerate(messages)]

    def _fit_message(self, message, remaining):
        if remaining <= 4:
            return None
        message = dict(message)
        cost = self.estimate([message])["input_tokens"]
        if cost <= remaining:
            return message
        content_budget = max(0, remaining - 4)
        if content_budget <= 0:
            return None
        message["content"] = truncate_to_tokens(
            str(message.get("content") or ""), content_budget, self.token_counter
        )
        return message if message["content"] else None


def estimate_tokens(text: str):
    text = str(text or "")
    units = re.findall(r"[A-Za-z0-9_./:-]+|[\u3400-\u9fff]|[^\s]", text)
    return len(units)


def truncate_to_tokens(text: str, limit: int, counter=estimate_tokens):
    if counter(text) <= limit:
        return text
    units = re.findall(r"[A-Za-z0-9_./:-]+|[\u3400-\u9fff]|[^\s]", str(text or ""))
    return " ".join(units[:limit]).strip()


def _structured_summary(all_messages, middle_messages, artifact_refs):
    all_lines = [str(item.get("content") or "").strip() for item in all_messages]
    middle_lines = [str(item.get("content") or "").strip() for item in middle_messages]
    goal_message = next((item for item in all_messages if item.get("role") == "user"), {})
    joined = "\n".join(all_lines)
    return {
        "goal": str(goal_message.get("content") or ""),
        "constraints": _select_lines(all_lines, ["必须", "不得", "中文", "引用", "排除", "保留", "must"]),
        "completed": _select_lines(middle_lines, ["完成", "已", "done"]),
        "in_progress": _select_lines(all_lines, ["正在", "进行中", "下一步", "running"]),
        "decisions": _select_lines(all_lines, ["决定", "选择", "采用", "decision"]),
        "entities": _entities(joined),
        "sources": [item["uri"] for item in artifact_refs] + _select_lines(all_lines, ["http://", "https://"]),
        "errors": _select_lines(all_lines, ["错误", "失败", "error", "exception"]),
        "todos": _select_lines(all_lines, ["待办", "下一步", "todo"]),
        "source_message_ids": [
            str(item.get("id") or "") for item in all_messages if item.get("id")
        ],
    }


def _normalize_summary(summary, messages):
    if not isinstance(summary, dict):
        raise ValueError("summarizer must return a dictionary")
    normalized = {}
    for field in SUMMARY_FIELDS:
        default = "" if field == "goal" else []
        normalized[field] = summary.get(field, default)
    if not normalized["goal"]:
        normalized["goal"] = next(
            (str(item.get("content") or "") for item in messages if item.get("role") == "user"),
            "",
        )
    if not normalized["source_message_ids"]:
        normalized["source_message_ids"] = [
            str(item.get("id") or "") for item in messages if item.get("id")
        ]
    return normalized


def _empty_summary(messages):
    return _normalize_summary({}, list(messages or []))


def _summary_text(summary):
    labels = {
        "goal": "Goal",
        "constraints": "Constraints",
        "completed": "Completed",
        "in_progress": "In progress",
        "decisions": "Decisions",
        "entities": "Entities",
        "sources": "Sources",
        "errors": "Errors",
        "todos": "Todos",
    }
    lines = ["Context handoff summary (derived; raw events remain restorable):"]
    for field, label in labels.items():
        value = summary.get(field)
        if value:
            rendered = value if isinstance(value, str) else " | ".join(str(item) for item in value)
            lines.append("{0}: {1}".format(label, rendered))
    return "\n".join(lines)


def _select_lines(lines, keywords):
    selected = []
    for line in lines:
        lowered = line.lower()
        if line and any(keyword.lower() in lowered for keyword in keywords) and line not in selected:
            selected.append(line)
    return selected[:8]


def _entities(text):
    candidates = re.findall(r"\b[A-Z][A-Z0-9-]{1,}\b", text)
    for term in ["无源互调", "神经网络", "Cross-Encoder"]:
        if term in text:
            candidates.append(term)
    output = []
    for item in candidates:
        if item not in output:
            output.append(item)
    return output[:20]


def _now():
    return datetime.now(timezone.utc).isoformat()
