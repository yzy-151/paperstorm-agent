import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .paperstorm_context_v56 import ContextEngine
from .paperstorm_memory import PaperStormMemoryStore
from .paperstorm_memory_v56 import LongTermMemoryService


@dataclass
class RuntimeEvent:
    event: str
    run_id: str
    task_id: str
    stage: str = ""
    tool: str = ""
    status: str = ""
    duration_sec: float = 0.0
    input_summary: Optional[Dict] = None
    output_summary: Optional[Dict] = None
    error: str = ""

    def to_trace_record(self):
        record = {
            "event": self.event,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": self.stage,
        }
        if self.tool:
            record["tool"] = self.tool
        if self.status:
            record["status"] = self.status
        if self.duration_sec:
            record["duration_sec"] = self.duration_sec
        if self.input_summary is not None:
            record["input_summary"] = self.input_summary
        if self.output_summary is not None:
            record["output_summary"] = self.output_summary
        if self.error:
            record["error"] = self.error
        return record


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, object] = {}

    def register(self, tool):
        if not getattr(tool, "name", ""):
            raise ValueError("PaperStorm tools must define a non-empty name.")
        self._tools[tool.name] = tool
        return tool

    def __contains__(self, name: str):
        return name in self._tools

    def get(self, name: str):
        if name not in self._tools:
            raise KeyError("Unknown PaperStorm tool: {0}".format(name))
        return self._tools[name]

    def list_schemas(self):
        schemas = []
        for name in sorted(self._tools):
            tool = self._tools[name]
            schemas.append(tool.to_schema() if hasattr(tool, "to_schema") else {"name": name})
        return schemas

    def validate_arguments(self, name: str, arguments: Dict):
        tool = self.get(name)
        schema = getattr(tool, "input_schema", {}) or {}
        required = schema.get("required") or []
        missing = [
            key
            for key in required
            if key not in (arguments or {}) or arguments.get(key) in ("", None)
        ]
        if missing:
            raise ValueError(
                "Tool '{0}' missing required argument(s): {1}".format(
                    name, ", ".join(missing)
                )
            )


class HookManager:
    def __init__(self):
        self._hooks: Dict[str, List[Callable[[RuntimeEvent], None]]] = {}

    def register(self, name: str, callback: Callable[[RuntimeEvent], None]):
        self._hooks.setdefault(name, []).append(callback)
        return callback

    def emit(self, name: str, event: RuntimeEvent):
        for callback in self._hooks.get(name, []):
            callback(event)


class PaperStormRuntimeSession:
    """Runtime shell for traceable PaperStorm tool calls and hooks."""

    def __init__(
        self,
        run_id: str,
        trace_path,
        task_id: Optional[str] = None,
        memory: Optional[PaperStormMemoryStore] = None,
        hooks: Optional[HookManager] = None,
        registry: Optional[ToolRegistry] = None,
        context_engine: Optional[ContextEngine] = None,
        long_term_memory: Optional[LongTermMemoryService] = None,
        memory_namespace: str = "user/local-user",
    ):
        self.run_id = run_id
        self.task_id = task_id or run_id
        self.trace_path = Path(trace_path)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory = memory or PaperStormMemoryStore()
        self.hooks = hooks or HookManager()
        self.registry = registry or ToolRegistry()
        self.context_engine = context_engine or ContextEngine()
        self.long_term_memory = long_term_memory
        self.memory_namespace = memory_namespace
        self.tools = self.registry._tools

    def register_tool(self, tool):
        return self.registry.register(tool)

    def call_tool(self, name: str, arguments: Dict, stage: str = "tool"):
        self.registry.validate_arguments(name, arguments or {})
        tool = self.registry.get(name)
        start = time.time()
        start_event = RuntimeEvent(
            event="tool_start",
            run_id=self.run_id,
            task_id=self.task_id,
            stage=stage,
            tool=name,
            status="running",
            input_summary=_summarize_input(arguments or {}),
        )
        self.hooks.emit("before_tool_call", start_event)
        self.record_event(start_event)
        if self.context_engine.store is not None:
            self.context_engine.store.append_tool_event(start_event.to_trace_record())
        self.memory.append_working(
            "tool_call {0}: {1}".format(
                name, json.dumps(_redact_arguments(arguments or {}), ensure_ascii=False)
            ),
            metadata={"tool": name},
        )
        try:
            result = tool.run(arguments or {})
        except Exception as error:
            error_event = RuntimeEvent(
                event="tool_error",
                run_id=self.run_id,
                task_id=self.task_id,
                stage=stage,
                tool=name,
                status="error",
                duration_sec=round(time.time() - start, 4),
                error=repr(error),
            )
            self.hooks.emit("on_tool_error", error_event)
            self.record_event(error_event)
            if self.context_engine.store is not None:
                self.context_engine.store.append_tool_event(error_event.to_trace_record())
            raise
        end_event = RuntimeEvent(
            event="tool_end",
            run_id=self.run_id,
            task_id=self.task_id,
            stage=stage,
            tool=name,
            status="success",
            duration_sec=round(time.time() - start, 4),
            output_summary=_summarize_output(result),
        )
        self.hooks.emit("after_tool_call", end_event)
        self.record_event(end_event)
        if self.context_engine.store is not None:
            self.context_engine.store.append_tool_event(end_event.to_trace_record())
        return result

    def compress_context(
        self,
        messages,
        expected_keywords=None,
        forbidden_keywords=None,
        max_chars: int = 1200,
    ):
        start = time.time()
        raw_result = self.context_engine.compact(
            list(messages or []),
            expected_constraints=list(expected_keywords or []),
            force=True,
        )
        summary_text = raw_result.get("summary_text") or ""
        expected_hits = [
            item for item in (expected_keywords or []) if item.lower() in summary_text.lower()
        ]
        forbidden_hits = [
            item for item in (forbidden_keywords or []) if item.lower() in summary_text.lower()
        ]
        result = dict(
            raw_result,
            handoff=raw_result.get("summary") or {},
            summary=summary_text,
            constraints={
                "expected_keywords": expected_keywords or [],
                "forbidden_keywords": forbidden_keywords or [],
                "legacy_max_chars": max_chars,
            },
            validation={
                "expected_keyword_hits": expected_hits,
                "forbidden_keyword_hits": forbidden_hits,
                "passed": len(expected_hits) == len(expected_keywords or [])
                and not forbidden_hits
                and raw_result.get("status") != "fallback_original",
            },
        )
        event = RuntimeEvent(
            event="context_compress",
            run_id=self.run_id,
            task_id=self.task_id,
            stage="compression",
            status="success" if result.get("validation", {}).get("passed") else "warning",
            duration_sec=round(time.time() - start, 4),
            input_summary={"message_count": len(list(messages or []))},
            output_summary={
                "summary_chars": len(result.get("summary", "")),
                "before_tokens": result.get("before_tokens", 0),
                "after_tokens": result.get("after_tokens", 0),
                "artifact_count": len(result.get("artifact_refs") or []),
                "compaction_id": result.get("compaction_id", ""),
                "validation": result.get("validation", {}),
            },
        )
        self.hooks.emit("on_context_compress", event)
        self.record_event(event)
        return result

    def remember(self, message: str, source_message_id: str = ""):
        if self.long_term_memory is None:
            result = {"status": "disabled", "reason": "no long-term memory service"}
        else:
            result = self.long_term_memory.ingest_message(
                namespace=self.memory_namespace,
                message=message,
                source_message_id=source_message_id,
            )
        self.record_event(
            "memory_write",
            stage="memory",
            status=result.get("status", "unknown"),
            namespace=self.memory_namespace,
            canonical_key=(result.get("memory") or {}).get("canonical_key", ""),
        )
        return result

    def recall_memory(self, query: str, top_k: int = 5):
        if self.long_term_memory is None:
            result = {
                "status": "disabled",
                "namespace": self.memory_namespace,
                "query": query,
                "results": [],
            }
        else:
            result = self.long_term_memory.search(
                namespace=self.memory_namespace,
                query=query,
                top_k=top_k,
            )
        self.record_event(
            "memory_recall",
            stage="memory",
            status=result.get("status", "unknown"),
            namespace=self.memory_namespace,
            result_count=len(result.get("results") or []),
            latency_ms=result.get("latency_ms", 0.0),
        )
        return result

    def record_event(self, event, **payload):
        if isinstance(event, RuntimeEvent):
            record = event.to_trace_record()
        else:
            record = {
                "event": event,
                "run_id": self.run_id,
                "task_id": self.task_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "stage": payload.pop("stage", ""),
            }
            record.update(payload)
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _redact_arguments(arguments: Dict):
    redacted = {}
    for key, value in arguments.items():
        if "key" in key.lower() or "token" in key.lower() or "secret" in key.lower():
            redacted[key] = "***REDACTED***"
        else:
            redacted[key] = value
    return redacted


def _summarize_output(output):
    if isinstance(output, dict):
        summary = {}
        for key, value in output.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                summary[key] = value
            elif isinstance(value, list):
                summary[key] = {"type": "list", "count": len(value)}
            elif isinstance(value, dict):
                summary[key] = {"type": "object", "keys": sorted(value.keys())[:10]}
        return summary
    return {"type": type(output).__name__}


def _summarize_input(arguments: Dict):
    return {
        key: _summarize_value(value)
        for key, value in _redact_arguments(arguments or {}).items()
    }


def _summarize_value(value):
    if isinstance(value, str):
        return value if len(value) <= 160 else value[:157] + "..."
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return {"type": "list", "count": len(value)}
    if isinstance(value, dict):
        return {"type": "object", "keys": sorted(value.keys())[:10]}
    return {"type": type(value).__name__}
