import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from .paperstorm_memory import PaperStormMemoryStore


class PaperStormRuntimeSession:
    """Lightweight runtime shell for traceable PaperStorm tool calls."""

    def __init__(
        self,
        run_id: str,
        trace_path,
        memory: Optional[PaperStormMemoryStore] = None,
    ):
        self.run_id = run_id
        self.trace_path = Path(trace_path)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory = memory or PaperStormMemoryStore()
        self.tools: Dict[str, object] = {}

    def register_tool(self, tool):
        self.tools[tool.name] = tool
        return tool

    def call_tool(self, name: str, arguments: Dict):
        if name not in self.tools:
            raise KeyError("Unknown PaperStorm tool: {0}".format(name))
        start = time.time()
        self.record_event(
            "tool_start",
            tool=name,
            arguments=_redact_arguments(arguments or {}),
        )
        self.memory.append_working(
            "tool_call {0}: {1}".format(
                name, json.dumps(_redact_arguments(arguments or {}), ensure_ascii=False)
            ),
            metadata={"tool": name},
        )
        try:
            result = self.tools[name].run(arguments or {})
        except Exception as error:
            self.record_event(
                "tool_error",
                tool=name,
                status="error",
                duration_sec=round(time.time() - start, 4),
                error=repr(error),
            )
            raise
        self.record_event(
            "tool_end",
            tool=name,
            status="success",
            duration_sec=round(time.time() - start, 4),
            output_summary=_summarize_output(result),
        )
        return result

    def record_event(self, event: str, **payload):
        record = {
            "event": event,
            "run_id": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
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
