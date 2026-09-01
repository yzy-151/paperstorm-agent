import json
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


_SECRET_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]+")
_BEARER_PATTERN = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|secret[_-]?key|token)"
    r"\s*[:=]\s*['\"]?[^\s,'\"}&]+"
)
_SECRET_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret_key",
    "access_token",
    "refresh_token",
}


def classify_runtime_error(error) -> str:
    message = str(error or "").lower()
    name = type(error).__name__.lower()
    if isinstance(error, TimeoutError) or "timeout" in name or "timed out" in message:
        return "timeout"
    if any(marker in message for marker in ("401", "403", "api key", "authentication", "unauthorized")):
        return "authentication"
    if any(marker in message for marker in ("arxiv", "retrieval", "retriever", "search api")):
        return "retrieval_error"
    if isinstance(error, (ValueError, json.JSONDecodeError)) and any(
        marker in message for marker in ("json", "parse", "invalid response")
    ):
        return "parse_error"
    if any(
        marker in message
        for marker in (
            "connection error",
            "connection refused",
            "failed to establish",
            "provider unavailable",
            "network is unreachable",
        )
    ):
        return "provider_unavailable"
    return "runtime_error"


def sanitize_trace_value(value, key="", max_string_length=512):
    lowered = str(key).lower()
    if lowered in _SECRET_KEYS or "password" in lowered or "secret" in lowered:
        return "***REDACTED***"
    if isinstance(value, dict):
        return {
            str(item_key): sanitize_trace_value(item, str(item_key), max_string_length)
            for item_key, item in list(value.items())[:50]
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_trace_value(item, "", max_string_length) for item in value[:50]]
    if isinstance(value, str):
        redacted = _SECRET_PATTERN.sub("sk-***REDACTED***", value)
        redacted = _BEARER_PATTERN.sub("Bearer ***REDACTED***", redacted)
        redacted = _SECRET_ASSIGNMENT_PATTERN.sub(
            lambda match: "{0}=***REDACTED***".format(match.group(1)), redacted
        )
        if len(redacted) > max_string_length:
            suffix = "...[truncated]"
            return redacted[: max_string_length - len(suffix)] + suffix
        return redacted
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return sanitize_trace_value(str(value), key, max_string_length)


class PaperStormTraceRecorder:
    def __init__(
        self,
        article_dir: str,
        enabled: bool = True,
        observation_parent=None,
    ):
        self.article_dir = article_dir
        self.enabled = enabled
        self.trace_path = str(Path(article_dir) / "paperstorm_trace.jsonl")
        self.summary_path = str(Path(article_dir) / "run_summary.json")
        self.started_at = time.time()
        self.events = []
        self.current_stage = None
        self._stage_started_at = None
        self._write_lock = threading.RLock()
        self.observation_parent = observation_parent
        self._observation_stages = {}
        if self.enabled:
            Path(article_dir).mkdir(parents=True, exist_ok=True)
            Path(self.trace_path).write_text("", encoding="utf-8")

    @staticmethod
    def _utc_now():
        return datetime.now(timezone.utc).isoformat()

    def emit(self, event: str, **payload):
        if not self.enabled:
            return None
        record = {
            "ts": self._utc_now(),
            "event": event,
            **sanitize_trace_value(payload),
        }
        with self._write_lock:
            self.events.append(record)
            with open(self.trace_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._export_observation_event(record)
        return record

    def generation(self, name: str, model: str = "", input=None, metadata=None):
        """Create a Generation under the currently active business stage."""
        parent = self._current_observation_parent()
        if parent is None:
            return None
        return parent.generation(
            name=name,
            model=model,
            input=input or {},
            metadata=metadata or {},
        )

    def _current_observation_parent(self):
        if self.current_stage:
            candidates = [
                handle
                for key, handle in self._observation_stages.items()
                if key[0] == self.current_stage
            ]
            if candidates:
                return candidates[-1]
        return self.observation_parent

    def _export_observation_event(self, record):
        if self.observation_parent is None:
            return
        try:
            event = record.get("event")
            if event == "stage_start":
                key = _observation_stage_key(record)
                previous = self._observation_stages.pop(key, None)
                if previous is not None:
                    previous.end(
                        output={"status": "superseded"},
                        metadata={"reason": "duplicate stage start"},
                    )
                handle = self.observation_parent.span(
                    record.get("stage") or "stage",
                    input=record.get("input") or {},
                    metadata=_stage_metadata(record),
                    as_type="chain",
                )
                handle.__enter__()
                self._observation_stages[key] = handle
            elif event in {"stage_end", "stage_error"}:
                key = _observation_stage_key(record)
                handle = self._observation_stages.pop(key, None)
                if handle is None:
                    matching = [
                        item
                        for item in self._observation_stages
                        if item[0] == record.get("stage")
                    ]
                    if matching:
                        handle = self._observation_stages.pop(matching[-1])
                if handle is not None:
                    handle.end(
                        output=record.get("output_summary")
                        or {"status": "failed" if event == "stage_error" else "completed"},
                        error=record.get("error_message") if event == "stage_error" else None,
                        metadata=_stage_metadata(record),
                    )
        except Exception:
            # Observability must not affect research execution. Remote failures are
            # already counted by PaperStormObservability; bridge shape errors are
            # intentionally fail-open as well.
            return

    def start_stage(self, stage: str, operation: str, input=None, **telemetry):
        if self.current_stage:
            self.end_stage(
                output_summary={"transition": "advanced to {0}".format(stage)}
            )
        self.current_stage = stage
        self._stage_started_at = time.perf_counter()
        return self.emit(
            "stage_start",
            stage=stage,
            operation=operation,
            input=input or {},
            **telemetry,
        )

    def progress_stage(self, operation: str = "", output_summary=None, **telemetry):
        if not self.current_stage:
            return None
        return self.emit(
            "stage_progress",
            stage=self.current_stage,
            operation=operation,
            output_summary=output_summary or {},
            **telemetry,
        )

    def end_stage(self, output_summary=None, **telemetry):
        if not self.current_stage:
            return None
        stage = self.current_stage
        duration_ms = max(
            0,
            round((time.perf_counter() - (self._stage_started_at or time.perf_counter())) * 1000, 2),
        )
        self.current_stage = None
        self._stage_started_at = None
        return self.emit(
            "stage_end",
            stage=stage,
            output_summary=output_summary or {},
            duration_ms=duration_ms,
            **telemetry,
        )

    def fail_current_stage(self, error, stage=None, **telemetry):
        failed_stage = stage or self.current_stage or "request"
        duration_ms = max(
            0,
            round((time.perf_counter() - (self._stage_started_at or time.perf_counter())) * 1000, 2),
        )
        self.current_stage = None
        self._stage_started_at = None
        return self.emit(
            "stage_error",
            stage=failed_stage,
            duration_ms=duration_ms,
            error_type=classify_runtime_error(error),
            error_message=str(error),
            **telemetry,
        )

    def write_summary(self, success: bool, artifacts=None, error=None, extra=None):
        if not self.enabled:
            return
        for key, handle in list(self._observation_stages.items()):
            handle.end(
                output={"status": "closed_at_run_end"},
                metadata={"stage": key[0]},
            )
            self._observation_stages.pop(key, None)
        artifacts = artifacts or []
        retrieval_starts = [
            event for event in self.events if event["event"] == "retrieval_start"
        ]
        summary = {
            "success": success,
            "duration_sec": round(time.time() - self.started_at, 4),
            "event_count": len(self.events),
            "retrieval_queries": sum(
                len(event.get("queries") or []) for event in retrieval_starts
            ),
            "retrieval_success": sum(
                event["event"] == "retrieval_end" for event in self.events
            ),
            "retrieval_failed": sum(
                event["event"] == "retrieval_error" for event in self.events
            ),
            "artifacts": artifacts,
        }
        if error:
            summary["error"] = sanitize_trace_value(error)
        if extra:
            summary.update(sanitize_trace_value(extra))
        Path(self.summary_path).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _observation_stage_key(record):
    return (
        str(record.get("stage") or "stage"),
        str(record.get("invocation_id") or ""),
    )


def _stage_metadata(record):
    ignored = {"ts", "event", "stage", "input", "output_summary", "error_message"}
    return {key: value for key, value in record.items() if key not in ignored}


class PaperStormStageCallback:
    def __init__(self, trace_recorder: PaperStormTraceRecorder, topic: str):
        self.trace_recorder = trace_recorder
        self.topic = topic
        self.dialogue_turn_count = 0

    def on_identify_perspective_start(self, **kwargs):
        self.trace_recorder.start_stage(
            "persona",
            "生成多视角调研角色",
            input={"topic": self.topic},
        )

    def on_identify_perspective_end(self, perspectives, **kwargs):
        self.trace_recorder.end_stage(
            output_summary={"perspectives": list(perspectives or [])}
        )
        self.trace_recorder.emit(
            "artifact_ready", stage="persona", artifact_name="personas.json"
        )

    def on_information_gathering_start(self, **kwargs):
        self.trace_recorder.start_stage(
            "dialogue",
            "WikiWriter 与领域专家开始多轮对话",
            input={"topic": self.topic},
        )

    def on_dialogue_turn_end(self, dlg_turn, **kwargs):
        self.dialogue_turn_count += 1
        question = _first_attribute(
            dlg_turn, "user_utterance", "question", "user_message"
        )
        answer = _first_attribute(
            dlg_turn, "agent_utterance", "answer", "agent_message"
        )
        self.trace_recorder.progress_stage(
            operation="完成第 {0} 轮研究对话".format(self.dialogue_turn_count),
            output_summary={
                "turn": self.dialogue_turn_count,
                "question": question,
                "answer_preview": answer,
            },
        )

    def on_information_gathering_end(self, **kwargs):
        self.trace_recorder.end_stage(
            output_summary={"dialogue_turns": self.dialogue_turn_count}
        )
        self.trace_recorder.emit(
            "artifact_ready", stage="dialogue", artifact_name="conversation_log.json"
        )
        self.trace_recorder.start_stage(
            "evidence",
            "汇总对话证据并建立信息表",
            input={"dialogue_turns": self.dialogue_turn_count},
        )
        self.trace_recorder.end_stage(
            output_summary={"status": "information_table_ready"}
        )
        self.trace_recorder.emit(
            "artifact_ready", stage="evidence", artifact_name="evidence_index.json"
        )

    def on_information_organization_start(self, **kwargs):
        self.trace_recorder.start_stage(
            "outline",
            "生成并细化文章大纲",
            input={"topic": self.topic},
        )

    def on_direct_outline_generation_end(self, outline, **kwargs):
        self.trace_recorder.progress_stage(
            operation="直接大纲已生成，正在结合证据细化",
            output_summary={"draft_outline": outline},
        )

    def on_outline_refinement_end(self, outline, **kwargs):
        self.trace_recorder.end_stage(output_summary={"outline": outline})
        self.trace_recorder.emit(
            "artifact_ready", stage="outline", artifact_name="storm_gen_outline.txt"
        )


class TracedRetrievalModel:
    def __init__(self, rm, trace_recorder: PaperStormTraceRecorder, retriever_name: str):
        self.rm = rm
        self.trace_recorder = trace_recorder
        self.retriever_name = retriever_name

    def __call__(self, query_or_queries, exclude_urls=None):
        queries = (
            list(query_or_queries)
            if isinstance(query_or_queries, (list, tuple))
            else [query_or_queries]
        )
        invocation_id = uuid.uuid4().hex
        started = time.perf_counter()
        self.trace_recorder.emit(
            "stage_start",
            stage="query",
            operation="规范化并提交检索查询",
            invocation_id=invocation_id,
            input={"queries": queries},
        )
        self.trace_recorder.emit(
            "stage_end",
            stage="query",
            operation="检索查询规划完成",
            invocation_id=invocation_id,
            output_summary={"queries": queries, "query_count": len(queries)},
            duration_ms=0,
        )
        self.trace_recorder.emit(
            "artifact_ready",
            stage="query",
            artifact_name="queries.json",
            output_summary={"query_count": len(queries)},
        )
        self.trace_recorder.emit(
            "stage_start",
            stage="retrieval",
            operation="使用 {0} 检索论文".format(self.retriever_name),
            invocation_id=invocation_id,
            input={"queries": queries, "retriever": self.retriever_name},
        )
        self.trace_recorder.emit(
            "tool_start",
            tool_name=self.retriever_name,
            tool_type="retriever",
            arguments={"queries": queries},
        )
        self.trace_recorder.emit(
            "retrieval_start",
            retriever=self.retriever_name,
            queries=queries,
        )
        try:
            results = self.rm(
                query_or_queries=query_or_queries,
                exclude_urls=exclude_urls or [],
            )
        except Exception as error:
            duration_sec = round(time.perf_counter() - started, 4)
            self.trace_recorder.emit(
                "retrieval_error",
                retriever=self.retriever_name,
                queries=queries,
                duration_sec=duration_sec,
                error_type=classify_runtime_error(error),
                error=str(error),
            )
            self.trace_recorder.emit(
                "tool_error",
                tool_name=self.retriever_name,
                tool_type="retriever",
                duration_sec=duration_sec,
                error_type=classify_runtime_error(error),
                error=str(error),
            )
            self.trace_recorder.emit(
                "stage_error",
                stage="retrieval",
                invocation_id=invocation_id,
                duration_ms=round(duration_sec * 1000, 2),
                error_type=classify_runtime_error(error),
                error_message=str(error),
            )
            raise
        result_list = list(results or [])
        selected_titles = [
            title for title in (_result_title(item) for item in result_list[:10]) if title
        ]
        duration_sec = round(time.perf_counter() - started, 4)
        output_summary = {
            "result_count": len(result_list),
            "selected_titles": selected_titles,
        }
        self.trace_recorder.emit(
            "retrieval_end",
            retriever=self.retriever_name,
            queries=queries,
            duration_sec=duration_sec,
            **output_summary,
        )
        self.trace_recorder.emit(
            "tool_end",
            tool_name=self.retriever_name,
            tool_type="retriever",
            duration_sec=duration_sec,
            **output_summary,
        )
        self.trace_recorder.emit(
            "stage_end",
            stage="retrieval",
            operation="论文检索完成",
            invocation_id=invocation_id,
            duration_ms=round(duration_sec * 1000, 2),
            output_summary=output_summary,
        )
        self.trace_recorder.emit(
            "artifact_ready",
            stage="retrieval",
            artifact_name="raw_search_results.json",
            output_summary=output_summary,
        )
        return results

    def get_usage_and_reset(self):
        if hasattr(self.rm, "get_usage_and_reset"):
            return self.rm.get_usage_and_reset()
        return {}


def _first_attribute(value, *names):
    for name in names:
        item = getattr(value, name, None)
        if item:
            return item
    return ""


def _result_title(value):
    if isinstance(value, dict):
        return str(value.get("title") or "")
    return str(getattr(value, "title", "") or "")
