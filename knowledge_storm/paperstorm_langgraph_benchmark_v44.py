import json
import time
import uuid
from pathlib import Path

from .paperstorm_langgraph_v44 import PaperStormLangGraphRuntime
from .paperstorm_service import PaperStormTaskService


class _FlakyBenchmarkTool:
    name = "storm_deep_research"

    def __init__(self):
        self.calls = 0

    def run(self, arguments):
        self.calls += 1
        if self.calls == 1:
            raise ConnectionError("controlled transient failure")
        return {
            "answer": "PIM 是 passive intermodulation。[1]",
            "citations": [{"id": 1, "url": "https://example.com/pim"}],
            "evidence": [{"content": "passive intermodulation RF"}],
            "grounded": True,
            "task_id": "retry-task",
            "artifact_uri": "artifact://retry-task/research",
            "evidence_sufficiency": {"sufficient": True, "score": 90},
            "retrieval_triggered": True,
        }


def run_langgraph_benchmark(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    run_dir = output_dir / "runs" / run_id
    service = PaperStormTaskService(run_dir / "service")
    runtime = PaperStormLangGraphRuntime(run_dir / "runtime", service)
    latencies = []
    path_checks = []
    try:
        casual = _timed_invoke(
            runtime,
            latencies,
            thread_id="casual",
            request_id="casual-1",
            user_id="alice",
            message="你好",
            run_mode="fake",
        )
        path_checks.append(casual["route"] == "casual_chat")
        remembered = _timed_invoke(
            runtime,
            latencies,
            thread_id="memory-write",
            request_id="memory-write-1",
            user_id="alice",
            message="请记住：回答我的问题时使用中文。",
            run_mode="fake",
        )
        path_checks.append(remembered["memory_write"]["status"] == "persisted")
        recalled = _timed_invoke(
            runtime,
            latencies,
            thread_id="memory-read",
            request_id="memory-read-1",
            user_id="alice",
            message="你记得我的回答偏好吗？",
            run_mode="fake",
        )
        path_checks.append(recalled["route"] == "memory_answer")
        isolated = _timed_invoke(
            runtime,
            latencies,
            thread_id="memory-bob",
            request_id="memory-bob-1",
            user_id="bob",
            message="你记得我的回答偏好吗？",
            run_mode="fake",
        )
        path_checks.append(isolated["route"] == "casual_chat")
        research = _timed_invoke(
            runtime,
            latencies,
            thread_id="research",
            request_id="research-1",
            user_id="alice",
            message="请调研 PIM 无源互调论文",
            topic="PIM 无源互调",
            run_mode="fake",
        )
        path_checks.append(research["route"] == "deep_research")
        task_count = len(service.list_tasks())
        replay = runtime.invoke(
            thread_id="research",
            request_id="research-1",
            user_id="alice",
            message="请调研 PIM 无源互调论文",
            topic="PIM 无源互调",
            run_mode="fake",
        )
        idempotency_rate = float(
            replay["idempotent_replay"] and len(service.list_tasks()) == task_count
        )
    finally:
        runtime.close()

    recreated = PaperStormLangGraphRuntime(run_dir / "runtime", service)
    try:
        restored = recreated.get_thread_state("research")
        checkpoint_restore_rate = float(
            (restored.get("values") or {}).get("request_id") == "research-1"
        )
    finally:
        recreated.close()

    flaky = _FlakyBenchmarkTool()
    retry_runtime = PaperStormLangGraphRuntime(
        run_dir / "retry_runtime", service, deep_research_tool=flaky
    )
    try:
        retry_result = retry_runtime.invoke(
            thread_id="retry",
            request_id="retry-1",
            user_id="alice",
            message="请调研 PIM 无源互调论文",
            run_mode="fake",
        )
        retry_recovery_rate = float(
            flaky.calls == 2 and retry_result["status"] == "succeeded"
        )
    finally:
        retry_runtime.close()

    node_events = (
        casual["node_events"]
        + remembered["node_events"]
        + recalled["node_events"]
        + isolated["node_events"]
        + research["node_events"]
    )
    trace_span_coverage = sum(bool(item.get("span_id")) for item in node_events) / max(
        1, len(node_events)
    )
    metrics = {
        "path_accuracy": round(sum(path_checks) / max(1, len(path_checks)), 4),
        "idempotency_rate": idempotency_rate,
        "checkpoint_restore_rate": checkpoint_restore_rate,
        "retry_recovery_rate": retry_recovery_rate,
        "cross_user_leakage_rate": float(
            bool((isolated.get("memory_recall") or {}).get("results"))
        ),
        "trace_span_coverage": round(trace_span_coverage, 4),
        "artifact_contract_rate": float(
            bool(research.get("artifact_uri")) and bool(research.get("citations"))
        ),
        "latency_p95_ms": round(_percentile(latencies, 0.95), 4),
        "average_executed_nodes": round(
            sum(len(item["executed_nodes"]) for item in [casual, remembered, recalled, isolated, research])
            / 5,
            2,
        ),
    }
    report = {
        "project": "PaperStorm LangGraph Runtime Benchmark v4.4",
        "run_id": run_id,
        "metrics": metrics,
        "paths": {
            "casual": casual["executed_nodes"],
            "memory_write": remembered["executed_nodes"],
            "memory_recall": recalled["executed_nodes"],
            "deep_research": research["executed_nodes"],
        },
        "runtime": {
            "graph": "LangGraph StateGraph",
            "checkpointer": "SqliteSaver",
            "memory": "PaperStorm V4.3 namespace store",
            "deep_research": "isolated STORM tool",
        },
        "limitations": [
            "The benchmark uses deterministic fake research and controlled routing cases.",
            "SQLite is suitable for a local demo, not multi-process production throughput.",
            "Sync node timeout metadata documents budgets; actual cancellation belongs in async clients.",
        ],
    }
    (output_dir / "langgraph_benchmark_v44.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "langgraph_benchmark_v44.md").write_text(
        _to_markdown(report), encoding="utf-8"
    )
    return report


def _timed_invoke(runtime, latencies, **payload):
    started = time.perf_counter()
    result = runtime.invoke(**payload)
    latencies.append((time.perf_counter() - started) * 1000)
    return result


def _percentile(values, quantile):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * quantile))))
    return ordered[index]


def _to_markdown(report):
    lines = ["# PaperStorm LangGraph Runtime Benchmark v4.4", "", "| Metric | Value |", "| --- | ---: |"]
    for key, value in report["metrics"].items():
        lines.append("| {0} | {1} |".format(key, value))
    return "\n".join(lines) + "\n"
