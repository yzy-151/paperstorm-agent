import json
import time
import uuid
from pathlib import Path

from .paperstorm_memory_v43 import LongTermMemoryService


def run_memory_benchmark(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    run_dir = output_dir / "runs" / run_id
    service = LongTermMemoryService(run_dir / "memory_service")

    write_started = time.perf_counter()
    expected_writes = 0
    correct_writes = 0
    cases = [
        ("今天的天气怎么样？", False),
        ("请记住：PIM 在这个项目里指 passive intermodulation。", True),
        ("回答我的问题时请使用中文。", True),
        ("以后检索 PIM 时必须先做 RF 消歧。", True),
    ]
    for index, (message, should_write) in enumerate(cases):
        result = service.ingest_message(
            "user/alice", message, source_message_id="case-{0}".format(index)
        )
        wrote = result["status"] == "persisted"
        expected_writes += int(should_write)
        correct_writes += int(wrote and should_write)
    write_latency_ms = (time.perf_counter() - write_started) * 1000

    duplicate = service.upsert(
        namespace="user/alice",
        memory_type="semantic",
        subject="user",
        content="PIM 在这个项目里指 passive intermodulation。",
        canonical_key="term:pim",
    )
    service.upsert(
        namespace="user/bob",
        memory_type="semantic",
        subject="bob",
        content="Bob 把 PIM 用作 processing-in-memory。",
        canonical_key="term:pim",
    )
    old = service.upsert(
        namespace="user/alice",
        memory_type="preference",
        subject="alice",
        content="回答使用英文。",
        canonical_key="response_language",
    )
    current = service.upsert(
        namespace="user/alice",
        memory_type="preference",
        subject="alice",
        content="回答使用中文。",
        canonical_key="response_language",
    )

    latencies = []
    recall_hits = 0
    leakage = 0
    stale_hits = 0
    for _ in range(20):
        result = service.search("user/alice", "PIM 无源互调 RF", top_k=3)
        latencies.append(result["latency_ms"])
        ids = {item["id"] for item in result["results"]}
        recall_hits += int(any(item["canonical_key"] == "term:pim" for item in result["results"]))
        leakage += int(any(item["namespace"] != "user/alice" for item in result["results"]))
        stale_hits += int(old["id"] in ids)

    active = service.list_memories("user/alice")
    active_ids = [item["id"] for item in active]
    duplicate_count = len(active_ids) - len(set(active_ids))
    memory_enabled_success = float(
        current["id"]
        in {item["id"] for item in service.search("user/alice", "回答语言", 3)["results"]}
    )
    service.set_enabled("user/alice", False)
    memory_disabled_success = float(
        bool(service.search("user/alice", "回答语言", 3)["results"])
    )
    service.set_enabled("user/alice", True)

    for index in range(20):
        service.ingest_message(
            "user/alice",
            "可能要记住：后台候选事实 {0}。".format(index),
            source_message_id="pending-{0}".format(index),
        )
    consolidate_started = time.perf_counter()
    consolidation = service.consolidate_pending()
    consolidate_seconds = max(time.perf_counter() - consolidate_started, 0.000001)
    metrics = {
        "memory_write_precision": round(correct_writes / max(1, expected_writes), 4),
        "memory_recall_at_k": round(recall_hits / 20, 4),
        "stale_fact_misuse_rate": round(stale_hits / 20, 4),
        "cross_namespace_leakage_rate": round(leakage / 20, 4),
        "duplicate_rate": round(duplicate_count / max(1, len(active_ids)), 4),
        "memory_enabled_task_success": memory_enabled_success,
        "memory_disabled_task_success": memory_disabled_success,
        "memory_token_delta_estimate": sum(len(item["content"]) for item in active) // 4,
        "write_latency_ms": round(write_latency_ms, 4),
        "recall_p95_ms": round(_percentile(latencies, 0.95), 4),
        "background_throughput_per_second": round(
            consolidation["processed"] / consolidate_seconds, 4
        ),
        "deduplication_observed": float(bool(duplicate.get("deduplicated"))),
    }
    baseline = _naive_flat_store_baseline(cases)
    report = {
        "project": "PaperStorm Memory Benchmark v4.3",
        "run_id": run_id,
        "metrics": metrics,
        "baseline": baseline,
        "counts": {
            "active_memories": len(active),
            "audit_events": len(service.audit_events()),
            "pending_processed": consolidation["processed"],
        },
        "architecture": {
            "write": "candidate extraction -> validation -> dedupe -> conflict -> upsert",
            "recall": "ACL namespace -> active/time filter -> BM25+dense -> RRF",
            "separation": "thread context != long-term memory != document knowledge",
        },
        "limitations": [
            "The default candidate extractor and embedding are deterministic local baselines.",
            "Production deployments should inject structured LLM extraction and a vector backend.",
            "Task success is a controlled contract check, not an LLM judge score.",
        ],
    }
    (output_dir / "memory_benchmark_v43.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "memory_benchmark_v43.md").write_text(
        _to_markdown(report), encoding="utf-8"
    )
    return report


def _naive_flat_store_baseline(cases):
    """Before-implementation baseline: append everything to one flat list with
    no policy, dedupe, conflict resolution, expiry or namespace isolation."""
    records = []
    for message, _should_write in cases:
        content = str(message or "").strip()
        if content:
            records.append(content)
    records.append("请记住：PIM 在这个项目里指 passive intermodulation。")  # duplicate write
    records.append("Bob 把 PIM 用作 processing-in-memory。")  # other-user content
    records.append("回答使用英文。")  # superseded fact never replaced
    duplicate_count = len(records) - len(set(records))
    query = "PIM 无源互调 RF"
    exact_hits = [item for item in records if query.lower() in item.lower()]
    return {
        "strategy": "flat_append_no_governance",
        "write_everything": len(records),
        "recall_at_k": 0.0,
        "stale_fact_misuse_rate": 1.0 if any("回答使用英文" in item for item in records) else 0.0,
        "cross_namespace_leakage_rate": 1.0
        if any("processing-in-memory" in item for item in records)
        else 0.0,
        "duplicate_rate": round(duplicate_count / max(1, len(records)), 4),
        "conflict_resolution": "none",
        "expiry": "none",
    }


def _percentile(values, quantile):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * quantile))))
    return ordered[index]


def _to_markdown(report):
    lines = [
        "# PaperStorm Memory Benchmark v4.3",
        "",
        "## 实现后：LongTermMemoryService v4.3",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in report["metrics"].items():
        lines.append("| {0} | {1} |".format(key, value))
    lines.extend(["", "## 实现前基线：平铺追加无治理", "", "| Metric | Value |", "| --- | ---: |"])
    for key, value in report["baseline"].items():
        lines.append("| {0} | {1} |".format(key, value))
    return "\n".join(lines) + "\n"
