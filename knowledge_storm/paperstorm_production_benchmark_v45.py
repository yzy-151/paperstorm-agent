import json
import statistics
import time
import uuid
from pathlib import Path

from .paperstorm_production_v45 import ProductionControlPlaneV45


def run_production_benchmark(output_dir, request_count: int = 100):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    run_dir = output_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    control = ProductionControlPlaneV45(run_dir / "production.sqlite")
    request_count = max(10, int(request_count))
    control.register_resource(
        tenant_id="tenant-a",
        resource_type="knowledge_base",
        resource_id="benchmark-kb",
        owner_user_id="alice",
        allowed_user_ids=["bob"],
    )

    latencies = []
    errors = 0
    started = time.perf_counter()
    for index in range(request_count):
        item_started = time.perf_counter()
        try:
            control.authorize(
                "tenant-a", "bob", "knowledge_base", "benchmark-kb", "read"
            )
            cached = control.get_cache("tenant-a/retrieval", "stable-query")
            if not cached["hit"]:
                control.set_cache(
                    "tenant-a/retrieval",
                    "stable-query",
                    {"chunks": ["benchmark"]},
                    ttl_seconds=60,
                    tags=["kb:benchmark-kb"],
                )
        except Exception:
            errors += 1
        latencies.append((time.perf_counter() - item_started) * 1000)
    elapsed = max(0.000001, time.perf_counter() - started)

    leakage = 0
    try:
        control.authorize(
            "tenant-b", "mallory", "knowledge_base", "benchmark-kb", "read"
        )
        leakage = 1
    except PermissionError:
        pass

    operation_calls = []
    first = control.execute_idempotent(
        "tenant-a/benchmark",
        "same-request",
        {"query": "stable"},
        lambda: operation_calls.append(1) or {"answer": "ok"},
    )
    replay = control.execute_idempotent(
        "tenant-a/benchmark",
        "same-request",
        {"query": "stable"},
        lambda: operation_calls.append(1) or {"answer": "duplicate"},
    )

    job_attempts = []
    control.enqueue_job(
        "tenant-a", "benchmark_job", {"item": 1}, "job-1", max_attempts=2
    )

    def flaky_job(payload):
        job_attempts.append(payload)
        if len(job_attempts) == 1:
            raise ConnectionError("controlled failure")
        return {"processed": True}

    control.run_worker_tick({"benchmark_job": flaky_job})
    recovered_job = control.run_worker_tick({"benchmark_job": flaky_job})

    provider_calls = []

    def unavailable():
        provider_calls.append(1)
        raise TimeoutError("controlled timeout")

    degraded = control.execute_resilient(
        "benchmark-provider",
        unavailable,
        fallback=lambda error: {"mode": "lexical_only"},
        max_attempts=2,
        failure_threshold=2,
        cooldown_seconds=60,
    )

    trace_id = uuid.uuid4().hex
    with control.trace_span(trace_id, "benchmark", "controlled_request"):
        time.sleep(0.001)
    spans = control.list_spans(trace_id)
    ordered = sorted(latencies)
    cache = control.cache_metrics()
    metrics = {
        "latency_p50_ms": round(_percentile(ordered, 0.50), 4),
        "latency_p95_ms": round(_percentile(ordered, 0.95), 4),
        "latency_p99_ms": round(_percentile(ordered, 0.99), 4),
        "latency_mean_ms": round(statistics.mean(ordered), 4),
        "qps": round(request_count / elapsed, 4),
        "error_rate": round(errors / request_count, 4),
        "degradation_rate": round(float(degraded["degraded"]) / request_count, 4),
        "cache_hit_rate": cache["hit_rate"],
        "acl_leakage_rate": float(leakage),
        "idempotency_rate": float(
            len(operation_calls) == 1
            and not first["idempotent_replay"]
            and replay["idempotent_replay"]
        ),
        "job_recovery_rate": float(recovered_job.get("status") == "succeeded"),
        "trace_span_coverage": float(bool(spans and spans[0].get("duration_ms") is not None)),
    }
    report = {
        "project": "PaperStorm Production Governance Benchmark v4.5",
        "run_id": run_id,
        "request_count": request_count,
        "metrics": metrics,
        "control_plane": control.status(),
        "slo": {
            "target_p95_ms": 50,
            "target_error_rate": 0.01,
            "p95_pass": metrics["latency_p95_ms"] <= 50,
            "error_rate_pass": metrics["error_rate"] <= 0.01,
        },
        "degradation": degraded,
        "limitations": [
            "The load path measures the local SQLite governance hot path, not real LLM or arXiv latency.",
            "The benchmark is single-process and does not represent a distributed deployment.",
            "Authentication is represented by explicit tenant/user inputs; OAuth or signed identity is not included.",
        ],
    }
    (output_dir / "production_benchmark_v45.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "production_benchmark_v45.md").write_text(
        _markdown(report), encoding="utf-8"
    )
    return report


def _percentile(values, quantile):
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, int(round((len(values) - 1) * quantile))))
    return values[index]


def _markdown(report):
    lines = [
        "# PaperStorm Production Governance Benchmark v4.5",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for name, value in report["metrics"].items():
        lines.append("| {0} | {1} |".format(name, value))
    return "\n".join(lines) + "\n"
