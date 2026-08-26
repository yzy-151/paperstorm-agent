"""Release-oriented benchmark registry and local process manager."""

from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LEGACY_BENCHMARK_IDS = {
    "scifact-retrieval-v55": "scifact-retrieval",
    "qasper-retrieval-v55": "qasper-retrieval",
    "qasper-answer-v55": "qasper-answer",
    "longmemeval-retrieval-v56": "longmemeval-retrieval",
    "qasper-context-v56": "qasper-context",
    "longbench-context-v56": "longbench-context",
    "context-pareto-v60": "context-pareto",
    "longmemeval-e2e-v60": "longmemeval-e2e",
}


@dataclass(frozen=True)
class BenchmarkDefinition:
    id: str
    name: str
    version: str
    category: str
    evidence_tier: str
    description: str
    runner: str
    required_inputs: tuple[str, ...]
    metrics: tuple[str, ...]
    estimated_time: str
    requires_llm: bool = False
    blocked_reason: str = ""


@dataclass(frozen=True)
class ReleaseGatePolicy:
    """Frozen offline release thresholds; zero ACL leakage is non-negotiable."""

    quality_metrics: tuple[str, ...] = ("recall_at_5",)
    max_quality_regression: float = 0.01
    max_p95_ratio: float = 1.20
    max_unsupported_claim_increase: float = 0.01
    max_failure_rate_increase: float = 0.01
    manifest_keys: tuple[str, ...] = ("dataset_sha256", "protocol_sha256")


@dataclass(frozen=True)
class ReleaseGateDecision:
    allowed: bool
    reasons: tuple[str, ...]
    checks: dict

    def to_dict(self):
        return {
            "allowed": self.allowed,
            "reasons": list(self.reasons),
            "checks": self.checks,
        }


class ReleaseGate:
    """Compare two fingerprint-compatible offline runs before release."""

    def evaluate(self, baseline, candidate, policy=None):
        policy = policy or ReleaseGatePolicy()
        baseline = dict(baseline or {})
        candidate = dict(candidate or {})
        checks = {}
        reasons = []

        baseline_manifest = dict(baseline.get("manifest") or {})
        candidate_manifest = dict(candidate.get("manifest") or {})
        mismatched = [
            key
            for key in policy.manifest_keys
            if not baseline_manifest.get(key)
            or baseline_manifest.get(key) != candidate_manifest.get(key)
        ]
        checks["manifest"] = {
            "status": "fail" if mismatched else "pass",
            "mismatched_keys": mismatched,
        }
        if mismatched:
            reasons.append("manifest_mismatch")

        baseline_metrics = dict(baseline.get("metrics") or {})
        candidate_metrics = dict(candidate.get("metrics") or {})
        paired_intervals = dict(candidate.get("paired_delta_ci") or {})
        for metric in policy.quality_metrics:
            before = _finite_metric(baseline_metrics.get(metric), metric)
            after = _finite_metric(candidate_metrics.get(metric), metric)
            delta = after - before
            passed = delta >= -float(policy.max_quality_regression)
            checks[metric] = {
                "status": "pass" if passed else "fail",
                "baseline": before,
                "candidate": after,
                "delta": round(delta, 10),
            }
            if not passed:
                reasons.append("quality_regression:{0}".format(metric))
            if metric in paired_intervals:
                interval = paired_intervals[metric]
                if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                    raise ValueError("paired_delta_ci[{0}] must contain lower and upper".format(metric))
                lower = _finite_metric(interval[0], metric + "_ci_lower")
                upper = _finite_metric(interval[1], metric + "_ci_upper")
                if lower > upper:
                    raise ValueError("paired_delta_ci[{0}] is not ordered".format(metric))
                checks[metric]["paired_delta_ci"] = [lower, upper]
                if upper < -float(policy.max_quality_regression):
                    checks[metric]["status"] = "fail"
                    reasons.append("quality_ci_regression:{0}".format(metric))

        before_p95 = _finite_metric(baseline.get("p95_ms"), "p95_ms")
        after_p95 = _finite_metric(candidate.get("p95_ms"), "p95_ms")
        ratio = after_p95 / before_p95 if before_p95 > 0 else (1.0 if after_p95 == 0 else float("inf"))
        p95_passed = ratio <= float(policy.max_p95_ratio)
        checks["p95"] = {
            "status": "pass" if p95_passed else "fail",
            "baseline": before_p95,
            "candidate": after_p95,
            "ratio": ratio,
        }
        if not p95_passed:
            reasons.append("p95_regression")

        _bounded_rate_check(
            checks, reasons, baseline, candidate, "unsupported_claim_rate",
            float(policy.max_unsupported_claim_increase), "unsupported_claim_regression",
        )
        _bounded_rate_check(
            checks, reasons, baseline, candidate, "failure_rate",
            float(policy.max_failure_rate_increase), "failure_rate_regression",
        )
        acl_leaks = int(candidate.get("acl_leak_count") or 0)
        checks["acl_leak"] = {
            "status": "pass" if acl_leaks == 0 else "fail",
            "candidate": acl_leaks,
        }
        if acl_leaks:
            reasons.append("acl_leak")
        return ReleaseGateDecision(not reasons, tuple(reasons), checks)


def load_offline_replay(run_dir):
    """Summarize frozen JSONL predictions without executing models or network."""
    root = Path(run_dir)
    manifest_path = root / "manifest.json"
    predictions_path = root / "predictions.jsonl"
    if not manifest_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError("offline replay requires manifest.json and predictions.jsonl")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    latencies = sorted(float(row.get("latency_ms") or 0.0) for row in rows)
    failures = sum(row.get("status") != "succeeded" for row in rows)
    unsupported = sum(int(row.get("unsupported_claim_count") or 0) for row in rows)
    validated = sum(int(row.get("validated_claim_count") or 0) for row in rows)
    return {
        "manifest": manifest,
        "case_count": len(rows),
        "failure_rate": failures / len(rows) if rows else 0.0,
        "acl_leak_count": sum(bool(row.get("acl_leak")) for row in rows),
        "unsupported_claim_rate": unsupported / validated if validated else 0.0,
        "p95_ms": _nearest_rank_percentile(latencies, 0.95),
    }


def run_production_governance_benchmark(output_dir):
    """Run the P4 governance contract without network or model calls."""
    from .control_plane import ProductionControlPlane, execute_batch
    from .paperstorm_enterprise_kb import _answer_cache_identity
    from .retrieval import HashEmbeddingProvider, HybridPaperIndex

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    fixture = [
        {
            "document_id": "public-doc",
            "title": "Public policy",
            "text": "public retrieval governance guidance",
        },
        {
            "document_id": "private-doc",
            "title": "Private policy",
            "text": "private salary secret needle",
        },
    ]
    protocol = "production-governance-v1"
    dataset_sha = _stable_sha256(fixture)
    protocol_sha = _stable_sha256(
        {
            "protocol": protocol,
            "checks": [
                "pre_retrieval_acl",
                "cache_partition",
                "deadline",
                "circuit_breaker",
                "batch_order",
                "trace_redaction",
                "release_gate",
            ],
        }
    )
    manifest = {
        "benchmark": "production-governance",
        "protocol": protocol,
        "dataset_sha256": dataset_sha,
        "protocol_sha256": protocol_sha,
        "requires_network": False,
        "requires_llm": False,
        "created_at": _now(),
    }

    predictions = []
    latencies = []

    def record(case_id, passed, latency_ms, **details):
        latency_ms = round(float(latency_ms), 6)
        latencies.append(latency_ms)
        predictions.append(
            {
                "case_id": case_id,
                "status": "succeeded" if passed else "failed",
                "latency_ms": latency_ms,
                **details,
            }
        )

    index = HybridPaperIndex.from_documents(
        fixture,
        embedding_provider=HashEmbeddingProvider(),
        chunk_size=500,
        chunk_overlap=0,
    )
    acl_leaks = 0
    for case_id, allowed_ids, expected_ids in (
        ("acl-public-only", ("public-doc",), {"public-doc"}),
        ("acl-private-only", ("private-doc",), {"private-doc"}),
        ("acl-deny-all", (), set()),
    ):
        started = time.perf_counter()
        results = index.search(
            "private secret needle",
            mode="hybrid",
            top_k=5,
            allowed_document_ids=allowed_ids,
        )
        returned_ids = {str(item.get("document_id") or "") for item in results}
        leaked = not returned_ids.issubset(set(allowed_ids))
        acl_leaks += int(leaked)
        record(
            case_id,
            returned_ids == expected_ids and not leaked,
            (time.perf_counter() - started) * 1000,
            acl_leak=leaked,
            allowed_document_ids=list(allowed_ids),
            returned_document_ids=sorted(returned_ids),
        )

    cache_common = {
        "kb_id": "kb-governed",
        "index_revision": {"index_version": 1, "schema_revision": 1},
        "top_k": 5,
        "query": "retrieval governance",
        "search_plan": {"standalone_query": "retrieval governance"},
    }
    cache_started = time.perf_counter()
    owner_cache = _answer_cache_identity(
        tenant_id="tenant-a",
        user_id="owner",
        policy_digest="policy-owner",
        **cache_common,
    )
    viewer_cache = _answer_cache_identity(
        tenant_id="tenant-a",
        user_id="viewer",
        policy_digest="policy-viewer",
        **cache_common,
    )
    cache_isolated = owner_cache != viewer_cache
    record(
        "cache-policy-isolation",
        cache_isolated,
        (time.perf_counter() - cache_started) * 1000,
        cache_collision=not cache_isolated,
    )

    control = ProductionControlPlane(root / "governance.sqlite")
    timeout_started = time.perf_counter()
    timeout_result = control.execute_resilient(
        "governance-provider",
        lambda: time.sleep(0.05),
        fallback=lambda _error: "fallback",
        max_attempts=1,
        failure_threshold=1,
        cooldown_seconds=60,
        timeout_seconds=0.005,
    )
    timeout_ms = (time.perf_counter() - timeout_started) * 1000
    timeout_classified = timeout_result.get("failure_type") == "timeout"
    record(
        "deadline-timeout",
        timeout_classified,
        timeout_ms,
        failure_type=timeout_result.get("failure_type"),
    )

    skipped_calls = []
    circuit_result = control.execute_resilient(
        "governance-provider",
        lambda: skipped_calls.append("called") or "unexpected",
        fallback=lambda _error: "fallback",
        max_attempts=1,
        failure_threshold=1,
        cooldown_seconds=60,
    )
    recovered = control.execute_resilient(
        "governance-provider",
        lambda: "recovered",
        fallback=lambda _error: "fallback",
        max_attempts=1,
        failure_threshold=1,
        cooldown_seconds=0,
    )
    circuit_recovered = (
        not skipped_calls
        and circuit_result.get("failure_type") == "circuit_open"
        and recovered.get("result") == "recovered"
        and recovered.get("circuit_state") == "closed"
    )
    record(
        "circuit-open-recovery",
        circuit_recovered,
        0.0,
        provider_calls_while_open=len(skipped_calls),
        recovered=bool(recovered.get("half_open_probe")),
    )

    batch_started = time.perf_counter()
    batch_output = execute_batch(
        (3, 1, 2),
        lambda value: (time.sleep(value * 0.001), value * 10)[1],
        max_workers=3,
    )
    batch_order_preserved = batch_output == [30, 10, 20]
    record(
        "concurrent-batch-order",
        batch_order_preserved,
        (time.perf_counter() - batch_started) * 1000,
        output=batch_output,
    )

    control.record_span(
        {
            "trace_id": "governance-trace",
            "component": "retrieval",
            "operation": "governance-check",
            "attributes": {
                "api_key": "sk-must-not-leak",
                "user_id": "private@example.com",
                "private_document": "PRIVATE" * 1000,
            },
        }
    )
    attributes = control.list_spans("governance-trace")[0]["attributes"]
    serialized_attributes = json.dumps(attributes, ensure_ascii=False)
    secret_leak_count = sum(
        marker in serialized_attributes
        for marker in ("sk-must-not-leak", "private@example.com")
    )
    trace_redacted = secret_leak_count == 0
    record(
        "trace-redaction",
        trace_redacted,
        0.0,
        secret_leak=not trace_redacted,
    )

    _write_json(root / "manifest.json", manifest)
    _write_jsonl(root / "predictions.jsonl", predictions)
    replay = load_offline_replay(root)
    release_input = {
        **replay,
        "metrics": {"recall_at_5": 1.0},
        "unsupported_claim_rate": 0.0,
    }
    decision = ReleaseGate().evaluate(release_input, dict(release_input))
    metrics = {
        "case_count": len(predictions),
        "acl_leak_count": acl_leaks,
        "secret_leak_count": secret_leak_count,
        "cache_isolated": cache_isolated,
        "timeout_classified": timeout_classified,
        "circuit_recovered": circuit_recovered,
        "batch_order_preserved": batch_order_preserved,
        "release_gate_allowed": decision.allowed,
        "failure_rate": replay["failure_rate"],
        "p95_ms": _nearest_rank_percentile(sorted(latencies), 0.95),
    }
    report = {
        "status": "completed" if not replay["failure_rate"] else "failed",
        "manifest": manifest,
        "metrics": metrics,
        "release_gate": decision.to_dict(),
        "predictions": predictions,
    }
    _write_json(root / "metrics.json", report)
    _write_jsonl(
        root / "case_dossiers.jsonl",
        [
            {
                "case_id": row["case_id"],
                "difficulty": "production governance contract",
                "before": "A missing boundary could leak data or hide a provider failure.",
                "root_cause": "Retrieval, cache, resilience, and trace controls require explicit contracts.",
                "change": "Validate the boundary with a deterministic offline case.",
                "after": row["status"],
                "resolved": row["status"] == "succeeded",
            }
            for row in predictions
        ],
    )
    _write_json(root / "run_status.json", {"status": report["status"], "finished_at": _now()})
    return report


def _stable_sha256(payload):
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_jsonl(path, rows):
    Path(path).write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _finite_metric(value, name):
    number = float(value)
    if not (number == number and abs(number) != float("inf")):
        raise ValueError("{0} must be finite".format(name))
    return number


def _bounded_rate_check(checks, reasons, baseline, candidate, key, allowed, reason):
    before = _finite_metric(baseline.get(key), key)
    after = _finite_metric(candidate.get(key), key)
    delta = after - before
    passed = delta <= allowed
    checks[key] = {
        "status": "pass" if passed else "fail",
        "baseline": before,
        "candidate": after,
        "delta": round(delta, 10),
    }
    if not passed:
        reasons.append(reason)


def _nearest_rank_percentile(values, quantile):
    if not values:
        return 0.0
    import math

    index = max(0, min(len(values) - 1, math.ceil(float(quantile) * len(values)) - 1))
    return float(values[index])


DEFINITIONS = (
    BenchmarkDefinition(
        "scifact-retrieval",
        "SciFact 科学论文检索",
        "current",
        "RAG Retrieval",
        "public_official",
        "在 SciFact 官方 test 上比较 BM25、Hybrid 与 Cross-Encoder 重排。",
        "run_paperstorm_public_benchmark.py",
        ("scifact_dir",),
        ("Recall@10", "MRR@10", "nDCG@10", "P95 latency"),
        "Smoke 约 1 分钟；完整真实向量约 20-40 分钟",
    ),
    BenchmarkDefinition(
        "qasper-retrieval",
        "QASPER 论文内证据检索",
        "current",
        "RAG Retrieval",
        "public_official",
        "在 QASPER 官方 test 的论文内检索人工证据段落。",
        "run_paperstorm_public_benchmark.py",
        ("qasper_json",),
        ("Recall@5", "MRR@5", "nDCG@5", "P95 latency"),
        "Smoke 约 1 分钟；完整真实向量约 1-2 小时",
    ),
    BenchmarkDefinition(
        "qasper-answer",
        "QASPER 端到端回答",
        "current",
        "RAG Generation",
        "public_official_llm",
        "冻结检索排名后调用 Reader LLM，按官方口径计算 Answer/Evidence F1。",
        "run_qasper_answer_benchmark.py",
        ("qasper_rankings", "qasper_cache"),
        ("Answer F1", "Evidence F1", "Exact Match", "success rate"),
        "Smoke 约 5 分钟；完整 1451 题会产生 API 成本",
        requires_llm=True,
    ),
    BenchmarkDefinition(
        "longmemeval-retrieval",
        "LongMemEval-S 长期记忆检索",
        "current",
        "Memory",
        "public_official_retrieval",
        "在官方 500 题上评估跨会话 evidence-session Recall@5 与延迟。",
        "run_longmemeval_benchmark.py",
        ("longmemeval_json",),
        ("Recall@5", "category recall", "P50 latency", "P95 latency"),
        "Smoke 约 1 分钟；完整真实向量约 15-30 分钟",
    ),
    BenchmarkDefinition(
        "qasper-context",
        "QASPER Context 预算治理",
        "current",
        "Context",
        "public_official_diagnostic",
        "在冻结检索结果上测量 token 压缩、金证据保留和结构校验。",
        "run_qasper_context_benchmark.py",
        ("qasper_json", "qasper_rankings"),
        ("token ratio", "gold evidence recall", "validation rate"),
        "约 1-3 分钟，不调用 LLM",
    ),
    BenchmarkDefinition(
        "longbench-context",
        "LongBench Context 配对评测",
        "current",
        "Context",
        "adapter_only",
        "适配器已完成，但缺少同模型 full/fixed/PaperStorm 配对预测，不能发布正式分数。",
        "run_longbench_context_benchmark.py",
        ("longbench_json", "longbench_predictions"),
        ("paired task score", "token reduction", "quality delta"),
        "尚未具备可发布输入",
        blocked_reason="缺少冻结的同模型配对预测文件",
    ),
    BenchmarkDefinition(
        "context-pareto",
        "LongBench 128K/256K/512K Pareto",
        "current",
        "Context",
        "public_official_llm_ablation",
        "同模型、同数据、同提示词下比较质量、输入 Token、TTFT、延迟与成本，并计算 Pareto 前沿。",
        "run_context_profile_pareto.py",
        ("longbench_json",),
        ("Accuracy", "Input tokens", "TTFT P50/P95", "Cost", "Pareto frontier"),
        "Smoke 约 5-15 分钟；完整运行取决于长上下文模型并产生 API 成本",
        requires_llm=True,
    ),
    BenchmarkDefinition(
        "longmemeval-e2e",
        "LongMemEval-S 端到端 Reader/Judge",
        "current",
        "Memory",
        "public_official_llm",
        "全量对比 Recent、FTS Session 与 PaperStorm Memory，统一 Reader 和 LLM Judge。",
        "run_longmemeval_e2e.py",
        ("longmemeval_json",),
        ("Judge accuracy", "Recall@5", "P50 latency", "Token usage", "Cost"),
        "Smoke 约 5-15 分钟；完整 500 题会产生 Reader 与 Judge API 成本",
        requires_llm=True,
    ),
)


class BenchmarkRegistry:
    def __init__(self, benchmark_root: Optional[Path] = None):
        self.root = _resolve_benchmark_root(benchmark_root)
        self.inputs = _discover_inputs(self.root)
        self._definitions = {item.id: item for item in DEFINITIONS}

    def catalog(self):
        benchmarks = []
        for definition in sorted(DEFINITIONS, key=lambda item: bool(item.blocked_reason)):
            inputs = [
                {
                    "key": key,
                    "path": str(self.inputs.get(key) or ""),
                    "available": bool(self.inputs.get(key)),
                }
                for key in definition.required_inputs
            ]
            missing = [item["key"] for item in inputs if not item["available"]]
            blocker = definition.blocked_reason
            if missing and not blocker:
                blocker = "缺少本地输入：{0}".format(", ".join(missing))
            ready = not blocker and not missing
            latest_path = _latest_result_path(definition.id, self.root)
            benchmarks.append(
                {
                    "id": definition.id,
                    "name": definition.name,
                    "version": definition.version,
                    "category": definition.category,
                    "evidence_tier": definition.evidence_tier,
                    "description": definition.description,
                    "metrics": list(definition.metrics),
                    "estimated_time": definition.estimated_time,
                    "requires_llm": definition.requires_llm,
                    "ready": ready,
                    "status": "ready" if ready else "blocked",
                    "blocker": blocker,
                    "inputs": inputs,
                    "profiles": ["smoke", "quality"],
                    "latest_result_path": str(latest_path or ""),
                    "latest_result": _read_json(latest_path) if latest_path else {},
                }
            )
        return {
            "benchmark_root": str(self.root),
            "model_cache": str(self.root / "models"),
            "python": sys.executable,
            "benchmarks": benchmarks,
        }

    def definition(self, benchmark_id: str):
        benchmark_id = LEGACY_BENCHMARK_IDS.get(benchmark_id, benchmark_id)
        try:
            return self._definitions[benchmark_id]
        except KeyError as exc:
            raise KeyError("unknown benchmark: {0}".format(benchmark_id)) from exc

    def build_command(
        self,
        benchmark_id: str,
        output_dir: Path,
        profile: str = "smoke",
        allow_paid_llm: bool = False,
    ):
        definition = self.definition(benchmark_id)
        benchmark_id = definition.id
        if profile not in {"smoke", "quality"}:
            raise ValueError("profile must be smoke or quality")
        if definition.blocked_reason:
            raise ValueError(definition.blocked_reason)
        missing = [key for key in definition.required_inputs if not self.inputs.get(key)]
        if missing:
            raise ValueError("缺少本地输入：{0}".format(", ".join(missing)))
        if definition.requires_llm:
            if not allow_paid_llm:
                raise ValueError("付费 LLM Benchmark 必须显式确认")
            if not os.getenv("DEEPSEEK_API_KEY"):
                raise ValueError("运行付费生成评测需要 DEEPSEEK_API_KEY")
            if benchmark_id == "longmemeval-e2e" and not os.getenv("OPENAI_API_KEY"):
                raise ValueError("官方 LongMemEval Judge 需要 OPENAI_API_KEY")

        runner = PROJECT_ROOT / "examples" / "storm_examples" / definition.runner
        command = [sys.executable, str(runner)]
        output_dir = Path(output_dir)
        if benchmark_id == "scifact-retrieval":
            command += [
                "--benchmark", "scifact", "--dataset-dir", str(self.inputs["scifact_dir"]),
                "--cache-dir", str(self.root), "--output-dir", str(output_dir),
                "--top-k", "10",
            ]
            command += ["--embedding", "hash", "--modes", "bm25", "hybrid", "--smoke-limit", "20"] if profile == "smoke" else ["--embedding", "real", "--modes", "bm25", "dense", "hybrid", "hybrid_rerank", "--reranker"]
        elif benchmark_id == "qasper-retrieval":
            command += [
                "--benchmark", "qasper", "--dataset-dir", str(self.inputs["qasper_json"]),
                "--cache-dir", str(self.root), "--output-dir", str(output_dir),
                "--top-k", "5",
            ]
            command += ["--embedding", "hash", "--modes", "bm25", "hybrid", "--smoke-limit", "20"] if profile == "smoke" else ["--embedding", "real", "--modes", "bm25", "dense", "hybrid", "hybrid_rerank", "--reranker"]
        elif benchmark_id == "qasper-answer":
            command += [
                "--split", "test", "--retrieval-predictions", str(self.inputs["qasper_rankings"]),
                "--cache-dir", str(self.inputs["qasper_cache"]), "--output-dir", str(output_dir),
            ]
            if profile == "smoke":
                command += ["--smoke-limit", "10"]
        elif benchmark_id == "longmemeval-retrieval":
            command += [
                "--dataset", str(self.inputs["longmemeval_json"]), "--output-dir", str(output_dir),
                "--model-cache", str(self.root / "models"), "--top-k", "5",
            ]
            command += ["--embedding", "hash", "--limit", "10"] if profile == "smoke" else ["--embedding", "sentence-transformer"]
        elif benchmark_id == "qasper-context":
            command += [
                "--dataset", str(self.inputs["qasper_json"]), "--rankings", str(self.inputs["qasper_rankings"]),
                "--output-dir", str(output_dir), "--mode", "hybrid_rerank",
            ]
        elif benchmark_id == "context-pareto":
            command += [
                "--dataset", str(self.inputs["longbench_json"]),
                "--output-dir", str(output_dir),
            ]
            if profile == "smoke":
                command += ["--limit", "3"]
        elif benchmark_id == "longmemeval-e2e":
            command += [
                "--dataset", str(self.inputs["longmemeval_json"]),
                "--output-dir", str(output_dir),
                "--model-cache", str(self.root / "models"),
                "--top-k", "5",
            ]
            if profile == "smoke":
                command += ["--limit", "3"]
        else:
            raise ValueError("benchmark is not runnable")
        return command


class BenchmarkRunManager:
    def __init__(
        self,
        service_root: Path,
        registry: Optional[BenchmarkRegistry] = None,
        popen_factory: Callable = subprocess.Popen,
        observability=None,
    ):
        self.root = Path(service_root) / "benchmark_runs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry = registry or BenchmarkRegistry()
        self.popen_factory = popen_factory
        self.observability = observability
        self._processes: Dict[str, object] = {}
        self._logs: Dict[str, object] = {}
        self._traces: Dict[str, object] = {}
        self._lock = threading.Lock()

    def catalog(self):
        return self.registry.catalog()

    def start(self, benchmark_id: str, profile="smoke", allow_paid_llm=False):
        benchmark_id = self.registry.definition(benchmark_id).id
        run_id = uuid.uuid4().hex
        run_dir = self.root / run_id
        output_dir = run_dir / "artifacts"
        run_dir.mkdir(parents=True)
        command = self.registry.build_command(
            benchmark_id,
            output_dir=output_dir,
            profile=profile,
            allow_paid_llm=allow_paid_llm,
        )
        log_path = run_dir / "benchmark.log"
        log_handle = log_path.open("w", encoding="utf-8")
        trace = None
        if self.observability is not None:
            trace = self.observability.trace(
                "paperstorm.benchmark",
                input={"benchmark_id": benchmark_id, "profile": profile},
                metadata={"run_id": run_id, "version": "6.0.0"},
                session_id=run_id,
                tags=["benchmark", profile],
            )
            trace.__enter__()
        try:
            process = self.popen_factory(
                command,
                cwd=str(PROJECT_ROOT),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
            )
        except Exception as error:
            log_handle.close()
            if trace is not None:
                trace.end(output={"status": "failed_to_start"}, error=error)
            raise
        manifest = {
            "run_id": run_id,
            "benchmark_id": benchmark_id,
            "profile": profile,
            "status": "running",
            "pid": getattr(process, "pid", None),
            "command": command,
            "command_preview": subprocess.list2cmdline(command),
            "created_at": _now(),
            "started_at": _now(),
            "output_dir": str(output_dir),
            "log_path": str(log_path),
            "manifest_path": str(run_dir / "run.json"),
        }
        self._write(manifest)
        with self._lock:
            self._processes[run_id] = process
            self._logs[run_id] = log_handle
            if trace is not None:
                self._traces[run_id] = trace
        return self.get(run_id)

    def get(self, run_id: str):
        manifest = self._read(run_id)
        process = self._processes.get(run_id)
        if manifest["status"] == "running" and process is not None:
            returncode = process.poll()
            if returncode is not None:
                self._close_log(run_id)
                manifest["returncode"] = returncode
                manifest["status"] = "succeeded" if returncode == 0 else "failed"
                manifest["finished_at"] = _now()
                self._write(manifest)
        elif manifest["status"] == "running" and process is None:
            manifest["status"] = "interrupted"
            manifest["finished_at"] = _now()
            self._write(manifest)
        manifest["log_tail"] = _tail(Path(manifest["log_path"]))
        result_path = Path(manifest["output_dir"]) / "metrics.json"
        manifest["result"] = _read_json(result_path)
        manifest["result_path"] = str(result_path) if result_path.exists() else ""
        if manifest["status"] != "running" and not manifest.get("observability_exported"):
            self._finish_observability(manifest)
            manifest["observability_exported"] = True
            self._write(manifest)
        return manifest

    def cancel(self, run_id: str):
        manifest = self._read(run_id)
        process = self._processes.get(run_id)
        if manifest["status"] == "running" and process is not None:
            process.terminate()
            manifest["status"] = "cancelled"
            manifest["finished_at"] = _now()
            self._close_log(run_id)
            self._write(manifest)
        return self.get(run_id)

    def _manifest_path(self, run_id):
        if not run_id or any(value in run_id for value in ("/", "\\", "..")):
            raise KeyError("invalid benchmark run id")
        return self.root / run_id / "run.json"

    def _read(self, run_id):
        path = self._manifest_path(run_id)
        if not path.exists():
            raise KeyError("benchmark run not found: {0}".format(run_id))
        return json.loads(path.read_text(encoding="utf-8"))

    def _write(self, manifest):
        path = Path(manifest["manifest_path"])
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    def _close_log(self, run_id):
        handle = self._logs.pop(run_id, None)
        if handle is not None and not handle.closed:
            handle.flush()
            handle.close()

    def _finish_observability(self, manifest):
        from .paperstorm_observability import numeric_scores

        trace = self._traces.pop(manifest["run_id"], None)
        if trace is None:
            return
        for name, value in numeric_scores(manifest.get("result") or {}).items():
            trace.score(name, value)
        trace.score("run_success", 1.0 if manifest.get("status") == "succeeded" else 0.0)
        trace.end(
            output={
                "status": manifest.get("status"),
                "result": manifest.get("result") or {},
                "result_path": manifest.get("result_path", ""),
            }
        )


def _resolve_benchmark_root(explicit=None):
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.getenv("PAPERSTORM_BENCHMARK_ROOT"):
        candidates.append(Path(os.environ["PAPERSTORM_BENCHMARK_ROOT"]))
    candidates.extend(
        [
            Path.home() / "Desktop" / "codex" / "paperstorm-benchmarks",
            PROJECT_ROOT / "data" / "benchmarks",
        ]
    )
    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate.resolve()
        except OSError:
            continue
    return candidates[0].expanduser().resolve() if candidates else (PROJECT_ROOT / "data" / "benchmarks")


def _discover_inputs(root):
    root = Path(root)
    qasper_json = root / "qasper-official-v0.3" / "qasper-test-v0.3.json"
    project_rankings = PROJECT_ROOT / "results" / "public_benchmarks" / "v55_qasper_test_real" / "predictions.jsonl"
    context_rankings = root / "v56" / "runs" / "qasper-context" / "predictions.jsonl"
    values = {
        "scifact_dir": root / "datasets" / "scifact",
        "qasper_json": qasper_json,
        "qasper_cache": root,
        "qasper_rankings": project_rankings if project_rankings.exists() else context_rankings,
        "longmemeval_json": root / "v56" / "longmemeval_s_cleaned.json",
        "longbench_json": root / "v56" / "longbench_v2_data.json",
        "longbench_predictions": root / "v56" / "runs" / "longbench-context" / "predictions.json",
    }
    return {key: path.resolve() for key, path in values.items() if _input_exists(key, path)}


def _input_exists(key, path):
    try:
        if key == "scifact_dir":
            return all((path / relative).exists() for relative in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"))
        return path.exists()
    except OSError:
        return False


def _latest_result_path(benchmark_id, root):
    candidates = {
        "scifact-retrieval": [PROJECT_ROOT / "results/public_benchmarks/v55_scifact_real/metrics.json"],
        "qasper-retrieval": [PROJECT_ROOT / "results/public_benchmarks/v55_qasper_test_real/metrics.json"],
        "qasper-answer": [PROJECT_ROOT / "results/public_benchmarks/v55_qasper_answer_test_real/metrics.json"],
        "longmemeval-retrieval": [root / "v56/runs/longmemeval-s-minilm/metrics.json"],
        "qasper-context": [root / "v56/runs/qasper-context/metrics.json"],
        "context-pareto": [root / "v60/runs/context-pareto/metrics.json"],
        "longmemeval-e2e": [root / "v60/runs/longmemeval-e2e/metrics.json"],
    }.get(benchmark_id, [])
    return next((path.resolve() for path in candidates if path.exists()), None)


def _read_json(path):
    if not path or not Path(path).exists():
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _tail(path, max_chars=12000):
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def _now():
    return datetime.now(timezone.utc).isoformat()
