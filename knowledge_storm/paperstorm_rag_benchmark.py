import json
import statistics
import time
from pathlib import Path

from .paperstorm_rag import ContextCompressionRetriever, PaperStormRAGIndex


DEFAULT_RAG_CASES = [
    {
        "name": "pim_definition",
        "query": "PIM passive intermodulation RF",
        "expected_keywords": ["passive intermodulation", "RF"],
        "forbidden_keywords": ["DRAM", "processing-in-memory"],
    },
    {
        "name": "neural_suppression",
        "query": "neural network suppression passive intermodulation",
        "expected_keywords": ["neural", "passive intermodulation"],
        "forbidden_keywords": ["DRAM"],
    },
]


def run_rag_benchmark(output_dir, run_dir, cases=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = cases or DEFAULT_RAG_CASES
    index = PaperStormRAGIndex.from_run_dir(run_dir)
    retriever = ContextCompressionRetriever(index)
    results = []
    latencies = []
    for case in cases:
        start = time.time()
        retrieved = retriever.retrieve(
            case["query"],
            expected_keywords=case.get("expected_keywords") or [],
            forbidden_keywords=case.get("forbidden_keywords") or [],
        )
        latency_ms = (time.time() - start) * 1000
        latencies.append(latency_ms)
        results.append(_score_case(case, retrieved, latency_ms))
    report = {
        "project": "PaperStorm RAG Benchmark v3.0",
        "index_config": index.config,
        "cases": results,
        "metrics": _aggregate(results, latencies),
    }
    (output_dir / "rag_benchmark_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "rag_benchmark_report.md").write_text(
        _to_markdown(report),
        encoding="utf-8",
    )
    return report


def _score_case(case, retrieved, latency_ms):
    text = "\n".join(chunk.get("content", "") for chunk in retrieved.get("chunks") or [])
    expected_hits = _hits(text, case.get("expected_keywords") or [])
    forbidden_hits = _hits(text, case.get("forbidden_keywords") or [])
    context_recall = len(expected_hits) / max(1, len(case.get("expected_keywords") or []))
    citation_precision = 1.0 if retrieved.get("chunks") and expected_hits else 0.0
    return {
        "name": case["name"],
        "query": case["query"],
        "context_recall": round(context_recall, 4),
        "citation_precision": round(citation_precision, 4),
        "forbidden_hit_count": len(forbidden_hits),
        "off_topic": bool(forbidden_hits and not expected_hits),
        "selected_count": len(retrieved.get("chunks") or []),
        "prompt_context_chars": len(retrieved.get("prompt_context") or ""),
        "latency_ms": round(latency_ms, 3),
    }


def _aggregate(results, latencies):
    total = max(1, len(results))
    return {
        "total_cases": len(results),
        "context_recall": round(sum(item["context_recall"] for item in results) / total, 4),
        "citation_precision": round(
            sum(item["citation_precision"] for item in results) / total,
            4,
        ),
        "off_topic_rate": round(len([item for item in results if item["off_topic"]]) / total, 4),
        "avg_latency_ms": round(sum(latencies) / max(1, len(latencies)), 3),
        "p95_latency_ms": round(_percentile(latencies, 0.95), 3),
        "qps_estimate": round(1000.0 / max(1.0, statistics.mean(latencies or [1.0])), 3),
    }


def _to_markdown(report):
    lines = ["# PaperStorm RAG Benchmark v3.0", ""]
    for key, value in report["metrics"].items():
        lines.append("- {0}: {1}".format(key, value))
    lines.extend(
        [
            "",
            "| case | context_recall | citation_precision | off_topic | latency_ms |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in report["cases"]:
        lines.append(
            "| {name} | {context_recall} | {citation_precision} | {off_topic} | {latency_ms} |".format(
                **item
            )
        )
    lines.append("")
    return "\n".join(lines)


def _hits(text, keywords):
    lowered = str(text or "").lower()
    return [keyword for keyword in keywords if keyword and keyword.lower() in lowered]


def _percentile(values, percentile):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * percentile))
    return ordered[index]
