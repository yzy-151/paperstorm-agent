"""Measure Exact versus USearch HNSW and label 2M values as estimates."""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from knowledge_storm.dense_index import ExactDenseBackend, HnswDenseBackend


def estimate_flat_bytes(vector_count, dimension, bytes_per_value=4):
    return int(vector_count) * int(dimension) * int(bytes_per_value)


def build_scale_report(measured, estimated_count=2_000_000):
    dimension = int(measured["dimension"])
    estimated_count = int(estimated_count)
    return {
        "schema": "paperstorm-dense-scale-v1",
        "measured": dict(measured, evidence_tier="measured_local"),
        "estimated": {
            "vector_count": estimated_count,
            "dimension": dimension,
            "flat_vector_bytes": estimate_flat_bytes(estimated_count, dimension),
            "evidence_tier": "estimate",
            "limitations": [
                "Latency is not extrapolated from the measured run.",
                "HNSW graph bytes depend on connectivity, allocator, and implementation.",
            ],
        },
    }


def run_benchmark(
    vector_count=100_000,
    dimension=384,
    query_count=200,
    top_k=10,
    seed=55,
    output_dir=None,
):
    rng = np.random.default_rng(int(seed))
    vectors = rng.normal(size=(int(vector_count), int(dimension))).astype(np.float32)
    queries = rng.normal(size=(int(query_count), int(dimension))).astype(np.float32)

    started = time.perf_counter()
    exact = ExactDenseBackend(vectors)
    exact_build_seconds = time.perf_counter() - started
    started = time.perf_counter()
    hnsw = HnswDenseBackend(vectors, ef_search=1200, ef_construction=400, m=32)
    hnsw_build_seconds = time.perf_counter() - started
    del vectors

    exact_latencies = []
    hnsw_latencies = []
    recalls = []
    for query in queries:
        started = time.perf_counter()
        exact_result = exact.search(query, int(top_k))
        exact_latencies.append((time.perf_counter() - started) * 1000.0)
        started = time.perf_counter()
        hnsw_result = hnsw.search(query, int(top_k))
        hnsw_latencies.append((time.perf_counter() - started) * 1000.0)
        recalls.append(
            len(set(exact_result.indices) & set(hnsw_result.indices))
            / max(1, int(top_k))
        )

    root = Path(output_dir) if output_dir else None
    index_bytes = 0
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
        index_path = root / "vectors.usearch"
        hnsw.save(index_path)
        index_bytes = index_path.stat().st_size + Path(
            str(index_path) + ".meta.json"
        ).stat().st_size
    measured = {
        "vector_count": int(vector_count),
        "dimension": int(dimension),
        "query_count": int(query_count),
        "top_k": int(top_k),
        "seed": int(seed),
        "exact_build_seconds": round(exact_build_seconds, 6),
        "hnsw_build_seconds": round(hnsw_build_seconds, 6),
        "exact_p50_ms": _percentile(exact_latencies, 0.50),
        "exact_p95_ms": _percentile(exact_latencies, 0.95),
        "hnsw_p50_ms": _percentile(hnsw_latencies, 0.50),
        "hnsw_p95_ms": _percentile(hnsw_latencies, 0.95),
        "hnsw_recall_at_k": round(statistics.mean(recalls), 6),
        "flat_vector_bytes": estimate_flat_bytes(vector_count, dimension),
        "hnsw_index_bytes": int(index_bytes),
        "rss_bytes": _rss_bytes(),
        "backend_implementation": "USearch HNSW",
        "hnsw_config": {"m": 32, "ef_construction": 400, "ef_search": 1200},
    }
    report = build_scale_report(measured)
    if root is not None:
        (root / "metrics.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        _write_markdown(root / "comparison.md", report)
    return report


def _percentile(values, fraction):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, int(len(ordered) * fraction + 0.999999) - 1))
    return round(ordered[index], 6)


def _rss_bytes():
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except ImportError:
        return 0


def _write_markdown(path, report):
    measured = report["measured"]
    estimated = report["estimated"]
    lines = [
        "# Dense Retrieval Scale Benchmark",
        "",
        "## Measured locally",
        "",
        "| Vectors | Dim | Exact P95 ms | HNSW P95 ms | HNSW Recall@K | HNSW bytes |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        "| {vector_count} | {dimension} | {exact_p95_ms:.3f} | {hnsw_p95_ms:.3f} | {hnsw_recall_at_k:.4f} | {hnsw_index_bytes} |".format(**measured),
        "",
        "## Capacity estimate only",
        "",
        "{0:,} vectors at dimension {1} require {2:,} bytes for raw float32 vectors. No 2M latency claim is made.".format(
            estimated["vector_count"], estimated["dimension"], estimated["flat_vector_bytes"]
        ),
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(description="Benchmark Exact versus USearch HNSW")
    parser.add_argument("--vectors", type=int, default=100_000)
    parser.add_argument("--dimension", type=int, default=384)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    report = run_benchmark(
        vector_count=args.vectors,
        dimension=args.dimension,
        query_count=args.queries,
        top_k=args.top_k,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main()
