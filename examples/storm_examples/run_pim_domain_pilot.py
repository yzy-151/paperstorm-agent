"""Run the private 50-case PIM retrieval and real-vector ANN pilot."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

try:
    from examples.storm_examples.benchmark_embedding_profiles import run_profile_benchmark
except ModuleNotFoundError as exc:  # Direct script execution from the repository root.
    if exc.name not in {"examples", "examples.storm_examples"}:
        raise
    from benchmark_embedding_profiles import run_profile_benchmark
from knowledge_storm.dense_index import ExactDenseBackend, HnswDenseBackend
from knowledge_storm.evaluation.domain_pilot import (
    DOMAIN_CATEGORIES,
    load_domain_dataset,
)
from knowledge_storm.retrieval import SentenceTransformerProvider
from knowledge_storm.retrieval_profiles import get_embedding_profile


DEFAULT_PROFILES = ("legacy-multilingual", "cpu-zh", "cpu-multilingual")


def select_best_profile(reports, top_k=5):
    recall_key = "recall_at_{0}".format(int(top_k))
    if not reports:
        raise ValueError("at least one profile report is required")
    return min(
        reports,
        key=lambda name: (
            -float(reports[name]["metrics"].get(recall_key, 0.0)),
            float(reports[name]["metrics"].get("query_p95_ms", float("inf"))),
            name,
        ),
    )


def run_real_vector_ann(dataset, provider, output_dir, top_k=5):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    texts = [document.text for document in dataset.documents]
    started = time.perf_counter()
    vectors = provider.embed_documents(texts)
    embedding_seconds = time.perf_counter() - started
    started = time.perf_counter()
    exact = ExactDenseBackend(vectors)
    exact_build_seconds = time.perf_counter() - started
    started = time.perf_counter()
    hnsw = HnswDenseBackend(vectors, ef_search=240, ef_construction=300, m=24)
    hnsw_build_seconds = time.perf_counter() - started
    index_path = root / "pim-real-vectors.usearch"
    hnsw.save(index_path)

    exact_latencies, hnsw_latencies, recalls = [], [], []
    rows = []
    for case in dataset.cases:
        query = provider.embed_query(case.query)
        started = time.perf_counter()
        exact_result = exact.search(query, top_k)
        exact_latencies.append((time.perf_counter() - started) * 1000.0)
        started = time.perf_counter()
        hnsw_result = hnsw.search(query, top_k)
        hnsw_latencies.append((time.perf_counter() - started) * 1000.0)
        denominator = min(int(top_k), len(exact_result.indices), len(hnsw_result.indices))
        recall = len(set(exact_result.indices) & set(hnsw_result.indices)) / max(1, denominator)
        recalls.append(recall)
        rows.append(
            {
                "case_id": case.case_id,
                "exact_indices": list(exact_result.indices),
                "hnsw_indices": list(hnsw_result.indices),
                "ann_recall_at_k": recall,
            }
        )
    report = {
        "schema": "paperstorm-pim-real-ann-v1",
        "evidence_tier": "measured_private_domain_pilot",
        "model_name": provider.model_name,
        "vector_count": len(vectors),
        "dimension": len(vectors[0]),
        "query_count": len(dataset.cases),
        "top_k": int(top_k),
        "embedding_seconds": round(embedding_seconds, 6),
        "exact_build_seconds": round(exact_build_seconds, 6),
        "hnsw_build_seconds": round(hnsw_build_seconds, 6),
        "exact_p50_ms": _percentile(exact_latencies, 0.50),
        "exact_p95_ms": _percentile(exact_latencies, 0.95),
        "hnsw_p50_ms": _percentile(hnsw_latencies, 0.50),
        "hnsw_p95_ms": _percentile(hnsw_latencies, 0.95),
        "hnsw_recall_at_k": round(statistics.mean(recalls), 6),
        "hnsw_index_bytes": index_path.stat().st_size,
        "predictions": rows,
    }
    _write_json(root / "metrics.json", report)
    return report


def run_pilot(corpus_path, cases_path, output_dir, model_cache, profiles=DEFAULT_PROFILES, top_k=5):
    dataset = load_domain_dataset(
        corpus_path,
        cases_path,
        expected_case_count=50,
        required_categories=DOMAIN_CATEGORIES,
    )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    reports = {}
    providers = {}
    for profile_name in profiles:
        provider = SentenceTransformerProvider(
            profile=get_embedding_profile(profile_name), cache_folder=model_cache, device="cpu"
        )
        providers[profile_name] = provider
        result = run_profile_benchmark(
            dataset,
            profile_name=profile_name,
            embedding_provider=provider,
            output_dir=root / "retrieval",
            sample_ratio=1.0,
            seed=55,
            top_k=top_k,
        )
        reports[profile_name] = result["report"]
    best = select_best_profile(reports, top_k=top_k)
    ann = run_real_vector_ann(
        dataset, providers[best], root / "ann" / best, top_k=top_k
    )
    summary = {
        "schema": "paperstorm-pim-domain-pilot-v1",
        "case_count": len(dataset.cases),
        "document_count": len(dataset.documents),
        "top_k": int(top_k),
        "profiles": {
            name: {
                "model_name": reports[name]["manifest"]["model_name"],
                "metrics": reports[name]["metrics"],
                "resources": reports[name]["resources"],
            }
            for name in profiles
        },
        "selected_profile": best,
        "selection_rule": "max Recall@K, then min query P95",
        "ann": {key: value for key, value in ann.items() if key != "predictions"},
    }
    _write_json(root / "summary.json", summary)
    _write_json(root / "metrics.json", summary)
    return summary


def _percentile(values, fraction):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, int(len(ordered) * fraction + 0.999999) - 1))
    return round(ordered[index], 6)


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-cache", required=True)
    parser.add_argument("--profiles", nargs="+", choices=DEFAULT_PROFILES, default=DEFAULT_PROFILES)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)
    summary = run_pilot(
        args.corpus,
        args.cases,
        args.output_dir,
        args.model_cache,
        profiles=tuple(args.profiles),
        top_k=args.top_k,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
