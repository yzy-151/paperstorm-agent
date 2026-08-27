"""Deterministic, resumable public retrieval comparison for embedding profiles.

Only query cases are sampled.  Every selected case is always searched against the
complete benchmark corpus, so the sampled run remains a valid smaller query set
rather than an easier reduced-corpus task.
"""

import argparse
import hashlib
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from knowledge_storm.evaluation.public_benchmarks.base import BenchmarkDataset
from knowledge_storm.evaluation.public_benchmarks.beir_scifact import load_scifact
from knowledge_storm.evaluation.public_benchmarks.metrics import retrieval_metrics
from knowledge_storm.evaluation.public_benchmarks.qasper import load_qasper_official_json
from knowledge_storm.retrieval import HybridPaperIndex, SentenceTransformerProvider
from knowledge_storm.retrieval_profiles import get_embedding_profile


DEFAULT_PROFILES = (
    "legacy-multilingual",
    "cpu-zh",
    "cpu-multilingual",
    "quality-multilingual",
)
REQUIRED_RESOURCE_METRICS = frozenset(
    {
        "build_seconds",
        "query_p50_ms",
        "query_p95_ms",
        "rss_peak_bytes",
        "index_bytes",
        "embedding_dimension",
    }
)


def stable_sample_cases(cases, sample_ratio=0.1, seed=55):
    """Return an order-independent SHA-256 sample of query cases.

    Sorting stable hashes instead of iterating a pseudo-random generator prevents
    changing loader order from changing the sampled public evaluation set.
    """
    cases = tuple(cases)
    if not cases:
        return ()
    ratio = float(sample_ratio)
    if not 0.0 < ratio <= 1.0:
        raise ValueError("sample_ratio must be in (0, 1]")
    count = max(1, int(round(len(cases) * ratio)))
    keyed = []
    for case in cases:
        identity = "{0}\\0{1}\\0{2}".format(seed, case.case_id, case.query)
        keyed.append((hashlib.sha256(identity.encode("utf-8")).hexdigest(), case))
    return tuple(case for _, case in sorted(keyed, key=lambda item: (item[0], item[1].case_id))[:count])


def sample_dataset_cases(dataset, ratio=0.1, seed=55):
    """Return a dataset with sampled cases and the untouched candidate corpus."""
    return BenchmarkDataset(
        name=dataset.name,
        version=dataset.version,
        documents=dataset.documents,
        cases=stable_sample_cases(dataset.cases, sample_ratio=ratio, seed=seed),
        metadata=dict(dataset.metadata, query_sample_ratio=float(ratio), sample_seed=int(seed)),
    )


def build_report(rows, manifest):
    """Summarize retrieved ranks and local machine resource measurements."""
    top_k = int(manifest["top_k"])
    metric_names = tuple(
        "{0}_at_{1}".format(prefix, top_k) for prefix in ("recall", "mrr", "ndcg")
    )
    latencies = sorted(float(row.get("latency_ms", 0.0)) for row in rows)
    metrics = {
        name: _mean(
            [float(row.get("metrics", row).get(name, 0.0)) for row in rows]
        )
        for name in metric_names
    }
    metrics.update({
        "case_count": len(rows),
        "query_p50_ms": round(statistics.median(latencies), 4) if latencies else 0.0,
        "query_p95_ms": round(_nearest_rank(latencies, 0.95), 4),
    })
    resources = {
        "build_seconds": round(float(manifest.get("build_seconds", manifest.get("build_time_ms", 0.0) / 1000.0)), 6),
        "rss_peak_bytes": int(manifest.get("rss_peak_bytes", manifest.get("rss_bytes", 0))),
        "index_bytes": int(manifest.get("index_bytes", 0)),
        "embedding_dimension": int(manifest.get("embedding_dimension", manifest.get("dimension", 0))),
    }
    reproducibility = {
        "profile": str(manifest.get("profile", "")),
        "model_name": str(manifest.get("model_name", manifest.get("model", ""))),
        "model_revision": manifest.get("model_revision"),
        "fingerprint": manifest.get("fingerprint"),
        "seed": manifest.get("seed"),
        "sample_ratio": manifest.get("sample_ratio"),
    }
    return {
        "manifest": manifest,
        "metrics": metrics,
        "resources": resources,
        "reproducibility": reproducibility,
        "predictions": list(rows),
    }


def load_completed_report(output_dir, expected_fingerprint):
    root = Path(output_dir)
    manifest = _read_json(root / "manifest.json") or {}
    if manifest.get("status") not in {"complete", "completed"}:
        return None
    if manifest.get("fingerprint") != expected_fingerprint:
        return None
    return _read_json(root / "metrics.json")


def run_profile_benchmark(
    dataset,
    profile_name,
    embedding_provider,
    output_dir,
    sample_ratio=0.1,
    seed=55,
    top_k=5,
):
    """Run or resume one profile, producing an independently auditable directory."""
    if not isinstance(dataset, BenchmarkDataset):
        raise TypeError("dataset must be a BenchmarkDataset")
    root = Path(output_dir) / _safe_path_part(profile_name)
    root.mkdir(parents=True, exist_ok=True)
    selected_cases = stable_sample_cases(dataset.cases, sample_ratio, seed)
    run_contract = _run_contract(dataset, selected_cases, profile_name, embedding_provider, sample_ratio, seed, top_k)
    manifest_path = root / "manifest.json"
    metrics_path = root / "metrics.json"
    predictions_path = root / "predictions.jsonl"

    completed = load_completed_report(root, run_contract["fingerprint"])
    if completed is not None and predictions_path.is_file():
        return {
            "status": "resumed",
            "output_dir": str(root),
            "manifest": completed.get("manifest", run_contract),
            "report": completed,
        }
    existing_manifest = _read_json(manifest_path)
    if existing_manifest:
        _assert_resume_compatible(existing_manifest, run_contract)

    manifest = dict(run_contract, status="running", started_at=_now())
    _write_json(manifest_path, manifest)
    rows_by_id = {row["case_id"]: row for row in _read_jsonl(predictions_path)}

    indexed_documents = [
        {
            "chunk_id": document.document_id,
            "document_id": document.document_id,
            "title": document.title,
            "content": document.text,
            "retrieval_content": document.text,
            "metadata": dict(document.metadata),
        }
        for document in dataset.documents
    ]
    build_started = time.perf_counter()
    index = HybridPaperIndex(
        indexed_documents,
        embedding_provider=embedding_provider,
        dense_backend_mode="exact",
    )
    manifest["build_seconds"] = round(time.perf_counter() - build_started, 6)
    manifest["dense_backend_mode"] = "exact"
    manifest["embedding_dimension"] = int(index.manifest.get("embedding_dimension", 0))
    manifest["index_bytes"] = _embedding_payload_bytes(index.embeddings)
    manifest["rss_peak_bytes"] = _rss_bytes()
    _write_json(manifest_path, manifest)

    for case in selected_cases:
        if case.case_id in rows_by_id:
            continue
        started = time.perf_counter()
        allowed_document_ids = _case_scope_document_ids(dataset, case)
        results = index.search(
            case.query,
            mode="hybrid",
            top_k=int(top_k),
            candidate_k=max(int(top_k) * 4, 20),
            allowed_document_ids=allowed_document_ids,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        ranked_ids = [str(item["document_id"]) for item in results]
        ranks = {
            document_id: rank
            for rank, document_id in enumerate(ranked_ids, start=1)
            if document_id in case.relevance
        }
        rows_by_id[case.case_id] = {
            "benchmark": dataset.name,
            "case_id": case.case_id,
            "query": case.query,
            "split": case.split,
            "ranked_document_ids": ranked_ids,
            "relevant_document_ids": list(case.relevant_document_ids),
            "relevant_document_ranks": ranks,
            "candidate_scope": "paper" if allowed_document_ids is not None else "full_corpus",
            "candidate_count": len(allowed_document_ids) if allowed_document_ids is not None else len(dataset.documents),
            "latency_ms": round(latency_ms, 4),
            "metrics": retrieval_metrics(ranked_ids, case.relevance, cutoffs=(int(top_k),)),
        }
        _write_jsonl(predictions_path, _ordered_rows(rows_by_id, selected_cases))

    rows = _ordered_rows(rows_by_id, selected_cases)
    manifest.update(status="complete", finished_at=_now(), rss_peak_bytes=max(int(manifest.get("rss_peak_bytes", 0)), _rss_bytes()))
    report = build_report(rows, manifest)
    _write_json(manifest_path, manifest)
    _write_json(metrics_path, report)
    _write_comparison_markdown(root / "comparison.md", report)
    return {"status": "completed", "output_dir": str(root), "manifest": manifest, "report": report}


def build_parser():
    parser = argparse.ArgumentParser(description="Compare PaperStorm embedding profiles on deterministic public query samples")
    parser.add_argument("--benchmarks", nargs="+", choices=("scifact", "qasper"), default=("scifact", "qasper"))
    parser.add_argument("--profiles", nargs="+", choices=DEFAULT_PROFILES, default=DEFAULT_PROFILES)
    parser.add_argument("--sample-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--model-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scifact-dir")
    parser.add_argument("--qasper-path")
    parser.add_argument("--scifact-split", default="test")
    parser.add_argument("--qasper-split", default="test")
    parser.add_argument("--top-k", type=int, default=5)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not 0.0 < args.sample_ratio <= 1.0:
        raise SystemExit("--sample-ratio must be in (0, 1]")
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {"benchmarks": {}, "profiles": list(args.profiles), "sample_ratio": args.sample_ratio, "seed": args.seed}
    for benchmark_name in args.benchmarks:
        dataset = _load_dataset(args, benchmark_name)
        benchmark_root = output_root / benchmark_name
        summary["benchmarks"][benchmark_name] = {}
        for profile_name in args.profiles:
            profile = get_embedding_profile(profile_name)
            provider = SentenceTransformerProvider(profile=profile, cache_folder=args.model_cache)
            result = run_profile_benchmark(
                dataset,
                profile_name=profile_name,
                embedding_provider=provider,
                output_dir=benchmark_root,
                sample_ratio=args.sample_ratio,
                seed=args.seed,
                top_k=args.top_k,
            )
            summary["benchmarks"][benchmark_name][profile_name] = {
                "status": result["status"],
                "output_dir": result["output_dir"],
            }
        _write_cross_profile_comparison(benchmark_root, summary["benchmarks"][benchmark_name])
    _write_json(output_root / "summary.json", summary)
    return summary


def _load_dataset(args, benchmark_name):
    if benchmark_name == "scifact":
        if not args.scifact_dir:
            raise FileNotFoundError("--scifact-dir is required when scifact is selected")
        return load_scifact(args.scifact_dir, split=args.scifact_split)
    if not args.qasper_path:
        raise FileNotFoundError("--qasper-path is required when qasper is selected")
    source = load_qasper_official_json(args.qasper_path, split=args.qasper_split)
    cases = tuple(case for case in source.cases if case.relevant_document_ids)
    return BenchmarkDataset(source.name, source.version, source.documents, cases, source.metadata)


def _run_contract(dataset, cases, profile_name, provider, sample_ratio, seed, top_k):
    profile = getattr(provider, "profile", None)
    contract = {
        "benchmark": dataset.name,
        "dataset_version": dataset.version,
        "document_count": len(dataset.documents),
        "full_case_count": len(dataset.cases),
        "sample_case_count": len(cases),
        "sample_case_ids": [case.case_id for case in cases],
        "sample_ratio": float(sample_ratio),
        "seed": int(seed),
        "top_k": int(top_k),
        "profile": profile_name,
        "model_name": str(getattr(provider, "model_name", getattr(provider, "name", "unknown"))),
        "model_revision": getattr(profile, "revision", None),
        "profile_contract": profile.manifest_contract() if profile is not None else None,
        "reranker": {"enabled": False, "reason": "embedding_profile_comparison"},
        "corpus_sha256": _corpus_sha256(dataset),
        "query_sample_sha256": _query_sample_sha256(cases),
    }
    contract["fingerprint"] = _sha256_json(contract)
    return contract


def _assert_resume_compatible(existing, contract):
    keys = (
        "benchmark", "dataset_version", "document_count", "sample_case_ids",
        "sample_ratio", "seed", "top_k", "profile", "model_name", "model_revision",
        "profile_contract", "reranker", "corpus_sha256", "query_sample_sha256",
    )
    mismatches = [key for key in keys if existing.get(key) != contract.get(key)]
    if mismatches:
        raise ValueError("existing benchmark run is incompatible: " + ", ".join(mismatches))


def _ordered_rows(rows_by_id, cases):
    return [rows_by_id[case.case_id] for case in cases if case.case_id in rows_by_id]


def _write_comparison_markdown(path, report):
    manifest = report["manifest"]
    metrics = report["metrics"]
    lines = [
        "# Embedding Profile Benchmark",
        "",
        "| Field | Value |",
        "| --- | --- |",
        "| Profile | `{0}` |".format(manifest["profile"]),
        "| Model | `{0}` |".format(manifest["model_name"]),
        "| Revision | `{0}` |".format(manifest.get("model_revision") or "unversioned"),
        "| Query sample / full corpus | {0} / {1} |".format(manifest["sample_case_count"], manifest["document_count"]),
        "| Reranker | disabled |",
        "",
        "| Recall@K | MRR@K | nDCG@K | Build s | Query P50 ms | Query P95 ms | RSS bytes | Index bytes | Dim |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| {0:.6f} | {1:.6f} | {2:.6f} | {3:.2f} | {4:.2f} | {5:.2f} | {6} | {7} | {8} |".format(
            metrics["recall_at_{0}".format(manifest["top_k"])],
            metrics["mrr_at_{0}".format(manifest["top_k"])],
            metrics["ndcg_at_{0}".format(manifest["top_k"])],
            report["resources"]["build_seconds"],
            metrics["query_p50_ms"],
            metrics["query_p95_ms"],
            report["resources"]["rss_peak_bytes"],
            report["resources"]["index_bytes"],
            report["resources"]["embedding_dimension"],
        ),
        "",
        "This comparison samples only query cases via stable SHA-256; the candidate corpus is complete.",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_cross_profile_comparison(root, statuses):
    rows = []
    for profile_name in sorted(statuses):
        payload = _read_json(Path(root) / _safe_path_part(profile_name) / "metrics.json") or {}
        metrics = payload.get("metrics") or {}
        rows.append((profile_name, metrics, payload.get("manifest") or {}, payload.get("resources") or {}))
    lines = ["# Embedding Profile Comparison", "", "| Profile | Recall@K | MRR@K | nDCG@K | P95 ms | Dim |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for profile_name, metrics, manifest, resources in rows:
        top_k = int(manifest.get("top_k", 5))
        lines.append("| {0} | {1:.6f} | {2:.6f} | {3:.6f} | {4:.2f} | {5} |".format(profile_name, float(metrics.get("recall_at_{0}".format(top_k), 0.0)), float(metrics.get("mrr_at_{0}".format(top_k), 0.0)), float(metrics.get("ndcg_at_{0}".format(top_k), 0.0)), float(metrics.get("query_p95_ms", 0.0)), int(resources.get("embedding_dimension", 0))))
    (Path(root) / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _corpus_sha256(dataset):
    payload = [{"id": item.document_id, "title": item.title, "text": item.text} for item in dataset.documents]
    return _sha256_json(payload)


def _query_sample_sha256(cases):
    payload = [{"id": item.case_id, "query": item.query, "relevance": sorted(item.relevance.items())} for item in cases]
    return _sha256_json(payload)


def _sha256_json(payload):
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _nearest_rank(values, fraction):
    if not values:
        return 0.0
    return values[max(0, min(len(values) - 1, int(len(values) * fraction + 0.999999) - 1))]


def _mean(values):
    return round(statistics.mean(values), 6) if values else 0.0


def _rss_bytes():
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except ImportError:
        pass
    if os.name == "nt":
        import ctypes

        counters = ctypes.create_string_buffer(72)
        size = ctypes.sizeof(counters)
        ctypes.memset(counters, 0, size)
        ctypes.cast(counters, ctypes.POINTER(ctypes.c_ulong))[0] = size
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(handle, counters, size):
            return int(ctypes.cast(counters, ctypes.POINTER(ctypes.c_size_t))[2])
        return 0


def _embedding_payload_bytes(embeddings):
    try:
        import numpy as np

        return int(np.asarray(embeddings, dtype=np.float32).nbytes)
    except ImportError:
        return sum(len(row) for row in embeddings) * 4


def _case_scope_document_ids(dataset, case):
    paper_id = str(case.metadata.get("paper_id") or "")
    if not paper_id:
        return None
    return [
        document.document_id
        for document in dataset.documents
        if str(document.metadata.get("paper_id") or "") == paper_id
    ]
    try:
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except (ImportError, AttributeError):
        return 0


def _read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _write_jsonl(path, rows):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    os.replace(temporary, path)


def _safe_path_part(value):
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in str(value))


def _now():
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
