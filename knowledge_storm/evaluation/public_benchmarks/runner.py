"""Shared PaperStorm retrieval runner for public datasets."""

import hashlib
import json
import random
import statistics
import subprocess
import time
from datetime import datetime, timezone

from ...paperstorm_retrieval_v41 import HybridPaperIndex
from .metrics import retrieval_metrics
from .report import write_benchmark_artifacts


class HashEmbeddingProvider:
    """Deterministic offline embedding for smoke tests, never headline results."""

    normalize = True

    def __init__(self, dim=128):
        self.dim = int(dim)
        self.name = "paperstorm-hash-smoke-{0}".format(self.dim)

    def embed(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        vector = [0.0] * self.dim
        for token in str(text or "").lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            vector[index] += 1.0 if digest[4] % 2 else -1.0
        norm = sum(value * value for value in vector) ** 0.5
        return [value / norm for value in vector] if norm else vector


def run_retrieval_benchmark(
    dataset,
    embedding_provider,
    modes=("bm25", "dense", "hybrid"),
    top_k=10,
    reranker=None,
    bootstrap_samples=2000,
    seed=55,
    output_dir=None,
    cache_state="warm_query_after_cold_index",
    scope_field=None,
):
    top_k = max(1, int(top_k))
    cases = tuple(dataset.cases)
    chunks = [
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
    index_started = time.perf_counter()
    index = HybridPaperIndex(chunks, embedding_provider=embedding_provider)
    scoped_indices = _build_scoped_indices(index, scope_field)
    index_time_ms = (time.perf_counter() - index_started) * 1000.0
    predictions = []
    bad_cases = []
    mode_reports = {}
    for mode in modes:
        per_case = []
        for case in cases:
            case_index = index
            if scope_field:
                scope_value = str(case.metadata.get(scope_field) or "")
                if scope_value not in scoped_indices:
                    raise ValueError(
                        "case {0} has unknown {1}: {2}".format(
                            case.case_id, scope_field, scope_value
                        )
                    )
                case_index = scoped_indices[scope_value]
            started = time.perf_counter()
            ranked = case_index.search(
                case.query,
                mode=mode,
                top_k=top_k,
                candidate_k=max(top_k * 4, 20),
                reranker=reranker,
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            ranked_ids = [item["document_id"] for item in ranked]
            metrics = retrieval_metrics(ranked_ids, case.relevance, cutoffs=(top_k,))
            row = {
                "benchmark": dataset.name,
                "case_id": case.case_id,
                "split": case.split,
                "mode": mode,
                "query": case.query,
                "ranked_document_ids": ranked_ids,
                "relevant_document_ids": list(case.relevant_document_ids),
                "latency_ms": round(latency_ms, 4),
                "metrics": metrics,
            }
            predictions.append(row)
            per_case.append(dict(metrics, latency_ms=latency_ms))
            if metrics.get("recall_at_{0}".format(top_k), 0.0) < 1.0:
                bad_cases.append(row)
        mode_reports[mode] = _summarize(
            per_case, top_k=top_k, bootstrap_samples=bootstrap_samples, seed=seed
        )
    git_commit, working_tree_dirty = _git_state()
    manifest = {
        "benchmark": dataset.name,
        "dataset_version": dataset.version,
        "split": _split_label(cases),
        "case_count": len(cases),
        "document_count": len(dataset.documents),
        "corpus_sha256": _corpus_hash(dataset.documents),
        "git_commit": git_commit,
        "working_tree_dirty": working_tree_dirty,
        "embedding_model": str(getattr(embedding_provider, "name", "unknown")),
        "modes": list(modes),
        "top_k": top_k,
        "seed": seed,
        "bootstrap_samples": bootstrap_samples,
        "cache_state": cache_state,
        "scope_field": scope_field,
        "index_time_ms": round(index_time_ms, 4),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence_tier": "public_official",
        "limitations": [
            "Retrieval metrics do not measure final answer faithfulness.",
            "Latency is machine-specific and must not be presented as an online SLA.",
        ],
    }
    report = {
        "benchmark": dataset.name,
        "case_count": len(cases),
        "modes": mode_reports,
        "manifest": manifest,
    }
    if output_dir is not None:
        write_benchmark_artifacts(output_dir, manifest, report, predictions, bad_cases)
    return report


def _build_scoped_indices(index, scope_field):
    if not scope_field:
        return {}
    grouped = {}
    for position, chunk in enumerate(index.chunks):
        value = str((chunk.get("metadata") or {}).get(scope_field) or "")
        grouped.setdefault(value, []).append(position)
    return {
        value: HybridPaperIndex(
            [index.chunks[position] for position in positions],
            embedding_provider=index.embedding_provider,
            embeddings=[index.embeddings[position] for position in positions],
        )
        for value, positions in grouped.items()
    }


def _summarize(rows, top_k, bootstrap_samples, seed):
    names = (
        "recall_at_{0}".format(top_k),
        "mrr_at_{0}".format(top_k),
        "ndcg_at_{0}".format(top_k),
    )
    output = {"case_count": len(rows)}
    intervals = {}
    for offset, name in enumerate(names):
        values = [float(row[name]) for row in rows]
        output[name] = round(statistics.mean(values), 6) if values else 0.0
        intervals[name] = _bootstrap_ci(
            values, samples=bootstrap_samples, seed=seed + offset
        )
    latencies = sorted(float(row["latency_ms"]) for row in rows)
    output["p50_latency_ms"] = (
        round(statistics.median(latencies), 4) if latencies else 0.0
    )
    output["p95_latency_ms"] = round(_nearest_rank(latencies, 0.95), 4)
    output["confidence_intervals"] = intervals
    return output


def _bootstrap_ci(values, samples, seed):
    if not values:
        return {"low": 0.0, "high": 0.0, "n": 0, "samples": samples}
    generator = random.Random(seed)
    means = sorted(
        statistics.mean(generator.choice(values) for _ in values)
        for _ in range(max(1, int(samples)))
    )
    return {
        "low": round(means[int(0.025 * (len(means) - 1))], 6),
        "high": round(means[int(0.975 * (len(means) - 1))], 6),
        "n": len(values),
        "samples": max(1, int(samples)),
    }


def _nearest_rank(values, fraction):
    if not values:
        return 0.0
    index = max(0, min(len(values) - 1, int(len(values) * fraction + 0.999999) - 1))
    return values[index]


def _corpus_hash(documents):
    payload = [
        {"document_id": item.document_id, "title": item.title, "text": item.text}
        for item in documents
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _split_label(cases):
    values = sorted({case.split for case in cases})
    return values[0] if len(values) == 1 else "+".join(values)


def _git_state():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return commit, bool(status)
    except (OSError, subprocess.SubprocessError):
        return "unknown", None
