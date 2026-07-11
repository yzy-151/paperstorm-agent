"""Runtime retrieval stack selection and legacy-vs-v4.1 improvement benchmark.

The chat / knowledge-base path historically used the light local index
(token-overlap + hash-embedding + keyword rerank). This module wires the V4.1
stack (BM25 + dense + RRF, optional Cross-Encoder) into that path and provides
a repeatable benchmark that proves how much retrieval improved.

Env knobs:
    PAPERSTORM_RETRIEVAL_STACK     auto | v41 | legacy   (default auto)
    PAPERSTORM_RETRIEVAL_EMBEDDING auto | real | hash    (default auto)
    PAPERSTORM_RETRIEVAL_MODE      hybrid | bm25 | dense | hybrid_rerank (default hybrid)
    PAPERSTORM_EMBEDDING_MODEL     sentence-transformers model name
    PAPERSTORM_MODEL_CACHE         huggingface cache folder
"""

import argparse
import collections
import json
import math
import os
import statistics
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional


def runtime_stack(override: Optional[str] = None) -> str:
    value = (override or os.getenv("PAPERSTORM_RETRIEVAL_STACK") or "auto").strip().lower()
    if value not in {"auto", "v41", "legacy"}:
        value = "auto"
    if value == "auto":
        try:
            import sentence_transformers  # noqa: F401

            value = "v41"
        except Exception:
            value = "legacy"
    return value


def runtime_embedding(override: Optional[str] = None) -> str:
    value = (
        override or os.getenv("PAPERSTORM_RETRIEVAL_EMBEDDING") or "auto"
    ).strip().lower()
    if value not in {"auto", "real", "hash"}:
        value = "auto"
    if value == "auto":
        # Runtime path stays fast and dependency-free by default; set
        # PAPERSTORM_RETRIEVAL_EMBEDDING=real to switch to a sentence model.
        value = "hash"
    return value


def runtime_mode(override: Optional[str] = None) -> str:
    value = (override or os.getenv("PAPERSTORM_RETRIEVAL_MODE") or "hybrid").strip().lower()
    if value not in {"hybrid", "bm25", "dense", "hybrid_rerank"}:
        value = "hybrid"
    return value


_REAL_EMBEDDING_PROVIDER = None

_INDEX_LRU_LOCK = threading.Lock()


def _index_cache_maxsize() -> int:
    try:
        return max(0, int(os.getenv("PAPERSTORM_RETRIEVAL_INDEX_CACHE_SIZE", "16")))
    except ValueError:
        return 16


class _IndexLRU:
    """Small process-local LRU for built retrieval indexes.

    Follow-up questions reuse the same run dir; rebuilding the BM25 model and
    embeddings on every query is wasteful. The LRU is keyed by run dir,
    stack/embedding config and the run-dir file signature, so any change to
    the article or raw search results invalidates the entry naturally.
    """

    def __init__(self, maxsize: int):
        self.maxsize = max(int(maxsize or 0), 0)
        self._data = collections.OrderedDict()

    def get(self, key):
        with _INDEX_LRU_LOCK:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def put(self, key, value):
        with _INDEX_LRU_LOCK:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = value
            while self.maxsize and len(self._data) > self.maxsize:
                self._data.popitem(last=False)


_INDEX_LRU = _IndexLRU(_index_cache_maxsize())


def _run_dir_signature(run_dir):
    """mtime+size signature of the files that feed the runtime index."""
    parts = []
    for name in (
        "storm_gen_article_polished.txt",
        "storm_gen_article.txt",
        "raw_search_results.json",
    ):
        path = Path(run_dir) / name
        if path.exists():
            stat = path.stat()
            parts.append((name, stat.st_mtime_ns, stat.st_size))
    return tuple(parts)


def _dense_provider(embedding: str):
    global _REAL_EMBEDDING_PROVIDER
    if embedding == "hash":
        from .paperstorm_rag import HashEmbeddingProvider

        return HashEmbeddingProvider(dim=64)
    if _REAL_EMBEDDING_PROVIDER is None:
        from .paperstorm_retrieval_v41 import SentenceTransformerProvider

        _REAL_EMBEDDING_PROVIDER = SentenceTransformerProvider(
            model_name=os.getenv("PAPERSTORM_EMBEDDING_MODEL")
            or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            cache_folder=os.getenv("PAPERSTORM_MODEL_CACHE") or os.getenv("HF_HOME"),
        )
    return _REAL_EMBEDDING_PROVIDER


def build_runtime_index(
    run_dir,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    stack: Optional[str] = None,
    embedding: Optional[str] = None,
):
    """Build the runtime retrieval index for a research run dir."""
    stack = runtime_stack(stack)
    embedding = runtime_embedding(embedding)
    signature = _run_dir_signature(run_dir)
    cache_key = (
        str(Path(run_dir).resolve()),
        stack,
        embedding,
        int(chunk_size),
        int(chunk_overlap),
        signature,
    )
    cached = _INDEX_LRU.get(cache_key)
    if cached is not None:
        index, meta = cached
        return index, dict(meta, cached=True)
    provider = _dense_provider(embedding)
    if stack == "v41":
        from .paperstorm_retrieval_v41 import HybridPaperIndex

        index = HybridPaperIndex.from_run_dir(
            run_dir,
            embedding_provider=provider,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:
        from .paperstorm_rag import PaperStormRAGIndex

        index = PaperStormRAGIndex.from_run_dir(
            run_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_provider=provider,
        )
    meta = {"stack": stack, "embedding": embedding, "cached": False}
    _INDEX_LRU.put(cache_key, (index, meta))
    return index, meta


def search_runtime_index(
    run_dir,
    query: str,
    top_k: int = 3,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    stack: Optional[str] = None,
    embedding: Optional[str] = None,
    mode: Optional[str] = None,
    reranker=None,
) -> Dict:
    """Search a research run dir with the configured runtime stack."""
    mode = runtime_mode(mode)
    index, meta = build_runtime_index(
        run_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        stack=stack,
        embedding=embedding,
    )
    if meta["stack"] == "v41" and mode == "hybrid_rerank" and reranker is None:
        from .paperstorm_retrieval_v41 import CrossEncoderReranker

        reranker = CrossEncoderReranker(
            cache_folder=os.getenv("PAPERSTORM_MODEL_CACHE") or os.getenv("HF_HOME")
        )
    if meta["stack"] == "v41":
        results = index.search(query, mode=mode, top_k=top_k, reranker=reranker)
        results = _relevance_gate(results, query, meta["embedding"])
    else:
        results = index.search(query, top_k=top_k)
    return {
        "results": results,
        "stack": meta["stack"],
        "embedding": meta["embedding"],
        "mode": mode if meta["stack"] == "v41" else "legacy_hybrid",
        "cached": bool(meta.get("cached")),
    }


def _meaningful_query_terms(text: str):
    """Latin words plus CJK bigrams; single CJK characters are noise here."""
    from .paperstorm_retrieval_v41 import multilingual_tokenize

    tokens = multilingual_tokenize(str(text or "").lower())
    return {
        token
        for token in tokens
        if not (len(token) == 1 and "\u4e00" <= token <= "\u9fff")
    }


def _relevance_gate(results: List[Dict], query: str, embedding: str) -> List[Dict]:
    """Drop weakly related chunks so unrelated questions abstain instead of
    being answered from volume-scored evidence. A chunk passes when it has
    meaningful lexical overlap (words/CJK bigrams) or a strong dense score."""
    if not results:
        return results
    dense_min = 0.45 if embedding == "real" else 0.20
    query_terms = _meaningful_query_terms(query)
    kept = []
    for item in results:
        dense = float(item.get("dense_score") or 0)
        text = "{0}\n{1}".format(item.get("title", ""), item.get("content", ""))
        overlap = query_terms & _meaningful_query_terms(text)
        if overlap or dense >= dense_min:
            kept.append(item)
    return kept


def run_retrieval_benchmark(
    output_dir,
    top_k: int = 5,
    embedding: str = "hash",
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> Dict:
    """Compare the legacy runtime index against the V4.1 stack on the auditable
    100-case seed set and write a JSON + Markdown report with deltas."""
    from .paperstorm_eval_v4 import build_seed_dataset
    from .paperstorm_rag import HashEmbeddingProvider, PaperStormRAGIndex
    from .paperstorm_retrieval_v41 import HybridPaperIndex

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_seed_dataset()
    corpus = dataset.get("corpus") or []
    cases = [case for case in dataset.get("cases") or [] if case.get("expected_behavior") != "abstain"]
    provider = _dense_provider(embedding)
    legacy_index = PaperStormRAGIndex.from_documents(
        corpus,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_provider=provider,
    )
    v41_index = HybridPaperIndex.from_documents(
        corpus,
        embedding_provider=provider,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    def run(stack, index):
        hits, reciprocal, ndcg_scores, latencies = [], [], [], []
        for case in cases:
            query = str(case.get("query") or "")
            relevant = set(case.get("relevant_chunk_ids") or [])
            started = time.perf_counter()
            if stack == "v41":
                ranked = index.search(query, mode="hybrid", top_k=top_k)
            else:
                ranked = index.search(query, top_k=top_k)
            latencies.append((time.perf_counter() - started) * 1000)
            ranked_ids = [str(item.get("chunk_id") or "") for item in ranked[:top_k]]
            hit, mrr, ndcg = _retrieval_metrics(ranked_ids, relevant, top_k)
            hits.append(hit)
            reciprocal.append(mrr)
            ndcg_scores.append(ndcg)
        return {
            "recall_at_k": round(statistics.mean(hits), 6) if hits else 0.0,
            "mrr": round(statistics.mean(reciprocal), 6) if reciprocal else 0.0,
            "ndcg_at_k": round(statistics.mean(ndcg_scores), 6) if ndcg_scores else 0.0,
            "p95_latency_ms": round(_percentile(latencies, 0.95), 4) if latencies else 0.0,
            "case_count": len(cases),
        }

    legacy = run("legacy", legacy_index)
    v41 = run("v41", v41_index)
    deltas = {
        "recall_at_k": round(v41["recall_at_k"] - legacy["recall_at_k"], 6),
        "mrr": round(v41["mrr"] - legacy["mrr"], 6),
        "ndcg_at_k": round(v41["ndcg_at_k"] - legacy["ndcg_at_k"], 6),
        "p95_latency_ms": round(v41["p95_latency_ms"] - legacy["p95_latency_ms"], 4),
    }
    deltas["relative_recall_gain_pct"] = round(
        (v41["recall_at_k"] - legacy["recall_at_k"])
        / max(1e-9, legacy["recall_at_k"])
        * 100.0,
        2,
    )
    report = {
        "project": "PaperStorm Runtime Retrieval Benchmark",
        "dataset": dataset.get("dataset_version", ""),
        "top_k": top_k,
        "embedding": embedding,
        "stack_meta": {"legacy": "token_overlap + dense + keyword rerank",
                       "v41": "BM25 + dense + RRF (hybrid)"},
        "legacy": legacy,
        "v41": v41,
        "deltas": deltas,
    }
    (output_dir / "retrieval_runtime_benchmark.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "retrieval_runtime_benchmark.md").write_text(
        _to_markdown(report), encoding="utf-8"
    )
    return report


def _retrieval_metrics(ranked_ids: List[str], relevant: set, top_k: int):
    relevant = set(relevant)
    recall = int(any(chunk_id in relevant for chunk_id in ranked_ids[:top_k]))
    mrr = 0.0
    for rank, chunk_id in enumerate(ranked_ids[:top_k], start=1):
        if chunk_id in relevant:
            mrr = 1.0 / rank
            break
    ndcg = 0.0
    for rank, chunk_id in enumerate(ranked_ids[:top_k], start=1):
        if chunk_id in relevant:
            ndcg += 1.0 / math.log2(rank + 1)
    ideal = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, min(top_k, len(relevant)) + 1)
    )
    ndcg = ndcg / ideal if ideal else 0.0
    return recall, mrr, ndcg


def _percentile(values, quantile):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * quantile))))
    return ordered[index]


def _to_markdown(report: Dict) -> str:
    lines = [
        "# PaperStorm Runtime Retrieval Benchmark",
        "",
        "数据集：{0}（排除 abstain 用例后 {1} 条）".format(
            report.get("dataset", ""), report.get("legacy", {}).get("case_count", 0)
        ),
        "",
        "| 指标 | legacy（词法重叠+hash向量） | v4.1（BM25+Dense+RRF） | 差值 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, label in [
        ("recall_at_k", "Recall@K"),
        ("mrr", "MRR"),
        ("ndcg_at_k", "nDCG@K"),
        ("p95_latency_ms", "P95 延迟(ms)"),
    ]:
        legacy = report["legacy"][key]
        v41 = report["v41"][key]
        delta = report["deltas"][key]
        lines.append("| {0} | {1} | {2} | {3} |".format(label, legacy, v41, delta))
    lines.append("")
    lines.append("relative recall gain: {0}%".format(
        report.get("deltas", {}).get("relative_recall_gain_pct", 0.0)
    ))
    lines.append("")
    lines.append("注意：v4.1 行默认使用 hybrid（BM25+Dense+RRF），可加 Cross-Encoder 二次重排。")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Compare legacy vs V4.1 runtime retrieval on the auditable seed set."
    )
    parser.add_argument("--output-dir", default="./results/paperstorm_retrieval_runtime")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embedding", choices=["auto", "real", "hash"], default="auto")
    args = parser.parse_args()
    report = run_retrieval_benchmark(
        Path(args.output_dir),
        top_k=args.top_k,
        embedding=args.embedding,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
