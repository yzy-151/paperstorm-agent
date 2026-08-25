"""Runtime adapter for the unified PaperStorm retrieval pipeline.

Env knobs:
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
    value = str(override or "unified").strip().lower()
    if value not in {"unified", "hybrid"}:
        raise ValueError("legacy retrieval stacks were removed; use unified")
    return "unified"


def runtime_embedding(override: Optional[str] = None) -> str:
    value = (
        override or os.getenv("PAPERSTORM_RETRIEVAL_EMBEDDING") or "auto"
    ).strip().lower()
    if value not in {"auto", "real", "hash"}:
        value = "auto"
    # ``auto`` means the production-quality provider. Tests and deterministic
    # smoke runs must request ``hash`` explicitly; a missing model dependency
    # must fail clearly instead of silently changing retrieval semantics.
    return "real" if value == "auto" else value


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

STOP_BIGRAMS = frozenset(
    {
        "效果", "方法", "用于", "可以", "问题", "相关", "研究", "核心", "主要",
        "关键", "需要", "通过", "进行", "以及", "这个", "什么", "为什么", "怎么",
        "如何", "为啥", "这种", "还有", "针对", "关于", "结合", "基于", "从而",
        "因此", "并且", "然后", "应该", "能够", "可能", "是否", "一下", "这里",
        "哪些", "区别", "关系", "时候", "地方", "方面", "部分", "内容", "情况",
        "方式", "思路", "建议", "帮助", "请问", "谢谢", "你好", "您好", "哈哈",
        "没事", "是的", "对的", "不是", "没有", "知道", "了解", "明白", "感觉",
        "觉得", "认为", "其实", "真的", "还是", "能帮", "帮我", "给我", "我想",
    }
)


def meaningful_terms(text: str):
    """Substantive retrieval terms: Latin words + CJK words/bigrams, with
    common function/quality bigrams filtered out so one shared word like
    "效果" cannot make an off-topic knowledge base look relevant."""
    from .retrieval import multilingual_tokenize

    tokens = multilingual_tokenize(str(text or "").lower())
    terms = set()
    for token in tokens:
        if len(token) == 1 and "\u4e00" <= token <= "\u9fff":
            continue
        if token in STOP_BIGRAMS:
            continue
        terms.add(token)
    return terms


def _run_dir_signature(run_dir):
    """mtime+size signature of the files that feed the runtime index."""
    parts = []
    for name in (
        "storm_gen_article_polished.txt",
        "storm_gen_article.txt",
        "raw_search_results.json",
        "url_to_info.json",
    ):
        path = Path(run_dir) / name
        if path.exists():
            stat = path.stat()
            parts.append((name, stat.st_mtime_ns, stat.st_size))
    return tuple(parts)


def _dense_provider(embedding: str):
    global _REAL_EMBEDDING_PROVIDER
    if embedding == "hash":
        from .retrieval import HashEmbeddingProvider

        return HashEmbeddingProvider(dim=64)
    if _REAL_EMBEDDING_PROVIDER is None:
        from .retrieval import SentenceTransformerProvider

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
    from .retrieval import HybridPaperIndex

    index = HybridPaperIndex.from_run_dir(
        run_dir,
        embedding_provider=provider,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    meta = {"stack": "unified", "embedding": embedding, "cached": False}
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
    history=(),
    search_plan=None,
    parent_budget_tokens: int = 0,
    metadata_filters=None,
    expected_keywords=(),
    forbidden_keywords=(),
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
    if mode == "hybrid_rerank" and reranker is None:
        from .retrieval import CrossEncoderReranker

        reranker = CrossEncoderReranker(
            cache_folder=os.getenv("PAPERSTORM_MODEL_CACHE") or os.getenv("HF_HOME")
        )
    from .retrieval_pipeline import RetrievalPipeline, RetrievalRequest

    pipeline = RetrievalPipeline(
        index=index,
        reranker=reranker,
        relevance_gate=lambda results, value: _relevance_gate(
            results, value, meta["embedding"]
        ),
    )
    outcome = pipeline.search(
        RetrievalRequest(
            query=query,
            top_k=top_k,
            candidate_k=max(top_k * 4, 20),
            mode=mode,
            enable_reranker=mode == "hybrid_rerank",
            history=tuple(history or ()),
            search_plan=search_plan,
            parent_budget_tokens=max(0, int(parent_budget_tokens)),
            metadata_filters=dict(metadata_filters or {}),
            expected_keywords=tuple(expected_keywords or ()),
            forbidden_keywords=tuple(forbidden_keywords or ()),
        )
    )
    outcome.update(
        {
            "stack": "retrieval_pipeline",
            "embedding": meta["embedding"],
            "cached": bool(meta.get("cached")),
        }
    )
    return outcome


def _meaningful_query_terms(text: str):
    """Back-compat alias of meaningful_terms (Latin words + CJK words/bigrams,
    single CJK characters and common function words are noise)."""
    return meaningful_terms(text)


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
