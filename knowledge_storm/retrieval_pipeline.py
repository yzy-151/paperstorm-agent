"""Single retrieval contract shared by PaperStorm product and benchmark paths."""

import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple


@dataclass(frozen=True)
class RetrievalRequest:
    query: str
    top_k: int = 5
    candidate_k: int = 20
    mode: str = "hybrid"
    expected_keywords: Tuple[str, ...] = ()
    forbidden_keywords: Tuple[str, ...] = ()
    enable_reranker: bool = False


class RetrievalPipeline:
    """Execute retrieval through a stable, auditable stage schema."""

    schema_revision = 1

    def __init__(self, index, reranker=None, relevance_gate: Optional[Callable] = None):
        self.index = index
        self.reranker = reranker
        self.relevance_gate = relevance_gate

    def search(self, request: RetrievalRequest):
        query = str(request.query or "").strip()
        if not query:
            raise ValueError("query is required")
        started = time.perf_counter()
        rerank_enabled = bool(request.enable_reranker or request.mode == "hybrid_rerank")
        mode = "hybrid_rerank" if rerank_enabled else request.mode
        retrieve_started = time.perf_counter()
        results = self.index.search(
            query,
            mode=mode,
            top_k=max(1, int(request.top_k)),
            candidate_k=max(int(request.candidate_k), int(request.top_k)),
            reranker=self.reranker if rerank_enabled else None,
        )
        retrieval_ms = _elapsed_ms(retrieve_started)
        stages = [
            _stage("retrieve", "completed", retrieval_ms, len(results)),
            _stage(
                "fuse",
                "completed" if mode in {"hybrid", "hybrid_rerank"} else "skipped",
                0.0,
                len(results),
            ),
            _stage(
                "rerank",
                "completed" if rerank_enabled else "skipped",
                0.0,
                len(results),
            ),
        ]
        gate_started = time.perf_counter()
        if self.relevance_gate is not None:
            results = list(self.relevance_gate(results, query))
            gate_status = "completed"
        else:
            gate_status = "skipped"
        if request.expected_keywords:
            expected = tuple(
                value.lower() for value in request.expected_keywords if value.strip()
            )
            results = [
                item
                for item in results
                if any(marker in _search_text(item).lower() for marker in expected)
            ]
            gate_status = "completed"
        if request.forbidden_keywords:
            forbidden = tuple(
                value.lower() for value in request.forbidden_keywords if value.strip()
            )
            results = [
                item
                for item in results
                if not any(
                    marker in _search_text(item).lower() for marker in forbidden
                )
            ]
            gate_status = "completed"
        stages.append(_stage("gate", gate_status, _elapsed_ms(gate_started), len(results)))
        provider = getattr(self.index, "embedding_provider", None)
        return {
            "schema": "paperstorm-retrieval-result",
            "schema_revision": self.schema_revision,
            "query": query,
            "results": results,
            "stages": stages,
            "models": {
                "embedding": str(getattr(provider, "name", "unknown")),
                "reranker": str(getattr(self.reranker, "model_name", "")),
            },
            "mode": mode,
            "latency_ms": _elapsed_ms(started),
        }


def _stage(name, status, latency_ms, output_count):
    return {
        "name": name,
        "status": status,
        "latency_ms": round(float(latency_ms), 3),
        "output_count": int(output_count),
    }


def _elapsed_ms(started):
    return (time.perf_counter() - started) * 1000.0


def _search_text(item):
    return "{0}\n{1}".format(item.get("title", ""), item.get("content", ""))
