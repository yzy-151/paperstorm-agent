"""Single retrieval contract shared by PaperStorm product and benchmarks."""

import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Tuple

from .retrieval import reciprocal_rank_fusion
from .search_planning import SearchPlan, SearchPlanner


class RetrievalCapabilityError(RuntimeError):
    """Raised when an index cannot satisfy a requested retrieval stage."""


@dataclass(frozen=True)
class RetrievalRequest:
    query: str
    top_k: int = 5
    candidate_k: int = 20
    mode: str = "hybrid"
    expected_keywords: Tuple[str, ...] = ()
    forbidden_keywords: Tuple[str, ...] = ()
    enable_reranker: bool = False
    search_plan: Optional[SearchPlan] = None
    history: Tuple[Mapping, ...] = ()
    parent_budget_tokens: int = 0
    metadata_filters: Mapping = field(default_factory=dict)


class RetrievalPipeline:
    """Execute planned retrieval through a stable, auditable stage schema."""

    schema_revision = 2

    def __init__(
        self,
        index,
        reranker=None,
        relevance_gate: Optional[Callable] = None,
        search_planner: Optional[SearchPlanner] = None,
    ):
        self.index = index
        self.reranker = reranker
        self.relevance_gate = relevance_gate
        self.search_planner = search_planner or SearchPlanner()

    def search(self, request: RetrievalRequest):
        query = " ".join(str(request.query or "").split())
        if not query:
            raise ValueError("query is required")
        started = time.perf_counter()
        plan_started = time.perf_counter()
        plan = request.search_plan or self.search_planner.plan(
            query, history=request.history
        )
        if not isinstance(plan, SearchPlan):
            raise TypeError("search_planner must return SearchPlan")
        if plan.original_query != query:
            raise ValueError("search_plan.original_query must match request.query")
        parent_budget = int(request.parent_budget_tokens)
        if parent_budget < 0:
            raise ValueError("parent_budget_tokens must not be negative")
        expand_parent = getattr(self.index, "expand_parent_context", None)
        if parent_budget > 0 and not callable(expand_parent):
            raise RetrievalCapabilityError(
                "index does not support parent context expansion"
            )
        queries = _planned_queries(plan)
        stages = [
            _stage(
                "plan",
                "completed",
                _elapsed_ms(plan_started),
                1,
                len(queries),
                "standalone query plus up to three unique subqueries",
            )
        ]

        rerank_enabled = bool(request.enable_reranker or request.mode == "hybrid_rerank")
        mode = "hybrid_rerank" if rerank_enabled else request.mode
        if rerank_enabled and self.reranker is None:
            raise ValueError("hybrid_rerank mode requires reranker")
        cheap_mode = "hybrid" if rerank_enabled else request.mode
        retrieve_started = time.perf_counter()
        candidate_k = max(int(request.candidate_k), int(request.top_k), 1)
        rankings = []
        for planned_query in queries:
            rankings.append(
                list(
                    self.index.search(
                        planned_query,
                        mode=cheap_mode,
                        top_k=candidate_k,
                        candidate_k=candidate_k,
                        reranker=None,
                        parent_budget_tokens=0,
                    )
                )
            )
        retrieved_count = sum(len(ranking) for ranking in rankings)
        stages.append(
            _stage(
                "retrieve",
                "completed",
                _elapsed_ms(retrieve_started),
                len(queries),
                retrieved_count,
                "{0} planned queries; mode={1}; candidate_k={2}".format(
                    len(queries), cheap_mode, candidate_k
                ),
            )
        )

        fuse_started = time.perf_counter()
        results = reciprocal_rank_fusion(rankings)[:candidate_k]
        stages.append(
            _stage(
                "fuse",
                "completed",
                _elapsed_ms(fuse_started),
                retrieved_count,
                len(results),
                "RRF by stable chunk_id across {0} ranking(s)".format(
                    len(rankings)
                ),
            )
        )

        if rerank_enabled:
            rerank_started = time.perf_counter()
            rerank_input = len(results)
            if hasattr(self.reranker, "rerank"):
                results = list(
                    self.reranker.rerank(
                        plan.standalone_query, results, top_k=candidate_k
                    )
                )
            else:
                results = list(self.reranker(plan.standalone_query, results))[
                    :candidate_k
                ]
            stages.append(
                _stage(
                    "rerank",
                    "completed",
                    _elapsed_ms(rerank_started),
                    rerank_input,
                    len(results),
                    "one cross-encoder pass over fused candidates",
                )
            )

        gate_started = time.perf_counter()
        gate_input = len(results)
        results = _metadata_filter(
            results, dict(plan.filters), dict(request.metadata_filters)
        )
        if self.relevance_gate is not None:
            results = list(self.relevance_gate(results, plan.standalone_query))
        expected = _normalized_terms(request.expected_keywords)
        if expected:
            results = [
                item
                for item in results
                if any(marker in _search_text(item).lower() for marker in expected)
            ]
        forbidden = _normalized_terms(
            tuple(request.forbidden_keywords) + tuple(plan.negative_terms)
        )
        if forbidden:
            results = [
                item
                for item in results
                if not any(
                    marker in _search_text(item).lower() for marker in forbidden
                )
            ]
        must_terms = _normalized_terms(plan.must_terms)
        if must_terms:
            results = [
                item
                for item in results
                if all(marker in _search_text(item).lower() for marker in must_terms)
            ]
        results = results[: max(1, int(request.top_k))]
        results = _rewrite_final_ranking(
            results, score_key="rerank_score" if rerank_enabled else "rrf_score"
        )
        filters_active = bool(
            self.relevance_gate
            or expected
            or forbidden
            or must_terms
            or plan.filters
            or request.metadata_filters
        )
        gate_stage = _stage(
            "gate",
            "completed" if filters_active else "skipped",
            _elapsed_ms(gate_started),
            gate_input,
            len(results),
            "metadata, must, expected, negative and forbidden evidence gates",
        )

        parent_started = time.perf_counter()
        parent_input = len(results)
        if parent_budget > 0 and results:
            results = list(expand_parent(results, parent_budget))
        expanded_count = sum(bool(item.get("parent_context")) for item in results)
        stages.append(
            _stage(
                "parent_expand",
                "completed" if parent_budget > 0 else "skipped",
                _elapsed_ms(parent_started),
                parent_input,
                len(results),
                "budget_tokens={0}; expanded={1}".format(
                    parent_budget, expanded_count
                ),
            )
        )
        stages.append(gate_stage)
        provider = getattr(self.index, "embedding_provider", None)
        return {
            "schema": "paperstorm-retrieval-result",
            "schema_revision": self.schema_revision,
            "query": query,
            "search_plan": plan.to_dict(),
            "results": results,
            "stages": stages,
            "models": {
                "embedding": str(getattr(provider, "name", "unknown")),
                "reranker": str(getattr(self.reranker, "model_name", "")),
            },
            "mode": mode,
            "latency_ms": _elapsed_ms(started),
        }


def _planned_queries(plan):
    output, seen = [], set()
    for query in (plan.standalone_query,) + tuple(plan.subqueries[:3]):
        normalized = str(query or "").strip()
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            output.append(normalized)
    return tuple(output)


def _metadata_filter(results, plan_filters, request_filters):
    filters = dict(plan_filters)
    filters.update(request_filters)
    if not filters:
        return list(results)
    return [
        item
        for item in results
        if _metadata_matches(item.get("metadata") or {}, filters)
    ]


def _metadata_matches(metadata, filters):
    for key, expected in filters.items():
        actual = metadata.get(key)
        if isinstance(actual, (list, tuple, set)):
            if isinstance(expected, (list, tuple, set)):
                if not any(value in actual for value in expected):
                    return False
            elif expected not in actual:
                return False
        elif isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def _normalized_terms(values):
    return tuple(
        str(value).strip().lower() for value in values if str(value).strip()
    )


def _stage(name, status, latency_ms, input_count, output_count, reason):
    return {
        "name": name,
        "status": status,
        "input_count": int(input_count),
        "output_count": int(output_count),
        "latency_ms": round(float(latency_ms), 3),
        "reason": str(reason),
    }


def _elapsed_ms(started):
    return (time.perf_counter() - started) * 1000.0


def _search_text(item):
    values = [item.get("title", ""), item.get("content", "")]
    values.extend(_searchable_metadata_values(item.get("metadata") or {}))
    return "\n".join(str(value) for value in values)


def _searchable_metadata_values(value):
    if isinstance(value, Mapping):
        output = []
        for key, item in value.items():
            output.append(str(key))
            output.extend(_searchable_metadata_values(item))
        return output
    if isinstance(value, (list, tuple, set, frozenset)):
        output = []
        for item in value:
            output.extend(_searchable_metadata_values(item))
        return output
    return ["" if value is None else str(value)]


def _rewrite_final_ranking(results, score_key):
    output = []
    for rank, item in enumerate(results, start=1):
        enriched = dict(item)
        score = float(
            enriched.get(
                score_key,
                enriched.get("rrf_score", enriched.get("score", 0.0)),
            )
        )
        enriched["final_rank"] = rank
        enriched["final_score"] = round(score, 8)
        enriched["score"] = round(score, 8)
        output.append(enriched)
    return output
