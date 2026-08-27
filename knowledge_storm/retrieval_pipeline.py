"""Single retrieval contract shared by PaperStorm product and benchmarks."""

import math
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Tuple

from .evidence_governance import select_evidence
from .retrieval import reciprocal_rank_fusion
from .search_planning import SearchPlan, SearchPlanner, normalize_filter_mapping


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
    governance_features: Mapping = field(default_factory=dict)
    tenant_id: str = ""
    user_id: str = ""
    policy_digest: str = ""
    allowed_document_ids: Optional[Tuple[str, ...]] = None
    allowed_chunk_ids: Optional[Tuple[str, ...]] = None


class RetrievalPipeline:
    """Execute planned retrieval through a stable, auditable stage schema."""

    schema_revision = 2

    def __init__(
        self,
        index,
        reranker=None,
        relevance_gate: Optional[Callable] = None,
        search_planner: Optional[SearchPlanner] = None,
        rerank_policy=None,
        evidence_assessor=None,
    ):
        self.index = index
        self.reranker = reranker
        self.relevance_gate = relevance_gate
        self.search_planner = search_planner or SearchPlanner()
        self.rerank_policy = rerank_policy
        self.evidence_assessor = evidence_assessor

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
        request_filters = dict(
            normalize_filter_mapping(request.metadata_filters, "metadata_filters")
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

        requested_rerank = bool(
            request.enable_reranker or request.mode == "hybrid_rerank"
        )
        governance_enabled = (
            self.rerank_policy is not None or self.evidence_assessor is not None
        )
        cheap_mode = (
            "hybrid"
            if requested_rerank or self.rerank_policy is not None
            else request.mode
        )
        retrieve_started = time.perf_counter()
        candidate_k = max(int(request.candidate_k), int(request.top_k), 1)
        rankings = []
        for planned_query in queries:
            search_kwargs = {
                "mode": cheap_mode,
                "top_k": candidate_k,
                "candidate_k": candidate_k,
                "reranker": None,
                "parent_budget_tokens": 0,
            }
            # Keep third-party/legacy index adapters compatible until a scope is
            # explicitly requested; scoped callers must receive pre-retrieval ACL.
            if request.allowed_document_ids is not None:
                search_kwargs["allowed_document_ids"] = request.allowed_document_ids
            if request.allowed_chunk_ids is not None:
                search_kwargs["allowed_chunk_ids"] = request.allowed_chunk_ids
            rankings.append(
                list(
                    self.index.search(planned_query, **search_kwargs)
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
        results = (
            list(rankings[0])
            if len(rankings) == 1
            else reciprocal_rank_fusion(rankings)
        )[:candidate_k]
        stages.append(
            _stage(
                "fuse",
                "completed",
                _elapsed_ms(fuse_started),
                retrieved_count,
                len(results),
                (
                    "preserved inner hybrid scores for one planned query"
                    if len(rankings) == 1
                    else "RRF by stable chunk_id across {0} rankings".format(len(rankings))
                ),
            )
        )

        rerank_decision = None
        rerank_enabled = requested_rerank
        rerank_limit = 0
        rerank_elapsed_ms = 0.0
        if self.rerank_policy is not None:
            policy_started = time.perf_counter()
            policy_features = dict(request.governance_features)
            policy_features["candidate_count"] = len(results)
            policy_features.setdefault("rrf_margin", _rrf_margin(results))
            policy_features.setdefault("answer_risk", 0.0)
            policy_features.setdefault("bm25_dense_overlap", 1.0)
            policy_features.setdefault("cache_state", "unknown")
            policy_features.setdefault("observed_p95_ms", 0.0)
            rerank_decision = self.rerank_policy.decide(policy_features)
            rerank_enabled = rerank_decision.enabled
            stages.append(
                _stage(
                    "policy",
                    "completed",
                    _elapsed_ms(policy_started),
                    len(results),
                    len(results),
                    rerank_decision.reason,
                )
            )
        mode = "hybrid_rerank" if rerank_enabled else (
            "hybrid" if requested_rerank else request.mode
        )
        if rerank_enabled and self.reranker is None:
            raise ValueError("hybrid_rerank mode requires reranker")

        if rerank_enabled:
            rerank_started = time.perf_counter()
            rerank_input = len(results)
            policy_limit = (
                min(len(results), int(self.rerank_policy.max_candidates))
                if self.rerank_policy is not None
                else len(results)
            )
            profile_limit = int(getattr(self.reranker, "max_candidates", 20) or 20)
            rerank_limit = min(policy_limit, profile_limit)
            rerank_candidates = results[:rerank_limit]
            if hasattr(self.reranker, "rerank"):
                reranked = list(
                    self.reranker.rerank(
                        plan.standalone_query,
                        rerank_candidates,
                        top_k=rerank_limit,
                    )
                )
                results = _normalize_reranker_results(
                    reranked, candidate_k=candidate_k, accept_score=False
                )
            else:
                reranked = list(
                    self.reranker(plan.standalone_query, rerank_candidates)
                )
                results = _normalize_reranker_results(
                    reranked, candidate_k=candidate_k, accept_score=True
                )
            rerank_elapsed_ms = _elapsed_ms(rerank_started)
            stages.append(
                _stage(
                    "rerank",
                    "completed",
                    rerank_elapsed_ms,
                    rerank_input,
                    len(results),
                    "one cross-encoder pass over fused candidates",
                )
            )

        gate_started = time.perf_counter()
        gate_input = len(results)
        results = _metadata_filter(
            results, dict(plan.filters), request_filters
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
        coverage_score = None
        if governance_enabled:
            coverage_started = time.perf_counter()
            top_k = max(1, int(request.top_k))
            selected = select_evidence(results[:top_k], top_k=top_k)
            coverage_score = selected.coverage_score
            results = list(selected)
            stages.append(
                _stage(
                    "coverage",
                    "completed",
                    _elapsed_ms(coverage_started),
                    gate_input,
                    len(results),
                    "recall-safe MMR within top-k; coverage_score={0:.6f}".format(
                        coverage_score
                    ),
                )
            )
        else:
            results = results[: max(1, int(request.top_k))]
        results = _rewrite_final_ranking(
            results,
            score_key="rerank_score" if rerank_enabled else "rrf_score",
            retrieval_mode=mode,
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
            "completed",
            _elapsed_ms(gate_started),
            gate_input,
            len(results),
            "top_k selection and final rank rewrite"
            + (
                "; metadata, must, expected, negative or forbidden filters applied"
                if filters_active
                else ""
            ),
        )
        stages.append(gate_stage)

        evidence_assessment = None
        if self.evidence_assessor is not None:
            assessment_started = time.perf_counter()
            evidence_assessment = self.evidence_assessor.assess(
                query, results, coverage_score=coverage_score
            )
            stages.append(
                _stage(
                    "assessment",
                    "completed",
                    _elapsed_ms(assessment_started),
                    len(results),
                    len(results),
                    evidence_assessment.next_action,
                )
            )

        parent_started = time.perf_counter()
        parent_input = len(results)
        if parent_budget > 0 and results:
            results = list(expand_parent(results, parent_budget))
        expanded_count = sum(bool(item.get("parent_context")) for item in results)
        parent_allocations = [
            item.get("parent_allocation") or {}
            for item in results
            if item.get("parent_allocation")
        ]
        parent_allocated_tokens = sum(
            int(item.get("allocated_tokens") or 0) for item in parent_allocations
        )
        parent_used_tokens = sum(
            int(item.get("used_tokens") or 0) for item in parent_allocations
        )
        stages.append(
            _stage(
                "parent_expand",
                "completed" if expanded_count > 0 else "skipped",
                _elapsed_ms(parent_started),
                parent_input,
                len(results),
                "budget_tokens={0}; expanded={1}".format(
                    parent_budget, expanded_count
                ),
            )
        )
        provider = getattr(self.index, "embedding_provider", None)
        output = {
            "schema": "paperstorm-retrieval-result",
            "schema_revision": self.schema_revision,
            "query": query,
            "search_plan": plan.to_dict(),
            "results": results,
            "stages": stages,
            "models": {
                "embedding": str(getattr(provider, "name", "unknown")),
                "reranker": str(getattr(self.reranker, "model_name", "")),
                "reranker_profile": str(
                    getattr(getattr(self.reranker, "profile", None), "name", "")
                ),
                "reranker_device": str(
                    getattr(self.reranker, "actual_device", "")
                ),
            },
            "mode": mode,
            "parent_context": {
                "budget_tokens": parent_budget,
                "allocated_tokens": parent_allocated_tokens,
                "used_tokens": parent_used_tokens,
                "expanded_parent_count": expanded_count,
            },
            "latency_ms": _elapsed_ms(started),
            "policy_digest": str(request.policy_digest or ""),
            "candidate_scope": {
                "allowed_document_count": (
                    None
                    if request.allowed_document_ids is None
                    else len(request.allowed_document_ids)
                ),
                "allowed_chunk_count": (
                    None
                    if request.allowed_chunk_ids is None
                    else len(request.allowed_chunk_ids)
                ),
            },
            "rerank_runtime": {
                "enabled": bool(rerank_enabled),
                "candidate_count": int(rerank_limit if rerank_enabled else 0),
                "candidate_limit": int(
                    getattr(self.reranker, "max_candidates", 0) or 0
                ),
                "batch_size": int(
                    getattr(self.reranker, "batch_size", 0) or 0
                ),
                "elapsed_ms": rerank_elapsed_ms,
            },
        }
        if rerank_decision is not None:
            output["rerank_decision"] = rerank_decision.to_dict()
        if evidence_assessment is not None:
            output["evidence_assessment"] = evidence_assessment.to_dict()
        return output


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
        if _item_matches_filters(item, filters)
    ]


_FILTER_PATHS = {
    "authors": (("authors",), ("metadata", "authors")),
    "document_id": (("document_id",), ("metadata", "document_id")),
    "domain": (("domain",), ("metadata", "domain")),
    "published": (("published",), ("metadata", "published")),
    "query": (("query",), ("metadata", "query")),
    "section": (("section",), ("metadata", "section")),
    "source": (
        ("source",),
        ("source_type",),
        ("metadata", "source"),
        ("metadata", "source_type"),
    ),
    "source_type": (("source_type",), ("metadata", "source_type")),
    "tags": (("tags",), ("metadata", "tags")),
    "venue": (("venue",), ("metadata", "venue")),
}
_YEAR_PATHS = (
    ("year",),
    ("published",),
    ("metadata", "year"),
    ("metadata", "published"),
)


def _item_matches_filters(item, filters):
    year_filters = {
        key: filters[key]
        for key in ("year", "year_from", "year_to")
        if key in filters
    }
    if year_filters:
        years = _item_years(item)
        if not any(_year_matches(year, year_filters) for year in years):
            return False
    for key, expected in filters.items():
        if key in {"year", "year_from", "year_to"}:
            continue
        actual = _path_values(item, _FILTER_PATHS[key])
        if not _matches_any(actual, expected):
            return False
    return True


def _year_matches(year, filters):
    return (
        ("year" not in filters or _matches_any((year,), filters["year"]))
        and ("year_from" not in filters or year >= filters["year_from"])
        and ("year_to" not in filters or year <= filters["year_to"])
    )


_MISSING = object()


def _path_values(item, paths):
    output = []
    for path in paths:
        value = item
        for part in path:
            if not isinstance(value, Mapping) or part not in value:
                value = _MISSING
                break
            value = value[part]
        if value is _MISSING:
            continue
        if isinstance(value, (list, tuple, set, frozenset)):
            output.extend(value)
        else:
            output.append(value)
    return output


def _item_years(item):
    years = []
    for value in _path_values(item, _YEAR_PATHS):
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            years.append(value)
            continue
        match = re.match(r"^\s*(\d{4})", str(value))
        if match:
            years.append(int(match.group(1)))
    return years


def _matches_any(actual_values, expected):
    expected_values = (
        tuple(expected)
        if isinstance(expected, (list, tuple, set, frozenset))
        else (expected,)
    )
    return any(
        _filter_scalar_equal(actual, wanted)
        for actual in actual_values
        for wanted in expected_values
    )


def _filter_scalar_equal(actual, expected):
    if isinstance(actual, str) and isinstance(expected, str):
        return actual.casefold() == expected.casefold()
    return actual == expected


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


def _rrf_margin(results):
    if len(results) < 2:
        return 1.0
    first = float(results[0].get("rrf_score", results[0].get("score", 0.0)))
    second = float(results[1].get("rrf_score", results[1].get("score", 0.0)))
    if not math.isfinite(first) or not math.isfinite(second):
        return 0.0
    return min(1.0, abs(first - second) / max(abs(first), 1e-12))


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


def _normalize_reranker_results(results, *, candidate_k, accept_score):
    output = []
    for item in results:
        enriched = dict(item)
        value = enriched.get("rerank_score")
        if value is None and accept_score:
            value = enriched.get("score")
        try:
            score = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("reranker result requires a numeric score") from exc
        if not math.isfinite(score):
            raise ValueError("reranker score must be finite")
        enriched["rerank_score"] = round(score, 8)
        output.append(enriched)
    output.sort(
        key=lambda item: (-item["rerank_score"], str(item.get("chunk_id") or ""))
    )
    return output[:candidate_k]


def _rewrite_final_ranking(results, score_key, retrieval_mode):
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
        enriched["retrieval_mode"] = retrieval_mode
        output.append(enriched)
    return output
