"""Deterministic evidence governance for selective reranking and answer safety."""

import math
import re
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence


_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
_ACTIONS = frozenset(
    {
        "rewrite",
        "expand_candidates",
        "switch_source",
        "abstain",
        "present_conflict",
        "answer",
    }
)


@dataclass(frozen=True)
class RerankDecision:
    enabled: bool
    reason: str
    candidate_count: int
    model: str
    latency_budget_ms: int

    def to_dict(self):
        return asdict(self)


class RerankPolicy:
    """Enable one fused-candidate rerank only for risky, uncertain requests."""

    def __init__(
        self,
        model="cross-encoder",
        max_p95_ms=None,
        high_risk_threshold=0.7,
        low_overlap_threshold=0.5,
        small_margin_threshold=0.08,
    ):
        self.model = str(model)
        self.max_p95_ms = max_p95_ms
        self.high_risk_threshold = float(high_risk_threshold)
        self.low_overlap_threshold = float(low_overlap_threshold)
        self.small_margin_threshold = float(small_margin_threshold)

    def decide(self, features):
        values = dict(features or {})
        candidate_count = max(0, int(values.get("candidate_count", 0)))
        budget = values.get("latency_budget_ms", self.max_p95_ms)
        budget = 0 if budget is None else max(0, int(budget))
        observed_p95 = _number(values.get("observed_p95_ms"), 0.0)
        if budget and observed_p95 > budget:
            return RerankDecision(
                False,
                "latency_budget_exceeded",
                candidate_count,
                self.model,
                budget,
            )
        if candidate_count < 2:
            return RerankDecision(
                False, "insufficient_candidates", candidate_count, self.model, budget
            )
        risk = _number(values.get("answer_risk"), 0.0)
        overlap = _number(values.get("bm25_dense_overlap"), 1.0)
        margin = _number(values.get("rrf_margin"), 1.0)
        uncertain = (
            overlap < self.low_overlap_threshold
            or margin < self.small_margin_threshold
        )
        if risk >= self.high_risk_threshold and uncertain:
            return RerankDecision(
                True, "high_risk_uncertain_evidence", candidate_count, self.model, budget
            )
        return RerankDecision(
            False,
            "risk_or_evidence_is_sufficient",
            candidate_count,
            self.model,
            budget,
        )


class EvidenceSelection(list):
    """Selected evidence retaining the aggregate coverage score."""

    def __init__(self, values=(), coverage_score=0.0):
        super().__init__(values)
        self.coverage_score = round(float(coverage_score), 6)


def select_evidence(candidates, top_k, lambda_mmr=0.65):
    """Select relevant but provenance-diverse evidence with stable MMR ties."""
    limit = max(0, int(top_k))
    if not candidates or not limit:
        return EvidenceSelection()
    tradeoff = min(1.0, max(0.0, float(lambda_mmr)))
    remaining = [dict(item) for item in candidates]
    remaining.sort(key=_candidate_order)
    selected = []
    used_parents, used_sources = set(), set()
    while remaining and len(selected) < limit:
        eligible = [
            item
            for item in remaining
            if _parent_key(item) not in used_parents
            and _source_key(item) not in used_sources
        ]
        pool = eligible or [
            item for item in remaining if _parent_key(item) not in used_parents
        ] or remaining
        choice = max(
            pool,
            key=lambda item: (
                round(
                    tradeoff * _relevance(item)
                    - (1.0 - tradeoff) * _max_similarity(item, selected),
                    12,
                ),
                _relevance(item),
                _stable_id(item),
            ),
        )
        selected.append(choice)
        remaining.remove(choice)
        used_parents.add(_parent_key(choice))
        used_sources.add(_source_key(choice))
    score = _coverage_score(selected)
    return EvidenceSelection(
        [dict(item, coverage_score=round(score, 6)) for item in selected], score
    )


@dataclass(frozen=True)
class EvidenceConflict:
    claim: str
    source: tuple
    relation: str
    conditions: tuple

    def to_dict(self):
        return {
            "claim": self.claim,
            "source": list(self.source),
            "relation": self.relation,
            "conditions": [dict(value) for value in self.conditions],
        }


@dataclass(frozen=True)
class EvidenceAssessment:
    relevance: float
    coverage: float
    answerability: float
    conflicts: tuple
    confidence: float
    failure_type: str
    next_action: str
    max_corrections: int = 1

    def __post_init__(self):
        if self.next_action not in _ACTIONS:
            raise ValueError("next_action is not supported")
        if self.max_corrections != 1:
            raise ValueError("max_corrections must be 1")

    def to_dict(self):
        return {
            "relevance": self.relevance,
            "coverage": self.coverage,
            "answerability": self.answerability,
            "conflicts": [conflict.to_dict() for conflict in self.conflicts],
            "confidence": self.confidence,
            "failure_type": self.failure_type,
            "next_action": self.next_action,
            "max_corrections": self.max_corrections,
        }


class EvidenceAssessor:
    def __init__(self, min_relevance=0.2, min_coverage=0.5):
        self.min_relevance = float(min_relevance)
        self.min_coverage = float(min_coverage)

    def assess(self, query, evidence, coverage_score=None, correction_count=0):
        values = list(evidence or ())
        if not values:
            return EvidenceAssessment(
                0.0, 0.0, 0.0, (), 0.0, "no_evidence", "abstain"
            )
        relevance = _assessment_relevance(query, values)
        coverage = (
            _number(coverage_score, _coverage_score(values))
            if coverage_score is not None
            else _coverage_score(values)
        )
        conflicts = tuple(_find_conflicts(values))
        answerability = min(relevance, coverage)
        confidence = answerability if not conflicts else min(answerability, 0.25)
        if conflicts:
            return EvidenceAssessment(
                _round(relevance),
                _round(coverage),
                _round(answerability),
                conflicts,
                _round(confidence),
                "conflict",
                "present_conflict",
            )
        if relevance < self.min_relevance:
            action = "abstain" if correction_count >= 1 else "rewrite"
            return EvidenceAssessment(
                _round(relevance),
                _round(coverage),
                _round(answerability),
                (),
                _round(confidence),
                "low_relevance",
                action,
            )
        if coverage < self.min_coverage:
            action = "abstain" if correction_count >= 1 else "expand_candidates"
            return EvidenceAssessment(
                _round(relevance),
                _round(coverage),
                _round(answerability),
                (),
                _round(confidence),
                "insufficient_coverage",
                action,
            )
        return EvidenceAssessment(
            _round(relevance),
            _round(coverage),
            _round(answerability),
            (),
            _round(confidence),
            "",
            "answer",
        )


def _number(value, default):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _round(value):
    return round(float(value), 6)


def _candidate_order(item):
    return (-_relevance(item), _stable_id(item))


def _stable_id(item):
    return str(item.get("chunk_id") or item.get("id") or "")


def _relevance(item):
    for key in ("rerank_score", "rrf_score", "score", "final_score"):
        if key in item:
            return _number(item[key], 0.0)
    return 0.0


def _parent_key(item):
    metadata = item.get("metadata") or {}
    return str(item.get("parent_id") or metadata.get("parent_id") or _stable_id(item))


def _source_key(item):
    metadata = item.get("metadata") or {}
    return str(
        item.get("source")
        or item.get("source_id")
        or item.get("document_id")
        or metadata.get("source")
        or metadata.get("source_id")
        or _parent_key(item)
    )


def _tokens(item):
    text = " ".join(str(item.get(key, "")) for key in ("title", "content", "claim"))
    return frozenset(_TOKEN_RE.findall(text.casefold()))


def _max_similarity(candidate, selected):
    return max((_similarity(candidate, item) for item in selected), default=0.0)


def _similarity(left, right):
    left_embedding, right_embedding = left.get("embedding"), right.get("embedding")
    if _is_vector(left_embedding) and _is_vector(right_embedding):
        return _cosine(left_embedding, right_embedding)
    left_tokens, right_tokens = _tokens(left), _tokens(right)
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def _is_vector(value):
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and bool(value)


def _cosine(left, right):
    if len(left) != len(right):
        return 0.0
    try:
        dot = sum(float(a) * float(b) for a, b in zip(left, right))
        left_norm = math.sqrt(sum(float(a) ** 2 for a in left))
        right_norm = math.sqrt(sum(float(b) ** 2 for b in right))
    except (TypeError, ValueError):
        return 0.0
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _coverage_score(values):
    if not values:
        return 0.0
    count = len(values)
    parents = len({_parent_key(item) for item in values}) / count
    sources = len({_source_key(item) for item in values}) / count
    return (parents + sources) / 2.0


def _assessment_relevance(query, values):
    query_tokens = frozenset(_TOKEN_RE.findall(str(query or "").casefold()))
    scores = []
    for item in values:
        explicit = _relevance(item)
        item_tokens = _tokens(item)
        lexical = len(query_tokens & item_tokens) / len(query_tokens) if query_tokens else 0.0
        scores.append(max(explicit, lexical))
    return min(1.0, sum(scores) / len(scores))


def _find_conflicts(values):
    grouped = {}
    for item in values:
        source = _source_key(item)
        for claim in _claims(item):
            key = str(claim.get("claim_id") or _claim_key(claim.get("claim", "")))
            grouped.setdefault(key, []).append((source, claim))
    output = []
    for records in grouped.values():
        for index, (left_source, left) in enumerate(records):
            for right_source, right in records[index + 1 :]:
                if _contradicts(left, right):
                    output.append(
                        EvidenceConflict(
                            str(left.get("claim") or right.get("claim") or ""),
                            tuple(sorted((left_source, right_source))),
                            "contradicted",
                            tuple(
                                dict(value)
                                for value in (left.get("conditions") or {}, right.get("conditions") or {})
                            ),
                        )
                    )
    return sorted(output, key=lambda item: (item.claim, item.source))


def _claims(item):
    claims = item.get("claims")
    if isinstance(claims, Mapping):
        claims = (claims,)
    if isinstance(claims, Sequence) and not isinstance(claims, (str, bytes)):
        return [dict(claim) for claim in claims if isinstance(claim, Mapping)]
    if item.get("claim"):
        return [
            {
                "claim_id": item.get("claim_id"),
                "claim": item.get("claim"),
                "value": item.get("value"),
                "conditions": item.get("conditions") or {},
            }
        ]
    return []


def _claim_key(claim):
    text = str(claim).casefold()
    text = re.sub(r"\d+(?:\.\d+)?", "#", text)
    text = re.sub(r"\b(?:not|no|never)\b", "", text)
    return " ".join(_TOKEN_RE.findall(text))


def _contradicts(left, right):
    left_value = str(left.get("value", left.get("claim", ""))).casefold()
    right_value = str(right.get("value", right.get("claim", ""))).casefold()
    if left_value != right_value and ("not" in left_value or "not" in right_value):
        return True
    if _numeric(left_value) is not None and _numeric(right_value) is not None:
        return _numeric(left_value) != _numeric(right_value)
    return dict(left.get("conditions") or {}) != dict(right.get("conditions") or {})


def _numeric(value):
    match = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*", value)
    return float(match.group(1)) if match else None
