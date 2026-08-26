"""Claim-level answer validation with strict schemas and bounded repair.

The model may propose claim text and citation identifiers, but source metadata is
always rehydrated from the trusted evidence registry before a claim is accepted.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Tuple, Union


class AnswerSchemaError(ValueError):
    """Raised when model output does not satisfy the answer contract."""


class ClaimVerdict(str, Enum):
    ENTAILED = "entailed"
    PARTIAL = "partial"
    CONTRADICTED = "contradicted"
    UNSUPPORTED = "unsupported"


def _strict_keys(data: Mapping[str, Any], required: set, optional: set, name: str) -> None:
    if not isinstance(data, Mapping):
        raise AnswerSchemaError(f"{name} must be an object")
    keys = set(data)
    missing = required - keys
    unknown = keys - required - optional
    if missing:
        raise AnswerSchemaError(f"{name} missing fields: {sorted(missing)}")
    if unknown:
        raise AnswerSchemaError(f"{name} has unknown fields: {sorted(unknown)}")


def _non_empty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AnswerSchemaError(f"{field} must be a non-empty string")
    return value


def _optional_string(value: Any, field: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise AnswerSchemaError(f"{field} must be null or a non-empty string")
    return value


@dataclass(frozen=True)
class Citation:
    citation_id: str
    source_id: str
    span: str
    title: str
    authors: Tuple[str, ...]
    page: Optional[Union[int, str]] = None
    section: Optional[str] = None
    url: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Citation":
        required = {"citation_id", "source_id", "span", "title", "authors"}
        optional = {"page", "section", "url"}
        _strict_keys(data, required, optional, "Citation")
        authors = data["authors"]
        if not isinstance(authors, (list, tuple)):
            raise AnswerSchemaError("Citation.authors must be an array")
        if any(not isinstance(author, str) or not author.strip() for author in authors):
            raise AnswerSchemaError("Citation.authors must contain non-empty strings")
        page = data.get("page")
        if isinstance(page, bool) or (
            page is not None and not isinstance(page, (int, str))
        ):
            raise AnswerSchemaError("Citation.page must be null, integer, or string")
        if isinstance(page, str) and not page.strip():
            raise AnswerSchemaError("Citation.page cannot be empty")
        return cls(
            citation_id=_non_empty_string(data["citation_id"], "Citation.citation_id"),
            source_id=_non_empty_string(data["source_id"], "Citation.source_id"),
            span=_non_empty_string(data["span"], "Citation.span"),
            title=_non_empty_string(data["title"], "Citation.title"),
            authors=tuple(authors),
            page=page,
            section=_optional_string(data.get("section"), "Citation.section"),
            url=_optional_string(data.get("url"), "Citation.url"),
        )

    def to_dict(self) -> dict:
        return {
            "citation_id": self.citation_id,
            "source_id": self.source_id,
            "span": self.span,
            "title": self.title,
            "authors": list(self.authors),
            "page": self.page,
            "section": self.section,
            "url": self.url,
        }


@dataclass(frozen=True)
class Claim:
    claim_id: str
    text: str
    citations: Tuple[Citation, ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Claim":
        _strict_keys(data, {"claim_id", "text", "citations"}, set(), "Claim")
        citations = data["citations"]
        if not isinstance(citations, (list, tuple)):
            raise AnswerSchemaError("Claim.citations must be an array")
        return cls(
            claim_id=_non_empty_string(data["claim_id"], "Claim.claim_id"),
            text=_non_empty_string(data["text"], "Claim.text"),
            citations=tuple(
                item if isinstance(item, Citation) else Citation.from_dict(item)
                for item in citations
            ),
        )

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "citations": [item.to_dict() for item in self.citations],
        }


@dataclass(frozen=True)
class AnswerDraft:
    answer: str
    claims: Tuple[Claim, ...]
    refusal: bool = False
    answer_type: str = "abstractive"
    uncertainty: float = 0.0
    abstain_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.answer, str):
            raise AnswerSchemaError("AnswerDraft.answer must be a string")
        _non_empty_string(self.answer_type, "AnswerDraft.answer_type")
        if type(self.refusal) is not bool:
            raise AnswerSchemaError("AnswerDraft.refusal must be a boolean")
        if (
            isinstance(self.uncertainty, bool)
            or not isinstance(self.uncertainty, (int, float))
            or not math.isfinite(float(self.uncertainty))
            or not 0.0 <= float(self.uncertainty) <= 1.0
        ):
            raise AnswerSchemaError(
                "AnswerDraft.uncertainty must be a finite number in [0, 1]"
            )
        reason = _optional_string(
            self.abstain_reason, "AnswerDraft.abstain_reason"
        )
        if self.refusal and not reason:
            raise AnswerSchemaError("A refusal requires abstain_reason")
        if not self.refusal and reason is not None:
            raise AnswerSchemaError("A non-refusal answer cannot have abstain_reason")
        if self.refusal and self.answer_type != "refusal":
            raise AnswerSchemaError("A refusal requires answer_type='refusal'")
        if not self.refusal and self.answer_type == "refusal":
            raise AnswerSchemaError("answer_type='refusal' requires refusal=true")
        if not self.refusal and not self.answer.strip():
            raise AnswerSchemaError("A non-refusal answer cannot be empty")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AnswerDraft":
        required = {
            "answer",
            "answer_type",
            "claims",
            "uncertainty",
            "refusal",
            "abstain_reason",
        }
        _strict_keys(data, required, {"citation_ids"}, "AnswerDraft")
        if not isinstance(data["answer"], str):
            raise AnswerSchemaError("AnswerDraft.answer must be a string")
        claims = data["claims"]
        if not isinstance(claims, (list, tuple)):
            raise AnswerSchemaError("AnswerDraft.claims must be an array")
        if type(data["refusal"]) is not bool:
            raise AnswerSchemaError("AnswerDraft.refusal must be a boolean")
        answer_type = _non_empty_string(
            data["answer_type"], "AnswerDraft.answer_type"
        )
        uncertainty = data["uncertainty"]
        if (
            isinstance(uncertainty, bool)
            or not isinstance(uncertainty, (int, float))
            or not math.isfinite(float(uncertainty))
            or not 0.0 <= float(uncertainty) <= 1.0
        ):
            raise AnswerSchemaError(
                "AnswerDraft.uncertainty must be a finite number in [0, 1]"
            )
        abstain_reason = _optional_string(
            data["abstain_reason"], "AnswerDraft.abstain_reason"
        )
        parsed = tuple(
            item if isinstance(item, Claim) else Claim.from_dict(item) for item in claims
        )
        ids = [item.claim_id for item in parsed]
        if len(ids) != len(set(ids)):
            raise AnswerSchemaError("AnswerDraft.claim_id values must be unique")
        if not data["refusal"] and not data["answer"].strip():
            raise AnswerSchemaError("A non-refusal answer cannot be empty")
        if data["refusal"] and not abstain_reason:
            raise AnswerSchemaError("A refusal requires abstain_reason")
        if not data["refusal"] and abstain_reason is not None:
            raise AnswerSchemaError("A non-refusal answer cannot have abstain_reason")
        if data["refusal"] and answer_type != "refusal":
            raise AnswerSchemaError("A refusal requires answer_type='refusal'")
        if not data["refusal"] and answer_type == "refusal":
            raise AnswerSchemaError("answer_type='refusal' requires refusal=true")
        derived_ids = _citation_ids(parsed)
        if "citation_ids" in data:
            supplied_ids = data["citation_ids"]
            if (
                not isinstance(supplied_ids, (list, tuple))
                or any(not isinstance(item, str) for item in supplied_ids)
                or tuple(supplied_ids) != derived_ids
            ):
                raise AnswerSchemaError(
                    "AnswerDraft.citation_ids must equal IDs derived from claims"
                )
        return cls(
            answer=data["answer"],
            claims=parsed,
            refusal=data["refusal"],
            answer_type=answer_type,
            uncertainty=float(uncertainty),
            abstain_reason=abstain_reason,
        )

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "answer_type": self.answer_type,
            "claims": [item.to_dict() for item in self.claims],
            "citation_ids": list(_citation_ids(self.claims)),
            "uncertainty": self.uncertainty,
            "refusal": self.refusal,
            "abstain_reason": self.abstain_reason,
        }


def _citation_ids(claims: Tuple[Claim, ...]) -> Tuple[str, ...]:
    ordered = []
    seen = set()
    for claim in claims:
        for citation in claim.citations:
            if citation.citation_id not in seen:
                ordered.append(citation.citation_id)
                seen.add(citation.citation_id)
    return tuple(ordered)


@dataclass(frozen=True)
class ClaimAssessment:
    claim_id: str
    verdict: ClaimVerdict
    rationale: str = ""
    attempt: int = 0

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "verdict": self.verdict.value,
            "rationale": self.rationale,
            "attempt": self.attempt,
        }


@dataclass(frozen=True)
class ValidationResult:
    draft: AnswerDraft
    assessments: Tuple[ClaimAssessment, ...]
    repaired_claim_ids: Tuple[str, ...] = ()
    downgraded_claim_ids: Tuple[str, ...] = ()
    deleted_claim_ids: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "draft": self.draft.to_dict(),
            "assessments": [item.to_dict() for item in self.assessments],
            "repaired_claim_ids": list(self.repaired_claim_ids),
            "downgraded_claim_ids": list(self.downgraded_claim_ids),
            "deleted_claim_ids": list(self.deleted_claim_ids),
        }


ParserRetry = Callable[[Any, AnswerSchemaError], Any]
VerifierResult = Union[ClaimVerdict, str, ClaimAssessment, Mapping[str, Any]]
Verifier = Callable[[Claim, Mapping[str, Citation]], VerifierResult]
Repairer = Callable[[Claim, ClaimAssessment, Mapping[str, Citation]], Union[Claim, Mapping[str, Any]]]


def _decode(raw: Any) -> AnswerDraft:
    if isinstance(raw, AnswerDraft):
        return raw
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as exc:
            raise AnswerSchemaError(f"AnswerDraft is not valid JSON: {exc}") from exc
    try:
        return AnswerDraft.from_dict(raw)
    except AnswerSchemaError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise AnswerSchemaError(f"Invalid AnswerDraft: {exc}") from exc


def parse_answer_draft(raw: Any, retry: Optional[ParserRetry] = None) -> AnswerDraft:
    """Parse strict model output; an optional correction callback runs at most once."""
    try:
        return _decode(raw)
    except AnswerSchemaError as first_error:
        if retry is None:
            raise
        corrected = retry(raw, first_error)
        try:
            return _decode(corrected)
        except AnswerSchemaError as second_error:
            raise AnswerSchemaError(
                f"AnswerDraft remained invalid after one retry: {second_error}"
            ) from second_error


class AnswerValidator:
    """Validate each claim and perform no more than one local repair attempt."""

    def __init__(
        self,
        verifier: Verifier,
        repair: Optional[Repairer] = None,
        refusal_text: str = "现有证据不足以可靠回答该问题。",
        partial_prefix: str = "现有证据仅部分支持：",
    ) -> None:
        if not callable(verifier):
            raise TypeError("verifier must be callable")
        if repair is not None and not callable(repair):
            raise TypeError("repair must be callable")
        self.verifier = verifier
        self.repair = repair
        self.refusal_text = refusal_text
        self.partial_prefix = partial_prefix

    @staticmethod
    def _registry(evidence: Mapping[str, Union[Citation, Mapping[str, Any]]]) -> dict:
        if not isinstance(evidence, Mapping):
            raise TypeError("evidence must map citation_id to Citation")
        canonical = {}
        for key, value in evidence.items():
            item = value if isinstance(value, Citation) else Citation.from_dict(value)
            if key != item.citation_id:
                raise AnswerSchemaError("Evidence key must equal Citation.citation_id")
            canonical[key] = item
        return canonical

    @staticmethod
    def _canonicalize(claim: Claim, evidence: Mapping[str, Citation]) -> Optional[Claim]:
        citations = []
        seen = set()
        for proposed in claim.citations:
            canonical = evidence.get(proposed.citation_id)
            if canonical is None:
                return None
            if canonical.citation_id not in seen:
                citations.append(canonical)
                seen.add(canonical.citation_id)
        return replace(claim, citations=tuple(citations))

    @staticmethod
    def _normalize_assessment(result: VerifierResult, claim_id: str, attempt: int) -> ClaimAssessment:
        rationale = ""
        if isinstance(result, ClaimAssessment):
            if result.claim_id != claim_id:
                raise AnswerSchemaError("Verifier returned assessment for another claim")
            return replace(result, attempt=attempt)
        if isinstance(result, Mapping):
            unknown = set(result) - {"verdict", "rationale"}
            if "verdict" not in result or unknown:
                raise AnswerSchemaError("Verifier result must contain verdict and optional rationale")
            verdict = result["verdict"]
            rationale = result.get("rationale", "")
            if not isinstance(rationale, str):
                raise AnswerSchemaError("Verifier rationale must be a string")
        else:
            verdict = result
        try:
            verdict = verdict if isinstance(verdict, ClaimVerdict) else ClaimVerdict(verdict)
        except (TypeError, ValueError) as exc:
            raise AnswerSchemaError(f"Unknown claim verdict: {verdict!r}") from exc
        return ClaimAssessment(claim_id, verdict, rationale, attempt)

    def assess(
        self,
        claim: Claim,
        evidence: Mapping[str, Union[Citation, Mapping[str, Any]]],
        attempt: int = 0,
    ) -> ClaimAssessment:
        registry = self._registry(evidence)
        canonical = self._canonicalize(claim, registry)
        if canonical is None or not canonical.citations:
            return ClaimAssessment(
                claim.claim_id,
                ClaimVerdict.UNSUPPORTED,
                "Claim cites no trusted evidence.",
                attempt,
            )
        return self._normalize_assessment(
            self.verifier(canonical, registry), claim.claim_id, attempt
        )

    def validate(
        self,
        draft: AnswerDraft,
        evidence: Mapping[str, Union[Citation, Mapping[str, Any]]],
    ) -> ValidationResult:
        registry = self._registry(evidence)
        kept = []
        assessments = []
        repaired = []
        downgraded = []
        deleted = []
        changed = False

        for proposed in draft.claims:
            claim = self._canonicalize(proposed, registry)
            if claim is None or not claim.citations:
                assessment = ClaimAssessment(
                    proposed.claim_id,
                    ClaimVerdict.UNSUPPORTED,
                    "Claim cites no trusted evidence.",
                    0,
                )
            else:
                assessment = self._normalize_assessment(
                    self.verifier(claim, registry), claim.claim_id, 0
                )
            assessments.append(assessment)

            if assessment.verdict is ClaimVerdict.ENTAILED:
                kept.append(claim)
                changed = changed or claim != proposed
                continue

            if self.repair is not None and claim is not None:
                try:
                    candidate = self.repair(claim, assessment, registry)
                    candidate = candidate if isinstance(candidate, Claim) else Claim.from_dict(candidate)
                    if candidate.claim_id != claim.claim_id:
                        raise AnswerSchemaError("Repair cannot change claim_id")
                    candidate = self._canonicalize(candidate, registry)
                except (AnswerSchemaError, KeyError, TypeError, ValueError):
                    candidate = None
                if candidate is not None and candidate.citations:
                    second = self._normalize_assessment(
                        self.verifier(candidate, registry), candidate.claim_id, 1
                    )
                    assessments.append(second)
                    repaired.append(candidate.claim_id)
                    assessment = second
                    claim = candidate
                    if second.verdict is ClaimVerdict.ENTAILED:
                        kept.append(candidate)
                        changed = True
                        continue

            if assessment.verdict is ClaimVerdict.PARTIAL and claim is not None:
                kept.append(replace(claim, text=self.partial_prefix + claim.text))
                downgraded.append(claim.claim_id)
            else:
                deleted.append(proposed.claim_id)
            changed = True

        refusal = not kept
        if refusal:
            answer = self.refusal_text
        elif changed:
            answer = "\n".join(item.text for item in kept)
        else:
            answer = draft.answer
        validated = AnswerDraft(
            answer=answer,
            claims=tuple(kept),
            refusal=refusal,
            answer_type="refusal" if refusal else draft.answer_type,
            uncertainty=1.0 if refusal else draft.uncertainty,
            abstain_reason=self.refusal_text if refusal else None,
        )
        return ValidationResult(
            draft=validated,
            assessments=tuple(assessments),
            repaired_claim_ids=tuple(repaired),
            downgraded_claim_ids=tuple(downgraded),
            deleted_claim_ids=tuple(deleted),
        )


__all__ = [
    "AnswerDraft",
    "AnswerSchemaError",
    "AnswerValidator",
    "Citation",
    "Claim",
    "ClaimAssessment",
    "ClaimVerdict",
    "ValidationResult",
    "parse_answer_draft",
]
