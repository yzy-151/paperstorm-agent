"""Structured query planning for retrieval backends.

This module intentionally does not execute retrieval. It converts a user query
into a validated, serializable contract that later pipeline stages can consume.
"""

import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple


ALLOWED_ANSWER_TYPES = frozenset(
    ("factoid", "list", "explanation", "comparison", "procedure", "survey")
)
_PLAN_FIELDS = frozenset(
    (
        "original_query",
        "standalone_query",
        "domain",
        "entities",
        "must_terms",
        "negative_terms",
        "filters",
        "subqueries",
        "answer_type",
    )
)
_RF_DOMAIN = "rf-passive-intermodulation"
_PROCESSING_DOMAIN = "processing-in-memory"
_PROCESSING_MARKERS = (
    "processing-in-memory",
    "processing in memory",
    "compute-in-memory",
    "compute in memory",
    "存内计算",
    "存算一体",
)
_RF_MARKERS = (
    "passive intermodulation",
    "无源互调",
    "射频互调",
    "射频 pim",
)
_RF_SUPPRESSION_MARKERS = (
    "抑制",
    "消除",
    "补偿",
    "suppression",
    "cancellation",
    "mitigation",
)
_FOLLOWUP_REFERENCE = re.compile(
    r"^(?:它|其|这个|该(?:问题|技术|现象)?|上述|这种|these\b|this\b|it\b)",
    re.I,
)


class PlanningError(RuntimeError):
    """Typed failure raised when a planner cannot produce a valid contract."""

    def __init__(self, message: str, *, error_type: str):
        super().__init__(message)
        self.error_type = error_type


@dataclass(frozen=True)
class SearchPlan:
    original_query: str
    standalone_query: str
    domain: str = ""
    entities: Tuple[str, ...] = ()
    must_terms: Tuple[str, ...] = ()
    negative_terms: Tuple[str, ...] = ()
    filters: Dict[str, Any] = field(default_factory=dict)
    subqueries: Tuple[str, ...] = ()
    answer_type: str = "factoid"

    def __post_init__(self) -> None:
        original_query = _normalize_required_text(
            self.original_query, "original_query"
        )
        standalone_query = _normalize_required_text(
            self.standalone_query, "standalone_query"
        )
        domain = _normalize_optional_text(self.domain, "domain")
        answer_type = _normalize_optional_text(self.answer_type, "answer_type")
        if answer_type not in ALLOWED_ANSWER_TYPES:
            raise ValueError(
                "answer_type must be one of: {0}".format(
                    ", ".join(sorted(ALLOWED_ANSWER_TYPES))
                )
            )
        entities = _normalize_string_sequence(self.entities, "entities")
        must_terms = _normalize_string_sequence(self.must_terms, "must_terms")
        negative_terms = _normalize_string_sequence(
            self.negative_terms, "negative_terms"
        )
        subqueries = _normalize_string_sequence(self.subqueries, "subqueries")
        if len(subqueries) > 3:
            raise ValueError("subqueries cannot contain more than 3 items")
        filters = _json_safe_mapping(self.filters, "filters")

        object.__setattr__(self, "original_query", original_query)
        object.__setattr__(self, "standalone_query", standalone_query)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "entities", entities)
        object.__setattr__(self, "must_terms", must_terms)
        object.__setattr__(self, "negative_terms", negative_terms)
        object.__setattr__(self, "filters", filters)
        object.__setattr__(self, "subqueries", subqueries)
        object.__setattr__(self, "answer_type", answer_type)

    def to_dict(self) -> Dict[str, Any]:
        """Return a detached JSON-safe representation."""

        return json.loads(
            json.dumps(asdict(self), ensure_ascii=False, allow_nan=False)
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SearchPlan":
        if not isinstance(value, Mapping):
            raise TypeError("search plan must be a mapping")
        unknown = set(value) - _PLAN_FIELDS
        if unknown:
            raise ValueError(
                "search plan contains unknown fields: {0}".format(
                    ", ".join(sorted(str(item) for item in unknown))
                )
            )
        if "original_query" not in value or "standalone_query" not in value:
            raise ValueError(
                "search plan requires original_query and standalone_query"
            )
        return cls(**dict(value))


class SearchPlanner:
    """Build deterministic or LLM-backed structured search plans."""

    def __init__(self, llm: Optional[Callable[[str], Any]] = None):
        if llm is not None and not callable(llm) and not callable(
            getattr(llm, "invoke", None)
        ):
            raise TypeError("llm must be callable or expose invoke(prompt)")
        self.llm = llm

    def plan(
        self,
        query: str,
        *,
        history: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> SearchPlan:
        original_query = _normalize_required_text(query, "query")
        normalized_history = _normalize_history(history)
        if self.llm is not None:
            return self._plan_with_llm(original_query, normalized_history)
        return _deterministic_plan(original_query, normalized_history)

    def _plan_with_llm(
        self, query: str, history: Sequence[Dict[str, str]]
    ) -> SearchPlan:
        prompt = _build_llm_prompt(query, history)
        last_error = None
        for attempt in range(2):
            attempt_prompt = prompt
            if attempt:
                attempt_prompt += (
                    "\nThe previous response was invalid. Return only one JSON object "
                    "that exactly matches the schema."
                )
            try:
                raw = _invoke_llm(self.llm, attempt_prompt)
                payload = _parse_json_object(_extract_llm_content(raw))
                _validate_llm_schema(payload, query)
                return SearchPlan.from_mapping(payload)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
        raise PlanningError(
            "LLM search planner returned invalid structured output: {0}".format(
                last_error
            ),
            error_type="invalid_structured_output",
        )


def _deterministic_plan(
    original_query: str, history: Sequence[Dict[str, str]]
) -> SearchPlan:
    standalone_query = original_query
    domain = _domain_for_text(original_query)

    if _looks_like_followup(original_query):
        antecedent_domain = _latest_explicit_domain(history)
        if antecedent_domain:
            domain = antecedent_domain
            subject = (
                "passive intermodulation (PIM)"
                if domain == _RF_DOMAIN
                else "processing-in-memory (PIM)"
            )
            remainder = _FOLLOWUP_REFERENCE.sub("", original_query, count=1).strip()
            standalone_query = "{0} {1}".format(subject, remainder).strip()

    entities = ()
    must_terms = ()
    negative_terms = ()
    subqueries = (standalone_query,)
    if domain == _RF_DOMAIN:
        entities = ("passive intermodulation",)
        must_terms = ("passive intermodulation",)
        negative_terms = ("dram", "processing-in-memory")
        subqueries = _unique_strings(
            (
                standalone_query,
                "passive intermodulation suppression RF",
                "PIM cancellation neural network RF",
            )
        )
    elif domain == _PROCESSING_DOMAIN:
        entities = ("processing-in-memory",)
        must_terms = ("processing-in-memory",)

    answer_type = "list" if _requests_list(original_query) else "factoid"
    return SearchPlan(
        original_query=original_query,
        standalone_query=standalone_query,
        domain=domain,
        entities=entities,
        must_terms=must_terms,
        negative_terms=negative_terms,
        filters={},
        subqueries=subqueries,
        answer_type=answer_type,
    )


def _domain_for_text(text: str) -> str:
    lowered = text.lower()
    if any(marker in lowered for marker in _PROCESSING_MARKERS):
        return _PROCESSING_DOMAIN
    if any(marker in lowered for marker in _RF_MARKERS):
        return _RF_DOMAIN
    if "pim" in lowered and "神经网络" in text and any(
        marker in lowered for marker in _RF_SUPPRESSION_MARKERS
    ):
        return _RF_DOMAIN
    return ""


def _latest_explicit_domain(history: Sequence[Dict[str, str]]) -> str:
    for item in reversed(history):
        domain = _domain_for_text(item["content"])
        if domain:
            return domain
    return ""


def _looks_like_followup(query: str) -> bool:
    return bool(_FOLLOWUP_REFERENCE.search(query.strip()))


def _requests_list(query: str) -> bool:
    lowered = query.lower()
    return any(marker in lowered for marker in ("哪些", "方法", "列出", "what are"))


def _normalize_history(
    history: Optional[Sequence[Mapping[str, Any]]],
) -> Tuple[Dict[str, str], ...]:
    if history is None:
        return ()
    if isinstance(history, (str, bytes)) or not isinstance(history, Sequence):
        raise TypeError("history must be a sequence of message mappings")
    normalized = []
    for item in history:
        if not isinstance(item, Mapping):
            raise TypeError("history items must be mappings")
        role = _normalize_optional_text(item.get("role", ""), "history role")
        content = _normalize_optional_text(
            item.get("content", ""), "history content"
        )
        if content:
            normalized.append({"role": role, "content": content})
    return tuple(normalized)


def _build_llm_prompt(query: str, history: Sequence[Dict[str, str]]) -> str:
    schema = {
        "original_query": "non-empty string, exactly the current user query",
        "standalone_query": "non-empty standalone retrieval query",
        "domain": "string; empty when ambiguous",
        "entities": ["string"],
        "must_terms": ["string"],
        "negative_terms": ["string"],
        "filters": {"JSON-safe filter": "value"},
        "subqueries": ["at most 3 unique non-empty strings"],
        "answer_type": sorted(ALLOWED_ANSWER_TYPES),
    }
    payload = {"query": query, "history": list(history), "schema": schema}
    return (
        "You are a search query planner. Return only one JSON object. "
        "Do not inherit a topic unless the supplied history contains a reliable "
        "antecedent. Preserve original_query exactly after whitespace normalization.\n"
        + json.dumps(payload, ensure_ascii=False, allow_nan=False)
    )


def _invoke_llm(llm: Any, prompt: str) -> Any:
    invoke = getattr(llm, "invoke", None)
    if callable(invoke):
        return invoke(prompt)
    return llm(prompt)


def _extract_llm_content(raw: Any) -> Any:
    if isinstance(raw, Mapping):
        if "content" in raw:
            return raw["content"]
        message = raw.get("message")
        if isinstance(message, Mapping) and "content" in message:
            return message["content"]
        choices = raw.get("choices")
        if isinstance(choices, Sequence) and choices:
            choice = choices[0]
            if isinstance(choice, Mapping):
                message = choice.get("message")
                if isinstance(message, Mapping) and "content" in message:
                    return message["content"]
        return raw
    content = getattr(raw, "content", None)
    return content if content is not None else raw


def _parse_json_object(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    if not isinstance(raw, str):
        raise TypeError("LLM output must be a JSON object or JSON string")
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("LLM output must decode to a JSON object")
    return value


def _validate_llm_schema(payload: Mapping[str, Any], query: str) -> None:
    missing = _PLAN_FIELDS - set(payload)
    unknown = set(payload) - _PLAN_FIELDS
    if missing:
        raise ValueError(
            "LLM search plan is missing fields: {0}".format(
                ", ".join(sorted(missing))
            )
        )
    if unknown:
        raise ValueError(
            "LLM search plan contains unknown fields: {0}".format(
                ", ".join(sorted(str(item) for item in unknown))
            )
        )
    if _normalize_required_text(payload["original_query"], "original_query") != query:
        raise ValueError("LLM search plan changed original_query")
    SearchPlan.from_mapping(payload)


def _normalize_required_text(value: Any, field_name: str) -> str:
    normalized = _normalize_optional_text(value, field_name)
    if not normalized:
        raise ValueError("{0} must be a non-empty string".format(field_name))
    return normalized


def _normalize_optional_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError("{0} must be a string".format(field_name))
    return " ".join(value.split())


def _normalize_string_sequence(value: Any, field_name: str) -> Tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise TypeError("{0} must be a list or tuple of strings".format(field_name))
    normalized = []
    seen = set()
    for item in value:
        text = _normalize_required_text(item, field_name)
        key = text.casefold()
        if key not in seen:
            seen.add(key)
            normalized.append(text)
    return tuple(normalized)


def _unique_strings(values: Sequence[str]) -> Tuple[str, ...]:
    return _normalize_string_sequence(tuple(values), "subqueries")[:3]


def _json_safe_mapping(value: Any, field_name: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("{0} must be a mapping".format(field_name))
    normalized = _normalize_json_value(dict(value), field_name)
    return json.loads(
        json.dumps(normalized, ensure_ascii=False, allow_nan=False)
    )


def _normalize_json_value(value: Any, field_name: str) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("{0} cannot contain non-finite floats".format(field_name))
        return value
    if isinstance(value, list):
        return [_normalize_json_value(item, field_name) for item in value]
    if isinstance(value, Mapping):
        output = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("{0} keys must be strings".format(field_name))
            output[key] = _normalize_json_value(item, field_name)
        return output
    raise TypeError("{0} must contain only JSON-safe values".format(field_name))
