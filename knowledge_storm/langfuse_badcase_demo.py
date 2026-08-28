"""Deterministic RAG badcase demo backed by PaperStorm observability."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .paperstorm_observability import PaperStormObservability, build_observability


TRACE_NAME = "paperstorm.rag.badcase"
STAGE_NAMES = ("route", "retrieve", "rerank", "context", "reader", "citation_validate")

DEFAULT_COMPOSITE_BADCASE = {
    "case_id": "fixed-composite-badcase",
    "question": "What did the trial recommend?",
    "expected_document_ids": ["paper-a", "paper-b"],
    "retrieved_documents": [
        {"document_id": "paper-b", "text": "The evidence says treatment should be avoided."}
    ],
    "reranked_documents": [
        {"document_id": "paper-b", "text": "The evidence says treatment should be avoided."}
    ],
    "context_documents": [
        {"document_id": "paper-b", "text": "The evidence says treatment should be avoided."}
    ],
    "answer": "I cannot answer this question.",
    "citations": ["paper-missing"],
    "answerable": True,
    "evidence_conflict": True,
    "latency_ms": 42.5,
}


def run_badcase_demo(
    case: Mapping[str, Any],
    output_dir: Any,
    observability: Optional[PaperStormObservability] = None,
) -> Dict[str, Any]:
    """Trace one RAG badcase and return its score and observability summary."""
    data = _validate_case(case)
    root_dir = Path(output_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    scores = _calculate_scores(data)
    badcase_types = _classify_badcase(data, scores)
    observer = observability or build_observability(root_dir)
    started = time.perf_counter()

    with observer.trace(
        TRACE_NAME,
        input={"case_id": data.get("case_id", ""), "question": data.get("question", "")},
        metadata={"badcase_types": badcase_types, "case_metadata": data.get("metadata", {})},
        tags=["paperstorm", "rag", "badcase"] + badcase_types,
    ) as trace:
        _record_stage_spans(trace, data, scores)
        for name, value in scores.items():
            trace.score(name, value)
        trace.end(
            output={"badcase_types": badcase_types, "scores": scores},
            metadata={"elapsed_ms": round((time.perf_counter() - started) * 1000, 3)},
        )

    observer.flush()
    status = observer.status()
    return {
        "paperstorm_trace_id": trace.trace_id,
        "remote_trace_id": _remote_trace_id(trace.remote),
        "scores": scores,
        "badcase_types": badcase_types,
        "observability": status,
        "local_events_path": status["local_events_path"],
    }


def _record_stage_spans(trace, case: Mapping[str, Any], scores: Mapping[str, float]) -> None:
    stages = {
        "route": ({"question": case.get("question", "")}, {"route": case.get("route", "rag")}),
        "retrieve": (
            {"expected_document_ids": case.get("expected_document_ids", [])},
            {"document_ids": _document_ids(case.get("retrieved_documents", []))},
        ),
        "rerank": (
            {"document_ids": _document_ids(case.get("retrieved_documents", []))},
            {"document_ids": _document_ids(case.get("reranked_documents", []))},
        ),
        "context": (
            {"document_ids": _document_ids(case.get("reranked_documents", []))},
            {"document_ids": _document_ids(case.get("context_documents", []))},
        ),
        "reader": (
            {"context_document_ids": _document_ids(case.get("context_documents", []))},
            {"answer": case.get("answer", ""), "groundedness": scores["answer_groundedness"]},
        ),
        "citation_validate": (
            {"citations": _citation_ids(case.get("citations", []))},
            {"citation_validity": scores["citation_validity"]},
        ),
    }
    for name in STAGE_NAMES:
        stage_input, stage_output = stages[name]
        with trace.span(name, input=stage_input) as span:
            span.end(output=stage_output)


def _calculate_scores(case: Mapping[str, Any]) -> Dict[str, float]:
    expected = set(_string_values(case.get("expected_document_ids", [])))
    retrieved = _document_ids(case.get("retrieved_documents", []))[:5]
    recall = float(len(expected.intersection(retrieved))) / len(expected) if expected else 1.0

    citations = _citation_ids(case.get("citations", []))
    available = set(_document_ids(case.get("context_documents", [])))
    citation_validity = float(sum(item in available for item in citations)) / len(citations) if citations else 1.0

    latency = float(case.get("latency_ms", 0.0))
    if latency < 0:
        raise ValueError("latency_ms must be non-negative")
    return {
        "retrieval_recall_at_5": round(recall, 6),
        "citation_validity": round(citation_validity, 6),
        "answer_groundedness": round(
            _groundedness(str(case.get("answer", "")), _document_text(case.get("context_documents", []))),
            6,
        ),
        "latency_ms": latency,
    }


def _classify_badcase(case: Mapping[str, Any], scores: Mapping[str, float]) -> List[str]:
    result = []
    if scores["retrieval_recall_at_5"] < 1.0:
        result.append("retrieval_miss")
    if scores["citation_validity"] < 1.0:
        result.append("invalid_citation")
    if case.get("evidence_conflict"):
        result.append("evidence_conflict")
    if bool(case.get("answerable")) and _is_abstention(str(case.get("answer", ""))):
        result.append("wrong_abstention")
    return result


def _document_ids(documents: Any) -> List[str]:
    result = []
    for document in _iterable(documents):
        value = (
            document.get("document_id", document.get("doc_id", document.get("id", "")))
            if isinstance(document, Mapping)
            else document
        )
        if value is not None and str(value):
            result.append(str(value))
    return result


def _citation_ids(citations: Any) -> List[str]:
    return _document_ids(citations)


def _document_text(documents: Any) -> str:
    return " ".join(
        str(document.get("text", document.get("content", "")))
        for document in _iterable(documents)
        if isinstance(document, Mapping)
    )


def _groundedness(answer: str, evidence: str) -> float:
    answer_terms = set(_lexical_terms(answer))
    if not answer_terms:
        return 0.0
    evidence_terms = set(_lexical_terms(evidence))
    return float(len(answer_terms.intersection(evidence_terms))) / len(answer_terms)


def _is_abstention(answer: str) -> bool:
    normalized = answer.strip().lower()
    return not normalized or any(
        marker in normalized
        for marker in ("cannot answer", "can't answer", "unable to answer", "i don't know", "unknown")
    )


def _iterable(value: Any) -> Iterable[Any]:
    return value if isinstance(value, (list, tuple, set)) else []


def _string_values(value: Any) -> List[str]:
    return [str(item) for item in _iterable(value) if item is not None]


def _validate_case(case: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(case, Mapping):
        raise TypeError("case must be a mapping")
    data = dict(case)
    required = (
        "case_id", "question", "expected_document_ids", "retrieved_documents",
        "reranked_documents", "context_documents", "answer", "citations",
        "answerable", "latency_ms",
    )
    missing = [field for field in required if field not in data]
    if missing:
        raise ValueError("case is missing required fields: {0}".format(", ".join(missing)))
    for field in ("case_id", "question", "answer"):
        if not isinstance(data[field], str):
            raise TypeError("{0} must be a string".format(field))
        if not data[field].strip():
            raise ValueError("{0} must not be empty".format(field))
    _validate_id_list("expected_document_ids", data["expected_document_ids"])
    _validate_document_list("retrieved_documents", data["retrieved_documents"])
    _validate_document_list("reranked_documents", data["reranked_documents"])
    _validate_document_list("context_documents", data["context_documents"], require_text=True)
    _validate_id_list("citations", data["citations"])
    if not isinstance(data["answerable"], bool):
        raise TypeError("answerable must be a boolean")
    if isinstance(data["latency_ms"], bool) or not isinstance(data["latency_ms"], (int, float)):
        raise TypeError("latency_ms must be a number")
    if data["latency_ms"] < 0:
        raise ValueError("latency_ms must be non-negative")
    if "metadata" in data and not isinstance(data["metadata"], Mapping):
        raise TypeError("metadata must be a mapping")
    return data


def _validate_id_list(field: str, value: Any) -> None:
    if not isinstance(value, list):
        raise TypeError("{0} must be a list".format(field))
    if not value:
        raise ValueError("{0} must not be empty".format(field))
    for item in value:
        if isinstance(item, Mapping):
            item = item.get("document_id", item.get("doc_id", item.get("id")))
        if not isinstance(item, str) or not item.strip():
            raise TypeError("{0} entries must be non-empty document IDs".format(field))


def _validate_document_list(field: str, value: Any, require_text: bool = False) -> None:
    if not isinstance(value, list):
        raise TypeError("{0} must be a list".format(field))
    for document in value:
        if not isinstance(document, Mapping):
            raise TypeError("{0} entries must be mappings".format(field))
        document_id = document.get("document_id", document.get("doc_id", document.get("id")))
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("{0} entries need a non-empty document ID".format(field))
        if require_text:
            text = document.get("text", document.get("content"))
            if not isinstance(text, str) or not text.strip():
                raise ValueError("context_documents entries need non-empty text or content")


def _lexical_terms(value: str) -> List[str]:
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", value.lower())


def _remote_trace_id(remote: Any) -> Optional[str]:
    if remote is None:
        return None
    for field in ("trace_id", "id"):
        value = getattr(remote, field, None)
        if value is not None and str(value):
            return str(value)
    return None
