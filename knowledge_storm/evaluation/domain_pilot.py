"""Contracts and metrics for a private, evidence-grounded domain pilot."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .public_benchmarks.base import (
    BenchmarkCase,
    BenchmarkDataset,
    BenchmarkDocument,
)


DOMAIN_CATEGORIES = (
    "definition",
    "mechanism",
    "method",
    "experiment",
    "comparison",
    "limitation",
)


def load_domain_dataset(
    corpus_path,
    cases_path,
    expected_case_count=50,
    required_categories=(),
):
    corpus_rows = _read_jsonl(corpus_path)
    case_rows = _read_jsonl(cases_path)
    validate_domain_rows(
        corpus_rows,
        case_rows,
        expected_case_count=expected_case_count,
        required_categories=required_categories,
    )
    documents = tuple(
        BenchmarkDocument(
            document_id=str(row["chunk_id"]),
            title=str(row.get("title") or row.get("document_id") or row["chunk_id"]),
            text=str(row["content"]),
            metadata={
                **dict(row.get("metadata") or {}),
                "source_document_id": str(row.get("document_id") or ""),
                "chunk_id": str(row["chunk_id"]),
            },
        )
        for row in corpus_rows
    )
    cases = tuple(
        BenchmarkCase(
            case_id=str(row["case_id"]),
            query=str(row["question"]),
            relevant_document_ids=tuple(
                str(value) for value in row["evidence_chunk_ids"]
            ),
            split="pilot",
            answers=(str(row["reference_answer"]),),
            evidence_ids=tuple(str(value) for value in row["evidence_chunk_ids"]),
            metadata={
                "category": str(row["category"]),
                "difficulty": str(row.get("difficulty") or "unspecified"),
                "evidence_quote": str(row["evidence_quote"]),
                "source_title": str(row.get("source_title") or ""),
                "review_status": str(row.get("review_status") or "validated"),
            },
        )
        for row in case_rows
    )
    return BenchmarkDataset(
        name="paperstorm-pim-domain-pilot",
        version="private-pilot-v1",
        documents=documents,
        cases=cases,
        metadata={
            "evidence_tier": "private_domain_pilot",
            "case_count": len(cases),
            "categories": sorted({case.metadata["category"] for case in cases}),
        },
    )


def validate_domain_rows(
    corpus_rows,
    case_rows,
    expected_case_count=50,
    required_categories=(),
):
    corpus_rows = tuple(corpus_rows or ())
    case_rows = tuple(case_rows or ())
    if len(case_rows) != int(expected_case_count):
        raise ValueError(
            "expected {0} domain cases, got {1}".format(
                int(expected_case_count), len(case_rows)
            )
        )
    chunks = {}
    for row in corpus_rows:
        chunk_id = str(row.get("chunk_id") or "").strip()
        content = str(row.get("content") or "").strip()
        if not chunk_id or not content:
            raise ValueError("corpus rows require chunk_id and content")
        if chunk_id in chunks:
            raise ValueError("duplicate chunk_id: {0}".format(chunk_id))
        chunks[chunk_id] = row

    questions = set()
    case_ids = set()
    categories = set()
    for row in case_rows:
        case_id = str(row.get("case_id") or "").strip()
        question = _normalized_text(row.get("question"))
        answer = _normalized_text(row.get("reference_answer"))
        quote = _normalized_text(row.get("evidence_quote"))
        category = str(row.get("category") or "").strip().lower()
        evidence_ids = tuple(str(value).strip() for value in row.get("evidence_chunk_ids") or ())
        if not case_id or not question or not answer or not quote or not category:
            raise ValueError("domain case fields must not be empty")
        if case_id in case_ids:
            raise ValueError("duplicate case_id: {0}".format(case_id))
        if question in questions:
            raise ValueError("duplicate question: {0}".format(question))
        if not evidence_ids:
            raise ValueError("domain case requires evidence_chunk_ids")
        missing = [value for value in evidence_ids if value not in chunks]
        if missing:
            raise ValueError("missing evidence chunk: {0}".format(", ".join(missing)))
        evidence_text = " ".join(_normalized_text(chunks[value]["content"]) for value in evidence_ids)
        if quote not in evidence_text:
            raise ValueError(
                "evidence quote is not grounded for case {0}".format(case_id)
            )
        case_ids.add(case_id)
        questions.add(question)
        categories.add(category)

    required = {str(value).strip().lower() for value in required_categories}
    missing_categories = sorted(required - categories)
    if missing_categories:
        raise ValueError(
            "missing required categories: {0}".format(", ".join(missing_categories))
        )
    return {
        "case_count": len(case_rows),
        "document_count": len(corpus_rows),
        "categories": sorted(categories),
    }


def _read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _normalized_text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()
