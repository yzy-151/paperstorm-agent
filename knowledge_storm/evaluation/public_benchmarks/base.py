"""Dataset-neutral contracts for public benchmark adapters."""

from dataclasses import dataclass, field
from typing import Dict, Mapping, Tuple


@dataclass(frozen=True)
class BenchmarkDocument:
    document_id: str
    title: str
    text: str
    metadata: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if not str(self.document_id).strip():
            raise ValueError("document_id is required")


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    query: str
    relevant_document_ids: Tuple[str, ...]
    split: str
    relevance: Mapping[str, int] = field(default_factory=dict)
    answers: Tuple[str, ...] = ()
    evidence_ids: Tuple[str, ...] = ()
    unanswerable: bool = False
    metadata: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if not str(self.case_id).strip():
            raise ValueError("case_id is required")
        if not str(self.query).strip():
            raise ValueError("query is required")
        if not self.relevance and self.relevant_document_ids:
            object.__setattr__(
                self,
                "relevance",
                {document_id: 1 for document_id in self.relevant_document_ids},
            )


@dataclass(frozen=True)
class BenchmarkDataset:
    name: str
    version: str
    documents: Tuple[BenchmarkDocument, ...]
    cases: Tuple[BenchmarkCase, ...]
    metadata: Mapping = field(default_factory=dict)

    def __post_init__(self):
        _assert_unique(
            [document.document_id for document in self.documents],
            "duplicate document_id",
        )
        _assert_unique([case.case_id for case in self.cases], "duplicate case_id")
        document_ids = {document.document_id for document in self.documents}
        unknown = {
            document_id
            for case in self.cases
            for document_id in case.relevant_document_ids
            if document_id not in document_ids
        }
        if unknown:
            raise ValueError(
                "qrels reference missing documents: {0}".format(sorted(unknown))
            )

    def document_map(self) -> Dict[str, BenchmarkDocument]:
        return {document.document_id: document for document in self.documents}


def _assert_unique(values, message):
    if len(values) != len(set(values)):
        raise ValueError(message)
