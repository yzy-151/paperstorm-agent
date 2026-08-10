"""Retrieval-only LongMemEval diagnostic for PaperStorm memory systems."""

import statistics
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ...paperstorm_memory_v56 import LongTermMemoryServiceV56


def run_memory_retrieval(dataset, output_dir, top_k=5, embedding_provider=None, limit=None):
    cases = list(dataset.cases[: int(limit)]) if limit else list(dataset.cases)
    document_map = dataset.document_map()
    by_question = {}
    for document in dataset.documents:
        by_question.setdefault(str(document.metadata.get("question_id")), []).append(document)
    service = LongTermMemoryServiceV56(Path(output_dir) / "memory", embedding_provider=embedding_provider)
    mode_rows = {"recent_window": [], "v56_memory": []}
    predictions = []
    for case in cases:
        documents = by_question.get(case.case_id, [])
        namespace = "longmemeval:" + case.case_id
        existing_document_ids = {
            str(item.get("metadata", {}).get("document_id") or "")
            for item in service.list_memories(namespace, include_inactive=True)
        }
        for index, document in enumerate(documents):
            if document.document_id in existing_document_ids:
                continue
            valid_from = _timestamp(document.metadata.get("timestamp"), index)
            service.upsert(
                namespace=namespace,
                memory_type="episodic",
                subject="conversation",
                content=document.text,
                canonical_key=str(document.metadata.get("session_id") or document.document_id),
                source_message_ids=[document.document_id],
                valid_from=valid_from,
                metadata={"document_id": document.document_id, "session_id": document.metadata.get("session_id"), "timestamp": document.metadata.get("timestamp")},
            )
        relevant = list(case.relevant_document_ids)
        recent_ids = [document.document_id for document in documents[-max(1, int(top_k)):]]
        mode_rows["recent_window"].append(_row(case, recent_ids, relevant, 0.0))
        started = time.perf_counter()
        result = service.search(namespace, case.query, top_k=top_k)
        latency_ms = (time.perf_counter() - started) * 1000
        retrieved_ids = [str(item.get("metadata", {}).get("document_id") or "") for item in result["results"]]
        mode_rows["v56_memory"].append(_row(case, retrieved_ids, relevant, latency_ms))
        predictions.append({"case_id": case.case_id, "question_type": case.metadata.get("question_type"), "evidence_ids": relevant, "recent_window": recent_ids, "v56_memory": retrieved_ids})
    return {
        "benchmark": "longmemeval",
        "dataset_version": dataset.version,
        "case_count": len(cases),
        "answerable_case_count": sum(bool(case.relevant_document_ids) for case in cases),
        "top_k": int(top_k),
        "evidence_tier": "public-official-retrieval-only",
        "embedding_backend": str(getattr(service.embedding_provider, "name", "unknown")),
        "modes": {name: _aggregate(rows, top_k) for name, rows in mode_rows.items()},
        "predictions": predictions,
        "limitations": ["Retrieval-only diagnostic; no reader LLM answer accuracy is claimed.", "Abstention cases have no positive evidence and are excluded from retrieval recall."],
    }


def _row(case, retrieved_ids, evidence_ids, latency_ms):
    return {"case_id": case.case_id, "question_type": case.metadata.get("question_type"), "retrieved_ids": retrieved_ids, "evidence_ids": evidence_ids, "latency_ms": latency_ms}


def _aggregate(rows, top_k):
    answerable = [row for row in rows if row["evidence_ids"]]
    recalls = [len(set(row["retrieved_ids"][:top_k]) & set(row["evidence_ids"])) / len(set(row["evidence_ids"])) for row in answerable]
    latencies = sorted(float(row["latency_ms"]) for row in rows)
    return {
        "case_count": len(rows),
        "retrieval_recall_at_{0}".format(top_k): round(statistics.mean(recalls), 6) if recalls else 0.0,
        "p50_latency_ms": round(_percentile(latencies, 50), 3),
        "p95_latency_ms": round(_percentile(latencies, 95), 3),
    }


def _timestamp(value, index):
    text = str(value or "").strip()
    for pattern in ("%Y/%m/%d (%a) %H:%M", "%Y/%m/%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, pattern).replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            pass
    return (datetime(2000, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=index)).isoformat()


def _percentile(values, percentile):
    if not values:
        return 0.0
    position = (len(values) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(len(values) - 1, lower + 1)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction
