"""LongMemEval adapter, metrics and resumable prediction checkpoint."""

import json
import random
from collections import defaultdict
from pathlib import Path

from .base import BenchmarkCase, BenchmarkDataset, BenchmarkDocument


def load_longmemeval(path, split="test", version="cleaned-2025-09", limit=None):
    documents, cases = [], []
    for row_index, row in enumerate(_iter_json_array(path, limit=limit)):
        case_id = str(row.get("question_id") or row.get("id") or "question-{0}".format(row_index))
        sessions = row.get("haystack_sessions") or row.get("history") or row.get("sessions") or []
        parallel_ids = row.get("haystack_session_ids") or []
        parallel_dates = row.get("haystack_dates") or []
        session_ids = []
        session_occurrences = defaultdict(int)
        for session_index, session in enumerate(sessions):
            session_mapping = session if isinstance(session, dict) else {}
            session_id = str(
                session_mapping.get("session_id")
                or session_mapping.get("id")
                or (parallel_ids[session_index] if session_index < len(parallel_ids) else None)
                or "session-{0}".format(session_index)
            )
            occurrence = session_occurrences[session_id]
            session_occurrences[session_id] += 1
            document_id = case_id + ":" + session_id
            if occurrence:
                document_id += ":occurrence-{0}".format(occurrence + 1)
            session_ids.append((session_id, document_id))
            messages = (
                session_mapping.get("content")
                or session_mapping.get("messages")
                or session_mapping.get("turns")
                or (session if isinstance(session, list) else [])
            )
            documents.append(
                BenchmarkDocument(
                    document_id=document_id,
                    title="LongMemEval session {0}".format(session_id),
                    text=_session_text(messages),
                    metadata={
                        "question_id": case_id,
                        "session_id": session_id,
                        "timestamp": str(
                            session_mapping.get("date")
                            or session_mapping.get("timestamp")
                            or session_mapping.get("session_date")
                            or (parallel_dates[session_index] if session_index < len(parallel_dates) else "")
                        ),
                        "messages": tuple(messages),
                    },
                )
            )
        evidence_raw = row.get("answer_session_ids") or row.get("evidence_session_ids") or row.get("evidence") or []
        evidence_session_ids = _evidence_session_ids(evidence_raw)
        relevant = tuple(
            document_id
            for evidence_session_id in evidence_session_ids
            for session_id, document_id in session_ids
            if session_id == evidence_session_id
        )
        answer = row.get("answer")
        answers = tuple(str(item) for item in answer) if isinstance(answer, list) else (str(answer or ""),)
        question_type = str(row.get("question_type") or row.get("type") or "unknown")
        cases.append(
            BenchmarkCase(
                case_id=case_id,
                query=str(row.get("question") or row.get("query") or ""),
                relevant_document_ids=relevant,
                split=split,
                answers=answers,
                evidence_ids=tuple(evidence_session_ids),
                unanswerable=question_type in {"abstention", "unanswerable"} or not evidence_session_ids,
                metadata={"question_type": question_type, "session_count": len(sessions), "raw_evidence": evidence_raw},
            )
        )
    return BenchmarkDataset(
        name="longmemeval",
        version=version,
        documents=tuple(documents),
        cases=tuple(cases),
        metadata={"split": split, "source": "xiaowu0162/longmemeval-cleaned"},
    )


def _iter_json_array(path, limit=None, chunk_size=64 * 1024):
    """Yield a top-level JSON array without duplicating the full file in memory."""
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    yielded = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(chunk_size)
            buffer += chunk
            if not started:
                buffer = buffer.lstrip()
                if not buffer:
                    if chunk:
                        continue
                    raise ValueError("LongMemEval dataset is empty")
                if buffer[0] != "[":
                    raise ValueError("LongMemEval dataset must be a top-level JSON array")
                buffer = buffer[1:]
                started = True

            while True:
                buffer = buffer.lstrip()
                if buffer.startswith("]"):
                    return
                if buffer.startswith(","):
                    buffer = buffer[1:].lstrip()
                try:
                    row, end = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    break
                if not isinstance(row, dict):
                    raise ValueError("LongMemEval rows must be JSON objects")
                yield row
                yielded += 1
                if limit and yielded >= int(limit):
                    return
                buffer = buffer[end:]

            if not chunk:
                if buffer.strip():
                    raise ValueError("LongMemEval dataset ended inside a JSON row")
                return


def score_longmemeval(rows, top_k=5, bootstrap_samples=2000, seed=42):
    rows = list(rows)
    category_rows = defaultdict(list)
    for row in rows:
        category_rows[str(row.get("question_type") or "unknown")].append(row)
    accuracy = _mean([float(bool(row.get("correct"))) for row in rows])
    answerable = [row for row in rows if row.get("evidence_ids")]
    retrieval = [
        len(set(row.get("retrieved_ids", [])[:top_k]) & set(row.get("evidence_ids", []))) / max(1, len(set(row.get("evidence_ids", []))))
        for row in answerable
    ]
    return {
        "case_count": len(rows),
        "accuracy": round(accuracy, 6),
        "accuracy_ci95": _bootstrap_ci([float(bool(row.get("correct"))) for row in rows], bootstrap_samples, seed),
        "retrieval_recall_at_{0}".format(top_k): round(_mean(retrieval), 6),
        "categories": {
            name: {
                "case_count": len(values),
                "accuracy": round(_mean([float(bool(item.get("correct"))) for item in values]), 6),
            }
            for name, values in sorted(category_rows.items())
        },
    }


class PredictionCheckpoint:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def rows(self):
        if not self.path.exists():
            return []
        output, seen = [], set()
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = str(row.get("case_id") or "")
            if case_id and case_id not in seen:
                output.append(row)
                seen.add(case_id)
        return output

    def completed_ids(self):
        return {str(row["case_id"]) for row in self.rows()}

    def append(self, row):
        case_id = str(row.get("case_id") or "")
        if not case_id:
            raise ValueError("case_id is required")
        if case_id in self.completed_ids():
            return False
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        return True


def _session_text(messages):
    if isinstance(messages, str):
        return messages
    lines = []
    for message in messages:
        if isinstance(message, dict):
            lines.append("{0}: {1}".format(message.get("role") or message.get("speaker") or "unknown", message.get("content") or message.get("text") or ""))
        else:
            lines.append(str(message))
    return "\n".join(lines)


def _evidence_session_ids(value):
    if isinstance(value, dict):
        value = value.get("session_ids") or value.get("sessions") or []
    output = []
    for item in value or []:
        if isinstance(item, dict):
            item = item.get("session_id") or item.get("id")
        if item is not None:
            output.append(str(item))
    return output


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _bootstrap_ci(values, samples, seed):
    values = list(values)
    if not values:
        return [0.0, 0.0]
    generator = random.Random(seed)
    estimates = sorted(_mean(generator.choice(values) for _ in values) for _ in range(max(1, int(samples))))
    low = estimates[int(0.025 * (len(estimates) - 1))]
    high = estimates[int(0.975 * (len(estimates) - 1))]
    return [round(low, 6), round(high, 6)]
