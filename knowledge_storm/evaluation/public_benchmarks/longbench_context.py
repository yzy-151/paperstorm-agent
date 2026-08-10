"""LongBench v2 selected-subset adapter and paired context metrics."""

import json
from pathlib import Path

from .base import BenchmarkCase, BenchmarkDataset, BenchmarkDocument


def load_longbench_v2(path, selected_subdomains=None, version="v2"):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        rows = payload if isinstance(payload, list) else payload.get("data", [])
    selected = set(selected_subdomains or [])
    documents, cases = [], []
    for index, row in enumerate(rows):
        subdomain = str(row.get("sub_domain") or row.get("dataset") or "unknown")
        if selected and subdomain not in selected:
            continue
        case_id = str(row.get("_id") or row.get("id") or "longbench-{0}".format(index))
        document_id = case_id + ":context"
        choices = {letter: str(row.get("choice_" + letter) or "") for letter in "ABCD"}
        documents.append(BenchmarkDocument(document_id=document_id, title="LongBench v2 context", text=str(row.get("context") or ""), metadata={"domain": row.get("domain"), "sub_domain": subdomain, "length": row.get("length")}))
        cases.append(BenchmarkCase(case_id=case_id, query=str(row.get("question") or row.get("query") or ""), relevant_document_ids=(document_id,), split="test", answers=(str(row.get("answer") or ""),), metadata={"domain": row.get("domain"), "sub_domain": subdomain, "difficulty": row.get("difficulty"), "length": row.get("length"), "choices": choices}))
    return BenchmarkDataset(name="longbench-v2-selected", version=version, documents=tuple(documents), cases=tuple(cases), metadata={"selected_subdomains": tuple(sorted(selected)), "official_aggregate": False})


def score_context_modes(predictions, baseline="full"):
    if baseline not in predictions:
        raise ValueError("baseline mode is required")
    baseline_rows = list(predictions[baseline])
    baseline_accuracy = _accuracy(baseline_rows)
    baseline_tokens = _mean([float(row.get("input_tokens", 0)) for row in baseline_rows])
    modes = {}
    for name, values in predictions.items():
        rows = list(values)
        accuracy = _accuracy(rows)
        tokens = _mean([float(row.get("input_tokens", 0)) for row in rows])
        modes[name] = {
            "case_count": len(rows),
            "accuracy": round(accuracy, 6),
            "mean_input_tokens": round(tokens, 3),
            "quality_delta": round(accuracy - baseline_accuracy, 6),
            "token_reduction": round(1.0 - tokens / baseline_tokens, 6) if baseline_tokens else 0.0,
        }
    return {"baseline": baseline, "paired_case_count": len(baseline_rows), "modes": modes}


def _accuracy(rows):
    rows = list(rows)
    return _mean([float(str(row.get("prediction", "")).strip().upper() == str(row.get("answer", "")).strip().upper()) for row in rows])


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0

