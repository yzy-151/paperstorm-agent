"""Dependency-free public benchmark metrics."""

import math
import re
from collections import Counter


def retrieval_metrics(ranked_ids, relevance, cutoffs=(5, 10)):
    ranked = list(dict.fromkeys(str(value) for value in ranked_ids))
    relevant = {
        str(key): int(value) for key, value in relevance.items() if int(value) > 0
    }
    output = {}
    for cutoff in sorted({max(1, int(value)) for value in cutoffs}):
        selected = ranked[:cutoff]
        hits = [document_id for document_id in selected if document_id in relevant]
        output["recall_at_{0}".format(cutoff)] = (
            round(len(hits) / len(relevant), 6) if relevant else 0.0
        )
        first_rank = next(
            (
                index
                for index, document_id in enumerate(selected, start=1)
                if document_id in relevant
            ),
            None,
        )
        output["mrr_at_{0}".format(cutoff)] = (
            round(1.0 / first_rank, 6) if first_rank else 0.0
        )
        dcg = sum(
            (2 ** relevant[document_id] - 1) / math.log2(index + 1)
            for index, document_id in enumerate(selected, start=1)
            if document_id in relevant
        )
        ideal = sorted(relevant.values(), reverse=True)[:cutoff]
        idcg = sum(
            (2**score - 1) / math.log2(index + 1)
            for index, score in enumerate(ideal, start=1)
        )
        output["ndcg_at_{0}".format(cutoff)] = round(dcg / idcg, 6) if idcg else 0.0
    return output


def answer_metrics(prediction, references):
    predicted_tokens = _answer_tokens(prediction)
    best_exact = 0.0
    best_f1 = 0.0
    for reference in references or [""]:
        reference_tokens = _answer_tokens(reference)
        exact = float(str(prediction).strip().lower() == str(reference).strip().lower())
        common = Counter(predicted_tokens) & Counter(reference_tokens)
        overlap = sum(common.values())
        precision = overlap / len(predicted_tokens) if predicted_tokens else 0.0
        recall = overlap / len(reference_tokens) if reference_tokens else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        best_exact = max(best_exact, exact)
        best_f1 = max(best_f1, f1)
    return {"exact_match": round(best_exact, 6), "token_f1": round(best_f1, 6)}


def evidence_metrics(predicted_ids, reference_ids):
    predicted = set(predicted_ids or [])
    reference = set(reference_ids or [])
    overlap = len(predicted & reference)
    precision = overlap / len(predicted) if predicted else float(not reference)
    recall = overlap / len(reference) if reference else float(not predicted)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def _answer_tokens(value):
    return re.findall(r"[a-z0-9]+|[\u3400-\u9fff]", str(value or "").lower())
