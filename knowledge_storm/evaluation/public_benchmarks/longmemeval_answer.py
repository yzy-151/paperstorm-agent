"""Resumable LongMemEval answer-generation evaluation (1/4 protocol run)."""

import json
import re
import string
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


class LiteLLMPlainGenerator:
    """Small retrying LiteLLM adapter for free-form answers (no JSON format)."""

    def __init__(
        self,
        model,
        api_key,
        api_base=None,
        max_tokens=128,
        timeout=45,
        max_attempts=4,
        completion=None,
        sleep=time.sleep,
    ):
        if not api_key:
            raise ValueError("an API key is required for generation evaluation")
        if completion is None:
            import litellm

            completion = litellm.completion
        self.model = str(model)
        self.api_key = str(api_key)
        self.api_base = str(api_base) if api_base else None
        self.max_tokens = max(1, int(max_tokens))
        self.timeout = max(1, int(timeout))
        self.max_attempts = max(1, int(max_attempts))
        self.completion = completion
        self.sleep = sleep

    def __call__(self, prompt):
        last_error = None
        for attempt in range(self.max_attempts):
            try:
                kwargs = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "api_key": self.api_key,
                    "temperature": 0.0,
                    "max_tokens": self.max_tokens,
                    "timeout": self.timeout,
                    "cache": {"no-cache": True, "no-store": True},
                }
                if self.api_base:
                    kwargs["api_base"] = self.api_base
                response = self.completion(**kwargs)
                choice = response["choices"][0]
                message = choice.get("message") or {}
                return {
                    "text": str(message.get("content") or "").strip(),
                    "usage": _usage_dict(response.get("usage") or {}),
                    "response_id": str(response.get("id") or ""),
                }
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.max_attempts:
                    self.sleep(min(30, 2**attempt))
        raise last_error


def run_longmemeval_answers(
    dataset,
    service,
    generate,
    output_dir,
    model_name,
    top_k=5,
    evidence_chars=5000,
    prompt_version="longmemeval-grounded-v1",
    on_progress=None,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / "predictions.jsonl"
    existing = _read_predictions(checkpoint_path)
    rows = []
    for case in dataset.cases:
        if existing.get(case.case_id, {}).get("status") == "succeeded":
            rows.append(existing[case.case_id])
            continue
        namespace = "longmemeval:" + case.case_id
        started = time.perf_counter()
        try:
            retrieval = service.search(namespace, case.query, top_k=max(1, int(top_k)))
            results = retrieval.get("results") or []
            retrieved_ids = [
                str(item.get("metadata", {}).get("document_id") or item.get("id") or "")
                for item in results
            ]
            evidence = "\n\n".join(
                "[session {0}]\n{1}".format(
                    index + 1,
                    _slice_evidence(
                        str(item.get("content") or ""), int(evidence_chars)
                    ),
                )
                for index, item in enumerate(results)
            )
            prompt = build_longmemeval_prompt(case.query, evidence)
            generated = generate(prompt)
            text = str(generated.get("text") or "")
            usage = generated.get("usage") or {}
            status = "succeeded"
            error = None
        except Exception as exc:
            text = ""
            usage = {}
            retrieved_ids = []
            evidence = ""
            status = "failed"
            error = "{0}: {1}".format(type(exc).__name__, exc)
        score = _score_case(case, text) if status == "succeeded" else {}
        row = {
            "case_id": case.case_id,
            "question_type": case.metadata.get("question_type", ""),
            "question": case.query,
            "prediction": text,
            "gold_answers": list(case.answers),
            "retrieved_ids": retrieved_ids,
            "evidence_recall_at_{0}".format(top_k): _recall(
                retrieved_ids[: max(1, int(top_k))], case.relevant_document_ids
            ),
            "score": score,
            "status": status,
            "error": error,
            "estimated_input_tokens": _estimate_tokens(prompt),
            "usage": usage,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 4),
            "model": model_name,
            "prompt_version": prompt_version,
        }
        _append_jsonl(checkpoint_path, row)
        rows.append(row)
        if on_progress is not None:
            on_progress(row)
    ordered = {row["case_id"]: row for row in rows}
    report = _aggregate(dataset, ordered, top_k=top_k)
    report.update(
        {
            "benchmark": "longmemeval-answer-generation",
            "dataset_version": dataset.version,
            "model": model_name,
            "prompt_version": prompt_version,
            "top_k": max(1, int(top_k)),
            "case_count": len(dataset.cases),
            "successful_predictions": sum(
                1 for row in rows if row.get("status") == "succeeded"
            ),
            "failed_predictions": sum(
                1 for row in rows if row.get("status") != "succeeded"
            ),
            "usage": _sum_usage(rows),
            "cost_estimate_usd": _cost_estimate(rows),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "evidence_tier": "public-official-1of4-protocol-run",
            "limitations": [
                "1/4 protocol run: 125 of 500 questions; unofficial answer judge "
                "(token-F1/EM/abstention rule) instead of the official LLM judge.",
                "Evidence is the top-K retrieved sessions truncated to evidence_chars.",
            ],
        }
    )
    _write_json(output_path / "metrics.json", report)
    return report


def build_longmemeval_prompt(question, evidence):
    return (
        "You are answering questions about a long user chat history with multiple "
        "sessions. The answer IS present in one of the sessions. Read every session "
        "carefully. Reply with the shortest factual answer span or a comma-separated "
        "list, or exactly Yes/No. Reply exactly Unanswerable ONLY if no session "
        "contains the answer. Do not explain.\n\nQuestion:\n{0}\n\nChat history:\n{1}\n\nAnswer:"
    ).format(str(question or ""), str(evidence or "No evidence was retrieved."))


def _score_case(case, prediction):
    golds = [str(gold) for gold in (case.answers or ()) if str(gold).strip()]
    question_type = str(case.metadata.get("question_type") or "unknown")
    abstain_gold = bool(case.unanswerable) or not golds
    pred_abstain = _is_unanswerable(prediction)
    if abstain_gold:
        return {
            "question_type": question_type,
            "abstention": True,
            "correct": bool(pred_abstain),
            "em": bool(pred_abstain),
            "token_f1": 1.0 if pred_abstain else 0.0,
        }
    if pred_abstain:
        return {
            "question_type": question_type,
            "abstention": False,
            "correct": False,
            "em": False,
            "token_f1": 0.0,
        }
    normalized = _normalize(prediction)
    em = any(normalized == _normalize(gold) for gold in golds)
    token_f1 = max(_token_f1(prediction, gold) for gold in golds)
    boolean = question_type in {"boolean", "abstention"}
    correct = em if boolean else em or token_f1 >= 1.0
    return {
        "question_type": question_type,
        "abstention": False,
        "correct": bool(correct),
        "em": em,
        "token_f1": token_f1,
    }


def _aggregate(dataset, ordered, top_k=5):
    by_type = defaultdict(list)
    answerable = []
    abstentions = []
    recalls = []
    for case in dataset.cases:
        row = ordered.get(case.case_id)
        if not row or row.get("status") != "succeeded":
            continue
        score = row.get("score") or {}
        question_type = score.get("question_type") or case.metadata.get(
            "question_type", "unknown"
        )
        by_type[question_type].append(score)
        if score.get("abstention"):
            abstentions.append(score)
        else:
            answerable.append(score)
        recalls.append(row.get("evidence_recall_at_{0}".format(top_k)) or 0.0)

    def summarize(scores):
        if not scores:
            return {}
        return {
            "n": len(scores),
            "accuracy": round(sum(s.get("correct", False) for s in scores) / len(scores), 4),
            "token_f1": round(sum(s.get("token_f1", 0.0) for s in scores) / len(scores), 4),
            "em": round(sum(s.get("em", False) for s in scores) / len(scores), 4),
        }

    return {
        "overall": summarize(answerable),
        "abstention": summarize(abstentions),
        "by_question_type": {key: summarize(values) for key, values in sorted(by_type.items())},
        "evidence_recall_at_{0}".format(top_k): round(
            sum(recalls) / len(recalls), 4
        )
        if recalls
        else 0.0,
    }


def _normalize(text):
    return re.sub(
        r"\s+",
        " ",
        str(text or "")
        .strip()
        .lower()
        .translate(str.maketrans("", "", string.punctuation)),
    ).strip()


def _is_unanswerable(text):
    normalized = _normalize(text)
    return any(
        marker in normalized
        for marker in (
            "unanswerable",
            "cannot be answered",
            "not enough information",
            "no information",
            "not mentioned",
            "insufficient",
            "unknown",
        )
    )


def _token_f1(prediction, gold):
    predicted = set(_normalize(prediction).split())
    gold_set = set(_normalize(gold).split())
    if not predicted and not gold_set:
        return 1.0
    if not predicted or not gold_set:
        return 0.0
    overlap = len(predicted & gold_set)
    precision = overlap / len(predicted)
    recall = overlap / len(gold_set)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _recall(ranked_ids, relevant_ids):
    relevant = set(str(item) for item in (relevant_ids or ()))
    if not relevant:
        return 0.0
    return len(set(ranked_ids) & relevant) / len(relevant)


def _estimate_tokens(text):
    return max(1, len(str(text or "")) // 4)


def _slice_evidence(content, head_chars):
    """Keep the head and tail of a long session so answers near either end survive."""
    head_chars = max(1, int(head_chars))
    if len(content) <= head_chars * 2:
        return content
    return "{0}\n...[truncated {1} chars]...\n{2}".format(
        content[:head_chars], len(content) - head_chars * 2, content[-head_chars:]
    )


def _read_predictions(path):
    if not path.exists():
        return {}
    output = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            output[str(row["case_id"])] = row
    return output


def _append_jsonl(path, row):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


def _write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _usage_dict(usage):
    return {
        key: int(usage.get(key) or 0)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def _sum_usage(rows):
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for row in rows:
        for key in totals:
            totals[key] += int((row.get("usage") or {}).get(key) or 0)
    return totals


def _cost_estimate(rows):
    usage = _sum_usage(rows)
    return round(
        usage["prompt_tokens"] / 1e6 * 0.27
        + usage["completion_tokens"] / 1e6 * 1.10,
        4,
    )
