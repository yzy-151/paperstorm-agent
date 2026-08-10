"""Resumable, official-style QASPER answer-generation evaluation."""

import json
import re
import string
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROMPT_VERSION = "qasper-grounded-json-v2"


class LiteLLMJsonGenerator:
    """Small retrying LiteLLM adapter that never serializes credentials."""

    def __init__(
        self,
        model,
        api_key,
        api_base=None,
        max_tokens=256,
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
                    "response_format": {"type": "json_object"},
                    "cache": {"no-cache": True, "no-store": True},
                }
                if self.api_base:
                    kwargs["api_base"] = self.api_base
                response = self.completion(**kwargs)
                choice = response["choices"][0]
                message = choice.get("message") or {}
                return {
                    "text": str(message.get("content") or ""),
                    "usage": _usage_dict(response.get("usage") or {}),
                    "response_id": str(response.get("id") or ""),
                }
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.max_attempts:
                    self.sleep(min(30, 2**attempt))
        raise last_error


def complete_qasper_rankings(
    dataset,
    initial_rankings,
    embedding_provider,
    mode="hybrid_rerank",
    top_k=5,
    reranker=None,
    on_ranking=None,
):
    """Fill missing rankings with paper-scoped retrieval."""
    from ...paperstorm_retrieval_v41 import HybridPaperIndex

    rankings = {
        str(case_id): [str(value) for value in values]
        for case_id, values in (initial_rankings or {}).items()
    }
    documents_by_paper = {}
    for document in dataset.documents:
        paper_id = str(document.metadata.get("paper_id") or "")
        documents_by_paper.setdefault(paper_id, []).append(document)
    cases_by_paper = {}
    for case in dataset.cases:
        if case.case_id not in rankings:
            paper_id = str(case.metadata.get("paper_id") or "")
            cases_by_paper.setdefault(paper_id, []).append(case)
    for paper_id, cases in cases_by_paper.items():
        documents = documents_by_paper.get(paper_id) or []
        if not documents:
            raise ValueError("no documents found for paper_id: {0}".format(paper_id))
        index = HybridPaperIndex(
            [
                {
                    "chunk_id": document.document_id,
                    "document_id": document.document_id,
                    "title": document.title,
                    "content": document.text,
                    "retrieval_content": document.text,
                    "metadata": dict(document.metadata),
                }
                for document in documents
            ],
            embedding_provider=embedding_provider,
        )
        for case in cases:
            results = index.search(
                case.query,
                mode=mode,
                top_k=max(1, int(top_k)),
                candidate_k=max(20, int(top_k) * 4),
                reranker=reranker,
            )
            ranked_ids = [str(item["document_id"]) for item in results]
            rankings[case.case_id] = ranked_ids
            if on_ranking is not None:
                on_ranking(case.case_id, ranked_ids)
    return rankings


def run_qasper_generation(
    dataset,
    rankings,
    generate,
    output_dir,
    model_name,
    top_k=5,
    prompt_version=PROMPT_VERSION,
    on_prediction=None,
    parse_attempts=2,
    context_mode="topk",
    input_budget_tokens=None,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / "predictions.jsonl"
    existing = _read_predictions(checkpoint_path)
    document_map = dataset.document_map()
    for case in dataset.cases:
        if existing.get(case.case_id, {}).get("status") == "succeeded":
            continue
        ranked_documents = [
            document_id
            for document_id in rankings.get(case.case_id, [])[: max(1, int(top_k))]
            if document_id in document_map
        ]
        if context_mode == "full":
            paper_id = str(case.metadata.get("paper_id") or "")
            selected_documents = [
                document
                for document in dataset.documents
                if str(document.metadata.get("paper_id") or "") == paper_id
            ]
        elif context_mode == "v56":
            paper_id = str(case.metadata.get("paper_id") or "")
            ranked_set = set(ranked_documents)
            paper_documents = [
                document
                for document in dataset.documents
                if str(document.metadata.get("paper_id") or "") == paper_id
            ]
            selected_documents = [
                document_map[document_id] for document_id in ranked_documents
            ] + [
                document
                for document in paper_documents
                if document.document_id not in ranked_set
            ]
            selected_documents = _trim_to_budget(
                selected_documents, input_budget_tokens
            )
        else:
            selected_documents = [
                document_map[document_id] for document_id in ranked_documents
            ]
        prompt = build_qasper_prompt(
            case.query, selected_documents
        )
        started = time.perf_counter()
        text = ""
        usage = {}
        raw_responses = []
        parse_errors = []
        generation_attempts = 0
        try:
            parsed = None
            for generation_attempts in range(1, max(1, int(parse_attempts)) + 1):
                generated = generate(prompt)
                text, call_usage = _generation_payload(generated)
                usage = _merge_usage(usage, call_usage)
                raw_responses.append(text)
                try:
                    parsed = parse_generation_json(text)
                    break
                except (json.JSONDecodeError, ValueError) as exc:
                    parse_errors.append("{0}: {1}".format(type(exc).__name__, exc))
                    if generation_attempts >= max(1, int(parse_attempts)):
                        raise
            cited_ids = [
                str(value)
                for value in parsed.get("evidence_ids") or []
                if str(value) in ranked_documents
            ]
            abstained = bool(parsed.get("abstained")) or _is_unanswerable(
                parsed.get("answer")
            )
            answer = "Unanswerable" if abstained else str(parsed.get("answer") or "")
            status = "succeeded"
            error = None
        except Exception as exc:
            answer = ""
            cited_ids = []
            abstained = False
            status = "failed"
            error = "{0}: {1}".format(type(exc).__name__, exc)
        row = {
            "case_id": case.case_id,
            "split": case.split,
            "question": case.query,
            "answer": answer,
            "abstained": abstained,
            "evidence_ids": cited_ids,
            "evidence_texts": [
                str(
                    document_map[document_id].metadata.get("raw_text")
                    or document_map[document_id].text
                )
                for document_id in cited_ids
            ],
            "ranked_document_ids": ranked_documents,
            "status": status,
            "error": error,
            "raw_response": text,
            "raw_responses": raw_responses,
            "parse_errors": parse_errors,
            "generation_attempts": generation_attempts,
            "context_mode": context_mode,
            "context_document_count": len(selected_documents),
            "context_estimated_tokens": _estimate_tokens(
                "".join(document.text for document in selected_documents)
            ),
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 4),
            "usage": usage,
            "model": model_name,
            "prompt_version": prompt_version,
        }
        _append_jsonl(checkpoint_path, row)
        existing[case.case_id] = row
        if on_prediction is not None:
            on_prediction(row)
    ordered = {
        case.case_id: existing[case.case_id]
        for case in dataset.cases
        if case.case_id in existing
    }
    metrics = official_qasper_metrics(dataset.cases, ordered)
    usage = _sum_usage(ordered.values())
    successful = sum(row.get("status") == "succeeded" for row in ordered.values())
    report = {
        "benchmark": "qasper-answer-generation",
        "dataset_version": dataset.version,
        "split": _split_label(dataset.cases),
        "model": model_name,
        "prompt_version": prompt_version,
        "top_k": max(1, int(top_k)),
        "context_mode": context_mode,
        "input_budget_tokens": input_budget_tokens,
        "case_count": len(dataset.cases),
        "successful_predictions": successful,
        "failed_predictions": len(dataset.cases) - successful,
        "metrics": metrics,
        "usage": usage,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_official_qasper_predictions(
        output_path / "official_predictions.jsonl", ordered
    )
    _write_json(output_path / "metrics.json", report)
    return report


def _estimate_tokens(text):
    return max(1, len(str(text or "")) // 4)


def _trim_to_budget(documents, budget_tokens):
    if not budget_tokens:
        return documents
    used = 0
    selected = []
    for document in documents:
        cost = _estimate_tokens(document.text)
        if used + cost <= max(1, int(budget_tokens)):
            selected.append(document)
            used += cost
    return selected or (documents[:1] if documents else [])


def build_qasper_prompt(question, documents):
    evidence = []
    for index, document in enumerate(documents, start=1):
        evidence.append(
            "[{0}] evidence_id={1}\n{2}".format(
                index, document.document_id, document.text
            )
        )
    evidence_text = "\n\n".join(evidence) or "No evidence was retrieved."
    return (
        "You answer questions about one scientific paper. Use only the supplied "
        "evidence. Return the shortest directly supported answer span or comma-separated "
        "list, never an explanation. For yes/no questions return exactly Yes or No. "
        "Infer a Yes or No when the evidence directly establishes it. Return "
        "Unanswerable only when none of the evidence addresses the question. Cite only evidence_id "
        "values shown below. Return strict JSON with keys answer, abstained, and "
        "evidence_ids.\n\nQuestion:\n{0}\n\nEvidence:\n{1}\n\nJSON:"
    ).format(question, evidence_text)


def parse_generation_json(text):
    value = str(text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", value, re.DOTALL)
    if fenced:
        value = fenced.group(1)
    else:
        start = value.find("{")
        end = value.rfind("}")
        if start >= 0 and end > start:
            value = value[start : end + 1]
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = json.loads(_repair_bare_evidence_ids(value))
    if not isinstance(parsed, dict):
        raise ValueError("generation must be a JSON object")
    return parsed


def _repair_bare_evidence_ids(value):
    def replace(match):
        items = []
        for raw_item in match.group(1).split(","):
            item = raw_item.strip()
            if not item:
                continue
            if item.startswith('"') and item.endswith('"'):
                items.append(item)
            else:
                items.append(json.dumps(item, ensure_ascii=False))
        return '"evidence_ids": [{0}]'.format(", ".join(items))

    return re.sub(
        r'"evidence_ids"\s*:\s*\[(.*?)\]',
        replace,
        value,
        flags=re.DOTALL,
    )


def official_qasper_metrics(cases, predictions):
    answer_scores = []
    evidence_scores = []
    by_type = {"extractive": [], "abstractive": [], "boolean": [], "none": []}
    missing = 0
    exact_matches = []
    for case in cases:
        prediction = predictions.get(case.case_id)
        if not prediction or prediction.get("status") == "failed":
            missing += 1
            answer_scores.append(0.0)
            evidence_scores.append(0.0)
            exact_matches.append(0.0)
            continue
        references = _references(case)
        scored = [
            (
                _token_f1(prediction.get("answer", ""), reference["answer"]),
                reference["answer_type"],
                float(
                    _normalize_answer(prediction.get("answer", ""))
                    == _normalize_answer(reference["answer"])
                ),
            )
            for reference in references
        ]
        best_answer, answer_type, exact = max(scored, key=lambda item: item[0])
        answer_scores.append(best_answer)
        exact_matches.append(exact)
        by_type.setdefault(answer_type, []).append(best_answer)
        use_text_evidence = "evidence_texts" in prediction and any(
            "evidence_texts" in reference for reference in references
        )
        predicted_evidence = (
            prediction.get("evidence_texts") or []
            if use_text_evidence
            else prediction.get("evidence_ids") or []
        )
        evidence_scores.append(
            max(
                _paragraph_f1(
                    predicted_evidence,
                    reference.get("evidence_texts") or []
                    if use_text_evidence
                    else reference["evidence_ids"],
                )
                for reference in references
            )
        )
    return {
        "case_count": len(tuple(cases)),
        "answer_f1": _mean(answer_scores),
        "answer_exact_match": _mean(exact_matches),
        "answer_f1_by_type": {
            name: _mean(values) for name, values in by_type.items()
        },
        "evidence_f1": _mean(evidence_scores),
        "missing_predictions": missing,
    }


def load_rankings(path, mode="hybrid_rerank"):
    rankings = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("mode") == mode:
            rankings[str(row["case_id"])] = [
                str(value) for value in row.get("ranked_document_ids") or []
            ]
    return rankings


def write_official_qasper_predictions(path, predictions):
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for question_id, prediction in predictions.items():
            stream.write(
                json.dumps(
                    {
                        "question_id": str(question_id),
                        "predicted_answer": str(prediction.get("answer") or ""),
                        "predicted_evidence": [
                            str(value)
                            for value in prediction.get("evidence_texts") or []
                        ],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    temporary.replace(target)


def _references(case):
    references = tuple(case.metadata.get("qasper_references") or ())
    if references:
        return references
    answers = case.answers or (("Unanswerable",) if case.unanswerable else ("",))
    return tuple(
        {
            "answer": answer,
            "answer_type": "none" if case.unanswerable else "extractive",
            "evidence_ids": tuple(case.evidence_ids),
        }
        for answer in answers
    )


def _normalize_answer(value):
    lowered = str(value or "").lower()
    without_punctuation = "".join(
        character for character in lowered if character not in set(string.punctuation)
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def _token_f1(prediction, reference):
    predicted = _normalize_answer(prediction).split()
    expected = _normalize_answer(reference).split()
    common = Counter(predicted) & Counter(expected)
    overlap = sum(common.values())
    if not overlap:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def _paragraph_f1(prediction, reference):
    predicted = set(prediction or [])
    expected = set(reference or [])
    if not predicted and not expected:
        return 1.0
    overlap = len(predicted & expected)
    if not overlap:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def _generation_payload(generated):
    if isinstance(generated, dict):
        return str(generated.get("text") or ""), dict(generated.get("usage") or {})
    return str(generated or ""), {}


def _usage_dict(usage):
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    elif not isinstance(usage, dict):
        try:
            usage = dict(usage)
        except (TypeError, ValueError):
            usage = {}
    return {
        key: int(usage.get(key) or 0)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def _merge_usage(left, right):
    left = _usage_dict(left)
    right = _usage_dict(right)
    return {
        key: int(left.get(key) or 0) + int(right.get(key) or 0)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def _is_unanswerable(answer):
    return _normalize_answer(answer) in {"unanswerable", "cannot be answered"}


def _read_predictions(path):
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["case_id"])] = row
    return rows


def _append_jsonl(path, row):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _sum_usage(rows):
    output = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for row in rows:
        usage = row.get("usage") or {}
        output["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        output["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        output["total_tokens"] += int(
            usage.get("total_tokens")
            or int(usage.get("prompt_tokens") or 0)
            + int(usage.get("completion_tokens") or 0)
        )
    return output


def _mean(values):
    return round(sum(values) / len(values), 6) if values else 0.0


def _split_label(cases):
    values = sorted({case.split for case in cases})
    return values[0] if len(values) == 1 else "+".join(values)
