"""Prepare and score 50 evidence-grounded PIM answers produced by Hermes."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path

try:
    from examples.storm_examples.prepare_pim_domain_pilot import parse_hermes_json_array
except ModuleNotFoundError as exc:
    if exc.name not in {"examples", "examples.storm_examples"}:
        raise
    from prepare_pim_domain_pilot import parse_hermes_json_array

from knowledge_storm.retrieval import multilingual_tokenize


def answer_f1(prediction, reference):
    predicted = multilingual_tokenize(str(prediction or ""))
    expected = multilingual_tokenize(str(reference or ""))
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return round(2.0 * precision * recall / (precision + recall), 6)


def build_answer_prompt(cases):
    payload = [
        {
            "case_id": case["case_id"],
            "question": case["question"],
            "contexts": case["contexts"],
        }
        for case in cases
    ]
    return """你是 PaperStorm 的中文文献问答 Reader。仅依据每题给出的检索证据回答，不得使用外部知识补齐。

严格要求：
1. 只返回 JSON 数组，不要 Markdown，不要额外解释。
2. 每项只含 case_id、answer、citations；case_id 原样保留。
3. citations 是实际支撑答案的 chunk_id 数组，只能引用 contexts 中出现的 ID。
4. 证据不足时 answer 明确说明“检索证据不足”，不得猜测。
5. 回答应直接、完整、专业，保留必要数字、条件和方法名称。

输入：
{0}
""".format(json.dumps(payload, ensure_ascii=False, indent=2))


def prepare_answer_cases(corpus_rows, case_rows, prediction_rows):
    corpus = {str(row["chunk_id"]): row for row in corpus_rows}
    cases = {str(row["case_id"]): row for row in case_rows}
    prepared = []
    for prediction in prediction_rows:
        case_id = str(prediction["case_id"])
        if case_id not in cases:
            raise ValueError("unknown prediction case_id: {0}".format(case_id))
        source = cases[case_id]
        contexts = []
        for chunk_id in prediction.get("ranked_document_ids") or ():
            chunk_id = str(chunk_id)
            if chunk_id not in corpus:
                raise ValueError("unknown retrieved chunk_id: {0}".format(chunk_id))
            row = corpus[chunk_id]
            contexts.append(
                {
                    "chunk_id": str(chunk_id),
                    "title": str(row.get("title") or ""),
                    "content": str(row["content"]),
                }
            )
        prepared.append(
            {
                "case_id": case_id,
                "question": source["question"],
                "reference_answer": source["reference_answer"],
                "evidence_chunk_ids": list(source["evidence_chunk_ids"]),
                "contexts": contexts,
            }
        )
    return prepared


def score_answers(prepared_cases, answer_rows):
    answers = {}
    for row in answer_rows:
        case_id = str(row.get("case_id") or "")
        if case_id in answers:
            raise ValueError("duplicate answer case: {0}".format(case_id))
        answers[case_id] = row
    rows = []
    for case in prepared_cases:
        answer = answers.get(case["case_id"])
        if answer is None:
            raise ValueError("missing answer case: {0}".format(case["case_id"]))
        retrieved = {item["chunk_id"] for item in case["contexts"]}
        gold = set(case["evidence_chunk_ids"])
        citations = {str(value) for value in answer.get("citations") or ()}
        valid_citations = citations & retrieved
        row = {
            "case_id": case["case_id"],
            "question": case["question"],
            "answer": str(answer.get("answer") or "").strip(),
            "reference_answer": case["reference_answer"],
            "citations": sorted(citations),
            "answer_f1": answer_f1(answer.get("answer"), case["reference_answer"]),
            "evidence_recall": len(retrieved & gold) / max(1, len(gold)),
            "citation_recall": len(valid_citations & gold) / max(1, len(gold)),
            "citation_precision": len(valid_citations) / max(1, len(citations)),
            "invalid_citation_count": len(citations - retrieved),
        }
        rows.append(row)
    metrics = {
        "case_count": len(rows),
        "answer_f1": _mean(row["answer_f1"] for row in rows),
        "evidence_recall": _mean(row["evidence_recall"] for row in rows),
        "citation_recall": _mean(row["citation_recall"] for row in rows),
        "citation_precision": _mean(row["citation_precision"] for row in rows),
        "invalid_citation_count": sum(row["invalid_citation_count"] for row in rows),
        "nonempty_answer_rate": _mean(bool(row["answer"]) for row in rows),
    }
    return {"schema": "paperstorm-pim-answer-pilot-v1", "metrics": metrics, "predictions": rows}


def prepare_prompts(corpus_path, cases_path, predictions_path, output_dir, batch_size=10):
    corpus = _read_jsonl(corpus_path)
    cases = _read_jsonl(cases_path)
    predictions = _read_jsonl(predictions_path)
    prepared = prepare_answer_cases(corpus, cases, predictions)
    root = Path(output_dir)
    prompt_dir = root / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(root / "prepared_cases.jsonl", prepared)
    paths = []
    for start in range(0, len(prepared), int(batch_size)):
        path = prompt_dir / "answer-{0:02d}.txt".format(start // int(batch_size) + 1)
        path.write_text(build_answer_prompt(prepared[start : start + int(batch_size)]), encoding="utf-8")
        paths.append(path)
    return prepared, paths


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _write_jsonl(path, rows):
    Path(path).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def _mean(values):
    values = [float(value) for value in values]
    return round(statistics.mean(values), 6) if values else 0.0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--responses-dir")
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args(argv)
    prepared, paths = prepare_prompts(
        args.corpus, args.cases, args.predictions, args.output_dir, args.batch_size
    )
    print("answer prompts: {0}".format(len(paths)))
    if args.responses_dir:
        answer_rows = []
        for path in sorted(Path(args.responses_dir).glob("answer-*.json")):
            answer_rows.extend(parse_hermes_json_array(path.read_text(encoding="utf-8-sig")))
        report = score_answers(prepared, answer_rows)
        report_path = Path(args.output_dir) / "metrics.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
