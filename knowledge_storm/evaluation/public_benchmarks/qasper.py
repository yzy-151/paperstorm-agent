"""QASPER adapter for scientific QA, evidence selection, and context budgets."""

import json
import re
import statistics
from collections import Counter
from pathlib import Path

from .base import BenchmarkCase, BenchmarkDataset, BenchmarkDocument
from .metrics import answer_metrics, evidence_metrics


def load_qasper_huggingface(
    split="validation", cache_dir=None, revision=None, smoke_limit=None
):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "QASPER download requires optional dependency datasets"
        ) from exc
    kwargs = {"split": split}
    if cache_dir:
        kwargs["cache_dir"] = str(cache_dir)
    if revision:
        kwargs["revision"] = revision
    records = load_dataset("allenai/qasper", **kwargs)
    if smoke_limit:
        records = records.select(range(min(len(records), int(smoke_limit))))
    fingerprint = getattr(records, "_fingerprint", None)
    version = "allenai/qasper@{0}".format(revision or fingerprint or "current")
    return load_qasper_records(records, split=split, version=version)


def load_qasper_official_json(path, split="test"):
    """Load the official QASPER v0.3 dictionary without Hugging Face/network."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for paper_id, paper in payload.items():
        sections = list(paper.get("full_text") or [])
        qas = list(paper.get("qas") or [])
        records.append(
            {
                "id": str(paper_id),
                "title": str(paper.get("title") or paper_id),
                "full_text": {
                    "section_name": [str(item.get("section_name") or "") for item in sections],
                    "paragraphs": [list(item.get("paragraphs") or []) for item in sections],
                },
                "qas": {
                    "question": [str(item.get("question") or "") for item in qas],
                    "question_id": [str(item.get("question_id") or "") for item in qas],
                    "answers": [
                        {"answer": [dict(answer.get("answer") or {}) for answer in item.get("answers") or []]}
                        for item in qas
                    ],
                },
            }
        )
    return load_qasper_records(
        records,
        split=split,
        version="qasper-official-v0.3:{0}".format(path.name),
    )


def evaluate_qasper_context_budget(
    dataset,
    ranking_rows,
    mode="hybrid_rerank",
    model_context_tokens=8192,
    output_reserve_tokens=1536,
    evidence_budget_ratio=0.7,
):
    """Measure evidence retention after real ranked passages enter Context v5.6.

    This is deliberately a context-governance diagnostic, not a generated-answer
    quality score. Retrieval quality remains separately attributable to the
    upstream ranking run.
    """
    from ...context_engine import estimate_tokens
    from ...context_engine import _ContextEngineConfigBase, _ContextEngineCore

    rankings = {
        str(row.get("case_id")): row
        for row in ranking_rows
        if str(row.get("mode")) == str(mode)
    }
    document_map = dataset.document_map()
    documents_by_paper = {}
    for document in dataset.documents:
        paper_id = str(document.metadata.get("paper_id") or "")
        documents_by_paper.setdefault(paper_id, []).append(document)
    config = _ContextEngineConfigBase(
        model_context_tokens=int(model_context_tokens),
        output_reserve_tokens=int(output_reserve_tokens),
        layer_caps={
            "pinned": 0.2,
            "active": 0.08,
            "summary": 0.02,
            "memory": 0.0,
            "evidence": float(evidence_budget_ratio),
            "artifact": 0.0,
        },
    )
    engine = _ContextEngineCore(config=config)
    rows = []
    for case in dataset.cases:
        ranking = rankings.get(str(case.case_id))
        if not ranking:
            continue
        ranked_ids = [
            str(value)
            for value in ranking.get("ranked_document_ids") or []
            if str(value) in document_map
        ]
        evidence = [
            {
                "id": document_id,
                "source_id": document_id,
                "content": document_map[document_id].text,
            }
            for document_id in ranked_ids
        ]
        assembled = engine.assemble(
            [
                {
                    "id": "question:" + str(case.case_id),
                    "role": "system",
                    "content": "Question: " + str(case.query),
                    "metadata": {"pinned": True},
                }
            ],
            evidence=evidence,
        )
        selected_ids = {
            str(item.get("metadata", {}).get("source_id"))
            for item in assembled["messages"]
            if item.get("metadata", {}).get("context_layer") == "evidence"
        }
        relevant = set(case.evidence_ids)
        paper_id = str(case.metadata.get("paper_id") or "")
        full_tokens = sum(
            max(1, estimate_tokens(document.text) + 4)
            for document in documents_by_paper.get(paper_id, [])
        )
        context_tokens = int(assembled["token_usage"]["total"])
        rows.append(
            {
                "case_id": str(case.case_id),
                "paper_id": paper_id,
                "retrieved_ids": ranked_ids,
                "selected_ids": sorted(selected_ids),
                "relevant_ids": sorted(relevant),
                "retrieved_evidence_retention": len(selected_ids.intersection(ranked_ids)) / max(1, len(ranked_ids)),
                "gold_evidence_recall_before_context": len(relevant.intersection(ranked_ids)) / max(1, len(relevant)),
                "gold_evidence_recall_after_context": len(relevant.intersection(selected_ids)) / max(1, len(relevant)),
                "full_document_tokens": full_tokens,
                "context_tokens": context_tokens,
                "input_limit_tokens": config.input_limit,
                "context_to_full_document_ratio": context_tokens / max(1, full_tokens),
                "over_budget": context_tokens > config.input_limit,
                "validation": assembled["validation"],
            }
        )
    report = {
        "benchmark": "qasper-context-budget-v56",
        "dataset_version": dataset.version,
        "case_count": len(rows),
        "retrieval_mode": mode,
        "model_context_tokens": int(model_context_tokens),
        "output_reserve_tokens": int(output_reserve_tokens),
        "input_limit_tokens": config.input_limit,
        "evidence_budget_ratio": float(evidence_budget_ratio),
        "retrieved_evidence_retention": _row_mean(rows, "retrieved_evidence_retention"),
        "gold_evidence_recall_before_context": _row_mean(rows, "gold_evidence_recall_before_context"),
        "gold_evidence_recall_after_context": _row_mean(rows, "gold_evidence_recall_after_context"),
        "mean_context_to_full_document_ratio": _row_mean(rows, "context_to_full_document_ratio"),
        "p50_context_tokens": round(statistics.median([row["context_tokens"] for row in rows]), 3) if rows else 0.0,
        "over_budget_rate": _row_mean(rows, "over_budget"),
        "validation_pass_rate": round(
            sum(bool(row["validation"].get("pinned_preserved") and row["validation"].get("tool_pairs_valid")) for row in rows) / max(1, len(rows)),
            6,
        ),
        "evidence_tier": "public-official-context-diagnostic",
        "limitations": [
            "This measures context-budget retention, not generated-answer correctness.",
            "Gold evidence missing from retrieval cannot be recovered by context assembly.",
        ],
    }
    return report, rows


def evaluate_qasper_parent_context_coverage(
    dataset,
    ranking_rows,
    mode="hybrid_governed",
    parent_budget_tokens=512,
):
    """Measure extra gold evidence made visible by bounded section expansion.

    Rankings are frozen: this diagnostic attributes any change only to the
    parent-context layer. It measures Reader input coverage, not Answer F1.
    """
    from .runner import HashEmbeddingProvider, _benchmark_nodes
    from ...retrieval import HybridPaperIndex

    budget = int(parent_budget_tokens)
    if budget <= 0:
        raise ValueError("parent_budget_tokens must be positive")
    rankings = {
        str(row.get("case_id")): row
        for row in ranking_rows
        if str(row.get("mode")) == str(mode)
    }
    document_map = dataset.document_map()
    index = HybridPaperIndex(
        _benchmark_nodes(dataset, structured=True),
        embedding_provider=HashEmbeddingProvider(dim=32),
    )
    chunk_map = {str(item["chunk_id"]): item for item in index.chunks}
    rows = []
    for case in dataset.cases:
        ranking = rankings.get(str(case.case_id))
        if not ranking:
            continue
        ranked_ids = [
            str(value)
            for value in ranking.get("ranked_document_ids") or []
            if str(value) in chunk_map
        ]
        gold_ids = {
            str(value)
            for value in (case.evidence_ids or case.relevant_document_ids)
            if str(value) in document_map
        }
        if not gold_ids:
            continue
        ranked = [
            dict(chunk_map[document_id], final_rank=rank, score=1.0 / rank)
            for rank, document_id in enumerate(ranked_ids, start=1)
        ]
        expanded = index.expand_parent_context(ranked, budget)
        child_visible_text = "\n".join(
            _normalized_evidence_text(item.get("content") or "")
            for item in ranked
        )
        visible_text = "\n".join(
            _normalized_evidence_text(item.get("expanded_content") or "")
            for item in expanded
        )
        child_covered = gold_ids.intersection(ranked_ids)
        expanded_covered = {
            document_id
            for document_id in gold_ids
            if _normalized_evidence_text(document_map[document_id].text)
            and _normalized_evidence_text(document_map[document_id].text)
            in visible_text
        }
        expanded_covered.update(child_covered)
        rows.append(
            {
                "case_id": str(case.case_id),
                "paper_id": str(case.metadata.get("paper_id") or ""),
                "ranked_document_ids": ranked_ids,
                "gold_evidence_ids": sorted(gold_ids),
                "child_covered_gold_ids": sorted(child_covered),
                "expanded_covered_gold_ids": sorted(expanded_covered),
                "additional_gold_evidence_ids": sorted(
                    expanded_covered - child_covered
                ),
                "child_gold_evidence_recall": len(child_covered) / len(gold_ids),
                "expanded_gold_evidence_recall": len(expanded_covered)
                / len(gold_ids),
                "child_gold_token_coverage": sum(
                    _evidence_token_coverage(
                        document_map[document_id].text, child_visible_text
                    )
                    for document_id in gold_ids
                )
                / len(gold_ids),
                "expanded_gold_token_coverage": sum(
                    _evidence_token_coverage(
                        document_map[document_id].text, visible_text
                    )
                    for document_id in gold_ids
                )
                / len(gold_ids),
                "expanded_parent_count": sum(
                    bool(item.get("parent_context")) for item in expanded
                ),
                "parent_tokens_used": sum(
                    int((item.get("parent_allocation") or {}).get("used_tokens") or 0)
                    for item in expanded
                ),
            }
        )
    child_recall = _row_mean(rows, "child_gold_evidence_recall")
    expanded_recall = _row_mean(rows, "expanded_gold_evidence_recall")
    child_token_coverage = _row_mean(rows, "child_gold_token_coverage")
    expanded_token_coverage = _row_mean(rows, "expanded_gold_token_coverage")
    report = {
        "benchmark": "qasper-parent-context-coverage-v1",
        "dataset_version": dataset.version,
        "case_count": len(rows),
        "retrieval_mode": mode,
        "parent_budget_tokens": budget,
        "child_gold_evidence_recall": child_recall,
        "expanded_gold_evidence_recall": expanded_recall,
        "recall_delta": round(expanded_recall - child_recall, 6),
        "child_gold_token_coverage": child_token_coverage,
        "expanded_gold_token_coverage": expanded_token_coverage,
        "token_coverage_delta": round(
            expanded_token_coverage - child_token_coverage, 6
        ),
        "improved_case_count": sum(
            row["expanded_gold_evidence_recall"]
            > row["child_gold_evidence_recall"]
            for row in rows
        ),
        "mean_parent_tokens_used": _row_mean(rows, "parent_tokens_used"),
        "evidence_tier": "public-official-context-diagnostic",
        "limitations": [
            "Frozen rankings isolate parent-context impact from retrieval changes.",
            "This measures visible gold evidence, not generated-answer correctness.",
        ],
    }
    return report, rows


def _normalized_evidence_text(text):
    return " ".join(str(text or "").split())


def _evidence_token_coverage(gold_text, visible_text):
    gold = Counter(re.findall(r"[a-z0-9]+", str(gold_text or "").lower()))
    if not gold:
        return 0.0
    visible = Counter(re.findall(r"[a-z0-9]+", str(visible_text or "").lower()))
    matched = sum(min(count, visible.get(token, 0)) for token, count in gold.items())
    return matched / sum(gold.values())


def _row_mean(rows, key):
    return round(
        sum(float(row[key]) for row in rows) / len(rows),
        6,
    ) if rows else 0.0


def load_qasper_records(records, split="validation", version="allenai/qasper"):
    documents = []
    cases = []
    for record in records:
        paper_id = str(record.get("id") or record.get("paper_id") or "")
        paragraph_lookup = {}
        full_text = record.get("full_text") or {}
        sections = full_text.get("section_name") or []
        paragraphs = full_text.get("paragraphs") or []
        for section_index, section_paragraphs in enumerate(paragraphs):
            section = (
                str(sections[section_index]) if section_index < len(sections) else ""
            )
            for paragraph_index, paragraph in enumerate(section_paragraphs or []):
                document_id = "{0}::section-{1}::paragraph-{2}".format(
                    paper_id, section_index, paragraph_index
                )
                paragraph_text = str(paragraph or "")
                paragraph_lookup.setdefault(paragraph_text.strip(), []).append(
                    document_id
                )
                documents.append(
                    BenchmarkDocument(
                        document_id=document_id,
                        title=str(record.get("title") or paper_id),
                        text="\n".join(
                            value for value in (section, paragraph_text) if value
                        ),
                        metadata={
                            "paper_id": paper_id,
                            "section": section,
                            "section_index": section_index,
                            "paragraph_index": paragraph_index,
                            "raw_text": paragraph_text,
                        },
                    )
                )
        qas = record.get("qas") or {}
        questions = qas.get("question") or []
        question_ids = qas.get("question_id") or []
        answer_groups = qas.get("answers") or []
        for index, question in enumerate(questions):
            answer_group = answer_groups[index] if index < len(answer_groups) else {}
            annotations = answer_group.get("answer") or []
            answers = []
            evidence_ids = []
            unanswerable_votes = []
            references = []
            for annotation in annotations:
                unanswerable_votes.append(bool(annotation.get("unanswerable")))
                answers.extend(
                    str(value)
                    for value in annotation.get("extractive_spans") or []
                    if str(value).strip()
                )
                if str(annotation.get("free_form_answer") or "").strip():
                    answers.append(str(annotation["free_form_answer"]).strip())
                if annotation.get("yes_no") is not None:
                    answers.append("yes" if annotation.get("yes_no") else "no")
                for evidence in annotation.get("evidence") or []:
                    evidence_ids.extend(paragraph_lookup.get(str(evidence).strip(), []))
                references.append(_qasper_reference(annotation, paragraph_lookup))
            evidence_ids = list(dict.fromkeys(evidence_ids))
            case_id = (
                str(question_ids[index])
                if index < len(question_ids)
                else "{0}::q-{1}".format(paper_id, index)
            )
            cases.append(
                BenchmarkCase(
                    case_id=case_id,
                    query=str(question),
                    relevant_document_ids=tuple(evidence_ids),
                    relevance={document_id: 1 for document_id in evidence_ids},
                    split=split,
                    answers=tuple(dict.fromkeys(answers)),
                    evidence_ids=tuple(evidence_ids),
                    unanswerable=bool(unanswerable_votes and all(unanswerable_votes)),
                    metadata={
                        "paper_id": paper_id,
                        "qasper_references": tuple(references),
                    },
                )
            )
    return BenchmarkDataset(
        name="qasper",
        version=version,
        documents=tuple(documents),
        cases=tuple(cases),
        metadata={"split": split},
    )


def evaluate_qasper_predictions(cases, predictions):
    cases = tuple(cases)
    answer_rows = []
    evidence_rows = []
    true_abstentions = 0
    predicted_abstentions = 0
    correct_abstentions = 0
    for case in cases:
        prediction = predictions.get(case.case_id) or {}
        abstained = bool(prediction.get("abstained"))
        if case.unanswerable:
            true_abstentions += 1
        if abstained:
            predicted_abstentions += 1
        if case.unanswerable and abstained:
            correct_abstentions += 1
        if not case.unanswerable:
            answer_rows.append(
                answer_metrics(prediction.get("answer", ""), case.answers)
            )
            evidence_rows.append(
                evidence_metrics(
                    prediction.get("evidence_ids") or [], case.evidence_ids
                )
            )
    return {
        "case_count": len(cases),
        "answer_case_count": len(answer_rows),
        "answer_exact_match": _mean(answer_rows, "exact_match"),
        "answer_token_f1": _mean(answer_rows, "token_f1"),
        "evidence_precision": _mean(evidence_rows, "precision"),
        "evidence_recall": _mean(evidence_rows, "recall"),
        "evidence_f1": _mean(evidence_rows, "f1"),
        "abstention_precision": (
            round(correct_abstentions / predicted_abstentions, 6)
            if predicted_abstentions
            else 0.0
        ),
        "abstention_recall": (
            round(correct_abstentions / true_abstentions, 6)
            if true_abstentions
            else 0.0
        ),
    }


def _mean(rows, key):
    return round(sum(float(row[key]) for row in rows) / len(rows), 6) if rows else 0.0


def _qasper_reference(annotation, paragraph_lookup):
    if annotation.get("unanswerable"):
        answer = "Unanswerable"
        answer_type = "none"
    elif annotation.get("extractive_spans"):
        answer = ", ".join(str(value) for value in annotation["extractive_spans"])
        answer_type = "extractive"
    elif str(annotation.get("free_form_answer") or "").strip():
        answer = str(annotation["free_form_answer"]).strip()
        answer_type = "abstractive"
    elif annotation.get("yes_no") is not None:
        answer = "Yes" if annotation.get("yes_no") else "No"
        answer_type = "boolean"
    else:
        answer = "Unanswerable"
        answer_type = "none"
    evidence_texts = tuple(
        str(value).strip()
        for value in annotation.get("evidence") or []
        if "FLOAT SELECTED" not in str(value)
    )
    evidence_ids = tuple(
        dict.fromkeys(
            document_id
            for evidence in evidence_texts
            for document_id in paragraph_lookup.get(evidence, [])
        )
    )
    return {
        "answer": answer,
        "answer_type": answer_type,
        "evidence_ids": evidence_ids,
        "evidence_texts": evidence_texts,
    }
