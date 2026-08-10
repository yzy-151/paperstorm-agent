"""可信的真实论文评测、人工审核与上下文工程工具。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import statistics
import tempfile
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional


QUERY_VALIDITY = {"valid", "invalid", "needs_edit"}
EVIDENCE_SUFFICIENCY = {"sufficient", "partial", "insufficient"}


def normalize_v54_corpus(dataset: Dict) -> List[Dict]:
    """把 v5.2 corpus 记录还原为 HybridPaperIndex 的 Chunk 契约。"""

    output = []
    for index, document in enumerate((dataset or {}).get("corpus") or [], start=1):
        metadata = dict(document.get("metadata") or {})
        source_document_id = str(
            document.get("source_document_id")
            or metadata.get("source_document_id")
            or document.get("document_id")
            or "document-{0}".format(index)
        )
        chunk_id = str(
            (document.get("chunk_ids") or [document.get("document_id")])[0]
            or "chunk-{0}".format(index)
        )
        content = str(document.get("content") or document.get("text") or "")
        output.append(
            {
                "chunk_id": chunk_id,
                "document_id": source_document_id,
                "title": str(document.get("title") or source_document_id),
                "content": content,
                "retrieval_content": str(metadata.get("context") or content),
                "metadata": metadata,
            }
        )
    return output


def enrich_context_cases(
    cases: Iterable[Dict], chunks: Iterable[Dict], max_chunks: int = 6, max_chars: int = 16000
) -> List[Dict]:
    """为每条用例附加来自同一篇真实论文的多段证据。"""

    grouped = {}
    for chunk in chunks or []:
        grouped.setdefault(str(chunk.get("document_id") or ""), []).append(chunk)
    output = []
    for case in cases or []:
        item = deepcopy(case)
        source_document_id = str(
            (item.get("metadata") or {}).get("source_document_id") or ""
        )
        sections = []
        for chunk in grouped.get(source_document_id, [])[: max(1, int(max_chunks))]:
            metadata = chunk.get("metadata") or {}
            sections.append(
                "[page {0} | {1}]\n{2}".format(
                    metadata.get("page_number") or "?",
                    chunk.get("chunk_id") or "chunk",
                    chunk.get("content") or "",
                )
            )
        if sections:
            item["context_evidence"] = "\n\n".join(sections)[: max(512, int(max_chars))]
        output.append(item)
    return output


def ranked_document_ids(chunks: Iterable[Dict]) -> List[str]:
    """将 Chunk 排名折叠为稳定的文档排名。"""

    output = []
    seen = set()
    for chunk in chunks or []:
        metadata = chunk.get("metadata") or {}
        document_id = str(
            chunk.get("document_id")
            or metadata.get("source_document_id")
            or metadata.get("document_id")
            or ""
        ).strip()
        if document_id and document_id not in seen:
            seen.add(document_id)
            output.append(document_id)
    return output


def retrieval_metrics(
    ranked_document_ids: Iterable[str],
    relevant_document_ids: Iterable[str],
    top_k: int = 5,
) -> Dict:
    """计算二元相关性的文档级排序指标。"""

    ranked = _unique_strings(ranked_document_ids)
    relevant = set(_unique_strings(relevant_document_ids))
    top_k = max(1, int(top_k))
    at_five = ranked[: min(5, top_k)]
    at_ten = ranked[: max(10, top_k)]
    relevant_count = max(1, len(relevant))
    hits_five = [document_id for document_id in at_five if document_id in relevant]
    hits_ten = [document_id for document_id in at_ten if document_id in relevant]
    first_rank = next(
        (index for index, document_id in enumerate(ranked, start=1) if document_id in relevant),
        None,
    )
    dcg = sum(
        1.0 / math.log2(index + 1)
        for index, document_id in enumerate(at_five, start=1)
        if document_id in relevant
    )
    ideal_hits = min(len(relevant), len(at_five) or min(5, top_k))
    ideal_dcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return {
        "recall_at_5": round(len(hits_five) / relevant_count, 6),
        "recall_at_10": round(len(hits_ten) / relevant_count, 6),
        "precision_at_5": round(len(hits_five) / 5.0, 6),
        "mrr": round(1.0 / first_rank, 6) if first_rank else 0.0,
        "ndcg_at_5": round(dcg / ideal_dcg, 6) if ideal_dcg else 0.0,
        "first_relevant_rank": first_rank,
    }


def summarize_retrieval_cases(
    per_case: Iterable[Dict], bootstrap_samples: int = 2000, seed: int = 17
) -> Dict:
    cases = list(per_case or [])
    metric_names = [
        "recall_at_5",
        "recall_at_10",
        "precision_at_5",
        "mrr",
        "ndcg_at_5",
    ]
    summary = {"case_count": len(cases)}
    confidence_intervals = {}
    for offset, name in enumerate(metric_names):
        values = [float(case.get(name) or 0.0) for case in cases]
        summary[name] = round(statistics.mean(values), 6) if values else 0.0
        confidence_intervals[name] = bootstrap_mean_ci(
            values, samples=bootstrap_samples, seed=seed + offset
        )
    latencies = [float(case.get("latency_ms") or 0.0) for case in cases]
    summary["p50_latency_ms"] = round(statistics.median(latencies), 4) if latencies else 0.0
    summary["p95_latency_ms"] = round(_nearest_rank(latencies, 0.95), 4)
    summary["confidence_intervals"] = confidence_intervals
    return summary


def bootstrap_mean_ci(
    values: Iterable[float], samples: int = 2000, seed: int = 17
) -> Dict:
    values = [float(value) for value in values]
    if not values:
        return {"mean": 0.0, "low": 0.0, "high": 0.0, "n": 0, "samples": samples}
    generator = random.Random(seed)
    means = []
    for _index in range(max(1, int(samples))):
        draw = [generator.choice(values) for _ in values]
        means.append(statistics.mean(draw))
    means.sort()
    return {
        "mean": round(statistics.mean(values), 6),
        "low": round(_linear_percentile(means, 0.025), 6),
        "high": round(_linear_percentile(means, 0.975), 6),
        "n": len(values),
        "samples": max(1, int(samples)),
    }


def paired_score_delta(
    baseline_scores: Iterable[float], candidate_scores: Iterable[float], tolerance=1e-12
) -> Dict:
    pairs = list(zip(baseline_scores, candidate_scores))
    deltas = [float(candidate) - float(baseline) for baseline, candidate in pairs]
    return {
        "wins": len([value for value in deltas if value > tolerance]),
        "ties": len([value for value in deltas if abs(value) <= tolerance]),
        "losses": len([value for value in deltas if value < -tolerance]),
        "mean_delta": round(statistics.mean(deltas), 6) if deltas else 0.0,
        "case_count": len(deltas),
    }


def select_dev_configuration(reports: Dict[str, Dict]) -> str:
    """只读取 dev 指标进行配置选择。"""

    if not reports:
        raise ValueError("没有可选择的检索配置")

    def key(item):
        name, report = item
        metrics = report.get("dev") or {}
        return (
            float(metrics.get("ndcg_at_5") or 0.0),
            float(metrics.get("mrr") or 0.0),
            float(metrics.get("recall_at_5") or 0.0),
            -float(metrics.get("p95_latency_ms") or float("inf")),
            name,
        )

    return max(reports.items(), key=key)[0]


def select_deployable_configuration(
    dev_reports: Dict[str, Dict],
    latency_budget_ms: float = 500.0,
    max_recall_drop: float = 0.02,
) -> Dict:
    wrapped = {name: {"dev": metrics} for name, metrics in dev_reports.items()}
    quality_best = select_dev_configuration(wrapped)
    non_reranked = {
        name: report for name, report in dev_reports.items() if name != "hybrid_rerank"
    }
    deployable_baseline = (
        select_dev_configuration(
            {name: {"dev": metrics} for name, metrics in non_reranked.items()}
        )
        if non_reranked
        else quality_best
    )
    if quality_best != "hybrid_rerank":
        return {
            "quality_best": quality_best,
            "selected": quality_best,
            "reranker_gate": {"enabled": False, "reason": "重排不是 dev 质量最优配置"},
        }
    gate = reranker_gate(
        dev_reports[deployable_baseline],
        dev_reports[quality_best],
        max_recall_drop=max_recall_drop,
        latency_budget_ms=latency_budget_ms,
    )
    return {
        "quality_best": quality_best,
        "selected": quality_best if gate["enabled"] else deployable_baseline,
        "reranker_gate": gate,
    }


def reranker_gate(
    baseline: Dict,
    reranked: Dict,
    max_recall_drop: float = 0.02,
    latency_budget_ms: float = 500.0,
) -> Dict:
    if float(reranked.get("ndcg_at_5") or 0.0) <= float(
        baseline.get("ndcg_at_5") or 0.0
    ):
        return {"enabled": False, "reason": "nDCG@5 未提升"}
    if float(baseline.get("recall_at_5") or 0.0) - float(
        reranked.get("recall_at_5") or 0.0
    ) > float(max_recall_drop):
        return {"enabled": False, "reason": "Recall@5 下降超过容忍值"}
    if float(reranked.get("p95_latency_ms") or 0.0) > float(latency_budget_ms):
        return {"enabled": False, "reason": "P95 延迟超出预算"}
    return {"enabled": True, "reason": "质量和延迟门禁通过"}


def run_retrieval_benchmark(
    dataset: Dict,
    search_fn,
    configurations: Iterable[str],
    top_k: int = 5,
    trust_level: str = "candidate",
    bootstrap_samples: int = 2000,
    reranker_latency_budget_ms: float = 500.0,
    max_recall_drop: float = 0.02,
) -> Dict:
    """在 dev 选型，并把 test 结果与证据状态分开报告。"""

    cases = list((dataset or {}).get("cases") or [])
    dev_cases = [case for case in cases if case.get("split") == "dev"]
    test_cases = [case for case in cases if case.get("split") == "test"]
    if not dev_cases:
        raise ValueError("数据集缺少 dev 用例，不能进行无泄漏选型")
    if not test_cases:
        raise ValueError("数据集缺少 test 用例，不能生成冻结结果")

    dev_reports = {}
    dev_per_case = {}
    for configuration in configurations:
        per_case = _evaluate_retrieval_cases(
            dev_cases, configuration, search_fn, top_k=top_k
        )
        dev_per_case[configuration] = per_case
        dev_reports[configuration] = summarize_retrieval_cases(
            per_case, bootstrap_samples=bootstrap_samples
        )

    selection = select_deployable_configuration(
        dev_reports,
        latency_budget_ms=reranker_latency_budget_ms,
        max_recall_drop=max_recall_drop,
    )
    selected = selection["selected"]
    baseline = "bm25" if "bm25" in dev_reports else next(iter(dev_reports))
    release_ready = trust_level == "release_ready"
    test_reports = {}
    paired_test_delta = None
    if release_ready:
        test_configurations = list(dict.fromkeys([baseline, selected]))
        test_per_case = {
            configuration: _evaluate_retrieval_cases(
                test_cases, configuration, search_fn, top_k=top_k
            )
            for configuration in test_configurations
        }
        test_reports = {
            configuration: summarize_retrieval_cases(
                per_case, bootstrap_samples=bootstrap_samples
            )
            for configuration, per_case in test_per_case.items()
        }
        baseline_scores = [
            item["ndcg_at_5"] for item in test_per_case.get(baseline, [])
        ]
        selected_scores = [
            item["ndcg_at_5"] for item in test_per_case.get(selected, [])
        ]
        paired_test_delta = paired_score_delta(baseline_scores, selected_scores)
    return {
        "project": "PaperStorm v5.4 真实论文检索评测",
        "protocol": "document_holdout_dev_selection_frozen_test",
        "selection_split": "dev",
        "final_reporting_split": "test" if release_ready else None,
        "selected_configuration": selected,
        "quality_best_configuration": selection["quality_best"],
        "reranker_gate": selection["reranker_gate"],
        "baseline_configuration": baseline,
        "evidence_status": trust_level,
        "release_claim_allowed": release_ready,
        "top_k": top_k,
        "dataset_sha256": _dataset_sha256(dataset),
        "dev": dev_reports,
        "test": test_reports,
        "paired_test_delta": paired_test_delta,
        "limitations": _retrieval_limitations(trust_level, len(test_cases)),
    }


def evaluate_context_scenarios(
    cases: Iterable[Dict], total_tokens: int = 420, recent_message_count: int = 3
) -> Dict:
    """在同一真实论文场景上比较三种上下文策略。"""

    from .paperstorm_context_v42 import (
        ContextEngine,
        ContextEngineConfig,
        ContextEventStore,
    )

    cases = list(cases or [])
    strategy_rows = {
        "full_history": [],
        "fixed_window": [],
        "structured_compaction": [],
    }
    for case in cases:
        messages, constraints, entities, source_document_id = _context_scenario(case)
        config = ContextEngineConfig(
            total_tokens=max(160, int(total_tokens)),
            output_reserve_tokens=max(48, int(total_tokens * 0.2)),
            recent_message_count=max(1, int(recent_message_count)),
            tool_inline_token_limit=24,
        )
        meter_engine = ContextEngine(config=config)
        full_tokens = meter_engine.estimate(messages)["input_tokens"]
        strategy_rows["full_history"].append(
            _context_strategy_metrics(
                messages,
                full_tokens,
                constraints,
                entities,
                source_document_id,
                restore_exact=True,
                artifact_references=False,
                repeated_retention=1.0,
            )
        )
        fixed = messages[-max(1, recent_message_count) :]
        strategy_rows["fixed_window"].append(
            _context_strategy_metrics(
                fixed,
                full_tokens,
                constraints,
                entities,
                source_document_id,
                restore_exact=False,
                artifact_references=False,
                repeated_retention=0.0,
            )
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ContextEventStore(Path(temp_dir) / "context_events.jsonl")
            for message in messages:
                store.append_message(message)
            engine = ContextEngine(config=config, store=store)
            compacted = engine.compact(
                messages, expected_constraints=constraints, force=True
            )
            restored = engine.restore(compacted["compaction_id"])
            repeated = ContextEngine(config=config).compact(
                compacted["messages"], expected_constraints=constraints, force=True
            )
        restored_exact = [item.get("content") for item in restored["messages"]] == [
            item.get("content") for item in messages
        ]
        repeated_text = "\n".join(
            [repeated.get("summary_text") or ""]
            + [str(item.get("content") or "") for item in repeated.get("messages") or []]
        ).lower()
        repeated_terms = constraints + entities
        repeated_retention = _term_retention(repeated_text, repeated_terms)
        strategy_rows["structured_compaction"].append(
            _context_strategy_metrics(
                compacted["messages"],
                full_tokens,
                constraints,
                entities,
                source_document_id,
                restore_exact=restored_exact,
                artifact_references=bool(compacted.get("artifact_refs")),
                repeated_retention=repeated_retention,
                summary_text=compacted.get("summary_text") or "",
            )
        )

    answer_probes = [case.get("answer_probe") for case in cases if case.get("answer_probe")]
    return {
        "project": "PaperStorm v5.4 真实论文上下文工程评测",
        "evidence_type": "deterministic_real_paper_probe",
        "scenario_count": len(cases),
        "strategies": {
            name: _aggregate_context_rows(rows) for name, rows in strategy_rows.items()
        },
        "answer_quality_claim_allowed": bool(cases)
        and len(answer_probes) == len(cases)
        and all(probe.get("human_reviewed") for probe in answer_probes),
        "limitations": [
            "确定性探针可以证明信息保留与 Token 权衡，不能证明回答质量提升。",
            "回答级质量需要人工审核的事实与引用标签，LLM 裁判只能作为次级证据。",
        ],
    }


def _context_scenario(case):
    metadata = case.get("metadata") or {}
    evidence = case.get("evidence") or {}
    query = str(case.get("query") or "请解释论文的核心方法。")
    source_document_id = str(metadata.get("source_document_id") or "unknown-document")
    source_title = str(metadata.get("source_title") or "Unknown paper")
    excerpt = str(
        case.get("context_evidence")
        or evidence.get("excerpt")
        or "论文证据尚未提供。"
    )
    entities = _unique_strings(metadata.get("query_terms") or [])
    constraints = ["中文", "引用"]
    messages = [
        {"id": "system", "role": "system", "content": "必须使用中文回答并保留论文引用。"},
        {"id": "goal", "role": "user", "content": query},
        {"id": "plan", "role": "assistant", "content": "决定先检索相关论文，再依据证据回答。"},
        {
            "id": "tool-call",
            "role": "assistant",
            "content": "call zotero_search",
            "tool_call_id": "zotero-v54",
        },
        {
            "id": "tool-output",
            "role": "tool",
            "name": "zotero_search",
            "tool_call_id": "zotero-v54",
            "content": "document_id={0}\ntitle={1}\nevidence={2}".format(
                source_document_id, source_title, excerpt
            ),
        },
        {
            "id": "decision",
            "role": "assistant",
            "content": "已找到来源论文，决定只使用可追溯证据。",
        },
        {"id": "correction", "role": "user", "content": "不要脱离论文，也不要省略引用。"},
        {"id": "follow-up", "role": "user", "content": "继续说明它与原问题的关系。"},
        {"id": "working", "role": "assistant", "content": "正在组织答案，下一步核对来源。"},
    ]
    return messages, constraints, entities, source_document_id


def _context_strategy_metrics(
    messages,
    full_tokens,
    constraints,
    entities,
    source_document_id,
    restore_exact,
    artifact_references,
    repeated_retention,
    summary_text="",
):
    from .paperstorm_context_v42 import estimate_tokens

    rendered = "\n".join(
        [summary_text] + [str(item.get("content") or "") for item in messages]
    ).lower()
    input_tokens = sum(estimate_tokens(str(item.get("content") or "")) + 4 for item in messages)
    source_retained = source_document_id.lower() in rendered or artifact_references
    return {
        "input_tokens": input_tokens,
        "token_reduction_rate": round(max(0, full_tokens - input_tokens) / max(1, full_tokens), 6),
        "constraint_retention_rate": _term_retention(rendered, constraints),
        "entity_retention_rate": _term_retention(rendered, entities),
        "source_retention_rate": float(source_retained),
        "restore_exact_rate": float(bool(restore_exact)),
        "artifact_reference_rate": float(bool(artifact_references)),
        "repeated_compaction_retention_rate": float(repeated_retention),
    }


def _aggregate_context_rows(rows):
    metric_names = [
        "input_tokens",
        "token_reduction_rate",
        "constraint_retention_rate",
        "entity_retention_rate",
        "source_retention_rate",
        "restore_exact_rate",
        "artifact_reference_rate",
        "repeated_compaction_retention_rate",
    ]
    output = {"scenario_count": len(rows)}
    for name in metric_names:
        values = [float(row.get(name) or 0.0) for row in rows]
        output[name] = round(statistics.mean(values), 6) if values else 0.0
    return output


def _term_retention(text, terms):
    terms = _unique_strings(terms)
    if not terms:
        return 1.0
    lowered = str(text or "").lower()
    return round(
        len([term for term in terms if term.lower() in lowered]) / len(terms), 6
    )


def _evaluate_retrieval_cases(cases, configuration, search_fn, top_k):
    output = []
    for case in cases:
        result = dict(search_fn(case, configuration, max(10, top_k)) or {})
        relevant = _case_relevant_documents(case)
        metrics = retrieval_metrics(
            result.get("ranked_document_ids") or [], relevant, top_k=top_k
        )
        metrics.update(
            case_id=str(case.get("case_id") or ""),
            latency_ms=float(result.get("latency_ms") or 0.0),
        )
        output.append(metrics)
    return output


def _case_relevant_documents(case):
    reviewed = (case.get("review") or {}).get("relevant_document_ids")
    direct = case.get("relevant_document_ids")
    source = (case.get("metadata") or {}).get("source_document_id")
    return _unique_strings(reviewed or direct or ([source] if source else []))


def _retrieval_limitations(trust_level, test_case_count):
    limitations = []
    if trust_level != "release_ready":
        limitations.append("人工审核尚未达到可发布门禁，结果只能称为小规模实验。")
    if test_case_count < 50:
        limitations.append("冻结测试问题少于 50 条，置信区间可能很宽。")
    return limitations


def sanitize_v54_report(report: Dict) -> Dict:
    """移除不应进入网页汇总或版本库的私有字段。"""

    blocked = {
        "dataset_path",
        "zotero_root",
        "pdf_path",
        "path",
        "excerpt",
        "evidence",
        "reviewer_notes",
        "per_case",
    }

    def clean(value):
        if isinstance(value, dict):
            return {
                str(key): clean(item)
                for key, item in value.items()
                if str(key).lower() not in blocked
            }
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    return clean(deepcopy(report or {}))


def validate_review(review: Dict) -> Dict:
    """校验并规范化一条人工审核记录。"""

    payload = deepcopy(dict(review or {}))
    case_id = str(payload.get("case_id") or "").strip()
    if not case_id:
        raise ValueError("审核记录缺少 case_id")

    query_validity = str(payload.get("query_validity") or "").strip()
    if query_validity not in QUERY_VALIDITY:
        raise ValueError("问题有效性必须是 valid、invalid 或 needs_edit")

    edited_query = str(payload.get("edited_query") or "").strip()
    if query_validity == "needs_edit" and not edited_query:
        raise ValueError("needs_edit 必须填写修改后的问题")

    document_ids = _unique_strings(payload.get("relevant_document_ids") or [])
    if query_validity in {"valid", "needs_edit"} and not document_ids:
        raise ValueError("有效问题必须至少选择一篇相关论文")

    evidence = str(payload.get("evidence_sufficiency") or "").strip()
    if evidence not in EVIDENCE_SUFFICIENCY:
        raise ValueError("证据充分性必须是 sufficient、partial 或 insufficient")

    payload.update(
        case_id=case_id,
        query_validity=query_validity,
        edited_query=edited_query,
        relevant_document_ids=document_ids,
        evidence_sufficiency=evidence,
        reviewer_notes=str(payload.get("reviewer_notes") or "").strip(),
        review_status="reviewed",
        reviewed_at=str(payload.get("reviewed_at") or _utc_now()),
    )
    return payload


class AnnotationStore:
    """将私有人工审核保存在 gitignored 的 JSONL 文件中。"""

    def __init__(
        self,
        root_dir,
        dataset: Dict,
        min_reviewed_frozen: int = 50,
        min_cases_per_domain: int = 10,
    ):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = deepcopy(dict(dataset or {}))
        self.cases = list(self.dataset.get("cases") or [])
        self.case_by_id = {
            str(case.get("case_id") or ""): case
            for case in self.cases
            if case.get("case_id")
        }
        self.dataset_sha256 = _dataset_sha256(self.dataset)
        self.min_reviewed_frozen = int(min_reviewed_frozen)
        self.min_cases_per_domain = int(min_cases_per_domain)
        self.review_path = self.root_dir / "reviews.jsonl"

    def list_cases(self) -> List[Dict]:
        reviews = self._current_reviews()
        output = []
        for case in self.cases:
            item = deepcopy(case)
            review = reviews.get(str(case.get("case_id") or ""))
            if review:
                item["review"] = deepcopy(review)
            output.append(item)
        return output

    def save_review(self, review: Dict) -> Dict:
        normalized = validate_review(review)
        case_id = normalized["case_id"]
        if case_id not in self.case_by_id:
            raise ValueError("找不到待审核用例：{0}".format(case_id))
        normalized["source_dataset_sha256"] = self.dataset_sha256
        records = self._all_reviews()
        records[case_id] = normalized
        self._write_reviews(records.values())
        return deepcopy(normalized)

    def progress(self) -> Dict:
        all_reviews = self._all_reviews()
        reviews = self._current_reviews()
        valid_test_reviews = []
        for case_id, review in reviews.items():
            case = self.case_by_id.get(case_id) or {}
            if (
                case.get("split") == "test"
                and review.get("query_validity") in {"valid", "needs_edit"}
            ):
                valid_test_reviews.append((case, review))

        domain_counts = Counter(
            str((case.get("metadata") or {}).get("domain") or "未分类")
            for case, _review in valid_test_reviews
        )
        enough_total = len(valid_test_reviews) >= self.min_reviewed_frozen
        enough_per_domain = bool(domain_counts) and all(
            count >= self.min_cases_per_domain for count in domain_counts.values()
        )
        release_ready = enough_total and enough_per_domain
        if not reviews:
            trust_level = "candidate"
        elif release_ready:
            trust_level = "release_ready"
        else:
            trust_level = "pilot"

        stale_count = len(
            [
                review
                for review in all_reviews.values()
                if review.get("source_dataset_sha256") != self.dataset_sha256
            ]
        )
        return {
            "trust_level": trust_level,
            "dataset_sha256": self.dataset_sha256,
            "candidate_count": len(self.cases),
            "reviewed_count": len(reviews),
            "valid_reviewed_test_count": len(valid_test_reviews),
            "stale_review_count": stale_count,
            "domain_counts": dict(sorted(domain_counts.items())),
            "minimum_reviewed_frozen": self.min_reviewed_frozen,
            "minimum_cases_per_domain": self.min_cases_per_domain,
            "frozen_test_allowed": release_ready,
        }

    def export_reviewed_dataset(self) -> Dict:
        reviews = self._current_reviews()
        cases = []
        for case in self.cases:
            case_id = str(case.get("case_id") or "")
            review = reviews.get(case_id)
            if not review or review.get("query_validity") not in {"valid", "needs_edit"}:
                continue
            item = deepcopy(case)
            if review.get("edited_query"):
                item["query"] = review["edited_query"]
            item["relevant_document_ids"] = list(review["relevant_document_ids"])
            item["review"] = deepcopy(review)
            cases.append(item)
        return {
            "metadata": {
                "schema_version": "paperstorm-reviewed-v5.4",
                "source_dataset_sha256": self.dataset_sha256,
                "annotation_status": "human_reviewed",
                "exported_at": _utc_now(),
                "case_count": len(cases),
            },
            "cases": cases,
        }

    def _current_reviews(self) -> Dict[str, Dict]:
        return {
            case_id: review
            for case_id, review in self._all_reviews().items()
            if review.get("source_dataset_sha256") == self.dataset_sha256
        }

    def _all_reviews(self) -> Dict[str, Dict]:
        if not self.review_path.exists():
            return {}
        records = {}
        for line in self.review_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            case_id = str(payload.get("case_id") or "")
            if case_id:
                records[case_id] = payload
        return records

    def _write_reviews(self, reviews: Iterable[Dict]) -> None:
        temporary = self.review_path.with_suffix(".jsonl.tmp")
        ordered = sorted(reviews, key=lambda item: str(item.get("case_id") or ""))
        content = "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
            for item in ordered
        )
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, self.review_path)


def _dataset_sha256(dataset: Dict) -> str:
    metadata = dataset.get("metadata") or {}
    declared = metadata.get("dataset_sha256") or metadata.get("sha256")
    if declared:
        return str(declared)
    payload = json.dumps(dataset, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _unique_strings(values: Iterable) -> List[str]:
    output = []
    seen = set()
    for value in values:
        rendered = str(value or "").strip()
        if rendered and rendered not in seen:
            seen.add(rendered)
            output.append(rendered)
    return output


def _nearest_rank(values: Iterable[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _linear_percentile(values: Iterable[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
