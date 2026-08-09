"""可信的真实论文评测、人工审核与上下文工程工具。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import statistics
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional


QUERY_VALIDITY = {"valid", "invalid", "needs_edit"}
EVIDENCE_SUFFICIENCY = {"sufficient", "partial", "insufficient"}


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

    selectable = {
        name: {"dev": metrics} for name, metrics in dev_reports.items()
    }
    selected = select_dev_configuration(selectable)
    baseline = "bm25" if "bm25" in dev_reports else next(iter(dev_reports))
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
    return {
        "project": "PaperStorm v5.4 真实论文检索评测",
        "protocol": "document_holdout_dev_selection_frozen_test",
        "selection_split": "dev",
        "final_reporting_split": "test",
        "selected_configuration": selected,
        "baseline_configuration": baseline,
        "evidence_status": trust_level,
        "release_claim_allowed": trust_level == "release_ready",
        "top_k": top_k,
        "dataset_sha256": _dataset_sha256(dataset),
        "dev": dev_reports,
        "test": test_reports,
        "paired_test_delta": paired_score_delta(baseline_scores, selected_scores),
        "limitations": _retrieval_limitations(trust_level, len(test_cases)),
    }


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
