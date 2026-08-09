"""可信的真实论文评测、人工审核与上下文工程工具。"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional


QUERY_VALIDITY = {"valid", "invalid", "needs_edit"}
EVIDENCE_SUFFICIENCY = {"sufficient", "partial", "insufficient"}


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
