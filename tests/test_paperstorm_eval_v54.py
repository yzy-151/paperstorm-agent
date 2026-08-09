import json
import tempfile
import unittest
from pathlib import Path


def _case(index=1, split="test", domain="rf"):
    return {
        "case_id": f"case-{index}",
        "split": split,
        "query": f"问题 {index}",
        "relevant_document_ids": [f"doc-{index}"],
        "metadata": {
            "source_document_id": f"doc-{index}",
            "source_title": f"Paper {index}",
            "domain": domain,
            "page_number": 1,
            "review_status": "needs_human_review",
        },
        "evidence": {"excerpt": f"evidence {index}", "content_sha256": f"hash-{index}"},
        "hard_negative_document_ids": ["negative-doc"],
    }


def _dataset(cases=None, dataset_hash="dataset-v1"):
    return {
        "metadata": {"dataset_sha256": dataset_hash, "annotation_status": "auto_candidate"},
        "cases": list(cases or [_case()]),
    }


def _valid_review(case_id="case-1", document_id="doc-1"):
    return {
        "case_id": case_id,
        "query_validity": "valid",
        "edited_query": "",
        "relevant_document_ids": [document_id],
        "evidence_sufficiency": "sufficient",
        "reviewer_notes": "证据能够回答问题",
    }


class AnnotationContractV54Test(unittest.TestCase):
    def test_unreviewed_dataset_is_candidate_and_not_release_ready(self):
        from knowledge_storm.paperstorm_eval_v54 import AnnotationStore

        with tempfile.TemporaryDirectory() as temp_dir:
            store = AnnotationStore(temp_dir, _dataset())
            progress = store.progress()

        self.assertEqual(progress["trust_level"], "candidate")
        self.assertEqual(progress["reviewed_count"], 0)
        self.assertFalse(progress["frozen_test_allowed"])

    def test_valid_review_requires_relevant_documents(self):
        from knowledge_storm.paperstorm_eval_v54 import validate_review

        review = _valid_review()
        review["relevant_document_ids"] = []
        with self.assertRaisesRegex(ValueError, "相关论文"):
            validate_review(review)

    def test_needs_edit_requires_an_edited_query(self):
        from knowledge_storm.paperstorm_eval_v54 import validate_review

        review = _valid_review()
        review["query_validity"] = "needs_edit"
        with self.assertRaisesRegex(ValueError, "修改后的问题"):
            validate_review(review)

    def test_save_review_merges_by_case_id_and_survives_reload(self):
        from knowledge_storm.paperstorm_eval_v54 import AnnotationStore

        with tempfile.TemporaryDirectory() as temp_dir:
            store = AnnotationStore(temp_dir, _dataset())
            first = store.save_review(_valid_review())
            changed = _valid_review()
            changed["reviewer_notes"] = "第二次审核"
            second = store.save_review(changed)
            reloaded = AnnotationStore(temp_dir, _dataset()).list_cases()[0]
            lines = Path(temp_dir, "reviews.jsonl").read_text(encoding="utf-8").splitlines()

        self.assertEqual(first["review_status"], "reviewed")
        self.assertEqual(second["reviewer_notes"], "第二次审核")
        self.assertEqual(reloaded["review"]["reviewer_notes"], "第二次审核")
        self.assertEqual(len(lines), 1)

    def test_reviewed_but_small_frozen_set_is_pilot(self):
        from knowledge_storm.paperstorm_eval_v54 import AnnotationStore

        with tempfile.TemporaryDirectory() as temp_dir:
            store = AnnotationStore(temp_dir, _dataset())
            store.save_review(_valid_review())
            progress = store.progress()

        self.assertEqual(progress["trust_level"], "pilot")
        self.assertEqual(progress["valid_reviewed_test_count"], 1)
        self.assertFalse(progress["frozen_test_allowed"])

    def test_fifty_reviewed_queries_with_ten_per_domain_are_release_ready(self):
        from knowledge_storm.paperstorm_eval_v54 import AnnotationStore

        domains = ["rf", "vlc", "mimo", "rag", "agent"]
        cases = [_case(index=i + 1, domain=domains[i // 10]) for i in range(50)]
        with tempfile.TemporaryDirectory() as temp_dir:
            store = AnnotationStore(temp_dir, _dataset(cases))
            for item in cases:
                store.save_review(
                    _valid_review(item["case_id"], item["metadata"]["source_document_id"])
                )
            progress = store.progress()

        self.assertEqual(progress["trust_level"], "release_ready")
        self.assertTrue(progress["frozen_test_allowed"])
        self.assertEqual(progress["valid_reviewed_test_count"], 50)
        self.assertEqual(progress["domain_counts"], {domain: 10 for domain in domains})

    def test_export_contains_only_valid_reviewed_cases_and_dataset_hash(self):
        from knowledge_storm.paperstorm_eval_v54 import AnnotationStore

        cases = [_case(1), _case(2)]
        with tempfile.TemporaryDirectory() as temp_dir:
            store = AnnotationStore(temp_dir, _dataset(cases))
            store.save_review(_valid_review("case-1", "doc-1"))
            invalid = _valid_review("case-2", "doc-2")
            invalid.update(
                query_validity="invalid",
                relevant_document_ids=[],
                evidence_sufficiency="insufficient",
            )
            store.save_review(invalid)
            exported = store.export_reviewed_dataset()

        self.assertEqual(exported["metadata"]["source_dataset_sha256"], "dataset-v1")
        self.assertEqual([item["case_id"] for item in exported["cases"]], ["case-1"])
        self.assertEqual(exported["cases"][0]["query"], "问题 1")

    def test_dataset_hash_change_marks_existing_reviews_stale(self):
        from knowledge_storm.paperstorm_eval_v54 import AnnotationStore

        with tempfile.TemporaryDirectory() as temp_dir:
            AnnotationStore(temp_dir, _dataset(dataset_hash="old")).save_review(
                _valid_review()
            )
            progress = AnnotationStore(
                temp_dir, _dataset(dataset_hash="new")
            ).progress()

        self.assertEqual(progress["stale_review_count"], 1)
        self.assertEqual(progress["reviewed_count"], 0)
        self.assertEqual(progress["trust_level"], "candidate")


if __name__ == "__main__":
    unittest.main()
