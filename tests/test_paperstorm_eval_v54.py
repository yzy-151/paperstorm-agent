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


class RetrievalMetricsV54Test(unittest.TestCase):
    def test_document_metrics_report_recall_precision_mrr_and_ndcg(self):
        from knowledge_storm.paperstorm_eval_v54 import retrieval_metrics

        metrics = retrieval_metrics(
            ranked_document_ids=["noise", "doc-a", "doc-b", "extra", "other"],
            relevant_document_ids=["doc-a", "doc-b"],
            top_k=5,
        )

        self.assertEqual(metrics["recall_at_5"], 1.0)
        self.assertEqual(metrics["recall_at_10"], 1.0)
        self.assertEqual(metrics["precision_at_5"], 0.4)
        self.assertEqual(metrics["mrr"], 0.5)
        self.assertGreater(metrics["ndcg_at_5"], 0.6)
        self.assertLess(metrics["ndcg_at_5"], 1.0)

    def test_duplicate_chunks_from_one_document_do_not_inflate_metrics(self):
        from knowledge_storm.paperstorm_eval_v54 import ranked_document_ids

        chunks = [
            {"document_id": "doc-a", "chunk_id": "a-1"},
            {"document_id": "doc-a", "chunk_id": "a-2"},
            {"document_id": "doc-b", "chunk_id": "b-1"},
        ]

        self.assertEqual(ranked_document_ids(chunks), ["doc-a", "doc-b"])

    def test_selection_uses_dev_only_with_latency_as_last_tiebreaker(self):
        from knowledge_storm.paperstorm_eval_v54 import select_dev_configuration

        reports = {
            "dense": {
                "dev": {"ndcg_at_5": 0.4, "mrr": 0.5, "recall_at_5": 0.6, "p95_latency_ms": 10},
                "test": {"ndcg_at_5": 0.99},
            },
            "hybrid": {
                "dev": {"ndcg_at_5": 0.6, "mrr": 0.4, "recall_at_5": 0.5, "p95_latency_ms": 100},
                "test": {"ndcg_at_5": 0.01},
            },
        }

        self.assertEqual(select_dev_configuration(reports), "hybrid")

    def test_paired_delta_counts_wins_ties_and_losses(self):
        from knowledge_storm.paperstorm_eval_v54 import paired_score_delta

        delta = paired_score_delta(
            baseline_scores=[0.2, 0.0, 0.5],
            candidate_scores=[0.4, 0.0, 0.3],
        )

        self.assertEqual(delta["wins"], 1)
        self.assertEqual(delta["ties"], 1)
        self.assertEqual(delta["losses"], 1)
        self.assertEqual(delta["mean_delta"], 0.0)

    def test_metric_summary_has_denominators_latency_and_confidence_intervals(self):
        from knowledge_storm.paperstorm_eval_v54 import summarize_retrieval_cases

        per_case = [
            {"recall_at_5": 1.0, "recall_at_10": 1.0, "precision_at_5": 0.2, "mrr": 1.0, "ndcg_at_5": 1.0, "latency_ms": 10.0},
            {"recall_at_5": 0.0, "recall_at_10": 1.0, "precision_at_5": 0.0, "mrr": 0.0, "ndcg_at_5": 0.0, "latency_ms": 30.0},
        ]

        summary = summarize_retrieval_cases(per_case, bootstrap_samples=200, seed=7)

        self.assertEqual(summary["case_count"], 2)
        self.assertEqual(summary["recall_at_5"], 0.5)
        self.assertEqual(summary["recall_at_10"], 1.0)
        self.assertEqual(summary["p50_latency_ms"], 20.0)
        self.assertEqual(summary["p95_latency_ms"], 30.0)
        self.assertEqual(summary["confidence_intervals"]["recall_at_5"]["n"], 2)

    def test_reranker_gate_rejects_quality_or_latency_regression(self):
        from knowledge_storm.paperstorm_eval_v54 import reranker_gate

        baseline = {"ndcg_at_5": 0.5, "recall_at_5": 0.6, "p95_latency_ms": 100}
        self.assertTrue(
            reranker_gate(
                baseline,
                {"ndcg_at_5": 0.6, "recall_at_5": 0.59, "p95_latency_ms": 180},
                max_recall_drop=0.02,
                latency_budget_ms=200,
            )["enabled"]
        )
        self.assertEqual(
            reranker_gate(
                baseline,
                {"ndcg_at_5": 0.49, "recall_at_5": 0.6, "p95_latency_ms": 180},
                max_recall_drop=0.02,
                latency_budget_ms=200,
            )["reason"],
            "nDCG@5 未提升",
        )
        self.assertEqual(
            reranker_gate(
                baseline,
                {"ndcg_at_5": 0.6, "recall_at_5": 0.59, "p95_latency_ms": 250},
                max_recall_drop=0.02,
                latency_budget_ms=200,
            )["reason"],
            "P95 延迟超出预算",
        )

    def test_benchmark_selects_on_dev_and_reports_test_as_pilot(self):
        from knowledge_storm.paperstorm_eval_v54 import run_retrieval_benchmark

        dataset = _dataset(
            [
                _case(1, split="dev"),
                _case(2, split="dev"),
                _case(3, split="test"),
            ]
        )
        rankings = {
            ("case-1", "bm25"): ["noise", "doc-1"],
            ("case-2", "bm25"): ["noise", "doc-2"],
            ("case-3", "bm25"): ["doc-3"],
            ("case-1", "dense"): ["doc-1"],
            ("case-2", "dense"): ["doc-2"],
            ("case-3", "dense"): ["noise"],
        }

        def search(case, mode, _retrieve_k):
            return {
                "ranked_document_ids": rankings[(case["case_id"], mode)],
                "latency_ms": 10.0 if mode == "bm25" else 20.0,
            }

        report = run_retrieval_benchmark(
            dataset,
            search_fn=search,
            configurations=["bm25", "dense"],
            top_k=5,
            trust_level="pilot",
            bootstrap_samples=100,
        )

        self.assertEqual(report["selected_configuration"], "dense")
        self.assertEqual(report["selection_split"], "dev")
        self.assertEqual(report["final_reporting_split"], "test")
        self.assertEqual(report["evidence_status"], "pilot")
        self.assertFalse(report["release_claim_allowed"])
        self.assertEqual(report["test"]["dense"]["recall_at_5"], 0.0)
        self.assertEqual(report["test"]["bm25"]["recall_at_5"], 1.0)
        self.assertEqual(report["paired_test_delta"]["losses"], 1)
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
