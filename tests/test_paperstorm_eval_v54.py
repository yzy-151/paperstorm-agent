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

    def test_deployable_selection_rejects_slow_reranker_despite_quality_gain(self):
        from knowledge_storm.paperstorm_eval_v54 import select_deployable_configuration

        reports = {
            "dense": {"ndcg_at_5": 0.35, "mrr": 0.31, "recall_at_5": 0.44, "p95_latency_ms": 243},
            "hybrid": {"ndcg_at_5": 0.28, "mrr": 0.25, "recall_at_5": 0.38, "p95_latency_ms": 239},
            "hybrid_rerank": {"ndcg_at_5": 0.45, "mrr": 0.44, "recall_at_5": 0.5, "p95_latency_ms": 3272},
        }

        selection = select_deployable_configuration(
            reports, latency_budget_ms=500, max_recall_drop=0.02
        )

        self.assertEqual(selection["quality_best"], "hybrid_rerank")
        self.assertEqual(selection["selected"], "dense")
        self.assertFalse(selection["reranker_gate"]["enabled"])
        self.assertEqual(selection["reranker_gate"]["reason"], "P95 延迟超出预算")

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

    def test_benchmark_selects_on_dev_without_reading_frozen_test_as_pilot(self):
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
        self.assertIsNone(report["final_reporting_split"])
        self.assertEqual(report["evidence_status"], "pilot")
        self.assertFalse(report["release_claim_allowed"])
        self.assertEqual(report["test"], {})
        self.assertIsNone(report["paired_test_delta"])

    def test_release_ready_benchmark_reports_frozen_test(self):
        from knowledge_storm.paperstorm_eval_v54 import run_retrieval_benchmark

        dataset = _dataset([_case(1, split="dev"), _case(2, split="test")])

        def search(case, mode, _retrieve_k):
            ranked = [case["metadata"]["source_document_id"]]
            return {"ranked_document_ids": ranked, "latency_ms": 10.0}

        report = run_retrieval_benchmark(
            dataset,
            search_fn=search,
            configurations=["bm25", "dense"],
            trust_level="release_ready",
            bootstrap_samples=100,
        )

        self.assertEqual(report["final_reporting_split"], "test")
        self.assertEqual(report["test"]["dense"]["recall_at_5"], 1.0)
        self.assertTrue(report["release_claim_allowed"])
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


class ContextEvaluationV54Test(unittest.TestCase):
    def test_structured_summary_never_reinlines_artifactized_tool_output(self):
        from knowledge_storm.paperstorm_context_v42 import ContextEngine, ContextEngineConfig

        long_output = "must be completed without error passive intermodulation evidence " * 250
        messages = [
            {"id": "system", "role": "system", "content": "必须中文回答。"},
            {"id": "goal", "role": "user", "content": "解释无源互调。"},
            {"id": "call", "role": "assistant", "content": "call search", "tool_call_id": "c1"},
            {"id": "tool", "role": "tool", "content": long_output, "tool_call_id": "c1"},
            {"id": "decision", "role": "assistant", "content": "决定依据论文回答。"},
            {"id": "follow", "role": "user", "content": "继续。"},
            {"id": "working", "role": "assistant", "content": "正在组织答案。"},
        ]
        engine = ContextEngine(
            ContextEngineConfig(
                total_tokens=360,
                output_reserve_tokens=72,
                recent_message_count=3,
                tool_inline_token_limit=24,
            )
        )

        result = engine.compact(messages, expected_constraints=["中文"], force=True)

        self.assertEqual(len(result["artifact_refs"]), 1)
        self.assertNotIn(long_output[:200], result["summary_text"])
        self.assertLess(result["after_tokens"], result["before_tokens"] * 0.5)

    def test_v52_corpus_is_normalized_to_source_document_ids(self):
        from knowledge_storm.paperstorm_eval_v54 import normalize_v54_corpus

        dataset = {
            "corpus": [
                {
                    "document_id": "doc-1::p1::c1",
                    "source_document_id": "doc-1",
                    "chunk_ids": ["doc-1::p1::c1"],
                    "title": "Paper 1",
                    "text": "paper evidence",
                    "metadata": {"context": "Page 1 paper evidence"},
                }
            ]
        }

        chunks = normalize_v54_corpus(dataset)

        self.assertEqual(chunks[0]["chunk_id"], "doc-1::p1::c1")
        self.assertEqual(chunks[0]["document_id"], "doc-1")
        self.assertEqual(chunks[0]["content"], "paper evidence")
        self.assertEqual(chunks[0]["retrieval_content"], "Page 1 paper evidence")

    def test_context_cases_use_multiple_real_chunks_from_same_paper(self):
        from knowledge_storm.paperstorm_eval_v54 import (
            enrich_context_cases,
            evaluate_context_scenarios,
        )

        case = _case(1)
        chunks = [
            {
                "chunk_id": f"doc-1-c{index}",
                "document_id": "doc-1",
                "content": (f"real evidence section {index} " * 80),
                "retrieval_content": (f"real evidence section {index} " * 80),
                "metadata": {"page_number": index},
            }
            for index in range(1, 5)
        ]

        enriched = enrich_context_cases([case], chunks)
        report = evaluate_context_scenarios(enriched, total_tokens=360)

        self.assertIn("real evidence section 4", enriched[0]["context_evidence"])
        self.assertGreater(
            report["strategies"]["structured_compaction"]["token_reduction_rate"],
            0.4,
        )
        self.assertEqual(
            report["strategies"]["structured_compaction"]["source_retention_rate"],
            1.0,
        )

    def test_context_report_compares_three_strategies_with_denominators(self):
        from knowledge_storm.paperstorm_eval_v54 import evaluate_context_scenarios

        case = _case(1, split="test", domain="rf")
        case["query"] = "无源互调为什么需要神经网络抑制？"
        case["metadata"]["query_terms"] = ["无源互调", "神经网络", "抑制"]
        case["evidence"]["excerpt"] = (
            "passive intermodulation cancellation uses a neural network to model nonlinear distortion "
            * 30
        )
        case["review"] = _valid_review("case-1", "doc-1")

        report = evaluate_context_scenarios([case], total_tokens=360)

        self.assertEqual(
            set(report["strategies"]),
            {"full_history", "fixed_window", "structured_compaction"},
        )
        for metrics in report["strategies"].values():
            self.assertEqual(metrics["scenario_count"], 1)
            self.assertIn("input_tokens", metrics)
            self.assertIn("constraint_retention_rate", metrics)
            self.assertIn("source_retention_rate", metrics)
        structured = report["strategies"]["structured_compaction"]
        self.assertGreater(structured["token_reduction_rate"], 0.0)
        self.assertEqual(structured["restore_exact_rate"], 1.0)
        self.assertEqual(structured["artifact_reference_rate"], 1.0)
        self.assertEqual(report["evidence_type"], "deterministic_real_paper_probe")

    def test_context_report_does_not_claim_answer_quality_without_human_probes(self):
        from knowledge_storm.paperstorm_eval_v54 import evaluate_context_scenarios

        report = evaluate_context_scenarios([_case()])

        self.assertFalse(report["answer_quality_claim_allowed"])
        self.assertIn("不能证明回答质量提升", " ".join(report["limitations"]))


class EvaluationApiV54Test(unittest.TestCase):
    def test_api_imports_dataset_saves_review_and_runs_context_pilot(self):
        from fastapi.testclient import TestClient

        from examples.storm_examples.paperstorm_service_api import create_app

        cases = [_case(1, split="dev"), _case(2, split="test")]
        dataset = _dataset(cases)
        dataset["corpus"] = [
            {
                "chunk_id": "doc-1-c1",
                "document_id": "doc-1",
                "title": "Paper 1",
                "content": "passive intermodulation neural cancellation",
                "retrieval_content": "passive intermodulation neural cancellation",
                "metadata": {"page_number": 1},
            },
            {
                "chunk_id": "doc-2-c1",
                "document_id": "doc-2",
                "title": "Paper 2",
                "content": "mimo channel estimation",
                "retrieval_content": "mimo channel estimation",
                "metadata": {"page_number": 1},
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_path = root / "candidate.json"
            dataset_path.write_text(
                json.dumps(dataset, ensure_ascii=False), encoding="utf-8"
            )
            client = TestClient(create_app(service_root=root / "service"))

            empty = client.get("/evaluations/v54/status")
            imported = client.post(
                "/evaluations/v54/dataset", json={"dataset_path": str(dataset_path)}
            )
            annotations = client.get("/evaluations/v54/annotations")
            saved = client.put(
                "/evaluations/v54/annotations/case-2", json=_valid_review("case-2", "doc-2")
            )
            context = client.post("/evaluations/v54/context")
            latest = client.get("/evaluations/v54/latest")

        self.assertEqual(empty.status_code, 200)
        self.assertFalse(empty.json()["configured"])
        self.assertEqual(imported.status_code, 200)
        self.assertEqual(imported.json()["candidate_count"], 2)
        self.assertEqual(len(annotations.json()["cases"]), 2)
        self.assertEqual(saved.json()["review_status"], "reviewed")
        self.assertEqual(context.status_code, 200)
        self.assertEqual(context.json()["evidence_type"], "deterministic_real_paper_probe")
        self.assertEqual(latest.status_code, 200)
        self.assertNotIn(str(dataset_path), json.dumps(latest.json(), ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
