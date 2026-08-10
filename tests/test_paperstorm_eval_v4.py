import json
import tempfile
import unittest
from pathlib import Path


class PaperStormEvalV4Test(unittest.TestCase):
    def test_seed_dataset_has_one_hundred_auditable_cases(self):
        from knowledge_storm.paperstorm_eval_v4 import build_seed_dataset, validate_dataset

        dataset = build_seed_dataset()
        validation = validate_dataset(dataset)

        self.assertEqual(len(dataset["cases"]), 100)
        self.assertEqual(len({case["case_id"] for case in dataset["cases"]}), 100)
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["error_count"], 0)
        self.assertEqual(dataset["metadata"]["provenance"], "synthetic_seed")
        self.assertTrue(dataset["metadata"]["domain_review_required"])

    def test_retrieval_metrics_use_relevant_chunk_ranks(self):
        from knowledge_storm.paperstorm_eval_v4 import evaluate_observation

        result = evaluate_observation(
            case={
                "case_id": "retrieval-1",
                "query": "what is passive intermodulation",
                "relevant_chunk_ids": ["pim-definition"],
                "expected_behavior": "answer",
                "required_answer_terms": ["passive intermodulation"],
                "allowed_citation_ids": ["pim-definition"],
            },
            observation={
                "candidates": [
                    {"chunk_id": "noise", "content": "DRAM accelerator"},
                    {"chunk_id": "pim-definition", "content": "Passive intermodulation is nonlinear distortion."},
                ],
                "selected": [
                    {"chunk_id": "pim-definition", "content": "Passive intermodulation is nonlinear distortion."},
                    {"chunk_id": "noise", "content": "DRAM accelerator"},
                ],
                "prompt_context": "Passive intermodulation is nonlinear distortion.",
                "answer": "Passive intermodulation is nonlinear distortion.",
                "citations": ["pim-definition"],
                "latency_ms": 12.5,
            },
            top_k=2,
        )

        self.assertEqual(result["retrieval"]["recall_at_k"], 1.0)
        self.assertEqual(result["retrieval"]["precision_at_k"], 0.5)
        self.assertEqual(result["retrieval"]["mrr"], 1.0)
        self.assertGreater(result["retrieval"]["ndcg_at_k"], 0.9)
        self.assertEqual(result["answer"]["citation_precision"], 1.0)
        self.assertEqual(result["failure_stage"], "passed")

    def test_failure_attribution_distinguishes_rerank_and_compression(self):
        from knowledge_storm.paperstorm_eval_v4 import evaluate_observation

        case = {
            "case_id": "failure-1",
            "query": "PIM suppression",
            "relevant_chunk_ids": ["relevant"],
            "expected_behavior": "answer",
            "required_answer_terms": ["cancellation"],
            "allowed_citation_ids": ["relevant"],
        }
        rerank_miss = evaluate_observation(
            case,
            {
                "candidates": [{"chunk_id": "relevant", "content": "Neural cancellation."}],
                "selected": [{"chunk_id": "noise", "content": "Unrelated memory system."}],
                "prompt_context": "Unrelated memory system.",
                "answer": "I do not know.",
                "citations": [],
            },
        )
        compression_loss = evaluate_observation(
            case,
            {
                "candidates": [{"chunk_id": "relevant", "content": "Neural cancellation."}],
                "selected": [{"chunk_id": "relevant", "content": "Neural cancellation."}],
                "prompt_context": "Neural method.",
                "answer": "Neural method.",
                "citations": ["relevant"],
            },
        )

        self.assertEqual(rerank_miss["failure_stage"], "rerank_miss")
        self.assertEqual(compression_loss["failure_stage"], "compression_loss")

    def test_abstention_and_citation_metrics_are_separate_from_retrieval(self):
        from knowledge_storm.paperstorm_eval_v4 import evaluate_observation

        result = evaluate_observation(
            {
                "case_id": "no-answer",
                "query": "unsupported claim",
                "relevant_chunk_ids": [],
                "expected_behavior": "abstain",
                "required_answer_terms": [],
                "allowed_citation_ids": [],
            },
            {
                "candidates": [],
                "selected": [],
                "prompt_context": "",
                "answer": "证据不足，无法可靠回答。",
                "citations": [],
                "abstained": True,
            },
        )

        self.assertEqual(result["answer"]["abstention_correct"], 1.0)
        self.assertEqual(result["answer"]["citation_precision"], 1.0)
        self.assertEqual(result["failure_stage"], "passed")

    def test_harness_writes_reports_and_bad_case_records(self):
        from knowledge_storm.paperstorm_eval_v4 import run_evaluation

        dataset = {
            "dataset_version": "test",
            "metadata": {"provenance": "unit_test", "domain_review_required": False},
            "cases": [
                {
                    "case_id": "ok",
                    "query": "PIM definition",
                    "relevant_chunk_ids": ["pim"],
                    "expected_behavior": "answer",
                    "required_answer_terms": ["passive intermodulation"],
                    "allowed_citation_ids": ["pim"],
                },
                {
                    "case_id": "miss",
                    "query": "PIM cause",
                    "relevant_chunk_ids": ["cause"],
                    "expected_behavior": "answer",
                    "required_answer_terms": ["nonlinear contact"],
                    "allowed_citation_ids": ["cause"],
                },
            ],
        }

        def runner(case):
            if case["case_id"] == "ok":
                chunk = {"chunk_id": "pim", "content": "Passive intermodulation is RF distortion."}
                return {
                    "candidates": [chunk],
                    "selected": [chunk],
                    "prompt_context": chunk["content"],
                    "answer": chunk["content"],
                    "citations": ["pim"],
                    "latency_ms": 5,
                }
            return {
                "candidates": [],
                "selected": [],
                "prompt_context": "",
                "answer": "No evidence.",
                "citations": [],
                "latency_ms": 8,
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            report = run_evaluation(dataset, runner, output_dir=temp_dir, top_k=5)
            output_dir = Path(temp_dir)
            saved = json.loads((output_dir / "rag_eval_v4_report.json").read_text(encoding="utf-8"))

            self.assertTrue((output_dir / "rag_eval_v4_report.md").exists())
            self.assertTrue((output_dir / "rag_eval_v4_bad_cases.jsonl").exists())

        self.assertEqual(saved["metrics"]["total_cases"], 2)
        self.assertEqual(saved["metrics"]["retrieval_recall_at_k"], 0.5)
        self.assertEqual(report["metrics"]["failed_cases"], 1)
        self.assertEqual(report["bad_cases"][0]["failure_stage"], "retrieval_miss")
        self.assertIn("uncategorized", report["category_slices"])


if __name__ == "__main__":
    unittest.main()
