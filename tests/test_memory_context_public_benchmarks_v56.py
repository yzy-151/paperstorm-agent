import tempfile
import unittest
from pathlib import Path


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "public_benchmarks"


class MemoryContextPublicBenchmarkV56Tests(unittest.TestCase):
    def test_qasper_official_json_and_context_budget_evaluation(self):
        import json

        from knowledge_storm.evaluation.public_benchmarks.qasper import (
            evaluate_qasper_context_budget,
            load_qasper_official_json,
        )

        payload = {
            "paper-1": {
                "title": "A paper",
                "abstract": "",
                "full_text": [
                    {"section_name": "Intro", "paragraphs": ["filler " * 80]},
                    {"section_name": "Results", "paragraphs": ["gold evidence"]},
                ],
                "qas": [
                    {
                        "question": "What is the result?",
                        "question_id": "q1",
                        "answers": [
                            {
                                "answer": {
                                    "unanswerable": False,
                                    "extractive_spans": ["gold"],
                                    "yes_no": None,
                                    "free_form_answer": "",
                                    "evidence": ["gold evidence"],
                                }
                            }
                        ],
                    }
                ],
            }
        }
        rankings = [
            {
                "case_id": "q1",
                "mode": "hybrid_rerank",
                "ranked_document_ids": [
                    "paper-1::section-1::paragraph-0",
                ],
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "qasper.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            dataset = load_qasper_official_json(path, split="test")
            report, rows = evaluate_qasper_context_budget(
                dataset,
                rankings,
                mode="hybrid_rerank",
                model_context_tokens=128,
                output_reserve_tokens=32,
                evidence_budget_ratio=0.6,
            )

        self.assertEqual(len(dataset.cases), 1)
        self.assertEqual(report["case_count"], 1)
        self.assertEqual(report["retrieved_evidence_retention"], 1.0)
        self.assertEqual(report["gold_evidence_recall_after_context"], 1.0)
        self.assertLess(report["mean_context_to_full_document_ratio"], 1.0)
        self.assertLessEqual(rows[0]["context_tokens"], rows[0]["input_limit_tokens"])

    def test_longmemeval_adapter_preserves_sessions_categories_and_evidence(self):
        from knowledge_storm.evaluation.public_benchmarks.longmemeval import load_longmemeval

        dataset = load_longmemeval(FIXTURE_ROOT / "longmemeval_tiny.json", split="test")

        self.assertEqual(dataset.name, "longmemeval")
        self.assertEqual(len(dataset.cases), 2)
        self.assertEqual(len(dataset.documents), 3)
        case = dataset.cases[0]
        self.assertEqual(case.metadata["question_type"], "knowledge_update")
        self.assertEqual(case.answers, ("中文",))
        self.assertEqual(case.evidence_ids, ("s2",))
        self.assertEqual(dataset.document_map()["q-update:s2"].metadata["timestamp"], "2026-02-01")

    def test_longmemeval_adapter_can_stop_streaming_at_limit(self):
        from knowledge_storm.evaluation.public_benchmarks.longmemeval import load_longmemeval

        dataset = load_longmemeval(FIXTURE_ROOT / "longmemeval_tiny.json", limit=1)

        self.assertEqual(len(dataset.cases), 1)
        self.assertTrue(dataset.documents)
        self.assertEqual({item.metadata["question_id"] for item in dataset.documents}, {dataset.cases[0].case_id})

    def test_longmemeval_adapter_accepts_official_parallel_session_arrays(self):
        import json

        from knowledge_storm.evaluation.public_benchmarks.longmemeval import load_longmemeval

        payload = [{
            "question_id": "official-1",
            "question_type": "single_hop",
            "question": "偏好是什么？",
            "answer": "中文",
            "answer_session_ids": ["answer-1"],
            "haystack_session_ids": ["filler-1", "answer-1"],
            "haystack_dates": ["2026/01/01", "2026/02/01"],
            "haystack_sessions": [
                [{"role": "user", "content": "无关信息"}],
                [{"role": "user", "content": "我偏好中文"}],
            ],
        }]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "official.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            dataset = load_longmemeval(path)

        self.assertEqual(dataset.cases[0].relevant_document_ids, ("official-1:answer-1",))
        self.assertEqual(dataset.documents[1].metadata["timestamp"], "2026/02/01")

    def test_longmemeval_duplicate_session_ids_get_unique_occurrence_ids(self):
        import json

        from knowledge_storm.evaluation.public_benchmarks.longmemeval import load_longmemeval

        payload = [{
            "question_id": "duplicate-1",
            "question_type": "single_hop",
            "question": "答案在哪？",
            "answer": "第二条",
            "answer_session_ids": ["same"],
            "haystack_session_ids": ["same", "same"],
            "haystack_dates": ["2026/01/01", "2026/01/02"],
            "haystack_sessions": [
                [{"role": "user", "content": "第一条"}],
                [{"role": "user", "content": "第二条"}],
            ],
        }]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "duplicate.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            dataset = load_longmemeval(path)

        self.assertEqual(len({item.document_id for item in dataset.documents}), 2)
        self.assertEqual(len(dataset.cases[0].relevant_document_ids), 2)

    def test_longbench_adapter_preserves_official_choices_and_context(self):
        from knowledge_storm.evaluation.public_benchmarks.longbench_context import load_longbench_v2

        dataset = load_longbench_v2(FIXTURE_ROOT / "longbench_tiny.jsonl", selected_subdomains={"Qasper"})

        self.assertEqual(dataset.name, "longbench-v2-selected")
        self.assertEqual(len(dataset.cases), 1)
        self.assertEqual(dataset.cases[0].answers, ("A",))
        self.assertEqual(dataset.cases[0].metadata["choices"]["A"], "Passive Intermodulation")
        self.assertIn("Passive Intermodulation", dataset.documents[0].text)

    def test_memory_metrics_report_categories_retrieval_and_bootstrap_ci(self):
        from knowledge_storm.evaluation.public_benchmarks.longmemeval import score_longmemeval

        rows = [
            {"question_type": "knowledge_update", "correct": 1, "retrieved_ids": ["s2"], "evidence_ids": ["s2"]},
            {"question_type": "knowledge_update", "correct": 0, "retrieved_ids": ["s8"], "evidence_ids": ["s9"]},
            {"question_type": "abstention", "correct": 1, "retrieved_ids": [], "evidence_ids": []},
        ]
        report = score_longmemeval(rows, top_k=1, bootstrap_samples=200, seed=7)

        self.assertAlmostEqual(report["accuracy"], 2 / 3, places=6)
        self.assertEqual(report["categories"]["knowledge_update"]["accuracy"], 0.5)
        self.assertEqual(report["retrieval_recall_at_1"], 0.5)
        self.assertEqual(len(report["accuracy_ci95"]), 2)

    def test_context_comparison_reports_paired_quality_and_token_delta(self):
        from knowledge_storm.evaluation.public_benchmarks.longbench_context import score_context_modes

        predictions = {
            "full": [{"case_id": "1", "prediction": "A", "answer": "A", "input_tokens": 100}],
            "v56": [{"case_id": "1", "prediction": "A", "answer": "A", "input_tokens": 55}],
        }
        report = score_context_modes(predictions, baseline="full")

        self.assertEqual(report["modes"]["v56"]["accuracy"], 1.0)
        self.assertEqual(report["modes"]["v56"]["token_reduction"], 0.45)
        self.assertEqual(report["modes"]["v56"]["quality_delta"], 0.0)

    def test_checkpoint_writer_resumes_without_duplicate_case_ids(self):
        from knowledge_storm.evaluation.public_benchmarks.longmemeval import PredictionCheckpoint

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "predictions.jsonl"
            checkpoint = PredictionCheckpoint(path)
            checkpoint.append({"case_id": "q1", "prediction": "中文"})
            checkpoint.append({"case_id": "q1", "prediction": "中文"})
            checkpoint.append({"case_id": "q2", "prediction": "未知"})

            self.assertEqual(checkpoint.completed_ids(), {"q1", "q2"})
            self.assertEqual(len(checkpoint.rows()), 2)

    def test_longmemeval_retrieval_runner_compares_recent_and_v56(self):
        import sqlite3
        from contextlib import closing

        from knowledge_storm.evaluation.public_benchmarks.longmemeval_runner import run_memory_retrieval
        from knowledge_storm.evaluation.public_benchmarks.longmemeval import load_longmemeval

        dataset = load_longmemeval(FIXTURE_ROOT / "longmemeval_tiny.json")
        with tempfile.TemporaryDirectory() as temp_dir:
            report = run_memory_retrieval(dataset, Path(temp_dir), top_k=1)
            database = Path(temp_dir) / "memory" / "memory_v56.sqlite3"
            with closing(sqlite3.connect(database)) as connection:
                before = connection.execute("SELECT COUNT(*) FROM memory_events").fetchone()[0]
            run_memory_retrieval(dataset, Path(temp_dir), top_k=1)
            with closing(sqlite3.connect(database)) as connection:
                after = connection.execute("SELECT COUNT(*) FROM memory_events").fetchone()[0]

        self.assertEqual(report["case_count"], 2)
        self.assertEqual(set(report["modes"]), {"recent_window", "v56_memory"})
        self.assertEqual(report["answerable_case_count"], 1)
        self.assertIn("retrieval_recall_at_1", report["modes"]["v56_memory"])
        self.assertEqual(report["evidence_tier"], "public-official-retrieval-only")
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
