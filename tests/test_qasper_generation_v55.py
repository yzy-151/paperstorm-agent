import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class QasperOfficialMetricTest(unittest.TestCase):
    def test_official_prediction_export_uses_qasper_field_names(self):
        from knowledge_storm.evaluation.public_benchmarks.qasper_generation import (
            write_official_qasper_predictions,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "official.jsonl"
            write_official_qasper_predictions(
                path,
                {
                    "q1": {
                        "answer": "Yes",
                        "evidence_texts": ["supporting paragraph"],
                    }
                },
            )
            row = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(
            row,
            {
                "question_id": "q1",
                "predicted_answer": "Yes",
                "predicted_evidence": ["supporting paragraph"],
            },
        )

    def test_official_answer_f1_removes_articles_and_punctuation(self):
        from knowledge_storm.evaluation.public_benchmarks.qasper_generation import (
            official_qasper_metrics,
        )
        from knowledge_storm.evaluation.public_benchmarks.base import BenchmarkCase

        case = BenchmarkCase(
            "q1",
            "What model?",
            ("p1",),
            "validation",
            {"p1": 1},
            metadata={
                "qasper_references": (
                    {
                        "answer": "neural model",
                        "answer_type": "extractive",
                        "evidence_ids": ("p1",),
                    },
                )
            },
        )

        report = official_qasper_metrics(
            (case,),
            {"q1": {"answer": "The neural model.", "evidence_ids": ["p1"]}},
        )

        self.assertEqual(report["answer_f1"], 1.0)
        self.assertEqual(report["evidence_f1"], 1.0)
        self.assertEqual(report["answer_f1_by_type"]["extractive"], 1.0)

    def test_official_metrics_score_correct_unanswerable_prediction(self):
        from knowledge_storm.evaluation.public_benchmarks.qasper_generation import (
            official_qasper_metrics,
        )
        from knowledge_storm.evaluation.public_benchmarks.base import BenchmarkCase

        case = BenchmarkCase(
            "q2",
            "Unknown?",
            (),
            "validation",
            metadata={
                "qasper_references": (
                    {
                        "answer": "Unanswerable",
                        "answer_type": "none",
                        "evidence_ids": (),
                    },
                )
            },
            unanswerable=True,
        )

        report = official_qasper_metrics(
            (case,),
            {"q2": {"answer": "Unanswerable", "evidence_ids": []}},
        )

        self.assertEqual(report["answer_f1"], 1.0)
        self.assertEqual(report["evidence_f1"], 1.0)
        self.assertEqual(report["missing_predictions"], 0)


class QasperReferenceAdapterTest(unittest.TestCase):
    def test_adapter_preserves_annotation_level_answer_and_evidence_sets(self):
        from knowledge_storm.evaluation.public_benchmarks.qasper import (
            load_qasper_records,
        )

        dataset = load_qasper_records(
            [
                {
                    "id": "paper-1",
                    "title": "Title",
                    "full_text": {
                        "section_name": ["Method"],
                        "paragraphs": [["first evidence", "second evidence"]],
                    },
                    "qas": {
                        "question": ["What was used?"],
                        "question_id": ["q1"],
                        "answers": [
                            {
                                "answer": [
                                    {
                                        "unanswerable": False,
                                        "extractive_spans": ["model A", "dataset B"],
                                        "free_form_answer": "",
                                        "yes_no": None,
                                        "evidence": [
                                            "first evidence",
                                            "FLOAT SELECTED: Table 1",
                                        ],
                                    },
                                    {
                                        "unanswerable": False,
                                        "extractive_spans": [],
                                        "free_form_answer": "a neural approach",
                                        "yes_no": None,
                                        "evidence": ["second evidence"],
                                    },
                                ]
                            }
                        ],
                    },
                }
            ],
            split="validation",
        )

        references = dataset.cases[0].metadata["qasper_references"]
        self.assertEqual(references[0]["answer"], "model A, dataset B")
        self.assertEqual(references[0]["answer_type"], "extractive")
        self.assertEqual(
            references[0]["evidence_ids"],
            ("paper-1::section-0::paragraph-0",),
        )
        self.assertEqual(references[0]["evidence_texts"], ("first evidence",))
        self.assertEqual(references[1]["answer_type"], "abstractive")


class QasperGenerationRunnerTest(unittest.TestCase):
    def test_parser_repairs_unquoted_qasper_evidence_ids_without_eval(self):
        from knowledge_storm.evaluation.public_benchmarks.qasper_generation import (
            parse_generation_json,
        )

        parsed = parse_generation_json(
            """{
              "answer": "HR-BiLSTM",
              "abstained": false,
              "evidence_ids": [1704.06194::section-5::paragraph-7]
            }"""
        )

        self.assertEqual(
            parsed["evidence_ids"], ["1704.06194::section-5::paragraph-7"]
        )

    def test_runner_retries_malformed_json_and_accounts_for_both_calls(self):
        from knowledge_storm.evaluation.public_benchmarks.base import (
            BenchmarkCase,
            BenchmarkDataset,
            BenchmarkDocument,
        )
        from knowledge_storm.evaluation.public_benchmarks.qasper_generation import (
            run_qasper_generation,
        )

        dataset = BenchmarkDataset(
            "qasper",
            "fixture",
            (BenchmarkDocument("p1", "Paper", "evidence"),),
            (BenchmarkCase("q1", "Question?", (), "validation"),),
        )
        outputs = iter(
            [
                {"text": '{"answer": broken}', "usage": {"total_tokens": 7}},
                {
                    "text": '{"answer":"yes","abstained":false,"evidence_ids":["p1"]}',
                    "usage": {"total_tokens": 9},
                },
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            report = run_qasper_generation(
                dataset,
                rankings={"q1": ["p1"]},
                generate=lambda _prompt: next(outputs),
                output_dir=temp_dir,
                model_name="fake/model",
                parse_attempts=2,
            )
            row = json.loads(
                (Path(temp_dir) / "predictions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )

        self.assertEqual(row["status"], "succeeded")
        self.assertEqual(row["generation_attempts"], 2)
        self.assertEqual(row["usage"]["total_tokens"], 16)
        self.assertEqual(len(row["raw_responses"]), 2)
        self.assertEqual(report["failed_predictions"], 0)

    def test_runner_covers_unanswerable_cases_and_resumes_checkpoint(self):
        from knowledge_storm.evaluation.public_benchmarks.base import (
            BenchmarkCase,
            BenchmarkDataset,
            BenchmarkDocument,
        )
        from knowledge_storm.evaluation.public_benchmarks.qasper_generation import (
            run_qasper_generation,
        )

        documents = (
            BenchmarkDocument("p1", "Paper", "retrieved evidence"),
            BenchmarkDocument("p2", "Paper", "other paragraph"),
        )
        cases = (
            BenchmarkCase(
                "q1",
                "What?",
                ("p1",),
                "validation",
                {"p1": 1},
                metadata={
                    "paper_id": "paper",
                    "qasper_references": (
                        {
                            "answer": "SECRET_GOLD",
                            "answer_type": "extractive",
                            "evidence_ids": ("p1",),
                        },
                    ),
                },
            ),
            BenchmarkCase(
                "q2",
                "Can this be answered?",
                (),
                "validation",
                metadata={
                    "paper_id": "paper",
                    "qasper_references": (
                        {
                            "answer": "Unanswerable",
                            "answer_type": "none",
                            "evidence_ids": (),
                        },
                    ),
                },
                unanswerable=True,
            ),
        )
        dataset = BenchmarkDataset("qasper", "fixture", documents, cases)
        calls = []

        def generate(prompt):
            calls.append(prompt)
            self.assertNotIn("SECRET_GOLD", prompt)
            if "Can this be answered?" in prompt:
                return {
                    "text": '{"answer":"Unanswerable","abstained":true,"evidence_ids":[]}',
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                }
            return {
                "text": '```json\n{"answer":"generated","abstained":false,"evidence_ids":["p1"]}\n```',
                "usage": {"prompt_tokens": 11, "completion_tokens": 3},
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            first = run_qasper_generation(
                dataset,
                rankings={"q1": ["p1"], "q2": ["p2"]},
                generate=generate,
                output_dir=temp_dir,
                model_name="fake/model",
            )
            second = run_qasper_generation(
                dataset,
                rankings={"q1": ["p1"], "q2": ["p2"]},
                generate=lambda _prompt: self.fail("checkpoint was not resumed"),
                output_dir=temp_dir,
                model_name="fake/model",
            )
            rows = [
                json.loads(line)
                for line in (Path(temp_dir) / "predictions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual(len(calls), 2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(first["metrics"]["case_count"], 2)
        self.assertEqual(second["metrics"], first["metrics"])
        self.assertEqual(first["usage"]["prompt_tokens"], 21)
        self.assertEqual(rows[1]["answer"], "Unanswerable")

    def test_ranking_completion_only_searches_missing_cases_in_same_paper(self):
        from knowledge_storm.evaluation.public_benchmarks.base import (
            BenchmarkCase,
            BenchmarkDataset,
            BenchmarkDocument,
        )
        from knowledge_storm.evaluation.public_benchmarks.qasper_generation import (
            complete_qasper_rankings,
        )
        from knowledge_storm.evaluation.public_benchmarks.runner import (
            HashEmbeddingProvider,
        )

        dataset = BenchmarkDataset(
            "qasper",
            "fixture",
            (
                BenchmarkDocument("a1", "A", "alpha", {"paper_id": "a"}),
                BenchmarkDocument("b1", "B", "beta", {"paper_id": "b"}),
            ),
            (
                BenchmarkCase("q1", "alpha", (), "validation", metadata={"paper_id": "a"}),
                BenchmarkCase("q2", "beta", (), "validation", metadata={"paper_id": "b"}),
            ),
        )

        rankings = complete_qasper_rankings(
            dataset,
            initial_rankings={"q1": ["a1"]},
            embedding_provider=HashEmbeddingProvider(),
            mode="hybrid",
            top_k=1,
        )

        self.assertEqual(rankings, {"q1": ["a1"], "q2": ["b1"]})

    def test_governed_label_uses_hybrid_for_missing_ranking_completion(self):
        from knowledge_storm.evaluation.public_benchmarks.base import (
            BenchmarkCase, BenchmarkDataset, BenchmarkDocument,
        )
        from knowledge_storm.evaluation.public_benchmarks.qasper_generation import (
            complete_qasper_rankings,
        )
        from knowledge_storm.evaluation.public_benchmarks.runner import HashEmbeddingProvider

        dataset = BenchmarkDataset(
            "qasper", "fixture",
            (BenchmarkDocument("p1", "Paper", "alpha", {"paper_id": "paper"}),),
            (BenchmarkCase("q1", "alpha", (), "test", metadata={"paper_id": "paper"}),),
        )

        rankings = complete_qasper_rankings(
            dataset, {}, HashEmbeddingProvider(), mode="hybrid_governed", top_k=1
        )

        self.assertEqual(rankings, {"q1": ["p1"]})

    def test_litellm_generator_retries_and_returns_usage_without_key_leakage(self):
        from knowledge_storm.evaluation.public_benchmarks.qasper_generation import (
            LiteLLMJsonGenerator,
        )

        completion = mock.Mock(
            side_effect=[
                TimeoutError("temporary"),
                {
                    "id": "response-1",
                    "choices": [{"message": {"content": '{"answer":"yes"}'}}],
                    "_hidden_params": {"response_cost": 0.00012},
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 3,
                        "total_tokens": 15,
                    },
                },
            ]
        )
        generator = LiteLLMJsonGenerator(
            model="deepseek/deepseek-chat",
            api_key="secret-key",
            api_base="https://api.deepseek.com",
            max_attempts=2,
            completion=completion,
            sleep=lambda _seconds: None,
        )

        result = generator("prompt")

        self.assertEqual(result["usage"]["total_tokens"], 15)
        self.assertEqual(result["usage"]["cost_usd"], 0.00012)
        self.assertNotIn("secret-key", json.dumps(result))
        self.assertEqual(completion.call_count, 2)


class QasperGenerationCliTest(unittest.TestCase):
    def test_cli_requires_explicit_split_rankings_and_output(self):
        from examples.storm_examples.run_qasper_answer_benchmark import build_parser

        args = build_parser().parse_args(
            [
                "--split",
                "validation",
                "--retrieval-predictions",
                "results/retrieval.jsonl",
                "--output-dir",
                "results/answers",
                "--smoke-limit",
                "20",
            ]
        )

        self.assertEqual(args.split, "validation")
        self.assertEqual(args.retrieval_mode, "hybrid_governed")
        self.assertEqual(args.model, "deepseek/deepseek-chat")
        self.assertEqual(args.smoke_limit, 20)
        self.assertTrue(args.claim_validation)

    def test_cli_accepts_official_local_dataset(self):
        from examples.storm_examples.run_qasper_answer_benchmark import build_parser

        args = build_parser().parse_args(
            [
                "--split", "test",
                "--retrieval-predictions", "retrieval.jsonl",
                "--output-dir", "answers",
                "--dataset-file", "qasper-test-v0.3.json",
                "--no-claim-validation",
            ]
        )

        self.assertEqual(args.dataset_file, "qasper-test-v0.3.json")
        self.assertFalse(args.claim_validation)


if __name__ == "__main__":
    unittest.main()
