import json
import tempfile
import unittest
from pathlib import Path

from knowledge_storm.evaluation.public_benchmarks.base import (
    BenchmarkCase,
    BenchmarkDataset,
    BenchmarkDocument,
)
from knowledge_storm.evaluation.public_benchmarks.v60_harness import (
    compute_pareto_frontier,
    run_context_profile_benchmark,
    run_longmemeval_end_to_end,
)
from knowledge_storm.evaluation.public_benchmarks.v60_llm import LongMemEvalJudge


class PaperStormV60BenchmarkTest(unittest.TestCase):
    def test_context_profiles_measure_quality_ttft_tokens_and_cost(self):
        dataset = BenchmarkDataset(
            name="tiny-longbench",
            version="test",
            documents=(BenchmarkDocument("d1", "doc", "evidence " * 100),),
            cases=(BenchmarkCase("c1", "question", ("d1",), "test", answers=("A",)),),
        )

        def reader(prompt, profile_tokens):
            return {
                "text": "A",
                "usage": {"prompt_tokens": min(profile_tokens, len(prompt)), "completion_tokens": 1},
                "ttft_ms": profile_tokens / 1000,
                "latency_ms": profile_tokens / 500,
                "cost_usd": profile_tokens / 1_000_000,
            }

        with tempfile.TemporaryDirectory() as directory:
            report = run_context_profile_benchmark(
                dataset,
                reader,
                Path(directory),
                profiles=(128_000, 256_000, 512_000),
            )
        self.assertEqual(set(report["profiles"]), {"128K", "256K", "512K"})
        self.assertEqual(report["profiles"]["128K"]["accuracy"], 1.0)
        self.assertIn("ttft_p50_ms", report["profiles"]["128K"])
        self.assertIn("cost_usd", report["profiles"]["128K"])
        self.assertTrue(report["pareto_frontier"])

    def test_pareto_removes_strictly_dominated_profiles(self):
        profiles = {
            "A": {"accuracy": 0.8, "mean_input_tokens": 100, "ttft_p50_ms": 10, "cost_usd": 0.1},
            "B": {"accuracy": 0.7, "mean_input_tokens": 120, "ttft_p50_ms": 12, "cost_usd": 0.2},
            "C": {"accuracy": 0.9, "mean_input_tokens": 150, "ttft_p50_ms": 14, "cost_usd": 0.3},
        }
        self.assertEqual(compute_pareto_frontier(profiles), ["A", "C"])

    def test_longmemeval_compares_three_modes_with_reader_and_judge(self):
        payload = [{
            "question_id": "q1",
            "question": "Where is the key?",
            "answer": "drawer",
            "question_type": "single-session-user",
            "haystack_session_ids": ["s1", "s2"],
            "haystack_sessions": [
                [{"role": "user", "content": "The key is in the drawer."}],
                [{"role": "user", "content": "We discussed lunch."}],
            ],
            "answer_session_ids": ["s1"],
        }]

        def reader(question, evidence, mode):
            return {"text": "drawer" if "drawer" in evidence else "unknown", "usage": {"prompt_tokens": 10, "completion_tokens": 1}}

        def judge(question, gold, prediction, question_type):
            return {"correct": prediction == gold, "explanation": "exact test judge"}

        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "longmem.json"
            dataset.write_text(json.dumps(payload), encoding="utf-8")
            report = run_longmemeval_end_to_end(
                dataset,
                Path(directory) / "run",
                reader=reader,
                judge=judge,
                embedding_provider=None,
                limit=1,
            )
        self.assertEqual(set(report["modes"]), {"recent", "fts_session", "v56_memory"})
        self.assertEqual(report["case_count"], 1)
        self.assertEqual(report["judge_protocol"], "reader_plus_llm_judge")
        self.assertIn("answer_accuracy", report["modes"]["fts_session"])

    def test_longmemeval_judge_uses_binary_type_specific_protocol(self):
        captured = {}

        def completion(**kwargs):
            captured.update(kwargs)
            return {
                "choices": [{"message": {"content": "yes"}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 1},
            }

        judge = LongMemEvalJudge(
            "openai/gpt-4o-2024-08-06",
            "test-key",
            completion=completion,
        )
        result = judge("How many days?", "18", "19", "temporal-reasoning")
        self.assertTrue(result["correct"])
        self.assertEqual(captured["max_tokens"], 10)
        self.assertIn("one-unit", captured["messages"][0]["content"])


if __name__ == "__main__":
    unittest.main()
