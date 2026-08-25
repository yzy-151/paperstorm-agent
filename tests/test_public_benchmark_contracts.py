import unittest
import json


class PublicBenchmarkContractTest(unittest.TestCase):
    def test_retrieval_predictions_expose_pipeline_stages(self):
        from knowledge_storm.evaluation.public_benchmarks.base import (
            BenchmarkCase,
            BenchmarkDataset,
            BenchmarkDocument,
        )
        from knowledge_storm.evaluation.public_benchmarks.runner import (
            HashEmbeddingProvider,
            run_retrieval_benchmark,
        )

        dataset = BenchmarkDataset(
            name="fixture",
            version="1",
            documents=(
                BenchmarkDocument("doc-1", "PIM", "passive intermodulation RF"),
                BenchmarkDocument("doc-2", "DRAM", "memory controller"),
            ),
            cases=(
                BenchmarkCase("case-1", "passive intermodulation", ("doc-1",), "test"),
            ),
        )

        report = run_retrieval_benchmark(
            dataset,
            embedding_provider=HashEmbeddingProvider(),
            modes=("hybrid",),
            top_k=1,
            bootstrap_samples=10,
        )

        self.assertEqual(
            ["plan", "retrieve", "fuse", "parent_expand", "gate"],
            [stage["name"] for stage in report["predictions"][0]["retrieval_stages"]],
        )
        self.assertIn("search_plan", report["predictions"][0])

    def test_prediction_includes_sanitized_explicit_milestone_only(self):
        from knowledge_storm.evaluation.public_benchmarks.base import BenchmarkCase, BenchmarkDataset, BenchmarkDocument
        from knowledge_storm.evaluation.public_benchmarks.runner import HashEmbeddingProvider, run_retrieval_benchmark

        dataset = BenchmarkDataset("fixture", "1", (BenchmarkDocument("doc", "T", "alpha"),), (BenchmarkCase("case", "alpha", ("doc",), "test"),))
        report = run_retrieval_benchmark(
            dataset,
            HashEmbeddingProvider(),
            modes=("bm25",),
            top_k=1,
            bootstrap_samples=5,
            milestone_metadata={"milestone": "P1", "api_key": "must-not-leak"},
        )
        plain = run_retrieval_benchmark(dataset, HashEmbeddingProvider(), modes=("bm25",), top_k=1, bootstrap_samples=5)

        self.assertEqual("P1", report["predictions"][0]["milestone"])
        self.assertNotIn("must-not-leak", json.dumps(report))
        self.assertNotIn("milestone", plain["predictions"][0])


if __name__ == "__main__":
    unittest.main()
