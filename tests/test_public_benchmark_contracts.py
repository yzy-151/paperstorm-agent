import unittest


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
            ["retrieve", "fuse", "rerank", "gate"],
            [stage["name"] for stage in report["predictions"][0]["retrieval_stages"]],
        )


if __name__ == "__main__":
    unittest.main()
