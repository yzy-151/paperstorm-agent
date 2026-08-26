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
            ["plan", "retrieve", "fuse", "gate", "parent_expand"],
            [stage["name"] for stage in report["predictions"][0]["retrieval_stages"]],
        )
        self.assertIn("search_plan", report["predictions"][0])

    def test_qasper_structured_run_uses_section_parent_without_changing_gold_id(self):
        from knowledge_storm.evaluation.public_benchmarks.base import BenchmarkCase, BenchmarkDataset, BenchmarkDocument
        from knowledge_storm.evaluation.public_benchmarks.runner import HashEmbeddingProvider, run_retrieval_benchmark

        paragraph_id = "paper-1::section-0::paragraph-0"
        dataset = BenchmarkDataset(
            "qasper",
            "fixture",
            (BenchmarkDocument(
                paragraph_id,
                "Paper",
                "Method\ncontrastive loss is optimized",
                {"paper_id": "paper-1", "section": "Method", "section_index": 0, "paragraph_index": 0, "raw_text": "contrastive loss is optimized"},
            ),),
            (BenchmarkCase("q1", "Which loss?", (paragraph_id,), "validation", metadata={"paper_id": "paper-1"}),),
        )
        report = run_retrieval_benchmark(
            dataset,
            HashEmbeddingProvider(),
            modes=("hybrid",),
            top_k=1,
            bootstrap_samples=5,
            scope_field="paper_id",
            structured_nodes=True,
            parent_budget_tokens=64,
        )

        prediction = report["predictions"][0]
        self.assertEqual([paragraph_id], prediction["ranked_document_ids"])
        parent = next(stage for stage in prediction["retrieval_stages"] if stage["name"] == "parent_expand")
        self.assertEqual("completed", parent["status"])
        self.assertEqual("structured-parent-child-v1", report["manifest"]["node_schema"])

    def test_qasper_parent_expansion_does_not_repeat_child_raw_text(self):
        from knowledge_storm.evaluation.public_benchmarks.base import BenchmarkDataset, BenchmarkDocument
        from knowledge_storm.evaluation.public_benchmarks.runner import HashEmbeddingProvider, _benchmark_nodes
        from knowledge_storm.retrieval import HybridPaperIndex

        raw = "contrastive loss is optimized"
        dataset = BenchmarkDataset(
            "qasper", "fixture",
            (BenchmarkDocument(
                "paper-1::section-0::paragraph-0", "Paper", "Method\n" + raw,
                {"paper_id": "paper-1", "section": "Method", "section_index": 0, "raw_text": raw},
            ),), (),
        )
        index = HybridPaperIndex(_benchmark_nodes(dataset, structured=True), HashEmbeddingProvider())
        result = index.search("contrastive loss", mode="hybrid", top_k=1, parent_budget_tokens=64)[0]
        self.assertEqual(1, result["expanded_content"].count(raw))

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
