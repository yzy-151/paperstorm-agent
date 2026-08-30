import unittest


class ParentContextBudgetTests(unittest.TestCase):
    def _index(self):
        from knowledge_storm.retrieval import HashEmbeddingProvider, HybridPaperIndex

        children = [
            {
                "chunk_id": "c1",
                "document_id": "d1",
                "parent_id": "p1",
                "content": "child one",
            },
            {
                "chunk_id": "c2",
                "document_id": "d2",
                "parent_id": "p2",
                "content": "child two",
            },
        ]
        parents = [
            {"node_id": "p1", "node_type": "section", "content": "left one child one right one"},
            {"node_id": "p2", "node_type": "section", "content": "left two child two right two"},
        ]
        return HybridPaperIndex(
            children,
            parents=parents,
            embedding_provider=HashEmbeddingProvider(dim=16),
        )

    def test_minimum_quota_prevents_first_parent_starvation(self):
        index = self._index()
        ranked = [
            dict(index.chunks[0], score=0.9, final_rank=1),
            dict(index.chunks[1], score=0.1, final_rank=2),
        ]

        expanded = index.expand_parent_context(ranked, parent_budget_tokens=6)

        self.assertGreater(expanded[0]["parent_allocation"]["allocated_tokens"], 0)
        self.assertGreater(expanded[1]["parent_allocation"]["allocated_tokens"], 0)
        self.assertLessEqual(
            sum(item["parent_allocation"]["used_tokens"] for item in expanded), 6
        )

    def test_expansion_preserves_child_rank_and_records_section_metadata(self):
        index = self._index()
        ranked = [
            dict(index.chunks[1], score=0.8, final_rank=1),
            dict(index.chunks[0], score=0.7, final_rank=2),
        ]

        expanded = index.expand_parent_context(ranked, parent_budget_tokens=8)

        self.assertEqual(["c2", "c1"], [item["chunk_id"] for item in expanded])
        self.assertEqual([1, 2], [item["final_rank"] for item in expanded])
        self.assertEqual("section", expanded[0]["parent_allocation"]["parent_type"])
        self.assertIn("used_tokens", expanded[0]["parent_allocation"])

    def test_context_window_keeps_text_around_child_without_duplicating_child(self):
        index = self._index()
        ranked = [dict(index.chunks[0], score=1.0, final_rank=1)]

        expanded = index.expand_parent_context(ranked, parent_budget_tokens=4)

        context = expanded[0]["parent_context"]
        self.assertIn("left", context)
        self.assertIn("right", context)
        self.assertNotIn("child one", context)

    def test_qasper_diagnostic_attributes_extra_gold_to_parent_context(self):
        from knowledge_storm.evaluation.public_benchmarks.base import (
            BenchmarkCase,
            BenchmarkDataset,
            BenchmarkDocument,
        )
        from knowledge_storm.evaluation.public_benchmarks.qasper import (
            evaluate_qasper_parent_context_coverage,
        )

        documents = (
            BenchmarkDocument(
                "p1:0:0",
                "Method",
                "The selected passage introduces Newton-Schulz.",
                {"paper_id": "p1", "section_index": 0, "section": "Method"},
            ),
            BenchmarkDocument(
                "p1:0:1",
                "Method",
                "The gold passage explains orthogonalized momentum.",
                {"paper_id": "p1", "section_index": 0, "section": "Method"},
            ),
        )
        dataset = BenchmarkDataset(
            "qasper",
            "fixture",
            documents,
            (
                BenchmarkCase(
                    "q1",
                    "Why Newton-Schulz?",
                    ("p1:0:1",),
                    "test",
                    evidence_ids=("p1:0:1",),
                    metadata={"paper_id": "p1"},
                ),
            ),
        )
        rankings = [
            {
                "case_id": "q1",
                "mode": "hybrid_governed",
                "ranked_document_ids": ["p1:0:0"],
            }
        ]

        report, rows = evaluate_qasper_parent_context_coverage(
            dataset,
            rankings,
            mode="hybrid_governed",
            parent_budget_tokens=128,
        )

        self.assertEqual(0.0, report["child_gold_evidence_recall"])
        self.assertEqual(1.0, report["expanded_gold_evidence_recall"])
        self.assertEqual(1.0, report["recall_delta"])
        self.assertEqual(["p1:0:1"], rows[0]["additional_gold_evidence_ids"])


if __name__ == "__main__":
    unittest.main()
