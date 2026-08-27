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


if __name__ == "__main__":
    unittest.main()
