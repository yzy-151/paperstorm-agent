import tempfile
import unittest
from pathlib import Path


class TextAnalyzerTests(unittest.TestCase):
    def test_domain_analyzer_preserves_pim_phrases_and_bigrams(self):
        from knowledge_storm.text_analyzers import JiebaDomainAnalyzer

        analyzer = JiebaDomainAnalyzer()
        tokens = analyzer.tokenize("无源互调神经网络抑制与DPD")

        self.assertIn("无源互调", tokens)
        self.assertIn("神经网络抑制", tokens)
        self.assertIn("源互", tokens)
        self.assertIn("dpd", tokens)

    def test_dictionary_content_changes_analyzer_revision(self):
        from knowledge_storm.text_analyzers import JiebaDomainAnalyzer

        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "first.txt"
            second = Path(temp_dir) / "second.txt"
            first.write_text("无源互调\n", encoding="utf-8")
            second.write_text("无源互调\n数字预失真\n", encoding="utf-8")

            self.assertNotEqual(
                JiebaDomainAnalyzer(first).revision,
                JiebaDomainAnalyzer(second).revision,
            )

    def test_legacy_bigram_analyzer_remains_dependency_free(self):
        from knowledge_storm.text_analyzers import CjkBigramAnalyzer

        tokens = CjkBigramAnalyzer().tokenize("无源互调 PIM-3")

        self.assertIn("无", tokens)
        self.assertIn("无源", tokens)
        self.assertIn("pim-3", tokens)


class RetrievalAnalyzerIntegrationTests(unittest.TestCase):
    def test_index_manifest_records_analyzer_contract(self):
        from knowledge_storm.retrieval import HashEmbeddingProvider, HybridPaperIndex
        from knowledge_storm.text_analyzers import CjkBigramAnalyzer

        analyzer = CjkBigramAnalyzer()
        index = HybridPaperIndex(
            [{"chunk_id": "c1", "document_id": "d1", "content": "无源互调"}],
            embedding_provider=HashEmbeddingProvider(dim=16),
            text_analyzer=analyzer,
        )

        self.assertEqual(analyzer.name, index.manifest["text_analyzer"])
        self.assertEqual(analyzer.revision, index.manifest["text_analyzer_revision"])

    def test_domain_bm25_ranks_pim_evidence_above_ram_distractor(self):
        from knowledge_storm.retrieval import HashEmbeddingProvider, HybridPaperIndex
        from knowledge_storm.text_analyzers import JiebaDomainAnalyzer

        index = HybridPaperIndex(
            [
                {
                    "chunk_id": "pim",
                    "document_id": "pim-paper",
                    "content": "无源互调抑制采用神经网络与数字预失真方法",
                },
                {
                    "chunk_id": "ram",
                    "document_id": "ram-paper",
                    "content": "PIM system memory uses RAM and DRAM acceleration",
                },
            ],
            embedding_provider=HashEmbeddingProvider(dim=16),
            text_analyzer=JiebaDomainAnalyzer(),
        )

        result = index.search("无源互调神经网络抑制", mode="bm25", top_k=1)

        self.assertEqual("pim-paper", result[0]["document_id"])


if __name__ == "__main__":
    unittest.main()
