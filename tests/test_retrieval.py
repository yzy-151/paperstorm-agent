import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class RetrievalProviderTest(unittest.TestCase):
    def test_hash_embedding_requires_explicit_selection(self):
        from knowledge_storm.retrieval import build_embedding_provider

        provider = build_embedding_provider("hash")

        self.assertEqual("hash", provider.name)
        self.assertEqual(provider.embed_query("PIM"), provider.embed_query("PIM"))

    def test_default_real_embedding_is_lazy(self):
        from knowledge_storm.retrieval import build_embedding_provider

        with mock.patch.dict(os.environ, {}, clear=True):
            provider = build_embedding_provider()

        self.assertTrue(provider.name.startswith("sentence-transformers:"))
        self.assertIsNone(provider.model)

    def test_index_save_uses_atomic_replace(self):
        from knowledge_storm.retrieval import HashEmbeddingProvider, HybridPaperIndex

        index = HybridPaperIndex.from_documents(
            [{"document_id": "doc-1", "title": "PIM", "text": "passive intermodulation"}],
            embedding_provider=HashEmbeddingProvider(),
        )
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch(
            "knowledge_storm.retrieval.os.replace", wraps=os.replace
        ) as replace:
            path = Path(temp_dir) / "index.json"
            index.save(path)
            replace.assert_called_once()
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
