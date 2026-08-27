import tempfile
import unittest
from pathlib import Path

import numpy as np


class ExactDenseBackendTests(unittest.TestCase):
    def test_exact_cosine_search_is_stable_and_scope_aware(self):
        from knowledge_storm.dense_index import ExactDenseBackend

        vectors = np.asarray([[1, 0], [0.8, 0.2], [0, 1]], dtype=np.float32)
        backend = ExactDenseBackend(vectors)

        global_result = backend.search([1, 0], top_k=2)
        scoped_result = backend.search([1, 0], top_k=2, allowed_indices=[1, 2])

        self.assertEqual((0, 1), global_result.indices)
        self.assertEqual((1, 2), scoped_result.indices)
        self.assertEqual("exact", scoped_result.backend)


class HnswDenseBackendTests(unittest.TestCase):
    def test_hnsw_recall_and_persistence_match_exact_or_dependency_is_explicit(self):
        from knowledge_storm.dense_index import ExactDenseBackend, HnswDenseBackend

        rng = np.random.default_rng(55)
        vectors = rng.normal(size=(200, 16)).astype(np.float32)
        query = rng.normal(size=16).astype(np.float32)
        exact = ExactDenseBackend(vectors).search(query, top_k=10)
        try:
            hnsw = HnswDenseBackend(vectors, ef_search=100)
        except RuntimeError as exc:
            self.assertIn("usearch", str(exc))
            self.skipTest("USearch is not installed")

        approximate = hnsw.search(query, top_k=10)
        recall = len(set(exact.indices) & set(approximate.indices)) / 10.0
        self.assertGreaterEqual(recall, 0.9)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dense.hnsw"
            hnsw.save(path)
            loaded = HnswDenseBackend.load(path)
            self.assertEqual(approximate.indices, loaded.search(query, 10).indices)


class AutoDenseBackendTests(unittest.TestCase):
    def test_scoped_search_falls_back_to_authorized_exact_candidates(self):
        from knowledge_storm.dense_index import AutoDenseBackend

        vectors = np.asarray([[1, 0], [0.9, 0.1], [0, 1]], dtype=np.float32)
        backend = AutoDenseBackend(vectors, mode="exact")

        result = backend.search([1, 0], top_k=2, allowed_indices=[1, 2])

        self.assertEqual((1, 2), result.indices)
        self.assertEqual("exact", result.backend)
        self.assertEqual("acl_exact_fallback", result.reason)

    def test_auto_exposes_why_exact_was_selected_below_threshold(self):
        from knowledge_storm.dense_index import AutoDenseBackend

        backend = AutoDenseBackend(np.eye(3, dtype=np.float32), mode="auto", ann_threshold=10)
        result = backend.search([1, 0, 0], top_k=1)

        self.assertEqual("exact", result.backend)
        self.assertEqual("below_ann_threshold", result.reason)


class HybridIndexDenseBackendIntegrationTests(unittest.TestCase):
    def test_hybrid_index_exposes_backend_decision_and_acl_fallback(self):
        from knowledge_storm.retrieval import HashEmbeddingProvider, HybridPaperIndex

        index = HybridPaperIndex(
            [
                {"chunk_id": "a", "document_id": "private", "content": "alpha"},
                {"chunk_id": "b", "document_id": "allowed", "content": "beta"},
            ],
            embedding_provider=HashEmbeddingProvider(dim=16),
            dense_backend_mode="exact",
        )

        global_result = index.search("alpha", mode="dense", top_k=1)[0]
        scoped_result = index.search(
            "alpha", mode="dense", top_k=1, allowed_document_ids=["allowed"]
        )[0]

        self.assertEqual("exact", global_result["dense_backend"])
        self.assertEqual("configured_exact", global_result["dense_backend_reason"])
        self.assertEqual("allowed", scoped_result["document_id"])
        self.assertEqual("acl_exact_fallback", scoped_result["dense_backend_reason"])


if __name__ == "__main__":
    unittest.main()
