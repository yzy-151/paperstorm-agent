import unittest
from unittest import mock


class RerankerProfileTests(unittest.TestCase):
    def test_default_and_quality_profiles_are_explicit(self):
        from knowledge_storm.retrieval_profiles import get_reranker_profile

        default = get_reranker_profile()
        quality = get_reranker_profile("quality-gpu")

        self.assertEqual("cpu-balanced", default.name)
        self.assertEqual("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", default.model_name)
        self.assertEqual("cpu", default.device)
        self.assertEqual("BAAI/bge-reranker-v2-m3", quality.model_name)
        self.assertEqual("cuda", quality.device)
        self.assertLessEqual(default.max_candidates, 20)

    def test_cpu_profile_reports_actual_runtime_and_batch_size(self):
        from knowledge_storm.retrieval import CrossEncoderReranker

        reranker = CrossEncoderReranker(
            profile="cpu-balanced", score_fn=lambda pairs: range(len(pairs))
        )
        output = reranker.rerank(
            "query",
            [
                {"chunk_id": "a", "content": "a"},
                {"chunk_id": "b", "content": "b"},
            ],
        )

        self.assertEqual("cpu", reranker.actual_device)
        self.assertEqual("cpu-balanced", reranker.profile.name)
        self.assertEqual("b", output[0]["chunk_id"])

    def test_quality_gpu_profile_fails_clearly_without_cuda(self):
        from knowledge_storm.retrieval import CrossEncoderReranker

        reranker = CrossEncoderReranker(profile="quality-gpu")
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
                reranker._scores([("query", "document")])


class RerankerTraceTests(unittest.TestCase):
    def test_pipeline_trace_exposes_profile_device_and_candidate_cap(self):
        from knowledge_storm.retrieval import CrossEncoderReranker, HashEmbeddingProvider
        from knowledge_storm.retrieval_pipeline import RetrievalPipeline, RetrievalRequest

        class Index:
            embedding_provider = HashEmbeddingProvider(dim=16)

            def search(self, query, **_kwargs):
                return [
                    {
                        "chunk_id": "c{0}".format(index),
                        "document_id": "d{0}".format(index),
                        "content": "content {0}".format(index),
                        "rrf_score": 1.0 / (index + 1),
                    }
                    for index in range(30)
                ]

        reranker = CrossEncoderReranker(
            profile="cpu-balanced", score_fn=lambda pairs: [1.0] * len(pairs)
        )
        output = RetrievalPipeline(Index(), reranker=reranker).search(
            RetrievalRequest(
                query="test query",
                mode="hybrid_rerank",
                enable_reranker=True,
                candidate_k=30,
                top_k=5,
            )
        )

        self.assertEqual("cpu-balanced", output["models"]["reranker_profile"])
        self.assertEqual("cpu", output["models"]["reranker_device"])
        self.assertEqual(20, output["rerank_runtime"]["candidate_count"])
        self.assertEqual(20, output["rerank_runtime"]["candidate_limit"])


if __name__ == "__main__":
    unittest.main()
