import unittest


class RecordingIndex:
    def __init__(self):
        self.embedding_provider = type("Provider", (), {"name": "test-embedding"})()
        self.calls = []

    def search(self, query, mode, top_k, candidate_k, reranker=None):
        self.calls.append(
            {
                "query": query,
                "mode": mode,
                "top_k": top_k,
                "candidate_k": candidate_k,
                "reranker": reranker,
            }
        )
        return [
            {
                "chunk_id": "chunk-1",
                "title": "Passive intermodulation",
                "content": "PIM is caused by nonlinear passive junctions.",
                "score": 0.8,
            }
        ]


class RetrievalPipelineTest(unittest.TestCase):
    def test_returns_stable_stage_schema_when_reranker_is_disabled(self):
        from knowledge_storm.retrieval_pipeline import (
            RetrievalPipeline,
            RetrievalRequest,
        )

        index = RecordingIndex()
        pipeline = RetrievalPipeline(index=index)

        result = pipeline.search(
            RetrievalRequest(query="PIM suppression", top_k=5, candidate_k=20)
        )

        self.assertEqual(["chunk-1"], [item["chunk_id"] for item in result["results"]])
        self.assertEqual(
            ["retrieve", "fuse", "rerank", "gate"],
            [stage["name"] for stage in result["stages"]],
        )
        self.assertEqual("skipped", result["stages"][2]["status"])
        self.assertEqual(1, result["schema_revision"])

    def test_rejects_empty_query(self):
        from knowledge_storm.retrieval_pipeline import (
            RetrievalPipeline,
            RetrievalRequest,
        )

        with self.assertRaisesRegex(ValueError, "query is required"):
            RetrievalPipeline(RecordingIndex()).search(RetrievalRequest(query=" "))

    def test_expected_keywords_filter_out_wrong_domain_candidates(self):
        from knowledge_storm.retrieval_pipeline import (
            RetrievalPipeline,
            RetrievalRequest,
        )

        result = RetrievalPipeline(RecordingIndex()).search(
            RetrievalRequest(
                query="PIM suppression",
                expected_keywords=("DRAM",),
            )
        )

        self.assertEqual([], result["results"])
        self.assertEqual("completed", result["stages"][3]["status"])

    def test_knowledge_base_uses_injected_pipeline_ranking(self):
        from knowledge_storm.paperstorm_qa import PaperStormKnowledgeBase
        from knowledge_storm.retrieval_pipeline import RetrievalPipeline

        index = RecordingIndex()
        knowledge_base = PaperStormKnowledgeBase(
            documents=[], retrieval_pipeline=RetrievalPipeline(index)
        )

        results = knowledge_base.search("PIM suppression", top_k=3)

        self.assertEqual(["chunk-1"], [item["chunk_id"] for item in results])
        self.assertEqual("test-embedding", knowledge_base.retrieval_meta["embedding"])
        self.assertEqual("hybrid", knowledge_base.retrieval_meta["mode"])


if __name__ == "__main__":
    unittest.main()
