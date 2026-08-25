import unittest
from dataclasses import FrozenInstanceError


class RecordingIndex:
    def __init__(self):
        self.embedding_provider = type("Provider", (), {"name": "test-embedding"})()
        self.calls = []

    def search(
        self,
        query,
        mode,
        top_k,
        candidate_k,
        reranker=None,
        parent_budget_tokens=0,
    ):
        self.calls.append(
            {
                "query": query,
                "mode": mode,
                "top_k": top_k,
                "candidate_k": candidate_k,
                "reranker": reranker,
                "parent_budget_tokens": parent_budget_tokens,
            }
        )
        return [
            {
                "chunk_id": "chunk-1",
                "title": "Passive intermodulation",
                "content": "PIM is caused by nonlinear passive junctions.",
                "score": 0.8,
                "metadata": {"year": 2024, "tags": ["rf", "pim"]},
                "parent_context": "parent evidence" if parent_budget_tokens else "",
                "expanded_content": "parent evidence\n\nPIM is caused by nonlinear passive junctions."
                if parent_budget_tokens
                else "PIM is caused by nonlinear passive junctions.",
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
            ["plan", "retrieve", "fuse", "parent_expand", "gate"],
            [stage["name"] for stage in result["stages"]],
        )
        self.assertEqual("completed", result["stages"][2]["status"])
        self.assertGreaterEqual(result["schema_revision"], 2)
        for stage in result["stages"]:
            self.assertEqual(
                {"name", "status", "input_count", "output_count", "latency_ms", "reason"},
                set(stage),
            )
        self.assertEqual("PIM suppression", result["search_plan"]["standalone_query"])

    def test_request_is_frozen_and_accepts_explicit_search_plan(self):
        from knowledge_storm.retrieval_pipeline import RetrievalPipeline, RetrievalRequest
        from knowledge_storm.search_planning import SearchPlan

        plan = SearchPlan("PIM", "passive intermodulation", filters={"year": 2024})
        request = RetrievalRequest(
            query="PIM", search_plan=plan, history=({"role": "user", "content": "RF"},)
        )
        with self.assertRaises(FrozenInstanceError):
            request.top_k = 9
        self.assertEqual(plan, request.search_plan)
        planner = type(
            "FailPlanner",
            (),
            {"plan": lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("planner should not run"))},
        )()
        result = RetrievalPipeline(RecordingIndex(), search_planner=planner).search(request)
        self.assertEqual("passive intermodulation", result["search_plan"]["standalone_query"])

    def test_multi_query_results_are_fused_and_deduplicated(self):
        from knowledge_storm.retrieval_pipeline import RetrievalPipeline, RetrievalRequest
        from knowledge_storm.search_planning import SearchPlan

        class MultiIndex(RecordingIndex):
            def search(self, query, **kwargs):
                self.calls.append(dict(query=query, **kwargs))
                shared = {"chunk_id": "shared", "content": "shared evidence", "metadata": {}}
                return [shared, {"chunk_id": query, "content": query, "metadata": {}}]

        plan = SearchPlan(
            "PIM", "passive intermodulation", subqueries=("RF PIM", "PIM suppression")
        )
        index = MultiIndex()
        result = RetrievalPipeline(index).search(
            RetrievalRequest(query="PIM", search_plan=plan, top_k=3)
        )

        self.assertEqual(
            ["passive intermodulation", "RF PIM", "PIM suppression"],
            [call["query"] for call in index.calls],
        )
        self.assertEqual(1, [item["chunk_id"] for item in result["results"]].count("shared"))
        self.assertEqual(3, result["stages"][1]["input_count"])

    def test_negative_and_metadata_filters_apply_before_final_top_k(self):
        from knowledge_storm.retrieval_pipeline import RetrievalPipeline, RetrievalRequest
        from knowledge_storm.search_planning import SearchPlan

        class FilterIndex(RecordingIndex):
            def search(self, query, **kwargs):
                return [
                    {"chunk_id": "dram", "content": "processing-in-memory DRAM", "metadata": {"year": 2024, "tags": ["memory"]}},
                    {"chunk_id": "old", "content": "passive intermodulation", "metadata": {"year": 2020, "tags": ["rf"]}},
                    {"chunk_id": "gold", "content": "passive intermodulation RF", "metadata": {"year": 2024, "tags": ["rf", "pim"]}},
                ]

        plan = SearchPlan(
            "PIM", "passive intermodulation", negative_terms=("DRAM",), filters={"year": 2024, "tags": "rf"}
        )
        result = RetrievalPipeline(FilterIndex()).search(
            RetrievalRequest(query="PIM", search_plan=plan, top_k=1)
        )
        self.assertEqual(["gold"], [item["chunk_id"] for item in result["results"]])

    def test_parent_expansion_keeps_gold_identifier(self):
        from knowledge_storm.retrieval_pipeline import RetrievalPipeline, RetrievalRequest

        result = RetrievalPipeline(RecordingIndex()).search(
            RetrievalRequest(query="PIM", top_k=1, parent_budget_tokens=64)
        )
        self.assertEqual("chunk-1", result["results"][0]["chunk_id"])
        self.assertEqual("parent evidence", result["results"][0]["parent_context"])
        self.assertEqual("completed", result["stages"][3]["status"])
        self.assertIn("64", result["stages"][3]["reason"])

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
        self.assertEqual("completed", result["stages"][4]["status"])

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
        self.assertIn("search_plan", knowledge_base.retrieval_meta)
        self.assertIn("parent_context", results[0])


if __name__ == "__main__":
    unittest.main()
