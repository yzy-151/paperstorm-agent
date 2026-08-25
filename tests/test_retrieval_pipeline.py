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
    def test_parent_expansion_runs_once_after_fusion(self):
        from knowledge_storm.retrieval_pipeline import RetrievalPipeline, RetrievalRequest
        from knowledge_storm.search_planning import SearchPlan

        class ExpandingIndex(RecordingIndex):
            def __init__(self):
                super().__init__()
                self.events = []

            def search(self, query, **kwargs):
                self.events.append(("search", query, kwargs["parent_budget_tokens"]))
                return [
                    {"chunk_id": "shared", "content": "child", "parent_id": "parent-1"},
                    {"chunk_id": query, "content": query},
                ]

            def expand_parent_context(self, results, parent_budget_tokens):
                self.events.append(("expand", len(results), parent_budget_tokens))
                return [dict(item, parent_context="parent", expanded_content="parent\n\n" + item["content"]) for item in results]

        plan = SearchPlan("PIM", "PIM", subqueries=("RF PIM", "PIM suppression"))
        index = ExpandingIndex()
        def gate(results, _query):
            index.events.append(("gate", len(results)))
            return results

        outcome = RetrievalPipeline(index, relevance_gate=gate).search(
            RetrievalRequest(query="PIM", search_plan=plan, top_k=3, parent_budget_tokens=32)
        )

        self.assertEqual([0, 0, 0], [event[2] for event in index.events if event[0] == "search"])
        expand_events = [event for event in index.events if event[0] == "expand"]
        self.assertEqual([("expand", 4, 32)], expand_events)
        self.assertEqual(["search", "search", "search", "expand", "gate"], [event[0] for event in index.events])
        self.assertEqual("completed", outcome["stages"][3]["status"])
        self.assertEqual(4, outcome["stages"][3]["input_count"])
        self.assertEqual(4, outcome["stages"][3]["output_count"])
        self.assertIn("expanded=4", outcome["stages"][3]["reason"])

    def test_parent_budget_requires_expansion_capability(self):
        from knowledge_storm.retrieval_pipeline import (
            RetrievalCapabilityError,
            RetrievalPipeline,
            RetrievalRequest,
        )

        with self.assertRaisesRegex(RetrievalCapabilityError, "parent context"):
            RetrievalPipeline(RecordingIndex()).search(
                RetrievalRequest(query="PIM", parent_budget_tokens=8)
            )

    def test_gate_can_use_evidence_added_by_parent_expansion(self):
        from knowledge_storm.retrieval_pipeline import RetrievalPipeline, RetrievalRequest

        class ParentOnlyIndex(RecordingIndex):
            def search(self, query, **kwargs):
                return [{"chunk_id": "child", "content": "local detail"}]

            def expand_parent_context(self, results, parent_budget_tokens):
                return [
                    dict(
                        results[0],
                        parent_context="passive intermodulation system context",
                        expanded_content="passive intermodulation system context\n\nlocal detail",
                    )
                ]

        outcome = RetrievalPipeline(ParentOnlyIndex()).search(
            RetrievalRequest(
                query="PIM",
                top_k=1,
                parent_budget_tokens=16,
                expected_keywords=("passive intermodulation",),
            )
        )

        self.assertEqual(["child"], [item["chunk_id"] for item in outcome["results"]])

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

    def test_hybrid_index_parent_expansion_is_deep_copied_and_keeps_rank(self):
        from knowledge_storm.retrieval import HashEmbeddingProvider, HybridPaperIndex

        child = {
            "chunk_id": "gold-child",
            "parent_id": "parent-1",
            "content": "child evidence",
            "metadata": {"nested": {"value": 1}},
            "final_rank": 2,
        }
        parent = {"node_id": "parent-1", "content": "parent-only-fact " * 20}
        index = HybridPaperIndex(
            [child], embedding_provider=HashEmbeddingProvider(), parents=[parent]
        )
        source = [dict(child)]
        expanded = index.expand_parent_context(source, 8)

        self.assertEqual("gold-child", expanded[0]["chunk_id"])
        self.assertEqual(2, expanded[0]["final_rank"])
        self.assertIn("parent-only-fact", expanded[0]["parent_context"])
        self.assertLessEqual(len(expanded[0]["parent_context"]), 32)
        expanded[0]["metadata"]["nested"]["value"] = 9
        self.assertEqual(1, child["metadata"]["nested"]["value"])

    def test_hybrid_index_parent_expansion_validates_budget_and_handles_missing_parent(self):
        from knowledge_storm.retrieval import HashEmbeddingProvider, HybridPaperIndex

        child = {
            "chunk_id": "orphan",
            "parent_id": "missing",
            "content": "child evidence",
            "metadata": {"nested": {"value": 1}},
        }
        index = HybridPaperIndex(
            [child], embedding_provider=HashEmbeddingProvider(), parents=[]
        )

        with self.assertRaisesRegex(ValueError, "parent_budget_tokens"):
            index.expand_parent_context([child], -1)

        not_expanded = index.expand_parent_context([child], 0)
        self.assertEqual([child], not_expanded)
        self.assertIsNot(child, not_expanded[0])
        self.assertIsNot(child["metadata"], not_expanded[0]["metadata"])

        orphan = index.expand_parent_context([child], 8)
        self.assertEqual("", orphan[0]["parent_context"])
        self.assertEqual("child evidence", orphan[0]["expanded_content"])

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

    def test_qa_uses_expanded_parent_content_but_cites_child(self):
        import json

        from knowledge_storm.paperstorm_qa import PaperStormKnowledgeBase
        from knowledge_storm.retrieval_pipeline import RetrievalPipeline

        class ParentIndex(RecordingIndex):
            def search(self, query, **kwargs):
                return [{"chunk_id": "child-gold", "content": "child has no answer", "parent_id": "parent-1", "metadata": {}}]

            def expand_parent_context(self, results, parent_budget_tokens):
                return [dict(results[0], parent_context="关键事实是 parent-only-fact。", expanded_content="关键事实是 parent-only-fact。\n\nchild has no answer")]

        captured = []
        kb = PaperStormKnowledgeBase([], retrieval_pipeline=RetrievalPipeline(ParentIndex()))
        answer = kb.answer_question(
            "关键事实是什么？",
            top_k=1,
            answer_generator=lambda prompt: captured.append(prompt) or "parent-only-fact[1]",
            retrieval_options={"parent_budget_tokens": 32},
        )

        self.assertIn("parent-only-fact", captured[0])
        self.assertEqual("child-gold", answer["citations"][0]["chunk_id"])
        self.assertIn("search_plan", answer["retrieval_metadata"])
        self.assertIn("stages", answer["retrieval_metadata"])
        self.assertIn("models", answer["retrieval_metadata"])
        self.assertIn("mode", answer["retrieval_metadata"])
        self.assertEqual("关键事实是什么？", answer["retrieval_metadata"]["query"])
        json.dumps(answer["retrieval_metadata"], allow_nan=False)

        composed = kb.answer_question(
            "关键事实是什么？",
            top_k=1,
            retrieval_options={"parent_budget_tokens": 32},
        )
        self.assertIn("parent-only-fact", composed["answer"])
        self.assertEqual("child-gold", composed["citations"][0]["chunk_id"])


if __name__ == "__main__":
    unittest.main()
