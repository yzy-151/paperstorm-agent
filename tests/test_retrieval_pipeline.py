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
    def test_parent_expansion_runs_once_after_gate_for_final_candidates(self):
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
                return [dict(item, parent_context="parent", expanded_content=item["content"] + "\n\nparent") for item in results]

        plan = SearchPlan("PIM", "PIM", subqueries=("RF PIM", "PIM suppression"))
        index = ExpandingIndex()
        def gate(results, _query):
            index.events.append(("gate", len(results)))
            return [item for item in results if item["chunk_id"] == "shared"]

        outcome = RetrievalPipeline(index, relevance_gate=gate).search(
            RetrievalRequest(query="PIM", search_plan=plan, top_k=1, parent_budget_tokens=32)
        )

        self.assertEqual([0, 0, 0], [event[2] for event in index.events if event[0] == "search"])
        expand_events = [event for event in index.events if event[0] == "expand"]
        self.assertEqual([("expand", 1, 32)], expand_events)
        self.assertEqual(["search", "search", "search", "gate", "expand"], [event[0] for event in index.events])
        self.assertEqual("completed", outcome["stages"][4]["status"])
        self.assertEqual(1, outcome["stages"][4]["input_count"])
        self.assertEqual(1, outcome["stages"][4]["output_count"])
        self.assertIn("expanded=1", outcome["stages"][4]["reason"])

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

    def test_parent_expansion_cannot_rescue_candidate_rejected_by_cheap_gate(self):
        from knowledge_storm.retrieval_pipeline import RetrievalPipeline, RetrievalRequest

        class ParentOnlyIndex(RecordingIndex):
            def __init__(self):
                super().__init__()
                self.expansion_inputs = []

            def search(self, query, **kwargs):
                return [{"chunk_id": "child", "content": "local detail"}]

            def expand_parent_context(self, results, parent_budget_tokens):
                self.expansion_inputs.append(list(results))
                return list(results)

        index = ParentOnlyIndex()
        outcome = RetrievalPipeline(index).search(
            RetrievalRequest(
                query="PIM",
                top_k=1,
                parent_budget_tokens=16,
                expected_keywords=("passive intermodulation",),
            )
        )

        self.assertEqual([], outcome["results"])
        self.assertEqual([], index.expansion_inputs)
        parent_stage = next(
            stage for stage in outcome["stages"] if stage["name"] == "parent_expand"
        )
        self.assertEqual("skipped", parent_stage["status"])

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
            ["plan", "retrieve", "fuse", "gate", "parent_expand"],
            [stage["name"] for stage in result["stages"]],
        )
        self.assertEqual("completed", result["stages"][3]["status"])
        self.assertIn("top_k selection", result["stages"][3]["reason"])
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

    def test_prebuilt_search_plan_must_preserve_request_original_query(self):
        from knowledge_storm.retrieval_pipeline import RetrievalPipeline, RetrievalRequest
        from knowledge_storm.search_planning import SearchPlan

        plan = SearchPlan(
            original_query="rewritten query",
            standalone_query="rewritten query",
        )

        with self.assertRaisesRegex(ValueError, "original_query"):
            RetrievalPipeline(RecordingIndex()).search(
                RetrievalRequest(query="user's exact words", search_plan=plan)
            )

    def test_request_query_uses_search_plan_whitespace_normalization(self):
        from knowledge_storm.retrieval_pipeline import RetrievalPipeline, RetrievalRequest

        result = RetrievalPipeline(RecordingIndex()).search(
            RetrievalRequest(query="  PIM   suppression  ")
        )

        self.assertEqual("PIM suppression", result["query"])
        self.assertEqual(
            "PIM suppression", result["search_plan"]["original_query"]
        )

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

    def test_typed_filters_map_fields_and_apply_inclusive_year_range(self):
        from knowledge_storm.retrieval_pipeline import RetrievalPipeline, RetrievalRequest
        from knowledge_storm.search_planning import SearchPlan

        class TypedFilterIndex(RecordingIndex):
            def search(self, query, **kwargs):
                return [
                    {
                        "chunk_id": "too-old",
                        "content": "old evidence",
                        "source_type": "arxiv",
                        "authors": ["Alice Zhang"],
                        "metadata": {"domain": "rf", "published": "2021-06-01"},
                    },
                    {
                        "chunk_id": "gold",
                        "content": "current evidence",
                        "source_type": "arxiv",
                        "authors": ["Alice Zhang", "Bob Smith"],
                        "metadata": {"domain": "rf", "published": "2023-04-02"},
                    },
                    {
                        "chunk_id": "wrong-source",
                        "content": "current evidence",
                        "source_type": "web",
                        "authors": ["Alice Zhang"],
                        "metadata": {"domain": "rf", "year": 2023},
                    },
                    {
                        "chunk_id": "split-range",
                        "content": "conflicting year fields",
                        "source_type": "arxiv",
                        "authors": ["Alice Zhang"],
                        "year": 2025,
                        "metadata": {
                            "domain": "rf",
                            "published": "2021-01-01",
                        },
                    },
                ]

        plan = SearchPlan(
            "q",
            "q",
            filters={
                "year_from": 2022,
                "year_to": 2024,
                "domain": "rf",
                "source": "arxiv",
                "authors": "Alice Zhang",
            },
        )
        result = RetrievalPipeline(TypedFilterIndex()).search(
            RetrievalRequest(query="q", search_plan=plan, top_k=5)
        )

        self.assertEqual(["gold"], [item["chunk_id"] for item in result["results"]])

    def test_all_must_terms_filter_wrong_domain_candidates_using_searchable_text(self):
        from knowledge_storm.retrieval_pipeline import RetrievalPipeline, RetrievalRequest
        from knowledge_storm.search_planning import SearchPlan

        class MustTermIndex(RecordingIndex):
            def search(self, query, **kwargs):
                return [
                    {
                        "chunk_id": "wrong-pim",
                        "title": "Processing-in-memory accelerator",
                        "content": "PIM implemented with DRAM arrays.",
                        "metadata": {"domain": "computer architecture"},
                    },
                    {
                        "chunk_id": "gold",
                        "title": "RF distortion",
                        "content": "Suppression method for nonlinear junctions.",
                        "metadata": {
                            "domain": "passive intermodulation",
                            "band": "radio frequency",
                        },
                    },
                    {
                        "chunk_id": "partial",
                        "title": "Passive intermodulation overview",
                        "content": "Cable distortion.",
                        "metadata": {},
                    },
                ]

        plan = SearchPlan(
            "PIM",
            "passive intermodulation RF",
            must_terms=("passive intermodulation", "radio frequency"),
        )
        result = RetrievalPipeline(MustTermIndex()).search(
            RetrievalRequest(query="PIM", search_plan=plan, top_k=3)
        )

        self.assertEqual(["gold"], [item["chunk_id"] for item in result["results"]])

    def test_three_planned_queries_rerank_fused_candidates_once_and_rewrite_scores(self):
        from knowledge_storm.retrieval_pipeline import RetrievalPipeline, RetrievalRequest
        from knowledge_storm.search_planning import SearchPlan

        class CheapIndex(RecordingIndex):
            def search(self, query, **kwargs):
                self.calls.append(dict(query=query, **kwargs))
                return [
                    {"chunk_id": "shared", "content": "shared", "metadata": {}},
                    {"chunk_id": query, "content": query, "metadata": {}},
                ]

        class OneShotReranker:
            model_name = "test-cross-encoder"

            def __init__(self):
                self.calls = []

            def rerank(self, query, candidates, top_k=None):
                self.calls.append((query, [item["chunk_id"] for item in candidates], top_k))
                output = [dict(item) for item in candidates]
                for item in output:
                    item["rerank_score"] = 0.9 if item["chunk_id"] == "q-2" else 0.2
                return sorted(output, key=lambda item: -item["rerank_score"])

        plan = SearchPlan("raw", "q-0", subqueries=("q-1", "q-2"))
        index = CheapIndex()
        reranker = OneShotReranker()
        result = RetrievalPipeline(index, reranker=reranker).search(
            RetrievalRequest(
                query="raw",
                search_plan=plan,
                mode="hybrid_rerank",
                top_k=2,
                candidate_k=10,
            )
        )

        self.assertEqual(3, len(index.calls))
        self.assertTrue(all(call["mode"] == "hybrid" for call in index.calls))
        self.assertTrue(all(call["reranker"] is None for call in index.calls))
        self.assertEqual(1, len(reranker.calls))
        self.assertEqual("q-0", reranker.calls[0][0])
        self.assertEqual("q-2", result["results"][0]["chunk_id"])
        self.assertEqual([1, 2], [item["final_rank"] for item in result["results"]])
        self.assertEqual([0.9, 0.2], [item["final_score"] for item in result["results"]])
        self.assertEqual([0.9, 0.2], [item["score"] for item in result["results"]])
        self.assertEqual(
            ["plan", "retrieve", "fuse", "rerank", "gate", "parent_expand"],
            [stage["name"] for stage in result["stages"]],
        )
        rerank_stage = result["stages"][3]
        self.assertEqual(len(reranker.calls[0][1]), rerank_stage["input_count"])
        self.assertEqual(len(reranker.calls[0][1]), rerank_stage["output_count"])

    def test_callable_reranker_score_is_normalized_and_controls_final_ranking(self):
        from knowledge_storm.retrieval_pipeline import RetrievalPipeline, RetrievalRequest
        from knowledge_storm.search_planning import SearchPlan

        class TwoCandidateIndex(RecordingIndex):
            def search(self, query, **kwargs):
                return [
                    {"chunk_id": "low", "content": "low", "score": 0.99},
                    {"chunk_id": "high", "content": "high", "score": 0.01},
                ]

        calls = []

        def callable_reranker(query, candidates):
            calls.append((query, len(candidates)))
            output = [dict(item) for item in candidates]
            for item in output:
                item["score"] = 0.9 if item["chunk_id"] == "high" else 0.2
            return output

        result = RetrievalPipeline(
            TwoCandidateIndex(), reranker=callable_reranker
        ).search(
            RetrievalRequest(
                query="raw",
                search_plan=SearchPlan("raw", "standalone"),
                mode="hybrid_rerank",
                top_k=2,
            )
        )

        self.assertEqual([("standalone", 2)], calls)
        self.assertEqual(["high", "low"], [item["chunk_id"] for item in result["results"]])
        self.assertEqual([0.9, 0.2], [item["rerank_score"] for item in result["results"]])
        self.assertEqual([0.9, 0.2], [item["final_score"] for item in result["results"]])
        self.assertEqual([0.9, 0.2], [item["score"] for item in result["results"]])
        self.assertEqual(
            ["hybrid_rerank", "hybrid_rerank"],
            [item["retrieval_mode"] for item in result["results"]],
        )

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
        self.assertTrue(expanded[0]["expanded_content"].startswith("child evidence"))
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

    def test_parent_expansion_deduplicates_siblings_and_uses_one_global_budget(self):
        from knowledge_storm.retrieval import HashEmbeddingProvider, HybridPaperIndex

        children = [
            {"chunk_id": "child-1", "parent_id": "parent-1", "content": "gold child fact"},
            {"chunk_id": "child-2", "parent_id": "parent-1", "content": "sibling detail"},
        ]
        parent = {
            "node_id": "parent-1",
            "content": "gold child fact alpha beta gamma delta epsilon zeta",
        }
        index = HybridPaperIndex(
            children, embedding_provider=HashEmbeddingProvider(), parents=[parent]
        )

        expanded = index.expand_parent_context(children, 4)

        self.assertTrue(expanded[0]["expanded_content"].startswith("gold child fact"))
        self.assertNotIn("gold child fact", expanded[0]["parent_context"])
        self.assertEqual("", expanded[1]["parent_context"])
        self.assertEqual("sibling detail", expanded[1]["expanded_content"])
        total_parent_words = sum(
            len(item["parent_context"].split()) for item in expanded
        )
        self.assertLessEqual(total_parent_words, 4)

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
        gate_stage = next(stage for stage in result["stages"] if stage["name"] == "gate")
        self.assertEqual("completed", gate_stage["status"])

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

    def test_qa_prompt_keeps_child_gold_fact_before_long_parent_context(self):
        from knowledge_storm.paperstorm_qa import _kb_answer_prompt

        prompt = _kb_answer_prompt(
            "关键事实是什么？",
            [
                {
                    "chunk_id": "gold",
                    "title": "Evidence",
                    "content": "CHILD_GOLD_FACT",
                    "parent_context": "parent filler " * 300,
                    "expanded_content": ("parent filler " * 300) + "CHILD_GOLD_FACT",
                }
            ],
        )

        self.assertIn("CHILD_GOLD_FACT", prompt)


if __name__ == "__main__":
    unittest.main()
