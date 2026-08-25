import json
import unittest
from pathlib import Path


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "pim_badcases.json"


class SearchPlanTest(unittest.TestCase):
    def test_pim_neural_suppression_is_disambiguated_as_rf(self):
        from knowledge_storm.search_planning import SearchPlanner

        plan = SearchPlanner().plan("  PIM   神经网络抑制  ")

        self.assertEqual(plan.original_query, "PIM 神经网络抑制")
        self.assertEqual(plan.domain, "rf-passive-intermodulation")
        self.assertIn("passive intermodulation", plan.must_terms)
        self.assertIn("dram", plan.negative_terms)
        self.assertIn("processing-in-memory", plan.negative_terms)
        self.assertLessEqual(len(plan.subqueries), 3)
        self.assertEqual(len(plan.subqueries), len(set(plan.subqueries)))
        self.assertTrue(all(item.strip() for item in plan.subqueries))

    def test_explicit_processing_in_memory_is_not_misclassified_as_rf(self):
        from knowledge_storm.search_planning import SearchPlanner

        plan = SearchPlanner().plan(
            "PIM processing-in-memory accelerator for DRAM inference"
        )

        self.assertEqual(plan.domain, "processing-in-memory")
        self.assertNotIn("passive intermodulation", plan.must_terms)

    def test_followup_rewrite_uses_explicit_history_and_keeps_original(self):
        from knowledge_storm.search_planning import SearchPlanner

        history = [
            {"role": "user", "content": "PIM 神经网络抑制"},
            {"role": "assistant", "content": "这里的 PIM 是射频无源互调。"},
        ]
        plan = SearchPlanner().plan("它有哪些抑制方法", history=history)

        self.assertEqual(plan.original_query, "它有哪些抑制方法")
        self.assertIn("passive intermodulation", plan.standalone_query.lower())
        self.assertEqual(plan.domain, "rf-passive-intermodulation")

    def test_followup_without_reliable_antecedent_does_not_invent_one(self):
        from knowledge_storm.search_planning import SearchPlanner

        plan = SearchPlanner().plan(
            "它有哪些抑制方法",
            history=[{"role": "assistant", "content": "我可以帮你。"}],
        )

        self.assertEqual(plan.standalone_query, "它有哪些抑制方法")
        self.assertEqual(plan.domain, "")

    def test_zero_pronoun_followups_use_adjacent_explicit_user_topic(self):
        from knowledge_storm.search_planning import SearchPlanner

        history = [
            {"role": "user", "content": "PIM 神经网络抑制"},
            {"role": "assistant", "content": "这里的 PIM 是射频无源互调。"},
        ]

        for query in ("有哪些抑制方法", "如何降低", "有哪些危害", "有哪些原因"):
            with self.subTest(query=query):
                plan = SearchPlanner().plan(query, history=history)
                self.assertEqual(plan.domain, "rf-passive-intermodulation")
                self.assertIn(
                    "passive intermodulation", plan.standalone_query.lower()
                )

    def test_new_user_topic_blocks_stale_domain_inheritance(self):
        from knowledge_storm.search_planning import SearchPlanner

        history = [
            {"role": "user", "content": "PIM 神经网络抑制"},
            {"role": "assistant", "content": "这里的 PIM 是射频无源互调。"},
            {"role": "user", "content": "Python 异步编程"},
            {"role": "assistant", "content": "可以使用 asyncio。"},
        ]

        plan = SearchPlanner().plan("它有哪些用途", history=history)

        self.assertEqual(plan.domain, "")
        self.assertEqual(plan.standalone_query, "它有哪些用途")

    def test_explicit_new_entity_is_not_treated_as_zero_pronoun_followup(self):
        from knowledge_storm.search_planning import SearchPlanner

        history = [
            {"role": "user", "content": "PIM 神经网络抑制"},
            {"role": "assistant", "content": "这里的 PIM 是射频无源互调。"},
        ]

        plan = SearchPlanner().plan("如何优化 Python 异步编程", history=history)

        self.assertEqual(plan.domain, "")
        self.assertEqual(plan.standalone_query, "如何优化 Python 异步编程")

    def test_llm_invalid_json_then_valid_json_retries_once(self):
        from knowledge_storm.search_planning import SearchPlanner

        responses = iter(
            [
                "not-json",
                json.dumps(
                    {
                        "original_query": "graph rag",
                        "standalone_query": "graph RAG retrieval methods",
                        "domain": "information-retrieval",
                        "entities": ["Graph RAG"],
                        "must_terms": ["graph retrieval"],
                        "negative_terms": [],
                        "filters": {"year_from": 2023},
                        "subqueries": ["graph RAG retrieval", "graph RAG indexing"],
                        "answer_type": "survey",
                    }
                ),
            ]
        )
        calls = []

        def llm(prompt):
            calls.append(prompt)
            return next(responses)

        plan = SearchPlanner(llm=llm).plan("graph rag")

        self.assertEqual(len(calls), 2)
        self.assertEqual(plan.domain, "information-retrieval")
        self.assertEqual(plan.answer_type, "survey")

    def test_llm_two_invalid_outputs_raise_typed_planning_error(self):
        from knowledge_storm.search_planning import PlanningError, SearchPlanner

        calls = []

        def llm(prompt):
            calls.append(prompt)
            return "still not json"

        with self.assertRaises(PlanningError) as captured:
            SearchPlanner(llm=llm).plan("graph rag")

        self.assertEqual(len(calls), 2)
        self.assertEqual(captured.exception.error_type, "invalid_structured_output")

    def test_llm_schema_invalid_twice_raises_planning_error(self):
        from knowledge_storm.search_planning import PlanningError, SearchPlanner

        invalid = json.dumps(
            {
                "original_query": "query",
                "standalone_query": "query",
                "subqueries": ["one", "two", "three", "four"],
                "answer_type": "invented-type",
            }
        )

        with self.assertRaises(PlanningError) as captured:
            SearchPlanner(llm=lambda _prompt: invalid).plan("query")

        self.assertEqual(captured.exception.error_type, "invalid_structured_output")

    def test_filters_are_not_shared_and_must_be_json_safe(self):
        from knowledge_storm.search_planning import SearchPlan

        first = SearchPlan(original_query="a", standalone_query="a")
        second = SearchPlan(original_query="b", standalone_query="b")
        first.filters["year"] = 2024

        self.assertEqual(second.filters, {})
        self.assertEqual(
            SearchPlan.from_mapping(first.to_dict()).to_dict(), first.to_dict()
        )
        with self.assertRaises((TypeError, ValueError)):
            SearchPlan(
                original_query="a",
                standalone_query="a",
                filters={"bad": object()},
            )

    def test_fixture_cases_capture_required_pim_badcases(self):
        from knowledge_storm.search_planning import SearchPlanner

        cases = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(cases), 5)
        for case in cases:
            with self.subTest(case=case["case_id"]):
                plan = SearchPlanner().plan(
                    case["query"], history=case.get("history")
                )
                self.assertEqual(plan.domain, case["expected_domain"])
                if case.get("expected_must_term"):
                    self.assertIn(case["expected_must_term"], plan.must_terms)
                for term in case.get("expected_negative_terms", []):
                    self.assertIn(term, plan.negative_terms)
                if case.get("standalone_contains"):
                    self.assertIn(
                        case["standalone_contains"], plan.standalone_query.lower()
                    )
                if case.get("expected_standalone"):
                    self.assertEqual(
                        plan.standalone_query, case["expected_standalone"]
                    )


if __name__ == "__main__":
    unittest.main()
