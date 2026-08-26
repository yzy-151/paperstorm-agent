import json
import unittest
from pathlib import Path


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "pim_badcases.json"


def valid_plan_payload(query="graph rag"):
    return {
        "original_query": query,
        "standalone_query": query,
        "domain": "information-retrieval",
        "entities": ["Graph RAG"],
        "must_terms": ["graph retrieval"],
        "negative_terms": [],
        "filters": {"year_from": 2023},
        "subqueries": [query],
        "answer_type": "survey",
    }


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

    def test_current_explicit_domain_overrides_followup_history(self):
        from knowledge_storm.search_planning import SearchPlanner

        history = [
            {
                "role": "user",
                "content": "processing-in-memory accelerator for DRAM",
            },
            {"role": "assistant", "content": "正在讨论存内计算。"},
        ]

        plan = SearchPlanner().plan("这种PIM无源互调抑制方法", history=history)

        self.assertEqual(plan.domain, "rf-passive-intermodulation")
        self.assertEqual(plan.standalone_query, "这种PIM无源互调抑制方法")

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

    def test_followup_history_may_include_current_turn_and_explicit_assistant_answer(self):
        from knowledge_storm.search_planning import SearchPlanner

        query = "那它为什么不是 DRAM？"
        history = [
            {"role": "user", "content": "PIM 是什么？"},
            {
                "role": "assistant",
                "content": "这里 PIM 指 passive intermodulation，是 RF 无源互调。",
            },
            {"role": "user", "content": query},
        ]

        plan = SearchPlanner().plan(query, history=history)

        self.assertEqual("rf-passive-intermodulation", plan.domain)
        self.assertEqual(query, plan.original_query)
        self.assertIn("passive intermodulation", plan.standalone_query.lower())

    def test_negated_processing_in_memory_meaning_is_rf_disambiguation(self):
        from knowledge_storm.search_planning import SearchPlanner

        plan = SearchPlanner().plan(
            "这里为什么不能把 PIM 理解成 DRAM processing-in-memory？"
        )

        self.assertEqual("rf-passive-intermodulation", plan.domain)
        self.assertEqual(("passive intermodulation",), plan.must_terms)

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

    def test_llm_adapter_accepts_supported_response_shapes(self):
        from knowledge_storm.search_planning import SearchPlanner

        payload = json.dumps(valid_plan_payload())

        class Message:
            def __init__(self, content):
                self.content = content

        class Choice:
            def __init__(self, content):
                self.message = Message(content)

        class Completion:
            def __init__(self, content):
                self.choices = [Choice(content)]

        responses = (
            payload,
            [payload],
            {"choices": [{"message": {"content": payload}}]},
            Completion(payload),
            Message(payload),
        )
        for response in responses:
            with self.subTest(response_type=type(response).__name__):
                plan = SearchPlanner(llm=lambda _prompt, value=response: value).plan(
                    "graph rag"
                )
                self.assertEqual(plan.domain, "information-retrieval")

    def test_llm_adapter_rejects_empty_or_ambiguous_output_lists(self):
        from knowledge_storm.search_planning import PlanningError, SearchPlanner

        payload = json.dumps(valid_plan_payload())
        for response in ([], [payload, payload]):
            with self.subTest(response=response):
                calls = []

                def llm(_prompt, value=response):
                    calls.append(1)
                    return value

                with self.assertRaises(PlanningError) as captured:
                    SearchPlanner(llm=llm).plan("graph rag")
                self.assertEqual(len(calls), 2)
                self.assertEqual(
                    captured.exception.error_type, "invalid_structured_output"
                )

    def test_provider_errors_are_typed_not_retried_and_preserve_cause(self):
        from knowledge_storm.search_planning import PlanningError, SearchPlanner

        class RateLimitError(RuntimeError):
            status_code = 429

        class AuthenticationError(RuntimeError):
            status_code = 401

        cases = (
            (TimeoutError("slow"), "provider_timeout"),
            (RateLimitError("limited"), "provider_rate_limited"),
            (AuthenticationError("denied"), "provider_auth_error"),
            (RuntimeError("offline"), "provider_error"),
        )
        for provider_error, expected_type in cases:
            with self.subTest(expected_type=expected_type):
                calls = []

                def llm(_prompt, error=provider_error):
                    calls.append(1)
                    raise error

                with self.assertRaises(PlanningError) as captured:
                    SearchPlanner(llm=llm).plan("graph rag")
                self.assertEqual(len(calls), 1)
                self.assertEqual(captured.exception.error_type, expected_type)
                self.assertIs(captured.exception.__cause__, provider_error)

    def test_filters_are_not_shared_and_must_be_json_safe(self):
        from knowledge_storm.search_planning import SearchPlan

        first = SearchPlan(original_query="a", standalone_query="a")
        second = SearchPlan(original_query="b", standalone_query="b")
        self.assertIsNot(first.filters, second.filters)
        self.assertEqual(dict(second.filters), {})
        self.assertEqual(
            SearchPlan.from_mapping(first.to_dict()).to_dict(), first.to_dict()
        )
        with self.assertRaises((TypeError, ValueError)):
            SearchPlan(
                original_query="a",
                standalone_query="a",
                filters={"bad": object()},
            )

    def test_filters_accept_only_json_scalars_or_scalar_lists(self):
        from knowledge_storm.search_planning import SearchPlan

        plan = SearchPlan(
            original_query="a",
            standalone_query="a",
            filters={
                "year": 2024,
                "source": "arxiv",
                "venue": None,
                "tags": ["rf", "pim"],
            },
        )

        with self.assertRaises(TypeError):
            plan.filters["new"] = "value"
        self.assertEqual(plan.filters["tags"], ("rf", "pim"))
        self.assertEqual(plan.to_dict()["filters"]["tags"], ["rf", "pim"])

        invalid_filters = (
            {"metadata": {"year": 2024}},
            {"ranges": [{"gte": 2020}]},
            {"tags": [["rf"]]},
            {"tags": {"rf", "pim"}},
        )
        for filters in invalid_filters:
            with self.subTest(filters=filters):
                with self.assertRaises((TypeError, ValueError)):
                    SearchPlan(
                        original_query="a",
                        standalone_query="a",
                        filters=filters,
                    )

    def test_filters_reject_unknown_keys_and_wrong_typed_values(self):
        from knowledge_storm.search_planning import SearchPlan

        with self.assertRaisesRegex(ValueError, "unknown filter"):
            SearchPlan("q", "q", filters={"unexecutable_field": "value"})
        with self.assertRaisesRegex((TypeError, ValueError), "year_from"):
            SearchPlan("q", "q", filters={"year_from": "recent"})

    def test_history_budget_keeps_recent_messages_and_marks_them_untrusted(self):
        from knowledge_storm.search_planning import (
            DEFAULT_HISTORY_MAX_CHARS_PER_MESSAGE,
            DEFAULT_HISTORY_MAX_MESSAGES,
            DEFAULT_HISTORY_MAX_TOTAL_CHARS,
            SearchPlanner,
        )

        self.assertGreater(DEFAULT_HISTORY_MAX_MESSAGES, 0)
        self.assertGreater(DEFAULT_HISTORY_MAX_CHARS_PER_MESSAGE, 0)
        self.assertGreater(DEFAULT_HISTORY_MAX_TOTAL_CHARS, 0)
        prompts = []

        def llm(prompt):
            prompts.append(prompt)
            return json.dumps(valid_plan_payload("current query"))

        SearchPlanner(
            llm=llm,
            history_max_messages=2,
            history_max_chars_per_message=12,
            history_max_total_chars=18,
        ).plan(
            "current query",
            history=[
                {"role": "user", "content": "PIM 神经网络抑制 OLD_TOPIC"},
                {"role": "assistant", "content": "OLD_ASSISTANT"},
                {"role": "user", "content": "RECENT_USER_123456"},
                {"role": "assistant", "content": "RECENT_ASSISTANT_123456"},
            ],
        )

        prompt = prompts[0]
        self.assertNotIn("OLD_TOPIC", prompt)
        self.assertNotIn("OLD_ASSISTANT", prompt)
        self.assertIn("RECENT", prompt)
        self.assertIn("untrusted data", prompt.lower())

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
