import tempfile
import unittest
from pathlib import Path
from unittest import mock


class PaperStormRouterLLMTest(unittest.TestCase):
    def test_router_tool_schema_constrains_memory_and_response_contract(self):
        from knowledge_storm.paperstorm_router_llm import _router_function_tool

        parameters = _router_function_tool()["function"]["parameters"]
        tool_call = parameters["properties"]["tool_calls"]["items"]
        arguments = tool_call["properties"]["arguments"]
        contract = parameters["properties"]["response_contract"]

        self.assertEqual(
            arguments["properties"]["memory_type"]["enum"],
            ["semantic", "episodic", "procedural", "preference"],
        )
        self.assertIn("requires_citations", contract["properties"])
        self.assertIn("requires_citations", contract["required"])

    def test_router_completion_uses_function_arguments_as_structured_content(self):
        from knowledge_storm.paperstorm_router_llm import _completion_result

        response = {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "submit_turn_plan",
                                    "arguments": '{"action":"respond","confidence":0.9}',
                                }
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        result = _completion_result(response, "openai/deepseek-v4-flash")

        self.assertIn('"action":"respond"', result["content"])
        self.assertEqual(result["structured_output"], "function_call")

    def test_router_output_budget_is_large_enough_for_complete_json_and_bounded(self):
        from knowledge_storm.paperstorm_router_llm import _router_output_tokens

        with mock.patch.dict("os.environ", {}, clear=False):
            self.assertEqual(2048, _router_output_tokens())
        with mock.patch.dict(
            "os.environ", {"PAPERSTORM_ROUTER_MAX_TOKENS": "256"}
        ):
            self.assertEqual(768, _router_output_tokens())
        with mock.patch.dict(
            "os.environ", {"PAPERSTORM_ROUTER_MAX_TOKENS": "99999"}
        ):
            self.assertEqual(8192, _router_output_tokens())

    def test_rewrite_query_preserves_original_without_calling_search_planner(self):
        from knowledge_storm.paperstorm_intent_router import rewrite_query
        from knowledge_storm.search_planning import SearchPlanner

        history = [
            {"role": "user", "content": "无源互调怎么抑制"},
            {"role": "assistant", "content": "可以做数字抵消"},
            {"role": "user", "content": "Python 的装饰器是什么"},
        ]
        with mock.patch.object(
            SearchPlanner,
            "plan",
            side_effect=AssertionError("router must not plan retrieval"),
        ):
            self.assertEqual(
                "它有哪些用途",
                rewrite_query("它有哪些用途", {"topic": "PIM"}, history),
            )

    def test_dynamic_output_budget_scales_with_response_contract(self):
        from knowledge_storm.paperstorm_router_llm import select_output_budget

        self.assertEqual(select_output_budget("你好", {}), 2048)
        self.assertEqual(
            select_output_budget(
                "继续写下去", {"continue_previous": True, "task": "续写长篇故事"}
            ),
            16384,
        )
        self.assertGreaterEqual(
            select_output_budget("请写一篇一万字的报告", {"task": "长篇报告"}),
            20000,
        )
        self.assertLessEqual(
            select_output_budget("请写一篇十万字小说", {"task": "超长小说"}),
            65536,
        )

    def test_length_finish_reason_continues_once_and_aggregates_telemetry(self):
        from knowledge_storm.paperstorm_router_llm import complete_chat_with_telemetry

        calls = []

        def completion(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return {
                    "choices": [{"finish_reason": "length", "message": {"content": "第一段"}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                }
            return {
                "choices": [{"finish_reason": "stop", "message": {"content": "第二段"}}],
                "usage": {"prompt_tokens": 40, "completion_tokens": 10, "total_tokens": 50},
            }

        result = complete_chat_with_telemetry(
            completion=completion,
            model="openai/deepseek-v4-flash",
            prompt="请续写",
            api_key="test",
            api_base="https://example.invalid",
            output_budget=16384,
            timeout=25,
        )

        self.assertEqual(result["content"], "第一段第二段")
        self.assertEqual(result["segments"], 2)
        self.assertFalse(result["truncated"])
        self.assertEqual(result["usage"]["total_tokens"], 170)
        self.assertIn("不要重复开头", calls[1]["messages"][0]["content"])

    def test_provider_failure_returns_typed_error_instead_of_greeting(self):
        from knowledge_storm.paperstorm_router_llm import complete_chat_with_telemetry

        def completion(**_kwargs):
            raise TimeoutError("provider timed out")

        result = complete_chat_with_telemetry(
            completion=completion,
            model="openai/deepseek-v4-flash",
            prompt="继续故事",
            api_key="test",
            api_base="https://example.invalid",
            output_budget=16384,
            timeout=25,
        )

        self.assertEqual(result["error"]["type"], "timeout")
        self.assertEqual(result["content"], "")
        self.assertNotIn("你好", result["content"])

    def make_service(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        from knowledge_storm.paperstorm_service import PaperStormTaskService

        return PaperStormTaskService(root_dir=Path(temp_dir.name))

    def test_fake_mode_router_is_rule_based(self):
        from knowledge_storm.paperstorm_router_llm import build_intent_router

        router = build_intent_router(run_mode="fake")
        self.assertIsNone(router.llm_router)
        decision = router.route(message="你好", session={}, context_window=[])
        self.assertEqual(decision["router"], "rule_fallback")

    def test_injected_llm_router_reaches_graph_result(self):
        from knowledge_storm.paperstorm_intent_router import PaperStormIntentRouter
        from knowledge_storm.control_plane import ProductionRuntime

        service = self.make_service()

        def fake_llm(_prompt):
            return (
                '{"intent":"system_help","need_retrieval":false,"tool":"chat_fallback",'
                '"rewritten_query":"你是什么模型？","confidence":0.96,"reason":"identity"}'
            )

        runtime = ProductionRuntime(
            root_dir=Path(service.root_dir) / "production_runtime_test",
            task_service=service,
            intent_router=PaperStormIntentRouter(llm_router=fake_llm),
        )
        result = runtime.invoke(
            tenant_id="local",
            thread_id="thread-llm-1",
            request_id="req-1",
            user_id="local-user",
            message="你是什么模型？",
            topic="pim 神经网络抑制",
            run_mode="fake",
            retriever="arxiv",
            output_language="zh",
            expected_keywords=[],
            forbidden_keywords=[],
            context_window=[],
            source_message_id="src-1",
        )
        self.assertEqual(result["router_decision"]["router"], "llm_planner")
        self.assertEqual(result["route"], "casual_chat")
        self.assertFalse(result["retrieval_triggered"])

    def test_paperstorm_mode_builds_llm_router_when_key_present(self):
        from knowledge_storm.paperstorm_router_llm import build_intent_router

        router = build_intent_router(run_mode="paperstorm")
        if router.llm_router is None:
            self.skipTest("no router API key configured in this environment")
        self.assertIsNotNone(router.llm_router)

    def test_chat_llm_builder_respects_enable_flag(self):
        from knowledge_storm.paperstorm_router_llm import build_chat_llm_callable

        self.assertIsNone(build_chat_llm_callable(enabled=False))
        callable_result = build_chat_llm_callable(enabled=True)
        if callable_result is None:
            self.skipTest("no chat LLM provider key configured in this environment")
        self.assertTrue(callable(callable_result))

    def test_judge_llm_builder_respects_enable_flag(self):
        from knowledge_storm.paperstorm_router_llm import build_judge_llm_callable

        self.assertIsNone(build_judge_llm_callable(enabled=False))
        callable_result = build_judge_llm_callable(enabled=True)
        if callable_result is None:
            self.skipTest("no judge LLM provider key configured in this environment")
        self.assertTrue(callable(callable_result))


if __name__ == "__main__":
    unittest.main()
