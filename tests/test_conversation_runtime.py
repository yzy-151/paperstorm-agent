import json
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock


class FlakyDeepResearchTool:
    name = "storm_deep_research"

    def __init__(self):
        self.calls = 0

    def run(self, arguments):
        self.calls += 1
        if self.calls == 1:
            raise ConnectionError("temporary retrieval outage")
        return {
            "answer": "PIM 是 passive intermodulation。[1]",
            "citations": [{"id": 1, "url": "https://example.com/pim"}],
            "evidence": [{"content": "passive intermodulation RF"}],
            "grounded": True,
            "task_id": "retry-task",
            "artifact_uri": "artifact://retry-task/research",
            "evidence_sufficiency": {"sufficient": True, "score": 90},
            "retrieval_triggered": True,
        }


@mock.patch.dict(
    os.environ,
    {
        "PAPERSTORM_RETRIEVAL_EMBEDDING": "hash",
        "PAPERSTORM_CHAT_LLM": "0",
        "PAPERSTORM_JUDGE_LLM": "0",
    },
)
class PaperStormLangGraphV44Test(unittest.TestCase):
    def test_structured_tool_query_is_used_for_contextual_evidence_followup(self):
        from knowledge_storm.conversation_runtime import _tool_query

        decision = {
            "tool_calls": [
                {
                    "name": "evidence.search",
                    "arguments": {
                        "query": "wavelet neural network versus feedforward network"
                    },
                }
            ]
        }

        self.assertEqual(
            _tool_query(decision, "evidence.search"),
            "wavelet neural network versus feedforward network",
        )

    def test_citation_contract_binds_existing_evidence_without_phrase_rules(self):
        from knowledge_storm.conversation_runtime import _enforce_response_contract

        decision = {
            "action": "respond",
            "tool_calls": [],
            "rewritten_query": "wavelet neural network comparison",
            "response_contract": {"requires_citations": True},
            "authorization": {"evidence.search": "allowed"},
        }

        adjusted = _enforce_response_contract(
            decision, task_id="task-wavelet", message="继续比较并附引用"
        )

        self.assertEqual(adjusted["action"], "tool_call")
        self.assertEqual(adjusted["tool_calls"][0]["name"], "evidence.search")
        self.assertEqual(
            adjusted["tool_calls"][0]["arguments"]["query"],
            "wavelet neural network comparison",
        )
        self.assertIn("citation_contract", adjusted["runtime_adjustments"])

    def test_structured_memory_write_updates_durable_fact_without_phrase_rules(self):
        from knowledge_storm.paperstorm_intent_router import PaperStormIntentRouter

        planner_output = json.dumps(
            {
                "action": "tool_call",
                "tool_calls": [
                    {
                        "name": "memory.write",
                        "arguments": {
                            "content": "用户希望回答先给一句结论，再列三条依据。",
                            "canonical_key": "answer_format_preference",
                            "memory_type": "preference",
                        },
                    }
                ],
                "tool_policy": {
                    "external_retrieval": "deny",
                    "new_research": "deny",
                },
                "confidence": 0.99,
                "reason": "persist durable preference update",
                "response_contract": {
                    "task": "确认偏好已更新",
                    "requires_citations": False,
                },
            },
            ensure_ascii=False,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            runtime, _ = self.make_runtime(
                temp_dir,
                intent_router=PaperStormIntentRouter(
                    llm_router=lambda _prompt: planner_output
                ),
                chat_llm=lambda _prompt, **_kwargs: "好的，已更新。",
            )
            result = runtime.invoke(
                thread_id="thread-memory-update",
                request_id="request-memory-update",
                user_id="alice",
                message="更新我的回答偏好。",
                run_mode="fake",
            )
            memories = runtime.memory_service.list_memories("user/alice")

        self.assertEqual(result["memory_write"]["status"], "persisted")
        self.assertTrue(result["memory_write"]["read_after_write"]["verified"])
        self.assertEqual(memories[0]["canonical_key"], "answer_format_preference")
        self.assertIn("三条依据", memories[0]["content"])

    def test_multilingual_topic_does_not_reject_strong_query_evidence_overlap(self):
        from knowledge_storm.paperstorm_research_qa import (
            evaluate_evidence_sufficiency,
        )

        grade = evaluate_evidence_sufficiency(
            question="wavelet neural network vs feedforward neural network difference",
            topic="小波神经网络",
            evidence=[
                {
                    "title": "Wavelet neural networks",
                    "content": (
                        "Wavelet neural network activation differs from a "
                        "feedforward neural network through scale and translation."
                    ),
                }
            ],
            citations=[{"url": "https://example.org/paper"}],
        )

        self.assertTrue(grade["sufficient"])

    def test_continuation_uses_respond_action_and_exposes_llm_telemetry(self):
        def chat_llm(_prompt, **_kwargs):
            return {
                "content": "城门在雨中缓缓打开。",
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150},
                "cost_usd": 0.000065,
                "latency_ms": 42.5,
                "output_budget": 16384,
                "segments": 1,
                "truncated": False,
                "error": None,
            }

        def planner(_prompt):
            return (
                '{"action":"respond","tool_calls":[],"rewritten_query":"继续写下去",'
                '"working_subject":"","confidence":0.99,"reason":"continuation",'
                '"response_contract":{"task":"续写故事","continue_previous":true,'
                '"requires_citations":false,"style_notes":["不要自我介绍"]}}'
            )

        from knowledge_storm.paperstorm_intent_router import PaperStormIntentRouter

        with tempfile.TemporaryDirectory() as temp_dir:
            runtime, _ = self.make_runtime(
                temp_dir,
                intent_router=PaperStormIntentRouter(llm_router=planner),
                chat_llm=chat_llm,
            )
            result = runtime.invoke(
                thread_id="thread-story",
                request_id="request-story",
                user_id="alice",
                message="继续写下去",
                run_mode="fake",
                context_window=[{"role": "user", "content": "从前有一座城……"}],
            )

        self.assertEqual(result["router_decision"]["action"], "respond")
        self.assertEqual(result["answer"], "城门在雨中缓缓打开。")
        chat_event = next(item for item in result["node_events"] if item["node"] == "casual_chat")
        self.assertEqual(chat_event["details"]["usage"]["total_tokens"], 150)
        self.assertEqual(chat_event["details"]["finish_reason"], "stop")
        self.assertEqual(chat_event["details"]["output_budget"], 16384)

    def test_chat_provider_error_is_explicit_and_never_becomes_self_introduction(self):
        def chat_llm(_prompt, **_kwargs):
            return {
                "content": "",
                "finish_reason": "error",
                "usage": {},
                "cost_usd": 0.0,
                "latency_ms": 5.0,
                "output_budget": 16384,
                "segments": 0,
                "truncated": False,
                "error": {"type": "timeout", "message": "provider timed out", "recoverable": True},
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            runtime, _ = self.make_runtime(temp_dir, chat_llm=chat_llm)
            result = runtime.invoke(
                thread_id="thread-timeout",
                request_id="request-timeout",
                user_id="alice",
                message="继续写下去",
                run_mode="fake",
                context_window=[{"role": "user", "content": "从前有一座城……"}],
            )

        self.assertIn("模型调用超时", result["answer"])
        self.assertNotIn("我是 PaperStorm", result["answer"])
        self.assertEqual(result["llm_error"]["type"], "timeout")

    def make_runtime(self, root, **kwargs):
        from knowledge_storm.conversation_runtime import PaperStormConversationRuntime
        from knowledge_storm.paperstorm_service import PaperStormTaskService

        service = kwargs.pop("task_service", None) or PaperStormTaskService(Path(root) / "service")
        runtime = PaperStormConversationRuntime(
            root_dir=Path(root) / "graph_runtime",
            task_service=service,
            **kwargs,
        )
        self.addCleanup(runtime.close)
        return runtime, service

    def test_casual_path_uses_real_langgraph_and_sqlite_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime, _ = self.make_runtime(temp_dir)
            result = runtime.invoke(
                thread_id="thread-casual",
                request_id="request-casual",
                user_id="alice",
                message="你好",
                run_mode="fake",
            )
            state = runtime.get_thread_state("thread-casual")
            history = runtime.get_thread_history("thread-casual")

            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(result["router_decision"]["intent"], "casual_chat")
            self.assertFalse(result["retrieval_triggered"])
            self.assertIn("classify", result["executed_nodes"])
            self.assertIn("casual_chat", result["executed_nodes"])
            self.assertTrue(history["checkpoints"])
            self.assertEqual(state["values"]["request_id"], "request-casual")
            self.assertTrue((Path(temp_dir) / "graph_runtime" / "checkpoints.sqlite").exists())

    def test_casual_chat_uses_injected_chat_llm(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime, _ = self.make_runtime(
                temp_dir,
                chat_llm=lambda _prompt: "在的！想聊面试还是论文调研？我都可以帮你。",
            )
            result = runtime.invoke(
                thread_id="thread-llm-chat",
                request_id="request-llm-chat",
                user_id="alice",
                message="莫西莫西",
                run_mode="fake",
            )
            self.assertEqual(
                result["answer"],
                "在的！想聊面试还是论文调研？我都可以帮你。",
            )
            self.assertFalse(result["retrieval_triggered"])
            self.assertIn("casual_chat", result["executed_nodes"])

    def test_casual_chat_escalates_to_research_when_llm_needs_retrieval(self):
        from knowledge_storm.conversation_runtime import RETRIEVE_MARKER

        with tempfile.TemporaryDirectory() as temp_dir:
            runtime, _ = self.make_runtime(
                temp_dir,
                chat_llm=lambda _prompt: RETRIEVE_MARKER,
            )
            result = runtime.invoke(
                thread_id="thread-escalate",
                request_id="request-escalate",
                user_id="alice",
                message="说说最近 5G 和 6G 的关键差异",
                run_mode="fake",
            )
            self.assertTrue(result["retrieval_triggered"])
            self.assertIn("deep_research", result["executed_nodes"])
            self.assertEqual(result["route"], "deep_research")
            self.assertTrue(result["answer"])

    def test_response_model_cannot_start_research_without_planner_authorization(self):
        from knowledge_storm.conversation_runtime import RETRIEVE_MARKER
        from knowledge_storm.paperstorm_intent_router import PaperStormIntentRouter

        planner = PaperStormIntentRouter(
            llm_router=lambda _prompt: json.dumps(
                {
                    "action": "respond",
                    "tool_calls": [],
                    "tool_policy": {
                        "external_retrieval": "deny",
                        "new_research": "deny",
                    },
                    "confidence": 0.99,
                    "reason": "直接解释处理流程",
                },
                ensure_ascii=False,
            )
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime, _ = self.make_runtime(
                temp_dir,
                intent_router=planner,
                chat_llm=lambda _prompt, **_kwargs: RETRIEVE_MARKER,
            )
            result = runtime.invoke(
                thread_id="thread-denied-research",
                request_id="request-denied-research",
                user_id="alice",
                message="解释论文观点冲突时如何处理，不要启动调研",
                run_mode="fake",
            )

        self.assertNotIn("deep_research", result["executed_nodes"])
        self.assertNotIn("knowledge_retrieval", result["executed_nodes"])
        self.assertFalse(result["retrieval_triggered"])
        self.assertTrue(result["answer"])

    def test_meta_question_never_escalates_even_if_llm_emits_marker(self):
        from knowledge_storm.conversation_runtime import RETRIEVE_MARKER

        with tempfile.TemporaryDirectory() as temp_dir:
            runtime, _ = self.make_runtime(
                temp_dir,
                chat_llm=lambda _prompt: RETRIEVE_MARKER,
            )
            result = runtime.invoke(
                thread_id="thread-meta",
                request_id="request-meta",
                user_id="alice",
                message="你说一下知识库问答那边的逻辑，具体实现",
                run_mode="fake",
            )
            self.assertFalse(result["retrieval_triggered"])
            self.assertNotIn("deep_research", result["executed_nodes"])
            self.assertTrue(result["answer"])

    def test_evidence_judge_can_force_research_for_off_topic_kb(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            from knowledge_storm.paperstorm_service import PaperStormTaskService

            service = PaperStormTaskService(Path(temp_dir) / "service")
            task = service.submit_research_task(
                topic="pim 神经网络抑制",
                run_mode="fake",
                expected_keywords=["passive intermodulation"],
                forbidden_keywords=["DRAM"],
            )
            service.run_task(task["task_id"])
            runtime, _ = self.make_runtime(
                temp_dir,
                task_service=service,
                evidence_judge=lambda _prompt: "需要更多检索",
            )
            result = runtime.invoke(
                thread_id="thread-judge",
                request_id="request-judge",
                user_id="alice",
                message="请检索一下 Transformer 注意力机制和大语言模型训练的关系",
                topic="pim 神经网络抑制",
                task_id=task["task_id"],
                run_mode="fake",
            )
            self.assertTrue(result["retrieval_triggered"])
            self.assertIn("deep_research", result["executed_nodes"])
            self.assertEqual(result["route"], "deep_research")

    def test_authorized_evidence_search_escalates_when_no_evidence_exists(self):
        from knowledge_storm.paperstorm_intent_router import PaperStormIntentRouter

        planner_output = json.dumps(
            {
                "action": "tool_call",
                "tool_calls": [
                    {
                        "name": "evidence.search",
                        "arguments": {"query": "小波神经网络最新研究进展"},
                    }
                ],
                "tool_policy": {
                    "external_retrieval": "allow",
                    "new_research": "allow",
                },
                "confidence": 0.99,
                "reason": "先复用证据，不足时允许调研",
            },
            ensure_ascii=False,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime, _ = self.make_runtime(
                temp_dir,
                intent_router=PaperStormIntentRouter(
                    llm_router=lambda _prompt: planner_output
                ),
            )
            result = runtime.invoke(
                thread_id="thread-authorized-escalation",
                request_id="request-authorized-escalation",
                user_id="alice",
                message="小波神经网络有哪些最新研究进展？",
                run_mode="fake",
            )

        self.assertTrue(result["retrieval_triggered"])
        self.assertIn("knowledge_retrieval", result["executed_nodes"])
        self.assertIn("deep_research", result["executed_nodes"])
        self.assertEqual(result["route"], "deep_research")

    def test_evidence_judge_can_accept_existing_kb(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            from knowledge_storm.paperstorm_service import PaperStormTaskService

            service = PaperStormTaskService(Path(temp_dir) / "service")
            task = service.submit_research_task(
                topic="pim 神经网络抑制",
                run_mode="fake",
                expected_keywords=["passive intermodulation"],
                forbidden_keywords=["DRAM"],
            )
            service.run_task(task["task_id"])
            runtime, _ = self.make_runtime(
                temp_dir,
                task_service=service,
                evidence_judge=lambda _prompt: "可以回答",
            )
            with mock.patch.dict(os.environ, {"PAPERSTORM_CHAT_LLM": "0"}):
                result = runtime.invoke(
                    thread_id="thread-judge-ok",
                    request_id="request-judge-ok",
                    user_id="alice",
                    message="PIM 是什么？",
                    topic="pim 神经网络抑制",
                    task_id=task["task_id"],
                    run_mode="fake",
                )
            self.assertEqual((result["evidence_grade"] or {}).get("judge"), "llm")
            self.assertEqual(result["route"], "existing_knowledge")
            self.assertFalse(result["retrieval_triggered"])
            self.assertIn("passive intermodulation", result["answer"])

    def test_runtime_plans_retrieval_once_with_original_query_and_full_history(self):
        from knowledge_storm.paperstorm_service import PaperStormTaskService
        from knowledge_storm.search_planning import SearchPlan, SearchPlanner

        with tempfile.TemporaryDirectory() as temp_dir:
            service = PaperStormTaskService(Path(temp_dir) / "service")
            task = service.submit_research_task(
                topic="pim 神经网络抑制",
                run_mode="fake",
                expected_keywords=["passive intermodulation"],
            )
            service.run_task(task["task_id"])
            history = [
                {"role": "user", "content": "无源互调会造成什么影响？"},
                {"role": "assistant", "content": "它会造成射频链路失真。"},
                {"role": "user", "content": "请保留这一整段历史给检索规划器。"},
            ]
            planner_calls = []

            def plan_once(_planner, query, *, history=None):
                planner_calls.append((query, tuple(history or ())))
                return SearchPlan(
                    original_query=query,
                    standalone_query="passive intermodulation PIM",
                    must_terms=("passive intermodulation",),
                )

            runtime, _ = self.make_runtime(
                temp_dir,
                task_service=service,
                evidence_judge=lambda _prompt: "可以回答",
            )
            with mock.patch.object(SearchPlanner, "plan", autospec=True, side_effect=plan_once):
                result = runtime.invoke(
                    thread_id="thread-single-plan",
                    request_id="request-single-plan",
                    user_id="alice",
                    message="PIM 到底是什么？",
                    topic="pim 神经网络抑制",
                    task_id=task["task_id"],
                    run_mode="fake",
                    context_window=history,
                )

        self.assertEqual(1, len(planner_calls))
        self.assertEqual("PIM 到底是什么？", planner_calls[0][0])
        self.assertEqual(tuple(history), planner_calls[0][1])
        self.assertEqual("PIM 到底是什么？", result["router_decision"]["rewritten_query"])
        self.assertEqual(
            "PIM 到底是什么？",
            result["retrieval_metadata"]["search_plan"]["original_query"],
        )

    def test_runtime_trace_exposes_typed_planning_error(self):
        from knowledge_storm.paperstorm_service import PaperStormTaskService
        from knowledge_storm.search_planning import PlanningError, SearchPlanner

        with tempfile.TemporaryDirectory() as temp_dir:
            service = PaperStormTaskService(Path(temp_dir) / "service")
            task = service.submit_research_task(topic="PIM", run_mode="fake")
            service.run_task(task["task_id"])
            runtime, _ = self.make_runtime(temp_dir, task_service=service)

            with mock.patch.object(
                SearchPlanner,
                "plan",
                side_effect=PlanningError(
                    "planner provider timed out", error_type="provider_timeout"
                ),
            ):
                with self.assertRaises(PlanningError):
                    runtime.invoke(
                        thread_id="thread-planning-error",
                        request_id="request-planning-error",
                        user_id="alice",
                        message="PIM 是什么？",
                        task_id=task["task_id"],
                        run_mode="fake",
                    )

            events = runtime._request_trace_events(
                "thread-planning-error", "request-planning-error"
            )

        event = next(
            item
            for item in events
            if item["node"] == "knowledge_retrieval" and item["status"] == "error"
        )
        self.assertEqual("PlanningError", event["details"]["exception_type"])
        self.assertEqual("provider_timeout", event["details"]["error_type"])
        self.assertIn("planner provider timed out", event["details"]["message"])

    def test_casual_chat_prompt_includes_conversation_history(self):
        recorded = {}

        def recorder(prompt):
            recorded["prompt"] = prompt
            return "继续聊！"

        with tempfile.TemporaryDirectory() as temp_dir:
            runtime, _ = self.make_runtime(temp_dir, chat_llm=recorder)
            runtime.invoke(
                thread_id="thread-history",
                request_id="request-history",
                user_id="alice",
                message="是因为要想我保密吗",
                run_mode="fake",
                context_window=[
                    {"role": "user", "content": "所以你跑在啥模型上"},
                    {"role": "assistant", "content": "我是基于大语言模型构建的智能助手。"},
                ],
            )
            self.assertIn("所以你跑在啥模型上", recorded["prompt"])
            self.assertIn("连续对话", recorded["prompt"])
            self.assertIn("BM25", recorded["prompt"])
            self.assertIn("RRF", recorded["prompt"])
            self.assertIn("当前运行模式", recorded["prompt"])

    def test_question_topic_follows_question_when_off_topic(self):
        from knowledge_storm.conversation_runtime import _question_topic

        state = {
            "message": "你去查一下muon优化器，这个优化器为啥效果好",
            "router_decision": {"rewritten_query": "muon 优化器 为什么效果好"},
            "topic": "pim 神经网络抑制",
        }
        topic = _question_topic(state)
        self.assertNotIn("pim", topic.lower())
        self.assertIn("muon", topic.lower())
        related = {
            "message": "PIM 是什么？",
            "router_decision": {"rewritten_query": "PIM 是什么？"},
            "topic": "pim 神经网络抑制",
        }
        self.assertIn("pim", _question_topic(related).lower())

    def test_long_term_memory_is_shared_across_threads_not_copied_into_store(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime, _ = self.make_runtime(temp_dir)
            written = runtime.invoke(
                thread_id="thread-one",
                request_id="remember-1",
                user_id="alice",
                message="请记住：回答我的问题时使用中文。",
                run_mode="fake",
            )
            recalled = runtime.invoke(
                thread_id="thread-two",
                request_id="recall-1",
                user_id="alice",
                message="你记得我的回答偏好吗？",
                run_mode="fake",
            )
            isolated = runtime.invoke(
                thread_id="thread-bob",
                request_id="recall-bob",
                user_id="bob",
                message="你记得我的回答偏好吗？",
                run_mode="fake",
            )

            self.assertEqual(written["memory_write"]["status"], "persisted")
            self.assertEqual(recalled["route"], "casual_chat")
            self.assertIn("中文", recalled["answer"])
            self.assertEqual(isolated["memory_recall"]["results"], [])

    def test_recalled_preference_is_context_for_substantive_task_not_terminal_answer(self):
        from knowledge_storm.paperstorm_intent_router import PaperStormIntentRouter

        def planner(_prompt):
            return (
                '{"action":"call_tool","tool_calls":[{"name":"memory.search",'
                '"arguments":{"query":"回答偏好"}}],"rewritten_query":"解释梯度下降",'
                '"working_subject":"梯度下降","confidence":0.98,'
                '"reason":"use preference","response_contract":{"task":"解释梯度下降",'
                '"requires_citations":false}}'
            )

        captured = {}

        def chat_llm(prompt, **_kwargs):
            captured["prompt"] = prompt
            return "结论：梯度下降通过沿负梯度方向更新参数来降低损失。"

        with tempfile.TemporaryDirectory() as temp_dir:
            runtime, _ = self.make_runtime(
                temp_dir,
                intent_router=PaperStormIntentRouter(llm_router=planner),
                chat_llm=chat_llm,
            )
            runtime.memory_service.ingest_message(
                namespace="user/alice",
                message="请记住我的回答偏好：先给结论，再给证据。",
                source_message_id="preference-1",
                subject="alice",
            )
            result = runtime.invoke(
                thread_id="thread-preference",
                request_id="request-preference",
                user_id="alice",
                message="按照我的偏好解释梯度下降，不要检索论文。",
                run_mode="fake",
            )

        self.assertEqual(result["route"], "casual_chat")
        self.assertIn("梯度下降", result["answer"])
        self.assertIn("先给结论", captured["prompt"])

    def test_memory_tool_mode_drives_behavior_without_phrase_classification(self):
        from knowledge_storm.paperstorm_intent_router import PaperStormIntentRouter

        planner = PaperStormIntentRouter(
            llm_router=lambda _prompt: json.dumps(
                {
                    "action": "tool_call",
                    "tool_calls": [
                        {
                            "name": "memory.search",
                            "arguments": {"query": "回答偏好", "mode": "answer"},
                        }
                    ],
                    "tool_policy": {
                        "external_retrieval": "deny",
                        "new_research": "deny",
                    },
                    "confidence": 0.99,
                    "reason": "读取长期记忆后回答",
                },
                ensure_ascii=False,
            )
        )
        captured = {}

        def chat_llm(prompt, **_kwargs):
            captured["prompt"] = prompt
            return "我记得你偏好先给结论，再给证据。"

        with tempfile.TemporaryDirectory() as temp_dir:
            runtime, _ = self.make_runtime(
                temp_dir, intent_router=planner, chat_llm=chat_llm
            )
            runtime.memory_service.ingest_message(
                namespace="user/alice",
                message="请记住我的回答偏好：先给结论，再给证据。",
                source_message_id="preference-mode",
                subject="alice",
            )
            result = runtime.invoke(
                thread_id="thread-memory-mode",
                request_id="request-memory-mode",
                user_id="alice",
                message="从长期记忆复述我的偏好，且不要调用外部工具。",
                run_mode="fake",
            )

        self.assertEqual(result["route"], "casual_chat")
        self.assertIn("先给结论", result["answer"])
        self.assertIn("先给结论", captured["prompt"])

    def test_deep_research_is_an_isolated_tool_with_artifact_contract(self):
        from knowledge_storm.conversation_runtime import StormDeepResearchTool
        from knowledge_storm.paperstorm_service import PaperStormTaskService

        with tempfile.TemporaryDirectory() as temp_dir:
            service = PaperStormTaskService(Path(temp_dir) / "service")
            tool = StormDeepResearchTool(service)
            result = tool.run(
                {
                    "question": "PIM 是什么？",
                    "topic": "PIM 无源互调",
                    "run_mode": "fake",
                    "retriever": "arxiv",
                    "output_language": "zh",
                }
            )

            self.assertEqual(tool.to_schema()["name"], "storm_deep_research")
            self.assertTrue(result["citations"])
            self.assertTrue(result["artifact_uri"].startswith("file:"))
            self.assertNotIn("messages", result)
            self.assertNotIn("qa_history", result)

    def test_transient_deep_research_failure_is_retried_by_node_policy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tool = FlakyDeepResearchTool()
            runtime, _ = self.make_runtime(temp_dir, deep_research_tool=tool)
            result = runtime.invoke(
                thread_id="thread-retry",
                request_id="request-retry",
                user_id="alice",
                message="请调研 PIM 无源互调论文",
                run_mode="fake",
            )

            self.assertEqual(tool.calls, 2)
            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(result["used_task_id"], "retry-task")
            self.assertIn("deep_research", result["executed_nodes"])
            self.assertIn("error", [event["status"] for event in result["node_events"]])

    def test_request_id_is_idempotent_across_runtime_recreation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime, service = self.make_runtime(temp_dir)
            first = runtime.invoke(
                thread_id="thread-idempotent",
                request_id="same-request",
                user_id="alice",
                message="请调研 PIM 无源互调论文",
                run_mode="fake",
            )
            task_count = len(service.list_tasks())
            runtime.close()

            from knowledge_storm.conversation_runtime import PaperStormConversationRuntime

            recreated = PaperStormConversationRuntime(
                root_dir=Path(temp_dir) / "graph_runtime",
                task_service=service,
            )
            self.addCleanup(recreated.close)
            replay = recreated.invoke(
                thread_id="thread-idempotent",
                request_id="same-request",
                user_id="alice",
                message="请调研 PIM 无源互调论文",
                run_mode="fake",
            )

            self.assertEqual(first["used_task_id"], replay["used_task_id"])
            self.assertTrue(replay["idempotent_replay"])
            self.assertEqual(len(service.list_tasks()), task_count)

    def test_request_idempotency_is_scoped_to_thread(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime, _ = self.make_runtime(temp_dir)
            first = runtime.invoke(
                thread_id="thread-alice",
                request_id="client-sequence-1",
                user_id="alice",
                message="你好",
                run_mode="fake",
            )
            second = runtime.invoke(
                thread_id="thread-bob",
                request_id="client-sequence-1",
                user_id="bob",
                message="你是谁？",
                run_mode="fake",
            )

            self.assertEqual(first["thread_id"], "thread-alice")
            self.assertEqual(second["thread_id"], "thread-bob")
            self.assertFalse(second["idempotent_replay"])

    def test_service_api_exposes_graph_invoke_state_and_history(self):
        try:
            from starlette.exceptions import StarletteDeprecationWarning

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", StarletteDeprecationWarning)
                from fastapi.testclient import TestClient
        except Exception as exc:  # pragma: no cover
            self.skipTest(str(exc))
        from examples.storm_examples.paperstorm_service_api import create_app

        with tempfile.TemporaryDirectory() as temp_dir:
            client = TestClient(create_app(service_root=Path(temp_dir)))
            invoked = client.post(
                "/conversation-graph/invoke",
                json={
                    "thread_id": "api-thread",
                    "request_id": "api-request",
                    "user_id": "alice",
                    "message": "你好",
                    "run_mode": "fake",
                },
            )
            state = client.get(
                "/conversation-graph/threads/api-thread/state",
                params={"tenant_id": "local", "user_id": "alice"},
            )
            history = client.get(
                "/conversation-graph/threads/api-thread/history",
                params={"tenant_id": "local", "user_id": "alice"},
            )
        self.assertEqual(invoked.status_code, 200)
        self.assertEqual(invoked.json()["runtime"], "paperstorm-production-runtime")
        self.assertEqual(invoked.json()["graph_runtime"], "conversation-runtime")
        self.assertEqual(state.json()["values"]["request_id"], "api-request")
        self.assertTrue(history.json()["checkpoints"])

    def test_chat_default_path_reports_langgraph_run(self):
        from knowledge_storm.paperstorm_service import PaperStormTaskService

        with tempfile.TemporaryDirectory() as temp_dir:
            service = PaperStormTaskService(Path(temp_dir))
            session = service.create_chat_session(user_id="alice", run_mode="fake")
            reply = service.send_chat_message(session["chat_id"], "你好")

            self.assertEqual(
                reply["conversation_runtime"], "paperstorm-production-runtime"
            )
            self.assertEqual(reply["graph_run"]["graph_runtime"], "conversation-runtime")
            self.assertEqual(reply["graph_run"]["status"], "succeeded")
            self.assertIn("classify", reply["graph_run"]["executed_nodes"])

    def test_graph_spec_exposes_nodes_edges_and_runtime_policies(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime, _ = self.make_runtime(temp_dir)
            spec = runtime.get_graph_spec()

            self.assertEqual(spec["runtime"], "conversation-runtime")
            self.assertIn("classify", spec["nodes"])
            self.assertIn("deep_research", spec["nodes"])
            self.assertTrue(spec["edges"])
            self.assertEqual(spec["checkpoint"]["backend"], "sqlite")
            self.assertEqual(spec["retry"]["max_attempts"], 2)


if __name__ == "__main__":
    unittest.main()
