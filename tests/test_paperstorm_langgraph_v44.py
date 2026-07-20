import json
import tempfile
import unittest
import warnings
from pathlib import Path


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


class PaperStormLangGraphV44Test(unittest.TestCase):
    def make_runtime(self, root, **kwargs):
        from knowledge_storm.paperstorm_langgraph_v44 import PaperStormLangGraphRuntime
        from knowledge_storm.paperstorm_service import PaperStormTaskService

        service = kwargs.pop("task_service", None) or PaperStormTaskService(Path(root) / "service")
        runtime = PaperStormLangGraphRuntime(
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
        from knowledge_storm.paperstorm_langgraph_v44 import RETRIEVE_MARKER

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

    def test_question_topic_follows_question_when_off_topic(self):
        from knowledge_storm.paperstorm_langgraph_v44 import _question_topic

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
            self.assertEqual(recalled["route"], "memory_answer")
            self.assertIn("中文", recalled["answer"])
            self.assertEqual(isolated["memory_recall"]["results"], [])

    def test_deep_research_is_an_isolated_tool_with_artifact_contract(self):
        from knowledge_storm.paperstorm_langgraph_v44 import StormDeepResearchToolV44
        from knowledge_storm.paperstorm_service import PaperStormTaskService

        with tempfile.TemporaryDirectory() as temp_dir:
            service = PaperStormTaskService(Path(temp_dir) / "service")
            tool = StormDeepResearchToolV44(service)
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

            from knowledge_storm.paperstorm_langgraph_v44 import PaperStormLangGraphRuntime

            recreated = PaperStormLangGraphRuntime(
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

    def test_runtime_benchmark_measures_paths_checkpoint_and_recovery(self):
        from knowledge_storm.paperstorm_langgraph_benchmark_v44 import run_langgraph_benchmark

        with tempfile.TemporaryDirectory() as temp_dir:
            report = run_langgraph_benchmark(Path(temp_dir))

            self.assertEqual(report["metrics"]["path_accuracy"], 1.0)
            self.assertEqual(report["metrics"]["idempotency_rate"], 1.0)
            self.assertEqual(report["metrics"]["checkpoint_restore_rate"], 1.0)
            self.assertEqual(report["metrics"]["retry_recovery_rate"], 1.0)
            self.assertEqual(report["metrics"]["cross_user_leakage_rate"], 0.0)

    def test_service_api_exposes_graph_invoke_state_history_and_benchmark(self):
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
            benchmark = client.post("/evaluations/runtime-v44")
            latest = client.get("/evaluations/runtime-v44/latest")

        self.assertEqual(invoked.status_code, 200)
        self.assertEqual(invoked.json()["runtime"], "paperstorm-production-v4.5")
        self.assertEqual(invoked.json()["graph_runtime"], "langgraph-v4.4")
        self.assertEqual(state.json()["values"]["request_id"], "api-request")
        self.assertTrue(history.json()["checkpoints"])
        self.assertEqual(benchmark.status_code, 200)
        self.assertEqual(latest.json()["run_id"], benchmark.json()["run_id"])

    def test_chat_default_path_reports_langgraph_run(self):
        from knowledge_storm.paperstorm_service import PaperStormTaskService

        with tempfile.TemporaryDirectory() as temp_dir:
            service = PaperStormTaskService(Path(temp_dir))
            session = service.create_chat_session(user_id="alice", run_mode="fake")
            reply = service.send_chat_message(session["chat_id"], "你好")

            self.assertEqual(
                reply["conversation_runtime"], "paperstorm-production-v4.5"
            )
            self.assertEqual(reply["graph_run"]["graph_runtime"], "langgraph-v4.4")
            self.assertEqual(reply["graph_run"]["status"], "succeeded")
            self.assertIn("classify", reply["graph_run"]["executed_nodes"])

    def test_graph_spec_exposes_nodes_edges_and_runtime_policies(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime, _ = self.make_runtime(temp_dir)
            spec = runtime.get_graph_spec()

            self.assertEqual(spec["runtime"], "langgraph-v4.4")
            self.assertIn("classify", spec["nodes"])
            self.assertIn("deep_research", spec["nodes"])
            self.assertTrue(spec["edges"])
            self.assertEqual(spec["checkpoint"]["backend"], "sqlite")
            self.assertEqual(spec["retry"]["max_attempts"], 2)

    def test_dashboard_exposes_graph_debugger_and_runtime_benchmark(self):
        root = Path(__file__).resolve().parents[1]
        index = (root / "frontend/paperstorm_dashboard/index.html").read_text(
            encoding="utf-8"
        )
        script = (root / "frontend/paperstorm_dashboard/app.js").read_text(
            encoding="utf-8"
        )

        self.assertIn("v4.4", index)
        self.assertIn("chat-graph-run", index)
        self.assertIn("chat-checkpoint-history", index)
        self.assertIn("runtime-v44-metrics", index)
        self.assertIn("/conversation-graph/threads/", script)
        self.assertIn("/evaluations/runtime-v44", script)


if __name__ == "__main__":
    unittest.main()
