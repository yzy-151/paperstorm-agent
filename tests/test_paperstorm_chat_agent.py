import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


@mock.patch.dict(
    os.environ,
    {
        "PAPERSTORM_RETRIEVAL_EMBEDDING": "hash",
        "PAPERSTORM_CHAT_LLM": "0",
        "PAPERSTORM_JUDGE_LLM": "0",
    },
)
class PaperStormChatAgentTest(unittest.TestCase):
    def test_chat_session_persists_explicit_memory_retrieval_mode(self):
        service = self.make_service()
        session = service.create_chat_session(
            run_mode="fake",
            memory_retrieval_mode="lexical",
        )

        self.assertEqual(session["memory_retrieval_mode"], "lexical")
        loaded = service.get_chat_session(session["chat_id"])
        self.assertEqual(loaded["memory_retrieval_mode"], "lexical")

    def make_service(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        from knowledge_storm.paperstorm_service import PaperStormTaskService

        return PaperStormTaskService(root_dir=Path(temp_dir.name))

    def test_chat_session_first_question_auto_researches_and_persists_context(self):
        service = self.make_service()
        session = service.create_chat_session(
            title="PIM research chat",
            topic="pim 神经网络抑制",
            run_mode="fake",
            expected_keywords=["passive intermodulation", "RF"],
            forbidden_keywords=["DRAM", "processing-in-memory"],
        )

        reply = service.send_chat_message(
            session["chat_id"],
            "PIM 是什么，神经网络如何抑制它？",
        )
        loaded = service.get_chat_session(session["chat_id"])

        self.assertEqual(reply["mode"], "chat")
        self.assertTrue(reply["retrieval_triggered"])
        self.assertTrue(reply["used_task_id"])
        self.assertEqual(
            reply["assistant_message"]["metadata"].get("retrieval_stack"),
            "storm_deep_research_tool",
        )
        self.assertTrue(reply["assistant_message"]["content"])
        user_telemetry = reply["message"]["metadata"]["telemetry"]
        assistant_telemetry = reply["assistant_message"]["metadata"]["telemetry"]
        self.assertGreater(user_telemetry["message_tokens"], 0)
        self.assertGreaterEqual(assistant_telemetry["duration_ms"], 0)
        self.assertIn("prompt_tokens", assistant_telemetry)
        self.assertIn("completion_tokens", assistant_telemetry)
        self.assertIn("total_tokens", assistant_telemetry)
        self.assertGreaterEqual(len(reply["messages"]), 2)
        self.assertEqual(loaded["chat_id"], session["chat_id"])
        self.assertEqual(loaded["messages"][-1]["role"], "assistant")
        self.assertIn("telemetry", loaded["messages"][-1]["metadata"])
        self.assertTrue(reply["context_window"])
        self.assertIn("summary", reply["compressed_context"])
        self.assertIn("semantic", reply["memory_context"])

    def test_chat_followup_reuses_previous_task_and_keeps_sliding_window(self):
        service = self.make_service()
        session = service.create_chat_session(
            topic="pim 神经网络抑制",
            run_mode="fake",
            expected_keywords=["passive intermodulation"],
            forbidden_keywords=["DRAM"],
            context_window_size=3,
        )

        first = service.send_chat_message(session["chat_id"], "PIM 是什么？")
        second = service.send_chat_message(session["chat_id"], "那它为什么不是 DRAM？")

        self.assertTrue(first["retrieval_triggered"])
        self.assertFalse(second["retrieval_triggered"])
        self.assertEqual(second["used_task_id"], first["used_task_id"])
        self.assertLessEqual(len(second["context_window"]), 3)
        self.assertIn("PIM 是什么", second["compressed_context"]["summary"])
        self.assertIn("qa_history_count", second["research_answer"])

    def test_chat_auto_researches_when_existing_task_has_insufficient_evidence(self):
        service = self.make_service()
        stale_task = service.submit_research_task(
            topic="Transformer 注意力机制",
            run_mode="fake",
            expected_keywords=["attention"],
            forbidden_keywords=[],
        )
        stale_state = service.run_task(stale_task["task_id"])
        stale_output = Path(stale_state["output_dir"])
        (stale_output / "storm_gen_article_polished.txt").write_text(
            "Transformer attention uses query, key, and value projections for language modeling.",
            encoding="utf-8",
        )
        (stale_output / "raw_search_results.json").write_text(
            json.dumps(
                [
                    {
                        "title": "Transformer attention",
                        "description": "Attention mechanisms for language models.",
                        "url": "https://example.com/attention",
                        "snippets": ["Query key value attention."],
                    }
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        session = service.create_chat_session(
            topic="pim 神经网络抑制",
            run_mode="fake",
            expected_keywords=["passive intermodulation"],
            forbidden_keywords=["DRAM"],
            task_id=stale_task["task_id"],
        )

        with mock.patch.dict(os.environ, {"PAPERSTORM_CHAT_LLM": "0"}):
            reply = service.send_chat_message(session["chat_id"], "PIM 是什么？")

        self.assertTrue(reply["retrieval_triggered"])
        self.assertNotEqual(reply["used_task_id"], stale_task["task_id"])
        self.assertEqual(
            reply["research_answer"]["decision"]["action"],
            "retrieve_then_answer",
        )
        self.assertIn("passive intermodulation", reply["assistant_message"]["content"])

    def test_chat_can_answer_casual_service_questions_without_research(self):
        service = self.make_service()
        session = service.create_chat_session(topic="pim 神经网络抑制", run_mode="fake")

        with mock.patch.dict(os.environ, {"PAPERSTORM_CHAT_LLM": "0"}):
            reply = service.send_chat_message(session["chat_id"], "你好，你能做什么？")

        self.assertFalse(reply["retrieval_triggered"])
        self.assertFalse(reply["used_task_id"])
        self.assertEqual(reply["research_answer"]["decision"]["action"], "chat_fallback")
        self.assertIn("论文调研", reply["assistant_message"]["content"])

    def test_chat_answers_bot_identity_model_and_ui_questions_without_research(self):
        service = self.make_service()
        session = service.create_chat_session(topic="pim 神经网络抑制", run_mode="fake")

        with mock.patch.dict(os.environ, {"PAPERSTORM_CHAT_LLM": "0"}):
            for question in ["你是什么模型？", "当前上下文怎么压缩？", "这个网页怎么使用？"]:
                reply = service.send_chat_message(session["chat_id"], question)
                self.assertFalse(reply["retrieval_triggered"], question)
                self.assertFalse(reply["used_task_id"], question)
                self.assertEqual(
                    reply["research_answer"]["decision"]["action"],
                    "chat_fallback",
                    question,
                )
                self.assertNotIn("无源器件非线性导致", reply["assistant_message"]["content"])

    def test_social_chat_does_not_leak_the_session_research_topic(self):
        service = self.make_service()
        session = service.create_chat_session(topic="pim 神经网络抑制", run_mode="fake")

        with mock.patch.dict(os.environ, {"PAPERSTORM_CHAT_LLM": "0"}):
            reply = service.send_chat_message(session["chat_id"], "莫西莫西")

        content = reply["assistant_message"]["content"].lower()
        self.assertFalse(reply["retrieval_triggered"])
        self.assertFalse(reply["used_task_id"])
        self.assertEqual(reply["router_decision"]["intent"], "casual_chat")
        self.assertNotIn("pim", content)
        self.assertNotIn("神经网络抑制", content)

    def test_casual_interview_question_gets_topic_aware_reply(self):
        service = self.make_service()
        session = service.create_chat_session(topic="pim 神经网络抑制", run_mode="fake")

        with mock.patch.dict(os.environ, {"PAPERSTORM_CHAT_LLM": "0"}):
            reply = service.send_chat_message(session["chat_id"], "帮我准备面试")

        content = reply["assistant_message"]["content"]
        self.assertFalse(reply["retrieval_triggered"])
        self.assertIn("面试", content)
        self.assertNotEqual(
            content,
            "你好，我是 PaperStorm Research Agent。你可以聊天、查询长期记忆、问已有知识库，或启动论文调研与深度研究。",
        )

    def test_casual_after_research_does_not_reuse_stale_retrieval_stack(self):
        service = self.make_service()
        session = service.create_chat_session(topic="pim 神经网络抑制", run_mode="fake")

        service.send_chat_message(session["chat_id"], "PIM 是什么？")
        with mock.patch.dict(os.environ, {"PAPERSTORM_CHAT_LLM": "0"}):
            casual = service.send_chat_message(session["chat_id"], "莫西莫西")

        self.assertFalse(casual["retrieval_triggered"])
        self.assertEqual(
            casual["assistant_message"]["metadata"].get("retrieval_stack"),
            "",
        )

    def test_fastapi_adapter_exposes_chat_session_routes(self):
        from fastapi.testclient import TestClient

        from examples.storm_examples.paperstorm_service_api import create_app

        with tempfile.TemporaryDirectory() as temp_dir:
            app = create_app(service_root=Path(temp_dir))
            client = TestClient(app)

            created = client.post(
                "/chat/sessions",
                json={
                    "topic": "pim 神经网络抑制",
                    "run_mode": "fake",
                    "expected_keywords": ["passive intermodulation"],
                    "forbidden_keywords": ["DRAM"],
                },
            )
            self.assertEqual(created.status_code, 200)
            chat_id = created.json()["chat_id"]

            reply = client.post(
                "/chat/sessions/{0}/messages".format(chat_id),
                json={"message": "PIM 是什么？"},
            )
            self.assertEqual(reply.status_code, 200)
            self.assertTrue(reply.json()["retrieval_triggered"])

            loaded = client.get("/chat/sessions/{0}".format(chat_id))
            self.assertEqual(loaded.status_code, 200)
            self.assertEqual(loaded.json()["chat_id"], chat_id)

    def test_list_sessions_and_regenerate_keep_history(self):
        service = self.make_service()
        session = service.create_chat_session(
            topic="pim 神经网络抑制",
            run_mode="fake",
            expected_keywords=["passive intermodulation"],
            forbidden_keywords=["DRAM"],
        )
        first = service.send_chat_message(session["chat_id"], "PIM 是什么？")

        sessions = service.list_chat_sessions()
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["chat_id"], session["chat_id"])
        self.assertGreaterEqual(sessions[0]["message_count"], 2)

        before = len(first["messages"])
        regenerated = service.regenerate_chat_message(session["chat_id"])
        messages = regenerated["messages"]
        self.assertTrue(regenerated.get("regenerated"))
        self.assertEqual(messages[-1]["role"], "assistant")
        self.assertEqual(messages[-1]["metadata"].get("version"), 2)
        # 旧回答保留为 v1，新回答追加为 v2，不覆盖历史
        self.assertEqual(len(messages), before + 1)
        self.assertEqual(messages[-2]["metadata"].get("version"), 1)
        self.assertIn("citations", messages[-1]["metadata"])

    def test_stop_generation_flag_is_best_effort_and_cleared_on_next_send(self):
        service = self.make_service()
        session = service.create_chat_session(topic="pim 神经网络抑制", run_mode="fake")

        stopped = service.stop_chat_generation(session["chat_id"])
        self.assertEqual(stopped["status"], "stopping")

        reply = service.send_chat_message(session["chat_id"], "你好")
        self.assertNotEqual(reply.get("status"), "stopped")
        self.assertGreaterEqual(len(reply["messages"]), 2)

    def test_loading_legacy_session_rehydrates_article_citation_sources(self):
        from knowledge_storm.paperstorm_chat_agent import PaperStormChatAgent

        service = self.make_service()
        task = service.submit_research_task(topic="Physical AI", run_mode="fake")
        run_dir = Path(task["output_dir"])
        (run_dir / "storm_gen_article_polished.txt").write_text(
            "# 定义\n\nPhysical AI connects software and the physical world.[1]",
            encoding="utf-8",
        )
        (run_dir / "url_to_info.json").write_text(
            json.dumps(
                {
                    "url_to_unified_index": {"https://example.com/paper": 1},
                    "url_to_info": {
                        "https://example.com/paper": {
                            "title": "Physical AI Paper",
                            "url": "https://example.com/paper",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        agent = PaperStormChatAgent(service)
        session = agent.create_session(task_id=task["task_id"])
        session["messages"] = [
            {
                "role": "assistant",
                "content": "Physical AI connects both worlds.[1]",
                "metadata": {
                    "used_task_id": task["task_id"],
                    "citations": [
                        {
                            "id": 1,
                            "title": "Generated article paragraph 1",
                            "url": str(run_dir / "storm_gen_article_polished.txt"),
                            "source_type": "article",
                            "chunk_id": "article-1",
                        }
                    ],
                },
            }
        ]
        agent._write_session(session)

        restored = agent.get_session(session["chat_id"])
        citation = restored["messages"][0]["metadata"]["citations"][0]

        self.assertEqual(citation["title"], "定义 · 第 1 段")
        self.assertEqual(citation["article_anchor"], "article-paragraph-1")
        self.assertEqual(citation["original_sources"][0]["title"], "Physical AI Paper")


if __name__ == "__main__":
    unittest.main()
