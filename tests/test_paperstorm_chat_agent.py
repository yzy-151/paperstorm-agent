import tempfile
import unittest
from pathlib import Path


class PaperStormChatAgentTest(unittest.TestCase):
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
        self.assertTrue(reply["assistant_message"]["content"])
        self.assertGreaterEqual(len(reply["messages"]), 2)
        self.assertEqual(loaded["chat_id"], session["chat_id"])
        self.assertEqual(loaded["messages"][-1]["role"], "assistant")
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


if __name__ == "__main__":
    unittest.main()
