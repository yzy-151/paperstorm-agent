import tempfile
import unittest
from pathlib import Path


class PaperStormSessionRecallTests(unittest.TestCase):
    def test_fts_search_finds_messages_across_sessions_and_returns_context(self):
        from knowledge_storm.paperstorm_session_recall import SessionRecallStore

        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionRecallStore(Path(temp_dir) / "sessions.sqlite3")
            store.append_message(
                user_id="user-1",
                chat_id="chat-old",
                message_id="m1",
                role="user",
                content="之前讨论了 PIM 无源互调抑制论文。",
            )
            store.append_message(
                user_id="user-1",
                chat_id="chat-old",
                message_id="m2",
                role="assistant",
                content="重点包括数字预失真和神经网络抵消。",
            )
            store.append_message(
                user_id="user-2",
                chat_id="chat-private",
                message_id="m3",
                role="user",
                content="PIM 论文不应跨用户泄漏。",
            )

            result = store.search("user-1", "之前聊过的 PIM 论文", top_k=3)

            self.assertEqual(result["retrieval"], "sqlite_fts5_bm25")
            self.assertEqual(result["results"][0]["chat_id"], "chat-old")
            self.assertIn("数字预失真", " ".join(result["results"][0]["context"]))
            self.assertNotIn("chat-private", [item["chat_id"] for item in result["results"]])

    def test_empty_or_unmatched_query_returns_no_results(self):
        from knowledge_storm.paperstorm_session_recall import SessionRecallStore

        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionRecallStore(Path(temp_dir) / "sessions.sqlite3")
            self.assertEqual(store.search("user-1", "", top_k=3)["results"], [])
            self.assertEqual(store.search("user-1", "不存在的主题", top_k=3)["results"], [])

    def test_chinese_query_recalls_prior_session_without_latin_keyword(self):
        from knowledge_storm.paperstorm_session_recall import SessionRecallStore

        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionRecallStore(Path(temp_dir) / "sessions.sqlite3")
            store.append_message(
                user_id="user-a",
                chat_id="old-chat",
                message_id="m-cn-1",
                role="assistant",
                content="我们讨论过无源互调抑制论文，并比较了数字预失真方案。",
            )

            result = store.search("user-a", "之前聊过的互调抑制论文", top_k=3)

            self.assertEqual(result["results"][0]["chat_id"], "old-chat")
            self.assertIn("无源互调", result["results"][0]["content"])


if __name__ == "__main__":
    unittest.main()
