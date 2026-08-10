import tempfile
import unittest
from pathlib import Path


class KeywordEmbedding:
    name = "keyword-test"

    def embed(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        lowered = str(text).lower()
        return [
            float("pim" in lowered or "无源互调" in lowered),
            float("python" in lowered),
            float("中文" in lowered),
        ]


class PaperStormMemoryV56Tests(unittest.TestCase):
    def _service(self, root):
        from knowledge_storm.paperstorm_memory_v56 import LongTermMemoryServiceV56

        return LongTermMemoryServiceV56(root, embedding_provider=KeywordEmbedding())

    def test_sqlite_wal_episode_is_idempotent_and_namespaced(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            service = self._service(Path(temp_dir) / "memory")
            first = service.ingest_episode(
                "tenant-a:user-1",
                "PIM means passive intermodulation.",
                source_id="message-1",
                occurred_at="2026-01-01T00:00:00+00:00",
            )
            second = service.ingest_episode(
                "tenant-a:user-1",
                "PIM means passive intermodulation.",
                source_id="message-1",
                occurred_at="2026-01-01T00:00:00+00:00",
            )

            self.assertEqual(first["episode_id"], second["episode_id"])
            self.assertTrue(second["deduplicated"])
            self.assertEqual(service.list_episodes("tenant-a:user-1")[0]["source_id"], "message-1")
            self.assertEqual(service.list_episodes("tenant-b:user-1"), [])
            self.assertEqual(service.storage_info()["journal_mode"].lower(), "wal")

    def test_fact_update_preserves_history_and_provenance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            service = self._service(Path(temp_dir) / "memory")
            old_episode = service.ingest_episode(
                "tenant:user",
                "用户偏好英文回答",
                source_id="old-message",
                occurred_at="2026-01-01T00:00:00+00:00",
            )
            old = service.upsert(
                namespace="tenant:user",
                memory_type="preference",
                subject="user",
                content="回答使用英文",
                canonical_key="answer_language",
                source_message_ids=["old-message"],
                valid_from="2026-01-01T00:00:00+00:00",
                metadata={"episode_id": old_episode["episode_id"]},
            )
            new_episode = service.ingest_episode(
                "tenant:user",
                "用户现在偏好中文回答",
                source_id="new-message",
                occurred_at="2026-02-01T00:00:00+00:00",
            )
            new = service.upsert(
                namespace="tenant:user",
                memory_type="preference",
                subject="user",
                content="回答使用中文",
                canonical_key="answer_language",
                source_message_ids=["new-message"],
                valid_from="2026-02-01T00:00:00+00:00",
                metadata={"episode_id": new_episode["episode_id"]},
            )

            current = service.search("tenant:user", "回答语言", as_of="2026-03-01T00:00:00+00:00")
            historical = service.search("tenant:user", "回答语言", as_of="2026-01-15T00:00:00+00:00")
            old_record = service.get_memory("tenant:user", old["id"], include_inactive=True)

            self.assertEqual(current["results"][0]["content"], "回答使用中文")
            self.assertEqual(historical["results"][0]["content"], "回答使用英文")
            self.assertEqual(old_record["status"], "superseded")
            self.assertEqual(old_record["valid_to"], "2026-02-01T00:00:00+00:00")
            self.assertEqual(new["supersedes_id"], old["id"])
            self.assertEqual(new["provenance"][0]["source_id"], "new-message")

    def test_hybrid_search_explains_scores_and_uses_mmr(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            service = self._service(Path(temp_dir) / "memory")
            for index, content in enumerate(
                [
                    "PIM 是无源互调，射频系统需要抑制它",
                    "无源互调 PIM 会污染射频接收链路",
                    "Python 项目使用 unittest",
                ]
            ):
                service.upsert(
                    namespace="tenant:user",
                    memory_type="semantic",
                    subject="paperstorm",
                    content=content,
                    canonical_key=f"fact-{index}",
                    source_message_ids=[f"source-{index}"],
                    metadata={"entities": ["PIM"] if index < 2 else ["Python"]},
                )

            result = service.search("tenant:user", "PIM 无源互调", top_k=2)

            self.assertEqual(result["embedding_backend"], "keyword-test")
            self.assertEqual(len(result["results"]), 2)
            self.assertIn("lexical", result["results"][0]["scores"])
            self.assertIn("dense", result["results"][0]["scores"])
            self.assertIn("entity", result["results"][0]["scores"])
            self.assertIn("temporal", result["results"][0]["scores"])
            self.assertTrue(result["results"][0]["retrieval_reasons"])
            self.assertNotIn("Python 项目使用 unittest", [item["content"] for item in result["results"]])


if __name__ == "__main__":
    unittest.main()
