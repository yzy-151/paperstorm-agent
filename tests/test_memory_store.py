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


class CountingEmbedding(KeywordEmbedding):
    def __init__(self):
        self.embed_calls = 0
        self.embed_query_calls = 0

    def embed(self, texts):
        self.embed_calls += 1
        return super().embed(texts)

    def embed_query(self, text):
        self.embed_query_calls += 1
        return super().embed_query(text)


class PaperStormMemoryV56Tests(unittest.TestCase):
    def test_lexical_mode_never_calls_embedding_provider(self):
        from knowledge_storm.memory_store import LongTermMemoryService

        with tempfile.TemporaryDirectory() as temp_dir:
            provider = CountingEmbedding()
            service = LongTermMemoryService(
                Path(temp_dir) / "memory",
                embedding_provider=provider,
                retrieval_mode="lexical",
            )
            service.upsert(
                namespace="tenant:user",
                memory_type="fact",
                subject="pim",
                content="PIM 指无源互调",
                canonical_key="pim",
            )
            result = service.search("tenant:user", "PIM", top_k=1)

        self.assertEqual(provider.embed_calls, 0)
        self.assertEqual(provider.embed_query_calls, 0)
        self.assertEqual(result["retrieval_mode"], "lexical")
        self.assertEqual(result["embedding_backend"], "disabled")
        self.assertNotIn("dense", result["results"][0]["scores"])

    def test_semantic_mode_rejects_hash_embedding(self):
        from knowledge_storm.memory_store import LongTermMemoryService
        from knowledge_storm.retrieval import HashEmbeddingProvider

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "real semantic embedding"):
                LongTermMemoryService(
                    Path(temp_dir) / "memory",
                    embedding_provider=HashEmbeddingProvider(64),
                    retrieval_mode="semantic",
                )

    def test_semantic_mode_reports_real_provider_name(self):
        from knowledge_storm.memory_store import LongTermMemoryService

        with tempfile.TemporaryDirectory() as temp_dir:
            service = LongTermMemoryService(
                Path(temp_dir) / "memory",
                embedding_provider=KeywordEmbedding(),
                retrieval_mode="semantic",
            )
            service.upsert(
                namespace="tenant:user",
                memory_type="fact",
                subject="pim",
                content="PIM 指无源互调",
                canonical_key="pim",
            )
            result = service.search("tenant:user", "无源互调", top_k=1)

        self.assertEqual(result["retrieval_mode"], "semantic")
        self.assertEqual(result["embedding_backend"], "keyword-test")
        self.assertIn("dense", result["results"][0]["scores"])

    def test_llm_candidate_extractor_is_validated_before_durable_write(self):
        from knowledge_storm.memory_store import LongTermMemoryService

        def extractor(_prompt):
            return {
                "memory_type": "preference",
                "subject": "user",
                "content": "用户偏好中文回答",
                "canonical_key": "answer_language",
                "confidence": 0.94,
                "importance": 0.8,
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            service = LongTermMemoryService(
                Path(temp_dir) / "memory",
                embedding_provider=KeywordEmbedding(),
                candidate_extractor=extractor,
            )
            result = service.ingest_message(
                "tenant:user", "以后回答简洁一些，用中文。", "message-1"
            )

            self.assertEqual(result["status"], "persisted")
            self.assertEqual(result["memory"]["canonical_key"], "answer_language")
            self.assertEqual(result["memory"]["metadata"]["extractor"], "llm_structured")

    def test_negative_memory_instruction_blocks_llm_candidate(self):
        from knowledge_storm.memory_store import LongTermMemoryService

        with tempfile.TemporaryDirectory() as temp_dir:
            service = LongTermMemoryService(
                Path(temp_dir) / "memory",
                embedding_provider=KeywordEmbedding(),
                candidate_extractor=lambda _prompt: {
                    "memory_type": "semantic",
                    "subject": "user",
                    "content": "不应写入",
                    "canonical_key": "blocked",
                    "confidence": 0.99,
                },
            )
            result = service.ingest_message(
                "tenant:user", "不要记住下面这句话。", "message-2"
            )

            self.assertEqual(result["status"], "skipped")
            self.assertEqual(service.list_memories("tenant:user"), [])

    def _service(self, root):
        from knowledge_storm.memory_store import LongTermMemoryService

        return LongTermMemoryService(root, embedding_provider=KeywordEmbedding())

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

    def test_fact_vectors_persist_and_are_reused_across_queries(self):
        from knowledge_storm.memory_store import LongTermMemoryService

        with tempfile.TemporaryDirectory() as temp_dir:
            provider = CountingEmbedding()
            service = LongTermMemoryService(Path(temp_dir) / "memory", embedding_provider=provider)
            service.upsert(
                namespace="tenant:user",
                memory_type="fact",
                subject="pim",
                content="PIM 指无源互调，中文射频场景常见。",
                canonical_key="pim",
                valid_from="2026-01-01T00:00:00+00:00",
            )
            service.upsert(
                namespace="tenant:user",
                memory_type="fact",
                subject="vlc",
                content="VLC 是可见光通信。",
                canonical_key="vlc",
                valid_from="2026-01-01T00:00:00+00:00",
            )
            provider.embed_calls = 0
            provider.embed_query_calls = 0

            first = service.search("tenant:user", "无源互调是什么", top_k=1)
            first_id = first["results"][0]["id"]
            embed_calls_after_first = provider.embed_calls

            second = service.search("tenant:user", "无源互调是什么", top_k=1)
            self.assertEqual(second["results"][0]["id"], first_id)
            self.assertEqual(provider.embed_calls, embed_calls_after_first)
            self.assertGreaterEqual(provider.embed_query_calls, 2)

            service2 = LongTermMemoryService(
                Path(temp_dir) / "memory", embedding_provider=CountingEmbedding()
            )
            same = service2.search("tenant:user", "无源互调是什么", top_k=1)
            self.assertEqual(same["results"][0]["id"], first_id)
            self.assertEqual(service2.embedding_provider.embed_calls, 0)

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
