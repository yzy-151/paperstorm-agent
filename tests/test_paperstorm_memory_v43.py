import tempfile
import unittest
from pathlib import Path


class PaperStormMemoryV43Test(unittest.TestCase):
    def make_service(self, temp_dir):
        from knowledge_storm.paperstorm_memory_v43 import LongTermMemoryService

        return LongTermMemoryService(Path(temp_dir) / "memory_service")

    def test_policy_only_persists_explicit_long_term_candidates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            service = self.make_service(temp_dir)

            skipped = service.ingest_message(
                namespace="user/alice",
                message="今天北京天气怎么样？",
                source_message_id="m-ordinary",
            )
            remembered = service.ingest_message(
                namespace="user/alice",
                message="请记住：PIM 在这个项目里指 passive intermodulation。",
                source_message_id="m-memory",
            )

            self.assertEqual(skipped["status"], "skipped")
            self.assertEqual(remembered["status"], "persisted")
            self.assertEqual(remembered["memory"]["memory_type"], "semantic")
            self.assertEqual(remembered["memory"]["source_message_ids"], ["m-memory"])
            self.assertEqual(len(service.list_memories("user/alice")), 1)

    def test_conflict_supersedes_old_fact_without_destroying_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            service = self.make_service(temp_dir)
            old = service.upsert(
                namespace="user/alice",
                memory_type="preference",
                subject="alice",
                content="回答使用英文。",
                canonical_key="response_language",
                source_message_ids=["m1"],
            )
            new = service.upsert(
                namespace="user/alice",
                memory_type="preference",
                subject="alice",
                content="回答使用中文。",
                canonical_key="response_language",
                source_message_ids=["m2"],
            )

            active = service.list_memories("user/alice")
            history = service.list_memories("user/alice", include_inactive=True)
            restored_old = next(item for item in history if item["id"] == old["id"])

            self.assertEqual(len(active), 1)
            self.assertEqual(active[0]["id"], new["id"])
            self.assertEqual(restored_old["status"], "superseded")
            self.assertEqual(new["supersedes_id"], old["id"])
            self.assertGreaterEqual(len(service.audit_events()), 3)

    def test_duplicate_write_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            service = self.make_service(temp_dir)
            first = service.upsert(
                namespace="user/alice",
                memory_type="semantic",
                subject="PaperStorm",
                content="PIM 指 passive intermodulation。",
                canonical_key="term:pim",
            )
            second = service.upsert(
                namespace="user/alice",
                memory_type="semantic",
                subject="PaperStorm",
                content="PIM 指 passive intermodulation。",
                canonical_key="term:pim",
            )

            self.assertEqual(first["id"], second["id"])
            self.assertTrue(second["deduplicated"])
            self.assertEqual(len(service.list_memories("user/alice")), 1)

    def test_hybrid_recall_filters_namespace_expiration_and_deleted_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            service = self.make_service(temp_dir)
            target = service.upsert(
                namespace="user/alice",
                memory_type="semantic",
                subject="PaperStorm",
                content="PIM 是射频无源互调 passive intermodulation。",
                canonical_key="term:pim",
                importance=0.9,
            )
            service.upsert(
                namespace="user/bob",
                memory_type="semantic",
                subject="Bob",
                content="Bob 的 PIM 表示 processing-in-memory。",
                canonical_key="term:pim",
            )
            deleted = service.upsert(
                namespace="user/alice",
                memory_type="episodic",
                subject="old-task",
                content="旧的 DRAM 调研任务。",
                canonical_key="episode:dram",
            )
            service.delete("user/alice", deleted["id"], reason="irrelevant")
            service.upsert(
                namespace="user/alice",
                memory_type="semantic",
                subject="expired",
                content="过期的 PIM 定义。",
                canonical_key="term:expired-pim",
                expires_at="2000-01-01T00:00:00+00:00",
            )

            result = service.search("user/alice", "PIM 无源互调 RF", top_k=5)

            self.assertEqual(result["results"][0]["id"], target["id"])
            self.assertTrue(result["results"][0]["scores"]["rrf"] > 0)
            self.assertTrue(all(item["namespace"] == "user/alice" for item in result["results"]))
            self.assertTrue(all(item["status"] == "active" for item in result["results"]))
            self.assertNotIn("DRAM", " ".join(item["content"] for item in result["results"]))

            unrelated = service.search("user/alice", "明天北京天气", top_k=5)
            self.assertEqual(unrelated["results"], [])

    def test_pending_consolidation_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            service = self.make_service(temp_dir)
            queued = service.ingest_message(
                "user/alice",
                "可能要记住：PIM 项目使用射频论文。",
                source_message_id="pending-1",
            )
            first = service.consolidate_pending()
            second = service.consolidate_pending()

            self.assertEqual(queued["status"], "queued")
            self.assertEqual(first["processed"], 1)
            self.assertEqual(second["processed"], 0)

    def test_edit_delete_export_and_memory_switch_are_governed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            service = self.make_service(temp_dir)
            record = service.upsert(
                namespace="user/alice",
                memory_type="procedural",
                subject="retrieval",
                content="检索 PIM 时先做 RF 消歧。",
                canonical_key="procedure:pim-retrieval",
            )
            edited = service.edit(
                namespace="user/alice",
                memory_id=record["id"],
                content="检索 PIM 时先做 RF 消歧并排除 DRAM。",
            )
            service.set_enabled("user/alice", False)
            disabled = service.search("user/alice", "PIM")
            exported = service.export_namespace("user/alice")
            service.delete("user/alice", edited["id"], reason="user_request")

            self.assertNotEqual(edited["id"], record["id"])
            self.assertEqual(disabled["status"], "disabled")
            self.assertEqual(exported["namespace"], "user/alice")
            self.assertGreaterEqual(len(exported["memories"]), 2)
            self.assertEqual(service.list_memories("user/alice"), [])

    def test_cross_session_chat_recall_uses_same_user_namespace(self):
        from knowledge_storm.paperstorm_service import PaperStormTaskService

        with tempfile.TemporaryDirectory() as temp_dir:
            service = PaperStormTaskService(Path(temp_dir))
            first = service.create_chat_session(
                topic="PaperStorm",
                run_mode="fake",
                user_id="alice",
            )
            remembered = service.send_chat_message(
                first["chat_id"],
                "请记住：回答我时使用中文。",
            )
            second = service.create_chat_session(
                topic="PaperStorm",
                run_mode="fake",
                user_id="alice",
            )
            recalled = service.send_chat_message(second["chat_id"], "你记得我的回答偏好吗？")
            third = service.create_chat_session(
                topic="PaperStorm",
                run_mode="fake",
                user_id="bob",
            )
            isolated = service.send_chat_message(third["chat_id"], "你记得我的回答偏好吗？")

            self.assertEqual(remembered["memory_write"]["status"], "persisted")
            self.assertTrue(any("中文" in item["content"] for item in recalled["long_term_memory"]["results"]))
            self.assertIn("中文", recalled["assistant_message"]["content"])
            self.assertFalse(recalled["retrieval_triggered"])
            self.assertEqual(recalled["router_decision"]["tool"], "memory_search")
            self.assertFalse(any("中文" in item["content"] for item in isolated["long_term_memory"]["results"]))

    def test_non_ascii_user_id_gets_stable_safe_namespace(self):
        from knowledge_storm.paperstorm_service import PaperStormTaskService

        with tempfile.TemporaryDirectory() as temp_dir:
            service = PaperStormTaskService(Path(temp_dir))
            first = service.create_chat_session(user_id="张三", run_mode="fake")
            second = service.create_chat_session(user_id="张三", run_mode="fake")

            self.assertTrue(first["memory_namespace"].startswith("user/user-"))
            self.assertEqual(first["memory_namespace"], second["memory_namespace"])

    def test_runtime_memory_operations_emit_trace_events(self):
        from knowledge_storm.paperstorm_memory_v43 import LongTermMemoryService
        from knowledge_storm.paperstorm_runtime import PaperStormRuntimeSession

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            memory = LongTermMemoryService(root / "memory")
            runtime = PaperStormRuntimeSession(
                run_id="memory-runtime",
                trace_path=root / "trace.jsonl",
                long_term_memory=memory,
                memory_namespace="user/alice",
            )
            written = runtime.remember("请记住：PIM 指 passive intermodulation。", "m1")
            recalled = runtime.recall_memory("PIM 无源互调", top_k=3)

            events = [
                __import__("json").loads(line)
                for line in (root / "trace.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(written["status"], "persisted")
            self.assertEqual(recalled["results"][0]["canonical_key"], "term:pim")
            self.assertEqual([item["event"] for item in events], ["memory_write", "memory_recall"])

    def test_memory_benchmark_is_reproducible_and_isolated(self):
        from knowledge_storm.paperstorm_memory_benchmark_v43 import run_memory_benchmark

        with tempfile.TemporaryDirectory() as temp_dir:
            first = run_memory_benchmark(Path(temp_dir))
            second = run_memory_benchmark(Path(temp_dir))

            self.assertNotEqual(first["run_id"], second["run_id"])
            self.assertEqual(first["metrics"]["memory_write_precision"], 1.0)
            self.assertEqual(first["metrics"]["memory_recall_at_k"], 1.0)
            self.assertEqual(first["metrics"]["stale_fact_misuse_rate"], 0.0)
            self.assertEqual(first["metrics"]["cross_namespace_leakage_rate"], 0.0)
            self.assertEqual(first["metrics"]["duplicate_rate"], 0.0)
            self.assertGreaterEqual(first["metrics"]["memory_enabled_task_success"], first["metrics"]["memory_disabled_task_success"])
            self.assertGreater(first["metrics"]["background_throughput_per_second"], 0)
            self.assertIn("baseline", first)
            self.assertGreater(first["baseline"]["duplicate_rate"], 0.0)
            self.assertGreater(
                first["metrics"]["memory_recall_at_k"],
                first["baseline"]["recall_at_k"],
            )

    def test_fastapi_exposes_memory_governance_and_benchmark(self):
        try:
            from fastapi.testclient import TestClient
        except Exception as exc:  # pragma: no cover - optional dependency
            self.skipTest(str(exc))
        from examples.storm_examples.paperstorm_service_api import create_app

        with tempfile.TemporaryDirectory() as temp_dir:
            client = TestClient(create_app(service_root=Path(temp_dir)))
            created = client.post(
                "/memories",
                json={
                    "namespace": "user/alice",
                    "memory_type": "semantic",
                    "subject": "PaperStorm",
                    "content": "PIM 指 passive intermodulation。",
                    "canonical_key": "term:pim",
                },
            )
            recalled = client.post(
                "/memories/search",
                json={"namespace": "user/alice", "query": "PIM", "top_k": 3},
            )
            exported = client.get("/memories/export", params={"namespace": "user/alice"})
            benchmark = client.post("/evaluations/memory-v43")

        root = Path(__file__).resolve().parents[1]
        index = (root / "frontend/paperstorm_dashboard/index.html").read_text(encoding="utf-8")
        script = (root / "frontend/paperstorm_dashboard/app.js").read_text(encoding="utf-8")
        self.assertEqual(created.status_code, 200)
        self.assertEqual(recalled.json()["results"][0]["canonical_key"], "term:pim")
        self.assertEqual(exported.json()["namespace"], "user/alice")
        self.assertEqual(benchmark.status_code, 200)
        self.assertIn("v4.3", index)
        self.assertIn("chat-memory-enabled", index)
        self.assertIn("/evaluations/memory-v43", script)
        self.assertIn("/memories/search", script)


if __name__ == "__main__":
    unittest.main()
