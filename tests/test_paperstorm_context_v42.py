import json
import tempfile
import unittest
from pathlib import Path


class PaperStormContextV42Test(unittest.TestCase):
    def _messages(self):
        return [
            {"id": "system", "role": "system", "content": "回答必须使用中文并保留引用。"},
            {"id": "goal", "role": "user", "content": "目标：调研 PIM 神经网络抑制。"},
            {"id": "assistant-1", "role": "assistant", "content": "我会先检索论文。"},
            {
                "id": "tool-1",
                "role": "tool",
                "name": "arxiv_search",
                "tool_call_id": "call-1",
                "content": "passive intermodulation evidence " * 120,
            },
            {"id": "assistant-2", "role": "assistant", "content": "决定：排除 DRAM，保留 RF 论文。"},
            {"id": "user-2", "role": "user", "content": "待办：比较 BM25 与 Dense。"},
            {"id": "assistant-3", "role": "assistant", "content": "正在比较，下一步运行 Reranker。"},
        ]

    def test_event_store_is_append_only_and_restores_exact_raw_messages(self):
        from knowledge_storm.paperstorm_context_v42 import ContextEventStore

        with tempfile.TemporaryDirectory() as temp_dir:
            store = ContextEventStore(Path(temp_dir) / "context.jsonl")
            for message in self._messages():
                store.append_message(message)
            before = store.read_events()
            store.append_compaction(
                {
                    "compaction_id": "compact-1",
                    "source_event_ids": [item["event_id"] for item in before],
                    "summary": {"goal": "调研 PIM"},
                }
            )
            restored = store.restore_messages("compact-1")
            after = store.read_events()

        self.assertEqual(len(before) + 1, len(after))
        self.assertEqual([item["message"] for item in before], restored)
        self.assertEqual(after[-1]["event_type"], "compaction")

    def test_engine_compacts_by_tokens_and_preserves_goal_constraints_recent_messages(self):
        from knowledge_storm.paperstorm_context_v42 import ContextEngine, ContextEngineConfig

        engine = ContextEngine(
            config=ContextEngineConfig(
                total_tokens=220,
                output_reserve_tokens=40,
                compact_threshold_ratio=0.55,
                high_watermark_ratio=0.8,
                recent_message_count=2,
                tool_inline_token_limit=20,
            )
        )
        decision = engine.should_compact(self._messages())
        compacted = engine.compact(
            self._messages(),
            expected_constraints=["中文", "引用", "DRAM", "RF"],
        )

        self.assertTrue(decision["should_compact"])
        self.assertLess(compacted["after_tokens"], compacted["before_tokens"])
        self.assertEqual(compacted["summary"]["goal"], "目标：调研 PIM 神经网络抑制。")
        self.assertTrue(compacted["validation"]["passed"])
        self.assertTrue(compacted["artifact_refs"])
        self.assertIn("context://message/tool-1", compacted["artifact_refs"][0]["uri"])
        compacted_ids = [item.get("id") for item in compacted["messages"]]
        self.assertIn("system", compacted_ids)
        self.assertIn("goal", compacted_ids)
        self.assertIn("assistant-3", compacted_ids)

    def test_summary_schema_contains_handoff_fields_and_source_range(self):
        from knowledge_storm.paperstorm_context_v42 import ContextEngine

        result = ContextEngine().compact(self._messages(), force=True)
        summary = result["summary"]

        for field in [
            "goal",
            "constraints",
            "completed",
            "in_progress",
            "decisions",
            "entities",
            "sources",
            "errors",
            "todos",
            "source_message_ids",
        ]:
            self.assertIn(field, summary)

    def test_compaction_failure_falls_back_to_original_messages(self):
        from knowledge_storm.paperstorm_context_v42 import ContextEngine

        def broken_summarizer(_messages):
            raise RuntimeError("summary backend unavailable")

        engine = ContextEngine(summarizer=broken_summarizer)
        result = engine.compact(self._messages(), force=True)

        self.assertEqual(result["status"], "fallback_original")
        self.assertEqual(result["messages"], self._messages())
        self.assertIn("summary backend unavailable", result["error"])

    def test_assemble_uses_dynamic_budget_and_stays_below_input_limit(self):
        from knowledge_storm.paperstorm_context_v42 import ContextEngine, ContextEngineConfig

        engine = ContextEngine(
            config=ContextEngineConfig(total_tokens=260, output_reserve_tokens=60)
        )
        assembled = engine.assemble(
            messages=self._messages(),
            memory=[{"role": "system", "content": "用户偏好：中文。" * 30}],
            rag_evidence=[{"role": "system", "content": "论文证据。" * 80}],
            tool_schemas=[{"name": "arxiv_search", "description": "检索论文" * 30}],
        )

        self.assertLessEqual(assembled["meter"]["input_tokens"], 200)
        self.assertEqual(assembled["meter"]["output_reserve_tokens"], 60)
        self.assertIn("allocation", assembled["meter"])
        self.assertTrue(assembled["messages"])
        assembled_ids = [item.get("id") for item in assembled["messages"]]
        self.assertIn("goal", assembled_ids)
        self.assertIn("assistant-3", assembled_ids)

    def test_context_benchmark_scores_savings_constraints_and_restore(self):
        from knowledge_storm.paperstorm_context_benchmark_v42 import run_context_benchmark

        with tempfile.TemporaryDirectory() as temp_dir:
            report = run_context_benchmark(temp_dir)
            repeated = run_context_benchmark(temp_dir)
            saved = json.loads(
                (Path(temp_dir) / "context_benchmark_v42.json").read_text(encoding="utf-8")
            )

        self.assertGreater(report["metrics"]["token_savings_rate"], 0)
        self.assertEqual(report["metrics"]["constraint_retention_rate"], 1.0)
        self.assertEqual(report["metrics"]["restore_exact"], 1.0)
        self.assertEqual(repeated["metrics"]["restore_exact"], 1.0)
        self.assertNotEqual(report["run_id"], repeated["run_id"])
        self.assertEqual(report["metrics"]["repeated_compaction_retention_rate"], 1.0)
        self.assertEqual(saved["project"], "PaperStorm Context Benchmark v4.2")

    def test_chat_service_exposes_context_meter_compaction_and_restore(self):
        from knowledge_storm.paperstorm_service import PaperStormTaskService

        with tempfile.TemporaryDirectory() as temp_dir:
            service = PaperStormTaskService(temp_dir)
            session = service.create_chat_session(
                topic="PIM",
                run_mode="fake",
                context_window_size=3,
                context_token_limit=180,
            )
            for question in ["你好", "你是谁", "上下文怎么压缩", "网页怎么使用"]:
                reply = service.send_chat_message(session["chat_id"], question)
            context = service.get_chat_context(session["chat_id"])
            compacted = service.compact_chat_context(session["chat_id"], force=True)
            restored = service.restore_chat_context(
                session["chat_id"], compacted["compaction_id"]
            )

        self.assertIn("context_meter", reply)
        self.assertTrue(context["raw_event_count"] >= 8)
        self.assertTrue(compacted["compaction_id"])
        self.assertGreater(compacted["context_meter"]["raw_usage_ratio"], 1.0)
        self.assertLessEqual(compacted["context_meter"]["usage_ratio"], 0.9)
        self.assertGreaterEqual(len(restored["messages"]), 8)

    def test_fastapi_and_dashboard_expose_v42_context_controls(self):
        from fastapi.testclient import TestClient

        from examples.storm_examples.paperstorm_service_api import create_app

        with tempfile.TemporaryDirectory() as temp_dir:
            client = TestClient(create_app(service_root=Path(temp_dir)))
            created = client.post(
                "/chat/sessions",
                json={"topic": "PIM", "run_mode": "fake", "context_token_limit": 180},
            ).json()
            chat_id = created["chat_id"]
            client.post("/chat/sessions/{0}/messages".format(chat_id), json={"message": "你好"})
            context = client.get("/chat/sessions/{0}/context".format(chat_id))
            compact = client.post(
                "/chat/sessions/{0}/context/compact".format(chat_id), json={"force": True}
            )
            benchmark = client.post("/evaluations/context-v42")

        root = Path(__file__).resolve().parents[1]
        index = (root / "frontend/paperstorm_dashboard/index.html").read_text(encoding="utf-8")
        script = (root / "frontend/paperstorm_dashboard/app.js").read_text(encoding="utf-8")
        self.assertEqual(context.status_code, 200)
        self.assertEqual(compact.status_code, 200)
        self.assertEqual(benchmark.status_code, 200)
        self.assertIn("v4.2", index)
        self.assertIn("context-meter", index)
        self.assertIn("compact-chat-context", index)
        self.assertIn("/evaluations/context-v42", script)


if __name__ == "__main__":
    unittest.main()
