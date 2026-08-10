import tempfile
import unittest
from pathlib import Path


class PaperStormContextV56Tests(unittest.TestCase):
    def _engine(self, root=None, summarizer=None):
        from knowledge_storm.paperstorm_context_v56 import (
            ContextEngineConfigV56,
            ContextEngineV56,
            ContextLedgerV56,
        )

        ledger = ContextLedgerV56(Path(root) / "context.sqlite3") if root else None
        config = ContextEngineConfigV56(
            model_context_tokens=220,
            output_reserve_tokens=40,
            soft_watermark=0.55,
            recent_messages=3,
        )
        return ContextEngineV56(config=config, ledger=ledger, summarizer=summarizer)

    @staticmethod
    def _messages():
        return [
            {"id": "sys", "role": "system", "content": "所有回答必须使用中文，并保留引用。"},
            {"id": "u1", "role": "user", "content": "调研 PIM 抑制方法。"},
            {"id": "a1", "role": "assistant", "content": "我会先检索论文。"},
            {"id": "call", "role": "assistant", "content": "", "tool_calls": [{"id": "tc1", "name": "search"}]},
            {"id": "result", "role": "tool", "tool_call_id": "tc1", "content": "论文结果 " * 35},
            {"id": "u2", "role": "user", "content": "重点比较神经网络与数字预失真。"},
            {"id": "a2", "role": "assistant", "content": "收到，我会进行对比。"},
        ]

    def test_assemble_protects_pinned_and_respects_typed_budget(self):
        engine = self._engine()
        result = engine.assemble(
            self._messages(),
            memories=[{"content": "用户偏好中文"}],
            evidence=[{"content": "PIM paper evidence", "source_id": "paper-1"}],
        )

        self.assertLessEqual(result["token_usage"]["total"], result["token_usage"]["input_limit"])
        self.assertIn("所有回答必须使用中文", result["messages"][0]["content"])
        self.assertEqual(set(result["token_usage"]["layers"]), {"pinned", "active", "summary", "memory", "evidence", "artifact"})
        self.assertTrue(result["validation"]["pinned_preserved"])

    def test_tool_call_and_result_are_selected_atomically(self):
        engine = self._engine()
        result = engine.assemble(self._messages())
        ids = {message.get("id") for message in result["messages"]}

        self.assertEqual("call" in ids, "result" in ids)
        self.assertTrue(result["validation"]["tool_pairs_valid"])

    def test_recursive_compaction_records_lineage_and_restores_raw_messages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = self._engine(temp_dir)
            first = engine.compact(self._messages(), force=True)
            second = engine.compact(first["messages"] + [
                {"id": "u3", "role": "user", "content": "补充可复现实验设置。"},
                {"id": "a3", "role": "assistant", "content": "将记录数据和指标。"},
            ], force=True)

            self.assertEqual(first["compaction"]["level"], 1)
            self.assertEqual(second["compaction"]["level"], 2)
            self.assertIn(first["compaction"]["compaction_id"], second["compaction"]["parent_ids"])
            restored = engine.restore(second["compaction"]["compaction_id"])
            restored_ids = {message.get("id") for message in restored}
            self.assertTrue({"sys", "u1", "call", "result"}.issubset(restored_ids))

    def test_broken_summarizer_falls_back_without_losing_constraints(self):
        def broken(_messages):
            raise RuntimeError("summarizer unavailable")

        engine = self._engine(summarizer=broken)
        result = engine.compact(self._messages(), force=True)

        self.assertEqual(result["compaction"]["strategy"], "deterministic_fallback")
        self.assertTrue(result["validation"]["pinned_preserved"])
        self.assertIn("所有回答必须使用中文", " ".join(item.get("content", "") for item in result["messages"]))

    def test_legacy_runtime_contract_is_backed_by_v56_engine(self):
        from knowledge_storm.paperstorm_context_v56 import ContextEngine, ContextEngineConfig

        config = ContextEngineConfig(
            total_tokens=220,
            output_reserve_tokens=40,
            compact_threshold_ratio=0.55,
            recent_message_count=3,
        )
        engine = ContextEngine(config=config)
        compacted = engine.compact(
            self._messages(), expected_constraints=["中文"], force=True
        )
        assembled = engine.assemble(
            compacted["messages"],
            memory=[{"role": "system", "content": "用户偏好中文"}],
            rag_evidence=[{"role": "system", "content": "PIM evidence"}],
        )

        self.assertEqual(compacted["status"], "compacted")
        self.assertTrue(compacted["validation"]["passed"])
        self.assertTrue(compacted["compaction_id"])
        self.assertIn("meter", assembled)
        self.assertIn("compaction", assembled)


if __name__ == "__main__":
    unittest.main()
