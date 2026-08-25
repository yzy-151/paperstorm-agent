import tempfile
import unittest
from pathlib import Path


class PaperStormContextV56Tests(unittest.TestCase):
    def test_default_profile_exposes_one_million_model_window_with_soft_budget(self):
        from knowledge_storm.context_engine import _ContextEngineConfigBase

        chat = _ContextEngineConfigBase.for_profile("chat")
        research = _ContextEngineConfigBase.for_profile("research")

        self.assertEqual(chat.model_context_tokens, 1_000_000)
        self.assertEqual(chat.operational_input_tokens, 128_000)
        self.assertEqual(research.operational_input_tokens, 512_000)
        self.assertLessEqual(chat.absolute_layer_caps["pinned"], 24_000)
        self.assertEqual(chat.recent_messages, 48)

    def test_structured_summary_prompt_preserves_decisions_constraints_and_sources(self):
        from knowledge_storm.context_engine import build_structured_summary_prompt

        prompt = build_structured_summary_prompt(self._messages())

        self.assertIn('"user_goals"', prompt)
        self.assertIn('"confirmed_facts"', prompt)
        self.assertIn('"open_questions"', prompt)
        self.assertIn('"evidence_refs"', prompt)
        self.assertIn("不得把推测写成事实", prompt)

    def test_summary_selection_uses_query_relevance_instead_of_last_two_only(self):
        engine = self._engine()
        messages = [
            {"id": "s1", "role": "system", "content": "PIM 无源互调论文与神经网络抑制", "metadata": {"context_summary": True}},
            {"id": "s2", "role": "system", "content": "午饭讨论与天气", "metadata": {"context_summary": True}},
            {"id": "s3", "role": "system", "content": "另一个无关项目", "metadata": {"context_summary": True}},
            {"id": "u", "role": "user", "content": "之前的 PIM 论文有哪些？"},
        ]

        result = engine.assemble(messages, query="PIM 无源互调论文")
        summary_text = " ".join(
            item["content"] for item in result["messages"]
            if item.get("metadata", {}).get("context_summary")
        )
        self.assertIn("PIM 无源互调", summary_text)

    def _engine(self, root=None, summarizer=None):
        from knowledge_storm.context_engine import (
            _ContextEngineConfigBase,
            _ContextEngineCore,
            ContextLedger,
        )

        ledger = ContextLedger(Path(root) / "context.sqlite3") if root else None
        config = _ContextEngineConfigBase(
            model_context_tokens=220,
            output_reserve_tokens=40,
            soft_watermark=0.55,
            recent_messages=3,
        )
        return _ContextEngineCore(config=config, ledger=ledger, summarizer=summarizer)

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
        from knowledge_storm.context_engine import ContextEngine, ContextEngineConfig

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
