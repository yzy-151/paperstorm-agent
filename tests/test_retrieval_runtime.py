import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class PaperStormRetrievalRuntimeTest(unittest.TestCase):
    def test_auto_embedding_resolves_to_real_provider(self):
        import builtins

        from knowledge_storm.retrieval_runtime import runtime_embedding

        real_import = builtins.__import__

        def import_without_sentence_transformers(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        with mock.patch.dict(os.environ, {}, clear=True), mock.patch(
            "builtins.__import__", side_effect=import_without_sentence_transformers
        ):
            self.assertEqual("real", runtime_embedding("auto"))

    def test_legacy_stack_is_rejected(self):
        from knowledge_storm.retrieval_runtime import runtime_stack

        with self.assertRaisesRegex(ValueError, "legacy retrieval stacks were removed"):
            runtime_stack("legacy")

    def test_kb_search_uses_unified_pipeline_and_reports_stages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            (run_dir / "storm_gen_article_polished.txt").write_text(
                "PIM 在本任务中指 passive intermodulation，是 RF 系统中由无源器件非线性导致的互调杂散问题。",
                encoding="utf-8",
            )
            (run_dir / "raw_search_results.json").write_text(
                json.dumps(
                    [
                        {
                            "title": "Neural PIM cancellation",
                            "description": "RF passive intermodulation suppression with neural networks.",
                            "url": "https://example.com/pim",
                            "snippets": ["PIM passive intermodulation neural suppression."],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            from knowledge_storm.paperstorm_qa import PaperStormKnowledgeBase

            kb = PaperStormKnowledgeBase.from_run_dir(run_dir)
            with mock.patch.dict(
                os.environ,
                {
                    "PAPERSTORM_RETRIEVAL_EMBEDDING": "hash",
                    "PAPERSTORM_RETRIEVAL_MODE": "hybrid",
                },
            ):
                evidence = kb.search("PIM 是什么？", top_k=3)
            self.assertTrue(evidence)
            self.assertEqual(kb.retrieval_meta["stack"], "retrieval_pipeline")
            self.assertEqual(kb.retrieval_meta["mode"], "hybrid")
            self.assertEqual(
                ["retrieve", "fuse", "rerank", "gate"],
                [stage["name"] for stage in kb.retrieval_meta["stages"]],
            )
            self.assertIn("score", evidence[0])
            self.assertIn("rrf_score", evidence[0])

    def test_runtime_search_exposes_unified_stack(self):
        from knowledge_storm.retrieval_runtime import search_runtime_index

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            (run_dir / "storm_gen_article.txt").write_text(
                "PIM passive intermodulation neural network suppression.",
                encoding="utf-8",
            )
            (run_dir / "raw_search_results.json").write_text("[]", encoding="utf-8")
            outcome = search_runtime_index(
                run_dir,
                "PIM 神经网络抑制",
                top_k=2,
                embedding="hash",
            )
            self.assertEqual(outcome["stack"], "retrieval_pipeline")
            self.assertEqual(outcome["mode"], "hybrid")

    def test_runtime_index_is_lru_cached_and_invalidated_on_file_change(self):
        from knowledge_storm.retrieval_runtime import search_runtime_index

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            (run_dir / "storm_gen_article.txt").write_text(
                "PIM passive intermodulation neural network suppression.",
                encoding="utf-8",
            )
            (run_dir / "raw_search_results.json").write_text("[]", encoding="utf-8")
            first = search_runtime_index(
                run_dir, "PIM 是什么？", top_k=2, embedding="hash"
            )
            second = search_runtime_index(
                run_dir, "PIM 是什么？", top_k=2, embedding="hash"
            )
            self.assertFalse(first["cached"])
            self.assertTrue(second["cached"])
            (run_dir / "storm_gen_article.txt").write_text(
                "PIM article changed with new content.",
                encoding="utf-8",
            )
            third = search_runtime_index(
                run_dir, "PIM 是什么？", top_k=2, embedding="hash"
            )
            self.assertFalse(third["cached"])


if __name__ == "__main__":
    unittest.main()
