import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class PaperStormRetrievalRuntimeTest(unittest.TestCase):
    def test_benchmark_reports_v41_improvement_over_legacy(self):
        from knowledge_storm.paperstorm_retrieval_runtime import run_retrieval_benchmark

        with tempfile.TemporaryDirectory() as temp_dir:
            report = run_retrieval_benchmark(
                Path(temp_dir),
                top_k=5,
                embedding="hash",
            )
            self.assertGreater(
                report["v41"]["recall_at_k"],
                report["legacy"]["recall_at_k"],
            )
            self.assertGreaterEqual(report["v41"]["recall_at_k"], 0.6)
            self.assertGreater(report["deltas"]["relative_recall_gain_pct"], 0)
            self.assertGreater(report["v41"]["mrr"], report["legacy"]["mrr"])
            self.assertTrue(
                (Path(temp_dir) / "retrieval_runtime_benchmark.json").exists()
            )
            self.assertTrue(
                (Path(temp_dir) / "retrieval_runtime_benchmark.md").exists()
            )

    def test_kb_search_uses_v41_stack_and_reports_mode(self):
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
                    "PAPERSTORM_RETRIEVAL_STACK": "v41",
                    "PAPERSTORM_RETRIEVAL_EMBEDDING": "hash",
                    "PAPERSTORM_RETRIEVAL_MODE": "hybrid",
                },
            ):
                evidence = kb.search("PIM 是什么？", top_k=3)
            self.assertTrue(evidence)
            self.assertEqual(kb.retrieval_meta["stack"], "v41")
            self.assertEqual(kb.retrieval_meta["mode"], "hybrid")
            self.assertIn("score", evidence[0])
            self.assertIn("rrf_score", evidence[0])

    def test_legacy_stack_still_works_as_fallback(self):
        from knowledge_storm.paperstorm_retrieval_runtime import search_runtime_index

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
                stack="legacy",
                embedding="hash",
            )
            self.assertEqual(outcome["stack"], "legacy")
            self.assertEqual(outcome["mode"], "legacy_hybrid")

    def test_runtime_index_is_lru_cached_and_invalidated_on_file_change(self):
        from knowledge_storm.paperstorm_retrieval_runtime import search_runtime_index

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            (run_dir / "storm_gen_article.txt").write_text(
                "PIM passive intermodulation neural network suppression.",
                encoding="utf-8",
            )
            (run_dir / "raw_search_results.json").write_text("[]", encoding="utf-8")
            first = search_runtime_index(
                run_dir, "PIM 是什么？", top_k=2, stack="legacy", embedding="hash"
            )
            second = search_runtime_index(
                run_dir, "PIM 是什么？", top_k=2, stack="legacy", embedding="hash"
            )
            self.assertFalse(first["cached"])
            self.assertTrue(second["cached"])
            (run_dir / "storm_gen_article.txt").write_text(
                "PIM article changed with new content.",
                encoding="utf-8",
            )
            third = search_runtime_index(
                run_dir, "PIM 是什么？", top_k=2, stack="legacy", embedding="hash"
            )
            self.assertFalse(third["cached"])


if __name__ == "__main__":
    unittest.main()
