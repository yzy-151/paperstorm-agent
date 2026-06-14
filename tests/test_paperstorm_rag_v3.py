import json
import tempfile
import unittest
from pathlib import Path


class PaperStormRAGV3Test(unittest.TestCase):
    def make_run_dir(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        run_dir = Path(temp_dir.name)
        (run_dir / "storm_gen_article_polished.txt").write_text(
            "# PIM\n\n"
            "Passive intermodulation is an RF nonlinearity problem caused by passive devices.\n\n"
            "Neural network cancellers can suppress passive intermodulation products.\n\n"
            "Processing-in-memory accelerators use DRAM and are unrelated in this RF task.",
            encoding="utf-8",
        )
        (run_dir / "raw_search_results.json").write_text(
            json.dumps(
                [
                    {
                        "title": "Neural passive intermodulation cancellation",
                        "description": "RF neural cancellation for passive intermodulation.",
                        "url": "https://example.com/pim",
                        "snippets": ["Neural cancellers reduce passive intermodulation."],
                    },
                    {
                        "title": "Processing in memory with DRAM",
                        "description": "DRAM accelerator.",
                        "url": "https://example.com/dram",
                        "snippets": ["Processing-in-memory uses RAM."],
                    },
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return run_dir

    def test_rag_index_chunks_embeds_persists_and_hybrid_searches(self):
        from knowledge_storm.paperstorm_rag import PaperStormRAGIndex

        run_dir = self.make_run_dir()
        index = PaperStormRAGIndex.from_run_dir(
            run_dir,
            chunk_size=90,
            chunk_overlap=20,
            embedding_dim=32,
        )
        saved_path = index.save(run_dir / "paperstorm_rag_index.json")
        loaded = PaperStormRAGIndex.load(saved_path)

        results = loaded.search(
            "PIM neural network RF suppression",
            top_k=3,
            expected_keywords=["passive intermodulation", "RF"],
            forbidden_keywords=["DRAM", "processing-in-memory"],
        )

        self.assertTrue(saved_path.exists())
        self.assertTrue(loaded.chunks)
        self.assertIn("embedding", loaded.chunks[0])
        self.assertIn("chunk_id", loaded.chunks[0])
        self.assertGreater(results[0]["hybrid_score"], 0)
        self.assertIn("vector_score", results[0])
        self.assertIn("lexical_score", results[0])
        self.assertIn("rerank_score", results[0])
        self.assertNotIn("DRAM accelerator", results[0]["title"])

    def test_context_compression_retriever_applies_budget_and_audit(self):
        from knowledge_storm.paperstorm_rag import (
            ContextCompressionRetriever,
            PaperStormRAGIndex,
        )

        index = PaperStormRAGIndex.from_run_dir(self.make_run_dir(), chunk_size=80, chunk_overlap=10)
        retriever = ContextCompressionRetriever(
            index,
            max_context_chars=360,
            history_ratio=0.3,
            evidence_ratio=0.7,
        )

        result = retriever.retrieve(
            "PIM 神经网络如何抑制？",
            history=[
                {"role": "user", "content": "之前我们讨论 PIM 指 passive intermodulation。"},
                {"role": "assistant", "content": "需要避免 DRAM processing-in-memory 跑题。"},
            ],
            expected_keywords=["passive intermodulation"],
            forbidden_keywords=["DRAM"],
        )

        self.assertLessEqual(len(result["prompt_context"]), 360)
        self.assertEqual(result["budget"]["history_ratio"], 0.3)
        self.assertTrue(result["chunks"])
        self.assertTrue(result["audit"]["coarse_filtered_count"] >= 0)
        self.assertIn("compressed_evidence", result)

    def test_long_term_memory_index_recalls_across_sessions(self):
        from knowledge_storm.paperstorm_memory import PaperStormMemoryStore
        from knowledge_storm.paperstorm_rag import PaperStormLongTermMemoryIndex

        store = PaperStormMemoryStore()
        store.remember_semantic("用户确认 PIM 在射频任务中指 passive intermodulation。")
        store.remember_episode("上次 arXiv 检索需要过滤 processing-in-memory 和 DRAM。")
        store.set_preference("output_language", "zh")

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "long_term_memory.json"
            index = PaperStormLongTermMemoryIndex.from_memory_store(store)
            index.save(path)
            recalled = PaperStormLongTermMemoryIndex.load(path).recall("PIM RF", top_k=2)

        self.assertTrue(recalled)
        self.assertEqual(recalled[0]["kind"], "semantic")
        self.assertIn("passive intermodulation", recalled[0]["content"])

    def test_rag_benchmark_reports_retrieval_and_latency_metrics(self):
        from knowledge_storm.paperstorm_rag_benchmark import run_rag_benchmark

        with tempfile.TemporaryDirectory() as temp_dir:
            report = run_rag_benchmark(output_dir=Path(temp_dir), run_dir=self.make_run_dir())

            self.assertTrue((Path(temp_dir) / "rag_benchmark_report.json").exists())
            self.assertTrue((Path(temp_dir) / "rag_benchmark_report.md").exists())

        metrics = report["metrics"]
        self.assertIn("context_recall", metrics)
        self.assertIn("citation_precision", metrics)
        self.assertIn("p95_latency_ms", metrics)
        self.assertGreaterEqual(metrics["context_recall"], 0)


if __name__ == "__main__":
    unittest.main()
