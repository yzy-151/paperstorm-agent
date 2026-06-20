import tempfile
import unittest
from pathlib import Path


class PaperStormEnterpriseV32Test(unittest.TestCase):
    def test_rag_index_accepts_callable_embedding_provider_and_records_backend(self):
        from knowledge_storm.paperstorm_rag import (
            CallableEmbeddingProvider,
            PaperStormRAGIndex,
        )

        provider = CallableEmbeddingProvider(
            name="test-real-embedding",
            dim=4,
            embed_fn=lambda texts: [[1.0, 0.0, 0.0, 0.0] for _ in texts],
        )
        index = PaperStormRAGIndex.from_documents(
            [
                {
                    "document_id": "doc-1",
                    "title": "PIM RF paper",
                    "text": "Passive intermodulation suppression in RF systems.",
                }
            ],
            embedding_provider=provider,
        )

        self.assertEqual(index.config["embedding_provider"], "test-real-embedding")
        self.assertEqual(index.chunks[0]["embedding"], [1.0, 0.0, 0.0, 0.0])
        self.assertTrue(index.search("RF suppression", top_k=1))

    def test_context_compression_can_use_llm_summarizer_with_fallback_shape(self):
        from knowledge_storm.paperstorm_rag import (
            ContextCompressionRetriever,
            PaperStormRAGIndex,
        )

        index = PaperStormRAGIndex.from_documents(
            [
                {
                    "document_id": "doc-1",
                    "title": "PIM",
                    "text": "Passive intermodulation is an RF issue. Neural cancellation suppresses PIM.",
                }
            ]
        )
        retriever = ContextCompressionRetriever(
            index,
            max_context_chars=240,
            llm_compressor=lambda payload: "LLM compressed evidence: passive intermodulation RF.",
        )
        result = retriever.retrieve(
            "PIM RF",
            history=[{"role": "user", "content": "之前讨论 RF PIM。"}],
            expected_keywords=["passive intermodulation"],
        )

        self.assertIn("LLM compressed evidence", result["compressed_evidence"])
        self.assertEqual(result["audit"]["compression"], "llm_context_compressor")

    def test_enterprise_knowledge_base_indexes_local_documents_and_answers(self):
        from knowledge_storm.paperstorm_enterprise_kb import EnterpriseKnowledgeBaseService

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paper = root / "paper.txt"
            paper.write_text(
                "Passive intermodulation is caused by nonlinear passive RF components.\n"
                "Neural network cancellation can suppress PIM products.",
                encoding="utf-8",
            )
            service = EnterpriseKnowledgeBaseService(root)
            kb = service.create_knowledge_base(
                name="pim-kb",
                source_paths=[str(paper)],
                expected_keywords=["passive intermodulation", "RF"],
                forbidden_keywords=["DRAM"],
            )
            answer = service.ask(kb["kb_id"], "PIM 是什么？", top_k=2)

            self.assertTrue(kb["kb_id"])
            self.assertEqual(kb["document_count"], 1)
            self.assertGreater(kb["chunk_count"], 0)
            self.assertTrue((root / "knowledge_bases" / kb["kb_id"] / "rag_index.json").exists())
            self.assertTrue(answer["grounded"])
            self.assertIn("Passive intermodulation", answer["answer"])
            self.assertTrue(answer["citations"])

    def test_task_service_exposes_enterprise_kb_workflow(self):
        from knowledge_storm.paperstorm_service import PaperStormTaskService

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paper = root / "manual.txt"
            paper.write_text(
                "Enterprise knowledge base agents use RAG over internal documents.",
                encoding="utf-8",
            )
            service = PaperStormTaskService(root_dir=root)
            kb = service.create_enterprise_knowledge_base(
                name="enterprise-kb",
                source_paths=[str(paper)],
            )
            answer = service.ask_enterprise_knowledge_base(
                kb["kb_id"],
                "企业知识库 agent 使用什么？",
            )

            self.assertIn("kb_id", kb)
            self.assertTrue(answer["grounded"])
            self.assertEqual(answer["kb_id"], kb["kb_id"])


if __name__ == "__main__":
    unittest.main()
