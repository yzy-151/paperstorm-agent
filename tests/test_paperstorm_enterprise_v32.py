import tempfile
import unittest
from pathlib import Path


class PaperStormEnterpriseV32Test(unittest.TestCase):
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
                embedding_provider="hash",
                expected_keywords=["passive intermodulation", "RF"],
                forbidden_keywords=["DRAM"],
            )
            answer = service.ask(kb["kb_id"], "PIM 是什么？", top_k=2)

            self.assertTrue(kb["kb_id"])
            self.assertEqual(kb["document_count"], 1)
            self.assertGreater(kb["chunk_count"], 0)
            self.assertTrue(Path(kb["index_path"]).exists())
            self.assertTrue(answer["grounded"])
            self.assertIn("Passive intermodulation", answer["answer"])
            self.assertTrue(answer["citations"])

    def test_incremental_update_commits_a_new_index_generation(self):
        from knowledge_storm.paperstorm_enterprise_kb import EnterpriseKnowledgeBaseService

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paper = root / "paper.txt"
            paper.write_text("passive intermodulation baseline", encoding="utf-8")
            service = EnterpriseKnowledgeBaseService(root)
            created = service.create_knowledge_base(
                "pim-kb", [str(paper)], embedding_provider="hash"
            )
            paper.write_text("passive intermodulation neural cancellation", encoding="utf-8")

            updated = service.update_knowledge_base(created["kb_id"], [str(paper)])

            self.assertEqual(2, updated["index_version"])
            self.assertNotEqual(created["index_path"], updated["index_path"])
            self.assertTrue(Path(updated["index_path"]).exists())

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
                embedding_provider="hash",
            )
            answer = service.ask_enterprise_knowledge_base(
                kb["kb_id"],
                "本地知识库 agent 使用什么？",
            )

            self.assertIn("kb_id", kb)
            self.assertTrue(answer["grounded"])
            self.assertEqual(answer["kb_id"], kb["kb_id"])


if __name__ == "__main__":
    unittest.main()
