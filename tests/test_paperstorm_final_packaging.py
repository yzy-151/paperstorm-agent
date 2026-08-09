import unittest
from pathlib import Path


class PaperStormFinalPackagingTest(unittest.TestCase):
    def test_readme_contains_final_project_positioning_and_architecture_map(self):
        root = Path(__file__).resolve().parents[1]
        readme = (root / "README.md").read_text(encoding="utf-8")

        self.assertIn("PaperStorm Agent", readme)
        self.assertIn("项目一眼看懂", readme)
        self.assertIn("最终能力地图", readme)
        self.assertIn("官方 STORM 基础架构", readme)
        self.assertIn(
            "STORM Workflow -> PaperStorm Runtime -> Service/Dashboard", readme
        )
        self.assertIn("启动服务", readme)

    def test_readme_documents_rag_memory_and_runtime(self):
        root = Path(__file__).resolve().parents[1]
        readme = (root / "README.md").read_text(encoding="utf-8")

        self.assertIn("PaperStormRAGIndex", readme)
        self.assertIn("Context v4.2", readme)
        self.assertIn("Memory v4.3", readme)
        self.assertIn("LangGraph v4.4", readme)
        self.assertIn("Production v4.5", readme)
        self.assertIn("本地文档知识库", readme)


if __name__ == "__main__":
    unittest.main()
