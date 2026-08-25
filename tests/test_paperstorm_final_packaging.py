import unittest
from pathlib import Path


class PaperStormFinalPackagingTest(unittest.TestCase):
    def test_readme_contains_final_project_positioning_and_architecture_map(self):
        root = Path(__file__).resolve().parents[1]
        readme = (root / "README.md").read_text(encoding="utf-8")

        self.assertIn("PaperStorm Agent", readme)
        self.assertIn("系统架构", readme)
        self.assertIn("核心能力", readme)
        self.assertIn("官方 STORM 基础架构", readme)
        self.assertIn("Stanford STORM Workflow", readme)
        self.assertIn("启动服务", readme)

    def test_readme_documents_rag_memory_and_runtime(self):
        root = Path(__file__).resolve().parents[1]
        readme = (root / "README.md").read_text(encoding="utf-8")

        self.assertIn("RetrievalPipeline", readme)
        self.assertIn("Context", readme)
        self.assertIn("Memory", readme)
        self.assertIn("Session Recall", readme)
        self.assertIn("LangGraph", readme)
        self.assertIn("Control Plane", readme)
        self.assertIn("本地文档知识库", readme)


if __name__ == "__main__":
    unittest.main()
