import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "knowledge_storm"
VERSION_MARKERS = ("_v3", "_v4", "_v5", "_v6")
REMOVED_LEGACY_MODULES = (
    "paperstorm_context_v42.py",
    "paperstorm_context_v56.py",
    "paperstorm_document_v41.py",
    "paperstorm_eval_v4.py",
    "paperstorm_eval_v54.py",
    "paperstorm_langgraph_v44.py",
    "paperstorm_memory_v43.py",
    "paperstorm_memory_v56.py",
    "paperstorm_production_v45.py",
    "paperstorm_rag.py",
    "paperstorm_retrieval_runtime.py",
    "paperstorm_retrieval_v41.py",
)


class PaperStormModuleBoundaryTest(unittest.TestCase):
    def test_production_modules_do_not_import_versioned_modules(self):
        violations = []
        for path in PACKAGE.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.ImportFrom) and node.module:
                    names.append(node.module)
                elif isinstance(node, ast.Import):
                    names.extend(alias.name for alias in node.names)
                for name in names:
                    if "paperstorm_" in name and any(
                        marker in name for marker in VERSION_MARKERS
                    ):
                        violations.append(f"{path.name}:{node.lineno}:{name}")
        self.assertEqual([], violations)

    def test_stable_runtime_modules_are_importable(self):
        from knowledge_storm import context_engine
        from knowledge_storm import control_plane
        from knowledge_storm import conversation_runtime
        from knowledge_storm import document_ingestion
        from knowledge_storm import memory_store
        from knowledge_storm import retrieval
        from knowledge_storm import retrieval_pipeline

        self.assertTrue(context_engine.ContextEngine)
        self.assertTrue(control_plane.ProductionControlPlane)
        self.assertTrue(conversation_runtime.PaperStormConversationRuntime)
        self.assertTrue(document_ingestion.chunk_pdf_pages)
        self.assertTrue(memory_store.LongTermMemoryService)
        self.assertTrue(retrieval.HybridPaperIndex)
        self.assertTrue(retrieval_pipeline.RetrievalPipeline)

    def test_removed_legacy_modules_are_not_reintroduced(self):
        present = [name for name in REMOVED_LEGACY_MODULES if (PACKAGE / name).exists()]
        self.assertEqual([], present)


if __name__ == "__main__":
    unittest.main()
