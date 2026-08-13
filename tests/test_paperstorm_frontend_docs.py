import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend" / "paperstorm_dashboard"


class PaperStormFrontendDocsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = (FRONTEND / "index.html").read_text(encoding="utf-8")
        cls.script = (FRONTEND / "app.js").read_text(encoding="utf-8")
        cls.styles = (FRONTEND / "styles.css").read_text(encoding="utf-8")

    def test_product_exposes_research_and_chat_without_developer_clutter(self):
        for marker in (
            'id="research-view"',
            'id="chat-view"',
            'id="start-research-workflow"',
            'id="chat-form"',
            'id="show-developer-mode"',
        ):
            self.assertIn(marker, self.index)
        self.assertIn("runResearchWorkflow", self.script)
        self.assertIn("sendChat", self.script)

    def test_developer_console_centers_public_v55_v56_evidence(self):
        for marker in (
            "BENCHMARK CONTROL PLANE",
            "PUBLIC EVIDENCE",
            "Benchmark Registry",
            "v5.5 / v5.6",
            "执行链路诊断",
        ):
            self.assertIn(marker, self.index)
        self.assertNotIn("RAG v4.0 评测", self.index)
        self.assertNotIn("Context v4.2 可恢复压缩 Benchmark", self.index)

    def test_frontend_loads_registry_runs_and_live_diagnostics(self):
        for marker in (
            'fetchJson("/benchmarks/catalog")',
            'fetchJson("/benchmarks/runs"',
            "pollBenchmarkRun",
            'fetchJson("/production/status")',
            'fetchJson("/research-tasks")',
        ):
            self.assertIn(marker, self.script)

    def test_styles_define_desktop_and_mobile_workbench_constraints(self):
        for marker in (
            ".benchmark-catalog",
            ".benchmark-run-workspace",
            ".benchmark-log-tail",
            ".status-band",
            "@media (max-width: 560px)",
            "minmax(0, 1fr)",
        ):
            self.assertIn(marker, self.styles)

    def test_demo_bundle_still_contains_research_artifacts(self):
        from knowledge_storm.paperstorm_demo import build_demo_bundle

        with tempfile.TemporaryDirectory() as temp:
            bundle = build_demo_bundle(temp)
            data = json.loads(Path(bundle["data_path"]).read_text(encoding="utf-8"))
        self.assertIn("tasks", data)
        self.assertIn("article", data)
        self.assertIn("trace", data)

    def test_official_chinese_docs_keep_storm_architecture(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        official = (ROOT / "docs" / "STORM_OFFICIAL_CN.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("官方 STORM 基础架构", readme)
        self.assertIn("知识整理", official)


if __name__ == "__main__":
    unittest.main()
