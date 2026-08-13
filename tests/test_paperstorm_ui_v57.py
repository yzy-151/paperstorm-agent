import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend" / "paperstorm_dashboard"


class PaperStormUIV57Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (FRONTEND / "index.html").read_text(encoding="utf-8")
        cls.css = (FRONTEND / "styles.css").read_text(encoding="utf-8")
        cls.readme = (ROOT / "README.md").read_text(encoding="utf-8")

    def test_product_uses_v57_workspace_shell(self):
        self.assertIn("v5.7", self.html)
        self.assertIn('class="workspace-rail"', self.html)
        self.assertIn('class="workspace-main"', self.html)
        self.assertIn('class="workspace-inspector product-only"', self.html)

    def test_visual_language_is_square_and_dense(self):
        self.assertIn("--radius: 2px", self.css)
        self.assertIn("grid-template-columns: 208px minmax(0, 1fr) 288px", self.css)
        self.assertIn(".benchmark-card::before", self.css)

    def test_readme_presents_new_screenshots_diagram_and_benchmark_icons(self):
        for marker in (
            "dashboard-research-v57.png",
            "dashboard-developer-v57.png",
            "paperstorm-executive-overview-v57.svg",
            "benchmark-icon-retrieval.svg",
            "benchmark-icon-memory.svg",
            "benchmark-icon-context.svg",
            "benchmark-icon-answer.svg",
        ):
            self.assertIn(marker, self.readme)

    def test_real_pipeline_is_the_default_for_research_and_chat(self):
        self.assertIn(
            '<option value="paperstorm" selected>真实检索与 LLM</option>',
            self.html,
        )
        self.assertIn(
            '<option value="paperstorm" selected>真实 API</option>',
            self.html,
        )

    def test_advanced_keyword_filters_are_empty_by_default(self):
        self.assertIn('id="task-expected-keyword" placeholder=', self.html)
        self.assertIn('id="task-forbidden-keyword" placeholder=', self.html)
        self.assertNotIn('id="task-expected-keyword" value=', self.html)
        self.assertNotIn('id="task-forbidden-keyword" value=', self.html)

    def test_public_copy_does_not_market_an_expert_edition(self):
        forbidden = ("专家版", "专业版", "专业工作台", "professional workspace")
        for marker in forbidden:
            self.assertNotIn(marker, self.html.lower())
            self.assertNotIn(marker, self.readme.lower())


if __name__ == "__main__":
    unittest.main()
