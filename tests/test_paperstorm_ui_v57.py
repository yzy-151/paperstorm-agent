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


if __name__ == "__main__":
    unittest.main()
