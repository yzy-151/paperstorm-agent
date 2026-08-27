import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend" / "paperstorm_dashboard"


class PaperStormDeveloperMilestonesUiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (FRONTEND / "index.html").read_text(encoding="utf-8")
        cls.script = (FRONTEND / "app.js").read_text(encoding="utf-8")
        cls.css = (FRONTEND / "styles.css").read_text(encoding="utf-8")

    def test_console_uses_only_cumulative_milestones(self):
        for marker in (
            "P0 baseline",
            "P1",
            "P1+P2",
            "P1+P2+P3",
            "P1+P2+P3+P4",
            "production-governance",
        ):
            self.assertIn(marker, self.script)
        self.assertIn('id="milestone-progress"', self.html)

    def test_milestones_disclose_scope_and_completed_p3(self):
        for marker in (
            "affected_benchmarks",
            "QASPER test 1451",
            "Answer F1 0.5083",
            "Claim support 0.9592",
            "completed",
            "production-governance",
        ):
            self.assertIn(marker, self.script)
        self.assertIn('id="milestone-detail"', self.html)

    def test_case_panel_explains_missing_evidence_fields(self):
        for marker in (
            "case-before-top-k",
            "case-after-top-k",
            "case-failure-stage",
            "case-trace",
            "case-citation",
            "未提供",
        ):
            self.assertIn(marker, self.html + self.script)

    def test_console_has_compact_dark_milestone_and_case_layout(self):
        for marker in (
            ".milestone-progress",
            ".milestone-card",
            ".case-comparison",
            ".case-field",
        ):
            self.assertIn(marker, self.css)


if __name__ == "__main__":
    unittest.main()
