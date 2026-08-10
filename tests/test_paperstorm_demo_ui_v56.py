import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PaperStormDemoUIV56Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        frontend = ROOT / "frontend" / "paperstorm_dashboard"
        cls.index = (frontend / "index.html").read_text(encoding="utf-8")
        cls.script = (frontend / "app.js").read_text(encoding="utf-8")
        cls.styles = (frontend / "styles.css").read_text(encoding="utf-8")

    def test_research_mode_has_one_click_demo_and_five_stage_progress(self):
        self.assertIn('id="start-research-demo"', self.index)
        self.assertIn('id="research-progress"', self.index)
        for stage in ("created", "retrieval", "outline", "writing", "completed"):
            self.assertIn('data-stage="{0}"'.format(stage), self.index)
        self.assertIn("renderResearchProgress", self.script)

    def test_chat_mode_keeps_context_aware_session_controls(self):
        for marker in (
            'id="chat-run-mode"',
            'id="chat-retriever"',
            'id="create-chat"',
            'id="chat-context-summary"',
        ):
            self.assertIn(marker, self.index)
        self.assertIn("/chat/sessions", self.script)

    def test_developer_mode_is_a_separate_surface(self):
        self.assertIn('id="developer-view"', self.index)
        self.assertIn('id="leave-developer-mode"', self.index)
        self.assertIn('setMode("developer")', self.script)
        self.assertIn(".developer-view", self.styles)

    def test_layout_uses_stable_responsive_dimensions(self):
        self.assertIn("width: min(1320px", self.styles)
        self.assertIn("grid-template-columns: repeat(3, minmax(0, 1fr))", self.styles)
        self.assertIn("@media (max-width: 680px)", self.styles)


if __name__ == "__main__":
    unittest.main()
