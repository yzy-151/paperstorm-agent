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

    def test_research_mode_has_one_click_demo_and_live_agent_graph(self):
        self.assertIn('id="start-research-demo"', self.index)
        self.assertIn('id="pipeline-canvas"', self.index)
        self.assertIn('id="pipeline-execution-wires"', self.index)
        self.assertIn('id="pipeline-artifact-wires"', self.index)

    def test_frontend_assets_are_versioned_to_prevent_mixed_releases(self):
        self.assertIn('href="styles.css?v=7.2.0"', self.index)
        self.assertIn('src="app.js?v=7.2.0"', self.index)
        for node in ("request", "persona", "dialogue", "retrieval", "evidence", "outline", "writer", "deliver"):
            self.assertIn('data-node="{0}"'.format(node), self.index)
        self.assertIn("renderResearchProgress", self.script)
        self.assertIn("applyPipelineTrace", self.script)

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
        self.assertIn('body[data-mode="developer"]', self.styles)

    def test_layout_uses_stable_responsive_dimensions(self):
        self.assertIn("grid-template-columns: 208px minmax(0, 1fr) 288px", self.styles)
        self.assertIn("grid-template-columns: repeat(3, minmax(0, 1fr))", self.styles)
        self.assertIn("@media (max-width: 560px)", self.styles)


if __name__ == "__main__":
    unittest.main()
