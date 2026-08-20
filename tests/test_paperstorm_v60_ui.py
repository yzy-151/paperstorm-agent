import unittest
from pathlib import Path


class PaperStormV60UITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1] / "frontend" / "paperstorm_dashboard"
        cls.html = (root / "index.html").read_text(encoding="utf-8")
        cls.css = (root / "styles.css").read_text(encoding="utf-8")
        cls.js = (root / "app.js").read_text(encoding="utf-8")

    def test_release_and_memory_mode_controls_are_visible(self):
        self.assertIn("v6.0", self.html)
        self.assertIn('id="chat-memory-mode"', self.html)
        self.assertIn('value="semantic"', self.html)

    def test_research_mode_hides_right_inspector_and_expands_workspace(self):
        self.assertIn('body[data-mode="research"] .workspace-inspector', self.css)
        self.assertIn('body[data-mode="research"] .workspace-shell', self.css)

    def test_pipeline_inspector_exposes_runtime_telemetry(self):
        for field_id in (
            "pipeline-node-activity",
            "pipeline-node-duration",
            "pipeline-node-tokens",
            "pipeline-node-cost",
            "pipeline-node-finish",
            "pipeline-node-error",
        ):
            self.assertIn('id="{0}"'.format(field_id), self.html)
        self.assertIn("formatPipelineTelemetry", self.js)
        self.assertIn("node-time", self.js)

    def test_active_node_has_flowing_border_and_breathing_animation(self):
        self.assertIn("@keyframes node-border-flow", self.css)
        self.assertIn("node-border-flow", self.css)
        self.assertIn("node-breathe", self.css)


if __name__ == "__main__":
    unittest.main()
