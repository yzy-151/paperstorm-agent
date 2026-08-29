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
        self.assertIn("v7.2", self.html)
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
        self.assertIn("@keyframes node-outline-flow", self.css)
        self.assertIn("node-outline-flow", self.css)
        self.assertIn("node-breathe", self.css)

    def test_pipeline_uses_named_ports_and_separate_edge_layers(self):
        for marker in (
            'class="node-port input-port"',
            'node-port output-port',
            'id="pipeline-execution-wires"',
            'id="pipeline-artifact-wires"',
            "pipelineExecutionEdges",
            "pipelineArtifactEdges",
            "artifact_ready",
        ):
            self.assertIn(marker, self.html + self.js)
        self.assertIn("artifact-wire", self.css)
        self.assertIn("execution-wire", self.css)
        self.assertIn('data-port="conversation"', self.html)
        self.assertIn('data-port="scorecard"', self.html)
        self.assertIn("port-row-2", self.css)
        self.assertIn("animation: artifact-flow", self.css)
        self.assertIn("animation: execution-flow", self.css)

    def test_chat_messages_show_local_avatars_and_telemetry(self):
        bundle = self.html + self.js + self.css
        for marker in (
            "avatar-paperstorm.svg",
            "avatar-user.svg",
            "message-telemetry",
            "message-usage",
            "总计",
            "估算用量",
            "真实用量",
            "耗时未记录",
            "prompt_tokens",
            "completion_tokens",
        ):
            self.assertIn(marker, bundle)

    def test_pipeline_uses_row_major_snake_layout_and_semantic_ports(self):
        for marker in (
            'data-node="retrieval" style="--col:1;--row:3"',
            'data-node="evidence" style="--col:2;--row:3"',
            'data-node="outline" style="--col:3;--row:3"',
            'data-node="writer" style="--col:4;--row:3"',
            'class="pipeline-node aux-node" data-node="polish"',
            'class="pipeline-node aux-node" data-node="evaluate"',
            'relay-output',
            'terminal-output',
            'id="pipeline-legend"',
        ):
            self.assertIn(marker, self.html)

    def test_pipeline_uses_monochrome_active_border_and_routed_curves(self):
        self.assertIn("node-outline-flow", self.css)
        self.assertNotIn("linear-gradient(90deg, #24c9b1, #72a7ff", self.css)
        self.assertIn("pipelineExecutionPath", self.js)
        self.assertIn("pipelineArtifactPath", self.js)
        self.assertIn("pipelineRowWrapPath", self.js)

    def test_artifact_edges_connect_port_markers_and_keep_labels_upright(self):
        self.assertIn('.output-port[data-port="${sourcePort}"] i', self.js)
        self.assertIn('.input-port[data-port="${targetPort}"] i', self.js)
        self.assertIn("positionArtifactLabels", self.js)
        self.assertNotIn("<textPath", self.js)
        self.assertIn("pipelineArtifactRoute", self.js)
        self.assertIn("artifactLaneOffset", self.js)
        self.assertNotIn(" L ${points.x2", self.js)

    def test_artifact_highlights_settle_and_cannot_reactivate(self):
        self.assertIn("settleArtifactStatuses", self.js)
        self.assertIn("failActiveArtifactStatuses", self.js)
        self.assertIn("markArtifactInputsActive", self.js)
        self.assertIn("completeArtifactInputs", self.js)
        self.assertIn('sourceStatus === "active"', self.js)
        self.assertIn('markArtifactInputsActive(stage)', self.js)
        self.assertIn('if (!stillActive) completeArtifactInputs(stage)', self.js)
        self.assertIn('state.artifactStatus[edge.id] !== "complete"', self.js)
        self.assertIn('payload.task_status === "succeeded"', self.js)
        self.assertIn('payload.task_status === "failed"', self.js)
        self.assertIn('state.artifactStatus[edgeId] = "failed"', self.js)
        self.assertIn("state.runFinished", self.js)
        self.assertIn("if (state.runFinished) return", self.js)


if __name__ == "__main__":
    unittest.main()
