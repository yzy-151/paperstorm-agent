import unittest
from pathlib import Path


class PaperStormDashboardStageUiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.html = (root / "frontend/paperstorm_dashboard/index.html").read_text(
            encoding="utf-8"
        )
        cls.javascript = (
            root / "frontend/paperstorm_dashboard/app.js"
        ).read_text(encoding="utf-8")
        cls.css = (root / "frontend/paperstorm_dashboard/styles.css").read_text(
            encoding="utf-8"
        )

    def test_deliver_card_has_pdf_choice_and_open_action(self):
        self.assertIn('id="task-generate-pdf"', self.html)
        self.assertIn('id="open-article-pdf"', self.html)
        self.assertIn("generate_pdf:", self.javascript)
        self.assertIn("paperstorm_report.pdf", self.javascript)

    def test_research_submission_does_not_guess_multiple_active_stages(self):
        self.assertNotIn('renderResearchProgress("retrieval")', self.javascript)
        self.assertNotIn('renderResearchProgress("completed")', self.javascript)
        running_block = self.javascript.split(
            'payload.task_status === "running"', 1
        )[1].split('payload.task_status === "failed"', 1)[0]
        self.assertNotIn('setPipelineNodeStatus("persona"', running_block)
        self.assertNotIn('setPipelineNodeStatus("dialogue"', running_block)

    def test_stage_events_drive_live_complete_error_and_skipped_states(self):
        for event_name in (
            'eventName === "stage_start"',
            'eventName === "stage_progress"',
            'eventName === "stage_end"',
            'eventName === "stage_error"',
        ):
            self.assertIn(event_name, self.javascript)
        self.assertIn('"skipped"', self.javascript)
        self.assertIn(".pipeline-node.skipped", self.css)
        self.assertIn("hasStageTrace", self.javascript)

    def test_inspector_maps_structured_stage_input_output_usage_and_error(self):
        for field in (
            "output_summary",
            "error_type",
            "error_message",
            "prompt_tokens",
            "completion_tokens",
            "estimated_cost",
        ):
            self.assertIn(field, self.javascript)

    def test_terminal_failure_does_not_create_a_second_failed_node(self):
        self.assertIn("existingFailedNode", self.javascript)
        self.assertIn("activeInvocations", self.javascript)
        self.assertIn('eventName === "stage_usage"', self.javascript)


if __name__ == "__main__":
    unittest.main()
