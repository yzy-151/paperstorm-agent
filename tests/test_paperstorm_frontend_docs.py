import json
import tempfile
import unittest
from pathlib import Path


class PaperStormFrontendDocsTest(unittest.TestCase):
    def test_demo_bundle_contains_dashboard_data(self):
        from knowledge_storm.paperstorm_demo import build_demo_bundle

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            bundle = build_demo_bundle(output_dir=output_dir)
            data_path = output_dir / "sample_data.json"
            js_path = output_dir / "sample_data.js"

            self.assertTrue(data_path.exists())
            self.assertTrue(js_path.exists())
            data = json.loads(data_path.read_text(encoding="utf-8"))
            self.assertIn("tasks", data)
            self.assertIn("article", data)
            self.assertIn("scorecard", data)
            self.assertIn("trace", data)
            self.assertIn("multi_agent", data)
            self.assertIn("pipeline_worker", data)
            self.assertIn("service_snapshot", data)
            self.assertIn("stress_report", data)
            self.assertEqual(data["project"]["version"], "v4.3")
            self.assertIn("rag_evaluation_v4", data)
            self.assertIn("process", data)
            self.assertEqual(bundle["data_path"], str(data_path))
            self.assertEqual(bundle["js_path"], str(js_path))

    def test_static_frontend_exposes_agent_dashboard_panels(self):
        root = Path(__file__).resolve().parents[1]
        index = (root / "frontend" / "paperstorm_dashboard" / "index.html").read_text(
            encoding="utf-8"
        )
        script = (root / "frontend" / "paperstorm_dashboard" / "app.js").read_text(
            encoding="utf-8"
        )

        self.assertIn("PaperStorm Agent Dashboard", index)
        self.assertIn("task-panel", index)
        self.assertIn("trace-panel", index)
        self.assertIn("scorecard-panel", index)
        self.assertIn("multi-agent-panel", index)
        self.assertIn("stress-panel", index)
        self.assertIn("sample_data.js", index)
        self.assertIn("sample_data.json", script)
        self.assertIn("PAPERSTORM_SAMPLE_DATA", script)

    def test_static_frontend_can_load_service_dashboard_bundle(self):
        root = Path(__file__).resolve().parents[1]
        index = (root / "frontend" / "paperstorm_dashboard" / "index.html").read_text(
            encoding="utf-8"
        )
        script = (root / "frontend" / "paperstorm_dashboard" / "app.js").read_text(
            encoding="utf-8"
        )

        self.assertIn("service-url", index)
        self.assertIn("service-task-id", index)
        self.assertIn("load-service-task", index)
        self.assertIn("/dashboard", script)
        self.assertIn("loadServiceTask", script)
        self.assertIn("pipeline-worker", index)
        self.assertIn("renderPipelineWorker", script)

    def test_static_frontend_exposes_task_control_console(self):
        root = Path(__file__).resolve().parents[1]
        index = (root / "frontend" / "paperstorm_dashboard" / "index.html").read_text(
            encoding="utf-8"
        )
        script = (root / "frontend" / "paperstorm_dashboard" / "app.js").read_text(
            encoding="utf-8"
        )

        self.assertIn("create-task-panel", index)
        self.assertIn("task-topic", index)
        self.assertIn("task-run-mode", index)
        self.assertIn("submit-task", index)
        self.assertIn("run-selected-task", index)
        self.assertIn("poll-selected-task", index)
        self.assertIn("task-error-panel", index)
        self.assertIn("submitTask", script)
        self.assertIn("runSelectedTask", script)
        self.assertIn("pollSelectedTask", script)
        self.assertIn("fetchTaskList", script)
        self.assertIn("/research-tasks", script)

    def test_static_frontend_exposes_sse_event_stream_panel(self):
        root = Path(__file__).resolve().parents[1]
        index = (root / "frontend" / "paperstorm_dashboard" / "index.html").read_text(
            encoding="utf-8"
        )
        script = (root / "frontend" / "paperstorm_dashboard" / "app.js").read_text(
            encoding="utf-8"
        )

        self.assertIn("sse-panel", index)
        self.assertIn("sse-event-list", index)
        self.assertIn("EventSource", script)
        self.assertIn("/events", script)
        self.assertIn("connectSSE", script)

    def test_dashboard_exposes_debuggable_v40_runtime_ui(self):
        root = Path(__file__).resolve().parents[1]
        index = (root / "frontend" / "paperstorm_dashboard" / "index.html").read_text(
            encoding="utf-8"
        )
        script = (root / "frontend" / "paperstorm_dashboard" / "app.js").read_text(
            encoding="utf-8"
        )
        styles = (root / "frontend" / "paperstorm_dashboard" / "styles.css").read_text(
            encoding="utf-8"
        )

        self.assertIn("v4.0", index)
        self.assertIn("操作说明", index)
        self.assertIn("Benchmark", index)
        self.assertIn("过程细节", index)
        self.assertIn("outline-content", index)
        self.assertIn("reflection-content", index)
        self.assertIn("conversation-content", index)
        self.assertIn("formatTimestamp", script)
        self.assertIn("setButtonBusy", script)
        self.assertIn("getServiceBaseUrl", script)
        self.assertIn("renderProcessDetails", script)
        self.assertIn("log-event-service", styles)
        self.assertIn("log-event-error", styles)
        self.assertIn("log-event-task_status", styles)

    def test_dashboard_exposes_research_qa_chat(self):
        root = Path(__file__).resolve().parents[1]
        index = (root / "frontend" / "paperstorm_dashboard" / "index.html").read_text(
            encoding="utf-8"
        )
        script = (root / "frontend" / "paperstorm_dashboard" / "app.js").read_text(
            encoding="utf-8"
        )
        styles = (root / "frontend" / "paperstorm_dashboard" / "styles.css").read_text(
            encoding="utf-8"
        )

        self.assertIn("research-qa-panel", index)
        self.assertIn("research-question", index)
        self.assertIn("ask-research-agent", index)
        self.assertIn("research-answer", index)
        self.assertIn("research-citations", index)
        self.assertIn("research-decision", index)
        self.assertIn("research-sufficiency", index)
        self.assertIn("/research-agent/ask", script)
        self.assertIn("askResearchAgent", script)
        self.assertIn("renderResearchQA", script)
        self.assertIn("chat-transcript", styles)

    def test_dashboard_exposes_dual_mode_research_and_chat_context(self):
        root = Path(__file__).resolve().parents[1]
        index = (root / "frontend" / "paperstorm_dashboard" / "index.html").read_text(
            encoding="utf-8"
        )
        script = (root / "frontend" / "paperstorm_dashboard" / "app.js").read_text(
            encoding="utf-8"
        )
        styles = (root / "frontend" / "paperstorm_dashboard" / "styles.css").read_text(
            encoding="utf-8"
        )

        self.assertIn("mode-switcher", index)
        self.assertIn("research-mode-panel", index)
        self.assertIn("chat-mode-panel", index)
        self.assertIn("chat-session-id", index)
        self.assertIn("create-chat-session", index)
        self.assertIn("send-chat-message", index)
        self.assertIn("chat-message-list", index)
        self.assertIn("chat-context-window", index)
        self.assertIn("chat-compressed-context", index)
        self.assertIn("chat-memory-context", index)
        self.assertIn("chat-router-decision", index)
        self.assertIn("chat-tool-decision", index)
        self.assertIn("chat-rewritten-query", index)
        self.assertIn("/chat/sessions", script)
        self.assertIn("createChatSession", script)
        self.assertIn("sendChatMessage", script)
        self.assertIn("renderChatSession", script)
        self.assertIn("router_decision", script)
        self.assertIn("tool_decision", script)
        self.assertIn("setDashboardMode", script)
        self.assertIn("chat-shell", styles)

    def test_dashboard_defaults_to_chat_and_separates_developer_console(self):
        root = Path(__file__).resolve().parents[1]
        index = (root / "frontend" / "paperstorm_dashboard" / "index.html").read_text(
            encoding="utf-8"
        )
        script = (root / "frontend" / "paperstorm_dashboard" / "app.js").read_text(
            encoding="utf-8"
        )
        styles = (root / "frontend" / "paperstorm_dashboard" / "styles.css").read_text(
            encoding="utf-8"
        )

        self.assertIn('id="show-developer-mode"', index)
        self.assertIn("开发者控制台", index)
        self.assertIn("developer-mode-panel", index)
        self.assertIn('setDashboardMode("chat")', script)
        self.assertIn('body[data-mode="developer"]', styles)

    def test_dashboard_exposes_retrieval_runtime_benchmark_panel(self):
        root = Path(__file__).resolve().parents[1]
        index = (root / "frontend" / "paperstorm_dashboard" / "index.html").read_text(
            encoding="utf-8"
        )
        script = (root / "frontend" / "paperstorm_dashboard" / "app.js").read_text(
            encoding="utf-8"
        )

        self.assertIn("retrieval-runtime-panel", index)
        self.assertIn("run-retrieval-runtime", index)
        self.assertIn("load-retrieval-runtime", index)
        self.assertIn("retrieval-runtime-table", index)
        self.assertIn("retrieval-runtime-summary", index)
        self.assertIn("runtime-integration-status", index)
        self.assertIn("runRetrievalRuntimeBenchmark", script)
        self.assertIn("loadRetrievalRuntimeBenchmark", script)
        self.assertIn("renderRetrievalRuntimeBenchmark", script)
        self.assertIn("/evaluations/retrieval-runtime", script)

    def test_dashboard_exposes_enterprise_kb_v32_workflow(self):
        root = Path(__file__).resolve().parents[1]
        index = (root / "frontend" / "paperstorm_dashboard" / "index.html").read_text(
            encoding="utf-8"
        )
        script = (root / "frontend" / "paperstorm_dashboard" / "app.js").read_text(
            encoding="utf-8"
        )

        self.assertIn("enterprise-kb-panel", index)
        self.assertIn("enterprise-kb-source-paths", index)
        self.assertIn("create-enterprise-kb", index)
        self.assertIn("enterprise-kb-question", index)
        self.assertIn("ask-enterprise-kb", index)
        self.assertIn("enterprise-kb-answer", index)
        self.assertIn("/enterprise-kbs", script)
        self.assertIn("createEnterpriseKB", script)
        self.assertIn("askEnterpriseKB", script)

    def test_official_chinese_doc_and_readme_include_storm_architecture(self):
        root = Path(__file__).resolve().parents[1]
        official_cn = (root / "docs" / "STORM_OFFICIAL_CN.md").read_text(
            encoding="utf-8"
        )
        readme = (root / "README.md").read_text(encoding="utf-8")

        self.assertIn("官方 STORM 架构", official_cn)
        self.assertIn("Perspective-Guided Question Asking", official_cn)
        self.assertIn("Simulated Conversation", official_cn)
        self.assertIn("assets/overview.svg", readme)
        self.assertIn("assets/two_stages.jpg", readme)
        self.assertIn("官方 STORM 基础架构", readme)


if __name__ == "__main__":
    unittest.main()
