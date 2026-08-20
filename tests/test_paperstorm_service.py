import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


@mock.patch.dict(
    os.environ,
    {
        "PAPERSTORM_RETRIEVAL_EMBEDDING": "hash",
        "PAPERSTORM_CHAT_LLM": "0",
        "PAPERSTORM_JUDGE_LLM": "0",
    },
)
class PaperStormServiceTest(unittest.TestCase):
    def make_service(self, **kwargs):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        from knowledge_storm.paperstorm_service import PaperStormTaskService

        return PaperStormTaskService(root_dir=Path(temp_dir.name), **kwargs)

    def make_service_with_pipeline_runner(self, runner):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        from knowledge_storm.paperstorm_service import PaperStormTaskService

        return PaperStormTaskService(
            root_dir=Path(temp_dir.name),
            pipeline_runner=runner,
        )

    def test_submit_research_task_creates_isolated_task_state(self):
        service = self.make_service()

        task = service.submit_research_task(
            topic="pim 神经网络抑制",
            retriever="arxiv",
            output_language="zh",
            run_mode="fake",
        )
        state = service.get_task(task["task_id"])

        self.assertEqual(state["status"], "queued")
        self.assertEqual(state["topic"], "pim 神经网络抑制")
        self.assertTrue(Path(state["output_dir"]).exists())
        self.assertIn(task["task_id"], state["output_dir"])

    def test_resolve_zotero_root_prefers_explicit_path_then_env(self):
        service = self.make_service()
        with tempfile.TemporaryDirectory() as temp_dir:
            (Path(temp_dir) / "zotero.sqlite").write_text("", encoding="utf-8")
            self.assertEqual(
                service._resolve_zotero_root(zotero_root=temp_dir),
                temp_dir,
            )
            with mock.patch.dict(
                os.environ, {"PAPERSTORM_ZOTERO_ROOT": temp_dir}, clear=False
            ):
                self.assertEqual(service._resolve_zotero_root(), temp_dir)

    def test_run_fake_task_writes_article_trace_summary_and_scorecard(self):
        service = self.make_service()
        task = service.submit_research_task(
            topic="pim 神经网络抑制",
            retriever="arxiv",
            output_language="zh",
            run_mode="fake",
            expected_keywords=["passive intermodulation", "RF"],
            forbidden_keywords=["processing-in-memory", "DRAM", "RAM"],
        )

        finished = service.run_task(task["task_id"])
        article = service.get_article(task["task_id"])
        scorecard = service.get_scorecard(task["task_id"])
        trace = service.get_trace(task["task_id"])

        self.assertEqual(finished["status"], "succeeded")
        self.assertIn("passive intermodulation", article["content"])
        self.assertGreater(scorecard["scores"]["total"], 50)
        self.assertTrue(trace["events"])
        stage_starts = [
            event.get("stage")
            for event in trace["events"]
            if event.get("event") == "stage_start"
        ]
        self.assertEqual(
            stage_starts,
            [
                "request",
                "persona",
                "dialogue",
                "query",
                "retrieval",
                "evidence",
                "outline",
                "writer",
                "polish",
                "evaluate",
                "deliver",
            ],
        )

    def test_query_knowledge_base_answers_from_task_artifacts(self):
        service = self.make_service()
        task = service.submit_research_task(
            topic="pim 神经网络抑制",
            retriever="arxiv",
            output_language="zh",
            run_mode="fake",
        )
        service.run_task(task["task_id"])

        with mock.patch.dict(os.environ, {"PAPERSTORM_CHAT_LLM": "0"}):
            answer = service.query_knowledge_base(task["task_id"], "PIM 是什么？")

        self.assertTrue(answer["grounded"])
        self.assertTrue(answer["citations"])
        self.assertIn("passive intermodulation", answer["answer"])

    def test_dashboard_bundle_collects_task_artifacts_for_frontend(self):
        service = self.make_service()
        task = service.submit_research_task(
            topic="pim 神经网络抑制",
            retriever="arxiv",
            output_language="zh",
            run_mode="fake",
            expected_keywords=["passive intermodulation"],
            forbidden_keywords=["DRAM"],
        )
        service.run_task(task["task_id"])
        with mock.patch.dict(os.environ, {"PAPERSTORM_CHAT_LLM": "0"}):
            service.query_knowledge_base(task["task_id"], "PIM 是什么？")

        bundle = service.get_dashboard_bundle(task["task_id"])

        self.assertEqual(bundle["project"]["version"], "v6.1")
        self.assertEqual(bundle["tasks"][0]["task_id"], task["task_id"])
        self.assertIn("passive intermodulation", bundle["article"]["content"])
        self.assertTrue(bundle["trace"]["events"])
        self.assertIn("神经网络抑制", bundle["process"]["outline"])
        self.assertIn("passive intermodulation", bundle["process"]["conversation"])
        self.assertGreater(bundle["scorecard"]["scores"]["total"], 50)
        self.assertTrue(bundle["qa"]["grounded"])
        self.assertIn("service_snapshot", bundle)

    def test_two_tasks_write_to_separate_output_dirs(self):
        service = self.make_service()

        first = service.submit_research_task(topic="topic one", run_mode="fake")
        second = service.submit_research_task(topic="topic two", run_mode="fake")
        service.run_task(first["task_id"])
        service.run_task(second["task_id"])
        first_state = service.get_task(first["task_id"])
        second_state = service.get_task(second["task_id"])

        self.assertNotEqual(first_state["output_dir"], second_state["output_dir"])
        self.assertTrue((Path(first_state["output_dir"]) / "run_summary.json").exists())
        self.assertTrue((Path(second_state["output_dir"]) / "run_summary.json").exists())

    def test_list_tasks_returns_recent_tasks_for_dashboard_polling(self):
        service = self.make_service()

        first = service.submit_research_task(topic="topic one", run_mode="fake")
        second = service.submit_research_task(topic="topic two", run_mode="manual")
        service.run_task(first["task_id"])
        service.worker_tick()

        tasks = service.list_tasks()
        running = service.list_tasks(status="running")
        succeeded = service.list_tasks(status="succeeded")

        self.assertEqual([task["task_id"] for task in tasks], [first["task_id"], second["task_id"]])
        self.assertEqual([task["task_id"] for task in running], [second["task_id"]])
        self.assertEqual([task["task_id"] for task in succeeded], [first["task_id"]])

    def test_task_failure_is_structured_and_secret_safe(self):
        service = self.make_service()
        task = service.submit_research_task(
            topic="broken",
            run_mode="fail",
            api_key="sk-secret-value",
        )

        failed = service.run_task(task["task_id"])
        serialized = json.dumps(failed, ensure_ascii=False)

        self.assertEqual(failed["status"], "failed")
        self.assertIn("error", failed)
        self.assertNotIn("sk-secret-value", serialized)

    def test_domain_keywords_are_preserved_in_task_state(self):
        service = self.make_service()

        task = service.submit_research_task(
            topic="pim",
            expected_keywords=["passive intermodulation"],
            forbidden_keywords=["DRAM"],
        )
        state = service.get_task(task["task_id"])

        self.assertEqual(state["expected_keywords"], ["passive intermodulation"])
        self.assertEqual(state["forbidden_keywords"], ["DRAM"])

    def test_fastapi_adapter_imports_without_required_runtime_dependency(self):
        from examples.storm_examples import paperstorm_service_api

        self.assertTrue(hasattr(paperstorm_service_api, "create_app"))
        self.assertTrue(hasattr(paperstorm_service_api, "DEFAULT_SERVICE_ROOT"))

    def test_fastapi_adapter_serves_dashboard_home_and_sse_events(self):
        from fastapi.testclient import TestClient

        from examples.storm_examples.paperstorm_service_api import create_app

        with tempfile.TemporaryDirectory() as temp_dir:
            app = create_app(service_root=Path(temp_dir))
            client = TestClient(app)

            home = client.get("/")
            self.assertEqual(home.status_code, 200)
            self.assertIn("PaperStorm Agent Workspace", home.text)

            styles = client.get("/styles.css")
            self.assertEqual(styles.status_code, 200)
            self.assertIn("workspace-shell", styles.text)
            self.assertIn("no-store", styles.headers["cache-control"])

            app_js = client.get("/app.js")
            self.assertEqual(app_js.status_code, 200)
            self.assertIn("loadBenchmarkCatalog", app_js.text)
            self.assertIn("no-store", app_js.headers["cache-control"])

            with client.stream("GET", "/events?once=true") as response:
                self.assertEqual(response.status_code, 200)
                self.assertIn("text/event-stream", response.headers["content-type"])
                first = next(response.iter_lines())
                self.assertIn("event: service", first)

    def test_fastapi_serves_only_allowlisted_task_artifacts(self):
        from fastapi.testclient import TestClient

        from examples.storm_examples.paperstorm_service_api import create_app

        with tempfile.TemporaryDirectory() as temp_dir:
            client = TestClient(create_app(service_root=Path(temp_dir)))
            task = client.post(
                "/research-tasks",
                json={"topic": "PIM", "run_mode": "fake"},
            ).json()
            client.post("/research-tasks/{0}/run".format(task["task_id"]))
            output_dir = Path(task["output_dir"])
            (output_dir / "paperstorm_report.pdf").write_bytes(b"%PDF-test")

            pdf = client.get(
                "/research-tasks/{0}/artifacts/paperstorm_report.pdf".format(
                    task["task_id"]
                )
            )
            forbidden = client.get(
                "/research-tasks/{0}/artifacts/secrets.toml".format(task["task_id"])
            )
            missing = client.get(
                "/research-tasks/{0}/artifacts/paperstorm_report.print.html".format(
                    task["task_id"]
                )
            )

        self.assertEqual(pdf.status_code, 200)
        self.assertEqual(pdf.content, b"%PDF-test")
        self.assertEqual(forbidden.status_code, 403)
        self.assertEqual(missing.status_code, 404)

    def test_paperstorm_run_mode_uses_injected_pipeline_runner(self):
        calls = []

        def runner(state):
            calls.append(state)
            output_dir = Path(state["output_dir"])
            (output_dir / "storm_gen_article_polished.txt").write_text(
                "# PIM\n\npassive intermodulation suppression with RF neural networks.",
                encoding="utf-8",
            )
            (output_dir / "paperstorm_trace.jsonl").write_text(
                json.dumps({"event": "run_start", "task_id": state["task_id"]})
                + "\n",
                encoding="utf-8",
            )
            (output_dir / "scorecard.json").write_text(
                json.dumps({"scores": {"total": 88}}, ensure_ascii=False),
                encoding="utf-8",
            )
            return {"artifacts": ["storm_gen_article_polished.txt"], "success": True}

        service = self.make_service_with_pipeline_runner(runner)
        task = service.submit_research_task(
            topic="pim 神经网络抑制",
            retriever="arxiv",
            output_language="zh",
            run_mode="paperstorm",
            llm_provider="deepseek",
            llm_model="flash",
        )

        finished = service.run_task(task["task_id"])

        self.assertEqual(finished["status"], "succeeded")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["run_mode"], "paperstorm")
        self.assertEqual(calls[0]["options"]["llm_provider"], "deepseek")
        self.assertIn("passive intermodulation", service.get_article(task["task_id"])["content"])
        self.assertEqual(service.get_scorecard(task["task_id"])["scores"]["total"], 88)
        self.assertTrue(service.get_trace(task["task_id"])["events"])

    def test_paperstorm_run_mode_records_structured_runner_failure(self):
        def runner(state):
            raise RuntimeError("provider failed with sk-secret-value")

        service = self.make_service_with_pipeline_runner(runner)
        task = service.submit_research_task(
            topic="broken",
            run_mode="paperstorm",
            api_key="sk-secret-value",
        )

        failed = service.run_task(task["task_id"])

        self.assertEqual(failed["status"], "failed")
        self.assertIn("provider failed", failed["error"])
        self.assertNotIn("sk-secret-value", json.dumps(failed, ensure_ascii=False))

    def test_optional_pdf_is_registered_and_exposed_in_dashboard_bundle(self):
        class FakePdfRenderer:
            def render(self, markdown_path, output_pdf, title):
                Path(output_pdf).write_bytes(b"%PDF-generated")
                Path(output_pdf).with_suffix(".print.html").write_text(
                    "<html>report</html>", encoding="utf-8"
                )
                return {
                    "path": str(output_pdf),
                    "html_path": str(Path(output_pdf).with_suffix(".print.html")),
                    "page_count": 2,
                    "text_length": 120,
                    "size_bytes": Path(output_pdf).stat().st_size,
                }

        service = self.make_service(pdf_renderer=FakePdfRenderer())
        task = service.submit_research_task(
            topic="PIM 神经网络抑制",
            run_mode="fake",
            generate_pdf=True,
        )

        finished = service.run_task(task["task_id"])
        bundle = service.get_dashboard_bundle(task["task_id"])
        artifact_path = service.get_artifact_path(
            task["task_id"], "paperstorm_report.pdf"
        )

        self.assertEqual(finished["status"], "succeeded")
        self.assertEqual(finished["artifacts"]["pdf"]["status"], "ready")
        self.assertEqual(finished["artifacts"]["pdf"]["page_count"], 2)
        self.assertEqual(bundle["artifacts"]["pdf"]["status"], "ready")
        self.assertTrue(bundle["artifacts"]["pdf"]["url"].endswith("paperstorm_report.pdf"))
        self.assertTrue(artifact_path.exists())
        trace_events = service.get_trace(task["task_id"])["events"]
        self.assertTrue(
            any(
                event.get("event") == "stage_end"
                and event.get("stage") == "deliver"
                and event.get("output_summary", {}).get("pdf") == "ready"
                for event in trace_events
            )
        )

    def test_pdf_failure_keeps_research_success_and_reports_typed_error(self):
        from knowledge_storm.paperstorm_pdf import PdfRenderError

        class BrokenPdfRenderer:
            def render(self, *args, **kwargs):
                raise PdfRenderError(
                    "pdf_renderer_unavailable", "没有找到 Edge 或 Chrome"
                )

        service = self.make_service(pdf_renderer=BrokenPdfRenderer())
        task = service.submit_research_task(
            topic="PIM",
            run_mode="fake",
            generate_pdf=True,
        )

        finished = service.run_task(task["task_id"])

        self.assertEqual(finished["status"], "succeeded")
        self.assertEqual(finished["artifacts"]["pdf"]["status"], "failed")
        self.assertEqual(
            finished["artifacts"]["pdf"]["error_type"],
            "pdf_renderer_unavailable",
        )
        self.assertTrue(service.get_article(task["task_id"])["content"])

    def test_artifact_lookup_is_allowlisted_and_rejects_traversal(self):
        service = self.make_service()
        task = service.submit_research_task(topic="PIM", run_mode="fake")
        service.run_task(task["task_id"])
        output_dir = Path(service.get_task(task["task_id"])["output_dir"])
        (output_dir / "paperstorm_report.pdf").write_bytes(b"%PDF-test")

        self.assertEqual(
            service.get_artifact_path(task["task_id"], "paperstorm_report.pdf"),
            output_dir / "paperstorm_report.pdf",
        )
        with self.assertRaises(PermissionError):
            service.get_artifact_path(task["task_id"], "../tasks.json")
        with self.assertRaises(PermissionError):
            service.get_artifact_path(task["task_id"], "secrets.toml")


if __name__ == "__main__":
    unittest.main()
