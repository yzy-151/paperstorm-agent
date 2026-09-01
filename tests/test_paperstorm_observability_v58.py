import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class _FakeObservation:
    def __init__(self, name, trace_id="trace-1", **payload):
        self.name = name
        self.trace_id = trace_id
        self.payload = payload
        self.children = []
        self.updates = []
        self.scores = []
        self.ended = False

    def start_observation(self, **payload):
        child = _FakeObservation(payload.pop("name"), trace_id=self.trace_id, **payload)
        self.children.append(child)
        return child

    def update(self, **payload):
        self.updates.append(payload)
        return self

    def score_trace(self, **payload):
        self.scores.append(payload)

    def end(self):
        self.ended = True


class _FakeLangfuseClient:
    def __init__(self, fail=False):
        self.fail = fail
        self.roots = []
        self.flushed = False

    def start_observation(self, **payload):
        if self.fail:
            raise RuntimeError("collector unavailable")
        root = _FakeObservation(payload.pop("name"), **payload)
        self.roots.append(root)
        return root

    def flush(self):
        self.flushed = True


class PaperStormObservabilityV58Test(unittest.TestCase):
    def test_recursive_redaction_masks_credentials_and_user_identity(self):
        from knowledge_storm.paperstorm_observability import sanitize_payload

        payload = sanitize_payload(
            {
                "api_key": "sk-secret",
                "authorization": "Bearer token",
                "user_id": "alice@example.com",
                "prompt_tokens": 321,
                "nested": [{"password": "secret", "question": "safe"}],
            }
        )

        self.assertEqual(payload["api_key"], "***REDACTED***")
        self.assertEqual(payload["authorization"], "***REDACTED***")
        self.assertNotEqual(payload["user_id"], "alice@example.com")
        self.assertTrue(payload["user_id"].startswith("user_"))
        self.assertEqual(payload["prompt_tokens"], 321)
        self.assertEqual(payload["nested"][0]["password"], "***REDACTED***")
        self.assertEqual(payload["nested"][0]["question"], "safe")

    def test_local_exporter_records_trace_span_score_and_completion(self):
        from knowledge_storm.paperstorm_observability import PaperStormObservability

        with tempfile.TemporaryDirectory() as root:
            observability = PaperStormObservability(Path(root), enabled=False)
            with observability.trace(
                "paperstorm.research",
                input={"topic": "RAG"},
                metadata={"task_id": "task-1"},
            ) as trace:
                with trace.span("retrieval", input={"query": "RAG"}) as span:
                    span.end(output={"documents": 3})
                trace.score("retrieval_recall", 0.8)
                trace.end(output={"status": "succeeded"})

            rows = [
                json.loads(line)
                for line in (Path(root) / "observability" / "events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                [row["event"] for row in rows],
                ["trace.start", "span.start", "span.end", "score", "trace.end"],
            )
            self.assertEqual(rows[1]["parent_id"], rows[0]["trace_id"])
            self.assertEqual(rows[3]["name"], "retrieval_recall")

    def test_langfuse_exporter_is_optional_and_fail_open(self):
        from knowledge_storm.paperstorm_observability import PaperStormObservability

        with tempfile.TemporaryDirectory() as root:
            observability = PaperStormObservability(
                Path(root), enabled=True, langfuse_client=_FakeLangfuseClient(fail=True)
            )
            with observability.trace("paperstorm.chat", input={"message": "hello"}) as trace:
                trace.end(output={"answer": "hi"})

            status = observability.status()
            self.assertEqual(status["provider"], "langfuse")
            self.assertEqual(status["status"], "degraded")
            self.assertEqual(status["export_failures"], 1)
            self.assertTrue((Path(root) / "observability" / "events.jsonl").exists())

    def test_local_mirror_failure_is_fail_open_and_visible_in_status(self):
        from knowledge_storm.paperstorm_observability import PaperStormObservability

        with tempfile.TemporaryDirectory() as root:
            observability = PaperStormObservability(Path(root), enabled=False)
            with mock.patch(
                "knowledge_storm.paperstorm_observability.Path.open",
                side_effect=OSError("disk unavailable"),
            ):
                with observability.trace("paperstorm.chat") as trace:
                    trace.end(output={"answer": "still returned"})

            status = observability.status()
            self.assertEqual(status["status"], "degraded")
            self.assertEqual(status["local_write_failures"], 2)
            self.assertIn("disk unavailable", status["last_error"])

    def test_langfuse_exporter_builds_nested_observations_and_scores(self):
        from knowledge_storm.paperstorm_observability import PaperStormObservability

        client = _FakeLangfuseClient()
        with tempfile.TemporaryDirectory() as root:
            observability = PaperStormObservability(
                Path(root), enabled=True, langfuse_client=client
            )
            with observability.trace(
                "paperstorm.chat",
                input={"message": "hello"},
                session_id="chat-1",
                user_id="alice@example.com",
            ) as trace:
                with trace.span("intent_route", as_type="chain") as span:
                    span.end(output={"intent": "chat"})
                trace.score("trajectory_success", 1.0)
                trace.end(output={"answer": "hi"})
            observability.flush()

        root = client.roots[0]
        self.assertEqual(root.name, "paperstorm.chat")
        self.assertEqual(root.payload["metadata"]["langfuse_session_id"], "chat-1")
        self.assertTrue(
            root.payload["metadata"]["langfuse_user_id"].startswith("user_")
        )
        self.assertEqual(root.children[0].name, "intent_route")
        self.assertTrue(root.children[0].ended)
        self.assertEqual(root.scores[0]["name"], "trajectory_success")
        self.assertTrue(root.ended)
        self.assertTrue(client.flushed)

    def test_nested_span_and_generation_preserve_parent_and_usage(self):
        from knowledge_storm.paperstorm_observability import PaperStormObservability

        client = _FakeLangfuseClient()
        with tempfile.TemporaryDirectory() as root:
            observability = PaperStormObservability(
                Path(root), enabled=True, langfuse_client=client
            )
            with observability.trace("paperstorm.research") as trace:
                with trace.span("research_pipeline", as_type="chain") as pipeline:
                    with pipeline.span("outline_generation", input={"topic": "RAG"}) as stage:
                        with stage.generation(
                            "outline_gen",
                            model="deepseek/deepseek-chat",
                            input={"messages": [{"role": "user", "content": "outline"}]},
                        ) as generation:
                            generation.end(
                                output={"content": "# Outline"},
                                usage={"prompt_tokens": 10, "completion_tokens": 4},
                                cost_usd=0.002,
                                finish_reason="stop",
                            )
                        stage.end(output={"sections": 3})
                    pipeline.end(output={"status": "succeeded"})
                trace.end(output={"status": "succeeded"})

            rows = [
                json.loads(line)
                for line in observability.events_path.read_text(encoding="utf-8").splitlines()
            ]

        root_observation = client.roots[0]
        pipeline = root_observation.children[0]
        stage = pipeline.children[0]
        generation = stage.children[0]
        self.assertEqual(stage.name, "outline_generation")
        self.assertEqual(generation.name, "outline_gen")
        self.assertEqual(generation.payload["as_type"], "generation")
        self.assertEqual(generation.payload["model"], "deepseek/deepseek-chat")
        self.assertEqual(
            generation.updates[-1]["usage_details"],
            {"input": 10, "output": 4, "total": 14},
        )
        self.assertEqual(generation.updates[-1]["cost_details"]["total"], 0.002)
        self.assertIn("generation.start", [row["event"] for row in rows])
        self.assertIn("generation.end", [row["event"] for row in rows])

    def test_litellm_model_exports_each_call_as_generation(self):
        from knowledge_storm.lm import LitellmModel
        from knowledge_storm.paperstorm_observability import PaperStormObservability

        class Response(dict):
            def json(self):
                return dict(self)

        response = Response(
            choices=[{"text": "answer", "finish_reason": "stop"}],
            usage={"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
            _hidden_params={"response_cost": 0.001},
        )
        client = _FakeLangfuseClient()
        with tempfile.TemporaryDirectory() as root:
            observer = PaperStormObservability(Path(root), enabled=True, langfuse_client=client)
            with observer.trace("paperstorm.research") as trace:
                with trace.span("research_pipeline") as pipeline:
                    model = LitellmModel(
                        model="test/model",
                        model_type="text",
                        cache=False,
                        generation_observer=pipeline.generation,
                        generation_name="outline_generation",
                    )
                    with mock.patch("knowledge_storm.lm.litellm_text_completion", return_value=response):
                        self.assertEqual(model(prompt="outline"), ["answer"])
                    pipeline.end(output={"status": "succeeded"})
                trace.end(output={"status": "succeeded"})

        generation = client.roots[0].children[0].children[0]
        self.assertEqual(generation.name, "outline_generation")
        self.assertEqual(generation.payload["as_type"], "generation")
        self.assertEqual(generation.updates[-1]["usage_details"]["total"], 10)
        self.assertEqual(generation.updates[-1]["cost_details"]["total"], 0.001)

    def test_trace_recorder_bridges_local_stages_to_nested_langfuse_spans(self):
        from knowledge_storm.paperstorm_observability import PaperStormObservability
        from knowledge_storm.paperstorm_trace import PaperStormTraceRecorder

        client = _FakeLangfuseClient()
        with tempfile.TemporaryDirectory() as root:
            observability = PaperStormObservability(
                Path(root), enabled=True, langfuse_client=client
            )
            with observability.trace("paperstorm.research") as trace:
                with trace.span("research_pipeline") as pipeline:
                    recorder = PaperStormTraceRecorder(
                        str(Path(root) / "article"), observation_parent=pipeline
                    )
                    recorder.start_stage("persona", "generate perspectives", input={"topic": "RAG"})
                    recorder.end_stage(output_summary={"perspective_count": 3})
                    recorder.emit(
                        "stage_start",
                        stage="retrieval",
                        invocation_id="query-1",
                        operation="arxiv search",
                        input={"queries": ["RAG"]},
                    )
                    recorder.emit(
                        "stage_end",
                        stage="retrieval",
                        invocation_id="query-1",
                        duration_ms=25,
                        output_summary={"result_count": 5},
                    )
                    pipeline.end(output={"status": "succeeded"})
                trace.end(output={"status": "succeeded"})

        pipeline = client.roots[0].children[0]
        self.assertEqual([item.name for item in pipeline.children], ["persona", "retrieval"])
        self.assertEqual(
            pipeline.children[1].updates[-1]["output"]["result_count"], 5
        )

    def test_chat_spans_export_router_memory_and_generation_details(self):
        from knowledge_storm.paperstorm_observability import PaperStormObservability
        from knowledge_storm.paperstorm_service import PaperStormTaskService

        client = _FakeLangfuseClient()
        with tempfile.TemporaryDirectory() as root:
            observer = PaperStormObservability(Path(root), enabled=True, langfuse_client=client)
            service = PaperStormTaskService(Path(root), observability=observer)
            session = service.create_chat_session(run_mode="fake", user_id="alice")
            service.send_chat_message(session["chat_id"], "请记住：回答先给结论")

        root_trace = next(item for item in client.roots if item.name == "paperstorm.chat")
        spans = {item.name: item for item in root_trace.children}
        self.assertIn("classify", spans)
        self.assertIn("memory_candidate_write", spans)
        memory_output = spans["memory_candidate_write"].updates[-1]["output"]
        self.assertIn("memory_write", memory_output)
        self.assertIn(memory_output["memory_write"]["status"], {"persisted", "queued", "skipped"})
        classify_output = spans["classify"].updates[-1]["output"]
        self.assertIn("router_decision", classify_output)
        score_names = {item["name"] for item in root_trace.scores}
        self.assertIn("planner_fallback", score_names)
        self.assertIn("memory_write_success", score_names)

    def test_regular_chat_does_not_score_skipped_memory_as_failure(self):
        from knowledge_storm.paperstorm_observability import PaperStormObservability
        from knowledge_storm.paperstorm_service import PaperStormTaskService

        client = _FakeLangfuseClient()
        with tempfile.TemporaryDirectory() as root:
            observer = PaperStormObservability(Path(root), enabled=True, langfuse_client=client)
            service = PaperStormTaskService(Path(root), observability=observer)
            session = service.create_chat_session(run_mode="fake", user_id="alice")
            service.send_chat_message(session["chat_id"], "你好")

        root_trace = next(item for item in client.roots if item.name == "paperstorm.chat")
        score_names = {item["name"] for item in root_trace.scores}
        self.assertNotIn("memory_write_success", score_names)

    def test_factory_is_disabled_without_credentials(self):
        from knowledge_storm.paperstorm_observability import build_observability

        with tempfile.TemporaryDirectory() as root, mock.patch.dict(os.environ, {}, clear=True):
            observability = build_observability(Path(root))
            self.assertFalse(observability.status()["remote_enabled"])
            self.assertEqual(observability.status()["status"], "local-only")

    def test_offline_test_mode_hard_disables_remote_export(self):
        from knowledge_storm.paperstorm_observability import build_observability

        environment = {
            "PAPERSTORM_TEST_OFFLINE": "1",
            "PAPERSTORM_OBSERVABILITY": "langfuse",
            "LANGFUSE_PUBLIC_KEY": "pk-test",
            "LANGFUSE_SECRET_KEY": "sk-test",
        }
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(
            os.environ, environment, clear=False
        ):
            observability = build_observability(Path(root))

        self.assertFalse(observability.status()["remote_enabled"])
        self.assertEqual(observability.status()["status"], "local-only")

    def test_task_service_emits_research_and_chat_harness_traces(self):
        from knowledge_storm.paperstorm_observability import PaperStormObservability
        from knowledge_storm.paperstorm_service import PaperStormTaskService

        with tempfile.TemporaryDirectory() as root:
            observability = PaperStormObservability(Path(root), enabled=False)
            service = PaperStormTaskService(Path(root), observability=observability)
            task = service.submit_research_task(topic="RAG", run_mode="fake")
            service.run_task(task["task_id"])
            session = service.create_chat_session(run_mode="fake", user_id="alice")
            service.send_chat_message(session["chat_id"], "你好")

            rows = [
                json.loads(line)
                for line in observability.events_path.read_text(encoding="utf-8").splitlines()
            ]
            trace_names = [row.get("name") for row in rows if row["event"] == "trace.start"]
            span_names = [row.get("name") for row in rows if row["event"] == "span.start"]
            self.assertIn("paperstorm.research", trace_names)
            self.assertIn("paperstorm.chat", trace_names)
            self.assertIn("research_pipeline", span_names)
            self.assertIn("classify", span_names)
            self.assertIn("casual_chat", span_names)
            self.assertIn("run_score", [row.get("name") for row in rows if row["event"] == "score"])

    def test_benchmark_completion_exports_numeric_metrics_as_scores(self):
        from knowledge_storm.paperstorm_benchmarks import BenchmarkRegistry, BenchmarkRunManager
        from knowledge_storm.paperstorm_observability import PaperStormObservability

        class Process:
            pid = 42
            returncode = None

            def poll(self):
                return self.returncode

            def terminate(self):
                self.returncode = -15

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            benchmark_root = root / "benchmarks"
            dataset = benchmark_root / "v56" / "longmemeval_s_cleaned.json"
            dataset.parent.mkdir(parents=True)
            dataset.write_text("{}", encoding="utf-8")
            process = Process()
            observability = PaperStormObservability(root, enabled=False)
            manager = BenchmarkRunManager(
                root / "service",
                registry=BenchmarkRegistry(benchmark_root=benchmark_root),
                popen_factory=lambda *args, **kwargs: process,
                observability=observability,
            )
            run = manager.start("longmemeval-retrieval", profile="smoke")
            output = Path(run["output_dir"])
            output.mkdir(parents=True)
            (output / "metrics.json").write_text(
                json.dumps({"recall_at_5": 0.8, "p95_latency_ms": 25.0}),
                encoding="utf-8",
            )
            process.returncode = 0
            manager.get(run["run_id"])

            rows = [
                json.loads(line)
                for line in observability.events_path.read_text(encoding="utf-8").splitlines()
            ]
            score_names = [row.get("name") for row in rows if row["event"] == "score"]
            self.assertIn("recall_at_5", score_names)
            self.assertIn("p95_latency_ms", score_names)

    def test_fastapi_exposes_observability_status_without_credentials(self):
        from fastapi.testclient import TestClient
        from examples.storm_examples.paperstorm_service_api import create_app

        clean = {
            "PAPERSTORM_OBSERVABILITY": "",
            "LANGFUSE_PUBLIC_KEY": "",
            "LANGFUSE_SECRET_KEY": "",
        }
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(
            os.environ, clean, clear=False
        ):
            client = TestClient(create_app(service_root=Path(root)))
            response = client.get("/observability/status")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["provider"], "langfuse")
        self.assertEqual(response.json()["status"], "local-only")

    def test_v58_ui_and_readme_expose_langfuse_configuration(self):
        root = Path(__file__).resolve().parents[1]
        html = (root / "frontend" / "paperstorm_dashboard" / "index.html").read_text(
            encoding="utf-8"
        )
        script = (root / "frontend" / "paperstorm_dashboard" / "app.js").read_text(
            encoding="utf-8"
        )
        readme = (root / "README.md").read_text(encoding="utf-8")

        self.assertIn('id="ready-langfuse"', html)
        self.assertIn('/observability/status', script)
        self.assertIn("PAPERSTORM_OBSERVABILITY", readme)
        self.assertIn("LANGFUSE_PUBLIC_KEY", readme)
        self.assertIn("Langfuse", readme)


if __name__ == "__main__":
    unittest.main()
