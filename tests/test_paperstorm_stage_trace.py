import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from knowledge_storm.paperstorm_trace import (
    PaperStormStageCallback,
    PaperStormTraceRecorder,
    TracedRetrievalModel,
    classify_runtime_error,
)


class PaperStormStageTraceTest(unittest.TestCase):
    def make_recorder(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        return PaperStormTraceRecorder(temp_dir.name)

    def test_stage_lifecycle_records_sanitized_live_telemetry(self):
        recorder = self.make_recorder()

        recorder.start_stage(
            "persona",
            "生成多视角角色",
            input={"topic": "PIM", "api_key": "sk-secret"},
        )
        recorder.progress_stage(
            operation="正在调用角色模型",
            output_summary={"preview": "x" * 2000},
            prompt_tokens=120,
        )
        recorder.end_stage(
            output_summary={"perspectives": ["射频专家", "算法专家"]},
            completion_tokens=80,
        )

        events = recorder.events
        self.assertEqual(
            [event["event"] for event in events],
            ["stage_start", "stage_progress", "stage_end"],
        )
        self.assertEqual(events[0]["stage"], "persona")
        self.assertEqual(events[0]["input"]["api_key"], "***REDACTED***")
        self.assertLessEqual(len(events[1]["output_summary"]["preview"]), 520)
        self.assertGreaterEqual(events[2]["duration_ms"], 0)
        self.assertEqual(events[2]["completion_tokens"], 80)
        self.assertIsNone(recorder.current_stage)

        written = [
            json.loads(line)
            for line in Path(recorder.trace_path).read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(written, events)

    def test_starting_next_stage_closes_previous_stage_once(self):
        recorder = self.make_recorder()

        recorder.start_stage("persona", "生成角色")
        recorder.start_stage("dialogue", "多 Agent 对话")

        self.assertEqual(
            [(event["event"], event["stage"]) for event in recorder.events],
            [
                ("stage_start", "persona"),
                ("stage_end", "persona"),
                ("stage_start", "dialogue"),
            ],
        )
        self.assertEqual(recorder.current_stage, "dialogue")

    def test_fail_current_stage_classifies_provider_connection_error(self):
        recorder = self.make_recorder()
        recorder.start_stage("request", "初始化模型")

        recorder.fail_current_stage(
            RuntimeError("OpenAIException - Connection error: sk-secret")
        )

        event = recorder.events[-1]
        self.assertEqual(event["event"], "stage_error")
        self.assertEqual(event["stage"], "request")
        self.assertEqual(event["error_type"], "provider_unavailable")
        self.assertNotIn("sk-secret", event["error_message"])
        self.assertIsNone(recorder.current_stage)

    def test_runtime_error_classifier_distinguishes_common_failures(self):
        cases = [
            (TimeoutError("request timed out"), "timeout"),
            (RuntimeError("401 invalid api key"), "authentication"),
            (RuntimeError("arxiv returned HTTP 503"), "retrieval_error"),
            (ValueError("invalid JSON response"), "parse_error"),
            (RuntimeError("connection refused"), "provider_unavailable"),
        ]

        for error, expected in cases:
            with self.subTest(error=error):
                self.assertEqual(classify_runtime_error(error), expected)

    def test_trace_redacts_bearer_and_key_value_credentials(self):
        recorder = self.make_recorder()

        recorder.fail_current_stage(
            RuntimeError(
                "Authorization: Bearer provider-token api_key=minimax-secret "
                "url=https://example.test?q=1&access_token=url-secret"
            ),
            stage="request",
        )

        message = recorder.events[-1]["error_message"]
        self.assertNotIn("provider-token", message)
        self.assertNotIn("minimax-secret", message)
        self.assertNotIn("url-secret", message)
        self.assertIn("REDACTED", message)

    def test_storm_callback_reports_persona_dialogue_and_evidence(self):
        recorder = self.make_recorder()
        callback = PaperStormStageCallback(recorder, topic="PIM 神经网络抑制")

        callback.on_identify_perspective_start()
        callback.on_identify_perspective_end(perspectives=["射频专家", "算法专家"])
        callback.on_information_gathering_start()
        callback.on_dialogue_turn_end(
            dlg_turn=SimpleNamespace(
                user_utterance="PIM 的主要来源是什么？",
                agent_utterance="连接器非线性是常见来源。",
            )
        )
        callback.on_information_gathering_end()

        pairs = [(event["event"], event["stage"]) for event in recorder.events]
        self.assertEqual(
            pairs,
            [
                ("stage_start", "persona"),
                ("stage_end", "persona"),
                ("artifact_ready", "persona"),
                ("stage_start", "dialogue"),
                ("stage_progress", "dialogue"),
                ("stage_end", "dialogue"),
                ("artifact_ready", "dialogue"),
                ("stage_start", "evidence"),
                ("stage_end", "evidence"),
                ("artifact_ready", "evidence"),
            ],
        )
        self.assertEqual(
            [
                event["artifact_name"]
                for event in recorder.events
                if event["event"] == "artifact_ready"
            ],
            ["personas.json", "conversation_log.json", "evidence_index.json"],
        )
        self.assertEqual(
            recorder.events[1]["output_summary"]["perspectives"],
            ["射频专家", "算法专家"],
        )

    def test_traced_retriever_reports_query_results_and_resumes_dialogue(self):
        recorder = self.make_recorder()
        recorder.start_stage("dialogue", "多 Agent 对话")

        retriever = TracedRetrievalModel(
            lambda query_or_queries, exclude_urls: [
                {"title": "Passive intermodulation survey", "url": "https://example.test/1"}
            ],
            trace_recorder=recorder,
            retriever_name="ArxivRM",
        )
        results = retriever(["passive intermodulation suppression"])

        self.assertEqual(len(results), 1)
        stages = [
            (event["event"], event.get("stage"))
            for event in recorder.events
            if event["event"].startswith("stage_")
        ]
        self.assertIn(("stage_start", "query"), stages)
        self.assertIn(("stage_end", "retrieval"), stages)
        self.assertEqual(recorder.current_stage, "dialogue")
        retrieval_end = next(
            event for event in recorder.events if event["event"] == "retrieval_end"
        )
        self.assertEqual(retrieval_end["result_count"], 1)
        self.assertEqual(
            retrieval_end["selected_titles"], ["Passive intermodulation survey"]
        )

    def test_concurrent_retrievals_keep_independent_stage_invocations(self):
        recorder = self.make_recorder()
        recorder.start_stage("dialogue", "多 Agent 对话")
        barrier = threading.Barrier(2)

        def retrieve(query_or_queries, exclude_urls):
            barrier.wait(timeout=2)
            return [{"title": str(query_or_queries), "url": "https://example.test"}]

        retriever = TracedRetrievalModel(retrieve, recorder, "ArxivRM")
        threads = [
            threading.Thread(target=retriever, args=(["query-a"],)),
            threading.Thread(target=retriever, args=(["query-b"],)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3)

        retrieval_events = [
            event
            for event in recorder.events
            if event.get("stage") == "retrieval"
            and event.get("event") in {"stage_start", "stage_end"}
        ]
        invocation_ids = {event.get("invocation_id") for event in retrieval_events}
        self.assertEqual(len(invocation_ids), 2)
        self.assertNotIn(None, invocation_ids)
        for invocation_id in invocation_ids:
            self.assertEqual(
                [
                    event["event"]
                    for event in retrieval_events
                    if event.get("invocation_id") == invocation_id
                ],
                ["stage_start", "stage_end"],
            )
        self.assertEqual(recorder.current_stage, "dialogue")

    def test_pipeline_attributes_bootstrap_connection_failure_to_request(self):
        from knowledge_storm.paperstorm_pipeline import (
            PaperStormPipelineConfig,
            run_paperstorm_pipeline,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            config = PaperStormPipelineConfig(
                topic="PIM",
                topic_for_storm="PIM",
                output_root=temp_dir,
                output_dir_name="run",
                article_dir=str(Path(temp_dir) / "run"),
            )
            with mock.patch(
                "knowledge_storm.paperstorm_pipeline._build_lm_configs",
                side_effect=RuntimeError("OpenAIException - Connection error"),
            ):
                with self.assertRaises(RuntimeError):
                    run_paperstorm_pipeline(config)

            events = [
                json.loads(line)
                for line in (Path(config.article_dir) / "paperstorm_trace.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        error = next(event for event in events if event["event"] == "stage_error")
        self.assertEqual(error["stage"], "request")
        self.assertEqual(error["error_type"], "provider_unavailable")

    def test_pipeline_emits_real_storm_callback_and_module_stages(self):
        from knowledge_storm.paperstorm_pipeline import (
            PaperStormPipelineConfig,
            run_paperstorm_pipeline,
        )

        class FakeRunner:
            def __init__(self, *_args):
                self.article_output_dir = ""

            def run_article_generation_module(self, *args, **kwargs):
                Path(self.article_output_dir, "storm_gen_article.txt").write_text(
                    "# Draft", encoding="utf-8"
                )
                return object()

            def run_article_polishing_module(self, *args, **kwargs):
                Path(
                    self.article_output_dir, "storm_gen_article_polished.txt"
                ).write_text("# Polished", encoding="utf-8")
                return object()

            def run(self, output_dir_name, callback_handler, **kwargs):
                self.article_output_dir = str(Path(config.output_root) / output_dir_name)
                callback_handler.on_identify_perspective_start()
                callback_handler.on_identify_perspective_end(["射频专家"])
                callback_handler.on_information_gathering_start()
                callback_handler.on_dialogue_turn_end(
                    SimpleNamespace(user_utterance="Q", agent_utterance="A")
                )
                callback_handler.on_information_gathering_end()
                callback_handler.on_information_organization_start()
                callback_handler.on_direct_outline_generation_end("# Draft outline")
                callback_handler.on_outline_refinement_end("# Refined outline")
                self.run_article_generation_module()
                self.run_article_polishing_module()

            def post_run(self):
                return None

            def summary(self):
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            config = PaperStormPipelineConfig(
                topic="PIM",
                topic_for_storm="PIM",
                output_root=temp_dir,
                output_dir_name="run",
                article_dir=str(Path(temp_dir) / "run"),
            )

            scorecard_write_states = []

            def write_scorecard(current_config):
                scorecard_write_states.append(
                    (Path(current_config.article_dir) / "run_summary.json").exists()
                )
                Path(current_config.article_dir, "scorecard.json").write_text(
                    '{"scores":{"total":88}}', encoding="utf-8"
                )

            with mock.patch(
                "knowledge_storm.paperstorm_pipeline._build_lm_configs",
                return_value=object(),
            ), mock.patch(
                "knowledge_storm.paperstorm_pipeline._build_paper_retriever",
                return_value=lambda *args, **kwargs: [],
            ), mock.patch(
                "knowledge_storm.paperstorm_pipeline.STORMWikiRunner",
                FakeRunner,
            ), mock.patch(
                "knowledge_storm.paperstorm_pipeline.ensure_research_sources",
                return_value=1,
            ), mock.patch(
                "knowledge_storm.paperstorm_pipeline._write_pipeline_scorecard",
                side_effect=write_scorecard,
            ):
                result = run_paperstorm_pipeline(config)

            self.assertTrue(scorecard_write_states[-1])

            self.assertTrue(result["success"])
            events = [
                json.loads(line)
                for line in (Path(config.article_dir) / "paperstorm_trace.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        started = [
            event["stage"] for event in events if event["event"] == "stage_start"
        ]
        for expected in (
            "request",
            "persona",
            "dialogue",
            "evidence",
            "outline",
            "writer",
            "polish",
            "evaluate",
            "deliver",
        ):
            self.assertIn(expected, started)

    def test_lm_history_delta_reports_actual_tokens_and_provider_cost(self):
        from knowledge_storm.paperstorm_pipeline import (
            _snapshot_lm_telemetry,
            _telemetry_delta,
        )

        lm = SimpleNamespace(history=[])
        configs = SimpleNamespace(article_gen_lm=lm)
        before = _snapshot_lm_telemetry(configs)
        lm.history.append(
            {
                "usage": {"prompt_tokens": 120, "completion_tokens": 45},
                "cost": 0.0123,
            }
        )
        after = _snapshot_lm_telemetry(configs)

        self.assertEqual(
            _telemetry_delta(before, after),
            {
                "prompt_tokens": 120,
                "completion_tokens": 45,
                "total_tokens": 165,
                "estimated_cost": 0.0123,
            },
        )

        without_cost = SimpleNamespace(history=[])
        no_cost_configs = SimpleNamespace(question_asker_lm=without_cost)
        before = _snapshot_lm_telemetry(no_cost_configs)
        without_cost.history.append(
            {"usage": {"prompt_tokens": 7, "completion_tokens": 3}}
        )
        delta = _telemetry_delta(
            before, _snapshot_lm_telemetry(no_cost_configs)
        )
        self.assertNotIn("estimated_cost", delta)


if __name__ == "__main__":
    unittest.main()
