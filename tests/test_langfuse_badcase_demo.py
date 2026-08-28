import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest import mock


class _FailingLangfuseClient:
    def start_observation(self, **_payload):
        raise RuntimeError("collector token=super-secret is unavailable")


class _RemoteObservation:
    trace_id = "langfuse-trace-123"

    def start_observation(self, **_payload):
        return _RemoteObservation()

    def score_trace(self, **_payload):
        return None

    def update(self, **_payload):
        return self

    def end(self):
        return None


class _RemoteLangfuseClient:
    def start_observation(self, **_payload):
        return _RemoteObservation()

    def flush(self):
        return None


class LangfuseBadcaseDemoTest(unittest.TestCase):
    _MODULE_NAMES = (
        "knowledge_storm",
        "knowledge_storm.paperstorm_observability",
        "knowledge_storm.langfuse_badcase_demo",
    )
    _MISSING = object()

    def setUp(self):
        self._original_modules = {
            name: sys.modules.get(name, self._MISSING)
            for name in self._MODULE_NAMES
        }

    def tearDown(self):
        for name, module in self._original_modules.items():
            if module is self._MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    @staticmethod
    def _load_module(module_name):
        root = Path(__file__).resolve().parents[1]
        package = types.ModuleType("knowledge_storm")
        package.__path__ = [str(root / "knowledge_storm")]
        sys.modules["knowledge_storm"] = package
        for name in (
            "knowledge_storm.paperstorm_observability",
            "knowledge_storm.langfuse_badcase_demo",
        ):
            sys.modules.pop(name, None)
        for name in (
            "knowledge_storm.paperstorm_observability",
            "knowledge_storm.langfuse_badcase_demo",
        ):
            path = root / "knowledge_storm" / (name.rsplit(".", 1)[-1] + ".py")
            spec = spec_from_file_location(name, path)
            module = module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return sys.modules["knowledge_storm.langfuse_badcase_demo"]

    def _composite_case(self):
        return {
            "case_id": "composite-1",
            "question": "What did the trial recommend?",
            "expected_document_ids": ["paper-a", "paper-b"],
            "retrieved_documents": [
                {
                    "document_id": "paper-b",
                    "text": "The evidence says the treatment should be avoided.",
                }
            ],
            "reranked_documents": [
                {
                    "document_id": "paper-b",
                    "text": "The evidence says the treatment should be avoided.",
                }
            ],
            "context_documents": [
                {
                    "document_id": "paper-b",
                    "text": "The evidence says the treatment should be avoided.",
                }
            ],
            "answer": "I cannot answer this question.",
            "citations": ["paper-missing"],
            "answerable": True,
            "evidence_conflict": True,
            "latency_ms": 42.5,
            "metadata": {"api_key": "never-export-this"},
        }

    @staticmethod
    def _local_only_environment():
        return {
            "PAPERSTORM_OBSERVABILITY": "",
            "LANGFUSE_PUBLIC_KEY": "",
            "LANGFUSE_SECRET_KEY": "",
        }

    def test_demo_records_required_trace_stages_and_scores(self):
        run_badcase_demo = self._load_module("knowledge_storm.langfuse_badcase_demo").run_badcase_demo

        with mock.patch.dict(os.environ, self._local_only_environment(), clear=False), tempfile.TemporaryDirectory() as temp_dir:
            result = run_badcase_demo(self._composite_case(), output_dir=temp_dir)
            events_path = Path(result["local_events_path"])
            events = events_path.read_text(encoding="utf-8")
            rows = [json.loads(line) for line in events.splitlines()]

        self.assertTrue(result["paperstorm_trace_id"])
        self.assertIsNone(result["remote_trace_id"])
        self.assertEqual(result["scores"]["retrieval_recall_at_5"], 0.5)
        self.assertEqual(result["scores"]["citation_validity"], 0.0)
        self.assertEqual(result["scores"]["latency_ms"], 42.5)
        self.assertLess(result["scores"]["answer_groundedness"], 1.0)
        self.assertEqual(
            result["badcase_types"],
            [
                "retrieval_miss",
                "invalid_citation",
                "evidence_conflict",
                "wrong_abstention",
            ],
        )
        self.assertEqual(result["observability"]["status"], "local-only")
        self.assertEqual(
            [row["name"] for row in rows if row["event"] == "trace.start"],
            ["paperstorm.rag.badcase"],
        )
        self.assertEqual(
            [row["name"] for row in rows if row["event"] == "span.start"],
            ["route", "retrieve", "rerank", "context", "reader", "citation_validate"],
        )
        self.assertEqual(
            {row["name"] for row in rows if row["event"] == "score"},
            {
                "retrieval_recall_at_5",
                "citation_validity",
                "answer_groundedness",
                "latency_ms",
            },
        )
        self.assertNotIn("never-export-this", events)

    def test_exporter_failure_is_fail_open_and_keeps_sanitized_local_event(self):
        demo = self._load_module("knowledge_storm.langfuse_badcase_demo")
        run_badcase_demo = demo.run_badcase_demo
        PaperStormObservability = sys.modules[
            "knowledge_storm.paperstorm_observability"
        ].PaperStormObservability

        with tempfile.TemporaryDirectory() as temp_dir:
            observability = PaperStormObservability(
                temp_dir, enabled=True, langfuse_client=_FailingLangfuseClient()
            )
            result = run_badcase_demo(
                self._composite_case(),
                output_dir=temp_dir,
                observability=observability,
            )
            events = observability.events_path.read_text(encoding="utf-8")

        self.assertEqual(result["observability"]["status"], "degraded")
        self.assertGreater(result["observability"]["export_failures"], 0)
        self.assertNotIn("super-secret", events)
        self.assertNotIn("never-export-this", events)

    def test_missing_credentials_stays_local_only(self):
        run_badcase_demo = self._load_module("knowledge_storm.langfuse_badcase_demo").run_badcase_demo

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(
            os.environ, self._local_only_environment(), clear=False
        ):
            result = run_badcase_demo(self._composite_case(), output_dir=temp_dir)

        self.assertEqual(result["observability"]["status"], "local-only")
        self.assertFalse(result["observability"]["remote_enabled"])

    def test_cli_writes_report_from_case_file_and_selects_scenario(self):
        root = Path(__file__).resolve().parents[1]
        script = root / "examples" / "storm_examples" / "run_langfuse_badcase_demo.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            case_file = temp_path / "cases.json"
            case_file.write_text(
                json.dumps(
                    {
                        "scenarios": {
                            "custom": {
                                **self._composite_case(),
                                "case_id": "from-file",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--output-dir",
                    str(temp_path / "out"),
                    "--case-file",
                    str(case_file),
                    "--scenario",
                    "custom",
                ],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, **self._local_only_environment()},
            )
            report_path = temp_path / "out" / "langfuse_badcase_report.json"

            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(report["scenario"], "custom")
        self.assertEqual(report["case_id"], "from-file")
        self.assertEqual(report["result"]["badcase_types"][0], "retrieval_miss")

    def test_groundedness_uses_cjk_and_alphanumeric_terms(self):
        run_badcase_demo = self._load_module("knowledge_storm.langfuse_badcase_demo").run_badcase_demo
        case = self._composite_case()
        case["answer"] = "临床试验建议避免使用该治疗"
        case["context_documents"] = [
            {"document_id": "paper-b", "text": "临床试验证据建议避免使用该治疗。"}
        ]
        with mock.patch.dict(os.environ, self._local_only_environment(), clear=False), tempfile.TemporaryDirectory() as temp_dir:
            result = run_badcase_demo(case, output_dir=temp_dir)

        self.assertGreaterEqual(result["scores"]["answer_groundedness"], 0.8)

    def test_citation_must_reference_context_not_only_retrieval_candidate(self):
        run_badcase_demo = self._load_module("knowledge_storm.langfuse_badcase_demo").run_badcase_demo
        case = self._composite_case()
        case["retrieved_documents"] = [{"document_id": "paper-a", "text": "candidate"}]
        case["reranked_documents"] = [{"document_id": "paper-a", "text": "candidate"}]
        case["context_documents"] = [{"document_id": "paper-b", "text": "context"}]
        case["citations"] = ["paper-a"]
        with mock.patch.dict(os.environ, self._local_only_environment(), clear=False), tempfile.TemporaryDirectory() as temp_dir:
            result = run_badcase_demo(case, output_dir=temp_dir)

        self.assertEqual(result["scores"]["citation_validity"], 0.0)
        self.assertIn("invalid_citation", result["badcase_types"])

    def test_case_schema_rejects_missing_and_invalid_required_values(self):
        run_badcase_demo = self._load_module("knowledge_storm.langfuse_badcase_demo").run_badcase_demo
        cases = []
        for field in (
            "case_id", "question", "expected_document_ids", "retrieved_documents",
            "reranked_documents", "context_documents", "answer", "citations",
            "answerable", "latency_ms",
        ):
            case = self._composite_case()
            case.pop(field)
            cases.append(case)
        invalid_expected = self._composite_case()
        invalid_expected["expected_document_ids"] = []
        cases.append(invalid_expected)
        invalid_answerable = self._composite_case()
        invalid_answerable["answerable"] = "true"
        cases.append(invalid_answerable)

        with tempfile.TemporaryDirectory() as temp_dir:
            for case in cases:
                with self.assertRaises((TypeError, ValueError)):
                    run_badcase_demo(case, output_dir=temp_dir)

    def test_remote_and_paperstorm_trace_ids_are_distinct(self):
        demo = self._load_module("knowledge_storm.langfuse_badcase_demo")
        PaperStormObservability = sys.modules[
            "knowledge_storm.paperstorm_observability"
        ].PaperStormObservability
        with tempfile.TemporaryDirectory() as temp_dir:
            observer = PaperStormObservability(
                temp_dir, enabled=True, langfuse_client=_RemoteLangfuseClient()
            )
            result = demo.run_badcase_demo(
                self._composite_case(), output_dir=temp_dir, observability=observer
            )

        self.assertTrue(result["paperstorm_trace_id"])
        self.assertEqual(result["remote_trace_id"], "langfuse-trace-123")
        self.assertNotEqual(result["paperstorm_trace_id"], result["remote_trace_id"])

    def test_cli_restores_preexisting_knowledge_storm_modules(self):
        root = Path(__file__).resolve().parents[1]
        script_path = root / "examples" / "storm_examples" / "run_langfuse_badcase_demo.py"
        spec = spec_from_file_location("langfuse_badcase_cli_test", script_path)
        cli = module_from_spec(spec)
        spec.loader.exec_module(cli)
        names = (
            "knowledge_storm",
            "knowledge_storm.paperstorm_observability",
            "knowledge_storm.langfuse_badcase_demo",
        )
        sentinel = {name: types.ModuleType(name) for name in names}
        original = {name: sys.modules.get(name) for name in names}
        sys.modules.update(sentinel)
        try:
            with mock.patch.dict(os.environ, self._local_only_environment(), clear=False), tempfile.TemporaryDirectory() as temp_dir:
                self.assertEqual(cli.main(["--output-dir", temp_dir]), 0)
            for name in names:
                self.assertIs(sys.modules[name], sentinel[name])
        finally:
            for name, module in original.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    def test_cli_loader_restores_modules_when_dynamic_loading_fails(self):
        root = Path(__file__).resolve().parents[1]
        script_path = root / "examples" / "storm_examples" / "run_langfuse_badcase_demo.py"
        spec = spec_from_file_location("langfuse_badcase_cli_failure_test", script_path)
        cli = module_from_spec(spec)
        spec.loader.exec_module(cli)
        names = (
            "knowledge_storm",
            "knowledge_storm.paperstorm_observability",
            "knowledge_storm.langfuse_badcase_demo",
        )
        sentinel = {name: types.ModuleType(name) for name in names}
        original = {name: sys.modules.get(name) for name in names}
        real_spec_from_file_location = cli.spec_from_file_location

        def fail_for_demo(module_name, path):
            if module_name.endswith("langfuse_badcase_demo"):
                raise RuntimeError("dynamic loader failed")
            return real_spec_from_file_location(module_name, path)

        sys.modules.update(sentinel)
        try:
            with mock.patch.object(cli, "spec_from_file_location", side_effect=fail_for_demo):
                with self.assertRaisesRegex(RuntimeError, "dynamic loader failed"):
                    cli._load_demo_module()
            for name in names:
                self.assertIs(sys.modules[name], sentinel[name])
        finally:
            for name, module in original.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    def test_default_case_is_a_fixed_composite_badcase(self):
        DEFAULT_COMPOSITE_BADCASE = self._load_module(
            "knowledge_storm.langfuse_badcase_demo"
        ).DEFAULT_COMPOSITE_BADCASE

        self.assertEqual(DEFAULT_COMPOSITE_BADCASE["case_id"], "fixed-composite-badcase")
        self.assertTrue(DEFAULT_COMPOSITE_BADCASE["evidence_conflict"])
        self.assertTrue(DEFAULT_COMPOSITE_BADCASE["answerable"])
        self.assertEqual(DEFAULT_COMPOSITE_BADCASE["citations"], ["paper-missing"])


if __name__ == "__main__":
    unittest.main()
