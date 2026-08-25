import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


class FakeProcess:
    def __init__(self):
        self.returncode = None
        self.pid = 4242

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15


class PaperStormBenchmarkRegistryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.benchmark_root = self.root / "benchmarks"
        (self.benchmark_root / "datasets" / "scifact" / "qrels").mkdir(
            parents=True
        )
        for relative in (
            "datasets/scifact/corpus.jsonl",
            "datasets/scifact/queries.jsonl",
            "datasets/scifact/qrels/test.tsv",
            "qasper-official-v0.3/qasper-test-v0.3.json",
            "v56/longmemeval_s_cleaned.json",
            "v56/runs/qasper-context/predictions.jsonl",
        ):
            path = self.benchmark_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_catalog_prioritizes_v55_v56_and_resolves_local_inputs(self):
        from knowledge_storm.paperstorm_benchmarks import BenchmarkRegistry

        with mock.patch.dict(
            os.environ,
            {"PAPERSTORM_BENCHMARK_ROOT": str(self.benchmark_root)},
            clear=False,
        ):
            catalog = BenchmarkRegistry().catalog()

        ids = [item["id"] for item in catalog["benchmarks"]]
        self.assertEqual(
            ids[:5],
            [
                "scifact-retrieval",
                "qasper-retrieval",
                "qasper-answer",
                "longmemeval-retrieval",
                "qasper-context",
            ],
        )
        self.assertEqual(catalog["benchmark_root"], str(self.benchmark_root))
        self.assertTrue(catalog["benchmarks"][0]["ready"])
        self.assertTrue(
            catalog["benchmarks"][0]["inputs"][0]["path"].endswith("scifact")
        )
        self.assertEqual(catalog["benchmarks"][-1]["status"], "blocked")

    def test_command_builder_is_allowlisted_and_applies_smoke_profile(self):
        from knowledge_storm.paperstorm_benchmarks import BenchmarkRegistry

        registry = BenchmarkRegistry(benchmark_root=self.benchmark_root)
        command = registry.build_command(
            "longmemeval-retrieval",
            output_dir=self.root / "run",
            profile="smoke",
        )

        self.assertEqual(Path(command[0]).name.lower(), "python.exe")
        self.assertIn("run_longmemeval_benchmark.py", command[1])
        self.assertEqual(command[command.index("--limit") + 1], "10")
        with self.assertRaises(KeyError):
            registry.build_command("arbitrary-shell-command", self.root / "bad")

    def test_paid_llm_benchmark_requires_explicit_confirmation_and_key(self):
        from knowledge_storm.paperstorm_benchmarks import BenchmarkRegistry

        registry = BenchmarkRegistry(benchmark_root=self.benchmark_root)
        with self.assertRaisesRegex(ValueError, "付费 LLM"):
            registry.build_command(
                "qasper-answer", self.root / "answer", profile="smoke"
            )
        with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}, clear=False):
            with self.assertRaisesRegex(ValueError, "DEEPSEEK_API_KEY"):
                registry.build_command(
                    "qasper-answer",
                    self.root / "answer",
                    profile="smoke",
                    allow_paid_llm=True,
                )

    def test_run_manager_persists_status_log_and_cancel(self):
        from knowledge_storm.paperstorm_benchmarks import (
            BenchmarkRegistry,
            BenchmarkRunManager,
        )

        process = FakeProcess()
        manager = BenchmarkRunManager(
            self.root / "service",
            registry=BenchmarkRegistry(benchmark_root=self.benchmark_root),
            popen_factory=lambda *args, **kwargs: process,
        )
        run = manager.start("longmemeval-retrieval", profile="smoke")
        self.assertEqual(run["status"], "running")
        self.assertTrue(Path(run["manifest_path"]).exists())
        cancelled = manager.cancel(run["run_id"])
        self.assertEqual(cancelled["status"], "cancelled")
        self.assertEqual(process.returncode, -15)


class PaperStormBenchmarkWorkbenchApiTest(unittest.TestCase):
    def test_fastapi_exposes_catalog_and_validates_paid_run(self):
        from fastapi.testclient import TestClient
        from examples.storm_examples.paperstorm_service_api import create_app

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with mock.patch.dict(
                os.environ,
                {"PAPERSTORM_BENCHMARK_ROOT": str(root / "missing")},
                clear=False,
            ):
                client = TestClient(create_app(service_root=root / "service"))
                response = client.get("/benchmarks/catalog")
                self.assertEqual(response.status_code, 200)
                self.assertIn("benchmarks", response.json())
                blocked = client.post(
                    "/benchmarks/runs",
                    json={"benchmark_id": "qasper-answer"},
                )
                self.assertEqual(blocked.status_code, 400)

    def test_frontend_is_a_registry_workbench_not_a_version_museum(self):
        html = (ROOT / "frontend/paperstorm_dashboard/index.html").read_text(
            encoding="utf-8"
        )
        script = (ROOT / "frontend/paperstorm_dashboard/app.js").read_text(
            encoding="utf-8"
        )
        for marker in (
            "benchmark-readiness",
            "benchmark-catalog",
            "benchmark-run-workspace",
            "benchmark-command-preview",
            "benchmark-log-tail",
            "benchmark-result-metrics",
        ):
            self.assertIn(marker, html)
        for obsolete in (
            "rag-eval-v4-panel",
            "rag-eval-v41-panel",
            "context-v42-panel",
            "memory-v43-panel",
        ):
            self.assertNotIn(obsolete, html)
        self.assertIn("loadBenchmarkCatalog", script)
        self.assertIn("startBenchmarkRun", script)
        self.assertIn("pollBenchmarkRun", script)
        self.assertIn("result.modes?.paperstorm_memory", script)


if __name__ == "__main__":
    unittest.main()
