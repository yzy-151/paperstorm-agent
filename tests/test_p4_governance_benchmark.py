import json
import tempfile
import unittest
from pathlib import Path


class P4GovernanceBenchmarkTest(unittest.TestCase):
    def test_offline_suite_writes_auditable_governance_artifacts(self):
        from knowledge_storm.paperstorm_benchmarks import (
            run_production_governance_benchmark,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "production-governance"
            report = run_production_governance_benchmark(output_dir)

            self.assertEqual("completed", report["status"])
            self.assertEqual(0, report["metrics"]["acl_leak_count"])
            self.assertEqual(0, report["metrics"]["secret_leak_count"])
            self.assertTrue(report["metrics"]["cache_isolated"])
            self.assertTrue(report["metrics"]["timeout_classified"])
            self.assertTrue(report["metrics"]["circuit_recovered"])
            self.assertTrue(report["metrics"]["batch_order_preserved"])
            self.assertTrue(report["metrics"]["release_gate_allowed"])
            self.assertTrue(report["metrics"]["release_gate_blocks_bad_candidate"])
            self.assertIn("acl_leak", report["negative_release_gate"]["reasons"])
            self.assertIn("p95_regression", report["negative_release_gate"]["reasons"])
            self.assertGreaterEqual(report["metrics"]["p95_ms"], 0.0)

            for name in (
                "manifest.json",
                "metrics.json",
                "predictions.jsonl",
                "case_dossiers.jsonl",
            ):
                self.assertTrue((output_dir / name).is_file(), name)

            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual("production-governance-v1", manifest["protocol"])
            self.assertFalse(manifest["requires_network"])
            self.assertFalse(manifest["requires_llm"])

    def test_milestone_cli_runs_only_the_p4_offline_suite(self):
        from examples.storm_examples.run_paperstorm_milestone import main

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            summary = main(
                [
                    "--milestone",
                    "P1+P2+P3+P4",
                    "--benchmark-root",
                    str(root / "unused-benchmarks"),
                    "--output-dir",
                    str(root / "output"),
                ]
            )

            self.assertEqual("completed", summary["status"])
            self.assertEqual(
                {"production-governance"}, set(summary["benchmarks"])
            )
            self.assertTrue(
                (root / "output" / "production-governance" / "metrics.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
