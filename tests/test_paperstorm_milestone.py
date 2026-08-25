import json
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


class PaperStormMilestoneTest(unittest.TestCase):
    def test_parser_defaults_to_affected_p1_benchmarks(self):
        from examples.storm_examples.run_paperstorm_milestone import build_parser, resolve_benchmarks

        args = build_parser().parse_args(["--output-dir", "out", "--benchmark-root", "data"])
        self.assertEqual("P1", args.milestone)
        self.assertEqual(("pim", "scifact", "qasper-retrieval"), resolve_benchmarks(args))

    def test_cli_rejects_non_p1_milestones(self):
        from examples.storm_examples.run_paperstorm_milestone import build_parser

        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["--milestone", "P1+P2", "--output-dir", "out", "--benchmark-root", "data"])

    def test_missing_dataset_returns_machine_readable_blocked_and_continues(self):
        from examples.storm_examples.run_paperstorm_milestone import main

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with redirect_stdout(io.StringIO()):
                summary = main([
                    "--benchmark", "scifact", "pim",
                    "--benchmark-root", str(root / "missing"),
                    "--output-dir", str(root / "out"),
                    "--embedding", "hash",
                    "--smoke-limit", "2",
                ])
            self.assertEqual("blocked", summary["benchmarks"]["scifact"]["status"])
            self.assertEqual("completed", summary["benchmarks"]["pim"]["status"])

    def test_pim_run_writes_manifest_and_case_dossiers(self):
        from examples.storm_examples.run_paperstorm_milestone import main

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with redirect_stdout(io.StringIO()):
                summary = main([
                    "--benchmark", "pim",
                    "--benchmark-root", str(root),
                    "--output-dir", str(root / "out"),
                    "--embedding", "hash",
                    "--smoke-limit", "3",
                ])
            run_dir = Path(summary["benchmarks"]["pim"]["output_dir"])
            manifest = json.loads((run_dir / "milestone_manifest.json").read_text(encoding="utf-8"))
            dossiers = [json.loads(line) for line in (run_dir / "case_dossiers.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual("P1", manifest["milestone"])
            self.assertIn("baseline_reference", manifest)
            self.assertNotIn("secret", json.dumps(manifest).lower())
            self.assertTrue(any(row["before"].get("source") == "archived_observation" for row in dossiers))
            by_id = {row["case_id"]: row for row in dossiers}
            self.assertTrue(by_id["pim-rf-zh"]["after"]["resolved"])
            self.assertTrue(by_id["pim-lexical"]["after"]["resolved"])
            self.assertFalse(by_id["pim-ambiguous"]["after"]["resolved"])
            self.assertTrue(by_id["pim-ambiguous"]["residual_risk"])
            self.assertIn("search_plan", by_id["pim-rf-zh"]["after"])


if __name__ == "__main__":
    unittest.main()
