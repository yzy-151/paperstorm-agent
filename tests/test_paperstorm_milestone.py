import json
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock


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
            self.assertEqual("dataset_missing", summary["benchmarks"]["scifact"]["reason_code"])
            self.assertEqual("completed", summary["benchmarks"]["pim"]["status"])

    def test_permission_error_is_blocked_and_next_benchmark_continues(self):
        from examples.storm_examples import run_paperstorm_milestone as module

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            module, "_run_one", side_effect=[PermissionError("denied"), {"status": "completed", "benchmark": "pim"}]
        ), redirect_stdout(io.StringIO()):
            summary = module.main([
                "--benchmark", "scifact", "pim",
                "--benchmark-root", temp_dir,
                "--output-dir", str(Path(temp_dir) / "out"),
                "--embedding", "hash",
            ])
        self.assertEqual("blocked", summary["benchmarks"]["scifact"]["status"])
        self.assertEqual("dataset_permission_denied", summary["benchmarks"]["scifact"]["reason_code"])
        self.assertEqual("completed", summary["benchmarks"]["pim"]["status"])

    def test_missing_real_model_has_stable_reason_code(self):
        from examples.storm_examples.run_paperstorm_milestone import main

        with tempfile.TemporaryDirectory() as temp_dir, redirect_stdout(io.StringIO()):
            summary = main([
                "--benchmark", "pim",
                "--benchmark-root", temp_dir,
                "--output-dir", str(Path(temp_dir) / "out"),
                "--embedding", "real",
            ])
        self.assertEqual("model_missing", summary["benchmarks"]["pim"]["reason_code"])

    def test_unexpected_execution_error_is_failed_with_stable_reason_code(self):
        from examples.storm_examples import run_paperstorm_milestone as module

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            module, "_run_one", side_effect=ValueError("api_key=must-not-leak")
        ), redirect_stdout(io.StringIO()):
            summary = module.main([
                "--benchmark", "pim",
                "--benchmark-root", temp_dir,
                "--output-dir", str(Path(temp_dir) / "out"),
                "--embedding", "hash",
            ])
        result = summary["benchmarks"]["pim"]
        self.assertEqual("failed", result["status"])
        self.assertEqual("benchmark_execution_failed", result["reason_code"])
        self.assertNotIn("must-not-leak", json.dumps(result))

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
            self.assertFalse(by_id["pim-rf-zh"]["after"]["resolved"])
            self.assertFalse(by_id["pim-lexical"]["after"]["resolved"])
            self.assertFalse(by_id["pim-ambiguous"]["after"]["resolved"])
            self.assertTrue(by_id["pim-ambiguous"]["residual_risk"])
            self.assertIn("search_plan", by_id["pim-rf-zh"]["after"])
            for case_id in ("pim-rf-zh", "pim-lexical"):
                after = by_id[case_id]["after"]
                self.assertEqual(after["relevant_document_ids"][0] in after["top_1"], True)
                self.assertEqual(["product-pim-1"], after["forbidden_hits_at_k"])
                self.assertIn("acceptance", after)

    def test_qasper_run_writes_auditable_non_fabricated_dossier(self):
        from examples.storm_examples.run_paperstorm_milestone import main

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            qasper = root / "qasper"
            qasper.mkdir()
            payload = {
                "paper-1": {
                    "title": "Optimization study",
                    "full_text": [{"section_name": "Method", "paragraphs": ["We minimize a contrastive loss objective."]}],
                    "qas": [{
                        "question": "Which training criterion is optimized?",
                        "question_id": "q-1",
                        "answers": [{"answer": {"extractive_spans": ["contrastive loss"], "free_form_answer": "", "yes_no": None, "unanswerable": False, "evidence": ["We minimize a contrastive loss objective."]}}],
                    }],
                }
            }
            (qasper / "qasper-test-v0.3.json").write_text(json.dumps(payload), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                summary = main([
                    "--benchmark", "qasper-retrieval",
                    "--benchmark-root", str(root),
                    "--output-dir", str(root / "out"),
                    "--embedding", "hash",
                    "--smoke-limit", "1",
                ])
            run_dir = Path(summary["benchmarks"]["qasper-retrieval"]["output_dir"])
            dossier = json.loads((run_dir / "case_dossiers.jsonl").read_text(encoding="utf-8").splitlines()[0])
        self.assertFalse(dossier["before"]["case_level_before_available"])
        self.assertEqual("archived_aggregate", dossier["before"]["source"])
        self.assertIn("aggregate", dossier["before"])
        self.assertEqual(0.505659, dossier["before"]["aggregate"]["recall"])
        self.assertTrue(dossier["after"]["ranked_document_ids"])
        self.assertTrue(dossier["root_cause"])
        self.assertTrue(dossier["change"])
        self.assertIn("resolved", dossier["after"])
        self.assertIn("lexical_overlap", dossier["after"])
        self.assertIn("search_plan", dossier["after"])

    def test_pim_top_k_hit_with_wrong_top1_is_not_resolved(self):
        from examples.storm_examples.run_paperstorm_milestone import _pim_dossiers

        fixture = Path(__file__).parent / "fixtures" / "pim_retrieval_badcases.json"
        report = {
            "predictions": [{
                "case_id": "pim-rf-zh",
                "ranked_document_ids": ["dram-pim-1", "rf-pim-1"],
                "search_plan": {"standalone_query": "PIM passive intermodulation"},
            }]
        }
        dossier = _pim_dossiers(fixture, report)[0].to_dict()
        self.assertFalse(dossier["after"]["resolved"])
        self.assertEqual(["dram-pim-1"], dossier["after"]["forbidden_hits_at_k"])
        self.assertEqual(["dram-pim-1"], dossier["after"]["top_1"])


if __name__ == "__main__":
    unittest.main()
