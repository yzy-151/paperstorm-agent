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
        self.assertEqual("development", args.evaluation_phase)
        self.assertEqual("sentence-transformers/all-MiniLM-L6-v2", args.model)
        self.assertIsNone(args.top_k)

    def test_benchmark_protocol_uses_development_splits_and_baseline_top_k(self):
        from examples.storm_examples.run_paperstorm_milestone import _protocol_for

        self.assertEqual(("dev", 10), _protocol_for("scifact", "development", None))
        self.assertEqual(("validation", 5), _protocol_for("qasper-retrieval", "development", None))
        self.assertEqual(("test", 10), _protocol_for("scifact", "final", None))

    def test_incompatible_protocol_is_explicit_and_has_no_delta(self):
        from examples.storm_examples.run_paperstorm_milestone import _comparison_metadata

        metadata = _comparison_metadata(
            benchmark="scifact",
            split="dev",
            top_k=10,
            model="sentence-transformers/all-MiniLM-L6-v2",
            embedding_kind="real",
            smoke_limit=None,
            actual_manifest={"corpus_sha256": "wrong", "case_count": 300, "document_count": 5183},
        )
        self.assertEqual("incomparable", metadata["status"])
        self.assertFalse(metadata["paired_comparison_allowed"])
        self.assertNotIn("delta", metadata)
        self.assertTrue(metadata["baseline_sha256"])

    def test_matching_legacy_protocol_stays_incomparable_without_query_gold_hash(self):
        from examples.storm_examples.run_paperstorm_milestone import _comparison_metadata

        metadata = _comparison_metadata(
            benchmark="scifact", split="test", top_k=10,
            model="sentence-transformers/all-MiniLM-L6-v2",
            embedding_kind="real", smoke_limit=None,
            actual_manifest={
                "corpus_sha256": "54e2468b7b03e164cd2a0d87bafe248e00e991cde4f5eab0d5122f540f6731a9",
                "case_count": 300,
                "document_count": 5183,
            },
        )
        self.assertEqual("incomparable", metadata["status"])
        self.assertFalse(metadata["aggregate_comparison_allowed"])
        self.assertIn("baseline_query_gold_fingerprint_missing", metadata["reasons"])

    def test_p2_defaults_to_only_affected_benchmarks(self):
        from examples.storm_examples.run_paperstorm_milestone import build_parser, resolve_benchmarks

        args = build_parser().parse_args([
            "--milestone", "P1+P2",
            "--output-dir", "out",
            "--benchmark-root", "data",
            "--baseline-dir", "p1",
        ])

        self.assertEqual(
            ("scifact", "qasper-retrieval", "evidence-governance"),
            resolve_benchmarks(args),
        )
        self.assertEqual("cross-encoder/ms-marco-MiniLM-L-6-v2", args.reranker_model)

    def test_p2_requires_previous_milestone_directory(self):
        from examples.storm_examples.run_paperstorm_milestone import main

        with tempfile.TemporaryDirectory() as temp_dir, redirect_stdout(io.StringIO()):
            summary = main([
                "--milestone", "P1+P2",
                "--benchmark", "scifact",
                "--benchmark-root", temp_dir,
                "--output-dir", str(Path(temp_dir) / "out"),
                "--embedding", "hash",
            ])

        self.assertEqual("baseline_missing", summary["benchmarks"]["scifact"]["reason_code"])

    def test_p2_comparison_uses_matching_p1_predictions(self):
        from examples.storm_examples.run_paperstorm_milestone import _p2_comparison

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "scifact"
            run_dir.mkdir()
            manifest = {
                "benchmark": "fixture", "dataset_version": "1", "split": "test",
                "case_count": 1, "document_count": 2, "corpus_sha256": "corpus",
                "query_gold_sha256": "qrels", "embedding_model": "embed",
                "top_k": 5, "seed": 55,
            }
            (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (run_dir / "metrics.json").write_text(json.dumps({
                "modes": {"hybrid": {"recall_at_5": 0.0, "mrr_at_5": 0.0, "ndcg_at_5": 0.0, "p95_latency_ms": 10.0}}
            }), encoding="utf-8")
            (run_dir / "predictions.jsonl").write_text(json.dumps({
                "case_id": "c1", "metrics": {"recall_at_5": 0.0, "mrr_at_5": 0.0, "ndcg_at_5": 0.0}
            }) + "\n", encoding="utf-8")
            candidate = {
                "manifest": dict(manifest),
                "modes": {"hybrid_governed": {"recall_at_5": 1.0, "mrr_at_5": 1.0, "ndcg_at_5": 1.0, "p95_latency_ms": 20.0}},
                "predictions": [{"case_id": "c1", "metrics": {"recall_at_5": 1.0, "mrr_at_5": 1.0, "ndcg_at_5": 1.0}}],
            }

            comparison = _p2_comparison(Path(temp_dir), "scifact", candidate, 5)

        self.assertEqual("comparable", comparison["status"])
        self.assertEqual(1.0, comparison["delta"]["recall_at_5"])
        self.assertEqual(10.0, comparison["delta"]["p95_latency_ms"])
        self.assertEqual(1, comparison["paired_case_count"])

    def test_evidence_governance_benchmark_writes_fixed_case_dossiers(self):
        from examples.storm_examples.run_paperstorm_milestone import _run_evidence_governance

        with tempfile.TemporaryDirectory() as temp_dir:
            result = _run_evidence_governance(Path(temp_dir))
            run_dir = Path(result["output_dir"])
            dossiers = [
                json.loads(line)
                for line in (run_dir / "case_dossiers.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

        self.assertEqual("completed", result["status"])
        self.assertEqual({"coverage", "conflict", "no-answer"}, {item["case_id"] for item in dossiers})
        self.assertEqual(1.0, metrics["pass_rate"])

    def test_p2_qasper_smoke_runs_governed_pipeline_against_p1(self):
        from examples.storm_examples import run_paperstorm_milestone as module
        from knowledge_storm.retrieval import CrossEncoderReranker

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            qasper = root / "qasper"
            qasper.mkdir()
            payload = {
                "paper-1": {
                    "title": "Optimization study",
                    "full_text": [{"section_name": "Method", "paragraphs": [
                        "We minimize a contrastive loss objective.",
                        "Training uses hard negative examples.",
                    ]}],
                    "qas": [{
                        "question": "Which training criterion is optimized?",
                        "question_id": "q-1",
                        "answers": [{"answer": {
                            "extractive_spans": ["contrastive loss"],
                            "free_form_answer": "", "yes_no": None,
                            "unanswerable": False,
                            "evidence": ["We minimize a contrastive loss objective."],
                        }}],
                    }],
                }
            }
            (qasper / "qasper-test-v0.3.json").write_text(json.dumps(payload), encoding="utf-8")
            common = [
                "--benchmark", "qasper-retrieval", "--benchmark-root", str(root),
                "--embedding", "hash", "--evaluation-phase", "final", "--smoke-limit", "1",
            ]
            with redirect_stdout(io.StringIO()):
                p1 = module.main(common + ["--output-dir", str(root / "p1")])
            fake = CrossEncoderReranker(score_fn=lambda pairs: [1.0 - index for index, _ in enumerate(pairs)])
            with mock.patch.object(module, "_reranker", return_value=fake), redirect_stdout(io.StringIO()):
                p2 = module.main(common + [
                    "--milestone", "P1+P2", "--baseline-dir", str(root / "p1"),
                    "--output-dir", str(root / "p2"),
                ])

            report = json.loads((root / "p2" / "qasper-retrieval" / "metrics.json").read_text(encoding="utf-8"))
            dossier = json.loads((root / "p2" / "qasper-retrieval" / "case_dossiers.jsonl").read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual("completed", p1["status"])
        self.assertEqual("completed", p2["status"])
        self.assertIn("hybrid_governed", report["modes"])
        self.assertEqual("comparable", report["manifest"]["comparison"]["status"])
        self.assertEqual("P1+P2", dossier["milestone"])
        self.assertTrue(dossier["before"]["case_level_before_available"])

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
                    "--evaluation-phase", "final",
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
            self.assertTrue(all(row["before"].get("baseline_sha256") for row in dossiers))
            by_id = {row["case_id"]: row for row in dossiers}
            self.assertTrue(by_id["pim-rf-zh"]["after"]["resolved"])
            self.assertTrue(by_id["pim-lexical"]["after"]["resolved"])
            self.assertFalse(by_id["pim-ambiguous"]["after"]["resolved"])
            self.assertTrue(by_id["pim-ambiguous"]["residual_risk"])
            self.assertIn("search_plan", by_id["pim-rf-zh"]["after"])
            for case_id in ("pim-rf-zh", "pim-lexical"):
                after = by_id[case_id]["after"]
                self.assertEqual(after["relevant_document_ids"][0] in after["top_1"], True)
                self.assertEqual([], after["forbidden_hits_at_k"])
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
                    "--evaluation-phase", "final",
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
        self.assertFalse(dossier["before"]["paired_comparison_allowed"])
        self.assertIn("baseline_sha256", dossier["before"])
        self.assertNotIn("query expansion", dossier["change"].lower())

    def test_failed_run_writes_lifecycle_and_cli_returns_nonzero(self):
        from examples.storm_examples import run_paperstorm_milestone as module

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            module, "_run_one", side_effect=RuntimeError("boom")
        ), redirect_stdout(io.StringIO()):
            output = Path(temp_dir) / "out"
            summary = module.main([
                "--benchmark", "pim",
                "--benchmark-root", temp_dir,
                "--output-dir", str(output),
                "--embedding", "hash",
            ])
            lifecycle = json.loads((output / "run_status.json").read_text(encoding="utf-8"))
        self.assertEqual("failed", summary["status"])
        self.assertEqual("failed", lifecycle["status"])
        self.assertEqual(1, module.exit_code(summary))

    def test_blocked_run_returns_distinct_nonzero_exit_code(self):
        from examples.storm_examples.run_paperstorm_milestone import exit_code

        self.assertEqual(2, exit_code({"status": "completed_with_blocks"}))

    def test_qasper_retrieval_excludes_cases_without_gold_evidence(self):
        from examples.storm_examples.run_paperstorm_milestone import _retrieval_cases_only
        from knowledge_storm.evaluation.public_benchmarks.base import BenchmarkCase, BenchmarkDataset, BenchmarkDocument

        dataset = BenchmarkDataset(
            "qasper", "fixture",
            (BenchmarkDocument("p", "T", "text"),),
            (
                BenchmarkCase("answerable", "q1", ("p",), "validation", evidence_ids=("p",)),
                BenchmarkCase("no-evidence", "q2", (), "validation", unanswerable=True),
            ),
        )
        filtered = _retrieval_cases_only(dataset)
        self.assertEqual(["answerable"], [case.case_id for case in filtered.cases])

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

    def test_change_does_not_claim_filtering_when_plan_has_no_filters(self):
        from examples.storm_examples.run_paperstorm_milestone import _actual_change

        change = _actual_change({
            "search_plan": {"subqueries": ["query"], "must_terms": [], "negative_terms": [], "filters": {}},
            "retrieval_stages": [{"name": "gate", "status": "completed"}],
        })
        self.assertNotIn("filter", change.lower())
        self.assertIn("final selection", change.lower())


if __name__ == "__main__":
    unittest.main()
