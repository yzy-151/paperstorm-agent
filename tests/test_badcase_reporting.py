import hashlib
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path


class CaseDossierTest(unittest.TestCase):
    def test_complete_dossier_is_frozen_and_json_safe(self):
        from knowledge_storm.badcase_reporting import CaseDossier

        dossier = CaseDossier(
            case_id="case-7",
            milestone="P1+P2",
            question="What changed?",
            before={"rank": 9, "evidence": ("d1",)},
            root_cause="Query terms were too broad.",
            change={"retrieval": "query expansion"},
            after={"rank": 2},
            residual_risk="Acronyms remain ambiguous.",
        )

        self.assertEqual(
            dossier.to_dict(),
            {
                "case_id": "case-7",
                "milestone": "P1+P2",
                "question": "What changed?",
                "before": {"rank": 9, "evidence": ["d1"]},
                "root_cause": "Query terms were too broad.",
                "change": {"retrieval": "query expansion"},
                "after": {"rank": 2},
                "residual_risk": "Acronyms remain ambiguous.",
            },
        )
        with self.assertRaises(FrozenInstanceError):
            dossier.question = "mutated"

    def test_dossier_rejects_unknown_milestone(self):
        from knowledge_storm.badcase_reporting import CaseDossier

        dossier = CaseDossier("case-1", "P2", "q", {}, "cause", "change", {})

        with self.assertRaisesRegex(ValueError, "milestone"):
            dossier.to_dict()

    def test_jsonl_writer_round_trips_and_preserves_target_on_json_error(self):
        from knowledge_storm.badcase_reporting import (
            CaseDossier,
            write_case_dossiers_jsonl,
        )

        dossiers = [
            CaseDossier("case-1", "P1", "q1", "old", "cause", "fix", "new"),
            CaseDossier(
                "case-2",
                "P1+P2+P3+P4",
                "q2",
                {"score": 0},
                "cause",
                "fix",
                {"score": 1},
            ),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dossiers.jsonl"
            write_case_dossiers_jsonl(path, dossiers)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

            self.assertEqual([row["case_id"] for row in rows], ["case-1", "case-2"])
            original = path.read_bytes()
            bad = CaseDossier("bad", "P1", "q", object(), "cause", "fix", "new")
            with self.assertRaises(TypeError):
                write_case_dossiers_jsonl(path, [bad])
            self.assertEqual(path.read_bytes(), original)


class MilestoneManifestTest(unittest.TestCase):
    def test_builder_records_required_fields_and_removes_credentials(self):
        from knowledge_storm.badcase_reporting import build_milestone_manifest

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "dataset.jsonl"
            dataset_path.write_bytes(b'{"id": 1}\n')
            manifest = build_milestone_manifest(
                milestone="P1",
                git_sha="abc123",
                dataset_path=dataset_path,
                split="test",
                models={"embedding": "fixture", "api_key": "model-secret-value"},
                top_k=10,
                seed=55,
                command=["python", "run.py", "--token", "command-secret-value"],
                started_at="2026-08-25T10:00:00+00:00",
                finished_at="2026-08-25T10:01:00+00:00",
                api_usage={
                    "requests": 2,
                    "prompt_tokens": 20,
                    "access_token": "usage-secret-value",
                },
                host_profile={"os": "test", "secret": "host-secret-value"},
            )

        self.assertEqual(manifest["milestone"], "P1")
        self.assertEqual(manifest["git_sha"], "abc123")
        self.assertEqual(manifest["dataset"]["path"], str(dataset_path))
        self.assertEqual(
            manifest["dataset"]["digest"],
            hashlib.sha256(b'{"id": 1}\n').hexdigest(),
        )
        for field in (
            "split",
            "models",
            "top_k",
            "seed",
            "command",
            "started_at",
            "finished_at",
            "api_usage",
            "host_profile",
        ):
            self.assertIn(field, manifest)
        self.assertEqual(manifest["api_usage"]["prompt_tokens"], 20)
        serialized = json.dumps(manifest, sort_keys=True)
        for leaked in (
            "api_key",
            "access_token",
            "model-secret-value",
            "command-secret-value",
            "usage-secret-value",
            "host-secret-value",
        ):
            self.assertNotIn(leaked, serialized)

    def test_manifest_writer_round_trips_json(self):
        from knowledge_storm.badcase_reporting import (
            build_milestone_manifest,
            write_milestone_manifest,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "dataset.jsonl"
            dataset_path.write_text("{}\n", encoding="utf-8")
            output_path = Path(temp_dir) / "manifest.json"
            manifest = build_milestone_manifest(
                milestone="P1+P2+P3",
                git_sha="abc123",
                dataset_path=dataset_path,
                split="validation",
                models=["reader", "judge"],
                top_k=5,
                seed=9,
                command="python benchmark.py",
                started_at="start",
                finished_at="finish",
                api_usage={"requests": 0},
                host_profile={"platform": "fixture"},
            )
            write_milestone_manifest(output_path, manifest)

            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), manifest)


class PairedBootstrapTest(unittest.TestCase):
    def test_confidence_interval_is_repeatable(self):
        from knowledge_storm.badcase_reporting import paired_bootstrap_ci

        baseline = [0.0, 0.25, 0.5, 0.75, 1.0]
        candidate = [0.25, 0.5, 0.5, 1.0, 1.0]

        first = paired_bootstrap_ci(baseline, candidate, samples=500, seed=71)
        second = paired_bootstrap_ci(baseline, candidate, samples=500, seed=71)

        self.assertEqual(first, second)
        self.assertEqual(first["delta"], 0.15)
        self.assertEqual(first["sample_count"], 5)
        self.assertLessEqual(first["low"], first["delta"])
        self.assertGreaterEqual(first["high"], first["delta"])

    def test_confidence_interval_rejects_empty_or_unpaired_input(self):
        from knowledge_storm.badcase_reporting import paired_bootstrap_ci

        with self.assertRaisesRegex(ValueError, "empty"):
            paired_bootstrap_ci([], [])
        with self.assertRaisesRegex(ValueError, "same length"):
            paired_bootstrap_ci([0.1], [0.1, 0.2])


class PublicBenchmarkMilestoneTest(unittest.TestCase):
    @staticmethod
    def _dataset():
        from knowledge_storm.evaluation.public_benchmarks.base import (
            BenchmarkCase,
            BenchmarkDataset,
            BenchmarkDocument,
        )

        return BenchmarkDataset(
            "fixture",
            "1",
            (BenchmarkDocument("doc-1", "Alpha", "alpha evidence"),),
            (BenchmarkCase("case-1", "alpha", ("doc-1",), "test"),),
        )

    def test_runner_only_emits_explicit_milestone_metadata(self):
        from knowledge_storm.evaluation.public_benchmarks.runner import (
            HashEmbeddingProvider,
            run_retrieval_benchmark,
        )

        with_milestone = run_retrieval_benchmark(
            self._dataset(),
            HashEmbeddingProvider(),
            modes=("bm25",),
            top_k=1,
            bootstrap_samples=5,
            milestone="P1",
        )
        without_milestone = run_retrieval_benchmark(
            self._dataset(),
            HashEmbeddingProvider(),
            modes=("bm25",),
            top_k=1,
            bootstrap_samples=5,
        )

        self.assertEqual(with_milestone["milestone"], "P1")
        self.assertEqual(with_milestone["manifest"]["milestone"], "P1")
        self.assertNotIn("milestone", without_milestone)
        self.assertNotIn("milestone", without_milestone["manifest"])


if __name__ == "__main__":
    unittest.main()
