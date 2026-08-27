import json
import tempfile
import unittest
from pathlib import Path

from knowledge_storm.evaluation.public_benchmarks.base import (
    BenchmarkCase,
    BenchmarkDataset,
    BenchmarkDocument,
)


class StableBenchmarkSamplingTests(unittest.TestCase):
    def test_stable_fraction_is_order_independent_and_query_only(self):
        from examples.storm_examples.benchmark_embedding_profiles import (
            sample_dataset_cases,
        )

        documents = tuple(
            BenchmarkDocument(str(index), "title", "text {0}".format(index))
            for index in range(100)
        )
        cases = tuple(
            BenchmarkCase(str(index), "query {0}".format(index), (str(index),), "test")
            for index in range(100)
        )
        dataset = BenchmarkDataset("fixture", "v1", documents, cases)
        reversed_dataset = BenchmarkDataset(
            "fixture", "v1", documents, tuple(reversed(cases))
        )

        selected = sample_dataset_cases(dataset, ratio=0.1, seed=55)
        selected_reversed = sample_dataset_cases(reversed_dataset, ratio=0.1, seed=55)

        self.assertEqual(100, len(selected.documents))
        self.assertEqual(10, len(selected.cases))
        self.assertEqual(
            [case.case_id for case in selected.cases],
            [case.case_id for case in selected_reversed.cases],
        )

    def test_positive_ratio_always_keeps_at_least_one_case(self):
        from examples.storm_examples.benchmark_embedding_profiles import (
            sample_dataset_cases,
        )

        dataset = BenchmarkDataset(
            "fixture",
            "v1",
            (BenchmarkDocument("doc", "title", "text"),),
            (BenchmarkCase("case", "query", ("doc",), "test"),),
        )

        self.assertEqual(1, len(sample_dataset_cases(dataset, 0.01, 7).cases))


class BenchmarkReportTests(unittest.TestCase):
    def test_milestone_parser_accepts_a_frozen_embedding_profile(self):
        from examples.storm_examples.run_paperstorm_milestone import build_parser

        args = build_parser().parse_args(
            [
                "--benchmark-root",
                "datasets",
                "--output-dir",
                "runs",
                "--embedding-profile",
                "quality-multilingual",
            ]
        )

        self.assertEqual("quality-multilingual", args.embedding_profile)

    def test_report_contains_quality_resource_and_reproducibility_fields(self):
        from examples.storm_examples.benchmark_embedding_profiles import build_report

        rows = [
            {
                "case_id": "q1",
                "latency_ms": 12.0,
                "ranked_document_ids": ["d1", "d2"],
                "recall_at_2": 1.0,
                "mrr_at_2": 1.0,
                "ndcg_at_2": 1.0,
            },
            {
                "case_id": "q2",
                "latency_ms": 28.0,
                "ranked_document_ids": ["d3", "d4"],
                "recall_at_2": 0.0,
                "mrr_at_2": 0.0,
                "ndcg_at_2": 0.0,
            },
        ]
        manifest = {
            "profile": "cpu-zh",
            "model_name": "example/model",
            "model_revision": "abc",
            "embedding_dimension": 384,
            "query_count": 2,
            "document_count": 10,
            "top_k": 2,
            "build_seconds": 1.25,
            "rss_peak_bytes": 1234,
            "index_bytes": 4567,
        }

        report = build_report(rows, manifest)

        self.assertEqual(0.5, report["metrics"]["recall_at_2"])
        self.assertEqual(20.0, report["metrics"]["query_p50_ms"])
        self.assertEqual(28.0, report["metrics"]["query_p95_ms"])
        self.assertEqual(384, report["resources"]["embedding_dimension"])
        self.assertEqual("abc", report["reproducibility"]["model_revision"])

    def test_completed_profile_checkpoint_is_reused(self):
        from examples.storm_examples.benchmark_embedding_profiles import (
            load_completed_report,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "manifest.json").write_text(
                json.dumps({"status": "complete", "fingerprint": "same"}),
                encoding="utf-8",
            )
            (root / "metrics.json").write_text(
                json.dumps({"metrics": {"recall_at_5": 0.75}}), encoding="utf-8"
            )

            report = load_completed_report(root, expected_fingerprint="same")

        self.assertEqual(0.75, report["metrics"]["recall_at_5"])

    def test_profile_run_writes_auditable_artifacts_and_resumes(self):
        from examples.storm_examples.benchmark_embedding_profiles import (
            run_profile_benchmark,
        )
        from knowledge_storm.retrieval import HashEmbeddingProvider

        dataset = BenchmarkDataset(
            "fixture",
            "v1",
            (
                BenchmarkDocument("d1", "alpha", "alpha evidence"),
                BenchmarkDocument("d2", "beta", "beta evidence"),
            ),
            (BenchmarkCase("q1", "alpha", ("d1",), "test"),),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            first = run_profile_benchmark(
                dataset,
                profile_name="hash-fixture",
                embedding_provider=HashEmbeddingProvider(dim=16),
                output_dir=temp_dir,
                sample_ratio=1.0,
                top_k=1,
            )
            root = Path(first["output_dir"])
            second = run_profile_benchmark(
                dataset,
                profile_name="hash-fixture",
                embedding_provider=HashEmbeddingProvider(dim=16),
                output_dir=temp_dir,
                sample_ratio=1.0,
                top_k=1,
            )

            self.assertEqual("complete", json.loads((root / "manifest.json").read_text())["status"])
            self.assertTrue((root / "predictions.jsonl").is_file())
            self.assertTrue((root / "metrics.json").is_file())
            self.assertTrue((root / "comparison.md").is_file())
            self.assertEqual("resumed", second["status"])
            self.assertEqual(
                "exact",
                first["report"]["manifest"]["dense_backend_mode"],
            )

    def test_paper_scoped_run_indexes_every_legal_candidate_only(self):
        from examples.storm_examples.benchmark_embedding_profiles import (
            run_profile_benchmark,
        )
        from knowledge_storm.retrieval import HashEmbeddingProvider

        dataset = BenchmarkDataset(
            "qasper-fixture",
            "v1",
            (
                BenchmarkDocument(
                    "paper-a::p1", "A", "gold evidence", {"paper_id": "paper-a"}
                ),
                BenchmarkDocument(
                    "paper-a::p2", "A", "hard negative", {"paper_id": "paper-a"}
                ),
                BenchmarkDocument(
                    "paper-b::p1", "B", "unreachable", {"paper_id": "paper-b"}
                ),
            ),
            (
                BenchmarkCase(
                    "q1",
                    "gold",
                    ("paper-a::p1",),
                    "test",
                    metadata={"paper_id": "paper-a"},
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_profile_benchmark(
                dataset,
                profile_name="hash-fixture",
                embedding_provider=HashEmbeddingProvider(dim=16),
                output_dir=temp_dir,
                sample_ratio=1.0,
                top_k=2,
            )

        manifest = result["report"]["manifest"]
        prediction = result["report"]["predictions"][0]
        self.assertEqual(3, manifest["document_count"])
        self.assertEqual(2, manifest["indexed_document_count"])
        self.assertEqual("selected_papers_complete", manifest["index_scope"])
        self.assertEqual(2, prediction["candidate_count"])
        self.assertNotIn("paper-b::p1", prediction["ranked_document_ids"])


if __name__ == "__main__":
    unittest.main()
