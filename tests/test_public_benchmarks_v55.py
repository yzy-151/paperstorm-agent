import json
import io
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class PublicBenchmarkOfflineGuardTest(unittest.TestCase):
    def test_guard_disables_model_flags_and_rejects_socket_connections(self):
        from tests.offline_guard import install_offline_test_guard

        with mock.patch.dict(os.environ, {}, clear=True):
            restore = install_offline_test_guard()
            try:
                self.assertEqual(os.environ["PAPERSTORM_CHAT_LLM"], "0")
                self.assertEqual(os.environ["PAPERSTORM_JUDGE_LLM"], "0")
                self.assertEqual(os.environ["PAPERSTORM_ROUTER_LLM"], "0")
                with self.assertRaisesRegex(
                    RuntimeError, "offline test blocked network"
                ):
                    socket.create_connection(("example.com", 443))
                with socket.socket() as server:
                    server.bind(("127.0.0.1", 0))
                    server.listen(1)
                    with socket.socket() as client:
                        client.connect(server.getsockname())
                        connection, _address = server.accept()
                        connection.close()
                with socket.socket() as client:
                    with self.assertRaisesRegex(
                        RuntimeError, "offline test blocked network"
                    ):
                        client.connect(("203.0.113.1", 9))
            finally:
                restore()

    def test_ci_enables_offline_guard_and_package_discovery(self):
        root = Path(__file__).resolve().parents[1]
        workflow_path = root / ".github" / "workflows" / "test.yml"
        if not workflow_path.exists():
            self.skipTest(
                "offline CI workflow (test.yml) is kept local: pushing it requires "
                "GitHub token with workflow scope"
            )
        workflow = workflow_path.read_text(encoding="utf-8")
        package_init = (root / "tests" / "__init__.py").read_text(encoding="utf-8")

        self.assertIn("PAPERSTORM_TEST_OFFLINE: 1", workflow)
        self.assertIn("discover -s tests -t . -v", workflow)
        self.assertIn("install_offline_test_guard", package_init)


class PublicBenchmarkCoreTest(unittest.TestCase):
    def test_dataset_rejects_duplicate_document_and_case_ids(self):
        from knowledge_storm.evaluation.public_benchmarks.base import (
            BenchmarkCase,
            BenchmarkDataset,
            BenchmarkDocument,
        )

        document = BenchmarkDocument("doc-1", "Title", "Text")
        case = BenchmarkCase("q-1", "query", ("doc-1",), split="test")
        with self.assertRaisesRegex(ValueError, "duplicate document_id"):
            BenchmarkDataset("demo", "1", (document, document), (case,))
        with self.assertRaisesRegex(ValueError, "duplicate case_id"):
            BenchmarkDataset("demo", "1", (document,), (case, case))

    def test_retrieval_metrics_match_hand_calculation(self):
        from knowledge_storm.evaluation.public_benchmarks.metrics import (
            retrieval_metrics,
        )

        metrics = retrieval_metrics(
            ranked_ids=["noise", "doc-a", "doc-b"],
            relevance={"doc-a": 1, "doc-b": 2},
            cutoffs=(1, 3),
        )

        self.assertEqual(metrics["recall_at_1"], 0.0)
        self.assertEqual(metrics["recall_at_3"], 1.0)
        self.assertEqual(metrics["mrr_at_3"], 0.5)
        self.assertEqual(metrics["ndcg_at_3"], 0.586883)

    def test_answer_evidence_and_abstention_metrics_are_separate(self):
        from knowledge_storm.evaluation.public_benchmarks.metrics import (
            answer_metrics,
            evidence_metrics,
        )

        answer = answer_metrics("The neural model", ["neural model"])
        evidence = evidence_metrics({"p1", "p2"}, {"p2", "p3"})

        self.assertEqual(answer["exact_match"], 0.0)
        self.assertGreater(answer["token_f1"], 0.7)
        self.assertEqual(evidence["precision"], 0.5)
        self.assertEqual(evidence["recall"], 0.5)
        self.assertEqual(evidence["f1"], 0.5)


class SciFactAdapterTest(unittest.TestCase):
    def test_loads_official_beir_layout_and_positive_qrels(self):
        from knowledge_storm.evaluation.public_benchmarks.beir_scifact import (
            load_scifact,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "qrels").mkdir()
            (root / "corpus.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"_id": "d1", "title": "Alpha", "text": "first"}),
                        json.dumps({"_id": "d2", "title": "Beta", "text": "second"}),
                    ]
                ),
                encoding="utf-8",
            )
            (root / "queries.jsonl").write_text(
                json.dumps({"_id": "q1", "text": "Which claim?"}), encoding="utf-8"
            )
            (root / "qrels" / "test.tsv").write_text(
                "query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td2\t0\n",
                encoding="utf-8",
            )

            dataset = load_scifact(root, split="test")

        self.assertEqual(dataset.name, "beir-scifact")
        self.assertEqual(len(dataset.documents), 2)
        self.assertEqual(dataset.cases[0].relevance, {"d1": 1})
        self.assertIn("Alpha", dataset.documents[0].text)


class QasperAdapterTest(unittest.TestCase):
    def test_expands_questions_answers_and_evidence_from_hf_record(self):
        from knowledge_storm.evaluation.public_benchmarks.qasper import (
            load_qasper_records,
        )

        records = [
            {
                "id": "paper-1",
                "title": "Paper title",
                "abstract": "Abstract text",
                "full_text": {
                    "section_name": ["Method", "Results"],
                    "paragraphs": [["method paragraph"], ["result paragraph"]],
                },
                "qas": {
                    "question": ["What was used?", "Was it evaluated?"],
                    "question_id": ["q1", "q2"],
                    "answers": [
                        {
                            "answer": [
                                {
                                    "unanswerable": False,
                                    "extractive_spans": ["method"],
                                    "free_form_answer": "",
                                    "yes_no": None,
                                    "evidence": ["method paragraph"],
                                }
                            ]
                        },
                        {
                            "answer": [
                                {
                                    "unanswerable": False,
                                    "extractive_spans": [],
                                    "free_form_answer": "",
                                    "yes_no": True,
                                    "evidence": ["result paragraph"],
                                }
                            ]
                        },
                    ],
                },
            }
        ]

        dataset = load_qasper_records(records, split="validation")

        self.assertEqual(len(dataset.documents), 2)
        self.assertEqual(len(dataset.cases), 2)
        self.assertEqual(dataset.cases[0].answers, ("method",))
        self.assertEqual(dataset.cases[1].answers, ("yes",))
        self.assertEqual(
            dataset.cases[0].evidence_ids, ("paper-1::section-0::paragraph-0",)
        )
        self.assertEqual(
            dataset.cases[0].relevance, {"paper-1::section-0::paragraph-0": 1}
        )


class PublicBenchmarkRunnerTest(unittest.TestCase):
    def test_bm25_mode_does_not_compute_unused_query_embedding(self):
        from knowledge_storm.retrieval import HybridPaperIndex

        class CountingEmbedding:
            name = "counting"
            dim = 2
            normalize = True

            def __init__(self):
                self.query_calls = 0

            def embed(self, texts):
                return [[1.0, 0.0] for _ in texts]

            def embed_query(self, _text):
                self.query_calls += 1
                return [1.0, 0.0]

        provider = CountingEmbedding()
        index = HybridPaperIndex(
            [{"chunk_id": "d1", "document_id": "d1", "content": "alpha"}],
            embedding_provider=provider,
        )

        index.search("alpha", mode="bm25", top_k=1)

        self.assertEqual(provider.query_calls, 0)

    def test_dense_search_uses_vectorized_cosine_without_python_pair_loop(self):
        from knowledge_storm.retrieval import HybridPaperIndex

        class Embedding:
            name = "fixture"
            dim = 2
            normalize = False

            def embed(self, _texts):
                return [[2.0, 0.0], [0.0, 3.0]]

            def embed_query(self, _text):
                return [4.0, 0.0]

        index = HybridPaperIndex(
            [
                {"chunk_id": "d1", "document_id": "d1", "content": "alpha"},
                {"chunk_id": "d2", "document_id": "d2", "content": "beta"},
            ],
            embedding_provider=Embedding(),
        )
        with mock.patch(
            "knowledge_storm.retrieval._cosine",
            side_effect=AssertionError("pairwise cosine must not be used"),
        ):
            results = index.search("alpha", mode="dense", top_k=2)

        self.assertEqual([item["document_id"] for item in results], ["d1", "d2"])
        self.assertEqual(results[0]["dense_score"], 1.0)

    def test_runner_compares_modes_and_writes_auditable_artifacts(self):
        from knowledge_storm.evaluation.public_benchmarks.base import (
            BenchmarkCase,
            BenchmarkDataset,
            BenchmarkDocument,
        )
        from knowledge_storm.evaluation.public_benchmarks.runner import (
            run_retrieval_benchmark,
        )

        class KeywordEmbedding:
            name = "test-keyword"
            dim = 3
            normalize = True

            def embed(self, texts):
                return [self.embed_query(text) for text in texts]

            def embed_query(self, text):
                lowered = str(text).lower()
                return [float(term in lowered) for term in ("alpha", "beta", "gamma")]

        dataset = BenchmarkDataset(
            "fixture",
            "1",
            (
                BenchmarkDocument("d1", "Alpha", "alpha evidence"),
                BenchmarkDocument("d2", "Beta", "beta evidence"),
                BenchmarkDocument("d3", "Gamma", "gamma evidence"),
            ),
            (
                BenchmarkCase("q1", "alpha", ("d1",), "test"),
                BenchmarkCase("q2", "beta", ("d2",), "test"),
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            report = run_retrieval_benchmark(
                dataset,
                embedding_provider=KeywordEmbedding(),
                modes=("bm25", "dense", "hybrid"),
                top_k=2,
                bootstrap_samples=50,
                output_dir=temp_dir,
            )
            names = {path.name for path in Path(temp_dir).iterdir()}

        self.assertEqual(report["case_count"], 2)
        self.assertEqual(set(report["modes"]), {"bm25", "dense", "hybrid"})
        self.assertEqual(report["modes"]["dense"]["recall_at_2"], 1.0)
        self.assertIn("confidence_intervals", report["modes"]["dense"])
        self.assertEqual(
            report["manifest"]["cache_state"], "warm_query_after_cold_index"
        )
        self.assertIn("working_tree_dirty", report["manifest"])
        self.assertTrue(
            {
                "manifest.json",
                "predictions.jsonl",
                "metrics.json",
                "bad_cases.jsonl",
                "report.md",
            }
            <= names
        )

    def test_qasper_prediction_metrics_keep_answer_evidence_and_abstention_separate(
        self,
    ):
        from knowledge_storm.evaluation.public_benchmarks.base import BenchmarkCase
        from knowledge_storm.evaluation.public_benchmarks.qasper import (
            evaluate_qasper_predictions,
        )

        cases = (
            BenchmarkCase(
                "q1",
                "question",
                ("p1",),
                "validation",
                {"p1": 1},
                answers=("neural model",),
                evidence_ids=("p1",),
            ),
            BenchmarkCase(
                "q2",
                "unknown",
                (),
                "validation",
                {},
                answers=(),
                evidence_ids=(),
                unanswerable=True,
            ),
        )
        report = evaluate_qasper_predictions(
            cases,
            {
                "q1": {"answer": "the neural model", "evidence_ids": ["p1"]},
                "q2": {"answer": "", "evidence_ids": [], "abstained": True},
            },
        )

        self.assertEqual(report["case_count"], 2)
        self.assertGreater(report["answer_token_f1"], 0.4)
        self.assertEqual(report["evidence_f1"], 1.0)
        self.assertEqual(report["abstention_recall"], 1.0)

    def test_scoped_runner_only_searches_paragraphs_from_question_paper(self):
        from knowledge_storm.evaluation.public_benchmarks.base import (
            BenchmarkCase,
            BenchmarkDataset,
            BenchmarkDocument,
        )
        from knowledge_storm.evaluation.public_benchmarks.runner import (
            HashEmbeddingProvider,
            run_retrieval_benchmark,
        )

        dataset = BenchmarkDataset(
            "qasper-fixture",
            "1",
            (
                BenchmarkDocument("a-1", "A", "method", {"paper_id": "a"}),
                BenchmarkDocument("a-2", "A", "result", {"paper_id": "a"}),
                BenchmarkDocument("b-1", "B", "method", {"paper_id": "b"}),
            ),
            (
                BenchmarkCase(
                    "q1",
                    "method",
                    ("a-1",),
                    "validation",
                    {"a-1": 1},
                    metadata={"paper_id": "a"},
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            run_retrieval_benchmark(
                dataset,
                HashEmbeddingProvider(),
                modes=("bm25",),
                top_k=2,
                bootstrap_samples=10,
                output_dir=temp_dir,
                scope_field="paper_id",
            )
            prediction = json.loads(
                (Path(temp_dir) / "predictions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )

        self.assertEqual(set(prediction["ranked_document_ids"]), {"a-1", "a-2"})


class PublicBenchmarkCliTest(unittest.TestCase):
    def test_benchmark_dependencies_are_declared_by_runtime_scope(self):
        root = Path(__file__).resolve().parents[1]
        requirements = (root / "requirements.txt").read_text(encoding="utf-8")
        setup_text = (root / "setup.py").read_text(encoding="utf-8")

        self.assertRegex(requirements, r"(?m)^certifi(?:[<>=].*)?$")
        self.assertIn('"benchmarks": ["datasets', setup_text)

    def test_cli_parser_supports_scifact_qasper_and_reproducibility_options(self):
        from examples.storm_examples.run_paperstorm_public_benchmark import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "--benchmark",
                "scifact",
                "--dataset-dir",
                "datasets/scifact",
                "--output-dir",
                "results/public",
                "--split",
                "test",
                "--modes",
                "bm25",
                "dense",
                "hybrid",
                "--embedding",
                "real",
                "--model",
                "sentence-transformers/all-MiniLM-L6-v2",
                "--top-k",
                "10",
                "--seed",
                "55",
                "--smoke-limit",
                "20",
            ]
        )

        self.assertEqual(args.benchmark, "scifact")
        self.assertEqual(args.modes, ["bm25", "dense", "hybrid"])
        self.assertEqual(args.smoke_limit, 20)

    def test_scifact_download_rejects_wrong_checksum(self):
        from knowledge_storm.evaluation.public_benchmarks.beir_scifact import (
            verify_md5,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            archive = Path(temp_dir) / "scifact.zip"
            archive.write_bytes(b"not the official archive")
            with self.assertRaisesRegex(ValueError, "MD5 mismatch"):
                verify_md5(archive, expected="00000000000000000000000000000000")

    def test_scifact_download_uses_verified_certifi_context(self):
        from knowledge_storm.evaluation.public_benchmarks.beir_scifact import (
            _download_file,
        )

        captured = {}

        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

        def fake_urlopen(request, context=None, timeout=None):
            captured["context"] = context
            captured["timeout"] = timeout
            captured["user_agent"] = request.headers.get("User-agent")
            return Response(b"official bytes")

        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "archive.zip"
            with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                _download_file("https://example.com/archive.zip", target)
            self.assertEqual(target.read_bytes(), b"official bytes")

        self.assertIsNotNone(captured["context"])
        self.assertGreater(captured["timeout"], 0)
        self.assertIn("PaperStorm", captured["user_agent"])

    def test_qasper_smoke_subset_keeps_all_paragraphs_from_selected_paper(self):
        from examples.storm_examples.run_paperstorm_public_benchmark import (
            _evaluation_subset,
        )
        from knowledge_storm.evaluation.public_benchmarks.qasper import (
            load_qasper_records,
        )

        dataset = load_qasper_records(
            [
                {
                    "id": "paper-1",
                    "title": "Title",
                    "full_text": {
                        "section_name": ["Section"],
                        "paragraphs": [["gold evidence", "hard negative"]],
                    },
                    "qas": {
                        "question": ["Question"],
                        "question_id": ["q1"],
                        "answers": [
                            {
                                "answer": [
                                    {
                                        "unanswerable": False,
                                        "extractive_spans": ["gold"],
                                        "free_form_answer": "",
                                        "yes_no": None,
                                        "evidence": ["gold evidence"],
                                    }
                                ]
                            }
                        ],
                    },
                }
            ]
        )

        subset = _evaluation_subset(dataset, smoke_limit=1, benchmark="qasper")

        self.assertEqual(len(subset.documents), 2)


if __name__ == "__main__":
    unittest.main()
