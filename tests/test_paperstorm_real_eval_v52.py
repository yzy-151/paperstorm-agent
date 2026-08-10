import tempfile
import unittest
from pathlib import Path


class _KeywordEmbedding:
    name = "keyword-test"
    dim = 4
    normalize = True

    def embed(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        lowered = str(text).lower()
        return [
            float("intermodulation" in lowered or "pim" in lowered),
            float("cancellation" in lowered or "抑制" in lowered),
            float("channel" in lowered or "信道" in lowered),
            float("estimation" in lowered or "估计" in lowered),
        ]


def _chunks():
    return [
        {
            "chunk_id": "pim-p1-c1",
            "document_id": "pim-paper",
            "title": "Neural Passive Intermodulation Cancellation",
            "content": "A neural cancellation model estimates passive intermodulation interference.",
            "retrieval_content": "Section: Cancellation\nA neural cancellation model estimates passive intermodulation interference.",
            "metadata": {"page_number": 1, "heading": "Cancellation Method"},
        },
        {
            "chunk_id": "pim-p2-c1",
            "document_id": "pim-paper",
            "title": "Neural Passive Intermodulation Cancellation",
            "content": "The suppression experiment reports cancellation performance.",
            "metadata": {"page_number": 2, "heading": "Experiments"},
        },
        {
            "chunk_id": "channel-p1-c1",
            "document_id": "channel-paper",
            "title": "MIMO Channel Estimation",
            "content": "Sparse channel estimation improves pilot efficiency in MIMO OFDM.",
            "metadata": {"page_number": 1, "heading": "Channel Estimation"},
        },
        {
            "chunk_id": "channel-p2-c1",
            "document_id": "channel-paper",
            "title": "MIMO Channel Estimation",
            "content": "The estimator is evaluated under frequency selective fading.",
            "metadata": {"page_number": 2, "heading": "Evaluation"},
        },
    ]


class PaperStormRealEvalV52Test(unittest.TestCase):
    def test_candidate_generator_supports_five_distinct_review_intents(self):
        from knowledge_storm.paperstorm_real_eval_v52 import _candidate_queries

        queries = _candidate_queries(["无源互调", "神经网络", "抑制"])

        self.assertGreaterEqual(len(queries), 5)
        self.assertEqual(len(queries), len(set(queries)))
        self.assertTrue(any("证据" in query for query in queries))
        self.assertTrue(any("比较" in query for query in queries))

    def test_dataset_is_document_split_auditable_and_has_hard_negatives(self):
        from knowledge_storm.paperstorm_real_eval_v52 import build_auditable_dataset

        dataset = build_auditable_dataset(_chunks(), test_ratio=0.5, split_seed=7)

        dev_documents = {
            case["metadata"]["source_document_id"]
            for case in dataset["cases"]
            if case["split"] == "dev"
        }
        test_documents = {
            case["metadata"]["source_document_id"]
            for case in dataset["cases"]
            if case["split"] == "test"
        }
        self.assertTrue(dev_documents)
        self.assertTrue(test_documents)
        self.assertTrue(dev_documents.isdisjoint(test_documents))
        self.assertEqual(dataset["metadata"]["split_unit"], "document_id")
        self.assertEqual(dataset["metadata"]["annotation_status"], "auto_candidate")
        self.assertIn("corpus_sha256", dataset["metadata"])
        for case in dataset["cases"]:
            self.assertNotIn(case["metadata"]["source_title"], case["query"])
            self.assertEqual(case["metadata"]["review_status"], "needs_human_review")
            self.assertEqual(case["metadata"]["target_granularity"], "document")
            self.assertTrue(case["hard_negative_chunk_ids"])
            self.assertTrue(case["evidence"]["content_sha256"])

    def test_bootstrap_interval_is_deterministic_and_contains_mean(self):
        from knowledge_storm.paperstorm_real_eval_v52 import bootstrap_mean_ci

        first = bootstrap_mean_ci([0.0, 1.0, 1.0, 0.0], seed=11, samples=400)
        second = bootstrap_mean_ci([0.0, 1.0, 1.0, 0.0], seed=11, samples=400)

        self.assertEqual(first, second)
        self.assertLessEqual(first["low"], first["mean"])
        self.assertGreaterEqual(first["high"], first["mean"])
        self.assertEqual(first["n"], 4)

    def test_case_cap_samples_across_documents_instead_of_pdf_order(self):
        from knowledge_storm.paperstorm_real_eval_v52 import build_auditable_dataset

        dataset = build_auditable_dataset(
            _chunks(), test_ratio=0.5, split_seed=7, max_cases=2
        )

        sampled_documents = {
            case["metadata"]["source_document_id"] for case in dataset["cases"]
        }
        self.assertEqual(sampled_documents, {"pim-paper", "channel-paper"})

    def test_frozen_evaluation_selects_only_on_dev_and_reports_test_denominator(self):
        from knowledge_storm.paperstorm_real_eval_v52 import (
            build_auditable_dataset,
            run_frozen_evaluation,
        )

        dataset = build_auditable_dataset(_chunks(), test_ratio=0.5, split_seed=7)
        with tempfile.TemporaryDirectory() as temp_dir:
            report = run_frozen_evaluation(
                dataset,
                output_dir=Path(temp_dir),
                embedding_provider=_KeywordEmbedding(),
                modes=["bm25", "dense", "hybrid"],
                top_k=2,
                bootstrap_samples=100,
            )

        self.assertIn(report["selected_config"], {"bm25", "dense", "hybrid"})
        self.assertEqual(report["selection_split"], "dev")
        self.assertEqual(report["final_reporting_split"], "test")
        self.assertEqual(
            report["test"]["selected"]["case_count"],
            report["dataset"]["test_case_count"],
        )
        self.assertIn("confidence_intervals", report["test"]["selected"])
        self.assertTrue(report["integrity"]["test_was_not_used_for_selection"])

    def test_cross_lingual_cases_use_chinese_aliases_not_english_title_words(self):
        from knowledge_storm.paperstorm_real_eval_v52 import build_auditable_dataset

        dataset = build_auditable_dataset(
            _chunks(), test_ratio=0.5, split_seed=7, cross_lingual_only=True
        )

        self.assertTrue(dataset["cases"])
        self.assertTrue(
            all(
                case["metadata"]["query_type"] == "cross_lingual_semantic"
                for case in dataset["cases"]
            )
        )
        self.assertTrue(any("无源互调" in case["query"] for case in dataset["cases"]))
        self.assertTrue(any("信道估计" in case["query"] for case in dataset["cases"]))

    def test_cross_lingual_queries_do_not_assign_same_query_to_different_papers(self):
        from knowledge_storm.paperstorm_real_eval_v52 import build_auditable_dataset

        dataset = build_auditable_dataset(
            _chunks(), test_ratio=0.5, split_seed=7, cross_lingual_only=True
        )
        query_documents = {}
        for case in dataset["cases"]:
            query_documents.setdefault(case["query"], set()).add(
                case["metadata"]["source_document_id"]
            )

        self.assertTrue(
            all(len(documents) == 1 for documents in query_documents.values())
        )


if __name__ == "__main__":
    unittest.main()
