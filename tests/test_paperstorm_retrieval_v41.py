import json
import tempfile
import unittest
from pathlib import Path


class _FakeEmbeddingProvider:
    name = "fake-multilingual"
    dim = 3
    normalize = True

    def embed(self, texts):
        vectors = []
        for text in texts:
            lowered = str(text).lower()
            vectors.append(
                [
                    float(
                        "passive intermodulation" in lowered or "无源互调" in lowered
                    ),
                    float("dram" in lowered or "processing-in-memory" in lowered),
                    float("cancellation" in lowered or "抵消" in lowered),
                ]
            )
        return vectors

    def embed_query(self, text):
        return self.embed([text])[0]


class PaperStormRetrievalV41Test(unittest.TestCase):
    def test_multilingual_tokenizer_keeps_terms_numbers_and_chinese_bigrams(self):
        from knowledge_storm.paperstorm_retrieval_v41 import multilingual_tokenize

        tokens = multilingual_tokenize("三阶无源互调 2f1-f2 PIM-3")

        self.assertIn("无源", tokens)
        self.assertIn("互调", tokens)
        self.assertIn("2f1-f2", tokens)
        self.assertIn("pim-3", tokens)

    def test_query_tokenizer_removes_question_boilerplate_but_keeps_domain_terms(self):
        from knowledge_storm.paperstorm_retrieval_v41 import retrieval_query_tokens

        tokens = retrieval_query_tokens(
            "相关研究中，micro-size 与 high-speed 的作用或关系是什么？"
        )

        self.assertIn("micro-size", tokens)
        self.assertIn("high-speed", tokens)
        self.assertNotIn("相关", tokens)
        self.assertNotIn("作用", tokens)
        self.assertNotIn("关系", tokens)

    def test_rrf_fuses_rankings_without_mixing_raw_score_scales(self):
        from knowledge_storm.paperstorm_retrieval_v41 import reciprocal_rank_fusion

        fused = reciprocal_rank_fusion(
            [
                [{"chunk_id": "a", "score": 1000}, {"chunk_id": "b", "score": 900}],
                [{"chunk_id": "b", "score": 0.91}, {"chunk_id": "c", "score": 0.90}],
            ],
            rank_constant=60,
        )

        self.assertEqual(fused[0]["chunk_id"], "b")
        self.assertEqual(fused[0]["fusion_hits"], 2)
        self.assertIn("rrf_score", fused[0])

    def test_hybrid_index_exposes_all_ablation_modes_and_stage_scores(self):
        from knowledge_storm.paperstorm_retrieval_v41 import HybridPaperIndex

        chunks = [
            {
                "chunk_id": "pim",
                "content": "Passive intermodulation is RF nonlinear distortion.",
            },
            {
                "chunk_id": "dram",
                "content": "Processing-in-memory reduces DRAM movement.",
            },
            {
                "chunk_id": "cancel",
                "content": "Neural cancellation suppresses 无源互调 interference.",
            },
        ]
        index = HybridPaperIndex(chunks, embedding_provider=_FakeEmbeddingProvider())

        bm25 = index.search("PIM passive intermodulation", mode="bm25", top_k=2)
        dense = index.search("无源互调", mode="dense", top_k=2)
        hybrid = index.search("无源互调 cancellation", mode="hybrid", top_k=3)
        reranked = index.search(
            "无源互调 cancellation",
            mode="hybrid_rerank",
            top_k=2,
            reranker=lambda query, items: list(reversed(items)),
        )

        self.assertEqual(bm25[0]["chunk_id"], "pim")
        self.assertEqual(dense[0]["chunk_id"], "pim")
        self.assertTrue(all("rrf_score" in item for item in hybrid))
        self.assertEqual(reranked[0]["chunk_id"], hybrid[-1]["chunk_id"])
        self.assertEqual(reranked[0]["retrieval_mode"], "hybrid_rerank")

    def test_cross_encoder_adapter_only_scores_first_stage_candidates(self):
        from knowledge_storm.paperstorm_retrieval_v41 import CrossEncoderReranker

        calls = []

        def score_pairs(pairs):
            calls.extend(pairs)
            return [1.0 if "relevant" in document else 0.0 for _, document in pairs]

        reranker = CrossEncoderReranker(model_name="fake", score_fn=score_pairs)
        output = reranker.rerank(
            "query",
            [
                {"chunk_id": "noise", "content": "noise"},
                {"chunk_id": "hit", "content": "relevant evidence"},
            ],
            top_k=1,
        )

        self.assertEqual(len(calls), 2)
        self.assertEqual(output[0]["chunk_id"], "hit")
        self.assertEqual(output[0]["rerank_score"], 1.0)

    def test_structured_contextual_chunks_preserve_page_heading_and_parent(self):
        from knowledge_storm.paperstorm_document_v41 import chunk_pdf_pages

        pages = [
            {
                "page_number": 7,
                "text": "2 PIM Suppression\nPassive intermodulation can be cancelled digitally.\n"
                "A neural model estimates the nonlinear interference.",
            }
        ]
        chunks = chunk_pdf_pages(
            pages,
            document_id="paper-1",
            title="PIM Study",
            chunk_tokens=12,
            overlap_tokens=3,
            strategy="contextual",
        )

        self.assertGreaterEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["metadata"]["page_number"], 7)
        self.assertEqual(chunks[0]["metadata"]["heading"], "2 PIM Suppression")
        self.assertTrue(chunks[0]["parent_id"].startswith("paper-1::section::"))
        self.assertIn("Document: PIM Study", chunks[0]["retrieval_content"])
        self.assertNotEqual(chunks[0]["retrieval_content"], chunks[0]["content"])

    def test_index_manifest_rejects_wrong_embedding_model_or_dimension(self):
        from knowledge_storm.paperstorm_retrieval_v41 import HybridPaperIndex

        chunks = [{"chunk_id": "one", "content": "passive intermodulation"}]
        index = HybridPaperIndex(chunks, embedding_provider=_FakeEmbeddingProvider())

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "index.json"
            index.save(path)
            loaded = HybridPaperIndex.load(
                path, embedding_provider=_FakeEmbeddingProvider()
            )

            class WrongProvider(_FakeEmbeddingProvider):
                name = "wrong-model"

            with self.assertRaisesRegex(ValueError, "embedding model mismatch"):
                HybridPaperIndex.load(path, embedding_provider=WrongProvider())

        self.assertEqual(loaded.manifest["embedding_dimension"], 3)

    def test_ablation_report_compares_four_retrievers_and_chunk_strategies(self):
        from knowledge_storm.paperstorm_ablation_v41 import run_ablation
        from knowledge_storm.paperstorm_eval_v4 import build_seed_dataset

        dataset = build_seed_dataset()
        with tempfile.TemporaryDirectory() as temp_dir:
            report = run_ablation(
                dataset,
                output_dir=temp_dir,
                embedding_provider=_FakeEmbeddingProvider(),
                reranker_score_fn=lambda pairs: [
                    float("无源互调" in doc) for _, doc in pairs
                ],
                modes=["bm25", "dense", "hybrid", "hybrid_rerank"],
                chunk_strategies=["ordinary", "contextual"],
                top_k=5,
            )
            saved = json.loads(
                (Path(temp_dir) / "rag_eval_v41_ablation.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(len(report["experiments"]), 8)
        self.assertEqual(saved["project"], "PaperStorm Retrieval Ablation v4.1")
        self.assertIn("retrieval_recall_at_k", saved["experiments"][0]["metrics"])

    def test_contextual_ablation_uses_document_context_from_pdf_pipeline(self):
        from knowledge_storm.paperstorm_ablation_v41 import _dataset_chunks

        dataset = {
            "corpus": [
                {
                    "document_id": "paper-page",
                    "chunk_ids": ["paper-page"],
                    "title": "PIM Paper",
                    "text": "raw chunk",
                    "metadata": {
                        "category": "real_paper",
                        "context": "Document: PIM Paper\nSection: Suppression\nPage: 7\nraw chunk",
                    },
                }
            ]
        }

        contextual = _dataset_chunks(dataset, "contextual")

        self.assertIn("Section: Suppression", contextual[0]["retrieval_content"])
        self.assertIn("Page: 7", contextual[0]["retrieval_content"])

    def test_zotero_source_resolves_storage_paths_and_deduplicates_titles(self):
        from knowledge_storm.paperstorm_zotero import (
            deduplicate_papers,
            resolve_attachment_path,
        )

        root = Path("D:/portable-zotero")
        resolved = resolve_attachment_path(root, "ABCD1234", "storage:paper.pdf")
        papers = deduplicate_papers(
            [
                {"title": "Passive Intermodulation Study", "path": "first.pdf"},
                {"title": " passive  intermodulation study ", "path": "duplicate.pdf"},
                {"title": "Neural PIM Cancellation", "path": "second.pdf"},
            ]
        )

        self.assertEqual(resolved, root / "storage" / "ABCD1234" / "paper.pdf")
        self.assertEqual(len(papers), 2)

    def test_real_paper_dataset_marks_weak_supervision_and_uses_chunk_citations(self):
        from knowledge_storm.paperstorm_zotero import build_weak_paper_dataset

        chunks = [
            {
                "chunk_id": "paper-a::p1::c1",
                "document_id": "paper-a",
                "title": "PIM Cancellation",
                "content": "Neural cancellation estimates passive intermodulation interference.",
                "retrieval_content": "Document: PIM Cancellation\nNeural cancellation estimates PIM.",
                "metadata": {"page_number": 1, "heading": "Abstract"},
            }
        ]
        dataset = build_weak_paper_dataset(chunks, source_label="local-zotero")

        self.assertEqual(dataset["metadata"]["provenance"], "local-zotero")
        self.assertTrue(dataset["metadata"]["domain_review_required"])
        self.assertEqual(dataset["cases"][0]["relevant_chunk_ids"], ["paper-a::p1::c1"])
        self.assertEqual(
            dataset["cases"][0]["allowed_citation_ids"], ["paper-a::p1::c1"]
        )


if __name__ == "__main__":
    unittest.main()
