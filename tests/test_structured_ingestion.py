import json
import tempfile
import unittest
from pathlib import Path


class StructuredIngestionTest(unittest.TestCase):
    def test_builds_stable_document_sections_and_cross_page_children(self):
        from knowledge_storm.document_ingestion import ingest_document

        pages = [
            {
                "page_number": 1,
                "text": "Preface\n1 引言\n第一页内容 alpha beta\n1.1 方法\n方法起始 gamma",
            },
            {
                "page_number": 2,
                "text": "方法跨页延续 delta\n2 Results\nresult epsilon",
            },
        ]
        first = ingest_document("paper-1", pages, title="PIM Study", chunk_tokens=8, overlap_tokens=2)
        second = ingest_document("paper-1", pages, title="PIM Study", chunk_tokens=8, overlap_tokens=2)

        self.assertEqual(first, second)
        documents = [node for node in first if node["node_type"] == "document"]
        sections = [node for node in first if node["node_type"] == "section"]
        children = [node for node in first if node["node_type"] == "passage"]
        self.assertEqual(len(documents), 1)
        self.assertEqual([item["title"] for item in sections], ["1 引言", "1.1 方法", "2 Results"])
        method = sections[1]
        self.assertEqual(method["metadata"]["page_start"], 1)
        self.assertEqual(method["metadata"]["page_end"], 2)
        method_children = [item for item in children if item["parent_id"] == method["node_id"]]
        self.assertEqual({item["metadata"]["page_number"] for item in method_children}, {1, 2})
        self.assertTrue(all(item["chunk_id"] == item["node_id"] for item in children))

    def test_preserves_formula_and_table_as_atomic_retrievable_nodes(self):
        from knowledge_storm.document_ingestion import ingest_document

        display_formula = r"\[P_{IM}=20\log_{10}(V_3/V_1)\]"
        pages = [
            {
                "page_number": 1,
                "text": (
                    "1 Method\nordinary before $y=x^2+1$ ordinary after\n"
                    + display_formula
                    + "\n| Method | Score |\n| --- | ---: |\n| Hybrid | 0.91 |\nend"
                ),
            }
        ]
        nodes = ingest_document("paper-2", pages, chunk_tokens=5, overlap_tokens=1)
        formulas = [node for node in nodes if node["node_type"] == "formula"]
        tables = [node for node in nodes if node["node_type"] == "table"]
        passages = [node for node in nodes if node["node_type"] == "passage"]

        self.assertEqual({node["content"] for node in formulas}, {"$y=x^2+1$", display_formula})
        self.assertTrue(all(node["retrieval_content"] == node["content"] for node in formulas))
        self.assertEqual(len(tables), 1)
        self.assertEqual(
            tables[0]["content"],
            "| Method | Score |\n| --- | ---: |\n| Hybrid | 0.91 |",
        )
        self.assertEqual(tables[0]["retrieval_content"], tables[0]["content"])
        passage_text = "\n".join(node["content"] for node in passages)
        self.assertNotIn("P_{IM}", passage_text)
        self.assertNotIn("| Hybrid |", passage_text)

    def test_recognizes_chinese_numbered_heading_and_multiline_formula(self):
        from knowledge_storm.document_ingestion import ingest_document

        formula = "$$\nP_{IM}=a_1a_2a_3\n+ b_1b_2b_3\n$$"
        nodes = ingest_document(
            "paper-cn",
            [
                {
                    "page_number": 7,
                    "text": "封面说明\n（一）实验方法\n正文\n" + formula + "\n结尾",
                }
            ],
            title="中文论文",
            chunk_tokens=4,
            overlap_tokens=1,
        )

        sections = [node for node in nodes if node["node_type"] == "section"]
        formulas = [node for node in nodes if node["node_type"] == "formula"]
        self.assertEqual([node["title"] for node in sections], ["（一）实验方法"])
        self.assertEqual([node["content"] for node in formulas], [formula])
        self.assertEqual(formulas[0]["metadata"]["page_number"], 7)

    def test_extracts_embedded_and_cross_page_display_formulas(self):
        from knowledge_storm.document_ingestion import ingest_document

        nodes = ingest_document(
            "paper-formulas",
            [
                {
                    "page_number": 1,
                    "text": "1 Method\nBefore $$x+y$$ after\nBefore \\[a+b\\] after\nBefore $$x+",
                },
                {"page_number": 2, "text": "y$$ after and $z=1$ done"},
            ],
            chunk_tokens=2,
            overlap_tokens=0,
        )
        formulas = [node for node in nodes if node["node_type"] == "formula"]
        passages = [node for node in nodes if node["node_type"] == "passage"]

        self.assertEqual(
            [node["content"] for node in formulas],
            ["$$x+y$$", "\\[a+b\\]", "$$x+\ny$$", "$z=1$"],
        )
        cross_page = formulas[2]
        self.assertEqual(cross_page["metadata"]["page_number"], 1)
        self.assertEqual(cross_page["metadata"]["page_start"], 1)
        self.assertEqual(cross_page["metadata"]["page_end"], 2)
        passage_text = "\n".join(node["content"] for node in passages)
        self.assertNotIn("$$", passage_text)
        self.assertNotIn("\\[", passage_text)
        self.assertNotIn("\\]", passage_text)

    def test_chunk_pdf_pages_keeps_legacy_shape(self):
        from knowledge_storm.document_ingestion import chunk_pdf_pages

        chunks = chunk_pdf_pages(
            [{"page_number": 1, "text": "1 Intro\nlegacy text"}],
            document_id="legacy",
            title="Legacy",
        )

        self.assertTrue(chunks)
        self.assertNotIn("node_id", chunks[0])
        self.assertNotIn("node_type", chunks[0])

    def test_chunk_pdf_pages_preserves_legacy_heading_snapshot(self):
        from knowledge_storm.document_ingestion import chunk_pdf_pages

        text = "\n".join(
            ["front matter {0}".format(index) for index in range(1, 9)]
            + ["2 Method", "body"]
        )
        chunks = chunk_pdf_pages(
            [{"page_number": 4, "text": text}],
            document_id="legacy-doc",
            title="Legacy Title",
            chunk_tokens=100,
            overlap_tokens=0,
            strategy="contextual",
        )

        self.assertEqual(chunks[0]["metadata"]["heading"], "")
        self.assertEqual(chunks[0]["parent_id"], "legacy-doc::section::page-4")
        self.assertTrue(
            chunks[0]["retrieval_content"].startswith(
                "Document: Legacy Title\nSection: unknown\nPage: 4\n"
            )
        )
        self.assertEqual(
            chunks[0]["content"],
            "front matter 1 front matter 2 front matter 3 front matter 4 "
            "front matter 5 front matter 6 front matter 7 front matter 8 2 Method body",
        )

    def test_table_preserves_cell_whitespace_and_line_structure(self):
        from knowledge_storm.document_ingestion import ingest_document

        nodes = ingest_document(
            "paper-table",
            [{"page_number": 1, "text": "1 Data\n  | A cell | B  |  \n | --- | --- | \n | x y |  3 | "}],
            chunk_tokens=2,
            overlap_tokens=0,
        )
        table = next(node for node in nodes if node["node_type"] == "table")
        self.assertEqual(
            table["content"],
            "| A cell | B  |\n| --- | --- |\n| x y |  3 |",
        )
        self.assertEqual(len(table["content"].splitlines()), 3)


class ParentChildRetrievalTest(unittest.TestCase):
    @staticmethod
    def _nodes():
        from knowledge_storm.document_ingestion import ingest_document

        return ingest_document(
            "paper-3",
            [
                {
                    "page_number": 1,
                    "text": "1 Background\ncontext alpha beta gamma delta epsilon zeta eta theta",
                },
                {
                    "page_number": 2,
                    "text": "2 Method\nneedle passive intermodulation suppression result",
                },
            ],
            title="PIM",
            chunk_tokens=6,
            overlap_tokens=1,
        )

    def test_parents_are_not_candidates_and_manifest_counts_actual_stores(self):
        from knowledge_storm.retrieval import HashEmbeddingProvider, HybridPaperIndex

        nodes = self._nodes()
        index = HybridPaperIndex(nodes, HashEmbeddingProvider())
        results = index.search("needle", mode="bm25", top_k=10)

        self.assertTrue(results)
        self.assertTrue(all(item.get("node_type") not in {"document", "section"} for item in results))
        self.assertEqual(index.manifest["schema_revision"], 3)
        self.assertEqual(index.manifest["node_schema"], "structured-parent-child-v1")
        self.assertEqual(index.manifest["retrievable_count"], len(index.chunks))
        self.assertEqual(index.manifest["parent_count"], len(index.parents))

    def test_parent_expansion_respects_budget_without_changing_gold_identity(self):
        from knowledge_storm.retrieval import HashEmbeddingProvider, HybridPaperIndex

        index = HybridPaperIndex(self._nodes(), HashEmbeddingProvider())
        baseline = index.search("needle", mode="bm25", top_k=1)
        expanded = index.search("needle", mode="bm25", top_k=1, parent_budget_tokens=7)

        self.assertEqual(expanded[0]["chunk_id"], baseline[0]["chunk_id"])
        self.assertEqual(expanded[0]["node_id"], baseline[0]["node_id"])
        self.assertEqual(expanded[0]["final_rank"], baseline[0]["final_rank"])
        self.assertIn("parent_context", expanded[0])
        self.assertIn("expanded_content", expanded[0])
        self.assertLessEqual(len(expanded[0]["parent_context"].split()), 7)
        self.assertNotIn("parent_context", baseline[0])

    def test_missing_parent_is_safe(self):
        from knowledge_storm.retrieval import HashEmbeddingProvider, HybridPaperIndex

        index = HybridPaperIndex(
            [{"chunk_id": "orphan", "node_id": "orphan", "node_type": "passage", "parent_id": "missing", "content": "needle"}],
            HashEmbeddingProvider(),
        )
        result = index.search("needle", mode="bm25", top_k=1, parent_budget_tokens=10)[0]
        self.assertEqual(result["parent_context"], "")
        self.assertEqual(result["expanded_content"], "needle")

    def test_save_load_roundtrips_parent_store_and_rejects_revision_two(self):
        from knowledge_storm.retrieval import (
            HashEmbeddingProvider,
            HybridPaperIndex,
            IndexMigrationRequiredError,
        )

        provider = HashEmbeddingProvider()
        index = HybridPaperIndex(self._nodes(), provider)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "index.json"
            index.save(path)
            loaded = HybridPaperIndex.load(path, provider)
            self.assertEqual(loaded.parents, index.parents)
            self.assertEqual(loaded.manifest["parent_count"], len(loaded.parents))

            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["manifest"]["schema_revision"] = 2
            legacy = Path(temp_dir) / "revision-2.json"
            legacy.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(IndexMigrationRequiredError):
                HybridPaperIndex.load(legacy, provider)

            payload["manifest"].pop("schema_revision")
            missing_revision = Path(temp_dir) / "missing-revision.json"
            missing_revision.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(IndexMigrationRequiredError):
                HybridPaperIndex.load(missing_revision, provider)

    def test_manifest_is_recomputed_and_corruption_is_rejected_on_load(self):
        from knowledge_storm.retrieval import (
            HashEmbeddingProvider,
            HybridPaperIndex,
            IndexIntegrityError,
            IndexMigrationRequiredError,
        )

        provider = HashEmbeddingProvider()
        index = HybridPaperIndex(
            self._nodes(),
            provider,
            manifest={
                "schema_version": "wrong",
                "schema_revision": 2,
                "node_schema": "wrong",
                "chunk_count": 999,
                "retrievable_count": 999,
                "parent_count": 999,
            },
        )
        self.assertEqual(index.manifest["schema_version"], HybridPaperIndex.schema_version)
        self.assertEqual(index.manifest["schema_revision"], 3)
        self.assertEqual(index.manifest["node_schema"], HybridPaperIndex.node_schema)
        self.assertEqual(index.manifest["chunk_count"], len(index.chunks))
        self.assertEqual(index.manifest["retrievable_count"], len(index.chunks))
        self.assertEqual(index.manifest["parent_count"], len(index.parents))

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "index.json"
            index.save(path)
            payload = json.loads(path.read_text(encoding="utf-8"))

            payload["manifest"]["node_schema"] = "unknown-schema"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(IndexMigrationRequiredError):
                HybridPaperIndex.load(path, provider)

            payload["manifest"]["node_schema"] = HybridPaperIndex.node_schema
            payload["manifest"]["parent_count"] += 1
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(IndexIntegrityError):
                HybridPaperIndex.load(path, provider)

    def test_plain_chunks_roundtrip_without_empty_node_type(self):
        from knowledge_storm.retrieval import HashEmbeddingProvider, HybridPaperIndex

        provider = HashEmbeddingProvider()
        index = HybridPaperIndex(
            [{"chunk_id": "legacy-1", "document_id": "doc", "content": "plain legacy"}],
            provider,
        )
        self.assertNotIn("node_type", index.chunks[0])
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "index.json"
            index.save(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertNotIn("node_type", payload["chunks"][0])


if __name__ == "__main__":
    unittest.main()
