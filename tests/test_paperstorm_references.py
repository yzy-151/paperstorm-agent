import json
import tempfile
import unittest
from pathlib import Path


class PaperStormReferencesTest(unittest.TestCase):
    def make_run_dir(self):
        temp_dir = tempfile.TemporaryDirectory()
        run_dir = Path(temp_dir.name)
        (run_dir / "storm_gen_article_polished.txt").write_text(
            "# Muon 优化器\n\nMuon 使用正交化动量更新。[1]",
            encoding="utf-8",
        )
        (run_dir / "url_to_info.json").write_text(
            json.dumps(
                {
                    "url_to_unified_index": {
                        "https://arxiv.org/abs/2502.16982v2": 1,
                    },
                    "url_to_info": {
                        "https://arxiv.org/abs/2502.16982v2": {
                            "title": "Muon is Scalable for LLM Training",
                            "url": "https://arxiv.org/abs/2502.16982v2",
                            "description": "Muon optimizer evidence.",
                            "snippets": ["Muon optimizer evidence."],
                            "meta": {
                                "authors": ["Keller Jordan", "Yuchen Jin"],
                                "published": "2025-02-24T00:00:00Z",
                                "pdf_url": "https://arxiv.org/pdf/2502.16982v2",
                            },
                        }
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return temp_dir, run_dir

    def test_loads_canonical_title_authors_and_arxiv_links(self):
        from knowledge_storm.paperstorm_references import load_reference_registry

        temp_dir, run_dir = self.make_run_dir()
        self.addCleanup(temp_dir.cleanup)

        references = load_reference_registry(run_dir)

        self.assertEqual(len(references), 1)
        self.assertEqual(references[0]["id"], 1)
        self.assertEqual(references[0]["title"], "Muon is Scalable for LLM Training")
        self.assertEqual(references[0]["authors"], ["Keller Jordan", "Yuchen Jin"])
        self.assertEqual(references[0]["url"], "https://arxiv.org/abs/2502.16982v2")
        self.assertEqual(references[0]["pdf_url"], "https://arxiv.org/pdf/2502.16982v2")

    def test_materializes_one_idempotent_reference_section_in_article(self):
        from knowledge_storm.paperstorm_references import materialize_article_references

        temp_dir, run_dir = self.make_run_dir()
        self.addCleanup(temp_dir.cleanup)

        first = materialize_article_references(run_dir)
        second = materialize_article_references(run_dir)
        content = (run_dir / "storm_gen_article_polished.txt").read_text(encoding="utf-8")

        self.assertEqual(first["reference_count"], 1)
        self.assertEqual(second["reference_count"], 1)
        self.assertEqual(content.count("## 参考文献"), 1)
        self.assertIn("[1] **Muon is Scalable for LLM Training**", content)
        self.assertIn("Keller Jordan, Yuchen Jin", content)
        self.assertIn("[原文](https://arxiv.org/abs/2502.16982v2)", content)

    def test_answer_reference_renderer_uses_structured_citations(self):
        from knowledge_storm.paperstorm_references import append_answer_references

        answer = append_answer_references(
            "Muon 适用于二维权重矩阵。[1]",
            [
                {
                    "id": 1,
                    "title": "Muon is Scalable for LLM Training",
                    "authors": ["Keller Jordan"],
                    "url": "https://arxiv.org/abs/2502.16982v2",
                }
            ],
        )

        self.assertIn("参考文献", answer)
        self.assertIn("Keller Jordan", answer)
        self.assertIn("https://arxiv.org/abs/2502.16982v2", answer)


if __name__ == "__main__":
    unittest.main()
