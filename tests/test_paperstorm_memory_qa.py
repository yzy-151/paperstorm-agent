import json
import tempfile
import unittest
from pathlib import Path


class PaperStormMemoryQATest(unittest.TestCase):
    def make_run_dir(self, files):
        temp_dir = tempfile.TemporaryDirectory()
        run_dir = Path(temp_dir.name)
        for relative_path, content in files.items():
            path = run_dir / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, (dict, list)):
                path.write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")
            else:
                path.write_text(content, encoding="utf-8")
        self.addCleanup(temp_dir.cleanup)
        return run_dir

    def test_memory_store_persists_three_memory_layers_and_preferences(self):
        from knowledge_storm.paperstorm_memory import PaperStormMemoryStore

        store = PaperStormMemoryStore()
        store.append_working("当前问题：PIM 神经网络抑制")
        store.remember_episode("arXiv 检索发现 PIM 容易被误召回为 processing-in-memory")
        store.remember_semantic(
            "PIM 在射频场景中应消歧为 passive intermodulation。",
            tags=["PIM", "RF"],
        )
        store.set_preference("output_language", "zh")

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "memory.json"
            store.save(path)
            loaded = PaperStormMemoryStore.load(path)

        bundle = loaded.get_context_bundle(query="PIM RF")

        self.assertIn("working", bundle)
        self.assertIn("episodic", bundle)
        self.assertIn("semantic", bundle)
        self.assertEqual(bundle["preferences"]["output_language"], "zh")
        self.assertIn("passive intermodulation", bundle["semantic"][0]["content"])

    def test_context_compression_keeps_constraints_and_reports_validation(self):
        from knowledge_storm.paperstorm_memory import compress_context

        messages = [
            {"role": "user", "content": "topic 是 PIM 神经网络抑制，PIM 指无源互调。"},
            {"role": "assistant", "content": "检索时必须使用 passive intermodulation 和 RF。"},
            {"role": "tool", "content": "误召回 processing-in-memory 和 DRAM 论文，需要过滤。"},
        ]

        compressed = compress_context(
            messages,
            expected_keywords=["passive intermodulation", "RF"],
            forbidden_keywords=["processing-in-memory", "DRAM"],
        )

        self.assertIn("passive intermodulation", compressed["summary"])
        self.assertIn("RF", compressed["validation"]["expected_keyword_hits"])
        self.assertIn("processing-in-memory", compressed["validation"]["forbidden_keyword_hits"])
        self.assertFalse(compressed["validation"]["passed"])

    def test_qa_answers_from_run_artifacts_with_citations_and_memory(self):
        from knowledge_storm.paperstorm_memory import PaperStormMemoryStore
        from knowledge_storm.paperstorm_qa import PaperStormKnowledgeBase

        run_dir = self.make_run_dir(
            {
                "storm_gen_article_polished.txt": (
                    "# 无源互调抑制\n\n"
                    "无源互调 passive intermodulation 是射频无源器件非线性导致的杂散问题。[1]\n\n"
                    "神经网络方法可以学习非线性抵消器，用于 suppression 和 cancellation。[2]\n"
                ),
                "raw_search_results.json": [
                    {
                        "title": "Neural cancellation of passive intermodulation",
                        "url": "https://arxiv.org/abs/0000.00001",
                        "description": "RF passive intermodulation suppression with neural networks.",
                        "snippets": ["Neural cancellers reduce passive intermodulation products."],
                    }
                ],
            }
        )
        memory = PaperStormMemoryStore()
        memory.remember_semantic("PIM 在本任务中固定指 passive intermodulation。")

        kb = PaperStormKnowledgeBase.from_run_dir(run_dir)
        answer = kb.answer_question("PIM 神经网络抑制是什么？", memory_store=memory)

        self.assertIn("passive intermodulation", answer["answer"])
        self.assertTrue(answer["citations"])
        self.assertTrue(answer["grounded"])
        self.assertIn("semantic", answer["memory_context"])
        self.assertEqual(answer["evidence"][0]["source_type"], "article")
        self.assertIn("chunk_id", answer["evidence"][0])
        self.assertIn("score", answer["evidence"][0])
        self.assertIn("metadata", answer["evidence"][0])
        self.assertEqual(answer["citations"][0]["source_type"], "article")
        self.assertIn("chunk_id", answer["citations"][0])

    def test_article_citation_keeps_article_locator_and_original_paper_sources(self):
        from knowledge_storm.paperstorm_qa import PaperStormKnowledgeBase

        run_dir = self.make_run_dir(
            {
                "storm_gen_article_polished.txt": (
                    "# 定义与核心概念\n\n"
                    "物理 AI 连接数字空间与物理世界。[1]\n\n"
                    "# 技术基础\n\n"
                    "具身智能依赖多模态感知。[2]\n"
                ),
                "url_to_info.json": {
                    "url_to_unified_index": {
                        "https://arxiv.org/abs/2407.06886": 1,
                        "https://arxiv.org/abs/2509.12989": 2,
                    },
                    "url_to_info": {
                        "https://arxiv.org/abs/2407.06886": {
                            "title": "Embodied AI Survey",
                            "url": "https://arxiv.org/abs/2407.06886",
                            "meta": {"authors": ["A. Author"], "published": "2024-07-09"},
                        },
                        "https://arxiv.org/abs/2509.12989": {
                            "title": "PANORAMA",
                            "url": "https://arxiv.org/abs/2509.12989",
                        },
                    },
                },
                "raw_search_results.json": [],
            }
        )

        kb = PaperStormKnowledgeBase.from_run_dir(run_dir)
        answer = kb.answer_question("物理 AI 如何连接现实世界？", top_k=1)
        citation = answer["citations"][0]

        self.assertEqual(citation["title"], "定义与核心概念 · 第 1 段")
        self.assertEqual(citation["article_anchor"], "article-paragraph-1")
        self.assertEqual(citation["paragraph_index"], 1)
        self.assertEqual(citation["url"], "")
        self.assertEqual(citation["original_sources"][0]["title"], "Embodied AI Survey")
        self.assertEqual(
            citation["original_sources"][0]["url"],
            "https://arxiv.org/abs/2407.06886",
        )
        self.assertIn("## 参考文献", answer["answer"])
        self.assertIn("Embodied AI Survey", answer["answer"])
        self.assertIn("A. Author", answer["answer"])
        self.assertIn(
            "https://arxiv.org/abs/2407.06886", answer["answer"]
        )
        self.assertNotIn("Generated article paragraph", json.dumps(citation))

    def test_kb_answer_generator_produces_generated_answer(self):
        from knowledge_storm.paperstorm_qa import PaperStormKnowledgeBase

        run_dir = self.make_run_dir(
            {
                "storm_gen_article_polished.txt": (
                    "PIM passive intermodulation suppression with neural networks."
                ),
                "raw_search_results.json": [],
            }
        )
        kb = PaperStormKnowledgeBase.from_run_dir(run_dir)
        generated = kb.answer_question(
            "PIM 是什么？",
            answer_generator=lambda _prompt: "PIM 指 passive intermodulation，可用神经网络抑制。[1]",
        )
        self.assertEqual(
            generated["answer"],
            "PIM 指 passive intermodulation，可用神经网络抑制。[1]",
        )
        fallback = kb.answer_question("PIM 是什么？")
        self.assertTrue(fallback["answer"])

    def test_kb_extracts_content_and_usage_from_structured_llm_response(self):
        from knowledge_storm.paperstorm_qa import PaperStormKnowledgeBase

        run_dir = self.make_run_dir(
            {
                "storm_gen_article.txt": "Muon optimizer uses orthogonalized momentum.",
                "raw_search_results.json": [],
            }
        )
        kb = PaperStormKnowledgeBase.from_run_dir(run_dir)

        result = kb.answer_question(
            "Muon 是什么？",
            answer_generator=lambda _prompt: {
                "content": "Muon 使用正交化动量。[1]",
                "usage": {"total_tokens": 321},
                "latency_ms": 456.7,
                "cost_usd": 0.001,
            },
        )

        self.assertTrue(result["answer"].startswith("Muon 使用正交化动量。[1]"))
        self.assertNotIn("{'content'", result["answer"])
        self.assertEqual(result["generation"]["usage"]["total_tokens"], 321)
        self.assertEqual(result["generation"]["latency_ms"], 456.7)

    def test_kb_qa_tool_exposes_schema_and_runs(self):
        from knowledge_storm.paperstorm_tools import KnowledgeBaseQATool

        run_dir = self.make_run_dir(
            {
                "storm_gen_article.txt": "PIM means passive intermodulation in RF systems. [1]",
                "raw_search_results.json": [
                    {
                        "title": "Passive intermodulation survey",
                        "url": "https://example.com/pim",
                        "description": "PIM in RF systems.",
                    }
                ],
            }
        )

        tool = KnowledgeBaseQATool()
        schema = tool.to_schema()
        result = tool.run({"run_dir": str(run_dir), "question": "PIM 是什么？"})

        self.assertEqual(schema["name"], "kb_qa")
        self.assertIn("run_dir", schema["input_schema"]["required"])
        self.assertIn("passive intermodulation", result["answer"])
        self.assertTrue(result["citations"])

    def test_research_qa_tool_exposes_schema_and_runs_fake_agent(self):
        from knowledge_storm.paperstorm_tools import ResearchQATool

        with tempfile.TemporaryDirectory() as temp_dir:
            tool = ResearchQATool(service_root=Path(temp_dir))
            schema = tool.to_schema()
            result = tool.run(
                {
                    "question": "PIM 是什么？",
                    "topic": "pim 神经网络抑制",
                    "run_mode": "fake",
                    "expected_keywords": ["passive intermodulation"],
                    "forbidden_keywords": ["DRAM"],
                }
            )

        self.assertEqual(schema["name"], "research_qa")
        self.assertIn("question", schema["input_schema"]["required"])
        self.assertEqual(result["decision"]["action"], "retrieve_then_answer")
        self.assertTrue(result["citations"])
        self.assertIn("evidence_sufficiency", result)

    def test_evaluate_qa_artifact_scores_grounded_answers(self):
        from knowledge_storm.paperstorm_eval import EvalCase, evaluate_qa_artifact

        run_dir = self.make_run_dir(
            {
                "qa_answer.json": {
                    "question": "PIM 是什么？",
                    "answer": "PIM 是 RF 系统中的 passive intermodulation 问题。[1]",
                    "citations": [{"id": 1, "url": "https://example.com/pim"}],
                    "grounded": True,
                }
            }
        )
        case = EvalCase(
            topic="pim 神经网络抑制",
            expected_keywords=["passive intermodulation", "RF"],
            forbidden_keywords=["processing-in-memory"],
            expected_language="zh",
            min_sources=1,
        )

        scorecard = evaluate_qa_artifact(run_dir, case)

        self.assertGreater(scorecard["scores"]["qa_quality"], 20)
        self.assertTrue(scorecard["checks"]["qa_has_citation"])
        self.assertTrue(scorecard["checks"]["qa_grounded"])

    def test_runtime_session_runs_tool_with_trace_and_working_memory(self):
        from knowledge_storm.paperstorm_runtime import PaperStormRuntimeSession
        from knowledge_storm.paperstorm_tools import KnowledgeBaseQATool

        run_dir = self.make_run_dir(
            {
                "storm_gen_article.txt": "PIM means passive intermodulation in RF systems. [1]",
                "raw_search_results.json": [
                    {
                        "title": "Passive intermodulation survey",
                        "url": "https://example.com/pim",
                        "description": "PIM in RF systems.",
                    }
                ],
            }
        )
        trace_path = run_dir / "paperstorm_trace.jsonl"
        session = PaperStormRuntimeSession(run_id="unit-test", trace_path=trace_path)
        session.register_tool(KnowledgeBaseQATool())

        result = session.call_tool(
            "kb_qa",
            {"run_dir": str(run_dir), "question": "PIM 是什么？"},
        )
        events = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
        ]
        memory = session.memory.get_context_bundle(query="kb_qa")

        self.assertIn("passive intermodulation", result["answer"])
        self.assertEqual(events[0]["event"], "tool_start")
        self.assertEqual(events[-1]["event"], "tool_end")
        self.assertEqual(events[-1]["status"], "success")
        self.assertTrue(memory["working"])


if __name__ == "__main__":
    unittest.main()
