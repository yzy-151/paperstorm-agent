import unittest
from types import SimpleNamespace

from knowledge_storm.rm import ArxivRM, LocalPDFRM


ARXIV_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v1</id>
    <updated>2024-01-02T00:00:00Z</updated>
    <published>2024-01-01T00:00:00Z</published>
    <title>Retrieval Augmented Generation Evaluation</title>
    <summary>
      This paper studies evaluation methods for retrieval augmented generation.
    </summary>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Lee</name></author>
    <arxiv:primary_category term="cs.CL" />
    <category term="cs.CL" />
    <link href="http://arxiv.org/abs/2401.12345v1" rel="alternate" type="text/html" />
    <link title="pdf" href="http://arxiv.org/pdf/2401.12345v1" rel="related" type="application/pdf" />
  </entry>
</feed>
"""


ARXIV_PIM_AMBIGUOUS_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2601.00001v1</id>
    <updated>2026-01-02T00:00:00Z</updated>
    <published>2026-01-01T00:00:00Z</published>
    <title>Neural Network Suppression of Passive Intermodulation in RF Front Ends</title>
    <summary>
      We study passive intermodulation mitigation using neural network cancellers
      for radio frequency systems and antenna feed networks.
    </summary>
    <author><name>Alice RF</name></author>
    <arxiv:primary_category term="eess.SP" />
    <category term="eess.SP" />
    <link href="http://arxiv.org/abs/2601.00001v1" rel="alternate" type="text/html" />
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2601.00002v1</id>
    <updated>2026-01-02T00:00:00Z</updated>
    <published>2026-01-01T00:00:00Z</published>
    <title>PIM: Processing-in-Memory System for Efficient RAM Access</title>
    <summary>
      This paper presents a processing-in-memory architecture for DRAM and RAM
      acceleration in computer systems.
    </summary>
    <author><name>Bob Memory</name></author>
    <arxiv:primary_category term="cs.AR" />
    <category term="cs.AR" />
    <link href="http://arxiv.org/abs/2601.00002v1" rel="alternate" type="text/html" />
  </entry>
</feed>
"""


ARXIV_MUON_AMBIGUOUS_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>https://arxiv.org/abs/2502.16982v2</id>
    <updated>2025-03-01T00:00:00Z</updated>
    <published>2025-02-24T00:00:00Z</published>
    <title>Muon is Scalable for LLM Training</title>
    <summary>
      We study the Muon optimizer, an orthogonalized momentum method based on
      Newton-Schulz iterations, for training large neural networks.
    </summary>
    <author><name>Optimizer Researcher</name></author>
    <arxiv:primary_category term="cs.LG" />
    <category term="cs.LG" />
    <link href="https://arxiv.org/abs/2502.16982v2" rel="alternate" type="text/html" />
  </entry>
  <entry>
    <id>https://arxiv.org/abs/hep-ex/0602035v1</id>
    <updated>2006-02-20T00:00:00Z</updated>
    <published>2006-02-20T00:00:00Z</published>
    <title>Final Report of the Muon E821 Anomalous Magnetic Moment Measurement</title>
    <summary>
      We report a particle physics measurement of the muon anomalous magnetic moment.
    </summary>
    <author><name>Muon Collaboration</name></author>
    <arxiv:primary_category term="hep-ex" />
    <category term="hep-ex" />
    <link href="https://arxiv.org/abs/hep-ex/0602035v1" rel="alternate" type="text/html" />
  </entry>
</feed>
"""


class PaperStormRetrieversTest(unittest.TestCase):
    def test_arxiv_rm_maps_atom_entries_to_storm_results(self):
        rm = ArxivRM(k=1)
        rm.request = lambda query: ARXIV_SAMPLE

        results = rm.forward("retrieval augmented generation evaluation")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["url"], "http://arxiv.org/abs/2401.12345v1")
        self.assertEqual(
            results[0]["title"], "Retrieval Augmented Generation Evaluation"
        )
        self.assertIn(
            "evaluation methods", results[0]["snippets"][0]
        )
        self.assertEqual(results[0]["meta"]["source_type"], "arxiv")
        self.assertEqual(results[0]["meta"]["authors"], ["Alice Smith", "Bob Lee"])
        self.assertEqual(
            results[0]["meta"]["pdf_url"], "http://arxiv.org/pdf/2401.12345v1"
        )

    def test_arxiv_rm_skips_empty_queries(self):
        rm = ArxivRM(k=1)
        calls = []
        rm.request = lambda query: calls.append(query) or ARXIV_SAMPLE

        results = rm.forward(["", "   "])

        self.assertEqual(results, [])
        self.assertEqual(calls, [])

    def test_arxiv_rm_disambiguates_pim_as_passive_intermodulation(self):
        rm = ArxivRM(k=2)
        calls = []
        rm.request = lambda query: calls.append(query) or ARXIV_PIM_AMBIGUOUS_SAMPLE

        results = rm.forward("pim 神经网络抑制")

        self.assertEqual(calls, ["passive intermodulation neural network suppression"])
        self.assertEqual(len(results), 1)
        self.assertIn("Passive Intermodulation", results[0]["title"])
        self.assertNotIn("Processing-in-Memory", results[0]["title"])

    def test_arxiv_rm_skips_memory_pim_queries_when_using_paperstorm(self):
        rm = ArxivRM(k=1)
        calls = []
        rm.request = lambda query: calls.append(query) or ARXIV_SAMPLE

        results = rm.forward("PIM RAM processing-in-memory system")

        self.assertEqual(results, [])
        self.assertEqual(calls, [])

    def test_arxiv_rm_compiles_muon_optimizer_into_constrained_queries(self):
        queries = ArxivRM._compile_queries_for_arxiv(
            "Muon optimizer neural network training"
        )

        self.assertGreaterEqual(len(queries), 2)
        self.assertIn('all:"Muon optimizer"', queries[0])
        self.assertIn("AND", queries[0])
        self.assertTrue(
            any("Newton-Schulz" in query or "orthogonalized momentum" in query for query in queries)
        )

    def test_arxiv_rm_translates_chinese_muon_topic_and_filters_particle_physics(self):
        rm = ArxivRM(k=3)
        calls = []

        def request(query):
            calls.append(query)
            if len(calls) == 1:
                return '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
            return ARXIV_MUON_AMBIGUOUS_SAMPLE

        rm.request = request
        results = rm.forward("muon优化器")

        self.assertGreaterEqual(len(calls), 2)
        self.assertTrue(
            all(
                "Muon" in query or "momentum" in query or "Newton-Schulz" in query
                for query in calls
            )
        )
        self.assertEqual([item["title"] for item in results], ["Muon is Scalable for LLM Training"])
        self.assertEqual(results[0]["meta"]["authors"], ["Optimizer Researcher"])

    def test_arxiv_rm_compiles_wavelet_neural_network_queries_in_english(self):
        queries = ArxivRM._compile_queries_for_arxiv("小波神经网络 核心结构")

        self.assertGreaterEqual(len(queries), 1)
        self.assertTrue(all("wavelet" in query.lower() for query in queries))
        self.assertTrue(all("neural network" in query.lower() for query in queries))

    def test_arxiv_rm_rejects_generic_neural_network_results_for_wavelet_query(self):
        rm = ArxivRM(k=3)
        rm.request = lambda _query: ARXIV_SAMPLE

        self.assertEqual(rm.forward("小波神经网络"), [])

    def test_arxiv_rm_rejects_neural_network_paper_without_muon_optimizer_method(self):
        rm = ArxivRM(k=3)
        response = ARXIV_MUON_AMBIGUOUS_SAMPLE.replace(
            "Muon is Scalable for LLM Training",
            "Physics-Informed Neural Networks for Magnetohydrodynamics",
        ).replace(
            "We study the Muon optimizer, an orthogonalized momentum method based on\n      Newton-Schulz iterations, for training large neural networks.",
            "Physics-informed neural networks solve magnetohydrodynamics optimization problems.",
        )
        rm.request = lambda _query: response

        results = rm.forward("Muon optimizer neural network training")

        self.assertEqual(results, [])

    def test_local_pdf_rm_retrieves_relevant_chunks_from_loaded_documents(self):
        rm = LocalPDFRM(
            documents=[
                {
                    "title": "RAG Evaluation Survey",
                    "path": "papers/rag-eval.pdf",
                    "text": (
                        "Retrieval augmented generation evaluation uses faithfulness "
                        "and answer relevance metrics. Wireless channel estimation is "
                        "a different topic."
                    ),
                }
            ],
            k=1,
            chunk_size=80,
            chunk_overlap=0,
        )

        results = rm.forward("faithfulness metrics")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "RAG Evaluation Survey")
        self.assertIn("faithfulness", results[0]["snippets"][0])
        self.assertEqual(results[0]["meta"]["source_type"], "local_pdf")
        self.assertEqual(results[0]["meta"]["pdf_path"], "papers/rag-eval.pdf")

    def test_local_pdf_rm_skips_empty_queries(self):
        rm = LocalPDFRM(
            documents=[
                {
                    "title": "RAG Evaluation Survey",
                    "path": "papers/rag-eval.pdf",
                    "text": "Retrieval augmented generation evaluation.",
                }
            ]
        )

        self.assertEqual(rm.forward(["", "   "]), [])

    def test_paper_storm_runner_builds_arxiv_retriever(self):
        from examples.storm_examples.run_paper_storm_minimax import (
            build_paper_retriever,
        )

        rm = build_paper_retriever(
            SimpleNamespace(retriever="arxiv", search_top_k=2, pdf_dir=None)
        )

        self.assertIsInstance(rm, ArxivRM)
        self.assertEqual(rm.k, 2)

    def test_paper_storm_runner_requires_pdf_dir_for_local_pdf(self):
        from examples.storm_examples.run_paper_storm_minimax import (
            build_paper_retriever,
        )

        with self.assertRaises(ValueError):
            build_paper_retriever(
                SimpleNamespace(retriever="local-pdf", search_top_k=2, pdf_dir=None)
            )

    def test_paper_storm_runner_builds_deepseek_flash_settings(self):
        from examples.storm_examples.run_paper_storm_minimax import (
            build_lm_settings,
        )

        settings = build_lm_settings(
            SimpleNamespace(llm_provider="deepseek", llm_model="flash")
        )

        self.assertEqual(settings["model"], "openai/deepseek-v4-flash")
        self.assertEqual(settings["api_env"], "DEEPSEEK_API_KEY")
        self.assertEqual(settings["api_base"], "https://api.deepseek.com")

    def test_deepseek_storm_pipeline_disables_default_thinking_mode(self):
        from unittest import mock

        from examples.storm_examples.run_paper_storm_minimax import build_lm_configs

        args = SimpleNamespace(llm_provider="deepseek", llm_model="flash")
        with mock.patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            configs = build_lm_configs(args)

        self.assertEqual(
            configs.outline_gen_lm.kwargs["extra_body"],
            {"thinking": {"type": "disabled"}},
        )

    def test_paper_storm_runner_builds_minimax_settings(self):
        from examples.storm_examples.run_paper_storm_minimax import (
            build_lm_settings,
        )

        settings = build_lm_settings(
            SimpleNamespace(llm_provider="minimax", llm_model=None)
        )

        self.assertEqual(settings["model"], "openai/MiniMax-M3")
        self.assertEqual(settings["api_env"], "MINIMAX_API_KEY")

    def test_paper_storm_runner_uses_larger_outline_token_budget(self):
        from examples.storm_examples.run_paper_storm_minimax import (
            build_lm_token_limits,
        )

        limits = build_lm_token_limits()

        self.assertGreaterEqual(limits["outline_gen"], 2600)
        self.assertGreaterEqual(limits["article_gen"], 5000)
        self.assertGreaterEqual(limits["article_polish"], 7000)

    def test_arxiv_search_tool_exposes_schema_and_runs(self):
        from knowledge_storm.paperstorm_tools import ArxivSearchTool

        rm = ArxivRM(k=1)
        rm.request = lambda query: ARXIV_SAMPLE
        tool = ArxivSearchTool(rm=rm)

        result = tool.run({"query": "retrieval augmented generation", "top_k": 1})

        self.assertEqual(tool.name, "arxiv_search")
        self.assertIn("input_schema", tool.to_schema())
        self.assertEqual(tool.to_schema()["input_schema"]["required"], ["query"])
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(
            result["results"][0]["title"], "Retrieval Augmented Generation Evaluation"
        )

    def test_local_pdf_search_tool_exposes_schema_and_runs(self):
        from knowledge_storm.paperstorm_tools import LocalPDFSearchTool

        rm = LocalPDFRM(
            documents=[
                {
                    "title": "PIM Neural Canceller",
                    "path": "papers/pim.pdf",
                    "text": "Passive intermodulation can be mitigated by neural network cancellers.",
                }
            ],
            k=1,
            chunk_size=80,
            chunk_overlap=0,
        )
        tool = LocalPDFSearchTool(rm=rm)

        result = tool.run({"query": "neural network cancellers", "top_k": 1})

        self.assertEqual(tool.name, "local_pdf_search")
        self.assertEqual(tool.to_schema()["input_schema"]["required"], ["query"])
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["results"][0]["meta"]["source_type"], "local_pdf")


if __name__ == "__main__":
    unittest.main()
