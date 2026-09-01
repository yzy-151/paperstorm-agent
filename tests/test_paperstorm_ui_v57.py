import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend" / "paperstorm_dashboard"


class PaperStormUIV57Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (FRONTEND / "index.html").read_text(encoding="utf-8")
        cls.css = (FRONTEND / "styles.css").read_text(encoding="utf-8")
        cls.script = (FRONTEND / "app.js").read_text(encoding="utf-8")
        cls.styles = cls.css
        cls.readme = (ROOT / "README.md").read_text(encoding="utf-8")

    def test_product_uses_v57_workspace_shell(self):
        self.assertIn("v7.2", self.html)
        self.assertIn('class="workspace-rail"', self.html)
        self.assertIn('class="workspace-main"', self.html)
        self.assertIn('class="workspace-inspector product-only"', self.html)

    def test_visual_language_is_square_and_dense(self):
        self.assertIn("--radius: 2px", self.css)
        self.assertIn("grid-template-columns: 208px minmax(0, 1fr) 288px", self.css)
        self.assertIn(".benchmark-card::before", self.css)

    def test_readme_presents_new_screenshots_diagram_and_benchmark_icons(self):
        for marker in (
            "dashboard-chat-v64.png",
            "paperstorm-research-flow-v65.gif",
            "paperstorm-agent-system-flow.svg",
            "paperstorm-async-runtime-sequence.svg",
            "benchmark-icon-retrieval.svg",
            "benchmark-icon-memory.svg",
            "benchmark-icon-context.svg",
            "benchmark-icon-answer.svg",
        ):
            self.assertIn(marker, self.readme)

    def test_real_pipeline_is_the_default_for_research_and_chat(self):
        self.assertIn(
            '<option value="paperstorm" selected>真实检索与 LLM</option>',
            self.html,
        )
        self.assertIn(
            '<option value="paperstorm" selected>真实 API</option>',
            self.html,
        )

    def test_advanced_keyword_filters_are_empty_by_default(self):
        self.assertIn('id="task-expected-keyword" placeholder=', self.html)
        self.assertIn('id="task-forbidden-keyword" placeholder=', self.html)
        self.assertNotIn('id="task-expected-keyword" value=', self.html)
        self.assertNotIn('id="task-forbidden-keyword" value=', self.html)

    def test_citations_link_to_article_anchor_and_original_source_title(self):
        self.assertIn("renderResearchArticle", self.script)
        self.assertIn("focusArticleCitation", self.script)
        self.assertIn('data-article-anchor=', self.script)
        self.assertIn("original_sources", self.script)
        self.assertIn("定位文章", self.script)
        self.assertIn("citation-target", self.styles)

    def test_public_copy_does_not_market_an_expert_edition(self):
        forbidden = ("专家版", "专业版", "专业工作台", "professional workspace")
        for marker in forbidden:
            self.assertNotIn(marker, self.html.lower())
            self.assertNotIn(marker, self.readme.lower())

    def test_research_pipeline_is_an_interactive_live_node_graph(self):
        bundle = self.html + self.script
        for marker in (
            'id="pipeline-canvas"',
            'class="pipeline-node',
            'id="pipeline-execution-wires"',
            'id="pipeline-artifact-wires"',
            'id="pipeline-node-title"',
            "applyPipelineTrace",
            "new EventSource",
            "pipelineExecutionEdges",
            "pipelineArtifactEdges",
        ):
            self.assertIn(marker, bundle)
        self.assertIn("@keyframes execution-flow", self.css)
        self.assertIn("@keyframes artifact-flow", self.css)

    def test_public_readme_links_professional_interview_materials(self):
        for marker in (
            "双 Agent 面试模拟器",
            "PAPERSTORM_RESUME_GUIDE.md",
            "RAG_AGENT_INTERVIEW_PLAYBOOK.md",
        ):
            self.assertIn(marker, self.readme)


if __name__ == "__main__":
    unittest.main()
