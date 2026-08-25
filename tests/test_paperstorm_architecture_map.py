from pathlib import Path
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]


class ArchitectureMapTests(unittest.TestCase):
    def test_architecture_source_contains_current_system_boundaries(self):
        source = (
            ROOT / "docs" / "architecture" / "paperstorm-system-architecture.html"
        ).read_text(encoding="utf-8")

        for label in (
            "PaperStorm Agent Platform",
            "Research Pipeline",
            "Chat Agent Runtime",
            "RAG Retrieval",
            "STORM CORE",
            "Benchmark Registry",
            "SciFact",
            "QASPER",
            "LongMemEval-S",
        ):
            with self.subTest(label=label):
                self.assertIn(label, source)

    def test_readme_embeds_architecture_png_and_source(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn(
            "docs/architecture/paperstorm-system-architecture.png", readme
        )
        self.assertIn(
            "docs/architecture/paperstorm-system-architecture.html", readme
        )

    def test_executive_and_detailed_drawio_sources_are_editable(self):
        architecture_dir = ROOT / "docs" / "architecture"
        expected = {
            "paperstorm-executive-overview.drawio": (10, 8),
            "paperstorm-agent-system-flow.drawio": (24, 20),
        }

        for filename, (minimum_nodes, minimum_edges) in expected.items():
            with self.subTest(filename=filename):
                root = ET.parse(architecture_dir / filename).getroot()
                self.assertEqual(root.tag, "mxfile")
                cells = root.findall(".//mxCell")
                self.assertGreaterEqual(
                    sum(cell.get("vertex") == "1" for cell in cells), minimum_nodes
                )
                self.assertGreaterEqual(
                    sum(cell.get("edge") == "1" for cell in cells), minimum_edges
                )

    def test_svg_exports_cover_executive_and_agent_flows(self):
        architecture_dir = ROOT / "docs" / "architecture"
        executive = (
            architecture_dir / "paperstorm-executive-overview.svg"
        ).read_text(encoding="utf-8")
        detailed = (
            architecture_dir / "paperstorm-agent-system-flow.svg"
        ).read_text(encoding="utf-8")

        for label in ("业务需求", "智能问答", "深度调研", "业务价值"):
            self.assertIn(label, executive)
        for label in (
            "意图路由",
            "Planner Agent",
            "WikiWriter",
            "TopicExpert",
            "RAG 算法",
            "Memory 算法",
            "Context 工程",
        ):
            self.assertIn(label, detailed)

    def test_readme_presents_both_diagrams_and_editable_sources(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        for filename in (
            "paperstorm-executive-overview.svg",
            "paperstorm-agent-system-flow.svg",
            "paperstorm-executive-overview.drawio",
            "paperstorm-agent-system-flow.drawio",
        ):
            with self.subTest(filename=filename):
                self.assertIn(f"docs/architecture/{filename}", readme)


if __name__ == "__main__":
    unittest.main()
