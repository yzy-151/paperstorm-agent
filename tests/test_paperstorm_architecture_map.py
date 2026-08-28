import importlib.util
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]


def load_diagram_generator():
    path = ROOT / "docs" / "architecture" / "generate_drawio_diagrams.py"
    spec = importlib.util.spec_from_file_location("paperstorm_diagram_generator", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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

    def test_drawio_sources_are_editable_and_include_runtime_sequence(self):
        architecture_dir = ROOT / "docs" / "architecture"
        expected = {
            "paperstorm-executive-overview.drawio": (10, 8),
            "paperstorm-agent-system-flow.drawio": (24, 20),
            "paperstorm-async-runtime-sequence.drawio": (9, 14),
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
                vertex_ids = {
                    cell.get("id") for cell in cells if cell.get("vertex") == "1"
                }
                for edge in (cell for cell in cells if cell.get("edge") == "1"):
                    with self.subTest(edge=edge.get("id")):
                        self.assertIn(edge.get("source"), vertex_ids)
                        self.assertIn(edge.get("target"), vertex_ids)

    def test_generator_outputs_match_committed_drawio_and_svg_artifacts(self):
        architecture_dir = ROOT / "docs" / "architecture"
        generator = load_diagram_generator()

        with TemporaryDirectory() as temporary_directory:
            original_root = generator.ROOT
            generator.ROOT = Path(temporary_directory)
            try:
                generator.main()
            finally:
                generator.ROOT = original_root

            for stem in (
                "paperstorm-executive-overview",
                "paperstorm-agent-system-flow",
                "paperstorm-async-runtime-sequence",
            ):
                for suffix in ("drawio", "svg"):
                    filename = f"{stem}.{suffix}"
                    with self.subTest(filename=filename):
                        self.assertEqual(
                            (Path(temporary_directory) / filename).read_bytes(),
                            (architecture_dir / filename).read_bytes(),
                        )

    def test_sequence_edges_reject_missing_or_out_of_range_offsets(self):
        generator = load_diagram_generator()
        with TemporaryDirectory() as temporary_directory:
            original_root = generator.ROOT
            generator.ROOT = Path(temporary_directory)
            try:
                for source_offset, target_offset in ((None, 0.5), (0.5, 1.1)):
                    with self.subTest(
                        source_offset=source_offset, target_offset=target_offset
                    ):
                        diagram = generator.Diagram(
                            "invalid-sequence",
                            400,
                            300,
                            "Invalid sequence",
                            "",
                            [
                                generator.Node("source", 20, 20, 100, 200, "Source"),
                                generator.Node("target", 220, 20, 100, 200, "Target"),
                            ],
                            [
                                generator.Edge(
                                    "source",
                                    "target",
                                    direction="sequence",
                                    source_offset=source_offset,
                                    target_offset=target_offset,
                                )
                            ],
                        )
                        with self.assertRaisesRegex(ValueError, "offset"):
                            generator.write_drawio(diagram)
            finally:
                generator.ROOT = original_root

    def test_async_runtime_sequence_covers_all_participants_and_messages(self):
        sequence = (
            ROOT / "docs" / "architecture" / "paperstorm-async-runtime-sequence.svg"
        ).read_text(encoding="utf-8")

        for participant in (
            "Browser",
            "FastAPI",
            "Async Queue",
            "Agent Runtime",
            "Retriever",
            "LLM",
            "Checkpoint",
            "SSE",
            "Langfuse",
        ):
            with self.subTest(participant=participant):
                self.assertIn(participant, sequence)

        root = ET.parse(
            ROOT / "docs" / "architecture" / "paperstorm-async-runtime-sequence.drawio"
        ).getroot()
        messages = [
            cell
            for cell in root.findall(".//mxCell")
            if cell.get("edge") == "1"
        ]
        self.assertGreaterEqual(len(messages), 14)

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
        for label in ("Async Queue", "Langfuse", "PIM Domain Pilot"):
            self.assertIn(label, executive)
        for label in (
            "意图路由",
            "Planner Agent",
            "WikiWriter",
            "TopicExpert",
            "RAG 算法",
            "Memory 算法",
            "Context 工程",
            "PIM Domain Pilot",
            "Langfuse Score",
            "Async Queue",
        ):
            self.assertIn(label, detailed)

    def test_readme_presents_both_diagrams_and_editable_sources(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        for filename in (
            "paperstorm-executive-overview.svg",
            "paperstorm-agent-system-flow.svg",
            "paperstorm-async-runtime-sequence.svg",
            "paperstorm-executive-overview.drawio",
            "paperstorm-agent-system-flow.drawio",
            "paperstorm-async-runtime-sequence.drawio",
        ):
            with self.subTest(filename=filename):
                self.assertIn(f"docs/architecture/{filename}", readme)


if __name__ == "__main__":
    unittest.main()
