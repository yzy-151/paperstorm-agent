import unittest
from pathlib import Path


class PaperStormReleaseDocsTest(unittest.TestCase):
    def test_public_docs_explain_design_and_reproducibility(self):
        root = Path(__file__).resolve().parents[1]
        readme = (root / "README.md").read_text(encoding="utf-8")
        design_sources = (root / "docs" / "DESIGN_SOURCES.md").read_text(
            encoding="utf-8"
        )

        self.assertIn("Benchmark", readme)
        self.assertIn("如何复现", readme)
        self.assertIn("真实论文", readme)
        self.assertIn("Claude Code", design_sources)
        self.assertIn("Hermes", design_sources)


if __name__ == "__main__":
    unittest.main()
