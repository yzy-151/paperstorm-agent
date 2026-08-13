import unittest
from pathlib import Path


class PaperStormReleaseDocsTest(unittest.TestCase):
    def test_public_docs_explain_design_and_reproducibility(self):
        root = Path(__file__).resolve().parents[1]
        readme = (root / "README.md").read_text(encoding="utf-8")
        evaluation = (root / "docs" / "PAPERSTORM_V55_PUBLIC_BENCHMARKS.md").read_text(
            encoding="utf-8"
        )

        self.assertIn("Benchmark", readme)
        self.assertIn("如何复现", readme)
        self.assertIn("真实论文", readme)
        self.assertIn("SciFact", evaluation)
        self.assertIn("QASPER", evaluation)

    def test_public_release_does_not_require_internal_handoff_docs(self):
        root = Path(__file__).resolve().parents[1]
        readme = (root / "README.md").read_text(encoding="utf-8")

        self.assertNotIn("HANDOFF", readme.upper())
        self.assertNotIn("docs/superpowers", readme)


if __name__ == "__main__":
    unittest.main()
