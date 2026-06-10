import tempfile
import unittest
from pathlib import Path


class PaperStormResearchQABenchmarkTest(unittest.TestCase):
    def test_research_qa_benchmark_writes_json_and_markdown_report(self):
        from knowledge_storm.paperstorm_research_qa_benchmark import (
            run_research_qa_benchmark,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            report = run_research_qa_benchmark(output_dir=Path(temp_dir))
            output_dir = Path(temp_dir)

            self.assertTrue((output_dir / "research_qa_benchmark_report.json").exists())
            self.assertTrue((output_dir / "research_qa_benchmark_report.md").exists())

        self.assertGreaterEqual(report["metrics"]["total_cases"], 3)
        self.assertIn("grounded_rate", report["metrics"])
        self.assertIn("retrieval_trigger_accuracy", report["metrics"])
        self.assertIn("low_confidence_rejection_rate", report["metrics"])
        self.assertTrue(report["cases"])


if __name__ == "__main__":
    unittest.main()
