import unittest
from pathlib import Path


class PaperStormDemoRunbookTest(unittest.TestCase):
    def test_service_launcher_parser_documents_local_demo_defaults(self):
        from examples.storm_examples.start_paperstorm_service import build_parser

        args = build_parser().parse_args(
            [
                "--service-root",
                "./results/paperstorm_demo_service",
                "--host",
                "127.0.0.1",
                "--port",
                "8765",
                "--reload",
            ]
        )

        self.assertEqual(args.service_root, "./results/paperstorm_demo_service")
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 8765)
        self.assertTrue(args.reload)

    def test_readme_documents_current_demo_runbook(self):
        root = Path(__file__).resolve().parents[1]
        readme = (root / "README.md").read_text(encoding="utf-8")

        self.assertIn("快速开始", readme)
        self.assertIn("paperstorm_service_api:app", readme)
        self.assertIn("http://127.0.0.1:8002", readme)
        self.assertIn("论文调研模式", readme)
        self.assertIn("智能问答模式", readme)


if __name__ == "__main__":
    unittest.main()
