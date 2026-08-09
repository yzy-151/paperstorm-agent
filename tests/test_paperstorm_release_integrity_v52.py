import os
import re
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


class PaperStormReleaseIntegrityV52Test(unittest.TestCase):
    def test_llm_builders_are_offline_by_default_even_when_key_exists(self):
        from knowledge_storm.paperstorm_router_llm import (
            build_chat_llm_callable,
            build_judge_llm_callable,
        )

        clean_environment = {
            key: value
            for key, value in os.environ.items()
            if key not in {"PAPERSTORM_CHAT_LLM", "PAPERSTORM_JUDGE_LLM"}
        }
        clean_environment["DEEPSEEK_API_KEY"] = "test-key-must-not-be-used"
        with mock.patch.dict(os.environ, clean_environment, clear=True):
            self.assertIsNone(build_chat_llm_callable())
            self.assertIsNone(build_judge_llm_callable())

    def test_service_dependencies_are_declared(self):
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        for package in ("fastapi", "uvicorn", "httpx"):
            self.assertRegex(requirements, rf"(?m)^{package}(?:[<>=].*)?$")

    def test_package_versions_match_v52(self):
        setup_text = (ROOT / "setup.py").read_text(encoding="utf-8")
        init_text = (ROOT / "knowledge_storm" / "__init__.py").read_text(
            encoding="utf-8"
        )
        setup_version = re.search(r'version="([^"]+)"', setup_text).group(1)
        init_version = re.search(r'__version__ = "([^"]+)"', init_text).group(1)
        self.assertEqual(setup_version, "5.2.0")
        self.assertEqual(init_version, setup_version)
        self.assertIn("PaperStorm Agent", setup_text)

    def test_ci_runs_offline_unit_tests(self):
        workflow = (ROOT / ".github" / "workflows" / "test.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("unittest discover", workflow)
        self.assertIn("PAPERSTORM_CHAT_LLM: 0", workflow)
        self.assertIn("PAPERSTORM_JUDGE_LLM: 0", workflow)
        self.assertIn("PAPERSTORM_RETRIEVAL_EMBEDDING: hash", workflow)


if __name__ == "__main__":
    unittest.main()
