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
        self.assertRegex(requirements, r"(?m)^huggingface-hub>=0\.34,<1\.0$")
        self.assertRegex(requirements, r"(?m)^sentence-transformers>=3\.4,<6\.0$")

    def test_public_tests_do_not_depend_on_gitignored_private_docs(self):
        private_docs = {"VERSION_PLAN.md", "RESUME_INTERVIEW_PLAN.md", "OPERATION_GUIDE.md"}
        for path in (ROOT / "tests").glob("test_*.py"):
            if path.resolve() == Path(__file__).resolve():
                continue
            content = path.read_text(encoding="utf-8")
            for filename in private_docs:
                self.assertNotIn(filename, content, str(path))

    def test_setup_declares_only_verified_python_versions(self):
        setup_text = (ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertIn('python_requires=">=3.10,<3.12"', setup_text)

    def test_package_versions_match_current_release(self):
        setup_text = (ROOT / "setup.py").read_text(encoding="utf-8")
        init_text = (ROOT / "knowledge_storm" / "__init__.py").read_text(
            encoding="utf-8"
        )
        setup_version = re.search(r'version="([^"]+)"', setup_text).group(1)
        init_version = re.search(r'__version__ = "([^"]+)"', init_text).group(1)
        self.assertEqual(setup_version, "7.2.0")
        self.assertEqual(init_version, setup_version)
        self.assertIn("PaperStorm Agent", setup_text)

    def test_readme_documents_v71_observability_and_interview_workflows(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        for marker in (
            "v7.2",
            "run_langfuse_badcase_demo.py",
            "LANGFUSE_BADCASE_GUIDE.md",
            "tags",
            "scores",
            "case_id",
            "events.jsonl",
            "run_rag_interview_simulator.py",
            "--mode deterministic",
            "--mode llm",
            "PAPERSTORM_RESUME_GUIDE.md",
            "RAG_AGENT_INTERVIEW_PLAYBOOK.md",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, readme)

        self.assertNotIn("v7.0", readme)
        self.assertNotIn("paperstorm-executive-overview", readme)
        self.assertNotIn("消除 `Muon optimizer` 与粒子物理 muon 的语义歧义", readme)
        self.assertIn('$env:LANGFUSE_PUBLIC_KEY="<Langfuse public key>"', readme)
        self.assertIn('$env:LANGFUSE_SECRET_KEY="<Langfuse secret key>"', readme)

    def test_litellm_is_bounded_to_verified_release_line(self):
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertRegex(requirements, r"(?m)^litellm>=1\.80,<1\.81$")

    def test_v55_public_benchmark_artifacts_are_documented_and_sanitized(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        guide = (ROOT / "docs" / "PAPERSTORM_V55_PUBLIC_BENCHMARKS.md").read_text(
            encoding="utf-8"
        )
        summary = (
            ROOT / "docs" / "benchmarks" / "paperstorm_public_v55_summary.json"
        ).read_text(encoding="utf-8")

        self.assertIn("SciFact", readme)
        self.assertIn("QASPER", readme)
        self.assertIn("public_official", summary)
        self.assertIn("1309", summary)
        self.assertIn("证据检索", guide)
        for content in (guide, summary):
            self.assertNotRegex(content, r"[A-Za-z]:\\")

    def test_ci_runs_offline_unit_tests(self):
        workflow_path = ROOT / ".github" / "workflows" / "test.yml"
        if not workflow_path.exists():
            self.skipTest(
                "offline CI workflow (test.yml) is kept local: pushing it requires "
                "GitHub token with workflow scope"
            )
        workflow = workflow_path.read_text(encoding="utf-8")
        self.assertIn("unittest discover", workflow)
        self.assertIn("PAPERSTORM_CHAT_LLM: 0", workflow)
        self.assertIn("PAPERSTORM_JUDGE_LLM: 0", workflow)
        self.assertIn("PAPERSTORM_RETRIEVAL_EMBEDDING: hash", workflow)


if __name__ == "__main__":
    unittest.main()
