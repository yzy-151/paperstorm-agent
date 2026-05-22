import json
import tempfile
import unittest
from pathlib import Path


class PaperStormServiceTest(unittest.TestCase):
    def make_service(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        from knowledge_storm.paperstorm_service import PaperStormTaskService

        return PaperStormTaskService(root_dir=Path(temp_dir.name))

    def test_submit_research_task_creates_isolated_task_state(self):
        service = self.make_service()

        task = service.submit_research_task(
            topic="pim 神经网络抑制",
            retriever="arxiv",
            output_language="zh",
            run_mode="fake",
        )
        state = service.get_task(task["task_id"])

        self.assertEqual(state["status"], "queued")
        self.assertEqual(state["topic"], "pim 神经网络抑制")
        self.assertTrue(Path(state["output_dir"]).exists())
        self.assertIn(task["task_id"], state["output_dir"])

    def test_run_fake_task_writes_article_trace_summary_and_scorecard(self):
        service = self.make_service()
        task = service.submit_research_task(
            topic="pim 神经网络抑制",
            retriever="arxiv",
            output_language="zh",
            run_mode="fake",
            expected_keywords=["passive intermodulation", "RF"],
            forbidden_keywords=["processing-in-memory", "DRAM", "RAM"],
        )

        finished = service.run_task(task["task_id"])
        article = service.get_article(task["task_id"])
        scorecard = service.get_scorecard(task["task_id"])
        trace = service.get_trace(task["task_id"])

        self.assertEqual(finished["status"], "succeeded")
        self.assertIn("passive intermodulation", article["content"])
        self.assertGreater(scorecard["scores"]["total"], 50)
        self.assertTrue(trace["events"])

    def test_query_knowledge_base_answers_from_task_artifacts(self):
        service = self.make_service()
        task = service.submit_research_task(
            topic="pim 神经网络抑制",
            retriever="arxiv",
            output_language="zh",
            run_mode="fake",
        )
        service.run_task(task["task_id"])

        answer = service.query_knowledge_base(task["task_id"], "PIM 是什么？")

        self.assertTrue(answer["grounded"])
        self.assertTrue(answer["citations"])
        self.assertIn("passive intermodulation", answer["answer"])

    def test_two_tasks_write_to_separate_output_dirs(self):
        service = self.make_service()

        first = service.submit_research_task(topic="topic one", run_mode="fake")
        second = service.submit_research_task(topic="topic two", run_mode="fake")
        service.run_task(first["task_id"])
        service.run_task(second["task_id"])
        first_state = service.get_task(first["task_id"])
        second_state = service.get_task(second["task_id"])

        self.assertNotEqual(first_state["output_dir"], second_state["output_dir"])
        self.assertTrue((Path(first_state["output_dir"]) / "run_summary.json").exists())
        self.assertTrue((Path(second_state["output_dir"]) / "run_summary.json").exists())

    def test_task_failure_is_structured_and_secret_safe(self):
        service = self.make_service()
        task = service.submit_research_task(
            topic="broken",
            run_mode="fail",
            api_key="sk-secret-value",
        )

        failed = service.run_task(task["task_id"])
        serialized = json.dumps(failed, ensure_ascii=False)

        self.assertEqual(failed["status"], "failed")
        self.assertIn("error", failed)
        self.assertNotIn("sk-secret-value", serialized)

    def test_fastapi_adapter_imports_without_required_runtime_dependency(self):
        from examples.storm_examples import paperstorm_service_api

        self.assertTrue(hasattr(paperstorm_service_api, "create_app"))
        self.assertTrue(hasattr(paperstorm_service_api, "DEFAULT_SERVICE_ROOT"))


if __name__ == "__main__":
    unittest.main()
