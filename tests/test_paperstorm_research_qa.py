import tempfile
import unittest
from pathlib import Path


class PaperStormResearchQATest(unittest.TestCase):
    def make_service(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        from knowledge_storm.paperstorm_service import PaperStormTaskService

        return PaperStormTaskService(root_dir=Path(temp_dir.name))

    def test_ask_without_task_id_runs_research_then_answers(self):
        service = self.make_service()

        answer = service.ask_research_agent(
            question="PIM 是什么，神经网络如何抑制它？",
            topic="pim 神经网络抑制",
            run_mode="fake",
            expected_keywords=["passive intermodulation", "RF"],
            forbidden_keywords=["DRAM", "RAM", "processing-in-memory"],
        )

        self.assertTrue(answer["retrieval_triggered"])
        self.assertEqual(answer["decision"]["action"], "retrieve_then_answer")
        self.assertEqual(answer["task_status"], "succeeded")
        self.assertTrue(answer["used_task_id"])
        self.assertTrue(answer["grounded"])
        self.assertTrue(answer["citations"])
        self.assertIn("passive intermodulation", answer["answer"])
        self.assertTrue(answer["trace"])

    def test_ask_with_finished_task_reuses_existing_knowledge_base(self):
        service = self.make_service()
        task = service.submit_research_task(
            topic="pim 神经网络抑制",
            run_mode="fake",
            expected_keywords=["passive intermodulation"],
            forbidden_keywords=["DRAM"],
        )
        service.run_task(task["task_id"])
        before_count = len(service.list_tasks())

        answer = service.ask_research_agent(
            question="这次调研里 PIM 指什么？",
            task_id=task["task_id"],
        )

        self.assertFalse(answer["retrieval_triggered"])
        self.assertEqual(answer["decision"]["action"], "answer_from_existing_kb")
        self.assertEqual(answer["used_task_id"], task["task_id"])
        self.assertEqual(len(service.list_tasks()), before_count)
        self.assertTrue(answer["grounded"])
        self.assertTrue(answer["citations"])
        self.assertEqual(answer["decision"]["action"], "answer_from_existing_kb")
        self.assertTrue(answer["evidence_sufficiency"]["sufficient"])
        self.assertGreaterEqual(answer["evidence_sufficiency"]["score"], 60)

    def test_ask_with_finished_task_rejects_low_confidence_unrelated_question(self):
        service = self.make_service()
        task = service.submit_research_task(
            topic="pim 神经网络抑制",
            run_mode="fake",
            expected_keywords=["passive intermodulation"],
            forbidden_keywords=["DRAM"],
        )
        service.run_task(task["task_id"])
        before_count = len(service.list_tasks())

        answer = service.ask_research_agent(
            question="Transformer 注意力机制和大语言模型训练有什么关系？",
            task_id=task["task_id"],
        )

        self.assertFalse(answer["retrieval_triggered"])
        self.assertEqual(answer["decision"]["action"], "reject_low_confidence")
        self.assertFalse(answer["grounded"])
        self.assertFalse(answer["evidence_sufficiency"]["sufficient"])
        self.assertEqual(len(service.list_tasks()), before_count)

    def test_evidence_sufficiency_records_forbidden_keyword_hits(self):
        service = self.make_service()
        task = service.submit_research_task(
            topic="pim 神经网络抑制",
            run_mode="fake",
            expected_keywords=["passive intermodulation"],
            forbidden_keywords=["DRAM", "processing-in-memory"],
        )
        service.run_task(task["task_id"])

        answer = service.ask_research_agent(
            question="这里为什么不能把 PIM 理解成 DRAM processing-in-memory？",
            task_id=task["task_id"],
        )

        self.assertIn("DRAM", answer["evidence_sufficiency"]["forbidden_keyword_hits"])
        self.assertIn(
            "processing-in-memory",
            answer["evidence_sufficiency"]["forbidden_keyword_hits"],
        )
        self.assertEqual(answer["decision"]["action"], "answer_from_existing_kb")

    def test_research_qa_persists_qa_history_for_followup(self):
        service = self.make_service()
        first = service.ask_research_agent(
            question="PIM 是什么？",
            topic="pim 神经网络抑制",
            run_mode="fake",
            expected_keywords=["passive intermodulation"],
            forbidden_keywords=["DRAM"],
        )
        second = service.ask_research_agent(
            question="那神经网络如何抑制它？",
            task_id=first["used_task_id"],
        )
        state = service.get_task(first["used_task_id"])
        history_path = Path(state["output_dir"]) / "qa_history.json"

        self.assertTrue(history_path.exists())
        self.assertEqual(second["qa_history_count"], 2)
        self.assertTrue(second["qa_history"])
        self.assertEqual(second["qa_history"][-1]["question"], "那神经网络如何抑制它？")

    def test_fastapi_adapter_exposes_research_agent_ask(self):
        from fastapi.testclient import TestClient

        from examples.storm_examples.paperstorm_service_api import create_app

        with tempfile.TemporaryDirectory() as temp_dir:
            app = create_app(service_root=Path(temp_dir))
            client = TestClient(app)

            response = client.post(
                "/research-agent/ask",
                json={
                    "question": "PIM 是什么？",
                    "topic": "pim 神经网络抑制",
                    "run_mode": "fake",
                    "expected_keywords": ["passive intermodulation"],
                    "forbidden_keywords": ["DRAM"],
                },
            )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertTrue(payload["retrieval_triggered"])
            self.assertEqual(payload["decision"]["action"], "retrieve_then_answer")
            self.assertTrue(payload["citations"])


if __name__ == "__main__":
    unittest.main()
