import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "docs" / "benchmarks" / "langfuse_observed_badcases_v2.json"


class LangfuseObservedBadcaseRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    def test_dataset_has_unique_stable_case_ids(self):
        case_ids = [case["case_id"] for case in self.dataset["cases"]]
        self.assertEqual(len(case_ids), len(set(case_ids)))
        self.assertGreaterEqual(len(case_ids), 7)

    def test_router_badcases_obey_expected_action_contracts(self):
        from knowledge_storm.paperstorm_intent_router import PaperStormIntentRouter

        route_cases = [case for case in self.dataset["cases"] if case["kind"] == "router"]
        for case in route_cases:
            with self.subTest(case_id=case["case_id"]):
                router = PaperStormIntentRouter(
                    llm_router=lambda _prompt, output=case["planner_output"]: output
                )
                decision = router.route(
                    message=case["message"],
                    session=case.get("session") or {},
                    context_window=case.get("context_window") or [],
                )
                expected = case["expected"]
                for key in ("action", "need_retrieval", "tool"):
                    self.assertEqual(expected[key], decision[key], (case["case_id"], decision))
                if expected.get("first_tool_call"):
                    self.assertEqual(
                        expected["first_tool_call"],
                        decision["tool_calls"][0]["name"],
                        (case["case_id"], decision),
                    )

    def test_explicit_memory_write_is_verified_and_recalled_cross_session(self):
        from knowledge_storm.paperstorm_service import PaperStormTaskService

        case = next(case for case in self.dataset["cases"] if case["kind"] == "memory_sequence")
        with tempfile.TemporaryDirectory() as temp_dir:
            service = PaperStormTaskService(Path(temp_dir))
            first = service.create_chat_session(run_mode="fake", user_id=case["user_id"])
            written = service.send_chat_message(first["chat_id"], case["write_message"])
            second = service.create_chat_session(run_mode="fake", user_id=case["user_id"])
            recalled = service.send_chat_message(second["chat_id"], case["recall_message"])

        expected = case["expected"]
        self.assertEqual(expected["write_status"], written["memory_write"]["status"])
        self.assertEqual(
            expected["read_after_write_verified"],
            written["memory_write"]["read_after_write"]["verified"],
        )
        self.assertTrue(
            any(
                expected["recall_contains"] in item["content"]
                for item in recalled["long_term_memory"]["results"]
            )
        )
        self.assertFalse(recalled["retrieval_triggered"])

    def test_dataset_sync_uses_stable_ids_and_source_traces(self):
        from knowledge_storm.evaluation.langfuse_badcases import sync_langfuse_dataset

        class FakeLangfuseClient:
            def __init__(self):
                self.datasets = []
                self.items = []
                self.flush_count = 0

            def get_dataset(self, _name):
                raise RuntimeError("not found")

            def create_dataset(self, **payload):
                self.datasets.append(payload)
                return payload

            def create_dataset_item(self, **payload):
                self.items.append(payload)
                return payload

            def flush(self):
                self.flush_count += 1

        client = FakeLangfuseClient()
        first = sync_langfuse_dataset(DATASET_PATH, client=client)
        first_ids = [item["id"] for item in client.items]
        client.items.clear()
        second = sync_langfuse_dataset(DATASET_PATH, client=client)
        second_ids = [item["id"] for item in client.items]

        self.assertEqual(first["dataset_name"], self.dataset["dataset_name"])
        self.assertEqual(first["item_count"], len(self.dataset["cases"]))
        self.assertEqual(first_ids, second_ids)
        self.assertEqual(len(first_ids), len(set(first_ids)))
        self.assertEqual(second["item_count"], len(self.dataset["cases"]))
        sourced = [item for item in client.items if item.get("source_trace_id")]
        self.assertGreaterEqual(len(sourced), 5)
        self.assertEqual(client.flush_count, 2)

    def test_local_dataset_runner_reports_every_badcase_passed(self):
        from knowledge_storm.evaluation.langfuse_badcases import run_badcase_regression

        with tempfile.TemporaryDirectory() as temp_dir:
            report = run_badcase_regression(DATASET_PATH, output_dir=Path(temp_dir))
            persisted = json.loads(
                (Path(temp_dir) / "langfuse_badcase_regression.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(report["passed"], report["total"])
        self.assertEqual(report["failed"], 0)
        self.assertEqual(persisted["dataset_name"], self.dataset["dataset_name"])

    def test_v2_dataset_covers_multilingual_evidence_gate(self):
        from knowledge_storm.evaluation.langfuse_badcases import _run_evidence_gate_case

        payload = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
        case = next(
            item
            for item in payload["cases"]
            if item["case_id"] == "multilingual-followup-reuses-relevant-evidence"
        )
        self.assertEqual(case["kind"], "evidence_gate")

        actual = _run_evidence_gate_case(case)
        self.assertEqual(case["expected"], actual)


if __name__ == "__main__":
    unittest.main()
