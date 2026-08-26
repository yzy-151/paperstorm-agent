import tempfile
import time
import unittest
from pathlib import Path


class RetrievalResilienceTest(unittest.TestCase):
    def _control(self, root):
        from knowledge_storm.control_plane import ProductionControlPlane

        return ProductionControlPlane(Path(root) / "control.sqlite")

    def test_deadline_timeout_has_explicit_failure_type(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            control = self._control(temp_dir)
            started = time.perf_counter()
            result = control.execute_resilient(
                "slow-reranker",
                lambda: time.sleep(0.2),
                fallback=lambda _error: {"mode": "hybrid_without_rerank"},
                max_attempts=1,
                failure_threshold=1,
                timeout_seconds=0.01,
            )

        self.assertLess(time.perf_counter() - started, 0.15)
        self.assertTrue(result["degraded"])
        self.assertEqual(result["failure_type"], "timeout")
        self.assertEqual(result["circuit_state"], "open")

    def test_open_circuit_skips_provider_and_cooldown_uses_half_open_probe(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            control = self._control(temp_dir)
            calls = []

            def fail():
                calls.append("fail")
                raise ConnectionError("offline")

            control.execute_resilient(
                "provider", fail, fallback=lambda _e: None,
                max_attempts=1, failure_threshold=1, cooldown_seconds=60,
            )
            blocked = control.execute_resilient(
                "provider", fail, fallback=lambda _e: None,
                max_attempts=1, failure_threshold=1, cooldown_seconds=60,
            )
            recovered = control.execute_resilient(
                "provider", lambda: calls.append("probe") or "ok",
                fallback=lambda _e: None,
                max_attempts=1, failure_threshold=1, cooldown_seconds=0,
            )

        self.assertEqual(calls, ["fail", "probe"])
        self.assertEqual(blocked["failure_type"], "circuit_open")
        self.assertEqual(recovered["result"], "ok")
        self.assertEqual(recovered["circuit_state"], "closed")
        self.assertTrue(recovered["half_open_probe"])

    def test_batch_execution_preserves_input_order(self):
        from knowledge_storm.control_plane import execute_batch

        values = [3, 1, 2]

        def worker(value):
            time.sleep(value * 0.002)
            return value * 10

        self.assertEqual(execute_batch(values, worker, max_workers=3), [30, 10, 20])

    def test_sqlite_trace_redacts_secrets_identity_and_private_document(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            control = self._control(temp_dir)
            control.record_span(
                {
                    "trace_id": "trace",
                    "component": "retrieval",
                    "operation": "rerank",
                    "attributes": {
                        "api_key": "sk-secret",
                        "user_id": "alice@example.com",
                        "private_document": "PRIVATE" * 1000,
                    },
                }
            )
            attributes = control.list_spans("trace")[0]["attributes"]

        self.assertEqual(attributes["api_key"], "***REDACTED***")
        self.assertTrue(attributes["user_id"].startswith("user_"))
        self.assertNotIn("alice@example.com", str(attributes))
        self.assertLessEqual(len(attributes["private_document"]), 530)


if __name__ == "__main__":
    unittest.main()
