import json
import tempfile
import unittest
from pathlib import Path


class RagReleaseGateTest(unittest.TestCase):
    def _baseline(self):
        return {
            "manifest": {"dataset_sha256": "same", "protocol_sha256": "p1"},
            "metrics": {"recall_at_5": 0.80},
            "p95_ms": 100.0,
            "unsupported_claim_rate": 0.02,
            "acl_leak_count": 0,
            "failure_rate": 0.01,
        }

    def test_gate_blocks_latency_regression_and_acl_leak(self):
        from knowledge_storm.paperstorm_benchmarks import ReleaseGate, ReleaseGatePolicy

        candidate = dict(self._baseline(), p95_ms=140.0, acl_leak_count=1)
        decision = ReleaseGate().evaluate(
            self._baseline(), candidate, ReleaseGatePolicy(max_p95_ratio=1.20)
        )

        self.assertFalse(decision.allowed)
        self.assertIn("p95_regression", decision.reasons)
        self.assertIn("acl_leak", decision.reasons)
        self.assertEqual(decision.checks["acl_leak"]["candidate"], 1)

    def test_gate_allows_bounded_regressions_and_records_checks(self):
        from knowledge_storm.paperstorm_benchmarks import ReleaseGate, ReleaseGatePolicy

        candidate = {
            **self._baseline(),
            "metrics": {"recall_at_5": 0.795},
            "p95_ms": 115.0,
            "unsupported_claim_rate": 0.025,
            "failure_rate": 0.015,
        }
        policy = ReleaseGatePolicy(
            quality_metrics=("recall_at_5",),
            max_quality_regression=0.01,
            max_p95_ratio=1.20,
            max_unsupported_claim_increase=0.01,
            max_failure_rate_increase=0.01,
        )

        decision = ReleaseGate().evaluate(self._baseline(), candidate, policy)

        self.assertTrue(decision.allowed)
        self.assertEqual(decision.reasons, ())
        self.assertEqual(decision.checks["recall_at_5"]["status"], "pass")

    def test_gate_refuses_incomparable_frozen_inputs(self):
        from knowledge_storm.paperstorm_benchmarks import ReleaseGate, ReleaseGatePolicy

        candidate = self._baseline()
        candidate["manifest"] = {"dataset_sha256": "different", "protocol_sha256": "p1"}
        decision = ReleaseGate().evaluate(
            self._baseline(), candidate, ReleaseGatePolicy()
        )

        self.assertFalse(decision.allowed)
        self.assertIn("manifest_mismatch", decision.reasons)

    def test_gate_uses_paired_bootstrap_interval_when_available(self):
        from knowledge_storm.paperstorm_benchmarks import ReleaseGate, ReleaseGatePolicy

        candidate = {
            **self._baseline(),
            "metrics": {"recall_at_5": 0.795},
            "paired_delta_ci": {"recall_at_5": [-0.03, -0.02]},
        }
        decision = ReleaseGate().evaluate(
            self._baseline(),
            candidate,
            ReleaseGatePolicy(max_quality_regression=0.01),
        )

        self.assertFalse(decision.allowed)
        self.assertIn("quality_ci_regression:recall_at_5", decision.reasons)
        self.assertEqual(decision.checks["recall_at_5"]["paired_delta_ci"], [-0.03, -0.02])

    def test_offline_replay_summarizes_frozen_predictions_without_network(self):
        from knowledge_storm.paperstorm_benchmarks import load_offline_replay

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "manifest.json").write_text(
                json.dumps({"dataset_sha256": "data", "protocol_sha256": "protocol"}),
                encoding="utf-8",
            )
            rows = [
                {"case_id": "ok", "status": "succeeded", "latency_ms": 10, "acl_leak": False,
                 "unsupported_claim_count": 0},
                {"case_id": "bad", "status": "failed", "latency_ms": 30, "acl_leak": True,
                 "unsupported_claim_count": 1, "validated_claim_count": 2},
            ]
            (root / "predictions.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
            )

            replay = load_offline_replay(root)

        self.assertEqual(replay["case_count"], 2)
        self.assertEqual(replay["failure_rate"], 0.5)
        self.assertEqual(replay["acl_leak_count"], 1)
        self.assertEqual(replay["unsupported_claim_rate"], 0.5)
        self.assertEqual(replay["p95_ms"], 30.0)


if __name__ == "__main__":
    unittest.main()
