import json
import unittest
from pathlib import Path


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "evidence_governance_badcases.json"


def load_cases():
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


class EvidenceGovernanceTest(unittest.TestCase):
    def test_policy_disables_rerank_when_latency_budget_is_exceeded(self):
        from knowledge_storm.evidence_governance import RerankPolicy

        decision = RerankPolicy(model="test-cross-encoder").decide(
            {
                "answer_risk": 0.95,
                "bm25_dense_overlap": 0.10,
                "rrf_margin": 0.01,
                "candidate_count": 8,
                "cache_state": "miss",
                "observed_p95_ms": 901,
                "latency_budget_ms": 800,
            }
        )

        self.assertFalse(decision.enabled)
        self.assertEqual("latency_budget_exceeded_cache_miss", decision.reason)
        self.assertEqual(8, decision.candidate_count)
        self.assertEqual("test-cross-encoder", decision.model)
        self.assertEqual(800, decision.latency_budget_ms)

    def test_policy_reuses_cached_rerank_when_latency_budget_is_exceeded(self):
        from knowledge_storm.evidence_governance import RerankPolicy

        decision = RerankPolicy(model="test-cross-encoder").decide(
            {
                "answer_risk": 0.95,
                "bm25_dense_overlap": 0.10,
                "rrf_margin": 0.01,
                "candidate_count": 8,
                "cache_state": "rerank_hit",
                "observed_p95_ms": 901,
                "latency_budget_ms": 800,
            }
        )

        self.assertTrue(decision.enabled)
        self.assertEqual("latency_budget_exceeded_cached_rerank", decision.reason)

    def test_policy_selectively_enables_only_high_risk_uncertain_requests(self):
        from knowledge_storm.evidence_governance import RerankPolicy

        policy = RerankPolicy()
        enabled = policy.decide(
            {
                "answer_risk": 0.9,
                "bm25_dense_overlap": 0.2,
                "rrf_margin": 0.3,
                "candidate_count": 5,
                "cache_state": "miss",
                "observed_p95_ms": 100,
                "latency_budget_ms": 800,
            }
        )
        disabled = policy.decide(
            {
                "answer_risk": 0.2,
                "bm25_dense_overlap": 0.1,
                "rrf_margin": 0.01,
                "candidate_count": 5,
                "cache_state": "hit",
                "observed_p95_ms": 100,
                "latency_budget_ms": 800,
            }
        )

        self.assertTrue(enabled.enabled)
        self.assertFalse(disabled.enabled)
        self.assertEqual("risk_or_evidence_is_sufficient", disabled.reason)

    def test_mmr_selects_diverse_parents_and_reports_coverage(self):
        from knowledge_storm.evidence_governance import select_evidence

        selected = select_evidence(load_cases()["coverage_candidates"], top_k=3)

        self.assertEqual(3, len(selected))
        self.assertGreaterEqual(len({item["parent_id"] for item in selected}), 2)
        self.assertGreater(selected.coverage_score, 0.5)
        self.assertEqual(selected.coverage_score, selected[0]["coverage_score"])

    def test_assessor_presents_conflicting_evidence_without_deciding(self):
        from knowledge_storm.evidence_governance import EvidenceAssessor

        assessment = EvidenceAssessor().assess(
            "Does the treatment improve survival?",
            load_cases()["conflict_evidence"],
        )

        self.assertEqual("present_conflict", assessment.next_action)
        self.assertEqual("conflict", assessment.failure_type)
        self.assertEqual("contradicted", assessment.conflicts[0].relation)
        self.assertEqual(1, assessment.max_corrections)

    def test_assessor_keeps_different_conditions_as_qualified_evidence(self):
        from knowledge_storm.evidence_governance import EvidenceAssessor

        evidence = [
            {
                "chunk_id": "adults",
                "source": "study-a",
                "content": "The adult dose is 10 mg.",
                "score": 0.9,
                "claims": [
                    {
                        "claim_id": "recommended-dose",
                        "claim": "recommended dose",
                        "value": 10,
                        "conditions": {"population": "adults"},
                    }
                ],
            },
            {
                "chunk_id": "children",
                "source": "study-b",
                "content": "The child dose is 5 mg.",
                "score": 0.9,
                "claims": [
                    {
                        "claim_id": "recommended-dose",
                        "claim": "recommended dose",
                        "value": 5,
                        "conditions": {"population": "children"},
                    }
                ],
            },
        ]

        assessment = EvidenceAssessor().assess("What is the recommended dose?", evidence)

        self.assertEqual((), assessment.conflicts)
        self.assertEqual("answer", assessment.next_action)

    def test_assessor_abstains_when_there_is_no_evidence(self):
        from knowledge_storm.evidence_governance import EvidenceAssessor

        assessment = EvidenceAssessor().assess(
            "What dose cures the disease?", load_cases()["no_answer_evidence"]
        )

        self.assertEqual("abstain", assessment.next_action)
        self.assertEqual("no_evidence", assessment.failure_type)
        self.assertEqual(0.0, assessment.answerability)

    def test_assessor_limits_corrections_to_one_round(self):
        from knowledge_storm.evidence_governance import EvidenceAssessor

        assessment = EvidenceAssessor().assess(
            "What dose cures the disease?", [], correction_count=1
        )

        self.assertEqual(1, assessment.max_corrections)
        self.assertEqual("abstain", assessment.next_action)

    def test_pipeline_uses_governance_components_and_keeps_default_p1_unchanged(self):
        from knowledge_storm.evidence_governance import EvidenceAssessor, RerankPolicy
        from knowledge_storm.retrieval_pipeline import RetrievalPipeline, RetrievalRequest

        class Index:
            embedding_provider = type("Provider", (), {"name": "test"})()

            def search(self, query, **_kwargs):
                return [
                    {"chunk_id": "a", "parent_id": "parent-a", "source": "a", "content": "Treatment survival improves.", "score": 0.8},
                    {"chunk_id": "b", "parent_id": "parent-b", "source": "b", "content": "Treatment survival outcomes.", "score": 0.7},
                ]

        class Reranker:
            model_name = "test-cross-encoder"

            def __init__(self):
                self.calls = 0

            def rerank(self, _query, candidates, top_k=None):
                self.calls += 1
                return [dict(item, rerank_score=0.9 - index * 0.1) for index, item in enumerate(candidates[:top_k])]

        reranker = Reranker()
        baseline = RetrievalPipeline(Index()).search(RetrievalRequest(query="treatment survival"))
        governed = RetrievalPipeline(
            Index(),
            reranker=reranker,
            rerank_policy=RerankPolicy(),
            evidence_assessor=EvidenceAssessor(),
        ).search(
            RetrievalRequest(
                query="treatment survival",
                governance_features={
                    "answer_risk": 0.9,
                    "bm25_dense_overlap": 0.1,
                    "rrf_margin": 0.01,
                    "cache_state": "miss",
                    "observed_p95_ms": 10,
                    "latency_budget_ms": 800,
                },
            )
        )

        self.assertEqual(["plan", "retrieve", "fuse", "gate", "parent_expand"], [stage["name"] for stage in baseline["stages"]])
        self.assertNotIn("rerank_decision", baseline)
        self.assertTrue(governed["rerank_decision"]["enabled"])
        self.assertEqual(1, reranker.calls)
        self.assertIn("evidence_assessment", governed)
        self.assertEqual(["policy", "coverage", "assessment"], [stage["name"] for stage in governed["stages"] if stage["name"] in {"policy", "coverage", "assessment"}])

    def test_pipeline_preserves_policy_default_latency_budget(self):
        from knowledge_storm.evidence_governance import RerankPolicy
        from knowledge_storm.retrieval_pipeline import RetrievalPipeline, RetrievalRequest

        class Index:
            def search(self, _query, **_kwargs):
                return [
                    {"chunk_id": "a", "content": "a", "score": 0.8},
                    {"chunk_id": "b", "content": "b", "score": 0.7},
                ]

        result = RetrievalPipeline(
            Index(), rerank_policy=RerankPolicy(max_p95_ms=5)
        ).search(
            RetrievalRequest(
                query="budget",
                governance_features={
                    "answer_risk": 0.9,
                    "bm25_dense_overlap": 0.1,
                    "observed_p95_ms": 6,
                },
            )
        )

        self.assertEqual(
            "latency_budget_exceeded_cache_miss",
            result["rerank_decision"]["reason"],
        )


if __name__ == "__main__":
    unittest.main()
