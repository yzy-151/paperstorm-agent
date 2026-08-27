import json
import tempfile
import unittest
from pathlib import Path


class QasperAnswerValidationMetricsTest(unittest.TestCase):
    def test_metrics_report_citation_claim_and_abstention_quality(self):
        from knowledge_storm.evaluation.public_benchmarks.base import BenchmarkCase
        from knowledge_storm.evaluation.public_benchmarks.qasper_generation import official_qasper_metrics

        cases = (
            BenchmarkCase(
                "supported", "What works?", ("e1",), "test",
                answers=("method",), evidence_ids=("e1",),
                metadata={"qasper_references": ({
                    "answer": "method", "answer_type": "extractive",
                    "evidence_ids": ("e1",), "evidence_texts": ("gold",),
                },)},
            ),
            BenchmarkCase(
                "unanswerable", "Unknown?", (), "test", unanswerable=True,
                answers=("Unanswerable",),
                metadata={"qasper_references": ({
                    "answer": "Unanswerable", "answer_type": "none",
                    "evidence_ids": (), "evidence_texts": (),
                },)},
            ),
        )
        predictions = {
            "supported": {
                "status": "succeeded", "answer": "method", "abstained": False,
                "evidence_ids": ["e1", "noise"],
                "claim_validation": {"assessments": [
                    {"verdict": "entailed"}, {"verdict": "unsupported"}
                ]},
            },
            "unanswerable": {
                "status": "succeeded", "answer": "Unanswerable", "abstained": True,
                "evidence_ids": [], "claim_validation": {"assessments": []},
            },
        }

        metrics = official_qasper_metrics(cases, predictions)

        self.assertEqual(0.75, metrics["citation_precision"])
        self.assertEqual(1.0, metrics["citation_recall"])
        self.assertEqual(0.5, metrics["claim_support_rate"])
        self.assertEqual(0.5, metrics["unsupported_claim_rate"])
        self.assertEqual(1.0, metrics["abstention_precision"])
        self.assertEqual(1.0, metrics["abstention_recall"])
        self.assertEqual(1.0, metrics["answer_f1_by_shape"]["single"])
        self.assertEqual(0.0, metrics["answer_f1_by_shape"]["list"])
        self.assertEqual(0.0, metrics["answer_f1_by_shape"]["comparison"])

    def test_failed_predictions_count_as_missing_not_supported_claims(self):
        from knowledge_storm.evaluation.public_benchmarks.base import BenchmarkCase
        from knowledge_storm.evaluation.public_benchmarks.qasper_generation import official_qasper_metrics

        case = BenchmarkCase("failed", "q", ("e1",), "test", answers=("a",), evidence_ids=("e1",))
        metrics = official_qasper_metrics((case,), {"failed": {"status": "failed"}})

        self.assertEqual(1, metrics["missing_predictions"])
        self.assertEqual(0.0, metrics["claim_support_rate"])
        self.assertEqual(0, metrics["validated_claim_count"])


class QasperGroundedGenerationTest(unittest.TestCase):
    def test_runner_retries_schema_invalid_draft_once(self):
        from knowledge_storm.evaluation.public_benchmarks.base import (
            BenchmarkCase, BenchmarkDataset, BenchmarkDocument,
        )
        from knowledge_storm.evaluation.public_benchmarks.qasper_generation import (
            run_qasper_generation,
        )

        document = BenchmarkDocument("e1", "Paper", "supported answer")
        dataset = BenchmarkDataset(
            "qasper", "fixture", (document,),
            (BenchmarkCase("q1", "Question?", ("e1",), "test"),),
        )
        invalid = {
            "answer": "supported answer", "answer_type": "span",
            "claims": [{"claim_id": 1, "text": "supported answer", "citation_ids": ["e1"]}],
            "uncertainty": 0.0, "refusal": False, "abstain_reason": "",
        }
        valid = {
            "answer": "supported answer", "answer_type": "extractive",
            "claims": [{"claim_id": "c1", "text": "supported answer", "citation_ids": ["e1"]}],
            "uncertainty": 0.0, "refusal": False, "abstain_reason": None,
        }
        outputs = iter((invalid, valid))
        prompts = []

        def generate(prompt):
            prompts.append(prompt)
            return {"text": json.dumps(next(outputs)), "usage": {"total_tokens": 5}}

        with tempfile.TemporaryDirectory() as temp_dir:
            run_qasper_generation(
                dataset, {"q1": ["e1"]}, generate, temp_dir, "fake/model",
                claim_verifier=lambda _q, _d, _e: {
                    "assessments": [{"claim_id": "c1", "verdict": "entailed", "rationale": "direct"}]
                },
                parse_attempts=2,
            )
            row = json.loads((Path(temp_dir) / "predictions.jsonl").read_text(encoding="utf-8"))

        self.assertEqual(row["status"], "succeeded")
        self.assertEqual(row["generation_attempts"], 2)
        self.assertEqual(row["usage"]["total_tokens"], 10)
        self.assertIn("previous response violated", prompts[1])
        self.assertIn("must be null when refusal is false", prompts[0])
        self.assertIn("Never return an empty string", prompts[1])

    def test_batch_verifier_builds_one_grounded_request(self):
        from knowledge_storm.answer_validation import AnswerDraft, Citation, Claim
        from knowledge_storm.evaluation.public_benchmarks.base import BenchmarkDocument
        from knowledge_storm.evaluation.public_benchmarks.qasper_generation import (
            LiteLLMClaimVerifier,
        )

        calls = []
        citation = Citation("e1", "paper", "supporting span", "Paper", ())
        draft = AnswerDraft(
            answer="claim",
            claims=(Claim("c1", "claim", (citation,)),),
            answer_type="abstractive",
        )

        def generate(prompt):
            calls.append(prompt)
            return {
                "text": json.dumps({"assessments": [{
                    "claim_id": "c1", "verdict": "entailed", "rationale": "direct"
                }]}),
                "usage": {"total_tokens": 8},
            }

        result = LiteLLMClaimVerifier(generate)(
            "question", draft, (BenchmarkDocument("e1", "Paper", "supporting span"),)
        )

        self.assertEqual(len(calls), 1)
        self.assertIn("c1", calls[0])
        self.assertIn("supporting span", calls[0])
        self.assertEqual(result["usage"]["total_tokens"], 8)

    def test_runner_rehydrates_citations_and_filters_unsupported_claim(self):
        from knowledge_storm.evaluation.public_benchmarks.base import (
            BenchmarkCase,
            BenchmarkDataset,
            BenchmarkDocument,
        )
        from knowledge_storm.evaluation.public_benchmarks.qasper_generation import (
            run_qasper_generation,
        )

        document = BenchmarkDocument(
            "paper::p1",
            "Trusted paper title",
            "The method uses a transformer encoder.",
            {"paper_id": "paper", "authors": ["A. Author"], "section": "Method"},
        )
        case = BenchmarkCase(
            "q1",
            "What encoder is used?",
            (document.document_id,),
            "test",
            metadata={
                "paper_id": "paper",
                "qasper_references": ({
                    "answer": "transformer encoder",
                    "answer_type": "extractive",
                    "evidence_ids": (document.document_id,),
                },),
            },
        )
        dataset = BenchmarkDataset("qasper", "fixture", (document,), (case,))
        generated = {
            "answer": "A transformer encoder is used. It improves accuracy.",
            "answer_type": "abstractive",
            "claims": [
                {
                    "claim_id": "c1",
                    "text": "A transformer encoder is used.",
                    "citation_ids": [document.document_id],
                },
                {
                    "claim_id": "c2",
                    "text": "It improves accuracy.",
                    "citation_ids": [document.document_id],
                },
            ],
            "uncertainty": 0.1,
            "refusal": False,
            "abstain_reason": None,
        }

        def verify_claims(_question, draft, _documents):
            self.assertEqual(
                draft.claims[0].citations[0].title, "Trusted paper title"
            )
            return {
                "assessments": [
                    {"claim_id": "c1", "verdict": "entailed", "rationale": "supported"},
                    {"claim_id": "c2", "verdict": "unsupported", "rationale": "missing"},
                ],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            report = run_qasper_generation(
                dataset,
                rankings={case.case_id: [document.document_id]},
                generate=lambda _prompt: {
                    "text": json.dumps(generated),
                    "usage": {"prompt_tokens": 11, "completion_tokens": 9, "total_tokens": 20},
                },
                output_dir=temp_dir,
                model_name="fake/model",
                claim_verifier=verify_claims,
            )
            row = json.loads(
                (Path(temp_dir) / "predictions.jsonl").read_text(encoding="utf-8")
            )

        self.assertEqual(row["answer"], "A transformer encoder is used.")
        self.assertEqual(row["evidence_ids"], [document.document_id])
        self.assertEqual(row["claim_validation"]["deleted_claim_ids"], ["c2"])
        self.assertEqual(row["claim_validation"]["assessments"][1]["verdict"], "unsupported")
        self.assertEqual(row["usage"]["total_tokens"], 30)
        self.assertEqual(report["metrics"]["claim_support_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
