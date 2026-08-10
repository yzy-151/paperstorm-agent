import unittest


class LongMemEvalAnswerV56Test(unittest.TestCase):
    def test_prompt_requires_careful_session_reading_and_abstention_only_as_fallback(self):
        from knowledge_storm.evaluation.public_benchmarks.longmemeval_answer import (
            build_longmemeval_prompt,
        )

        prompt = build_longmemeval_prompt("What degree did I graduate with?", "session text")
        self.assertIn("answer IS present", prompt)
        self.assertIn("Unanswerable", prompt)
        self.assertIn("What degree did I graduate with?", prompt)
        self.assertIn("session text", prompt)

    def test_evidence_slice_keeps_head_and_tail(self):
        from knowledge_storm.evaluation.public_benchmarks.longmemeval_answer import (
            _slice_evidence,
        )

        long_text = "A" * 9000
        sliced = _slice_evidence(long_text, 3000)
        self.assertIn("A" * 3000, sliced)
        self.assertIn("truncated 3000 chars", sliced)
        self.assertEqual(len(sliced), 3000 + 3000 + len("\n...[truncated 3000 chars]...\n"))

    def test_scoring_handles_extractive_boolean_and_abstention(self):
        from knowledge_storm.evaluation.public_benchmarks.base import BenchmarkCase
        from knowledge_storm.evaluation.public_benchmarks.longmemeval_answer import (
            _score_case,
        )

        extractive = BenchmarkCase(
            case_id="c1",
            query="q",
            answers=("Business Administration",),
            relevant_document_ids=(),
            split="test",
            unanswerable=False,
            metadata={"question_type": "single-session-user"},
        )
        score = _score_case(extractive, "Business Administration")
        self.assertTrue(score["em"])
        self.assertEqual(score["token_f1"], 1.0)

        boolean = BenchmarkCase(
            case_id="c2",
            query="q",
            answers=("Yes",),
            relevant_document_ids=(),
            split="test",
            unanswerable=False,
            metadata={"question_type": "boolean"},
        )
        self.assertTrue(_score_case(boolean, "Yes")["correct"])
        self.assertFalse(_score_case(boolean, "No")["correct"])

        abstain = BenchmarkCase(
            case_id="c3",
            query="q",
            answers=("",),
            relevant_document_ids=(),
            split="test",
            unanswerable=True,
            metadata={"question_type": "abstention"},
        )
        self.assertTrue(_score_case(abstain, "Unanswerable")["correct"])
        self.assertFalse(_score_case(abstain, "Target")["correct"])


if __name__ == "__main__":
    unittest.main()
