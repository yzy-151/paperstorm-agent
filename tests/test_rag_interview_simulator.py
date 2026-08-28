import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch


MODULE_NAME = "rag_interview_simulator_under_test"
MODULE_PATH = Path(__file__).parents[1] / "knowledge_storm" / "rag_interview_simulator.py"
CLI_PATH = Path(__file__).parents[1] / "examples" / "storm_examples" / "run_rag_interview_simulator.py"


def load_simulator_module():
    if not MODULE_PATH.exists():
        return None
    spec = spec_from_file_location(MODULE_NAME, MODULE_PATH)
    module = module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def load_cli_module():
    if not CLI_PATH.exists():
        return None
    spec = spec_from_file_location("rag_interview_simulator_cli_under_test", CLI_PATH)
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RagInterviewSimulatorTest(unittest.TestCase):
    def test_custom_questions_reject_missing_required_categories(self):
        simulator_module = load_simulator_module()
        self.assertIsNotNone(simulator_module)

        questions = tuple(
            question
            for question in simulator_module.DEFAULT_QUESTIONS
            if question.category != "Langfuse"
        )

        with self.assertRaisesRegex(
            ValueError, "missing required categories: Langfuse"
        ):
            simulator_module.RagInterviewSimulator(
                project_context="PaperStorm interview", questions=questions
            )

    def test_custom_questions_allow_multiple_questions_for_one_category(self):
        simulator_module = load_simulator_module()
        self.assertIsNotNone(simulator_module)
        duplicate_rag_question = simulator_module.InterviewQuestion(
            identifier="rag-evaluation",
            category="RAG",
            prompt="How would you evaluate retrieval quality?",
            reference_answer="Discuss retrieval metrics and failure analysis.",
        )

        session = simulator_module.RagInterviewSimulator(
            project_context="PaperStorm interview",
            questions=simulator_module.DEFAULT_QUESTIONS + (duplicate_rag_question,),
        ).run(rounds=len(simulator_module.REQUIRED_CATEGORIES))

        self.assertEqual(
            session.covered_categories, set(simulator_module.REQUIRED_CATEGORIES)
        )

    def test_deterministic_session_covers_every_required_category(self):
        simulator_module = load_simulator_module()
        self.assertIsNotNone(simulator_module)

        session = simulator_module.RagInterviewSimulator(
            project_context="PaperStorm is a research RAG agent."
        ).run(rounds=len(simulator_module.REQUIRED_CATEGORIES))

        self.assertEqual(
            [turn.question.category for turn in session.turns],
            list(simulator_module.REQUIRED_CATEGORIES),
        )
        self.assertEqual(
            session.covered_categories, set(simulator_module.REQUIRED_CATEGORIES)
        )

    def test_interviewer_followup_uses_the_previous_answer_after_coverage(self):
        simulator_module = load_simulator_module()
        self.assertIsNotNone(simulator_module)

        simulator = simulator_module.RagInterviewSimulator(
            project_context="PaperStorm interview"
        )
        simulator.run(rounds=len(simulator_module.REQUIRED_CATEGORIES))
        previous_answer = simulator.session.turns[-1].answer

        followup = simulator.run_round()

        self.assertTrue(followup.is_follow_up)
        self.assertIn(previous_answer, followup.question.prompt)
        self.assertEqual(
            followup.follow_up_to, len(simulator_module.REQUIRED_CATEGORIES)
        )

    def test_candidate_prompt_never_contains_reference_answer(self):
        simulator_module = load_simulator_module()
        self.assertIsNotNone(simulator_module)

        prompts = []

        def llm(prompt):
            prompts.append(prompt)
            if '"role": "interviewer"' in prompt:
                return '{"question": "What would you measure first?"}'
            return '{"answer": "I would validate retrieved evidence before use."}'

        secret_question = simulator_module.InterviewQuestion(
            identifier="secret",
            category="RAG",
            prompt="How would you validate retrieval?",
            reference_answer="REFERENCE_ANSWER_MUST_REMAIN_PRIVATE",
        )
        questions = (secret_question,) + tuple(
            question
            for question in simulator_module.DEFAULT_QUESTIONS
            if question.category != "RAG"
        )
        simulator = simulator_module.RagInterviewSimulator(
            project_context="A public project context.",
            questions=questions,
            llm=llm,
            mode="llm",
        )

        simulator.run(rounds=1)

        candidate_prompts = [item for item in prompts if '"role": "candidate"' in item]
        self.assertEqual(len(candidate_prompts), 1)
        self.assertNotIn("REFERENCE_ANSWER_MUST_REMAIN_PRIVATE", candidate_prompts[0])
        self.assertIn("What would you measure first?", candidate_prompts[0])
        self.assertIn("A public project context.", candidate_prompts[0])

    def test_injected_llm_drives_both_roles_with_litellm_response_shape(self):
        simulator_module = load_simulator_module()
        self.assertIsNotNone(simulator_module)
        roles = []

        class Message:
            def __init__(self, content):
                self.content = content

        class Choice:
            def __init__(self, content):
                self.message = Message(content)

        class Completion:
            def __init__(self, content):
                self.choices = [Choice(content)]

        def llm(prompt):
            role = "interviewer" if '"role": "interviewer"' in prompt else "candidate"
            roles.append(role)
            if role == "interviewer":
                return '{"question": "Which retrieval metric would you inspect first?"}'
            return Completion('{"answer": "I would inspect recall and citation validity."}')

        turn = simulator_module.RagInterviewSimulator(
            project_context="PaperStorm interview",
            llm=llm,
            mode="llm",
        ).run(rounds=1).turns[0]

        self.assertEqual(roles, ["interviewer", "candidate"])
        self.assertEqual(turn.question.category, "RAG")
        self.assertEqual(
            turn.question.prompt, "Which retrieval metric would you inspect first?"
        )
        self.assertEqual(turn.answer, "I would inspect recall and citation validity.")

    def test_invalid_llm_json_raises_a_typed_candidate_error(self):
        simulator_module = load_simulator_module()
        self.assertIsNotNone(simulator_module)

        simulator = simulator_module.RagInterviewSimulator(
            project_context="PaperStorm interview",
            llm=lambda prompt: (
                '{"question": "What would you validate?"}'
                if '"role": "interviewer"' in prompt
                else "not valid json"
            ),
            mode="llm",
        )

        with self.assertRaises(simulator_module.InterviewResponseError) as captured:
            simulator.run(rounds=1)

        self.assertEqual(captured.exception.role, "candidate")
        self.assertEqual(captured.exception.error_type, "invalid_structured_output")
        self.assertIn("not valid json", str(captured.exception))

    def test_interviewer_uses_fallback_question_after_invalid_llm_json(self):
        simulator_module = load_simulator_module()
        self.assertIsNotNone(simulator_module)

        def llm(prompt):
            if '"role": "interviewer"' in prompt:
                return "not valid json"
            return '{"answer": "I would inspect the retrieved evidence."}'

        turn = simulator_module.RagInterviewSimulator(
            project_context="PaperStorm interview",
            llm=llm,
            mode="llm",
            fallback_on_parse_error=True,
        ).run(rounds=1).turns[0]

        self.assertEqual(turn.question.prompt, simulator_module.DEFAULT_QUESTIONS[0].prompt)
        self.assertEqual(turn.answer, "I would inspect the retrieved evidence.")

    def test_candidate_uses_deterministic_fallback_after_invalid_llm_json(self):
        simulator_module = load_simulator_module()
        self.assertIsNotNone(simulator_module)

        def llm(prompt):
            if '"role": "interviewer"' in prompt:
                return '{"question": "Which retrieval metric would you inspect first?"}'
            return "not valid json"

        turn = simulator_module.RagInterviewSimulator(
            project_context="PaperStorm interview",
            llm=llm,
            mode="llm",
            fallback_on_parse_error=True,
        ).run(rounds=1).turns[0]

        self.assertEqual(
            turn.question.prompt, "Which retrieval metric would you inspect first?"
        )
        self.assertIn("RAG", turn.answer)

    def test_mapping_content_does_not_evaluate_choices_fallback(self):
        simulator_module = load_simulator_module()
        self.assertIsNotNone(simulator_module)

        class ContentOnlyMapping(dict):
            def get(self, key, default=None):
                if key == "choices":
                    raise AssertionError("choices fallback should not be evaluated")
                return super().get(key, default)

        response = simulator_module._parse_role_response(
            ContentOnlyMapping(content='{"answer": "Use the provided content."}'),
            "candidate",
            "answer",
        )

        self.assertEqual(response, {"answer": "Use the provided content."})

    def test_cli_writes_structured_markdown_in_deterministic_mode(self):
        cli_module = load_cli_module()
        self.assertIsNotNone(cli_module)
        previous_module = sys.modules.get(cli_module.MODULE_NAME)
        sentinel = object()
        sys.modules[cli_module.MODULE_NAME] = sentinel
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "interview.md"
            try:
                result = cli_module.main(
                    [
                        "--mode",
                        "deterministic",
                        "--rounds",
                        "6",
                        "--output",
                        str(output_path),
                        "--model",
                        "test-model",
                    ]
                )

                report = output_path.read_text(encoding="utf-8")
                self.assertIs(sys.modules[cli_module.MODULE_NAME], sentinel)
            finally:
                if previous_module is None:
                    sys.modules.pop(cli_module.MODULE_NAME, None)
                else:
                    sys.modules[cli_module.MODULE_NAME] = previous_module
        self.assertEqual(result, 0)
        self.assertIn("# RAG Agent 双角色面试模拟", report)
        self.assertIn("## 类别覆盖", report)
        self.assertIn("Langfuse", report)

    def test_cli_restores_modules_after_dynamic_load_failure(self):
        cli_module = load_cli_module()
        self.assertIsNotNone(cli_module)
        previous_module = sys.modules.get(cli_module.MODULE_NAME)
        sentinel = object()
        sys.modules[cli_module.MODULE_NAME] = sentinel
        with tempfile.TemporaryDirectory() as directory:
            broken_module_path = Path(directory) / "broken_simulator.py"
            broken_module_path.write_text("raise RuntimeError('load failure')\n", encoding="utf-8")

            try:
                with patch.object(cli_module, "MODULE_PATH", broken_module_path):
                    with self.assertRaisesRegex(RuntimeError, "load failure"):
                        cli_module._load_simulator_module()
                self.assertIs(sys.modules[cli_module.MODULE_NAME], sentinel)
            finally:
                if previous_module is None:
                    sys.modules.pop(cli_module.MODULE_NAME, None)
                else:
                    sys.modules[cli_module.MODULE_NAME] = previous_module


if __name__ == "__main__":
    unittest.main()
