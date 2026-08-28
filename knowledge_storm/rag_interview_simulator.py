"""A deterministic or LLM-backed RAG interview simulator.

The candidate receives only public question data, project context, and prior
turns. Reference answers remain interviewer-side evaluation material.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple


REQUIRED_CATEGORIES = (
    "RAG",
    "Memory",
    "Context",
    "Runtime",
    "Langfuse",
    "项目复盘",
)


class InterviewResponseError(RuntimeError):
    """Raised when an LLM response cannot be used as structured interview data."""

    def __init__(self, role: str, error_type: str, raw_response: Any):
        summary = _response_summary(raw_response)
        super().__init__(
            "{0} LLM response {1}: {2}".format(role, error_type, summary)
        )
        self.role = role
        self.error_type = error_type
        self.raw_response = raw_response


@dataclass(frozen=True)
class InterviewQuestion:
    identifier: str
    category: str
    prompt: str
    reference_answer: str

    def __post_init__(self) -> None:
        for name in ("identifier", "category", "prompt", "reference_answer"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError("{0} must be a non-empty string".format(name))

    def public_view(self) -> Dict[str, str]:
        return {
            "identifier": self.identifier,
            "category": self.category,
            "prompt": self.prompt,
        }


@dataclass(frozen=True)
class InterviewTurn:
    number: int
    question: InterviewQuestion
    answer: str
    is_follow_up: bool = False
    follow_up_to: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.number, int) or self.number < 1:
            raise ValueError("number must be a positive integer")
        if not isinstance(self.answer, str) or not self.answer.strip():
            raise ValueError("answer must be a non-empty string")
        if self.is_follow_up != (self.follow_up_to is not None):
            raise ValueError("follow_up_to must be set exactly for follow-up turns")

    def public_view(self) -> Dict[str, Any]:
        return {
            "number": self.number,
            "question": self.question.public_view(),
            "answer": self.answer,
            "is_follow_up": self.is_follow_up,
            "follow_up_to": self.follow_up_to,
        }


@dataclass
class InterviewSession:
    project_context: str
    questions: Tuple[InterviewQuestion, ...]
    turns: List[InterviewTurn] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.project_context, str) or not self.project_context.strip():
            raise ValueError("project_context must be a non-empty string")
        if not self.questions:
            raise ValueError("questions must not be empty")

    @property
    def covered_categories(self) -> set:
        return {turn.question.category for turn in self.turns}

    def public_history(self) -> List[Dict[str, Any]]:
        return [turn.public_view() for turn in self.turns]

    def append(self, turn: InterviewTurn) -> None:
        if turn.number != len(self.turns) + 1:
            raise ValueError("turn number must be sequential")
        self.turns.append(turn)


DEFAULT_QUESTIONS = (
    InterviewQuestion(
        "rag-design",
        "RAG",
        "请说明你会如何设计可评估的 RAG 检索、重排和引用校验链路。",
        "应覆盖召回、重排、证据归因、离线指标和失败样本闭环。",
    ),
    InterviewQuestion(
        "memory-policy",
        "Memory",
        "请说明长期记忆如何写入、检索、过期并避免污染当前回答。",
        "应说明显式策略、来源、时间衰减、隔离和可审计删除。",
    ),
    InterviewQuestion(
        "context-budget",
        "Context",
        "请说明你会如何在上下文预算内保留任务状态、记忆和检索证据。",
        "应说明分层预算、压缩、优先级与可恢复的原始事件。",
    ),
    InterviewQuestion(
        "runtime-resilience",
        "Runtime",
        "请说明 Agent 运行时如何处理超时、重试、取消和可观测状态。",
        "应说明边界超时、幂等性、资源限制与可诊断的错误分类。",
    ),
    InterviewQuestion(
        "langfuse-observability",
        "Langfuse",
        "请说明如何用 Langfuse 追踪 RAG 阶段并将 badcase 变成改进依据。",
        "应说明 trace/span、输入输出脱敏、指标、标签与本地降级。",
    ),
    InterviewQuestion(
        "project-retrospective",
        "项目复盘",
        "请复盘一个 RAG 项目中的关键失败，说明根因、修复和如何验证没有回归。",
        "应以事实区分症状和根因，并说明回归测试及可量化结果。",
    ),
)


class RagInterviewSimulator:
    """Coordinate interviewer question selection and candidate responses."""

    def __init__(
        self,
        project_context: str,
        questions: Sequence[InterviewQuestion] = DEFAULT_QUESTIONS,
        llm: Optional[Callable[[str], Any]] = None,
        *,
        mode: str = "deterministic",
        model: Optional[str] = None,
        fallback_on_parse_error: bool = False,
    ) -> None:
        if mode not in ("deterministic", "llm"):
            raise ValueError("mode must be deterministic or llm")
        if mode == "llm" and llm is None:
            raise ValueError("llm mode requires an LLM callable")
        if llm is not None and not callable(llm) and not callable(getattr(llm, "invoke", None)):
            raise TypeError("llm must be callable or expose invoke(prompt)")
        question_tuple = tuple(questions)
        if not all(isinstance(question, InterviewQuestion) for question in question_tuple):
            raise TypeError("questions must contain InterviewQuestion instances")
        question_categories = {question.category for question in question_tuple}
        missing_categories = [
            category
            for category in REQUIRED_CATEGORIES
            if category not in question_categories
        ]
        if missing_categories:
            raise ValueError(
                "questions missing required categories: {0}".format(
                    ", ".join(missing_categories)
                )
            )
        self.mode = mode
        self.llm = llm
        self.model = model
        self.fallback_on_parse_error = fallback_on_parse_error
        self.session = InterviewSession(project_context, question_tuple)

    def run(self, rounds: int) -> InterviewSession:
        if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 1:
            raise ValueError("rounds must be a positive integer")
        for _ in range(rounds):
            self.run_round()
        return self.session

    def run_round(self) -> InterviewTurn:
        question, is_follow_up, follow_up_to = self._choose_question()
        answer = self._candidate_answer(question)
        turn = InterviewTurn(
            number=len(self.session.turns) + 1,
            question=question,
            answer=answer,
            is_follow_up=is_follow_up,
            follow_up_to=follow_up_to,
        )
        self.session.append(turn)
        return turn

    def _choose_question(self) -> Tuple[InterviewQuestion, bool, Optional[int]]:
        uncovered = [
            question
            for question in self.session.questions
            if question.category not in self.session.covered_categories
        ]
        if uncovered:
            question = uncovered[0]
            return self._interviewer_question(question, False), False, None
        previous = self.session.turns[-1]
        prompt = "针对你刚才的回答“{0}”，请说明你会如何验证这个方案的效果。".format(
            previous.answer
        )
        question = InterviewQuestion(
            "follow-up-{0}".format(len(self.session.turns) + 1),
            previous.question.category,
            prompt,
            "面试官应检查回答是否给出可执行的验证指标和失败处理。",
        )
        return self._interviewer_question(question, True), True, previous.number

    def _interviewer_question(
        self, question: InterviewQuestion, is_follow_up: bool
    ) -> InterviewQuestion:
        if self.mode == "deterministic":
            return question
        prompt = _interviewer_prompt(self.session, question, is_follow_up, self.model)
        try:
            response = _parse_role_response(
                _invoke_llm(self.llm, prompt), "interviewer", "question"
            )
            return InterviewQuestion(
                question.identifier,
                question.category,
                response["question"],
                question.reference_answer,
            )
        except InterviewResponseError:
            if not self.fallback_on_parse_error:
                raise
            return question

    def _candidate_answer(self, question: InterviewQuestion) -> str:
        if self.mode == "deterministic":
            return self._deterministic_answer(question)
        prompt = _candidate_prompt(self.session, question, self.model)
        try:
            response = _parse_role_response(_invoke_llm(self.llm, prompt), "candidate", "answer")
            return response["answer"]
        except InterviewResponseError:
            if not self.fallback_on_parse_error:
                raise
            return self._deterministic_answer(question)

    def _deterministic_answer(self, question: InterviewQuestion) -> str:
        return (
            "我会先明确 {0} 的可验证目标，再用项目上下文中的证据、指标和失败样本迭代方案。"
        ).format(question.category)


def render_markdown(session: InterviewSession) -> str:
    """Render only public interview data as a stable Markdown report."""
    lines = ["# RAG Agent 双角色面试模拟", "", "## 项目上下文", session.project_context, "", "## 面试记录"]
    for turn in session.turns:
        lines.extend(
            (
                "",
                "### 第 {0} 轮".format(turn.number),
                "- 类别: {0}".format(turn.question.category),
                "- 追问: {0}".format("是" if turn.is_follow_up else "否"),
                "- Interviewer: {0}".format(turn.question.prompt),
                "- Candidate: {0}".format(turn.answer),
            )
        )
    lines.extend(("", "## 类别覆盖", "- " + "、".join(sorted(session.covered_categories))))
    return "\n".join(lines) + "\n"


def _candidate_prompt(session: InterviewSession, question: InterviewQuestion, model: Optional[str]) -> str:
    payload = {
        "role": "candidate",
        "model": model or "",
        "project_context": session.project_context,
        "question": question.public_view(),
        "history": session.public_history(),
        "response_schema": {"answer": "non-empty string"},
    }
    return (
        "Return exactly one JSON object. You are the candidate. Use only the supplied "
        "public project context, question, and history.\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def _interviewer_prompt(
    session: InterviewSession,
    question: InterviewQuestion,
    is_follow_up: bool,
    model: Optional[str],
) -> str:
    payload = {
        "role": "interviewer",
        "model": model or "",
        "project_context": session.project_context,
        "target_category": question.category,
        "question": question.public_view(),
        "history": session.public_history(),
        "is_follow_up": is_follow_up,
        "response_schema": {"question": "non-empty string"},
    }
    return (
        "Return exactly one JSON object. You are the interviewer. Preserve the target "
        "category and use the supplied history only as interview context.\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def _invoke_llm(llm: Any, prompt: str) -> Any:
    invoke = getattr(llm, "invoke", None)
    return invoke(prompt) if callable(invoke) else llm(prompt)


def _parse_role_response(raw: Any, role: str, required_field: str) -> Dict[str, str]:
    try:
        if isinstance(raw, Mapping):
            content = raw.get("content")
            if content is None:
                content = _choice_content(raw.get("choices"), raw)
        else:
            content = getattr(raw, "content", None)
            if content is None:
                content = _choice_content(getattr(raw, "choices", None), raw)
        if isinstance(content, Mapping):
            payload = dict(content)
        elif isinstance(content, str):
            payload = json.loads(content.strip())
        else:
            raise TypeError("response must be a JSON string or mapping")
        if not isinstance(payload, dict):
            raise TypeError("response JSON must be an object")
        value = payload.get(required_field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError("response requires a non-empty {0}".format(required_field))
        return {required_field: value.strip()}
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise InterviewResponseError(role, "invalid_structured_output", raw) from error


def _choice_content(choices: Any, raw: Any) -> Any:
    if choices is None:
        return raw
    if isinstance(choices, (str, bytes)) or not isinstance(choices, Sequence) or len(choices) != 1:
        raise TypeError("LLM choices must contain exactly one completion")
    choice = choices[0]
    if isinstance(choice, Mapping):
        message = choice.get("message")
        return message.get("content") if isinstance(message, Mapping) else choice.get("text")
    message = getattr(choice, "message", None)
    if message is not None:
        return getattr(message, "content", None)
    return getattr(choice, "text", None)


def _response_summary(raw: Any) -> str:
    value = raw if isinstance(raw, str) else repr(raw)
    return value[:240] + ("..." if len(value) > 240 else "")
