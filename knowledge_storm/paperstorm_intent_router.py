"""Intent routing and tool decision layer for PaperStorm chat.

The router is intentionally small and dependency-light: production deployments
can inject an LLM callable, while local tests and fake demos use a deterministic
fallback. The public output is structured so the runtime, UI, and trace can all
explain why the agent chatted, reused RAG evidence, or triggered research.
"""

import json
import re
from typing import Callable, Dict, List, Optional


ROUTER_SCHEMA = {
    "action": "respond | tool_call | clarify",
    "tool_calls": [
        {
            "name": "memory.search | evidence.search | research.start",
            "arguments": "JSON object",
        }
    ],
    "rewritten_query": "standalone Chinese or English query",
    "working_subject": "current subject or empty string; never inherit stale topic",
    "response_contract": {
        "task": "free-form description of the requested response",
        "continue_previous": "boolean",
        "requires_citations": "boolean",
        "requested_output_tokens": "integer or zero",
        "style_notes": ["free-form generation constraints"],
    },
    "confidence": "0.0-1.0",
    "reason": "short routing reason",
}

ALLOWED_ACTIONS = {"respond", "tool_call", "clarify"}
ALLOWED_TOOLS = {"memory.search", "evidence.search", "research.start"}


class PaperStormIntentRouter:
    """Route a chat turn into an intent, tool decision, and rewritten query."""

    def __init__(
        self,
        llm_router: Optional[Callable[[str], str]] = None,
        confidence_threshold: float = 0.65,
    ):
        self.llm_router = llm_router
        self.confidence_threshold = confidence_threshold

    def route(
        self,
        message: str,
        session: Optional[Dict] = None,
        context_window: Optional[List[Dict]] = None,
        memory_context: Optional[Dict] = None,
        evidence_sufficiency: Optional[Dict] = None,
    ) -> Dict:
        message = str(message or "").strip()
        session = session or {}
        context_window = context_window or []

        if not message:
            return route_high_confidence_rules(message, session, context_window)

        if self.llm_router:
            prompt = build_router_prompt(
                message=message,
                session=session,
                context_window=context_window,
                memory_context=memory_context or {},
                evidence_sufficiency=evidence_sufficiency or {},
            )
            try:
                raw_result = self.llm_router(prompt)
                content, telemetry = _planner_content(raw_result)
                if telemetry.get("error"):
                    raise PlannerProviderError(telemetry["error"])
                decision = parse_llm_router_json(content)
                decision = normalize_decision(decision, message, session, context_window)
                if decision["confidence"] >= self.confidence_threshold:
                    decision["router"] = "llm_planner"
                    decision["planner_status"] = "success"
                    decision["planner_telemetry"] = telemetry
                    return decision
                fallback = route_by_rules(message, session, context_window)
                fallback["planner_status"] = "fallback"
                fallback["planner_error"] = {
                    "type": "low_confidence",
                    "message": "planner confidence below threshold",
                    "recoverable": True,
                }
                fallback["planner_telemetry"] = telemetry
                return fallback
            except Exception as exc:
                fallback = route_by_rules(message, session, context_window)
                fallback["planner_status"] = "fallback"
                fallback["planner_error"] = _planner_error(exc)
                fallback["router_error"] = str(exc)  # compatibility
                return fallback

        guard_decision = route_high_confidence_rules(message, session, context_window)
        if guard_decision:
            return guard_decision
        return route_by_rules(message, session, context_window)


def build_router_prompt(
    message: str,
    session: Dict,
    context_window: List[Dict],
    memory_context: Dict,
    evidence_sufficiency: Dict,
) -> str:
    compact_context = [
        {
            "role": item.get("role", ""),
            "content": item.get("content", "")[:1200],
            "context_layer": item.get("metadata", {}).get("context_layer", "active"),
        }
        for item in context_window[-24:]
    ]
    payload = {
        "user_message": message,
        "task_id": session.get("task_id", ""),
        "expected_keywords": session.get("expected_keywords") or [],
        "forbidden_keywords": session.get("forbidden_keywords") or [],
        "context_window": compact_context,
        "memory_context": memory_context,
        "evidence_sufficiency": evidence_sufficiency,
        "schema": ROUTER_SCHEMA,
    }
    return (
        "你是 PaperStorm 的 Turn Planner。只输出一个符合 Schema 的 JSON 对象。\n"
        "为当前这一轮选择动作。普通对话、续写、翻译、代码和改写都使用 respond，"
        "并通过 response_contract 描述生成约束；它们不是新的路由类型。只有确实需要"
        "外部能力时才使用 tool_call。不要生成最终答案。\n"
        "旧任务主题不得自动继承。只有当前消息或最近对话明确承接旧主题时，才设置 "
        "working_subject；创作、闲聊、系统问题默认不检索。\n"
        "短追问必须根据最近对话改写，不能仅凭 task_id 猜测主题。论文事实需要证据，"
        "用户偏好和稳定事实查 memory.search，论文事实查 evidence.search，明确要求完整"
        "调研或现有证据不足时用 research.start。续写必须设置 continue_previous=true，"
        "并要求保持原风格、禁止自我介绍。\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def parse_llm_router_json(text: str) -> Dict:
    text = str(text or "").strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("router response does not contain json")
    return json.loads(match.group(0))


def normalize_decision(
    decision: Dict,
    message: str,
    session: Dict,
    context_window: List[Dict],
) -> Dict:
    action = str(decision.get("action") or "").strip()
    tool_calls = _normalize_tool_calls(decision.get("tool_calls"))
    if not action:
        action, tool_calls = _legacy_action(decision)
    if action not in ALLOWED_ACTIONS:
        action = "clarify"
        tool_calls = []
    if action == "respond":
        tool_calls = []
    elif action == "clarify":
        tool_calls = []
    elif not tool_calls:
        action = "clarify"
    confidence = _to_float(decision.get("confidence"), 0.7)
    rewritten_query = str(decision.get("rewritten_query") or "").strip()
    if not rewritten_query:
        rewritten_query = rewrite_query(message, session, context_window)
    response_contract = _response_contract(
        decision.get("response_contract"), message
    )
    legacy = _legacy_view(action, tool_calls, message)
    return {
        "action": action,
        "tool_calls": tool_calls,
        "response_contract": response_contract,
        **legacy,
        "rewritten_query": rewritten_query,
        "working_subject": str(decision.get("working_subject") or "").strip(),
        "confidence": max(0.0, min(1.0, confidence)),
        "reason": str(decision.get("reason") or "normalized router decision"),
        "router": str(decision.get("router") or "unknown"),
    }


def route_by_rules(message: str, session: Dict, context_window: List[Dict]) -> Dict:
    guarded = route_high_confidence_rules(message, session, context_window)
    if guarded:
        return guarded
    return _decision(
        "casual_chat",
        "chat_fallback",
        False,
        message,
        0.62,
        "no explicit retrieval intent; default safely to conversation",
    )


def route_high_confidence_rules(
    message: str, session: Dict, context_window: List[Dict]
) -> Optional[Dict]:
    if not message:
        return _decision(
            "clarify", "clarify", False, "", 1.0, "empty message needs clarification"
        )
    if is_memory_query(message):
        return _decision(
            "memory_recall",
            "memory_search",
            False,
            rewrite_query(message, session, context_window),
            0.9,
            "message explicitly asks about prior conversation or durable memory",
        )
    if is_system_help(message):
        return _decision(
            "system_help",
            "chat_fallback",
            False,
            message,
            0.9,
            "user asks about agent identity, UI, service, memory, or context",
        )
    if is_casual_chat(message):
        return _decision(
            "casual_chat",
            "chat_fallback",
            False,
            message,
            0.82,
            "casual message does not need retrieval",
        )
    if is_direct_research_request(message):
        rewritten = rewrite_query(message, session, context_window)
        return _decision(
            "run_research",
            "paper_research",
            True,
            rewritten,
            0.86,
            "message asks for papers, literature, survey, citations, or technical evidence",
        )
    if is_research_knowledge_question(message):
        return _decision(
            "research_qa",
            "research_qa",
            True,
            rewrite_query(message, session, context_window),
            0.82,
            "message asks a domain knowledge question that benefits from evidence",
        )
    if looks_like_followup(message) and _has_prior_user_context(context_window):
        if session.get("task_id") or session.get("topic"):
            return _decision(
                "research_qa",
                "research_qa",
                True,
                rewrite_query(message, session, context_window),
                0.84,
                "follow-up question reuses the active research context",
            )
    if looks_like_followup(message) and not _has_prior_user_context(context_window):
        return _decision(
            "clarify",
            "clarify",
            False,
            message,
            0.88,
            "short follow-up has no usable conversational antecedent",
        )
    return None


def _llm_decision_safe(decision: Dict, guard_decision: Optional[Dict]) -> bool:
    """Refuse LLM decisions that fight a high-confidence rule outcome."""
    if not guard_decision:
        return True
    guard_tool = str((guard_decision or {}).get("tool") or "")
    if guard_tool in {"chat_fallback", "clarify"}:
        # The rules already decided this turn must not touch retrieval.
        if decision.get("need_retrieval"):
            return False
        # A plain chat/system guard should not be downgraded to "clarify".
        if guard_tool == "chat_fallback" and decision.get("tool") == "clarify":
            return False
        return True
    # The rules already decided this turn needs retrieval; do not let the LLM
    # downgrade it to chat or clarification.
    if decision.get("tool") in {"chat_fallback", "clarify"}:
        return False
    return True


def rewrite_query(message: str, session: Dict, context_window: List[Dict]) -> str:
    text = str(message or "").strip()
    if not looks_like_followup(text):
        return text
    previous_user = ""
    for item in reversed(context_window or []):
        if item.get("role") == "user":
            content = str(item.get("content") or "").strip()
            if content and content != text:
                previous_user = content
                break
    parts = [previous_user, text]
    return "\n".join(part for part in parts if part)


def looks_like_followup(text: str) -> bool:
    text = str(text or "").strip()
    if not text or len(text) >= 80:
        return False
    lead_markers = ("那", "它", "这个", "上述", "继续", "还有呢", "然后呢", "然后", "再")
    if text.startswith(lead_markers):
        return True
    # Short pronoun/demonstrative phrases without their own subject.
    return len(text) <= 20 and any(
        marker in text for marker in ("这个", "它", "那", "上述", "继续", "还有呢")
    )


def is_system_help(message: str) -> bool:
    text = str(message or "").strip().lower()
    hits = [
        "你是什么模型",
        "你是模型",
        "你用的什么模型",
        "你是谁",
        "你的身份",
        "你叫什么",
        "你能做什么",
        "你可以做什么",
        "介绍一下你",
        "怎么使用",
        "如何使用",
        "这个网页",
        "这个界面",
        "按钮",
        "service",
        "serve",
        "端口",
        "api",
        "sse",
        "上下文",
        "记忆",
        "压缩",
        "聊天模式",
        "调研模式",
        "什么算法",
        "用什么算法",
        "怎么检索",
        "如何检索",
        "检索方式",
        "检索算法",
        "技术栈",
        "怎么实现",
        "如何实现",
        "具体实现",
        "实现原理",
        "工作流程",
        "架构",
        "底层",
        "原理",
        "逻辑",
        "知识库问答",
        "知识库",
        "版本",
        "帮助",
        "报错",
    ]
    if any(hit in text for hit in hits):
        return not any(marker in text for marker in _research_domain_markers())
    return False


def is_memory_query(message: str) -> bool:
    text = str(message or "").strip().lower()
    return any(
        marker in text
        for marker in (
            "你记得",
            "还记得",
            "之前聊过",
            "以前聊过",
            "我的偏好",
            "记忆里",
            "我之前说",
        )
    )


def is_casual_chat(message: str) -> bool:
    text = str(message or "").strip().lower()
    social_phrases = [
        "你好",
        "您好",
        "hello",
        "hi",
        "早上好",
        "晚上好",
        "莫西莫西",
        "もしもし",
        "谢谢",
        "感谢",
        "再见",
        "你在干嘛",
        "你在做什么",
        "今天天气不错",
        "聊聊天",
    ]
    return any(hit in text for hit in social_phrases) and not is_direct_research_request(text)


def is_direct_research_request(message: str) -> bool:
    text = str(message or "").strip().lower()
    hits = [
        "论文",
        "文献",
        "检索",
        "调研",
        "综述",
        "survey",
        "paper",
        "citation",
        "引用",
        "最新研究",
        "深入研究",
        "查一下",
        "搜索一下",
    ]
    return any(hit in text for hit in hits)


def is_research_knowledge_question(message: str) -> bool:
    text = str(message or "").strip().lower()
    domain_hits = [
        "pim",
        "无源互调",
        "passive intermodulation",
        "神经网络抑制",
        "rag",
        "retrieval augmented generation",
    ]
    question_hits = ["什么", "为何", "为什么", "如何", "怎么", "区别", "原理", "吗", "？", "?"]
    return any(hit in text for hit in domain_hits) and any(hit in text for hit in question_hits)


def _has_prior_user_context(context_window: List[Dict]) -> bool:
    return any(
        item.get("role") == "user" and str(item.get("content") or "").strip()
        for item in context_window or []
    )


def _research_domain_markers():
    return ["pim", "无源互调", "passive intermodulation", "论文", "文献", "调研"]


def _tool_for_intent(intent: str) -> str:
    return {
        "casual_chat": "chat_fallback",
        "system_help": "chat_fallback",
        "research_qa": "research_qa",
        "run_research": "paper_research",
        "clarify": "clarify",
    }.get(intent, "clarify")


def _decision(intent, tool, need_retrieval, rewritten_query, confidence, reason):
    action, tool_calls = _legacy_action(
        {"intent": intent, "tool": tool, "need_retrieval": need_retrieval}
    )
    return {
        "action": action,
        "tool_calls": tool_calls,
        "response_contract": _response_contract(None, rewritten_query),
        "intent": intent,
        "need_retrieval": need_retrieval,
        "tool": tool,
        "rewritten_query": rewritten_query,
        "confidence": confidence,
        "reason": reason,
        "router": "rule_fallback",
        "planner_status": "offline_fallback",
    }


class PlannerProviderError(RuntimeError):
    def __init__(self, error):
        self.error = error or {}
        super().__init__(str(self.error.get("message") or "planner provider failed"))


def _planner_content(result):
    if isinstance(result, dict) and "content" in result:
        telemetry = {
            key: result.get(key)
            for key in ("finish_reason", "usage", "cost_usd", "latency_ms", "error")
        }
        return str(result.get("content") or ""), telemetry
    return str(result or ""), {}


def _planner_error(error):
    if isinstance(error, PlannerProviderError):
        payload = dict(error.error)
        payload.setdefault("type", "provider_error")
        payload.setdefault("message", str(error))
        payload.setdefault("recoverable", True)
        return payload
    if isinstance(error, (ValueError, json.JSONDecodeError)):
        error_type = "invalid_response"
    elif isinstance(error, TimeoutError):
        error_type = "timeout"
    else:
        error_type = "provider_error"
    return {"type": error_type, "message": str(error), "recoverable": True}


def _normalize_tool_calls(value):
    output = []
    for item in value or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if name not in ALLOWED_TOOLS:
            continue
        arguments = item.get("arguments")
        output.append({"name": name, "arguments": arguments if isinstance(arguments, dict) else {}})
    return output


def _legacy_action(decision):
    tool = str(decision.get("tool") or "").strip()
    intent = str(decision.get("intent") or "").strip()
    if tool in {"paper_research"} or intent == "run_research":
        return "tool_call", [{"name": "research.start", "arguments": {}}]
    if tool in {"research_qa", "kb_qa"} or intent == "research_qa":
        return "tool_call", [{"name": "evidence.search", "arguments": {}}]
    if tool == "memory_search" or intent == "memory_recall":
        return "tool_call", [{"name": "memory.search", "arguments": {}}]
    if tool == "clarify" or intent == "clarify":
        return "clarify", []
    return "respond", []


def _legacy_view(action, tool_calls, message):
    tool_name = tool_calls[0]["name"] if tool_calls else ""
    if action == "clarify":
        return {"intent": "clarify", "tool": "clarify", "need_retrieval": False}
    if tool_name == "research.start":
        return {"intent": "run_research", "tool": "paper_research", "need_retrieval": True}
    if tool_name == "evidence.search":
        return {"intent": "research_qa", "tool": "research_qa", "need_retrieval": True}
    if tool_name == "memory.search":
        return {"intent": "memory_recall", "tool": "memory_search", "need_retrieval": False}
    intent = "system_help" if is_system_help(message) else "casual_chat"
    return {"intent": intent, "tool": "chat_fallback", "need_retrieval": False}


def _response_contract(value, message):
    value = value if isinstance(value, dict) else {}
    continuation = bool(value.get("continue_previous")) or _looks_like_continuation(message)
    notes = [str(item).strip() for item in value.get("style_notes") or [] if str(item).strip()]
    if continuation:
        for note in ("保持上一段的叙事、人物和语言风格", "不要自我介绍", "不要重复开头"):
            if note not in notes:
                notes.append(note)
    try:
        requested = max(0, int(value.get("requested_output_tokens") or 0))
    except (TypeError, ValueError):
        requested = 0
    return {
        "task": str(value.get("task") or message or "直接回答用户").strip(),
        "continue_previous": continuation,
        "requires_citations": bool(value.get("requires_citations")),
        "requested_output_tokens": requested,
        "style_notes": notes,
    }


def _looks_like_continuation(message):
    text = str(message or "").strip().lower()
    return text.startswith(("继续", "接着", "续写", "往下写", "continue"))


def _to_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
