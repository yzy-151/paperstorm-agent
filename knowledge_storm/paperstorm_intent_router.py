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
    "intent": "casual_chat | system_help | research_qa | run_research | clarify",
    "need_retrieval": "boolean",
    "tool": "chat_fallback | kb_qa | research_qa | paper_research | clarify",
    "rewritten_query": "standalone Chinese or English query",
    "confidence": "0.0-1.0",
    "reason": "short routing reason",
}


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

        guard_decision = route_high_confidence_rules(message, session, context_window)

        if self.llm_router:
            prompt = build_router_prompt(
                message=message,
                session=session,
                context_window=context_window,
                memory_context=memory_context or {},
                evidence_sufficiency=evidence_sufficiency or {},
            )
            try:
                decision = parse_llm_router_json(self.llm_router(prompt))
                decision = normalize_decision(decision, message, session, context_window)
                if (
                    decision["confidence"] >= self.confidence_threshold
                    and _llm_decision_safe(decision, guard_decision)
                ):
                    decision["router"] = "llm"
                    return decision
            except Exception as exc:  # pragma: no cover - defensive trace path.
                fallback = guard_decision or route_by_rules(message, session, context_window)
                fallback["router_error"] = str(exc)
                return fallback

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
        {"role": item.get("role", ""), "content": item.get("content", "")[:300]}
        for item in context_window[-6:]
    ]
    payload = {
        "user_message": message,
        "topic": session.get("topic", ""),
        "task_id": session.get("task_id", ""),
        "expected_keywords": session.get("expected_keywords") or [],
        "forbidden_keywords": session.get("forbidden_keywords") or [],
        "context_window": compact_context,
        "memory_context": memory_context,
        "evidence_sufficiency": evidence_sufficiency,
        "schema": ROUTER_SCHEMA,
    }
    return (
        "你是 Agent Runtime 的意图路由器。只输出一个 JSON 对象，不要解释。\n"
        "判断用户是在聊天/问系统能力，还是需要论文知识库问答，还是需要触发新调研。\n"
        "如果用户问“你是什么模型/你是谁/怎么用/上下文/记忆/按钮/端口”等系统问题，"
        "不要被 topic 诱导去检索。\n"
        "如果是代词追问，把 topic、上一轮用户问题和当前问题重写成独立 query。\n"
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
    intent = str(decision.get("intent") or "").strip()
    if intent not in {"casual_chat", "system_help", "research_qa", "run_research", "clarify"}:
        intent = "clarify"
    tool = str(decision.get("tool") or "").strip()
    if tool not in {"chat_fallback", "kb_qa", "research_qa", "paper_research", "clarify"}:
        tool = _tool_for_intent(intent)
    need_retrieval = bool(decision.get("need_retrieval"))
    if tool in {"research_qa", "paper_research", "kb_qa"}:
        need_retrieval = True
    if tool in {"chat_fallback", "clarify"}:
        need_retrieval = False
    confidence = _to_float(decision.get("confidence"), 0.7)
    rewritten_query = str(decision.get("rewritten_query") or "").strip()
    if not rewritten_query:
        rewritten_query = rewrite_query(message, session, context_window)
    return {
        "intent": intent,
        "need_retrieval": need_retrieval,
        "tool": tool,
        "rewritten_query": rewritten_query,
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
    parts = [session.get("topic", ""), previous_user, text]
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
    return {
        "intent": intent,
        "need_retrieval": need_retrieval,
        "tool": tool,
        "rewritten_query": rewritten_query,
        "confidence": confidence,
        "reason": reason,
        "router": "rule_fallback",
    }


def _to_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
