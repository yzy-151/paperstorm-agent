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
                if decision["confidence"] >= self.confidence_threshold:
                    decision["router"] = "llm"
                    return decision
            except Exception as exc:  # pragma: no cover - defensive trace path.
                fallback = route_by_rules(message, session, context_window)
                fallback["router_error"] = str(exc)
                return fallback

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
        "你是企业级 Agent Runtime 的意图路由器。只输出一个 JSON 对象，不要解释。\n"
        "判断用户是在闲聊/问系统能力，还是需要论文知识库问答，还是需要触发新调研。\n"
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
        intent = "research_qa"
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
    rewritten = rewrite_query(message, session, context_window)
    if is_direct_research_request(message):
        return _decision(
            "run_research",
            "paper_research",
            True,
            rewritten,
            0.86,
            "message asks for papers, literature, survey, citations, or technical evidence",
        )
    return _decision(
        "research_qa",
        "research_qa",
        True,
        rewritten,
        0.76,
        "technical question should be grounded by PaperStorm evidence",
    )


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
    markers = ["那", "它", "这个", "上述", "继续", "为什么", "如何", "区别", "还有呢"]
    return any(marker in text for marker in markers) and len(text) < 120


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
        "版本",
        "帮助",
        "报错",
    ]
    if any(hit in text for hit in hits):
        return not any(marker in text for marker in _research_domain_markers())
    return False


def is_casual_chat(message: str) -> bool:
    text = str(message or "").strip().lower()
    greetings = ["你好", "您好", "hello", "hi", "早上好", "晚上好"]
    return any(hit in text for hit in greetings) and not is_direct_research_request(text)


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
        "rag",
        "无源互调",
        "passive intermodulation",
        "神经网络抑制",
    ]
    return any(hit in text for hit in hits)


def _research_domain_markers():
    return ["pim", "无源互调", "passive intermodulation", "论文", "文献", "调研"]


def _tool_for_intent(intent: str) -> str:
    return {
        "casual_chat": "chat_fallback",
        "system_help": "chat_fallback",
        "research_qa": "research_qa",
        "run_research": "paper_research",
        "clarify": "clarify",
    }.get(intent, "research_qa")


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
