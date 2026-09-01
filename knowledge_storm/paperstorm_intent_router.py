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
            "name": "memory.search | memory.write | evidence.search | research.start",
            "arguments": (
                "JSON object; memory.search supports query and mode=context|answer; "
                "memory.write requires content, canonical_key, and memory_type; "
                "memory_type must be semantic, episodic, procedural, or preference"
            ),
        }
    ],
    "rewritten_query": "original user query preserved for compatibility",
    "working_subject": "current subject or empty string; never inherit stale topic",
    "response_contract": {
        "task": "free-form description of the requested response",
        "continue_previous": "boolean",
        "requires_citations": "boolean",
        "requested_output_tokens": "integer or zero",
        "style_notes": ["free-form generation constraints"],
    },
    "tool_policy": {
        "external_retrieval": "allow | deny | unspecified",
        "new_research": "allow | deny",
    },
    "confidence": "0.0-1.0",
    "reason": "short routing reason",
}

ALLOWED_ACTIONS = {"respond", "tool_call", "clarify"}
ALLOWED_TOOLS = {"memory.search", "memory.write", "evidence.search", "research.start"}


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
            return _empty_message_decision()

        if self.llm_router:
            prompt = build_router_prompt(
                message=message,
                session=session,
                context_window=context_window,
                memory_context=memory_context or {},
                evidence_sufficiency=evidence_sufficiency or {},
            )
            telemetry = {}
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
                    return enforce_tool_authorization(decision, telemetry=telemetry)
                return _safe_planner_fallback(
                    message,
                    telemetry,
                    {
                    "type": "low_confidence",
                    "message": "planner confidence below threshold",
                    "recoverable": True,
                    },
                )
            except Exception as exc:
                return _safe_planner_fallback(
                    message, telemetry, _planner_error(exc), router_error=str(exc)
                )

        # Deterministic rules are an offline/demo substitute for the planner.
        # They never act as a production recovery path after an LLM failure.
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
        "JSON 必须完整且精简：reason 不超过 30 个汉字，style_notes 最多 3 条，"
        "不要复述上下文、不要输出 Markdown、不要生成思考过程。\n"
        "为当前这一轮选择动作。普通对话、续写、翻译、代码和改写都使用 respond，"
        "并通过 response_contract 描述生成约束；它们不是新的路由类型。只有确实需要"
        "外部能力时才使用 tool_call。不要生成最终答案。\n"
        "旧任务主题不得自动继承。只有当前消息或最近对话明确承接旧主题时，才设置 "
        "working_subject；创作、闲聊、系统问题默认不检索。\n"
        "短追问需识别是否承接最近对话，但 rewritten_query 保留用户原话；独立检索改写"
        "由 retrieval pipeline 的 SearchPlanner 完成。论文事实需要证据，"
        "用户偏好和稳定事实查 memory.search；用户要求保存或更新长期事实时调用 "
        "memory.write，并在 arguments 中提供 context-independent 的 content、稳定的 "
        "canonical_key 和 memory_type。论文事实查 evidence.search，明确要求完整"
        "调研或现有证据不足时用 research.start。续写必须设置 continue_previous=true，"
        "并要求保持原风格、禁止自我介绍。\n"
        "memory.search 的 arguments.mode 必须是 context 或 answer：若记忆只是完成当前任务"
        "的背景（如‘按我的偏好解释’）用 context；若用户直接询问记住了什么用 answer。"
        "两者都由回复模型结合召回结果生成答案，不要新增内容关键词分类。\n"
        "tool_policy 是 Runtime 的工具授权声明，不是意图标签。使用 evidence.search 时"
        "将 external_retrieval 设为 allow；只有在本轮确实允许创建新调研任务时，才将 "
        "new_research 设为 allow。用户要求不启动调研时必须设为 deny。无法确定、指令冲突"
        "或仅需解释流程时也设为 deny。\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def parse_llm_router_json(text: str) -> Dict:
    text = str(text or "").strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("router response does not contain json")
    value, _end = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(value, dict):
        raise ValueError("router response must contain one JSON object")
    return value


def normalize_decision(
    decision: Dict,
    message: str,
    session: Dict,
    context_window: List[Dict],
) -> Dict:
    action = str(decision.get("action") or "").strip()
    if action == "call_tool":
        action = "tool_call"
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
    rewritten_query = rewrite_query(message, session, context_window)
    response_contract = _response_contract(
        decision.get("response_contract"), message
    )
    tool_policy = _normalize_tool_policy(decision.get("tool_policy"))
    legacy = _legacy_view(action, tool_calls, message)
    return {
        "action": action,
        "tool_calls": tool_calls,
        "response_contract": response_contract,
        "tool_policy": tool_policy,
        **legacy,
        "rewritten_query": rewritten_query,
        "working_subject": str(decision.get("working_subject") or "").strip(),
        "confidence": max(0.0, min(1.0, confidence)),
        "reason": str(decision.get("reason") or "normalized router decision"),
        "router": str(decision.get("router") or "unknown"),
    }


def enforce_tool_authorization(decision: Dict, telemetry: Optional[Dict] = None) -> Dict:
    """Authorize planner-proposed tools at the runtime boundary.

    New research is costly and side-effecting, so it requires an explicit
    structured authorization. A malformed or incomplete planner response can
    never acquire that permission by fallback.
    """

    policy = _normalize_tool_policy(decision.get("tool_policy"))
    tool_name = _first_tool_call_name(decision)
    authorization = {
        "memory.search": "allowed",
        "memory.write": "allowed",
        # Evidence search reads the task's already-materialized local index. It
        # is not an external side effect and must remain available when a
        # response contract requires provenance. Only research.start may fetch
        # new remote material.
        "evidence.search": "allowed",
        "research.start": (
            "allowed"
            if policy["new_research"] == "allow"
            and policy["external_retrieval"] != "deny"
            else "denied"
        ),
    }
    decision["tool_policy"] = policy
    decision["authorization"] = authorization
    if tool_name and authorization.get(tool_name) != "allowed":
        denied_tool = tool_name
        decision["action"] = "respond"
        decision["tool_calls"] = []
        decision.update(_legacy_view("respond", [], decision.get("rewritten_query", "")))
        decision["planner_status"] = "policy_denied"
        decision["planner_error"] = {
            "type": "tool_not_authorized",
            "message": "planner proposed {0} without runtime authorization".format(
                denied_tool
            ),
            "recoverable": True,
        }
        if telemetry is not None:
            decision["planner_telemetry"] = telemetry
    return decision


def _safe_planner_fallback(
    message: str,
    telemetry: Dict,
    error: Dict,
    router_error: str = "",
) -> Dict:
    """Fail closed: retain conversation, deny every external tool."""

    decision = _decision(
        "casual_chat",
        "chat_fallback",
        False,
        message,
        0.0,
        "planner unavailable; continue without external tools",
    )
    decision["router"] = "planner_fail_closed"
    decision["planner_status"] = "fallback"
    decision["planner_error"] = error
    decision["planner_telemetry"] = telemetry
    decision["tool_policy"] = {
        "external_retrieval": "deny",
        "new_research": "deny",
    }
    decision["authorization"] = {
        "memory.search": "denied",
        "evidence.search": "denied",
        "research.start": "denied",
    }
    if router_error:
        decision["router_error"] = router_error
    return decision


def _empty_message_decision() -> Dict:
    decision = _decision(
        "clarify", "clarify", False, "", 1.0, "empty message needs clarification"
    )
    decision["authorization"] = {
        "memory.search": "denied",
        "evidence.search": "denied",
        "research.start": "denied",
    }
    return decision


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
    if is_explicit_memory_write(message):
        return _decision(
            "memory_write",
            "chat_fallback",
            False,
            message,
            0.99,
            "explicit durable memory instruction must use the memory write path",
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
    if prohibits_external_retrieval(message):
        return _decision(
            "casual_chat",
            "chat_fallback",
            False,
            message,
            0.99,
            "user explicitly prohibited external retrieval for this turn",
        )
    if is_conversation_meta_query(message):
        return _decision(
            "casual_chat",
            "chat_fallback",
            False,
            message,
            0.94,
            "question asks about the active conversation rather than external knowledge",
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
    if is_contextual_citation_request(message, session, context_window):
        return _decision(
            "research_qa",
            "research_qa",
            True,
            rewrite_query(message, session, context_window),
            0.94,
            "citation follow-up must reuse existing evidence before starting new research",
        )
    if session.get("task_id") and is_existing_evidence_request(message):
        return _decision(
            "research_qa",
            "research_qa",
            True,
            rewrite_query(message, session, context_window),
            0.96,
            "an existing research task must be queried before starting another one",
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
    if (
        looks_like_followup(message)
        and _has_prior_user_context(context_window)
        and _recent_context_is_research_domain(message, context_window)
    ):
        return _decision(
            "research_qa",
            "research_qa",
            True,
            rewrite_query(message, session, context_window),
            0.84,
            "follow-up question reuses research evidence in the recent conversation",
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


def _first_tool_call_name(decision: Optional[Dict]) -> str:
    calls = (decision or {}).get("tool_calls") or []
    if calls and isinstance(calls[0], dict):
        return str(calls[0].get("name") or "")
    return ""


def rewrite_query(message: str, session: Dict, context_window: List[Dict]) -> str:
    """Preserve the user query; retrieval planning belongs to the pipeline."""

    return str(message or "").strip()


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
            "偏好什么",
            "我叫什么",
            "我的名字",
            "记忆里",
            "我之前说",
            "今天讨论了什么",
            "刚才聊了什么",
        )
    )


def is_explicit_memory_write(message: str) -> bool:
    text = str(message or "").strip().lower()
    return bool(
        re.search(r"请记住|记住[：:]|remember that", text)
        and not re.search(r"不要记住|别记住|无需记住|forget this", text)
    )


def prohibits_external_retrieval(message: str) -> bool:
    text = str(message or "").strip().lower()
    if not text or re.search(r"不要只(?:检索|搜索|查|看)", text):
        return False
    return bool(
        re.search(
            r"(?:不需要|无需|不要|不用|禁止|别)(?:再|去|进行|启动|使用|给我)?"
            r"(?:检索|搜索|查(?:询|找)?|调研|研究)(?:论文|文献|资料)?",
            text,
        )
        or re.search(r"(?:不需要|不要|无需|不用)(?:检索|搜索)?(?:论文|文献)", text)
    )


def is_conversation_meta_query(message: str) -> bool:
    text = str(message or "").strip().lower()
    markers = (
        "刚才的它指什么",
        "刚才说的它",
        "上一句什么意思",
        "你刚才说了什么",
        "我们刚才聊了什么",
        "这里的它指什么",
        "这个代词指什么",
    )
    if any(marker in text for marker in markers):
        return True
    has_conversation_anchor = any(
        marker in text for marker in ("刚才", "前面", "上一轮", "上几轮", "连续几轮")
    )
    asks_about_reference = bool(
        re.search(r"(?:它|这|这个|该词|代词).{0,12}(?:指|指代|具体指|是什么|什么意思)", text)
    )
    return has_conversation_anchor and asks_about_reference


def is_contextual_citation_request(
    message: str, session: Dict, context_window: List[Dict]
) -> bool:
    text = str(message or "").strip().lower()
    asks_for_citation = any(
        marker in text for marker in ("引用", "出处", "来源", "参考文献")
    )
    refers_back = any(
        marker in text for marker in ("上一", "刚才", "前面", "这个结论", "上述")
    )
    has_context = bool(session.get("task_id")) or _has_prior_user_context(context_window)
    return asks_for_citation and refers_back and has_context


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


def is_existing_evidence_request(message: str) -> bool:
    """Recognize evidence questions without treating every citation word as a new job."""
    text = str(message or "").strip().lower()
    explicit_new_research = any(
        marker in text
        for marker in (
            "请调研",
            "重新调研",
            "启动调研",
            "深入调研",
            "检索论文",
            "搜索论文",
            "查找论文",
            "文献综述",
            "literature review",
        )
    )
    if explicit_new_research:
        return False
    return any(
        marker in text
        for marker in (
            "论文证据",
            "文献证据",
            "原文链接",
            "参考文献",
            "引用出处",
            "论文名称",
            "作者和原文",
        )
    )


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


def _recent_context_is_research_domain(message: str, context_window: List[Dict]) -> bool:
    """Use active turns, never a stale session topic, to classify a follow-up."""
    recent = "\n".join(
        str(item.get("content") or "")
        for item in (context_window or [])[-6:]
    )
    text = "{0}\n{1}".format(message or "", recent).lower()
    return any(marker in text for marker in _research_domain_markers())


def _research_domain_markers():
    return [
        "pim",
        "无源互调",
        "passive intermodulation",
        "小波神经网络",
        "wavelet neural network",
        "论文",
        "文献",
        "调研",
    ]


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
            for key in (
                "model",
                "finish_reason",
                "usage",
                "cost_usd",
                "latency_ms",
                "structured_output",
                "error",
            )
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


def _normalize_tool_policy(value):
    value = value if isinstance(value, dict) else {}
    external = str(value.get("external_retrieval") or "unspecified").strip().lower()
    research = str(value.get("new_research") or "deny").strip().lower()
    if external not in {"allow", "deny", "unspecified"}:
        external = "unspecified"
    if research not in {"allow", "deny"}:
        research = "deny"
    return {"external_retrieval": external, "new_research": research}


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
    if tool_name == "memory.write":
        return {"intent": "memory_write", "tool": "memory_write", "need_retrieval": False}
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
