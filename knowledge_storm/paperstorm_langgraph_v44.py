import hashlib
import json
import logging
import re
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, TypedDict

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy
from pydantic import BaseModel, Field

from .paperstorm_intent_router import PaperStormIntentRouter
from .paperstorm_memory_v43 import LongTermMemoryService
from .paperstorm_research_qa import evaluate_evidence_sufficiency


logging.getLogger("langgraph.pregel._retry").setLevel(logging.WARNING)

RETRIEVE_MARKER = "__NEED_RESEARCH__"


class ConversationRequestV44(BaseModel):
    thread_id: str = Field(min_length=1, max_length=200)
    request_id: str = Field(min_length=1, max_length=200)
    user_id: str = "local-user"
    message: str = Field(min_length=1)
    topic: str = ""
    task_id: str = ""
    run_mode: str = "fake"
    retriever: str = "arxiv"
    output_language: str = "zh"
    expected_keywords: List[str] = Field(default_factory=list)
    forbidden_keywords: List[str] = Field(default_factory=list)
    context_window: List[Dict] = Field(default_factory=list)
    allow_deep_research: bool = True
    source_message_id: str = ""


class ConversationStateV44(TypedDict, total=False):
    thread_id: str
    request_id: str
    user_id: str
    namespace: str
    message: str
    topic: str
    task_id: str
    run_mode: str
    retriever: str
    output_language: str
    expected_keywords: List[str]
    forbidden_keywords: List[str]
    context_window: List[Dict]
    allow_deep_research: bool
    source_message_id: str
    router_decision: Dict
    memory_recall: Dict
    memory_write: Dict
    knowledge_result: Dict
    evidence_grade: Dict
    deep_research_result: Dict
    answer: str
    citations: List[Dict]
    evidence: List[Dict]
    grounded: bool
    retrieval_triggered: bool
    used_task_id: str
    artifact_uri: str
    route: str
    status: str
    error: str
    executed_nodes: List[str]
    node_events: List[Dict]
    retrieval_stack: str
    retrieval_mode: str
    escalate_to_retrieval: bool


class StormDeepResearchToolV44:
    name = "storm_deep_research"
    description = (
        "Run PaperStorm/STORM as an isolated deep-research tool and return only a "
        "structured conclusion, citations, task id, evidence summary, and artifact URI."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "topic": {"type": "string"},
            "task_id": {"type": "string"},
            "run_mode": {"type": "string", "default": "fake"},
            "retriever": {"type": "string", "default": "arxiv"},
            "output_language": {"type": "string", "default": "zh"},
            "expected_keywords": {"type": "array", "items": {"type": "string"}},
            "forbidden_keywords": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["question"],
    }
    output_schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "citations": {"type": "array", "items": {"type": "object"}},
            "evidence": {"type": "array", "items": {"type": "object"}},
            "grounded": {"type": "boolean"},
            "task_id": {"type": "string"},
            "artifact_uri": {"type": "string"},
            "evidence_sufficiency": {"type": "object"},
        },
        "required": ["answer", "citations", "grounded", "task_id", "artifact_uri"],
    }

    def __init__(self, task_service):
        self.task_service = task_service

    def to_schema(self):
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
        }

    def run(self, arguments: Dict):
        arguments = dict(arguments or {})
        question = str(arguments.get("question") or "").strip()
        if not question:
            raise ValueError("Tool argument 'question' is required.")
        allowed = {
            "question",
            "topic",
            "task_id",
            "run_mode",
            "retriever",
            "output_language",
            "expected_keywords",
            "forbidden_keywords",
        }
        payload = {key: value for key, value in arguments.items() if key in allowed and value not in (None, "")}
        result = self.task_service.ask_research_agent(**payload)
        if (
            payload.get("task_id")
            and (result.get("decision") or {}).get("action") == "reject_low_confidence"
        ):
            payload.pop("task_id", None)
            result = self.task_service.ask_research_agent(**payload)
        task_id = result.get("used_task_id") or ""
        artifact_uri = ""
        if task_id:
            state = self.task_service.get_task(task_id)
            artifact_uri = Path(state["output_dir"]).resolve().as_uri()
        evidence = []
        for item in (result.get("evidence") or [])[:5]:
            evidence.append(
                {
                    "chunk_id": item.get("chunk_id", ""),
                    "title": item.get("title", ""),
                    "content": str(item.get("content") or "")[:1200],
                    "score": item.get("score", 0),
                    "source_type": item.get("source_type", ""),
                }
            )
        return {
            "answer": result.get("answer", ""),
            "citations": result.get("citations") or [],
            "evidence": evidence,
            "grounded": bool(result.get("grounded")),
            "task_id": task_id,
            "artifact_uri": artifact_uri,
            "evidence_sufficiency": result.get("evidence_sufficiency") or {},
            "retrieval_triggered": bool(result.get("retrieval_triggered")),
            "decision": result.get("decision") or {},
            "retrieval_stack": "storm_deep_research_tool",
            "retrieval_mode": "",
        }


class PaperStormLangGraphRuntime:
    runtime_name = "langgraph-v4.4"

    def __init__(
        self,
        root_dir,
        task_service,
        intent_router: Optional[PaperStormIntentRouter] = None,
        memory_service: Optional[LongTermMemoryService] = None,
        deep_research_tool=None,
        chat_llm: Optional[Callable[[str], str]] = None,
        evidence_judge: Optional[Callable[[str], str]] = None,
    ):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir = self.root_dir / "request_results"
        self.trace_dir = self.root_dir / "traces"
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.task_service = task_service
        self.intent_router = intent_router or PaperStormIntentRouter()
        self.chat_llm = chat_llm
        self.evidence_judge = evidence_judge
        self.memory_service = memory_service or LongTermMemoryService(
            Path(task_service.root_dir) / "memory_service_v43"
        )
        self.deep_research_tool = deep_research_tool or StormDeepResearchToolV44(
            task_service
        )
        self._connection = None
        self.checkpointer = None
        self.graph = None

    def invoke(self, **payload):
        request = ConversationRequestV44(**payload)
        cached = self._read_result(request.thread_id, request.request_id)
        if cached:
            return dict(cached, idempotent_replay=True)
        self._ensure_open()
        initial = {
            **_model_dump(request),
            "namespace": _memory_namespace(request.user_id),
            "router_decision": {},
            "memory_recall": {},
            "memory_write": {"status": "not_evaluated"},
            "knowledge_result": {},
            "evidence_grade": {},
            "deep_research_result": {},
            "answer": "",
            "citations": [],
            "evidence": [],
            "grounded": False,
            "retrieval_triggered": False,
            "used_task_id": request.task_id,
            "artifact_uri": "",
            "route": "",
            "status": "running",
            "error": "",
            "executed_nodes": [],
            "node_events": [],
            "retrieval_stack": "",
            "retrieval_mode": "",
        }
        config = {"configurable": {"thread_id": request.thread_id}}
        try:
            final_state = self.graph.invoke(initial, config=config)
            result = self._public_result(final_state)
            self._write_result(request.thread_id, request.request_id, result)
            return result
        finally:
            self.close()

    def get_thread_state(self, thread_id: str):
        self._ensure_open()
        try:
            snapshot = self.graph.get_state(
                {"configurable": {"thread_id": str(thread_id)}}
            )
            return _snapshot_payload(snapshot)
        finally:
            self.close()

    def get_thread_history(self, thread_id: str, limit: int = 50):
        self._ensure_open()
        config = {"configurable": {"thread_id": str(thread_id)}}
        try:
            checkpoints = []
            for snapshot in self.graph.get_state_history(config, limit=max(1, int(limit))):
                checkpoints.append(_snapshot_payload(snapshot, include_values=False))
            return {"thread_id": thread_id, "checkpoints": checkpoints}
        finally:
            self.close()

    def get_graph_spec(self):
        return {
            "runtime": self.runtime_name,
            "nodes": [
                "classify",
                "memory_recall",
                "casual_chat",
                "memory_answer",
                "knowledge_retrieval",
                "evidence_grade",
                "deep_research",
                "answer_with_citations",
                "refuse_or_clarify",
                "memory_candidate_write",
                "final_trace",
            ],
            "edges": [
                ["START", "classify"],
                ["classify", "memory_recall|refuse_or_clarify"],
                ["memory_recall", "memory_answer|casual_chat|knowledge_retrieval|refuse_or_clarify"],
                ["knowledge_retrieval", "evidence_grade"],
                ["evidence_grade", "answer_with_citations|deep_research|refuse_or_clarify"],
                ["deep_research", "answer_with_citations"],
                ["answer_nodes", "memory_candidate_write"],
                ["memory_candidate_write", "final_trace"],
                ["final_trace", "END"],
            ],
            "checkpoint": {
                "backend": "sqlite",
                "scope": "thread_id",
                "path": str(self.root_dir / "checkpoints.sqlite"),
            },
            "retry": {
                "nodes": ["knowledge_retrieval", "deep_research"],
                "max_attempts": 2,
                "retry_on": ["ConnectionError", "TimeoutError"],
            },
            "long_term_memory": "PaperStorm V4.3 namespace store",
            "deep_research_tool": self.deep_research_tool.to_schema()
            if hasattr(self.deep_research_tool, "to_schema")
            else {"name": self.deep_research_tool.name},
        }

    def graph_spec(self):
        return self.get_graph_spec()

    def close(self):
        if self._connection is not None:
            self._connection.close()
        self._connection = None
        self.checkpointer = None
        self.graph = None

    def _ensure_open(self):
        if self._connection is not None:
            return
        self._connection = sqlite3.connect(
            str(self.root_dir / "checkpoints.sqlite"), check_same_thread=False
        )
        serializer = JsonPlusSerializer(allowed_msgpack_modules=())
        self.checkpointer = SqliteSaver(self._connection, serde=serializer)
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(ConversationStateV44)
        retry = RetryPolicy(
            initial_interval=0.01,
            backoff_factor=1.0,
            max_interval=0.05,
            max_attempts=2,
            jitter=False,
            retry_on=(ConnectionError, TimeoutError),
        )
        builder.add_node("classify", self._classify, metadata={"timeout_sec": 5})
        builder.add_node("memory_recall", self._memory_recall, metadata={"timeout_sec": 2})
        builder.add_node("casual_chat", self._casual_chat)
        builder.add_node("memory_answer", self._memory_answer)
        builder.add_node(
            "knowledge_retrieval",
            self._knowledge_retrieval,
            retry_policy=retry,
            metadata={"timeout_sec": 15},
        )
        builder.add_node("evidence_grade", self._evidence_grade)
        builder.add_node(
            "deep_research",
            self._deep_research,
            retry_policy=retry,
            metadata={"timeout_sec": 300, "isolated_context": True},
        )
        builder.add_node("answer_with_citations", self._answer_with_citations)
        builder.add_node("refuse_or_clarify", self._refuse_or_clarify)
        builder.add_node("memory_candidate_write", self._memory_candidate_write)
        builder.add_node("final_trace", self._final_trace)

        builder.add_edge(START, "classify")
        builder.add_conditional_edges(
            "classify",
            self._after_classify,
            {"memory_recall": "memory_recall", "refuse_or_clarify": "refuse_or_clarify"},
        )
        builder.add_conditional_edges(
            "memory_recall",
            self._after_memory_recall,
            {
                "memory_answer": "memory_answer",
                "casual_chat": "casual_chat",
                "knowledge_retrieval": "knowledge_retrieval",
                "refuse_or_clarify": "refuse_or_clarify",
            },
        )
        builder.add_edge("knowledge_retrieval", "evidence_grade")
        builder.add_conditional_edges(
            "evidence_grade",
            self._after_evidence_grade,
            {
                "answer_with_citations": "answer_with_citations",
                "deep_research": "deep_research",
                "refuse_or_clarify": "refuse_or_clarify",
            },
        )
        builder.add_edge("deep_research", "answer_with_citations")
        builder.add_conditional_edges(
            "casual_chat",
            self._after_casual_chat,
            {
                "knowledge_retrieval": "knowledge_retrieval",
                "memory_candidate_write": "memory_candidate_write",
            },
        )
        for node in [
            "memory_answer",
            "answer_with_citations",
            "refuse_or_clarify",
        ]:
            builder.add_edge(node, "memory_candidate_write")
        builder.add_edge("memory_candidate_write", "final_trace")
        builder.add_edge("final_trace", END)
        return builder.compile(checkpointer=self.checkpointer)

    def _classify(self, state: ConversationStateV44):
        started = time.perf_counter()
        try:
            decision = self.intent_router.route(
                message=state["message"],
                session={
                    "topic": state.get("topic", ""),
                    "task_id": state.get("task_id", ""),
                    "expected_keywords": state.get("expected_keywords") or [],
                    "forbidden_keywords": state.get("forbidden_keywords") or [],
                },
                context_window=state.get("context_window") or [],
                memory_context={},
            )
            return self._success_update(
                state, "classify", started, {"router_decision": decision}
            )
        except Exception as error:
            self._error_event(state, "classify", started, error)
            raise

    def _memory_recall(self, state: ConversationStateV44):
        started = time.perf_counter()
        try:
            recalled = self.memory_service.search(
                state["namespace"], state["message"], top_k=5
            )
            return self._success_update(
                state,
                "memory_recall",
                started,
                {"memory_recall": recalled},
                result_count=len(recalled.get("results") or []),
            )
        except Exception as error:
            self._error_event(state, "memory_recall", started, error)
            raise

    def _casual_chat(self, state: ConversationStateV44):
        started = time.perf_counter()
        decision = state.get("router_decision") or {}
        escalate = False
        if _is_explicit_memory_write(state["message"]):
            answer = _casual_answer(state["message"], state.get("memory_recall") or {})
            decision = {
                "intent": "memory_write",
                "need_retrieval": False,
                "tool": "memory_write",
                "rewritten_query": state["message"],
                "confidence": 0.98,
                "reason": "explicit durable memory instruction",
                "router": "langgraph_memory_policy_v44",
            }
        elif _is_memory_question(state["message"]):
            answer = _casual_answer(state["message"], state.get("memory_recall") or {})
            decision = {
                "intent": "memory_recall",
                "need_retrieval": False,
                "tool": "memory_search",
                "rewritten_query": state["message"],
                "confidence": 0.98,
                "reason": "memory question has no relevant hit",
                "router": "langgraph_memory_policy_v44",
            }
        else:
            answer = self._casual_answer(state)
            if answer == RETRIEVE_MARKER:
                answer = ""
                escalate = True
            elif not answer:
                answer = _casual_answer(state["message"], state.get("memory_recall") or {})
                escalate = _needs_research_fallback(state)
        return self._success_update(
            state,
            "casual_chat",
            started,
            {
                "answer": answer,
                "citations": [],
                "evidence": [],
                "grounded": False,
                "retrieval_triggered": False,
                "route": "casual_chat",
                "router_decision": decision,
                "escalate_to_retrieval": escalate,
            },
        )

    def _casual_answer(self, state: ConversationStateV44) -> str:
        """Generate the casual reply: LLM first when available, local fallback."""
        if self.chat_llm is not None:
            try:
                reply = self._chat_llm_answer(state)
                if reply:
                    return reply
            except Exception:
                pass
        return _casual_answer(
            state.get("message") or "",
            state.get("memory_recall") or {},
        )

    def _chat_llm_answer(self, state: ConversationStateV44) -> str:
        return str(self.chat_llm(_casual_chat_prompt(state)) or "").strip()

    @staticmethod
    def _after_casual_chat(state: ConversationStateV44):
        """Answer-first: escalate to retrieval only when the chat layer says so."""
        if state.get("escalate_to_retrieval"):
            return "knowledge_retrieval"
        return "memory_candidate_write"

    def _memory_answer(self, state: ConversationStateV44):
        started = time.perf_counter()
        results = (state.get("memory_recall") or {}).get("results") or []
        answer = "我从你的跨会话长期记忆中找到了：{0}".format(
            "；".join(item.get("content", "") for item in results[:3])
        )
        return self._success_update(
            state,
            "memory_answer",
            started,
            {
                "answer": answer,
                "citations": [],
                "evidence": [],
                "grounded": False,
                "retrieval_triggered": False,
                "route": "memory_answer",
                "router_decision": {
                    "intent": "memory_recall",
                    "need_retrieval": False,
                    "tool": "memory_search",
                    "rewritten_query": state["message"],
                    "confidence": 0.98,
                    "reason": "relevant cross-session memory is available",
                    "router": "langgraph_memory_policy_v44",
                },
            },
        )

    def _knowledge_retrieval(self, state: ConversationStateV44):
        started = time.perf_counter()
        try:
            task_id = state.get("task_id") or ""
            if not task_id:
                result = {"answer": "", "citations": [], "evidence": [], "grounded": False}
            else:
                result = self.task_service.query_knowledge_base(
                    task_id,
                    question=(state.get("router_decision") or {}).get("rewritten_query")
                    or state["message"],
                    top_k=3,
                )
            return self._success_update(
                state,
                "knowledge_retrieval",
                started,
                {"knowledge_result": result},
                evidence_count=len(result.get("evidence") or []),
            )
        except (KeyError, FileNotFoundError) as error:
            result = {
                "answer": "",
                "citations": [],
                "evidence": [],
                "grounded": False,
                "retrieval_error": str(error),
            }
            return self._success_update(
                state,
                "knowledge_retrieval",
                started,
                {"knowledge_result": result},
                degraded=True,
            )
        except Exception as error:
            self._error_event(state, "knowledge_retrieval", started, error)
            raise

    def _evidence_grade(self, state: ConversationStateV44):
        started = time.perf_counter()
        result = state.get("knowledge_result") or {}
        question = (state.get("router_decision") or {}).get("rewritten_query") or state["message"]
        grade = evaluate_evidence_sufficiency(
            question=question,
            evidence=result.get("evidence") or [],
            citations=result.get("citations") or [],
            topic=state.get("topic") or "",
            expected_keywords=state.get("expected_keywords") or [],
            forbidden_keywords=state.get("forbidden_keywords") or [],
        )
        judge = "local"
        if self.evidence_judge is not None:
            try:
                verdict = _parse_judge_verdict(
                    self.evidence_judge(
                        _evidence_judge_prompt(
                            question=question,
                            topic=state.get("topic") or "",
                            expected_keywords=state.get("expected_keywords") or [],
                            evidence=result.get("evidence") or [],
                        )
                    )
                )
                if verdict == "can_answer":
                    grade["sufficient"] = True
                    grade["score"] = max(float(grade.get("score") or 0), 80.0)
                    grade["reason"] = "llm_judge: evidence is sufficient"
                    judge = "llm"
                elif verdict == "need_retrieval":
                    grade["sufficient"] = False
                    grade["score"] = min(float(grade.get("score") or 0), 40.0)
                    grade["reason"] = "llm_judge: evidence is insufficient"
                    judge = "llm"
            except Exception:
                pass
        grade["judge"] = judge
        return self._success_update(
            state, "evidence_grade", started, {"evidence_grade": grade}, score=grade["score"]
        )

    def _deep_research(self, state: ConversationStateV44):
        started = time.perf_counter()
        try:
            decision = state.get("router_decision") or {}
            result = self.deep_research_tool.run(
                {
                    "question": decision.get("rewritten_query") or state["message"],
                    "topic": _question_topic(state),
                    "task_id": state.get("task_id") or "",
                    "run_mode": state.get("run_mode") or "fake",
                    "retriever": state.get("retriever") or "arxiv",
                    "output_language": state.get("output_language") or "zh",
                    "expected_keywords": state.get("expected_keywords") or [],
                    "forbidden_keywords": state.get("forbidden_keywords") or [],
                }
            )
            return self._success_update(
                state,
                "deep_research",
                started,
                {"deep_research_result": result},
                task_id=result.get("task_id", ""),
                artifact_uri=result.get("artifact_uri", ""),
            )
        except Exception as error:
            self._error_event(state, "deep_research", started, error)
            raise

    def _answer_with_citations(self, state: ConversationStateV44):
        started = time.perf_counter()
        source = state.get("deep_research_result") or state.get("knowledge_result") or {}
        route = "deep_research" if state.get("deep_research_result") else "existing_knowledge"
        return self._success_update(
            state,
            "answer_with_citations",
            started,
            {
                "answer": source.get("answer", ""),
                "citations": source.get("citations") or [],
                "evidence": source.get("evidence") or [],
                "grounded": bool(source.get("grounded")),
                "retrieval_triggered": bool(
                    source.get("retrieval_triggered") or route == "deep_research"
                ),
                "used_task_id": source.get("task_id") or state.get("task_id") or "",
                "artifact_uri": source.get("artifact_uri", ""),
                "route": route,
                "evidence_grade": source.get("evidence_sufficiency")
                or state.get("evidence_grade")
                or {},
                "retrieval_stack": source.get("retrieval_stack", ""),
                "retrieval_mode": source.get("retrieval_mode", ""),
            },
        )

    def _refuse_or_clarify(self, state: ConversationStateV44):
        started = time.perf_counter()
        if not state.get("allow_deep_research", True):
            answer = "现有证据不足，并且本次请求关闭了深度调研。请补充 task_id 或允许执行调研。"
            route = "refuse_low_evidence"
        else:
            answer = "请明确你希望基于现有知识库回答，还是允许我启动新的论文调研任务。"
            route = "clarify"
        return self._success_update(
            state,
            "refuse_or_clarify",
            started,
            {
                "answer": answer,
                "citations": [],
                "evidence": [],
                "grounded": False,
                "retrieval_triggered": False,
                "route": route,
            },
        )

    def _memory_candidate_write(self, state: ConversationStateV44):
        started = time.perf_counter()
        result = self.memory_service.ingest_message(
            namespace=state["namespace"],
            message=state["message"],
            source_message_id=state.get("source_message_id") or state["request_id"],
            subject=state.get("user_id") or "local-user",
        )
        return self._success_update(
            state,
            "memory_candidate_write",
            started,
            {"memory_write": result},
            write_status=result.get("status", ""),
        )

    def _final_trace(self, state: ConversationStateV44):
        started = time.perf_counter()
        return self._success_update(
            state, "final_trace", started, {"status": "succeeded"}, route=state.get("route", "")
        )

    @staticmethod
    def _after_classify(state: ConversationStateV44):
        if (state.get("router_decision") or {}).get("tool") == "clarify":
            return "refuse_or_clarify"
        return "memory_recall"

    @staticmethod
    def _after_memory_recall(state: ConversationStateV44):
        message = state.get("message") or ""
        results = (state.get("memory_recall") or {}).get("results") or []
        decision = state.get("router_decision") or {}
        if _is_memory_question(message):
            return "memory_answer" if results else "casual_chat"
        if _is_explicit_memory_write(message):
            return "casual_chat"
        if decision.get("tool") == "chat_fallback":
            return "casual_chat"
        if decision.get("tool") == "clarify":
            return "refuse_or_clarify"
        return "knowledge_retrieval"

    @staticmethod
    def _after_evidence_grade(state: ConversationStateV44):
        if (state.get("evidence_grade") or {}).get("sufficient"):
            return "answer_with_citations"
        if state.get("allow_deep_research", True):
            return "deep_research"
        return "refuse_or_clarify"

    def _success_update(self, state, node, started, values, **details):
        event = _node_event(
            state, node, "success", started, details=details
        )
        self._append_trace(state["thread_id"], event)
        return dict(
            values,
            node_events=list(state.get("node_events") or []) + [event],
            executed_nodes=list(state.get("executed_nodes") or []) + [node],
        )

    def _error_event(self, state, node, started, error):
        event = _node_event(
            state,
            node,
            "error",
            started,
            details={"error": repr(error)},
        )
        self._append_trace(state["thread_id"], event)

    def _append_trace(self, thread_id: str, event: Dict):
        path = self.trace_dir / "{0}.jsonl".format(_safe_digest(thread_id))
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _public_result(self, state: ConversationStateV44):
        config = {"configurable": {"thread_id": state["thread_id"]}}
        checkpoint_count = sum(1 for _ in self.graph.get_state_history(config, limit=200))
        trace_events = self._request_trace_events(
            state["thread_id"], state["request_id"]
        )
        return {
            "runtime": self.runtime_name,
            "thread_id": state["thread_id"],
            "request_id": state["request_id"],
            "status": state.get("status", ""),
            "route": state.get("route", ""),
            "answer": state.get("answer", ""),
            "citations": state.get("citations") or [],
            "evidence": state.get("evidence") or [],
            "grounded": bool(state.get("grounded")),
            "retrieval_triggered": bool(state.get("retrieval_triggered")),
            "used_task_id": state.get("used_task_id") or "",
            "artifact_uri": state.get("artifact_uri") or "",
            "router_decision": state.get("router_decision") or {},
            "memory_recall": state.get("memory_recall") or {},
            "memory_write": state.get("memory_write") or {},
            "evidence_grade": state.get("evidence_grade") or {},
            "executed_nodes": state.get("executed_nodes") or [],
            "node_events": trace_events or state.get("node_events") or [],
            "checkpoint_count": checkpoint_count,
            "idempotent_replay": False,
            "retrieval_stack": state.get("retrieval_stack", ""),
            "retrieval_mode": state.get("retrieval_mode", ""),
        }

    def _request_trace_events(self, thread_id: str, request_id: str):
        path = self.trace_dir / "{0}.jsonl".format(_safe_digest(thread_id))
        if not path.exists():
            return []
        events = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("request_id") == request_id:
                events.append(event)
        return events

    def _result_path(self, thread_id: str, request_id: str):
        key = "{0}\0{1}".format(thread_id, request_id)
        return self.result_dir / "{0}.json".format(_safe_digest(key))

    def _read_result(self, thread_id: str, request_id: str):
        path = self._result_path(thread_id, request_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_result(self, thread_id: str, request_id: str, result: Dict):
        path = self._result_path(thread_id, request_id)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)


def _node_event(state, node, status, started, details=None):
    return {
        "event": "graph_node_{0}".format(status),
        "span_id": uuid.uuid4().hex,
        "runtime": "langgraph-v4.4",
        "thread_id": state.get("thread_id", ""),
        "request_id": state.get("request_id", ""),
        "node": node,
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_ms": round((time.perf_counter() - started) * 1000, 4),
        "details": details or {},
    }


def _snapshot_payload(snapshot, include_values=True):
    payload = {
        "next": list(snapshot.next or []),
        "config": snapshot.config or {},
        "metadata": snapshot.metadata or {},
        "created_at": snapshot.created_at or "",
        "parent_config": snapshot.parent_config or {},
    }
    if include_values:
        payload["values"] = dict(snapshot.values or {})
    return payload


def _casual_answer(message: str, memory_recall: Dict):
    text = str(message or "").lower()
    if _is_memory_question(text):
        return "我目前没有召回与你这个问题相关的长期记忆。你可以用“请记住：...”明确保存稳定偏好或事实。"
    if _is_explicit_memory_write(text):
        return "我会通过 V4.3 Memory Policy 校验这条信息；符合稳定事实、偏好或规则时才会跨会话保存。"
    if "你是谁" in text or "模型" in text:
        return "我是 PaperStorm 的 LangGraph Conversation Runtime 演示层，基础模型由运行时配置决定。"
    if "能做什么" in text or "可以做什么" in text or "介绍一下" in text:
        return (
            "我是 PaperStorm Research Agent，可以陪你闲聊、回答论文调研与技术问题，"
            "也能基于 arXiv/本地 PDF 做深度调研、生成带引用的中文综述，并管理跨会话记忆。"
            "你可以直接问‘PIM 是什么？’试试深度调研，或者问‘你能做什么？’了解能力边界。"
        )
    if any(token in text for token in ["面试", "求职", "简历", "offer", "hr"]):
        return (
            "面试准备可以按四条主线来：1) 项目定位——PaperStorm 是在 Stanford STORM 上做的"
            "工程化增强，不是从零写聊天机器人；2) 实现细节——RAG（BM25+Dense+RRF）、可恢复"
            " Context、可治理 Memory、LangGraph 编排、SQLite WAL 治理；3) 数据说话——seed 集"
            "检索 Recall@K 从 0.36 提到 0.78~0.99，Context 压缩省 66% 且可恢复；4) 边界——"
            "本地治理基线与真实生产的区别。想深入哪条，我可以展开讲。"
        )
    if any(token in text for token in ["人话", "听不懂", "说人话", "换个说法", "风格"]):
        return (
            "抱歉刚才回答太像说明书了。简单说：我能帮你做三件事——闲聊、用已有资料回答"
            "技术问题、需要时自动去检索论文再回答。你想聊哪个方向，我换个更自然的说法陪你聊。"
        )
    if _is_greeting(message):
        return _greeting_reply(message)
    return "你好，我是 PaperStorm Research Agent。你可以闲聊、查询长期记忆、问已有知识库，或启动论文调研与深度研究。"


def _casual_chat_prompt(state: ConversationStateV44) -> str:
    memory = state.get("memory_recall") or {}
    memory_lines = [
        "- {0}".format(item.get("content", ""))
        for item in (memory.get("results") or [])[:3]
    ]
    topic = str(state.get("topic") or "").strip()
    history_lines = []
    for item in (state.get("context_window") or [])[-6:]:
        role = item.get("role") or ""
        if role == "user":
            label = "用户"
        elif role == "assistant":
            label = "助手"
        else:
            label = "系统"
        content = str(item.get("content") or "")[:200]
        history_lines.append("{0}: {1}".format(label, content))
    return (
        "你是 PaperStorm Research Agent 的聊天回复生成器。用户可能在闲聊、问系统能力，"
        "或聊面试/求职话题。请用自然、简洁、有温度的中文回复（3-5 句），不要提内部实现"
        "细节，不要编造不存在的功能。如果用户提到面试准备，可以基于项目背景给出可执行的建议。\n"
        "这是同一会话的连续对话，你有完整的会话上下文（不是没有记忆），请自然地接着聊。\n"
        "如果你能直接回答，就直接回答；只有当你认为必须检索外部资料/论文才能回答时，"
        "才只回复一行：{0}\n"
        "最近对话记录：\n{1}\n"
        "会话主题（仅供背景，不要直接复述）：{2}\n"
        "跨会话记忆（仅供参考）：{3}\n"
        "用户消息：{4}\n"
        "回复：".format(
            RETRIEVE_MARKER,
            "\n".join(history_lines) or "（无）",
            topic or "无",
            "\n".join(memory_lines) or "无",
            str(state.get("message") or ""),
        )
    )


def _evidence_judge_prompt(
    question: str,
    topic: str,
    expected_keywords: List[str],
    evidence: List[Dict],
) -> str:
    lines = [
        "你是严谨的证据裁判。只判断现有检索证据能否回答用户问题，不要生成答案。",
        "用户问题：{0}".format(question),
    ]
    if topic:
        lines.append("会话主题：{0}".format(topic))
    if expected_keywords:
        lines.append("期望关键词：{0}".format("、".join(expected_keywords)))
    lines.append("检索到的证据：")
    for index, item in enumerate((evidence or [])[:5], start=1):
        content = "{0}：{1}".format(
            str(item.get("title") or ""),
            str(item.get("content") or "")[:220],
        )
        lines.append("[{0}] {1}".format(index, content))
    lines.append("只回复三个词之一：可以回答 / 需要更多检索 / 无法回答")
    return "\n".join(lines)


def _parse_judge_verdict(text: str) -> Optional[str]:
    lowered = str(text or "").lower()
    if any(marker in lowered for marker in ("可以回答", "能回答", "足够")):
        return "can_answer"
    if any(
        marker in lowered
        for marker in ("需要更多检索", "需要检索", "无法回答", "不能回答", "不足", "无法")
    ):
        return "need_retrieval"
    return None


def _question_topic(state: ConversationStateV44) -> str:
    """Use the session topic only when the question still relates to it;
    otherwise let a fresh research task follow the question itself."""
    message = str(state.get("message") or "")
    rewritten = str(
        (state.get("router_decision") or {}).get("rewritten_query") or message
    )
    session_topic = str(state.get("topic") or "").strip()
    if session_topic and _meaningful_overlap(session_topic, rewritten):
        return session_topic
    return rewritten


def _meaningful_overlap(left: str, right: str) -> bool:
    from .paperstorm_retrieval_runtime import meaningful_terms

    return bool(meaningful_terms(left) & meaningful_terms(right))


def _needs_research_fallback(state: ConversationStateV44) -> bool:
    """Local safety net: escalate when the message clearly needs retrieval and
    the chat layer produced no LLM answer (e.g. offline fake mode)."""
    from .paperstorm_intent_router import route_high_confidence_rules

    decision = route_high_confidence_rules(
        str(state.get("message") or ""),
        {
            "topic": str(state.get("topic") or ""),
            "task_id": str(state.get("task_id") or ""),
        },
        state.get("context_window") or [],
    )
    return bool(
        decision and decision.get("intent") in {"research_qa", "run_research"}
    )


def _is_greeting(message: str) -> bool:
    normalized = str(message or "").strip().lower().strip(" ！!。？?～~")
    return normalized in {
        "你好",
        "您好",
        "hello",
        "hi",
        "hey",
        "早上好",
        "晚上好",
        "下午好",
        "莫西莫西",
        "もしもし",
    }


def _greeting_reply(message: str) -> str:
    variants = [
        "你好呀！我是 PaperStorm，论文调研和知识库问答都能帮你。想聊点啥？",
        "嗨，我在呢。你可以问我技术问题，也可以让我去检索论文，或者就是随便聊聊。",
        "你好！直接说需求就行——提问、调研论文、管理记忆都可以。",
        "哈喽！有什么想聊的？论文、面试、技术问题都行。",
    ]
    index = sum(ord(character) for character in str(message or "")) % len(variants)
    return variants[index]


def _is_memory_question(message: str):
    lowered = str(message or "").lower()
    return any(token in lowered for token in ["记得", "偏好", "之前", "上次", "remember"])


def _is_explicit_memory_write(message: str):
    lowered = str(message or "").lower()
    return any(token in lowered for token in ["请记住", "记住：", "偏好", "以后必须", "remember that"])


def _memory_namespace(user_id: str):
    raw = str(user_id or "local-user").strip().lower()
    value = "".join(
        character
        if character.isascii() and (character.isalnum() or character in "._-")
        else "-"
        for character in raw
    ).strip("-.")
    if not value:
        value = "user-{0}".format(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12])
    return "user/{0}".format(value[:128])


def _safe_digest(value: str):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _model_dump(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()
