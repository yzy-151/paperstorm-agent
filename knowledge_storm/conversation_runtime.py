import hashlib
import json
import logging
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
from .memory_store import LongTermMemoryService
from .paperstorm_research_qa import evaluate_evidence_sufficiency


logging.getLogger("langgraph.pregel._retry").setLevel(logging.WARNING)

RETRIEVE_MARKER = "__NEED_RESEARCH__"


class ConversationRequest(BaseModel):
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
    memory_retrieval_mode: str = "lexical"


class ConversationState(TypedDict, total=False):
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
    memory_retrieval_mode: str
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
    retrieval_metadata: Dict
    escalate_to_retrieval: bool
    llm_call: Dict
    llm_error: Dict


class StormDeepResearchTool:
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
            "history": {"type": "array", "items": {"type": "object"}},
            "search_plan": {"type": "object"},
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
            "history",
            "search_plan",
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
            "retrieval_metadata": result.get("retrieval_metadata") or {},
            "llm_call": {},
            "llm_error": {},
        }


class PaperStormConversationRuntime:
    runtime_name = "conversation-runtime"

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
            Path(task_service.root_dir) / "memory_service"
        )
        self.deep_research_tool = deep_research_tool or StormDeepResearchTool(
            task_service
        )
        self._connection = None
        self.checkpointer = None
        self.graph = None

    def invoke(self, **payload):
        request = ConversationRequest(**payload)
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
            "retrieval_metadata": {},
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
                ["memory_recall", "casual_chat|knowledge_retrieval|refuse_or_clarify"],
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
            "long_term_memory": "PaperStorm namespace memory store",
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
        builder = StateGraph(ConversationState)
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
            {
                "casual_chat": "casual_chat",
                "memory_recall": "memory_recall",
                "knowledge_retrieval": "knowledge_retrieval",
                "deep_research": "deep_research",
                "refuse_or_clarify": "refuse_or_clarify",
            },
        )
        builder.add_conditional_edges(
            "memory_recall",
            self._after_memory_recall,
            {
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
            "answer_with_citations",
            "refuse_or_clarify",
        ]:
            builder.add_edge(node, "memory_candidate_write")
        builder.add_edge("memory_candidate_write", "final_trace")
        builder.add_edge("final_trace", END)
        return builder.compile(checkpointer=self.checkpointer)

    def _classify(self, state: ConversationState):
        started = time.perf_counter()
        try:
            memory_preview = self.memory_service.search(
                state["namespace"], state["message"], top_k=3
            )
            decision = self.intent_router.route(
                message=state["message"],
                session={
                    "topic": state.get("topic", ""),
                    "task_id": state.get("task_id", ""),
                    "expected_keywords": state.get("expected_keywords") or [],
                    "forbidden_keywords": state.get("forbidden_keywords") or [],
                },
                context_window=state.get("context_window") or [],
                memory_context=memory_preview,
            )
            decision = _enforce_response_contract(
                decision,
                task_id=state.get("task_id") or "",
                message=state.get("message") or "",
            )
            return self._success_update(
                state,
                "classify",
                started,
                {"router_decision": decision, "planner_memory_preview": memory_preview},
            )
        except Exception as error:
            self._error_event(state, "classify", started, error)
            raise

    def _memory_recall(self, state: ConversationState):
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
                input=str(state.get("message") or "")[:500],
                activity="检索跨会话长期记忆",
                retrieval_mode=recalled.get("retrieval_mode", ""),
                embedding_backend=recalled.get("embedding_backend", ""),
            )
        except Exception as error:
            self._error_event(state, "memory_recall", started, error)
            raise

    def _casual_chat(self, state: ConversationState):
        started = time.perf_counter()
        decision = state.get("router_decision") or {}
        escalate = False
        llm_call = {}
        llm_error = {}
        offline_routing = decision.get("planner_status") == "offline_fallback"
        if offline_routing and _is_explicit_memory_write(state["message"]):
            answer = _casual_answer(state["message"], state.get("memory_recall") or {})
            decision = {
                "intent": "memory_write",
                "need_retrieval": False,
                "tool": "memory_write",
                "rewritten_query": state["message"],
                "confidence": 0.98,
                "reason": "explicit durable memory instruction",
                "router": "conversation_memory_policy",
            }
        else:
            llm_call = self._casual_answer(state)
            answer = str(llm_call.get("content") or "")
            llm_error = llm_call.get("error") or {}
            if llm_error:
                answer = _llm_failure_message(llm_error)
            if answer == RETRIEVE_MARKER:
                if _can_escalate_to_retrieval(state):
                    answer = ""
                    escalate = True
                else:
                    answer = (
                        "当前回合没有获得外部检索或新调研授权。"
                        "我不会自行启动高成本工具；请明确允许检索后再继续。"
                    )
                    escalate = False
            elif not answer and not llm_error:
                answer = _casual_answer(state["message"], state.get("memory_recall") or {})
                escalate = False
        telemetry = {
            key: llm_call.get(key)
            for key in (
                "finish_reason",
                "usage",
                "cost_usd",
                "latency_ms",
                "output_budget",
                "segments",
                "truncated",
                "error",
            )
            if key in llm_call
        }
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
                "llm_call": telemetry,
                "llm_error": llm_error,
            },
            input=str(state.get("message") or "")[:500],
            activity=(decision.get("response_contract") or {}).get("task", "直接回复"),
            **telemetry,
        )

    def _casual_answer(self, state: ConversationState) -> Dict:
        """Generate a typed reply result; never hide provider failures."""
        if self.chat_llm is not None:
            try:
                reply = self._chat_llm_answer(state)
                if reply.get("content") or reply.get("error"):
                    return reply
            except Exception as error:
                from .paperstorm_router_llm import classify_llm_error

                return {
                    "content": "",
                    "finish_reason": "error",
                    "usage": {},
                    "cost_usd": 0.0,
                    "latency_ms": 0.0,
                    "output_budget": 0,
                    "segments": 0,
                    "truncated": False,
                    "error": {
                        "type": classify_llm_error(error),
                        "message": str(error),
                        "recoverable": True,
                    },
                }
        return {
            "content": _casual_answer(
                state.get("message") or "", state.get("memory_recall") or {}
            ),
            "finish_reason": "local_fallback",
            "usage": {},
            "cost_usd": 0.0,
            "latency_ms": 0.0,
            "output_budget": 0,
            "segments": 1,
            "truncated": False,
            "error": None,
        }

    def _chat_llm_answer(self, state: ConversationState) -> Dict:
        from .paperstorm_router_llm import select_output_budget

        contract = (state.get("router_decision") or {}).get("response_contract") or {}
        budget = select_output_budget(state.get("message") or "", contract)
        try:
            value = self.chat_llm(
                _casual_chat_prompt(state),
                response_contract=contract,
                user_message=state.get("message") or "",
                output_budget=budget,
            )
        except TypeError:
            value = self.chat_llm(_casual_chat_prompt(state))
        if isinstance(value, dict):
            result = dict(value)
            result.setdefault("output_budget", budget)
            return result
        return {
            "content": str(value or "").strip(),
            "finish_reason": "unknown",
            "usage": {},
            "cost_usd": 0.0,
            "latency_ms": 0.0,
            "output_budget": budget,
            "segments": 1,
            "truncated": False,
            "error": None,
        }

    @staticmethod
    def _after_casual_chat(state: ConversationState):
        """Answer-first: escalate to retrieval only when the chat layer says so."""
        if state.get("escalate_to_retrieval"):
            return "knowledge_retrieval"
        return "memory_candidate_write"

    def _knowledge_retrieval(self, state: ConversationState):
        started = time.perf_counter()
        try:
            task_id = state.get("task_id") or ""
            question = _tool_query(
                state.get("router_decision") or {}, "evidence.search"
            ) or state["message"]
            if not task_id:
                result = {"answer": "", "citations": [], "evidence": [], "grounded": False}
            else:
                result = self.task_service.query_knowledge_base(
                    task_id,
                    question=question,
                    top_k=3,
                    history=state.get("context_window") or [],
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

    def _evidence_grade(self, state: ConversationState):
        started = time.perf_counter()
        result = state.get("knowledge_result") or {}
        search_plan = (result.get("retrieval_metadata") or {}).get("search_plan") or {}
        question = search_plan.get("standalone_query") or state["message"]
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

    def _deep_research(self, state: ConversationState):
        started = time.perf_counter()
        try:
            decision = state.get("router_decision") or {}
            result = self.deep_research_tool.run(
                {
                    "question": decision.get("rewritten_query") or state["message"],
                    "topic": _question_topic(state),
                    # This node is reached only after evidence_grade rejected the
                    # existing KB. Start fresh without querying that task again.
                    "task_id": "",
                    "run_mode": state.get("run_mode") or "fake",
                    "retriever": state.get("retriever") or "arxiv",
                    "output_language": state.get("output_language") or "zh",
                    "expected_keywords": state.get("expected_keywords") or [],
                    "forbidden_keywords": state.get("forbidden_keywords") or [],
                    "history": state.get("context_window") or [],
                    "search_plan": (
                        ((state.get("knowledge_result") or {}).get("retrieval_metadata") or {}).get(
                            "search_plan"
                        )
                        or None
                    ),
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

    def _answer_with_citations(self, state: ConversationState):
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
                "retrieval_metadata": source.get("retrieval_metadata") or {},
            },
        )

    def _refuse_or_clarify(self, state: ConversationState):
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

    def _memory_candidate_write(self, state: ConversationState):
        started = time.perf_counter()
        decision = state.get("router_decision") or {}
        source_message_id = state.get("source_message_id") or state["request_id"]
        arguments = _tool_arguments(decision, "memory.write")
        if arguments:
            result = self.memory_service.ingest_structured(
                namespace=state["namespace"],
                content=arguments.get("content") or "",
                canonical_key=arguments.get("canonical_key") or "",
                memory_type=arguments.get("memory_type") or "semantic",
                source_message_id=source_message_id,
                subject=state.get("user_id") or "local-user",
                confidence=arguments.get("confidence") or 0.99,
                importance=arguments.get("importance") or 0.8,
            )
        else:
            result = self.memory_service.ingest_message(
                namespace=state["namespace"],
                message=state["message"],
                source_message_id=source_message_id,
                subject=state.get("user_id") or "local-user",
            )
        return self._success_update(
            state,
            "memory_candidate_write",
            started,
            {"memory_write": result},
            write_status=result.get("status", ""),
        )

    def _final_trace(self, state: ConversationState):
        started = time.perf_counter()
        return self._success_update(
            state, "final_trace", started, {"status": "succeeded"}, route=state.get("route", "")
        )

    @staticmethod
    def _after_classify(state: ConversationState):
        decision = state.get("router_decision") or {}
        action = decision.get("action")
        if action == "clarify":
            return "refuse_or_clarify"
        if action == "respond":
            return "casual_chat"
        tool_name = _first_tool_name(decision)
        if not _tool_is_authorized(decision, tool_name):
            return "refuse_or_clarify"
        return {
            "memory.search": "memory_recall",
            "memory.write": "casual_chat",
            "evidence.search": "knowledge_retrieval",
            "research.start": "deep_research",
        }.get(tool_name, "refuse_or_clarify")

    @staticmethod
    def _after_memory_recall(state: ConversationState):
        decision = state.get("router_decision") or {}
        if _first_tool_name(decision) == "memory.search":
            # Memory is retrieved context, never a phrase-selected terminal.
            # The response model uses the planner contract to finish the task.
            return "casual_chat"
        return "refuse_or_clarify"

    @staticmethod
    def _after_evidence_grade(state: ConversationState):
        if (state.get("evidence_grade") or {}).get("sufficient"):
            return "answer_with_citations"
        if _research_is_authorized(state):
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
        details = {
            "error": repr(error),
            "exception_type": type(error).__name__,
            "message": str(error),
        }
        error_type = getattr(error, "error_type", None)
        if error_type:
            details["error_type"] = str(error_type)
        event = _node_event(
            state,
            node,
            "error",
            started,
            details=details,
        )
        self._append_trace(state["thread_id"], event)

    def _append_trace(self, thread_id: str, event: Dict):
        path = self.trace_dir / "{0}.jsonl".format(_safe_digest(thread_id))
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _public_result(self, state: ConversationState):
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
            "retrieval_metadata": state.get("retrieval_metadata") or {},
            "llm_call": state.get("llm_call") or {},
            "llm_error": state.get("llm_error") or {},
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
        "runtime": "conversation-runtime",
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
    if _is_explicit_memory_write(text):
        return "我会通过长期记忆写入策略校验这条信息；符合稳定事实、偏好或规则时才会跨会话保存。"
    recalled = (memory_recall or {}).get("results") or []
    if recalled:
        return "我从你的跨会话长期记忆中找到了：{0}".format(
            "；".join(str(item.get("content") or "") for item in recalled[:3])
        )
    if "你是谁" in text or "模型" in text:
        return "我是 PaperStorm 的 LangGraph Conversation Runtime 演示层，基础模型由运行时配置决定。"
    if any(token in text for token in ["逻辑", "实现", "流程", "知识库", "工作方式"]):
        return (
            "知识库问答的流程是：问题进来先做意图路由和记忆召回，然后用混合检索"
            "（BM25+Dense+RRF）从当前任务或文档里找证据，证据裁判判断够不够；"
            "够就带着引用编号生成中文回答；新调研必须由 Planner 明确授权。"
            "你想问某一步的细节，我可以展开讲。"
        )
    if "能做什么" in text or "可以做什么" in text or "介绍一下" in text:
        return (
            "我是 PaperStorm Research Agent，可以陪你聊天、回答论文调研与技术问题，"
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
            "抱歉刚才回答太像说明书了。简单说：我能帮你做三件事——聊天、用已有资料回答"
            "技术问题、需要时自动去检索论文再回答。你想聊哪个方向，我换个更自然的说法陪你聊。"
        )
    if _is_greeting(message):
        return _greeting_reply(message)
    if str(message or "").strip().lower().startswith(
        ("继续", "接着", "续写", "往下写", "continue")
    ):
        return "当前未连接可用的文本生成模型，无法可靠续写。请切换到真实 API 后重试；已有上下文不会被清空。"
    return "当前本地回退模式无法可靠生成这类开放回答。请切换到真实 API，或更明确地说明需要检索的资料。"


def _casual_chat_prompt(state: ConversationState) -> str:
    memory = state.get("memory_recall") or {}
    memory_lines = [
        "- {0}".format(item.get("content", ""))
        for item in (memory.get("results") or [])[:3]
    ]
    history_lines = []
    for item in (state.get("context_window") or [])[-24:]:
        role = item.get("role") or ""
        if role == "user":
            label = "用户"
        elif role == "assistant":
            label = "助手"
        else:
            label = "系统"
        content = str(item.get("content") or "")[:1200]
        history_lines.append("{0}: {1}".format(label, content))
    decision = state.get("router_decision") or {}
    contract = decision.get("response_contract") or {}
    return (
        "你是 PaperStorm Research Agent 的聊天回复生成器。用户可能在聊天、问系统能力，"
        "也可能要求创作或讨论技术。请用自然、简洁、有温度的中文回复，不要提内部实现"
        "细节；用户问到算法/实现细节时，按【系统事实】如实简要回答，不要编造，"
        "也不要主动展开未问到的内容。不要编造不存在的功能。除非用户明确询问身份，"
        "禁止用‘你好，我是 PaperStorm’或类似自我介绍作为回答或错误回退。"
        "用户问系统自身（算法、知识库逻辑、实现细节）时，必须直接按【系统事实】回答，"
        "禁止使用检索标记。\n"
        "【系统事实】\n"
        "- 检索算法：默认 BM25（稀疏）+ Dense 向量 + RRF 融合的混合检索，"
        "可选 Cross-Encoder 二次重排；真实语义向量模型可用时自动启用。\n"
        "- 动作规划：真实模式由 LLM Turn Planner 结构化选择工具，Runtime 单独授权；"
        "规则仅供离线演示，Planner 失败时禁止外部工具。\n"
        "- 证据判定：LLM 证据裁判判断已有证据能否回答；证据不足时只有本轮已经授权"
        "新调研才执行，否则澄清或拒绝。\n"
        "- 知识库问答：意图路由 → 记忆召回 → 混合检索 → 证据裁判 → 带引用回答；"
        "新调研作为独立高成本工具，需要显式授权。\n"
        "- 记忆：短期对话、FTS5 跨会话历史、长期用户事实和论文证据彼此隔离。\n"
        "- 当前运行模式：{run_mode}（fake=本地模拟调研；paperstorm=真实检索+LLM）。\n"
        "这是同一会话的连续对话，你有完整的会话上下文（不是没有记忆），请自然地接着聊。\n"
        "如果你能直接回答，就直接回答；只有当你认为必须检索外部资料/论文才能回答时，"
        "才只回复一行：{0}\n"
        "最近对话记录：\n{1}\n"
        "跨会话记忆（仅供参考）：{2}\n"
        "本轮响应契约：{3}\n"
        "用户消息：{4}\n"
        "回复：".format(
            RETRIEVE_MARKER,
            "\n".join(history_lines) or "（无）",
            "\n".join(memory_lines) or "无",
            json.dumps(contract, ensure_ascii=False),
            str(state.get("message") or ""),
            run_mode=str(state.get("run_mode") or "fake"),
        )
    )


def _first_tool_name(decision: Dict) -> str:
    calls = (decision or {}).get("tool_calls") or []
    if calls and isinstance(calls[0], dict):
        return str(calls[0].get("name") or "")
    legacy = str((decision or {}).get("tool") or "")
    return {
        "memory_search": "memory.search",
        "research_qa": "evidence.search",
        "kb_qa": "evidence.search",
        "paper_research": "research.start",
    }.get(legacy, "")


def _tool_query(decision: Dict, tool_name: str) -> str:
    for call in (decision or {}).get("tool_calls") or []:
        if not isinstance(call, dict) or call.get("name") != tool_name:
            continue
        arguments = call.get("arguments") or {}
        if isinstance(arguments, dict):
            return str(arguments.get("query") or "").strip()
    return ""


def _tool_arguments(decision: Dict, tool_name: str) -> Dict:
    for call in (decision or {}).get("tool_calls") or []:
        if isinstance(call, dict) and call.get("name") == tool_name:
            arguments = call.get("arguments") or {}
            return dict(arguments) if isinstance(arguments, dict) else {}
    return {}


def _enforce_response_contract(decision: Dict, task_id: str, message: str) -> Dict:
    """Bind citation-bearing responses to provenance before generation.

    This is an execution invariant over the planner's structured contract, not
    semantic intent classification. The planner remains free to describe any
    task; the Runtime only prevents an ungrounded response from claiming
    citations when an existing evidence store is available.
    """
    output = dict(decision or {})
    contract = output.get("response_contract") or {}
    if (
        output.get("action") == "respond"
        and bool(contract.get("requires_citations"))
        and str(task_id or "").strip()
        and (output.get("authorization") or {}).get("evidence.search") == "allowed"
    ):
        query = str(output.get("rewritten_query") or message or "").strip()
        output["action"] = "tool_call"
        output["tool_calls"] = [
            {
                "name": "evidence.search",
                "arguments": {"query": query},
            }
        ]
        output["intent"] = "research_qa"
        output["tool"] = "research_qa"
        output["need_retrieval"] = True
        output["runtime_adjustments"] = list(
            dict.fromkeys(
                list(output.get("runtime_adjustments") or [])
                + ["citation_contract"]
            )
        )
    return output


def _llm_failure_message(error: Dict) -> str:
    error_type = str((error or {}).get("type") or "provider_error")
    labels = {
        "timeout": "模型调用超时",
        "rate_limit": "模型服务触发限流",
        "authentication": "模型 API 认证失败",
        "provider_unavailable": "模型服务当前不可用",
        "invalid_response": "模型返回格式无效",
        "provider_error": "模型调用失败",
    }
    suggestion = "请稍后重试；当前会话和上下文已保留。"
    if error_type == "authentication":
        suggestion = "请检查 API Key 与服务地址；当前会话和上下文已保留。"
    return "{0}。{1}".format(labels.get(error_type, labels["provider_error"]), suggestion)


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


def _question_topic(state: ConversationState) -> str:
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
    from .retrieval_runtime import meaningful_terms

    return bool(meaningful_terms(left) & meaningful_terms(right))


def _tool_is_authorized(decision: Dict, tool_name: str) -> bool:
    if not tool_name:
        return True
    authorization = (decision or {}).get("authorization") or {}
    if authorization:
        return authorization.get(tool_name) == "allowed"
    # Deterministic routing exists only for fake/offline demos. Production
    # planners always emit an authorization map before the graph executes.
    return (decision or {}).get("planner_status") == "offline_fallback"


def _research_is_authorized(state: ConversationState) -> bool:
    if not state.get("allow_deep_research", True):
        return False
    decision = state.get("router_decision") or {}
    if (decision.get("planner_status") or "") == "offline_fallback":
        return True
    # The planner may deliberately search existing evidence first while also
    # authorizing a fresh research task if the evidence grader rejects it.
    # Authorization is the execution boundary; the first selected tool only
    # determines the initial node and must not cancel that staged permission.
    return _tool_is_authorized(decision, "research.start")


def _can_escalate_to_retrieval(state: ConversationState) -> bool:
    decision = state.get("router_decision") or {}
    if (decision.get("planner_status") or "") == "offline_fallback":
        return bool(
            state.get("allow_deep_research", True)
            and decision.get("intent") not in {
                "system_help",
                "clarify",
                "memory_recall",
                "memory_write",
            }
        )
    return _tool_is_authorized(decision, "evidence.search") and (
        _first_tool_name(decision) == "evidence.search"
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
    return any(
        token in lowered
        for token in ["请记住", "记住：", "记住:", "以后必须", "remember that"]
    )


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
