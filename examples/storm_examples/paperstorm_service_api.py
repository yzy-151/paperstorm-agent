"""
Optional FastAPI adapter for the PaperStorm service core.

Example:
    uvicorn examples.storm_examples.paperstorm_service_api:app --reload
"""

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from knowledge_storm.paperstorm_service import PaperStormTaskService


DEFAULT_SERVICE_ROOT = Path("./results/paperstorm_service")
DEFAULT_DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "frontend" / "paperstorm_dashboard"


def create_app(service_root=DEFAULT_SERVICE_ROOT, dashboard_dir=DEFAULT_DASHBOARD_DIR):
    try:
        from fastapi import FastAPI
        from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
        from pydantic import BaseModel, Field
        from starlette.middleware.cors import CORSMiddleware
        from starlette.staticfiles import StaticFiles
    except ImportError as exc:
        raise RuntimeError(
            "FastAPI service adapter requires optional dependencies: fastapi and uvicorn."
        ) from exc

    service = PaperStormTaskService(root_dir=service_root)
    dashboard_dir = Path(dashboard_dir)

    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            service.observability.flush()

    app = FastAPI(title="PaperStorm Agent Service", version="6.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def disable_dashboard_asset_cache(request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path in {"/styles.css", "/app.js", "/sample_data.js"} or path.startswith("/dashboard"):
            response.headers["Cache-Control"] = "no-store, max-age=0"
            response.headers["Pragma"] = "no-cache"
        return response

    @app.exception_handler(PermissionError)
    async def permission_error_handler(_request, error):
        return JSONResponse(status_code=403, content={"detail": str(error)})

    @app.exception_handler(KeyError)
    async def not_found_error_handler(_request, error):
        return JSONResponse(status_code=404, content={"detail": str(error)})

    @app.exception_handler(ValueError)
    async def validation_error_handler(_request, error):
        return JSONResponse(status_code=400, content={"detail": str(error)})

    @app.exception_handler(FileNotFoundError)
    async def file_not_found_error_handler(_request, error):
        return JSONResponse(status_code=404, content={"detail": str(error)})

    if dashboard_dir.exists():
        app.mount(
            "/dashboard",
            StaticFiles(directory=str(dashboard_dir), html=True),
            name="dashboard",
        )

    class ResearchTaskRequest(BaseModel):
        topic: str
        retriever: str = "arxiv"
        output_language: str = "zh"
        run_mode: str = "fake"
        pdf_dir: Optional[str] = None
        llm_provider: str = "deepseek"
        llm_model: str = "flash"
        max_thread_num: int = 1
        max_conv_turn: int = 1
        max_perspective: int = 1
        search_top_k: int = 2
        retrieve_top_k: int = 3
        do_research: bool = True
        do_generate_outline: bool = True
        do_generate_article: bool = True
        do_polish_article: bool = True
        remove_duplicate: bool = False
        generate_pdf: bool = False
        disable_trace: bool = False
        verbose: bool = False
        expected_keywords: list[str] = []
        forbidden_keywords: list[str] = []

    class KnowledgeBaseQueryRequest(BaseModel):
        question: str
        top_k: int = 3

    class ResearchAgentAskRequest(BaseModel):
        question: str
        topic: Optional[str] = None
        task_id: Optional[str] = None
        mode: str = "auto"
        top_k: int = 3
        run_mode: str = "fake"
        retriever: str = "arxiv"
        output_language: str = "zh"
        expected_keywords: list[str] = []
        forbidden_keywords: list[str] = []

    class ChatSessionRequest(BaseModel):
        title: str = ""
        topic: str = ""
        run_mode: str = "fake"
        retriever: str = "arxiv"
        output_language: str = "zh"
        expected_keywords: list[str] = []
        forbidden_keywords: list[str] = []
        context_window_size: int = Field(default=48, ge=2, le=200)
        context_token_limit: int = Field(default=1_000_000, ge=128, le=1_000_000)
        user_id: str = "local-user"
        tenant_id: str = "local"
        memory_enabled: bool = True
        memory_retrieval_mode: str = Field(default="lexical", pattern="^(lexical|semantic)$")

    class ChatMessageRequest(BaseModel):
        message: str

    class ConversationGraphInvokeRequest(BaseModel):
        tenant_id: str = "local"
        thread_id: str
        request_id: str
        user_id: str = "local-user"
        message: str
        topic: str = ""
        task_id: str = ""
        run_mode: str = "fake"
        retriever: str = "arxiv"
        output_language: str = "zh"
        expected_keywords: list[str] = []
        forbidden_keywords: list[str] = []
        context_window: list[dict] = []
        source_message_id: str = ""
        memory_retrieval_mode: str = Field(default="lexical", pattern="^(lexical|semantic)$")

    class CompactContextRequest(BaseModel):
        force: bool = True

    class RestoreContextRequest(BaseModel):
        compaction_id: str

    class MemoryCreateRequest(BaseModel):
        namespace: str
        memory_type: str
        subject: str
        content: str
        canonical_key: str
        source_message_ids: list[str] = []
        confidence: float = Field(default=0.9, ge=0.0, le=1.0)
        importance: float = Field(default=0.7, ge=0.0, le=1.0)
        valid_from: Optional[str] = None
        valid_to: Optional[str] = None
        expires_at: Optional[str] = None
        metadata: dict = {}

    class MemorySearchRequest(BaseModel):
        namespace: str
        query: str
        top_k: int = Field(default=5, ge=1, le=50)

    class MemoryEditRequest(BaseModel):
        namespace: str
        content: str
        confidence: Optional[float] = None
        importance: Optional[float] = None
        expires_at: Optional[str] = None

    class MemorySettingRequest(BaseModel):
        namespace: str
        enabled: bool

    class EnterpriseKnowledgeBaseCreateRequest(BaseModel):
        name: str = "Enterprise Knowledge Base"
        source_paths: list[str] = []
        expected_keywords: list[str] = []
        forbidden_keywords: list[str] = []
        chunk_size: int = 500
        chunk_overlap: int = 100
        embedding_provider: str = "hash"
        tenant_id: str = "local"
        owner_user_id: str = "local-user"
        allowed_user_ids: list[str] = []

    class EnterpriseKnowledgeBaseFromZoteroRequest(BaseModel):
        zotero_root: Optional[str] = None
        query_terms: list[str] = []
        max_papers: int = Field(default=8, ge=1, le=100)
        name: str = "Zotero 论文知识库"
        expected_keywords: list[str] = []
        forbidden_keywords: list[str] = []
        chunk_size: int = Field(default=500, ge=64, le=4000)
        chunk_overlap: int = Field(default=100, ge=0, le=500)
        embedding_provider: str = "hash"
        tenant_id: str = "local"
        owner_user_id: str = "local-user"
        allowed_user_ids: list[str] = []

    class EnterpriseKnowledgeBaseAskRequest(BaseModel):
        question: str
        top_k: int = 4
        tenant_id: str = "local"
        user_id: str = "local-user"

    class EnterpriseKnowledgeBaseUpdateRequest(BaseModel):
        source_paths: list[str]
        tenant_id: str = "local"
        user_id: str = "local-user"
        idempotency_key: str

    class ProductionBenchmarkRequest(BaseModel):
        request_count: int = Field(default=100, ge=10, le=10000)

    class EvaluationDatasetV54Request(BaseModel):
        dataset_path: str

    class EvaluationReviewV54Request(BaseModel):
        query_validity: str
        edited_query: str = ""
        relevant_document_ids: list[str] = []
        evidence_sufficiency: str
        reviewer_notes: str = ""

    class EvaluationRetrievalV54Request(BaseModel):
        embedding: str = "hash"
        top_k: int = Field(default=5, ge=1, le=20)
        candidate_k: int = Field(default=20, ge=5, le=200)
        configurations: list[str] = ["bm25", "dense", "hybrid"]
        enable_reranker: bool = False

    class BenchmarkRunRequest(BaseModel):
        benchmark_id: str
        profile: str = "smoke"
        allow_paid_llm: bool = False

    @app.get("/")
    def get_dashboard_home():
        index_path = dashboard_dir / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return {"service": "PaperStorm Agent Service", "dashboard": "not found"}

    @app.get("/styles.css")
    def get_dashboard_styles():
        return _dashboard_file_response(dashboard_dir, "styles.css")

    @app.get("/app.js")
    def get_dashboard_app_js():
        return _dashboard_file_response(dashboard_dir, "app.js")

    @app.get("/sample_data.js")
    def get_dashboard_sample_data_js():
        return _dashboard_file_response(dashboard_dir, "sample_data.js")

    @app.get("/sample_data.json")
    def get_dashboard_sample_data_json():
        return _dashboard_file_response(dashboard_dir, "sample_data.json")

    @app.get("/benchmarks/catalog")
    def get_benchmark_catalog():
        return service.get_benchmark_catalog()

    @app.get("/observability/status")
    def get_observability_status():
        return service.get_observability_status()

    @app.post("/benchmarks/runs")
    def start_benchmark_run(request: BenchmarkRunRequest):
        return service.start_benchmark_run(
            benchmark_id=request.benchmark_id,
            profile=request.profile,
            allow_paid_llm=request.allow_paid_llm,
        )

    @app.get("/benchmarks/runs/{run_id}")
    def get_benchmark_run(run_id: str):
        return service.get_benchmark_run(run_id)

    @app.post("/benchmarks/runs/{run_id}/cancel")
    def cancel_benchmark_run(run_id: str):
        return service.cancel_benchmark_run(run_id)

    @app.get("/events")
    def stream_events(task_id: Optional[str] = None, once: bool = False):
        def event_stream():
            yield _sse_event(
                "service",
                {
                    "status": "connected",
                    "message": "PaperStorm SSE connected",
                    "task_id": task_id or "",
                },
            )
            last_status = None
            seen_trace_events = 0
            while True:
                payload = {
                    "status": "heartbeat",
                    "task_count": len(service.list_tasks()),
                    "task_id": task_id or "",
                    "timestamp": time.time(),
                }
                if task_id:
                    try:
                        task = service.get_task(task_id)
                        payload["task_status"] = task.get("status", "")
                        payload["topic"] = task.get("topic", "")
                        payload["error"] = task.get("error", "")
                        payload["artifacts"] = task.get("artifacts", {})
                    except KeyError:
                        payload["task_status"] = "unknown"
                    try:
                        trace_events = service.get_trace(task_id).get("events") or []
                    except (KeyError, FileNotFoundError):
                        trace_events = []
                    for trace_event in trace_events[seen_trace_events:]:
                        yield _sse_event(
                            "trace",
                            {"task_id": task_id, "trace": trace_event},
                        )
                    seen_trace_events = len(trace_events)
                    if payload.get("task_status") != last_status:
                        last_status = payload.get("task_status")
                        yield _sse_event("task_status", payload)
                yield _sse_event("heartbeat", payload)
                if once:
                    break
                time.sleep(2)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/research-tasks")
    def submit_research_task(request: ResearchTaskRequest):
        return service.submit_research_task(**_request_payload(request))

    @app.get("/research-tasks")
    def list_research_tasks(status: Optional[str] = None):
        return {"tasks": service.list_tasks(status=status)}

    @app.post("/research-tasks/{task_id}/run")
    def run_research_task(task_id: str):
        return service.run_task(task_id)

    @app.get("/research-tasks/{task_id}")
    def get_research_task(task_id: str):
        return service.get_task(task_id)

    @app.get("/research-tasks/{task_id}/article")
    def get_research_article(task_id: str):
        return service.get_article(task_id)

    @app.get("/research-tasks/{task_id}/scorecard")
    def get_research_scorecard(task_id: str):
        return service.get_scorecard(task_id)

    @app.get("/research-tasks/{task_id}/trace")
    def get_research_trace(task_id: str):
        return service.get_trace(task_id)

    @app.get("/research-tasks/{task_id}/dashboard")
    def get_research_dashboard(task_id: str):
        return service.get_dashboard_bundle(task_id)

    @app.get("/research-tasks/{task_id}/artifacts/{artifact_name}")
    def get_research_artifact(task_id: str, artifact_name: str):
        return FileResponse(
            service.get_artifact_path(task_id, artifact_name),
            filename=artifact_name,
        )

    @app.post("/knowledge-bases/{task_id}/query")
    def query_knowledge_base(task_id: str, request: KnowledgeBaseQueryRequest):
        return service.query_knowledge_base(
            task_id,
            question=request.question,
            top_k=request.top_k,
        )

    @app.post("/enterprise-kbs")
    def create_enterprise_kb(request: EnterpriseKnowledgeBaseCreateRequest):
        return service.create_enterprise_knowledge_base(**_request_payload(request))

    @app.post("/enterprise-kbs/from-zotero")
    def create_enterprise_kb_from_zotero(
        request: EnterpriseKnowledgeBaseFromZoteroRequest,
    ):
        return service.create_enterprise_knowledge_base_from_zotero(
            **_request_payload(request)
        )

    @app.get("/enterprise-kbs")
    def list_enterprise_kbs(
        tenant_id: str = "local", user_id: str = "local-user"
    ):
        return {
            "knowledge_bases": service.list_enterprise_knowledge_bases(
                tenant_id=tenant_id, user_id=user_id
            )
        }

    @app.post("/enterprise-kbs/{kb_id}/ask")
    def ask_enterprise_kb(kb_id: str, request: EnterpriseKnowledgeBaseAskRequest):
        return service.ask_enterprise_knowledge_base(
            kb_id=kb_id, **_request_payload(request)
        )

    @app.post("/enterprise-kbs/{kb_id}/index-jobs")
    def enqueue_enterprise_kb_update(
        kb_id: str, request: EnterpriseKnowledgeBaseUpdateRequest
    ):
        return service.enqueue_enterprise_kb_update(
            kb_id=kb_id, **_request_payload(request)
        )

    @app.post("/production/worker/tick")
    def run_production_worker_tick():
        return service.run_production_worker_tick()

    @app.post("/evaluations/v54/dataset")
    def import_evaluation_v54_dataset(request: EvaluationDatasetV54Request):
        return service.import_evaluation_v54_dataset(request.dataset_path)

    @app.get("/evaluations/v54/status")
    def get_evaluation_v54_status():
        return service.get_evaluation_v54_status()

    @app.get("/evaluations/v54/annotations")
    def list_evaluation_v54_annotations(offset: int = 0, limit: int = 50):
        return service.list_evaluation_v54_annotations(offset=offset, limit=limit)

    @app.put("/evaluations/v54/annotations/{case_id}")
    def save_evaluation_v54_review(case_id: str, request: EvaluationReviewV54Request):
        return service.save_evaluation_v54_review(case_id, _request_payload(request))

    @app.post("/evaluations/v54/retrieval")
    def run_evaluation_v54_retrieval(request: EvaluationRetrievalV54Request):
        return service.run_evaluation_v54_retrieval(**_request_payload(request))

    @app.post("/evaluations/v54/context")
    def run_evaluation_v54_context():
        return service.run_evaluation_v54_context()

    @app.get("/evaluations/v54/latest")
    def get_evaluation_v54_latest():
        return service.get_evaluation_v54_latest()

    @app.post("/research-agent/ask")
    def ask_research_agent(request: ResearchAgentAskRequest):
        return service.ask_research_agent(**_request_payload(request))

    @app.post("/chat/sessions")
    def create_chat_session(request: ChatSessionRequest):
        return service.create_chat_session(**_request_payload(request))

    @app.get("/chat/sessions/{chat_id}")
    def get_chat_session(chat_id: str):
        return service.get_chat_session(chat_id)

    @app.get("/chat/sessions")
    def list_chat_sessions(limit: int = 50):
        return service.list_chat_sessions(limit=limit)

    @app.post("/chat/sessions/{chat_id}/regenerate")
    def regenerate_chat_session(chat_id: str):
        return service.regenerate_chat_message(chat_id)

    @app.post("/chat/sessions/{chat_id}/stop")
    def stop_chat_session(chat_id: str):
        return service.stop_chat_generation(chat_id)

    @app.post("/chat/sessions/{chat_id}/messages")
    def send_chat_message(chat_id: str, request: ChatMessageRequest):
        return service.send_chat_message(
            chat_id,
            message=request.message,
        )

    @app.get("/chat/sessions/{chat_id}/context")
    def get_chat_context(chat_id: str):
        return service.get_chat_context(chat_id)

    @app.post("/chat/sessions/{chat_id}/context/compact")
    def compact_chat_context(chat_id: str, request: CompactContextRequest):
        return service.compact_chat_context(chat_id, force=request.force)

    @app.post("/chat/sessions/{chat_id}/context/restore")
    def restore_chat_context(chat_id: str, request: RestoreContextRequest):
        return service.restore_chat_context(chat_id, request.compaction_id)

    @app.post("/conversation-graph/invoke")
    def invoke_conversation_graph(request: ConversationGraphInvokeRequest):
        return service.invoke_conversation_graph(**_request_payload(request))

    @app.get("/conversation-graph/spec")
    def get_conversation_graph_spec():
        return service.get_conversation_graph_spec()

    @app.get("/conversation-graph/threads/{thread_id}/state")
    def get_conversation_thread_state(
        thread_id: str, tenant_id: str = "local", user_id: str = "local-user"
    ):
        return service.get_conversation_thread_state(thread_id, tenant_id, user_id)

    @app.get("/conversation-graph/threads/{thread_id}/history")
    def get_conversation_thread_history(
        thread_id: str,
        limit: int = 50,
        tenant_id: str = "local",
        user_id: str = "local-user",
    ):
        return service.get_conversation_thread_history(
            thread_id, limit=limit, tenant_id=tenant_id, user_id=user_id
        )

    @app.post("/memories")
    def create_memory(request: MemoryCreateRequest):
        return service.create_memory(**_request_payload(request))

    @app.get("/memories")
    def list_memories(namespace: str, include_inactive: bool = False):
        return service.list_memories(namespace, include_inactive=include_inactive)

    @app.post("/memories/search")
    def search_memories(request: MemorySearchRequest):
        return service.search_memories(**_request_payload(request))

    @app.patch("/memories/{memory_id}")
    def edit_memory(memory_id: str, request: MemoryEditRequest):
        payload = _request_payload(request)
        namespace = payload.pop("namespace")
        content = payload.pop("content")
        payload = {key: value for key, value in payload.items() if value is not None}
        return service.edit_memory(namespace, memory_id, content, **payload)

    @app.delete("/memories/{memory_id}")
    def delete_memory(memory_id: str, namespace: str, reason: str = "user_request"):
        return service.delete_memory(namespace, memory_id, reason=reason)

    @app.get("/memories/export")
    def export_memories(namespace: str):
        return service.export_memories(namespace)

    @app.post("/memories/settings")
    def update_memory_settings(request: MemorySettingRequest):
        return service.set_memory_enabled(request.namespace, request.enabled)

    @app.post("/evaluations/runtime-v44")
    def run_langgraph_benchmark_v44():
        return service.run_langgraph_benchmark_v44()

    @app.get("/evaluations/runtime-v44/latest")
    def get_langgraph_benchmark_v44():
        return service.get_langgraph_benchmark_v44()

    @app.get("/production/status")
    def get_production_status():
        return service.get_production_status()

    @app.get("/production/traces/{trace_id}")
    def get_production_trace(
        trace_id: str, tenant_id: str, user_id: str
    ):
        return service.get_production_trace(trace_id, tenant_id, user_id)

    @app.post("/evaluations/production-v45")
    def run_production_benchmark_v45(request: ProductionBenchmarkRequest):
        return service.run_production_benchmark_v45(request.request_count)

    @app.get("/evaluations/production-v45/latest")
    def get_production_benchmark_v45():
        return service.get_production_benchmark_v45()

    return app


def _sse_event(event, payload):
    return "event: {0}\ndata: {1}\n\n".format(
        event,
        json.dumps(payload, ensure_ascii=False),
    )


def _dashboard_file_response(dashboard_dir, filename):
    from fastapi.responses import FileResponse

    path = Path(dashboard_dir) / filename
    if not path.exists():
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail=f"{filename} not found")
    return FileResponse(path)


def _request_payload(request):
    if hasattr(request, "model_dump"):
        return request.model_dump()
    return request.dict()


app = create_app()
