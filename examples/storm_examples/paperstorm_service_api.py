"""
Optional FastAPI adapter for the PaperStorm service core.

Example:
    uvicorn examples.storm_examples.paperstorm_service_api:app --reload
"""

import json
import time
from pathlib import Path
from typing import Optional

from knowledge_storm.paperstorm_service import PaperStormTaskService


DEFAULT_SERVICE_ROOT = Path("./results/paperstorm_service")
DEFAULT_DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "frontend" / "paperstorm_dashboard"


def create_app(service_root=DEFAULT_SERVICE_ROOT, dashboard_dir=DEFAULT_DASHBOARD_DIR):
    try:
        from fastapi import FastAPI
        from fastapi.responses import FileResponse, StreamingResponse
        from pydantic import BaseModel, Field
        from starlette.middleware.cors import CORSMiddleware
        from starlette.staticfiles import StaticFiles
    except ImportError as exc:
        raise RuntimeError(
            "FastAPI service adapter requires optional dependencies: fastapi and uvicorn."
        ) from exc

    service = PaperStormTaskService(root_dir=service_root)
    dashboard_dir = Path(dashboard_dir)
    app = FastAPI(title="PaperStorm Agent Service", version="4.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
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
        context_window_size: int = 6

    class ChatMessageRequest(BaseModel):
        message: str

    class EnterpriseKnowledgeBaseCreateRequest(BaseModel):
        name: str = "Enterprise Knowledge Base"
        source_paths: list[str] = []
        expected_keywords: list[str] = []
        forbidden_keywords: list[str] = []
        chunk_size: int = 500
        chunk_overlap: int = 100
        embedding_provider: str = "hash"

    class EnterpriseKnowledgeBaseAskRequest(BaseModel):
        question: str
        top_k: int = 4

    class RAGEvaluationV4Request(BaseModel):
        top_k: int = Field(default=5, ge=1, le=20)

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
                    except KeyError:
                        payload["task_status"] = "unknown"
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

    @app.get("/enterprise-kbs")
    def list_enterprise_kbs():
        return {"knowledge_bases": service.list_enterprise_knowledge_bases()}

    @app.post("/enterprise-kbs/{kb_id}/ask")
    def ask_enterprise_kb(kb_id: str, request: EnterpriseKnowledgeBaseAskRequest):
        return service.ask_enterprise_knowledge_base(
            kb_id=kb_id,
            question=request.question,
            top_k=request.top_k,
        )

    @app.post("/evaluations/rag-v4")
    def run_rag_evaluation_v4(request: RAGEvaluationV4Request):
        return service.run_rag_evaluation_v4(
            top_k=request.top_k,
        )

    @app.get("/evaluations/rag-v4/latest")
    def get_rag_evaluation_v4():
        return service.get_rag_evaluation_v4()

    @app.post("/research-agent/ask")
    def ask_research_agent(request: ResearchAgentAskRequest):
        return service.ask_research_agent(**_request_payload(request))

    @app.post("/chat/sessions")
    def create_chat_session(request: ChatSessionRequest):
        return service.create_chat_session(**_request_payload(request))

    @app.get("/chat/sessions/{chat_id}")
    def get_chat_session(chat_id: str):
        return service.get_chat_session(chat_id)

    @app.post("/chat/sessions/{chat_id}/messages")
    def send_chat_message(chat_id: str, request: ChatMessageRequest):
        return service.send_chat_message(
            chat_id,
            message=request.message,
        )

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


try:
    app = create_app()
except RuntimeError:
    app = None
