"""
Optional FastAPI adapter for the PaperStorm service core.

Example:
    uvicorn examples.storm_examples.paperstorm_service_api:app --reload
"""

from pathlib import Path

from knowledge_storm.paperstorm_service import PaperStormTaskService


DEFAULT_SERVICE_ROOT = Path("./results/paperstorm_service")


def create_app(service_root=DEFAULT_SERVICE_ROOT):
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
    except ImportError as exc:
        raise RuntimeError(
            "FastAPI service adapter requires optional dependencies: fastapi and uvicorn."
        ) from exc

    service = PaperStormTaskService(root_dir=service_root)
    app = FastAPI(title="PaperStorm Agent Service", version="0.5")

    class ResearchTaskRequest(BaseModel):
        topic: str
        retriever: str = "arxiv"
        output_language: str = "zh"
        run_mode: str = "fake"
        expected_keywords: list[str] = []
        forbidden_keywords: list[str] = []

    class KnowledgeBaseQueryRequest(BaseModel):
        question: str
        top_k: int = 3

    @app.post("/research-tasks")
    def submit_research_task(request: ResearchTaskRequest):
        return service.submit_research_task(**request.dict())

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

    @app.post("/knowledge-bases/{task_id}/query")
    def query_knowledge_base(task_id: str, request: KnowledgeBaseQueryRequest):
        return service.query_knowledge_base(
            task_id,
            question=request.question,
            top_k=request.top_k,
        )

    return app


try:
    app = create_app()
except RuntimeError:
    app = None
