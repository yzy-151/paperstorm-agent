from typing import Dict, Any

from .paperstorm_qa import PaperStormKnowledgeBase
from .paperstorm_service import PaperStormTaskService
from .rm import ArxivRM, LocalPDFRM


class PaperStormTool:
    name = ""
    description = ""
    input_schema: Dict[str, Any] = {}
    output_schema: Dict[str, Any] = {}

    def to_schema(self):
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
        }

    def run(self, arguments: Dict[str, Any]):
        raise NotImplementedError


class RetrievalTool(PaperStormTool):
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query for the retrieval backend.",
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum number of results to return.",
                "default": None,
            },
        },
        "required": ["query"],
    }
    output_schema = {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {"type": "object"},
            }
        },
        "required": ["results"],
    }

    def __init__(self, rm):
        self.rm = rm

    @staticmethod
    def _serialize_result(result):
        return {
            "url": result.get("url"),
            "title": result.get("title"),
            "description": result.get("description"),
            "snippets": result.get("snippets") or [],
            "meta": result.get("meta") or {},
        }

    def run(self, arguments: Dict[str, Any]):
        query = (arguments or {}).get("query")
        if not query or not str(query).strip():
            raise ValueError("Tool argument 'query' is required.")

        original_k = getattr(self.rm, "k", None)
        top_k = (arguments or {}).get("top_k")
        if top_k is not None and hasattr(self.rm, "k"):
            self.rm.k = int(top_k)
        try:
            results = self.rm.forward(str(query))
        finally:
            if top_k is not None and original_k is not None:
                self.rm.k = original_k

        return {"results": [self._serialize_result(result) for result in results]}


class ArxivSearchTool(RetrievalTool):
    name = "arxiv_search"
    description = (
        "Search paper metadata from arXiv. Useful for academic literature retrieval."
    )

    def __init__(self, rm=None):
        super().__init__(rm=rm or ArxivRM())


class LocalPDFSearchTool(RetrievalTool):
    name = "local_pdf_search"
    description = (
        "Search relevant chunks from a local PDF paper collection."
    )

    def __init__(self, rm=None, pdf_dir=None):
        super().__init__(rm=rm or LocalPDFRM(pdf_dir=pdf_dir))


class KnowledgeBaseQATool(PaperStormTool):
    name = "kb_qa"
    description = (
        "Answer a question from an existing PaperStorm run directory using generated "
        "article text and saved retrieval results."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "run_dir": {
                "type": "string",
                "description": "PaperStorm topic output directory containing article and retrieval artifacts.",
            },
            "question": {
                "type": "string",
                "description": "User question to answer from the run artifacts.",
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum evidence items to use.",
                "default": 3,
            },
        },
        "required": ["run_dir", "question"],
    }
    output_schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "citations": {"type": "array", "items": {"type": "object"}},
            "grounded": {"type": "boolean"},
            "memory_context": {"type": "object"},
        },
        "required": ["answer", "citations", "grounded"],
    }

    def run(self, arguments: Dict[str, Any]):
        arguments = arguments or {}
        run_dir = arguments.get("run_dir")
        question = arguments.get("question")
        if not run_dir or not str(run_dir).strip():
            raise ValueError("Tool argument 'run_dir' is required.")
        if not question or not str(question).strip():
            raise ValueError("Tool argument 'question' is required.")
        kb = PaperStormKnowledgeBase.from_run_dir(run_dir)
        return kb.answer_question(
            question=str(question),
            top_k=int(arguments.get("top_k") or 3),
        )


class ResearchQATool(PaperStormTool):
    name = "research_qa"
    description = (
        "Ask the PaperStorm Research QA Agent. It can reuse an existing task_id or "
        "create a research task before answering with citations and evidence."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "User question."},
            "topic": {"type": "string", "description": "Research topic when no task_id is available."},
            "task_id": {"type": "string", "description": "Existing PaperStorm task id."},
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
            "decision": {"type": "object"},
            "evidence_sufficiency": {"type": "object"},
            "grounded": {"type": "boolean"},
        },
        "required": ["answer", "citations", "decision", "grounded"],
    }

    def __init__(self, service_root="./results/paperstorm_tool_service"):
        self.service = PaperStormTaskService(root_dir=service_root)

    def run(self, arguments: Dict[str, Any]):
        arguments = arguments or {}
        question = arguments.get("question")
        if not question or not str(question).strip():
            raise ValueError("Tool argument 'question' is required.")
        return self.service.ask_research_agent(**arguments)


def list_paperstorm_tools(pdf_dir=None):
    tools = [ArxivSearchTool(), KnowledgeBaseQATool(), ResearchQATool()]
    if pdf_dir:
        tools.append(LocalPDFSearchTool(pdf_dir=pdf_dir))
    return tools
