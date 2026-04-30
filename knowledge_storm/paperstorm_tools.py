from typing import Dict, Any

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


def list_paperstorm_tools(pdf_dir=None):
    tools = [ArxivSearchTool()]
    if pdf_dir:
        tools.append(LocalPDFSearchTool(pdf_dir=pdf_dir))
    return tools
