import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .paperstorm_qa import _citation_from_doc, _compose_answer, _rag_chunk_to_doc
from .paperstorm_rag import (
    ContextCompressionRetriever,
    PaperStormRAGIndex,
    build_embedding_provider,
)


class EnterpriseKnowledgeBaseService:
    """File-backed enterprise knowledge base workflow.

    This service is a local demo baseline for "upload documents -> build RAG
    index -> ask with citations". It keeps production replacement points clear:
    embedding_provider can become bge/openai, and the saved JSON index can be
    replaced by Qdrant/FAISS/Milvus without changing the public service API.
    """

    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.kb_dir = self.root_dir / "knowledge_bases"
        self.kb_dir.mkdir(parents=True, exist_ok=True)

    def create_knowledge_base(
        self,
        name: str,
        source_paths: Iterable[str],
        expected_keywords: Optional[List[str]] = None,
        forbidden_keywords: Optional[List[str]] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        embedding_provider: str = "hash",
    ) -> Dict:
        documents = []
        for index, source_path in enumerate(source_paths or [], start=1):
            path = Path(source_path)
            text = _read_document_text(path)
            if not text.strip():
                continue
            documents.append(
                {
                    "document_id": "doc-{0}".format(index),
                    "title": path.name,
                    "text": text,
                    "url": str(path),
                    "source_type": path.suffix.lower().lstrip(".") or "document",
                    "metadata": {"path": str(path)},
                }
            )
        if not documents:
            raise ValueError("No readable documents were provided.")

        provider = build_embedding_provider(embedding_provider)
        index = PaperStormRAGIndex.from_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_provider=provider,
        )
        kb_id = uuid.uuid4().hex
        kb_path = self.kb_dir / kb_id
        kb_path.mkdir(parents=True, exist_ok=True)
        index_path = index.save(kb_path / "rag_index.json")
        manifest = {
            "kb_id": kb_id,
            "name": name or "Enterprise Knowledge Base",
            "created_at": _now(),
            "document_count": len(documents),
            "chunk_count": len(index.chunks),
            "source_paths": [str(path) for path in source_paths or []],
            "expected_keywords": expected_keywords or [],
            "forbidden_keywords": forbidden_keywords or [],
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding_provider": index.config.get("embedding_provider", ""),
            "index_path": str(index_path),
        }
        self._write_manifest(kb_id, manifest)
        return manifest

    def get_knowledge_base(self, kb_id: str):
        return self._read_manifest(kb_id)

    def list_knowledge_bases(self):
        items = []
        for path in sorted(self.kb_dir.glob("*/manifest.json")):
            items.append(json.loads(path.read_text(encoding="utf-8")))
        return items

    def ask(self, kb_id: str, question: str, top_k: int = 4) -> Dict:
        question = str(question or "").strip()
        if not question:
            raise ValueError("question is required")
        manifest = self._read_manifest(kb_id)
        index = PaperStormRAGIndex.load(self.kb_dir / kb_id / "rag_index.json")
        retriever = ContextCompressionRetriever(index, max_context_chars=2200)
        retrieval_query = _expand_query(question)
        retrieved = retriever.retrieve(
            retrieval_query,
            top_k=top_k,
            expected_keywords=manifest.get("expected_keywords") or [],
            forbidden_keywords=manifest.get("forbidden_keywords") or [],
        )
        evidence = [_rag_chunk_to_doc(chunk) for chunk in retrieved.get("chunks") or []]
        answer = _compose_answer(question, evidence, memory_context={})
        result = {
            "kb_id": kb_id,
            "question": question,
            "retrieval_query": retrieval_query,
            "answer": answer,
            "grounded": bool(evidence),
            "citations": [
                _citation_from_doc(index, doc)
                for index, doc in enumerate(evidence, start=1)
            ],
            "evidence": evidence,
            "retrieval": retrieved,
            "manifest": manifest,
            "trace": [
                {
                    "event": "enterprise_kb_ask",
                    "timestamp": _now(),
                    "payload": {
                        "kb_id": kb_id,
                        "top_k": top_k,
                        "chunk_count": len(evidence),
                        "retrieval_query": retrieval_query,
                        "embedding_provider": manifest.get("embedding_provider", ""),
                    },
                }
            ],
        }
        (self.kb_dir / kb_id / "last_answer.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return result

    def _read_manifest(self, kb_id: str):
        path = self.kb_dir / kb_id / "manifest.json"
        if not path.exists():
            raise KeyError("Unknown kb_id: {0}".format(kb_id))
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_manifest(self, kb_id: str, manifest: Dict):
        (self.kb_dir / kb_id / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _read_document_text(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf_text(path)
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf_text(path: Path):
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("PDF ingestion requires optional dependency pypdf.") from exc
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _expand_query(question: str):
    text = str(question or "").strip()
    additions = []
    lowered = text.lower()
    bilingual_terms = {
        "企业知识库": "enterprise knowledge base internal documents",
        "知识库": "knowledge base documents",
        "使用": "use uses using",
        "agent": "agent agents",
        "智能体": "agent agents",
        "检索": "retrieval search",
        "问答": "question answering qa",
    }
    for key, value in bilingual_terms.items():
        if key.lower() in lowered and value not in additions:
            additions.append(value)
    if additions:
        return "{0}\n{1}".format(text, " ".join(additions))
    return text


def _now():
    return datetime.now(timezone.utc).isoformat()
