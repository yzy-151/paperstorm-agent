import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from .paperstorm_qa import (
    _citation_from_doc,
    _compose_answer,
    _kb_answer_prompt,
    _rag_chunk_to_doc,
)
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

    def __init__(self, root_dir, control_plane=None):
        self.root_dir = Path(root_dir)
        self.kb_dir = self.root_dir / "knowledge_bases"
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        if control_plane is None:
            from .paperstorm_production_v45 import ProductionControlPlaneV45

            control_plane = ProductionControlPlaneV45(
                self.root_dir / "production_control_v45.sqlite"
            )
        self.control = control_plane

    def create_knowledge_base(
        self,
        name: str,
        source_paths: Iterable[str],
        expected_keywords: Optional[List[str]] = None,
        forbidden_keywords: Optional[List[str]] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        embedding_provider: str = "hash",
        tenant_id: str = "local",
        owner_user_id: str = "local-user",
        allowed_user_ids: Optional[List[str]] = None,
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
                    "metadata": {
                        "path": str(path),
                        "tenant_id": tenant_id,
                        "owner_user_id": owner_user_id,
                        "allowed_user_ids": allowed_user_ids or [],
                    },
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
            "tenant_id": tenant_id,
            "owner_user_id": owner_user_id,
            "allowed_user_ids": allowed_user_ids or [],
            "index_version": 1,
            "documents": [
                {
                    "document_id": item["document_id"],
                    "path": item["metadata"]["path"],
                    "sha256": _file_digest(Path(item["metadata"]["path"])),
                }
                for item in documents
            ],
        }
        self._write_manifest(kb_id, manifest)
        self.control.register_resource(
            tenant_id=tenant_id,
            resource_type="knowledge_base",
            resource_id=kb_id,
            owner_user_id=owner_user_id,
            allowed_user_ids=allowed_user_ids,
            metadata={"name": manifest["name"]},
            version=1,
        )
        self._register_document_resources(manifest)
        return manifest

    def get_knowledge_base(
        self,
        kb_id: str,
        tenant_id: str = "local",
        user_id: str = "local-user",
    ):
        self.control.authorize(
            tenant_id, user_id, "knowledge_base", kb_id, "read"
        )
        return self._read_manifest(kb_id)

    def list_knowledge_bases(
        self, tenant_id: str = "local", user_id: str = "local-user"
    ):
        accessible_ids = {
            resource["resource_id"]
            for resource in self.control.list_accessible_resources(
                tenant_id, user_id, "knowledge_base"
            )
        }
        items = []
        for path in sorted(self.kb_dir.glob("*/manifest.json")):
            if path.parent.name in accessible_ids:
                items.append(json.loads(path.read_text(encoding="utf-8")))
        return items

    def ask(
        self,
        kb_id: str,
        question: str,
        top_k: int = 4,
        tenant_id: str = "local",
        user_id: str = "local-user",
        answer_generator: Optional[Callable[[str], str]] = None,
    ) -> Dict:
        question = str(question or "").strip()
        if not question:
            raise ValueError("question is required")
        self.control.authorize(
            tenant_id, user_id, "knowledge_base", kb_id, "read"
        )
        manifest = self._read_manifest(kb_id)
        cache_namespace = "{0}/knowledge_base/{1}/answers".format(tenant_id, kb_id)
        cache_key = hashlib.sha256(
            json.dumps(
                {
                    "question": question,
                    "top_k": int(top_k),
                    "index_version": manifest.get("index_version", 1),
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        cached = self.control.get_cache(cache_namespace, cache_key)
        if cached["hit"]:
            return dict(cached["value"], cache_hit=True)
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
        if answer_generator is not None:
            try:
                generated = str(
                    answer_generator(_kb_answer_prompt(question, evidence)) or ""
                ).strip()
                if generated:
                    answer = generated
            except Exception:
                pass
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
            "cache_hit": False,
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
        self.control.set_cache(
            cache_namespace,
            cache_key,
            result,
            ttl_seconds=300,
            tags=["kb:{0}".format(kb_id)],
        )
        return result

    def update_knowledge_base(
        self,
        kb_id: str,
        source_paths: Iterable[str],
        tenant_id: str = "local",
        user_id: str = "local-user",
    ):
        self.control.authorize(
            tenant_id, user_id, "knowledge_base", kb_id, "write"
        )
        manifest = self._read_manifest(kb_id)
        index = PaperStormRAGIndex.load(self.kb_dir / kb_id / "rag_index.json")
        records = {item["path"]: dict(item) for item in manifest.get("documents") or []}
        changed = []
        for source_path in source_paths or []:
            path = Path(source_path)
            digest = _file_digest(path)
            previous = records.get(str(path))
            if previous and previous.get("sha256") == digest:
                continue
            document_id = previous.get("document_id") if previous else "doc-{0}".format(len(records) + 1)
            text = _read_document_text(path)
            if not text.strip():
                continue
            index.chunks = [
                chunk for chunk in index.chunks
                if chunk.get("document_id") != document_id
            ]
            incremental = PaperStormRAGIndex.from_documents(
                [
                    {
                        "document_id": document_id,
                        "title": path.name,
                        "text": text,
                        "url": str(path),
                        "source_type": path.suffix.lower().lstrip(".") or "document",
                        "metadata": {
                            "path": str(path),
                            "tenant_id": tenant_id,
                            "owner_user_id": manifest.get("owner_user_id", user_id),
                            "allowed_user_ids": manifest.get("allowed_user_ids") or [],
                        },
                    }
                ],
                chunk_size=int(manifest.get("chunk_size") or 500),
                chunk_overlap=int(manifest.get("chunk_overlap") or 100),
                embedding_provider=index.embedding_provider,
            )
            index.chunks.extend(incremental.chunks)
            records[str(path)] = {
                "document_id": document_id,
                "path": str(path),
                "sha256": digest,
            }
            changed.append(str(path))
        if changed:
            index.save(self.kb_dir / kb_id / "rag_index.json")
            manifest["documents"] = list(records.values())
            manifest["source_paths"] = [item["path"] for item in records.values()]
            manifest["document_count"] = len(records)
            manifest["chunk_count"] = len(index.chunks)
            manifest["index_version"] = int(manifest.get("index_version") or 1) + 1
            manifest["updated_at"] = _now()
            self._write_manifest(kb_id, manifest)
            self.control.register_resource(
                tenant_id=manifest.get("tenant_id") or tenant_id,
                resource_type="knowledge_base",
                resource_id=kb_id,
                owner_user_id=manifest.get("owner_user_id") or user_id,
                allowed_user_ids=manifest.get("allowed_user_ids") or [],
                metadata={"name": manifest.get("name", "")},
                version=manifest["index_version"],
            )
            self._register_document_resources(manifest)
            self.control.invalidate_cache(tag="kb:{0}".format(kb_id))
        return dict(manifest, changed_source_paths=changed)

    def _register_document_resources(self, manifest: Dict):
        for document in manifest.get("documents") or []:
            self.control.register_resource(
                tenant_id=manifest.get("tenant_id") or "local",
                resource_type="document",
                resource_id="{0}:{1}".format(
                    manifest["kb_id"], document["document_id"]
                ),
                owner_user_id=manifest.get("owner_user_id") or "local-user",
                allowed_user_ids=manifest.get("allowed_user_ids") or [],
                metadata={
                    "kb_id": manifest["kb_id"],
                    "path": document.get("path", ""),
                },
                version=int(manifest.get("index_version") or 1),
            )

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
        "本地知识库": "local knowledge base internal documents",
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


def _file_digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
