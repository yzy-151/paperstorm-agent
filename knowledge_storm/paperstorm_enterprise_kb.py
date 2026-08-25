import hashlib
import json
import os
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
from .retrieval import (
    HybridPaperIndex,
    IndexMigrationRequiredError,
    build_embedding_provider,
)
from .retrieval_pipeline import RetrievalPipeline, RetrievalRequest


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
            from .control_plane import ProductionControlPlane

            current = self.root_dir / "production_control.sqlite"
            legacy = self.root_dir / "production_control_v45.sqlite"
            control_plane = ProductionControlPlane(
                legacy if legacy.exists() else current
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
        embedding_provider: str = "sentence-transformer",
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
        index = HybridPaperIndex.from_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_provider=provider,
        )
        kb_id = uuid.uuid4().hex
        kb_path = self.kb_dir / kb_id
        kb_path.mkdir(parents=True, exist_ok=True)
        index_file = "rag_index.1.json"
        index_path = index.save(kb_path / index_file)
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
            "index_schema": HybridPaperIndex.schema_version,
            "schema_revision": HybridPaperIndex.schema_revision,
            "embedding_provider": index.manifest.get("embedding_model", ""),
            "retrieval_mode": "hybrid",
            "index_file": index_file,
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
        index = self._load_index(kb_id, manifest)
        pipeline = RetrievalPipeline(index)
        retrieval_query = _expand_query(question)
        retrieved = pipeline.search(
            RetrievalRequest(
                query=retrieval_query,
                top_k=top_k,
                candidate_k=max(top_k * 4, 20),
                mode=manifest.get("retrieval_mode") or "hybrid",
                expected_keywords=tuple(manifest.get("expected_keywords") or []),
                forbidden_keywords=tuple(manifest.get("forbidden_keywords") or []),
            )
        )
        evidence = [_rag_chunk_to_doc(chunk) for chunk in retrieved.get("results") or []]
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
        records = {item["path"]: dict(item) for item in manifest.get("documents") or []}
        changed = []
        for source_path in source_paths or []:
            path = Path(source_path)
            digest = _file_digest(path)
            previous = records.get(str(path))
            if previous and previous.get("sha256") == digest:
                continue
            document_id = previous.get("document_id") if previous else "doc-{0}".format(len(records) + 1)
            if not _read_document_text(path).strip():
                continue
            records[str(path)] = {
                "document_id": document_id,
                "path": str(path),
                "sha256": digest,
            }
            changed.append(str(path))
        if changed:
            provider = _provider_from_manifest(manifest)
            documents = _documents_from_records(
                records.values(),
                tenant_id=tenant_id,
                owner_user_id=manifest.get("owner_user_id", user_id),
                allowed_user_ids=manifest.get("allowed_user_ids") or [],
            )
            index = HybridPaperIndex.from_documents(
                documents,
                chunk_size=int(manifest.get("chunk_size") or 500),
                chunk_overlap=int(manifest.get("chunk_overlap") or 100),
                embedding_provider=provider,
            )
            next_version = int(manifest.get("index_version") or 1) + 1
            index_file = "rag_index.{0}.json".format(next_version)
            index_path = index.save(self.kb_dir / kb_id / index_file)
            manifest["documents"] = list(records.values())
            manifest["source_paths"] = [item["path"] for item in records.values()]
            manifest["document_count"] = len(records)
            manifest["chunk_count"] = len(index.chunks)
            manifest["index_version"] = next_version
            manifest["index_file"] = index_file
            manifest["index_path"] = str(index_path)
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

    def _load_index(self, kb_id: str, manifest: Dict):
        index_file = str(manifest.get("index_file") or "").strip()
        if index_file:
            if Path(index_file).name != index_file:
                raise ValueError("invalid knowledge base index_file")
            index_path = self.kb_dir / kb_id / index_file
        else:
            configured = Path(str(manifest.get("index_path") or ""))
            index_path = configured if configured.exists() else self.kb_dir / kb_id / "rag_index.json"
        try:
            return HybridPaperIndex.load(
                index_path,
                embedding_provider=_provider_from_manifest(manifest),
            )
        except IndexMigrationRequiredError as error:
            raise IndexMigrationRequiredError(
                "Knowledge base {0} uses a legacy index. Enqueue an index job "
                "to rebuild schema revision {1}.".format(
                    kb_id, HybridPaperIndex.schema_revision
                )
            ) from error

    def _read_manifest(self, kb_id: str):
        path = self.kb_dir / kb_id / "manifest.json"
        if not path.exists():
            raise KeyError("Unknown kb_id: {0}".format(kb_id))
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_manifest(self, kb_id: str, manifest: Dict):
        path = self.kb_dir / kb_id / "manifest.json"
        temporary = path.with_name(path.name + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(manifest, ensure_ascii=False, indent=2))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _read_document_text(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf_text(path)
    return path.read_text(encoding="utf-8", errors="replace")


def _provider_from_manifest(manifest: Dict):
    name = str(manifest.get("embedding_provider") or "sentence-transformer")
    if name == "hash" or name.startswith("hash_"):
        return build_embedding_provider("hash")
    prefix = "sentence-transformers:"
    model_name = name[len(prefix) :] if name.startswith(prefix) else None
    return build_embedding_provider("sentence-transformer", model_name=model_name)


def _documents_from_records(
    records,
    tenant_id: str,
    owner_user_id: str,
    allowed_user_ids,
):
    documents = []
    for record in records:
        path = Path(record["path"])
        text = _read_document_text(path)
        if not text.strip():
            continue
        documents.append(
            {
                "document_id": record["document_id"],
                "title": path.name,
                "text": text,
                "url": str(path),
                "source_type": path.suffix.lower().lstrip(".") or "document",
                "metadata": {
                    "path": str(path),
                    "tenant_id": tenant_id,
                    "owner_user_id": owner_user_id,
                    "allowed_user_ids": list(allowed_user_ids or []),
                },
            }
        )
    return documents


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
