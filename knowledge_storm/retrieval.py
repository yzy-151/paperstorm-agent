import copy
import json
import hashlib
import math
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np


_SAVE_LOCK_REGISTRY = {}
_SAVE_LOCK_REGISTRY_GUARD = threading.Lock()


def _save_lock_for(path):
    key = str(Path(path).resolve())
    with _SAVE_LOCK_REGISTRY_GUARD:
        lock = _SAVE_LOCK_REGISTRY.get(key)
        if lock is None:
            lock = threading.RLock()
            _SAVE_LOCK_REGISTRY[key] = lock
        return lock


def _acquire_sidecar_lock(path, timeout_seconds, stale_seconds):
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    token = "{0}:{1}:{2}".format(os.getpid(), threading.get_ident(), time.time_ns())
    encoded = token.encode("utf-8")
    while True:
        try:
            descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(descriptor, encoded)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            return token
        except FileExistsError:
            try:
                age = time.time() - path.stat().st_mtime
                existing = path.read_text(encoding="utf-8")
                if age > float(stale_seconds) and existing == path.read_text(encoding="utf-8"):
                    path.unlink()
                    continue
            except FileNotFoundError:
                continue
            except OSError:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for index sidecar lock")
            time.sleep(0.01)


def _release_sidecar_lock(path, token):
    try:
        if path.read_text(encoding="utf-8") == token:
            path.unlink()
    except OSError:
        pass


def _replace_with_retry(source, target, attempts, backoff_seconds):
    attempts = max(1, int(attempts))
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except OSError as exc:
            sharing_violation = isinstance(exc, PermissionError) or getattr(
                exc, "winerror", None
            ) in {5, 32, 33}
            if not sharing_violation or attempt + 1 >= attempts:
                raise
            time.sleep(max(0.0, float(backoff_seconds)) * (attempt + 1))


def _hash_embedding(text: str, dim: int):
    vector = [0.0] * dim
    for token in multilingual_tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dim
        vector[index] += -1.0 if digest[4] % 2 else 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def _embedding_role_contract(provider):
    getter = getattr(provider, "manifest_identity", None)
    if callable(getter):
        return getter()
    return {
        "name": str(getattr(provider, "name", "unknown")),
        "normalize": bool(getattr(provider, "normalize", False)),
        "query": "embed_query",
        "document": "embed",
    }


def multilingual_tokenize(text: str) -> List[str]:
    """Tokenize exact Latin terms and add CJK unigrams/bigrams for BM25."""
    text = str(text or "").lower()
    tokens = re.findall(r"[a-z0-9]+(?:[-./][a-z0-9]+)*", text)
    for sequence in re.findall(r"[\u3400-\u9fff]+", text):
        tokens.extend(sequence)
        tokens.extend(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return tokens


_QUERY_STOP_TOKENS = set(
    multilingual_tokenize(
        "相关研究中 作用 关系 是什么 哪些论文 场景 解决问题 同时讨论 请问 介绍 的 与 和"
    )
)


def retrieval_query_tokens(text: str) -> List[str]:
    """Tokenize a query while dropping question boilerplate from BM25."""
    return [
        token
        for token in multilingual_tokenize(text)
        if token not in _QUERY_STOP_TOKENS
    ]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Dict]],
    rank_constant: int = 60,
    weights: Optional[Sequence[float]] = None,
) -> List[Dict]:
    """Fuse independent rankings without assuming comparable raw scores."""
    weights = list(weights or [1.0] * len(rankings))
    if len(weights) != len(rankings):
        raise ValueError("weights must match rankings")
    merged = {}
    for source_index, ranking in enumerate(rankings):
        for rank, item in enumerate(ranking, start=1):
            chunk_id = str(item.get("chunk_id") or "")
            if not chunk_id:
                continue
            target = merged.setdefault(
                chunk_id,
                dict(item, rrf_score=0.0, fusion_hits=0, source_ranks={}),
            )
            for score_key in ("bm25_score", "dense_score"):
                if item.get(score_key) is not None and target.get(score_key) is None:
                    target[score_key] = item[score_key]
            target["rrf_score"] += weights[source_index] / (rank_constant + rank)
            target["fusion_hits"] += 1
            target["source_ranks"][str(source_index)] = rank
    output = list(merged.values())
    for item in output:
        item["rrf_score"] = round(item["rrf_score"], 8)
    return sorted(output, key=lambda item: (-item["rrf_score"], item["chunk_id"]))


class SentenceTransformerProvider:
    """Lazy real dense-embedding adapter with explicit index metadata."""

    normalize = True

    def __init__(
        self,
        model_name: Optional[str] = None,
        cache_folder: Optional[str] = None,
        device: Optional[str] = None,
        profile=None,
        model=None,
    ):
        from .retrieval_profiles import EmbeddingProfile, resolve_embedding_profile

        if isinstance(profile, EmbeddingProfile):
            if model_name and model_name != profile.model_name:
                raise ValueError("model_name conflicts with the selected embedding profile")
            self.profile = profile
        else:
            self.profile = resolve_embedding_profile(profile_name=profile, model_name=model_name)
        self.model_name = self.profile.model_name
        self.name = "sentence-transformers:{0}".format(self.model_name)
        self.cache_folder = cache_folder or os.getenv("PAPERSTORM_MODEL_CACHE")
        self.device = device
        self.model = model
        self.dim = 0
        self.token_codec = None
        self.normalize = bool(self.profile.document.normalize)

    def _ensure_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer

            offline = str(os.getenv("PAPERSTORM_OFFLINE_TESTS", "0")).lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=self.cache_folder,
                device=self.device,
                local_files_only=offline,
                trust_remote_code=self.profile.trust_remote_code,
            )
        if not self.dim:
            dimension_getter = getattr(
                self.model,
                "get_embedding_dimension",
                self.model.get_sentence_embedding_dimension,
            )
            self.dim = int(dimension_getter())
        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is not None and self.token_codec is None:
            self.token_codec = HuggingFaceTokenizerCodec(tokenizer)
        return self.model

    def _encode(self, texts: Iterable[str], role):
        encoded_texts = ["{0}{1}".format(role.prompt, str(text or "")) for text in texts]
        options = role.encode_kwargs()
        options["show_progress_bar"] = False
        vectors = self._ensure_model().encode(encoded_texts, **options)
        return [[float(value) for value in vector] for vector in vectors]

    def embed_documents(self, texts: Iterable[str]):
        return self._encode(texts, self.profile.document)

    def embed(self, texts: Iterable[str]):
        """Back-compatible alias for document/passages encoding."""
        return self.embed_documents(texts)

    def embed_query(self, text: str):
        return self._encode([text], self.profile.query)[0]

    def manifest_identity(self):
        return self.profile.manifest_contract()

    def get_token_codec(self):
        """Lazily expose the model tokenizer under the provider's offline policy."""
        self._ensure_model()
        return self.token_codec


class HashEmbeddingProvider:
    """Deterministic embedding fixture for tests and offline smoke runs."""

    name = "hash"
    normalize = True

    def __init__(self, dim: int = 64):
        self.dim = int(dim or 64)

    def embed(self, texts: Iterable[str]):
        return [_hash_embedding(str(text or ""), self.dim) for text in texts]

    def embed_query(self, text: str):
        return self.embed([text])[0]


class HuggingFaceTokenizerCodec:
    """Minimal token-budget adapter around a Hugging Face tokenizer."""

    def __init__(self, tokenizer):
        if not callable(getattr(tokenizer, "encode", None)) or not callable(
            getattr(tokenizer, "decode", None)
        ):
            raise ValueError("Hugging Face tokenizer must provide encode and decode")
        self.tokenizer = tokenizer

    def encode(self, text):
        return self.tokenizer.encode(str(text or ""), add_special_tokens=False)

    def decode(self, tokens):
        return self.tokenizer.decode(
            list(tokens),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )


def build_embedding_provider(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    cache_folder: Optional[str] = None,
):
    selected = str(
        provider
        or os.getenv("PAPERSTORM_EMBEDDING_PROVIDER")
        or "sentence-transformer"
    ).strip().lower()
    if selected in {"hash", "local", "smoke"}:
        return HashEmbeddingProvider()
    if selected in {"real", "sentence-transformer", "sentence-transformers"}:
        return SentenceTransformerProvider(
            model_name=model_name,
            cache_folder=cache_folder,
        )
    raise ValueError("unsupported embedding provider: {0}".format(selected))


class CrossEncoderReranker:
    """Second-stage reranker. Unit tests can inject score_fn without a model."""

    def __init__(
        self,
        model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        score_fn: Optional[Callable[[List], Iterable[float]]] = None,
        cache_folder: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.score_fn = score_fn
        self.cache_folder = cache_folder or os.getenv("PAPERSTORM_MODEL_CACHE")
        self.device = device
        self._model = None

    def _scores(self, pairs):
        if self.score_fn is not None:
            return list(self.score_fn(pairs))
        if self._model is None:
            from sentence_transformers import CrossEncoder

            offline = str(os.getenv("PAPERSTORM_OFFLINE_TESTS", "0")).lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            self._model = CrossEncoder(
                self.model_name,
                cache_folder=self.cache_folder,
                device=self.device,
                local_files_only=offline,
            )
        return [
            float(value)
            for value in self._model.predict(pairs, show_progress_bar=False)
        ]

    def rerank(
        self, query: str, candidates: Sequence[Dict], top_k: Optional[int] = None
    ):
        candidates = [dict(item) for item in candidates]
        pairs = [
            (query, str(item.get("retrieval_content") or item.get("content") or ""))
            for item in candidates
        ]
        scores = self._scores(pairs) if pairs else []
        for item, score in zip(candidates, scores):
            item["rerank_score"] = round(float(score), 8)
        candidates.sort(
            key=lambda item: (
                -item.get("rerank_score", float("-inf")),
                item.get("chunk_id", ""),
            )
        )
        return candidates[:top_k] if top_k is not None else candidates


class IndexMigrationRequiredError(RuntimeError):
    """Raised when a legacy index must be rebuilt instead of silently reused."""


class IndexIntegrityError(ValueError):
    """Raised when an index payload disagrees with its manifest."""


class HybridPaperIndex:
    """Persistent BM25 + dense index with RRF and optional Cross-Encoder."""

    schema_version = "paperstorm-hybrid-index"
    schema_revision = 3
    node_schema = "structured-parent-child-v1"
    default_max_nodes = 250_000
    default_max_node_chars = 2_000_000
    default_max_node_bytes = 8_000_000
    default_max_embedding_values = 100_000_000
    default_max_file_bytes = 512 * 1024 * 1024

    def __init__(
        self,
        chunks: Iterable[Dict],
        embedding_provider,
        embeddings=None,
        manifest=None,
        parents=None,
        token_codec=None,
        max_nodes=default_max_nodes,
        max_node_chars=default_max_node_chars,
        max_node_bytes=default_max_node_bytes,
        max_embedding_values=default_max_embedding_values,
    ):
        self.max_nodes = int(max_nodes)
        self.max_node_chars = int(max_node_chars)
        self.max_node_bytes = int(max_node_bytes)
        self.max_embedding_values = int(max_embedding_values)
        if min(
            self.max_nodes,
            self.max_node_chars,
            self.max_node_bytes,
            self.max_embedding_values,
        ) <= 0:
            raise ValueError("index resource limits must be positive")
        self.token_codec = self._validated_token_codec(
            token_codec
            if token_codec is not None
            else getattr(embedding_provider, "token_codec", None)
        )
        supplied_nodes = self._bounded_node_copies(chunks or [], self.max_nodes)
        parent_nodes = [
            item for item in supplied_nodes if item.get("node_type") in {"document", "section"}
        ]
        retrievable_nodes = [
            item for item in supplied_nodes if item.get("node_type") not in {"document", "section"}
        ]
        if parents is not None:
            parent_nodes = self._bounded_node_copies(parents, self.max_nodes)
        if len(parent_nodes) + len(retrievable_nodes) > self.max_nodes:
            raise ValueError("index node count exceeds max_nodes")
        self._validate_nodes(parent_nodes, retrievable_nodes)
        self.parents = {
            str(item["node_id"]): self._normalize_parent(copy.deepcopy(item))
            for item in parent_nodes
        }
        self.chunks = [
            self._normalize_chunk(copy.deepcopy(item), index)
            for index, item in enumerate(retrievable_nodes)
        ]
        self._validate_node_payloads(list(self.parents.values()) + self.chunks)
        self.embedding_provider = embedding_provider
        self._tokens = [
            multilingual_tokenize(self._search_text(item)) for item in self.chunks
        ]
        self._bm25 = None
        if self.chunks:
            try:
                from rank_bm25 import BM25Okapi
            except ImportError as exc:
                raise RuntimeError("Hybrid retrieval requires dependency rank-bm25") from exc
            self._bm25 = BM25Okapi(self._tokens)
        declared_dimension = int(getattr(embedding_provider, "dim", 0) or 0)
        if (
            self.chunks
            and declared_dimension
            and len(self.chunks) * declared_dimension > self.max_embedding_values
        ):
            raise ValueError("embedding value count exceeds max_embedding_values")
        if embeddings is None:
            self.embeddings = (
                embedding_provider.embed([self._search_text(item) for item in self.chunks])
                if self.chunks
                else []
            )
        else:
            self.embeddings = embeddings
        if self.token_codec is None:
            self.token_codec = self._validated_token_codec(
                getattr(embedding_provider, "token_codec", None)
            )
        embedding_matrix, dimension = self._validated_embedding_matrix(
            self.embeddings, embedding_provider
        )
        self.embeddings = embedding_matrix.tolist()
        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        self._normalized_embedding_matrix = np.divide(
            embedding_matrix,
            norms,
            out=np.zeros_like(embedding_matrix),
            where=norms != 0,
        )
        self.manifest = dict(manifest or {})
        self.manifest.setdefault(
            "embedding_model", str(getattr(embedding_provider, "name", "unknown"))
        )
        self.manifest.setdefault("embedding_dimension", dimension)
        self.manifest.setdefault(
            "normalized", bool(getattr(embedding_provider, "normalize", False))
        )
        self.manifest.setdefault(
            "embedding_profile", str(
                getattr(getattr(embedding_provider, "profile", None), "name", "unknown")
            ),
        )
        self.manifest.setdefault(
            "embedding_role_contract", _embedding_role_contract(embedding_provider)
        )
        self._refresh_manifest_integrity()

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 5,
        candidate_k: Optional[int] = None,
        reranker=None,
        rank_constant: int = 60,
        parent_budget_tokens: int = 0,
        allowed_document_ids: Optional[Sequence[str]] = None,
        allowed_chunk_ids: Optional[Sequence[str]] = None,
    ) -> List[Dict]:
        if mode not in {"bm25", "dense", "hybrid", "hybrid_rerank"}:
            raise ValueError("unsupported retrieval mode: {0}".format(mode))
        if parent_budget_tokens < 0:
            raise ValueError("parent_budget_tokens must not be negative")
        if not self.chunks:
            return []
        candidate_indices = self._candidate_indices(
            allowed_document_ids=allowed_document_ids,
            allowed_chunk_ids=allowed_chunk_ids,
        )
        if not candidate_indices:
            return []
        candidate_k = min(
            len(candidate_indices), candidate_k or max(top_k * 4, 20)
        )
        if mode == "bm25":
            selected = self._bm25_search(query, candidate_k, candidate_indices)
        elif mode == "dense":
            selected = self._dense_search(query, candidate_k, candidate_indices)
        else:
            bm25 = self._bm25_search(query, candidate_k, candidate_indices)
            dense = self._dense_search(query, candidate_k, candidate_indices)
            selected = reciprocal_rank_fusion(
                [bm25, dense], rank_constant=rank_constant
            )
            selected = selected[:candidate_k]
        if mode == "hybrid_rerank":
            if reranker is None:
                raise ValueError("hybrid_rerank mode requires reranker")
            if hasattr(reranker, "rerank"):
                selected = reranker.rerank(query, selected, top_k=top_k)
            else:
                selected = list(reranker(query, selected))[:top_k]
        else:
            selected = selected[:top_k]
        output = []
        for rank, item in enumerate(selected, start=1):
            enriched = dict(item)
            enriched["retrieval_mode"] = mode
            enriched["final_rank"] = rank
            enriched["score"] = round(
                float(
                    item.get(
                        "rrf_score",
                        item.get("bm25_score", item.get("dense_score", 0.0)),
                    )
                ),
                8,
            )
            output.append(copy.deepcopy(enriched))
        return self.expand_parent_context(output, parent_budget_tokens)

    def expand_parent_context(
        self, results: Iterable[Dict], parent_budget_tokens: int
    ) -> List[Dict]:
        """Attach bounded parent text without mutating ranked child results."""
        budget = int(parent_budget_tokens)
        if budget < 0:
            raise ValueError("parent_budget_tokens must not be negative")
        expanded = copy.deepcopy(list(results or []))
        if budget == 0:
            return expanded
        remaining_budget = budget
        expanded_parent_ids = set()
        for item in expanded:
            parent_id = str(item.get("parent_id") or "")
            parent = self.parents.get(parent_id)
            child_content = str(item.get("content") or "")
            parent_content = str(parent.get("content", "") if parent else "")
            if child_content:
                parent_content = parent_content.replace(child_content, "")
            raw_child = str((item.get("metadata") or {}).get("raw_text") or "")
            if raw_child:
                parent_content = parent_content.replace(raw_child, "")
            if not parent_id or parent_id in expanded_parent_ids:
                parent_content = ""
            elif parent:
                expanded_parent_ids.add(parent_id)
            parent_context = self._truncate_to_budget(
                parent_content, remaining_budget
            )
            remaining_budget = max(
                0, remaining_budget - self._token_count(parent_context)
            )
            item["parent_context"] = parent_context
            item["expanded_content"] = (
                child_content + "\n\n" + parent_context
                if parent_context
                else child_content
            )
        return expanded

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Dict],
        embedding_provider,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ):
        from .document_ingestion import chunk_text

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        chunks = []
        for doc_index, document in enumerate(documents or [], start=1):
            text = str(document.get("text") or "")
            document_id = document.get("document_id") or "doc-{0}".format(doc_index)
            for chunk_index, content in enumerate(
                chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap),
                start=1,
            ):
                chunk_id = "{0}-chunk-{1}".format(document_id, chunk_index)
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "title": document.get("title") or document_id,
                        "content": content,
                        "url": document.get("url") or "",
                        "source_type": document.get("source_type") or "document",
                        "metadata": dict(
                            document.get("metadata") or {}, chunk_index=chunk_index
                        ),
                    }
                )
        return cls(chunks=chunks, embedding_provider=embedding_provider)

    @classmethod
    def from_run_dir(
        cls,
        run_dir,
        embedding_provider,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ):
        """Build an index from polished article passages and raw search results."""
        run_dir = Path(run_dir)
        documents = []
        from .paperstorm_sources import load_article_passages

        for passage in load_article_passages(run_dir):
            documents.append(
                {
                    "document_id": "article-{0}".format(passage["paragraph_index"]),
                    "title": passage["title"],
                    "text": passage["content"],
                    "source_type": "article",
                    "url": "",
                    "metadata": passage,
                }
            )
        for index, result in enumerate(
            _read_json(run_dir / "raw_search_results.json", []), start=1
        ):
            result_meta = result.get("meta") or {}
            snippets = result.get("snippets") or []
            text = "\n".join(
                [
                    str(result.get("title") or ""),
                    str(result.get("description") or ""),
                    "\n".join(str(item) for item in snippets),
                ]
            ).strip()
            if text:
                documents.append(
                    {
                        "document_id": "retrieval-{0}".format(index),
                        "title": result.get("title")
                        or "Retrieved source {0}".format(index),
                        "text": text,
                        "source_type": result_meta.get("source_type") or result.get("source_type") or "retrieval",
                        "url": result.get("url") or "",
                        "metadata": {
                            "result_index": index,
                            "query": result.get("query", ""),
                            "authors": result_meta.get("authors") or result.get("authors") or [],
                            "published": result_meta.get("published") or result.get("published") or "",
                            "original_title": result.get("title") or "",
                        },
                    }
                )
        return cls.from_documents(
            documents,
            embedding_provider=embedding_provider,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def _candidate_indices(self, allowed_document_ids=None, allowed_chunk_ids=None):
        """Resolve authorization scope before either retrieval scorer executes."""
        document_scope = (
            None
            if allowed_document_ids is None
            else {str(value) for value in allowed_document_ids}
        )
        chunk_scope = (
            None
            if allowed_chunk_ids is None
            else {str(value) for value in allowed_chunk_ids}
        )
        return [
            index
            for index, chunk in enumerate(self.chunks)
            if (document_scope is None or str(chunk.get("document_id") or "") in document_scope)
            and (chunk_scope is None or str(chunk.get("chunk_id") or "") in chunk_scope)
        ]

    def _bm25_search(self, query: str, top_k: int, candidate_indices=None):
        indices = (
            list(range(len(self.chunks)))
            if candidate_indices is None
            else list(candidate_indices)
        )
        if not indices:
            return []
        if len(indices) == len(self.chunks):
            scores = self._bm25.get_scores(retrieval_query_tokens(query))
        else:
            from rank_bm25 import BM25Okapi

            scores = BM25Okapi([self._tokens[index] for index in indices]).get_scores(
                retrieval_query_tokens(query)
            )
        ranked = []
        for position in sorted(
            range(len(scores)), key=lambda value: (-scores[value], indices[value])
        )[:top_k]:
            index = indices[position]
            item = dict(self.chunks[index])
            item["bm25_score"] = round(float(scores[position]), 8)
            ranked.append(item)
        return ranked

    def _dense_search(self, query: str, top_k: int, candidate_indices=None):
        indices = (
            list(range(len(self.chunks)))
            if candidate_indices is None
            else list(candidate_indices)
        )
        if not indices:
            return []
        query_vector = np.asarray(
            self.embedding_provider.embed_query(query), dtype=np.float32
        )
        query_norm = float(np.linalg.norm(query_vector))
        if query_norm:
            query_vector = query_vector / query_norm
        scores = self._normalized_embedding_matrix[indices] @ query_vector
        ranked = []
        for position in sorted(
            range(len(scores)), key=lambda value: (-scores[value], indices[value])
        )[:top_k]:
            index = indices[position]
            item = dict(self.chunks[index])
            item["dense_score"] = round(float(scores[position]), 8)
            ranked.append(item)
        return ranked

    def save(
        self,
        path,
        lock_timeout_seconds=10.0,
        stale_lock_seconds=60.0,
        replace_attempts=5,
        replace_backoff_seconds=0.02,
    ):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        resolved_path = path.resolve()
        process_lock = _save_lock_for(resolved_path)
        sidecar = Path(str(resolved_path) + ".lock")
        with process_lock:
            token = _acquire_sidecar_lock(
                sidecar, lock_timeout_seconds, stale_lock_seconds
            )
            temporary = None
            try:
                self._refresh_manifest_integrity()
                payload = json.dumps(
                    {
                        "manifest": self.manifest,
                        "chunks": self.chunks,
                        "parents": list(self.parents.values()),
                        "embeddings": self.embeddings,
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                    indent=2,
                )
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=str(resolved_path.parent),
                    prefix=resolved_path.name + ".",
                    suffix=".tmp",
                    delete=False,
                ) as handle:
                    temporary = Path(handle.name)
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                _replace_with_retry(
                    temporary,
                    resolved_path,
                    replace_attempts,
                    replace_backoff_seconds,
                )
            finally:
                if temporary is not None:
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError:
                        pass
                _release_sidecar_lock(sidecar, token)
        return path

    @classmethod
    def load(
        cls,
        path,
        embedding_provider,
        token_codec=None,
        max_nodes=default_max_nodes,
        max_node_chars=default_max_node_chars,
        max_node_bytes=default_max_node_bytes,
        max_embedding_values=default_max_embedding_values,
        max_file_bytes=default_max_file_bytes,
    ):
        path = Path(path)
        if path.stat().st_size > int(max_file_bytes):
            raise ValueError("index file size exceeds max_file_bytes")
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifest = payload.get("manifest") or {}
        if manifest.get("schema_version") != cls.schema_version:
            raise IndexMigrationRequiredError(
                "legacy retrieval index detected; rebuild the knowledge base index"
            )
        if "schema_revision" not in manifest:
            raise IndexMigrationRequiredError(
                "retrieval index revision is missing; rebuild the knowledge base index"
            )
        if int(manifest.get("schema_revision") or 0) != cls.schema_revision:
            raise IndexMigrationRequiredError(
                "unsupported retrieval index revision; rebuild the knowledge base index"
            )
        if manifest.get("node_schema") != cls.node_schema:
            raise IndexMigrationRequiredError(
                "unsupported retrieval node schema; rebuild the knowledge base index"
            )
        chunks = payload.get("chunks") or []
        parents = payload.get("parents") or []
        if not isinstance(chunks, list) or not isinstance(parents, list):
            raise IndexIntegrityError("index integrity error: stores must be lists")
        if any(item.get("node_type") in {"document", "section"} for item in chunks):
            raise IndexIntegrityError(
                "index integrity error: parent node found in retrievable store"
            )
        parent_ids = [str(item.get("node_id") or "") for item in parents]
        if any(not value for value in parent_ids) or len(set(parent_ids)) != len(parent_ids):
            raise IndexIntegrityError(
                "index integrity error: parent store contains missing or duplicate node_id"
            )
        actual_counts = {
            "chunk_count": len(chunks),
            "retrievable_count": len(chunks),
            "parent_count": len(parent_ids),
        }
        for field, actual in actual_counts.items():
            try:
                declared = int(manifest.get(field))
            except (TypeError, ValueError):
                raise IndexIntegrityError(
                    "index integrity error: manifest {0} is invalid".format(field)
                )
            if declared != actual:
                raise IndexIntegrityError(
                    "index integrity error: manifest {0}={1}, payload={2}".format(
                        field, declared, actual
                    )
                )
        actual_name = str(getattr(embedding_provider, "name", "unknown"))
        actual_dim = int(getattr(embedding_provider, "dim", 0) or 0)
        if manifest.get("embedding_model") != actual_name:
            raise ValueError(
                "embedding model mismatch: index={0}, provider={1}".format(
                    manifest.get("embedding_model"), actual_name
                )
            )
        if actual_dim and int(manifest.get("embedding_dimension") or 0) != actual_dim:
            raise ValueError(
                "embedding dimension mismatch: index={0}, provider={1}".format(
                    manifest.get("embedding_dimension"), actual_dim
                )
            )
        if bool(manifest.get("normalized")) != bool(
            getattr(embedding_provider, "normalize", False)
        ):
            raise ValueError("embedding normalization mismatch")
        actual_profile = str(
            getattr(getattr(embedding_provider, "profile", None), "name", "unknown")
        )
        if manifest.get("embedding_profile", "unknown") != actual_profile:
            raise ValueError(
                "embedding profile mismatch: index={0}, provider={1}".format(
                    manifest.get("embedding_profile", "unknown"), actual_profile
                )
            )
        if manifest.get("embedding_role_contract") != _embedding_role_contract(
            embedding_provider
        ):
            raise ValueError("embedding role contract mismatch")
        return cls(
            chunks,
            embedding_provider=embedding_provider,
            embeddings=payload.get("embeddings") or [],
            manifest=manifest,
            parents=parents,
            token_codec=token_codec,
            max_nodes=max_nodes,
            max_node_chars=max_node_chars,
            max_node_bytes=max_node_bytes,
            max_embedding_values=max_embedding_values,
        )

    def _refresh_manifest_integrity(self):
        self.manifest.update(
            {
                "schema_version": self.schema_version,
                "schema_revision": self.schema_revision,
                "node_schema": self.node_schema,
                "chunk_count": len(self.chunks),
                "retrievable_count": len(self.chunks),
                "parent_count": len(self.parents),
            }
        )

    @staticmethod
    def _normalize_chunk(item, index):
        chunk = dict(item)
        chunk["chunk_id"] = str(chunk.get("chunk_id") or "")
        chunk["content"] = str(chunk.get("content") or chunk.get("text") or "")
        chunk.setdefault("retrieval_content", chunk["content"])
        return chunk

    @staticmethod
    def _normalize_parent(item):
        parent = dict(item)
        parent["node_id"] = str(parent.get("node_id") or "")
        parent["content"] = str(parent.get("content") or "")
        parent.setdefault("retrieval_content", parent["content"])
        return parent

    def _truncate_to_budget(self, text, token_budget):
        if token_budget <= 0:
            return ""
        text = str(text or "")
        self._refresh_token_codec()
        if self.token_codec is not None:
            encoded = self.token_codec.encode(text)
            return str(self.token_codec.decode(encoded[:token_budget]))
        from .document_ingestion import _join_units, _token_units

        units = _token_units(text)[:token_budget]
        return _join_units(units).strip()[: token_budget * 4]

    def _token_count(self, text):
        text = str(text or "")
        if not text:
            return 0
        self._refresh_token_codec()
        if self.token_codec is not None:
            return len(self.token_codec.encode(text))
        from .document_ingestion import _token_units

        return len(_token_units(text))

    def _refresh_token_codec(self):
        if self.token_codec is not None:
            return self.token_codec
        getter = getattr(self.embedding_provider, "get_token_codec", None)
        codec = getter() if callable(getter) else getattr(
            self.embedding_provider, "token_codec", None
        )
        self.token_codec = self._validated_token_codec(codec)
        return self.token_codec

    @staticmethod
    def _bounded_node_copies(nodes, max_nodes):
        output = []
        for index, item in enumerate(nodes, start=1):
            if index > max_nodes:
                raise ValueError("index node count exceeds max_nodes")
            if not isinstance(item, dict):
                raise ValueError("index nodes must be mappings")
            output.append(dict(item))
        return output

    def _validate_nodes(self, parents, chunks):
        self._validate_node_payloads(parents + chunks)
        parent_ids = [str(item.get("node_id") or "") for item in parents]
        if any(not value for value in parent_ids):
            raise ValueError("parent node_id must be non-empty")
        if len(set(parent_ids)) != len(parent_ids):
            raise ValueError("duplicate parent node_id")
        chunk_ids = [str(item.get("chunk_id") or "") for item in chunks]
        if any(not value for value in chunk_ids):
            raise ValueError("chunk_id must be non-empty")
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("duplicate chunk_id")
        child_node_ids = []
        for item in chunks:
            if item.get("node_type") and not str(item.get("node_id") or ""):
                raise ValueError("structured child node_id must be non-empty")
            if "node_id" in item:
                value = str(item.get("node_id") or "")
                if not value:
                    raise ValueError("child node_id must be non-empty")
                child_node_ids.append(value)
        if len(set(child_node_ids)) != len(child_node_ids):
            raise ValueError("duplicate child node_id")
        if set(parent_ids) & set(child_node_ids):
            raise ValueError("parent and child node_id collision")

    def _validate_node_payloads(self, nodes):
        for item in nodes:
            try:
                serialized = json.dumps(
                    item,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("index node must be strict JSON") from exc
            if len(serialized) > self.max_node_chars:
                raise ValueError("index node character count exceeds max_node_chars")
            if len(serialized.encode("utf-8")) > self.max_node_bytes:
                raise ValueError("index node byte count exceeds max_node_bytes")

    @staticmethod
    def _validated_token_codec(codec):
        if codec is None:
            return None
        if not (
            callable(getattr(codec, "encode", None))
            and callable(getattr(codec, "decode", None))
        ):
            raise ValueError("token_codec must provide encode and decode")
        return codec

    def _validated_embedding_matrix(self, embeddings, provider):
        if not isinstance(embeddings, (list, tuple, np.ndarray)):
            raise ValueError("embedding matrix must be a two-dimensional sequence")
        row_count = len(embeddings)
        if row_count != len(self.chunks):
            raise ValueError("embedding row count must match chunk count")
        provider_dim = int(getattr(provider, "dim", 0) or 0)
        if row_count == 0:
            return np.zeros((0, provider_dim), dtype=np.float32), provider_dim
        try:
            column_counts = [len(row) for row in embeddings]
        except TypeError as exc:
            raise ValueError("embedding matrix must be two-dimensional") from exc
        if len(set(column_counts)) != 1:
            raise ValueError("embedding matrix rows must have equal column count")
        column_count = column_counts[0]
        if row_count * column_count > self.max_embedding_values:
            raise ValueError("embedding value count exceeds max_embedding_values")
        if provider_dim and column_count != provider_dim:
            raise ValueError(
                "embedding column count does not match provider dimension"
            )
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("embedding matrix must be two-dimensional")
        if not np.isfinite(matrix).all():
            raise ValueError("embedding matrix values must be finite")
        return matrix, provider_dim or column_count

    @staticmethod
    def _search_text(item):
        return "{0}\n{1}".format(
            item.get("title", ""), item.get("retrieval_content", "")
        )


def _cosine(left, right):
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def _read_first_existing(paths):
    for path in paths:
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
    return ""


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default
