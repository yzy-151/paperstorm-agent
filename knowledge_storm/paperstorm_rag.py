import hashlib
import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .paperstorm_memory import compress_context


class PaperStormRAGIndex:
    """Local RAG index with lexical + hash-embedding hybrid retrieval."""

    def __init__(
        self,
        chunks: Optional[List[Dict]] = None,
        embedding_dim: int = 64,
        config: Optional[Dict] = None,
    ):
        self.chunks = chunks or []
        self.embedding_dim = int(embedding_dim or 64)
        self.config = config or {
            "index_type": "local_json_hash_embedding",
            "ann": "linear_scan_baseline",
            "hnsw_ready_params": {"M": 16, "ef_construct": 100, "ef_search": 64},
        }

    @classmethod
    def from_run_dir(
        cls,
        run_dir,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        embedding_dim: int = 64,
    ):
        run_dir = Path(run_dir)
        documents = []
        article = _read_first_existing(
            [
                run_dir / "storm_gen_article_polished.txt",
                run_dir / "storm_gen_article.txt",
            ]
        )
        if article:
            documents.append(
                {
                    "document_id": "generated_article",
                    "title": "Generated PaperStorm Article",
                    "text": article,
                    "source_type": "article",
                    "url": str(run_dir / "storm_gen_article_polished.txt"),
                }
            )
        for index, result in enumerate(_read_json(run_dir / "raw_search_results.json", []), start=1):
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
                        "title": result.get("title") or "Retrieved source {0}".format(index),
                        "text": text,
                        "source_type": result.get("source_type") or "retrieval",
                        "url": result.get("url") or "",
                        "metadata": {
                            "result_index": index,
                            "query": result.get("query", ""),
                        },
                    }
                )
        return cls.from_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_dim=embedding_dim,
        )

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Dict],
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        embedding_dim: int = 64,
    ):
        chunks = []
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
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
                        "metadata": dict(document.get("metadata") or {}, chunk_index=chunk_index),
                        "embedding": hash_embedding(content, dim=embedding_dim),
                        "token_count_estimate": estimate_tokens(content),
                    }
                )
        return cls(
            chunks=chunks,
            embedding_dim=embedding_dim,
            config={
                "index_type": "local_json_hash_embedding",
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "ann": "linear_scan_baseline",
                "hnsw_ready_params": {"M": 16, "ef_construct": 100, "ef_search": 64},
            },
        )

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "embedding_dim": self.embedding_dim,
                    "config": self.config,
                    "chunks": self.chunks,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            chunks=data.get("chunks") or [],
            embedding_dim=data.get("embedding_dim") or 64,
            config=data.get("config") or {},
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.65,
        expected_keywords: Optional[List[str]] = None,
        forbidden_keywords: Optional[List[str]] = None,
        rerank: bool = True,
    ):
        query_embedding = hash_embedding(query, dim=self.embedding_dim)
        query_terms = tokenize(query)
        expected_keywords = expected_keywords or []
        forbidden_keywords = forbidden_keywords or []
        scored = []
        for index, chunk in enumerate(self.chunks):
            text = "{0}\n{1}".format(chunk.get("title", ""), chunk.get("content", ""))
            lexical = lexical_score(query_terms, tokenize(text))
            vector = cosine_similarity(query_embedding, chunk.get("embedding") or [])
            hybrid = alpha * vector + (1.0 - alpha) * lexical
            rerank_score = hybrid
            expected_hits = keyword_hits(text, expected_keywords)
            forbidden_hits = keyword_hits(text, forbidden_keywords)
            if rerank:
                rerank_score += min(0.25, 0.08 * len(expected_hits))
                rerank_score -= min(0.35, 0.12 * len(forbidden_hits))
            enriched = dict(chunk)
            enriched.update(
                {
                    "lexical_score": round(lexical, 6),
                    "vector_score": round(vector, 6),
                    "hybrid_score": round(hybrid, 6),
                    "rerank_score": round(rerank_score, 6),
                    "expected_keyword_hits": expected_hits,
                    "forbidden_keyword_hits": forbidden_hits,
                    "score": round(rerank_score, 6),
                }
            )
            scored.append((rerank_score, index, enriched))
        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        return [item for score, _, item in scored[:top_k] if score > 0]


class ContextCompressionRetriever:
    """Wrapper between retriever/index and prompt assembly."""

    def __init__(
        self,
        index: PaperStormRAGIndex,
        max_context_chars: int = 2400,
        history_ratio: float = 0.3,
        evidence_ratio: float = 0.7,
        candidate_multiplier: int = 4,
    ):
        self.index = index
        self.max_context_chars = int(max_context_chars)
        self.history_ratio = float(history_ratio)
        self.evidence_ratio = float(evidence_ratio)
        self.candidate_multiplier = max(1, int(candidate_multiplier))

    def retrieve(
        self,
        query: str,
        history: Optional[List[Dict]] = None,
        top_k: int = 5,
        expected_keywords: Optional[List[str]] = None,
        forbidden_keywords: Optional[List[str]] = None,
    ):
        expected_keywords = expected_keywords or []
        forbidden_keywords = forbidden_keywords or []
        candidates = self.index.search(
            query,
            top_k=max(top_k, top_k * self.candidate_multiplier),
            expected_keywords=expected_keywords,
            forbidden_keywords=forbidden_keywords,
        )
        filtered = [
            chunk
            for chunk in candidates
            if not chunk.get("forbidden_keyword_hits") or chunk.get("expected_keyword_hits")
        ]
        if not filtered:
            filtered = candidates
        selected = filtered[:top_k]
        history_budget = int(self.max_context_chars * self.history_ratio)
        evidence_budget = self.max_context_chars - history_budget
        compressed_history = compress_context(
            history or [],
            expected_keywords=expected_keywords,
            forbidden_keywords=forbidden_keywords,
            max_chars=max(120, history_budget),
        )
        compressed_evidence = _compress_chunks(
            selected,
            query=query,
            max_chars=max(120, evidence_budget),
            expected_keywords=expected_keywords,
        )
        prompt_context = _trim_join(
            [
                compressed_history.get("summary", ""),
                compressed_evidence,
            ],
            max_chars=self.max_context_chars,
        )
        return {
            "query": query,
            "chunks": selected,
            "compressed_history": compressed_history,
            "compressed_evidence": compressed_evidence,
            "prompt_context": prompt_context,
            "budget": {
                "max_context_chars": self.max_context_chars,
                "history_ratio": self.history_ratio,
                "evidence_ratio": self.evidence_ratio,
                "history_chars": len(compressed_history.get("summary", "")),
                "evidence_chars": len(compressed_evidence),
            },
            "audit": {
                "candidate_count": len(candidates),
                "coarse_filtered_count": len(candidates) - len(filtered),
                "selected_count": len(selected),
                "compression": "coarse_filter_then_rule_sentence_extract",
            },
        }


class PaperStormLongTermMemoryIndex:
    """Persistent local vector-like memory index for cross-session recall."""

    def __init__(self, records: Optional[List[Dict]] = None, embedding_dim: int = 64):
        self.records = records or []
        self.embedding_dim = int(embedding_dim)

    @classmethod
    def from_memory_store(cls, memory_store, embedding_dim: int = 64):
        records = []
        data = memory_store.to_dict()
        for kind in ("semantic", "episodic", "working"):
            for item in data.get(kind, []):
                record = dict(item)
                record["kind"] = kind
                record["embedding"] = hash_embedding(record.get("content", ""), dim=embedding_dim)
                records.append(record)
        if data.get("preferences"):
            records.append(
                {
                    "kind": "preferences",
                    "content": json.dumps(data["preferences"], ensure_ascii=False),
                    "metadata": {},
                    "embedding": hash_embedding(
                        json.dumps(data["preferences"], ensure_ascii=False),
                        dim=embedding_dim,
                    ),
                }
            )
        return cls(records=records, embedding_dim=embedding_dim)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"embedding_dim": self.embedding_dim, "records": self.records},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            records=data.get("records") or [],
            embedding_dim=data.get("embedding_dim") or 64,
        )

    def recall(self, query: str, top_k: int = 5):
        query_embedding = hash_embedding(query, dim=self.embedding_dim)
        query_terms = tokenize(query)
        scored = []
        for index, record in enumerate(self.records):
            content = record.get("content", "")
            score = 0.7 * cosine_similarity(query_embedding, record.get("embedding") or [])
            score += 0.3 * lexical_score(query_terms, tokenize(content))
            enriched = dict(record)
            enriched["score"] = round(score, 6)
            scored.append((score, index, enriched))
        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        return [item for score, _, item in scored[:top_k] if score > 0]


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100):
    text = " ".join(str(text or "").split())
    if not text:
        return []
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    chunks = []
    start = 0
    step = chunk_size - chunk_overlap
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def hash_embedding(text: str, dim: int = 64):
    vector = [0.0] * int(dim)
    for token in tokenize(text):
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % dim
        sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [round(value / norm, 8) for value in vector]


def cosine_similarity(left: List[float], right: List[float]):
    if not left or not right:
        return 0.0
    length = min(len(left), len(right))
    return max(0.0, sum(left[i] * right[i] for i in range(length)))


def lexical_score(query_terms, text_terms):
    if not query_terms or not text_terms:
        return 0.0
    return len(query_terms & text_terms) / max(1, len(query_terms))


def tokenize(text: str):
    return set(re.findall(r"[a-zA-Z0-9_\-]+|[\u4e00-\u9fff]+", str(text).lower()))


def keyword_hits(text: str, keywords: Iterable[str]):
    lowered = str(text or "").lower()
    return [keyword for keyword in keywords or [] if keyword and keyword.lower() in lowered]


def estimate_tokens(text: str):
    return max(1, int(len(str(text or "")) / 4))


def _compress_chunks(chunks: List[Dict], query: str, max_chars: int, expected_keywords: List[str]):
    query_terms = tokenize(query)
    lines = []
    for chunk in chunks:
        sentences = re.split(r"(?<=[。.!?])\s+", chunk.get("content", ""))
        kept = []
        for sentence in sentences:
            if not sentence:
                continue
            if tokenize(sentence) & query_terms or keyword_hits(sentence, expected_keywords):
                kept.append(sentence)
        if not kept:
            kept = [chunk.get("content", "")[:220]]
        lines.append(
            "[{0}] {1}".format(chunk.get("chunk_id", ""), " ".join(kept).strip())
        )
    return _trim_join(lines, max_chars=max_chars)


def _trim_join(parts: List[str], max_chars: int):
    text = "\n".join(part for part in parts if part)
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def _read_first_existing(paths):
    for path in paths:
        if Path(path).exists():
            return Path(path).read_text(encoding="utf-8", errors="replace")
    return ""


def _read_json(path, default):
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default
