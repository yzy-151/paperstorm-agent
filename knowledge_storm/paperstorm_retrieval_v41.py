import json
import math
import os
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence


def multilingual_tokenize(text: str) -> List[str]:
    """Tokenize exact Latin terms and add CJK unigrams/bigrams for BM25."""
    text = str(text or "").lower()
    tokens = re.findall(r"[a-z0-9]+(?:[-./][a-z0-9]+)*", text)
    for sequence in re.findall(r"[\u3400-\u9fff]+", text):
        tokens.extend(sequence)
        tokens.extend(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return tokens


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
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        cache_folder: Optional[str] = None,
        device: Optional[str] = None,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.name = "sentence-transformers:{0}".format(model_name)
        self.model = SentenceTransformer(
            model_name,
            cache_folder=cache_folder or os.getenv("PAPERSTORM_MODEL_CACHE"),
            device=device,
        )
        dimension_getter = getattr(
            self.model,
            "get_embedding_dimension",
            self.model.get_sentence_embedding_dimension,
        )
        self.dim = int(dimension_getter())

    def embed(self, texts: Iterable[str]):
        vectors = self.model.encode(
            list(texts),
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return [[float(value) for value in vector] for vector in vectors]

    def embed_query(self, text: str):
        return self.embed([text])[0]


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

            self._model = CrossEncoder(
                self.model_name,
                cache_folder=self.cache_folder,
                device=self.device,
            )
        return [float(value) for value in self._model.predict(pairs, show_progress_bar=False)]

    def rerank(self, query: str, candidates: Sequence[Dict], top_k: Optional[int] = None):
        candidates = [dict(item) for item in candidates]
        pairs = [
            (query, str(item.get("retrieval_content") or item.get("content") or ""))
            for item in candidates
        ]
        scores = self._scores(pairs) if pairs else []
        for item, score in zip(candidates, scores):
            item["rerank_score"] = round(float(score), 8)
        candidates.sort(
            key=lambda item: (-item.get("rerank_score", float("-inf")), item.get("chunk_id", ""))
        )
        return candidates[:top_k] if top_k is not None else candidates


class HybridPaperIndex:
    """Persistent BM25 + dense index with RRF and optional Cross-Encoder."""

    schema_version = "paperstorm-hybrid-index-v4.1"

    def __init__(self, chunks: Iterable[Dict], embedding_provider, embeddings=None, manifest=None):
        self.chunks = [self._normalize_chunk(item, index) for index, item in enumerate(chunks)]
        self.embedding_provider = embedding_provider
        self._tokens = [multilingual_tokenize(self._search_text(item)) for item in self.chunks]
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError("V4.1 BM25 requires dependency rank-bm25") from exc
        self._bm25 = BM25Okapi(self._tokens)
        self.embeddings = embeddings or embedding_provider.embed(
            [self._search_text(item) for item in self.chunks]
        )
        if len(self.embeddings) != len(self.chunks):
            raise ValueError("embedding count must match chunk count")
        dimension = int(getattr(embedding_provider, "dim", 0) or 0)
        if not dimension and self.embeddings:
            dimension = len(self.embeddings[0])
        self.manifest = manifest or {
            "schema_version": self.schema_version,
            "embedding_model": str(getattr(embedding_provider, "name", "unknown")),
            "embedding_dimension": dimension,
            "normalized": bool(getattr(embedding_provider, "normalize", False)),
            "chunk_count": len(self.chunks),
        }

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 5,
        candidate_k: Optional[int] = None,
        reranker=None,
        rank_constant: int = 60,
    ) -> List[Dict]:
        if mode not in {"bm25", "dense", "hybrid", "hybrid_rerank"}:
            raise ValueError("unsupported retrieval mode: {0}".format(mode))
        candidate_k = min(len(self.chunks), candidate_k or max(top_k * 4, 20))
        bm25 = self._bm25_search(query, candidate_k)
        dense = self._dense_search(query, candidate_k)
        if mode == "bm25":
            selected = bm25
        elif mode == "dense":
            selected = dense
        else:
            selected = reciprocal_rank_fusion([bm25, dense], rank_constant=rank_constant)
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
            output.append(enriched)
        return output

    def _bm25_search(self, query: str, top_k: int):
        scores = self._bm25.get_scores(multilingual_tokenize(query))
        ranked = []
        for index in sorted(range(len(scores)), key=lambda value: (-scores[value], value))[:top_k]:
            item = dict(self.chunks[index])
            item["bm25_score"] = round(float(scores[index]), 8)
            ranked.append(item)
        return ranked

    def _dense_search(self, query: str, top_k: int):
        query_vector = self.embedding_provider.embed_query(query)
        scores = [_cosine(query_vector, vector) for vector in self.embeddings]
        ranked = []
        for index in sorted(range(len(scores)), key=lambda value: (-scores[value], value))[:top_k]:
            item = dict(self.chunks[index])
            item["dense_score"] = round(float(scores[index]), 8)
            ranked.append(item)
        return ranked

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"manifest": self.manifest, "chunks": self.chunks, "embeddings": self.embeddings},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path, embedding_provider):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        manifest = payload.get("manifest") or {}
        actual_name = str(getattr(embedding_provider, "name", "unknown"))
        actual_dim = int(getattr(embedding_provider, "dim", 0) or 0)
        if manifest.get("embedding_model") != actual_name:
            raise ValueError(
                "embedding model mismatch: index={0}, provider={1}".format(
                    manifest.get("embedding_model"), actual_name
                )
            )
        if int(manifest.get("embedding_dimension") or 0) != actual_dim:
            raise ValueError(
                "embedding dimension mismatch: index={0}, provider={1}".format(
                    manifest.get("embedding_dimension"), actual_dim
                )
            )
        if bool(manifest.get("normalized")) != bool(
            getattr(embedding_provider, "normalize", False)
        ):
            raise ValueError("embedding normalization mismatch")
        return cls(
            payload.get("chunks") or [],
            embedding_provider=embedding_provider,
            embeddings=payload.get("embeddings") or [],
            manifest=manifest,
        )

    @staticmethod
    def _normalize_chunk(item, index):
        chunk = dict(item)
        chunk["chunk_id"] = str(chunk.get("chunk_id") or "chunk-{0}".format(index + 1))
        chunk["content"] = str(chunk.get("content") or chunk.get("text") or "")
        chunk.setdefault("retrieval_content", chunk["content"])
        return chunk

    @staticmethod
    def _search_text(item):
        return "{0}\n{1}".format(item.get("title", ""), item.get("retrieval_content", ""))


def _cosine(left, right):
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)
