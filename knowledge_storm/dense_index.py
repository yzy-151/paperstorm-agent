"""Dense retrieval backends with an Exact oracle and optional HNSW acceleration."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DenseSearchResult:
    indices: tuple
    scores: tuple
    backend: str
    reason: str = ""


def _normalized_matrix(vectors):
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("dense vectors must be a two-dimensional matrix")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("dense vectors must not be empty")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)


def _normalized_query(query, dimension):
    vector = np.asarray(query, dtype=np.float32).reshape(-1)
    if len(vector) != int(dimension):
        raise ValueError(
            "query dimension mismatch: expected {0}, got {1}".format(
                dimension, len(vector)
            )
        )
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


class ExactDenseBackend:
    name = "exact"

    def __init__(self, vectors):
        self.vectors = _normalized_matrix(vectors)
        self.count, self.dimension = self.vectors.shape

    def search(self, query, top_k, allowed_indices=None, reason=""):
        if allowed_indices is None:
            indices = np.arange(self.count, dtype=np.int64)
        else:
            indices = np.asarray(
                sorted({int(value) for value in allowed_indices}), dtype=np.int64
            )
            if len(indices) and (indices[0] < 0 or indices[-1] >= self.count):
                raise ValueError("allowed dense index is out of range")
        if not len(indices):
            return DenseSearchResult((), (), self.name, reason)
        query_vector = _normalized_query(query, self.dimension)
        scores = self.vectors[indices] @ query_vector
        order = sorted(
            range(len(indices)), key=lambda position: (-scores[position], indices[position])
        )[: max(0, min(int(top_k), len(indices)))]
        return DenseSearchResult(
            tuple(int(indices[position]) for position in order),
            tuple(float(scores[position]) for position in order),
            self.name,
            reason,
        )


class HnswDenseBackend:
    name = "hnsw"
    implementation = "usearch"

    def __init__(self, vectors, ef_search=100, ef_construction=200, m=16):
        try:
            from usearch.index import Index
        except ImportError as exc:
            raise RuntimeError(
                "HNSW dense retrieval requires optional dependency usearch"
            ) from exc
        matrix = _normalized_matrix(vectors)
        self.count, self.dimension = matrix.shape
        self.ef_search = max(int(ef_search), 1)
        self.ef_construction = max(int(ef_construction), 1)
        self.m = max(int(m), 2)
        self._index = Index(
            ndim=self.dimension,
            metric="cos",
            dtype="f32",
            connectivity=self.m,
            expansion_add=self.ef_construction,
            expansion_search=self.ef_search,
        )
        self._index.add(np.arange(self.count, dtype=np.uint64), matrix)

    def search(self, query, top_k):
        limit = max(0, min(int(top_k), self.count))
        if limit == 0:
            return DenseSearchResult((), (), self.name)
        vector = _normalized_query(query, self.dimension)
        matches = self._index.search(vector, count=limit)
        pairs = sorted(
            (
                (int(label), 1.0 - float(distance))
                for label, distance in zip(matches.keys, matches.distances)
            ),
            key=lambda item: (-item[1], item[0]),
        )
        return DenseSearchResult(
            tuple(item[0] for item in pairs),
            tuple(item[1] for item in pairs),
            self.name,
        )

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._index.save(str(path))
        metadata = {
            "schema": "paperstorm-hnsw-usearch-v1",
            "implementation": self.implementation,
            "count": self.count,
            "dimension": self.dimension,
            "ef_search": self.ef_search,
            "ef_construction": self.ef_construction,
            "m": self.m,
        }
        self._metadata_path(path).write_text(
            json.dumps(metadata, sort_keys=True, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path):
        try:
            from usearch.index import Index
        except ImportError as exc:
            raise RuntimeError(
                "HNSW dense retrieval requires optional dependency usearch"
            ) from exc
        path = Path(path)
        metadata = json.loads(cls._metadata_path(path).read_text(encoding="utf-8"))
        if metadata.get("schema") != "paperstorm-hnsw-usearch-v1":
            raise ValueError("unsupported HNSW index metadata")
        instance = cls.__new__(cls)
        instance.count = int(metadata["count"])
        instance.dimension = int(metadata["dimension"])
        instance.ef_search = int(metadata["ef_search"])
        instance.ef_construction = int(metadata["ef_construction"])
        instance.m = int(metadata["m"])
        instance._index = Index.restore(str(path))
        return instance

    @staticmethod
    def _metadata_path(path):
        return Path(str(path) + ".meta.json")


class AutoDenseBackend:
    """Choose HNSW by scale while preserving pre-retrieval ACL semantics."""

    def __init__(
        self,
        vectors,
        mode="auto",
        ann_threshold=20_000,
        ef_search=100,
        ef_construction=200,
        m=16,
    ):
        mode = str(mode or "auto").lower()
        if mode not in {"auto", "exact", "hnsw"}:
            raise ValueError("unsupported dense backend: {0}".format(mode))
        self.mode = mode
        self.ann_threshold = max(1, int(ann_threshold))
        self.exact = ExactDenseBackend(vectors)
        self.hnsw = None
        should_build_hnsw = mode == "hnsw" or (
            mode == "auto" and self.exact.count >= self.ann_threshold
        )
        if should_build_hnsw:
            try:
                self.hnsw = HnswDenseBackend(
                    vectors,
                    ef_search=ef_search,
                    ef_construction=ef_construction,
                    m=m,
                )
            except RuntimeError:
                if mode == "hnsw":
                    raise

    def search(self, query, top_k, allowed_indices=None):
        if allowed_indices is not None:
            return self.exact.search(
                query,
                top_k,
                allowed_indices=allowed_indices,
                reason="acl_exact_fallback",
            )
        if self.hnsw is not None:
            return self.hnsw.search(query, top_k)
        reason = (
            "configured_exact"
            if self.mode == "exact"
            else "hnsw_dependency_unavailable"
            if self.exact.count >= self.ann_threshold
            else "below_ann_threshold"
        )
        return self.exact.search(query, top_k, reason=reason)
