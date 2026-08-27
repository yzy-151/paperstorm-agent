"""Frozen embedding profiles shared by PaperStorm retrieval surfaces."""

import os
from dataclasses import dataclass
from types import MappingProxyType
from typing import Optional, Tuple


DEFAULT_EMBEDDING_PROFILE = "cpu-multilingual"
CUSTOM_EMBEDDING_PROFILE = "custom"


@dataclass(frozen=True)
class EmbeddingEncoding:
    """One role's documented SentenceTransformers encoding contract."""

    role: str
    intended_role: str
    prompt: str = ""
    prompt_name: Optional[str] = None
    normalize: bool = True
    encode_options: Tuple[Tuple[str, object], ...] = ()

    def encode_kwargs(self):
        options = dict(self.encode_options)
        options["normalize_embeddings"] = self.normalize
        if self.prompt_name:
            options["prompt_name"] = self.prompt_name
        return options

    def manifest_contract(self):
        return {
            "role": self.role,
            "intended_role": self.intended_role,
            "prompt": self.prompt,
            "prompt_name": self.prompt_name,
            "normalize": self.normalize,
            "encode_options": [list(item) for item in self.encode_options],
        }


@dataclass(frozen=True)
class EmbeddingProfile:
    """Immutable model and role-encoding metadata for a dense retriever."""

    name: str
    model_name: str
    revision: Optional[str]
    dimension: int
    max_seq_length: int
    query: EmbeddingEncoding
    document: EmbeddingEncoding
    trust_remote_code: bool
    intended_role: str

    def manifest_contract(self):
        return {
            "name": self.name,
            "model_name": self.model_name,
            "revision": self.revision,
            "dimension": self.dimension,
            "max_seq_length": self.max_seq_length,
            "query": self.query.manifest_contract(),
            "document": self.document.manifest_contract(),
            "trust_remote_code": self.trust_remote_code,
            "intended_role": self.intended_role,
        }


def _role(role, intended_role, prompt="", prompt_name=None):
    return EmbeddingEncoding(
        role=role,
        intended_role=intended_role,
        prompt=prompt,
        prompt_name=prompt_name,
        normalize=True,
    )


EMBEDDING_PROFILES = MappingProxyType({
    "legacy-multilingual": EmbeddingProfile(
        name="legacy-multilingual",
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        revision="e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
        dimension=384,
        max_seq_length=128,
        query=_role("query", "multilingual semantic-similarity query"),
        document=_role("document", "multilingual semantic-similarity document"),
        trust_remote_code=False,
        intended_role="legacy multilingual compatibility",
    ),
    "cpu-zh": EmbeddingProfile(
        name="cpu-zh",
        model_name="BAAI/bge-small-zh-v1.5",
        revision="7999e1d3359715c523056ef9478215996d62a620",
        dimension=512,
        max_seq_length=512,
        query=_role(
            "query",
            "short Chinese retrieval query",
            prompt="为这个句子生成表示以用于检索相关文章：",
        ),
        document=_role("document", "Chinese retrieval passage"),
        trust_remote_code=False,
        intended_role="CPU-oriented Chinese passage retrieval",
    ),
    "cpu-multilingual": EmbeddingProfile(
        name="cpu-multilingual",
        model_name="Alibaba-NLP/gte-multilingual-base",
        revision="9bbca17d9273fd0d03d5725c7a4b0f6b45142062",
        dimension=768,
        max_seq_length=8192,
        query=_role("query", "multilingual retrieval query"),
        document=_role("document", "multilingual retrieval passage"),
        trust_remote_code=True,
        intended_role="CPU-oriented multilingual retrieval",
    ),
    "quality-multilingual": EmbeddingProfile(
        name="quality-multilingual",
        model_name="Qwen/Qwen3-Embedding-0.6B",
        revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        dimension=1024,
        max_seq_length=32768,
        query=_role(
            "query",
            "instruction-aware multilingual retrieval query",
            prompt_name="query",
        ),
        document=_role("document", "multilingual retrieval passage"),
        trust_remote_code=False,
        intended_role="quality-oriented multilingual retrieval",
    ),
})


def get_embedding_profile(name=None):
    key = str(
        name or os.getenv("PAPERSTORM_EMBEDDING_PROFILE") or DEFAULT_EMBEDDING_PROFILE
    ).strip().lower()
    try:
        return EMBEDDING_PROFILES[key]
    except KeyError as exc:
        raise ValueError(
            "unsupported embedding profile: {0}; expected one of {1}".format(
                key, ", ".join(sorted(EMBEDDING_PROFILES))
            )
        ) from exc


def custom_embedding_profile(model_name):
    model_name = str(model_name or "").strip()
    if not model_name:
        raise ValueError("custom embedding model name is required")
    return EmbeddingProfile(
        name=CUSTOM_EMBEDDING_PROFILE,
        model_name=model_name,
        revision=None,
        dimension=0,
        max_seq_length=0,
        query=_role("query", "custom model query"),
        document=_role("document", "custom model document"),
        trust_remote_code=False,
        intended_role="explicit custom model override",
    )


def resolve_embedding_profile(profile_name=None, model_name=None):
    override = str(model_name or os.getenv("PAPERSTORM_EMBEDDING_MODEL") or "").strip()
    if override:
        for profile in EMBEDDING_PROFILES.values():
            if profile.model_name == override:
                return profile
        return custom_embedding_profile(override)
    return get_embedding_profile(profile_name)
