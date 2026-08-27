import os
import json
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock


class EmbeddingProfileRegistryTests(unittest.TestCase):
    def test_profile_lookup_exposes_declared_model_contract(self):
        from knowledge_storm.retrieval_profiles import get_embedding_profile

        profile = get_embedding_profile("cpu-zh")

        self.assertEqual("cpu-zh", profile.name)
        self.assertEqual("BAAI/bge-small-zh-v1.5", profile.model_name)
        self.assertEqual("为这个句子生成表示以用于检索相关文章：", profile.query.prompt)
        self.assertEqual("", profile.document.prompt)
        self.assertTrue(profile.query.normalize)
        self.assertFalse(profile.trust_remote_code)

    def test_known_profiles_freeze_official_revision_dimension_and_length(self):
        from knowledge_storm.retrieval_profiles import get_embedding_profile

        expected = {
            "legacy-multilingual": (
                "e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
                384,
                128,
            ),
            "cpu-zh": ("7999e1d3359715c523056ef9478215996d62a620", 512, 512),
            "cpu-multilingual": (
                "9bbca17d9273fd0d03d5725c7a4b0f6b45142062",
                768,
                8192,
            ),
            "quality-multilingual": (
                "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
                1024,
                32768,
            ),
        }

        self.assertEqual(
            expected,
            {
                name: (
                    get_embedding_profile(name).revision,
                    get_embedding_profile(name).dimension,
                    get_embedding_profile(name).max_seq_length,
                )
                for name in expected
            },
        )

    def test_unknown_profile_is_rejected_with_valid_names(self):
        from knowledge_storm.retrieval_profiles import get_embedding_profile

        with self.assertRaisesRegex(ValueError, "unsupported embedding profile: missing"):
            get_embedding_profile("missing")

    def test_gte_and_qwen_keep_their_documented_role_contracts(self):
        from knowledge_storm.retrieval_profiles import get_embedding_profile

        gte = get_embedding_profile("cpu-multilingual")
        qwen = get_embedding_profile("quality-multilingual")

        self.assertTrue(gte.trust_remote_code)
        self.assertEqual("", gte.query.prompt)
        self.assertIsNone(gte.query.prompt_name)
        self.assertEqual("", gte.document.prompt)
        self.assertEqual("", qwen.query.prompt)
        self.assertEqual("query", qwen.query.prompt_name)
        self.assertEqual("", qwen.document.prompt)
        self.assertIsNone(qwen.document.prompt_name)
        self.assertEqual({"batch_size": 1}, dict(qwen.query.encode_options))
        self.assertEqual({"batch_size": 2}, dict(qwen.document.encode_options))

    def test_known_model_overrides_resolve_back_to_frozen_profiles(self):
        from knowledge_storm.retrieval_profiles import (
            EMBEDDING_PROFILES,
            resolve_embedding_profile,
        )

        self.assertEqual(
            "cpu-zh",
            resolve_embedding_profile(
                model_name=EMBEDDING_PROFILES["cpu-zh"].model_name
            ).name,
        )
        self.assertEqual(
            "quality-multilingual",
            resolve_embedding_profile(
                model_name=EMBEDDING_PROFILES["quality-multilingual"].model_name
            ).name,
        )


class FakeEncodeModel:
    def __init__(self, dimension=3):
        self.calls = []
        self.tokenizer = None
        self.dimension = dimension
        self.max_seq_length = None

    def get_sentence_embedding_dimension(self):
        return self.dimension

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), dict(kwargs)))
        return [[1.0] + [0.0] * (self.dimension - 1) for _ in texts]


class SentenceTransformerProfileProviderTests(unittest.TestCase):
    def test_query_and_document_calls_keep_qwen_prompt_name_asymmetric(self):
        from knowledge_storm.retrieval import SentenceTransformerProvider

        model = FakeEncodeModel(dimension=1024)
        provider = SentenceTransformerProvider(
            profile="quality-multilingual", model=model
        )

        self.assertEqual(1024, len(provider.embed_documents(["passage"])[0]))
        self.assertEqual(1024, len(provider.embed_query("question")))
        self.assertEqual(1024, len(provider.embed(["legacy document"])[0]))
        self.assertEqual(
            [
                (
                    ["passage"],
                    {
                        "batch_size": 2,
                        "normalize_embeddings": True,
                        "show_progress_bar": False,
                    },
                ),
                (
                    ["question"],
                    {
                        "batch_size": 1,
                        "normalize_embeddings": True,
                        "prompt_name": "query",
                        "show_progress_bar": False,
                    },
                ),
                (
                    ["legacy document"],
                    {
                        "batch_size": 2,
                        "normalize_embeddings": True,
                        "show_progress_bar": False,
                    },
                ),
            ],
            model.calls,
        )

    def test_custom_model_override_uses_stable_custom_profile(self):
        from knowledge_storm.retrieval import build_embedding_provider

        with mock.patch.dict(
            os.environ,
            {
                "PAPERSTORM_EMBEDDING_MODEL": "example/custom-embedding",
                "PAPERSTORM_EMBEDDING_PROFILE": "cpu-zh",
            },
            clear=True,
        ):
            provider = build_embedding_provider()

        self.assertEqual("custom", provider.profile.name)
        self.assertEqual("example/custom-embedding", provider.model_name)
        self.assertEqual("explicit custom model override", provider.profile.intended_role)

    def test_profile_metadata_pins_model_loading_and_encode_length(self):
        from knowledge_storm.retrieval import SentenceTransformerProvider

        loaded = []

        def fake_loader(*args, **kwargs):
            model = FakeEncodeModel(dimension=512)
            loaded.append((args, kwargs, model))
            return model

        with mock.patch.dict(
            sys.modules,
            {"sentence_transformers": types.SimpleNamespace(SentenceTransformer=fake_loader)},
        ):
            provider = SentenceTransformerProvider(profile="cpu-zh")
            provider.embed_documents(["passage"])

        args, kwargs, model = loaded[0]
        self.assertEqual(("BAAI/bge-small-zh-v1.5",), args)
        self.assertEqual("7999e1d3359715c523056ef9478215996d62a620", kwargs["revision"])
        self.assertFalse(kwargs["trust_remote_code"])
        self.assertEqual(512, provider.dim)
        self.assertEqual(512, model.max_seq_length)


class RetrievalProfileIntegrationTests(unittest.TestCase):
    def test_memory_provider_reacts_to_environment_without_cache_clear(self):
        from knowledge_storm.memory_store import build_memory_embedding_provider

        with mock.patch.dict(
            os.environ, {"PAPERSTORM_EMBEDDING_PROFILE": "cpu-zh"}, clear=True
        ):
            first = build_memory_embedding_provider()
        with mock.patch.dict(
            os.environ,
            {"PAPERSTORM_EMBEDDING_PROFILE": "quality-multilingual"},
            clear=True,
        ):
            second = build_memory_embedding_provider()

        self.assertEqual("cpu-zh", first.profile.name)
        self.assertEqual("quality-multilingual", second.profile.name)
        self.assertIsNot(first, second)

    def test_runtime_and_memory_share_profile_default(self):
        from knowledge_storm import memory_store, retrieval_runtime

        with mock.patch.dict(
            os.environ, {"PAPERSTORM_EMBEDDING_PROFILE": "cpu-zh"}, clear=True
        ):
            retrieval_runtime._REAL_EMBEDDING_PROVIDER = None
            runtime_profile = retrieval_runtime.runtime_embedding_profile()
            memory_provider = memory_store.build_memory_embedding_provider()

        self.assertEqual("cpu-zh", runtime_profile.name)
        self.assertEqual(runtime_profile.name, memory_provider.profile.name)

    def test_index_manifest_rejects_changed_role_encoding_contract(self):
        from knowledge_storm.retrieval import HybridPaperIndex, SentenceTransformerProvider
        from knowledge_storm.retrieval_profiles import get_embedding_profile

        with tempfile.TemporaryDirectory() as temp_dir:
            profile = get_embedding_profile("cpu-zh")
            index = HybridPaperIndex.from_documents(
                [{"document_id": "doc", "title": "title", "text": "passage"}],
                embedding_provider=SentenceTransformerProvider(
                    profile=profile, model=FakeEncodeModel(dimension=512)
                ),
            )
            path = Path(temp_dir) / "index.json"
            index.save(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["manifest"].pop("embedding_profile_contract", None)
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                RuntimeError, "embedding profile contract is missing; rebuild"
            ):
                HybridPaperIndex.load(
                    path,
                    embedding_provider=SentenceTransformerProvider(
                        profile=profile, model=FakeEncodeModel(dimension=512)
                    ),
                )
            index.save(path)
            loaded = HybridPaperIndex.load(
                path,
                embedding_provider=SentenceTransformerProvider(
                    profile=profile, model=FakeEncodeModel(dimension=512)
                ),
            )
            changed_profile = replace(
                profile,
                query=replace(profile.query, prompt="changed query contract"),
            )
            changed_provider = SentenceTransformerProvider(
                profile=changed_profile, model=FakeEncodeModel(dimension=512)
            )

            with self.assertRaisesRegex(ValueError, "embedding role contract mismatch"):
                HybridPaperIndex.load(path, embedding_provider=changed_provider)

        self.assertEqual(index.chunks, loaded.chunks)
        self.assertEqual("cpu-zh", index.manifest["embedding_profile"])
        self.assertEqual(
            profile.manifest_contract(), index.manifest["embedding_profile_contract"]
        )


if __name__ == "__main__":
    unittest.main()
