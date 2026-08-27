import os
import tempfile
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


class FakeEncodeModel:
    def __init__(self):
        self.calls = []
        self.tokenizer = None

    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), dict(kwargs)))
        return [[1.0, 0.0, 0.0] for _ in texts]


class SentenceTransformerProfileProviderTests(unittest.TestCase):
    def test_query_and_document_calls_keep_qwen_prompt_name_asymmetric(self):
        from knowledge_storm.retrieval import SentenceTransformerProvider

        model = FakeEncodeModel()
        provider = SentenceTransformerProvider(
            profile="quality-multilingual", model=model
        )

        self.assertEqual([[1.0, 0.0, 0.0]], provider.embed_documents(["passage"]))
        self.assertEqual([1.0, 0.0, 0.0], provider.embed_query("question"))
        self.assertEqual([[1.0, 0.0, 0.0]], provider.embed(["legacy document"]))
        self.assertEqual(
            [
                (["passage"], {"normalize_embeddings": True, "show_progress_bar": False}),
                (
                    ["question"],
                    {
                        "normalize_embeddings": True,
                        "prompt_name": "query",
                        "show_progress_bar": False,
                    },
                ),
                (
                    ["legacy document"],
                    {"normalize_embeddings": True, "show_progress_bar": False},
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


class RetrievalProfileIntegrationTests(unittest.TestCase):
    def test_runtime_and_memory_share_profile_default(self):
        from knowledge_storm import memory_store, retrieval_runtime

        with mock.patch.dict(
            os.environ, {"PAPERSTORM_EMBEDDING_PROFILE": "cpu-zh"}, clear=True
        ):
            retrieval_runtime._REAL_EMBEDDING_PROVIDER = None
            memory_store.build_memory_embedding_provider.cache_clear()
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
                    profile=profile, model=FakeEncodeModel()
                ),
            )
            path = Path(temp_dir) / "index.json"
            index.save(path)
            loaded = HybridPaperIndex.load(
                path,
                embedding_provider=SentenceTransformerProvider(
                    profile=profile, model=FakeEncodeModel()
                ),
            )
            changed_profile = replace(
                profile,
                query=replace(profile.query, prompt="changed query contract"),
            )
            changed_provider = SentenceTransformerProvider(
                profile=changed_profile, model=FakeEncodeModel()
            )

            with self.assertRaisesRegex(ValueError, "embedding role contract mismatch"):
                HybridPaperIndex.load(path, embedding_provider=changed_provider)

        self.assertEqual(index.chunks, loaded.chunks)
        self.assertEqual("cpu-zh", index.manifest["embedding_profile"])
        self.assertEqual(
            profile.manifest_contract(), index.manifest["embedding_role_contract"]
        )


if __name__ == "__main__":
    unittest.main()
