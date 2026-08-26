import tempfile
import unittest
from pathlib import Path


class RetrievalGovernanceTest(unittest.TestCase):
    def _index(self):
        from knowledge_storm.retrieval import HashEmbeddingProvider, HybridPaperIndex

        return HybridPaperIndex.from_documents(
            [
                {
                    "document_id": "public-doc",
                    "title": "Public document",
                    "text": "public guidance for general retrieval governance",
                },
                {
                    "document_id": "private-doc",
                    "title": "Private document",
                    "text": "private secret needle retrieval governance",
                },
            ],
            embedding_provider=HashEmbeddingProvider(),
            chunk_size=500,
            chunk_overlap=0,
        )

    def test_scope_excludes_private_chunks_before_bm25_and_dense_ranking(self):
        index = self._index()
        seen_scopes = []
        original_bm25 = index._bm25_search
        original_dense = index._dense_search

        def record_bm25(query, top_k, candidate_indices=None):
            seen_scopes.append(tuple(candidate_indices or ()))
            return original_bm25(query, top_k, candidate_indices)

        def record_dense(query, top_k, candidate_indices=None):
            seen_scopes.append(tuple(candidate_indices or ()))
            return original_dense(query, top_k, candidate_indices)

        index._bm25_search = record_bm25
        index._dense_search = record_dense

        for mode in ("bm25", "dense", "hybrid"):
            results = index.search(
                "private secret needle",
                mode=mode,
                top_k=5,
                allowed_document_ids=("public-doc",),
            )
            self.assertTrue(results)
            self.assertEqual({"public-doc"}, {item["document_id"] for item in results})

        public_indices = tuple(
            index
            for index, chunk in enumerate(index.chunks)
            if chunk["document_id"] == "public-doc"
        )
        self.assertTrue(seen_scopes)
        self.assertTrue(all(scope == public_indices for scope in seen_scopes))

        self.assertEqual(
            [],
            index.search(
                "private secret needle",
                mode="hybrid",
                top_k=5,
                allowed_document_ids=(),
            ),
        )

    def test_unscoped_index_search_remains_compatible(self):
        index = self._index()

        legacy = index.search("private secret needle", mode="hybrid", top_k=2)
        explicit_none = index.search(
            "private secret needle",
            mode="hybrid",
            top_k=2,
            allowed_document_ids=None,
        )

        self.assertEqual(legacy, explicit_none)
        self.assertEqual("private-doc", legacy[0]["document_id"])

    def test_cache_identity_changes_with_tenant_or_user_policy(self):
        from knowledge_storm.paperstorm_enterprise_kb import _answer_cache_identity

        shared = {
            "kb_id": "kb-1",
            "index_revision": {"index_version": 1, "schema_revision": 3},
            "top_k": 4,
            "query": "governance",
            "search_plan": {"standalone_query": "governance"},
        }
        owner = _answer_cache_identity(
            tenant_id="tenant-a", user_id="owner", policy_digest="policy-a", **shared
        )
        viewer = _answer_cache_identity(
            tenant_id="tenant-a", user_id="viewer", policy_digest="policy-b", **shared
        )
        other_tenant = _answer_cache_identity(
            tenant_id="tenant-b", user_id="owner", policy_digest="policy-a", **shared
        )

        self.assertNotEqual(owner, viewer)
        self.assertNotEqual(owner, other_tenant)

    def test_enterprise_ask_scopes_accessible_documents_and_cache_by_policy(self):
        from knowledge_storm.paperstorm_enterprise_kb import EnterpriseKnowledgeBaseService

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            public = root / "public.txt"
            private = root / "private.txt"
            public.write_text("public employee handbook", encoding="utf-8")
            private.write_text("private salary secret needle", encoding="utf-8")
            service = EnterpriseKnowledgeBaseService(root)
            kb = service.create_knowledge_base(
                "governed-kb",
                [str(public), str(private)],
                embedding_provider="hash",
                allowed_user_ids=["viewer"],
            )

            service.control.register_resource(
                tenant_id="local",
                resource_type="document",
                resource_id="{0}:doc-2".format(kb["kb_id"]),
                owner_user_id="private-owner",
                allowed_user_ids=[],
                metadata={"kb_id": kb["kb_id"]},
                version=1,
            )

            owner = service.ask(kb["kb_id"], "needle", user_id="local-user")
            viewer = service.ask(kb["kb_id"], "needle", user_id="viewer")

            self.assertFalse(owner["cache_hit"])
            self.assertFalse(viewer["cache_hit"])
            self.assertEqual(
                {"doc-1"},
                {item["document_id"] for item in owner["retrieval"]["results"]},
            )
            self.assertEqual(
                {"doc-1"},
                {item["document_id"] for item in viewer["retrieval"]["results"]},
            )
            self.assertNotEqual(
                owner["retrieval"]["policy_digest"],
                viewer["retrieval"]["policy_digest"],
            )


if __name__ == "__main__":
    unittest.main()
