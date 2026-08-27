# PaperStorm Retrieval Stack Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace fragmented MiniLM defaults with measured embedding profiles and add scalable, auditable ANN, Chinese lexical analysis, fair parent expansion, and reranker runtime profiles.

**Architecture:** Keep `RetrievalPipeline` as the public contract. Move model metadata, lexical analysis, and dense search behind focused adapters, then let `HybridPaperIndex` compose them. Exact search remains the correctness oracle; HNSW is an optional acceleration backend and scoped requests fail closed to authorized Exact search when a safe ANN partition is unavailable.

**Tech Stack:** Python 3.10/3.11, NumPy, SentenceTransformers, rank-bm25, USearch HNSW, Jieba, unittest, SciFact, QASPER.

---

## File Structure

- Create `knowledge_storm/retrieval_profiles.py`: immutable embedding/reranker profile registry and model-specific encode options.
- Create `knowledge_storm/dense_index.py`: Exact/HNSW backend contract, persistence, auto selection and ACL-safe fallback.
- Create `knowledge_storm/text_analyzers.py`: CJK bigram and Jieba-domain analyzers.
- Create `knowledge_storm/resources/paperstorm_domain_terms.txt`: versioned RF/PIM terminology.
- Modify `knowledge_storm/retrieval.py`: compose profiles, analyzers, dense backend and fair parent expansion.
- Modify `knowledge_storm/retrieval_pipeline.py`: emit backend, analyzer, parent allocation and reranker device Trace.
- Modify `knowledge_storm/retrieval_runtime.py`: one profile-based runtime default and cache identity.
- Modify `knowledge_storm/memory_store.py`: use the same profile registry instead of a separate MiniLM default.
- Modify `examples/storm_examples/run_paperstorm_milestone.py`: profile-aware reproducible benchmark arguments.
- Create `examples/storm_examples/benchmark_embedding_profiles.py`: deterministic 10% four-model comparison.
- Create `examples/storm_examples/benchmark_dense_scale.py`: Exact/HNSW 100k comparison and 2M estimator.
- Create focused tests under `tests/test_retrieval_profiles.py`, `tests/test_dense_index.py`, `tests/test_text_analyzers.py`, and `tests/test_parent_context_budget.py`.

## Task 1: Unified Retrieval Profiles

**Files:**
- Create: `knowledge_storm/retrieval_profiles.py`
- Modify: `knowledge_storm/retrieval.py`
- Modify: `knowledge_storm/retrieval_runtime.py`
- Modify: `knowledge_storm/memory_store.py`
- Test: `tests/test_retrieval_profiles.py`

- [ ] **Step 1: Write profile and asymmetric encoding tests**

```python
def test_profiles_expose_stable_model_contracts():
    gte = get_embedding_profile("cpu-multilingual")
    qwen = get_embedding_profile("quality-multilingual")
    assert gte.model_name == "Alibaba-NLP/gte-multilingual-base"
    assert qwen.model_name == "Qwen/Qwen3-Embedding-0.6B"
    assert gte.query_prompt != gte.document_prompt

def test_runtime_and_memory_share_default_profile(monkeypatch):
    monkeypatch.setenv("PAPERSTORM_EMBEDDING_PROFILE", "cpu-multilingual")
    assert runtime_embedding_profile().name == build_memory_embedding_provider().profile.name
```

- [ ] **Step 2: Run tests and verify failure**

Run: `D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest tests.test_retrieval_profiles -v`

Expected: import failure for `knowledge_storm.retrieval_profiles`.

- [ ] **Step 3: Implement frozen profile records and provider role encoding**

```python
@dataclass(frozen=True)
class EmbeddingProfile:
    name: str
    model_name: str
    query_prompt: str = ""
    document_prompt: str = ""
    trust_remote_code: bool = False

def get_embedding_profile(name=None):
    key = str(name or os.getenv("PAPERSTORM_EMBEDDING_PROFILE") or "cpu-multilingual")
    return EMBEDDING_PROFILES[key]
```

Extend `SentenceTransformerProvider.embed_query()` and `embed_documents()` so each role applies its declared prompt and writes the full contract into index manifests.

- [ ] **Step 4: Remove split defaults and run focused regression**

Run: `D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest tests.test_retrieval_profiles tests.test_retrieval_runtime tests.test_memory_store tests.test_retrieval -q`

Expected: all tests pass; no production path hard-codes a MiniLM model outside the registry.

- [ ] **Step 5: Commit**

```text
feat(retrieval): unify embedding profiles
```

## Task 2: Deterministic 10% Embedding Benchmark

**Files:**
- Create: `examples/storm_examples/benchmark_embedding_profiles.py`
- Modify: `examples/storm_examples/run_paperstorm_milestone.py`
- Test: `tests/test_embedding_profile_benchmark.py`

- [ ] **Step 1: Write deterministic sampling and manifest tests**

```python
def test_stable_tenth_keeps_full_candidate_corpus():
    selected = stable_tenth([{"case_id": str(i)} for i in range(100)])
    assert 8 <= len(selected) <= 12
    assert selected == stable_tenth(list(reversed([{"case_id": str(i)} for i in range(100)])))

def test_report_records_model_and_resource_metrics():
    assert REQUIRED <= set(build_report(fixture_rows, fixture_manifest)["metrics"])
```

- [ ] **Step 2: Run tests and verify failure**

Run: `D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest tests.test_embedding_profile_benchmark -v`

Expected: benchmark module is missing.

- [ ] **Step 3: Implement stable SHA-256 sampling, checkpointing and metrics**

The CLI accepts `--benchmarks scifact qasper`, `--profiles ...`, `--sample-ratio 0.1`, `--seed`, `--model-cache`, and `--output-dir`. Sampling changes query rows only; candidate corpora remain intact. Save `manifest.json`, `predictions.jsonl`, `metrics.json`, and `comparison.md` per profile.

- [ ] **Step 4: Run smoke fixtures**

Run: `D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest tests.test_embedding_profile_benchmark tests.test_public_benchmarks_v55 -q`

Expected: deterministic IDs and complete resource metrics.

- [ ] **Step 5: Commit**

```text
feat(eval): add embedding profile benchmark
```

## Task 3: Exact and HNSW Dense Backends

**Files:**
- Create: `knowledge_storm/dense_index.py`
- Modify: `knowledge_storm/retrieval.py`
- Modify: `requirements.txt`
- Test: `tests/test_dense_index.py`

- [ ] **Step 1: Write exact parity, ANN recall, persistence and ACL tests**

```python
def test_hnsw_recall_against_exact():
    exact = ExactDenseBackend(vectors, ids)
    hnsw = HnswDenseBackend(vectors, ids, ef_search=100)
    assert recall_at_k(exact.search(query, 10), hnsw.search(query, 10)) >= 0.9

def test_scoped_search_never_uses_global_hnsw():
    result = auto.search(query, 5, allowed_indices=[1, 4])
    assert result.backend == "exact"
    assert result.reason == "acl_exact_fallback"
```

- [ ] **Step 2: Run tests and verify failure**

Run: `D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest tests.test_dense_index -v`

Expected: backend classes are missing.

- [ ] **Step 3: Implement backend contract and optional HNSW adapter**

```python
@dataclass(frozen=True)
class DenseSearchResult:
    indices: tuple
    scores: tuple
    backend: str
    reason: str = ""

class AutoDenseBackend:
    def search(self, query, top_k, allowed_indices=None):
        if allowed_indices is not None:
            return self.exact.search(query, top_k, allowed_indices, "acl_exact_fallback")
        return self.hnsw.search(query, top_k) if self.hnsw else self.exact.search(query, top_k)
```

Pin `usearch>=2.16,<3.0`; it provides a Windows wheel while retaining an HNSW backend. Missing dependency is an explicit error only when `hnsw` is requested. `auto` may use Exact below threshold, but must expose that decision. The original `hnswlib` choice was rejected after its Windows install required a local MSVC build toolchain.

- [ ] **Step 4: Integrate `HybridPaperIndex` and verify parity**

Run: `D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest tests.test_dense_index tests.test_retrieval tests.test_retrieval_governance tests.test_retrieval_pipeline -q`

Expected: Exact ranking matches the previous NumPy implementation; scoped tests leak zero chunks.

- [ ] **Step 5: Commit**

```text
feat(retrieval): add HNSW dense backend
```

## Task 4: Jieba Domain Analyzer

**Files:**
- Create: `knowledge_storm/text_analyzers.py`
- Create: `knowledge_storm/resources/paperstorm_domain_terms.txt`
- Modify: `knowledge_storm/retrieval.py`
- Modify: `requirements.txt`
- Test: `tests/test_text_analyzers.py`

- [ ] **Step 1: Write domain phrase, fallback and revision tests**

```python
def test_domain_analyzer_preserves_passive_intermodulation():
    tokens = JiebaDomainAnalyzer().tokenize("无源互调神经网络抑制")
    assert "无源互调" in tokens
    assert "神经网络抑制" in tokens
    assert "源互" in tokens

def test_analyzer_revision_changes_with_dictionary():
    assert analyzer_revision(["无源互调"]) != analyzer_revision(["无源互调", "数字预失真"])
```

- [ ] **Step 2: Run tests and verify failure**

Run: `D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest tests.test_text_analyzers -v`

Expected: analyzer module is missing.

- [ ] **Step 3: Implement analyzers and versioned dictionary**

`JiebaDomainAnalyzer` uses a private `jieba.Tokenizer`, loads the repository dictionary, preserves Latin technical terms, and appends CJK bigrams as OOV fallback. Add `jieba>=0.42,<1.0` to requirements.

- [ ] **Step 4: Integrate BM25 query/corpus analysis and verify PIM case**

Run: `D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest tests.test_text_analyzers tests.test_retrieval tests.test_paperstorm_domain_retrieval -q`

Expected: index/query analyzer revisions match and PIM terms rank above RAM distractors.

- [ ] **Step 5: Commit**

```text
feat(retrieval): add domain-aware analyzer
```

## Task 5: Fair Parent Context Allocation

**Files:**
- Modify: `knowledge_storm/retrieval.py`
- Modify: `knowledge_storm/retrieval_pipeline.py`
- Modify: `knowledge_storm/document_ingestion.py`
- Test: `tests/test_parent_context_budget.py`

- [ ] **Step 1: Write starvation, local-window and rank-preservation tests**

```python
def test_parent_budget_reaches_multiple_unique_sections():
    expanded = index.expand_parent_context(results, parent_budget_tokens=160)
    assert all(item["parent_allocation"]["allocated_tokens"] > 0 for item in expanded[:2])

def test_parent_expansion_never_changes_child_order():
    assert ids(index.expand_parent_context(results, 160)) == ids(results)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest tests.test_parent_context_budget -v`

Expected: first parent consumes the shared budget or allocation metadata is absent.

- [ ] **Step 3: Implement minimum quota plus relevance-weighted remainder**

Group unique section parents, reserve `min(64, budget // parent_count)` tokens per parent, then distribute remaining tokens by normalized child score. Locate child/raw text in the section and truncate a bidirectional window around the match. Emit `parent_type`, requested, allocated, used and truncated fields.

- [ ] **Step 4: Run ingestion and pipeline regressions**

Run: `D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest tests.test_parent_context_budget tests.test_document_ingestion tests.test_retrieval tests.test_retrieval_pipeline -q`

Expected: tables/formulas stay atomic, section fallback is explicit, child ordering is unchanged.

- [ ] **Step 5: Commit**

```text
fix(retrieval): prevent parent budget starvation
```

## Task 6: Reranker Profiles and Runtime Trace

**Files:**
- Modify: `knowledge_storm/retrieval_profiles.py`
- Modify: `knowledge_storm/retrieval.py`
- Modify: `knowledge_storm/retrieval_pipeline.py`
- Test: `tests/test_reranker_profiles.py`

- [ ] **Step 1: Write profile, device and trace tests**

```python
def test_quality_reranker_requires_explicit_profile():
    assert get_reranker_profile().name == "cpu-balanced"
    assert get_reranker_profile("quality-gpu").model_name == "BAAI/bge-reranker-v2-m3"

def test_trace_exposes_actual_reranker_runtime():
    assert output["models"]["reranker_device"] == "cpu"
    assert output["rerank_decision"]["candidate_count"] <= 20
```

- [ ] **Step 2: Run tests and verify failure**

Run: `D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest tests.test_reranker_profiles -v`

Expected: profile/device fields are absent.

- [ ] **Step 3: Implement explicit profiles, batch size and actual device reporting**

CPU remains the default. `quality-gpu` raises an actionable configuration error when CUDA is unavailable unless the caller explicitly accepts CPU. Preserve selective rerank and the candidate cap.

- [ ] **Step 4: Run reranker and resilience tests**

Run: `D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest tests.test_reranker_profiles tests.test_retrieval_pipeline tests.test_retrieval_resilience -q`

- [ ] **Step 5: Commit**

```text
feat(retrieval): add reranker runtime profiles
```

## Task 7: Scale Benchmark and Real Model Comparison

**Files:**
- Create: `examples/storm_examples/benchmark_dense_scale.py`
- Create: `tests/test_dense_scale_benchmark.py`
- Modify: `docs/RAG_BADCASE_PROGRESSIVE_RESULTS.md`
- Modify: `README.md`

- [ ] **Step 1: Implement deterministic synthetic scale benchmark tests**

```python
def test_memory_estimate_is_dimension_aware():
    assert estimate_flat_bytes(2_000_000, 1024) == 8_192_000_000

def test_scale_report_separates_measured_and_estimated():
    assert report["measured"]["vector_count"] == 100_000
    assert report["estimated"]["vector_count"] == 2_000_000
```

- [ ] **Step 2: Run offline 100k Exact/HNSW comparison**

Run: `D:\SOFTWARE\spyder\envs\storm\python.exe examples\storm_examples\benchmark_dense_scale.py --vectors 100000 --dimension 384 --queries 200 --output-dir C:\Users\yzy\Desktop\codex\paperstorm-benchmarks\retrieval-stack-upgrade\ann-100k`

Expected: measured ANN Recall@10, P50/P95, build time and index bytes; 2M values are labelled estimates.

- [ ] **Step 3: Install/cache the three candidate models and run 10% public comparison**

Run the four profiles through `benchmark_embedding_profiles.py` with the frozen SciFact/QASPER sample manifest. Failed downloads resume from the model cache; completed profiles are not recomputed.

- [ ] **Step 4: Write the Pareto and concrete Bad Case report**

Document model-by-model quality, latency, memory, English/Chinese trade-offs, and at least three queries whose Top K changed. Do not call a metric improvement when protocols differ or confidence intervals overlap.

- [ ] **Step 5: Run full offline validation**

Run:

```powershell
$env:PAPERSTORM_OFFLINE_TESTS="1"
$env:PAPERSTORM_RETRIEVAL_EMBEDDING="hash"
D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest discover -s tests -p "test_*.py" -q
D:\SOFTWARE\spyder\envs\storm\python.exe -m compileall -q knowledge_storm examples tests
git diff --check
```

Expected: zero failures; network/model tests are skipped under explicit offline mode.

- [ ] **Step 6: Commit final report and packaging**

```text
docs(retrieval): publish stack benchmark
```

## Plan Self-Review

- Every design requirement maps to Tasks 1-7.
- Model comparison preserves full candidate corpora and changes only a deterministic 10% query sample.
- HNSW cannot weaken pre-retrieval ACL; unsafe dynamic scopes use explicit Exact fallback.
- Parent expansion cannot change child ranking.
- Real model and scale artifacts stay outside the repository.
- No step claims a 2M measured result without executing it.
