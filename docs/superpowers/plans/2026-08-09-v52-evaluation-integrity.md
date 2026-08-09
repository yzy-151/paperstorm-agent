# PaperStorm v5.2 Evaluation Integrity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace resume-facing synthetic-only claims with a reproducible, source-grounded real-paper benchmark while restoring a hermetic green test baseline.

**Architecture:** Keep the 100-case synthetic seed as a deterministic smoke suite. Add a document-disjoint Zotero benchmark whose cases retain paper, page, section, evidence, hash, split, and review status; tune retrieval only on dev and report frozen test results separately. Keep contract benchmarks, but expose denominators and limitations instead of bare 100%/0 claims.

**Tech Stack:** Python 3.10+, unittest, sentence-transformers, CrossEncoder, BM25, RRF, Zotero PDFs, GitHub Actions.

---

### Task 1: Restore a hermetic engineering baseline

**Files:**
- Modify: `tests/__init__.py`
- Modify: `tests/test_paperstorm_demo_runbook.py`
- Modify: `tests/test_paperstorm_final_packaging.py`
- Modify: `tests/test_paperstorm_release_docs.py`
- Create: `.github/workflows/test.yml`
- Modify: `requirements.txt`
- Modify: `setup.py`

- [ ] Add tests proving the default suite disables router/chat/judge LLM calls.
- [ ] Replace assertions against ignored private plans with public README/design-source assertions.
- [ ] Run the affected tests and verify the pre-fix failures.
- [ ] Implement the smallest environment and document-test changes.
- [ ] Add a Python 3.10/3.11 CI matrix that runs `unittest discover` without secrets.
- [ ] Add service dependencies and PaperStorm fork metadata.
- [ ] Run the affected tests until green.

### Task 2: Build an auditable real-paper dataset

**Files:**
- Create: `knowledge_storm/paperstorm_real_eval_v52.py`
- Create: `tests/test_paperstorm_real_eval_v52.py`
- Modify: `knowledge_storm/paperstorm_zotero.py`

- [ ] Write tests for evidence provenance, SHA-256, unique IDs, document-disjoint splits, hard negatives, and pending-human-review status.
- [ ] Verify tests fail because the v5.2 builder does not exist.
- [ ] Build cases from real PDF chunks with paper/page/section/evidence metadata.
- [ ] Split by stable document hash into tune/dev and frozen test partitions.
- [ ] Export JSON, JSONL review sheet, manifest, and leakage audit.
- [ ] Verify the dataset tests pass.

### Task 3: Evaluate frozen retrieval configurations

**Files:**
- Modify: `knowledge_storm/paperstorm_real_eval_v52.py`
- Create: `examples/storm_examples/run_paperstorm_real_eval_v52.py`
- Modify: `tests/test_paperstorm_real_eval_v52.py`

- [ ] Write tests that dev selection cannot read test metrics and that reports separate split results.
- [ ] Evaluate legacy, BM25, dense, weighted RRF, and CrossEncoder on dev.
- [ ] Freeze the selected configuration in the report before running test.
- [ ] Report Recall@5, MRR, nDCG@5, citation hit rate, latency, bootstrap confidence intervals, and per-group bad cases.
- [ ] Persist exact model names, corpus/case hashes, runtime environment, and limitations.

### Task 4: Make contract benchmarks interview-safe

**Files:**
- Modify: `knowledge_storm/paperstorm_context_benchmark_v42.py`
- Modify: `knowledge_storm/paperstorm_memory_benchmark_v43.py`
- Modify: `knowledge_storm/paperstorm_langgraph_benchmark_v44.py`
- Modify: `knowledge_storm/paperstorm_production_benchmark_v45.py`
- Modify: corresponding tests under `tests/`

- [ ] Add explicit numerators, denominators, scenario counts, and benchmark type.
- [ ] Separate local SQLite governance latency from end-to-end Agent latency.
- [ ] Preserve zero/one rates for compatibility, but make reports render `passed/total` first.
- [ ] Verify controlled failures and SLO misses remain visible.

### Task 5: Run real data and publish honest claims

**Files:**
- Modify: `README.md`
- Create: `docs/PAPERSTORM_V52_EVALUATION_REPORT.md`
- Modify: resume LaTeX source under `C:/Users/yzy/Desktop/简历类/HITSZ_Resume/`

- [ ] Build the candidate set from `D:/FILEEEEEEEEEEE/zotero`.
- [ ] Run real sentence embeddings and reranker on dev, freeze selection, then run test once.
- [ ] Record positive and negative results without selecting only favorable task groups.
- [ ] Replace the resume metric bullet with sample size, provenance, split, and honest outcome.
- [ ] Build and visually inspect the updated resume PDF.
- [ ] Run all tests with LLM access disabled and confirm no network/model logs appear.

