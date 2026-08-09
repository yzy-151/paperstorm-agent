# PaperStorm v5.4 Real RAG and Context Evaluation Design

## 1. Goal

Build an interview-defensible evaluation system over real Zotero papers. The release must distinguish automatic annotation candidates from human-reviewed labels, evaluate retrieval and reranking without test leakage, measure context compression against realistic research conversations, and explain every metric in the developer dashboard.

The release version is `5.4.0`. GitHub publishing is outside this task.

## 2. Current Baseline and Problems

PaperStorm v5.2 introduced document-disjoint dev/test splitting, corpus and dataset hashes, bootstrap confidence intervals, and a real cross-lingual Zotero pilot. Its frozen test has 12 automatic candidate queries. Dense retrieval reaches Recall@5 `0.4167`, MRR `0.2986`, and nDCG@5 `0.2463`, with Recall@5 95% CI `[0.1667, 0.6667]`.

These results demonstrate evaluation discipline but do not establish mature RAG quality:

- all 46 queries still require human review;
- the frozen split is too small for a narrow confidence interval;
- the release result has no active reranker;
- the earlier v4.1 weak-label Cross-Encoder experiment degraded nDCG and added seconds of latency;
- the v4.2 context benchmark is one deterministic constructed conversation, not a real-paper answer-retention study;
- the dashboard exposes several raw reports without a unified trust status or plain-language metric definitions.

## 3. Scope

### 3.1 Included

- A versioned annotation schema and browser annotation workbench.
- Import, incremental save, validation, and export for human review records.
- Dev-only retrieval configuration selection and a frozen-test evaluation gate.
- BM25, multilingual dense retrieval, RRF hybrid retrieval, and optional multilingual Cross-Encoder reranking.
- Retrieval metrics with denominators, confidence intervals, latency, deltas, and failure taxonomy.
- A real-paper context-compression benchmark with full-history and fixed-window baselines.
- A benchmark API and an understandable developer console.
- Documentation that separates measured facts, pilot results, contract tests, and future targets.

### 3.2 Excluded

- Claiming expert annotation before the user completes review.
- Publishing private PDF text, Zotero paths, or full review artifacts to Git.
- Using the frozen test to tune models, thresholds, fusion weights, or candidate depth.
- Treating LLM-as-judge scores as ground truth.
- Replacing the evaluation core with Ragas, DeepEval, or a hosted platform in v5.4.
- Pushing to GitHub in this task.

## 4. Data and Annotation Contract

Each case has a stable `case_id`, split, query, source document metadata, page and evidence excerpt, candidate relevant document IDs, hard negatives, hashes, and review fields.

Human review records contain:

- `query_validity`: `valid`, `invalid`, or `needs_edit`;
- `edited_query`: optional natural user-style replacement;
- `relevant_document_ids`: one or more relevant documents;
- `evidence_sufficiency`: `sufficient`, `partial`, or `insufficient`;
- `reviewer_notes` and `reviewed_at`;
- `review_status`: `reviewed` only when required fields pass validation.

Automatic candidates remain usable for development diagnostics but not for externally stated frozen-test quality. The dashboard labels a dataset as:

- `candidate`: no human-reviewed cases;
- `pilot`: at least one reviewed case but fewer than 50 reviewed frozen-test queries;
- `release_ready`: at least 50 valid reviewed frozen-test queries and at least 10 queries in every reported domain.

Review progress is stored under `results/`, which remains gitignored. A sanitized aggregate report may be committed, but it must contain no private paths, excerpts, or query-level private data.

## 5. Retrieval and Reranking Evaluation

### 5.1 Protocol

Documents, not chunks, are the split unit. All chunks from a paper belong to one split. Configuration search runs on dev only. The frozen test is evaluated only when the annotation gate permits it, and every frozen run records the dataset hash, corpus hash, code commit, model identifiers, parameters, and timestamp.

Candidate configurations are:

- BM25;
- multilingual dense retrieval;
- BM25 plus dense retrieval fused with weighted RRF;
- hybrid candidates reranked by a multilingual Cross-Encoder.

Dev search may compare `candidate_k`, RRF weights, rank constant, and reranker model. Selection order is nDCG@5, MRR, Recall@5, then P95 latency. Test metrics never participate in selection.

### 5.2 Metrics

The report includes:

- Recall@5 and Recall@10;
- Precision@5;
- MRR;
- nDCG@5;
- P50 and P95 latency;
- 95% bootstrap confidence intervals;
- paired per-query deltas against BM25 and the selected non-reranked configuration;
- win, tie, and loss counts for reranking;
- failures classified as lexical mismatch, cross-language mismatch, chunking miss, candidate-generation miss, rerank demotion, ambiguous label, or annotation defect.

A reranker is enabled by default only if it improves dev nDCG@5, does not reduce dev Recall@5 beyond a declared tolerance, and stays inside the configured latency budget. A negative result is reported as a valid engineering finding.

## 6. Real-Paper Context Evaluation

### 6.1 Scenarios

Reviewed Zotero cases are converted into multi-turn research conversations containing user goals, constraints, retrieval calls, evidence artifacts, interim decisions, corrections, and follow-up questions. No source PDF content is committed.

Three strategies are compared on the same scenario:

- `full_history`: reference condition, bounded only by the benchmark model limit;
- `fixed_window`: recent-message truncation baseline;
- `structured_compaction`: PaperStorm's structured state summary, recent-turn preservation, artifact references, and reversible event store inspired by Claude Code and Hermes-style context management.

### 6.2 Metrics

Deterministic probes measure token reduction, constraint retention, entity retention, source-reference retention, tool-call pairing, restore exactness, and repeated-compaction drift. Reviewed retrieval probes measure whether the same relevant documents remain reachable after compaction. Answer probes measure required-fact recall and citation support against reviewed evidence. Optional LLM judge output is secondary and visibly identified as model-judged.

The report must not claim that compression improves answer quality unless the reviewed answer probes support it. The main advantage may instead be lower token use at equivalent retention.

## 7. Backend Boundaries

The v5.4 evaluation package has four focused responsibilities:

- annotation store and validation;
- retrieval experiment orchestration and statistics;
- context scenario generation and evaluation;
- dashboard-safe report projection.

The service exposes endpoints to load candidate cases, save a review, report progress, run a dev experiment, run an eligible frozen evaluation, run context evaluation, and load the latest sanitized benchmark summary. Long-running actions return explicit status and error information instead of blocking the UI without feedback.

## 8. Dashboard Design

The developer console contains four sections:

1. **Trust status**: dataset state, reviewed count, frozen-test eligibility, hashes, sample sizes, and limitations.
2. **Retrieval comparison**: compact method table, selected configuration, confidence interval, latency, relative delta, and rerank win/tie/loss.
3. **Context engineering**: strategy comparison for token use, retention, answer support, and restore behavior.
4. **Annotation workbench**: one case at a time with query, paper, page, excerpt, hard negatives, review controls, save state, progress, and export.

Every metric shows a Chinese name, definition, direction, denominator, and evidence type. Colors express state rather than decoration: green for a passed gate, amber for pilot or uncertainty, red for invalid or failed, and neutral for unavailable. Raw JSON remains available in a collapsible diagnostics area.

## 9. Error Handling and Privacy

- Missing Zotero roots, PDFs, model dependencies, or model caches produce actionable errors.
- Reranker initialization failure skips only that configuration and records the reason.
- Invalid or incomplete reviews cannot enter the frozen release set.
- Dataset hash changes invalidate prior frozen results.
- Private paths and excerpts are removed by the dashboard-safe projection.
- Paid or remote LLM evaluation remains opt-in.

## 10. Testing and Acceptance

Unit tests cover annotation validation, progress states, document-disjoint splits, dev-only selection, frozen gating, metrics, paired deltas, context strategy comparison, sanitization, service endpoints, and dashboard rendering contracts.

Acceptance requires:

- all existing offline tests plus new v5.4 tests pass;
- the real Zotero pipeline completes locally without modifying Zotero;
- an unreviewed dataset is visibly blocked from release claims;
- a small reviewed pilot can run end to end;
- the dashboard clearly distinguishes automatic, human-reviewed, deterministic, and model-judged evidence;
- package and visible dashboard versions read `5.4.0` / `v5.4`;
- no GitHub push occurs.

## 11. Interpretation for Interviews

The defensible contribution is the evaluation architecture and the measured trade-off, not a guaranteed high score. The project demonstrates document-level split discipline, frozen-test governance, human review workflow, hybrid retrieval, conditional reranking, confidence intervals, failure analysis, token-budgeted context assembly, structured compaction, artifactization, restoration, and observability.

Until human review reaches the release gate, all real-paper numbers must be introduced as a pilot. After review, only the sanitized frozen report may be used in the resume.
