# Langfuse RAG Badcase Demo

`run_langfuse_badcase_demo.py` records a deterministic composite RAG failure case.
It always writes a local event mirror and a compact JSON report.

## Environment

Local-only is the default: leave the variables below unset. The report will show
`observability.status: local-only`, and events are at `<output-dir>/observability/events.jsonl`.

To enable Langfuse export, install the observability extra and configure:

```powershell
pip install -e ".[observability]"
$env:PAPERSTORM_OBSERVABILITY = "langfuse"
$env:LANGFUSE_PUBLIC_KEY = "pk-..."
$env:LANGFUSE_SECRET_KEY = "sk-..."
```

Set `LANGFUSE_HOST` for self-hosted Langfuse and `LANGFUSE_TRACING_ENVIRONMENT` to label
an environment. Credentials are redacted from telemetry. Exporter failures are fail-open:
the demo returns normally, keeps the local mirror, and reports `degraded` status.

## Run

Run the fixed composite badcase:

```powershell
python examples/storm_examples/run_langfuse_badcase_demo.py --output-dir results/langfuse_badcase_demo
```

Select a named case from a JSON file:

```powershell
python examples/storm_examples/run_langfuse_badcase_demo.py `
  --output-dir results/badcase_custom `
  --case-file examples/storm_examples/badcases.json `
  --scenario citation_failure
```

The input may be one case object or `{ "scenarios": { "name": { ...case... } } }`.
Each case must include non-empty `case_id`, `question`, `expected_document_ids`, `citations`,
and `answer`, plus list-valued `retrieved_documents`, `reranked_documents`, and
`context_documents`, boolean `answerable`, and numeric `latency_ms`. Invalid or missing values
are rejected before a trace is created.

`answer_groundedness` is a deterministic lexical-overlap heuristic: it compares CJK characters
and English/digit terms in the answer with final context text. It is not an LLM judge and should
not be interpreted as semantic entailment.

The report is `<output-dir>/langfuse_badcase_report.json` and includes the PaperStorm-local
`paperstorm_trace_id`, optional SDK-provided `remote_trace_id`, scores, badcase types,
observability status, and local event path. These IDs are not assumed to be equivalent.

## Langfuse Investigation

1. Filter traces by tag `badcase`, then by `retrieval_miss`, `invalid_citation`,
   `evidence_conflict`, or `wrong_abstention`.
2. Add score filters such as `retrieval_recall_at_5 < 1`, `citation_validity < 1`,
   or low `answer_groundedness`; compare `latency_ms` by environment and release.
3. When `remote_trace_id` is present, open that Langfuse trace. Otherwise search Langfuse by
   `case_id` together with the `badcase` and classification tags. Open the root trace
   `paperstorm.rag.badcase` and inspect, in order: `route`, `retrieve`,
   `rerank`, `context`, `reader`, and `citation_validate`.
4. The first divergent span identifies the root cause: retrieval loss, evidence selection loss,
   reader/evidence conflict or abstention, or unsupported citations.
5. When export is unavailable or degraded, cross-check the same trace in local `events.jsonl`.
