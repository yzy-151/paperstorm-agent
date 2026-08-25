"""Production-oriented v6.0 context and long-memory evaluation harnesses."""

import json
import math
import re
import sqlite3
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .longmemeval import _iter_json_array, _session_text
from ...memory_store import LongTermMemoryService


def run_context_profile_benchmark(dataset, reader, output_dir, profiles=(128_000, 256_000, 512_000)):
    """Run identical LongBench cases at several context budgets.

    ``reader`` receives ``(prompt, profile_tokens)`` and returns text, usage,
    TTFT, latency and cost. Rows are checkpointed after every model call.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "predictions.jsonl"
    existing = _read_checkpoint(checkpoint, keys=("case_id", "profile"))
    documents = dataset.document_map()
    rows = []
    for profile_tokens in tuple(int(value) for value in profiles):
        profile = _profile_name(profile_tokens)
        for case in dataset.cases:
            key = (case.case_id, profile)
            if existing.get(key, {}).get("status") == "succeeded":
                rows.append(existing[key])
                continue
            context = "\n\n".join(documents[item].text for item in case.relevant_document_ids)
            prompt = _longbench_prompt(case, context, profile_tokens)
            started = time.perf_counter()
            try:
                generated = reader(prompt, profile_tokens)
                prediction = str(generated.get("text") or "").strip()
                usage = dict(generated.get("usage") or {})
                row = {
                    "case_id": case.case_id,
                    "profile": profile,
                    "profile_tokens": profile_tokens,
                    "prediction": prediction,
                    "answer": str((case.answers or ("",))[0]),
                    "correct": prediction.upper() == str((case.answers or ("",))[0]).strip().upper(),
                    "input_tokens": int(usage.get("prompt_tokens") or _estimate_tokens(prompt)),
                    "output_tokens": int(usage.get("completion_tokens") or 0),
                    "ttft_ms": float(generated.get("ttft_ms") or 0.0),
                    "latency_ms": float(generated.get("latency_ms") or ((time.perf_counter() - started) * 1000)),
                    "cost_usd": float(generated.get("cost_usd") or 0.0),
                    "status": "succeeded",
                    "error": None,
                }
            except Exception as exc:
                row = {
                    "case_id": case.case_id,
                    "profile": profile,
                    "profile_tokens": profile_tokens,
                    "status": "failed",
                    "error": "{0}: {1}".format(type(exc).__name__, exc),
                }
            _append_jsonl(checkpoint, row)
            rows.append(row)
    by_profile = {}
    for profile in (_profile_name(value) for value in profiles):
        completed = [row for row in rows if row.get("profile") == profile and row.get("status") == "succeeded"]
        by_profile[profile] = _context_metrics(completed)
    report = {
        "benchmark": "context-profile-pareto",
        "dataset": dataset.name,
        "dataset_version": dataset.version,
        "case_count": len(dataset.cases),
        "profiles": by_profile,
        "pareto_frontier": compute_pareto_frontier(by_profile),
        "generated_at": _now(),
        "evidence_tier": "public-dataset-paid-model-run",
        "limitations": [
            "Profiles are upper budgets; actual token use depends on each case and provider limits.",
            "Compare profiles only when dataset, model, prompt and provider region are held constant.",
        ],
    }
    _write_json(output_dir / "metrics.json", report)
    return report


def compute_pareto_frontier(profiles):
    """Return profiles not dominated on quality, input tokens, TTFT and cost."""
    frontier = []
    for name, candidate in profiles.items():
        dominated = False
        for other_name, other in profiles.items():
            if other_name == name:
                continue
            no_worse = (
                float(other.get("accuracy", 0)) >= float(candidate.get("accuracy", 0))
                and float(other.get("mean_input_tokens", math.inf)) <= float(candidate.get("mean_input_tokens", math.inf))
                and float(other.get("ttft_p50_ms", math.inf)) <= float(candidate.get("ttft_p50_ms", math.inf))
                and float(other.get("cost_usd", math.inf)) <= float(candidate.get("cost_usd", math.inf))
            )
            strictly_better = any((
                float(other.get("accuracy", 0)) > float(candidate.get("accuracy", 0)),
                float(other.get("mean_input_tokens", math.inf)) < float(candidate.get("mean_input_tokens", math.inf)),
                float(other.get("ttft_p50_ms", math.inf)) < float(candidate.get("ttft_p50_ms", math.inf)),
                float(other.get("cost_usd", math.inf)) < float(candidate.get("cost_usd", math.inf)),
            ))
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(name)
    return frontier


def run_longmemeval_end_to_end(
    dataset_path,
    output_dir,
    reader,
    judge,
    embedding_provider,
    limit=None,
    top_k=5,
    recent_sessions=5,
):
    """Evaluate retrieval plus answer generation plus an LLM judge on all modes."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "predictions.jsonl"
    existing = _read_checkpoint(checkpoint, keys=("case_id", "mode"))
    rows = []
    case_count = 0
    for raw in _iter_json_array(dataset_path, limit=limit):
        case_count += 1
        case = _longmem_case(raw, case_count - 1)
        evidence_by_mode = {
            "recent": case["sessions"][-max(1, int(recent_sessions)):],
            "fts_session": _fts_retrieve(case["sessions"], case["question"], top_k),
            "paperstorm_memory": _paperstorm_memory_retrieve(case, output_dir / "memory-index", embedding_provider, top_k),
        }
        for mode, evidence_sessions in evidence_by_mode.items():
            key = (case["case_id"], mode)
            if existing.get(key, {}).get("status") == "succeeded":
                rows.append(existing[key])
                continue
            started = time.perf_counter()
            evidence = "\n\n".join("[session {0}]\n{1}".format(item["session_id"], item["text"]) for item in evidence_sessions)
            try:
                generated = reader(case["question"], evidence, mode)
                prediction = str(generated.get("text") or "").strip()
                judge_type = "abstention" if case["unanswerable"] else case["question_type"]
                judged = judge(case["question"], case["answer"], prediction, judge_type)
                retrieved = [item["session_id"] for item in evidence_sessions]
                relevant = set(case["answer_session_ids"])
                row = {
                    "case_id": case["case_id"],
                    "mode": mode,
                    "question_type": case["question_type"],
                    "prediction": prediction,
                    "gold_answer": case["answer"],
                    "retrieved_session_ids": retrieved,
                    "recall_at_{0}".format(top_k): len(set(retrieved[:top_k]) & relevant) / len(relevant) if relevant else 0.0,
                    "judge_correct": bool(judged.get("correct")),
                    "judge_explanation": str(judged.get("explanation") or ""),
                    "usage": dict(generated.get("usage") or {}),
                    "judge_usage": dict(judged.get("usage") or {}),
                    "cost_usd": float(generated.get("cost_usd") or 0) + float(judged.get("cost_usd") or 0),
                    "latency_ms": round((time.perf_counter() - started) * 1000, 4),
                    "status": "succeeded",
                    "error": None,
                }
            except Exception as exc:
                row = {"case_id": case["case_id"], "mode": mode, "status": "failed", "error": "{0}: {1}".format(type(exc).__name__, exc)}
            _append_jsonl(checkpoint, row)
            rows.append(row)
    modes = {}
    for mode in ("recent", "fts_session", "paperstorm_memory"):
        completed = [row for row in rows if row.get("mode") == mode and row.get("status") == "succeeded"]
        modes[mode] = {
            "successful_cases": len(completed),
            "answer_accuracy": _mean([float(row["judge_correct"]) for row in completed]),
            "recall_at_{0}".format(top_k): _mean([float(row.get("recall_at_{0}".format(top_k), 0)) for row in completed]),
            "latency_p50_ms": _percentile([row["latency_ms"] for row in completed], 0.5),
            "total_cost_usd": round(sum(float(row.get("cost_usd") or 0) for row in completed), 6),
        }
    report = {
        "benchmark": "longmemeval-s-end-to-end",
        "case_count": case_count,
        "modes": modes,
        "judge_protocol": "reader_plus_llm_judge",
        "official_compatible": getattr(judge, "protocol", "") == "longmemeval-official-qa-compatible" and str(getattr(judge, "model", "")) in {
            "gpt-4o-2024-08-06",
            "openai/gpt-4o-2024-08-06",
        },
        "generated_at": _now(),
        "limitations": [
            "Official comparability requires the official LongMemEval-S file and pinned official judge model/prompt.",
            "A run using another reader or judge remains an ablation, not an official leaderboard score.",
        ],
    }
    _write_json(output_dir / "metrics.json", report)
    return report


def _longmem_case(raw, index):
    case_id = str(raw.get("question_id") or raw.get("id") or "question-{0}".format(index))
    sessions_raw = raw.get("haystack_sessions") or raw.get("sessions") or raw.get("history") or []
    ids = raw.get("haystack_session_ids") or []
    sessions = []
    for offset, session in enumerate(sessions_raw):
        mapping = session if isinstance(session, dict) else {}
        messages = mapping.get("content") or mapping.get("messages") or mapping.get("turns") or session
        sessions.append({
            "session_id": str(mapping.get("session_id") or mapping.get("id") or (ids[offset] if offset < len(ids) else "session-{0}".format(offset))),
            "text": _session_text(messages),
        })
    answer = raw.get("answer")
    if isinstance(answer, list):
        answer = ", ".join(str(item) for item in answer)
    evidence = raw.get("answer_session_ids") or raw.get("evidence_session_ids") or []
    return {
        "case_id": case_id,
        "question": str(raw.get("question") or raw.get("query") or ""),
        "answer": str(answer or ""),
        "question_type": str(raw.get("question_type") or raw.get("type") or "unknown"),
        "unanswerable": "_abs" in case_id or str(raw.get("question_type") or raw.get("type") or "") in {"abstention", "unanswerable"},
        "answer_session_ids": [str(item.get("session_id") or item.get("id")) if isinstance(item, dict) else str(item) for item in evidence],
        "sessions": sessions,
    }


def _fts_retrieve(sessions, query, top_k):
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("CREATE VIRTUAL TABLE sessions USING fts5(session_id UNINDEXED, content, tokenize='unicode61')")
        connection.executemany("INSERT INTO sessions(session_id, content) VALUES (?, ?)", [(item["session_id"], item["text"]) for item in sessions])
        terms = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}", str(query))
        match = " OR ".join('"{0}"'.format(term.replace('"', '""')) for term in dict.fromkeys(terms))
        if not match:
            return []
        rows = connection.execute("SELECT session_id, content FROM sessions WHERE sessions MATCH ? ORDER BY bm25(sessions) LIMIT ?", (match, max(1, int(top_k)))).fetchall()
        return [{"session_id": row[0], "text": row[1]} for row in rows]
    finally:
        connection.close()


def _paperstorm_memory_retrieve(case, root, embedding_provider, top_k):
    namespace = "longmemeval:{0}".format(case["case_id"])
    service = LongTermMemoryService(
        root,
        embedding_provider=embedding_provider,
        retrieval_mode="semantic" if embedding_provider is not None else "lexical",
    )
    for index, session in enumerate(case["sessions"]):
        service.upsert(
            namespace=namespace,
            memory_type="episode",
            subject="session",
            content=session["text"],
            canonical_key="session:{0}".format(session["session_id"]),
            source_message_ids=[session["session_id"]],
            metadata={"session_id": session["session_id"], "document_id": session["session_id"], "order": index},
        )
    result = service.search(namespace, case["question"], top_k=max(1, int(top_k)))
    output = []
    for item in result.get("results") or []:
        metadata = item.get("metadata") or {}
        output.append({"session_id": str(metadata.get("session_id") or metadata.get("document_id") or item.get("canonical_key") or item.get("id")), "text": str(item.get("content") or "")})
    return output


def _longbench_prompt(case, context, profile_tokens):
    reserve = 2048
    max_chars = max(1, (int(profile_tokens) - reserve) * 4)
    choices = case.metadata.get("choices") or {}
    choice_text = "\n".join("{0}. {1}".format(key, value) for key, value in choices.items() if value)
    return "Context budget: {0} tokens\nContext:\n{1}\n\nQuestion:\n{2}\n{3}\nReturn only A, B, C or D.".format(profile_tokens, context[:max_chars], case.query, choice_text)


def _context_metrics(rows):
    return {
        "successful_cases": len(rows),
        "accuracy": round(_mean([float(row["correct"]) for row in rows]), 6),
        "mean_input_tokens": round(_mean([row["input_tokens"] for row in rows]), 3),
        "ttft_p50_ms": _percentile([row["ttft_ms"] for row in rows], 0.5),
        "ttft_p95_ms": _percentile([row["ttft_ms"] for row in rows], 0.95),
        "latency_p50_ms": _percentile([row["latency_ms"] for row in rows], 0.5),
        "cost_usd": round(sum(row["cost_usd"] for row in rows), 6),
    }


def _profile_name(tokens):
    return "{0}K".format(int(tokens) // 1000)


def _percentile(values, fraction):
    values = sorted(float(value) for value in values)
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, math.ceil(len(values) * fraction) - 1))
    return round(values[index], 4)


def _mean(values):
    values = list(values)
    return round(sum(values) / len(values), 6) if values else 0.0


def _estimate_tokens(text):
    return max(1, len(str(text or "")) // 4)


def _read_checkpoint(path, keys):
    output = {}
    if not Path(path).exists():
        return output
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            output[tuple(str(row.get(key) or "") for key in keys)] = row
    return output


def _append_jsonl(path, row):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _now():
    return datetime.now(timezone.utc).isoformat()
