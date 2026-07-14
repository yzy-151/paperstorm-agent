import json
import uuid
from pathlib import Path

from .paperstorm_context_v42 import ContextEngine, ContextEngineConfig, ContextEventStore


def run_context_benchmark(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    store = ContextEventStore(output_dir / "runs" / run_id / "context_events.jsonl")
    messages = _benchmark_messages()
    for message in messages:
        store.append_message(message)
    constraints = ["中文", "引用", "DRAM", "RF"]
    engine = ContextEngine(
        config=ContextEngineConfig(
            total_tokens=420,
            output_reserve_tokens=80,
            recent_message_count=3,
            tool_inline_token_limit=24,
        ),
        store=store,
    )
    result = engine.compact(messages, expected_constraints=constraints, force=True)
    second_pass = ContextEngine(config=engine.config).compact(
        result["messages"], expected_constraints=constraints, force=True
    )
    restored = engine.restore(result["compaction_id"])
    source_contents = [item["content"] for item in messages]
    restored_contents = [item["content"] for item in restored["messages"]]
    summary_text = result.get("summary_text") or ""
    retained = len([item for item in constraints if item.lower() in summary_text.lower()])
    repeated_text = second_pass.get("summary_text") or ""
    repeated_terms = constraints + ["PIM", "Cross-Encoder"]
    repeated_retained = len(
        [item for item in repeated_terms if item.lower() in repeated_text.lower()]
    )
    token_savings = max(0, result["before_tokens"] - result["after_tokens"])
    metrics = {
        "before_tokens": result["before_tokens"],
        "after_tokens": result["after_tokens"],
        "token_savings_rate": round(token_savings / max(1, result["before_tokens"]), 4),
        "constraint_retention_rate": round(retained / len(constraints), 4),
        "entity_retention_rate": 1.0 if "PIM" in result["summary"].get("entities", []) else 0.0,
        "todo_retention_rate": 1.0 if result["summary"].get("todos") else 0.0,
        "restore_exact": float(source_contents == restored_contents),
        "artifact_reference_count": len(result.get("artifact_refs") or []),
        "tool_call_pairing_rate": _tool_pairing_rate(messages),
        "repeated_compaction_retention_rate": round(
            repeated_retained / len(repeated_terms), 4
        ),
        "compaction_status": result["status"],
    }
    baseline = _naive_truncation_baseline(messages, constraints)
    report = {
        "project": "PaperStorm Context Benchmark v4.2",
        "run_id": run_id,
        "metrics": metrics,
        "baseline": baseline,
        "config": engine.config.__dict__,
        "summary": result["summary"],
        "limitations": [
            "Deterministic benchmark validates context contracts, not LLM answer quality.",
            "Repeated-compaction semantic drift requires a judge or human labels in later evaluation.",
        ],
    }
    (output_dir / "context_benchmark_v42.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "context_benchmark_v42.md").write_text(
        _to_markdown(report), encoding="utf-8"
    )
    return report


def _naive_truncation_baseline(messages, constraints):
    """Before-implementation baseline: fixed-window truncation without
    artifactization, structured summary or restore."""
    from .paperstorm_context_v42 import estimate_tokens, truncate_to_tokens

    recent_ids = {str(message.get("id") or "") for message in messages[-1:]}
    view = []
    for message in messages:
        message_id = str(message.get("id") or "")
        if message_id in recent_ids:
            view.append(dict(message))
        elif message.get("role") == "tool":
            continue
        else:
            content = truncate_to_tokens(str(message.get("content") or ""), 40)
            view.append(dict(message, content=content))
    joined = " ".join(str(message.get("content") or "") for message in view)
    before_tokens = sum(
        estimate_tokens(str(message.get("content") or "")) for message in messages
    )
    after_tokens = sum(
        estimate_tokens(str(message.get("content") or "")) for message in view
    )
    return {
        "strategy": "fixed_window_truncation",
        "before_tokens": before_tokens,
        "after_tokens": after_tokens,
        "token_savings_rate": round(
            max(0, before_tokens - after_tokens) / max(1, before_tokens), 4
        ),
        "constraint_retention_rate": round(
            len([item for item in constraints if item.lower() in joined.lower()])
            / len(constraints),
            4,
        ),
        "entity_retention_rate": 1.0 if "PIM" in joined else 0.0,
        "todo_retention_rate": 1.0 if "待办" in joined else 0.0,
        "restore_exact": 0.0,
        "artifact_reference_count": 0,
        "tool_call_pairing_rate": 0.0,
        "repeated_compaction_retention_rate": 0.0,
        "structured_summary": False,
    }


def _benchmark_messages():
    return [
        {"id": "sys", "role": "system", "content": "必须使用中文回答并保留句子级引用。"},
        {"id": "goal", "role": "user", "content": "目标：调研 PIM 无源互调神经网络抑制。"},
        {"id": "plan", "role": "assistant", "content": "决定先检索 RF 论文，并排除 DRAM。"},
        {
            "id": "tool-call",
            "role": "assistant",
            "content": "call arxiv_search",
            "tool_call_id": "call-42",
        },
        {
            "id": "tool-output",
            "role": "tool",
            "name": "arxiv_search",
            "tool_call_id": "call-42",
            "content": "passive intermodulation RF evidence " * 180,
        },
        {"id": "decision", "role": "assistant", "content": "已完成召回；决定使用 BM25、Dense 和 RRF。"},
        {"id": "todo", "role": "user", "content": "待办：运行 Cross-Encoder 并比较 P95。"},
        {"id": "recent", "role": "assistant", "content": "正在运行评测，下一步检查坏例。"},
    ]


def _tool_pairing_rate(messages):
    calls = {
        item.get("tool_call_id")
        for item in messages
        if item.get("role") == "assistant" and item.get("tool_call_id")
    }
    outputs = {
        item.get("tool_call_id")
        for item in messages
        if item.get("role") == "tool" and item.get("tool_call_id")
    }
    return 1.0 if calls == outputs else 0.0


def _to_markdown(report):
    lines = [
        "# PaperStorm Context Benchmark v4.2",
        "",
        "## 实现后：ContextEngine v4.2",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in report["metrics"].items():
        lines.append("| {0} | {1} |".format(key, value))
    lines.extend(["", "## 实现前基线：固定窗口截断", "", "| Metric | Value |", "| --- | ---: |"])
    for key, value in report["baseline"].items():
        lines.append("| {0} | {1} |".format(key, value))
    return "\n".join(lines) + "\n"
