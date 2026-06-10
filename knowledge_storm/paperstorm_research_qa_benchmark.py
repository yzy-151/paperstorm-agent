import json
import tempfile
from pathlib import Path

from .paperstorm_service import PaperStormTaskService


DEFAULT_CASES = [
    {
        "name": "auto_retrieve_definition",
        "question": "PIM 是什么？",
        "topic": "pim 神经网络抑制",
        "expect_retrieval": True,
        "expect_grounded": True,
    },
    {
        "name": "reuse_existing_kb",
        "question": "这次调研里 PIM 指什么？",
        "topic": "pim 神经网络抑制",
        "reuse_previous": True,
        "expect_retrieval": False,
        "expect_grounded": True,
    },
    {
        "name": "reject_unrelated_question",
        "question": "Transformer 注意力机制和大语言模型训练有什么关系？",
        "topic": "pim 神经网络抑制",
        "reuse_previous": True,
        "expect_retrieval": False,
        "expect_grounded": False,
        "expect_action": "reject_low_confidence",
    },
]


def run_research_qa_benchmark(output_dir, cases=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = cases or DEFAULT_CASES
    with tempfile.TemporaryDirectory() as service_root:
        service = PaperStormTaskService(root_dir=Path(service_root))
        previous_task_id = None
        results = []
        for case in cases:
            task_id = previous_task_id if case.get("reuse_previous") else None
            answer = service.ask_research_agent(
                question=case["question"],
                topic=case.get("topic"),
                task_id=task_id,
                run_mode="fake",
                expected_keywords=["passive intermodulation", "RF"],
                forbidden_keywords=["DRAM", "RAM", "processing-in-memory"],
            )
            previous_task_id = answer.get("used_task_id") or previous_task_id
            results.append(_score_case(case, answer))
    report = {
        "project": "PaperStorm Research QA Benchmark",
        "cases": results,
        "metrics": _aggregate(results),
    }
    (output_dir / "research_qa_benchmark_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "research_qa_benchmark_report.md").write_text(
        _to_markdown(report),
        encoding="utf-8",
    )
    return report


def _score_case(case, answer):
    action = (answer.get("decision") or {}).get("action", "")
    passed = True
    passed = passed and answer.get("retrieval_triggered") == case.get("expect_retrieval")
    passed = passed and answer.get("grounded") == case.get("expect_grounded")
    if case.get("expect_action"):
        passed = passed and action == case["expect_action"]
    return {
        "name": case["name"],
        "passed": bool(passed),
        "question": case["question"],
        "decision_action": action,
        "retrieval_triggered": bool(answer.get("retrieval_triggered")),
        "grounded": bool(answer.get("grounded")),
        "citation_count": len(answer.get("citations") or []),
        "sufficiency_score": (answer.get("evidence_sufficiency") or {}).get("score", 0),
    }


def _aggregate(results):
    total = max(1, len(results))
    grounded = len([item for item in results if item["grounded"]])
    correct_retrieval = len([item for item in results if item["passed"]])
    rejected = len([item for item in results if item["decision_action"] == "reject_low_confidence"])
    return {
        "total_cases": len(results),
        "pass_rate": round(len([item for item in results if item["passed"]]) / total, 4),
        "grounded_rate": round(grounded / total, 4),
        "retrieval_trigger_accuracy": round(correct_retrieval / total, 4),
        "low_confidence_rejection_rate": round(rejected / total, 4),
        "avg_citation_count": round(
            sum(item["citation_count"] for item in results) / total,
            4,
        ),
    }


def _to_markdown(report):
    lines = ["# PaperStorm Research QA Benchmark", ""]
    metrics = report["metrics"]
    for key, value in metrics.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "| case | passed | action | grounded | citations |", "|---|---:|---|---:|---:|"])
    for item in report["cases"]:
        lines.append(
            "| {name} | {passed} | {decision_action} | {grounded} | {citation_count} |".format(
                **item
            )
        )
    lines.append("")
    return "\n".join(lines)
