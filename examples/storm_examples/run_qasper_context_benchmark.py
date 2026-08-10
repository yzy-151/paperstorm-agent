"""Evaluate PaperStorm v5.6 context budgets on official QASPER rankings."""

import argparse
import json
from pathlib import Path

from knowledge_storm.evaluation.public_benchmarks.qasper import (
    evaluate_qasper_context_budget,
    load_qasper_official_json,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run the QASPER context-budget diagnostic without an LLM."
    )
    parser.add_argument("--dataset", required=True, help="Official qasper-test-v0.3.json")
    parser.add_argument("--rankings", required=True, help="PaperStorm retrieval predictions.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", default="hybrid_rerank")
    parser.add_argument("--model-context-tokens", type=int, default=8192)
    parser.add_argument("--output-reserve-tokens", type=int, default=1536)
    parser.add_argument("--evidence-budget-ratio", type=float, default=0.7)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    dataset = load_qasper_official_json(args.dataset, split="test")
    ranking_rows = [
        json.loads(line)
        for line in Path(args.rankings).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report, rows = evaluate_qasper_context_budget(
        dataset,
        ranking_rows,
        mode=args.mode,
        model_context_tokens=args.model_context_tokens,
        output_reserve_tokens=args.output_reserve_tokens,
        evidence_budget_ratio=args.evidence_budget_ratio,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
