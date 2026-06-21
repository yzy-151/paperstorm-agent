"""Run the PaperStorm v4.0 retrieval and answer evaluation harness."""

import argparse
import json
from pathlib import Path

from knowledge_storm.paperstorm_eval_v4 import (
    build_seed_dataset,
    load_dataset,
    run_evaluation,
    run_seed_baseline,
)
from knowledge_storm.paperstorm_rag import ContextCompressionRetriever, PaperStormRAGIndex


def parse_args():
    parser = argparse.ArgumentParser(description="PaperStorm RAG Evaluation v4.0")
    parser.add_argument("--dataset", help="Optional JSON/JSONL evaluation dataset")
    parser.add_argument("--output-dir", default="./results/paperstorm_eval_v4")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--export-seed-dataset",
        help="Write the built-in 100-case seed dataset to this JSON path and exit",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.export_seed_dataset:
        path = Path(args.export_seed_dataset)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(build_seed_dataset(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("Seed dataset: {0}".format(path.resolve()))
        return

    if not args.dataset:
        report = run_seed_baseline(args.output_dir, top_k=args.top_k)
    else:
        dataset = load_dataset(args.dataset)
        if not dataset.get("corpus"):
            raise ValueError("A runnable dataset must include a corpus array.")
        index = PaperStormRAGIndex.from_documents(
            dataset["corpus"],
            chunk_size=2000,
            chunk_overlap=0,
        )
        retriever = ContextCompressionRetriever(index)

        def case_runner(case):
            candidates = index.search(case["query"], top_k=max(20, args.top_k * 4), rerank=False)
            retrieved = retriever.retrieve(case["query"], top_k=args.top_k)
            selected = retrieved.get("chunks") or []
            if case.get("expected_behavior") == "abstain":
                selected = []
            return {
                "candidates": candidates,
                "selected": selected,
                "prompt_context": retrieved.get("prompt_context") or "",
                "answer": selected[0].get("content", "") if selected else "现有资料不足以可靠回答该问题。",
                "citations": [selected[0].get("chunk_id")] if selected else [],
                "abstained": not selected,
            }

        report = run_evaluation(dataset, case_runner, args.output_dir, top_k=args.top_k)

    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print("Report: {0}".format((Path(args.output_dir) / "rag_eval_v4_report.md").resolve()))


if __name__ == "__main__":
    main()
