"""Run PaperStorm v5.6 LongMemEval retrieval diagnostics.

This command deliberately labels the result retrieval-only.  End-to-end answer
accuracy requires a separately frozen reader LLM and the official evaluator.
"""

import argparse
import json
from pathlib import Path

from knowledge_storm.evaluation.public_benchmarks.longmemeval import load_longmemeval
from knowledge_storm.evaluation.public_benchmarks.longmemeval_runner import run_memory_retrieval
from knowledge_storm.paperstorm_rag import HashEmbeddingProvider
from knowledge_storm.paperstorm_retrieval_v41 import SentenceTransformerProvider


def build_parser():
    parser = argparse.ArgumentParser(description="PaperStorm LongMemEval retrieval benchmark")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--embedding", choices=["hash", "sentence-transformer"], default="sentence-transformer")
    parser.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--model-cache", default="")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provider = HashEmbeddingProvider(128) if args.embedding == "hash" else SentenceTransformerProvider(model_name=args.model, cache_folder=args.model_cache or None)
    dataset = load_longmemeval(args.dataset, limit=args.limit or None)
    report = run_memory_retrieval(dataset, output_dir, top_k=args.top_k, embedding_provider=provider, limit=args.limit or None)
    predictions = report.pop("predictions")
    (output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main()
