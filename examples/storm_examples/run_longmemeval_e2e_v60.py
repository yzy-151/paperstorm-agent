"""Run full LongMemEval-S reader/judge comparison across three memory modes."""

import argparse
import json
import os
from pathlib import Path

from knowledge_storm.evaluation.public_benchmarks.v60_harness import run_longmemeval_end_to_end
from knowledge_storm.evaluation.public_benchmarks.v60_llm import LongMemEvalJudge, StreamingReader
from knowledge_storm.paperstorm_retrieval_v41 import SentenceTransformerProvider
from knowledge_storm.paperstorm_router_llm import _load_flat_toml_env


def main(argv=None):
    parser = argparse.ArgumentParser(description="PaperStorm LongMemEval-S full end-to-end benchmark")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--reader-model", default="deepseek/deepseek-chat")
    parser.add_argument("--reader-api-base", default=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com"))
    parser.add_argument("--reader-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--judge-model", default="openai/gpt-4o-2024-08-06")
    parser.add_argument("--judge-api-base", default="")
    parser.add_argument("--judge-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--embedding-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--model-cache", default="")
    args = parser.parse_args(argv)
    _load_flat_toml_env()
    reader = StreamingReader(args.reader_model, os.getenv(args.reader_key_env), api_base=args.reader_api_base, max_tokens=256, input_price=0.27, output_price=1.10)
    judge = LongMemEvalJudge(args.judge_model, os.getenv(args.judge_key_env), api_base=args.judge_api_base or None)
    embedding = SentenceTransformerProvider(args.embedding_model, cache_folder=args.model_cache or None)
    report = run_longmemeval_end_to_end(
        args.dataset,
        Path(args.output_dir),
        reader=reader,
        judge=judge,
        embedding_provider=embedding,
        limit=args.limit or None,
        top_k=args.top_k,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main()
