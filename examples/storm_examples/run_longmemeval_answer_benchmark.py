"""Run PaperStorm LongMemEval answer generation on a subset of questions.

1/4 protocol run: uses the persisted v5.6 memory index for evidence retrieval and
a DeepSeek reader for answers. Checkpoint/resume per question; costs real API
tokens, so the default limit is 125 (1/4 of the official 500).
"""

import argparse
import json
import logging
import os
import warnings
from pathlib import Path

from knowledge_storm.evaluation.public_benchmarks.longmemeval import (
    load_longmemeval,
)
from knowledge_storm.evaluation.public_benchmarks.longmemeval_answer import (
    LiteLLMPlainGenerator,
    run_longmemeval_answers,
)
from knowledge_storm.paperstorm_memory_v56 import LongTermMemoryServiceV56
from knowledge_storm.paperstorm_retrieval_v41 import SentenceTransformerProvider
from knowledge_storm.paperstorm_router_llm import _load_flat_toml_env


def build_parser():
    parser = argparse.ArgumentParser(
        description="PaperStorm LongMemEval end-to-end answer benchmark (subset)"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--memory-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=125)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--evidence-chars", type=int, default=5000)
    parser.add_argument("--model", default="deepseek/deepseek-chat")
    parser.add_argument(
        "--api-base",
        default=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    parser.add_argument("--model-cache", default="")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    for name in ("LiteLLM", "litellm", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
    warnings.filterwarnings(
        "ignore",
        message=r".*Pydantic serializer warnings.*",
        category=UserWarning,
    )
    dataset = load_longmemeval(args.dataset, limit=args.limit or None)
    provider = SentenceTransformerProvider(
        model_name=args.embedding_model,
        cache_folder=args.model_cache or None,
    )
    service = LongTermMemoryServiceV56(
        Path(args.memory_root), embedding_provider=provider
    )
    _load_flat_toml_env()
    generator = LiteLLMPlainGenerator(
        model=args.model,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        api_base=args.api_base,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        max_attempts=args.max_attempts,
    )
    completed = {"count": 0}

    def progress(row):
        completed["count"] += 1
        if completed["count"] == 1 or completed["count"] % 10 == 0:
            print(
                "Answered {0}/{1}: {2} ({3})".format(
                    completed["count"],
                    len(dataset.cases),
                    row["case_id"],
                    row["status"],
                ),
                flush=True,
            )

    report = run_longmemeval_answers(
        dataset,
        service=service,
        generate=generator,
        output_dir=Path(args.output_dir),
        model_name=args.model,
        top_k=args.top_k,
        evidence_chars=args.evidence_chars,
        on_progress=progress,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main()
