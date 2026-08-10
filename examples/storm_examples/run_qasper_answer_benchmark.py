"""Run resumable QASPER answer generation over frozen retrieval rankings."""

import argparse
import json
import logging
import os
import warnings
from pathlib import Path

from knowledge_storm.evaluation.public_benchmarks.base import BenchmarkDataset
from knowledge_storm.evaluation.public_benchmarks.qasper import (
    load_qasper_huggingface,
)
from knowledge_storm.evaluation.public_benchmarks.qasper_generation import (
    LiteLLMJsonGenerator,
    complete_qasper_rankings,
    load_rankings,
    run_qasper_generation,
)
from knowledge_storm.paperstorm_retrieval_v41 import (
    CrossEncoderReranker,
    SentenceTransformerProvider,
)
from knowledge_storm.paperstorm_router_llm import _load_flat_toml_env


def build_parser():
    parser = argparse.ArgumentParser(
        description="PaperStorm QASPER end-to-end Answer F1 benchmark"
    )
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--retrieval-predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--cache-dir",
        default=str(Path.home() / ".cache" / "paperstorm"),
    )
    parser.add_argument("--retrieval-mode", default="hybrid_rerank")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--smoke-limit", type=int)
    parser.add_argument("--model", default="deepseek/deepseek-chat")
    parser.add_argument(
        "--api-base",
        default=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--parse-attempts", type=int, default=2)
    parser.add_argument(
        "--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument(
        "--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    _configure_logging()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    dataset = load_qasper_huggingface(
        split=args.split,
        cache_dir=cache_dir / "datasets",
    )
    dataset = _subset(dataset, args.smoke_limit)
    rankings = load_rankings(args.retrieval_predictions, mode=args.retrieval_mode)
    rankings.update(_read_ranking_checkpoint(output_dir / "rankings.jsonl"))
    missing = [case for case in dataset.cases if case.case_id not in rankings]
    if missing:
        print("Completing {0} missing paper-scoped rankings...".format(len(missing)))
        embedding = SentenceTransformerProvider(
            model_name=args.embedding_model,
            cache_folder=str(cache_dir / "models"),
        )
        reranker = CrossEncoderReranker(
            model_name=args.reranker_model,
            cache_folder=str(cache_dir / "models"),
        )
        rankings = complete_qasper_rankings(
            dataset,
            initial_rankings=rankings,
            embedding_provider=embedding,
            mode=args.retrieval_mode,
            top_k=args.top_k,
            reranker=reranker,
            on_ranking=lambda case_id, values: _append_ranking(
                output_dir / "rankings.jsonl", case_id, values
            ),
        )
    _load_flat_toml_env()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    generator = LiteLLMJsonGenerator(
        model=args.model,
        api_key=api_key,
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
                "Generated {0}/{1}: {2} ({3})".format(
                    completed["count"],
                    len(dataset.cases),
                    row["case_id"],
                    row["status"],
                ),
                flush=True,
            )

    report = run_qasper_generation(
        dataset,
        rankings=rankings,
        generate=generator,
        output_dir=output_dir,
        model_name=args.model,
        top_k=args.top_k,
        on_prediction=progress,
        parse_attempts=args.parse_attempts,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _subset(dataset, limit):
    if not limit:
        return dataset
    cases = tuple(dataset.cases[: max(1, int(limit))])
    paper_ids = {str(case.metadata.get("paper_id") or "") for case in cases}
    documents = tuple(
        document
        for document in dataset.documents
        if str(document.metadata.get("paper_id") or "") in paper_ids
    )
    return BenchmarkDataset(
        dataset.name,
        dataset.version,
        documents,
        cases,
        dict(dataset.metadata, evaluation_scope="answer_generation_smoke"),
    )


def _read_ranking_checkpoint(path):
    if not path.exists():
        return {}
    output = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            output[str(row["case_id"])] = [
                str(value) for value in row.get("ranked_document_ids") or []
            ]
    return output


def _append_ranking(path, case_id, values):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {"case_id": case_id, "ranked_document_ids": values},
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
        stream.flush()


def _configure_logging():
    for name in (
        "LiteLLM",
        "litellm",
        "httpx",
        "httpcore",
        "sentence_transformers",
        "sentence_transformers.base.model",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)
    warnings.filterwarnings(
        "ignore",
        message=r".*Pydantic serializer warnings.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Transformer `cache_dir` argument is deprecated.*",
        category=FutureWarning,
    )


if __name__ == "__main__":
    main()
