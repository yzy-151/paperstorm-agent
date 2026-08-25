"""Run PaperStorm public retrieval benchmarks from official local data."""

import argparse
import json
from pathlib import Path

from knowledge_storm.evaluation.public_benchmarks.base import BenchmarkDataset
from knowledge_storm.evaluation.public_benchmarks.beir_scifact import (
    download_scifact,
    load_scifact,
)
from knowledge_storm.evaluation.public_benchmarks.qasper import (
    load_qasper_huggingface,
    load_qasper_records,
)
from knowledge_storm.evaluation.public_benchmarks.runner import (
    HashEmbeddingProvider,
    run_retrieval_benchmark,
)
from knowledge_storm.retrieval import (
    CrossEncoderReranker,
    SentenceTransformerProvider,
)


def build_parser():
    parser = argparse.ArgumentParser(description="PaperStorm public benchmark")
    parser.add_argument("--benchmark", choices=("scifact", "qasper"), required=True)
    parser.add_argument("--dataset-dir")
    parser.add_argument(
        "--cache-dir", default=str(Path.home() / ".cache" / "paperstorm")
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--modes", nargs="+", default=["bm25", "dense", "hybrid"])
    parser.add_argument("--embedding", choices=("hash", "real"), default="real")
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model used when --embedding real",
    )
    parser.add_argument("--reranker", action="store_true")
    parser.add_argument(
        "--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--smoke-limit", type=int)
    parser.add_argument("--download", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    cache_dir = Path(args.cache_dir)
    dataset = _load_dataset(args, cache_dir)
    dataset = _evaluation_subset(dataset, args.smoke_limit, args.benchmark)
    embedding = (
        HashEmbeddingProvider()
        if args.embedding == "hash"
        else SentenceTransformerProvider(
            model_name=args.model, cache_folder=str(cache_dir / "models")
        )
    )
    reranker = None
    if args.reranker or "hybrid_rerank" in args.modes:
        reranker = CrossEncoderReranker(
            model_name=args.reranker_model, cache_folder=str(cache_dir / "models")
        )
    report = run_retrieval_benchmark(
        dataset,
        embedding_provider=embedding,
        modes=args.modes,
        top_k=args.top_k,
        reranker=reranker,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        output_dir=args.output_dir,
        cache_state="warm_query_after_cold_index",
        scope_field="paper_id" if args.benchmark == "qasper" else None,
    )
    print(json.dumps(report["modes"], ensure_ascii=False, indent=2))
    return report


def _load_dataset(args, cache_dir):
    if args.benchmark == "scifact":
        dataset_dir = (
            Path(args.dataset_dir)
            if args.dataset_dir
            else cache_dir / "datasets" / "scifact"
        )
        if args.download:
            dataset_dir = download_scifact(cache_dir / "datasets")
        return load_scifact(dataset_dir, split=args.split)
    if args.dataset_dir:
        records = json.loads(Path(args.dataset_dir).read_text(encoding="utf-8"))
        return load_qasper_records(
            records, split=args.split, version="local-qasper-json"
        )
    return load_qasper_huggingface(
        split=args.split,
        cache_dir=cache_dir / "datasets",
        smoke_limit=args.smoke_limit,
    )


def _evaluation_subset(dataset, smoke_limit, benchmark):
    cases = tuple(case for case in dataset.cases if case.relevant_document_ids)
    if smoke_limit:
        cases = cases[: max(1, int(smoke_limit))]
    if benchmark == "qasper":
        selected_paper_ids = {
            str(case.metadata.get("paper_id") or "") for case in cases
        }
        documents = tuple(
            document
            for document in dataset.documents
            if str(document.metadata.get("paper_id") or "") in selected_paper_ids
        )
    else:
        documents = dataset.documents
    return BenchmarkDataset(
        dataset.name,
        dataset.version,
        documents,
        cases,
        dict(dataset.metadata, evaluation_scope="evidence_retrieval"),
    )


if __name__ == "__main__":
    main()
