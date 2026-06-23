"""Run PaperStorm v4.1 hybrid-retrieval ablations on seed or local Zotero papers."""

import argparse
import logging
import os
from pathlib import Path

from knowledge_storm.paperstorm_ablation_v41 import run_ablation
from knowledge_storm.paperstorm_eval_v4 import build_seed_dataset
from knowledge_storm.paperstorm_retrieval_v41 import SentenceTransformerProvider
from knowledge_storm.paperstorm_zotero import build_weak_paper_dataset, load_zotero_chunks


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["seed", "zotero"], default="seed")
    parser.add_argument("--zotero-root", default=os.getenv("PAPERSTORM_ZOTERO_ROOT", ""))
    parser.add_argument(
        "--terms",
        nargs="*",
        default=["无源互调", "passive intermodulation", "PIM"],
    )
    parser.add_argument("--max-papers", type=int, default=8)
    parser.add_argument("--max-pages", type=int, default=15)
    parser.add_argument("--max-cases", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    parser.add_argument(
        "--reranker-model",
        default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    )
    parser.add_argument("--model-cache", default=os.getenv("PAPERSTORM_MODEL_CACHE", ""))
    parser.add_argument("--output-dir", default="./results/paperstorm_v41_eval")
    return parser


def main():
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.dataset == "zotero":
        if not args.zotero_root:
            raise ValueError("--zotero-root is required for the Zotero dataset")
        chunks = load_zotero_chunks(
            args.zotero_root,
            query_terms=args.terms,
            max_papers=args.max_papers,
            max_pages=args.max_pages,
            strategy="contextual",
        )
        dataset = build_weak_paper_dataset(chunks, max_cases=args.max_cases)
    else:
        dataset = build_seed_dataset()
    provider = SentenceTransformerProvider(
        model_name=args.embedding_model,
        cache_folder=args.model_cache or None,
    )
    report = run_ablation(
        dataset,
        output_dir=Path(args.output_dir),
        embedding_provider=provider,
        reranker_model=args.reranker_model,
        top_k=args.top_k,
    )
    print("PaperStorm v4.1 ablation complete")
    print("dataset:", report["dataset_version"])
    print("experiments:", len(report["experiments"]))
    print("best recall:", report["best_by_recall"])
    print("best nDCG:", report["best_by_ndcg"])
    print("report:", Path(args.output_dir) / "rag_eval_v41_ablation.md")


if __name__ == "__main__":
    main()
