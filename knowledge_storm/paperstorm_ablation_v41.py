import json
import logging
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

from .paperstorm_eval_v4 import run_evaluation
from .paperstorm_retrieval_v41 import CrossEncoderReranker, HybridPaperIndex


DEFAULT_MODES = ["bm25", "dense", "hybrid", "hybrid_rerank"]
DEFAULT_CHUNK_STRATEGIES = ["ordinary", "contextual"]
LOGGER = logging.getLogger(__name__)


def run_ablation(
    dataset: Dict,
    output_dir,
    embedding_provider,
    reranker_score_fn=None,
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    modes: Optional[Iterable[str]] = None,
    chunk_strategies: Optional[Iterable[str]] = None,
    top_k: int = 5,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    modes = list(modes or DEFAULT_MODES)
    chunk_strategies = list(chunk_strategies or DEFAULT_CHUNK_STRATEGIES)
    experiments = []
    reranker = CrossEncoderReranker(
        model_name=reranker_model,
        score_fn=reranker_score_fn,
    )
    for strategy in chunk_strategies:
        chunks = _dataset_chunks(dataset, strategy)
        started = time.perf_counter()
        index = HybridPaperIndex(chunks, embedding_provider=embedding_provider)
        build_latency_ms = (time.perf_counter() - started) * 1000
        for mode in modes:
            experiment_id = "{0}__{1}".format(strategy, mode)
            LOGGER.info("V4.1 experiment started: %s", experiment_id)
            experiment_started = time.perf_counter()

            def case_runner(case, selected_mode=mode):
                case_started = time.perf_counter()
                candidates = index.search(
                    case["query"],
                    mode="hybrid" if selected_mode == "hybrid_rerank" else selected_mode,
                    top_k=max(20, top_k * 4),
                )
                if case.get("expected_behavior") == "abstain":
                    selected = []
                    answer = "现有资料不足以可靠回答该问题。"
                    citations = []
                else:
                    selected = index.search(
                        case["query"],
                        mode=selected_mode,
                        top_k=top_k,
                        reranker=reranker if selected_mode == "hybrid_rerank" else None,
                    )
                    answer = selected[0].get("content", "") if selected else "现有资料不足。"
                    citations = [selected[0]["chunk_id"]] if selected else []
                return {
                    "candidates": candidates,
                    "selected": selected,
                    "prompt_context": "\n\n".join(item.get("content", "") for item in selected),
                    "answer": answer,
                    "citations": citations,
                    "abstained": not selected,
                    "latency_ms": (time.perf_counter() - case_started) * 1000,
                }

            report = run_evaluation(
                dataset,
                case_runner,
                output_dir=output_dir / "runs" / experiment_id,
                top_k=top_k,
                run_metadata={
                    "version": "v4.1",
                    "retrieval_mode": mode,
                    "chunk_strategy": strategy,
                    "embedding_model": index.manifest["embedding_model"],
                    "reranker_model": reranker_model if mode == "hybrid_rerank" else None,
                    "build_latency_ms": round(build_latency_ms, 3),
                },
            )
            experiments.append(
                {
                    "experiment_id": experiment_id,
                    "retrieval_mode": mode,
                    "chunk_strategy": strategy,
                    "metrics": report["metrics"],
                    "failure_counts": report["metrics"]["failure_counts"],
                    "build_latency_ms": round(build_latency_ms, 3),
                }
            )
            LOGGER.info(
                "V4.1 experiment completed: %s recall=%.4f ndcg=%.4f elapsed=%.2fs",
                experiment_id,
                report["metrics"].get("retrieval_recall_at_k", 0.0),
                report["metrics"].get("ndcg_at_k", 0.0),
                time.perf_counter() - experiment_started,
            )
    summary = {
        "project": "PaperStorm Retrieval Ablation v4.1",
        "dataset_version": dataset.get("dataset_version"),
        "dataset_metadata": dataset.get("metadata") or {},
        "top_k": top_k,
        "experiments": experiments,
        "best_by_recall": _best(experiments, "retrieval_recall_at_k"),
        "best_by_ndcg": _best(experiments, "ndcg_at_k"),
        "notes": [
            "RRF combines ranks, not uncalibrated BM25 and cosine scores.",
            "Cross-Encoder latency includes query-candidate joint scoring.",
            "Synthetic seed and weakly supervised paper sets must be reported separately.",
        ],
    }
    (output_dir / "rag_eval_v41_ablation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "rag_eval_v41_ablation.md").write_text(
        _to_markdown(summary), encoding="utf-8"
    )
    return summary


def _dataset_chunks(dataset: Dict, strategy: str):
    chunks = []
    for index, document in enumerate(dataset.get("corpus") or [], start=1):
        chunk_ids = document.get("chunk_ids") or ["doc-{0}-chunk-1".format(index)]
        content = str(document.get("text") or document.get("content") or "")
        title = str(document.get("title") or document.get("document_id") or "")
        context = content
        if strategy == "contextual":
            metadata = document.get("metadata") or {}
            context = metadata.get("context") or "Document: {0}\nCategory: {1}\n{2}".format(
                title,
                metadata.get("category") or "unknown",
                content,
            )
        for chunk_id in chunk_ids:
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": document.get("document_id") or chunk_id,
                    "title": title,
                    "content": content,
                    "retrieval_content": context,
                    "metadata": dict(document.get("metadata") or {}, chunk_strategy=strategy),
                }
            )
    return chunks


def _best(experiments, metric):
    if not experiments:
        return None
    return max(experiments, key=lambda item: item["metrics"].get(metric, 0.0))["experiment_id"]


def _to_markdown(report):
    lines = [
        "# PaperStorm V4.1 检索消融报告",
        "",
        "| 实验 | Recall@K | MRR | nDCG@K | P95(ms) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for item in report.get("experiments") or []:
        metrics = item["metrics"]
        lines.append(
            "| {0} | {1:.4f} | {2:.4f} | {3:.4f} | {4:.3f} |".format(
                item["experiment_id"],
                metrics.get("retrieval_recall_at_k", 0.0),
                metrics.get("mrr", 0.0),
                metrics.get("ndcg_at_k", 0.0),
                metrics.get("p95_latency_ms", 0.0),
            )
        )
    lines.extend(
        [
            "",
            "- Recall 最优：`{0}`".format(report.get("best_by_recall")),
            "- nDCG 最优：`{0}`".format(report.get("best_by_ndcg")),
            "- 真实论文弱标注集与 synthetic seed 必须分开解读。",
        ]
    )
    return "\n".join(lines) + "\n"
