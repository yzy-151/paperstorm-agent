"""Multi-task before/after retrieval benchmark over the local Zotero library.

Designs several task groups from real papers, builds a weak-supervision
dataset per group (section provenance labels), then compares the legacy
runtime stack (token-overlap + dense + keyword rerank) against the V4.1 stack
(BM25 + Dense + RRF) on every group so the improvement is visible per task.
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional


TASK_GROUPS = [
    {
        "name": "pim",
        "label": "射频无源互调 PIM",
        "terms": ["无源互调", "passive intermodulation", "PIM", "intermodulation"],
    },
    {
        "name": "vlc",
        "label": "可见光通信 VLC",
        "terms": ["visible light", "VLC", "DCO-OFDM", "ACO-OFDM", "可见光"],
    },
    {
        "name": "channel_estimation",
        "label": "MIMO-OFDM 信道估计",
        "terms": ["channel estimation", "MIMO-OFDM", "mmWave", "信道估计"],
    },
    {
        "name": "noma_power",
        "label": "NOMA 与功率分配",
        "terms": ["NOMA", "power allocation", "non-orthogonal", "功率分配"],
    },
    {
        "name": "dpd_nonlinearity",
        "label": "非线性建模与数字预失真",
        "terms": ["predistortion", "预失真", "nonlinear", "非线性"],
    },
    {
        "name": "neural_net",
        "label": "神经网络与深度学习",
        "terms": ["neural network", "deep learning", "convolutional", "神经网络", "深度学习"],
    },
]


def compare_stacks_on_dataset(
    dataset: Dict,
    embedding_provider,
    top_k: int = 5,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> Dict:
    """Compare legacy vs V4.1 retrieval on an arbitrary dataset (corpus+cases)."""
    from .paperstorm_ablation_v41 import _dataset_chunks
    from .paperstorm_rag import PaperStormRAGIndex
    from .paperstorm_retrieval_v41 import HybridPaperIndex
    from .paperstorm_retrieval_runtime import _retrieval_metrics, _percentile

    cases = [
        case
        for case in dataset.get("cases") or []
        if case.get("expected_behavior") != "abstain"
    ]
    chunks = _dataset_chunks(dataset, "contextual")
    texts = [
        chunk.get("retrieval_content") or chunk.get("content") or ""
        for chunk in chunks
    ]
    embeddings = embedding_provider.embed(texts)
    legacy_chunks = [
        dict(chunk, embedding=embeddings[index]) for index, chunk in enumerate(chunks)
    ]
    legacy_index = PaperStormRAGIndex(
        chunks=legacy_chunks,
        embedding_provider=embedding_provider,
    )
    v41_index = HybridPaperIndex(
        chunks,
        embedding_provider=embedding_provider,
    )

    def run(index, legacy):
        hits, reciprocal, ndcg_scores, latencies = [], [], [], []
        for case in cases:
            query = str(case.get("query") or "")
            relevant = set(case.get("relevant_chunk_ids") or [])
            started = time.perf_counter()
            ranked = index.search(query, top_k=top_k)
            latencies.append((time.perf_counter() - started) * 1000)
            ranked_ids = [str(item.get("chunk_id") or "") for item in ranked[:top_k]]
            hit, mrr, ndcg = _retrieval_metrics(ranked_ids, relevant, top_k)
            hits.append(hit)
            reciprocal.append(mrr)
            ndcg_scores.append(ndcg)
        return {
            "recall_at_k": round(statistics.mean(hits), 6) if hits else 0.0,
            "mrr": round(statistics.mean(reciprocal), 6) if reciprocal else 0.0,
            "ndcg_at_k": round(statistics.mean(ndcg_scores), 6) if ndcg_scores else 0.0,
            "p95_latency_ms": round(_percentile(latencies, 0.95), 4) if latencies else 0.0,
            "case_count": len(cases),
        }

    legacy = run(legacy_index, True)
    v41 = run(v41_index, False)
    deltas = {
        key: round(v41[key] - legacy[key], 6) for key in ("recall_at_k", "mrr", "ndcg_at_k", "p95_latency_ms")
    }
    deltas["relative_recall_gain_pct"] = round(
        (v41["recall_at_k"] - legacy["recall_at_k"])
        / max(1e-9, legacy["recall_at_k"])
        * 100.0,
        2,
    )
    return {
        "legacy": legacy,
        "v41": v41,
        "deltas": deltas,
    }


def run_multi_task_benchmark(
    zotero_root,
    output_dir,
    top_k: int = 5,
    embedding: str = "hash",
    max_papers: int = 8,
    max_pages: int = 15,
    max_cases: int = 60,
) -> Dict:
    from .paperstorm_retrieval_runtime import _dense_provider
    from .paperstorm_zotero import (
        build_weak_paper_dataset,
        discover_zotero_papers,
        load_zotero_chunks,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provider = _dense_provider(embedding)
    groups = []
    for group in TASK_GROUPS:
        papers = discover_zotero_papers(
            zotero_root, query_terms=group["terms"], max_papers=max_papers
        )
        paper_titles = [str(item.get("title") or "") for item in papers]
        if not papers:
            groups.append(
                {
                    "name": group["name"],
                    "label": group["label"],
                    "terms": group["terms"],
                    "paper_count": 0,
                    "paper_titles": [],
                    "corpus_chunks": 0,
                    "case_count": 0,
                    "legacy": {},
                    "v41": {},
                    "deltas": {},
                    "skipped": "no matching papers in Zotero",
                }
            )
            continue
        chunks = load_zotero_chunks(
            zotero_root,
            query_terms=group["terms"],
            max_papers=max_papers,
            max_pages=max_pages,
            strategy="contextual",
        )
        dataset = build_weak_paper_dataset(chunks, max_cases=max_cases)
        comparison = compare_stacks_on_dataset(dataset, provider, top_k=top_k)
        groups.append(
            {
                "name": group["name"],
                "label": group["label"],
                "terms": group["terms"],
                "paper_count": len(papers),
                "paper_titles": paper_titles,
                "corpus_chunks": len(chunks),
                "case_count": comparison["legacy"]["case_count"],
                "legacy": comparison["legacy"],
                "v41": comparison["v41"],
                "deltas": comparison["deltas"],
            }
        )
    ran = [item for item in groups if item.get("case_count")]
    overall = {
        "groups_ran": len(ran),
        "total_cases": sum(item["case_count"] for item in ran),
        "recall_before": round(
            statistics.mean(item["legacy"]["recall_at_k"] for item in ran), 6
        ) if ran else 0.0,
        "recall_after": round(
            statistics.mean(item["v41"]["recall_at_k"] for item in ran), 6
        ) if ran else 0.0,
        "mrr_before": round(statistics.mean(item["legacy"]["mrr"] for item in ran), 6) if ran else 0.0,
        "mrr_after": round(statistics.mean(item["v41"]["mrr"] for item in ran), 6) if ran else 0.0,
        "ndcg_before": round(statistics.mean(item["legacy"]["ndcg_at_k"] for item in ran), 6) if ran else 0.0,
        "ndcg_after": round(statistics.mean(item["v41"]["ndcg_at_k"] for item in ran), 6) if ran else 0.0,
    }
    if ran:
        overall["relative_recall_gain_pct"] = round(
            (overall["recall_after"] - overall["recall_before"])
            / max(1e-9, overall["recall_before"])
            * 100.0,
            2,
        )
    report = {
        "project": "PaperStorm Multi-Task Retrieval Benchmark (Zotero)",
        "zotero_root": str(Path(zotero_root)),
        "embedding": embedding,
        "top_k": top_k,
        "task_groups": groups,
        "overall": overall,
        "notes": [
            "Labels are section-provenance weak supervision, pending domain review.",
            "legacy = token-overlap + dense + keyword rerank; v41 = BM25 + Dense + RRF.",
        ],
    }
    (output_dir / "multi_task_benchmark.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "multi_task_benchmark.md").write_text(
        _to_markdown(report), encoding="utf-8"
    )
    return report


def _to_markdown(report: Dict) -> str:
    lines = [
        "# PaperStorm Multi-Task Retrieval Benchmark (Zotero)",
        "",
        "embedding: {0} · top_k: {1}".format(report.get("embedding"), report.get("top_k")),
        "",
        "| 任务组 | 论文 | chunk | case | legacy Recall@K | V4.1 Recall@K | Δ | MRR Δ | nDCG Δ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in report.get("task_groups") or []:
        if not group.get("case_count"):
            lines.append(
                "| {0} | 0 | 0 | 0 | - | - | - | - | - |".format(group.get("label", ""))
            )
            continue
        lines.append(
            "| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} |".format(
                group.get("label", ""),
                group.get("paper_count", 0),
                group.get("corpus_chunks", 0),
                group.get("case_count", 0),
                group["legacy"]["recall_at_k"],
                group["v41"]["recall_at_k"],
                group["deltas"]["recall_at_k"],
                group["deltas"]["mrr"],
                group["deltas"]["ndcg_at_k"],
            )
        )
    overall = report.get("overall") or {}
    lines.extend(
        [
            "",
            "Overall: recall {0} -> {1} ({2}%), MRR {3} -> {4}, nDCG {5} -> {6}".format(
                overall.get("recall_before", 0),
                overall.get("recall_after", 0),
                overall.get("relative_recall_gain_pct", 0),
                overall.get("mrr_before", 0),
                overall.get("mrr_after", 0),
                overall.get("ndcg_before", 0),
                overall.get("ndcg_after", 0),
            ),
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zotero-root", default=os.getenv("PAPERSTORM_ZOTERO_ROOT", ""))
    parser.add_argument("--output-dir", default="./results/paperstorm_multi_task")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embedding", choices=["auto", "real", "hash"], default="hash")
    parser.add_argument("--max-papers", type=int, default=8)
    parser.add_argument("--max-pages", type=int, default=15)
    parser.add_argument("--max-cases", type=int, default=60)
    args = parser.parse_args()
    if not args.zotero_root:
        raise ValueError("--zotero-root is required (or set PAPERSTORM_ZOTERO_ROOT)")
    report = run_multi_task_benchmark(
        args.zotero_root,
        Path(args.output_dir),
        top_k=args.top_k,
        embedding=args.embedding,
        max_papers=args.max_papers,
        max_pages=args.max_pages,
        max_cases=args.max_cases,
    )
    print(json.dumps(report["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
