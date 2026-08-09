"""Auditable real-paper retrieval evaluation with document-level holdout.

The generated cases are annotation *candidates*, not expert labels. They keep
source evidence and hashes so a domain reviewer can approve or rewrite them.
Retrieval configuration is selected on document-disjoint dev cases and then
reported once on the frozen test split.
"""

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .paperstorm_retrieval_v41 import CrossEncoderReranker, HybridPaperIndex


DATASET_VERSION = "paperstorm-real-paper-v5.2-candidate-v1"
_STOPWORDS = {
    "abstract",
    "introduction",
    "method",
    "methods",
    "result",
    "results",
    "discussion",
    "conclusion",
    "figure",
    "table",
    "paper",
    "using",
    "based",
    "proposed",
    "model",
    "study",
    "performance",
    "analysis",
    "experiment",
    "experiments",
    "ieee",
    "transactions",
    "letters",
    "journal",
    "volume",
    "technol",
    "technology",
    "september",
    "november",
    "january",
    "february",
    "march",
    "april",
    "august",
    "october",
    "december",
    "enabled",
    "novel",
    "considerate",
    "great",
    "importance",
    "along",
    "increase",
}

_DOMAIN_ALIASES = [
    ("near-field downlink beamforming", "近场下行波束成形"),
    ("hybrid message passing", "混合消息传递"),
    ("tensor cp decomposition", "张量CP分解"),
    ("alternating least squares", "交替最小二乘"),
    ("blind and semi-blind", "盲与半盲估计"),
    ("simultaneous channel estimation and sensing", "信道估计与感知一体化"),
    ("beam squint", "波束偏斜"),
    ("block sparse", "块稀疏"),
    ("block sparsity", "块稀疏"),
    ("residual difference", "残差差分"),
    ("gridless", "无网格估计"),
    ("millimeter wave", "毫米波"),
    ("sum rate maximization", "和速率最大化"),
    ("machine learning", "机器学习"),
    ("adaptive diversity combining", "自适应分集合并"),
    ("passive intermodulation", "无源互调"),
    ("intermodulation", "互调失真"),
    ("digital predistortion", "数字预失真"),
    ("predistortion", "预失真"),
    ("visible light communication", "可见光通信"),
    ("noma-vlc", "非正交多址可见光通信"),
    ("noma", "非正交多址"),
    ("dco-ofdm", "直流偏置光正交频分复用"),
    ("mimo-ofdm", "多输入多输出正交频分复用"),
    ("mimo ofdm", "多输入多输出正交频分复用"),
    ("mimo", "多输入多输出"),
    ("channel estimation", "信道估计"),
    ("location estimation", "定位估计"),
    ("power allocation", "功率分配"),
    ("user association", "用户关联"),
    ("full-duplex", "全双工"),
    ("precoding", "预编码"),
    ("precoder", "预编码器"),
    ("mmwave", "毫米波"),
    ("beam squint", "波束偏斜"),
    ("block sparsity", "块稀疏"),
    ("neural network", "神经网络"),
    ("deep learning", "深度学习"),
    ("network compression", "网络压缩"),
    ("emitter identification", "辐射源识别"),
    ("demodulator", "解调器"),
    ("spatial diversity", "空间分集"),
    ("underwater", "水下通信"),
    ("fairness", "公平性"),
    ("inner-product", "内积"),
    ("periodic functions", "周期函数"),
    ("nonlinear", "非线性"),
    ("cancellation", "干扰抵消"),
    ("suppression", "抑制"),
]


def build_auditable_dataset(
    chunks: Iterable[Dict],
    test_ratio: float = 0.25,
    split_seed: int = 52,
    max_cases: Optional[int] = None,
    query_variants_per_document: int = 2,
    cross_lingual_only: bool = False,
) -> Dict:
    """Build source-grounded document-retrieval cases without title leakage."""
    chunks = [_normalise_chunk(item, index) for index, item in enumerate(chunks)]
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be between 0 and 1")
    document_ids = sorted({item["document_id"] for item in chunks})
    if len(document_ids) < 2:
        raise ValueError("document-level evaluation requires at least two papers")
    grouped = _chunks_by_document(chunks)
    eligible_document_ids = document_ids
    if cross_lingual_only:
        eligible_document_ids = [
            document_id
            for document_id in document_ids
            if len(_semantic_title_terms(grouped[document_id][0].get("title") or ""))
            >= 2
        ]
    if len(eligible_document_ids) < 2:
        raise ValueError("evaluation requires at least two eligible papers")
    corpus = [_corpus_record(item) for item in chunks]
    document_frequency = _document_frequency(chunks)
    cases = []
    for document_id in sorted(grouped):
        document_chunks = grouped[document_id]
        source = document_chunks[0]
        semantic_terms = _semantic_title_terms(source.get("title") or "")
        if cross_lingual_only and len(semantic_terms) < 2:
            continue
        terms = semantic_terms
        if len(terms) < 2:
            terms = _distinctive_terms(
                source.get("title") or "",
                limit=4,
                document_frequency=document_frequency,
                total_documents=len(document_ids),
            )
        if len(terms) < 2:
            continue
        title = str(source.get("title") or "")
        for variant_index, query in enumerate(
            _candidate_queries(terms)[: max(1, query_variants_per_document)]
        ):
            if title and title.lower() in query.lower():
                continue
            hard_negatives = _hard_negatives(source, chunks, query, limit=3)
            if not hard_negatives:
                continue
            metadata = dict(source.get("metadata") or {})
            case_id = "real-{0}".format(
                hashlib.sha256(
                    "{0}|{1}|{2}".format(document_id, variant_index, query).encode(
                        "utf-8"
                    )
                ).hexdigest()[:14]
            )
            relevant_chunk_ids = [item["chunk_id"] for item in document_chunks]
            cases.append(
                {
                    "case_id": case_id,
                    "query": query,
                    "split": "pending",
                    "expected_behavior": "answer",
                    "relevant_chunk_ids": relevant_chunk_ids,
                    "allowed_citation_ids": relevant_chunk_ids,
                    "hard_negative_chunk_ids": hard_negatives,
                    "evidence": {
                        "chunk_id": source["chunk_id"],
                        "page_number": metadata.get("page_number"),
                        "heading": metadata.get("heading") or "",
                        "excerpt": str(source.get("content") or "")[:500],
                        "content_sha256": _sha256_text(source.get("content") or ""),
                    },
                    "metadata": {
                        "source_document_id": document_id,
                        "source_title": title,
                        "query_terms": terms[:2],
                        "query_variant": variant_index,
                        "query_type": (
                            "cross_lingual_semantic"
                            if semantic_terms
                            else "lexical_title_concept"
                        ),
                        "target_granularity": "document",
                        "label_method": "title_concept_source_grounded_auto_candidate",
                        "review_status": "needs_human_review",
                    },
                }
            )
    if max_cases:
        cases = _balanced_case_cap(cases, max_cases)
    cases = _drop_cross_document_duplicate_queries(cases)
    evaluated_document_ids = sorted(
        {(case.get("metadata") or {}).get("source_document_id") for case in cases}
    )
    if len(evaluated_document_ids) < 2:
        raise ValueError("evaluation requires at least two unambiguous papers")
    test_documents = _stable_test_documents(
        evaluated_document_ids, test_ratio, split_seed
    )
    for case in cases:
        document_id = (case.get("metadata") or {}).get("source_document_id")
        case["split"] = "test" if document_id in test_documents else "dev"
    dataset = {
        "dataset_version": DATASET_VERSION,
        "metadata": {
            "provenance": "local_zotero_pdf_text",
            "annotation_status": "auto_candidate",
            "human_review_required_for_resume_claims": True,
            "split_unit": "document_id",
            "target_granularity": "document",
            "cross_lingual_only": cross_lingual_only,
            "split_seed": split_seed,
            "test_ratio": test_ratio,
            "contains_private_paths": False,
            "document_count": len(document_ids),
            "evaluated_document_count": len(evaluated_document_ids),
            "corpus_chunk_count": len(corpus),
            "case_count": len(cases),
            "corpus_sha256": _corpus_hash(corpus),
        },
        "corpus": corpus,
        "cases": cases,
    }
    dataset["metadata"]["dataset_sha256"] = _sha256_text(
        json.dumps(dataset, ensure_ascii=False, sort_keys=True)
    )
    return dataset


def bootstrap_mean_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    samples: int = 2000,
    seed: int = 52,
) -> Dict:
    values = [float(value) for value in values]
    if not values:
        return {"mean": 0.0, "low": 0.0, "high": 0.0, "n": 0, "samples": samples}
    if samples < 1:
        raise ValueError("samples must be positive")
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        means.append(statistics.mean(rng.choice(values) for _ in values))
    means.sort()
    tail = (1.0 - confidence) / 2.0
    low_index = max(0, min(len(means) - 1, int(math.floor(tail * len(means)))))
    high_index = max(
        0,
        min(len(means) - 1, int(math.ceil((1.0 - tail) * len(means))) - 1),
    )
    return {
        "mean": round(statistics.mean(values), 6),
        "low": round(means[low_index], 6),
        "high": round(means[high_index], 6),
        "n": len(values),
        "samples": samples,
        "confidence": confidence,
    }


def run_frozen_evaluation(
    dataset: Dict,
    output_dir,
    embedding_provider,
    modes: Optional[Sequence[str]] = None,
    top_k: int = 5,
    reranker=None,
    bootstrap_samples: int = 2000,
) -> Dict:
    """Select retrieval mode on dev, then evaluate that frozen mode on test."""
    modes = list(modes or ["bm25", "dense", "hybrid", "hybrid_rerank"])
    chunks = _dataset_chunks(dataset)
    index = HybridPaperIndex(chunks, embedding_provider=embedding_provider)
    if "hybrid_rerank" in modes and reranker is None:
        reranker = CrossEncoderReranker()
    dev_cases = [
        case for case in dataset.get("cases") or [] if case.get("split") == "dev"
    ]
    test_cases = [
        case for case in dataset.get("cases") or [] if case.get("split") == "test"
    ]
    if not dev_cases or not test_cases:
        raise ValueError(
            "dataset must contain non-empty document-disjoint dev and test cases"
        )
    dev_results = {
        mode: _evaluate_mode(
            index,
            dev_cases,
            mode,
            top_k,
            reranker,
            bootstrap_samples,
        )
        for mode in modes
    }
    selected = max(
        modes,
        key=lambda mode: (
            dev_results[mode]["ndcg_at_k"],
            dev_results[mode]["mrr"],
            dev_results[mode]["recall_at_k"],
            -dev_results[mode]["p95_latency_ms"],
        ),
    )
    baseline_mode = "bm25" if "bm25" in modes else modes[0]
    test_selected = _evaluate_mode(
        index,
        test_cases,
        selected,
        top_k,
        reranker,
        bootstrap_samples,
    )
    test_baseline = _evaluate_mode(
        index,
        test_cases,
        baseline_mode,
        top_k,
        reranker,
        bootstrap_samples,
    )
    report = {
        "project": "PaperStorm Real-Paper Retrieval Evaluation v5.2",
        "protocol": "document_holdout_dev_selection_frozen_test",
        "selection_split": "dev",
        "final_reporting_split": "test",
        "selected_config": selected,
        "baseline_config": baseline_mode,
        "dataset": {
            "version": dataset.get("dataset_version"),
            "dataset_sha256": (dataset.get("metadata") or {}).get("dataset_sha256"),
            "corpus_sha256": (dataset.get("metadata") or {}).get("corpus_sha256"),
            "document_count": (dataset.get("metadata") or {}).get("document_count"),
            "dev_case_count": len(dev_cases),
            "test_case_count": len(test_cases),
            "annotation_status": (dataset.get("metadata") or {}).get(
                "annotation_status"
            ),
        },
        "models": {
            "embedding": str(getattr(embedding_provider, "name", "unknown")),
            "reranker": (
                str(getattr(reranker, "model_name", "custom"))
                if selected == "hybrid_rerank"
                else None
            ),
        },
        "code_commit": _git_commit(),
        "top_k": top_k,
        "dev": dev_results,
        "test": {"baseline": test_baseline, "selected": test_selected},
        "integrity": {
            "split_unit": "document_id",
            "test_was_not_used_for_selection": True,
            "all_cases_have_source_evidence": all(
                case.get("evidence") for case in dev_cases + test_cases
            ),
            "human_review_complete": all(
                (case.get("metadata") or {}).get("review_status") == "approved"
                for case in dev_cases + test_cases
            ),
        },
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "real_paper_eval_v52.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "real_paper_eval_v52.md").write_text(
        _report_markdown(report), encoding="utf-8"
    )
    _write_review_sheet(dataset, output_dir / "real_paper_review_candidates.jsonl")
    return report


def _evaluate_mode(index, cases, mode, top_k, reranker, bootstrap_samples):
    from .paperstorm_retrieval_runtime import _percentile, _retrieval_metrics

    per_case = []
    for case in cases:
        started = time.perf_counter()
        ranked = index.search(
            case["query"],
            mode=mode,
            top_k=top_k,
            reranker=reranker if mode == "hybrid_rerank" else None,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        ranked_ids = [str(item.get("chunk_id") or "") for item in ranked]
        recall, reciprocal_rank, ndcg = _retrieval_metrics(
            ranked_ids, set(case.get("relevant_chunk_ids") or []), top_k
        )
        per_case.append(
            {
                "case_id": case["case_id"],
                "recall_at_k": recall,
                "mrr": reciprocal_rank,
                "ndcg_at_k": ndcg,
                "latency_ms": round(latency_ms, 4),
                "ranked_chunk_ids": ranked_ids,
            }
        )
    recall_values = [item["recall_at_k"] for item in per_case]
    mrr_values = [item["mrr"] for item in per_case]
    ndcg_values = [item["ndcg_at_k"] for item in per_case]
    latencies = [item["latency_ms"] for item in per_case]
    return {
        "mode": mode,
        "case_count": len(per_case),
        "recall_at_k": round(statistics.mean(recall_values), 6),
        "mrr": round(statistics.mean(mrr_values), 6),
        "ndcg_at_k": round(statistics.mean(ndcg_values), 6),
        "p95_latency_ms": round(_percentile(latencies, 0.95), 4),
        "confidence_intervals": {
            "recall_at_k": bootstrap_mean_ci(recall_values, samples=bootstrap_samples),
            "mrr": bootstrap_mean_ci(mrr_values, samples=bootstrap_samples),
            "ndcg_at_k": bootstrap_mean_ci(ndcg_values, samples=bootstrap_samples),
        },
        "cases": per_case,
    }


def _normalise_chunk(item, index):
    item = dict(item)
    item["chunk_id"] = str(item.get("chunk_id") or "chunk-{0}".format(index + 1))
    item["document_id"] = str(item.get("document_id") or item["chunk_id"])
    item["title"] = str(item.get("title") or item["document_id"])
    item["content"] = str(item.get("content") or "")
    item["retrieval_content"] = str(item.get("retrieval_content") or item["content"])
    item["metadata"] = dict(item.get("metadata") or {})
    return item


def _corpus_record(chunk):
    metadata = chunk.get("metadata") or {}
    return {
        "document_id": chunk["chunk_id"],
        "source_document_id": chunk["document_id"],
        "chunk_ids": [chunk["chunk_id"]],
        "title": chunk["title"],
        "text": chunk["content"],
        "source_type": "local_pdf_chunk",
        "metadata": {
            "category": "real_paper",
            "page_number": metadata.get("page_number"),
            "heading": metadata.get("heading") or "",
            "context": chunk["retrieval_content"],
            "content_sha256": _sha256_text(chunk["content"]),
        },
    }


def _dataset_chunks(dataset):
    output = []
    for document in dataset.get("corpus") or []:
        metadata = dict(document.get("metadata") or {})
        chunk_id = (document.get("chunk_ids") or [document["document_id"]])[0]
        output.append(
            {
                "chunk_id": chunk_id,
                "document_id": document.get("source_document_id")
                or document["document_id"],
                "title": document.get("title") or "",
                "content": document.get("text") or "",
                "retrieval_content": metadata.get("context")
                or document.get("text")
                or "",
                "metadata": metadata,
            }
        )
    return output


def _stable_test_documents(document_ids, test_ratio, seed):
    ranked = sorted(
        document_ids,
        key=lambda value: hashlib.sha256(
            "{0}|{1}".format(seed, value).encode("utf-8")
        ).hexdigest(),
    )
    count = max(1, min(len(ranked) - 1, int(round(len(ranked) * test_ratio))))
    return set(ranked[:count])


def _round_robin_by_document(chunks):
    grouped = {}
    order = []
    for chunk in chunks:
        document_id = chunk["document_id"]
        if document_id not in grouped:
            grouped[document_id] = []
            order.append(document_id)
        grouped[document_id].append(chunk)
    depth = 0
    while True:
        emitted = False
        for document_id in order:
            items = grouped[document_id]
            if depth < len(items):
                emitted = True
                yield items[depth]
        if not emitted:
            return
        depth += 1


def _candidate_tokens(text):
    tokens = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*", str(text)):
        lowered = token.lower()
        if len(lowered) < 4 or lowered in _STOPWORDS:
            continue
        if lowered.isdigit() or re.fullmatch(r"vol|no|pp|et|al", lowered):
            continue
        tokens.append(lowered)
    tokens.extend(re.findall(r"[\u4e00-\u9fff]{2,6}", str(text)))
    return tokens


def _document_frequency(chunks):
    by_document = {}
    for chunk in chunks:
        by_document.setdefault(chunk["document_id"], set()).update(
            _candidate_tokens(
                "{0} {1}".format(chunk.get("title") or "", chunk.get("content") or "")
            )
        )
    frequency = Counter()
    for terms in by_document.values():
        frequency.update(terms)
    return frequency


def _distinctive_terms(
    text,
    limit=3,
    document_frequency=None,
    total_documents=None,
):
    tokens = _candidate_tokens(text)
    counts = Counter(tokens)
    document_frequency = document_frequency or Counter(tokens)
    total_documents = max(1, int(total_documents or 1))
    first_position = {token: tokens.index(token) for token in counts}
    scored = []
    for token, count in counts.items():
        df = max(1, int(document_frequency.get(token, 1)))
        idf = math.log((total_documents + 1.0) / (df + 0.5))
        acronym_bonus = 0.8 if token.isupper() or "-" in token else 0.0
        score = (1.0 + math.log(count)) * idf + acronym_bonus
        scored.append((score, first_position[token], token))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [token for _, _, token in scored[:limit]]


def _candidate_query(terms):
    if len(terms) == 1:
        return "相关研究中的 {0} 解决了什么问题？".format(terms[0])
    if len(terms) >= 3:
        return "相关研究中，{0}、{1} 与 {2} 的作用或关系是什么？".format(
            terms[0], terms[1], terms[2]
        )
    return "相关研究中，{0} 与 {1} 的作用或关系是什么？".format(terms[0], terms[1])


def _candidate_queries(terms):
    queries = [_candidate_query(terms)]
    if len(terms) >= 3:
        queries.append(
            "哪些论文研究了 {0} 场景中的 {1} 与 {2}？".format(
                terms[0], terms[1], terms[2]
            )
        )
    else:
        queries.append("哪些论文同时讨论 {0} 和 {1}？".format(terms[0], terms[1]))
    return queries


def _semantic_title_terms(title):
    lowered = str(title or "").lower()
    terms = []
    consumed = []
    for phrase, alias in _DOMAIN_ALIASES:
        if phrase not in lowered:
            continue
        if any(phrase in existing or existing in phrase for existing in consumed):
            continue
        consumed.append(phrase)
        if alias not in terms:
            terms.append(alias)
    return terms


def _hard_negatives(source, chunks, query, limit=3):
    query_terms = set(_distinctive_terms(query, limit=20))
    candidates = []
    for chunk in chunks:
        if chunk["document_id"] == source["document_id"]:
            continue
        terms = set(_distinctive_terms(chunk.get("retrieval_content") or "", limit=30))
        candidates.append((len(query_terms & terms), chunk["chunk_id"]))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [chunk_id for _, chunk_id in candidates[:limit]]


def _relevant_chunks(source, chunks, terms):
    required = set(term.lower() for term in terms[:2])
    matches = []
    for chunk in chunks:
        if chunk["document_id"] != source["document_id"]:
            continue
        text_terms = set(_candidate_tokens(chunk.get("retrieval_content") or ""))
        if required <= text_terms:
            matches.append(chunk["chunk_id"])
    return matches or [source["chunk_id"]]


def _chunks_by_document(chunks):
    grouped = {}
    for chunk in chunks:
        grouped.setdefault(chunk["document_id"], []).append(chunk)
    return grouped


def _balanced_case_cap(cases, max_cases):
    grouped = {}
    order = []
    for case in cases:
        document_id = (case.get("metadata") or {}).get("source_document_id")
        if document_id not in grouped:
            grouped[document_id] = []
            order.append(document_id)
        grouped[document_id].append(case)
    selected = []
    depth = 0
    while len(selected) < max_cases:
        emitted = False
        for document_id in order:
            items = grouped[document_id]
            if depth < len(items):
                selected.append(items[depth])
                emitted = True
                if len(selected) >= max_cases:
                    return selected
        if not emitted:
            return selected
        depth += 1
    return selected


def _drop_cross_document_duplicate_queries(cases):
    owners = {}
    for case in cases:
        owners.setdefault(case["query"], set()).add(
            (case.get("metadata") or {}).get("source_document_id")
        )
    ambiguous = {query for query, documents in owners.items() if len(documents) > 1}
    return [case for case in cases if case["query"] not in ambiguous]


def _corpus_hash(corpus):
    canonical = [
        {
            "document_id": item.get("document_id"),
            "source_document_id": item.get("source_document_id"),
            "title": item.get("title"),
            "text_sha256": _sha256_text(item.get("text") or ""),
        }
        for item in corpus
    ]
    return _sha256_text(json.dumps(canonical, ensure_ascii=False, sort_keys=True))


def _sha256_text(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_review_sheet(dataset, path):
    with Path(path).open("w", encoding="utf-8") as handle:
        for case in dataset.get("cases") or []:
            record = {
                "case_id": case.get("case_id"),
                "split": case.get("split"),
                "question": case.get("query"),
                "source_title": (case.get("metadata") or {}).get("source_title"),
                "page_number": (case.get("evidence") or {}).get("page_number"),
                "evidence_excerpt": (case.get("evidence") or {}).get("excerpt"),
                "review_status": (case.get("metadata") or {}).get("review_status"),
                "reviewer_notes": "",
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _report_markdown(report):
    selected = report["test"]["selected"]
    baseline = report["test"]["baseline"]
    integrity = report["integrity"]
    return "\n".join(
        [
            "# PaperStorm v5.2 真实论文检索评测",
            "",
            "- 协议：按论文切分 dev/test；仅在 dev 选择配置，test 冻结报告。",
            "- 标注状态：`{0}`；人工审核完成：`{1}`。".format(
                report["dataset"].get("annotation_status"),
                integrity.get("human_review_complete"),
            ),
            "- 测试集：{0} 条；Top-K={1}；选择配置：`{2}`。".format(
                report["dataset"]["test_case_count"],
                report["top_k"],
                report["selected_config"],
            ),
            "",
            "| test 配置 | Recall@K | MRR | nDCG@K | P95(ms) |",
            "| --- | ---: | ---: | ---: | ---: |",
            "| {0}（baseline） | {1:.4f} | {2:.4f} | {3:.4f} | {4:.2f} |".format(
                report["baseline_config"],
                baseline["recall_at_k"],
                baseline["mrr"],
                baseline["ndcg_at_k"],
                baseline["p95_latency_ms"],
            ),
            "| {0}（dev 选出） | {1:.4f} | {2:.4f} | {3:.4f} | {4:.2f} |".format(
                report["selected_config"],
                selected["recall_at_k"],
                selected["mrr"],
                selected["ndcg_at_k"],
                selected["p95_latency_ms"],
            ),
            "",
            "> 未完成人工审核前，这些结果只能称为真实语料上的自动候选标注评测，不能称专家 QA benchmark。",
            "",
        ]
    )


def main():
    from .paperstorm_retrieval_runtime import _dense_provider
    from .paperstorm_zotero import load_zotero_chunks

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zotero-root", required=True)
    parser.add_argument("--output-dir", default="./results/paperstorm_real_eval_v52")
    parser.add_argument("--embedding", choices=["auto", "real", "hash"], default="real")
    parser.add_argument("--max-papers", type=int, default=24)
    parser.add_argument("--max-pages", type=int, default=15)
    parser.add_argument("--max-cases", type=int, default=160)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--test-ratio", type=float, default=0.25)
    parser.add_argument("--cross-lingual-only", action="store_true")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["bm25", "dense", "hybrid", "hybrid_rerank"],
        default=["bm25", "dense", "hybrid", "hybrid_rerank"],
    )
    args = parser.parse_args()
    chunks = load_zotero_chunks(
        args.zotero_root,
        max_papers=args.max_papers,
        max_pages=args.max_pages,
        strategy="contextual",
    )
    dataset = build_auditable_dataset(
        chunks,
        test_ratio=args.test_ratio,
        max_cases=args.max_cases,
        cross_lingual_only=args.cross_lingual_only,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "real_paper_dataset_v52.json").write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = run_frozen_evaluation(
        dataset,
        output_dir=output_dir,
        embedding_provider=_dense_provider(args.embedding),
        modes=args.modes,
        top_k=args.top_k,
    )
    print(json.dumps(report["test"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
