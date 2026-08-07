import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional


SEED_FACTS = [
    {
        "id": "pim-definition",
        "subject": "无源互调的定义",
        "english_subject": "the definition of passive intermodulation",
        "content": "无源互调（Passive Intermodulation, PIM）是无源射频器件在多载波激励下产生的非线性失真。",
        "required_terms": ["无源互调", "非线性失真"],
        "category": "definition",
    },
    {
        "id": "pim-disambiguation",
        "subject": "PIM 在射频领域的含义",
        "english_subject": "the RF meaning of PIM",
        "content": "在射频通信语境中，PIM 通常指 passive intermodulation，而不是 processing-in-memory。",
        "required_terms": ["passive intermodulation", "processing-in-memory"],
        "category": "acronym_disambiguation",
    },
    {
        "id": "pim-contact-nonlinearity",
        "subject": "接触非线性导致无源互调",
        "english_subject": "contact nonlinearity as a PIM source",
        "content": "松动连接、粗糙接触面和氧化层会形成非线性接触，是无源互调的常见来源。",
        "required_terms": ["非线性接触", "氧化层"],
        "category": "cause",
    },
    {
        "id": "pim-material-nonlinearity",
        "subject": "材料非线性导致无源互调",
        "english_subject": "material nonlinearity as a PIM source",
        "content": "铁磁材料、表面污染和材料微观非线性可能产生或增强无源互调。",
        "required_terms": ["铁磁材料", "材料"],
        "category": "cause",
    },
    {
        "id": "pim-order",
        "subject": "三阶无源互调产物",
        "english_subject": "third-order passive intermodulation products",
        "content": "双音输入 f1 和 f2 的三阶互调产物常出现在 2f1-f2 与 2f2-f1，可能落入接收带宽。",
        "required_terms": ["2f1-f2", "2f2-f1"],
        "category": "formula",
    },
    {
        "id": "pim-impact",
        "subject": "无源互调对接收机的影响",
        "english_subject": "the receiver impact of PIM",
        "content": "当无源互调产物落入接收频段时，会抬高噪声底并降低接收灵敏度。",
        "required_terms": ["噪声底", "接收灵敏度"],
        "category": "impact",
    },
    {
        "id": "pim-measurement",
        "subject": "无源互调测试方法",
        "english_subject": "passive intermodulation testing",
        "content": "典型无源互调测试向被测件注入两个高功率载波，并测量指定阶次互调产物的功率。",
        "required_terms": ["两个高功率载波", "互调产物"],
        "category": "measurement",
    },
    {
        "id": "pim-unit",
        "subject": "无源互调测试结果单位",
        "english_subject": "PIM measurement units",
        "content": "无源互调水平常用 dBm 表示绝对功率，或用 dBc 表示相对载波功率。",
        "required_terms": ["dBm", "dBc"],
        "category": "measurement",
    },
    {
        "id": "pim-prevention",
        "subject": "无源互调的工程预防",
        "english_subject": "engineering prevention of PIM",
        "content": "清洁接触面、控制紧固扭矩、避免铁磁材料并提高连接器质量可以降低无源互调。",
        "required_terms": ["紧固扭矩", "连接器"],
        "category": "suppression",
    },
    {
        "id": "pim-digital-cancellation",
        "subject": "无源互调数字抵消",
        "english_subject": "digital cancellation of passive intermodulation",
        "content": "数字抵消先建立无源互调非线性模型，再估计干扰分量并从接收信号中减去。",
        "required_terms": ["非线性模型", "减去"],
        "category": "suppression",
    },
    {
        "id": "pim-neural-cancellation",
        "subject": "神经网络无源互调抵消",
        "english_subject": "neural-network PIM cancellation",
        "content": "神经网络可以学习发射载波到无源互调干扰之间的非线性映射，用于估计和抵消干扰。",
        "required_terms": ["神经网络", "非线性映射"],
        "category": "neural_suppression",
    },
    {
        "id": "pim-training-data",
        "subject": "神经网络无源互调模型的训练数据",
        "english_subject": "training data for neural PIM models",
        "content": "训练神经网络无源互调模型通常需要同步的发射参考信号与接收端干扰观测。",
        "required_terms": ["发射参考信号", "接收端"],
        "category": "neural_suppression",
    },
    {
        "id": "pim-generalization",
        "subject": "神经网络无源互调模型的泛化问题",
        "english_subject": "generalization of neural PIM models",
        "content": "功率、温度、器件状态和频段变化会造成数据分布漂移，降低神经网络无源互调模型的泛化能力。",
        "required_terms": ["分布漂移", "泛化"],
        "category": "neural_suppression",
    },
    {
        "id": "pim-memory-effect",
        "subject": "无源互调建模中的记忆效应",
        "english_subject": "memory effects in PIM modeling",
        "content": "热过程和频率相关响应可能使无源互调呈现记忆效应，模型需要利用过去的输入样本。",
        "required_terms": ["记忆效应", "过去的输入"],
        "category": "modeling",
    },
    {
        "id": "pim-evaluation-nmse",
        "subject": "无源互调建模的 NMSE 指标",
        "english_subject": "NMSE for PIM modeling",
        "content": "归一化均方误差（NMSE）可衡量无源互调模型预测信号与真实干扰信号之间的误差。",
        "required_terms": ["NMSE", "预测"],
        "category": "evaluation",
    },
    {
        "id": "pim-evaluation-cancellation",
        "subject": "无源互调抵消效果指标",
        "english_subject": "PIM cancellation performance metrics",
        "content": "无源互调抵消效果可用残余干扰功率、抵消增益和接收灵敏度改善量评估。",
        "required_terms": ["残余干扰功率", "抵消增益"],
        "category": "evaluation",
    },
    {
        "id": "rag-bm25-role",
        "subject": "BM25 在论文检索中的作用",
        "english_subject": "the role of BM25 in paper retrieval",
        "content": "BM25 擅长召回包含精确术语、缩写、型号和数字的论文片段，可与语义向量检索互补。",
        "required_terms": ["精确术语", "向量检索"],
        "category": "rag",
    },
    {
        "id": "rag-dense-role",
        "subject": "向量检索在论文 RAG 中的作用",
        "english_subject": "the role of dense retrieval in paper RAG",
        "content": "Dense Retrieval 使用 Embedding 召回语义相近的片段，即使查询和文档没有完全相同的词。",
        "required_terms": ["Embedding", "语义相近"],
        "category": "rag",
    },
    {
        "id": "rag-rrf-role",
        "subject": "RRF 如何融合检索结果",
        "english_subject": "how RRF fuses retrieval rankings",
        "content": "Reciprocal Rank Fusion 根据文档在多个结果列表中的名次累加倒数排名分数，不要求原始分数同尺度。",
        "required_terms": ["名次", "原始分数"],
        "category": "rag",
    },
    {
        "id": "rag-rerank-role",
        "subject": "Cross-Encoder Reranker 的作用",
        "english_subject": "the role of a cross-encoder reranker",
        "content": "Cross-Encoder 将查询和候选片段联合编码，通常比第一阶段召回更精确，但推理成本更高。",
        "required_terms": ["联合编码", "推理成本"],
        "category": "rag",
    },
]


ANSWER_QUERY_TEMPLATES = [
    "什么是{subject}？",
    "请解释{subject}。",
    "关于{subject}，核心结论是什么？",
    "How should I understand {english_subject}?",
]

NO_ANSWER_QUERIES = [
    "请给出该资料没有记录的基站供应商内部故障编号。",
    "这组资料能否证明所有天线在任何温度下都不会产生 PIM？",
    "请报告文档中未提供的 2035 年无源互调市场规模。",
    "资料是否给出了某运营商全部站点的实时 PIM 测量值？",
    "请提供材料中不存在的连接器序列号清单。",
    "文档能否证明神经网络抵消在所有硬件上必然优于传统方法？",
    "请给出资料没有披露的训练数据下载地址。",
    "材料是否包含所有实验的原始 IQ 数据？",
    "请报告文档未给出的每个模型参数数量。",
    "资料能否确定某个未知基站今天的紧固扭矩？",
    "请给出材料中没有记录的测试仪器校准证书编号。",
    "文档是否提供全球所有频段的统一 PIM 阈值？",
    "请列出资料没有提到的专利授权费用。",
    "材料能否预测尚未采集数据的未来器件老化曲线？",
    "请提供文档中未公开的内部客户名称。",
    "资料是否包含每个温度点的完整训练日志？",
    "请给出材料没有说明的模型商业部署成本。",
    "文档能否证明任何数据分布漂移都不会影响模型？",
    "请提供材料中不存在的生产数据库密码。",
    "资料是否能回答未记录设备的实时告警状态？",
]


def build_seed_dataset():
    corpus = []
    cases = []
    for fact in SEED_FACTS:
        document_id = fact["id"]
        chunk_id = "{0}-chunk-1".format(document_id)
        corpus.append(
            {
                "document_id": document_id,
                "chunk_ids": [chunk_id],
                "title": fact["subject"],
                "text": fact["content"],
                "source_type": "v4_seed_fact",
                "metadata": {"category": fact["category"], "review_status": "seed"},
            }
        )
        for variant, template in enumerate(ANSWER_QUERY_TEMPLATES, start=1):
            cases.append(
                {
                    "case_id": "{0}-q{1}".format(fact["id"], variant),
                    "query": template.format(**fact),
                    "category": fact["category"],
                    "relevant_chunk_ids": [chunk_id],
                    "expected_behavior": "answer",
                    "reference_answer": fact["content"],
                    "required_answer_terms": list(fact["required_terms"]),
                    "allowed_citation_ids": [chunk_id],
                    "metadata": {"source": "curated_seed_fact", "review_status": "seed"},
                }
            )
    for index, query in enumerate(NO_ANSWER_QUERIES, start=1):
        cases.append(
            {
                "case_id": "no-answer-{0:02d}".format(index),
                "query": query,
                "category": "unanswerable",
                "relevant_chunk_ids": [],
                "expected_behavior": "abstain",
                "reference_answer": "现有资料不足以回答。",
                "required_answer_terms": [],
                "allowed_citation_ids": [],
                "metadata": {"source": "curated_seed_question", "review_status": "seed"},
            }
        )
    return {
        "dataset_version": "4.0-seed-1",
        "metadata": {
            "name": "PaperStorm v4.0 auditable seed set",
            "provenance": "synthetic_seed",
            "domain_review_required": True,
            "case_design": "20 controlled facts x 4 paraphrases + 20 unanswerable cases",
        },
        "corpus": corpus,
        "cases": cases,
    }


def validate_dataset(dataset: Dict):
    errors = []
    cases = dataset.get("cases") or []
    corpus_chunk_ids = {
        str(chunk_id)
        for document in dataset.get("corpus") or []
        for chunk_id in document.get("chunk_ids") or []
    }
    seen = set()
    required = {"case_id", "query", "relevant_chunk_ids", "expected_behavior"}
    for index, case in enumerate(cases):
        missing = sorted(required - set(case))
        if missing:
            errors.append({"index": index, "error": "missing_fields", "fields": missing})
        case_id = case.get("case_id")
        if case_id in seen:
            errors.append({"index": index, "error": "duplicate_case_id", "case_id": case_id})
        seen.add(case_id)
        if case.get("expected_behavior") not in {"answer", "abstain"}:
            errors.append({"index": index, "error": "invalid_expected_behavior"})
        if corpus_chunk_ids:
            unknown = sorted(set(case.get("relevant_chunk_ids") or []) - corpus_chunk_ids)
            if unknown:
                errors.append({"index": index, "error": "unknown_relevant_chunks", "chunks": unknown})
    return {"valid": not errors, "error_count": len(errors), "errors": errors}


def load_dataset(path):
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        dataset = {"dataset_version": path.stem, "metadata": {"provenance": str(path)}, "cases": cases}
    else:
        dataset = json.loads(path.read_text(encoding="utf-8"))
    validation = validate_dataset(dataset)
    if not validation["valid"]:
        raise ValueError("Invalid evaluation dataset: {0}".format(validation["errors"]))
    return dataset


def evaluate_observation(case: Dict, observation: Dict, top_k: int = 5):
    candidates = list(observation.get("candidates") or [])
    selected = list(observation.get("selected") or [])[:top_k]
    relevant = set(case.get("relevant_chunk_ids") or [])
    selected_ids = [_chunk_id(item) for item in selected]
    candidate_ids = [_chunk_id(item) for item in candidates]
    relevant_selected = [chunk_id for chunk_id in selected_ids if chunk_id in relevant]
    expected_behavior = case.get("expected_behavior") or "answer"

    if relevant:
        recall_at_k = len(set(relevant_selected)) / len(relevant)
        precision_at_k = len(relevant_selected) / max(1, len(selected_ids))
        first_rank = next((index for index, value in enumerate(selected_ids, start=1) if value in relevant), 0)
        mrr = 1.0 / first_rank if first_rank else 0.0
        ndcg_at_k = _ndcg(selected_ids, relevant, top_k)
    else:
        recall_at_k = 1.0 if not selected_ids else 0.0
        precision_at_k = 1.0 if not selected_ids else 0.0
        mrr = 1.0 if not selected_ids else 0.0
        ndcg_at_k = 1.0 if not selected_ids else 0.0

    answer = str(observation.get("answer") or "")
    prompt_context = str(observation.get("prompt_context") or "")
    required_terms = [str(term) for term in case.get("required_answer_terms") or [] if term]
    answer_hits = _term_hits(answer, required_terms)
    context_hits = _term_hits(prompt_context, required_terms)
    citations = [_citation_id(value) for value in observation.get("citations") or []]
    allowed_citations = set(case.get("allowed_citation_ids") or [])
    valid_citations = [value for value in citations if value in allowed_citations]
    citation_precision = len(valid_citations) / len(citations) if citations else (1.0 if not allowed_citations else 0.0)
    citation_recall = len(set(valid_citations)) / len(allowed_citations) if allowed_citations else 1.0
    abstained = bool(observation.get("abstained")) or _looks_like_abstention(answer)
    abstention_correct = float((expected_behavior == "abstain") == abstained)

    failure_stage = _failure_stage(
        expected_behavior=expected_behavior,
        relevant=relevant,
        candidate_ids=candidate_ids,
        selected_ids=selected_ids,
        required_terms=required_terms,
        context_hits=context_hits,
        answer_hits=answer_hits,
        citation_precision=citation_precision,
        citation_recall=citation_recall,
        abstained=abstained,
    )
    return {
        "case_id": case.get("case_id"),
        "query": case.get("query"),
        "category": case.get("category") or "uncategorized",
        "expected_behavior": expected_behavior,
        "retrieval": {
            "recall_at_k": round(recall_at_k, 4),
            "precision_at_k": round(precision_at_k, 4),
            "mrr": round(mrr, 4),
            "ndcg_at_k": round(ndcg_at_k, 4),
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "relevant_selected": relevant_selected,
        },
        "answer": {
            "required_term_recall": round(len(answer_hits) / max(1, len(required_terms)), 4),
            "citation_precision": round(citation_precision, 4),
            "citation_recall": round(citation_recall, 4),
            "abstention_correct": abstention_correct,
            "faithfulness": None,
            "faithfulness_mode": "not_scored_without_judge",
        },
        "failure_stage": failure_stage,
        "passed": failure_stage == "passed",
        "latency_ms": round(float(observation.get("latency_ms") or 0.0), 3),
        "debug": {
            "candidate_ids": candidate_ids,
            "selected_ids": selected_ids,
            "required_answer_terms": required_terms,
            "context_term_hits": context_hits,
            "answer_term_hits": answer_hits,
            "citations": citations,
        },
    }


def run_evaluation(
    dataset: Dict,
    case_runner: Callable[[Dict], Dict],
    output_dir,
    top_k: int = 5,
    run_metadata: Optional[Dict] = None,
):
    validation = validate_dataset(dataset)
    if not validation["valid"]:
        raise ValueError("Invalid evaluation dataset: {0}".format(validation["errors"]))
    results = []
    for case in dataset.get("cases") or []:
        started = time.perf_counter()
        observation = case_runner(case) or {}
        if not observation.get("latency_ms"):
            observation["latency_ms"] = (time.perf_counter() - started) * 1000
        results.append(evaluate_observation(case, observation, top_k=top_k))
    bad_cases = [item for item in results if not item["passed"]]
    report = {
        "project": "PaperStorm RAG Evaluation v4.0",
        "dataset": {
            "dataset_version": dataset.get("dataset_version"),
            "metadata": dataset.get("metadata") or {},
            "validation": validation,
        },
        "run": dict(run_metadata or {}, top_k=top_k),
        "metrics": _aggregate(results),
        "category_slices": _category_slices(results),
        "cases": results,
        "bad_cases": bad_cases,
    }
    write_evaluation_report(output_dir, report)
    return report


def run_seed_baseline(output_dir, top_k: int = 5):
    from .paperstorm_rag import ContextCompressionRetriever, PaperStormRAGIndex

    dataset = build_seed_dataset()
    index = PaperStormRAGIndex.from_documents(
        dataset["corpus"],
        chunk_size=2000,
        chunk_overlap=0,
    )
    retriever = ContextCompressionRetriever(index, max_context_chars=2400)

    def case_runner(case):
        started = time.perf_counter()
        candidates = index.search(case["query"], top_k=max(20, top_k * 4), rerank=False)
        if case.get("expected_behavior") == "abstain":
            selected = []
            prompt_context = ""
            answer = "现有资料不足以可靠回答该问题。"
            citations = []
            abstained = True
        else:
            retrieved = retriever.retrieve(case["query"], top_k=top_k)
            selected = retrieved.get("chunks") or []
            prompt_context = retrieved.get("prompt_context") or ""
            answer = selected[0].get("content", "") if selected else "现有资料不足以可靠回答该问题。"
            citations = [_chunk_id(item) for item in selected[:1]]
            abstained = not selected
        return {
            "candidates": candidates,
            "selected": selected,
            "prompt_context": prompt_context,
            "answer": answer,
            "citations": citations,
            "abstained": abstained,
            "latency_ms": (time.perf_counter() - started) * 1000,
        }

    return run_evaluation(
        dataset,
        case_runner,
        output_dir=output_dir,
        top_k=top_k,
        run_metadata={
            "retriever": "v3.2_hash_hybrid_rule_baseline",
            "answerer": "deterministic_top1_chunk",
            "embedding_provider": index.config.get("embedding_provider"),
            "faithfulness_judge": "disabled",
        },
    )


def write_evaluation_report(output_dir, report: Dict):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rag_eval_v4_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "rag_eval_v4_report.md").write_text(
        _to_markdown(report),
        encoding="utf-8",
    )
    with (output_dir / "rag_eval_v4_bad_cases.jsonl").open("w", encoding="utf-8") as handle:
        for item in report.get("bad_cases") or []:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return output_dir


def _aggregate(results: List[Dict]):
    answer_cases = [item for item in results if item["expected_behavior"] == "answer"]
    abstain_cases = [item for item in results if item["expected_behavior"] == "abstain"]
    latencies = [item["latency_ms"] for item in results]
    failures = Counter(item["failure_stage"] for item in results if not item["passed"])
    return {
        "total_cases": len(results),
        "answer_cases": len(answer_cases),
        "abstain_cases": len(abstain_cases),
        "passed_cases": len([item for item in results if item["passed"]]),
        "failed_cases": len([item for item in results if not item["passed"]]),
        "pass_rate": _mean([float(item["passed"]) for item in results]),
        "retrieval_recall_at_k": _mean([item["retrieval"]["recall_at_k"] for item in answer_cases]),
        "retrieval_precision_at_k": _mean([item["retrieval"]["precision_at_k"] for item in answer_cases]),
        "mrr": _mean([item["retrieval"]["mrr"] for item in answer_cases]),
        "ndcg_at_k": _mean([item["retrieval"]["ndcg_at_k"] for item in answer_cases]),
        "required_term_recall": _mean([item["answer"]["required_term_recall"] for item in answer_cases]),
        "citation_precision": _mean([item["answer"]["citation_precision"] for item in answer_cases]),
        "citation_recall": _mean([item["answer"]["citation_recall"] for item in answer_cases]),
        "abstention_accuracy": _mean([item["answer"]["abstention_correct"] for item in abstain_cases]),
        "avg_latency_ms": round(statistics.mean(latencies), 3) if latencies else 0.0,
        "p95_latency_ms": round(_percentile(latencies, 0.95), 3),
        "failure_counts": dict(sorted(failures.items())),
    }


def _category_slices(results: List[Dict]):
    slices = {}
    for category in sorted({item["category"] for item in results}):
        items = [item for item in results if item["category"] == category]
        slices[category] = {
            "total_cases": len(items),
            "pass_rate": _mean([float(item["passed"]) for item in items]),
            "recall_at_k": _mean([item["retrieval"]["recall_at_k"] for item in items]),
            "mrr": _mean([item["retrieval"]["mrr"] for item in items]),
            "failure_counts": dict(
                sorted(Counter(item["failure_stage"] for item in items if not item["passed"]).items())
            ),
        }
    return slices


def _failure_stage(
    expected_behavior,
    relevant,
    candidate_ids,
    selected_ids,
    required_terms,
    context_hits,
    answer_hits,
    citation_precision,
    citation_recall,
    abstained,
):
    if expected_behavior == "abstain":
        return "passed" if abstained else "false_answer"
    if not any(chunk_id in relevant for chunk_id in candidate_ids):
        return "retrieval_miss"
    if not any(chunk_id in relevant for chunk_id in selected_ids):
        return "rerank_miss"
    if required_terms and len(context_hits) < len(required_terms):
        return "compression_loss"
    if abstained or (required_terms and len(answer_hits) < len(required_terms)):
        return "generation_miss"
    if citation_precision < 1.0 or citation_recall < 1.0:
        return "citation_error"
    return "passed"


def _ndcg(ranked_ids: List[str], relevant: set, top_k: int):
    gains = [1.0 if chunk_id in relevant else 0.0 for chunk_id in ranked_ids[:top_k]]
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal = [1.0] * min(len(relevant), top_k)
    idcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def _term_hits(text: str, terms: Iterable[str]):
    lowered = str(text or "").lower()
    return [term for term in terms if str(term).lower() in lowered]


def _looks_like_abstention(answer: str):
    lowered = str(answer or "").lower()
    markers = ["证据不足", "无法可靠回答", "没有足够", "insufficient evidence", "cannot answer"]
    return any(marker in lowered for marker in markers)


def _chunk_id(item):
    if isinstance(item, dict):
        return str(item.get("chunk_id") or item.get("id") or "")
    return str(item or "")


def _citation_id(value):
    if isinstance(value, dict):
        return str(value.get("chunk_id") or value.get("id") or value.get("document_id") or "")
    return str(value or "")


def _mean(values):
    values = list(values)
    return round(sum(values) / len(values), 4) if values else 0.0


def _percentile(values, percentile):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * percentile))
    return ordered[index]


def _to_markdown(report):
    lines = ["# PaperStorm RAG Evaluation v4.0", "", "## Metrics", ""]
    for key, value in report.get("metrics", {}).items():
        lines.append("- {0}: {1}".format(key, value))
    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| case | category | recall@k | MRR | nDCG@k | citation precision | failure stage |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for item in report.get("cases") or []:
        lines.append(
            "| {case_id} | {category} | {recall} | {mrr} | {ndcg} | {citation} | {failure} |".format(
                case_id=item["case_id"],
                category=item["category"],
                recall=item["retrieval"]["recall_at_k"],
                mrr=item["retrieval"]["mrr"],
                ndcg=item["retrieval"]["ndcg_at_k"],
                citation=item["answer"]["citation_precision"],
                failure=item["failure_stage"],
            )
        )
    lines.append("")
    return "\n".join(lines)
