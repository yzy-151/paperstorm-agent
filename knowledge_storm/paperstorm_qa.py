import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .answer_validation import (
    AnswerValidator,
    Citation,
    ClaimVerdict,
    parse_answer_draft,
)
from .paperstorm_memory import PaperStormMemoryStore
from .paperstorm_sources import load_article_passages


class PaperStormKnowledgeBase:
    def __init__(
        self,
        documents: List[Dict],
        run_dir: Optional[Path] = None,
        retrieval_pipeline=None,
    ):
        self.documents = documents
        self.run_dir = Path(run_dir) if run_dir else None
        self.retrieval_pipeline = retrieval_pipeline
        self.retrieval_meta: Dict = {}

    @classmethod
    def from_run_dir(cls, run_dir):
        run_dir = Path(run_dir)
        documents = []
        for passage in load_article_passages(run_dir):
            index = passage["paragraph_index"]
            paragraph = passage["content"]
            documents.append(
                {
                    "id": "article-{0}".format(index),
                    "chunk_id": "article-{0}".format(index),
                    "title": passage["title"],
                    "content": paragraph,
                    "url": "",
                    "source": "article",
                    "source_type": "article",
                    "score": 0,
                    "metadata": passage,
                }
            )

        raw_results = _read_json(run_dir / "raw_search_results.json", [])
        for index, result in enumerate(raw_results if isinstance(raw_results, list) else [], start=1):
            result_meta = result.get("meta") or {}
            snippets = result.get("snippets") or []
            content = "\n".join(
                [
                    str(result.get("title") or ""),
                    str(result.get("description") or ""),
                    "\n".join(str(snippet) for snippet in snippets),
                ]
            ).strip()
            if content:
                documents.append(
                    {
                        "id": "retrieval-{0}".format(index),
                        "chunk_id": "retrieval-{0}".format(index),
                        "title": result.get("title") or "Retrieved source {0}".format(index),
                        "content": content,
                        "url": result.get("url") or "",
                        "source": "retrieval",
                        "source_type": result_meta.get("source_type") or result.get("source_type") or "retrieval",
                        "score": 0,
                        "metadata": {
                            "result_index": index,
                            "query": result.get("query", ""),
                            "authors": result_meta.get("authors") or result.get("authors") or [],
                            "published": result_meta.get("published") or result.get("published") or "",
                            "original_title": result.get("title") or "",
                        },
                    }
                )
        return cls(documents=documents, run_dir=run_dir)

    def answer_question(
        self,
        question: str,
        memory_store: Optional[PaperStormMemoryStore] = None,
        top_k: int = 3,
        answer_generator: Optional[Callable[[str], object]] = None,
        retrieval_options: Optional[Dict] = None,
        answer_validator: Optional[AnswerValidator] = None,
        answer_parse_retry: Optional[Callable] = None,
    ):
        question = str(question or "").strip()
        if not question:
            raise ValueError("question is required")
        evidence = self.search(
            question, top_k=top_k, **dict(retrieval_options or {})
        )
        memory_context = (
            memory_store.get_context_bundle(query=question, max_items=3)
            if memory_store
            else {}
        )
        answer = _compose_answer(question, evidence, memory_context)
        validation_metadata = None
        citations = [
            _citation_from_doc(index, doc)
            for index, doc in enumerate(evidence, start=1)
        ]
        if answer_generator is not None:
            if answer_validator is None:
                try:
                    generated = str(
                        answer_generator(_kb_answer_prompt(question, evidence)) or ""
                    ).strip()
                    if generated:
                        answer = generated
                except Exception:
                    pass
            else:
                raw_draft = answer_generator(
                    _kb_answer_draft_prompt(question, evidence)
                )
                draft = parse_answer_draft(
                    raw_draft,
                    retry=answer_parse_retry,
                )
                validation = answer_validator.validate(
                    draft, _answer_citation_registry(evidence)
                )
                answer = validation.draft.answer
                citations = _validated_citations(validation.draft.claims)
                validation_metadata = _answer_validation_metadata(validation)
        elif answer_validator is not None:
            raise ValueError("answer_validator requires answer_generator")

        retrieval_metadata = _json_safe_copy(self.retrieval_meta)
        payload = {
            "question": question,
            "answer": answer,
            "citations": citations,
            "grounded": bool(citations) if validation_metadata is not None else bool(evidence),
            "memory_context": memory_context,
            "evidence": evidence,
            "retrieval_stack": self.retrieval_meta.get("stack", ""),
            "retrieval_mode": self.retrieval_meta.get("mode", ""),
            "retrieval_metadata": retrieval_metadata,
        }
        if validation_metadata is not None:
            payload["answer_validation"] = validation_metadata
            payload["failure_type"] = validation_metadata["failure_type"]
            payload["unsupported_claim_count"] = validation_metadata[
                "unsupported_claim_count"
            ]
            retrieval_metadata["answer_validation"] = validation_metadata
        return payload

    def search(self, query: str, top_k: int = 3, **retrieval_options):
        if self.retrieval_pipeline is not None:
            from .retrieval_pipeline import RetrievalRequest

            outcome = self.retrieval_pipeline.search(
                RetrievalRequest(query=query, top_k=top_k, **retrieval_options)
            )
            self.retrieval_meta = _retrieval_metadata(
                outcome, stack="retrieval_pipeline"
            )
            return [_rag_chunk_to_doc(item) for item in outcome["results"]]
        if self.run_dir:
            from .retrieval_runtime import search_runtime_index

            outcome = search_runtime_index(
                self.run_dir, query, top_k=top_k, **retrieval_options
            )
            self.retrieval_meta = _retrieval_metadata(
                outcome, stack=outcome.get("stack", "")
            )
            return [_rag_chunk_to_doc(item) for item in outcome.get("results") or []]
        raise RuntimeError(
            "PaperStormKnowledgeBase requires a run_dir or RetrievalPipeline"
        )


def write_qa_artifact(run_dir, answer) -> Path:
    run_dir = Path(run_dir)
    path = run_dir / "qa_answer.json"
    path.write_text(json.dumps(answer, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _compose_answer(question: str, evidence: List[Dict], memory_context: Dict):
    if not evidence:
        return "没有找到足够证据回答该问题。"
    sentences = []
    for index, doc in enumerate(evidence, start=1):
        content = _first_sentence(
            doc.get("expanded_content") or doc.get("content", "")
        )
        if content:
            sentences.append("{0}[{1}]".format(content, index))
    memory_hint = _memory_hint(memory_context)
    if memory_hint:
        sentences.insert(0, memory_hint)
    return " ".join(sentences)


def _memory_hint(memory_context: Dict):
    for layer in ("semantic", "episodic", "working"):
        records = memory_context.get(layer) or []
        if records:
            return records[0].get("content", "")
    return ""


def _citation_from_doc(index: int, doc: Dict):
    metadata = doc.get("metadata") or {}
    citation = {
        "id": index,
        "title": doc.get("title") or "",
        "url": doc.get("url") or "",
        "source": doc.get("source") or "",
        "source_type": doc.get("source_type") or doc.get("source") or "",
        "document_id": doc.get("id") or "",
        "chunk_id": doc.get("chunk_id") or doc.get("id") or "",
        "score": doc.get("score", 0),
        "authors": metadata.get("authors") or doc.get("authors") or [],
        "published": metadata.get("published") or doc.get("published") or "",
    }
    if citation["source_type"] == "article":
        citation.update(
            {
                "article_anchor": metadata.get("article_anchor") or "",
                "paragraph_index": metadata.get("paragraph_index"),
                "section": metadata.get("section") or "",
                "original_sources": metadata.get("original_sources") or [],
            }
        )
    return citation


def _kb_answer_prompt(question: str, evidence: List[Dict]) -> str:
    lines = [
        "你是论文/文档知识库问答助手。请用中文回答问题，并基于给出的证据组织答案；",
        "引用证据时保留编号如 [1]、[2]。不要编造证据之外的内容；如果证据不足，直接说'现有资料不足以回答'。",
        "问题：{0}".format(question),
        "证据：",
    ]
    for index, doc in enumerate((evidence or [])[:6], start=1):
        child_content = str(doc.get("content") or "")
        parent_context = str(doc.get("parent_context") or "")
        if child_content:
            parent_context = parent_context.replace(child_content, "")
        evidence_text = child_content
        if parent_context:
            evidence_text += "\n父级补充：" + parent_context[:260]
        lines.append(
            "[{0}] {1}：{2}".format(
                index,
                str(doc.get("title") or doc.get("id") or "")[:60],
                evidence_text,
            )
        )
    lines.append("回答：")
    return "\n".join(lines)


def _kb_answer_draft_prompt(question: str, evidence: List[Dict]) -> str:
    registry = _answer_citation_registry(evidence)
    evidence_payload = [item.to_dict() for item in registry.values()]
    contract = {
        "answer": "final answer text",
        "answer_type": "extractive|abstractive|boolean|list|comparison|factoid|refusal",
        "claims": [
            {
                "claim_id": "stable unique ID",
                "text": "one atomic claim",
                "citations": [
                    {
                        "citation_id": "ID selected from evidence",
                        "source_id": "copy from evidence",
                        "span": "copy exact evidence span",
                        "title": "copy exact title",
                        "authors": ["copy exact authors"],
                        "page": "copy page or null",
                        "section": "copy section or null",
                        "url": "copy URL or null",
                    }
                ],
            }
        ],
        "uncertainty": "number from 0 to 1",
        "refusal": False,
        "abstain_reason": None,
    }
    return "\n".join(
        [
            "Answer the question only from the supplied evidence.",
            "Return one JSON object matching the schema exactly; no markdown fence.",
            "Use atomic claims. Cite only supplied citation_id values.",
            "If evidence is insufficient, set answer_type='refusal', refusal=true, "
            "provide abstain_reason, and return no unsupported claims.",
            "Question: {0}".format(question),
            "Schema: {0}".format(json.dumps(contract, ensure_ascii=False)),
            "Evidence: {0}".format(
                json.dumps(evidence_payload, ensure_ascii=False)
            ),
        ]
    )


def _answer_citation_registry(evidence: List[Dict]):
    registry = {}
    for index, doc in enumerate(evidence or [], start=1):
        metadata = doc.get("metadata") or {}
        span = str(doc.get("content") or doc.get("expanded_content") or "").strip()
        if not span:
            continue
        citation_id = str(index)
        registry[citation_id] = Citation(
            citation_id=citation_id,
            source_id=str(
                doc.get("chunk_id") or doc.get("id") or "evidence-{0}".format(index)
            ),
            span=span,
            title=str(
                metadata.get("original_title")
                or doc.get("title")
                or doc.get("id")
                or "Untitled source"
            ),
            authors=tuple(
                str(author)
                for author in (
                    metadata.get("authors") or doc.get("authors") or []
                )
                if str(author).strip()
            ),
            page=metadata.get("page"),
            section=metadata.get("section") or None,
            url=doc.get("url") or None,
        )
    return registry


def _validated_citations(claims):
    citations = []
    seen = set()
    for claim in claims:
        for citation in claim.citations:
            if citation.citation_id in seen:
                continue
            serialized = citation.to_dict()
            serialized["id"] = (
                int(citation.citation_id)
                if citation.citation_id.isdigit()
                else citation.citation_id
            )
            serialized["document_id"] = citation.source_id
            serialized["chunk_id"] = citation.source_id
            citations.append(serialized)
            seen.add(citation.citation_id)
    return citations


def _answer_validation_metadata(validation):
    latest = {}
    for assessment in validation.assessments:
        latest[assessment.claim_id] = assessment
    counts = {
        verdict.value: sum(
            item.verdict is verdict for item in latest.values()
        )
        for verdict in ClaimVerdict
    }
    if counts[ClaimVerdict.UNSUPPORTED.value]:
        failure_type = "unsupported_claims"
    elif counts[ClaimVerdict.CONTRADICTED.value]:
        failure_type = "contradicted_claims"
    elif counts[ClaimVerdict.PARTIAL.value]:
        failure_type = "partial_support"
    elif validation.draft.refusal:
        failure_type = "abstained"
    else:
        failure_type = ""
    serialized = validation.to_dict()
    serialized.update(
        {
            "failure_type": failure_type,
            "entailed_claim_count": counts[ClaimVerdict.ENTAILED.value],
            "partial_claim_count": counts[ClaimVerdict.PARTIAL.value],
            "contradicted_claim_count": counts[
                ClaimVerdict.CONTRADICTED.value
            ],
            "unsupported_claim_count": counts[
                ClaimVerdict.UNSUPPORTED.value
            ],
            "repair_attempt_count": len(validation.repaired_claim_ids),
            "refusal": validation.draft.refusal,
        }
    )
    return serialized


def _with_score(doc: Dict, score: int):
    enriched = dict(doc)
    enriched["score"] = score
    enriched.setdefault("source_type", enriched.get("source", ""))
    enriched.setdefault("chunk_id", enriched.get("id", ""))
    enriched.setdefault("metadata", {})
    return enriched


def _rag_chunk_to_doc(chunk: Dict):
    return {
        "id": chunk.get("chunk_id") or "",
        "chunk_id": chunk.get("chunk_id") or "",
        "parent_id": chunk.get("parent_id") or "",
        "title": chunk.get("title") or "",
        "content": chunk.get("content") or "",
        "expanded_content": chunk.get("expanded_content") or chunk.get("content") or "",
        "parent_context": chunk.get("parent_context") or "",
        "url": chunk.get("url") or "",
        "source": chunk.get("source_type") or "",
        "source_type": chunk.get("source_type") or "",
        "score": chunk.get("score", 0),
        "metadata": chunk.get("metadata") or {},
        "authors": (chunk.get("metadata") or {}).get("authors") or chunk.get("authors") or [],
        "published": (chunk.get("metadata") or {}).get("published") or chunk.get("published") or "",
        "lexical_score": chunk.get("lexical_score", 0),
        "vector_score": chunk.get("vector_score", 0),
        "hybrid_score": chunk.get("hybrid_score", 0),
        "rerank_score": chunk.get("rerank_score", 0),
        "bm25_score": chunk.get("bm25_score", 0),
        "dense_score": chunk.get("dense_score", 0),
        "rrf_score": chunk.get("rrf_score", 0),
        "retrieval_mode": chunk.get("retrieval_mode", ""),
        "final_rank": chunk.get("final_rank", 0),
    }


def _json_safe_copy(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe_copy(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_copy(item) for item in value]
    return str(value)


def _retrieval_metadata(outcome, stack):
    metadata = {
        key: value for key, value in dict(outcome or {}).items() if key != "results"
    }
    metadata["stack"] = stack
    models = dict(metadata.get("models") or {})
    if not models and metadata.get("embedding"):
        models["embedding"] = metadata["embedding"]
    metadata["models"] = models
    metadata.setdefault("mode", "")
    metadata.setdefault("search_plan", {})
    metadata.setdefault("stages", [])
    metadata["embedding"] = models.get(
        "embedding", metadata.get("embedding", "")
    )
    return _json_safe_copy(metadata)


def _read_first_existing(paths):
    for path in paths:
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
    return ""


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _split_paragraphs(text: str):
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()]
    return [item for item in paragraphs if not item.startswith("#")]


def _first_sentence(text: str):
    text = " ".join(str(text or "").split())
    if not text:
        return ""
    match = re.search(r"(.+?[。.!?])\s", text + " ")
    if match:
        return match.group(1)
    return text[:240]


def _tokenize(text: str):
    return set(re.findall(r"[a-zA-Z0-9_\-]+|[\u4e00-\u9fff]+", str(text).lower()))


def _contains_cjk(text: str):
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _cjk_overlap(query: str, text: str):
    query_chars = set(re.findall(r"[\u4e00-\u9fff]", query))
    text_chars = set(re.findall(r"[\u4e00-\u9fff]", text))
    return len(query_chars & text_chars)
