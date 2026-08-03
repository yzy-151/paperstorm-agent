import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .paperstorm_memory import PaperStormMemoryStore


class PaperStormKnowledgeBase:
    def __init__(self, documents: List[Dict], run_dir: Optional[Path] = None):
        self.documents = documents
        self.run_dir = Path(run_dir) if run_dir else None
        self.retrieval_meta: Dict = {}

    @classmethod
    def from_run_dir(cls, run_dir):
        run_dir = Path(run_dir)
        documents = []
        article = _read_first_existing(
            [
                run_dir / "storm_gen_article_polished.txt",
                run_dir / "storm_gen_article.txt",
            ]
        )
        if article:
            for index, paragraph in enumerate(_split_paragraphs(article), start=1):
                documents.append(
                    {
                        "id": "article-{0}".format(index),
                        "chunk_id": "article-{0}".format(index),
                        "title": "Generated article paragraph {0}".format(index),
                        "content": paragraph,
                        "url": str(run_dir / "storm_gen_article_polished.txt"),
                        "source": "article",
                        "source_type": "article",
                        "score": 0,
                        "metadata": {"paragraph_index": index},
                    }
                )

        raw_results = _read_json(run_dir / "raw_search_results.json", [])
        for index, result in enumerate(raw_results if isinstance(raw_results, list) else [], start=1):
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
                        "source_type": result.get("source_type") or "retrieval",
                        "score": 0,
                        "metadata": {
                            "result_index": index,
                            "query": result.get("query", ""),
                        },
                    }
                )
        return cls(documents=documents, run_dir=run_dir)

    def answer_question(
        self,
        question: str,
        memory_store: Optional[PaperStormMemoryStore] = None,
        top_k: int = 3,
        answer_generator: Optional[Callable[[str], str]] = None,
    ):
        question = str(question or "").strip()
        if not question:
            raise ValueError("question is required")
        evidence = self.search(question, top_k=top_k)
        memory_context = (
            memory_store.get_context_bundle(query=question, max_items=3)
            if memory_store
            else {}
        )
        answer = _compose_answer(question, evidence, memory_context)
        if answer_generator is not None:
            try:
                generated = str(
                    answer_generator(_kb_answer_prompt(question, evidence)) or ""
                ).strip()
                if generated:
                    answer = generated
            except Exception:
                pass
        return {
            "question": question,
            "answer": answer,
            "citations": [_citation_from_doc(index, doc) for index, doc in enumerate(evidence, start=1)],
            "grounded": bool(evidence),
            "memory_context": memory_context,
            "evidence": evidence,
            "retrieval_stack": self.retrieval_meta.get("stack", ""),
            "retrieval_mode": self.retrieval_meta.get("mode", ""),
        }

    def search(self, query: str, top_k: int = 3):
        if self.run_dir:
            try:
                from .paperstorm_retrieval_runtime import search_runtime_index

                outcome = search_runtime_index(self.run_dir, query, top_k=top_k)
                self.retrieval_meta = {
                    "stack": outcome.get("stack", ""),
                    "mode": outcome.get("mode", ""),
                    "embedding": outcome.get("embedding", ""),
                }
                if outcome.get("results"):
                    return [_rag_chunk_to_doc(item) for item in outcome["results"]]
            except Exception:
                pass
        self.retrieval_meta = {"stack": "legacy_fallback", "mode": "set_overlap"}
        terms = _tokenize(query)
        scored = []
        for index, doc in enumerate(self.documents):
            text = "{0}\n{1}".format(doc.get("title", ""), doc.get("content", ""))
            score = len(terms & _tokenize(text))
            if score == 0 and _contains_cjk(query):
                score = _cjk_overlap(query, text)
            scored.append((score, index, doc))
        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        selected = [_with_score(doc, score) for score, _, doc in scored if score > 0]
        if not selected:
            selected = [_with_score(doc, 0) for doc in self.documents[:top_k]]
        return selected[:top_k]


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
        content = _first_sentence(doc.get("content", ""))
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
    return {
        "id": index,
        "title": doc.get("title") or "",
        "url": doc.get("url") or "",
        "source": doc.get("source") or "",
        "source_type": doc.get("source_type") or doc.get("source") or "",
        "document_id": doc.get("id") or "",
        "chunk_id": doc.get("chunk_id") or doc.get("id") or "",
        "score": doc.get("score", 0),
    }


def _kb_answer_prompt(question: str, evidence: List[Dict]) -> str:
    lines = [
        "你是论文/文档知识库问答助手。请用中文回答问题，并基于给出的证据组织答案；",
        "引用证据时保留编号如 [1]、[2]。不要编造证据之外的内容；如果证据不足，直接说'现有资料不足以回答'。",
        "问题：{0}".format(question),
        "证据：",
    ]
    for index, doc in enumerate((evidence or [])[:6], start=1):
        lines.append(
            "[{0}] {1}：{2}".format(
                index,
                str(doc.get("title") or doc.get("id") or "")[:60],
                str(doc.get("content") or "")[:260],
            )
        )
    lines.append("回答：")
    return "\n".join(lines)


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
        "title": chunk.get("title") or "",
        "content": chunk.get("content") or "",
        "url": chunk.get("url") or "",
        "source": chunk.get("source_type") or "",
        "source_type": chunk.get("source_type") or "",
        "score": chunk.get("score", 0),
        "metadata": chunk.get("metadata") or {},
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
