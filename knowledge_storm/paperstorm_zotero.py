import hashlib
import re
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .paperstorm_document_v41 import chunk_pdf_pages, extract_pdf_pages


def resolve_attachment_path(zotero_root, attachment_key: str, stored_path: str) -> Path:
    root = Path(zotero_root)
    stored_path = str(stored_path or "")
    if stored_path.startswith("storage:"):
        return root / "storage" / attachment_key / stored_path.split(":", 1)[1]
    if stored_path.startswith("attachments:"):
        return root / stored_path.split(":", 1)[1]
    return Path(stored_path)


def deduplicate_papers(papers: Iterable[Dict]) -> List[Dict]:
    output = []
    seen = set()
    for paper in papers:
        title_key = re.sub(r"\s+", " ", str(paper.get("title") or "")).strip().lower()
        fallback = str(Path(str(paper.get("path") or "")).name).lower()
        key = title_key or fallback
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(dict(paper))
    return output


def discover_zotero_papers(
    zotero_root,
    query_terms: Optional[Iterable[str]] = None,
    max_papers: Optional[int] = None,
) -> List[Dict]:
    """Read Zotero in read-only mode and return deduplicated local PDF records."""
    root = Path(zotero_root)
    database = root / "zotero.sqlite"
    if not database.exists():
        raise FileNotFoundError("Zotero database not found: {0}".format(database))
    uri = "file:{0}?mode=ro".format(database.as_posix())
    connection = sqlite3.connect(uri, uri=True)
    try:
        rows = connection.execute(
            """
            SELECT attachment_item.key, attachment.path,
                   COALESCE(title_value.value, '') AS title
            FROM itemAttachments AS attachment
            JOIN items AS attachment_item ON attachment_item.itemID = attachment.itemID
            LEFT JOIN itemData AS title_data
              ON title_data.itemID = COALESCE(attachment.parentItemID, attachment.itemID)
             AND title_data.fieldID = (SELECT fieldID FROM fields WHERE fieldName = 'title')
            LEFT JOIN itemDataValues AS title_value ON title_value.valueID = title_data.valueID
            WHERE lower(COALESCE(attachment.contentType, '')) = 'application/pdf'
            ORDER BY title_value.value, attachment_item.key
            """
        ).fetchall()
    finally:
        connection.close()
    terms = [str(term).strip().lower() for term in (query_terms or []) if str(term).strip()]
    papers = []
    for attachment_key, stored_path, title in rows:
        path = resolve_attachment_path(root, attachment_key, stored_path)
        haystack = "{0} {1}".format(title, path.name).lower()
        if terms and not any(term in haystack for term in terms):
            continue
        if not path.exists() or path.suffix.lower() != ".pdf":
            continue
        document_id = "paper-{0}".format(
            hashlib.sha256("{0}|{1}".format(title, path.stat().st_size).encode("utf-8")).hexdigest()[:16]
        )
        papers.append(
            {
                "document_id": document_id,
                "title": str(title or path.stem),
                "path": str(path),
                "page_count": None,
            }
        )
    papers = deduplicate_papers(papers)
    return papers[:max_papers] if max_papers else papers


def load_zotero_chunks(
    zotero_root,
    query_terms: Optional[Iterable[str]] = None,
    max_papers: int = 8,
    max_pages: int = 20,
    chunk_tokens: int = 320,
    overlap_tokens: int = 48,
    strategy: str = "contextual",
) -> List[Dict]:
    papers = discover_zotero_papers(zotero_root, query_terms=query_terms, max_papers=max_papers)
    chunks = []
    for paper in papers:
        pages = extract_pdf_pages(paper["path"])[:max_pages]
        chunks.extend(
            chunk_pdf_pages(
                pages,
                document_id=paper["document_id"],
                title=paper["title"],
                chunk_tokens=chunk_tokens,
                overlap_tokens=overlap_tokens,
                strategy=strategy,
            )
        )
    return chunks


def build_weak_paper_dataset(
    chunks: Iterable[Dict],
    source_label: str = "zotero_weak_supervision",
    max_cases: int = 60,
) -> Dict:
    """Build retrieval labels from page/section provenance, pending human review."""
    chunks = list(chunks)
    corpus = []
    cases = []
    seen_sections = set()
    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        metadata = dict(chunk.get("metadata") or {})
        corpus.append(
            {
                "document_id": chunk_id,
                "chunk_ids": [chunk_id],
                "title": chunk.get("title") or chunk.get("document_id") or "paper",
                "text": chunk.get("content") or "",
                "source_type": "local_pdf_chunk",
                "metadata": {
                    "category": "real_paper",
                    "page_number": metadata.get("page_number"),
                    "heading": metadata.get("heading"),
                    "context": chunk.get("retrieval_content") or chunk.get("content") or "",
                },
            }
        )
        section_key = (chunk.get("document_id"), metadata.get("heading") or metadata.get("page_number"))
        if section_key in seen_sections or len(cases) >= max_cases:
            continue
        seen_sections.add(section_key)
        heading = metadata.get("heading") or "第 {0} 页".format(metadata.get("page_number") or "未知")
        title = chunk.get("title") or "该论文"
        required_terms = _required_terms(chunk.get("content") or "")
        cases.append(
            {
                "case_id": "paper-case-{0}".format(hashlib.sha1(chunk_id.encode("utf-8")).hexdigest()[:12]),
                "query": "论文《{0}》的“{1}”部分主要讨论什么？".format(title, heading),
                "relevant_chunk_ids": [chunk_id],
                "expected_behavior": "answer",
                "required_answer_terms": required_terms,
                "allowed_citation_ids": [chunk_id],
                "category": "real_paper",
                "metadata": {
                    "label_method": "section_provenance_weak_supervision",
                    "review_status": "needs_domain_review",
                },
            }
        )
    return {
        "dataset_version": "paperstorm-v4.1-local-paper-weak-v1",
        "metadata": {
            "provenance": source_label,
            "domain_review_required": True,
            "contains_private_paths": False,
            "corpus_chunk_count": len(corpus),
            "case_count": len(cases),
            "label_limitations": "Section provenance is a weak retrieval label, not expert QA annotation.",
        },
        "corpus": corpus,
        "cases": cases,
    }


def _required_terms(text: str):
    latin = [
        word.lower()
        for word in re.findall(r"[A-Za-z][A-Za-z0-9-]{5,}", text)
        if word.lower() not in {"abstract", "introduction", "figure", "results", "method"}
    ]
    chinese = re.findall(r"[\u4e00-\u9fff]{2,6}", text)
    terms = []
    for term in latin + chinese:
        if term not in terms:
            terms.append(term)
        if len(terms) >= 2:
            break
    return terms
