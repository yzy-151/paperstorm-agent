import hashlib
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List


def extract_pdf_pages(path) -> List[Dict]:
    """Extract page text while retaining stable, auditable page metadata."""
    from pypdf import PdfReader

    path = Path(path)
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    reader = PdfReader(str(path))
    pages = []
    for index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").replace("\x00", "").strip()
        if text:
            pages.append({"page_number": index, "text": text})
    return pages


def chunk_pdf_pages(
    pages: Iterable[Dict],
    document_id: str,
    title: str,
    chunk_tokens: int = 320,
    overlap_tokens: int = 48,
    strategy: str = "structured",
) -> List[Dict]:
    if strategy not in {"ordinary", "structured", "contextual"}:
        raise ValueError("unsupported chunk strategy: {0}".format(strategy))
    if chunk_tokens <= 0 or overlap_tokens < 0 or overlap_tokens >= chunk_tokens:
        raise ValueError("chunk token and overlap settings are invalid")
    chunks = []
    for page in pages:
        page_number = int(page.get("page_number") or 0)
        text = str(page.get("text") or "").strip()
        if not text:
            continue
        heading = _find_heading(text) if strategy != "ordinary" else ""
        parent_key = heading or "page-{0}".format(page_number)
        parent_id = "{0}::section::{1}".format(document_id, _slug(parent_key))
        units = _token_units(text)
        step = chunk_tokens - overlap_tokens
        for chunk_index, start in enumerate(range(0, len(units), step), start=1):
            window = units[start : start + chunk_tokens]
            if not window:
                continue
            content = _join_units(window).strip()
            context = ""
            if strategy == "contextual":
                context = "Document: {0}\nSection: {1}\nPage: {2}\n".format(
                    title, heading or "unknown", page_number
                )
            raw_id = "{0}|{1}|{2}|{3}".format(document_id, page_number, chunk_index, content)
            chunks.append(
                {
                    "chunk_id": "{0}::p{1}::c{2}::{3}".format(
                        document_id,
                        page_number,
                        chunk_index,
                        hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:10],
                    ),
                    "document_id": document_id,
                    "parent_id": parent_id,
                    "title": title,
                    "content": content,
                    "retrieval_content": context + content,
                    "metadata": {
                        "page_number": page_number,
                        "heading": heading,
                        "chunk_index": chunk_index,
                        "chunk_strategy": strategy,
                        "token_count": len(window),
                    },
                }
            )
            if start + chunk_tokens >= len(units):
                break
    return chunks


def _find_heading(text: str) -> str:
    for line in text.splitlines()[:8]:
        candidate = re.sub(r"\s+", " ", line).strip()
        if not candidate or len(candidate) > 100:
            continue
        if re.match(r"^(?:\d+(?:\.\d+)*|[IVX]+)[\s.)、:-]+\S+", candidate, re.I):
            return candidate
        if re.match(r"^(?:abstract|introduction|conclusion|摘要|引言|结论)\b", candidate, re.I):
            return candidate
    return ""


def _token_units(text: str):
    return re.findall(r"[A-Za-z0-9]+(?:[-./][A-Za-z0-9]+)*|[\u3400-\u9fff]|[^\s]", text)


def _join_units(units):
    output = ""
    previous_latin = False
    for unit in units:
        latin = bool(re.match(r"^[A-Za-z0-9]", unit))
        if output and latin and previous_latin:
            output += " "
        output += unit
        previous_latin = latin
    return output


def _slug(value: str):
    cleaned = re.sub(r"[^A-Za-z0-9\u3400-\u9fff]+", "-", value).strip("-")
    return cleaned[:60] or "unknown"
