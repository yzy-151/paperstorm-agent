import hashlib
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List


_HEADING_PATTERN = re.compile(
    r"^(?:(?:\d+(?:\.\d+)*)|(?:[IVXLC]+)|(?:第[一二三四五六七八九十百0-9]+[章节]))"
    r"(?:[\s.)、:：-]+|(?=[\u3400-\u9fff]))\S+",
    re.I,
)
_NAMED_HEADING_PATTERN = re.compile(
    r"^(?:abstract|introduction|background|method(?:s|ology)?|results?|discussion|"
    r"conclusion|references|摘要|引言|背景|方法|结果|讨论|结论|参考文献)\s*$",
    re.I,
)
_CHINESE_NUMBERED_HEADING_PATTERN = re.compile(
    r"^(?:[（(][一二三四五六七八九十百0-9]+[）)]|[一二三四五六七八九十百]+[、.．])\s*\S+"
)


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100):
    """Split text into deterministic overlapping character windows."""
    text = str(text or "")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    chunks = []
    step = chunk_size - chunk_overlap
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
    return chunks


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


def ingest_document(
    document_id: str,
    pages: Iterable[Dict],
    title: str = "",
    chunk_tokens: int = 320,
    overlap_tokens: int = 48,
) -> List[Dict]:
    """Build a stable hierarchy while keeping tables and formulas atomic."""
    if chunk_tokens <= 0 or overlap_tokens < 0 or overlap_tokens >= chunk_tokens:
        raise ValueError("chunk token and overlap settings are invalid")
    document_id = str(document_id or "").strip()
    if not document_id:
        raise ValueError("document_id is required")
    normalized_pages = []
    for page in pages or []:
        text = str(page.get("text") or "").replace("\x00", "").strip()
        if text:
            normalized_pages.append(
                {"page_number": int(page.get("page_number") or 0), "text": text}
            )
    document_title = str(title or document_id)
    document_node_id = _stable_node_id(document_id, "document", document_title)
    document_content = "\n\n".join(page["text"] for page in normalized_pages)
    document_node = _node(
        node_id=document_node_id,
        node_type="document",
        document_id=document_id,
        parent_id=None,
        title=document_title,
        content=document_content,
        metadata={
            "page_start": normalized_pages[0]["page_number"] if normalized_pages else 0,
            "page_end": normalized_pages[-1]["page_number"] if normalized_pages else 0,
        },
    )

    sections = []
    current = None
    preamble = []
    section_ordinal = 0
    for page in normalized_pages:
        page_number = page["page_number"]
        lines = page["text"].splitlines()
        content_lines = []
        for line in lines:
            candidate = re.sub(r"\s+", " ", line).strip()
            if _is_heading(candidate):
                if content_lines:
                    _append_section_content(current, preamble, page_number, content_lines)
                    content_lines = []
                section_ordinal += 1
                current = {
                    "ordinal": section_ordinal,
                    "title": candidate,
                    "page_start": page_number,
                    "page_end": page_number,
                    "page_lines": [],
                }
                sections.append(current)
            else:
                content_lines.append(line)
        if content_lines:
            _append_section_content(current, preamble, page_number, content_lines)

    nodes = [document_node]
    if preamble:
        nodes.extend(
            _build_child_nodes(
                document_id,
                document_node_id,
                document_title,
                preamble,
                chunk_tokens,
                overlap_tokens,
            )
        )
    for section in sections:
        section_id = _stable_node_id(
            document_id,
            "section",
            str(section["ordinal"]),
            section["title"],
        )
        child_nodes = _build_child_nodes(
            document_id,
            section_id,
            section["title"],
            section["page_lines"],
            chunk_tokens,
            overlap_tokens,
        )
        section_content = "\n".join(
            node["content"] for node in child_nodes if node["content"]
        )
        nodes.append(
            _node(
                node_id=section_id,
                node_type="section",
                document_id=document_id,
                parent_id=document_node_id,
                title=section["title"],
                content=section_content,
                metadata={
                    "page_start": section["page_start"],
                    "page_end": section["page_end"],
                    "section_index": section["ordinal"],
                },
            )
        )
        nodes.extend(child_nodes)
    return nodes


def _append_section_content(current, preamble, page_number, lines):
    target = current["page_lines"] if current is not None else preamble
    target.append({"page_number": page_number, "text": "\n".join(lines).strip()})
    if current is not None:
        current["page_end"] = page_number


def _build_child_nodes(
    document_id, parent_id, parent_title, page_parts, chunk_tokens, overlap_tokens
):
    nodes = []
    child_ordinal = 0
    for part in page_parts:
        page_number = part["page_number"]
        for node_type, content in _atomic_blocks(part["text"]):
            contents = [content]
            if node_type == "passage":
                contents = _chunk_units(content, chunk_tokens, overlap_tokens)
            for child_content in contents:
                if not child_content.strip():
                    continue
                child_ordinal += 1
                node_id = _stable_node_id(
                    document_id,
                    parent_id,
                    node_type,
                    str(page_number),
                    str(child_ordinal),
                    child_content,
                )
                metadata = {
                    "page_number": page_number,
                    "child_index": child_ordinal,
                    "token_count": len(_token_units(child_content)),
                }
                node = _node(
                    node_id=node_id,
                    node_type=node_type,
                    document_id=document_id,
                    parent_id=parent_id,
                    title=parent_title,
                    content=child_content,
                    metadata=metadata,
                )
                node["chunk_id"] = node_id
                nodes.append(node)
    return nodes


def _atomic_blocks(text):
    lines = str(text or "").splitlines()
    output = []
    ordinary = []

    def flush_ordinary():
        if ordinary:
            value = "\n".join(ordinary).strip()
            if value:
                output.extend(_split_inline_formulas(value))
            ordinary[:] = []

    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if _is_table_line(stripped):
            flush_ordinary()
            table = []
            while index < len(lines) and _is_table_line(lines[index].strip()):
                table.append(lines[index].strip())
                index += 1
            output.append(("table", "\n".join(table)))
            continue
        delimiter = "$$" if stripped.startswith("$$") else "\\[" if stripped.startswith("\\[") else ""
        if delimiter:
            flush_ordinary()
            closing = "$$" if delimiter == "$$" else "\\]"
            formula = [lines[index].strip()]
            closed = closing in formula[0][len(delimiter) :]
            while not closed and index + 1 < len(lines):
                index += 1
                formula.append(lines[index].strip())
                closed = closing in formula[-1]
            output.append(("formula", "\n".join(formula).strip()))
        else:
            ordinary.append(lines[index])
        index += 1
    flush_ordinary()
    return output


def _split_inline_formulas(text):
    output = []
    cursor = 0
    for match in re.finditer(r"(?<!\$)\$(?!\$).+?(?<!\$)\$(?!\$)", text, re.S):
        before = text[cursor : match.start()].strip()
        if before:
            output.append(("passage", before))
        output.append(("formula", match.group(0)))
        cursor = match.end()
    after = text[cursor:].strip()
    if after:
        output.append(("passage", after))
    return output


def _chunk_units(text, chunk_tokens, overlap_tokens):
    units = _token_units(text)
    step = chunk_tokens - overlap_tokens
    chunks = []
    for start in range(0, len(units), step):
        window = units[start : start + chunk_tokens]
        if window:
            chunks.append(_join_units(window).strip())
        if start + chunk_tokens >= len(units):
            break
    return chunks


def _node(node_id, node_type, document_id, parent_id, title, content, metadata):
    return {
        "node_id": node_id,
        "node_type": node_type,
        "document_id": document_id,
        "parent_id": parent_id,
        "title": title,
        "content": content,
        "retrieval_content": content,
        "metadata": metadata,
    }


def _stable_node_id(*parts):
    lineage = "\x1f".join(str(part or "") for part in parts)
    digest = hashlib.sha256(lineage.encode("utf-8")).hexdigest()[:20]
    return "{0}::{1}::{2}".format(_slug(parts[0]), _slug(parts[1]), digest)


def _is_heading(candidate):
    if not candidate or len(candidate) > 120:
        return False
    return bool(
        _HEADING_PATTERN.match(candidate)
        or _CHINESE_NUMBERED_HEADING_PATTERN.match(candidate)
        or _NAMED_HEADING_PATTERN.match(candidate)
    )


def _is_table_line(line):
    return bool(line.startswith("|") and line.endswith("|") and line.count("|") >= 2)


def _find_heading(text: str) -> str:
    for line in text.splitlines():
        candidate = re.sub(r"\s+", " ", line).strip()
        if _is_heading(candidate):
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
