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

DEFAULT_MAX_PAGES = 5000
DEFAULT_MAX_PAGE_CHARS = 2_000_000


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
    max_pages: int = DEFAULT_MAX_PAGES,
    max_page_chars: int = DEFAULT_MAX_PAGE_CHARS,
) -> List[Dict]:
    """Build a stable hierarchy while keeping tables and formulas atomic."""
    if chunk_tokens <= 0 or overlap_tokens < 0 or overlap_tokens >= chunk_tokens:
        raise ValueError("chunk token and overlap settings are invalid")
    document_id = str(document_id or "").strip()
    if not document_id:
        raise ValueError("document_id is required")
    if max_pages <= 0 or max_page_chars <= 0:
        raise ValueError("ingestion limits must be positive")
    normalized_pages = []
    for page_index, page in enumerate(pages or [], start=1):
        if page_index > max_pages:
            raise ValueError("document page count exceeds max_pages")
        text = str(page.get("text") or "").replace("\x00", "").strip()
        if len(text) > max_page_chars:
            raise ValueError("document page character count exceeds max_page_chars")
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
    formula_state = None
    for page in normalized_pages:
        page_number = page["page_number"]
        lines = page["text"].splitlines()
        content_lines = []
        for line in lines:
            candidate = re.sub(r"\s+", " ", line).strip()
            starts_inside_formula = formula_state is not None
            formula_state = _display_formula_state_after(line, formula_state)
            if not starts_inside_formula and _is_heading(candidate):
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
            part["text"] for part in section["page_lines"] if part["text"]
        ).strip()
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
    for block in _atomic_blocks(page_parts):
        node_type = block["node_type"]
        contents = [block["content"]]
        if node_type == "passage":
            contents = _chunk_units(block["content"], chunk_tokens, overlap_tokens)
        for child_content in contents:
            if not child_content.strip():
                continue
            child_ordinal += 1
            node_id = _stable_node_id(
                document_id,
                parent_id,
                node_type,
                str(block["page_start"]),
                str(block["page_end"]),
                str(child_ordinal),
                child_content,
            )
            metadata = {
                "page_number": block["page_start"],
                "page_start": block["page_start"],
                "page_end": block["page_end"],
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


def _atomic_blocks(page_parts):
    source, page_map = _section_stream(page_parts)
    table_ranges = {
        match.start(): match.end()
        for match in re.finditer(
            r"(?m)(?:^[ \t]*\|[^\n]*\|[ \t]*(?:\n|$))+",
            source,
        )
    }
    output = []
    passage_start = 0
    position = 0

    def emit(node_type, start, end):
        if node_type == "passage":
            segment_start = start
            while segment_start < end:
                page_number = page_map[segment_start] if page_map else 0
                segment_end = segment_start + 1
                while segment_end < end and page_map[segment_end] == page_number:
                    segment_end += 1
                append_block(node_type, segment_start, segment_end)
                segment_start = segment_end
            return
        append_block(node_type, start, end)

    def append_block(node_type, start, end):
        value = source[start:end].strip()
        if not value:
            return
        if node_type == "table":
            value = "\n".join(line.strip() for line in value.splitlines())
        page_start, page_end = _page_span(page_map, start, end)
        output.append(
            {
                "node_type": node_type,
                "content": value,
                "page_start": page_start,
                "page_end": page_end,
            }
        )

    while position < len(source):
        atom_end = None
        node_type = ""
        if position in table_ranges:
            atom_end = table_ranges[position]
            node_type = "table"
        elif source.startswith("$$", position):
            close = source.find("$$", position + 2)
            atom_end = close + 2 if close >= 0 else len(source)
            node_type = "formula"
        elif source.startswith("\\[", position):
            close = source.find("\\]", position + 2)
            atom_end = close + 2 if close >= 0 else len(source)
            node_type = "formula"
        elif source[position] == "$" and not _is_escaped(source, position):
            if not source.startswith("$$", position):
                close = _find_inline_formula_close(source, position + 1)
                if close >= 0:
                    atom_end = close + 1
                    node_type = "formula"
        if atom_end is None:
            position += 1
            continue
        emit("passage", passage_start, position)
        emit(node_type, position, atom_end)
        position = atom_end
        passage_start = position
    emit("passage", passage_start, len(source))
    return output


def _section_stream(page_parts):
    pieces = []
    page_map = []
    for index, part in enumerate(page_parts):
        page_number = int(part.get("page_number") or 0)
        if index:
            pieces.append("\n")
            page_map.append(page_number)
        text = str(part.get("text") or "")
        pieces.append(text)
        page_map.extend([page_number] * len(text))
    return "".join(pieces), page_map


def _page_span(page_map, start, end):
    pages = [page for page in page_map[start:end] if page]
    return (min(pages), max(pages)) if pages else (0, 0)


def _find_inline_formula_close(source, start):
    position = start
    while position < len(source):
        if source[position] == "$" and not _is_escaped(source, position):
            if not source.startswith("$$", position):
                return position
        position += 1
    return -1


def _is_escaped(source, position):
    backslashes = 0
    cursor = position - 1
    while cursor >= 0 and source[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 1


def _display_formula_state_after(line, state):
    position = 0
    while position < len(line):
        if state == "$$":
            close = line.find("$$", position)
            if close < 0:
                return state
            state = None
            position = close + 2
            continue
        if state == "\\]":
            close = line.find("\\]", position)
            if close < 0:
                return state
            state = None
            position = close + 2
            continue
        display = line.find("$$", position)
        bracket = line.find("\\[", position)
        candidates = [(offset, marker) for offset, marker in ((display, "$$"), (bracket, "\\]")) if offset >= 0]
        if not candidates:
            return None
        offset, state = min(candidates, key=lambda item: item[0])
        position = offset + 2
    return state


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
