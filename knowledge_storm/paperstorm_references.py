import json
import re
from pathlib import Path


REFERENCE_HEADING_PATTERN = re.compile(
    r"^#{1,3}\s*(参考文献|references)\s*$", flags=re.I | re.M
)


def load_reference_registry(run_dir):
    """Load canonical sources in the same order used by STORM citation ids."""
    source_path = Path(run_dir) / "url_to_info.json"
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    url_to_index = payload.get("url_to_unified_index") or {}
    url_to_info = payload.get("url_to_info") or {}
    references = []
    for url, raw_index in sorted(url_to_index.items(), key=_reference_sort_key):
        info = url_to_info.get(url) or {}
        metadata = info.get("meta") or {}
        title = str(info.get("title") or metadata.get("title") or "").strip()
        canonical_url = str(info.get("url") or url or "").strip()
        if not title or not canonical_url:
            continue
        references.append(
            {
                "id": _safe_int(raw_index, len(references) + 1),
                "title": title,
                "authors": _normalize_authors(metadata.get("authors")),
                "url": canonical_url,
                "pdf_url": str(metadata.get("pdf_url") or "").strip(),
                "published": str(metadata.get("published") or "").strip(),
                "source_type": str(metadata.get("source_type") or "retrieval"),
            }
        )
    return references


def normalize_citations(citations):
    """Normalize direct citations and expand article citations to original sources."""
    normalized = []
    seen = set()
    for fallback_id, citation in enumerate(citations or [], start=1):
        citation = citation or {}
        candidates = citation.get("original_sources") or [citation]
        for candidate in candidates:
            candidate = candidate or {}
            url = str(candidate.get("url") or candidate.get("source_url") or "").strip()
            title = str(
                candidate.get("title")
                or candidate.get("original_title")
                or citation.get("title")
                or ""
            ).strip()
            if not title or not url:
                continue
            key = (url.casefold(), title.casefold())
            if key in seen:
                continue
            seen.add(key)
            normalized.append(
                {
                    "id": _safe_int(
                        candidate.get("id")
                        or candidate.get("citation_index")
                        or citation.get("id"),
                        fallback_id,
                    ),
                    "title": title,
                    "authors": _normalize_authors(
                        candidate.get("authors") or citation.get("authors")
                    ),
                    "url": url,
                    "pdf_url": str(candidate.get("pdf_url") or "").strip(),
                    "published": str(candidate.get("published") or "").strip(),
                }
            )
    return normalized


def render_reference_markdown(references, heading="参考文献"):
    lines = []
    for fallback_id, reference in enumerate(references or [], start=1):
        title = str(reference.get("title") or "").strip()
        url = str(reference.get("url") or "").strip()
        if not title or not url:
            continue
        reference_id = _safe_int(reference.get("id"), fallback_id)
        authors = _normalize_authors(reference.get("authors"))
        author_text = ", ".join(authors) if authors else "作者信息未提供"
        published = str(reference.get("published") or "").strip()
        date_suffix = " · {0}".format(published[:10]) if published else ""
        lines.append(
            "[{0}] **{1}** — {2}{3}. [原文]({4})".format(
                reference_id, title, author_text, date_suffix, url
            )
        )
    if not lines:
        return ""
    return "## {0}\n\n{1}\n".format(heading, "\n\n".join(lines))


def append_reference_section(text, references, heading="参考文献"):
    value = str(text or "")
    if REFERENCE_HEADING_PATTERN.search(value):
        return value
    rendered = render_reference_markdown(references, heading=heading)
    if not rendered:
        return value
    return value.rstrip() + "\n\n" + rendered


def materialize_article_references(run_dir):
    run_path = Path(run_dir)
    references = load_reference_registry(run_path)
    written_paths = []
    for filename in ("storm_gen_article.txt", "storm_gen_article_polished.txt"):
        article_path = run_path / filename
        if not article_path.is_file():
            continue
        original = article_path.read_text(encoding="utf-8", errors="replace")
        enriched = append_reference_section(original, references)
        if enriched != original:
            article_path.write_text(enriched, encoding="utf-8")
        written_paths.append(str(article_path))
    return {
        "reference_count": len(references),
        "paths": written_paths,
        "references": references,
    }


def append_answer_references(answer, citations):
    return append_reference_section(answer, normalize_citations(citations))


def _normalize_authors(authors):
    if isinstance(authors, str):
        authors = [item.strip() for item in authors.split(",")]
    return [str(item).strip() for item in authors or [] if str(item).strip()]


def _safe_int(value, fallback):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


def _reference_sort_key(item):
    try:
        return int(item[1])
    except (TypeError, ValueError):
        return 10**9
