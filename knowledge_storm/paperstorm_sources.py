"""Normalize generated article passages and their original research sources."""

import json
import re
from pathlib import Path


ARTICLE_FILES = ("storm_gen_article_polished.txt", "storm_gen_article.txt")


def load_article_passages(run_dir):
    run_dir = Path(run_dir)
    article = _read_article(run_dir)
    if not article:
        return []
    sources = _load_unified_sources(run_dir / "url_to_info.json")
    passages = []
    section = "调研文章"
    section_paragraph = 0
    for block in re.split(r"\n\s*\n", article):
        block = block.strip()
        if not block:
            continue
        if block.startswith("#"):
            section = block.lstrip("#").strip() or section
            section_paragraph = 0
            continue
        section_paragraph += 1
        paragraph_index = len(passages) + 1
        citation_indices = _citation_indices(block)
        passages.append(
            {
                "paragraph_index": paragraph_index,
                "article_anchor": "article-paragraph-{0}".format(paragraph_index),
                "section": section,
                "section_paragraph": section_paragraph,
                "title": "{0} · 第 {1} 段".format(section, section_paragraph),
                "content": block,
                "citation_indices": citation_indices,
                "original_sources": [
                    sources[index] for index in citation_indices if index in sources
                ],
            }
        )
    return passages


def _read_article(run_dir):
    for name in ARTICLE_FILES:
        path = run_dir / name
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
    return ""


def _load_unified_sources(path):
    payload = _read_json(path, {})
    url_to_index = payload.get("url_to_unified_index") or {}
    url_to_info = payload.get("url_to_info") or {}
    sources = {}
    for url, raw_index in url_to_index.items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            continue
        info = url_to_info.get(url) or {}
        metadata = info.get("meta") or {}
        sources[index] = {
            "citation_index": index,
            "title": info.get("title") or "来源 {0}".format(index),
            "url": info.get("url") or url,
            "authors": metadata.get("authors") or [],
            "published": metadata.get("published") or "",
            "source_type": metadata.get("source_type") or "retrieval",
        }
    return sources


def _citation_indices(text):
    output = []
    for value in re.findall(r"\[(\d+)\]", text):
        index = int(value)
        if index not in output:
            output.append(index)
    return output


def _read_json(path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default
