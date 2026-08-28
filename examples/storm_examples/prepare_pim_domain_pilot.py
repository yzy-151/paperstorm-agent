"""Prepare a private, evidence-grounded PIM evaluation pilot.

The generated corpus and cases are private benchmark artifacts. Keep them outside
the repository (the default output directory is under the user's Codex cache).
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from knowledge_storm.document_ingestion import chunk_pdf_pages, extract_pdf_pages
from knowledge_storm.evaluation.domain_pilot import (
    DOMAIN_CATEGORIES,
    validate_domain_rows,
)


PIM_TERMS = (
    "无源互调",
    "passive intermodulation",
    "pim",
    "非线性",
    "互调",
    "抑制",
    "对消",
    "波束赋形",
    "神经网络",
)


def select_generation_chunks(rows, count=50, seed=55):
    """Select stable, source-balanced evidence chunks for question generation."""
    grouped = defaultdict(list)
    for row in rows or ():
        content = str(row.get("content") or "").strip()
        document_id = str(row.get("document_id") or "").strip()
        if not document_id or len(content) < 80:
            continue
        lowered = content.lower()
        relevance = sum(lowered.count(term) for term in PIM_TERMS)
        digest = hashlib.sha1(
            "{0}|{1}|{2}".format(seed, row.get("chunk_id"), content).encode("utf-8")
        ).hexdigest()
        grouped[document_id].append((relevance, digest, row))
    if not grouped:
        raise ValueError("no eligible PIM evidence chunks were found")
    queues = {
        document_id: [item[2] for item in sorted(items, key=lambda item: (-item[0], item[1]))]
        for document_id, items in sorted(grouped.items())
    }
    selected = []
    while len(selected) < int(count):
        progressed = False
        for document_id in sorted(queues):
            if queues[document_id] and len(selected) < int(count):
                selected.append(queues[document_id].pop(0))
                progressed = True
        if not progressed:
            break
    if len(selected) != int(count):
        raise ValueError(
            "expected {0} eligible chunks, found {1}".format(count, len(selected))
        )
    return selected


def parse_hermes_json_array(text):
    """Parse a JSON array from a defensive Hermes one-shot response."""
    value = str(text or "").strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, flags=re.I | re.S)
    if fence:
        value = fence.group(1).strip()
    start, end = value.find("["), value.rfind("]")
    if start < 0 or end < start:
        raise ValueError("Hermes response must contain a JSON array")
    try:
        payload = json.loads(value[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError("Hermes response contains invalid JSON: {0}".format(exc)) from exc
    if not isinstance(payload, list):
        raise ValueError("Hermes response must contain a JSON array")
    return payload


def build_corpus(pdf_paths, chunk_tokens=320, overlap_tokens=48):
    rows = []
    manifest = []
    for index, raw_path in enumerate(pdf_paths, start=1):
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        document_id = "pim-paper-{0:02d}".format(index)
        title = path.stem
        pages = extract_pdf_pages(path)
        chunks = chunk_pdf_pages(
            pages,
            document_id=document_id,
            title=title,
            chunk_tokens=chunk_tokens,
            overlap_tokens=overlap_tokens,
            strategy="contextual",
        )
        rows.extend(chunks)
        manifest.append(
            {
                "document_id": document_id,
                "title": title,
                "source_path": str(path),
                "page_count": len(pages),
                "chunk_count": len(chunks),
            }
        )
    return rows, manifest


def build_generation_slots(selected_rows):
    slots = []
    for index, row in enumerate(selected_rows, start=1):
        slots.append(
            {
                "slot_id": "slot-{0:03d}".format(index),
                "case_id": "pim-{0:03d}".format(index),
                "category": DOMAIN_CATEGORIES[(index - 1) % len(DOMAIN_CATEGORIES)],
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "source_title": row["title"],
                "page_number": int((row.get("metadata") or {}).get("page_number") or 0),
                "excerpt": row["content"],
            }
        )
    return slots


def build_generation_prompt(slots):
    payload = [
        {
            "slot_id": slot["slot_id"],
            "category": slot["category"],
            "chunk_id": slot["chunk_id"],
            "source_title": slot["source_title"],
            "page_number": slot["page_number"],
            "excerpt": slot["excerpt"],
        }
        for slot in slots
    ]
    return """你是中文无源互调（PIM）领域评测集编写员。根据每个槽位给出的唯一论文原文，生成恰好一道可由该原文独立回答的问题。

严格要求：
1. 只返回 JSON 数组，不要 Markdown，不要解释。
2. 每项只含 slot_id、question、reference_answer、evidence_quote、difficulty。
3. slot_id 必须原样保留；不得合并或遗漏槽位。
4. evidence_quote 必须逐字复制 excerpt 中一个连续片段，长度 12-160 个字符。
5. reference_answer 必须忠于原文，不得引入外部事实；问题应明确、自然、有区分度。
6. category 表示考查方向，但不要把英文类别直接写进问题。
7. difficulty 只能是 basic、intermediate、advanced。

槽位数据：
{0}
""".format(json.dumps(payload, ensure_ascii=False, indent=2))


def write_generation_prompts(output_dir, slots, batch_size=10):
    output_dir = Path(output_dir)
    prompt_dir = output_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for start in range(0, len(slots), int(batch_size)):
        batch = slots[start : start + int(batch_size)]
        path = prompt_dir / "generate-{0:02d}.txt".format(start // int(batch_size) + 1)
        path.write_text(build_generation_prompt(batch), encoding="utf-8")
        paths.append(path)
    return paths


def bind_generated_cases(slots, generated_rows):
    generated = {}
    for row in generated_rows:
        slot_id = str(row.get("slot_id") or "")
        if slot_id in generated:
            raise ValueError("duplicate generated slot: {0}".format(slot_id))
        generated[slot_id] = row
    cases = []
    for slot in slots:
        row = generated.get(slot["slot_id"])
        if row is None:
            raise ValueError("missing generated slot: {0}".format(slot["slot_id"]))
        quote = re.sub(r"\s+", " ", str(row.get("evidence_quote") or "")).strip()
        excerpt = re.sub(r"\s+", " ", slot["excerpt"]).strip()
        if quote not in excerpt:
            quote = _repair_near_exact_quote(excerpt, quote)
        if not quote:
            raise ValueError("ungrounded generated quote: {0}".format(slot["slot_id"]))
        if not 12 <= len(quote) <= 160:
            raise ValueError("generated quote length must be in [12, 160]: {0}".format(slot["slot_id"]))
        cases.append(
            {
                "case_id": slot["case_id"],
                "question": str(row.get("question") or "").strip(),
                "reference_answer": str(row.get("reference_answer") or "").strip(),
                "evidence_chunk_ids": [slot["chunk_id"]],
                "evidence_quote": quote,
                "category": slot["category"],
                "difficulty": str(row.get("difficulty") or "intermediate"),
                "source_title": slot["source_title"],
                "review_status": "hermes-generated",
            }
        )
    return cases


def _repair_near_exact_quote(excerpt, quote, minimum_ratio=0.88):
    if not excerpt or not quote:
        return ""
    best = (0.0, "")
    minimum = max(8, len(quote) - 3)
    maximum = min(len(excerpt), len(quote) + 6)
    for width in range(minimum, maximum + 1):
        for start in range(0, len(excerpt) - width + 1):
            candidate = excerpt[start : start + width]
            ratio = difflib.SequenceMatcher(None, quote, candidate).ratio()
            if ratio > best[0]:
                best = (ratio, candidate)
    return best[1] if best[0] >= float(minimum_ratio) else ""


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", action="append", required=True, dest="pdf_paths")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument(
        "--responses-dir",
        help="Optional directory containing generate-*.json Hermes responses.",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).resolve()
    corpus, manifest = build_corpus(args.pdf_paths)
    selected = select_generation_chunks(corpus, count=args.count)
    slots = build_generation_slots(selected)
    write_jsonl(output_dir / "corpus.jsonl", corpus)
    write_jsonl(output_dir / "generation_slots.jsonl", slots)
    prompt_paths = write_generation_prompts(output_dir, slots)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("corpus={0} selected_slots={1}".format(len(corpus), len(slots)))
    print("generation slots: {0}".format(output_dir / "generation_slots.jsonl"))
    print("generation prompts: {0}".format(len(prompt_paths)))
    if args.responses_dir:
        generated = []
        response_paths = sorted(Path(args.responses_dir).glob("generate-*.json"))
        for path in response_paths:
            generated.extend(parse_hermes_json_array(path.read_text(encoding="utf-8-sig")))
        cases = bind_generated_cases(slots, generated)
        validate_domain_rows(
            corpus,
            cases,
            expected_case_count=args.count,
            required_categories=DOMAIN_CATEGORIES,
        )
        write_jsonl(output_dir / "cases.jsonl", cases)
        print("validated cases: {0}".format(output_dir / "cases.jsonl"))
    print("Generate cases with Hermes, bind them with bind_generated_cases(), then run:")
    print("validate_domain_rows(corpus, cases, {0}, DOMAIN_CATEGORIES)".format(args.count))


if __name__ == "__main__":
    main()
