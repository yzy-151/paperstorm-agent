"""Loader for the official BEIR SciFact JSONL/TSV layout."""

import csv
import hashlib
import json
import shutil
import ssl
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from .base import BenchmarkCase, BenchmarkDataset, BenchmarkDocument


SCIFACT_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
)
SCIFACT_MD5 = "5f7d1de60b170fc8027bb7898e2efca1"


def verify_md5(path, expected=SCIFACT_MD5):
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual.lower() != str(expected).lower():
        raise ValueError("MD5 mismatch: expected {0}, got {1}".format(expected, actual))
    return actual


def download_scifact(cache_dir, url=SCIFACT_URL, expected_md5=SCIFACT_MD5):
    cache_root = Path(cache_dir)
    target = cache_root / "scifact"
    if (target / "corpus.jsonl").is_file() and (target / "queries.jsonl").is_file():
        return target
    cache_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(cache_root)) as temp_dir:
        archive = Path(temp_dir) / "scifact.zip"
        _download_file(url, archive)
        verify_md5(archive, expected=expected_md5)
        extracted = Path(temp_dir) / "extracted"
        extracted.mkdir()
        with zipfile.ZipFile(archive) as handle:
            _safe_extract(handle, extracted)
        source = extracted / "scifact"
        if not source.is_dir():
            raise ValueError("SciFact archive does not contain scifact directory")
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(source), str(target))
    return target


def load_scifact(dataset_dir, split="test"):
    root = Path(dataset_dir)
    corpus_path = root / "corpus.jsonl"
    query_path = root / "queries.jsonl"
    qrels_path = root / "qrels" / "{0}.tsv".format(split)
    for path in (corpus_path, query_path, qrels_path):
        if not path.is_file():
            raise FileNotFoundError("missing SciFact file: {0}".format(path))

    documents = []
    for item in _read_jsonl(corpus_path):
        title = str(item.get("title") or "")
        body = str(item.get("text") or "")
        documents.append(
            BenchmarkDocument(
                document_id=str(item["_id"]),
                title=title,
                text="\n".join(value for value in (title, body) if value),
                metadata=dict(item.get("metadata") or {}),
            )
        )
    queries = {str(item["_id"]): str(item["text"]) for item in _read_jsonl(query_path)}
    qrels = {}
    with qrels_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            score = int(row["score"])
            if score > 0:
                qrels.setdefault(str(row["query-id"]), {})[
                    str(row["corpus-id"])
                ] = score
    cases = tuple(
        BenchmarkCase(
            case_id=query_id,
            query=queries[query_id],
            relevant_document_ids=tuple(relevance),
            relevance=relevance,
            split=split,
        )
        for query_id, relevance in sorted(qrels.items())
        if query_id in queries
    )
    return BenchmarkDataset(
        name="beir-scifact",
        version="official-beir-scifact",
        documents=tuple(documents),
        cases=cases,
        metadata={
            "source_url": SCIFACT_URL,
            "expected_md5": SCIFACT_MD5,
            "split": split,
        },
    )


def _read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _safe_extract(archive, destination):
    root = Path(destination).resolve()
    for item in archive.infolist():
        resolved = (root / item.filename).resolve()
        if root != resolved and root not in resolved.parents:
            raise ValueError("unsafe archive path: {0}".format(item.filename))
    archive.extractall(root)


def _download_file(url, target, timeout=60):
    try:
        import certifi
    except ImportError as exc:
        raise RuntimeError("verified benchmark downloads require certifi") from exc
    request = urllib.request.Request(
        url, headers={"User-Agent": "PaperStorm/5.5 public-benchmark"}
    )
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(request, context=context, timeout=timeout) as response:
        with Path(target).open("wb") as output:
            shutil.copyfileobj(response, output)
