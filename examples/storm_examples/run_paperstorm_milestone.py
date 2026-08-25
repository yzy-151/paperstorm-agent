"""Run cumulative PaperStorm retrieval milestones without implicit downloads."""

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from knowledge_storm.badcase_reporting import (
    CaseDossier,
    build_milestone_manifest,
    sanitize_json_payload,
    write_case_dossiers,
    write_milestone_manifest,
)
from knowledge_storm.evaluation.public_benchmarks.base import (
    BenchmarkCase,
    BenchmarkDataset,
    BenchmarkDocument,
)
from knowledge_storm.evaluation.public_benchmarks.beir_scifact import load_scifact
from knowledge_storm.evaluation.public_benchmarks.qasper import load_qasper_official_json
from knowledge_storm.evaluation.public_benchmarks.runner import (
    HashEmbeddingProvider,
    run_retrieval_benchmark,
)
from knowledge_storm.retrieval import SentenceTransformerProvider


AFFECTED_P1 = ("pim", "scifact", "qasper-retrieval")
BASELINE_REFERENCE = "docs/benchmarks/paperstorm_public_v55_summary.json"


def build_parser():
    parser = argparse.ArgumentParser(description="Run a PaperStorm cumulative milestone")
    parser.add_argument("--milestone", choices=("P1",), default="P1")
    parser.add_argument("--benchmark", nargs="+", choices=AFFECTED_P1)
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--model-cache")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--embedding", choices=("hash", "real"), default="real")
    parser.add_argument(
        "--model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--smoke-limit", type=int)
    return parser


def resolve_benchmarks(args):
    return tuple(args.benchmark or AFFECTED_P1)


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.command = [sys.executable, str(Path(__file__).resolve())] + list(
        argv if argv is not None else sys.argv[1:]
    )
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {"milestone": args.milestone, "benchmarks": {}}
    for benchmark in resolve_benchmarks(args):
        try:
            result = _run_one(args, benchmark, output_root)
        except (FileNotFoundError, RuntimeError) as exc:
            result = {
                "status": "blocked",
                "benchmark": benchmark,
                "error_type": type(exc).__name__,
                "reason": str(exc),
            }
        except Exception as exc:
            result = {
                "status": "failed",
                "benchmark": benchmark,
                "error_type": type(exc).__name__,
                "reason": str(exc),
            }
        result = sanitize_json_payload(result)
        summary["benchmarks"][benchmark] = result
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    (output_root / "milestone_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def _run_one(args, benchmark, output_root):
    started_at = _now()
    dataset, dataset_path, scope_field = _load_dataset(args, benchmark)
    dataset = _subset(dataset, args.smoke_limit, scope_field)
    provider = _embedding_provider(args)
    run_dir = output_root / benchmark
    run_dir.mkdir(parents=True, exist_ok=True)
    report = run_retrieval_benchmark(
        dataset,
        embedding_provider=provider,
        modes=("hybrid",),
        top_k=args.top_k,
        seed=args.seed,
        bootstrap_samples=100 if args.smoke_limit else 2000,
        output_dir=run_dir,
        scope_field=scope_field,
        milestone_metadata={
            "milestone": "P1",
            "baseline_reference": BASELINE_REFERENCE,
        },
    )
    manifest = build_milestone_manifest(
        milestone="P1",
        git_sha=_git_sha(),
        dataset_path=dataset_path,
        split="test",
        models={"embedding": getattr(provider, "name", args.model)},
        top_k=args.top_k,
        seed=args.seed,
        command=args.command,
        started_at=started_at,
        finished_at=_now(),
        api_usage={"requests": 0, "tokens": 0},
        host_profile={
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
    )
    manifest["baseline_reference"] = BASELINE_REFERENCE
    manifest["benchmark"] = benchmark
    write_milestone_manifest(run_dir / "milestone_manifest.json", manifest)
    if benchmark == "pim":
        write_case_dossiers(
            run_dir / "case_dossiers.jsonl", _pim_dossiers(dataset_path, report)
        )
    return {
        "status": "completed",
        "benchmark": benchmark,
        "output_dir": str(run_dir),
        "case_count": report["case_count"],
    }


def _load_dataset(args, benchmark):
    root = Path(args.benchmark_root)
    if benchmark == "pim":
        fixture = (
            Path(__file__).resolve().parents[2]
            / "tests"
            / "fixtures"
            / "pim_retrieval_badcases.json"
        )
        return _load_pim_fixture(fixture), fixture, None
    if benchmark == "scifact":
        path = _first_existing((root / "scifact", root / "datasets" / "scifact", root))
        return load_scifact(path, split="test"), path, None
    path = _first_existing(
        (
            root / "qasper" / "qasper-test-v0.3.json",
            root / "qasper-test-v0.3.json",
            root / "qasper" / "test.json",
        ),
        file_required=True,
    )
    return load_qasper_official_json(path, split="test"), path, "paper_id"


def _load_pim_fixture(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    documents = tuple(
        BenchmarkDocument(
            item["id"], item["title"], item["text"], {"domain": item["domain"]}
        )
        for item in payload["documents"]
    )
    cases = tuple(
        BenchmarkCase(
            item["id"],
            item["query"],
            tuple(item["relevant"]),
            "test",
            metadata={
                "before": item["before"],
                "unresolved": bool(item.get("unresolved")),
            },
        )
        for item in payload["cases"]
    )
    return BenchmarkDataset("pim-domain-badcases", "p1-fixed-v1", documents, cases)


def _embedding_provider(args):
    if args.embedding == "hash":
        return HashEmbeddingProvider()
    if not args.model_cache:
        raise FileNotFoundError(
            "real embedding requires --model-cache; implicit download is disabled"
        )
    cache = Path(args.model_cache)
    model_dir = cache / ("models--" + args.model.replace("/", "--"))
    if not cache.is_dir() or not model_dir.exists():
        raise FileNotFoundError(
            "embedding model is not present in local cache: {0}".format(model_dir)
        )
    os.environ["PAPERSTORM_OFFLINE_TESTS"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return SentenceTransformerProvider(model_name=args.model, cache_folder=str(cache))


def _subset(dataset, limit, scope_field):
    if not limit:
        return dataset
    cases = tuple(dataset.cases[:max(1, int(limit))])
    if scope_field:
        values = {str(case.metadata.get(scope_field) or "") for case in cases}
        documents = tuple(
            doc
            for doc in dataset.documents
            if str(doc.metadata.get(scope_field) or "") in values
        )
    else:
        documents = dataset.documents
    return BenchmarkDataset(dataset.name, dataset.version, documents, cases, dataset.metadata)


def _pim_dossiers(fixture_path, report):
    payload = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
    predictions = {row["case_id"]: row for row in report["predictions"]}
    dossiers = []
    for case in payload["cases"]:
        prediction = predictions.get(case["id"], {})
        if not prediction:
            continue
        unresolved = bool(case.get("unresolved"))
        ranked_ids = prediction.get("ranked_document_ids", [])
        relevant_ids = case.get("relevant", [])
        dossiers.append(
            CaseDossier(
                case_id=case["id"],
                milestone="P1",
                question=case["query"],
                before=case["before"],
                root_cause=(
                    "PIM acronym/domain ambiguity or bilingual lexical mismatch"
                ),
                change=(
                    "structured SearchPlan, subquery expansion, RRF and "
                    "negative-term gate"
                ),
                after={
                    "ranked_document_ids": ranked_ids,
                    "relevant_document_ids": relevant_ids,
                    "resolved": bool(
                        not unresolved and set(ranked_ids).intersection(relevant_ids)
                    ),
                    "search_plan": prediction.get("search_plan", {}),
                },
                residual_risk=(
                    "Ambiguous PIM without reliable context still requires clarification"
                    if unresolved
                    else ""
                ),
            )
        )
    return dossiers


def _first_existing(candidates, file_required=False):
    for path in candidates:
        if path.is_file() if file_required else path.is_dir():
            return path
    raise FileNotFoundError(
        "local benchmark data not found; checked: {0}".format(
            ", ".join(str(path) for path in candidates)
        )
    )


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _now():
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
