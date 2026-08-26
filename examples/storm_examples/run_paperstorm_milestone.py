"""Run cumulative PaperStorm retrieval milestones without implicit downloads."""

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import uuid
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


class BenchmarkPreflightError(RuntimeError):
    """A machine-classifiable local prerequisite failure."""

    def __init__(self, reason_code, message):
        super().__init__(message)
        self.reason_code = str(reason_code)


def build_parser():
    parser = argparse.ArgumentParser(description="Run a PaperStorm cumulative milestone")
    parser.add_argument("--milestone", choices=("P1",), default="P1")
    parser.add_argument("--benchmark", nargs="+", choices=AFFECTED_P1)
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--model-cache")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--embedding", choices=("hash", "real"), default="real")
    parser.add_argument(
        "--evaluation-phase", choices=("development", "final"), default="development"
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--top-k", type=int)
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
    summary = {"milestone": args.milestone, "status": "running", "benchmarks": {}}
    _atomic_write_json(output_root / "run_status.json", {"status": "running", "started_at": _now()})
    for benchmark in resolve_benchmarks(args):
        try:
            result = _run_one(args, benchmark, output_root)
        except BenchmarkPreflightError as exc:
            result = {
                "status": "blocked",
                "benchmark": benchmark,
                "reason_code": exc.reason_code,
                "error_type": type(exc).__name__,
                "reason": str(exc),
            }
        except PermissionError as exc:
            result = {
                "status": "blocked",
                "benchmark": benchmark,
                "reason_code": "dataset_permission_denied",
                "error_type": type(exc).__name__,
                "reason": str(exc),
            }
        except FileNotFoundError as exc:
            result = {
                "status": "blocked",
                "benchmark": benchmark,
                "reason_code": "dataset_missing",
                "error_type": type(exc).__name__,
                "reason": str(exc),
            }
        except Exception as exc:
            result = {
                "status": "failed",
                "benchmark": benchmark,
                "reason_code": "benchmark_execution_failed",
                "error_type": type(exc).__name__,
                "reason": str(exc),
            }
        result = sanitize_json_payload(result)
        if result.get("status") in {"failed", "blocked"}:
            _write_benchmark_status(output_root / benchmark, result)
        summary["benchmarks"][benchmark] = result
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    statuses = {item.get("status") for item in summary["benchmarks"].values()}
    summary["status"] = (
        "failed" if "failed" in statuses else
        "completed_with_blocks" if "blocked" in statuses else
        "completed"
    )
    _atomic_write_json(output_root / "milestone_summary.json", summary)
    _atomic_write_json(
        output_root / "run_status.json",
        {"status": summary["status"], "finished_at": _now()},
    )
    return summary


def _run_one(args, benchmark, output_root):
    started_at = _now()
    split, top_k = _protocol_for(benchmark, args.evaluation_phase, args.top_k)
    dataset, dataset_path, scope_field = _load_dataset(args, benchmark, split)
    dataset = _subset(dataset, args.smoke_limit, scope_field)
    provider = _embedding_provider(args)
    run_dir = output_root / benchmark
    run_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(run_dir / "run_status.json", {"status": "running", "started_at": started_at})
    comparison = _comparison_metadata(
        benchmark=benchmark,
        split=split,
        top_k=top_k,
        model=args.model,
        embedding_kind=args.embedding,
        smoke_limit=args.smoke_limit,
    )
    report = run_retrieval_benchmark(
        dataset,
        embedding_provider=provider,
        modes=("hybrid",),
        top_k=top_k,
        seed=args.seed,
        bootstrap_samples=100 if args.smoke_limit else 2000,
        output_dir=run_dir,
        scope_field=scope_field,
        milestone_metadata={
            "milestone": "P1",
            "baseline_reference": BASELINE_REFERENCE,
            "comparison": comparison,
        },
        structured_nodes=benchmark in {"scifact", "qasper-retrieval"},
        parent_budget_tokens=512 if benchmark == "qasper-retrieval" else (256 if benchmark == "scifact" else 0),
    )
    manifest = build_milestone_manifest(
        milestone="P1",
        git_sha=_git_sha(),
        dataset_path=dataset_path,
        split=split,
        models={"embedding": getattr(provider, "name", args.model)},
        top_k=top_k,
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
    manifest["comparison"] = comparison
    manifest["benchmark"] = benchmark
    write_milestone_manifest(run_dir / "milestone_manifest.json", manifest)
    dossiers = []
    if benchmark == "pim":
        dossiers = _pim_dossiers(dataset_path, report)
        write_case_dossiers(
            run_dir / "case_dossiers.jsonl", dossiers
        )
    elif benchmark == "qasper-retrieval":
        dossiers = _qasper_dossiers(dataset, report)
        write_case_dossiers(run_dir / "case_dossiers.jsonl", dossiers)
    result = {
        "status": "completed",
        "benchmark": benchmark,
        "output_dir": str(run_dir),
        "case_count": report["case_count"],
        "dossier_count": len(dossiers),
        "dossier_status": "written" if dossiers else "no_cases_available",
        "comparison": comparison,
    }
    _atomic_write_json(run_dir / "run_status.json", {"status": "completed", "finished_at": _now()})
    return result


def _load_dataset(args, benchmark, split):
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
        return load_scifact(path, split=split), path, None
    filename = "qasper-{0}-v0.3.json".format(split)
    path = _first_existing(
        (
            root / "qasper" / filename,
            root / "qasper-official-v0.3" / filename,
            root / filename,
            root / "qasper" / (split + ".json"),
        ),
        file_required=True,
    )
    return load_qasper_official_json(path, split=split), path, "paper_id"


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
        raise BenchmarkPreflightError(
            "model_missing",
            "real embedding requires --model-cache; implicit download is disabled"
        )
    cache = Path(args.model_cache)
    model_dir = cache / ("models--" + args.model.replace("/", "--"))
    if not cache.is_dir() or not model_dir.exists():
        raise BenchmarkPreflightError(
            "model_missing",
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
    domains = {item["id"]: item.get("domain") for item in payload["documents"]}
    predictions = {row["case_id"]: row for row in report["predictions"]}
    dossiers = []
    for case in payload["cases"]:
        prediction = predictions.get(case["id"], {})
        if not prediction:
            continue
        unresolved = bool(case.get("unresolved"))
        ranked_ids = prediction.get("ranked_document_ids", [])
        relevant_ids = case.get("relevant", [])
        acceptance = dict(case.get("acceptance") or {})
        top_1 = ranked_ids[:1]
        forbidden_ids = set(acceptance.get("forbidden_document_ids") or [])
        forbidden_domains = set(acceptance.get("forbidden_domains") or [])
        forbidden_hits = [item for item in ranked_ids if item in forbidden_ids]
        forbidden_domain_hits = [
            item for item in ranked_ids if domains.get(item) in forbidden_domains
        ]
        top1_relevant = bool(top_1 and top_1[0] in relevant_ids)
        resolved = bool(
            not unresolved
            and top1_relevant
            and not forbidden_hits
            and not forbidden_domain_hits
        )
        dossiers.append(
            CaseDossier(
                case_id=case["id"],
                milestone="P1",
                question=case["query"],
                before=dict(
                    case["before"],
                    evidence_level="anecdotal",
                    paired_comparison_allowed=False,
                ),
                root_cause=(
                    "PIM acronym/domain ambiguity or bilingual lexical mismatch"
                ),
                change=_actual_change(prediction),
                after={
                    "ranked_document_ids": ranked_ids,
                    "relevant_document_ids": relevant_ids,
                    "top_1": top_1,
                    "acceptance": acceptance,
                    "actual": {
                        "top1_relevant": top1_relevant,
                        "forbidden_document_ids": forbidden_hits,
                        "forbidden_domains": [domains.get(item) for item in forbidden_domain_hits],
                    },
                    "forbidden_hits_at_k": forbidden_hits,
                    "forbidden_domain_hits_at_k": forbidden_domain_hits,
                    "resolved": resolved,
                    "search_plan": prediction.get("search_plan", {}),
                },
                residual_risk=(
                    "Ambiguous PIM without reliable context still requires clarification"
                    if unresolved
                    else (
                        "Forbidden-domain documents remain inside Top-K despite a relevant Top-1."
                        if not resolved
                        else ""
                    )
                ),
            )
        )
    return dossiers


def _qasper_dossiers(dataset, report):
    """Create auditable current-run evidence without inventing a case baseline."""
    predictions = {row["case_id"]: row for row in report.get("predictions", [])}
    document_map = dataset.document_map()
    candidates = []
    for case in dataset.cases:
        prediction = predictions.get(case.case_id)
        if not prediction:
            continue
        relevant_ids = list(case.relevant_document_ids)
        gold_text = " ".join(
            document_map[item].text for item in relevant_ids if item in document_map
        )
        overlap = _lexical_overlap(case.query, gold_text)
        ranked_ids = list(prediction.get("ranked_document_ids") or [])
        top1_relevant = bool(ranked_ids and ranked_ids[0] in relevant_ids)
        recalled = bool(set(ranked_ids).intersection(relevant_ids))
        candidates.append((not recalled, overlap, case, prediction, top1_relevant))
    if not candidates:
        return []
    _, overlap, case, prediction, top1_relevant = sorted(
        candidates, key=lambda item: (not item[0], item[1], str(item[2].case_id))
    )[0]
    ranked_ids = list(prediction.get("ranked_document_ids") or [])
    relevant_ids = list(case.relevant_document_ids)
    before = {
        "source": "archived_aggregate",
        "case_level_before_available": False,
        "baseline_reference": BASELINE_REFERENCE,
        "baseline_sha256": _sha256_file(_baseline_path()),
        "paired_comparison_allowed": False,
        "benchmark": "qasper_test",
        "aggregate": _archived_qasper_aggregate(),
        "note": "Archived aggregate only; no case-level baseline was recorded.",
    }
    return [
        CaseDossier(
            case_id=str(case.case_id),
            milestone="P1",
            question=case.query,
            before=before,
            root_cause=(
                "Low query-to-gold lexical overlap and/or failure to retrieve gold evidence"
            ),
            change=_actual_change(prediction),
            after={
                "ranked_document_ids": ranked_ids,
                "relevant_document_ids": relevant_ids,
                "lexical_overlap": overlap,
                "top1_relevant": top1_relevant,
                "resolved": top1_relevant,
                "search_plan": prediction.get("search_plan", {}),
                "retrieval_stages": prediction.get("retrieval_stages", []),
            },
            residual_risk=(
                "Current run does not place gold evidence at rank 1; the case remains unresolved."
                if not top1_relevant
                else "Case-level improvement cannot be paired because the archived baseline is aggregate-only."
            ),
        )
    ]


def _lexical_overlap(left, right):
    left_tokens = set(re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", str(left).lower()))
    right_tokens = set(re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", str(right).lower()))
    if not left_tokens:
        return 0.0
    return round(len(left_tokens.intersection(right_tokens)) / len(left_tokens), 6)


def _archived_qasper_aggregate():
    path = _baseline_path()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload["qasper_test"]["methods"]["hybrid"])


def _actual_change(prediction):
    plan = prediction.get("search_plan") or {}
    stages = prediction.get("retrieval_stages") or []
    stage_by_name = {str(item.get("name")): item for item in stages}
    changes = ["RRF candidate fusion"]
    if len(plan.get("subqueries") or []) > 1:
        changes.insert(0, "structured multi-query planning")
    elif plan.get("must_terms") or plan.get("negative_terms"):
        changes.insert(0, "structured domain constraints")
    parent = stage_by_name.get("parent_expand") or {}
    if parent.get("status") == "completed":
        changes.append("section parent context expansion")
    gate = stage_by_name.get("gate") or {}
    if gate.get("status") == "completed":
        changes.append("typed evidence filtering and final selection")
    return "; ".join(changes)


def _protocol_for(benchmark, evaluation_phase, requested_top_k):
    split = {
        ("scifact", "development"): "dev",
        ("scifact", "final"): "test",
        ("qasper-retrieval", "development"): "validation",
        ("qasper-retrieval", "final"): "test",
    }.get((benchmark, evaluation_phase), "test")
    default_top_k = 10 if benchmark == "scifact" else 5
    return split, max(1, int(requested_top_k or default_top_k))


def _comparison_metadata(
    benchmark, split, top_k, model, embedding_kind, smoke_limit
):
    baseline_path = _baseline_path()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_key = {
        "scifact": "scifact_test",
        "qasper-retrieval": "qasper_test",
    }.get(benchmark)
    reasons = []
    if not baseline_key:
        reasons.append("no_frozen_baseline")
    else:
        expected = baseline[baseline_key]
        if split != "test":
            reasons.append("split_mismatch")
        if int(top_k) != int(expected["top_k"]):
            reasons.append("top_k_mismatch")
        if str(model) != str(baseline["protocol"]["embedding_model"]):
            reasons.append("embedding_model_mismatch")
        if embedding_kind != "real":
            reasons.append("smoke_embedding")
        if smoke_limit:
            reasons.append("partial_case_set")
    return {
        "status": "incomparable" if reasons else "comparable_aggregate_only",
        "reasons": reasons,
        "baseline_reference": BASELINE_REFERENCE,
        "baseline_sha256": _sha256_file(baseline_path),
        "paired_comparison_allowed": False,
        "aggregate_comparison_allowed": not reasons,
    }


def _baseline_path():
    return Path(__file__).resolve().parents[2] / BASELINE_REFERENCE


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    try:
        temporary.write_text(
            json.dumps(sanitize_json_payload(payload), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_benchmark_status(run_dir, result):
    _atomic_write_json(
        Path(run_dir) / "run_status.json",
        {"status": result.get("status"), "reason_code": result.get("reason_code"), "finished_at": _now()},
    )


def exit_code(summary):
    return 1 if summary.get("status") == "failed" else 0


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
    raise SystemExit(exit_code(main()))
