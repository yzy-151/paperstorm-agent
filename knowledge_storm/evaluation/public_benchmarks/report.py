"""Artifact writers for public benchmark runs."""

import json
from pathlib import Path


def write_benchmark_artifacts(output_dir, manifest, report, predictions, bad_cases):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "manifest.json", manifest)
    _write_json(root / "metrics.json", report)
    _write_jsonl(root / "predictions.jsonl", predictions)
    _write_jsonl(root / "bad_cases.jsonl", bad_cases)
    (root / "report.md").write_text(
        _markdown_report(manifest, report), encoding="utf-8"
    )


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _markdown_report(manifest, report):
    lines = [
        "# {0} Public Benchmark".format(manifest["benchmark"]),
        "",
        "- Dataset version: `{0}`".format(manifest["dataset_version"]),
        "- Split: `{0}`".format(manifest["split"]),
        "- Cases: `{0}`".format(report["case_count"]),
        "- Evidence tier: `{0}`".format(manifest["evidence_tier"]),
        "",
        "| Mode | Recall | MRR | nDCG | P95 ms |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    cutoff = manifest["top_k"]
    for mode, metrics in report["modes"].items():
        lines.append(
            "| {0} | {1:.4f} | {2:.4f} | {3:.4f} | {4:.2f} |".format(
                mode,
                metrics.get("recall_at_{0}".format(cutoff), 0.0),
                metrics.get("mrr_at_{0}".format(cutoff), 0.0),
                metrics.get("ndcg_at_{0}".format(cutoff), 0.0),
                metrics.get("p95_latency_ms", 0.0),
            )
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend("- {0}".format(item) for item in manifest.get("limitations", []))
    return "\n".join(lines) + "\n"
