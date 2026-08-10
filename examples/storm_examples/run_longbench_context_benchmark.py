"""Validate LongBench v2 selected-subset loading and paired context outputs."""

import argparse
import json
from pathlib import Path

from knowledge_storm.evaluation.public_benchmarks.longbench_context import load_longbench_v2, score_context_modes


def main(argv=None):
    parser = argparse.ArgumentParser(description="PaperStorm LongBench selected-subset scorer")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--predictions", required=True, help="JSON mapping mode -> prediction rows")
    parser.add_argument("--output", required=True)
    parser.add_argument("--subdomain", action="append", default=[])
    args = parser.parse_args(argv)
    dataset = load_longbench_v2(args.dataset, selected_subdomains=set(args.subdomain))
    predictions = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    report = score_context_modes(predictions, baseline="full")
    report.update({"benchmark": dataset.name, "dataset_version": dataset.version, "dataset_case_count": len(dataset.cases), "selected_subdomains": list(dataset.metadata["selected_subdomains"]), "official_aggregate": False})
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main()
