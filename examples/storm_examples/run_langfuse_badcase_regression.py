"""Run the frozen Langfuse-derived badcase regression dataset."""

import argparse
import json
from pathlib import Path

from knowledge_storm.evaluation.langfuse_badcases import (
    run_badcase_regression,
    sync_langfuse_dataset,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "docs" / "benchmarks" / "langfuse_observed_badcases_v2.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-dir", default="./results/langfuse_badcase_regression")
    parser.add_argument("--sync-langfuse", action="store_true")
    parser.add_argument("--dataset-name", default="")
    args = parser.parse_args()

    report = run_badcase_regression(args.dataset, output_dir=args.output_dir)
    output = {"regression": report}
    if args.sync_langfuse:
        output["langfuse_sync"] = sync_langfuse_dataset(
            args.dataset, dataset_name=args.dataset_name
        )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if report["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
