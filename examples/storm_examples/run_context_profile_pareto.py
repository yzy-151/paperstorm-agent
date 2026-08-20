"""Run the v6.0 128K/256K/512K context Pareto experiment."""

import argparse
import json
import os
from pathlib import Path

from knowledge_storm.evaluation.public_benchmarks.longbench_context import load_longbench_v2
from knowledge_storm.evaluation.public_benchmarks.v60_harness import run_context_profile_benchmark
from knowledge_storm.evaluation.public_benchmarks.v60_llm import StreamingReader
from knowledge_storm.paperstorm_router_llm import _load_flat_toml_env


def main(argv=None):
    parser = argparse.ArgumentParser(description="PaperStorm context profile Pareto benchmark")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="deepseek/deepseek-chat")
    parser.add_argument("--api-base", default=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com"))
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--profile", action="append", type=int, default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--subdomain", action="append", default=[])
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--input-price", type=float, default=0.27)
    parser.add_argument("--output-price", type=float, default=1.10)
    args = parser.parse_args(argv)
    _load_flat_toml_env()
    dataset = load_longbench_v2(args.dataset, selected_subdomains=set(args.subdomain))
    if args.limit:
        dataset = type(dataset)(dataset.name, dataset.version, dataset.documents, dataset.cases[: args.limit], dataset.metadata)
    client = StreamingReader(
        args.model,
        os.getenv(args.api_key_env),
        api_base=args.api_base,
        max_tokens=args.max_output_tokens,
        input_price=args.input_price,
        output_price=args.output_price,
    )
    report = run_context_profile_benchmark(
        dataset,
        client.complete_prompt,
        Path(args.output_dir),
        profiles=tuple(args.profile or (128_000, 256_000, 512_000)),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main()
