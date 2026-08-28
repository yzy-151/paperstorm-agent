"""Run the RAG Agent dual-role interview simulator."""

from __future__ import annotations

import argparse
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_NAME = "knowledge_storm.rag_interview_simulator"
MODULE_PATH = PROJECT_ROOT / "knowledge_storm" / "rag_interview_simulator.py"


def _load_simulator_module():
    had_existing_module = MODULE_NAME in sys.modules
    existing_module = sys.modules.get(MODULE_NAME)
    try:
        spec = spec_from_file_location(MODULE_NAME, MODULE_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load RAG interview simulator from {0}".format(MODULE_PATH))
        module = module_from_spec(spec)
        sys.modules[MODULE_NAME] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if had_existing_module:
            sys.modules[MODULE_NAME] = existing_module
        else:
            sys.modules.pop(MODULE_NAME, None)


def build_parser():
    parser = argparse.ArgumentParser(description="Run a RAG Agent dual-role interview simulation.")
    parser.add_argument("--mode", choices=("deterministic", "llm"), default="deterministic")
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--output", default="results/rag_interview_simulation.md")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--project-context",
        default="PaperStorm is a production-oriented research and RAG agent.",
    )
    parser.add_argument(
        "--fallback-on-parse-error",
        action="store_true",
        help="Use a deterministic candidate answer after invalid LLM JSON.",
    )
    return parser


def build_litellm_callable(model):
    """Create an explicit LiteLLM adapter only when LLM mode is requested."""
    try:
        from litellm import completion
    except ImportError as error:
        raise RuntimeError(
            "--mode llm requires the optional litellm dependency; use --mode deterministic instead"
        ) from error

    def invoke(prompt):
        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response

    return invoke


def main(argv=None):
    args = build_parser().parse_args(argv)
    simulator_module = _load_simulator_module()
    llm = build_litellm_callable(args.model) if args.mode == "llm" else None
    simulator = simulator_module.RagInterviewSimulator(
        project_context=args.project_context,
        llm=llm,
        mode=args.mode,
        model=args.model,
        fallback_on_parse_error=args.fallback_on_parse_error,
    )
    session = simulator.run(args.rounds)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        simulator_module.render_markdown(session), encoding="utf-8"
    )
    print("Wrote {0}".format(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
