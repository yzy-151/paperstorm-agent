"""Run PaperStorm's deterministic Langfuse badcase demonstration."""

from __future__ import annotations

import argparse
import json
import sys
import types
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODULE_NAMES = (
    "knowledge_storm",
    "knowledge_storm.paperstorm_observability",
    "knowledge_storm.langfuse_badcase_demo",
)
_MISSING = object()


def _load_demo_module():
    previous_modules = {name: sys.modules.get(name, _MISSING) for name in _MODULE_NAMES}
    loaded = False
    try:
        package = types.ModuleType("knowledge_storm")
        package.__path__ = [str(PROJECT_ROOT / "knowledge_storm")]
        sys.modules["knowledge_storm"] = package
        for module_name in _MODULE_NAMES[1:]:
            path = PROJECT_ROOT / "knowledge_storm" / (module_name.rsplit(".", 1)[-1] + ".py")
            spec = spec_from_file_location(module_name, path)
            module = module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        loaded = True
        return sys.modules["knowledge_storm.langfuse_badcase_demo"], previous_modules
    finally:
        if not loaded:
            _restore_modules(previous_modules)


def _restore_modules(previous_modules):
    for name, module in previous_modules.items():
        if module is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _select_case(payload, scenario):
    if isinstance(payload, dict) and isinstance(payload.get("scenarios"), dict):
        try:
            return payload["scenarios"][scenario]
        except KeyError as error:
            raise ValueError("scenario not found: {0}".format(scenario)) from error
    if isinstance(payload, dict):
        return payload
    raise ValueError("case file must be a JSON object or contain a scenarios object")


def main(argv=None):
    demo, previous_modules = _load_demo_module()
    try:
        parser = argparse.ArgumentParser(description="Trace a deterministic PaperStorm RAG badcase.")
        parser.add_argument("--output-dir", default="results/langfuse_badcase_demo")
        parser.add_argument("--case-file", default=None, help="JSON case or a scenarios mapping.")
        parser.add_argument("--scenario", default="composite", help="Scenario name in --case-file.")
        args = parser.parse_args(argv)

        if args.case_file:
            payload = json.loads(Path(args.case_file).read_text(encoding="utf-8"))
            case = _select_case(payload, args.scenario)
        else:
            case = dict(demo.DEFAULT_COMPOSITE_BADCASE)

        output_dir = Path(args.output_dir)
        result = demo.run_badcase_demo(case, output_dir=output_dir)
        report = {"scenario": args.scenario, "case_id": case.get("case_id", ""), "result": result}
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "langfuse_badcase_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("Wrote {0}".format(report_path))
        print("PaperStorm trace {0} ({1})".format(
            result["paperstorm_trace_id"], result["observability"]["status"]
        ))
        return 0
    finally:
        _restore_modules(previous_modules)


if __name__ == "__main__":
    raise SystemExit(main())
