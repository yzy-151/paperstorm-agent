"""Release-oriented benchmark registry and local process manager."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BenchmarkDefinition:
    id: str
    name: str
    version: str
    category: str
    evidence_tier: str
    description: str
    runner: str
    required_inputs: tuple[str, ...]
    metrics: tuple[str, ...]
    estimated_time: str
    requires_llm: bool = False
    blocked_reason: str = ""


DEFINITIONS = (
    BenchmarkDefinition(
        "scifact-retrieval-v55",
        "SciFact 科学论文检索",
        "v5.5",
        "RAG Retrieval",
        "public_official",
        "在 SciFact 官方 test 上比较 BM25、Hybrid 与 Cross-Encoder 重排。",
        "run_paperstorm_public_benchmark.py",
        ("scifact_dir",),
        ("Recall@10", "MRR@10", "nDCG@10", "P95 latency"),
        "Smoke 约 1 分钟；完整真实向量约 20-40 分钟",
    ),
    BenchmarkDefinition(
        "qasper-retrieval-v55",
        "QASPER 论文内证据检索",
        "v5.5",
        "RAG Retrieval",
        "public_official",
        "在 QASPER 官方 test 的论文内检索人工证据段落。",
        "run_paperstorm_public_benchmark.py",
        ("qasper_json",),
        ("Recall@5", "MRR@5", "nDCG@5", "P95 latency"),
        "Smoke 约 1 分钟；完整真实向量约 1-2 小时",
    ),
    BenchmarkDefinition(
        "qasper-answer-v55",
        "QASPER 端到端回答",
        "v5.5",
        "RAG Generation",
        "public_official_llm",
        "冻结检索排名后调用 Reader LLM，按官方口径计算 Answer/Evidence F1。",
        "run_qasper_answer_benchmark.py",
        ("qasper_rankings", "qasper_cache"),
        ("Answer F1", "Evidence F1", "Exact Match", "success rate"),
        "Smoke 约 5 分钟；完整 1451 题会产生 API 成本",
        requires_llm=True,
    ),
    BenchmarkDefinition(
        "longmemeval-retrieval-v56",
        "LongMemEval-S 长期记忆检索",
        "v5.6",
        "Memory",
        "public_official_retrieval",
        "在官方 500 题上评估跨会话 evidence-session Recall@5 与延迟。",
        "run_longmemeval_benchmark.py",
        ("longmemeval_json",),
        ("Recall@5", "category recall", "P50 latency", "P95 latency"),
        "Smoke 约 1 分钟；完整真实向量约 15-30 分钟",
    ),
    BenchmarkDefinition(
        "qasper-context-v56",
        "QASPER Context 预算治理",
        "v5.6",
        "Context",
        "public_official_diagnostic",
        "在冻结检索结果上测量 token 压缩、金证据保留和结构校验。",
        "run_qasper_context_benchmark.py",
        ("qasper_json", "qasper_rankings"),
        ("token ratio", "gold evidence recall", "validation rate"),
        "约 1-3 分钟，不调用 LLM",
    ),
    BenchmarkDefinition(
        "longbench-context-v56",
        "LongBench Context 配对评测",
        "v5.6",
        "Context",
        "adapter_only",
        "适配器已完成，但缺少同模型 full/fixed/v5.6 配对预测，不能发布正式分数。",
        "run_longbench_context_benchmark.py",
        ("longbench_json", "longbench_predictions"),
        ("paired task score", "token reduction", "quality delta"),
        "尚未具备可发布输入",
        blocked_reason="缺少冻结的同模型配对预测文件",
    ),
)


class BenchmarkRegistry:
    def __init__(self, benchmark_root: Optional[Path] = None):
        self.root = _resolve_benchmark_root(benchmark_root)
        self.inputs = _discover_inputs(self.root)
        self._definitions = {item.id: item for item in DEFINITIONS}

    def catalog(self):
        benchmarks = []
        for definition in DEFINITIONS:
            inputs = [
                {
                    "key": key,
                    "path": str(self.inputs.get(key) or ""),
                    "available": bool(self.inputs.get(key)),
                }
                for key in definition.required_inputs
            ]
            missing = [item["key"] for item in inputs if not item["available"]]
            blocker = definition.blocked_reason
            if missing and not blocker:
                blocker = "缺少本地输入：{0}".format(", ".join(missing))
            ready = not blocker and not missing
            latest_path = _latest_result_path(definition.id, self.root)
            benchmarks.append(
                {
                    "id": definition.id,
                    "name": definition.name,
                    "version": definition.version,
                    "category": definition.category,
                    "evidence_tier": definition.evidence_tier,
                    "description": definition.description,
                    "metrics": list(definition.metrics),
                    "estimated_time": definition.estimated_time,
                    "requires_llm": definition.requires_llm,
                    "ready": ready,
                    "status": "ready" if ready else "blocked",
                    "blocker": blocker,
                    "inputs": inputs,
                    "profiles": ["smoke", "quality"],
                    "latest_result_path": str(latest_path or ""),
                    "latest_result": _read_json(latest_path) if latest_path else {},
                }
            )
        return {
            "benchmark_root": str(self.root),
            "model_cache": str(self.root / "models"),
            "python": sys.executable,
            "benchmarks": benchmarks,
        }

    def definition(self, benchmark_id: str):
        try:
            return self._definitions[benchmark_id]
        except KeyError as exc:
            raise KeyError("unknown benchmark: {0}".format(benchmark_id)) from exc

    def build_command(
        self,
        benchmark_id: str,
        output_dir: Path,
        profile: str = "smoke",
        allow_paid_llm: bool = False,
    ):
        definition = self.definition(benchmark_id)
        if profile not in {"smoke", "quality"}:
            raise ValueError("profile must be smoke or quality")
        if definition.blocked_reason:
            raise ValueError(definition.blocked_reason)
        missing = [key for key in definition.required_inputs if not self.inputs.get(key)]
        if missing:
            raise ValueError("缺少本地输入：{0}".format(", ".join(missing)))
        if definition.requires_llm:
            if not allow_paid_llm:
                raise ValueError("付费 LLM Benchmark 必须显式确认")
            if not os.getenv("DEEPSEEK_API_KEY"):
                raise ValueError("运行 QASPER Answer F1 需要 DEEPSEEK_API_KEY")

        runner = PROJECT_ROOT / "examples" / "storm_examples" / definition.runner
        command = [sys.executable, str(runner)]
        output_dir = Path(output_dir)
        if benchmark_id == "scifact-retrieval-v55":
            command += [
                "--benchmark", "scifact", "--dataset-dir", str(self.inputs["scifact_dir"]),
                "--cache-dir", str(self.root), "--output-dir", str(output_dir),
                "--top-k", "10",
            ]
            command += ["--embedding", "hash", "--modes", "bm25", "hybrid", "--smoke-limit", "20"] if profile == "smoke" else ["--embedding", "real", "--modes", "bm25", "dense", "hybrid", "hybrid_rerank", "--reranker"]
        elif benchmark_id == "qasper-retrieval-v55":
            command += [
                "--benchmark", "qasper", "--dataset-dir", str(self.inputs["qasper_json"]),
                "--cache-dir", str(self.root), "--output-dir", str(output_dir),
                "--top-k", "5",
            ]
            command += ["--embedding", "hash", "--modes", "bm25", "hybrid", "--smoke-limit", "20"] if profile == "smoke" else ["--embedding", "real", "--modes", "bm25", "dense", "hybrid", "hybrid_rerank", "--reranker"]
        elif benchmark_id == "qasper-answer-v55":
            command += [
                "--split", "test", "--retrieval-predictions", str(self.inputs["qasper_rankings"]),
                "--cache-dir", str(self.inputs["qasper_cache"]), "--output-dir", str(output_dir),
            ]
            if profile == "smoke":
                command += ["--smoke-limit", "10"]
        elif benchmark_id == "longmemeval-retrieval-v56":
            command += [
                "--dataset", str(self.inputs["longmemeval_json"]), "--output-dir", str(output_dir),
                "--model-cache", str(self.root / "models"), "--top-k", "5",
            ]
            command += ["--embedding", "hash", "--limit", "10"] if profile == "smoke" else ["--embedding", "sentence-transformer"]
        elif benchmark_id == "qasper-context-v56":
            command += [
                "--dataset", str(self.inputs["qasper_json"]), "--rankings", str(self.inputs["qasper_rankings"]),
                "--output-dir", str(output_dir), "--mode", "hybrid_rerank",
            ]
        else:
            raise ValueError("benchmark is not runnable")
        return command


class BenchmarkRunManager:
    def __init__(
        self,
        service_root: Path,
        registry: Optional[BenchmarkRegistry] = None,
        popen_factory: Callable = subprocess.Popen,
    ):
        self.root = Path(service_root) / "benchmark_runs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry = registry or BenchmarkRegistry()
        self.popen_factory = popen_factory
        self._processes: Dict[str, object] = {}
        self._logs: Dict[str, object] = {}
        self._lock = threading.Lock()

    def catalog(self):
        return self.registry.catalog()

    def start(self, benchmark_id: str, profile="smoke", allow_paid_llm=False):
        run_id = uuid.uuid4().hex
        run_dir = self.root / run_id
        output_dir = run_dir / "artifacts"
        run_dir.mkdir(parents=True)
        command = self.registry.build_command(
            benchmark_id,
            output_dir=output_dir,
            profile=profile,
            allow_paid_llm=allow_paid_llm,
        )
        log_path = run_dir / "benchmark.log"
        log_handle = log_path.open("w", encoding="utf-8")
        try:
            process = self.popen_factory(
                command,
                cwd=str(PROJECT_ROOT),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
            )
        except Exception:
            log_handle.close()
            raise
        manifest = {
            "run_id": run_id,
            "benchmark_id": benchmark_id,
            "profile": profile,
            "status": "running",
            "pid": getattr(process, "pid", None),
            "command": command,
            "command_preview": subprocess.list2cmdline(command),
            "created_at": _now(),
            "started_at": _now(),
            "output_dir": str(output_dir),
            "log_path": str(log_path),
            "manifest_path": str(run_dir / "run.json"),
        }
        self._write(manifest)
        with self._lock:
            self._processes[run_id] = process
            self._logs[run_id] = log_handle
        return self.get(run_id)

    def get(self, run_id: str):
        manifest = self._read(run_id)
        process = self._processes.get(run_id)
        if manifest["status"] == "running" and process is not None:
            returncode = process.poll()
            if returncode is not None:
                self._close_log(run_id)
                manifest["returncode"] = returncode
                manifest["status"] = "succeeded" if returncode == 0 else "failed"
                manifest["finished_at"] = _now()
                self._write(manifest)
        elif manifest["status"] == "running" and process is None:
            manifest["status"] = "interrupted"
            manifest["finished_at"] = _now()
            self._write(manifest)
        manifest["log_tail"] = _tail(Path(manifest["log_path"]))
        result_path = Path(manifest["output_dir"]) / "metrics.json"
        manifest["result"] = _read_json(result_path)
        manifest["result_path"] = str(result_path) if result_path.exists() else ""
        return manifest

    def cancel(self, run_id: str):
        manifest = self._read(run_id)
        process = self._processes.get(run_id)
        if manifest["status"] == "running" and process is not None:
            process.terminate()
            manifest["status"] = "cancelled"
            manifest["finished_at"] = _now()
            self._close_log(run_id)
            self._write(manifest)
        return self.get(run_id)

    def _manifest_path(self, run_id):
        if not run_id or any(value in run_id for value in ("/", "\\", "..")):
            raise KeyError("invalid benchmark run id")
        return self.root / run_id / "run.json"

    def _read(self, run_id):
        path = self._manifest_path(run_id)
        if not path.exists():
            raise KeyError("benchmark run not found: {0}".format(run_id))
        return json.loads(path.read_text(encoding="utf-8"))

    def _write(self, manifest):
        path = Path(manifest["manifest_path"])
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    def _close_log(self, run_id):
        handle = self._logs.pop(run_id, None)
        if handle is not None and not handle.closed:
            handle.flush()
            handle.close()


def _resolve_benchmark_root(explicit=None):
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.getenv("PAPERSTORM_BENCHMARK_ROOT"):
        candidates.append(Path(os.environ["PAPERSTORM_BENCHMARK_ROOT"]))
    candidates.extend(
        [
            Path.home() / "Desktop" / "codex" / "paperstorm-benchmarks",
            PROJECT_ROOT / "data" / "benchmarks",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].expanduser().resolve() if candidates else (PROJECT_ROOT / "data" / "benchmarks")


def _discover_inputs(root):
    root = Path(root)
    qasper_json = root / "qasper-official-v0.3" / "qasper-test-v0.3.json"
    project_rankings = PROJECT_ROOT / "results" / "public_benchmarks" / "v55_qasper_test_real" / "predictions.jsonl"
    context_rankings = root / "v56" / "runs" / "qasper-context-v56" / "predictions.jsonl"
    values = {
        "scifact_dir": root / "datasets" / "scifact",
        "qasper_json": qasper_json,
        "qasper_cache": root,
        "qasper_rankings": project_rankings if project_rankings.exists() else context_rankings,
        "longmemeval_json": root / "v56" / "longmemeval_s_cleaned.json",
        "longbench_json": root / "v56" / "longbench_v2_data.json",
        "longbench_predictions": root / "v56" / "runs" / "longbench-context-v56" / "predictions.json",
    }
    return {key: path.resolve() for key, path in values.items() if _input_exists(key, path)}


def _input_exists(key, path):
    if key == "scifact_dir":
        return all((path / relative).exists() for relative in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"))
    return path.exists()


def _latest_result_path(benchmark_id, root):
    candidates = {
        "scifact-retrieval-v55": [PROJECT_ROOT / "results/public_benchmarks/v55_scifact_real/metrics.json"],
        "qasper-retrieval-v55": [PROJECT_ROOT / "results/public_benchmarks/v55_qasper_test_real/metrics.json"],
        "qasper-answer-v55": [PROJECT_ROOT / "results/public_benchmarks/v55_qasper_answer_test_real/metrics.json"],
        "longmemeval-retrieval-v56": [root / "v56/runs/longmemeval-s-minilm/metrics.json"],
        "qasper-context-v56": [root / "v56/runs/qasper-context-v56/metrics.json"],
    }.get(benchmark_id, [])
    return next((path.resolve() for path in candidates if path.exists()), None)


def _read_json(path):
    if not path or not Path(path).exists():
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _tail(path, max_chars=12000):
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def _now():
    return datetime.now(timezone.utc).isoformat()
