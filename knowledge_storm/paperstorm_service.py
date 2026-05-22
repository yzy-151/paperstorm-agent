import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .paperstorm_eval import EvalCase, evaluate_run, write_scorecards
from .paperstorm_qa import PaperStormKnowledgeBase, write_qa_artifact


class PaperStormTaskService:
    """File-backed service core for PaperStorm task APIs."""

    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.tasks_dir = self.root_dir / "tasks"
        self.results_dir = self.root_dir / "results"
        self.tasks_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)

    def submit_research_task(
        self,
        topic: str,
        retriever: str = "arxiv",
        output_language: str = "zh",
        run_mode: str = "fake",
        expected_keywords: Optional[List[str]] = None,
        forbidden_keywords: Optional[List[str]] = None,
        **options,
    ):
        task_id = uuid.uuid4().hex
        output_dir = self.results_dir / task_id
        output_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "task_id": task_id,
            "topic": topic,
            "retriever": retriever,
            "output_language": output_language,
            "run_mode": run_mode,
            "status": "queued",
            "output_dir": str(output_dir),
            "created_at": _now(),
            "updated_at": _now(),
            "expected_keywords": expected_keywords or [],
            "forbidden_keywords": forbidden_keywords or [],
            "options": _redact(options),
        }
        self._write_state(task_id, state)
        return state

    def get_task(self, task_id: str):
        return self._read_state(task_id)

    def run_task(self, task_id: str):
        state = self._read_state(task_id)
        state["status"] = "running"
        state["started_at"] = _now()
        state["updated_at"] = _now()
        self._write_state(task_id, state)
        try:
            if state.get("run_mode") == "fail":
                raise RuntimeError("simulated task failure for service testing")
            if state.get("run_mode") != "fake":
                raise ValueError("Only run_mode='fake' is supported by the service core.")
            self._run_fake_research(state)
            state["status"] = "succeeded"
            state["finished_at"] = _now()
        except Exception as error:
            state["status"] = "failed"
            state["finished_at"] = _now()
            state["error"] = _redact_error(str(error))
        state["updated_at"] = _now()
        self._write_state(task_id, state)
        return state

    def get_article(self, task_id: str):
        state = self._read_state(task_id)
        output_dir = Path(state["output_dir"])
        path = _first_existing(
            [
                output_dir / "storm_gen_article_polished.txt",
                output_dir / "storm_gen_article.txt",
            ]
        )
        return {
            "task_id": task_id,
            "path": str(path) if path else "",
            "content": path.read_text(encoding="utf-8", errors="replace") if path else "",
        }

    def get_scorecard(self, task_id: str):
        state = self._read_state(task_id)
        return _read_json(Path(state["output_dir"]) / "scorecard.json", {})

    def get_trace(self, task_id: str):
        state = self._read_state(task_id)
        trace_path = Path(state["output_dir"]) / "paperstorm_trace.jsonl"
        return {"task_id": task_id, "events": _load_jsonl(trace_path)}

    def query_knowledge_base(self, task_id: str, question: str, top_k: int = 3):
        state = self._read_state(task_id)
        output_dir = Path(state["output_dir"])
        kb = PaperStormKnowledgeBase.from_run_dir(output_dir)
        answer = kb.answer_question(question, top_k=top_k)
        write_qa_artifact(output_dir, answer)
        return answer

    def _run_fake_research(self, state: Dict):
        output_dir = Path(state["output_dir"])
        topic = state["topic"]
        article = (
            "# {topic}\n\n"
            "PIM 在本任务中指 passive intermodulation，是 RF 系统中由无源器件"
            "非线性导致的互调杂散问题。[1]\n\n"
            "神经网络方法可以学习非线性抵消器，用于 passive intermodulation "
            "suppression 和 cancellation。[2]\n"
        ).format(topic=topic)
        raw_results = [
            {
                "title": "Neural passive intermodulation cancellation",
                "description": "RF passive intermodulation suppression with neural networks.",
                "url": "https://example.com/pim",
                "snippets": ["Neural cancellers reduce passive intermodulation products."],
            }
        ]
        summary = {
            "success": True,
            "task_id": state["task_id"],
            "topic": topic,
            "artifacts": [
                "storm_gen_article_polished.txt",
                "raw_search_results.json",
                "paperstorm_trace.jsonl",
            ],
        }
        (output_dir / "storm_gen_outline.txt").write_text(
            "# {0}\n## 定义\n## 神经网络抑制".format(topic),
            encoding="utf-8",
        )
        (output_dir / "storm_gen_article_polished.txt").write_text(
            article,
            encoding="utf-8",
        )
        (output_dir / "raw_search_results.json").write_text(
            json.dumps(raw_results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / "run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        trace_events = [
            {"event": "run_start", "task_id": state["task_id"], "success": True},
            {"event": "tool_start", "task_id": state["task_id"], "tool": "fake_research"},
            {"event": "tool_end", "task_id": state["task_id"], "tool": "fake_research"},
            {"event": "run_end", "task_id": state["task_id"], "success": True},
        ]
        (output_dir / "paperstorm_trace.jsonl").write_text(
            "\n".join(json.dumps(event, ensure_ascii=False) for event in trace_events) + "\n",
            encoding="utf-8",
        )
        case = EvalCase(
            topic=topic,
            expected_keywords=state.get("expected_keywords")
            or ["passive intermodulation", "RF"],
            forbidden_keywords=state.get("forbidden_keywords")
            or ["processing-in-memory", "DRAM", "RAM"],
            expected_language=state.get("output_language", "zh"),
            min_sources=1,
        )
        write_scorecards(output_dir, evaluate_run(output_dir, case))

    def _state_path(self, task_id: str):
        return self.tasks_dir / "{0}.json".format(task_id)

    def _read_state(self, task_id: str):
        path = self._state_path(task_id)
        if not path.exists():
            raise KeyError("Unknown task_id: {0}".format(task_id))
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_state(self, task_id: str, state: Dict):
        self._state_path(task_id).write_text(
            json.dumps(_redact(state), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _now():
    return datetime.now(timezone.utc).isoformat()


def _redact(value):
    if isinstance(value, dict):
        return {key: _redact_secret(key, _redact(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _redact_secret(key, value):
    lowered = str(key).lower()
    if "key" in lowered or "token" in lowered or "secret" in lowered:
        return "***REDACTED***"
    return value


def _redact_error(message: str):
    return re.sub(r"sk-[A-Za-z0-9_\-]+", "sk-***REDACTED***", message)


def _first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _load_jsonl(path: Path):
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            events.append({"event": "decode_error", "raw": line})
    return events
