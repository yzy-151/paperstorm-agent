import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .paperstorm_eval import EvalCase, evaluate_run, write_scorecards
from .paperstorm_qa import PaperStormKnowledgeBase, write_qa_artifact
from . import __version__


def _chat_node_observation(node, message, graph_run, event):
    """Return sanitized business input/output for one conversation graph node."""
    details = dict(event.get("details") or {})
    common_input = {"message": message}
    common_output = {
        "status": event.get("status", "success"),
        "duration_ms": event.get("duration_ms"),
    }
    if node == "classify":
        return common_input, {
            **common_output,
            "router_decision": graph_run.get("router_decision") or {},
        }, details
    if node == "memory_recall":
        return common_input, {
            **common_output,
            "memory_recall": graph_run.get("memory_recall") or {},
        }, details
    if node == "memory_candidate_write":
        return common_input, {
            **common_output,
            "memory_write": graph_run.get("memory_write")
            or {"status": "not_evaluated"},
        }, details
    if node == "knowledge_retrieval":
        evidence = graph_run.get("evidence") or []
        return common_input, {
            **common_output,
            "evidence_count": len(evidence),
            "evidence": evidence,
            "retrieval_stack": graph_run.get("retrieval_stack", ""),
            "retrieval_metadata": graph_run.get("retrieval_metadata") or {},
        }, details
    if node == "evidence_grade":
        return {"evidence_count": len(graph_run.get("evidence") or [])}, {
            **common_output,
            "evidence_grade": graph_run.get("evidence_grade") or {},
        }, details
    if node == "deep_research":
        return common_input, {
            **common_output,
            "used_task_id": graph_run.get("used_task_id", ""),
            "retrieval_triggered": bool(graph_run.get("retrieval_triggered")),
            "artifact_uri": graph_run.get("artifact_uri", ""),
        }, details
    if node in {"casual_chat", "memory_answer", "answer_with_citations"}:
        return common_input, {
            **common_output,
            "answer": graph_run.get("answer", ""),
            "citation_count": len(graph_run.get("citations") or []),
            "citations": graph_run.get("citations") or [],
            "grounded": bool(graph_run.get("grounded")),
        }, details
    if node == "final_trace":
        return {}, {
            **common_output,
            "route": graph_run.get("route", ""),
            "executed_nodes": graph_run.get("executed_nodes") or [],
        }, details
    return common_input, common_output, details


def _export_chat_generations(span, node, message, graph_run):
    if node == "classify":
        decision = graph_run.get("router_decision") or {}
        telemetry = decision.get("planner_telemetry") or {}
        if telemetry:
            with span.generation(
                "intent_router",
                model=str(telemetry.get("model") or "configured-router-model"),
                input={"message": message},
                metadata={"planner_status": decision.get("planner_status", "")},
            ) as generation:
                generation.end(
                    output={"router_decision": decision},
                    usage=telemetry.get("usage") or {},
                    cost_usd=telemetry.get("cost_usd"),
                    finish_reason=telemetry.get("finish_reason", ""),
                    error=(telemetry.get("error") or {}).get("message")
                    if isinstance(telemetry.get("error"), dict)
                    else telemetry.get("error"),
                    metadata={"latency_ms": telemetry.get("latency_ms")},
                )
    if node in {"casual_chat", "memory_answer", "answer_with_citations"}:
        llm_call = graph_run.get("llm_call") or {}
        if llm_call:
            with span.generation(
                "answer_generation",
                model=str(llm_call.get("model") or "configured-chat-model"),
                input={"message": message},
            ) as generation:
                generation.end(
                    output={"content": graph_run.get("answer", "")},
                    usage=llm_call.get("usage") or {},
                    cost_usd=llm_call.get("cost_usd"),
                    finish_reason=llm_call.get("finish_reason", ""),
                    error=(llm_call.get("error") or {}).get("message")
                    if isinstance(llm_call.get("error"), dict)
                    else llm_call.get("error"),
                    metadata={"latency_ms": llm_call.get("latency_ms")},
                )


class PaperStormTaskService:
    """File-backed service core for PaperStorm task APIs."""

    def __init__(
        self,
        root_dir,
        max_concurrent_tasks: int = 1,
        pipeline_runner=None,
        pdf_renderer=None,
        observability=None,
    ):
        from .paperstorm_benchmarks import BenchmarkRunManager
        from .paperstorm_observability import build_observability

        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.tasks_dir = self.root_dir / "tasks"
        self.results_dir = self.root_dir / "results"
        self.tasks_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        self.max_concurrent_tasks = max(1, int(max_concurrent_tasks))
        self.pipeline_runner = pipeline_runner
        self.pdf_renderer = pdf_renderer
        self.observability = observability or build_observability(self.root_dir)
        self.benchmark_runs = BenchmarkRunManager(
            self.root_dir, observability=self.observability
        )

    def get_observability_status(self):
        return self.observability.status()

    def get_benchmark_catalog(self):
        return self.benchmark_runs.catalog()

    def start_benchmark_run(
        self,
        benchmark_id: str,
        profile: str = "smoke",
        allow_paid_llm: bool = False,
    ):
        return self.benchmark_runs.start(
            benchmark_id,
            profile=profile,
            allow_paid_llm=allow_paid_llm,
        )

    def get_benchmark_run(self, run_id: str):
        return self.benchmark_runs.get(run_id)

    def cancel_benchmark_run(self, run_id: str):
        return self.benchmark_runs.cancel(run_id)

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
            "queue_index": self._next_queue_index(),
            "expected_keywords": expected_keywords or [],
            "forbidden_keywords": forbidden_keywords or [],
            "options": _redact(options),
        }
        self._write_state(task_id, state)
        return state

    def get_task(self, task_id: str):
        return self._read_state(task_id)

    def list_tasks(self, status: Optional[str] = None):
        tasks = []
        for path in sorted(self.tasks_dir.glob("*.json")):
            state = json.loads(path.read_text(encoding="utf-8"))
            if status is None or state.get("status") == status:
                tasks.append(state)
        return sorted(
            tasks,
            key=lambda item: (
                int(item.get("queue_index", 0)),
                item.get("created_at", ""),
            ),
        )

    def run_task(self, task_id: str):
        state = self._read_state(task_id)
        state.pop("error", None)
        state.pop("finished_at", None)
        state["status"] = "running"
        state["started_at"] = _now()
        state["updated_at"] = _now()
        self._write_state(task_id, state)
        with self.observability.trace(
            "paperstorm.research",
            input={"topic": state.get("topic"), "options": state.get("options", {})},
            metadata={
                "task_id": task_id,
                "run_mode": state.get("run_mode"),
                "retriever": state.get("retriever"),
                "output_language": state.get("output_language"),
                "version": __version__,
            },
            session_id=task_id,
            tags=["research", str(state.get("run_mode") or "")],
        ) as trace:
            try:
                with trace.span(
                    "research_pipeline",
                    input={"topic": state.get("topic")},
                    metadata={"retriever": state.get("retriever")},
                    as_type="chain",
                ) as pipeline_span:
                    if state.get("run_mode") == "fail":
                        raise RuntimeError("simulated task failure for service testing")
                    if state.get("run_mode") == "manual":
                        pipeline_span.end(output={"status": "manual"})
                        trace.end(output={"status": "running"})
                        return state
                    if state.get("run_mode") == "paperstorm":
                        self._run_paperstorm_pipeline(
                            state, observation_parent=pipeline_span
                        )
                    elif state.get("run_mode") != "fake":
                        raise ValueError(
                            "Supported run modes are 'fake', 'paperstorm', 'manual', and 'fail'."
                        )
                    else:
                        self._run_fake_research(state)
                    with pipeline_span.span(
                        "pdf_export",
                        input={
                            "generate_pdf": bool(
                                (state.get("options") or {}).get("generate_pdf", False)
                            )
                        },
                        as_type="chain",
                    ) as pdf_span:
                        self._maybe_generate_pdf(state)
                        pdf_span.end(
                            output={
                                "pdf": (state.get("artifacts") or {}).get("pdf", {})
                            }
                        )
                    pipeline_span.end(
                        output={
                            "status": "succeeded",
                            "pdf": (state.get("artifacts") or {}).get("pdf", {}),
                        }
                    )
                state["status"] = "succeeded"
                state["finished_at"] = _now()
                scorecard = self.get_scorecard(task_id)
                total = (scorecard.get("scores") or {}).get("total")
                if isinstance(total, (int, float)):
                    trace.score("run_score", float(total))
                trace.score("run_success", 1.0)
                trace.end(output={"status": "succeeded", "scorecard": scorecard})
            except Exception as error:
                state["status"] = "failed"
                state["finished_at"] = _now()
                state["error"] = _redact_error(str(error))
                trace.score("run_success", 0.0, comment=state["error"])
                trace.end(output={"status": "failed"}, error=error)
        state["updated_at"] = _now()
        self._write_state(task_id, state)
        return state

    def worker_tick(self):
        running = self._list_tasks_by_status("running")
        capacity = max(0, self.max_concurrent_tasks - len(running))
        queued = self._list_tasks_by_status("queued")
        started = []
        for state in queued[:capacity]:
            state["status"] = "running"
            state["started_at"] = _now()
            state["updated_at"] = _now()
            self._write_state(state["task_id"], state)
            started.append(state["task_id"])
        return {
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "running_count": len(running) + len(started),
            "queued_count": max(0, len(queued) - len(started)),
            "started_count": len(started),
            "started_task_ids": started,
        }

    def complete_task(self, task_id: str, success: bool = True, error: str = ""):
        state = self._read_state(task_id)
        state["status"] = "succeeded" if success else "failed"
        if not success:
            state["error"] = _redact_error(error or "task failed")
        state["finished_at"] = _now()
        state["updated_at"] = _now()
        self._write_state(task_id, state)
        return state

    def recover_stale_running_tasks(self, max_age_seconds: float):
        now_ts = time.time()
        failed = []
        for state in self._list_tasks_by_status("running"):
            started_at = _parse_timestamp(state.get("started_at")) or 0.0
            if now_ts - started_at >= max_age_seconds:
                state["status"] = "failed"
                state["error"] = "stale running task recovered after timeout"
                state["finished_at"] = _now()
                state["updated_at"] = _now()
                self._write_state(state["task_id"], state)
                failed.append(state["task_id"])
        return {"failed_count": len(failed), "failed_task_ids": failed}

    def run_stress_benchmark(self, total_tasks: int, fail_every: int = 0):
        created = []
        latencies = []
        max_observed_running = 0
        for index in range(total_tasks):
            mode = "fail" if fail_every and (index + 1) % fail_every == 0 else "fake"
            created.append(
                self.submit_research_task(
                    topic="stress topic {0}".format(index + 1),
                    run_mode=mode,
                )["task_id"]
            )
        while True:
            queued = self._list_tasks_by_status("queued")
            running = self._list_tasks_by_status("running")
            max_observed_running = max(max_observed_running, len(running))
            if not queued and not running:
                break
            batch = self.worker_tick()
            max_observed_running = max(max_observed_running, batch["running_count"])
            for task_id in batch["started_task_ids"]:
                start = time.time()
                state = self._read_state(task_id)
                if state.get("run_mode") == "manual":
                    self.complete_task(task_id, success=True)
                else:
                    self.run_task(task_id)
                latencies.append(time.time() - start)
        states = [self._read_state(task_id) for task_id in created]
        succeeded = len([state for state in states if state["status"] == "succeeded"])
        failed = len([state for state in states if state["status"] == "failed"])
        report = {
            "total_tasks": total_tasks,
            "succeeded": succeeded,
            "failed": failed,
            "failure_rate": round(failed / max(1, total_tasks), 4),
            "avg_latency_sec": round(sum(latencies) / max(1, len(latencies)), 4),
            "p95_latency_sec": round(_percentile(latencies, 0.95), 4),
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "max_observed_running": max_observed_running,
            "retry_count": 0,
        }
        (self.root_dir / "stress_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return report

    def get_article(self, task_id: str):
        from .paperstorm_references import (
            append_reference_section,
            load_reference_registry,
        )

        state = self._read_state(task_id)
        output_dir = Path(state["output_dir"])
        path = _first_existing(
            [
                output_dir / "storm_gen_article_polished.txt",
                output_dir / "storm_gen_article.txt",
            ]
        )
        references = load_reference_registry(output_dir)
        content = path.read_text(encoding="utf-8", errors="replace") if path else ""
        return {
            "task_id": task_id,
            "path": str(path) if path else "",
            "content": append_reference_section(content, references),
            "references": references,
        }

    def get_scorecard(self, task_id: str):
        state = self._read_state(task_id)
        return _read_json(Path(state["output_dir"]) / "scorecard.json", {})

    def get_trace(self, task_id: str):
        state = self._read_state(task_id)
        trace_path = Path(state["output_dir"]) / "paperstorm_trace.jsonl"
        return {"task_id": task_id, "events": _load_jsonl(trace_path)}

    def get_dashboard_bundle(self, task_id: str):
        state = self._read_state(task_id)
        output_dir = Path(state["output_dir"])
        return {
            "project": {
                "name": "PaperStorm Agent",
                "version": "v{0}".format(__version__.rsplit(".", 1)[0]),
                "description": "Service-backed PaperStorm dashboard snapshot",
            },
            "tasks": [state],
            "article": self.get_article(task_id),
            "qa": _read_json(output_dir / "qa_answer.json", {}),
            "scorecard": self.get_scorecard(task_id),
            "trace": self.get_trace(task_id),
            "process": self.get_process_artifacts(task_id),
            "artifacts": self.get_artifacts(task_id),
            "pipeline_worker": _read_json(output_dir / "pipeline_worker.json", {}),
            "service_snapshot": {
                "task_id": task_id,
                "output_dir": str(output_dir),
                "status": state.get("status", ""),
                "run_mode": state.get("run_mode", ""),
                "retriever": state.get("retriever", ""),
                "updated_at": state.get("updated_at", ""),
            },
        }

    def get_artifacts(self, task_id: str):
        state = self._read_state(task_id)
        output_dir = Path(state["output_dir"])
        stored = dict(state.get("artifacts") or {})
        pdf_path = output_dir / "paperstorm_report.pdf"
        pdf = dict(stored.get("pdf") or {})
        if pdf_path.exists() and not pdf:
            pdf = {
                "status": "ready",
                "name": pdf_path.name,
                "size_bytes": pdf_path.stat().st_size,
            }
        pdf.setdefault("status", "not_requested")
        if pdf.get("status") == "ready":
            pdf["url"] = "/research-tasks/{0}/artifacts/{1}".format(
                task_id, pdf_path.name
            )
        return {
            "markdown": {
                "status": "ready" if self.get_article(task_id)["path"] else "missing",
                "url": "/research-tasks/{0}/article".format(task_id),
            },
            "pdf": pdf,
        }

    def get_artifact_path(self, task_id: str, artifact_name: str):
        allowed = {
            "paperstorm_report.pdf",
            "paperstorm_report.print.html",
        }
        if artifact_name not in allowed:
            raise PermissionError("Artifact is not allow-listed: {0}".format(artifact_name))
        state = self._read_state(task_id)
        output_dir = Path(state["output_dir"]).resolve()
        path = (output_dir / artifact_name).resolve()
        if path.parent != output_dir:
            raise PermissionError("Artifact path escapes the task directory.")
        if not path.is_file():
            raise FileNotFoundError("Artifact does not exist: {0}".format(artifact_name))
        return path

    def query_knowledge_base(
        self,
        task_id: str,
        question: str,
        top_k: int = 3,
        history=None,
        **retrieval_options,
    ):
        from .paperstorm_router_llm import build_chat_llm_callable

        state = self._read_state(task_id)
        output_dir = Path(state["output_dir"])
        kb = PaperStormKnowledgeBase.from_run_dir(output_dir)
        retrieval_options = dict(retrieval_options)
        retrieval_options.setdefault(
            "parent_budget_tokens",
            1600 if state.get("retriever") == "local-pdf" else 900,
        )
        retrieval_options["history"] = tuple(history or ())
        search_plan = retrieval_options.get("search_plan")
        if search_plan is not None:
            from .search_planning import SearchPlan

            if isinstance(search_plan, dict):
                retrieval_options["search_plan"] = SearchPlan.from_mapping(search_plan)
        answer = kb.answer_question(
            question,
            top_k=top_k,
            answer_generator=build_chat_llm_callable(
                enabled=state.get("run_mode") == "paperstorm"
            ),
            retrieval_options=retrieval_options,
        )
        write_qa_artifact(output_dir, answer)
        return answer

    def create_enterprise_knowledge_base(
        self,
        name: str,
        source_paths: List[str],
        expected_keywords: Optional[List[str]] = None,
        forbidden_keywords: Optional[List[str]] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        embedding_provider: str = "hash",
        tenant_id: str = "local",
        owner_user_id: str = "local-user",
        allowed_user_ids: Optional[List[str]] = None,
    ):
        from .paperstorm_enterprise_kb import EnterpriseKnowledgeBaseService

        return EnterpriseKnowledgeBaseService(self.root_dir).create_knowledge_base(
            name=name,
            source_paths=source_paths,
            expected_keywords=expected_keywords,
            forbidden_keywords=forbidden_keywords,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_provider=embedding_provider,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            allowed_user_ids=allowed_user_ids,
        )

    def create_enterprise_knowledge_base_from_zotero(
        self,
        zotero_root: Optional[str] = None,
        query_terms: Optional[List[str]] = None,
        max_papers: int = 8,
        name: str = "Zotero 论文知识库",
        expected_keywords: Optional[List[str]] = None,
        forbidden_keywords: Optional[List[str]] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        embedding_provider: str = "hash",
        tenant_id: str = "local",
        owner_user_id: str = "local-user",
        allowed_user_ids: Optional[List[str]] = None,
    ):
        """Create an enterprise KB directly from the local Zotero library.

        The Zotero data directory resolves in this order: explicit argument ->
        PAPERSTORM_ZOTERO_ROOT -> repo-local local_zotero_root.txt -> ~/Zotero.
        """
        from .paperstorm_zotero import discover_zotero_papers

        root = self._resolve_zotero_root(zotero_root)
        papers = discover_zotero_papers(
            root,
            query_terms=query_terms,
            max_papers=max_papers,
        )
        if not papers:
            raise ValueError("Zotero 中没有匹配的 PDF 论文：请检查目录与检索词")
        result = self.create_enterprise_knowledge_base(
            name=name,
            source_paths=[item["path"] for item in papers],
            expected_keywords=expected_keywords,
            forbidden_keywords=forbidden_keywords,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_provider=embedding_provider,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            allowed_user_ids=allowed_user_ids,
        )
        result["zotero_root"] = root
        result["source_papers"] = [
            {"title": item.get("title") or "", "path": item.get("path") or ""}
            for item in papers
        ]
        return result

    def _resolve_zotero_root(self, zotero_root: Optional[str] = None) -> str:
        candidates = []
        if zotero_root:
            candidates.append(Path(zotero_root))
        if os.getenv("PAPERSTORM_ZOTERO_ROOT"):
            candidates.append(Path(os.getenv("PAPERSTORM_ZOTERO_ROOT")))
        local_file = Path(__file__).resolve().parents[1] / "local_zotero_root.txt"
        if local_file.exists():
            value = local_file.read_text(encoding="utf-8").strip()
            if value:
                candidates.append(Path(value))
        candidates.append(Path.home() / "Zotero")
        for candidate in candidates:
            if (candidate / "zotero.sqlite").exists():
                return str(candidate)
        raise ValueError(
            "未找到 Zotero 数据目录：请填写目录，或设置 PAPERSTORM_ZOTERO_ROOT，"
            "或在项目根目录放 local_zotero_root.txt"
        )

    def get_enterprise_knowledge_base(
        self,
        kb_id: str,
        tenant_id: str = "local",
        user_id: str = "local-user",
    ):
        from .paperstorm_enterprise_kb import EnterpriseKnowledgeBaseService

        return EnterpriseKnowledgeBaseService(self.root_dir).get_knowledge_base(
            kb_id, tenant_id=tenant_id, user_id=user_id
        )

    def list_enterprise_knowledge_bases(
        self, tenant_id: str = "local", user_id: str = "local-user"
    ):
        from .paperstorm_enterprise_kb import EnterpriseKnowledgeBaseService

        return EnterpriseKnowledgeBaseService(self.root_dir).list_knowledge_bases(
            tenant_id=tenant_id, user_id=user_id
        )

    def ask_enterprise_knowledge_base(
        self,
        kb_id: str,
        question: str,
        top_k: int = 4,
        tenant_id: str = "local",
        user_id: str = "local-user",
    ):
        from .paperstorm_router_llm import build_chat_llm_callable

        from .paperstorm_enterprise_kb import EnterpriseKnowledgeBaseService

        return EnterpriseKnowledgeBaseService(self.root_dir).ask(
            kb_id=kb_id,
            question=question,
            top_k=top_k,
            tenant_id=tenant_id,
            user_id=user_id,
            answer_generator=build_chat_llm_callable(),
        )

    def enqueue_enterprise_kb_update(
        self,
        kb_id: str,
        source_paths: List[str],
        tenant_id: str,
        user_id: str,
        idempotency_key: str,
    ):
        control = self._production_control()
        control.authorize(tenant_id, user_id, "knowledge_base", kb_id, "write")
        return control.enqueue_job(
            tenant_id=tenant_id,
            job_type="incremental_index",
            payload={
                "kb_id": kb_id,
                "source_paths": source_paths,
                "tenant_id": tenant_id,
                "user_id": user_id,
            },
            idempotency_key=idempotency_key,
            max_attempts=3,
        )

    def run_production_worker_tick(self):
        from .paperstorm_enterprise_kb import EnterpriseKnowledgeBaseService

        enterprise = EnterpriseKnowledgeBaseService(
            self.root_dir, control_plane=self._production_control()
        )
        return self._production_control().run_worker_tick(
            {
                "incremental_index": lambda payload: enterprise.update_knowledge_base(
                    **payload
                )
            }
        )

    def ask_research_agent(
        self,
        question: str,
        topic: Optional[str] = None,
        task_id: Optional[str] = None,
        mode: str = "auto",
        top_k: int = 3,
        run_mode: str = "fake",
        retriever: str = "arxiv",
        output_language: str = "zh",
        expected_keywords: Optional[List[str]] = None,
        forbidden_keywords: Optional[List[str]] = None,
        **options,
    ):
        from .paperstorm_research_qa import ResearchQAAgent

        return ResearchQAAgent(self).ask(
            question=question,
            topic=topic,
            task_id=task_id,
            mode=mode,
            top_k=top_k,
            run_mode=run_mode,
            retriever=retriever,
            output_language=output_language,
            expected_keywords=expected_keywords,
            forbidden_keywords=forbidden_keywords,
            **options,
        )

    def create_chat_session(
        self,
        title: str = "",
        topic: str = "",
        run_mode: str = "fake",
        retriever: str = "arxiv",
        output_language: str = "zh",
        expected_keywords: Optional[List[str]] = None,
        forbidden_keywords: Optional[List[str]] = None,
        context_window_size: int = 48,
        context_token_limit: int = 1_000_000,
        user_id: str = "local-user",
        tenant_id: str = "local",
        memory_enabled: bool = True,
        memory_retrieval_mode: str = "lexical",
        **options,
    ):
        from .paperstorm_chat_agent import PaperStormChatAgent

        return PaperStormChatAgent(self).create_session(
            title=title,
            topic=topic,
            run_mode=run_mode,
            retriever=retriever,
            output_language=output_language,
            expected_keywords=expected_keywords,
            forbidden_keywords=forbidden_keywords,
            context_window_size=context_window_size,
            context_token_limit=context_token_limit,
            user_id=user_id,
            tenant_id=tenant_id,
            memory_enabled=memory_enabled,
            memory_retrieval_mode=memory_retrieval_mode,
            **options,
        )

    def get_chat_session(self, chat_id: str):
        from .paperstorm_chat_agent import PaperStormChatAgent

        return PaperStormChatAgent(self).get_session(chat_id)

    def send_chat_message(self, chat_id: str, message: str):
        from .paperstorm_chat_agent import PaperStormChatAgent

        session = PaperStormChatAgent(self).get_session(chat_id)
        with self.observability.trace(
            "paperstorm.chat",
            input={"message": message, "topic": session.get("topic", "")},
            metadata={
                "chat_id": chat_id,
                "run_mode": session.get("run_mode", ""),
                "retriever": session.get("retriever", ""),
                "version": __version__,
            },
            session_id=chat_id,
            user_id=session.get("user_id", ""),
            tags=["chat", str(session.get("run_mode") or "")],
        ) as trace:
            try:
                result = PaperStormChatAgent(self).send_message(chat_id, message)
                graph_run = result.get("graph_run") or {}
                events = graph_run.get("node_events") or []
                event_by_node = {event.get("node"): event for event in events}
                for node in graph_run.get("executed_nodes") or []:
                    event = event_by_node.get(node) or {}
                    span_input, span_output, span_metadata = _chat_node_observation(
                        node=node,
                        message=message,
                        graph_run=graph_run,
                        event=event,
                    )
                    with trace.span(
                        node,
                        input=span_input,
                        metadata={
                            "status": event.get("status", "success"),
                            "duration_ms": event.get("duration_ms"),
                            **span_metadata,
                        },
                        as_type="chain",
                    ) as span:
                        _export_chat_generations(span, node, message, graph_run)
                        span.end(output=span_output)
                router = graph_run.get("router_decision") or {}
                planner_status = str(router.get("planner_status") or "")
                trace.score(
                    "planner_fallback",
                    1.0
                    if planner_status
                    in {"fallback", "guarded_fallback", "offline_fallback"}
                    else 0.0,
                    comment=str((router.get("planner_error") or {}).get("message") or ""),
                )
                memory_write = graph_run.get("memory_write") or {}
                if router.get("intent") == "memory_write":
                    trace.score(
                        "memory_write_success",
                        1.0 if memory_write.get("status") == "persisted" else 0.0,
                        comment=str(
                            memory_write.get("reason")
                            or memory_write.get("status")
                            or "not_evaluated"
                        ),
                    )
                trace.score(
                    "trajectory_success",
                    1.0 if graph_run.get("status") == "succeeded" else 0.0,
                )
                trace.score(
                    "retrieval_triggered",
                    1.0 if result.get("retrieval_triggered") else 0.0,
                )
                trace.end(
                    output={
                        "answer": (result.get("assistant_message") or {}).get("content", ""),
                        "route": graph_run.get("route", ""),
                        "executed_nodes": graph_run.get("executed_nodes") or [],
                    }
                )
                return result
            except Exception as error:
                trace.score("trajectory_success", 0.0, comment=str(error))
                trace.end(output={"status": "failed"}, error=error)
                raise

    def list_chat_sessions(self, limit: int = 50):
        from .paperstorm_chat_agent import PaperStormChatAgent

        return PaperStormChatAgent(self).list_sessions(limit=limit)

    def regenerate_chat_message(self, chat_id: str):
        from .paperstorm_chat_agent import PaperStormChatAgent

        return PaperStormChatAgent(self).regenerate_last(chat_id)

    def stop_chat_generation(self, chat_id: str):
        from .paperstorm_chat_agent import PaperStormChatAgent

        return PaperStormChatAgent(self).stop_generation(chat_id)

    def get_chat_context(self, chat_id: str):
        from .paperstorm_chat_agent import PaperStormChatAgent

        return PaperStormChatAgent(self).get_context(chat_id)

    def compact_chat_context(self, chat_id: str, force: bool = True):
        from .paperstorm_chat_agent import PaperStormChatAgent

        return PaperStormChatAgent(self).compact_context(chat_id, force=force)

    def restore_chat_context(self, chat_id: str, compaction_id: str):
        from .paperstorm_chat_agent import PaperStormChatAgent

        return PaperStormChatAgent(self).restore_context(chat_id, compaction_id)

    def create_memory(self, **payload):
        return self._memory_service().upsert(**payload)

    def list_memories(self, namespace: str, include_inactive: bool = False):
        return {
            "namespace": namespace,
            "memories": self._memory_service().list_memories(
                namespace, include_inactive=include_inactive
            ),
        }

    def search_memories(self, namespace: str, query: str, top_k: int = 5):
        return self._memory_service().search(namespace, query, top_k=top_k)

    def edit_memory(self, namespace: str, memory_id: str, content: str, **updates):
        return self._memory_service().edit(
            namespace=namespace, memory_id=memory_id, content=content, **updates
        )

    def delete_memory(
        self, namespace: str, memory_id: str, reason: str = "user_request"
    ):
        return self._memory_service().delete(namespace, memory_id, reason=reason)

    def export_memories(self, namespace: str):
        return self._memory_service().export_namespace(namespace)

    def set_memory_enabled(self, namespace: str, enabled: bool):
        return self._memory_service().set_enabled(namespace, enabled)

    def invoke_conversation_graph(self, **payload):
        return self._production_runtime().invoke(**payload)

    def get_conversation_graph_spec(self):
        return self._production_runtime().get_graph_spec()

    def get_conversation_thread_state(
        self, thread_id: str, tenant_id: str = "local", user_id: str = "local-user"
    ):
        self._production_control().authorize(
            tenant_id, user_id, "conversation_thread", thread_id, "read_state"
        )
        return self._production_runtime().get_thread_state(thread_id)

    def get_conversation_thread_history(
        self,
        thread_id: str,
        limit: int = 50,
        tenant_id: str = "local",
        user_id: str = "local-user",
    ):
        self._production_control().authorize(
            tenant_id, user_id, "conversation_thread", thread_id, "read_history"
        )
        return self._production_runtime().get_thread_history(thread_id, limit=limit)

    def get_production_trace(self, trace_id: str, tenant_id: str, user_id: str):
        self._production_control().authorize(
            tenant_id, user_id, "trace", trace_id, "read"
        )
        return {
            "trace_id": trace_id,
            "spans": self._production_control().list_spans(trace_id),
        }

    def get_production_status(self):
        return self._production_control().status()

    def _conversation_runtime(self):
        from .conversation_runtime import PaperStormConversationRuntime

        return PaperStormConversationRuntime(
            root_dir=self.root_dir / "conversation_runtime",
            task_service=self,
            memory_service=self._memory_service(),
        )

    def _production_runtime(self):
        from .control_plane import ProductionRuntime

        current = self.root_dir / "production_runtime"
        legacy = self.root_dir / "production_runtime_v45"
        return ProductionRuntime(
            root_dir=legacy if legacy.exists() else current,
            task_service=self,
            control_plane=self._production_control(),
        )

    def _production_control(self):
        from .control_plane import ProductionControlPlane

        current = self.root_dir / "production_control.sqlite"
        legacy = self.root_dir / "production_control_v45.sqlite"
        return ProductionControlPlane(legacy if legacy.exists() else current)

    def _memory_service(self):
        from .memory_store import LongTermMemoryService

        current = self.root_dir / "memory_service"
        legacy = self.root_dir / "memory_service_v56"
        return LongTermMemoryService(legacy if legacy.exists() else current)

    def _run_fake_research(self, state: Dict):
        output_dir = Path(state["output_dir"])
        topic = state["topic"]
        topic_lower = str(topic or "").lower()
        pim_topic = (
            "pim" in topic_lower
            or "无源互调" in topic_lower
            or "passive intermodulation" in topic_lower
        )
        if pim_topic:
            keyword = "passive intermodulation"
            framing = "是 RF 系统中由无源器件非线性导致的互调杂散问题"
        else:
            keyword = topic
            framing = "是该方向的核心研究问题"
        article = (
            "# {topic}\n\n"
            "围绕“{topic}”的调研结论：{keyword} {framing}；"
            "模型驱动与数据驱动（神经网络）方法可用于建模、抑制与对消，"
            "并需要可复现的 benchmark 验证效果。[1]\n\n"
            "工程化要点：混合检索（BM25+Dense+RRF）、证据门控、可恢复上下文"
            "与跨会话记忆共同保证回答可溯源。[2]\n"
        ).format(topic=topic, keyword=keyword, framing=framing)
        raw_results = [
            {
                "title": "{0} 研究综述".format(topic),
                "description": "围绕 {0} 的检索结果与关键方法。".format(topic),
                "url": "https://example.com/topic-0",
                "snippets": ["{0} 的模型驱动与数据驱动方法对比。".format(topic)],
            },
            {
                "title": "Neural methods for {0}".format(topic),
                "description": "Neural network approaches for {0} modeling and suppression.".format(
                    topic
                ),
                "url": "https://example.com/topic-1",
                "snippets": ["Neural modeling and cancellation for {0}.".format(topic)],
            },
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
        (output_dir / "conversation_log.json").write_text(
            json.dumps(
                (
                    [
                        {
                            "role": "researcher",
                            "message": "如何定义 RF 场景下的 PIM？",
                        },
                        {
                            "role": "expert",
                            "message": "这里 PIM 指 passive intermodulation，不是 processing-in-memory。",
                        },
                    ]
                    if pim_topic
                    else [
                        {
                            "role": "researcher",
                            "message": "如何界定这个主题的核心问题？",
                        },
                        {
                            "role": "expert",
                            "message": "围绕“{0}”，先给出定义与关键方法，"
                            "再比较模型驱动与数据驱动路线。".format(topic),
                        },
                    ]
                ),
                ensure_ascii=False,
                indent=2,
            ),
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
        (output_dir / "reflection.txt").write_text(
            (
                "Critic: 检索结果需要过滤 processing-in-memory / DRAM 语义，"
                "保留 RF passive intermodulation 方向。"
                if pim_topic
                else "Critic: 检索结果需要区分相关与跑题内容，围绕 {0} "
                "保留模型驱动与数据驱动两条主线，并记录证据来源。".format(topic)
            ),
            encoding="utf-8",
        )
        trace_events = [
            {
                "event": "run_start",
                "task_id": state["task_id"],
                "timestamp": _now(),
                "success": True,
            },
            {
                "event": "tool_start",
                "task_id": state["task_id"],
                "timestamp": _now(),
                "tool": "fake_research",
            },
        ]
        fake_stages = [
            ("request", "解析任务并初始化运行配置", {"topic": topic}, {"mode": "fake"}),
            ("persona", "生成多视角研究角色", {"topic": topic}, {"perspectives": ["领域专家"]}),
            ("dialogue", "执行多 Agent 研究对话", {"perspectives": 1}, {"turns": 1}),
            ("query", "规划检索查询", {"topic": topic}, {"query_count": 2}),
            ("retrieval", "检索论文证据", {"retriever": state.get("retriever")}, {"result_count": 2}),
            ("evidence", "清洗并组织证据", {"candidate_count": 2}, {"selected_count": 2}),
            ("outline", "生成并细化文章大纲", {"topic": topic}, {"sections": 2}),
            ("writer", "按章节撰写文章", {"sections": 2}, {"article": "storm_gen_article_polished.txt"}),
            ("polish", "去重并润色文章", {"article": "storm_gen_article_polished.txt"}, {"status": "polished"}),
            ("evaluate", "评估证据与文章质量", {"source_count": 2}, {"status": "scored"}),
            ("deliver", "整理 Markdown 交付物", {"article": "storm_gen_article_polished.txt"}, {"markdown": "ready"}),
        ]
        for stage, operation, stage_input, output_summary in fake_stages:
            trace_events.extend(
                [
                    {
                        "event": "stage_start",
                        "stage": stage,
                        "operation": operation,
                        "input": stage_input,
                        "timestamp": _now(),
                    },
                    {
                        "event": "stage_end",
                        "stage": stage,
                        "operation": operation,
                        "output_summary": output_summary,
                        "duration_ms": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "estimated_cost": 0.0,
                        "timestamp": _now(),
                    },
                ]
            )
        trace_events.extend(
            [
                {
                    "event": "tool_end",
                    "task_id": state["task_id"],
                    "timestamp": _now(),
                    "tool": "fake_research",
                },
                {
                    "event": "run_end",
                    "task_id": state["task_id"],
                    "timestamp": _now(),
                    "success": True,
                },
            ]
        )
        (output_dir / "paperstorm_trace.jsonl").write_text(
            "\n".join(json.dumps(event, ensure_ascii=False) for event in trace_events)
            + "\n",
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

    def get_process_artifacts(self, task_id: str):
        state = self._read_state(task_id)
        output_dir = Path(state["output_dir"])
        return {
            "outline": _read_text(output_dir / "storm_gen_outline.txt"),
            "conversation": _read_text(output_dir / "conversation_log.json"),
            "reflection": _read_text(output_dir / "reflection.txt"),
            "run_summary": _read_text(output_dir / "run_summary.json"),
            "raw_search_results": _read_text(output_dir / "raw_search_results.json"),
            "plan": _read_text(output_dir / "query_plan.json")
            or _read_text(output_dir / "raw_search_results.json"),
        }

    def _run_paperstorm_pipeline(self, state: Dict, observation_parent=None):
        runner = self.pipeline_runner
        if runner is None:
            from .paperstorm_pipeline import run_paperstorm_pipeline_task

            return run_paperstorm_pipeline_task(
                state, observation_parent=observation_parent
            )
        return runner(state)

    def _maybe_generate_pdf(self, state: Dict):
        options = state.get("options") or {}
        artifacts = state.setdefault("artifacts", {})
        if not bool(options.get("generate_pdf", False)):
            artifacts["pdf"] = {"status": "not_requested"}
            return

        output_dir = Path(state["output_dir"])
        article = _first_existing(
            [
                output_dir / "storm_gen_article_polished.txt",
                output_dir / "storm_gen_article.txt",
            ]
        )
        pdf_path = output_dir / "paperstorm_report.pdf"
        self._append_trace_event(
            state,
            {
                "event": "stage_start",
                "stage": "deliver",
                "operation": "将 Markdown 文章渲染为 PDF",
                "input": {
                    "article": article.name if article else "",
                    "output": pdf_path.name,
                },
            },
        )
        try:
            if article is None:
                from .paperstorm_pdf import PdfRenderError

                raise PdfRenderError(
                    "pdf_source_missing", "没有找到可用于生成 PDF 的文章。"
                )
            renderer = self.pdf_renderer
            if renderer is None:
                from .paperstorm_pdf import PaperStormPdfRenderer

                renderer = PaperStormPdfRenderer()
            result = renderer.render(
                markdown_path=article,
                output_pdf=pdf_path,
                title=state.get("topic") or "PaperStorm 调研报告",
            )
            artifacts["pdf"] = {
                "status": "ready",
                "name": pdf_path.name,
                "page_count": int(result.get("page_count", 0)),
                "text_length": int(result.get("text_length", 0)),
                "size_bytes": int(result.get("size_bytes", pdf_path.stat().st_size)),
                "html_name": Path(
                    result.get("html_path") or pdf_path.with_suffix(".print.html")
                ).name,
            }
            self._append_trace_event(
                state,
                {
                    "event": "artifact_written",
                    "stage": "deliver",
                    "path": str(pdf_path),
                    "artifact_name": pdf_path.name,
                },
            )
            self._append_trace_event(
                state,
                {
                    "event": "stage_end",
                    "stage": "deliver",
                    "operation": "PDF 交付物生成完成",
                    "output_summary": {
                        "pdf": "ready",
                        "page_count": artifacts["pdf"]["page_count"],
                        "size_bytes": artifacts["pdf"]["size_bytes"],
                    },
                },
            )
        except Exception as error:
            error_type = getattr(error, "code", "pdf_render_error")
            artifacts["pdf"] = {
                "status": "failed",
                "error_type": error_type,
                "error_message": _redact_error(str(error)),
            }
            self._append_trace_event(
                state,
                {
                    "event": "stage_error",
                    "stage": "deliver",
                    "operation": "PDF 交付物生成失败",
                    "error_type": error_type,
                    "error_message": _redact_error(str(error)),
                },
            )

    def _append_trace_event(self, state: Dict, event: Dict):
        trace_path = Path(state["output_dir"]) / "paperstorm_trace.jsonl"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"ts": _now(), **_redact(event)}
        with trace_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

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

    def _list_tasks_by_status(self, status: str):
        tasks = []
        for path in sorted(self.tasks_dir.glob("*.json")):
            state = json.loads(path.read_text(encoding="utf-8"))
            if state.get("status") == status:
                tasks.append(state)
        return sorted(
            tasks,
            key=lambda item: (
                int(item.get("queue_index", 0)),
                item.get("created_at", ""),
            ),
        )

    def _next_queue_index(self):
        return len(list(self.tasks_dir.glob("*.json"))) + 1


def _now():
    return datetime.now(timezone.utc).isoformat()


def _parse_timestamp(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def _percentile(values, percentile):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * percentile))
    return ordered[index]


def _redact(value):
    if isinstance(value, dict):
        return {key: _redact_secret(key, _redact(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _redact_secret(key, value):
    lowered = str(key).lower()
    if lowered in {"api_key", "apikey", "access_key", "secret_key"}:
        return "***REDACTED***"
    if (
        "token" in lowered
        or "secret" in lowered
        or "password" in lowered
        or lowered.endswith("_key")
    ):
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


def _read_text(path: Path):
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


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
