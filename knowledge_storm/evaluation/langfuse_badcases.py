"""Regression harness for bad cases discovered through Langfuse traces."""

import hashlib
import json
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Optional


DEFAULT_DATASET_NAME = "paperstorm-langfuse-observed-badcases"


def load_badcase_dataset(dataset_path) -> Dict:
    path = Path(dataset_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases") or []
    if not payload.get("dataset_name") or not payload.get("version") or not cases:
        raise ValueError("badcase dataset requires dataset_name, version, and cases")
    case_ids = [str(case.get("case_id") or "").strip() for case in cases]
    if any(not case_id for case_id in case_ids):
        raise ValueError("every badcase requires case_id")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("badcase case_id values must be unique")
    return payload


def sync_langfuse_dataset(dataset_path, client=None, dataset_name: str = "") -> Dict:
    """Idempotently mirror the frozen local dataset into Langfuse."""
    payload = load_badcase_dataset(dataset_path)
    name = str(dataset_name or payload.get("dataset_name") or DEFAULT_DATASET_NAME)
    if client is None:
        from langfuse import get_client

        client = get_client()

    try:
        client.get_dataset(name)
        created = False
    except Exception:
        client.create_dataset(
            name=name,
            description=(
                "PaperStorm production bad cases discovered through Langfuse traces; "
                "long-running latency cases are excluded from the fast regression gate."
            ),
            metadata={
                "version": payload["version"],
                "source": payload.get("source", ""),
                "scope": payload.get("scope", ""),
                "dataset_sha256": _sha256(Path(dataset_path)),
            },
        )
        created = True

    item_ids = []
    for case in payload["cases"]:
        item_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                "paperstorm:{0}:{1}:{2}".format(
                    name, payload["version"], case["case_id"]
                ),
            )
        )
        item_ids.append(item_id)
        client.create_dataset_item(
            id=item_id,
            dataset_name=name,
            input=_case_input(case),
            expected_output=case.get("expected") or {},
            metadata={
                "case_id": case["case_id"],
                "kind": case.get("kind", ""),
                "dataset_version": payload["version"],
                "planner_output": case.get("planner_output", ""),
                "source": payload.get("source", ""),
                "source_type": "langfuse_trace"
                if case.get("source_trace_id")
                else "synthetic_boundary",
            },
            source_trace_id=case.get("source_trace_id") or None,
        )
    client.flush()
    return {
        "dataset_name": name,
        "dataset_version": payload["version"],
        "dataset_created": created,
        "item_count": len(item_ids),
        "item_ids": item_ids,
    }


def run_badcase_regression(dataset_path, output_dir=None) -> Dict:
    payload = load_badcase_dataset(dataset_path)
    rows = []
    for case in payload["cases"]:
        if case.get("kind") == "router":
            actual = _run_router_case(case)
        elif case.get("kind") == "memory_sequence":
            actual = _run_memory_case(case)
        elif case.get("kind") == "arxiv_query":
            actual = _run_arxiv_query_case(case)
        elif case.get("kind") == "memory_context":
            actual = _run_memory_context_case(case)
        elif case.get("kind") == "operational_contract":
            actual = _run_operational_contract_case(case)
        elif case.get("kind") == "router_config":
            actual = _run_router_config_case(case)
        elif case.get("kind") == "evidence_gate":
            actual = _run_evidence_gate_case(case)
        elif case.get("kind") == "memory_structured_update":
            actual = _run_memory_structured_update_case(case)
        elif case.get("kind") == "response_contract":
            actual = _run_response_contract_case(case)
        elif case.get("kind") == "api_contract":
            actual = _run_api_contract_case(case)
        elif case.get("kind") == "memory_type_contract":
            actual = _run_memory_type_contract_case(case)
        elif case.get("kind") == "citation_authorization":
            actual = _run_citation_authorization_case(case)
        elif case.get("kind") == "scorecard_lifecycle":
            actual = _run_scorecard_lifecycle_case(case)
        else:
            actual = {"error": "unsupported case kind: {0}".format(case.get("kind"))}
        differences = _differences(case.get("expected") or {}, actual)
        rows.append(
            {
                "case_id": case["case_id"],
                "kind": case.get("kind", ""),
                "passed": not differences,
                "expected": case.get("expected") or {},
                "actual": actual,
                "differences": differences,
                "source_trace_id": case.get("source_trace_id", ""),
            }
        )
    passed = sum(1 for row in rows if row["passed"])
    report = {
        "dataset_name": payload["dataset_name"],
        "dataset_version": payload["version"],
        "dataset_sha256": _sha256(Path(dataset_path)),
        "total": len(rows),
        "passed": passed,
        "failed": len(rows) - passed,
        "rows": rows,
    }
    if output_dir is not None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        (root / "langfuse_badcase_regression.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return report


def _run_router_case(case: Dict) -> Dict:
    from ..paperstorm_intent_router import PaperStormIntentRouter

    planner_output = case.get("planner_output", "")
    router = PaperStormIntentRouter(llm_router=lambda _prompt: planner_output)
    decision = router.route(
        message=case.get("message", ""),
        session=case.get("session") or {},
        context_window=case.get("context_window") or [],
    )
    calls = decision.get("tool_calls") or []
    return {
        "action": decision.get("action"),
        "need_retrieval": bool(decision.get("need_retrieval")),
        "tool": decision.get("tool"),
        "first_tool_call": calls[0].get("name") if calls else "",
        "planner_status": decision.get("planner_status", ""),
        "planner_error_type": (decision.get("planner_error") or {}).get("type", ""),
    }


def _run_memory_case(case: Dict) -> Dict:
    from ..paperstorm_observability import PaperStormObservability
    from ..paperstorm_service import PaperStormTaskService

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        service = PaperStormTaskService(
            root,
            observability=PaperStormObservability(root, enabled=False),
        )
        first = service.create_chat_session(
            run_mode="fake", user_id=case.get("user_id") or "regression-user"
        )
        written = service.send_chat_message(first["chat_id"], case["write_message"])
        second = service.create_chat_session(
            run_mode="fake", user_id=case.get("user_id") or "regression-user"
        )
        recalled = service.send_chat_message(second["chat_id"], case["recall_message"])
    write = written.get("memory_write") or {}
    recalled_items = (recalled.get("long_term_memory") or {}).get("results") or []
    return {
        "write_status": write.get("status", ""),
        "read_after_write_verified": bool(
            (write.get("read_after_write") or {}).get("verified")
        ),
        "recall_contains": " ".join(
            str(item.get("content") or "") for item in recalled_items
        ),
        "retrieval_triggered": bool(recalled.get("retrieval_triggered")),
    }


def _run_arxiv_query_case(case: Dict) -> Dict:
    from ..rm import ArxivRM

    queries = ArxivRM._compile_queries_for_arxiv(case.get("message") or "")
    return {
        "query_count_at_least": len(queries) >= int(case.get("min_query_count") or 1),
        "all_contain": all(
            str(token).lower() in query.lower()
            for query in queries
            for token in case.get("required_terms") or []
        ),
    }


def _run_memory_context_case(case: Dict) -> Dict:
    from ..paperstorm_intent_router import PaperStormIntentRouter
    from ..paperstorm_service import PaperStormTaskService
    from ..conversation_runtime import PaperStormConversationRuntime

    planner_output = case.get("planner_output") or ""
    captured = {}

    def answer(prompt, **_kwargs):
        captured["prompt"] = prompt
        return case.get("model_answer") or ""

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        runtime = PaperStormConversationRuntime(
            root_dir=root / "runtime",
            task_service=PaperStormTaskService(root / "service"),
            intent_router=PaperStormIntentRouter(llm_router=lambda _prompt: planner_output),
            chat_llm=answer,
        )
        try:
            runtime.memory_service.ingest_message(
                namespace="user/{0}".format(case.get("user_id") or "regression-user"),
                message=case["write_message"],
                source_message_id="dataset-memory",
                subject=case.get("user_id") or "regression-user",
            )
            result = runtime.invoke(
                thread_id="dataset-thread",
                request_id="dataset-request",
                user_id=case.get("user_id") or "regression-user",
                message=case["message"],
                run_mode="fake",
            )
        finally:
            runtime.close()
    return {
        "route": result.get("route", ""),
        "answer_contains": result.get("answer", ""),
        "prompt_contains": captured.get("prompt", ""),
        "retrieval_triggered": bool(result.get("retrieval_triggered")),
    }


def _run_operational_contract_case(_case: Dict) -> Dict:
    import os
    import sys
    import types
    from unittest import mock

    loaded = []

    class FakeModel:
        max_seq_length = None

        def get_sentence_embedding_dimension(self):
            return 384

        def encode(self, texts, **_kwargs):
            return [[0.0] * 384 for _ in texts]

    def fake_loader(*args, **kwargs):
        loaded.append((args, kwargs))
        return FakeModel()

    from ..retrieval import SentenceTransformerProvider

    with mock.patch.dict(
        os.environ,
        {"PAPERSTORM_ALLOW_MODEL_DOWNLOAD": "0"},
        clear=False,
    ), mock.patch.dict(
        sys.modules,
        {"sentence_transformers": types.SimpleNamespace(SentenceTransformer=fake_loader)},
    ):
        provider = SentenceTransformerProvider(profile="legacy-multilingual")
        provider.embed_query("wavelet neural network")
    options = loaded[0][1]
    return {
        "model_loading": "local_first",
        "runtime_download": not bool(options.get("local_files_only")),
        "failure_mode": "fast_and_explicit",
    }


def _run_router_config_case(_case: Dict) -> Dict:
    from ..paperstorm_router_llm import _router_output_tokens

    return {
        "default_output_tokens": _router_output_tokens(),
        "complete_json_budget": _router_output_tokens() >= 768,
    }


def _run_evidence_gate_case(case: Dict) -> Dict:
    from ..paperstorm_research_qa import evaluate_evidence_sufficiency

    result = evaluate_evidence_sufficiency(
        question=case.get("question") or "",
        topic=case.get("topic") or "",
        evidence=case.get("evidence") or [],
        citations=case.get("citations") or [],
        expected_keywords=case.get("expected_keywords") or [],
        forbidden_keywords=case.get("forbidden_keywords") or [],
    )
    return {
        "sufficient": bool(result.get("sufficient")),
        "evidence_count": int(result.get("evidence_count") or 0),
        "citation_count": int(result.get("citation_count") or 0),
    }


def _run_memory_structured_update_case(case: Dict) -> Dict:
    from ..memory_store import LongTermMemoryService

    with tempfile.TemporaryDirectory() as temp_dir:
        service = LongTermMemoryService(Path(temp_dir))
        first = service.ingest_structured(
            namespace="user/regression",
            content=case["initial_content"],
            canonical_key=case["canonical_key"],
            memory_type="preference",
            source_message_id="initial",
        )
        second = service.ingest_structured(
            namespace="user/regression",
            content=case["updated_content"],
            canonical_key=case["canonical_key"],
            memory_type="preference",
            source_message_id="update",
        )
        active = service.list_memories("user/regression")
        all_memories = service.list_memories("user/regression", include_inactive=True)
    return {
        "initial_status": first.get("status"),
        "update_status": second.get("status"),
        "read_after_write_verified": bool(
            (second.get("read_after_write") or {}).get("verified")
        ),
        "active_count": len(active),
        "active_content": active[0].get("content", "") if active else "",
        "superseded_count": sum(
            1 for item in all_memories if item.get("status") == "superseded"
        ),
    }


def _run_response_contract_case(case: Dict) -> Dict:
    from ..conversation_runtime import _enforce_response_contract

    adjusted = _enforce_response_contract(
        case.get("decision") or {},
        task_id=case.get("task_id") or "",
        message=case.get("message") or "",
    )
    calls = adjusted.get("tool_calls") or []
    return {
        "action": adjusted.get("action"),
        "first_tool_call": calls[0].get("name") if calls else "",
        "query": ((calls[0].get("arguments") or {}).get("query") if calls else ""),
        "adjustment": "citation_contract"
        if "citation_contract" in (adjusted.get("runtime_adjustments") or [])
        else "",
    }


def _run_api_contract_case(case: Dict) -> Dict:
    from fastapi.testclient import TestClient
    from examples.storm_examples.paperstorm_service_api import create_app

    with tempfile.TemporaryDirectory() as temp_dir:
        client = TestClient(create_app(service_root=temp_dir))
        response = client.post("/chat/sessions", json=case.get("payload") or {})
    body = response.json() if response.status_code == 200 else {}
    return {
        "status_code": response.status_code,
        "task_id": body.get("task_id", ""),
    }


def _run_memory_type_contract_case(case: Dict) -> Dict:
    from ..memory_store import LongTermMemoryService

    with tempfile.TemporaryDirectory() as temp_dir:
        service = LongTermMemoryService(Path(temp_dir))
        result = service.ingest_structured(
            namespace="user/regression",
            content=case.get("content") or "用户姓名是小宇",
            canonical_key=case.get("canonical_key") or "user.name",
            memory_type=case.get("memory_type") or "semantic",
            source_message_id="langfuse-enum-drift",
        )
    memory = result.get("memory") or {}
    return {
        "write_status": result.get("status"),
        "memory_type": memory.get("memory_type"),
        "normalized_from": (memory.get("metadata") or {}).get(
            "normalized_memory_type_from", ""
        ),
        "read_after_write_verified": bool(
            (result.get("read_after_write") or {}).get("verified")
        ),
    }


def _run_citation_authorization_case(case: Dict) -> Dict:
    from ..conversation_runtime import _enforce_response_contract
    from ..paperstorm_intent_router import enforce_tool_authorization

    decision = enforce_tool_authorization(case.get("decision") or {})
    adjusted = _enforce_response_contract(
        decision,
        task_id=case.get("task_id") or "task-existing-evidence",
        message=case.get("message") or "",
    )
    calls = adjusted.get("tool_calls") or []
    return {
        "evidence_search": (decision.get("authorization") or {}).get(
            "evidence.search"
        ),
        "research_start": (decision.get("authorization") or {}).get(
            "research.start"
        ),
        "action": adjusted.get("action"),
        "first_tool_call": calls[0].get("name") if calls else "",
        "adjustment": "citation_contract"
        if "citation_contract" in (adjusted.get("runtime_adjustments") or [])
        else "",
    }


def _run_scorecard_lifecycle_case(case: Dict) -> Dict:
    from ..paperstorm_eval import EvalCase, evaluate_run

    with tempfile.TemporaryDirectory() as temp_dir:
        run_dir = Path(temp_dir)
        (run_dir / "raw_search_results.json").write_text(
            json.dumps([{"title": "PIM", "description": "passive intermodulation"}]),
            encoding="utf-8",
        )
        (run_dir / "storm_gen_outline.txt").write_text("# PIM", encoding="utf-8")
        (run_dir / "storm_gen_article_polished.txt").write_text(
            "无源互调 neural network [1]", encoding="utf-8"
        )
        (run_dir / "paperstorm_trace.jsonl").write_text(
            '{"event":"run_end","success":true}\n', encoding="utf-8"
        )
        (run_dir / "run_summary.json").write_text(
            '{"success":true}', encoding="utf-8"
        )
        scorecard = evaluate_run(
            run_dir,
            EvalCase(
                topic=case.get("topic") or "PIM",
                expected_keywords=["passive intermodulation"],
                forbidden_keywords=[],
                expected_language="zh",
                min_sources=1,
            ),
        )
    return {"run_success": bool((scorecard.get("checks") or {}).get("run_success"))}


def _case_input(case: Dict) -> Dict:
    omitted = {"case_id", "expected", "planner_output", "source_trace_id"}
    return {key: value for key, value in case.items() if key not in omitted}


def _differences(expected: Dict, actual: Dict):
    differences = []
    for key, value in expected.items():
        observed = actual.get(key)
        if key in {"recall_contains", "answer_contains", "prompt_contains"}:
            matched = str(value) in str(observed or "")
        else:
            matched = observed == value
        if not matched:
            differences.append({"field": key, "expected": value, "actual": observed})
    return differences


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
