import tempfile
import threading
import time
import unittest
from pathlib import Path


class PaperStormProductionV45Test(unittest.TestCase):
    def make_control(self, root):
        from knowledge_storm.paperstorm_production_v45 import ProductionControlPlaneV45

        return ProductionControlPlaneV45(Path(root) / "production_v45.sqlite")

    def test_acl_is_checked_before_resource_access_and_audited(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            control = self.make_control(temp_dir)
            control.register_resource(
                tenant_id="tenant-a",
                resource_type="knowledge_base",
                resource_id="kb-1",
                owner_user_id="alice",
                allowed_user_ids=["bob"],
            )

            granted = control.authorize(
                tenant_id="tenant-a",
                user_id="bob",
                resource_type="knowledge_base",
                resource_id="kb-1",
                action="read",
            )
            with self.assertRaises(PermissionError):
                control.authorize(
                    tenant_id="tenant-b",
                    user_id="mallory",
                    resource_type="knowledge_base",
                    resource_id="kb-1",
                    action="read",
                )

            self.assertTrue(granted["allowed"])
            audit = control.list_audit_events(limit=10)
            self.assertEqual([item["decision"] for item in audit[-2:]], ["allow", "deny"])

    def test_transactional_idempotency_executes_once_under_concurrency(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            control = self.make_control(temp_dir)
            calls = []
            results = []
            lock = threading.Lock()

            def operation():
                with lock:
                    calls.append(time.time())
                time.sleep(0.05)
                return {"value": 42}

            def worker():
                results.append(
                    control.execute_idempotent(
                        scope="tenant-a/thread-1",
                        key="request-1",
                        payload={"message": "hello"},
                        operation=operation,
                    )
                )

            threads = [threading.Thread(target=worker) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            self.assertEqual(len(calls), 1)
            self.assertEqual(len(results), 8)
            self.assertEqual(sum(item["idempotent_replay"] for item in results), 7)
            self.assertTrue(all(item["result"] == {"value": 42} for item in results))
            with self.assertRaises(ValueError):
                control.execute_idempotent(
                    scope="tenant-a/thread-1",
                    key="request-1",
                    payload={"message": "different"},
                    operation=operation,
                )

    def test_cache_supports_ttl_metrics_and_tag_invalidation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            control = self.make_control(temp_dir)
            control.set_cache(
                namespace="tenant-a/retrieval",
                key="query-1",
                value={"chunks": ["c1"]},
                ttl_seconds=60,
                tags=["kb:kb-1"],
            )

            hit = control.get_cache("tenant-a/retrieval", "query-1")
            miss = control.get_cache("tenant-a/retrieval", "missing")
            invalidated = control.invalidate_cache(tag="kb:kb-1")

            self.assertTrue(hit["hit"])
            self.assertFalse(miss["hit"])
            self.assertEqual(invalidated, 1)
            self.assertFalse(control.get_cache("tenant-a/retrieval", "query-1")["hit"])
            self.assertGreater(control.cache_metrics()["hit_rate"], 0)

    def test_durable_job_retries_then_succeeds_and_circuit_degrades(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            control = self.make_control(temp_dir)
            job = control.enqueue_job(
                tenant_id="tenant-a",
                job_type="incremental_index",
                payload={"kb_id": "kb-1"},
                idempotency_key="update-1",
                max_attempts=3,
            )
            duplicate = control.enqueue_job(
                tenant_id="tenant-a",
                job_type="incremental_index",
                payload={"kb_id": "kb-1"},
                idempotency_key="update-1",
                max_attempts=3,
            )
            attempts = []

            def flaky(payload):
                attempts.append(payload)
                if len(attempts) == 1:
                    raise ConnectionError("temporary index outage")
                return {"indexed": 1}

            first = control.run_worker_tick({"incremental_index": flaky})
            second = control.run_worker_tick({"incremental_index": flaky})

            self.assertEqual(job["job_id"], duplicate["job_id"])
            self.assertEqual(first["status"], "retrying")
            self.assertEqual(second["status"], "succeeded")
            self.assertEqual(second["result"], {"indexed": 1})

            calls = []

            def unavailable():
                calls.append(1)
                raise TimeoutError("provider timeout")

            degraded = control.execute_resilient(
                operation_name="embedding-provider",
                operation=unavailable,
                fallback=lambda error: {"mode": "lexical_only", "error": str(error)},
                max_attempts=2,
                failure_threshold=2,
                cooldown_seconds=60,
            )
            open_degraded = control.execute_resilient(
                operation_name="embedding-provider",
                operation=unavailable,
                fallback=lambda error: {"mode": "lexical_only", "error": str(error)},
                max_attempts=2,
                failure_threshold=2,
                cooldown_seconds=60,
            )

            self.assertTrue(degraded["degraded"])
            self.assertEqual(open_degraded["circuit_state"], "open")
            self.assertEqual(len(calls), 2)

    def test_enterprise_kb_enforces_acl_and_processes_incremental_update(self):
        from knowledge_storm.paperstorm_service import PaperStormTaskService

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first.txt"
            second = root / "second.txt"
            first.write_text("PIM means passive intermodulation in RF systems.", encoding="utf-8")
            second.write_text("Neural cancellers suppress passive intermodulation distortion.", encoding="utf-8")
            service = PaperStormTaskService(root / "service")
            manifest = service.create_enterprise_knowledge_base(
                name="RF KB",
                source_paths=[str(first)],
                tenant_id="tenant-a",
                owner_user_id="alice",
                allowed_user_ids=["bob"],
            )

            answer = service.ask_enterprise_knowledge_base(
                manifest["kb_id"], "What is PIM?", tenant_id="tenant-a", user_id="bob"
            )
            visible = service.list_enterprise_knowledge_bases(
                tenant_id="tenant-a", user_id="bob"
            )
            hidden = service.list_enterprise_knowledge_bases(
                tenant_id="tenant-b", user_id="mallory"
            )
            with self.assertRaises(PermissionError):
                service.ask_enterprise_knowledge_base(
                    manifest["kb_id"], "What is PIM?", tenant_id="tenant-b", user_id="mallory"
                )
            with self.assertRaises(PermissionError):
                service.get_enterprise_knowledge_base(
                    manifest["kb_id"], tenant_id="tenant-b", user_id="mallory"
                )
            job = service.enqueue_enterprise_kb_update(
                kb_id=manifest["kb_id"],
                source_paths=[str(second)],
                tenant_id="tenant-a",
                user_id="alice",
                idempotency_key="add-second-doc",
            )
            completed = service.run_production_worker_tick()
            updated = service.get_enterprise_knowledge_base(
                manifest["kb_id"], tenant_id="tenant-a", user_id="alice"
            )
            control = service._production_control_v45()
            document_resource = control.get_resource(
                "document",
                "{0}:{1}".format(manifest["kb_id"], updated["documents"][0]["document_id"]),
            )

            self.assertTrue(answer["grounded"])
            self.assertEqual([item["kb_id"] for item in visible], [manifest["kb_id"]])
            self.assertEqual(hidden, [])
            self.assertEqual(job["status"], "queued")
            self.assertEqual(completed["status"], "succeeded")
            self.assertEqual(updated["document_count"], 2)
            self.assertEqual(updated["index_version"], 2)
            self.assertEqual(document_resource["tenant_id"], "tenant-a")

    def test_v45_runtime_and_benchmark_expose_trace_and_slo_metrics(self):
        from knowledge_storm.paperstorm_production_benchmark_v45 import run_production_benchmark
        from knowledge_storm.paperstorm_service import PaperStormTaskService

        with tempfile.TemporaryDirectory() as temp_dir:
            service = PaperStormTaskService(Path(temp_dir) / "service")
            result = service.invoke_conversation_graph(
                tenant_id="tenant-a",
                thread_id="thread-1",
                request_id="request-1",
                user_id="alice",
                message="你好",
                run_mode="fake",
            )
            replay = service.invoke_conversation_graph(
                tenant_id="tenant-a",
                thread_id="thread-1",
                request_id="request-1",
                user_id="alice",
                message="你好",
                run_mode="fake",
            )
            trace = service.get_production_trace(
                result["trace_id"], tenant_id="tenant-a", user_id="alice"
            )
            report = run_production_benchmark(Path(temp_dir) / "benchmark", request_count=30)

            self.assertEqual(result["runtime"], "paperstorm-production-v4.5")
            self.assertTrue(replay["governance"]["idempotent_replay"])
            self.assertTrue(trace["spans"])
            for metric in ["latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "qps", "error_rate", "degradation_rate"]:
                self.assertIn(metric, report["metrics"])
            self.assertEqual(report["metrics"]["acl_leakage_rate"], 0.0)

    def test_api_and_dashboard_expose_v45_governance(self):
        root = Path(__file__).resolve().parents[1]
        index = (root / "frontend/paperstorm_dashboard/index.html").read_text(encoding="utf-8")
        script = (root / "frontend/paperstorm_dashboard/app.js").read_text(encoding="utf-8")

        self.assertIn("v4.5", index)
        self.assertIn("production-v45-metrics", index)
        self.assertIn("production-v45-trace", index)
        self.assertIn("/evaluations/production-v45", script)
        self.assertIn("/production/traces/", script)

    def test_production_api_exposes_governed_runtime_and_benchmark(self):
        try:
            from fastapi.testclient import TestClient
        except Exception as exc:  # pragma: no cover
            self.skipTest(str(exc))
        from examples.storm_examples.paperstorm_service_api import create_app

        with tempfile.TemporaryDirectory() as temp_dir:
            client = TestClient(create_app(service_root=Path(temp_dir)))
            invoked = client.post(
                "/conversation-graph/invoke",
                json={
                    "tenant_id": "tenant-a",
                    "thread_id": "api-v45-thread",
                    "request_id": "api-v45-request",
                    "user_id": "alice",
                    "message": "你好",
                    "run_mode": "fake",
                },
            )
            result = invoked.json()
            state = client.get(
                "/conversation-graph/threads/api-v45-thread/state",
                params={"tenant_id": "tenant-a", "user_id": "alice"},
            )
            history = client.get(
                "/conversation-graph/threads/api-v45-thread/history",
                params={"tenant_id": "tenant-a", "user_id": "alice"},
            )
            trace = client.get(
                "/production/traces/{0}".format(result["trace_id"]),
                params={"tenant_id": "tenant-a", "user_id": "alice"},
            )
            denied = client.get(
                "/production/traces/{0}".format(result["trace_id"]),
                params={"tenant_id": "tenant-b", "user_id": "mallory"},
            )
            status = client.get("/production/status")
            benchmark = client.post(
                "/evaluations/production-v45", json={"request_count": 10}
            )

        self.assertEqual(invoked.status_code, 200)
        self.assertEqual(result["runtime"], "paperstorm-production-v4.5")
        self.assertEqual(state.json()["values"]["request_id"], "api-v45-request")
        self.assertTrue(history.json()["checkpoints"])
        self.assertTrue(trace.json()["spans"])
        self.assertEqual(denied.status_code, 403)
        self.assertEqual(status.json()["backend"], "sqlite-wal")
        self.assertEqual(status.json()["version"], "v4.5")
        self.assertIn("latency_p95_ms", benchmark.json()["metrics"])


if __name__ == "__main__":
    unittest.main()
