import tempfile
import unittest
from pathlib import Path


class PaperStormRouterLLMTest(unittest.TestCase):
    def make_service(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        from knowledge_storm.paperstorm_service import PaperStormTaskService

        return PaperStormTaskService(root_dir=Path(temp_dir.name))

    def test_fake_mode_router_is_rule_based(self):
        from knowledge_storm.paperstorm_router_llm import build_intent_router

        router = build_intent_router(run_mode="fake")
        self.assertIsNone(router.llm_router)
        decision = router.route(message="你好", session={}, context_window=[])
        self.assertEqual(decision["router"], "rule_fallback")

    def test_injected_llm_router_reaches_graph_result(self):
        from knowledge_storm.paperstorm_intent_router import PaperStormIntentRouter
        from knowledge_storm.paperstorm_production_v45 import PaperStormProductionRuntimeV45

        service = self.make_service()

        def fake_llm(_prompt):
            return (
                '{"intent":"system_help","need_retrieval":false,"tool":"chat_fallback",'
                '"rewritten_query":"你是什么模型？","confidence":0.96,"reason":"identity"}'
            )

        runtime = PaperStormProductionRuntimeV45(
            root_dir=Path(service.root_dir) / "production_runtime_test",
            task_service=service,
            intent_router=PaperStormIntentRouter(llm_router=fake_llm),
        )
        result = runtime.invoke(
            tenant_id="local",
            thread_id="thread-llm-1",
            request_id="req-1",
            user_id="local-user",
            message="你是什么模型？",
            topic="pim 神经网络抑制",
            run_mode="fake",
            retriever="arxiv",
            output_language="zh",
            expected_keywords=[],
            forbidden_keywords=[],
            context_window=[],
            source_message_id="src-1",
        )
        self.assertEqual(result["router_decision"]["router"], "llm")
        self.assertEqual(result["route"], "casual_chat")
        self.assertFalse(result["retrieval_triggered"])

    def test_paperstorm_mode_builds_llm_router_when_key_present(self):
        from knowledge_storm.paperstorm_router_llm import build_intent_router

        router = build_intent_router(run_mode="paperstorm")
        if router.llm_router is None:
            self.skipTest("no router API key configured in this environment")
        self.assertIsNotNone(router.llm_router)


if __name__ == "__main__":
    unittest.main()
