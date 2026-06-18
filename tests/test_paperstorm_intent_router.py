import unittest


class PaperStormIntentRouterTest(unittest.TestCase):
    def test_llm_router_accepts_structured_json_decision(self):
        from knowledge_storm.paperstorm_intent_router import PaperStormIntentRouter

        def fake_llm(prompt):
            return """
            {"intent":"system_help","need_retrieval":false,"tool":"chat_fallback",
             "rewritten_query":"你是什么模型？","confidence":0.96,
             "reason":"用户询问系统身份"}
            """

        router = PaperStormIntentRouter(llm_router=fake_llm)
        decision = router.route(
            message="你是什么模型？",
            session={"topic": "pim 神经网络抑制", "task_id": "task-1"},
            context_window=[],
        )

        self.assertEqual(decision["intent"], "system_help")
        self.assertFalse(decision["need_retrieval"])
        self.assertEqual(decision["tool"], "chat_fallback")
        self.assertEqual(decision["router"], "llm")

    def test_router_rewrites_followup_into_standalone_research_query(self):
        from knowledge_storm.paperstorm_intent_router import PaperStormIntentRouter

        router = PaperStormIntentRouter()
        decision = router.route(
            message="那它为什么不是 DRAM？",
            session={
                "topic": "pim 神经网络抑制",
                "expected_keywords": ["passive intermodulation"],
                "forbidden_keywords": ["DRAM"],
            },
            context_window=[
                {"role": "user", "content": "PIM 是什么？"},
                {"role": "assistant", "content": "PIM 指 passive intermodulation。"},
            ],
        )

        self.assertEqual(decision["intent"], "research_qa")
        self.assertTrue(decision["need_retrieval"])
        self.assertIn("pim 神经网络抑制", decision["rewritten_query"])
        self.assertIn("PIM 是什么", decision["rewritten_query"])

    def test_router_distinguishes_chat_from_research_without_topic_bias(self):
        from knowledge_storm.paperstorm_intent_router import PaperStormIntentRouter

        router = PaperStormIntentRouter()
        chat = router.route(
            message="你是什么模型？",
            session={"topic": "pim 神经网络抑制"},
            context_window=[],
        )
        research = router.route(
            message="PIM 神经网络抑制有哪些论文方向？",
            session={"topic": "pim 神经网络抑制"},
            context_window=[],
        )

        self.assertEqual(chat["intent"], "system_help")
        self.assertFalse(chat["need_retrieval"])
        self.assertEqual(research["intent"], "run_research")
        self.assertTrue(research["need_retrieval"])


if __name__ == "__main__":
    unittest.main()
