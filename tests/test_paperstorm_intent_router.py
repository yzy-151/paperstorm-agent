import unittest


class PaperStormIntentRouterTest(unittest.TestCase):
    def test_planner_routes_response_action_without_creative_intent_enum(self):
        from knowledge_storm.paperstorm_intent_router import PaperStormIntentRouter

        def fake_llm(_prompt):
            return {
                "content": (
                    '{"action":"respond","tool_calls":[],"rewritten_query":"继续写故事",'
                    '"working_subject":"","confidence":0.98,"reason":"continue current text",'
                    '"response_contract":{"task":"续写用户未完成的故事",'
                    '"continue_previous":true,"requires_citations":false,'
                    '"style_notes":["保持原文风格","不要自我介绍"]}}'
                ),
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 80, "completion_tokens": 42},
                "cost_usd": 0.0001,
                "error": None,
            }

        decision = PaperStormIntentRouter(llm_router=fake_llm).route(
            message="继续写下去",
            session={"topic": "PIM 无源互调", "task_id": "task-pim"},
            context_window=[{"role": "user", "content": "从前有一座城……"}],
        )

        self.assertEqual(decision["action"], "respond")
        self.assertEqual(decision["tool_calls"], [])
        self.assertTrue(decision["response_contract"]["continue_previous"])
        self.assertEqual(decision["planner_status"], "success")
        self.assertNotIn("story", decision.get("intent", ""))

    def test_invalid_planner_json_exposes_typed_error_on_safe_fallback(self):
        from knowledge_storm.paperstorm_intent_router import PaperStormIntentRouter

        decision = PaperStormIntentRouter(llm_router=lambda _prompt: "not-json").route(
            message="继续写下去",
            session={},
            context_window=[{"role": "user", "content": "从前有一座城……"}],
        )

        self.assertEqual(decision["action"], "respond")
        self.assertEqual(decision["planner_status"], "fallback")
        self.assertEqual(decision["planner_error"]["type"], "invalid_response")

    def test_rule_fallback_routes_registered_tools_not_content_types(self):
        from knowledge_storm.paperstorm_intent_router import PaperStormIntentRouter

        router = PaperStormIntentRouter()
        evidence = router.route("PIM 是什么？", session={}, context_window=[])
        research = router.route("请调研 PIM 相关论文", session={}, context_window=[])

        self.assertEqual(evidence["action"], "tool_call")
        self.assertEqual(evidence["tool_calls"][0]["name"], "evidence.search")
        self.assertEqual(research["action"], "tool_call")
        self.assertEqual(research["tool_calls"][0]["name"], "research.start")

    def test_llm_planner_is_not_vetoed_by_topic_biased_keyword_rules(self):
        from knowledge_storm.paperstorm_intent_router import PaperStormIntentRouter

        def fake_llm(_prompt):
            return (
                '{"intent":"casual_chat","need_retrieval":false,'
                '"tool":"chat_fallback","rewritten_query":"写一个太空故事",'
                '"working_subject":"","confidence":0.98,'
                '"reason":"creative chat does not need research"}'
            )

        decision = PaperStormIntentRouter(llm_router=fake_llm).route(
            message="继续写一个太空故事",
            session={"topic": "PIM 无源互调", "task_id": "task-pim"},
            context_window=[{"role": "user", "content": "给我写一个故事"}],
        )

        self.assertEqual(decision["router"], "llm_planner")
        self.assertEqual(decision["tool"], "chat_fallback")
        self.assertFalse(decision["need_retrieval"])

    def test_planner_prompt_does_not_inject_stale_topic_as_current_intent(self):
        from knowledge_storm.paperstorm_intent_router import build_router_prompt

        prompt = build_router_prompt(
            message="给我写一个太空故事",
            session={"topic": "PIM 无源互调", "task_id": "task-pim"},
            context_window=[{"role": "user", "content": "我们换个话题"}],
            memory_context={},
            evidence_sufficiency={},
        )

        self.assertNotIn('"topic": "PIM 无源互调"', prompt)
        self.assertIn("旧任务主题不得自动继承", prompt)

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
        self.assertEqual(decision["router"], "llm_planner")

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
        self.assertEqual("那它为什么不是 DRAM？", decision["rewritten_query"])
        self.assertNotIn("pim 神经网络抑制", decision["rewritten_query"].lower())

    def test_router_keeps_standalone_research_request_without_topic_pollution(self):
        from knowledge_storm.paperstorm_intent_router import PaperStormIntentRouter

        router = PaperStormIntentRouter()
        decision = router.route(
            message="你去查一下muon优化器，这个优化器为啥效果好",
            session={"topic": "pim 神经网络抑制", "task_id": "task-pim"},
            context_window=[
                {"role": "user", "content": "PIM 是什么？"},
                {"role": "assistant", "content": "PIM 指 passive intermodulation。"},
            ],
        )
        self.assertEqual(decision["intent"], "run_research")
        self.assertTrue(decision["need_retrieval"])
        self.assertIn("muon", decision["rewritten_query"].lower())
        self.assertNotIn("pim", decision["rewritten_query"].lower())

    def test_router_treats_algorithm_meta_question_as_system_help(self):
        from knowledge_storm.paperstorm_intent_router import PaperStormIntentRouter

        decision = PaperStormIntentRouter().route(
            message="你当前用什么算法检索？",
            session={"topic": "pim 神经网络抑制", "task_id": "task-pim"},
            context_window=[],
        )
        self.assertEqual(decision["intent"], "system_help")
        self.assertFalse(decision["need_retrieval"])

    def test_router_treats_kb_logic_meta_question_as_system_help(self):
        from knowledge_storm.paperstorm_intent_router import PaperStormIntentRouter

        decision = PaperStormIntentRouter().route(
            message="你说一下知识库问答那边的逻辑，具体实现",
            session={"topic": "pim 神经网络抑制", "task_id": "task-pim"},
            context_window=[],
        )
        self.assertEqual(decision["intent"], "system_help")
        self.assertFalse(decision["need_retrieval"])

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

    def test_social_messages_never_inherit_the_research_topic(self):
        from knowledge_storm.paperstorm_intent_router import PaperStormIntentRouter

        router = PaperStormIntentRouter()
        for message in ["莫西莫西", "谢谢", "你在干嘛", "今天天气不错"]:
            with self.subTest(message=message):
                decision = router.route(
                    message=message,
                    session={"topic": "pim 神经网络抑制", "task_id": "task-pim"},
                    context_window=[],
                )
                self.assertEqual(decision["intent"], "casual_chat")
                self.assertFalse(decision["need_retrieval"])
                self.assertEqual(decision["rewritten_query"], message)

    def test_ambiguous_short_followup_without_context_asks_for_clarification(self):
        from knowledge_storm.paperstorm_intent_router import PaperStormIntentRouter

        decision = PaperStormIntentRouter().route(
            message="这个呢？",
            session={"topic": "pim 神经网络抑制"},
            context_window=[],
        )

        self.assertEqual(decision["intent"], "clarify")
        self.assertFalse(decision["need_retrieval"])
        self.assertNotIn("pim", decision["rewritten_query"].lower())

    def test_invalid_or_low_confidence_llm_decision_falls_back_safely(self):
        from knowledge_storm.paperstorm_intent_router import PaperStormIntentRouter

        for response in [
            "not-json",
            '{"intent":"research_qa","need_retrieval":true,"tool":"research_qa",'
            '"confidence":0.2,"reason":"uncertain"}',
        ]:
            with self.subTest(response=response):
                decision = PaperStormIntentRouter(
                    llm_router=lambda _prompt, value=response: value
                ).route(
                    message="莫西莫西",
                    session={"topic": "pim 神经网络抑制"},
                    context_window=[],
                )
                self.assertEqual(decision["intent"], "casual_chat")
                self.assertFalse(decision["need_retrieval"])


if __name__ == "__main__":
    unittest.main()
