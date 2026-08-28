import re
import unittest
from pathlib import Path


class PaperStormCareerDocsTest(unittest.TestCase):
    EXPECTED_PLAYBOOK_SECTIONS = {
        "基础原理": 30,
        "Bad Case 与排查": 25,
        "假设性系统设计": 20,
        "PaperStorm 针对性追问": 25,
    }
    REQUIRED_QUESTION_FIELDS = (
        "参考回答",
        "项目实例",
        "排查/设计步骤",
        "追问",
        "考察点",
        "常见失误",
    )
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.resume = (cls.root / "docs" / "PAPERSTORM_RESUME_GUIDE.md").read_text(
            encoding="utf-8"
        )
        cls.playbook = (cls.root / "docs" / "RAG_AGENT_INTERVIEW_PLAYBOOK.md").read_text(
            encoding="utf-8"
        )

    def assert_every_metric_line_has_context(self, document):
        line_contracts = (
            (
                "SciFact recall@10 0.8264",
                ("n=300",),
            ),
            (
                "QASPER retrieval recall@5 0.5526",
                ("n=1309",),
            ),
            (
                "Answer F1 0.5083",
                ("独立 full 1451 端到端协议",),
            ),
            (
                "Evidence F1 0.5500",
                ("独立 full 1451 端到端协议",),
            ),
            (
                "LongMemEval-S cleaned 500/500 retrieval-only：recall@5 0.8003",
                ("P95 359.3 ms",),
            ),
            (
                "Raw Citation Precision 0.9237",
                ("规则型原始引用映射精度", "不是语义/人工验证准确率"),
            ),
        )
        for needle, required_context in line_contracts:
            metric_lines = [line for line in document.splitlines() if needle in line]
            self.assertTrue(metric_lines, f"missing metric statement: {needle}")
            for line in metric_lines:
                for context in required_context:
                    with self.subTest(needle=needle, context=context, line=line):
                        self.assertIn(context, line)

    def test_resume_contains_professional_material_and_honest_metric_boundary(self):
        required = (
            "STAR",
            "60 秒",
            "3 分钟",
            "SciFact recall@10 0.8264（n=300）",
            "QASPER retrieval recall@5 0.5526（n=1309）",
            "独立 full 1451 端到端协议",
            "Answer F1 0.5083",
            "Evidence F1 0.5500",
            "claim support 0.9592",
            "unsupported 0.0214",
            "LongMemEval-S cleaned 500/500 retrieval-only：recall@5 0.8003",
            "P95 359.3 ms",
            "PIM",
            "5 篇论文",
            "797 个 chunk",
            "50 题",
            "GTE recall@5 0.7200",
            "Raw Citation Precision 0.9237",
            "规则型原始引用映射精度",
            "不是语义/人工验证准确率",
            "HNSW recall@5 1.0000",
            "公开基准",
            "私有领域 pilot",
            "离线治理指标",
            "不能",
            "不等同于",
        )
        for item in required:
            with self.subTest(item=item):
                self.assertIn(item, self.resume)

        self.assert_every_metric_line_has_context(self.resume)

        bullets = re.findall(r"^[-*] .+", self.resume, flags=re.MULTILINE)
        self.assertGreaterEqual(len(bullets), 3)
        self.assertLessEqual(len(bullets), 5)

    def test_interview_playbook_has_one_hundred_questions_in_fixed_modules(self):
        matches = list(re.finditer(r"^### (\d+)\. .+$", self.playbook, flags=re.MULTILINE))
        self.assertEqual([int(match.group(1)) for match in matches], list(range(1, 101)))

        for index, match in enumerate(matches):
            section_end = matches[index + 1].start() if index + 1 < len(matches) else len(self.playbook)
            section = self.playbook[match.start() : section_end]
            for label in self.REQUIRED_QUESTION_FIELDS:
                with self.subTest(question=match.group(1), label=label):
                    self.assertEqual(section.count(f"**{label}**"), 1)

        module_matches = list(
            re.finditer(
                r"^## (基础原理|Bad Case 与排查|假设性系统设计|PaperStorm 针对性追问)$",
                self.playbook,
                flags=re.MULTILINE,
            )
        )
        self.assertEqual(
            [match.group(1) for match in module_matches],
            list(self.EXPECTED_PLAYBOOK_SECTIONS),
        )
        for index, match in enumerate(module_matches):
            module_end = (
                module_matches[index + 1].start()
                if index + 1 < len(module_matches)
                else len(self.playbook)
            )
            module = self.playbook[match.end() : module_end]
            count = len(re.findall(r"^### \d+\. ", module, flags=re.MULTILINE))
            self.assertEqual(count, self.EXPECTED_PLAYBOOK_SECTIONS[match.group(1)])

        topics = (
            "RAG 基础",
            "Chunk",
            "Embedding",
            "Hybrid",
            "BM25",
            "Dense",
            "RRF",
            "Rerank",
            "证据",
            "引用",
            "冲突",
            "Memory",
            "Context Compression",
            "Agent Runtime",
            "Tool",
            "Multi-Agent",
            "Langfuse",
            "Benchmark",
            "稳定性",
            "并发",
            "项目难点",
            "项目成就",
        )
        for topic in topics:
            with self.subTest(topic=topic):
                self.assertIn(topic, self.playbook)

    def test_playbook_covers_real_debugging_and_system_design_cases(self):
        required = (
            "PIM / RAM / DRAM",
            "100% 重排",
            "recall-safe MMR",
            "Parent Context 预算饥饿",
            "Cross-Encoder 误排",
            "引用映射",
            "Memory 召回",
            "ACL",
            "百万级知识库",
            "多租户企业知识库 Agent",
            "高并发 RAG",
            "长期记忆 Agent",
            "论文调研 Agent",
        )
        for item in required:
            with self.subTest(item=item):
                self.assertIn(item, self.playbook)

    def test_playbook_repeats_metric_provenance_and_prohibits_exaggeration(self):
        required = (
            "SciFact recall@10 0.8264（n=300）",
            "QASPER retrieval recall@5 0.5526（n=1309）",
            "独立 full 1451 端到端协议",
            "Answer F1 0.5083",
            "Evidence F1 0.5500",
            "claim support 0.9592",
            "unsupported 0.0214",
            "LongMemEval-S cleaned 500/500 retrieval-only：recall@5 0.8003",
            "P95 359.3 ms",
            "PIM pilot：5 篇论文 / 797 chunks / 50 题",
            "GTE recall@5 0.7200",
            "Raw Citation Precision 0.9237",
            "规则型原始引用映射精度",
            "不是语义/人工验证准确率",
            "HNSW recall@5 1.0000",
            "公开基准",
            "私有领域 pilot",
            "离线治理指标",
            "禁止夸大",
        )
        for item in required:
            with self.subTest(item=item):
                self.assertIn(item, self.playbook)

        self.assert_every_metric_line_has_context(self.playbook)


if __name__ == "__main__":
    unittest.main()
