import re
import unittest
from pathlib import Path


class PaperStormCareerDocsTest(unittest.TestCase):
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

    def test_interview_playbook_has_sixty_structured_questions_and_required_topics(self):
        matches = list(re.finditer(r"^### (\d+)\. .+$", self.playbook, flags=re.MULTILINE))
        self.assertGreaterEqual(len(matches), 60)

        for index, match in enumerate(matches):
            section_end = matches[index + 1].start() if index + 1 < len(matches) else len(self.playbook)
            section = self.playbook[match.start() : section_end]
            for label in ("参考回答", "追问", "考察点", "常见失误"):
                with self.subTest(question=match.group(1), label=label):
                    self.assertEqual(section.count(f"**{label}**"), 1)

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
