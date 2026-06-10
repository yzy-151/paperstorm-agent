import unittest
from pathlib import Path


class PaperStormFinalPackagingTest(unittest.TestCase):
    def test_readme_contains_final_project_positioning_and_architecture_map(self):
        root = Path(__file__).resolve().parents[1]
        readme = (root / "README.md").read_text(encoding="utf-8")

        self.assertIn("v1.2 Final Packaging", readme)
        self.assertIn("项目一眼看懂", readme)
        self.assertIn("最终能力地图", readme)
        self.assertIn("Architecture Map", readme)
        self.assertIn("STORM Workflow -> PaperStorm Runtime -> Service/Dashboard", readme)
        self.assertIn("最终演示命令", readme)

    def test_plans_mark_project_as_ready_for_resume_and_maintenance(self):
        root = Path(__file__).resolve().parents[1]
        version_plan = (root / "docs" / "VERSION_PLAN.md").read_text(encoding="utf-8")
        resume_plan = (root / "docs" / "RESUME_INTERVIEW_PLAN.md").read_text(
            encoding="utf-8"
        )

        self.assertIn("v1.2：最终包装与投递收口", version_plan)
        self.assertIn("状态：已完成第一阶段", version_plan)
        self.assertIn("后续进入维护和面试准备阶段", version_plan)
        self.assertIn("最终简历 bullet", resume_plan)
        self.assertIn("最终面试 FAQ 精简版", resume_plan)
        self.assertIn("不要继续堆版本", resume_plan)

    def test_readme_and_plans_include_research_qa_v2_direction(self):
        root = Path(__file__).resolve().parents[1]
        readme = (root / "README.md").read_text(encoding="utf-8")
        version_plan = (root / "docs" / "VERSION_PLAN.md").read_text(encoding="utf-8")
        resume_plan = (root / "docs" / "RESUME_INTERVIEW_PLAN.md").read_text(
            encoding="utf-8"
        )

        self.assertIn("Research QA Agent", readme)
        self.assertIn("/research-agent/ask", readme)
        self.assertIn("research_qa_benchmark_report", readme)
        self.assertIn("v2.0：完整演示版与简历包装", version_plan)
        self.assertIn("v1.9：Benchmark", version_plan)
        self.assertIn("Evidence Sufficiency", resume_plan)


if __name__ == "__main__":
    unittest.main()
