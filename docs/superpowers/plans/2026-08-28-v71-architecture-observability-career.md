# PaperStorm v7.1 Architecture, Observability and Career Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 发布可编辑架构图、异步时序图、Langfuse Bad Case 演示、项目简历指南和双 Agent RAG 面试模拟器。

**Architecture:** 继续使用现有 Draw.io 生成器和 `PaperStormObservability`，不引入新的 Web 产品页面。Langfuse 演示与面试模拟器分别作为小型领域模块和 CLI，默认离线可测试、配置后可连接真实服务。

**Tech Stack:** Python 3.10/3.11、Draw.io XML、SVG、Langfuse SDK、JSONL、Markdown、unittest。

---

## 文件结构

- `docs/architecture/generate_drawio_diagrams.py`：三张可编辑图的单一生成源。
- `docs/architecture/paperstorm-async-runtime-sequence.{drawio,svg}`：异步时序图产物。
- `knowledge_storm/langfuse_badcase_demo.py`：Bad Case 分类、Trace/Span/Score 演示合同。
- `examples/storm_examples/run_langfuse_badcase_demo.py`：演示 CLI。
- `knowledge_storm/rag_interview_simulator.py`：题库状态、Interviewer/Candidate Agent 与会话导出。
- `examples/storm_examples/run_rag_interview_simulator.py`：模拟面试 CLI。
- `docs/PAPERSTORM_RESUME_GUIDE.md`：STAR、bullet、自我介绍与指标边界。
- `docs/RAG_AGENT_INTERVIEW_PLAYBOOK.md`：完整题库、答案、追问与评分点。
- `docs/LANGFUSE_BADCASE_GUIDE.md`：部署、演示和定位流程。
- `tests/test_paperstorm_architecture_map.py`、`tests/test_langfuse_badcase_demo.py`、
  `tests/test_rag_interview_simulator.py`：契约与行为测试。

### Task 1: 更新架构图并新增异步时序图

**Files:**
- Modify: `docs/architecture/generate_drawio_diagrams.py`
- Modify: `docs/architecture/paperstorm-executive-overview.drawio`
- Modify: `docs/architecture/paperstorm-executive-overview.svg`
- Modify: `docs/architecture/paperstorm-agent-system-flow.drawio`
- Modify: `docs/architecture/paperstorm-agent-system-flow.svg`
- Create: `docs/architecture/paperstorm-async-runtime-sequence.drawio`
- Create: `docs/architecture/paperstorm-async-runtime-sequence.svg`
- Test: `tests/test_paperstorm_architecture_map.py`

- [ ] **Step 1: 写失败测试**

要求时序图至少含 `Browser/FastAPI/Queue/Runtime/Retriever/LLM/Checkpoint/SSE/Langfuse` 九个参与者，
至少 14 条消息；详细架构图必须出现 `PIM Domain Pilot`、`Langfuse Score`、`Async Queue`。

```python
def test_async_sequence_drawio_is_editable_and_complete(self):
    root = ET.parse(ARCH / "paperstorm-async-runtime-sequence.drawio").getroot()
    cells = root.findall(".//mxCell")
    self.assertGreaterEqual(sum(c.get("vertex") == "1" for c in cells), 9)
    self.assertGreaterEqual(sum(c.get("edge") == "1" for c in cells), 14)
```

- [ ] **Step 2: 运行测试并确认因文件不存在而失败**

Run: `python -m unittest tests.test_paperstorm_architecture_map`

- [ ] **Step 3: 扩展生成器**

增加 `sequence_diagram()`，使用横向 participant、纵向 lifeline 和编号消息；执行阶段包含任务提交、
202 Accepted、后台出队、Checkpoint、检索/重排、LLM、Trace/Score、SSE 推送、产物读取和失败重试。
更新现有图节点与边，不手工维护生成产物。

- [ ] **Step 4: 生成并验证**

Run: `python docs/architecture/generate_drawio_diagrams.py`

Run: `python -m unittest tests.test_paperstorm_architecture_map`

Expected: PASS，三个 Draw.io 与 SVG 均更新。

### Task 2: 实现 Langfuse Bad Case 演示

**Files:**
- Create: `knowledge_storm/langfuse_badcase_demo.py`
- Create: `examples/storm_examples/run_langfuse_badcase_demo.py`
- Create: `tests/test_langfuse_badcase_demo.py`
- Create: `docs/LANGFUSE_BADCASE_GUIDE.md`

- [ ] **Step 1: 写失败测试**

```python
def test_demo_emits_stage_spans_scores_and_badcase_tags(self):
    report = run_badcase_demo(root, fixture_case(), observability=fake_observability)
    self.assertEqual(report["badcase_types"], ["retrieval_miss", "invalid_citation"])
    self.assertIn("retrieval_recall_at_5", report["scores"])
    self.assertIn("citation_validity", report["scores"])
```

另测：远程 exporter 抛错时 CLI 仍成功；任何 API Key 被脱敏；输出包含 `trace_id` 和本地 JSONL 路径。

- [ ] **Step 2: 确认测试失败**

Run: `python -m unittest tests.test_langfuse_badcase_demo`

- [ ] **Step 3: 实现最小领域合同**

定义：

```python
BADCASE_RULES = {
    "retrieval_miss": lambda case: case["gold_ids"] and not set(case["gold_ids"]) & set(case["retrieved_ids"]),
    "invalid_citation": lambda case: bool(set(case["citation_ids"]) - set(case["retrieved_ids"])),
    "evidence_conflict": lambda case: bool(case.get("conflict_detected")),
    "wrong_abstention": lambda case: bool(case.get("answerable")) and bool(case.get("abstained")),
}
```

`run_badcase_demo` 建立 `paperstorm.rag.badcase` 根 Trace，按 route/retrieve/rerank/context/reader/
citation_validate 建 Span，写入 numeric score 与 badcase metadata，最终返回可序列化报告。

- [ ] **Step 4: 实现 CLI 和文档**

CLI 参数：`--output-dir`、`--case-file`、`--scenario`；默认运行一个同时包含漏召回和非法引用的
固定示例。文档给出 Langfuse 环境变量、运行命令、按 tag/score 筛选和从 Trace 回溯根因的步骤。

- [ ] **Step 5: 验证**

Run: `python examples/storm_examples/run_langfuse_badcase_demo.py --output-dir <temp>`

Expected: 输出 report JSON 和 events JSONL；未配置密钥时显示 `local-only`。

### Task 3: 实现双 Agent RAG 面试模拟器

**Files:**
- Create: `knowledge_storm/rag_interview_simulator.py`
- Create: `examples/storm_examples/run_rag_interview_simulator.py`
- Create: `tests/test_rag_interview_simulator.py`

- [ ] **Step 1: 写失败测试**

覆盖题目去重、分类覆盖、追问引用上一轮回答、Candidate 不接收 reference answer、JSON 解析失败降级、
Markdown 导出和 deterministic 模式。

```python
def test_candidate_prompt_does_not_leak_reference_answer(self):
    prompt = build_candidate_prompt(question_public_view(fixture_question()), project_context="...")
    self.assertNotIn("reference_answer", prompt)
    self.assertNotIn(fixture_question()["reference_answer"], prompt)
```

- [ ] **Step 2: 确认测试失败**

Run: `python -m unittest tests.test_rag_interview_simulator`

- [ ] **Step 3: 实现 Agent 与状态机**

定义 `InterviewQuestion`、`InterviewTurn`、`InterviewSession`。Interviewer 根据未覆盖类别选题，
并根据回答生成场景追问；Candidate 只收到问题、项目事实摘要与历史对话。LLM callable 注入，
deterministic callable 用于测试。每轮保存 question/answer/follow_up/assessment/category。

- [ ] **Step 4: 实现 CLI**

CLI 参数：`--mode deterministic|llm`、`--rounds`、`--output`、`--model`。LLM 模式复用项目现有
OpenAI-compatible provider 配置；失败时记录 `error_type`，不会伪造回答。

- [ ] **Step 5: 验证**

Run: `python examples/storm_examples/run_rag_interview_simulator.py --mode deterministic --rounds 12 --output <temp.md>`

Expected: 至少覆盖 RAG、Memory、Context、Runtime、Langfuse、项目复盘六类，并输出 Markdown。

### Task 4: 编写简历指南与完整面试手册

**Files:**
- Create: `docs/PAPERSTORM_RESUME_GUIDE.md`
- Create: `docs/RAG_AGENT_INTERVIEW_PLAYBOOK.md`
- Test: `tests/test_paperstorm_career_docs.py`

- [ ] **Step 1: 写文档契约测试**

简历指南必须含 STAR、简历 bullet、60 秒介绍、3 分钟介绍、指标边界；面试手册至少 60 个主问题，
每题含参考答案、追问、评价点和失分点，且覆盖 12 个主题。

- [ ] **Step 2: 写简历指南**

使用真实结果：SciFact/QASPER/LongMemEval/PIM Pilot；所有数字同时注明数据集、样本量或证据等级。
形成“原始 STORM 检索与工程治理不足 → 统一 RAG/Memory/Context/Runtime/Eval → 可量化结果”的故事。

- [ ] **Step 3: 写面试手册**

按基础原理、方案选型、故障定位、规模化、项目深挖、Langfuse 六层组织。情景题必须要求候选人给出
指标、Trace 观察点、定位顺序和回归测试，而不只背概念。

- [ ] **Step 4: 验证文档合同**

Run: `python -m unittest tests.test_paperstorm_career_docs`

### Task 5: 更新 README 与 v7.1 发布标识

**Files:**
- Modify: `README.md`
- Modify: `setup.py`
- Modify: `knowledge_storm/__init__.py`
- Modify: `knowledge_storm/paperstorm_service.py`
- Modify: `examples/storm_examples/paperstorm_service_api.py`
- Modify: `frontend/paperstorm_dashboard/index.html`
- Modify: existing version contract tests

- [ ] **Step 1: 先更新版本契约测试为 7.1.0/v7.1 并确认失败**

- [ ] **Step 2: 统一版本号**

包、OpenAPI、Research/Chat Trace、Dashboard bundle、静态资源 cachebuster 均改为 `7.1.0`，
前端展示 `v7.1`。

- [ ] **Step 3: 重写 README 相关章节**

展示三张 SVG 与 Draw.io 链接；添加 Langfuse 5 分钟演示、Bad Case 定位步骤、简历指南和面试手册入口；
保留现有安装、Benchmark 与可信边界，不加入内部 handoff 内容。

- [ ] **Step 4: 运行相关测试**

Run: `python -m unittest tests.test_paperstorm_release_integrity_v52 tests.test_paperstorm_service tests.test_paperstorm_demo_ui_v56 tests.test_paperstorm_career_docs`

### Task 6: 发布验证与 GitHub

**Files:** All changed files above.

- [ ] **Step 1: 运行完整离线测试**

```powershell
$env:PAPERSTORM_OFFLINE_TESTS='1'
$env:PAPERSTORM_RETRIEVAL_EMBEDDING='hash'
python -m unittest discover -s tests -p 'test_*.py'
```

- [ ] **Step 2: 运行编译、XML 与差异检查**

Run: `python -m compileall -q knowledge_storm examples tests`

Run: `git diff --check`

- [ ] **Step 3: 扫描本次新增行中的真实格式 API Key**

只允许 README 中的 `pk-lf-...` / `sk-lf-...` 占位符，不允许长格式真实值。

- [ ] **Step 4: 精确暂存并提交**

排除既有 `docs/DESIGN_SOURCES.md` 删除、`.codex-temp` 与补丁文件。

Commit: `feat(observability): add v7.1 project kit`

- [ ] **Step 5: 推送**

Push `main`，创建并推送 `version/v7.1`，最后确认两个远端 ref 指向同一 commit。
