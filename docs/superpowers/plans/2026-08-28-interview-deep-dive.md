# PaperStorm Interview Deep-Dive Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 PaperStorm 求职材料扩展为 100 道可追问、可复盘、可验证的 RAG / Agent 面试题，并形成职责清晰、指标克制的简历叙事。

**Architecture:** 保留两份正式 Markdown 文档，不增加运行时依赖。测试按模块和每题六字段解析题库，确保数量、编号、配额和事实边界不会回退；正文只引用冻结报告与可核验外部资料。

**Tech Stack:** Markdown、Python `unittest`、正则解析、PaperStorm Benchmark 报告、公开论文与官方模型卡。

---

## 文件职责

- `docs/RAG_AGENT_INTERVIEW_PLAYBOOK.md`：100 道结构化面试题。
- `docs/PAPERSTORM_RESUME_GUIDE.md`：职责、架构演进、改进矩阵与岗位化 Bullet。
- `tests/test_paperstorm_career_docs.py`：题量、模块、六字段、案例与指标边界合同。

### Task 1: 建立 100 题结构合同

**Files:**
- Modify: `tests/test_paperstorm_career_docs.py`

- [ ] **Step 1: 写入失败测试**

增加题目解析器并声明：

```python
EXPECTED_SECTIONS = {"基础原理": 30, "Bad Case 与排查": 25, "假设性系统设计": 20, "PaperStorm 针对性追问": 25}
REQUIRED_FIELDS = ("**参考回答**：", "**项目实例**：", "**排查/设计步骤**：", "**追问**：", "**考察点**：", "**常见失误**：")
```

断言题号严格等于 `1..100`、模块配额等于 `30/25/20/25`、每题六字段完整。

- [ ] **Step 2: 验证 RED**

Run: `D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest tests.test_paperstorm_career_docs`

Expected: FAIL，因为当前只有 60 题且缺少两个字段。

- [ ] **Step 3: 增加内容合同**

要求出现 `PIM / RAM / DRAM`、`100% 重排`、`recall-safe MMR`、`Parent Context 预算饥饿`、`Cross-Encoder 误排`、`引用映射`、`Memory 召回`、`ACL`、`百万级知识库`、`多租户企业知识库 Agent` 和 `高并发 RAG`。

- [ ] **Step 4: 提交测试合同**

```powershell
git add tests/test_paperstorm_career_docs.py
git commit -m "test(career): define 100-question interview contract"
```

### Task 2: 编写 30 道基础原理与 25 道真实排查题

**Files:**
- Modify: `docs/RAG_AGENT_INTERVIEW_PLAYBOOK.md`
- Test: `tests/test_paperstorm_career_docs.py`

- [ ] **Step 1: 编写基础原理题**

覆盖 Chunk/Overlap、结构化解析、Jieba、BM25、Dense、RRF、Embedding Profile、Cross-Encoder、Parent-Child、HNSW 和评测协议。定义题的“排查/设计步骤”也必须给出冻结数据、单变量实验、Recall/nDCG/P95 与 case-level 检查方法。

- [ ] **Step 2: 编写 Bad Case 题**

每题按 `现象 -> 发现 -> 分层定位 -> 根因 -> 修复 -> Benchmark -> 结果 -> 残余边界` 回答。至少覆盖：

- PIM 歧义：RF 文档 Top-1、forbidden hit=0；无上下文问题仍需澄清。
- QASPER 低词面重叠与 SciFact Vitamin D：gold 分别升至 Top-1。
- PPM1D：Cross-Encoder 将 gold 从第 3 移出 Top-10，明确尚未完全解决。
- 重复 RRF 导致 100% rerank；修后触发率 SciFact 36.33%、QASPER 45.68%。
- MMR 令 QASPER Recall@5 降至 0.4631；recall-safe 后为 0.5526。
- P2 总体 SciFact 0.8114 -> 0.8264、QASPER 0.5057 -> 0.5526，并报告 P95 代价。

- [ ] **Step 3: 对照冻结报告**

逐项检查 `docs/RAG_BADCASE_PROGRESSIVE_RESULTS.md`、`docs/RAG_BAD_CASES_AND_ROADMAP.md`、`docs/PAPERSTORM_RETRIEVAL_STACK_UPGRADE.md` 和 `docs/PAPERSTORM_DOMAIN_PILOT.md`，禁止新增无来源数字。

- [ ] **Step 4: 运行测试并提交**

Run: `D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest tests.test_paperstorm_career_docs`

```powershell
git add docs/RAG_AGENT_INTERVIEW_PLAYBOOK.md
git commit -m "docs(interview): add RAG debugging case studies"
```

### Task 3: 编写 20 道系统设计与 25 道项目追问题

**Files:**
- Modify: `docs/RAG_AGENT_INTERVIEW_PLAYBOOK.md`
- Test: `tests/test_paperstorm_career_docs.py`

- [ ] **Step 1: 编写系统设计题**

每题按 `需求澄清 -> SLO/数据边界 -> 组件与数据流 -> 状态/权限 -> 失败与降级 -> 评测与门禁` 展开。覆盖企业文档 Agent、百万级索引、调研 Agent、客服问答、长期记忆、多租户 ACL、高并发、离线部署、降级、可观测性与成本。

- [ ] **Step 2: 编写项目追问题**

直接回答 BGE/GTE/Qwen 选型、GTE CPU 默认、Parent-Child 和 parent 粒度、SciFact/QASPER/LongMemEval-S 选型、数据适配与 fingerprint、PIM 5 篇/797 chunks/50 题构造，以及外部成绩不可直接横比的原因。

- [ ] **Step 3: 增加来源说明**

社区面经只归纳高频问题；答案依据项目报告、论文、官方仓库与模型卡，不复制社区答案。外部数字同时写 split、样本量、Top K、粒度和 evaluator，否则只作量级参考。

- [ ] **Step 4: 验证 GREEN 并提交**

Run: `D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest tests.test_paperstorm_career_docs`

Expected: 100 题、四类配额和六字段全部 PASS。

```powershell
git add docs/RAG_AGENT_INTERVIEW_PLAYBOOK.md tests/test_paperstorm_career_docs.py
git commit -m "docs(interview): complete 100-question agent playbook"
```

### Task 4: 重构简历职责与技术改进矩阵

**Files:**
- Modify: `docs/PAPERSTORM_RESUME_GUIDE.md`
- Modify: `tests/test_paperstorm_career_docs.py`

- [ ] **Step 1: 添加失败测试**

要求存在 `个人职责与原项目边界`、`为什么进行架构改造`、`技术改进矩阵`、`可组合的简历 Bullet`、`面试叙事：难题、决策与结果`。矩阵至少包含 Jieba、结构化 Chunk、Parent-Child、BM25、Dense、RRF、Embedding Profile、HNSW、选择性 Cross-Encoder、证据治理、Memory/Context 和 Langfuse。

- [ ] **Step 2: 验证 RED**

Run: `D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest tests.test_paperstorm_career_docs`

Expected: FAIL on missing sections。

- [ ] **Step 3: 重写职责边界与改造动机**

明确 Stanford STORM 原有多视角调研、访谈、大纲和写作；个人负责论文/本地 PDF、统一 RAG、Memory/Context、Runtime、Evaluation、Observability 与演示层，禁止冒领上游能力。

- [ ] **Step 4: 编写改进矩阵**

列固定为 `原结构/问题 | 如何发现 | 技术决策 | 为什么这样选 | 结果 | 局限`，覆盖 CJK->Jieba、固定段落->Parent-Child、单路->Hybrid/RRF、MiniLM->多 Profile、Exact->HNSW、全量->选择性 Cross-Encoder、无约束->证据治理、不可定位->Langfuse/local trace。

- [ ] **Step 5: 编写岗位化 Bullet**

分别提供 RAG/知识库、Agent Runtime/Harness、Evaluation/Observability 各 4 条，以及校招精简版 3 条；每条使用“负责范围 + 难点 + 技术行动 + 有边界结果”。

- [ ] **Step 6: 验证并提交**

```powershell
D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest tests.test_paperstorm_career_docs
git add docs/PAPERSTORM_RESUME_GUIDE.md tests/test_paperstorm_career_docs.py
git commit -m "docs(career): expand PaperStorm project narrative"
```

### Task 5: 事实审查与发布验证

**Files:**
- Review: `docs/RAG_AGENT_INTERVIEW_PLAYBOOK.md`
- Review: `docs/PAPERSTORM_RESUME_GUIDE.md`
- Review: `tests/test_paperstorm_career_docs.py`

- [ ] **Step 1: 扫描夸大表述**

Run: `rg -n "生产准确率|行业领先|SOTA|全部解决|零幻觉|无损" docs/RAG_AGENT_INTERVIEW_PLAYBOOK.md docs/PAPERSTORM_RESUME_GUIDE.md`

Expected: 只出现在“不能声称”或反例语境。

- [ ] **Step 2: 审查数字与协议**

确保每个小数指标可在 README 或冻结报告找到；不可混写 SciFact Recall、QASPER Evidence Recall、QASPER Answer F1 与 LongMemEval session recall。

- [ ] **Step 3: 运行定向与全量测试**

```powershell
D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest tests.test_paperstorm_career_docs
$env:PAPERSTORM_OFFLINE_TESTS='1'
$env:PAPERSTORM_RETRIEVAL_EMBEDDING='hash'
D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest discover -s tests -p 'test_*.py'
```

Expected: 不低于 v7.1 基线 `605 passed, 5 skipped`，无失败。

- [ ] **Step 4: 最终内容审查并提交修正**

检查 100 题无重复填充、排查题先定位再给方案、社区问题未被当作技术事实、职责未冒领上游实现。

```powershell
git add docs/RAG_AGENT_INTERVIEW_PLAYBOOK.md docs/PAPERSTORM_RESUME_GUIDE.md tests/test_paperstorm_career_docs.py
git commit -m "docs: finalize PaperStorm interview deep dive"
```
