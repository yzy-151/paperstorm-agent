# PaperStorm Agent 更新计划

更新时间：2026-07-27

本文档是后续版本计划。每次项目更新都必须维护本文件，记录版本目标、完成情况、验收标准和下一步。

## 1. 项目长期目标

项目新定位：

```text
PaperStorm Agent：面向论文调研与知识库问答的 RAG + Memory + Multi-Agent 实验平台
```

与 `nonlinear-nn-agent` 的互补关系：

- `nonlinear-nn-agent`：从零实现轻量 Agent Harness，重点是 Runtime、ToolRegistry、Hook、Session、Trace、Async、Retry、实验自动化。
- `PaperStorm Agent`：基于成熟 STORM/RAG 框架二次开发，重点是论文 RAG、知识库、Memory、MCP、多 Agent 调研协作、评测和前端展示。

目标岗位能力映射：

- 企业级 AI 平台架构：AI 网关、Agent 构建平台、知识库平台、组件库。
- 生产级 Agent 系统：任务规划、工具调用、记忆管理、Multi-Agent 编排。
- 稳定性保障：响应延迟、并发、错误容灾、可观测性、评测指标。
- 文档能力：结构化技术文档、版本计划、简历/面试材料、指标驱动决策。

## 2. 已完成版本

### v0.1：PaperStorm MVP 与运行稳定性

状态：已完成。

核心能力：

- 基于 STORM 四阶段 pipeline：research -> outline -> article -> polish。
- 接入 DeepSeek / MiniMax。
- 接入 arXiv 和 Local PDF 检索。
- 支持中文输出。
- 修复 Windows UTF-8、surrogate、输出目录名、run_config 脱敏等问题。
- 修复 LLM 空 query、Markdown/JSON query 噪声。
- 修复空检索结果导致文章生成崩溃。
- 修复 Wikipedia 辅助抓取失败时日志过噪。
- PIM 缩写消歧：`passive intermodulation` vs `processing-in-memory`。

关键产物：

```text
examples/storm_examples/run_paper_storm_minimax.py
knowledge_storm/rm.py
tests/test_paperstorm_retrievers.py
tests/test_paperstorm_logging.py
tests/test_minimax_runtime_fixes.py
```

### v0.1.1：Runtime Trace / Tool Schema / MCP / Eval

状态：已完成。

核心能力：

- `paperstorm_trace.jsonl`：记录工具调用、耗时、失败和产物。
- `run_summary.json`：记录一次运行摘要。
- `PaperStormTool` / `RetrievalTool` / `ArxivSearchTool` / `LocalPDFSearchTool`。
- MCP-style stdio server：`tools/list`、`tools/call`、结构化错误返回。
- Eval Harness v1：输出 `scorecard.json`、`scorecard.md`。

关键产物：

```text
knowledge_storm/paperstorm_tools.py
knowledge_storm/paperstorm_eval.py
examples/storm_examples/paperstorm_mcp_server.py
examples/storm_examples/evaluate_paperstorm_run.py
examples/storm_examples/paperstorm_eval_cases.json
tests/test_paperstorm_mcp_server.py
tests/test_paperstorm_eval.py
```

### v0.1.2：文档归并与中文 README

状态：已完成。

核心能力：

- 保留官方 STORM README 为 `README_STORM_OFFICIAL.md`。
- 将项目 README 改为中文 `PaperStorm Agent` 首页。
- 将 `docs` 根目录收敛为三份主文档：
  - `OPERATION_GUIDE.md`
  - `VERSION_PLAN.md`
  - `RESUME_INTERVIEW_PLAN.md`
- 旧文档归档到 `docs/archive/2026-07-27-legacy-docs/`。
- 根据 Agent 岗 JD 和面试经历，明确后续版本主线：RAG、Memory、Multi-Agent、MCP、Eval、前端展示。

分支：

```text
version/v0.1.2-docs-roadmap
```

### v0.1.3：GitHub 改名与分支治理

状态：进行中。

已完成：

- GitHub 仓库从 `yzy-151/storm` 改名为 `yzy-151/paperstorm-agent`。
- 本地 `fork` remote 更新为 `https://github.com/yzy-151/paperstorm-agent.git`。
- `version/v0.1.2-docs-roadmap` 已合并进 `fork/main`。
- 已删除旧阶段远程分支：
  - `codex/minimax-agent-storm`
  - `feature/paperstorm-query-quality`
  - `feature/paperstorm-retrieval-quality`
  - `feature/paperstorm-runtime-tracing`
  - `feature/paperstorm-tool-schema`
  - `feature/paperstorm-mcp-server`

保留：

```text
main
version/v0.1.2-docs-roadmap
feature/paperstorm-eval-harness
```

待确认：

- 是否删除 fork 中继承自 Stanford 官方仓库的旧分支，例如 `dev-gemini`、`dev-chinese`、`costorm-integration` 等。

## 3. v0.2：RAG 质量与 Memory 模块

目标：把 PaperStorm 从“检索后生成文章”升级为“带记忆和可评估检索质量的论文 RAG Agent”。

### 功能目标

1. Query Planner
   - 区分探索型 query、定义型 query、方法型 query、评测型 query。
   - 保存每条 query 的来源、persona、轮次和意图。

2. Memory Store v1
   - 短期记忆：一次运行中的 persona、query、已读论文、已拒绝跑题论文。
   - 长期记忆：跨运行保存 topic summary、paper summary、已知缩写消歧规则。
   - 存储格式先用本地 JSON/JSONL，不急着上数据库。

3. RAG 审计
   - 记录每条检索结果为什么保留/过滤。
   - 输出 `retrieval_audit.json`。
   - 支持 `expected_keywords` / `forbidden_keywords` 规则。

4. Eval Harness v2
   - 增加 retrieval precision@k。
   - 增加 off-topic rate。
   - 增加 citation coverage。
   - 增加 answer groundedness 的规则版检查。

### 验收标准

- 能解释一次运行中每条 query 的来源。
- 能看到哪些论文被过滤以及原因。
- 同一 topic 第二次运行可以复用 memory 中的领域消歧信息。
- 至少 3 个 eval cases。
- 测试不少于 45 个。

### 简历价值

可写：

```text
为 PaperStorm 增加 Memory Store 与 RAG 审计链路，记录 query intent、检索来源、过滤原因和跨任务 topic summary，并基于 precision@k、off-topic rate、citation coverage 评估检索质量。
```

## 4. v0.3：Multi-Agent 论文调研协作

目标：把 STORM 原有多视角对话思想显式工程化，形成可解释的 Multi-Agent 调研编排。

### Agent 角色建议

- `PlannerAgent`：拆解调研任务，生成 query plan。
- `RetrieverAgent`：执行 arXiv / LocalPDF / Web 检索。
- `CriticAgent`：识别跑题、重复、引用不足。
- `MemoryAgent`：维护 topic memory、paper memory、缩写规则。
- `WriterAgent`：生成 outline 和 article。
- `EvaluatorAgent`：运行 scorecard 并给出改进建议。

### 功能目标

- 明确每个 Agent 的输入、输出和状态字段。
- 记录多 Agent 消息流。
- 支持中心化 orchestrator。
- 先不做复杂并发，优先保证可审计。

### 验收标准

- 输出 `agent_trace.jsonl`。
- 每个 Agent 的决策可在 trace 中复盘。
- 至少一个 case 展示 CriticAgent 发现 PIM 跑题结果。
- Eval 能比较单 Agent 与 Multi-Agent 流程的结果差异。

### 简历价值

可写：

```text
设计多角色论文调研 Agent 编排，将规划、检索、记忆、批判、写作和评估拆分为独立 Agent，并通过中心化 orchestrator 记录 agent_trace，实现多 Agent 协作链路可观测。
```

## 5. v0.4：知识库平台化与服务 API

目标：靠近企业级知识库平台和 Agent 构建平台。

### 功能目标

- FastAPI 服务化。
- 任务提交：`POST /research-tasks`。
- 状态查询：`GET /research-tasks/{task_id}`。
- 报告读取：`GET /research-tasks/{task_id}/article`。
- 知识库导入：PDF / Markdown / arXiv。
- 本地任务状态存储。
- 敏感信息脱敏。

### 验收标准

- 不真实调用 LLM 的 API 层测试可通过。
- 支持 task_id、queued/running/succeeded/failed 状态。
- 支持读取 scorecard。
- 测试不少于 55 个。

### 简历价值

可写：

```text
将 PaperStorm 命令行 pipeline 服务化为 FastAPI Agent API，支持任务提交、状态追踪、报告读取、scorecard 获取和错误脱敏，为后续前端展示和企业知识库平台化奠定基础。
```

## 6. v0.5：前端展示 Demo

目标：像 `nonlinear-nn-agent` 最后要做前端一样，PaperStorm 也需要可展示界面。

### 功能目标

- 前端输入 topic / PDF 目录。
- 展示任务状态。
- 展示 query plan。
- 展示检索结果与过滤原因。
- 展示 memory 摘要。
- 展示 trace 时间线。
- 展示最终文章和 scorecard。

技术建议：

```text
FastAPI + 简单 HTML/React/Vite
```

第一版不追求华丽 UI，重点是展示 Agent 执行链路。

### 验收标准

- 一键启动本地 demo。
- 能展示一次已完成 run 的 report、trace、scorecard。
- 不依赖真实 API key 也能用样例数据预览。

### 简历价值

可写：

```text
实现 PaperStorm Agent 前端 Demo，展示任务状态、检索审计、memory 摘要、工具调用 trace、最终报告和 scorecard，使 Agent 执行链路从黑盒变为可视化调试界面。
```

## 7. v1.0：Agent 平台化 Demo

目标：形成可投递、可演示、可面试讲 5 分钟的完整项目。

### 能力边界

做到：

- RAG 知识库。
- Memory。
- 多 Agent 编排。
- Tool Schema / MCP。
- Runtime Trace。
- Eval Harness。
- API 服务化。
- 前端展示。
- README 中文文档。
- 简历问答文档。

暂不承诺：

- 多租户。
- 权限系统。
- 高并发线上服务。
- 云部署。
- 企业级监控告警。

## 8. 每次版本更新模板

```markdown
## vX.Y：版本名称

### 状态

计划中 / 开发中 / 已完成

### 本次完成

- ...

### 验证命令

```powershell
...
```

### 验证结果

```text
...
```

### 文档更新

- README.md
- docs/VERSION_PLAN.md
- docs/RESUME_INTERVIEW_PLAN.md

### 下一步

- ...
```
