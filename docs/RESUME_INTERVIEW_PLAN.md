# PaperStorm Agent 简历与面试计划

更新时间：2026-07-27

本文档面向投递简历和面试准备。它把 PaperStorm Agent、Nonlinear NN Agent、面试经历汇总和目标岗位 JD 统一起来，后续每次项目更新都要维护。

## 1. 目标岗位

目标方向：

```text
Agent 开发工程师 / LLM 应用工程师 / Agent Harness / RAG 工程方向
```

目标 JD 关键词：

- 企业级 AI 平台架构设计与研发。
- AI 网关、MarketPlace、智能体构建平台、知识库平台。
- 通用 Agent 开发框架与组件库。
- 任务规划、工具调用、记忆管理、Multi-Agent 编排。
- 响应延迟、并发能力、错误容灾。
- Python / Java / Go / TypeScript。
- LLM 基础原理、AI Agent、RAG、知识库系统、Multi-Agent。
- 结构化技术文档、跨角色沟通、可量化指标、数据驱动决策。

## 2. 两个项目的互补定位

### PaperStorm Agent

定位：

```text
基于成熟 STORM/RAG 框架二次开发的论文调研 Agent，重点展示 RAG、Memory、MCP、Multi-Agent、Eval、知识库方向。
```

适合回答：

- RAG 全流程怎么做？
- query 怎么规划和清洗？
- 检索跑题怎么解决？
- MCP 工具怎么暴露？
- Agent 怎么评测？
- 论文/知识库 Agent 怎么设计？

### Nonlinear NN Agent

定位：

```text
从零实现轻量 Agent Harness Runtime，重点展示 ToolRegistry、Hook、Session、Trace、Async、Retry、实验自动化。
```

适合回答：

- Agent Harness 和直接调用 LLM API 有什么区别？
- 工具系统怎么设计？
- Hook 怎么做？
- session 和 trace 为什么分开？
- 失败重试和可观测怎么做？
- Agentic Loop 如何异步执行？

## 3. PaperStorm 简历项目描述

项目名称建议：

```text
PaperStorm Agent：中文论文调研与知识库 RAG Agent
```

简历短描述：

```text
基于 Stanford STORM 二次开发中文论文调研 Agent，接入 DeepSeek/MiniMax、arXiv 与本地 PDF 检索，围绕 RAG、Tool Calling、MCP、Runtime Trace 和 Eval Harness 构建可观测、可评估的论文调研流程。
```

推荐 bullet：

- 基于 Stanford STORM 二次开发 PaperStorm Agent，接入 DeepSeek/MiniMax、arXiv API 与本地 PDF 检索，将论文摘要、PDF 片段和网页资料统一为 `Information` schema，复用 research-outline-article-polish 多阶段流程生成中文综述报告。
- 改进 RAG 工具链稳定性：实现 LLM query sanitizer、PIM 缩写消歧、空检索防护、arXiv 单 query 失败降级和 Wikipedia 抓取防御，减少外部工具异常对主流程的影响。
- 新增 Runtime Trace，输出 `paperstorm_trace.jsonl` 与 `run_summary.json`，记录工具调用、耗时、结果数量、失败原因和产物路径，使 Agent 执行链路可复盘。
- 抽象 `PaperStormTool` schema，将 arXiv / Local PDF 检索封装为可发现、可调用工具，并实现 MCP-style stdio server 支持 `tools/list`、`tools/call` 和结构化错误返回。
- 实现规则版 Eval Harness，读取检索结果、文章、trace 和 summary 输出 `scorecard.json/md`，从任务完成度、检索相关性、跑题率、文章质量和 runtime 可观测性评估 Agent 效果。

压缩版 bullet：

- 基于 Stanford STORM 二次开发 PaperStorm Agent，接入 DeepSeek/MiniMax、arXiv 与本地 PDF 检索，复用多阶段 RAG 流程生成中文论文综述。
- 实现 query 清洗、PIM 歧义消解、空检索防护、外部 API 失败降级和 JSONL runtime trace，提升 Agent 工具链稳定性与可观测性。
- 抽象 Tool Schema 并实现 MCP-style server 与 Eval Harness，支持工具发现/调用、结构化错误和 scorecard 评估。

## 4. Nonlinear NN Agent 简历项目描述

项目名称建议：

```text
Agentic Experiment Harness for Nonlinear System Modeling
```

推荐 bullet：

- 从零实现面向算法实验的轻量 Agent Harness Runtime，将非线性神经网络 MPDPD 拟合实验拆解为配置生成、训练执行、NMSE 评估、PSD 验证和报告生成等工具链。
- 抽象 `ToolRegistry`、`HookManager`、`SessionStore`、`TraceLogger` 等 runtime 组件，支持异步工具调用、timeout/retry、before/after/error/metric hooks、session resume 与 JSONL trace。
- 将原始单脚本 CNN 仿真实验重构为可配置、可测试、可审计实验工程，支持 YAML 配置、固定随机种子、metrics JSON 解析、PSD 产物验证和 Markdown 摘要。
- 在 4000 参数约束下完成小模型搜索，引入复数记忆多项式特征 + 闭式最小二乘，获得 3626 参数、NMSE -37.42 dB 的轻量可解释模型，并生成实验对比报告。

## 5. 面试高频问题与项目回答

面试经历汇总显示，Agent 开发高频考点排序：

1. 上下文管理 / 记忆系统。
2. RAG 全流程。
3. Agent 架构设计。
4. 工具系统 / Tool Calling。
5. MCP。
6. Agent Harness / Runtime。
7. RAG 评测体系。
8. 幻觉处理。
9. Skill 封装。
10. ReAct。
11. LangChain / LangGraph。
12. Token 优化 / 成本。
13. AI Coding 方法论。
14. Multi-Agent 协作。
15. Agent 发展趋势。

### Q1：Agent Harness 和直接调用 LLM API 有什么区别？

答题素材：

```text
直接调 LLM API 只解决生成问题，Agent Harness 解决执行问题。它需要工具注册、参数校验、调用调度、timeout/retry、Hook、session、trace、错误恢复和评测。我的 Nonlinear 项目从零实现了 ToolRegistry、HookManager、SessionStore 和 TraceLogger；PaperStorm 则把这些思想落到 RAG 论文调研流程里，增加了 tool schema、MCP server、runtime trace 和 eval harness。
```

### Q2：RAG 全流程怎么做？

答题结构：

```text
文档/论文来源 -> 清洗与解析 -> chunk / metadata -> query planning -> hybrid retrieval -> rerank/filter -> context assembly -> grounded generation -> citation/eval
```

结合 PaperStorm：

```text
当前 PaperStorm 已完成 arXiv / Local PDF 检索、Information schema、query 清洗、PIM 消歧、结果过滤、文章生成和 scorecard。v0.2 计划补 retrieval_audit、memory store、precision@k、off-topic rate 和 citation coverage。
```

### Q3：上下文和记忆怎么设计？

当前回答：

```text
短期记忆保存一次运行内的 persona、query、已读论文、已过滤论文和当前 outline；长期记忆保存跨运行的 topic summary、paper summary、缩写消歧规则和用户偏好。原始 trace 和 summary 分开存，trace 用于审计，memory 用于下次召回。
```

项目状态：

```text
Nonlinear 已有 SessionStore 和 context_summary 字段；PaperStorm v0.2 将实现 Memory Store v1。
```

### Q4：工具调用失败怎么办？

答题素材：

```text
工具失败分级处理：参数错误直接返回结构化错误；外部 API 超时/429 视为部分失败，可跳过或重试；关键工具失败写入 trace 和 summary；上层 Agent 根据失败类型决定重试、换 query、换工具或向用户解释。PaperStorm 已经把 arXiv 单 query 失败降为 INFO 并继续主流程，MCP server 也会对未知工具和内部异常返回 JSON-RPC 风格错误。
```

### Q5：MCP 是什么？你写过什么？

答题素材：

```text
MCP 可以理解成工具发现和工具调用的标准协议边界。我的 PaperStorm 先抽象 PaperStormTool schema，包含 name、description、input_schema、output_schema 和 run(arguments)，再实现 MCP-style stdio server，支持 tools/list 返回工具 schema，tools/call 调用 arXiv 和本地 PDF 检索，并返回结构化错误。
```

### Q6：怎么评估 Agent 变好了？

答题素材：

```text
不能只靠主观感觉。我给 PaperStorm 做了 Eval Harness，读取 raw_search_results、outline、polished article、paperstorm_trace 和 run_summary，输出 scorecard.json/md。指标包括任务完成度、检索相关性、跑题率、文章质量和 runtime 可观测性。比如 PIM case 用 passive intermodulation/RF/neural network 作为正向关键词，用 processing-in-memory/RAM/DRAM 作为负向关键词，量化检索是否跑题。
```

### Q7：如何控制幻觉？

答题结构：

```text
源头控制：query planning 和领域消歧。
检索控制：expected/forbidden keywords、rerank/filter。
生成控制：要求引用 evidence，不足时说缺信息。
运行控制：trace 审计每次工具调用。
评估控制：scorecard 记录 citation coverage、off-topic rate、groundedness。
```

### Q8：单 Agent 和 Multi-Agent 怎么取舍？

当前观点：

```text
简单线性任务优先单 Agent，减少通信和状态复杂度；当任务天然包含规划、检索、批判、记忆、写作、评估等不同职责时适合 Multi-Agent。PaperStorm v0.3 计划拆分 PlannerAgent、RetrieverAgent、CriticAgent、MemoryAgent、WriterAgent、EvaluatorAgent，并用中心化 orchestrator 记录 agent_trace。
```

### Q9：Skill 和 MCP 有什么区别？

答题素材：

```text
Skill 更像 Agent 内部能力封装和触发策略，强调什么时候用、怎么用、上下文如何组织；MCP 更像外部工具协议，强调工具如何被发现、schema 如何描述、调用如何传参、结果和错误如何返回。两者可以配合：Skill 决定何时调用某类能力，MCP 提供标准化工具入口。
```

### Q10：如何面对生产级性能与稳定性？

结合未来规划：

```text
当前项目处于本地原型阶段，已经做了错误降级、trace、eval 和测试。后续版本会补 FastAPI task_id、异步任务、状态查询、前端展示、错误分类、检索缓存、memory 召回和固定 benchmark。真正生产级还需要鉴权、限流、队列、并发压测、监控告警和数据权限。
```

## 6. 针对目标 JD 的项目强化方向

### AI 网关 / MarketPlace / Agent 构建平台

PaperStorm 可以演进成一个垂直 Agent 构建平台 demo：

- 工具市场：arXiv、LocalPDF、WebSearch、Eval、Memory。
- Agent 模板：论文综述 Agent、领域调研 Agent、实验规划 Agent。
- 组件库：Retriever、Memory、Reranker、Evaluator、Writer。

### 知识库平台

v0.4 后重点展示：

- 文档导入。
- chunk/metadata。
- query planning。
- 检索审计。
- memory recall。
- scorecard 评估。

### 生产级 Agent 系统

当前可讲：

- Tool schema。
- MCP server。
- trace。
- structured error。
- eval。

后续补：

- task queue。
- retry policy。
- timeout policy。
- concurrent runs。
- task status。
- frontend timeline。

## 7. 简历投递策略

### Agent Harness / Runtime 岗

项目顺序：

1. Nonlinear NN Agentic Experiment Harness。
2. PaperStorm Agent。
3. 华为通信算法实习。

### RAG / 知识库岗

项目顺序：

1. PaperStorm Agent。
2. Nonlinear NN Agentic Experiment Harness。
3. 华为通信算法实习。

### AI 平台 / Agent 平台岗

重点写：

- Tool Schema / MCP。
- Memory 计划。
- Eval Harness。
- API / 前端版本计划。
- 文档和指标驱动。

## 8. 后续每次更新必须补充

每完成一个版本，在本文件追加：

```markdown
## YYYY-MM-DD 面试与简历更新

### 新增项目能力

### 简历 bullet 更新

### 面试可能追问

### 推荐回答

### 不能夸大的边界
```
