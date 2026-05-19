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
- 设计 `PaperStormMemoryStore` 三层记忆、`compress_context` 上下文压缩和 `PaperStormKnowledgeBase` 问答模块，将调研产物转化为可问答知识库，并要求 QA 返回 citations、grounded 和 evidence。
- 新增 `PaperStormRuntimeSession`，统一 tool registry、tool call trace 和 working memory 写入，形成轻量 Agent Runtime 雏形。
- 将 runtime 升级为 ToolRegistry + HookManager + RuntimeEvent 结构，统一工具注册、参数校验、生命周期 hook 和 JSONL trace 字段，并让 MCP server 共用同一套工具注册模型。
- 设计多 Agent 调研编排层，将任务拆分为 Planner、Retriever、Critic、Memory、Evaluator 等角色，通过中心化 orchestrator 输出 `agent_trace.jsonl` 和 `multi_agent_report.json`，使规划、检索、过滤、记忆和评估过程可复盘。

压缩版 bullet：

- 基于 Stanford STORM 二次开发 PaperStorm Agent，接入 DeepSeek/MiniMax、arXiv 与本地 PDF 检索，复用多阶段 RAG 流程生成中文论文综述。
- 实现 query 清洗、PIM 歧义消解、空检索防护、外部 API 失败降级和 JSONL runtime trace，提升 Agent 工具链稳定性与可观测性。
- 抽象 Tool Schema 并实现 MCP-style server 与 Eval Harness，支持工具发现/调用、结构化错误和 scorecard 评估。
- 增加三层记忆、上下文压缩、知识库 QA 和轻量 Runtime Session，使调研结果可追踪、可问答、可评估。
- 设计 ToolRegistry、HookManager 与统一 RuntimeEvent，将 PaperStorm 工具链升级为可观测、可扩展的轻量 Agent Harness。
- 增加 Planner/Retriever/Critic/Memory/Evaluator 多 Agent 编排和 agent trace，支持对 PIM 跑题检索结果给出过滤理由。

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
直接调 LLM API 只解决生成问题，Agent Harness 解决执行问题。它需要工具注册、参数校验、调用调度、timeout/retry、Hook、session、trace、错误恢复和评测。我的 Nonlinear 项目从零实现了 ToolRegistry、HookManager、SessionStore 和 TraceLogger；PaperStorm 则把这些思想落到 RAG 论文调研流程里，已经实现 PaperStormTool、ToolRegistry、HookManager、RuntimeEvent、MCP server、runtime trace、memory 和 eval harness。
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
我把记忆拆成 working、episodic、semantic 和 preferences。working 保存一次任务中的工具调用和当前上下文；episodic 保存一次运行经历，例如某个 topic 容易召回哪些跑题结果；semantic 保存可复用知识，例如 PIM 在射频场景中指 passive intermodulation；preferences 保存用户偏好，例如中文输出。trace 用来复盘执行链路，memory 用来影响后续检索和问答，两者职责分开。
```

项目状态：

```text
Nonlinear 已有 SessionStore 和 context_summary 字段；PaperStorm 已实现 `PaperStormMemoryStore`、`compress_context` 和 `PaperStormRuntimeSession`，并在 v0.3 补了 `HookManager`、`ToolRegistry` 和统一 `RuntimeEvent` trace schema。
```

### Q3.3：Workflow 和 Runtime 有什么区别？为什么这个项目需要 Runtime？

答题素材：

```text
Workflow 描述业务步骤，比如 research、outline、article、polish、qa；Runtime 描述这些步骤如何被稳定执行，比如工具怎么注册、参数怎么校验、失败怎么记录、hook 怎么插入、trace 怎么统一、memory 怎么写入。PaperStorm 原来更像 workflow，我在 v0.3 加了 ToolRegistry、HookManager、RuntimeEvent 和 RuntimeSession，让工具调用和上下文压缩都能被统一追踪。这样后续 Multi-Agent、服务化和前端 trace timeline 都有底座。
```

### Q3.4：Hook 机制有什么用？

答题素材：

```text
Hook 是 Agent Runtime 的生命周期扩展点。比如 before_tool_call 可以记录输入摘要或做参数校验，after_tool_call 可以记录耗时和输出摘要，on_tool_error 可以统一收敛错误，on_context_compress 可以记录压缩是否保留关键约束。它的价值是让核心业务代码不用到处写日志、指标和错误处理，后续也方便接可观测面板和告警。
```

### Q3.1：上下文压缩怎么做，如何避免压缩丢关键信息？

答题素材：

```text
我没有把压缩理解成简单截断，而是输出结构化摘要、保留事实、约束和 validation。比如 PaperStorm 的 compress_context 会检查 expected_keywords 是否仍然保留，同时检查 forbidden_keywords 是否混入摘要。这样可以解释压缩后的上下文是否还能约束下一步工具调用和生成。
```

### Q3.2：为什么要做知识库 QA，会不会降低项目格调？

答题素材：

```text
普通聊天式 QA 价值不高，但企业内部文档知识库 Agent 是真实需求。我的做法不是套一个聊天壳，而是把 PaperStorm 调研产物和本地 PDF 检索结果作为证据层，QA 必须返回 citations、grounded 和 evidence，并纳入 Eval Harness 检查。这能对应企业知识库平台、RAG grounded answer 和可追踪问答链路。
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
简单线性任务优先单 Agent，减少通信和状态复杂度；当任务天然包含规划、检索、批判、记忆、写作、评估等不同职责时适合 Multi-Agent。PaperStorm 里我没有为了形式硬拆，而是把论文调研中真实存在的职责拆成 PlannerAgent、RetrieverAgent、CriticAgent、MemoryAgent 和 EvaluatorAgent，并用中心化 orchestrator 记录 agent_trace。比如 PIM case 中 CriticAgent 会把 processing-in-memory / DRAM / RAM 这类跑题结果过滤掉，并记录 reason。
```

### Q8.1：你的 Multi-Agent 是不是几个 prompt 拼起来？

答题素材：

```text
不是。我先做的是可测试的工程编排层，而不是依赖 LLM 随机输出。每个 Agent 都有明确输入输出：Planner 输出 query plan，Retriever 通过 runtime 调工具，Critic 输出 kept/rejected 和 reason，Memory 写 episodic memory，Evaluator 输出 scorecard。所有 Agent 的 start/end 都写入 agent_trace.jsonl。这让 Multi-Agent 的价值可以被测试和复盘，后续再把 Planner/Critic 替换成 LLM 版本，也有规则版 baseline。
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

企业内部文档知识库 Agent 是非常真实的落地场景。很多企业会把制度文档、研发文档、接口文档、故障手册、产品手册、客户支持文档放进知识库，让 Agent 做检索问答、引用溯源、流程辅助和问题定位。

所以 PaperStorm 做“普通知识库 QA”不会降低项目格调。会降低格调的是只做一个没有评测、没有引用、没有审计的“上传 PDF 聊天”demo。

正确定位应该是：

```text
企业内部文档 / 论文资料知识库 Agent：支持文档导入、chunk metadata、检索审计、引用溯源、memory、scorecard 和前端 trace 展示。
```

v0.4 后重点展示：

- 文档导入。
- chunk/metadata。
- 普通知识库 QA。
- query planning。
- 检索审计。
- memory recall。
- scorecard 评估。
- 引用来源和召回片段展示。

面试可说：

```text
PaperStorm 最早是论文调研 Agent，但底层能力可以迁移到企业内部文档知识库。论文、研发文档、接口文档本质上都需要解析、切块、metadata、检索、rerank、引用溯源和评测。我的规划不是做泛聊天，而是把知识库 QA 做成可审计 RAG：每次回答都能看到召回片段、来源、过滤原因和 scorecard。
```

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
- rate limit。
- stress test report。
- frontend timeline。

高并发回答边界：

```text
当前 PaperStorm 还不是线上高并发系统。后续会按工程路径推进：先 FastAPI task_id 和状态隔离，再单 worker 后台执行，再加任务队列、并发数限制、timeout/retry、rate limit 和 fake runner 压测。真正生产级还需要鉴权、权限、监控告警、分布式队列和成本治理。
```

### Q11：做普通知识库 QA 会不会降低项目水平？

推荐回答：

```text
不会，企业内部文档知识库 Agent 是真实需求。关键在于不要只做“上传 PDF 问答”的 demo，而要做企业知识库需要的工程能力：文档导入、chunk metadata、混合检索、rerank、引用溯源、权限/版本意识、检索审计、评测指标和前端可观测。PaperStorm 的优势是已经有论文 RAG、LocalPDF、trace、MCP 和 eval，普通知识库 QA 可以作为平台能力补齐，亮点仍然放在 Memory、Multi-Agent、MCP 和 Eval 上。
```

### Q12：内部文档知识库 Agent 和 PaperStorm 有什么关系？

推荐回答：

```text
两者底层链路相同：文档解析、chunk、metadata、query planning、retrieval、context assembly、grounded generation、citation 和 eval。区别是 PaperStorm 的领域对象是论文和调研报告，企业知识库的对象是内部文档、接口文档、故障手册和产品文档。PaperStorm 后续可以把论文知识库抽象成通用 KB 层，再保留论文调研作为一个 Agent template。
```

### Q13：高并发怎么做？难点是什么？

推荐回答：

```text
我会分阶段做，不会一上来声称高并发。第一阶段 FastAPI task_id，保证每个任务状态、output_dir、trace、scorecard 隔离；第二阶段后台 worker 和队列，限制 max_concurrent_tasks；第三阶段给 LLM、检索、embedding 工具加 timeout/retry/rate limit；第四阶段用 fake runner 做压测，统计平均延迟、P95、失败率和 retry 次数。难点是外部 API 限流、embedding 模型复用、文件写入隔离、阻塞调用与 async 混用、失败任务状态恢复。
```

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
