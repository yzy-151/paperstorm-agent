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
- 抽象 `PaperStormTaskService` 服务核心层，支持 task_id、queued/running/succeeded/failed 状态、独立 output_dir、文章/trace/scorecard 读取、知识库 QA、fake runner 和错误脱敏，并提供可选 FastAPI 适配器。
- 增加服务层并发与稳定性 baseline，支持 `max_concurrent_tasks`、`worker_tick`、running 任务容量释放、stale running 恢复和 fake runner 压测报告。
- 实现静态 PaperStorm Dashboard，基于样例数据展示任务状态、文章、知识库 QA、runtime trace、scorecard、Multi-Agent 保留/过滤结果和 stress report，并补充官方 STORM 架构中文说明与架构图。
- 将真实 STORM pipeline 封装为可注入 worker，`PaperStormTaskService` 支持 `run_mode="paperstorm"`，统一 fake 和真实任务的 task_id、状态、产物目录、trace、scorecard 和错误处理。
- 为 Dashboard 增加 service-backed snapshot 接口，聚合 task、article、QA、scorecard、trace、pipeline worker 元数据和 service snapshot，支持前端通过 task_id 读取真实任务产物。
- 将 Dashboard 从结果查看器扩展为本地 Agent 控制台，支持创建任务、运行任务、刷新任务列表、轮询选中任务并展示结构化失败原因。
- 新增 v1.0 release demo，一条命令复现 service task、文章、QA、trace、scorecard 和 Dashboard 样例数据，形成可投递、可演示、可面试讲解的 Agent 平台原型。

压缩版 bullet：

- 基于 Stanford STORM 二次开发 PaperStorm Agent，接入 DeepSeek/MiniMax、arXiv 与本地 PDF 检索，复用多阶段 RAG 流程生成中文论文综述。
- 实现 query 清洗、PIM 歧义消解、空检索防护、外部 API 失败降级和 JSONL runtime trace，提升 Agent 工具链稳定性与可观测性。
- 抽象 Tool Schema 并实现 MCP-style server 与 Eval Harness，支持工具发现/调用、结构化错误和 scorecard 评估。
- 增加三层记忆、上下文压缩、知识库 QA 和轻量 Runtime Session，使调研结果可追踪、可问答、可评估。
- 设计 ToolRegistry、HookManager 与统一 RuntimeEvent，将 PaperStorm 工具链升级为可观测、可扩展的轻量 Agent Harness。
- 增加 Planner/Retriever/Critic/Memory/Evaluator 多 Agent 编排和 agent trace，支持对 PIM 跑题检索结果给出过滤理由。
- 抽象文件存储版 Agent Task Service，支持任务状态、产物隔离、知识库 QA、scorecard/trace 查询和可选 FastAPI 路由。
- 增加任务队列、并发上限、stale task 恢复和 stress benchmark，输出平均延迟、P95、失败率和最大观察并发。
- 实现静态 Dashboard 和官方 STORM 中文架构文档，将 Agent 执行链路、评估结果和稳定性报告可视化展示。
- 接入真实 PaperStorm pipeline worker，并通过 runner 注入让单元测试不依赖真实 LLM/API，保留 fake baseline 作为稳定回归测试。
- 将 Dashboard 接入 service dashboard bundle，支持输入 service URL 和 task_id 查看真实任务的 article、trace、scorecard 和 worker 元数据。
- 将 Dashboard 升级为轻量 Agent 控制台，支持提交/运行/轮询 task，并把失败 error 纳入可观测面板。
- 增加 release demo 生成器，把 RAG、Memory、Runtime Trace、Eval、Task Service 和 Dashboard 串成可复现的 5 分钟本地演示。

## 3.0 最终简历 bullet

投 Agent / RAG / 知识库平台方向时，建议最终只放 4 到 5 条，不要把所有版本都塞进简历。

```text
- 基于 Stanford STORM 二次开发 PaperStorm Agent，接入 DeepSeek/MiniMax、arXiv 与本地 PDF 检索，复用 research -> outline -> article -> polish 多阶段 RAG 流程生成中文论文综述。
- 抽象 PaperStormTool、ToolRegistry、HookManager 和 RuntimeEvent，统一工具 schema、参数校验、生命周期 hook、JSONL trace 和结构化错误，使 Agent 执行链路可复盘。
- 构建 Memory / Context Compression / QA 模块，将调研产物转化为可问答知识库，支持 working/episodic/semantic 三层记忆、grounded answer、citations 和 QA scorecard。
- 设计 Planner/Retriever/Critic/Memory/Evaluator 多 Agent 编排，对 PIM 场景中的 processing-in-memory、RAM、DRAM 跑题检索结果进行过滤并记录 critic reason。
- 抽象 PaperStormTaskService 与 FastAPI 适配器，支持 task_id、queued/running/succeeded/failed 状态、产物隔离、Dashboard 展示、并发 baseline、release demo 和 scorecard 评估。
```

最终简历短描述：

```text
PaperStorm Agent 是一个基于 Stanford STORM 二次开发的中文论文调研与知识库 Agent，覆盖 RAG、Memory、Tool Calling、MCP-style tools、Multi-Agent、Runtime Trace、Eval Harness、Task Service 和 Dashboard 演示。
```

## 3.0.1 最终面试 FAQ 精简版

1. 你这个项目解决什么问题？
   - 把原本偏离线的论文调研 pipeline 工程化成可观测、可评估、可服务化的 Agent 平台原型。

2. 和普通 RAG demo 有什么区别？
   - 不只是上传文档问答，而是包含 query planning/清洗、领域消歧、工具 schema、trace、memory、multi-agent critic、scorecard 和 Dashboard。

3. Runtime 在这里有什么价值？
   - Workflow 描述业务步骤，Runtime 负责稳定执行：工具注册、参数校验、hook、trace、错误、memory 和上下文压缩。

4. 你怎么评估 Agent 好不好？
   - 用 scorecard 检查任务完成度、检索相关性、跑题风险、文章质量、trace 可观测性和 QA groundedness。

5. Multi-Agent 是不是硬凑？
   - 不是。Planner、Retriever、Critic、Memory、Evaluator 分别对应调研链路里的真实职责，并且每个角色都有可测试输入输出和 agent trace。

6. 项目边界是什么？
   - 当前是本地可演示平台原型，不是生产级多租户系统。权限、鉴权、分布式队列、企业监控和真实大规模压测属于后续工程化方向。

7. 为什么不要继续堆版本？
   - 现在项目的求职叙事已经闭环，继续加零散功能会稀释重点。后续应该围绕真实面试反馈、bug 和明确岗位需求维护。

## 3.1 v1.0 项目讲法

### 30 秒项目介绍

```text
PaperStorm Agent 是我基于 Stanford STORM 二次开发的中文论文调研和知识库 Agent。我保留了原项目 research、outline、article、polish 的长文生成流程，并在此基础上补了 arXiv/本地 PDF 检索、PIM 领域消歧、Tool Schema、MCP-style server、三层记忆、上下文压缩、知识库 QA、Multi-Agent 编排、Eval Harness、Task Service 和 Dashboard。v1.0 已经能一条命令生成本地 release demo，展示一次 Agent 任务从提交、运行、问答、trace 到 scorecard 的完整链路。
```

### 2 分钟技术介绍

```text
这个项目的核心不是简单套一个 RAG，而是把论文调研 Agent 做成可观测、可评估、可服务化的系统。底层复用 STORM 的多阶段 workflow：先多视角调研，再生成大纲、文章和润色结果。我主要做了几层工程化增强：

第一是 RAG 稳定性，接入 arXiv 和 Local PDF，并做 query sanitizer、PIM 缩写消歧、空检索防护和单 query 失败降级，避免 PIM 被误召回成 processing-in-memory、RAM、DRAM。

第二是 Agent Runtime，把工具封装成统一 PaperStormTool schema，通过 ToolRegistry 管理工具发现和参数校验，用 HookManager 和 RuntimeEvent 记录工具调用、错误、耗时和上下文压缩，输出 JSONL trace。

第三是 Memory 和 QA，把调研产物转成可问答知识库，支持 working/episodic/semantic 三层记忆、结构化上下文压缩、grounded QA、citations 和 QA eval。

第四是 Multi-Agent 和评测，把调研拆成 Planner、Retriever、Critic、Memory、Evaluator，用 agent_trace 记录每个角色决策，并用 scorecard 评估任务完成度、检索相关性、跑题率、文章质量和可观测性。

最后我抽象了 PaperStormTaskService 和 FastAPI 适配器，并做 Dashboard 展示任务状态、文章、QA、trace、scorecard 和错误信息。v1.0 release demo 可以不依赖真实 API key 复现完整链路，真实 worker 则可以接 DeepSeek/arXiv 手工运行。
```

### 5 分钟演示路线

1. 打开 README，先指官方 STORM 架构图，说明原流程是 research -> outline -> article -> polish。
2. 运行 `run_paperstorm_release_demo.py`，说明这个命令复用 service 层，不是单独伪造前端数据。
3. 打开 `release_demo_summary.json`，展示 task_id、task_status、article_path、trace_path、scorecard_path 和 QA answer。
4. 打开 `frontend/paperstorm_dashboard/index.html`，展示 Dashboard 的 task、article、QA、trace、scorecard。
5. 讲 PIM case：expected keywords 是 `passive intermodulation / RF`，forbidden keywords 是 `processing-in-memory / DRAM / RAM`，对应检索跑题治理。
6. 讲系统设计：ToolRegistry 负责工具，Hook/Trace 负责可观测，Memory/QA 负责知识沉淀，Eval 负责量化，Task Service 负责生命周期，Dashboard 负责调试展示。
7. 主动讲边界：当前 v1.0 是本地可演示原型，不是生产级多租户平台；后续要补鉴权、权限、分布式队列、监控告警和真实 API 压测。

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
当前项目处于本地原型阶段，已经做了错误降级、trace、eval、测试和 PaperStormTaskService 服务核心层。v0.5 已经支持 task_id、状态查询、文章/trace/scorecard 读取、知识库 QA、fake runner、独立 output_dir 和错误脱敏，并提供可选 FastAPI 适配器。真正生产级还需要鉴权、限流、队列、并发压测、监控告警、数据权限和真实 worker 编排。
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
当前 PaperStorm 还不是线上高并发系统，但已经完成了单进程服务层稳定性 baseline：task_id、状态隔离、独立 output_dir、trace/scorecard 隔离、max_concurrent_tasks、worker_tick、stale running 恢复和 fake runner 压测报告。下一步如果接真实 worker，会给 LLM、检索、embedding 工具加 timeout/retry/rate limit，并用 10/50/100 任务压测统计平均延迟、P95、失败率和 retry 次数。真正生产级还需要鉴权、权限、监控告警、分布式队列和成本治理。
```

### Q13.1：为什么先做 fake runner，不直接接真实 LLM 服务？

推荐回答：

```text
因为服务层首先要验证任务状态、路径隔离、trace、scorecard、错误脱敏和 QA API 这些工程语义，不能让测试依赖真实 API、网络和模型波动。fake runner 是稳定 baseline，可以支撑单元测试、前端预览和后续压测。真实 LLM pipeline 后续作为 worker runner 接入，同一套 task_id 和状态模型不用变。
```

### Q13.2：为什么要做前端 Dashboard？

推荐回答：

```text
Agent 系统的问题定位不能只看最终回答。Dashboard 的价值是把 task 状态、runtime trace、scorecard、QA 引用、Multi-Agent 决策和 stress report 展示出来，让执行链路可解释、可复盘、可沟通。对企业 Agent 平台来说，这对应可观测性和跨角色协作：算法、后端、产品都能看到 Agent 为什么这么答、工具是否失败、检索是否跑题、评估指标是否达标。
```

### Q13.3：真实 LLM pipeline 怎么接入服务层？为什么要 runner 注入？

推荐回答：

```text
我没有让 service 直接散落调用命令行脚本，而是把真实 STORM pipeline 封装成 `run_paperstorm_pipeline_task(state)`，由 `PaperStormTaskService` 在 `run_mode="paperstorm"` 时调用。service 只关心 task_id、状态、output_dir、trace、scorecard 和错误处理，真实 worker 负责把 task state 映射成 STORM runner 配置。这样 fake runner 和真实 worker 共用同一套服务语义，测试可以注入本地 runner，不依赖真实 LLM、arXiv、网络和余额；生产上也可以把这个 runner 换成队列 worker 或异步 worker。
```

### Q13.4：这次为什么修了 keywords 脱敏？

推荐回答：

```text
我在新增 service CLI smoke test 时发现 `expected_keywords` 和 `forbidden_keywords` 被错误脱敏成 `***REDACTED***`。根因是旧脱敏规则只要字段名包含 `key` 就脱敏，误伤了 `keywords`，导致 eval 约束丢失。修复后只对 `api_key`、`secret_key`、`token`、`password` 等真正敏感字段脱敏，同时新增回归测试保证领域关键词不会再丢。
```

### Q13.5：Dashboard 为什么要后端聚合 snapshot，而不是前端分别调很多接口？

推荐回答：

```text
我给 service 增加了 `get_dashboard_bundle(task_id)` 和 `/research-tasks/{task_id}/dashboard`，把 task、article、QA、scorecard、trace、pipeline worker 元数据聚合成一个前端 snapshot。这样前端只负责展示，后端负责产物结构和路径处理，减少前端对内部文件结构的依赖。对 Agent 平台来说，这是一种常见的可观测性接口设计：调试页面需要的是一次运行的完整视图，而不是让页面自己拼很多低层接口。
```

### Q13.6：为什么 v0.9 还要做提交、运行和轮询？

推荐回答：

```text
因为只展示结果还不算 Agent 平台，最多是报告查看器。v0.9 把 Dashboard 扩成一个本地 Agent 控制台：前端可以创建 task、运行 task、刷新任务列表、轮询 selected task，并展示结构化 error。这样面试时可以讲清楚 Agent runtime 的完整生命周期：submit -> queued -> running -> succeeded/failed -> artifacts -> trace/scorecard，而不是只讲一个离线脚本。
```

### Q13.7：v1.1 面试讲法：为什么还要做 Demo Runbook？

推荐回答：

```text
演示不是只给静态截图。Agent 平台面试里，面试官更关心系统是不是能复现、能定位问题、能解释状态变化。v1.1 我补了 start_paperstorm_service.py 和 README runbook，让项目可以按固定步骤启动 service、打开 Dashboard、提交任务、运行任务、轮询任务，再查看 article、QA、trace 和 scorecard。这个链路能讲清楚 submit -> queued -> running -> succeeded/failed -> artifacts -> trace/scorecard，也能说明 fake 模式用于稳定演示，paperstorm 模式用于真实 LLM 任务。
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
我会分阶段做，不会一上来声称高并发。当前已经完成第一阶段和第二阶段的单进程 baseline：task_id、状态隔离、output_dir/trace/scorecard 隔离、worker_tick 队列、max_concurrent_tasks、stale running 恢复和 fake runner 压测。下一步真实 worker 接入后，再给 LLM、检索、embedding 工具加 timeout/retry/rate limit。难点是外部 API 限流、embedding 模型复用、文件写入隔离、阻塞调用与 async 混用、失败任务状态恢复。
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

## 2026-07-29 面试与简历更新：v1.3 Research QA Agent 核心入口

### 新增项目能力

- 新增 `ResearchQAAgent`，把“论文调研任务”和“基于调研产物问答”串成统一 `ask` 入口。
- 新增 `PaperStormTaskService.ask_research_agent(...)`。
- 新增 FastAPI 路由 `POST /research-agent/ask`。
- 当用户没有提供 `task_id` 时，系统会自动创建并运行 research task，再基于产物回答。
- 当用户提供已完成 `task_id` 时，系统直接复用已有知识库回答，不重复调研。
- 回答返回 `answer`、`citations`、`evidence`、`grounded`、`used_task_id`、`retrieval_triggered`、`decision` 和 `trace`。

### 简历 bullet 更新

可加入压缩版项目描述：

```text
- 将 PaperStorm 从“先离线调研、再手动问答”升级为 Research QA Agent，新增统一 ask 入口：系统可根据 task_id 状态自动复用已有调研产物或触发 research task，再生成带 citations、evidence、decision 和 trace 的 grounded answer。
```

### 面试可能追问

1. 这个版本和普通 RAG QA 有什么区别？
2. 为什么要让 ask 入口自动创建 research task？
3. 如何避免每次提问都重新检索，造成成本浪费？
4. 当前 decision 是规则还是 LLM？
5. 这个版本离豆包那种自动检索回答还差什么？

### 推荐回答

```text
v1.3 的核心是把调研 pipeline 和知识库 QA 打通成统一 Research QA Agent。普通 RAG QA 通常假设知识库已经建好，而我的场景是用户可能直接问一个新技术问题，所以 ask 入口需要先判断是否有可用 task_id：如果没有，就自动创建 research task，跑完后再基于产物回答；如果有已完成 task_id，就直接复用已有 evidence，避免重复检索和重复消耗。当前 decision 先用规则 baseline，保证可测试、可复现；后续 v1.4 再做 evidence sufficiency 评分，v1.5 做前端聊天式问答。
```

### 不能夸大的边界

- v1.3 还不是完整豆包式联网搜索 Agent。
- 当前自动检索决策是规则版，不是 LLM planner。
- 证据充分性评分还在 v1.4 计划中。
- 前端聊天式问答还在 v1.5 计划中。
- 真实 `paperstorm` 模式仍依赖网络、API key、模型稳定性和外部检索服务。

## 2026-07-29 面试与简历更新：v1.4 Evidence Sufficiency

### 新增项目能力

- `ResearchQAAgent` 返回 `evidence_sufficiency`，包含 `score`、`sufficient`、`evidence_count`、`citation_count`、`keyword_overlap`、`topic_relevance`、`expected_keyword_hits`、`forbidden_keyword_hits`。
- 对已有 task_id 的问答先做证据充分性判断。
- 证据足够时返回 `answer_from_existing_kb`。
- 证据不足时返回 `reject_low_confidence`，避免把无关证据硬拼成答案。
- 对 PIM 消歧问题记录 forbidden keywords，例如 `DRAM`、`processing-in-memory`，但如果证据明确命中 `passive intermodulation`，仍可回答消歧问题。
- 修复中文单字 overlap 造成的误判，避免无关问题因为“和/关系/模型”等低信息量词被误判相关。

### 简历 bullet 更新

```text
- 为 Research QA Agent 增加 evidence sufficiency 评分和低置信拒答机制，基于 evidence_count、citation_count、keyword_overlap、topic_relevance、expected/forbidden keyword hits 判断已有知识是否足以回答，避免无关检索片段被拼接成 hallucinated answer。
```

### 面试可能追问

1. 为什么 RAG 系统需要 evidence sufficiency？
2. 你怎么判断已有知识是否足够回答？
3. 为什么证据不足时要拒答，而不是让 LLM 自己发挥？
4. forbidden keywords 是不是一命中就拒绝？
5. 这个评分是规则还是模型判断？

### 推荐回答

```text
RAG 不能只要召回了片段就回答，因为检索器常常会返回“看似相关但无法支撑问题”的材料。v1.4 我加了 evidence sufficiency：综合 evidence_count、citation_count、问题和证据的关键词重叠、topic relevance、expected keyword hits 和 forbidden keyword hits。对于已有 task_id，如果证据不足，系统返回 reject_low_confidence，而不是把无关证据硬拼成 grounded answer。forbidden keywords 不是简单一票否决，比如用户问“为什么 PIM 不是 processing-in-memory”时，DRAM/processing-in-memory 是消歧上下文；只要证据明确命中 passive intermodulation，就可以回答并记录 forbidden hits。
```

### 不能夸大的边界

- 当前 sufficiency 是规则版 baseline，不是训练出的 judge 模型。
- 当前证据不足时先拒答，尚未自动触发补充检索。
- 前端还没有把 sufficiency 单独可视化，后续 v1.5 聊天式问答再展示。

## 2026-07-29 面试与简历更新：v2.0 Research QA Agent 收口

### 新增项目能力

- 前端新增“文献检索问答”聊天区，支持直接调用 `/research-agent/ask`。
- 统一 evidence schema，证据和 citation 包含 `source_type`、`chunk_id`、`score`、`metadata`。
- Research QA 支持 `qa_history.json`，保留最近问答历史，支撑连续追问。
- 新增 `research_qa` 工具 schema，可被工具系统发现和调用。
- 新增 Research QA benchmark，输出 `research_qa_benchmark_report.json/md`。
- README 增加 v2.0 Research QA Agent 演示说明。

### 最终简历 bullet 更新

```text
- 将 PaperStorm 从离线论文调研流程升级为 Research QA Agent：新增 `/research-agent/ask` 统一入口、Evidence Sufficiency 评分、低置信拒答、统一 evidence schema、QA history、research_qa Tool Schema、Benchmark 和 Dashboard 聊天式问答，实现“自动调研/复用知识库 -> grounded answer -> citations/decision/trace 可观测”的闭环。
```

### 推荐回答

```text
第二阶段我把 PaperStorm 的调研结果和问答能力融合起来。用户不需要先手动跑完调研再记 task_id，而是可以直接调用 `/research-agent/ask` 或在 Dashboard 聊天区提问。系统会根据 task_id 和 evidence sufficiency 判断是否能直接回答；如果没有 task_id，会先创建 fake research task，产出文章和检索结果后再回答。回答会返回 citations、evidence、decision、sufficiency、qa_history 和 trace。这样项目从“RAG 报告生成器”进一步接近企业知识库/科研助理 Agent。
```

### 不能夸大的边界

- 当前 v2.0 仍是本地可演示原型，不是线上多租户系统。
- Evidence Sufficiency 仍是规则版 baseline。
- 真实 paperstorm 模式仍依赖网络、API key 和外部模型稳定性。
- 前端是零构建静态 Dashboard，不是生产级 React 管理后台。

## 2026-07-29 面试与简历更新：v2.1 Research Chat Agent

### 新增项目能力

- 新增 `PaperStormChatAgent`，在 Research QA Agent 外层增加聊天会话层。
- 新增 `chat_sessions/{chat_id}.json` 文件持久化，保存消息、主题、task_id、上下文摘要和记忆上下文。
- 新增 FastAPI 路由：
  - `POST /chat/sessions`
  - `GET /chat/sessions/{chat_id}`
  - `POST /chat/sessions/{chat_id}/messages`
- Dashboard 增加双模式：
  - `调研写文章`：保留 submit/run/article/trace/scorecard 流程。
  - `聊天问答`：用户直接提问，系统自动判断是否复用知识库或触发调研。
- 聊天回答返回：
  - `context_window`
  - `compressed_context`
  - `memory_context`
  - `retrieval_triggered`
  - `used_task_id`
  - `research_answer`
- 修复“上下文压缩不像聊天机器人”的问题：最近对话作为滑窗展示，被滑出窗口的信息通过上一轮压缩摘要继续进入下一轮压缩。

### 简历 bullet 更新

```text
- 在 Research QA Agent 外层设计会话式 Chat Agent，支持 chat session 持久化、上下文滑窗、结构化上下文压缩、三层 memory context 展示和自动检索触发；Dashboard 提供“调研写文章 / 聊天问答”双模式，实现从离线调研 workflow 到可交互 RAG Chat Agent 的升级。
```

### 面试可能追问

1. 这和普通聊天机器人有什么区别？
2. 为什么还要保留“调研写文章”模式？
3. 你的上下文压缩具体压缩了什么？
4. 记忆系统现在是真三层吗？
5. 用户第二轮追问时如何避免重复检索？
6. 这个版本离生产级 Agent Chat 还差什么？

### 推荐回答

```text
v2.1 我把 PaperStorm 拆成两个产品模式。调研写文章模式负责完整 research -> outline -> article -> polish，适合生成长文综述和展示 trace/scorecard；聊天问答模式负责用户直接提问，系统维护 chat session、最近上下文滑窗、compressed_context 和 memory_context。第一轮没有 task_id 时自动创建 research task 并基于证据回答，第二轮会复用上一轮 task_id，如果 evidence sufficiency 足够就不重复检索。这样它不是简单聊天壳，而是把 STORM 的深度调研能力包装成一个可追踪的 RAG Chat Agent。
```

### 不能夸大的边界

- 当前 chat session 是文件持久化，不是数据库或分布式 session store。
- `compressed_context` 是规则压缩 baseline，不是 LLM summary，也没有 token-level 预算调度。
- 三层 memory 目前可展示和参与上下文组装，但还不是长期用户画像系统。
- 自动检索触发依赖 ResearchQAAgent 的规则 decision 和 evidence sufficiency，不是复杂 LLM planner。
- 前端是本地演示 Dashboard，不是生产级 SaaS 前端。

## 2026-07-29 面试与简历更新：RAG / Memory 后续主线

### 当前真实状态

```text
已做：
- arXiv / LocalPDF 检索接入。
- LocalPDFRM 支持字符 chunk，默认 chunk_size=1200，chunk_overlap=150。
- PaperStormKnowledgeBase 能从文章段落和 raw_search_results 构造 evidence。
- QA 返回 citations、chunk_id、source_type、score、metadata。
- Evidence Sufficiency 能做低置信拒答和跑题关键词检测。
- Chat Agent 有短期上下文窗口、压缩摘要和本地 JSON 记忆。

未做成主链路：
- PaperStorm Chat 主链路还没接正式 embedding 向量库。
- 没有完成 Hybrid BM25 + Vector 检索。
- 没有完成 cross-encoder / LLM rerank。
- 没有 HNSW 参数调优。
- 没有动态 token 预算分配。
- 没有长期向量记忆。
```

### 面试推荐回答：当前 RAG 链路

```text
当前 PaperStorm 的主链路是可追踪 RAG baseline：文档来源包括 arXiv、本地 PDF 和一次 STORM 运行产物；LocalPDFRM 支持 chunk_size 和 chunk_overlap；QA 层会把生成文章段落和 raw_search_results 转成 evidence，并返回 citations、chunk_id、source_type 和 metadata。当前排序主要是 lexical / CJK overlap 和 evidence sufficiency，不会夸大成已经完成生产级向量库 RAG。下一步我计划把原项目已有的 Qdrant/VectorRM 能力接入 PaperStorm Chat 主链路，形成 Chunk -> Embedding -> Vector Store -> Hybrid Retrieval -> Rerank -> Context Compression -> Prompt 的完整链路。
```

### 面试推荐回答：什么时候用 RAG，什么时候微调

```text
知识频繁变化、需要引用溯源、面向企业私有文档或论文资料时优先 RAG；模型行为风格、固定任务格式、领域表达习惯或工具调用模式需要稳定提升时考虑 SFT/微调。PaperStorm 这种论文调研和知识库问答场景优先 RAG，因为资料变化快、必须返回 citation，而且用户会追问具体来源。微调可以后续用于 query planner、reranker 或 answer style，但不应该替代知识检索。
```

### 面试推荐回答：召回低、幻觉严重怎么排查

```text
我会先拆链路排查：query 是否被正确改写，chunk 是否切得过大或过小，embedding 是否适合中英文和领域术语，top_k 是否太小，rerank 是否把正确证据过滤掉，prompt 是否没有明确要求引用证据。指标上看 context_recall、citation_precision、forbidden_hit_rate、grounded_rate 和 hallucination_rate。PaperStorm 已经有 PIM expected/forbidden keyword 和 scorecard baseline，后续会扩展成标准 RAG benchmark。
```

### 面试推荐回答：Hybrid 检索为什么更稳

```text
纯向量检索适合语义相近表达，但对专有名词、缩写、公式、型号、接口名不一定稳；BM25/关键词检索对精确词命中稳，但处理同义表达弱。Hybrid 把两者合并，既能召回语义相关 chunk，又不丢专业关键词。PIM 这种缩写消歧场景很适合 Hybrid，因为 passive intermodulation、RF、DRAM、processing-in-memory 这些词面信号非常关键。
```

### 面试推荐回答：多轮对话 RAG 如何保存历史上下文

```text
我会分短期窗口和长期记忆。短期窗口保存当前 session 最近 N 轮对话，直接参与当前回答；超出窗口后生成 compressed_context，保留用户问题、约束、关键事实和引用。长期记忆保存跨会话偏好、稳定事实和历史任务经验，可以用 JSON baseline 起步，后续写入向量库做 semantic recall。PaperStorm v2.1 已经有短期窗口、压缩摘要和 JSON memory，v2.4 计划补长期向量记忆。
```

### 后续简历增强 bullet

```text
- 规划并推进 PaperStorm RAG 主链路升级：将当前 lexical evidence baseline 演进为 Chunk/Embedding/VectorStore/Hybrid Retrieval/Rerank/ContextCompressionRetriever，补充动态 token 预算、长期向量记忆和 RAG benchmark，指标覆盖 context_recall、citation_precision、grounded_rate、hallucination_rate、p95 latency 和 QPS。
```

## 2026-07-29 面试与简历更新：v3.0 RAG Memory Benchmark

### 新增项目能力

- 新增 `PaperStormRAGIndex`，支持从 PaperStorm run artifacts 构建 chunk、metadata、hash embedding 和本地 JSON 索引。
- 新增 RAG Benchmark 能力，用可复现指标评估检索、引用、跑题率和延迟。
- 新增 Hybrid Retrieval baseline，同时返回 `lexical_score`、`vector_score`、`hybrid_score` 和 `rerank_score`。
- 新增规则 rerank：expected keywords 加分，forbidden keywords 扣分。
- 新增 `ContextCompressionRetriever`，在 retriever 和 prompt 拼接之间做粗过滤、细压缩和上下文预算分配。
- 新增 `PaperStormLongTermMemoryIndex`，将三层记忆写入本地长期记忆索引，支持跨会话 recall。
- 新增 `run_rag_benchmark`，输出 `rag_benchmark_report.json/md`。
- `PaperStormKnowledgeBase.search()` 优先接入 v3.0 RAG index，失败时回退旧检索。

### 简历 bullet 更新

```text
- 为 PaperStorm 设计 RAG v3.0 检索与压缩链路：实现 Chunk/Metadata/Hash Embedding/Local Vector Index/Hybrid Retrieval/Rule Rerank/ContextCompressionRetriever/Long-term Memory Index，并建立 RAG benchmark，指标覆盖 context_recall、citation_precision、off_topic_rate、p95 latency 和 QPS estimate。
```

### 面试推荐回答：Hybrid 检索和 rerank

```text
v3.0 我先做了一个无外部依赖 baseline：每个 chunk 同时计算 lexical score 和 hash embedding vector score，然后按 alpha 融合成 hybrid score，最后用 expected/forbidden keywords 做规则 rerank。这个版本不是为了替代 Qdrant 或 cross-encoder，而是把检索链路、score 字段、audit 和 benchmark 接口做通。后续把 hash embedding 换成 bge-m3 / text-embedding，把本地 JSON 换成 Qdrant，把规则 rerank 换成 cross-encoder，业务接口不需要大改。
```

### 面试推荐回答：ContextCompressionRetriever

```text
我没有直接把所有召回片段塞进 prompt，而是在 retriever 和 prompt assembly 中间加了 ContextCompressionRetriever wrapper。它先召回多一些候选 chunk，再粗过滤跑题或低分 chunk，然后按历史 30%、证据 70% 做字符预算分配，最后抽取 query 命中句和 expected keyword 命中句形成 compressed evidence。这样既能解释上下文怎么被压缩，也方便后续无侵入替换为小模型总结器。
```

### 面试推荐回答：长期记忆和短期窗口

```text
短期窗口是 chat session 最近 N 轮对话，用来处理指代和连续追问；长期记忆是跨会话保存的 semantic/episodic/preferences。v3.0 的长期记忆索引是本地 JSON + hash embedding baseline，能展示 recall 链路，但我不会把它夸大成生产级用户画像或向量数据库。生产化下一步会接 Qdrant、增加记忆写入策略、过期策略和 memory eval。
```

### 不能夸大的边界

- v3.0 是本地 baseline，不是生产级向量数据库系统。
- 当前 embedding 是 hash embedding，不是语义 embedding 模型。
- 当前 ANN 仍是 linear scan，不是真 HNSW。
- 当前 rerank 是规则版，不是 cross-encoder。
- 当前 token 预算是字符近似，不是 tokenizer 精确估算。
- 当前 QPS 是本地估算值，不代表真实线上吞吐。

## 2026-07-29 面试与简历更新：v3.1 Enterprise Intent Router

### 新增项目能力

- 新增 `PaperStormIntentRouter`，把聊天模式中的意图路由从 ChatAgent 内部抽成独立 runtime 层。
- 支持 LLM JSON Router 注入：模型只负责输出结构化决策，不直接写业务逻辑。
- 支持 rule fallback：没有 API key 或本地测试时仍可稳定复现。
- Router 输出字段包括 `intent`、`need_retrieval`、`tool`、`rewritten_query`、`confidence`、`reason`、`router`。
- ChatAgent 返回新增 `router_decision` 和 `tool_decision`，Dashboard 可以直接看到每轮回答为什么检索、为什么不检索。
- 多轮追问支持 query rewrite，例如把“那它为什么不是 DRAM？”改写为包含 topic 和上一轮用户问题的独立 query。

### 简历 bullet 更新

```text
- 设计 PaperStorm v3.1 企业 Agent 路由层：将聊天/知识库问答/论文调研的判断抽象为 LLM JSON Router + rule fallback，输出 intent、tool、confidence、reason 与 rewritten_query，并接入 ChatAgent、RAG QA 和 Dashboard trace，解决多轮对话中闲聊问题被 topic 误导、追问 query 不完整、工具调用不可观测等问题。
```

### 面试推荐回答：豆包、千问或企业内部知识库如何判断是否检索

```text
不会只靠一个 if/else。更常见的是先有一个 Router 或 Planner：输入当前用户问题、最近对话、用户画像/记忆、已有知识状态和工具列表，输出结构化决策，例如 intent、是否需要检索、该调用哪个工具、是否需要 query rewrite、置信度和原因。简单问题直接走 chat fallback；需要事实依据或内部文档时走 RAG；证据不足时触发检索或反问澄清；涉及外部动作时走 tool calling。PaperStorm v3.1 就是按这个思路做的，Router 可由 LLM 产出 JSON，也有本地 fallback，决策结果会进入 trace 和前端面板。
```

### 面试推荐回答：为什么不能直接让 LLM 决定一切

```text
生产系统不能只相信模型口头判断，因为要稳定、可测、可审计。我的做法是让 LLM 只输出受 schema 约束的 JSON 决策，再由 runtime 校验 intent/tool/confidence，低置信或格式异常就 fallback。这样既保留模型对复杂语义的判断能力，又不会让业务链路被自由文本带偏。这个设计也便于单元测试、灰度、日志审计和后续替换模型。
```

### 面试推荐回答：这个版本离企业级还差什么

```text
v3.1 已经补齐了企业 Agent 的核心形态：Router、Tool Decision、RAG/Memory、Trace/UI 四层，但还不是生产系统。后续如果继续做，要补权限 ACL、租户隔离、真实向量库、真实 embedding/cross-encoder rerank、模型路由灰度、限流熔断、分布式 trace、线上指标看板和安全审计。简历上我会明确说这是可演示的工程化 Agent baseline，而不是已上线的企业平台。
```

### 不能夸大的边界

- 默认演示不调用真实 LLM Router，使用 fallback；真实 LLM Router 需要注入 provider。
- 当前 `PaperStormIntentRouter` 是路由层，不是完整 LangGraph/PydanticAI 编排框架。
- 当前 tool decision 只覆盖 PaperStorm 本项目的 chat/research/RAG 工具，不是通用企业 Tool Marketplace。
- 当前 trace 是本地 JSON/前端面板级别，不是 OpenTelemetry 或分布式 tracing。
