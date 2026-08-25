# PaperStorm RAG 治理与代码清理设计

## 1. 背景

PaperStorm 已经具备论文调研、知识库问答、混合检索、长期记忆、上下文压缩、
运行时治理和公开 benchmark，但多个阶段性实现仍以 `v41`、`v42`、`v43`、
`v44`、`v45`、`v52`、`v54`、`v56`、`v60` 等版本号存在于模块名、类名、
测试名、Trace 和 README 中。

这些编号曾用于区分迭代阶段，现在已经产生三个问题：

1. 当前生产入口仍直接引用多个历史版本模块，无法判断哪个实现是主路径。
2. 研究问答、企业知识库和 benchmark 使用不同检索栈，指标不能直接代表网页行为。
3. toy seed、历史兼容契约和公开 benchmark 混在同一测试面，增加维护成本并容易误读结果。

本次整理采用破坏旧内部 Python 导入路径的清洁迁移。对外保留最新版网页、CLI、
主要 HTTP API、公开 benchmark 和正式数据产物，不保证历史内部模块路径继续可用。

## 2. 目标

### 2.1 产品目标

- 研究问答和企业知识库使用同一套检索、证据治理与答案生成主链路。
- bad case 可以定位到召回、融合、重排、压缩、生成、引用或拒答中的具体阶段。
- README 和 Trace 只展示稳定能力名称，产品版本号只用于发布，不进入模块名。
- 保留 SciFact、QASPER、LongMemEval 等公开、可复现的质量证据。

### 2.2 工程目标

- 当前有效模块改为无版本命名。
- 合并仍有价值的历史实现，删除不再被生产代码引用的旧模块。
- 删除只验证旧版本字符串、toy 页面或已废弃入口的测试。
- 新增依赖边界测试，阻止生产代码重新导入带内部版本号的模块。
- 清理后测试面按核心单元、服务集成、公开 benchmark 契约和前端契约组织。

## 3. 非目标

- 不重写 Stanford 官方 STORM、Co-STORM 或其官方示例。
- 不删除历史 benchmark 结果文件；结果是审计证据，可移入归档清单但不伪造或覆盖。
- 不在本次整理中更换 LLM、Embedding 或 Cross-Encoder 模型。
- 不为了目录整洁而改变公开数据集口径或重新宣称历史分数。
- 不处理工作区中用户现有的补丁文件、`.codex-temp` 或 `docs/DESIGN_SOURCES.md` 删除状态。

## 4. 目标架构

```text
API / Web / CLI
       |
       v
Conversation Runtime ---- Memory Store
       |
       +---- Research Orchestrator ---- arXiv / Local PDF / Zotero
       |
       +---- Retrieval Pipeline
                 |
                 +-- Query Plan / Rewrite
                 +-- BM25 + Dense
                 +-- RRF Fusion
                 +-- Optional Cross-Encoder
                 +-- Relevance / Coverage / Conflict Gates
                 +-- Context Engine
                 +-- Grounded Answer + Citation Validation
       |
       +---- Trace / Langfuse / Evaluation Harness
```

所有消费检索能力的入口都依赖 `RetrievalPipeline`，不再直接实例化旧
`PaperStormRAGIndex` 或复制检索参数。

## 5. 模块迁移

| 当前模块 | 目标模块 | 处理方式 |
|---|---|---|
| `paperstorm_retrieval_v41.py` | `retrieval.py` | 迁移 BM25、Dense、RRF、Cross-Encoder 和持久索引 |
| `paperstorm_retrieval_runtime.py` | `retrieval_runtime.py` | 保留运行时缓存、模型选择和检索 Trace，移除 legacy 分支 |
| `paperstorm_document_v41.py` | `document_ingestion.py` | 保留文档解析与 chunk 元数据，移除版本名 |
| `paperstorm_context_v56.py` | `context_engine.py` | 作为唯一 Context 实现 |
| `paperstorm_memory_v56.py` | `memory_store.py` | 作为唯一长期 Memory 实现 |
| `paperstorm_langgraph_v44.py` | `conversation_runtime.py` | 保留 LangGraph 会话、checkpoint 和工具编排 |
| `paperstorm_production_v45.py` | `control_plane.py` | 保留 ACL、幂等、缓存、队列、熔断和 Trace 存储 |
| `evaluation/public_benchmarks/v60_harness.py` | `evaluation/public_benchmarks/harness.py` | 保留公开评测执行器 |
| `evaluation/public_benchmarks/v60_llm.py` | `evaluation/public_benchmarks/llm_reader.py` | 保留端到端 reader/judge 适配 |

类名、schema 名和 runtime 名同步去版本化。例如：

- `LongTermMemoryServiceV56` 改为 `LongTermMemoryService`。
- `ContextEngineConfigV56` 改为 `ContextEngineConfig`。
- `PaperStormProductionRuntimeV45` 改为 `ProductionRuntime`。
- `langgraph-v4.4` 改为 `conversation-runtime`。
- `paperstorm-hybrid-index-v4.1` 改为稳定 schema 名，并单独保留整数 schema revision。

## 6. 合并与删除边界

### 6.1 合并后删除

- 将 `paperstorm_memory_v43.py` 中仍被使用的写入策略、候选模型和序列化工具迁入
  `memory_store.py`，再删除旧模块。
- 将 `paperstorm_context_v42.py` 中仍被使用的事件、token 估算和恢复逻辑迁入
  `context_engine.py`，再删除旧模块。
- 将 `paperstorm_rag.py` 中仍需要的 chunk 工具和压缩接口迁入
  `document_ingestion.py`、`retrieval.py` 和 `context_engine.py`；企业知识库切换主检索栈后
  删除旧 hash 混合检索实现。

### 6.2 删除的历史评测入口

满足“无生产引用、公开评测已有替代、测试可覆盖当前行为”三个条件后删除：

- `paperstorm_eval_v4.py` 及其 deterministic top-1 seed 入口。
- `paperstorm_ablation_v41.py` 的历史 seed ablation 入口。
- `paperstorm_real_eval_v52.py` 和 `paperstorm_eval_v54.py` 中已被公开 benchmark
  替代的本地候选评测入口。
- `paperstorm_rag_benchmark.py`、`paperstorm_multi_task_benchmark.py` 中仅比较 legacy
  hash 与当前栈的历史代码。
- `paperstorm_release.py` 和 `run_paperstorm_release_demo.py` 中固定 `v1.0` 的旧打包演示。
- 对应的 version-only、toy UI 和历史 facade 测试。

如果其中某个模块仍提供公开 benchmark 不具备的独立能力，则先迁移该能力，再删除文件。

### 6.3 必须保留

- Stanford STORM 和 Co-STORM 主体。
- SciFact、QASPER、LongMemEval 数据适配、指标计算、冻结配置和报告生成。
- 当前 Web UI、研究任务、聊天问答、企业知识库、PDF 交付和 Langfuse 可观测能力。
- ACL、租户隔离、持久化、幂等、缓存和失败恢复测试。
- 历史结果文件及其指标诚信说明。

## 7. 统一检索链

研究问答和企业知识库统一为：

```text
Query
 -> QueryPlan（改写、子查询、过滤条件、答案类型）
 -> BM25 Top-N + Dense Top-N
 -> RRF
 -> 去重与 metadata/ACL 过滤
 -> 选择性 Cross-Encoder
 -> Evidence Gate（相关性、覆盖率、冲突、可回答性）
 -> Context Engine
 -> Grounded Answer
 -> Claim/Citation Validator
```

本次清理首先统一接口与现有行为，不强行一次实现所有高级策略。未实现的增强项通过
稳定接口和 roadmap 表达，不能用占位分数冒充结果。

## 8. Bad Case 治理文档

新增正式文档 `docs/RAG_BAD_CASES_AND_ROADMAP.md`，每个问题采用统一结构：

1. 难点。
2. 真实案例和来源数据集。
3. 改进前行为与指标。
4. 根因定位。
5. 成熟工业 RAG 的处理方式。
6. PaperStorm 当前应对方案。
7. Gap。
8. 可执行改进任务。
9. 验收指标。
10. 改进后实测结果；尚未实施时明确标记“计划中”，不得假装完成。

首批覆盖以下 bad case：

- 领域缩写歧义：PIM、RAM、DRAM。
- 查询与证据表达不同导致的召回失败。
- 同一论文内相似章节导致的段落定位失败。
- 多证据问题缺少覆盖和聚合。
- 检索命中但答案遗漏关键事实。
- 引用存在但不能支持具体 claim。
- 无答案问题误答或无法自动纠错。
- Cross-Encoder 提升质量但引入明显 P95 延迟。
- 长对话中会话事实、Memory 和论文 Evidence 边界混乱。
- 文档版本或来源冲突时缺少显式呈现。

## 9. 改进前后对比模型

| 维度 | 改进前 | 目标状态 | 验收方法 |
|---|---|---|---|
| 检索入口 | 研究、企业 KB、评测存在多套栈 | 统一 `RetrievalPipeline` | 依赖边界测试与服务集成测试 |
| 模块命名 | 内部版本号代表实现身份 | 能力名 + 独立 schema revision | 禁止生产模块导入 `*_vNN` |
| toy 评测 | synthetic seed 与公开 benchmark 混放 | 公开 benchmark 为主，seed 只可作测试 fixture | benchmark catalog 契约测试 |
| Bad Case | JSONL 分散，缺少统一根因说明 | 可按 failure stage 和数据集追踪 | 文档链接真实产物与指标 |
| Cross-Encoder | 显式全候选重排 | 保留显式模式，为选择性门控提供接口 | Recall/nDCG/P95 联合验收 |
| 引用 | chunk 级引用 | 逐 claim 支持关系可校验 | Citation precision/recall |
| Trace | 暴露历史版本名 | 暴露稳定阶段名、输入输出和耗时 | Trace schema 测试 |
| 企业 KB | legacy hash/关键词链仍存在 | 与研究问答共享混合检索 | 同数据同 query 结果一致性测试 |

## 10. 错误处理与迁移策略

- 每次重命名先增加面向目标模块的失败测试，再迁移实现和调用方。
- 删除文件前使用 Git 引用审计确认生产代码、示例和测试均不再导入。
- 持久化文件使用显式 schema revision；模块改名不能导致现有索引和 Memory 数据静默损坏。
- 无法读取旧 schema 时必须给出迁移错误，不得无提示回退到 hash 或空索引。
- 模型缺失时允许 CI 使用确定性 provider，但真实服务必须在状态和 Trace 中明确显示后端。
- LLM、Embedding、Reranker 异常不得静默吞掉；返回可观测错误或执行显式降级。

## 11. 测试策略

### 11.1 核心单元测试

- BM25、Dense、RRF、Cross-Encoder 候选边界。
- Context budget、压缩、恢复和 lineage。
- Memory 写入策略、跨会话召回、时效性和 supersede。
- query plan、evidence gate、引用校验和拒答。

### 11.2 集成测试

- 研究产物和企业文档经过相同检索接口得到一致的 stage schema。
- Web/CLI/API 可以创建任务、检索、回答、查看引用和 Trace。
- 现有持久化数据可迁移或产生明确错误。
- 生产代码不再导入已删除的版本模块。

### 11.3 公开评测

- SciFact：Recall@10、MRR、nDCG、P95。
- QASPER：Evidence Recall/F1、Answer F1、Citation 指标、P95。
- LongMemEval：跨会话 Recall 与端到端 reader/judge，按问题类型拆分。
- 所有新指标必须记录数据版本、模型、配置、样本量和失败数。

## 12. 验收标准

1. 官方 STORM 测试和 PaperStorm 当前核心测试通过。
2. 生产模块、README 当前架构和 Trace 中不再出现内部阶段版本号。
3. 研究问答与企业知识库共享同一检索实现。
4. 被删除模块无运行时、示例或测试引用。
5. benchmark catalog 只展示当前公开评测和必要的 smoke profile。
6. `RAG_BAD_CASES_AND_ROADMAP.md` 中所有“已完成”结果都可追溯到真实产物。
7. 未完成的工业增强项有明确任务、数据集和验收指标，不写虚构结果。
8. 工作区原有用户改动和临时补丁未被覆盖、删除或提交。

## 13. 风险与控制

| 风险 | 控制措施 |
|---|---|
| 大规模改名导致循环依赖 | 先建立稳定接口，再逐消费者迁移 |
| 删除 facade 破坏隐藏调用方 | Git 全仓引用审计，加导入边界测试 |
| 历史索引不能加载 | schema revision 与显式迁移测试 |
| benchmark 分数变化 | 冻结数据与配置，前后分别保存报告 |
| 清理范围膨胀 | 仅处理 PaperStorm 扩展和直接依赖，不重构官方 STORM |
| 误删用户工作 | 只暂存本次明确修改的文件 |
