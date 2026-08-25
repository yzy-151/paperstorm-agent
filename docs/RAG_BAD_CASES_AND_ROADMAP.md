# PaperStorm RAG Bad Case、工业方案与改进路线

## 1. 结论

PaperStorm 已完成 BM25、Dense、RRF、可选 Cross-Encoder、公开检索评测和基础引用
问答，但当前仍属于“可审计的工程原型”，还不是成熟工业 RAG。现阶段最关键的问题不是
继续更换模型，而是把研究问答、企业知识库和 benchmark 统一到同一条检索与证据治理
链路，并补齐查询规划、结构化文档解析、证据覆盖、冲突检测、claim 级引用和纠错检索。

当前公开结果说明：

- SciFact test：Hybrid Recall@10 为 `0.811444`；Hybrid+Rerank 为 `0.837889`，
  但 P95 从 `67.7748 ms` 上升到 `2733.4805 ms`。
- QASPER test：Hybrid Evidence Recall@5 为 `0.505659`；Hybrid+Rerank 为
  `0.618648`，但 P95 从 `15.3228 ms` 上升到 `1316.6630 ms`。
- QASPER 端到端 test：Answer F1 为 `0.544147`，Evidence F1 为 `0.581404`；
  abstractive 问题 F1 仅 `0.265068`。
- LongMemEval-S：Recent Recall@5 为 `0.1358`，真实持久化 Memory Recall@5 为
  `0.800333`，但这是 retrieval-only 结果，不能视为端到端回答准确率。
- QASPER Context：压缩前后 Gold Evidence Recall 都是 `0.618648`，说明当前
  Context 没有进一步丢失已召回证据；它也没有解决检索阶段没有召回的 `38%` 左右证据缺口。

内部 v4 seed 中记录了 `51` 个 retrieval miss 和 `10` 个 generation miss，但该数据是
synthetic seed，生成器还是 deterministic top-1 chunk，只适合定位阶段性故障，不能作为
最新版真实 LLM 的质量结论。

## 2. 证据等级

| 等级 | 可用于什么结论 | 当前产物 |
|---|---|---|
| A：公开官方 | 对外展示与简历指标 | SciFact、QASPER、LongMemEval-S |
| B：公开诊断 | 分析 Context、Memory 或检索阶段 | QASPER Context、LongMemEval retrieval-only |
| C：本地真实 | 本机论文和 Zotero 回归 | 本地 PDF 与人工审核集 |
| D：合成 seed | 单元测试、失败阶段定位 | v4 PIM seed |

任何“改进结果”必须标注证据等级、样本量、数据 split、模型、Top K 和延迟口径。

## 3. 成熟工业 RAG 的共同结构

成熟 RAG 通常采用分阶段、可观测的检索链：

```text
文档解析与权限
 -> 结构化 Chunk + Metadata + Embedding
 -> Query Rewrite / Decomposition / Filters
 -> BM25 + Dense 并行召回
 -> RRF 或校准后的加权融合
 -> Metadata / ACL Filter + Dedup
 -> 候选级 Semantic Rerank
 -> Relevance / Coverage / Conflict / Answerability Gate
 -> Context Selection / Compression
 -> Grounded Generation
 -> Claim-Citation Validation
 -> 低置信度时改写、扩召回、外部搜索或拒答
```

工业依据：

- Microsoft 的 RAG 架构将 chunk、metadata enrichment、embedding、索引、查询编排和
  分阶段评测作为独立环节，而不是单次向量搜索：
  [Azure RAG 设计与评估指南](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-solution-design-and-evaluation-guide)。
- Azure 和 Elastic 都把 BM25 与向量检索的 Hybrid+RRF 作为通用生产起点：
  [Azure Hybrid Search](https://learn.microsoft.com/en-us/azure/search/hybrid-search-overview)、
  [Elastic Hybrid Search](https://www.elastic.co/docs/solutions/search/hybrid-search)。
- Elastic 将 Semantic Rerank 定义为第一阶段候选之上的昂贵二阶段操作：
  [Elastic Ranking and Reranking](https://www.elastic.co/docs/solutions/search/ranking)。
- Azure Semantic Ranker 只重排已有候选，不能补回第一阶段完全漏掉的文档：
  [Azure Semantic Ranking](https://learn.microsoft.com/en-us/azure/search/semantic-search-overview)。
- CRAG 使用检索质量评估器触发正确、错误、模糊等不同纠错动作：
  [Corrective RAG](https://arxiv.org/abs/2401.15884)。
- Self-RAG 的核心是按需检索并反思证据与生成，而不是每个请求固定检索相同数量：
  [Self-RAG](https://arxiv.org/abs/2310.11511)。
- RAPTOR 用层次聚类与递归摘要处理跨段落和长文档整体理解：
  [RAPTOR](https://arxiv.org/abs/2401.18059)。

## 4. Bad Case 1：领域缩写歧义

### 难点与真实案例

`PIM` 在射频领域表示 Passive Intermodulation，在计算机体系结构中又可能表示
Processing-in-Memory。用户输入“PIM 神经网络抑制”时，普通语义召回会同时匹配
RAM、DRAM、存内计算和射频非线性论文。

### 当前应对

- ArxivRM 增加领域查询扩展、期望关键词和排除关键词。
- Evidence sufficiency 同时检查问题、topic、证据和 forbidden keywords。
- 真实问答不再只根据旧 topic 判断是否复用证据。

### 工业方案

- Query Planner 先抽取实体、领域、时间和任务约束，再生成检索查询。
- 在 BM25 和 Dense 之前应用领域、来源、文档类型和 ACL metadata filter。
- 缩写无法确定时先澄清；可从会话实体和用户选定知识库继承领域，但不能继承陈旧 topic。
- Reranker 输入标题、摘要、章节和领域 metadata，而不是只输入正文片段。

### Gap 与计划

- 当前排除词主要由调用方提供，缺少统一术语词典和可解释的 SearchPlan。
- 企业知识库的 legacy 检索链尚未共享同一套领域过滤。
- 建立 `SearchPlan(domain, entities, must_terms, negative_terms, filters, subqueries)`。
- 以 PIM/RAM/DRAM、ACP、A2A 等高歧义词建立固定回归集，分别测试有上下文、无上下文和
  topic 过期三种情况。

### 改进前后

| 项目 | 改进前 | 当前结果 | 下一验收目标 |
|---|---|---|---|
| PIM 歧义 | 可召回 RAM/DRAM 论文 | 有领域词和 forbidden guard；仅内部诊断 | 公开或人工审核集上错误领域 Recall@5 为 0 |
| 可解释性 | 只看到最终结果 | 可看到 query、keywords 和 sufficiency | Trace 输出完整 SearchPlan 与每个 filter 原因 |

## 5. Bad Case 2：查询表达与证据词汇不一致

### 难点与真实案例

QASPER 中“which datasets did they experiment with?”、“what language pairs are explored?”
等问题，答案所在段落不一定重复问题中的词。只用 BM25 会漏掉同义表达，只用 Dense 又容易
把概念相近但不包含答案的段落放在前面。

### 当前应对与结果

当前使用 BM25 + Dense + RRF。QASPER Recall@5 从 BM25 的 `0.427937` 提升到
Hybrid 的 `0.505659`；SciFact Recall@10 从 BM25 的 `0.759167` 提升到 Hybrid 的
`0.811444`。这证明混合召回有效，但仍有明显未召回证据。

### 工业方案

- 将多轮问题改写为自包含问题，解析代词和省略对象。
- 为缩写、别名、实体和学术术语生成有限查询扩展。
- 对复杂问题生成多个子查询，各自召回后再融合。
- Hybrid 第一阶段扩大候选窗口，RRF 负责融合不同分数空间。
- 用字段级 BM25 对标题、章节、正文、作者和关键词设置不同权重。

Microsoft 的高级 RAG 指南明确包含 query rewrite、acronym expansion、step-back、HyDE
和 subqueries：[Advanced RAG](https://learn.microsoft.com/en-us/azure/developer/ai/advanced-retrieval-augmented-generation)。

### Gap 与计划

- 当前 rewrite 在 Arxiv 调研、聊天路由和企业 KB 中实现不一致。
- BM25 主要搜索拼接文本，没有字段级权重。
- 统一 Query Planner；为 research、existing evidence 和 enterprise KB 输出同一 schema。
- 运行 QASPER 按 question type 的消融：raw query、rewrite、subquery、rewrite+subquery。

### 改进前后

| 配置 | QASPER Recall@5 | P95 | 结论 |
|---|---:|---:|---|
| BM25 | 0.427937 | 0.3391 ms | 精确但语义泛化不足 |
| Dense | 0.477107 | 14.5165 ms | 语义更好但仍会错段 |
| Hybrid | 0.505659 | 15.3228 ms | 当前默认的合理底座 |
| 目标 | 待实测 | 预算不高于当前 P95 的 2 倍 | rewrite/subquery 必须用公开 test 之外的 dev 调参 |

## 6. Bad Case 3：同一论文内段落定位和多证据覆盖

### 难点与真实案例

QASPER 有 `55.5%` 的问题需要多个段落证据，`13%` 涉及表格或图：
[QASPER 论文](https://aclanthology.org/2021.naacl-main.365)。当前 paragraph Top-5 容易选到
同一章节的相似段落，却漏掉互补证据；固定字符 chunk 还可能切断公式、表格和章节语义。

### 当前应对

- 本地文档默认固定字符 chunk 和 overlap。
- Context 对已召回证据保留率达到 `0.999847`，压缩前后 Gold Recall 都是
  `0.618648`。
- 该结果只证明 Context 没丢证据，不能证明第一阶段找到完整证据集。

### 工业方案

- 依据标题、章节、段落、列表、表格和公式做 layout-aware chunk。
- chunk 保存 `document -> section -> passage` 父子关系；命中 passage 后补充 parent context。
- 用 MMR、覆盖率优化或子问题覆盖选择互补证据，避免 Top-K 被近重复段落占满。
- 对全局总结和多跳问题使用层次检索；RAPTOR 是可研究方案，不应未经 benchmark 直接全量引入。
- PDF 解析必须保留页码、章节、公式和表格来源。

Microsoft 将文档结构和 layout analysis 作为 chunk 策略选择依据：
[Azure Chunking Phase](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-chunking-phase)。

### Gap 与计划

- 当前 `500/100` 是字符窗口，不是 token 或结构边界。
- 本地 PDF 对表格、公式、图注支持不足。
- 增加结构化 `Document/Section/Passage/Table/Formula` schema。
- 比较 fixed、section-aware、parent-child 三种索引；指标除 Recall 外增加完整 evidence-set coverage。

### 改进前后

| 项目 | 当前 | 目标 |
|---|---|---|
| Chunk | 固定字符 + overlap | 结构优先，超长章节再按 token 切分 |
| 多证据 | Top-K 排名截断 | 子问题覆盖 + MMR/coverage selector |
| Context | 已召回证据几乎不丢 | 保持 retention，同时提高完整证据集覆盖率 |

## 7. Bad Case 4：Cross-Encoder 提升质量但尾延迟过高

### 难点与真实结果

Cross-Encoder 需要对 query-document pair 联合前向推理，不是简单倒序排序。

| 数据集 | Hybrid Recall | Rerank Recall | Recall 增量 | Hybrid P95 | Rerank P95 |
|---|---:|---:|---:|---:|---:|
| SciFact | 0.811444 | 0.837889 | +0.026445 | 67.7748 ms | 2733.4805 ms |
| QASPER | 0.505659 | 0.618648 | +0.112989 | 15.3228 ms | 1316.6630 ms |

### 当前应对

- 默认 `hybrid`，显式 `hybrid_rerank` 才启用 Cross-Encoder。
- 默认候选窗口是 `max(top_k * 4, 20)`。
- 当前没有按分差、查询难度或证据覆盖自动门控。

### 工业方案

- 第一阶段高召回，第二阶段只重排有限窗口。
- 模型常驻、批量推理、GPU/ONNX、结果缓存和请求合并。
- 根据答案风险、BM25/Dense 一致性、RRF 分差和 answerability 选择性启用。
- 同时优化质量和 P50/P95/吞吐，不以 Recall 单指标决定上线。

AWS 和 Elastic 都将 reranker 置于候选结果之后：
[Amazon Bedrock Rerank](https://docs.aws.amazon.com/bedrock/latest/userguide/rerank.html)、
[Elastic Semantic Reranking](https://www.elastic.co/docs/solutions/search/ranking/semantic-reranking)。

### Gap 与计划

- 增加 `RerankPolicy`，输入 query features、候选重叠、RRF margin、是否要求引用和延迟预算。
- 输出 `enabled/reason/candidate_count/model/latency` 到 Trace。
- 在 QASPER validation 调门限，test 只运行一次最终冻结配置。
- 验收采用 Recall/nDCG/P95 Pareto；不能只追求最高 Recall。

## 8. Bad Case 5：检索命中，但生成遗漏关键事实

### 难点与真实案例

内部 seed 有 10 个 generation miss；QASPER test Answer F1 为 `0.544147`，其中
abstractive F1 只有 `0.265068`。这说明“Evidence Recall 较高”不等于“答案正确”。

### 当前应对

- 问答 prompt 要求基于证据并保留 `[1]`、`[2]` 引用。
- 输出 citation metadata，包含 title、authors、URL 和 chunk ID。
- Evidence 不足时允许拒答。

### 工业方案

- 先生成结构化 claim，再为每个 claim 绑定 supporting evidence。
- 区分 extractive、boolean、abstractive、comparison 和 list 问题的回答合同。
- 生成后执行 entailment/LLM verifier：检查 unsupported claim、遗漏问题部分和引用错位。
- 校验失败时只重写失败 claim，不必重新生成整篇答案。

### Gap 与计划

- 当前 `grounded=True` 主要表示存在 evidence，不代表每个 claim 被支持。
- prompt 中每条 evidence 的正文长度有限，可能截断真正支持句。
- 建立 `AnswerDraft {claims[], citations[], abstain, uncertainty}` schema。
- 增加 claim-citation verifier 和一次受控修复循环。
- 用 QASPER Answer F1、Evidence F1、citation precision/recall 和 unsupported-claim rate 联合评估。

## 9. Bad Case 6：引用存在但不能支持结论

### 难点

“有引用”只说明输出了编号；引用可能只是主题相关，不能推出对应结论。当前内部 seed 的
citation error 分类能力存在，但真实问答链还没有逐 claim 对齐。

### 当前应对

- 引用记录原始标题、作者、发布时间、URL、段落和文章 anchor。
- UI 支持显示来源并跳到文章位置。

### 工业方案与计划

- 引用验证粒度改为 `claim -> evidence span`。
- 对 extractive 事实保存原文 span；对综合结论保存多个支持 span。
- verifier 输出 `entailed / partial / contradicted / unsupported`。
- PDF 和 UI 显示论文原名、作者、页码/章节和原文证据。
- 验收：citation precision、citation recall、claim support rate，并人工复核固定样本。

## 10. Bad Case 7：无答案、低质量检索和自动纠错

### 难点

当前 relevance gate 使用 lexical overlap 或 dense threshold，evidence sufficiency 使用规则分数。
这些规则能阻止部分跑题，但不能判断“语义相关却没有回答问题”，也不能稳定决定下一步动作。

### 当前应对

- Evidence 不足时拒答或启动新的调研任务。
- 聊天 planner 可以选择 `evidence.search` 或 `research.start`。
- 真实问答有基础 sufficiency score。

### 工业方案

- 检索评估器给出 `correct / ambiguous / incorrect` 或连续置信度。
- 低质量时依次执行 query rewrite、subquery、扩大候选、切换数据源和外部检索。
- 纠错仍失败时明确拒答，并返回缺失的证据类型。
- Self-RAG/CRAG 提供方法参考，但项目应复用其控制思想，不直接复制训练代码。

### Gap 与计划

- 当前阈值来自规则，没有使用标注集校准。
- fallback 动作与失败类型没有一一对应。
- 建立 `EvidenceAssessment` schema 和纠错状态机；限制最大检索轮数和成本。
- 在 QASPER none、内部无答案集和对抗性跨域问题上评估 abstention precision/recall。

## 11. Bad Case 8：证据冲突、版本与来源权威性

### 难点

不同论文可能因实验条件、数据集、年份或定义不同而给出冲突结论。简单按相似度排序会把
冲突证据当成重复事实，LLM 可能擅自选择其中一方。

### 当前应对

- 保存 title、authors、published、URL 等来源信息。
- 当前没有正式 conflict detector、source authority 或 temporal validity 策略。

### 工业方案与计划

- 抽取结构化 claim、实验条件、时间和来源类型。
- 对同一 claim 执行 entailment/contradiction 聚类。
- 以官方标准、同行评审论文、预印本、生成文章等来源等级控制答案措辞，而不是直接删除低等级来源。
- 冲突无法消解时并列呈现双方结论、条件和引用，明确“不足以得出统一结论”。
- 对时间敏感事实优先使用在查询时间有效的版本，保留历史 lineage。
- 新增 conflict benchmark：同结论、条件差异、时间更新、直接矛盾四类人工样本。

## 12. Bad Case 9：Memory、Conversation Context 与 Evidence 边界

### 难点

用户说“之前聊过的 PIM 论文”时：用户偏好和会话事实应来自 Memory，论文事实必须来自
Evidence。若把旧调研摘要当作权威证据，会出现陈旧信息和错误引用。

### 当前应对与结果

- Memory 保存偏好、稳定事实和会话事件；Evidence 保存带来源的论文片段。
- LongMemEval-S real persisted Memory Recall@5 为 `0.800333`，明显高于 Recent 的
  `0.1358`，但该结果没有验证最终回答是否正确。
- Context 具备预算、压缩、恢复和 lineage。

### 工业方案与计划

- Memory 只影响检索意图、个性化和上下文恢复，不作为科学 claim 的最终证据。
- 用户说“之前那篇”时，Memory 返回 document identity，再由 Evidence 层重新检索原文。
- prompt 分区明确标记 system、conversation、memory、evidence 和 tool result 的信任等级。
- 验收跨会话引用恢复、过期事实更新和错误 Memory 不污染 Evidence 三类场景。

## 13. 当前能力与工业方案 Gap

| 能力 | 当前实现 | 工业目标 | Gap 等级 |
|---|---|---|---|
| 文档解析 | PDF 文本与固定窗口 chunk | layout、section、table、formula、page lineage | 高 |
| Query Planner | 路由和局部 rewrite | 自包含改写、subquery、filters、answer type | 高 |
| Hybrid | BM25 + Dense + RRF | 统一到所有入口，支持字段权重和 metadata | 中 |
| Rerank | 显式 Cross-Encoder | 选择性策略、批量、缓存、Pareto 门限 | 高 |
| Evidence Gate | lexical/dense + 规则分数 | relevance、coverage、conflict、answerability | 高 |
| Context | 分层预算、压缩、恢复 | 与结构化证据、claim 和 token budget 联动 | 中 |
| Generation | 基础 grounded prompt | answer contract、claim-citation validator、修复循环 | 高 |
| Conflict | 仅保存来源 metadata | 条件化冲突检测、来源权威、时间有效性 | 高 |
| ACL | KB 级控制面 | 检索前 chunk/document 级权限过滤 | 中 |
| Evaluation | 公开 retrieval/answer/memory/context | failure taxonomy、在线反馈、回归门禁 | 中 |
| Observability | Trace、SSE、Langfuse | 每阶段候选、门限、原因、成本与版本 | 中 |

## 14. 分阶段改进计划

### P0：统一主链与代码清理

- 研究问答、企业知识库和 benchmark 统一使用 `RetrievalPipeline`。
- 去除内部版本模块名，删除 legacy hash/关键词生产链和 toy benchmark。
- Trace 固定输出 query、候选、融合、重排、过滤、Context 和引用阶段。
- 验收：相同 corpus/query 在不同入口得到相同 Top-K 和 stage schema。

### P1：提高第一阶段召回

- 实现 `SearchPlan`、query rewrite、subquery 和 metadata filter。
- 实现结构化 chunk 和 parent-child retrieval。
- 在 QASPER validation、SciFact dev 和 PIM 人工集做消融。
- 验收：Recall/nDCG 提升必须有 bootstrap CI，P95 增量可解释。

### P2：选择性重排与证据治理

- 实现 `RerankPolicy`、MMR/coverage selector 和 EvidenceAssessment。
- 低置信度触发有限纠错循环。
- 实现 source/temporal/conflict schema。
- 验收：质量-P95 Pareto 优于“所有请求都 Cross-Encoder”。

### P3：答案与引用闭环

- 结构化 AnswerDraft、claim-citation validator 和局部修复。
- 按 extractive、abstractive、boolean、list、comparison 分类型评测。
- 加入无答案、冲突和跨会话引用测试。
- 验收：Answer F1、Evidence F1、citation precision/recall、unsupported claim 和 abstention 指标。

### P4：生产运行与在线治理

- 模型常驻、批量、缓存、超时、熔断、并发和索引增量更新。
- 检索前执行 tenant/document/chunk ACL。
- Langfuse 记录 retrieval policy、模型、token、P95、失败阶段和用户反馈。
- 建立离线冻结集、发布回归门禁和线上 shadow/canary 对比。

## 15. 改进记录模板

每次完成一项 RAG 改进，按以下格式维护：

```text
难点：真实用户或公开 benchmark 中遇到了什么问题。
真实案例：case_id、query、gold evidence、当前输出。
根因：召回、融合、重排、压缩、生成、引用或运行时哪个阶段导致。
改进方案：修改了哪些算法、参数、接口和 Trace。
改进前：冻结数据、模型、Top K、指标、P50/P95、失败数。
改进后：相同协议下的指标和置信区间。
结果判断：解决、部分解决或未解决。
残余风险：哪些数据、语言、文档类型或线上条件仍未覆盖。
```

## 16. 面试表述边界

可以陈述：

> 我通过公开 SciFact、QASPER 和 LongMemEval 将 RAG 拆成召回、重排、Context、
> 生成、引用和 Memory 阶段评测。Hybrid+RRF 改善了第一阶段召回，Cross-Encoder
> 进一步提升 QASPER Evidence Recall，但引入明显 P95 延迟，因此后续采用选择性重排、
> 结构化 chunk、证据覆盖和 claim 级引用，按质量与延迟 Pareto 决定上线策略。

不能陈述：

- 当前已达到行业 SOTA。
- LongMemEval `0.800333` 是端到端回答准确率。
- QASPER Answer F1 能单独证明无幻觉。
- synthetic seed 的高分代表真实论文问答质量。
- 尚未实现的 CRAG、Self-RAG、RAPTOR 已经取得结果。
