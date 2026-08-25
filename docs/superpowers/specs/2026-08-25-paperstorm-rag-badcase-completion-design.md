# PaperStorm RAG Bad Case 完整治理设计

## 1. 目标

在已完成 P0 检索主链统一的基础上，连续完成 P1-P4：提高第一阶段召回、建立选择性重排与证据治理、
形成答案和引用闭环，并补齐生产运行与在线治理。最终产物必须同时包含可运行代码、离线测试、分阶段
消融报告、公开 Benchmark 最终结果和面向工程复盘的中文文档。

## 2. 基线约束

- 不重新运行修改前基线，不重复消耗 Embedding、Cross-Encoder 或 LLM API。
- 冻结基线来自：
  - `docs/benchmarks/paperstorm_public_v55_summary.json`
  - `docs/benchmarks/paperstorm_memory_context_v56_summary.json`
  - `docs/benchmarks/paperstorm_v56_paid_quarter_summary.json`
- 历史 baseline 只在其原有数据 split、模型、Top K 和硬件口径内比较；字段缺失时标记为不可比，
  不补造数值。
- validation/dev 用于策略和门限选择；公开 test 只在最终冻结配置上运行一次。
- API Key 只从环境变量读取，不进入源码、命令输出、Trace、报告或 Git。

## 3. 方案选择

采用“分阶段能力开关 + 累积候选系统”的消融方式。每项能力都有独立 feature flag 和 Trace 决策，
既可单独关闭，也可按 P1、P2、P3、P4 累积启用。一次性大改后再测无法判断提升来源；只做功能不做
公开评测也无法形成可信质量结论，因此均不采用。

## 4. 系统架构

```text
Question
  -> SearchPlanner
       -> standalone rewrite
       -> entity/domain/time/source filters
       -> bounded subqueries
  -> Structured Ingestion
       -> Section / Passage / Table / Formula
       -> parent-child lineage
  -> RetrievalPipeline
       -> BM25 + Dense
       -> RRF
       -> Metadata / ACL filter
       -> RerankPolicy
       -> MMR / evidence coverage
  -> EvidenceAssessment
       -> relevance / coverage / conflict / answerability
       -> bounded corrective retrieval
  -> ContextEngine
       -> trust-separated context assembly
  -> AnswerDraft
       -> typed claims and citation spans
       -> ClaimCitationValidator
       -> local repair or abstention
  -> Control Plane / Langfuse / Benchmark Gate
```

## 5. P1：召回与文档结构

新增稳定 `SearchPlan`，字段包括原始问题、自包含问题、领域、实体、时间范围、must terms、negative
terms、metadata filters、answer type 和最多三个 subquery。默认采用确定性 planner；真实模式可调用 LLM
输出同一 JSON schema，解析失败时返回可观察错误，不静默改写。

文档摄取优先按标题、段落、列表、表格、公式和页码建立结构；超长节点再按 token 上限切分。Passage
保存 parent section ID，检索命中后按预算补回父级上下文。第一阶段仍为 BM25 + Dense + RRF，避免在
召回阶段引入昂贵生成模型。

验收：QASPER validation、SciFact dev 和 PIM 人工歧义集分别比较 raw、rewrite、subquery、
parent-child 及累积配置，输出 Recall、MRR、nDCG、evidence-set coverage、P50/P95 和 Bootstrap CI。

## 6. P2：选择性重排与证据治理

`RerankPolicy` 根据答案风险、BM25/Dense 候选重叠、RRF margin、候选规模和延迟预算决定是否启用
Cross-Encoder。MMR/coverage selector 优先保留互补证据，避免 Top K 被同一章节近重复段落占满。

`EvidenceAssessment` 输出 relevance、coverage、answerability、conflicts、confidence、failure type
和 next action。低置信度时仅允许有限次数的 rewrite、扩大候选或切换来源；达到预算后明确拒答。
冲突证据保留来源、时间、条件和支持/反对关系，不由 LLM 静默选择一方。

验收：比较 never/always/policy rerank 的质量-P95 Pareto；对无答案、跨域、冲突和多证据样本评估
abstention precision/recall、conflict detection F1 和完整 evidence coverage。

## 7. P3：答案与引用闭环

Reader 输出结构化 `AnswerDraft`：answer type、claims、每个 claim 的 citation IDs、uncertainty 和
abstain reason。引用定位到原始 evidence span，并保留论文原名、作者、页码/章节、URL 和时间。

`ClaimCitationValidator` 将 claim 判定为 entailed、partial、contradicted 或 unsupported。仅对失败
claim 做一次局部修复；修复失败则删除无支持 claim、降低措辞或拒答，禁止整篇无界重写。

验收：按 extractive、abstractive、boolean、list、comparison 分类报告 Answer F1、Evidence F1、
citation precision/recall、claim support rate、unsupported-claim rate 和 abstention 指标。

## 8. P4：生产治理

- 模型进程内常驻；Cross-Encoder 支持批量、超时、熔断和缓存。
- tenant/document/chunk ACL 在检索前过滤，禁止先召回再隐藏。
- Retrieval、Rerank、Assessment、Repair 全部记录策略原因、模型、候选数、延迟、Token 和成本。
- Langfuse 不可用时本地 JSONL 继续记录；远程观测失败不阻断业务。
- 冻结 Benchmark manifest 保存 git SHA、数据摘要、模型、参数、环境和随机种子。
- 发布门禁比较基线与候选系统；质量下降超阈值、P95 超预算或泄漏测试失败时阻断发布。
- shadow/canary 只实现可复用的离线 replay 与策略接口，不在本地项目伪造真实线上流量。

## 9. 分阶段消融协议

| 阶段 | 相对上一阶段唯一新增能力 | 主要数据 | 主要指标 |
| --- | --- | --- | --- |
| A | SearchPlan rewrite | SciFact dev、QASPER validation、PIM | Recall/MRR/nDCG、歧义错误率 |
| B | bounded subquery | QASPER validation | evidence-set coverage、P95 |
| C | structured + parent-child chunk | QASPER validation、本地 PDF | Recall、页码/章节 lineage |
| D | RerankPolicy | SciFact/QASPER validation | Recall-P95 Pareto、启用率 |
| E | MMR/coverage | QASPER validation | 完整证据覆盖、重复率 |
| F | EvidenceAssessment + corrective retrieval | 无答案/跨域/冲突集 | abstention、conflict F1、纠错成功率 |
| G | AnswerDraft + citation validation | QASPER validation | Answer/Evidence F1、citation、unsupported claim |
| H | 缓存/ACL/门禁/观测 | 离线 replay 与并发测试 | P50/P95、吞吐、泄漏、恢复 |

每阶段输出独立 manifest、metrics、predictions、failure cases 和与上一阶段的 delta。最终冻结累积配置后，
只运行一次 SciFact test、QASPER test 和 LongMemEval-S 完整评测。

## 10. 错误处理与兼容

- 所有 LLM 结构化输出均严格校验 schema；失败类型进入 Trace，并允许一次受控重试。
- 真实模型、Reranker 或数据集缺失时 fail fast；smoke 使用 hash 必须显式声明。
- 旧索引继续显式要求迁移；新结构化索引增加 schema revision 和重建命令。
- 保留 P0 稳定入口；新能力通过配置对象注入，不复制新的版本化模块。
- Memory 只协助恢复用户、会话和文档 identity；科学 claim 仍必须由 Evidence 支持。

## 11. 完成标准

1. P1-P4 功能、feature flags、Trace schema 和离线测试全部通过。
2. 每阶段至少有一个确定性消融测试与一份真实 validation 报告。
3. 最终三个公开 Benchmark 使用冻结配置运行一次，并与现有 baseline 对齐可比较字段。
4. 报告明确提升、退化、置信区间、延迟和 API 成本，不选择性隐藏失败指标。
5. README、Bad Case 路线图和实施复盘同步更新，生产代码不出现新的内部版本模块。
