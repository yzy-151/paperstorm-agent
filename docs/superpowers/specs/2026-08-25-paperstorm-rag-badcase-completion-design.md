# PaperStorm RAG Bad Case 完整治理设计

## 1. 目标

在已完成 P0 检索主链统一的基础上，连续完成 P1-P4：提高第一阶段召回、建立选择性重排与证据治理、
形成答案和引用闭环，并补齐生产运行与在线治理。最终产物必须同时包含可运行代码、离线测试、分阶段
递进对比报告、受影响 Benchmark 结果和面向工程复盘的中文文档。

## 2. 基线约束

- 不重新运行修改前基线，不重复消耗 Embedding、Cross-Encoder 或 LLM API。
- 冻结基线来自：
  - `docs/benchmarks/paperstorm_public_v55_summary.json`
  - `docs/benchmarks/paperstorm_memory_context_v56_summary.json`
  - `docs/benchmarks/paperstorm_v56_paid_quarter_summary.json`
- 历史 baseline 只在其原有数据 split、模型、Top K 和硬件口径内比较；字段缺失时标记为不可比，
  不补造数值。
- validation/dev 用于策略和门限选择；每个里程碑只运行会被该阶段改动影响的 Benchmark。
- API Key 只从环境变量读取，不进入源码、命令输出、Trace、报告或 Git。

## 3. 方案选择

采用四个累积里程碑递进开发：`P1`、`P1+P2`、`P1+P2+P3`、`P1+P2+P3+P4`。不为每个
内部能力增加回退开关，不运行单能力排列组合。每个里程碑完成后与现有冻结基线及上一里程碑比较，
仅运行其代码改动可能影响的 Benchmark，避免无意义的 API 与算力消耗。

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

验收：将完整 P1 与现有 P0 基线比较，在 QASPER validation、SciFact dev 和 PIM 人工歧义集输出
Recall、MRR、nDCG、evidence-set coverage、P50/P95 和 Bootstrap CI，不单独发布 rewrite、
subquery 或 parent-child 的开关消融。

## 6. P2：选择性重排与证据治理

`RerankPolicy` 根据答案风险、BM25/Dense 候选重叠、RRF margin、候选规模和延迟预算决定是否启用
Cross-Encoder。MMR/coverage selector 优先保留互补证据，避免 Top K 被同一章节近重复段落占满。

`EvidenceAssessment` 输出 relevance、coverage、answerability、conflicts、confidence、failure type
和 next action。低置信度时仅允许有限次数的 rewrite、扩大候选或切换来源；达到预算后明确拒答。
冲突证据保留来源、时间、条件和支持/反对关系，不由 LLM 静默选择一方。

验收：将 `P1+P2` 与 P1 比较，报告质量-P95 Pareto；对无答案、跨域、冲突和多证据样本评估
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

## 9. 递进评测协议

| 里程碑 | 累积能力 | 运行的 Benchmark | 不重复运行的 Benchmark |
| --- | --- | --- | --- |
| P1 | SearchPlan、rewrite、subquery、结构化与 parent-child chunk | SciFact Retrieval、QASPER Retrieval、PIM 歧义集 | QASPER Answer、LongMemEval-S |
| P1+P2 | P1 + RerankPolicy、MMR、EvidenceAssessment、冲突与纠错 | SciFact Retrieval、QASPER Retrieval、无答案/冲突治理集 | LongMemEval-S；QASPER Answer 留到 P3 |
| P1+P2+P3 | P2 + AnswerDraft、claim-citation validator、局部修复 | QASPER Answer/Evidence、citation 与 unsupported-claim 评测 | SciFact；LongMemEval-S |
| P1+P2+P3+P4 | P3 + 缓存、ACL、并发、观测与发布门禁 | 离线 replay、并发/P95、ACL 泄漏、恢复与发布门禁 | 质量算法未变化时不重复跑 SciFact/QASPER/LongMemEval-S |

每个里程碑输出独立 manifest、metrics、predictions、failure cases 和与上一里程碑的 delta。只有当某个
阶段实际修改 Memory 或 Context 算法时才重跑 LongMemEval-S；仅修改 RAG、Reader 或生产治理时沿用
已有 LongMemEval-S 基线并明确标记“未受影响、未重跑”。

## 10. 具体 Bad Case 验收

除聚合指标外，每个里程碑必须产出可复现的 `CaseDossier`，至少包含：

- `case_id`、来源数据集、问题和 gold evidence/answer。
- 改进前 Top K、回答、引用、Trace 和失败类型。
- 根因定位到 ingestion、rewrite、recall、fusion、rerank、coverage、assessment、context、reader、
  citation 或 runtime 中的具体阶段。
- 修改的算法、数据结构、prompt 或治理策略，以及该修改为何能针对根因生效。
- 改进后 Top K、回答、引用、Trace、指标变化和是否真正解决。
- 未解决部分、适用边界和防止回归的测试名称。

首批固定案例：

| Case | 改进前问题 | 对应里程碑 | 预期解决机制 |
| --- | --- | --- | --- |
| PIM/RAM/DRAM 缩写歧义 | “PIM 神经网络抑制”召回存内计算论文 | P1 | SearchPlan 领域解析、must/negative terms、metadata filter |
| QASPER 词汇不一致 | 问题与证据不用相同词，BM25 漏召回 | P1 | standalone rewrite、bounded subquery、BM25+Dense+RRF |
| QASPER 多证据遗漏 | Top K 被同章节近重复段落占满 | P1+P2 | parent-child、MMR 与 evidence coverage selector |
| 文献结论冲突 | 不同实验条件的相反结论被模型擅自合并 | P1+P2 | 条件化 claim、conflict group、来源与时间并列呈现 |
| 引用不支持结论 | 回答带编号，但对应片段不能推出 claim | P1+P2+P3 | claim-evidence span 校验和局部修复/拒答 |
| 无答案问题误答 | 仅主题相关就生成肯定答案 | P1+P2+P3 | answerability、abstention 和 unsupported-claim gate |
| ACL/缓存污染 | 无权限文档进入候选或跨租户缓存 | P1+P2+P3+P4 | 检索前 ACL、tenant cache namespace、泄漏测试 |

最终中文报告按“难点与真实案例 → 改进前现象 → 根因 → 方案 → 改进后证据 → 是否解决”的顺序展示，
聚合 Benchmark 用于证明整体趋势，Case Dossier 用于证明机制确实解决了具体问题。不得只选择成功案例；
至少保留一个未完全解决的案例并说明下一步。

## 11. 错误处理与兼容

- 所有 LLM 结构化输出均严格校验 schema；失败类型进入 Trace，并允许一次受控重试。
- 真实模型、Reranker 或数据集缺失时 fail fast；smoke 使用 hash 必须显式声明。
- 旧索引继续显式要求迁移；新结构化索引增加 schema revision 和重建命令。
- 保留 P0 稳定入口；新能力直接演进稳定接口，不复制新的版本化模块，也不为旧算法增加产品回退开关。
- Memory 只协助恢复用户、会话和文档 identity；科学 claim 仍必须由 Evidence 支持。

## 12. 完成标准

1. P1-P4 功能、Trace schema 和离线测试全部通过。
2. 四个累积里程碑分别有确定性回归测试；受影响的里程碑具有真实 validation 报告。
3. 每个里程碑只运行受影响 Benchmark，并与现有 baseline 或上一里程碑对齐可比较字段。
4. 报告明确提升、退化、置信区间、延迟和 API 成本，不选择性隐藏失败指标。
5. 至少完成上述七类固定 Case Dossier，并保留至少一个未完全解决的案例。
6. README、Bad Case 路线图和实施复盘同步更新，生产代码不出现新的内部版本模块。
