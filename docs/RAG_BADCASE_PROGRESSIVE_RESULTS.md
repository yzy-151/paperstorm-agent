# PaperStorm RAG Bad Case 递进结果

## P1：查询规划与结构化召回

运行目录：`C:\Users\yzy\Desktop\codex\paperstorm-benchmarks\p1\runs\2026-08-26-final`

| 数据集 | 样本 | Recall | MRR | nDCG | P95 | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| SciFact test | 300 | 0.811444 | 0.629782 | 0.668713 | 70.56 ms | 与 v5.5 数值一致，无回归 |
| QASPER test | 1309 | 0.505659 | 0.459524 | 0.415529 | 27.83 ms | 与 v5.5 数值一致；parent context 不改变排序 |
| PIM 固定集 | 4 | 见 dossier | 见 dossier | 见 dossier | 本机参考 | 领域歧义案例用于 case-level 验收 |

### 可比性边界

- 未重跑冻结基线。
- 新 manifest 增加 corpus、query+gold、case/document 数、模型、split、Top K 指纹。
- v5.5 基线缺 query+gold 指纹；所有 aggregate delta 均禁止计算。
- QASPER 当前 corpus 20221 段，与归档报告 19914 段不一致，进一步确认不可直接比较。
- 延迟为单机 CPU warm-query 参考，不代表线上 SLA。

### 具体案例

| 案例 | 改进前 | 根因 | P1 改动 | 改进后 |
| --- | --- | --- | --- | --- |
| `PIM神经网络抑制` | 召回 DRAM / processing-in-memory | 缩写跨领域歧义 | RF 领域 SearchPlan、多查询、must/negative term gate | RF 文档 Top-1；forbidden hit=0；已解决 |
| `无源互调 neural cancellation` | 中英文术语不一致，RF 论文靠后 | 词汇错配 | 中英文子查询 + RRF + 领域约束 | RF 文档 Top-1；forbidden hit=0；已解决 |
| `PIM 有哪些应用` | 可落入任意 PIM 领域 | 查询本身缺上下文 | 保留歧义，不伪造领域 | 仍未解决；应澄清用户意图 |
| QASPER `what existing databases were used?` | 无 case-level 历史预测 | query 与 gold evidence 词汇重叠为 0 | section parent context | gold 未进 Top-1；仍未解决，交给 P2 coverage/rerank |

### P1 工程修复

- SearchPlan 在创建外部调研任务前生成一次，后续检索复用。
- 多查询先廉价召回，RRF 后 Cross-Encoder 最多调用一次。
- gate 后再扩 parent；全局预算；child-first；同 parent 去重。
- typed filters 支持 `year/year_from/year_to`；未知键显式报错。
- 低证据升级由 `old → old → new` 修为 `old → new`。
- Trace 按真实执行顺序记录；blocked/failed 返回非零退出码。

## P2：选择性重排与证据治理

最终目录：`C:\Users\yzy\Desktop\codex\paperstorm-benchmarks\p2\runs\2026-08-26-final-recall-safe`

P1 与 P2 的 split、Top K、语料及 query+gold 指纹一致，允许 300 条 SciFact 与 1309 条 QASPER 做配对 Bootstrap 比较（2000 次采样）。

| 数据集 | P1 Recall / 95% CI | P2 Recall / 95% CI | Recall delta / 配对 95% CI | P2 MRR / delta | P2 nDCG / delta | P1 -> P2 P95 | Rerank 触发率 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SciFact test, K=10, n=300 | 0.8114 / [0.7693, 0.8550] | 0.8264 / [0.7834, 0.8678] | +0.0149 / [+0.0010, +0.0305] | 0.6571 / +0.0274，CI [+0.0038, +0.0508] | 0.6911 / +0.0224，CI [+0.0034, +0.0415] | 70.56 -> 1376.54 ms | 109/300，36.33% |
| QASPER test, K=5, n=1309 | 0.5057 / [0.4824, 0.5276] | 0.5526 / [0.5305, 0.5749] | +0.0469 / [+0.0336, +0.0597] | 0.5057 / +0.0462，CI [+0.0307, +0.0604] | 0.4591 / +0.0435，CI [+0.0317, +0.0554] | 27.83 -> 1029.34 ms | 598/1309，45.68% |

固定证据治理集 3/3 通过：多来源覆盖、条件冲突显式呈现、无证据拒答。延迟为本机 CPU warm-query 结果，不代表线上 SLA；P2 的主要残余问题是 Cross-Encoder 尾延迟。

### 两次失败候选与修复

| 候选 | 观察结果 | 根因 | 修复 | 最终结果 |
| --- | --- | --- | --- | --- |
| `2026-08-26-final` | SciFact 300/300 均触发 rerank，P95 2738.33 ms；QASPER 未完成，目录状态仍为 running | 单查询内部 Hybrid 已完成 RRF，Pipeline 又做一次 RRF，原始分数差被压扁，`rrf_margin` 长期落入不确定区，策略退化为 100% 重排 | 单查询保留内部 Hybrid 分数；多查询时才再次 RRF；校准 margin 门限并限制 Cross-Encoder 候选为 20 | 触发率降至 SciFact 36.33%、QASPER 45.68%；P95 分别降至 1376.54/1029.34 ms |
| `2026-08-26-final-selective` | QASPER Recall@5 0.4631，较 P1 下降 0.0425，配对 CI [-0.0618, -0.0227] | MMR 从整个 candidate pool 选择 Top K；为追求来源多样性，把原 Top K 的 gold evidence 替换出去 | 先冻结相关性 Top K，再仅在 Top K 内做 recall-safe MMR 重排，不改变成员集合 | QASPER Recall@5 升至 0.5526，较 P1 +0.0469；SciFact Recall@10 升至 0.8264，较 P1 +0.0149 |

### 具体改善与退化案例

| 数据集 / 问题 | P1 | P2 | 解释 | 状态 |
| --- | --- | --- | --- | --- |
| QASPER `what evaluation metrics did they use?` | gold 不在 Top 5；Recall/MRR/nDCG 均为 0 | gold `2002.08902::section-9::paragraph-2` 升至 Top 1；三项均为 1 | 低词面重叠触发 Cross-Encoder，语义重排恢复目标段落 | 已解决 |
| SciFact `Vitamin D deficiency effects the term of delivery.` | gold 不在 Top 10 | gold `2425364` 升至 Top 1 | 选择性语义重排修复词面表达错配 | 已解决 |
| SciFact `Activation of PPM1D suppresses p53 function.` | gold 位于第 3 | gold 退出 Top 10 | Cross-Encoder 对该 claim-document 对打分失准 | 未解决；需 P3 claim-citation/判定层兜底 |
| QASPER `How significant are the improvements over previous approaches?` | gold 位于 Top 1 | gold 退出 Top 5 | 宽泛比较型问题使语义重排偏向背景/方法段 | 未解决；需查询类型感知 rerank 与 evidence-type 约束 |

Case 级统计：SciFact Recall 改善 9、退化 1；QASPER Recall 改善 151、退化 52。最终提升是总体趋势，不等于所有查询均改善。

### P2 工程结果

- 基于风险、BM25/Dense 一致性、RRF margin、候选数与延迟预算选择性启用 Cross-Encoder。
- 单次 Cross-Encoder 只处理融合后最多 20 个候选；避免每路查询重复重排。
- recall-safe MMR 保留 Top K 成员，只优化覆盖顺序与来源多样性。
- `EvidenceAssessment` 输出 relevance、coverage、answerability、conflict、confidence、failure type 与下一动作；纠错最多一轮。
- P2 只运行受影响的 SciFact Retrieval、QASPER Retrieval 和固定证据治理集；未重复运行 QASPER Answer、LongMemEval-S 或 P0 基线。

## P3：Claim-Citation 闭环与 Grounded Answer

全量目录：`C:\Users\yzy\Desktop\codex\paperstorm-benchmarks\p3\runs\2026-08-27-full`（QASPER test 1451/1451 完成）

真实 API smoke 目录：`C:\Users\yzy\Desktop\codex\paperstorm-benchmarks\p3\runs\2026-08-27-smoke-3-retry`

| 评测 | 样本 | Answer F1 | EM | Evidence F1 | Citation P/R | Claim support | Unsupported claim | Token | 成本 |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 严格 JSON smoke | 3 | 0.7200 | 0.6667 | 0.5556 | 0.6667 / 0.5000 | 1.0000 | 0 | 10469 | $0.00238856 |
| QASPER test 全量 | 1451 | 0.5083 | 0.3039 | 0.5500 | 0.5844 / 0.5776 | 0.9592 | 0.0214 | 4616659 | $1.43615524 |

smoke 仅用于验证真实 LLM、结构化解析、引用校验和成本采集链路，不代表公开数据集总体水平。

### 用户授权的跨指纹方向性对比

归档 v5.5 与 P3 都是 QASPER test、1451 条、DeepSeek Chat、Top K=5，但 dataset loader/fingerprint、prompt 与输出合同不同。按用户授权可做方向性比较并认可新增能力，不能把差值解释为严格配对因果提升。

| 指标 | v5.5 | P3 | 方向性变化 | 结论 |
| --- | ---: | ---: | ---: | --- |
| Answer F1 | 0.5441 | 0.5083 | -0.0358 | 下降；严格短答案、拒答和 claim 约束损失部分覆盖 |
| Exact Match | 0.3274 | 0.3039 | -0.0234 | 下降 |
| Evidence F1 | 0.5814 | 0.5500 | -0.0314 | 下降；P2 冻结排名与旧运行来源不同亦会影响结果 |
| 生成失败 | 0 | 0 | 0 | 最终断点修复后无失败 |
| Claim support | 未测 | 0.9592 | 新增 | 可信回答能力可量化 |
| Unsupported claim | 未测 | 0.0214 | 新增 | 可定位无支持/矛盾 claim |
| Citation P/R | 未测 | 0.5844 / 0.5776 | 新增 | 引用正确性和覆盖率可分开评估 |
| Total Token | 1371303 | 4616659 | 约 3.37x | 每条答案增加 verifier 调用，成本显著上升；累计值包含断点前失败尝试 |

因此 P3 的“提升”成立在工程可信度、审计维度和失败治理上，不成立在传统 Answer/Evidence F1 上。下一轮应以质量-成本 Pareto 为目标，而不是继续堆叠 verifier。

### 严格输出 Bad Case

| 难点 | 改进前 | 根因 | 改进 | 改进后 | 状态 |
| --- | --- | --- | --- | --- | --- |
| 模型返回 `claim_id` 数字、空字符串 `abstain_reason` | 首次 smoke 3/3 解析失败；如果宽松强转会掩盖模型契约违约 | LLM 结构化输出仍可能偏离 schema | 保持严格 schema；把解析失败纳入最多一次、带具体错误的纠正重试；仍失败则记录明确 failure type | 同 3 条输入重跑 3/3 成功；3 条已验证 claim 均有证据支持 | 已解决链路问题；全量稳定性待最终指标 |
| 引用编号存在但来源不可信 | 模型可能自造 citation metadata | 生成结果与检索证据的信任边界不清 | 模型只返回 citation ID；标题、作者、URL、段落由可信 evidence registry 回填 | Citation precision/recall 与 claim verdict 可独立审计 | 已完成工程闭环 |
| Claim 无证据支持 | 主题相关引用可能无法推出结论 | “有引用”不等于 entailment | verifier 输出 entailed/partial/contradicted/unsupported；仅局部修复一次，仍不支持则删除、降调或拒答 | 全量 Claim support 0.9592，unsupported claim 0.0214 | 已量化但未完全解决；需分析不支持 claim 分布 |
| `abstain_reason=""` 合同违约 | 首次全量 1444/1451 成功，7 条失败；失败问题本身已有可用答案 | 主提示词没有明确 non-refusal 必须返回 `null`，纠正提示也未禁止空字符串 | 主提示和 repair prompt 同时加入 `false -> null / true -> 非空原因`；断点只重跑失败项 | 1451/1451 成功，失败 7 -> 0；F1 0.5049 -> 0.5083 | 已解决该错误类型；仍需监控新模型版本的 schema drift |

复现说明：全量报告中的 1444 条首次成功预测沿用原提示，7 条失败预测使用补充 nullability
约束后的 repair 提示断点重跑。因此该目录是同一输出 schema 下的恢复型最终报告，但不是 1451 条全部
重新生成的单一提示快照；累计 Token/成本包含失败尝试，`latest_prediction_usage` 另记录最终预测用量。

## P4：生产治理与发布门禁

正式目录：`C:\Users\yzy\Desktop\codex\paperstorm-benchmarks\p4\runs\2026-08-27-final\production-governance`

该阶段没有改变相关性算法，因此没有重跑 SciFact、QASPER Retrieval 或 LongMemEval-S。验收完全离线、不联网、不调用 LLM，覆盖 8 个生产治理合同。

| 指标 | 结果 | 验收结论 |
| --- | ---: | --- |
| ACL leak count | 0 | 通过；BM25/Dense 排序前即限定授权候选集 |
| Secret leak count | 0 | 通过；API Key、用户标识和私有正文进入 Trace 前脱敏/截断 |
| Cache isolation | true | 通过；tenant/user/policy/index/SearchPlan 共同决定缓存身份 |
| Timeout classified | true | 通过；超时返回独立 failure type |
| Circuit recovered | true | 通过；open 状态跳过 provider，cooldown 后 half-open 探针恢复 |
| Batch order preserved | true | 通过；并发执行保持输入输出对应关系 |
| Failure rate | 0 | 8/8 合同通过 |
| 本机 P95 | 27.8796 ms | 仅为离线治理夹具参考，不是线上 SLA |
| Release Gate | allowed | 指纹、质量、P95、unsupported claim、失败率和 ACL 门禁全部通过 |

### P4 具体案例

| 难点 | 改进前风险 | 根因 | 改进方案 | 改进结果 | 残余风险 |
| --- | --- | --- | --- | --- | --- |
| 私有文档可能进入候选集 | 先全库召回再过滤时，敏感 chunk 仍可能进入 rerank、Trace 或缓存 | ACL 边界位于检索之后 | 从 Control Plane 生成 fail-closed scope；BM25 在授权子语料评分，Dense 只计算授权矩阵 | public-only、private-only、deny-all 三例均无越权，leak=0 | 大规模租户需倒排 posting/向量库原生 filter，避免每次构造子集 |
| 相同问题跨用户复用答案 | 只按 query/KB 缓存会把高权限答案返回低权限用户 | 缓存 key 缺少身份与策略版本 | namespace/key 加入 tenant、user、policy digest、index revision 和 SearchPlan digest | owner/viewer 缓存身份不同，collision=0 | 权限变更必须可靠更新 policy digest |
| Provider 卡死或连续失败 | 请求长时间挂起，失败后继续打满下游 | 缺少 deadline、熔断状态和恢复探针 | deadline + explicit timeout；open 时快速降级；cooldown 后 half-open | timeout 分类正确；open 期间 provider 调用 0；探针后 closed | 线程超时不能强杀底层系统调用，线上应配合客户端级 timeout/取消 |
| 发布只看平均质量 | ACL 泄漏或 P95/失败率恶化仍可能上线 | 缺少多维自动门禁 | 冻结 manifest/predictions 离线 replay；联合质量 CI、P95、unsupported claim、failure、ACL | 正向候选 allowed；注入 1 条 ACL 泄漏和 50% P95 回归的负向候选被拒绝 | 仍需接入 CI artifact 与真实预发布 canary，离线报告不能冒充线上结果 |
