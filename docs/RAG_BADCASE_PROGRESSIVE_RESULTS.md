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
