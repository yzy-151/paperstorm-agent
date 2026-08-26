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
