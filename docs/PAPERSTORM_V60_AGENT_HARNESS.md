# PaperStorm v6.0 Agent Harness 发布说明

## 1. 发布目标

v6.0 解决三个直接影响真实使用的问题：路由被有限“内容类型”约束、生成达到固定
输出上限后静默截断、Memory 的 lexical / semantic 边界与评测证据不够清晰。同时，
执行图从阶段展示升级为节点级遥测界面，并提供两组可断点续跑的公开数据集实验。

## 2. 原因、影响、改动与验收

| 原因 | 导致什么 | 如何改 | 做到了什么 |
| --- | --- | --- | --- |
| 路由 schema 把 chat、story、QA 等内容形式当成 intent | 路由空间随产品功能膨胀；续写被误判后可能进入旧主题检索 | Planner 只决定 `respond / tool_call / clarify`，工具限定为 Memory、Evidence、Research；文体和长度进入 response contract | 输出内容开放，副作用动作有限且可审计；保留旧字段仅用于兼容 |
| LLM JSON 无效或 Provider 失败后使用普通文本兜底 | 故障会伪装成“你好/自我介绍”，难以定位 | Planner 与 Chat 统一生成 typed error：`invalid_response / timeout / rate_limit / authentication / provider_unavailable / provider_error` | 用户得到明确故障提示；Trace 保存错误类型，服务不丢失会话 |
| 所有请求共用固定 `max_tokens` | 短答浪费预算，长文与续写容易在半句处停止 | 根据任务复杂度与显式长度计算 2K、4K、8K、16K 到最高 64K；检测 `finish_reason=length` 后续接一次 | 请求成本与任务匹配，长内容完成率提高；仍保留硬上限避免失控 |
| Memory 构造器默认存在 Hash embedding | 离线确定性向量可能被误读成真实语义模型 | 默认 `lexical` 不计算向量；`semantic` 强制真实 SentenceTransformer；Hash 在显式 semantic 模式直接报错 | UI 可选择 FTS/BM25 或真实语义召回，运行报告暴露 backend |
| 调研页右侧配置与节点图抢占空间 | 图难以阅读，节点只能看到状态词 | 调研模式隐藏右栏；节点累计输入、活动、输出、耗时、Token、费用、finish reason 和 error | 完成节点显示耗时，活动节点使用流动边框与呼吸动画 |
| 长上下文和 Memory 只有局部指标 | 不能回答“质量提升是否值得额外延迟和费用” | 加入 Context profile Pareto 与 LongMemEval-S Reader/Judge 三模式同条件实验 | 具备质量、Token、TTFT、延迟、成本与 Recall 的统一证据链 |

## 3. 动态输出预算

动态预算的价值不是“把上限一律调大”，而是让资源随任务变化：

| 请求 | 初始输出预算 |
| --- | ---: |
| 极短问答 | 2,048 tokens |
| 普通对话 | 4,096 tokens |
| 详细知识回答 / 带证据回答 | 8,192 tokens |
| 创作、续写、长篇输出 | 16,384 tokens |
| 用户明确指定较长字数 | 按长度估计，最高 65,536 tokens |

如果 Provider 返回 `finish_reason=length`，Runtime 会携带原结果请求一次连续输出，并明确
要求“不重复开头、不自我介绍”。最终遥测记录分段数、聚合 token、费用、延迟和是否仍
截断。这样兼顾短请求 TTFT/费用与长请求完整性，也比固定 64K 更容易做容量治理。

## 4. Dense 的真实边界

- `Memory lexical`：SQLite FTS/BM25，不加载 embedding，适合关键词明确、跨会话历史搜索和低延迟场景。
- `Memory semantic`：真实 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`，参与 dense、RRF 与 MMR，适合同义表达和语义改写。
- `HashEmbeddingProvider`：只用于 CI 与算法契约测试；v6.0 禁止把它配置成 semantic 后端。
- Evidence RAG：继续使用真实 Dense + BM25 Hybrid、RRF 和可选 Cross-Encoder，因为论文段落的同义表达与术语改写更强。

## 5. Context Pareto 实验

该实验在 LongBench v2 相同样本、相同模型和相同提示词下运行 128K、256K、512K
三个 profile，逐条 checkpoint：

```powershell
python examples/storm_examples/run_context_profile_pareto.py `
  --dataset C:\path\to\longbench_v2_data.json `
  --output-dir C:\path\to\runs\context-pareto-v60
```

输出包含 Accuracy、实际输入 Token、TTFT P50/P95、总延迟、费用和非支配 Pareto
profile。Profile 是预算，不代表 Provider 一定支持该窗口；正式比较必须固定模型、区域、
数据和 prompt，并使用支持对应上下文的模型。

## 6. LongMemEval-S 端到端实验

三种模式共享同一 Reader 与 Judge：

1. `recent`：仅使用最近 K 个 session。
2. `fts_session`：SQLite FTS5/BM25 在完整 session 上检索。
3. `v56_memory`：真实 SentenceTransformer + v5.6 temporal memory 排序。

```powershell
python examples/storm_examples/run_longmemeval_e2e_v60.py `
  --dataset C:\path\to\longmemeval_s_cleaned.json `
  --output-dir C:\path\to\runs\longmemeval-e2e-v60 `
  --model-cache C:\path\to\models
```

每个 `(case_id, mode)` 完成后立即写入 `predictions.jsonl`，中断后可续跑。正式官方兼容
口径要求官方 500 题文件、按题型区分的官方兼容 yes/no prompt，以及固定的
`gpt-4o-2024-08-06` Judge；改用 DeepSeek 等 Judge
仍可做消融，但不得称为官方榜单成绩。

## 7. 验收状态

- Action Planner、动态输出、typed errors：单元测试通过。
- lexical / semantic Memory 边界：单元测试通过，semantic 使用真实模型名称与 backend。
- 节点遥测与 UI 状态动画：前端契约测试通过。
- Context Pareto 与 LongMemEval-S E2E：Fake Reader/Judge 离线闭环、checkpoint、聚合测试通过。
- 完整付费公开数据集分数：尚未执行，不在本发布中宣称成绩。

公开评测协议参考 [LongMemEval 官方仓库](https://github.com/xiaowu0162/LongMemEval)
与 [LongBench v2](https://longbench2.github.io/)。
