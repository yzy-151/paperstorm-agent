# PaperStorm v5.9：Memory、Context 与 Agent Planner 改进记录

## 1. 目标与边界

v5.9 解决的不是“把窗口数字调大”，而是让对话、记忆、论文证据和任务状态各自有清晰
的数据边界，并能被统一的 Turn Planner 正确使用。

| 数据层 | 保存什么 | 不保存什么 | 检索方式 | 典型问题 |
| --- | --- | --- | --- | --- |
| Active Context | 当前会话最近消息、当前回合工具状态 | 跨会话全文、整篇论文 | 时间顺序 + token 预算 | “接着刚才说” |
| Session Recall | 同一用户的完整历史消息与邻近上下文 | 稳定用户画像、论文全文 | SQLite FTS5 BM25；中文 n-gram 兜底 | “之前聊过哪篇 PIM 论文” |
| Long-term Memory | 稳定偏好、用户事实、明确决定、可复用流程 | 一次性闲聊、外部论文事实、模型推测 | BM25 + Dense + entity/time + RRF/MMR | “以后都用中文回答” |
| Evidence | 论文/PDF 的 chunk、来源、页码、引用关系 | 用户偏好、聊天原文 | BM25 + Dense + RRF；可选 Cross-Encoder | “论文如何证明该结论” |
| Recursive Summary | 被压缩历史的结构化状态 | 大段工具输出、无来源推断 | 当前 query 的 BM25 风格相关性 | “此前决定和未完成事项是什么” |

### “之前聊过的 PIM 论文”应查哪里

1. Session Recall 查旧会话，找出讨论过的消息和 citation/source 指针。
2. 若只需复述讨论结论，可在清楚标注“会话回忆”的前提下回答。
3. 若用户要求论文事实、原文或可靠引用，继续查询 Evidence。
4. 只有“用户偏好某类 PIM 论文”这类稳定事实才进入 Long-term Memory。

边界原则是：Memory 回答“关于用户和长期决定的事实”，Evidence 回答“外部世界的
可引用事实”，Session Recall 回答“我们过去说过什么”。

## 2. 为什么旧 Memory 写入简单、提取复杂

旧实现使用关键词和正则决定是否写入，是为了低成本、可复现和避免无意存储敏感内容；
提取阶段则要同时处理相关性、时效、实体、重要性和去重，因此采用更复杂的混合排序。
这种不对称本身合理，但纯规则写入漏掉隐式偏好，也会把带“记住”字样的一次性内容误存。

v5.9 改为两段式写入：

1. 真实模式由 LLM 按 JSON Schema 提取候选，区分 preference / fact / decision /
   procedure，并输出 confidence、validity 和来源。
2. 确定性策略执行禁止写入、显式“不要记住”、置信度门槛、去重、时间有效性和审计。
3. LLM 不可用或解析失败时回退旧规则；fake 测试始终走规则，保证 CI 离线可复现。

这不是让 LLM 直接写数据库，而是让 LLM 做语义提取，让策略层保留最终写权限。

## 3. DeepSeek V4 百万窗口与预算

DeepSeek V4 的模型硬窗口按 1,000,000 token 建模，但系统不应每轮填满窗口。长上下文
会增加首 token 延迟、成本和无关信息干扰，因此使用“模型硬上限 + 任务软工作集”：

| Profile | 输入软上限 | 输出预留 | 主要优先级 |
| --- | ---: | ---: | --- |
| chat | 128K | 16K | Active → Summary → Memory |
| qa | 256K | 32K | Evidence → Active → Summary |
| research | 512K | 64K | Evidence → Active → Artifact |

每层先获得 profile 目标份额，未使用预算可动态借给高优先级层，同时受绝对上限约束。
Pinned/System 不再固定占 25%；其目标份额为 chat 8%、QA 6%、research 4%，绝对上限
24K。系统提示通常远小于该上限，空闲预算会自动借出，不会浪费。

压缩触发规则：工作集估算 token 达到输入软上限的 78% 时触发；90% 是高水位保护。
最近 48 条消息作为 Active 候选，旧消息进入结构化 summary；原始事件写入 ledger，可按
`compaction_id` 恢复。窗口“支持 1M”与“每轮都发送 1M”是两件事。

## 4. 结构化上下文压缩

LLM 摘要提示词要求严格 JSON，并固定以下字段：

```text
user_goals, confirmed_facts, decisions, constraints, open_questions,
completed_actions, pending_actions, evidence_refs, memory_candidates,
topic_transitions
```

压缩规则：

- 不把推测写成事实，事实、决定、待办和问题分栏；
- 原样保留否定条件、数字、路径、错误、task/document/source ID；
- 显式记录 topic transition，旧主题不能自动成为当前主题；
- 工具大输出只留结论、来源指针和恢复线索；
- JSON 解析后再次按 Schema 归一化，异常时回退确定性摘要。

Summary 不再固定取“最后两条”，而是按当前 query 对候选摘要做 BM25 风格相关性排序，
最多选 4 条。完整跨会话原文不塞进 Summary，而由 Session Recall 按需召回。

## 5. 为什么 Context 不使用 Dense

对话上下文具有强时间性、指代关系和精确关键词，且候选规模远小于论文库。默认 Dense
会引入模型加载、向量写入和查询延迟，对“上一条命令”“task_id”“不要执行”等精确
信息未必优于词法检索。因此：

- Active Context：按时间窗口选择，不做向量检索；
- Session Recall / Summary：SQLite FTS5 BM25，中文分词失败时做 n-gram 子串兜底；
- Long-term Memory：因跨会话语义改写明显，保留 BM25 + Dense 混合检索；
- Evidence：论文语言改写和术语同义表达多，保留 chunk + Dense + BM25 + RRF，按需
  Cross-Encoder Rerank。

这减少了 Context 热路径延迟，同时没有削弱论文证据的语义召回。

## 6. Turn Planner 与旧 topic 污染

旧路由器先用关键词规则分类，再让 LLM 补充；session topic 长期不更新，并被注入每轮
提示。结果是“写个故事”也可能被旧 PIM topic 拉回论文检索。

v5.9 在真实模式改为 LLM-first Turn Planner：

- 输入最近 24 条消息、相关 summary、长期记忆和当前请求；
- 输出结构化 intent、need_retrieval、tool、rewritten_query、working_subject、confidence；
- session.topic 不再作为当前事实自动注入；working_subject 可在话题切换时更新或清空；
- 只有 LLM 不可用、超时或结构化解析失败时才走规则 fallback；
- fake 模式继续规则路由，用于无网络、无费用的确定性测试。

规则仍然有价值，但从“语义决策者”降级为“故障保险”。

## 7. 依赖与启动可靠性

- `fastapi`、`uvicorn`、`httpx` 已列入正式依赖；
- FastAPI 模块级应用直接 `app = create_app()`，依赖缺失时 fail fast，不再吞异常后
  暴露一个启动成功但请求全 500 的 `None` 应用；
- `huggingface-hub>=0.34,<1.0` 与 `sentence-transformers>=3.4,<6.0` 成组约束；
- `litellm>=1.80,<1.81` 固定到已验证 release line；
- litellm 磁盘缓存不再 import 时自动写用户目录，需显式 opt-in；
- benchmark 默认目录不可访问时返回 BLOCKED/readiness，而不是拖垮 API 启动。

## 8. 改进前后对照

| 改进前 | 造成的问题 | 根因 | v5.9 改进 | 验收结果 |
| --- | --- | --- | --- | --- |
| Memory 与论文内容共用模糊概念 | “之前聊过的论文”来源不清 | 未区分用户事实、会话历史和外部证据 | Session Recall / Memory / Evidence 三域隔离 | 边界已解决；按用户隔离测试通过 |
| Memory 只靠“记住”等规则写入 | 隐式偏好漏写，一次性内容误写 | 规则缺少语义理解 | LLM Schema 提取 + 确定性策略门禁 | 提取、拒绝、fallback 测试通过 |
| 模型窗口固定 32K | 长任务过早压缩 | 旧模型配置遗留 | 1M 硬上限 + 128K/256K/512K profile | 配置与预算测试通过 |
| System 固定可占 25% | 大窗口下配额不合理 | 静态百分比 | 4%~8% 目标 + 24K 绝对 cap + 动态借用 | 预算约束测试通过 |
| 最近只保留 8 条 | 多轮对话过早遗忘 | toy 默认值 | Active 候选提升到 48 条 | 长窗口配置测试通过 |
| Summary 固定最后 2 条 | 跨主题召回错误 | 按 ID/时间而非 query | BM25 风格相关性选 4 条 | 相关摘要选择测试通过 |
| 中文历史依赖 unicode61 | 无英文关键词时召回为空 | 中文无空格分词限制 | FTS5 BM25 + CJK n-gram fallback | 纯中文跨会话测试通过 |
| 路由规则可否决 LLM | 闲聊被旧主题拉去调研 | stale topic + 关键词优先 | LLM-first Planner，规则只 fallback | 路由与 topic 污染测试通过 |
| 摘要是自由文本 | 决策、事实、待办混淆 | 无 Schema 与校验 | 10 字段 JSON、解析归一化、失败回退 | prompt/normalization 测试通过 |
| Context 也倾向 Dense | 热路径延迟和依赖增加 | 未按数据规模和访问模式分层 | 对话 BM25/时间窗；Memory/Evidence 才用 Dense | 架构边界已落地 |
| 五段静态进度条 | 看不到 Multi-Agent 内部阶段 | UI 与 Runtime trace 粒度不一致 | 11 节点执行图、实时 SSE、节点检查器 | DOM/JS 契约与浏览器验收 |

## 9. 评测边界与剩余工作

v5.9 的单元测试证明隔离、预算、压缩、路由和 UI 契约符合设计，但不等于新的公开数据集
分数。已有 LongMemEval-S、QASPER Context、SciFact 和 QASPER 指标仍按 v5.5/v5.6
冻结口径展示，不因代码版本升级而改写。

后续正式实验应补：

1. LongMemEval-S 端到端全量 reader/judge，比较 Recent / FTS Session / v5.6 Memory；
2. 长对话 topic-switch 数据集，统计 Planner 路由准确率、误检索率和 fallback 率；
3. 128K/256K/512K profile 的质量、输入 token、TTFT 与成本 Pareto 曲线；
4. 中文跨会话真实语料上的 Recall@K，评估 n-gram fallback 是否需要升级为专用 tokenizer。
