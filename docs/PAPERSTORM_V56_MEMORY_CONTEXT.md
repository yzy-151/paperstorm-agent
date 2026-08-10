# PaperStorm v5.6 Memory 与 Context 工程复盘

## 1. 为什么要重做

v4.2/v4.3 已经具备压缩、恢复、namespace、去重和 supersede 等功能契约，但实现仍偏原型：

- Memory 每次读取都重放完整 JSONL；所谓 dense 是 64 维 hash；episode、事实和来源没有规范化存储；
- Context 只有单层摘要和固定 recent window；没有递归 lineage、typed budget、pinned 层和 tool pair 原子选择；
- 评测主要是 1 个 Context 场景和 4 个 Memory 场景，能防回归，不能证明公开数据效果。

v5.6 的目标不是增加更多 `if/else`，而是把事实源、派生状态、生命周期、检索、压缩和评测拆成可解释的工程模块。

## 2. 调研与架构取舍

参考方案：

- LangGraph：线程短期状态与 namespace 长期记忆分离；semantic/episodic/procedural 分类；
- MemGPT：把 Context Window 视为有限工作内存，历史位于外部存储并按需分页；
- Graphiti/Zep：episode 是 provenance，事实带时间有效期，旧事实失效但不删除；
- Mem0：写入过滤、关键词/语义/实体/时间融合；
- Anthropic Context Engineering：优化单位 Token 的信息效用，而不是机械塞满窗口。

没有直接引入 Graphiti/Neo4j 或托管 Memory。原因是 PaperStorm 的目标包含本地演示、离线 CI 和实现可解释性；完整图数据库会显著增加部署与故障面。最终方案采用 SQLite WAL，并保留 embedding/extractor provider 接口，未来可替换外部底座。

## 3. Memory v5.6 做了什么

### 3.1 分层与数据模型

- Working：仍由 LangGraph checkpoint 管理；
- Episodic：不可变 session/tool episode；
- Semantic：稳定事实；
- Preference：用户偏好；
- Procedural：长期规则和流程。

SQLite 表分别保存 episode、fact、fact source、entity、fact-entity、event、setting 和待确认 candidate。数据库启用 WAL、foreign key 和 busy timeout。

### 3.2 时间与来源

事实更新不是覆盖：新事实写入后，旧事实变为 `superseded`，`valid_to` 等于新事实生效时间，新事实保存 `supersedes_id`。检索支持 `as_of`，因此既能回答“现在偏好什么”，也能回答“1 月时偏好什么”。每条事实都能回到 source message/episode。

### 3.3 混合检索

候选先按 namespace、状态和时间过滤，再融合：

```text
BM25 + real dense embedding + entity overlap
  -> RRF
  -> importance + recency + temporal compatibility
  -> MMR 去重复
```

结果返回每个分项分数和 `retrieval_reasons`。真实 embedding 可注入；hash 只保留作离线测试和协议基线。

## 4. Context v5.6 做了什么

### 4.1 五层工作集

1. Pinned：system/developer 和硬约束；
2. Active：近期对话与未完成工具组；
3. Summary：可递归 summary DAG；
4. Memory/Evidence：按问题召回的长期记忆与 RAG 证据；
5. Artifact：大工具结果的外置引用。

预算先扣除 output reserve，再按 layer cap 装配。任何低优先级层都不能挤掉 pinned。tool call 与对应 result 被视为一个原子组，要么一起进入，要么一起不进入。

### 4.2 递归压缩与恢复

每次压缩记录：

- `compaction_id` 与 level；
- parent compaction；
- source event IDs；
- Token before/after；
- summarizer/fallback 策略；
- pinned 与 tool pair 校验。

第二次压缩会先展开上一层摘要的原始 lineage，再生成 level 2 节点。摘要器异常时切换 deterministic fallback。SQLite ledger 只追加，原始消息可按任意 compaction ID 恢复。

### 4.3 运行时兼容

聊天、Runtime 和 LangGraph 已切换到 v5.6 Memory/Context。兼容 facade 保留 v4.2 的参数名和返回字段，因此现有服务 API、Trace 和网页端不需要同时重写。

## 5. 公开评测

### 5.1 LongMemEval-S 完整 500 题

数据：官方 `longmemeval_s_cleaned.json`，cleaned-2025-09，500/500；Top-K=5。该实验只测 evidence session retrieval，未运行 reader LLM，因此不宣称 end-to-end QA accuracy。

| 模式 | Embedding | Recall@5 | P50 | P95 |
| --- | --- | ---: | ---: | ---: |
| Recent 5 sessions | 无 | 0.1358 | 0 ms | 0 ms |
| v5.6 Memory | hash（协议基线） | 0.4813 | 146.8 ms | 202.6 ms |
| v5.6 Memory | all-MiniLM-L6-v2，CPU | **0.7930** | 1586.1 ms | 1857.3 ms |

结论：时间/namespace 过滤和混合检索明显优于 recent window；真实语义向量进一步提升召回，但当前按 query 现场编码所有 session，CPU P95 约 1.86 秒，后续应将 fact embedding 预计算并建立 ANN 索引。

原始数据、SQLite 和逐题 prediction 位于仓库外：

```text
%USERPROFILE%\Desktop\codex\paperstorm-benchmarks\v56\
```

### 5.2 QASPER Context 预算治理（1309 题）

复用 v5.5 冻结的官方 QASPER test Hybrid+Rerank Top-5 排名，将真实检索段落送入
v5.6 Context。模型窗口设为 8192，输出预留 1536，Evidence 层上限为可用输入的
70%。这是 Context diagnostic，不调用 reader LLM。

| 指标 | 结果 |
| --- | ---: |
| Retrieved evidence retention | **0.999847** |
| Gold evidence recall（进入 Context 前） | 0.618648 |
| Gold evidence recall（Context 装配后） | **0.618648** |
| 平均 Context / 完整论文 token 比 | **0.166570** |
| Context token P50 | 554 |
| 超预算率 | **0** |
| pinned/tool 结构校验通过率 | **1.0** |

结论：当前预算下 Context 没有进一步损害上游金证据召回，并把完整论文输入缩减到
平均 16.657%。`retrieved evidence retention` 不是严格 1.0，是因为极少数超长单段超过
Evidence 单层预算；装配器选择整段丢弃而不是静默截断来源。该实验不能证明答案质量，
但能把“检索没找到”和“Context 找到后又丢了”两个故障阶段分开。

### 5.3 LongBench 状态

adapter、paired scorer、checkpoint 和离线 fixture 已完成并通过。官方 LongBench v2 下载在 58.9 MB 时连接提前中断，JSON 校验失败；随后 Hugging Face 连续多次连接超时。残缺文件未用于成绩，正式 task accuracy/token paired comparison 仍标记为外部网络阻塞，不伪装成已完成榜单结果。

## 6. 遇到的困难与解决

### 6.1 官方 session ID 会重复

LongMemEval 的同一 haystack 中 session ID 可能重复，最初触发统一 Benchmark 的 duplicate document ID 检查。修复为 occurrence-aware ID，并把 gold evidence 映射到所有对应 occurrence，而不是静默丢弃重复内容。

### 6.2 Windows SQLite 文件无法删除

`with sqlite3.connect(...)` 只提交/回滚事务，不会关闭连接。Linux 上可能因文件语义不明显，Windows 会直接报文件占用。修复为 `contextmanager + finally: close()`，Memory 和 Context ledger 同步处理。

### 6.3 Tool call/result 被拆开

第一版 group 算法遇到 call 时先当单条加入，随后 result 因 call 已使用而被跳过。测试复现后改为先建立 `tool_call_id -> result indices`，再以 call 为入口组装原子组。

### 6.4 第一层摘要被预算丢弃

确定性摘要没有按 summary layer cap 截断，装配器可能整条拒绝，导致下一次压缩 level 又从 1 开始。修复为生成后按 layer budget 截断，并增加两级压缩与 restore 测试。

### 6.5 正确但慢的批量导入

完整 hash 基线约 14 分钟，主要来自 20,000+ facts 的逐条 SQLite 事务。增加已索引 namespace/document 检查，使不同 embedding 复跑可复用同一事实库，不再重复写入审计事件。下一步仍应提供显式 bulk transaction。

## 7. 测试证据

联合定向回归覆盖 v4.2/v4.3 兼容、v5.6 原生模块、官方 schema、重复 ID、checkpoint、聊天、Runtime 和 LangGraph；最终全量数字以本次仓库回归结果为准。

重点契约：

- WAL、episode 幂等和 namespace 隔离；
- duplicate/supersede/expiry/current/historical/provenance；
- embedding 注入、BM25/entity/time/RRF/MMR 和 score explanation；
- pinned、typed budget、tool pair、recursive lineage、restore、fallback；
- LongMemEval 官方并行数组 schema、重复 session、分类指标和 Bootstrap CI；
- LongBench paired quality/token delta 与 checkpoint 去重。
- QASPER 官方原始 JSON 离线适配、真实排名下的 Context token/证据保留评测。

## 8. 面试怎么讲

> 我没有直接把 Mem0 或 Graphiti 包进项目，而是提取其关键数据模型：不可变 episode、带 provenance 和有效期的派生事实、混合时间感知召回；存储用 SQLite WAL 保证本地部署和可审计。Context 借鉴 MemGPT 的虚拟内存思路，把 pinned、active、summary、retrieved evidence 和 artifact 分层，并用递归 compaction DAG 保证恢复。评测上我先用单元测试锁定生命周期，再跑 LongMemEval-S 官方 500 题和 QASPER 1309 题 Context diagnostic；真实 Memory Recall@5 为 0.7930，Context 将完整论文 token 降至平均 16.657% 且没有额外损失金证据召回。边界也很明确：Memory CPU P95 1.86 秒，LongBench 端到端任务分仍待网络恢复后完成。
