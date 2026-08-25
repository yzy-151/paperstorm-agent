# PaperStorm RAG P0 实施报告

## 1. 目标与结论

本轮工作的目标是统一 PaperStorm 的检索主链、清理版本化生产模块与 toy 评测，并建立可持续的
离线回归边界。研究问答、企业知识库、运行目录检索和公开 Benchmark 现在共同依赖
`RetrievalPipeline`，其固定阶段为 `retrieve -> fuse -> rerank -> gate`。

本轮证明的是工程一致性、可追踪性和离线可测试性。它没有重新运行公开数据集质量实验，因此不宣称
Recall、F1 或 P95 相比既有报告进一步提升。现有公开指标仍以冻结报告为准。

## 2. 难点、案例与结果

### 2.1 多条检索链产生行为漂移

- **真实案例**：研究问答、企业知识库、运行目录和 Benchmark 分别维护检索逻辑，同一个 query
  可能得到不同候选顺序和不同 metadata。
- **根因**：检索、融合、重排和过滤散落在多个入口，缺少稳定请求与阶段输出契约。
- **改进方案**：新增 `RetrievalRequest` 与 `RetrievalPipeline`；所有产品和评测入口注入同一 pipeline。
- **改进结果**：跨入口契约测试验证统一的 `retrieve/fuse/rerank/gate` schema；候选、模型、延迟和
  gate 原因可被 Trace 消费。
- **结论**：工程漂移问题已解决；真实质量仍需在 P1 的冻结数据集上比较。

### 2.2 静默 fallback 掩盖生产退化

- **真实案例**：真实 embedding 不可用时，旧链可能静默退化为集合重叠或 hash 表示，服务仍返回结果，
  但语义检索质量不可知。
- **根因**：测试替身和生产 provider 没有清晰边界，异常捕获范围过大。
- **改进方案**：生产默认使用延迟加载的 SentenceTransformer；hash provider 必须显式选择；索引 schema
  不兼容时抛出 `IndexMigrationRequiredError`；问答入口不再吞掉检索错误并切到集合重叠。
- **改进结果**：离线 CI 可显式使用 hash，生产配置缺失会快速失败；旧索引迁移原因可诊断。
- **结论**：静默退化已解决；模型部署和索引迁移仍需运维 runbook。

### 2.3 内部版本号成为模块边界

- **真实案例**：`paperstorm_context_v56.py`、`paperstorm_memory_v43.py` 等文件名被生产代码直接依赖，
  新旧实现长期并存，调用者无法判断主路径。
- **根因**：版本演进通过复制模块完成，而不是稳定接口、schema revision 和数据迁移完成。
- **改进方案**：迁移为 `context_engine.py`、`memory_policy.py`、`memory_store.py`、
  `conversation_runtime.py`、`control_plane.py` 等稳定模块；仅在持久化层保留旧路径兼容读取。
- **改进结果**：AST 边界测试保证生产模块不再导入版本化模块；legacy 文件缺失也有回归测试。
- **结论**：代码边界已稳定；历史文档中的版本记录可保留为追溯信息。

### 2.4 Toy 评测与公开 Benchmark 混用

- **真实案例**：synthetic seed、确定性 top-1 生成器和公开 QASPER/SciFact 指标曾出现在同一服务接口，
  容易把 smoke test 误解为质量评测。
- **根因**：测试目的、证据等级和数据集 split 没有在接口层分离。
- **改进方案**：删除旧 toy benchmark 服务端点与模块；公开 runner 统一走产品检索主链；README 明确
  smoke 只验证确定性，quality profile 才能形成质量结论。
- **改进结果**：开发者控制台与公开 runner 只暴露稳定 Benchmark ID；旧 ID 仅保留输入兼容映射。
- **结论**：评测口径混淆已解决；P1-P3 仍需补充置信区间和 failure taxonomy。

### 2.5 Cross-Encoder 的质量与延迟冲突

- **真实案例**：SciFact R@10 从 `0.811444` 提升至 `0.837889`，P95 从 `67.7748 ms`
  增至 `2733.4805 ms`；QASPER R@5 从 `0.505659` 提升至 `0.618648`，P95 从
  `15.3228 ms` 增至 `1316.6630 ms`。
- **根因**：Cross-Encoder 对 query-document pair 做联合前向推理，不是简单排序；全量启用造成尾延迟。
- **改进方案**：默认 Hybrid，Cross-Encoder 作为 pipeline 的显式可选阶段，并记录候选数、模型与耗时。
- **改进结果**：产品可以在同一主链上比较 Hybrid 与 Hybrid+Rerank，而不会切换另一套检索实现。
- **结论**：可控与可测问题已解决；动态 `RerankPolicy` 属于 P2，尚未完成。

### 2.6 Memory、Context 与 Evidence 边界模糊

- **真实案例**：用户问“之前聊过的 PIM 论文”时，旧会话摘要可能被误当成论文事实来源。
- **根因**：会话恢复、用户长期记忆和外部证据都属于“可召回文本”，但可信度和用途不同。
- **改进方案**：Memory 只保存用户事实、偏好、决策和可复用流程；Context 负责 token 预算、摘要和恢复；
  Evidence 保存带来源的外部片段。Memory 可返回文档 identity，但科学结论必须重新查询 Evidence。
- **改进结果**：模块和 README 明确三层边界；持久化 Memory 和 Context 分别拥有稳定存储与接口。
- **结论**：工程边界已明确；跨会话引用恢复和陈旧事实污染仍需 P3 的端到端评测。

### 2.7 离线 CI 发生网络泄漏

- **真实案例**：全量单测曾尝试连接 Langfuse 并下载 HuggingFace 模型，导致慢网络下超时。
- **根因**：离线开关名称不一致，真实 provider 在构造阶段加载模型，观测 SDK 仍可能初始化远程连接。
- **改进方案**：统一识别离线环境变量；SentenceTransformer 改为延迟加载；离线时使用
  `local_files_only`；测试显式选择 hash；Langfuse key 为空且 OpenTelemetry 禁用。
- **改进结果**：定向回归可在断网配置下完成，真实模型不会在模块导入阶段下载。
- **结论**：单元测试网络泄漏已解决；真实模型集成测试应在独立、可缓存模型的 job 中运行。

### 2.8 升级兼容与索引一致性

- **真实案例**：稳定模块改名后，如果服务直接创建新目录，旧 ACL、Memory、会话 checkpoint 会表现为
  “全部消失”；知识库更新中断也可能留下半写 JSON 或 index/manifest 版本不一致。
- **根因**：代码模块迁移与持久化路径迁移混在一起；索引和 manifest 采用覆盖写入。
- **改进方案**：服务优先读取已存在的 legacy 状态目录；Control Plane 的 invoke/state/history/spec 统一
  使用同一个 graph-root resolver；索引采用临时文件、`fsync` 和原子替换；企业 KB 使用 generation
  index，最后原子切换 manifest 指针。
- **改进结果**：升级兼容路径、统一 graph root、原子索引和 generation 更新均有回归测试；更新中断时
  旧 manifest 仍指向上一代完整索引。
- **结论**：本地单进程升级与写入一致性风险已解决；跨机器迁移和多写者事务仍属于 P4 范围。

## 3. 公开指标基线

| 数据集 | 配置 | 指标 | 结果 | 解释边界 |
| --- | --- | --- | ---: | --- |
| SciFact | Hybrid | Recall@10 | 0.811444 | 第一阶段证据召回 |
| SciFact | Hybrid+Rerank | Recall@10 | 0.837889 | 质量提升伴随显著 P95 增长 |
| QASPER | Hybrid | Evidence Recall@5 | 0.505659 | 仍存在明显召回缺口 |
| QASPER | Hybrid+Rerank | Evidence Recall@5 | 0.618648 | 不等于最终答案正确率 |
| QASPER | 端到端 | Answer / Evidence F1 | 0.544147 / 0.581404 | 生成与引用仍有改进空间 |
| LongMemEval-S | Persisted Memory | Recall@5 | 0.800333 | retrieval-only，不是回答准确率 |

## 4. 验证与剩余工作

本轮验收包括检索 pipeline、持久化 Memory/Context、会话 Runtime、Control Plane、公开 Benchmark
契约、README、模块边界、离线全量测试、Python 编译与 FastAPI 导入。最终离线套件共运行 `342`
项测试（其中 `2` 项按本地 CI 环境条件跳过）；命令和结果以本轮提交日志为准。

后续按路线图推进：P1 提升第一阶段召回和结构化 chunk；P2 做选择性重排、覆盖率与冲突治理；
P3 建立 claim-citation 闭环；P4 完成 ACL、并发、发布门禁和在线治理。
