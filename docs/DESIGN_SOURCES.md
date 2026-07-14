# 设计借鉴来源

PaperStorm 在工程化过程中参考了以下公开设计与方案。所有借鉴都是设计层面的
思路迁移，代码为自研实现；能直接对照的差异点也在文中说明。

## Claude Code（Anthropic）

参考点：

- **上下文分层与压缩**：新会话从新的上下文窗口开始；持久规则（`CLAUDE.md`）
  与自动记忆在压缩后重新注入；长会话压缩前先清理旧工具输出，再对历史做摘要；
  研究型子 Agent 使用独立上下文，只把摘要返回主会话。
  → PaperStorm v4.2 Context Engine 对应实现：append-only 事件存储、工具输出
  artifact 化、结构化 handoff 摘要、系统消息/约束保留、`compaction_id` 精确恢复；
  `storm_deep_research` 作为隔离工具只回传答案/引用/证据/artifact URI。
- **MCP 工具协议**：Model Context Protocol 定义了工具发现与调用边界。
  → PaperStorm 的 `paperstorm_mcp_server.py` 实现 `tools/list` / `tools/call`
  JSON-RPC 桥接，项目刻意使用 "MCP-style" 表述，不冒充完整 MCP 实现。
- **项目工作流**：spec → plan → 实现的文档化流程。
  → 仓库 `docs/superpowers/specs`、`docs/superpowers/plans`、handoff 文档沿用
  这一思路，供 Codex / Claude Code / DeepSeek 接手时保持可追踪。

参考来源：

- Claude Code Context Window / Memory / How Claude Code Works 文档。

## Hermes（NousResearch hermes-agent）

参考点：

- **会话历史搜索**：Hermes 将完整消息与元数据保存在 SQLite，`session_search`
  返回"目标开头 → 命中附近窗口 → 结尾"，而不是把整段历史塞进 prompt。
  → PaperStorm 的上下文视图与按 `compaction_id` 恢复采用同类"目标-过程-结果"
  的窗口化思路。
- **Context Compressor 策略**：头尾消息保护、结构化摘要、迭代重压缩、工具调用
  配对修复。
  → v4.2 `ContextEngine` 直接对应：`recent_message_count` 保留最近完整消息、
  `_structured_summary` 摘要 schema、重复压缩保留率指标、`tool_call_pairing_rate`。
- **已知局限（不照抄）**：平面 Markdown 记忆、仅 FTS5 的历史搜索、缺少类型/时间
  冲突与语义召回、摘要失败静默丢历史。
  → PaperStorm 据此补上：Memory 的类型/有效期/冲突 supersede/软删除、
  BM25+Dense+RRF 混合召回、压缩失败回退原文、以及对应 benchmark 契约。

参考来源：

- Hermes Context Compression 与 Sessions 文档。

## Anthropic Contextual Retrieval

- 思路：为每个 Chunk 补充其在整篇文档中的位置与主题，再同时建立 Embedding 与
  BM25 索引，可叠加 Rerank。
  → v4.1 的 contextual chunk 采用"确定性 metadata 注入"（Document/Category/
  标题与页号），不是 LLM 逐块生成上下文，因此文档中明确不声称实现了
  Anthropic 版本。

## Stanford STORM

- 项目本体：保留 research → outline → article → polish 的多视角长文生成链路，
  在其上做工程化增强（模型适配、检索质量、运行时、治理与评测）。

## 自己动手的部分（不归功于以上来源）

- LangGraph 状态图编排、SQLite checkpoint 与节点级重试。
- SQLite WAL 生产控制面（ACL/审计/幂等/TTL 缓存/持久任务/熔断/span）。
- 意图路由（规则兜底 + LLM 增强 + 双向安全门）。
- 运行时检索索引 LRU、LLM 调用双层缓存（lru_cache + 磁盘）。
- 前后对比 Benchmark 体系（seed 集 + Zotero 真实论文多任务组）。
