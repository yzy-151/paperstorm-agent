# PaperStorm 面试深挖材料设计

## 目标

将现有求职材料扩展为一套面向 RAG / Agent 开发岗位的 100 题项目深挖手册，并重构简历指导，使候选人不仅能解释组件原理，还能讲清问题如何被发现、定位、修复和验证。

## 交付物

- `docs/RAG_AGENT_INTERVIEW_PLAYBOOK.md`：100 道结构化问题。
- `docs/PAPERSTORM_RESUME_GUIDE.md`：职责、难点、改造动机、技术组合和结果表达。
- `tests/test_paperstorm_career_docs.py`：结构、数量、证据边界和关键主题回归测试。

## 题库结构

| 模块 | 数量 | 核心内容 |
| --- | ---: | --- |
| 基础原理 | 30 | Chunk、Jieba、BM25、Dense、RRF、Embedding、Cross-Encoder、Parent-Child、HNSW、评测指标 |
| Bad Case 与排查 | 25 | 现象、发现、定位、根因、修复、对照实验、残余风险 |
| 假设性系统设计 | 20 | 企业知识库、百万级索引、多租户 ACL、高并发、长期记忆、调研 Agent、可观测性 |
| PaperStorm 针对性追问 | 25 | 模型与数据集选型、评测协议、本地题集、架构演进、个人贡献和取舍 |

每道题必须包含：

1. `参考回答`
2. `项目实例`
3. `排查/设计步骤`
4. `追问`
5. `考察点`
6. `常见失误`

纯定义题的“排查/设计步骤”允许写成验证或选型步骤，但不得留空。

## Bad Case 叙事合同

真实案例统一采用：

```text
现象 -> 如何发现 -> 分层定位 -> 根因 -> 修复 -> 受影响 Benchmark -> 结果 -> 未解决边界
```

重点覆盖：

- PIM / RAM / DRAM 缩写歧义。
- 查询与证据低词面重叠。
- Parent 上下文预算饥饿。
- Cross-Encoder 全量触发造成高 P95。
- Cross-Encoder 个别误排。
- MMR 从候选池重选导致 gold evidence 被移出 Top K。
- 多来源冲突、无证据拒答和引用映射错误。
- Memory、Context、路由与长期会话污染。
- ACL、缓存隔离、超时、熔断和 trace 泄漏。

## 简历叙事结构

简历指导必须回答：

- 在 Stanford STORM 基础上主要负责哪些扩展，哪些能力属于原项目。
- 为什么从单一网络搜索流程扩展为统一 Retrieval、Evidence、Memory、Context、Runtime、Evaluation 和 Observability。
- 遇到什么具体问题，使用什么证据判断根因，而不是只列技术名词。
- Jieba/领域词典、结构化 Chunk、Parent-Child、BM25 + Dense + RRF、Embedding Profile、HNSW、选择性 Cross-Encoder、证据治理分别解决什么问题。
- 哪些改进有严格配对结果，哪些只是诊断实验或私有领域 pilot。
- 提供按 RAG、Agent Runtime、评测可观测性三个岗位侧重点组合的简历 Bullet。

## 外部资料使用原则

- 牛客、小红书和社区题库只用于识别高频提问方向。
- 技术答案和数值优先引用论文、官方数据集、模型卡及仓库中的冻结报告。
- 外部结果只在协议足够清楚时列入，并同时写明 split、样本量、Top K、任务粒度和 evaluator。
- 不把 SciFact 的 Recall、QASPER Evidence Recall、QASPER Answer F1、LongMemEval session recall 当成同一类指标横向排名。
- 无法严格对齐的外部数字只作为量级参考，并明确不可直接比较。

## 验收标准

- 题库恰好 100 道，四类配额为 30/25/20/25。
- 每题六个字段完整，题号连续且无重复。
- 至少 12 个真实 PaperStorm Bad Case 具备完整排查闭环。
- 至少 10 道假设题给出需求澄清、架构、数据流、失败处理和评测方案。
- 简历指导包含职责边界、原版对比、至少 10 项技术改进组合和至少 3 套岗位化 Bullet。
- 所有数字保留数据集、样本量、协议和外推边界。
- 文档结构测试和全量测试通过。
