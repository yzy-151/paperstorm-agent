# PaperStorm 最终改进复盘与公开 Benchmark 实施交接

> 用途：项目复盘、面试准备，以及交给 DeepSeek / Claude Code / Codex 继续实现公开数据集评测。
>
> 当前基线：PaperStorm v5.4 工作区。本文不会把候选集、受控契约测试或开发集结果包装成正式线上指标。

## 1. 结论先行

PaperStorm 当前已经不是一个只会调用 LLM 写文章的 Demo，而是一个围绕论文调研构建的 Agent 工程化原型，包含：

- STORM 多角色调研、提纲、写作和润色工作流；
- arXiv、本地 PDF、Zotero 论文语料接入；
- BM25、Dense、Hybrid、可选 Rerank 的检索链路；
- 对话、检索问答、深度调研三类意图路由；
- 分层 Context、可恢复压缩、Artifact 外置和持久化 Memory；
- LangGraph 状态编排、Checkpoint、重试、幂等和 Trace；
- Web 控制台、SSE 事件、任务服务、ACL 原型和评测入口；
- synthetic seed、本地真实论文候选集、Context/Memory/Runtime 契约评测。

它适合作为校招 Agent/RAG 岗位的高完成度项目，但还不能宣称是成熟生产平台。当前最需要补齐的不是继续堆功能，而是：

1. 使用公开、可复现、有标准标签的数据集验证检索、问答、上下文和记忆能力；
2. 完成本地 v5.4 候选问题的人工审核，解锁冻结测试集；
3. 让单元测试彻底离线，建立真正运行测试的 CI；
4. 将安全、并发、协议兼容和依赖管理从“原型”推进到可部署实现。

公开 Benchmark **可以做，而且应该做**。最佳方案不是只选一个总榜，而是建立三条互补证据链：

| 证据层 | 数据 | 回答什么问题 |
| --- | --- | --- |
| 工程回归 | synthetic seed、固定故障注入 | 代码改动是否破坏既有契约 |
| 本地业务验证 | Zotero 真实论文 + 人工审核 | 对目标论文领域是否有效 |
| 公开可比评测 | BEIR、QASPER、LongMemEval、MIRACL 等 | 能否按公开协议复现并与基线公平比较 |

## 2. 最后阶段做了什么

### 2.1 从“高分 Benchmark”改成“可信 Benchmark”

早期简历使用过以下结果：

```text
Recall@K / MRR / nDCG
0.3625 / 0.2804 / 0.3006 -> 0.9875 / 0.8688 / 0.8986
```

这些数字并非伪造，但来源是 100 条 synthetic seed，其中 80 条为检索用例，数据生成规则与 PIM 消歧、tokenizer 和检索规则高度同分布。它适合做快速回归和消融，不足以代表真实论文检索。

改进措施：

- 明确标注数据来源、样本数、任务粒度、embedding 类型和运行模式；
- 将 synthetic、真实 PDF 弱标注、人工金标和公开数据集分开报告；
- 增加 document-level dev/test 隔离，避免同一论文相邻 chunk 泄漏；
- 只在 dev 选择检索配置，冻结 test 不参与调参；
- 保存数据哈希、代码 commit、模型名、参数和原始预测；
- 使用 Bootstrap 95% CI 表达小样本不确定性；
- 在证据不足时直接禁止生成“可发布结果”。

### 2.2 从 v5.2 小样本 Pilot 推进到 v5.4 人工门禁

v5.2 使用 40 篇真实 Zotero PDF、868 个 chunk，最终只有 46 条唯一候选，冻结 test 仅 12 条。Dense 在中文问题检索英文论文的 test pilot 上取得 Recall@5=0.4167，但 95% CI 为 `[0.1667, 0.6667]`，区间很宽，无法外推。

v5.4 将候选扩展为：

- 23 篇可评测论文；
- 115 条中文跨语言问题候选；
- 55 条 dev 候选；
- 60 条冻结 test 候选；
- 至少 50 条有效 test 人工审核后才允许读取冻结指标；
- 数据哈希变化后旧审核自动失效；
- 网页端提供候选审核和证据等级展示。

当前人工审核数仍为 0，因此 v5.4 的检索结果只能称为“真实语料 dev 候选实验”，不能写成正式 test 成绩。

### 2.3 建立质量与延迟联合选型

v5.4 dev 候选实验结果：

| 方法 | Recall@5 | MRR | nDCG@5 | P95 延迟 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.0909 | 0.0909 | 0.0909 | 248.5 ms |
| Dense | 0.3273 | 0.2606 | 0.2773 | 234.8 ms |
| Hybrid | 0.3636 | 0.2530 | 0.2814 | 335.7 ms |
| Hybrid + Rerank | 0.4000 | 0.3485 | 0.3617 | 3252.7 ms |

Rerank 的离线质量最好，但 CPU P95 达到约 3.25 秒，超过 500 ms 延迟预算，因此默认选择 Hybrid。这项改进体现了一个重要工程判断：**离线分数最高的方案不一定是部署方案**。

### 2.4 将 Context 压缩从 Toy Summary 改为可恢复状态管理

早期“上下文压缩”容易退化为截断或不可审计的 LLM 摘要。v4.2-v5.4 采用了分层设计：

- append-only 原始事件作为事实源；
- 当前状态、用户约束和近期消息保留在 Prompt；
- 长工具结果外置为 Artifact，只在上下文中保留引用；
- 结构化摘要记录目标、已完成步骤、约束、来源和下一步；
- 使用 `compaction_id` 恢复压缩前原文；
- 压缩失败时回退，不静默丢失历史；
- 对工具调用和工具结果做配对检查。

真实论文 Context 对照包含 20 个场景：

| 策略 | 平均输入 Token | 节省率 | 约束保留 | 来源保留 | 精确恢复 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full history | 1799.2 | 0% | 100% | 100% | 100% |
| Fixed window | 55.0 | 96.94% | 50% | 0% | 0% |
| Structured compaction | 301.9 | 83.20% | 100% | 100% | 100% |

实验还发现一个真实缺陷：摘要曾读取 Artifact 化之前的长工具输出，导致正文被重新内联，出现“刚压缩又膨胀”。修复方式是让摘要仅消费 Artifact 化后的上下文视图。

这些结果证明的是机制、Token 和信息保持之间的权衡，尚未证明最终回答正确率提升。

### 2.5 将 Memory 从聊天记录提升为受策略约束的持久记忆

当前 Memory 的关键设计包括：

- 短期上下文与长期持久记忆分离；
- 写入策略判断，不把每句话都存成长记忆；
- namespace 隔离不同 tenant、user 和 session；
- 相似记忆去重；
- 新事实可 supersede 旧事实，而不是直接覆盖审计历史；
- 支持软删除、有效期和来源元数据；
- 检索结果按相关性、时间和类型组合。

早期 100% 指标只来自 4 个功能案例和重复 query，属于确定性契约回归，不是公开 Memory Benchmark。后续必须使用 LongMemEval 一类公开协议验证跨会话信息提取、更新、时间推理和拒答。

### 2.6 将 Agent Workflow 提升为可观察 Runtime

项目在原 STORM 工作流之上增加：

- LangGraph 显式状态图；
- SQLite Checkpoint；
- 节点级 retry；
- 幂等键与恢复；
- 节点、工具、检索、LLM 的结构化 Trace；
- SSE 事件流和 Web 可视化；
- 任务状态、取消、失败原因和结果 Artifact。

这使面试叙述从“我串了几个 Prompt”升级为“我设计了可恢复、可追踪、可评测的 Agent 执行链路”。

## 3. 遇到的主要困难与解决办法

### 困难一：LLM 输出格式不稳定，空 query 进入检索

**现象**：LLM 生成的 query 列表含空行、项目符号和解释文本，旧解析逻辑把空字符串或指令片段送入检索器。

**根因**：模型输出是概率文本，而下游检索接口需要严格结构；中文约束会改变格式，但不是唯一根因。

**解决**：在模型边界做 strip、空值过滤、去重、长度限制、前缀清洗和结构校验；把 Prompt 约束作为第一层，把确定性校验作为最终防线。

**面试点**：不要相信 LLM 输出天然符合协议，Agent 系统必须在模型与工具之间建立 schema、validation 和 fallback。

### 困难二：PIM 歧义导致检索到 RAM/系统建模论文

**现象**：`PIM 神经网络抑制` 被解释为 processing-in-memory、physics-informed model 等含义，而不是 passive intermodulation。

**根因**：缩写本身多义；arXiv query 缺少射频领域锚点；只依赖向量相似度无法保证领域一致性。

**解决**：增加领域词、期望词和排除词；执行 query expansion；在候选层做领域一致性过滤；评测中保留 hard negatives 和错误类型。

**面试点**：RAG 质量问题不能只靠换 embedding 解决，应拆成 query understanding、recall、filter、rerank 和 generation 分层定位。

### 困难三：真实 PDF 不等于真实 Benchmark

**现象**：第一版从 PDF chunk 开头自动取词，产生期刊页眉、作者单位和 OCR 断词；后来直接使用标题又让 BM25 达到 1.0。

**根因**：数据是真实的，但 query 和 label 生成方式不真实；标题泄漏使任务过于简单，同主题论文只标一个正例又会制造 false negative。

**解决**：改为中文释义检索英文论文；按 document 划分；剔除冲突 query；保存证据和 hard negatives；建立人工审核门禁。

**面试点**：Benchmark 的可信度取决于任务定义、标注、划分和评测协议，不取决于文件扩展名是不是 PDF。

### 困难四：Hybrid/Rerank 并非始终更好

**现象**：早期真实弱标注数据上新检索栈反而低于 legacy；v5.4 中 Rerank 提升排序质量，却产生数秒 CPU 延迟。

**根因**：词法和语义方法对不同 query 分布各有优势；融合权重、候选数和跨语言特征影响结果；Cross-Encoder 计算成本高。

**解决**：固定 corpus、chunk 和 Top-K 做 BM25/Dense/Hybrid/Rerank 消融；同时测 Recall、MRR、nDCG 和 P95；用延迟预算做部署 gate。

**面试点**：检索优化是多目标决策，而不是只追求一个 nDCG 数字。

### 困难五：压缩率很高，但关键信息丢失

**现象**：固定窗口节省约 97% Token，却丢失一半约束、全部来源，无法恢复原始证据。

**根因**：仅按时间截断不理解 Agent 状态；纯摘要没有可靠事实源。

**解决**：事件溯源 + Artifact + 结构化状态摘要 + 最近窗口 + 精确恢复；把约束保留、来源保留和恢复率作为 guardrail。

**面试点**：Context Engineering 的目标不是最小 Token，而是在预算内最大化任务状态、约束和证据的保真度。

### 困难六：测试看似离线，实际触发真实 LLM

**现象**：完整测试发现部分测试会调用 DeepSeek/LiteLLM，并输出 Pydantic warning。

**根因**：环境开关放在 `tests/__init__.py`，使用某些 `unittest discover` 方式时测试模块按顶层导入，初始化文件不一定执行。

**解决方向**：在测试入口和 CI 显式设置禁网/禁 LLM 环境变量；为 LLM、Retriever 和网络客户端使用 dependency injection/fake；单元测试默认遇到外网调用就失败。

**面试点**：可靠测试不仅要“能跑”，还要保证 hermetic，避免成本、网络和随机性污染回归。

### 困难七：项目功能成熟度与宣传用词不一致

**现象**：宽松 CORS、参数式 tenant/user、MCP-style JSON-RPC、单进程 SQLite 等原型能力容易被写成生产级安全、完整 MCP 和线上 SLA。

**解决**：把“契约通过”与“生产验证”分开；说明 fake/real、单机/分布式、认证参数/真实认证、MCP-style/协议兼容的边界。

**面试点**：工程能力的一部分是知道系统哪里还不能被称为生产级。

## 4. 面试讲述模板

### 4.1 90 秒项目介绍

> 我基于 Stanford STORM 做了 PaperStorm，目标是把一次性论文生成脚本改造成可检索、可对话、可恢复和可评测的调研 Agent。系统支持 arXiv、本地 PDF 和 Zotero 语料，检索侧实现 BM25、Dense、Hybrid 和条件式 Rerank；Agent 侧用 LangGraph 管理状态、Checkpoint、重试和 Trace；Context 使用 append-only 事件、Artifact 外置和结构化压缩，长期 Memory 支持隔离、去重和冲突更新。项目后期我重点重做了评测诚信：原来 synthetic 数据能跑到接近 0.99，但真实论文集没有同等提升，所以我引入文档级 dev/test 隔离、冻结测试、人工审核门禁、Bootstrap CI，以及质量和延迟联合选型。真实候选 dev 上 Rerank 质量最好但 P95 超过 3 秒，因此默认选择约 336 ms 的 Hybrid。这个项目最重要的结果不是一个漂亮分数，而是形成了从失败归因到可审计实验的完整闭环。

### 4.2 深挖问题及回答要点

**为什么不直接用一个 LLM 完成全部工作？**

不同阶段的输入、输出、上下文预算、温度和失败模式不同。角色拆分便于单独配置、观测、重试和评测，也能并发执行独立章节；代价是状态和成本管理更复杂。

**为什么 Hybrid 比 Dense 好，但 MRR 可能更低？**

Hybrid 可通过词法补充提高 Top-K 覆盖，但融合后的文档顺序不一定更优。Recall 反映是否召回，MRR/nDCG 反映排序位置，必须分开看。

**为什么不默认开启 Rerank？**

当前候选 dev 上 Rerank 提高 Recall、MRR 和 nDCG，但 CPU P95 约 3.25 秒，超过 500 ms 预算。线上方案要同时满足质量、延迟、成本和稳定性约束。

**如何防止 Benchmark 泄漏？**

按 document 而不是 chunk 划分；只在 dev 调参；test 冻结且受人工门禁保护；保存数据哈希、代码版本、模型和参数；禁止根据 test 坏例继续调权重后仍使用同一 test 报分。

**Context 压缩如何避免失忆？**

原始事件不删除，长结果外置为 Artifact，Prompt 只保留状态摘要、约束、来源和近期窗口；需要时按 `compaction_id` 恢复。压缩率是优化目标，约束/来源保留和可恢复性是 guardrail。

**Memory 和普通聊天记录有什么区别？**

聊天记录是事件事实源；长期 Memory 是经过写入策略筛选、带 namespace、类型、来源、有效期和冲突关系的派生知识。两者生命周期和召回方式不同。

### 4.3 简历建议表述

推荐：

> 构建 synthetic 回归、本地 Zotero 真实论文候选与公开 Benchmark 三层 Eval Harness，覆盖 Recall@K、MRR、nDCG、Bootstrap CI、引用支持率、延迟与坏例归因；通过文档级 dev/test 隔离、冻结测试和人工审核门禁治理评测泄漏，并以质量/延迟联合 gate 在 Hybrid 与 Rerank 间完成部署选型。

> 设计可恢复 Context Engine：以 append-only 事件保存事实源，将长工具结果 Artifact 化，并保留结构化状态与近期窗口；20 个真实论文场景中相对完整历史减少 83.2% 输入 Token，同时保持约束、来源和精确恢复契约。

暂时不要写：

- “真实业务 Recall 从 0.36 提升到 0.99”；
- “错误率、ACL 泄漏率均为 0，达到生产级”；
- “通过完整 MCP 兼容认证”；
- “Benchmark 已完成人工专家标注”；
- “上下文压缩后回答质量提升 83.2%”。

## 5. 是否可以使用公开数据集评测

可以。公开数据集能补足本地 Zotero 数据不可公开、难以横向比较、标签尚未完成人工审核的问题。但不能用一个数据集给整个 Agent 打一个总分，因为检索、问答、Memory、Context 和深度调研是不同任务。

### 5.1 推荐数据集

| 优先级 | 数据集 | 评测对象 | 选择原因 | 注意事项 |
| --- | --- | --- | --- | --- |
| P0 | BEIR SciFact | 科学语料检索 | 规模较小、有 corpus/query/qrels，适合先验证适配器和指标 | 主要是英文摘要检索，不代表全文论文问答 |
| P0 | QASPER | 论文 QA、证据定位 | 问题、论文正文、答案和 evidence 与 PaperStorm 高度匹配 | 需按官方 split 和 evaluator 处理可回答/不可回答问题 |
| P1 | LongMemEval | 跨会话长期记忆 | 覆盖信息提取、多会话推理、信息更新、时间推理和拒答 | 必须固定底座模型，区分 Agent Memory 与模型本身能力 |
| P1 | MIRACL zh | 中文检索 | 有人工标注中文查询，可验证中文语义检索和跨领域泛化 | 中文 corpus 很大，先 smoke，正式结果必须使用官方完整协议 |
| P1 | LongBench 子集 | 长上下文与压缩 | 中英双语，可做同一 LLM 下 full/截断/structured compaction 配对实验 | 结果受底座 LLM 影响，不能全部归因于 Context Engine |
| P2 | STORM FreshWiki | 多来源长文调研 | 与原 STORM 论文和文章生成链路最接近 | 偏 Wikipedia/英文，不是论文问答金标 |
| P2 | DeepResearch Bench | 深度调研与引用 | 覆盖报告质量、引用正确性和完整性 | 成本高，且当前只靠 arXiv/PDF 工具覆盖不足，补通用 Web 检索后再做 |

官方来源：

- BEIR：[beir-cellar/beir](https://github.com/beir-cellar/beir)
- MIRACL：[project-miracl/miracl](https://github.com/project-miracl/miracl)
- QASPER：[AllenAI QASPER 数据集卡](https://huggingface.co/datasets/allenai/qasper)
- LongMemEval：[xiaowu0162/longmemeval](https://github.com/xiaowu0162/longmemeval)
- LongBench：[THUDM/LongBench](https://github.com/THUDM/LongBench)
- STORM/FreshWiki：[stanford-oval/storm](https://github.com/stanford-oval/storm)
- DeepResearch Bench：[Ayanami0730/deep_research_bench](https://github.com/Ayanami0730/deep_research_bench)

### 5.2 推荐顺序

第一批只做 **BEIR SciFact + QASPER**。两者体量和任务都适合论文 RAG，能够尽快获得第一个公开、可复现的端到端结果。

第二批做 **LongMemEval + LongBench 选定子集**，验证项目最容易被质疑为 Toy 的 Memory 和 Context。

第三批做 **MIRACL zh**，验证中文检索，但需要单独确认磁盘、索引时间和内存。

暂不把 DeepResearch Bench 作为 v5.5 的硬目标。当前工具面主要是论文源，而该 Benchmark 更接近开放 Web 深度研究；在工具覆盖不匹配时强跑，测到的是工具缺失，不是 Agent 上限。

## 6. 统一评测指标

### 6.1 Retrieval

- Recall@5、Recall@10；
- MRR@10；
- nDCG@10；
- 查询 P50/P95；
- 建索引时间、索引大小、峰值内存；
- 冷缓存与热缓存分别报告。

### 6.2 QA 与引用

- 官方 EM/F1 或数据集原生指标；
- Evidence Recall/F1；
- Citation Precision、Citation Recall；
- 无答案问题的 abstention precision/recall；
- Faithfulness：优先使用可验证 evidence；LLM Judge 只作为辅助，并用人工样本校准一致率。

### 6.3 Memory

- 官方总分及五类能力分数；
- Memory retrieval Recall@K；
- 新事实更新成功率与 stale fact 命中率；
- 跨用户/跨会话泄漏率；
- 平均输入 Token、压缩次数和恢复次数。

### 6.4 Context

- 相同 LLM、相同问题下的 full history / fixed window / structured compaction 配对任务分；
- Token reduction；
- 约束保留、来源保留、工具配对和 restore exact；
- 回答质量下降幅度；
- 压缩耗时和 LLM 成本。

### 6.5 Runtime

- 端到端成功率；
- P50/P95/P99；
- 每任务 LLM token、检索次数和费用；
- retry、timeout、checkpoint restore 成功率；
- Trace 完整率；
- 1/5/10/20 并发下吞吐与错误分布。

## 7. DeepSeek 实施计划（三个里程碑）

### 7.1 总原则

1. 不覆盖当前工作区未提交修改，先执行 `git status` 和 `git diff --stat`。
2. 不把公开数据集、模型权重、索引、Zotero 路径和完整预测提交到 Git。
3. 数据缓存统一放到：

```text
C:\Users\yzy\Desktop\codex\paperstorm-benchmarks\
```

4. 仓库只保存 adapter、测试、小型 fixture、聚合报告和协议说明。
5. 单元测试禁止真实网络、真实 LLM 和真实下载。
6. 所有 headline 结果必须写明 dataset、split、n、模型、embedding、reranker、时间、commit 和 CI。
7. smoke/subsample 只能叫 smoke，不得伪装成官方完整结果。
8. dev 用于调参，test 只运行一次；若根据 test 修改系统，必须建立新版本 test 或明确结果已被污染。

### 7.2 建议目录

```text
knowledge_storm/
  evaluation/
    public_benchmarks/
      __init__.py
      base.py
      metrics.py
      runner.py
      report.py
      beir_scifact.py
      qasper.py
      longmemeval.py
      miracl.py
      longbench.py
tests/
  fixtures/public_benchmarks/
  test_public_benchmark_base.py
  test_beir_scifact_adapter.py
  test_qasper_adapter.py
  test_public_benchmark_metrics.py
examples/
  storm_examples/
    run_paperstorm_public_benchmark.py
docs/
  benchmarks/
    PUBLIC_BENCHMARK_PROTOCOL.md
    public_benchmark_latest_summary.json
```

建议统一数据结构：

```python
@dataclass
class BenchmarkDocument:
    document_id: str
    title: str
    text: str
    metadata: dict

@dataclass
class BenchmarkCase:
    case_id: str
    query: str
    relevant_document_ids: list[str]
    answer: object | None
    evidence: list[dict]
    split: str
    metadata: dict
```

Adapter 只负责将官方格式转换为统一结构；指标计算、预测保存和报告生成不得散落在各 adapter 中。

### 7.3 v5.5 阶段 A：公开评测底座与 SciFact

**目标**：完成第一条完全公开、可复现的检索结果。

任务：

1. 新建统一 Benchmark adapter、runner、metric 和 manifest schema。
2. 接入 BEIR SciFact 官方 corpus、queries 和 qrels。
3. 比较以下基线，固定相同 corpus、Top-K 和 query：
   - BM25；
   - Dense；
   - Hybrid RRF；
   - Hybrid + Rerank；
   - 当前 legacy retriever。
4. 使用官方 BEIR evaluator 或与其逐项对拍，确保 nDCG/Recall 实现一致。
5. 保存 raw ranking、聚合指标、P50/P95、manifest 和错误案例。
6. 单元测试只使用 3-5 篇微型 fixture，不下载正式数据。
7. CI 显式设置 `PAPERSTORM_CHAT_LLM=0`、`PAPERSTORM_JUDGE_LLM=0`、hash embedding，并阻止外网。

验收标准：

- adapter 在离线 fixture 上可重复运行；
- 自研指标与官方 evaluator 对拍一致；
- 完整 SciFact dev/test 严格按官方协议运行；
- 报告同时给出质量、延迟、配置和限制；
- 不再出现单元测试调用 DeepSeek/LiteLLM；
- GitHub Actions 至少运行核心 unit tests，而不只检查 Black。

### 7.4 v5.5 阶段 B：QASPER 论文问答与引用

**目标**：从 document retrieval 推进到 answer/evidence/citation 闭环。

任务：

1. 按 QASPER 官方 split 读取论文、问题、答案和 evidence。
2. 建立全文 chunk，记录 paper、section、paragraph 和字符偏移。
3. 比较 BM25/Dense/Hybrid/Rerank 的 evidence recall。
4. 固定同一回答模型，比较：
   - no retrieval；
   - dense RAG；
   - hybrid RAG；
   - hybrid + rerank；
   - hybrid + compression。
5. 使用官方答案指标，并补充 evidence recall、引用 precision/recall 和 abstention。
6. LLM Judge 只评无法用原生指标覆盖的维度，先抽样至少 50 条人工校准。
7. 输出错误分类：未召回、召回未采用、证据冲突、引用错误、生成幻觉、错误拒答。

验收标准：

- retrieval 与 generation 指标分开；
- 每条答案能追溯到 chunk、论文和原始 evidence；
- 不用 Judge 分数替代官方指标；
- 报告给出成本、延迟和失败样例，不只给平均分。

v5.5 统一交付内容为：测试隔离与 CI、公开 Benchmark 通用底座、SciFact 检索、QASPER 论文问答与引用评测。阶段 A/B 可以分提交开发，但只发布一个 `v5.5` 版本。

### 7.5 v5.6 阶段 A：LongMemEval 与 LongBench Context

**目标**：用公开协议证明 Memory 和 Context 不只是四五个固定案例。

#### 7.5.1 先重构、后跑分

现有 v4.2/v4.3 只能作为兼容基线，不能直接代表成熟 Memory/Context：Memory 每次读取重放 JSONL，dense 实际为 hash embedding；Context 只有单层摘要、固定近期窗口和一个构造场景。v5.6 必须先完成以下底座升级，再运行公开 benchmark。

Memory 采用本地可部署方案：

1. SQLite WAL 替代全量 JSONL replay；原始 episode、派生 fact、source provenance、entity、event 和 namespace setting 分表存储；
2. 分离 working/episodic/semantic/preference/procedural memory；
3. episode 只追加，事实更新关闭旧事实 `valid_to` 并建立 `supersedes_id`，支持当前与历史 `as_of` 查询；
4. 检索融合 BM25、可注入真实 embedding、entity、time、importance/recency，使用 RRF 和 MMR；
5. 每条结果返回 score breakdown、命中原因、有效期和来源；hash 仅允许 CI fallback；
6. 保留 v4.3 API 兼容层，运行时切换到 v5.6 实现。

Context 采用分层虚拟工作内存：

1. Pinned 指令/硬约束；
2. Active recent turns 和未闭合 tool call；
3. 可递归 summary DAG；
4. 按需召回的 Memory/RAG evidence；
5. 外置 tool artifact 引用。

Token 预算按类型设置 floor/cap，先扣输出 reserve；soft watermark 触发正常压缩，high watermark 触发应急压缩。tool call/result 必须作为原子组保留。每次 compaction 记录 level、parent/source event、Token 前后、策略和校验结果；摘要失败、硬约束丢失或 tool pair 被破坏时必须回退。

设计依据与完整验收见 `docs/superpowers/specs/2026-08-10-v56-memory-context-design.md`，实施步骤见 `docs/superpowers/plans/2026-08-10-v56-memory-context.md`。

LongMemEval：

1. 先跑官方小规模版本或固定 smoke，确认接口；再跑完整 500 问题。
2. 固定同一 LLM，对比：
   - full conversation；
   - recent window；
   - PaperStorm Context only；
   - PaperStorm Context + persistent Memory。
3. 按五类能力分别报告，重点查看 information update 和 abstention。
4. 检查 namespace 隔离、过期事实和 supersede 是否造成错误召回。

LongBench：

1. 选择与项目相关且中英兼顾的任务，如 Qasper、MultiFieldQA-zh 和多文档 QA。
2. 同一底座模型做 paired comparison，禁止不同模型之间直接归因于压缩器。
3. 同时报告任务分、Token、压缩耗时和来源保留。

验收标准：

- 不再使用“4 个案例全通过”作为 Memory headline；
- 公开数据上报告分类分数与置信区间；
- Context 至少证明在 Token 减少时，任务分下降处于预设容忍范围；
- 所有压缩结果可关联到原始 event/artifact。

### 7.6 v5.6 阶段 B：MIRACL 中文检索

**执行前先向用户确认**：可用磁盘、内存、CPU/GPU、允许下载的数据量和预计运行时间。

任务：

1. 使用 MIRACL 中文官方 corpus、query 和 qrels。
2. smoke 可以子集运行，但正式结果必须标明是否为完整 corpus。
3. 对比中文 Dense、BM25、Hybrid、Rerank。
4. 记录建索引时间、索引大小、峰值内存、查询 P95。
5. 分析实体、术语、跨语言和长尾 query 的失败类型。

验收标准：

- 不把子集结果称为 MIRACL 官方分数；
- 公开完整配置和数据版本；
- 质量与资源消耗一起报告。

v5.6 统一交付内容为：LongMemEval 长期记忆、LongBench Context 对照、MIRACL 中文检索，以及本地 v5.4 冻结候选的人工审核结果。MIRACL 因资源不足无法完成时，明确标记为资源依赖的待办，不为它单独创建版本。

### 7.7 可选扩展：Deep Research 评测（不单独发版）

仅在 PaperStorm 增加通用 Web Search、网页正文抽取、去重、可信来源过滤和稳定引用解析后实施。

1. 先跑 10 条开发任务估算成本与工具覆盖率。
2. 验证网页可访问率、有效引用率和 Judge 稳定性。
3. 固定搜索预算、轮数、模型和最大 Token。
4. 满足预算后再决定是否运行完整 100 题。
5. 报告研究质量、引用正确性、引用完整性、成本和总耗时。

不得在只支持 arXiv/PDF 时将低分归因于 Agent 推理能力，也不得为了跑榜临时人工补答案。

### 7.8 v6.0：精简 UI 与正式项目演示（第一阶段已提前完成）

**目标**：把当前偏开发者控制台的网页整理为招聘者能在 3-5 分钟内看懂和操作的正式演示，同时保留完整调试能力。

界面分为两个层级：

1. 普通用户默认界面只保留两种主模式：
   - **论文问答**：像普通聊天机器人一样连续对话；知识不足时自动检索，回答展示引用、页码和来源；
   - **深度调研**：提交主题后执行 research → outline → article → polish，持续展示阶段、进度和结果。
2. 开发者面板默认折叠，包含：
   - Router 判定、置信度和触发原因；
   - Query rewrite、检索候选、Rerank 分数和引用映射；
   - Context Token 预算、压缩事件和 Artifact；
   - Memory 读取、写入、去重和 supersede；
   - LangGraph 节点、Tool Call、retry、checkpoint 和 Trace；
   - Benchmark 选择、运行状态、指标和坏例列表。

任务状态：

| 项目 | 状态 | 当前证据 / 缺口 |
| --- | --- | --- |
| 合并重复入口与内部参数 | **已完成** | 普通界面只保留论文调研、智能问答；KB/Benchmark/Task ID/Memory ID 进入开发者模式 |
| 默认 fake demo + 高级设置 | **已完成** | `runOfficialDemo -> create -> run -> dashboard`，无需 Key |
| 任务状态机 | **部分完成** | idle/queued/running/succeeded/failed 与五阶段进度已接入；waiting_input/cancel 尚无后端契约 |
| SSE 分类与本地时间 | **已完成历史能力** | Trace/SSE 开发者面板保留，任务状态同步到产品进度 |
| 聊天产品能力 | **部分完成** | 连续上下文、新建会话已有；会话列表、引用展开、重新生成、停止生成未完成 |
| 调研过程展示 | **部分完成** | 五阶段、文章、评分已有；大纲/检索/Multi-Agent 仍只在开发者模式，章节进度和下载未完成 |
| 官方演示案例 | **部分完成** | 无 Key fake 案例已完成；QASPER/SciFact 真实案例快捷入口未完成 |
| 根路径静态服务 | **已完成** | `/`、`styles.css`、`app.js` 实测均为 HTTP 200 |
| 单一启动与环境检查 | **部分完成** | README 已统一推荐命令；端口冲突、依赖/API Key preflight 尚未实现 |
| Playwright 验收 | **已完成** | 1366x768 一键调研、390x844 聊天、无横向溢出、真实按钮/API 流程通过 |

正式演示脚本：

```text
第 1 分钟：打开首页，展示两种工作模式和系统状态。
第 2 分钟：询问一个普通问题，展示 Router 不触发检索。
第 3 分钟：询问论文知识，展示自动检索、Rerank 和带证据回答。
第 4 分钟：追问上一轮内容，展示 Context 和 Memory；打开开发者面板查看 Trace。
第 5 分钟：提交深度调研任务，展示多 Agent 阶段、文章结果和 Benchmark 报告。
```

验收标准：

- 新用户不阅读 README，也能完成一次聊天和一次 fake 调研；
- 首页所有 CSS/JS/favicon 请求均为 200，不存在静态资源 404；
- 每个按钮都有明确名称、禁用态、加载态、成功态和失败反馈；
- 用户能判断任务是否仍在执行，以及在哪个 Agent/节点；
- 普通界面不暴露调试噪声，开发者面板可以追踪完整执行链；
- fake demo 无网络、无 Key 可运行，真实 demo 对缺失依赖和 Key 给出清晰错误；
- README 只保留一条推荐启动命令、演示 URL、两种模式和 5 分钟演示流程；
- 完成桌面与移动端视觉检查，并保存正式截图用于 README 和简历项目介绍。

## 8. 实验治理清单

每次正式实验必须生成：

```json
{
  "benchmark": "beir-scifact",
  "dataset_version": "...",
  "split": "test",
  "sample_count": 0,
  "corpus_sha256": "...",
  "dataset_sha256": "...",
  "git_commit": "...",
  "python_version": "...",
  "retriever": "hybrid",
  "embedding_model": "...",
  "reranker_model": null,
  "top_k": 10,
  "seed": 42,
  "cache_state": "cold",
  "started_at": "...",
  "finished_at": "...",
  "estimated_cost": 0.0
}
```

报告必须回答：

- 数据从哪里来，许可证是什么；
- 使用哪个官方 split；
- 是否调过 test；
- 是否使用完整 corpus；
- 使用真实 embedding 还是 hash；
- 使用 fake 还是真实 LLM/工具；
- 运行几次，随机种子是什么；
- 均值之外的不确定性多大；
- 失败样例是什么；
- 指标提升是否以延迟或成本为代价。

## 9. DeepSeek 开工提示词

> 本节原 v5.5 提示已成为历史背景。新的接手任务和量化验收以第 15 节为唯一准绳。

可将下面内容直接交给 DeepSeek：

```text
你正在接手 D:\FILEEEEEEEEEEE\projects\storm 的 PaperStorm 公开 Benchmark 建设。

先阅读：
1. docs/PAPERSTORM_FINAL_REVIEW_AND_PUBLIC_BENCHMARK_HANDOFF.md
2. docs/PAPERSTORM_V54_EVALUATION.md
3. docs/PAPERSTORM_V52_EVALUATION.md
4. docs/DESIGN_SOURCES.md
5. README.md 的 Benchmark 与限制部分

先执行 git status 和 git diff --stat。当前工作区可能有用户或其他 Agent 的未提交改动，
不得 reset、checkout 或覆盖。先执行 v5.5 阶段 A：统一公开 Benchmark adapter + BEIR SciFact；协议验证通过后继续同一版本的 QASPER 阶段 B。

要求：
- 测试默认完全离线，不得调用真实 LLM、Embedding API 或下载数据；
- 正式数据缓存放在 C:\Users\yzy\Desktop\codex\paperstorm-benchmarks；
- 仓库只提交代码、微型 fixture、协议、聚合摘要，不提交数据集和模型；
- BM25/Dense/Hybrid/Rerank 使用相同 corpus、query、Top-K；
- 指标与官方 BEIR evaluator 对拍；
- 同时报告 Recall、MRR、nDCG、P50/P95、索引时间和大小；
- 保存 dataset/version/hash、commit、模型、参数、seed 和原始 ranking；
- 先修复测试隔离和 CI，再运行正式实验；
- 不修改简历 headline，直到公开正式结果完成并经过审查；
- 完成后更新本交接文档的“执行记录”，列出成功、失败、命令、结果和下一步。

不要并行接入所有数据集。v5.5 内先完成评测底座和 SciFact，验证协议后再做 QASPER；A/B 两阶段验收后统一发布 v5.5。
版本只保留三个里程碑：v5.5、v5.6、v6.0。完成公开评测主链路后按 v6.0 精简 UI 和制作正式演示，不要为每个数据集单独发版，也不要把 Benchmark 开发参数全部堆到普通用户首页。
```

## 10. 当前工作区提醒

编写本文时，工作区存在 v5.4 相关的未提交代码、前端、测试和文档改动。接手者必须将其视为现有工作，不得回退。应先确认这些改动的测试状态和归属，再创建版本提交。

此外，项目仍需处理以下基础问题：

- `setup.py`、包版本、项目名称和上游 Stanford 元数据是否一致；
- `requirements.txt` 是否包含服务所需的 FastAPI/Uvicorn 等可选依赖；
- CI 是否真正执行测试；
- 测试是否完全离线；
- 服务 CORS、认证和租户身份是否仍为原型实现；
- MCP 是否继续准确使用 “MCP-style”，或改用官方 SDK 完成协议兼容。

这些问题不阻止开展公开 Benchmark，但会影响项目被称为“成熟生产系统”的可信度。

## 11. 最终判断

PaperStorm 最值得在面试中展示的，不是“我把一个指标调到 0.99”，而是以下能力：

1. 能识别高分实验中的同分布和标签泄漏；
2. 能把 RAG 拆成 query、recall、rerank、generation 和 citation 分层诊断；
3. 能在质量、延迟、成本和稳定性之间做部署取舍；
4. 能设计可恢复 Context 和有生命周期的 Memory；
5. 能把 Agent Loop 做成有状态、可重试、可观察的 Runtime；
6. 能承认样本量、标注、协议、安全和并发边界，并给出可执行的补齐路径。

下一阶段优先级应为：

```text
修复测试隔离与 CI
  -> BEIR SciFact 公开检索
  -> QASPER 论文 QA/证据/引用
  -> LongMemEval + LongBench
  -> MIRACL 中文检索
  -> 人工完成 v5.4 冻结集
  -> 视工具覆盖决定是否做 DeepResearch Bench
  -> v6.0 精简 UI、一键启动与 5 分钟正式演示
```

这条路线既能提高项目可信度，也直接对应 Agent/RAG 面试中最常被追问的评测、上下文、记忆、检索、Runtime 和工程治理问题。

## 12. v5.5 执行记录（2026-08-09）

### 已完成

- 建立统一公开 Benchmark 数据契约、指标、runner、manifest、raw ranking、坏例与报告；
- 接入并校验 BEIR SciFact 官方数据，完整运行 5,183 文档、300 test query；
- 接入 AllenAI QASPER，按 `paper_id` 对完整论文段落做 evidence retrieval；
- 完整运行 QASPER validation 888 条和冻结 test 1,309 条有效 evidence query；
- 冻结 validation Prompt 后，完整运行 QASPER validation 1,005 题和 test 1,451 题的真实 LLM 生成；
- 四种检索配置使用相同 query、corpus、Top-K 和候选预算；
- 正式实验使用真实 embedding、Cross-Encoder 和 5,000 次 Bootstrap；
- 单元测试增加 socket 双路径阻断，CI 使用 `unittest discover -t .`；
- 修复 BM25 无条件计算 Dense 的性能错误；
- 将 Dense exact cosine 从 Python pair loop 改为数学等价的 NumPy matrix computation；
- 修复 QASPER smoke 只保留 gold evidence 造成的标签泄漏。

### 正式结果

- SciFact test：Hybrid+Rerank Recall@10 `0.8379`、MRR@10 `0.6659`、nDCG@10 `0.7001`；Hybrid nDCG@10 `0.6687`；
- QASPER test evidence retrieval：Hybrid+Rerank Recall@5 `0.6186`、MRR@5 `0.5802`、nDCG@5 `0.5327`；Hybrid nDCG@5 `0.4155`；
- QASPER test answer generation：官方 Answer F1 `0.5441`、Evidence F1 `0.5814`、Exact Match `0.3274`，1,451/1,451 成功；
- Rerank CPU P95 分别约 `2.73s` 和 `1.32s`，因此质量最优与低延迟配置应分开。

### 失败与处理

1. SciFact HTTPS 因本地 CA 链失败：使用 `certifi` 验证证书，并保留官方 MD5 校验，没有关闭 SSL；
2. Windows 反斜杠把 Hugging Face 模型 ID 变成本地路径：统一使用标准 `/` 模型 ID；
3. 初次 SciFact exact Dense 超过十分钟：定位到 Python pairwise cosine 和 BM25 白算 Dense，修复后保持 exact quality；
4. QASPER 第一版 subset 存在 gold evidence 泄漏：改为保留问题所属论文的全部段落，并加入防泄漏测试。
5. DeepSeek 对 `::` evidence ID 输出无引号 JSON：增加限定# PaperStorm v5.5：公开论文 RAG Benchmark

## 结论先行

v5.5 将 PaperStorm 的评测从 synthetic seed 和本地 Zotero 候选扩展到两个公开、可复现的数据集：

- **BEIR SciFact**：验证科学文献全库检索；
- **AllenAI QASPER**：验证已知论文内部的证据检索、证据段落选择与端到端答案生成。

正式实验均使用真实 `all-MiniLM-L6-v2` Dense embedding、BM25、RRF Hybrid 和 `ms-marco-MiniLM-L-6-v2` Cross-Encoder，保存原始 ranking、坏例、manifest、数据哈希、模型、seed、延迟和 5,000 次 Bootstrap 95% CI。

## SciFact 官方 test

- 语料：5,183 篇科学摘要；
- Query：官方 test 300 条；
- 数据完整性：官方下载压缩包 MD5 `5f7d1de60b170fc8027bb7898e2efca1`；
- Top-K：10；
- Corpus SHA-256：`54e2468b7b03e164cd2a0d87bafe248e00e991cde4f5eab0d5122f540f6731a9`。

| 方法 | Recall@10 | MRR@10 | nDCG@10 | P95 query |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.7592 | 0.6069 | 0.6395 | 42.2 ms |
| Dense | 0.7857 | 0.6071 | 0.6492 | 20.5 ms |
| Hybrid | 0.8114 | 0.6298 | 0.6687 | 67.8 ms |
| Hybrid + Rerank | **0.8379** | **0.6659** | **0.7001** | 2733.5 ms |

Hybrid 相比 BM25 同时提高 Recall、MRR 和 nDCG；Rerank 继续提高排序质量，但 CPU P95 约 2.73 秒。因此质量最优配置是 Hybrid+Rerank，低延迟配置是 Hybrid。

## QASPER 官方 test

QASPER 的标准 evidence-selection 场景已经知道问题属于哪篇论文。因此评测不是在全部论文间找文档，而是在问题所属论文的完整段落中寻找人工 evidence。代码按 `paper_id` 建立 scoped index，保留同论文非证据段落作为 hard negatives；不会只把金标段落送给检索器。

- 官方 test 全部论文段落：19,914；
- 有人工 evidence、进入 Retrieval 分母的问题：1,309；
- Dataset fingerprint：`462b1c1545733a5e`；
- Top-K：5；
- Corpus SHA-256：`a48f71cfc64807cfd665e4f521c31cc8db6a0402464085bb60c278b55fde5c72`。

| 方法 | Evidence Recall@5 | MRR@5 | nDCG@5 | P95 query |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.4279 | 0.3714 | 0.3396 | 0.34 ms |
| Dense | 0.4771 | 0.4527 | 0.4026 | 14.5 ms |
| Hybrid | 0.5057 | 0.4595 | 0.4155 | 15.3 ms |
| Hybrid + Rerank | **0.6186** | **0.5802** | **0.5327** | 1316.7 ms |

validation 上共有 888 条有效问题，趋势与 test 一致：Hybrid+Rerank Recall@5 为 0.6126、nDCG@5 为 0.5024。正式 test 使用相同配置运行一次，结果产生后不再基于 test 调参。

### QASPER 端到端生成

在 20 条 validation smoke 上冻结 `qasper-grounded-json-v2` 后，使用相同 Hybrid+Rerank Top-5 和 `deepseek/deepseek-chat` 完整运行 validation 1,005 题与 test 1,451 题。test 不再用于调参。

| split | Answer F1 | Exact Match | Evidence F1 | 成功率 |
| --- | ---: | ---: | ---: | ---: |
| validation | 0.4722 | 0.2428 | 0.5025 | 100% |
| test | **0.5441** | **0.3274** | **0.5814** | **100%** |

test Answer F1 已与 QASPER v0.3 自带 `qasper_evaluator.py` 精确对拍；Evidence F1 使用官方 `--text_evidence_only` 口径。详细协议与排查过程见 `docs/PAPERSTORM_V55_QASPER_ANSWER_F1.md`。

## v5.5 新增实现

- 统一 `BenchmarkDocument / BenchmarkCase / BenchmarkDataset` 契约；
- BEIR JSONL/TSV 和 Hugging Face QASPER adapter；
- Recall、MRR、graded nDCG、Token F1、Evidence P/R/F1、Abstention P/R；
- SciFact 全库检索和 QASPER `paper_id` scoped retrieval；
- manifest、raw predictions、bad cases、JSON/Markdown 报告；
- 数据 MD5、corpus SHA-256、dataset fingerprint、Git commit 和工作区 dirty 状态；
- 单元测试显式阻断 socket，CI 使用 package-aware discovery；
- BM25 不再计算未使用的 Dense；
- Dense cosine 从 Python pair loop 改为数学等价的 NumPy exact matrix computation。

## 遇到的困难及解决

### 1. 官方 SciFact HTTPS 证书失败

storm Python 环境缺少正确 CA 链，`urllib` 报 `CERTIFICATE_VERIFY_FAILED`。没有关闭 SSL 校验，而是显式使用 `certifi` CA bundle、User-Agent、timeout 和 MD5 校验。

### 2. Windows 模型 ID 被写成本地路径

首次命令使用 `sentence-transformers\all-MiniLM-L6-v2`，被解释为本地目录。改用 Hugging Face 标准 ID `sentence-transformers/all-MiniLM-L6-v2`。

### 3. 精确 Dense 运行超过十分钟

旧 `_dense_search` 对每个 query、每个文档逐元素执行 Python cosine；同时 BM25 模式也无条件计算 Dense。修复后 BM25 只计算词法排名，Dense 使用 NumPy 归一化矩阵乘法。算法仍是 exact cosine，没有换 ANN 或降低候选质量。

### 4. QASPER 的潜在标签泄漏

第一版 smoke subset 只保留 gold evidence 文档，会让任务变得虚假简单。回归测试发现后改为保留所选论文的全部段落，并让每个问题只搜索对应 `paper_id` 的完整段落集合。

### 5. 不能混淆 Evidence Retrieval 与 Answer QA

QASPER Retrieval 只对 1,309 个有文本 evidence 的问题计算 Recall/MRR/nDCG；Answer QA 则覆盖全部 1,451 个 test 问题，包括 79 个不可答问题。两层独立报告，避免把“找到证据”误写成“回答正确”。

### 6. 模型 JSON 与官方口径对拍

DeepSeek 会把带 `::` 的 evidence ID 输出为未加引号的无效 JSON。实现了只修复 `evidence_ids` 裸 ID 的安全解析器，并增加解析重试与断点续跑。官方 Evidence F1 首次与本地不一致，最终定位到 `--text_evidence_only` 对 `FLOAT SELECTED` 的过滤；修正口径后，以官方 evaluator 为权威结果。

## 指标与延迟边界

- Query 延迟是在模型加载和冷建索引完成后测量的 warm-query 延迟；冷建索引时间单独保存在 manifest。
- 本轮正式实验基于未提交工作区执行，因此 `working_tree_dirty=true`；commit 只表示基线版本，完整实现差异仍以当前工作区和本报告为准。
- 延迟来自单机 CPU，只能用于同机同协议比较，不能写成线上 SLA。
- 检索指标由确定性手算 fixture 验证；实验环境未安装可选 BEIR/pytrec_eval，因此不声称完成 BEIR evaluator 自动对拍。QASPER Answer/Evidence F1 已使用数据集自带 evaluator 对拍。
- SciFact/QASPER test 已经用于本版本冻结报告，后续不能依据这些坏例调参后继续称其为未见 test。

## 复现

数据与模型缓存应放在仓库外。下面路径仅为示例：

```powershell
# SciFact：显式下载、校验并运行完整检索
python examples/storm_examples/run_paperstorm_public_benchmark.py `
  --benchmark scifact --download `
  --cache-dir <external-cache> `
  --output-dir results/public_benchmarks/v55_scifact_real `
  --split test --modes bm25 dense hybrid hybrid_rerank `
  --embedding real --model sentence-transformers/all-MiniLM-L6-v2 `
  --reranker --reranker-model cross-encoder/ms-marco-MiniLM-L-6-v2 `
  --top-k 10 --bootstrap-samples 5000 --seed 55

# QASPER：Hugging Face 官方 test，按 paper_id 搜索全文段落
python examples/storm_examples/run_paperstorm_public_benchmark.py `
  --benchmark qasper --cache-dir <external-cache> `
  --output-dir results/public_benchmarks/v55_qasper_test_real `
  --split test --modes bm25 dense hybrid hybrid_rerank `
  --embedding real --model sentence-transformers/all-MiniLM-L6-v2 `
  --reranker --reranker-model cross-encoder/ms-marco-MiniLM-L-6-v2 `
  --top-k 5 --bootstrap-samples 5000 --seed 55

# QASPER：冻结检索结果后的完整生成式 Answer F1
python examples/storm_examples/run_qasper_answer_benchmark.py `
  --split test `
  --retrieval-predictions results/public_benchmarks/v55_qasper_test_real/predictions.jsonl `
  --output-dir results/public_benchmarks/v55_qasper_answer_test_real `
  --cache-dir <external-cache> --top-k 5
```

## 面试推荐讲法

> 我没有继续使用 synthetic 0.99 作为主结果，而是接入了 SciFact 和 QASPER 两个公开论文 Benchmark。SciFact 300 个 test query 上，Hybrid+Rerank 的 nDCG@10 从 BM25 的 0.6395 提升到 0.7001；QASPER 1,309 个 evidence query 上，nDCG@5 从 0.3396 提升到 0.5327。冻结检索和 Prompt 后，我又在全部 1,451 个 QASPER test 问题上得到官方 Answer F1 0.5441、Evidence F1 0.5814，且无缺失预测。与此同时 Rerank 的 CPU P95 达到约 1.32 秒，所以我把质量最优和部署配置分开报告，并实现断点续跑、Token 审计与官方 evaluator 对拍。
字段的安全修复器、解析重试和逐题 checkpoint；
6. 本地 Evidence F1 与官方脚本不一致：定位到 `--text_evidence_only` 对 `FLOAT SELECTED` 的过滤并统一口径；Windows 官方脚本读取 UTF-8 时设置 `PYTHONUTF8=1`。

### 证据边界

- SciFact 是公开 test 文档检索；QASPER 是公开 test 论文内证据检索；
- QASPER 已运行真实 LLM 并公布官方 Answer F1，但仍未测 LLM-as-Judge Faithfulness；
- Query latency 不含模型加载和冷建索引；
- test 已用于冻结报告，后续不能再用于调参；
- 实验环境未安装可选 BEIR evaluator，标准指标由确定性手算 fixture 验证。

完整记录见 `docs/PAPERSTORM_V55_PUBLIC_BENCHMARKS.md`，脱敏摘要见 `docs/benchmarks/paperstorm_public_v55_summary.json`。下一里程碑为 v5.6：LongMemEval、LongBench Context、MIRACL 中文检索，以及 v5.4 本地候选人工审核。

## 13. v5.6 Memory/Context 执行记录（2026-08-10）

### 已完成

- 调研 Anthropic Context Engineering、LangGraph Memory、MemGPT、Graphiti、Mem0、LongMemEval 与 LongBench；
- 新增 SQLite WAL Memory：episode/fact/source/entity/event/setting/candidate 分表；
- 新增 temporal supersede 与 `as_of`，事实更新不覆盖历史；
- 新增 BM25 + injected dense + entity + time + importance/recency + RRF + MMR；
- 新增五层 Context、typed budget、tool pair 原子组、递归 summary DAG、lineage、restore 和 fallback；
- 通过兼容 facade 将聊天、Runtime、LangGraph 切换到 v5.6；
- 新增 LongMemEval/LongBench adapter、metrics、checkpoint、CLI 和离线 fixture；
- 完整运行 LongMemEval-S cleaned 500/500 evidence retrieval；
- 复用官方 QASPER test 1309 条冻结检索排名，完成 Context 预算与证据保留诊断；
- 联合回归 66 tests 通过。

### LongMemEval-S 结果

| 方法 | Recall@5 | P50 | P95 |
| --- | ---: | ---: | ---: |
| Recent 5 sessions | 0.1358 | 0 ms | 0 ms |
| v5.6 hash | 0.4813 | 146.8 ms | 202.6 ms |
| v5.6 all-MiniLM-L6-v2 CPU | **0.7930** | 1586.1 ms | 1857.3 ms |

证据等级为 `public-official-retrieval-only`。没有 reader LLM，因此不能写成 LongMemEval QA accuracy。真实向量按 query 编码 session，质量高但延迟偏高；下一步是预计算 embedding 与 ANN。

### QASPER Context 结果

| 指标 | 结果 |
| --- | ---: |
| 题数 | 1309 |
| 已召回 evidence 保留率 | **0.999847** |
| Gold evidence recall：装配前 / 后 | **0.618648 / 0.618648** |
| 平均 Context / 完整论文 token 比 | **0.166570** |
| Context token P50 | 554 |
| 超预算率 / 结构校验失败率 | **0 / 0** |

配置为 8192 模型窗口、1536 输出预留、Evidence 层占可用输入 70%。结果证明 V5.6
Context 没有在真实论文证据上造成二次召回损失，并显著控制输入预算；它不等于生成
答案 F1。逐题结果与官方数据位于仓库外统一缓存目录。

### 排查过程与修复

1. 官方 `haystack_sessions` 是 list-of-lists，并通过平行 `haystack_session_ids/haystack_dates` 给元数据；adapter 已兼容 fixture dict 与官方数组。
2. 同一 haystack 可能出现重复 session ID；改用 occurrence-aware document ID，不丢弃重复内容。
3. Windows SQLite 文件被占用；确认 Python connection context manager 不负责 close，改为 `finally: close()`。
4. tool call 先入组会导致 result 被跳过；改为预建 call-result 映射并原子选择。
5. 确定性摘要超过 summary cap 会整条被拒绝，破坏递归 level；增加按层截断与两级 restore 测试。
6. 完整 hash 导入约 14 分钟；增加已索引 document 检查，使真实 embedding 复跑复用同一事实库。

### 未完成与边界

- LongBench adapter/paired scorer 已通过离线测试；官方 v2 下载在 58.9 MB 时提前中断，JSON 校验失败，随后多次连接超时。残缺文件未计分。
- LongMemEval 尚未运行 reader LLM 端到端 QA、分类 accuracy 与官方 Judge；当前只证明 evidence retrieval。
- 未实现 ANN 和 embedding 持久化，因此正式 CPU P95 仍偏高。
- MIRACL 与 v5.4 人工审核不属于本次 Memory/Context 改造，仍按 v5.6 后续阶段执行。

完整复盘见 `docs/PAPERSTORM_V56_MEMORY_CONTEXT.md`，脱敏摘要见 `docs/benchmarks/paperstorm_memory_context_v56_summary.json`。

## 14. v5.6 精简 UI 与正式演示执行记录（2026-08-10）

### 已完成

- 产品首页重构为“论文调研 / 智能问答”两种模式；
- 用户页只保留论文调研与智能问答；旧 Task ID、submit/run/poll、v4.x synthetic Benchmark 和重复知识库入口已从网页移除；
- 新增无 Key 正式样例，一次点击组合既有任务创建、运行、SSE 和 Dashboard API；
- 新增创建任务、检索证据、生成大纲、撰写文章、完成五阶段进度；
- idle/running/succeeded/failed 使用一致的文字、状态灯和进度样式；
- 初始页不再用内置 sample 冒充已经完成的用户任务；无错误时不展示失败卡片；
- README 统一为一个启动命令和 `http://127.0.0.1:8002` 演示入口；
- 新增 `tests/test_paperstorm_demo_ui_v56.py` 和 `tests/browser/paperstorm_demo.spec.js`；
- 浏览器实际点击 fake 流程成功，文章和评分正常回显；根路径、CSS、JS 均为 200；
- 修复 390px 手机端 chat session bar 导致的 492px 横向溢出。
- 重构开发者控制台为统一 Benchmark Registry：只展示 SciFact、QASPER Retrieval/Answer、LongMemEval-S、QASPER Context 与待补输入的 LongBench；
- 新增 Benchmark 子进程运行管理 API，支持 Smoke/Quality、付费 LLM 门禁、PID/状态/日志/结果轮询与取消；
- 自动发现 `%USERPROFILE%\Desktop\codex\paperstorm-benchmarks` 并回填本地数据集、模型缓存与冻结 ranking 路径；
- 删除旧 v4.2 Context、v4.3 Memory toy benchmark 模块及旧网页/API 入口，保留仍有价值的底层算法回归；
- 修复 277 MB LongMemEval JSON 一次性 `read_text + json.loads` 导致的 `MemoryError`，改为顶层数组流式解析，Smoke 在达到 limit 后立即停止读取。

### 验收证据

```text
Python offline regression: 283/283 passed
Focused workbench/UI regression: 16/16 passed
Playwright: 3/3 passed
Desktop: 1366x768, one-click research completed
Mobile: 390x844, no horizontal overflow
Developer: 1440x1000, 6 Registry entries, 5 ready / 1 blocked
Static: /, /styles.css, /app.js -> HTTP 200
Real API smoke: LongMemEval 10 cases succeeded, Recall@5=0.4, P95=399.558 ms
```

截图位于 `docs/screenshots/dashboard-research-v56.png`、
`docs/screenshots/dashboard-chat-mobile-v56.png` 与
`docs/screenshots/dashboard-benchmark-workbench-v56.png`。临时完整截图和服务运行目录位于
`%USERPROFILE%\Desktop\codex\paperstorm-ui-validation-8013`。

## 15. DeepSeek 接手计划与量化验收目标

### 15.1 开工规则

1. 先执行 `git status --short`、`git diff --stat`，当前工作区包含 V5.4-V5.6 未提交改动，不得 reset/checkout/覆盖。
2. 先运行全量离线测试，再修改；单元测试不得联网或调用真实 LLM。
3. 数据集、模型、完整 prediction 放 `%USERPROFILE%\Desktop\codex\paperstorm-benchmarks`，不得提交 Git。
4. 每个正式实验保存 dataset/version/hash、commit、配置、seed、逐题预测、坏例和延迟分位。
5. 只完成下面 P0/P1，不再新增版本号；全部验收后统一进入 v6.0 候选。

### 15.2 P0：Memory 性能闭环

**工作**：为 V5.6 fact/session embedding 增加持久化缓存和 FAISS/HNSW ANN；批量写入使用单事务；查询不再现场编码全部 session。

**验收**：

- LongMemEval-S 同一 500 题、同一 MiniLM、Top-5；
- Recall@5 不低于 `0.773`（相对当前 0.793 最多下降 0.02）；
- 单进程 Windows CPU warm-query P95 从 `1857ms` 降到 `<=500ms`，目标 `<=300ms`；
- 重启后索引可复用，第二次启动不得重新编码全部事实；
- namespace 泄漏率 0，supersede/as_of/provenance 契约继续通过；
- 报告索引构建时间、磁盘大小、峰值内存、P50/P95。

### 15.3 P0：Memory 端到端答案评测

**工作**：在 LongMemEval-S 固定同一 reader LLM，对比 recent window、Context only、Context + V5.6 Memory；使用逐题 checkpoint，按能力类型报告。

**验收**：

- 500/500 完成，失败重试后成功率 >=99%；
- 报告官方总分，以及 information extraction、multi-session、knowledge update、temporal reasoning、abstention 分类；
- 同时报告 evidence Recall@5、答案分、Token、成本、P50/P95；
- 不把 retrieval Recall 写成 QA accuracy；
- 至少抽查 50 条错误，分类 stale fact、未召回、召回未采用、时间错误、错误拒答。

### 15.4 P0：Context 端到端质量闭环

**工作**：网络恢复后获取 LongBench 官方完整数据，使用同一 LLM 配对比较 full context、fixed window、V5.6 structured context；同时在 QASPER 冻结 test 上比较 full 与 V5.6 的 Answer/Evidence F1。

**验收**：

- 残缺下载必须做 JSON/schema/count 校验，失败不得计分；
- V5.6 平均 Token reduction >=50%；
- 相对 full context 的任务分下降 <=2 个绝对百分点；
- QASPER Gold evidence recall 不低于当前 `0.618648`，Answer F1 相对 full 下降 <=0.02；
- pinned/tool pair/over-budget guardrail 分别为 100%/100%/0%；
- 报告压缩 P50/P95、LLM 摘要成本、fallback 率和至少 50 条压缩坏例。

### 15.5 P0：正式演示第二阶段

**工作**：补齐会话列表、引用展开、重新生成、停止生成；调研结果增加 Outline/证据/章节进度和 Markdown 下载；启动脚本增加依赖、Key、端口 preflight。

**验收**：

- 新用户只看首页能在 3 次点击内完成 fake 聊天和 fake 调研；
- stop 在 1 秒内改变 UI 状态并阻止后续 token/event 写入；
- regenerate 产生新 message version，不覆盖旧回答；
- 每条引用可展开到 title/url/page/chunk，失效引用有明确状态；
- 8002 被占用时自动建议 8003，缺 uvicorn/API Key 时输出可执行安装/配置命令；
- Playwright 覆盖 desktop/mobile、stop/regenerate、citation、download、developer toggle；
- 浏览器 console error 为 0，所有本地静态请求无 404。

### 15.6 P1：中文与人工金标

- MIRACL zh 运行前先向用户确认下载量、磁盘、内存和预计时间；subsample 只能标 smoke；
- 完成 V5.4 frozen test 至少 50 条人工审核，保存 reviewer、时间、修改理由；
- 人工门禁通过前不得更新简历中的真实论文质量 headline；
- 若做 DeepResearch Bench，必须先补通用 Web search/正文抽取/来源去重，不能只用 arXiv 硬跑。

### 15.7 完成定义

DeepSeek 只有在以下条件全部满足后才能写“接手完成”：

```text
全量离线测试 = 0 failures
Playwright 正式演示 = 0 failures
LongMemEval 500/500 + 分类结果 + 成本/延迟
Context full/fixed/v5.6 同模型配对完成
Memory P95 <= 500ms 且 Recall@5 >= 0.773
数据/模型/完整预测未进入 Git
README、HANDOFF、benchmark summary 与实际结果一致
```
