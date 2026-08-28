# PaperStorm RAG Agent 求职简历素材

## 定位

面向 RAG / Agent / LLM 应用工程岗位。以下内容是可由项目文档、评测产物和代码契约支持的候选人素材；应按实际参与范围、团队分工和面试岗位要求裁剪。不要把项目实验说成线上生产结果，也不要把组件能力说成模型能力。

## 个人职责与原项目边界

| 范围 | 已有能力 / 个人职责 | 面试表达边界 |
| --- | --- | --- |
| Stanford STORM 原项目 | 已有多视角 Persona、Conv Simulator 访谈式知识整理、大纲生成、章节写作和文章润色。 | 可以说明基于其开源架构二次开发，不应把这些上游能力全部表述为个人从零实现。 |
| 个人主要负责：RAG 与论文数据 | 接入 arXiv、本地 PDF，设计结构化 Chunk、Jieba/领域词典、BM25 + Dense、RRF、Embedding Profile、Parent-Child、HNSW 与选择性 Cross-Encoder。 | 重点讲检索边界、真实 Bad Case、递进实验和局部回归，不把小样本结果外推。 |
| 个人主要负责：Agent 工程 | 扩展问答动作路由、Tool/MCP 契约、Memory/Context、任务状态、Checkpoint/SSE、Trace、证据治理与 PDF/Web 交付。 | 可以讲 Runtime/Harness 设计，但当前仍是本地工程系统，不宣称已通过企业生产流量验证。 |
| 个人主要负责：质量闭环 | 建立 SciFact、QASPER、LongMemEval-S、PIM pilot、case diff、发布门禁和 Langfuse 可观测集成。 | 公开基准、私有 pilot、ANN 治理指标严格分开；外部成绩协议不一致时不横比。 |

## 为什么进行架构改造

原版 STORM 的核心目标是从 Web 检索到长文调研，适合展示 Multi-Agent 调研思想；求职目标中的企业 RAG / Agent Harness 更关注私有文档、权限、长期状态、错误容灾、可观测和量化评测。早期 PaperStorm 还暴露出固定切分损伤结构、PIM 被 RAM/DRAM 污染、重复 RRF 导致 100% 重排、MMR 误删 gold、Parent 预算饥饿、引用断链以及 Memory 跨会话不可诊断等问题。因此改造目标不是堆框架，而是把每个失败拆成可观测阶段，并形成“发现 Bad Case -> 最小复现 -> 分层定位 -> 单变量/递进实验 -> 发布门禁”的闭环。

## 技术改进矩阵

| 原结构/问题 | 如何发现 | 技术决策 | 为什么这样选 | 结果 | 局限 |
| --- | --- | --- | --- | --- | --- |
| 固定字符切分破坏标题、公式和论证 | 检查 gold 跨段与 PDF 视觉结果 | 结构化 Chunk + provenance | 先保护文档语义边界，再调长度/overlap | 支持页码、章节和原文回绑 | 扫描件、复杂表格仍需专门解析 |
| CJK bigram 把“无源互调”拆碎 | 查看 query/gold token 与 PIM 误召回 | Jieba + 领域词典，bigram 回退 | 兼顾中文术语完整性与离线可用性 | 改善 RF 精确术语召回与可解释 token | 词典需持续治理，不能覆盖全部新词 |
| 单路检索无法同时处理术语和改写 | Sparse/Dense 分桶与 union oracle | BM25 + Dense + RRF | Sparse/Dense 错误面互补，RRF 避免原始分数量纲对齐 | QASPER retrieval recall@5 由 0.5057 提升至 0.5526 | 增加候选和延迟，仍有 query-level 回归 |
| MiniLM 单一模型上限和适配性不清 | 冻结小集只替换模型 | Legacy/BGE/GTE/Qwen Embedding Profile + fingerprint | 用领域质量、CPU 延迟、许可和硬件 Pareto 选型 | QASPER 小集 GTE 接近 Qwen 且 CPU 更快；支持可复现实验 | 小样本诊断不等同于全量结论 |
| child 精确但上下文不足，长段又稀释相关性 | gold child 命中而最终 Evidence 不完整 | Parent-Child + child 邻域 | 检索粒度与阅读粒度解耦 | 能追踪 parent allocation、used、truncated/reason | Parent Context 预算饥饿仍需门禁和调度 |
| 全库 Dense 线性扫描无法扩展 | 比较 exact 延迟并审查容量上限 | HNSW + exact 对照 | 用 ANN 加速，exact 作为治理 gold | 小规模固定 PIM 配置 HNSW recall@5 1.0000 | 不是百万级或高并发无损证明 |
| 重复 RRF 压平 margin，触发 100% 重排 | Trace 发现 rerank trigger 全命中 | 保留内部 Hybrid score，仅多查询再次 RRF | 修正分数语义而非提高门限掩盖问题 | SciFact/QASPER 触发率降至 36.33%/45.68% | 门限仍需跨域校准 |
| 全量 Cross-Encoder 昂贵且会误排 | P95 与 PPM1D rank movement | 选择性 Cross-Encoder、候选上限与超时回退 | 只在不确定 query 上支付联合编码成本 | P2 Recall 提升且避免 100% 调用 | SciFact/QASPER P95 明显增加，PPM1D 回归未完全解决 |
| MMR 从全池重选导致 gold 丢失 | QASPER Recall@5 降至 0.4631 | recall-safe MMR：冻结 TopK 内去重 | 多样性不能破坏召回门禁 | Recall@5 恢复至 0.5526 | TopK 外潜在多样性收益被限制 |
| 生成引用只有 placeholder，证据冲突不可解释 | UI 断链与 claim-level 抽查 | Evidence schema、引用映射、冲突/时效字段、证据治理 | 回答必须回到 title/author/source span | 可区分引用映射与语义支持，支持点击定位 | Raw Citation Precision 不是人工语义准确率 |
| 近期消息、摘要和长期事实混在一起 | 跨会话问题无法稳定返回且旧 topic 污染路由 | Recent + Session Summary + Long-term Memory/Context，FTS 优先、Dense 可选 | 生命周期、成本、ACL 与可引用性不同 | LongMemEval-S cleaned 500/500 retrieval-only：recall@5 0.8003，P95 359.3 ms | retrieval 指标不代表 reader 端答案正确 |
| 单条日志无法定位回答失败 | 复现时无法知道 gold 在哪层丢失 | 本地 Trace + Langfuse span/score + dataset fingerprint | 本地事实源保证可审计，Langfuse 支持聚合和回放 | 能按路由、检索、重排、压缩、生成分层定位 | Langfuse 本身不替代固定 Benchmark |

## 可组合的简历 Bullet

以下候选句按岗位选取，正式简历仍建议保留 3-5 条。

- 设计并实现 PaperStorm 研究问答 RAG 链路：结构化解析与 chunk、BM25 + Dense 双路召回、RRF 融合、Cross-Encoder rerank、上下文压缩及证据约束引用，形成可追溯的检索到回答闭环。
- 建立公开评测协议与可复现 harness：SciFact recall@10 0.8264（n=300）；QASPER retrieval recall@5 0.5526（n=1309）；独立 full 1451 端到端协议的 Answer F1 0.5083、Evidence F1 0.5500、claim support 0.9592、unsupported 0.0214。
- 构建带 provenance、实体和时间有效性的 Memory 检索与治理链路；LongMemEval-S cleaned 500/500 retrieval-only：recall@5 0.8003，P95 359.3 ms，分别反映该固定协议下的召回和尾延迟。
- 在私有领域 PIM pilot 中完成 5 篇论文、797 个 chunks、50 题的端到端验证：GTE recall@5 0.7200、Answer F1 0.3983、Raw Citation Precision 0.9237（规则型原始引用映射精度，不是语义/人工验证准确率）；另在该小规模离线索引上验证 HNSW recall@5 1.0000。

| 岗位方向 | 可替换表达 |
| --- | --- |
| RAG / 知识库 | 负责中文论文知识库检索升级，以 Jieba 领域分词、结构化 Chunk、Parent-Child、BM25 + Dense + RRF 和选择性 Cross-Encoder 处理术语、低词面重叠与长上下文；通过 SciFact/QASPER 固定协议和 case diff 约束质量与 P95。 |
| Agent Runtime / Harness | 负责问答与调研 Agent 的动作路由、Tool/MCP Schema、任务状态、幂等、Checkpoint、SSE、预算和降级，保留 STORM Multi-Agent 主链并将开放决策纳入可恢复 Runtime。 |
| Evaluation / Observability | 建立 retrieval、reader、evidence、memory 分层 Eval Harness 与 Langfuse/local trace，定位重复 RRF、MMR 回归、Parent 预算饥饿和引用断链，并将真实 Bad Case 纳入发布门禁。 |
| 领域落地 | 面向无源互调论文构建 5 篇/797 chunks/50 题私有 pilot，以查询消歧、领域词典和多 Profile Embedding 处理 PIM/RAM/DRAM 跑题；明确私有试点与公开基准边界。 |

## 60 秒自我介绍

我专注于把 RAG 从“能检索、能回答”做成可评测、可治理、可排障的 Agent 系统。在 PaperStorm 中，我负责的重点是检索和运行时闭环：上游将论文做结构化解析和切分，中间采用 BM25 与 Dense 双路召回、RRF 融合和重排，下游通过压缩、证据校验和引用约束降低无依据回答。

我不只看单一答案分数，而是把检索、证据、回答、延迟和可观测性拆开度量。例如公开协议下，SciFact recall@10 0.8264（n=300），QASPER retrieval recall@5 0.5526（n=1309）；独立 full 1451 端到端协议的 Answer F1 0.5083、Evidence F1 0.5500。对我来说，这些是固定数据、版本和配置下的评测结果，而不是泛化承诺。我也做了 PIM 私有领域 pilot，规模是 5 篇论文、797 chunks、50 题，用来验证领域链路，不能替代公开基准或线上指标。

## 3 分钟自我介绍

我是面向 RAG 和 Agent 应用工程的候选人，关注的不是把大模型调用拼起来，而是把不确定性显式放进系统设计。PaperStorm 是一个研究型问答系统，我围绕它完成了检索、记忆、证据治理、运行时和评测的工程化连接。

第一部分是检索质量。我将文档解析后的结构、chunk 边界与 parent 上下文一起纳入检索链路，先用 BM25 覆盖术语和精确匹配，再用 Dense 覆盖语义表达，使用 RRF 避免强行比较不同检索器的原始分数，随后用 Cross-Encoder rerank 精排候选。对于回答端，我不把“检索到了”直接等同于“可以作答”：回答需要绑定可展示的原文证据和引用；若证据不足、冲突或引用无法校验，应说明不确定性或拒答。

第二部分是评测与指标解释。公开基准方面，SciFact recall@10 0.8264（n=300），QASPER retrieval recall@5 0.5526（n=1309）。独立 full 1451 端到端协议的 Answer F1 0.5083、Evidence F1 0.5500、claim support 0.9592、unsupported 0.0214。LongMemEval-S cleaned 500/500 retrieval-only：recall@5 0.8003，P95 359.3 ms。这些数字必须和数据集、任务定义、样本量、模型版本、硬件与运行配置一起陈述；它们不代表所有领域、所有负载下的效果。

第三部分是领域验证和可运行性。我做过私有领域 PIM pilot：5 篇论文、797 chunks、50 题。这里 GTE recall@5 0.7200，Answer F1 0.3983，Raw Citation Precision 0.9237（规则型原始引用映射精度，不是语义/人工验证准确率）。该 pilot 用于暴露领域术语、图表数值和多段证据等问题，不应对外表述成生产准确率。HNSW recall@5 1.0000 也是在这个小规模离线治理指标中得到的，说明近似索引在该设置下没有漏掉 Top-5，不证明大规模、不同参数或线上压力下同样成立。

最后，我会把运行时和可观测性视为产品能力的一部分：Agent runtime 的工具调用有超时、幂等和降级边界；多 Agent 协作有状态与责任边界；Langfuse 用于 trace、span、score 和 badcase 分析，但不替代固定 evaluator 或本地审计事实源。我的贡献可以概括为：把 RAG 的质量问题转化成可复现的评测、可定位的 trace 和可讨论的工程决策。

## 面试叙事：难题、决策与结果

### 主线一：从“结果跑题”定位到查询语义污染

现象是“PIM 神经网络抑制”召回 RAM/DRAM 和 processing-in-memory。先检查 arXiv 原始候选，确认错误发生在首召回而非生成；再检查 query expansion，发现缩写缺少 RF 域约束。技术决策不是简单屏蔽几个词，而是建立领域 SearchPlan：保留原 query，补充 passive intermodulation / RF / suppression 必须语义，并设置 RAM/DRAM negative terms。固定案例中 RF 文档进入 Top1、forbidden hit=0；但对没有上下文的“PIM 有哪些应用”仍应澄清，不能强制解释成无源互调。

### 主线二：从“Rerank 很慢”定位到重复融合

现象是 Cross-Encoder 几乎每条 query 都运行，P95 显著上升。通过 Trace 查看 trigger_reason 和 margin，发现内部 Hybrid 已完成 RRF，外层再次 RRF 把分数间隔压平，门限因此触发 100% 重排。技术决策是修复分数语义：保留内部 Hybrid score，只在多查询合并时再次 RRF，并限制候选、设置超时回退。修复后 SciFact/QASPER 触发率分别为 36.33%/45.68%。这说明可观测性帮助找到了系统错误，但 Cross-Encoder 本身仍昂贵且存在 PPM1D 误排，不能宣称问题全部消失。

### 主线三：从“去重后更差”定位到目标函数错误

现象是加入 MMR 后 QASPER Recall@5 从基线跌至 0.4631。query-level diff 显示 MMR 从整个候选池重新选择 TopK，为多样性移除了 gold。技术决策是 recall-safe MMR：先冻结检索 TopK，只在其中做多样性排序。Recall@5 恢复为 0.5526。面试时重点不是背 MMR 公式，而是说明任何后处理都必须服从上游 Recall 门禁，并用案例移动验证目标函数没有错位。

### 主线四：从“能召回但回答缺证据”定位到上下文与引用

现象是 child 命中 gold，但长 parent 抢占预算、后续证据无法展开；生成引用又只有 placeholder。Trace 增加 parent allocation/used/truncated/reason 后定位到 Parent Context 预算饥饿。技术决策包括 unique-parent 最低配额、加权剩余预算、child 邻域和稳定 Citation 对象（title、author、locator、source span）。结果是上下文分配和引用断链可以被定位；局限是复杂表格、跨页公式和语义 claim support 仍需独立 evaluator。

### 统一排查话术

1. 先冻结输入、数据 fingerprint、模型 revision、TopK 和硬件，保证问题可复现。
2. 判断 gold 是否在首召回、融合、Rerank、Parent Context、Prompt 或最终答案中丢失。
3. 对照 Trace 的 rank、score、trigger、token、耗时、错误类型和 source ID，提出最小根因假设。
4. 写 case-level 回归，再做单变量或递进 Benchmark；同时看总体、分桶、P95、成本和回归数。
5. 发布时说明解决范围与残余边界，把失败案例留在门禁而不是删除。

## STAR 故事

### 故事一：把“答案看起来对”变成可验证的证据链

**Situation：** 研究问答场景中，纯向量召回对术语变体和精确名词不稳定；即使答案流畅，也可能没有足够证据支撑。

**Task：** 建立既覆盖关键词又覆盖语义表达的检索链路，并使生成答案能回到可核验的 chunk、页面或段落来源。

**Action：** 我将结构化 chunk 和 parent 上下文结合，采用 BM25 + Dense 双路召回，再以 RRF 融合、Cross-Encoder 重排；回答端引入证据白名单、claim 支持校验和引用校验。评测时把 retrieval recall、Evidence F1、Answer F1、claim support 与 unsupported 分开记录，并保留 badcase。

**Result：** 在公开 QASPER 独立 full 1451 端到端协议中，得到 Answer F1 0.5083、Evidence F1 0.5500、claim support 0.9592、unsupported 0.0214。结果说明该配置下多数声明可被证据支撑，但不能据此宣称“解决了幻觉”或适用于所有数据域。

### 故事二：将领域试点与公开结论严格分开

**Situation：** 通用公开集无法充分暴露 PIM 论文中的中英文术语、图表数值和跨段证据问题，而小样本领域数据又很容易被过度解读。

**Task：** 设计一个可复盘的私有领域 pilot，验证端到端链路，同时给出清晰的适用边界。

**Action：** 我固定为 5 篇论文、797 chunks、50 题，记录检索、回答和引用指标，并将 GTE、HNSW 的结果与数据规模、索引设置绑定。对漏召回、图表数值错配、多个 chunk 共同支撑一条结论等失败类型逐项保留。

**Result：** 该 PIM pilot 中，GTE recall@5 0.7200，Answer F1 0.3983，Raw Citation Precision 0.9237（规则型原始引用映射精度，不是语义/人工验证准确率）；小规模离线 HNSW recall@5 1.0000。结论是该配置具备领域验证价值和可诊断性，而不是“PIM 生产可用”或“HNSW 无损”的证明。

### 故事三：让 Agent 出问题时能定位，而非只知道失败

**Situation：** RAG 系统的问题可能发生在改写、检索、重排、压缩、工具调用或生成任一环节，单条应用日志很难定位责任。

**Task：** 提供不阻断主链路的可观测能力，并把线上式 trace 与离线评测关联起来。

**Action：** 我为 runtime 定义统一 trace、span、指标与版本标签，保留本地 JSONL 审计事实源，并将 Langfuse 作为可选分析后端。Exporter 异常、凭据缺失或网络问题采取 fail-open；质量结论仍由固定 benchmark evaluator 产生。

**Result：** 调试时可以按版本、检索器、阶段延迟与评分回放 badcase，避免把可观测平台当成评测真相。该设计提升的是定位与治理能力，不是直接提升模型分数的承诺。

## 指标真实性边界

| 类别 | 可说的事实 | 不应说成 |
| --- | --- | --- |
| 公开基准 | SciFact recall@10 0.8264（n=300）；QASPER retrieval recall@5 0.5526（n=1309）；独立 full 1451 端到端协议的 Answer F1 0.5083、Evidence F1 0.5500、claim support 0.9592、unsupported 0.0214；LongMemEval-S cleaned 500/500 retrieval-only：recall@5 0.8003，P95 359.3 ms。 | 所有知识库、语言、硬件或业务流量下的通用性能，更不能称为生产 SLA。 |
| 私有领域 pilot | PIM pilot：5 篇论文、797 chunks、50 题；GTE recall@5 0.7200、Answer F1 0.3983、Raw Citation Precision 0.9237（规则型原始引用映射精度，不是语义/人工验证准确率）。 | 行业生产准确率、客户收益，或对任意 PIM 文档的保证。 |
| 离线治理指标 | 小规模离线 PIM 索引上的 HNSW recall@5 1.0000，用于检查该配置下 ANN 与 Exact 的 Top-5 一致性。 | HNSW 在大规模数据、不同 ef/search 参数、不同硬件或并发下“零损失”。 |

面试和简历中应先说任务、数据、样本量、配置与指标定义，再说数值；没有 A/B 对照、线上流量、人工盲评或业务收益数据时，不推导节省成本、提升转化或生产稳定性。公开基准、私有领域 pilot 与离线治理指标不等同于彼此，也不能代替尚未完成的验证。
