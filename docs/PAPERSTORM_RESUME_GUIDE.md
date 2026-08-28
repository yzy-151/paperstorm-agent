# PaperStorm RAG Agent 求职简历素材

## 定位

面向 RAG / Agent / LLM 应用工程岗位。以下内容是可由项目文档、评测产物和代码契约支持的候选人素材；应按实际参与范围、团队分工和面试岗位要求裁剪。不要把项目实验说成线上生产结果，也不要把组件能力说成模型能力。

## 专业简历 Bullet（择 3-5 条使用）

- 设计并实现 PaperStorm 研究问答 RAG 链路：结构化解析与 chunk、BM25 + Dense 双路召回、RRF 融合、Cross-Encoder rerank、上下文压缩及证据约束引用，形成可追溯的检索到回答闭环。
- 建立公开评测协议与可复现 harness：SciFact recall@10 0.8264（n=300）；QASPER retrieval recall@5 0.5526（n=1309）；独立 full 1451 端到端协议的 Answer F1 0.5083、Evidence F1 0.5500、claim support 0.9592、unsupported 0.0214。
- 构建带 provenance、实体和时间有效性的 Memory 检索与治理链路；LongMemEval-S cleaned 500/500 retrieval-only：recall@5 0.8003，P95 359.3 ms，分别反映该固定协议下的召回和尾延迟。
- 在私有领域 PIM pilot 中完成 5 篇论文、797 个 chunks、50 题的端到端验证：GTE recall@5 0.7200、Answer F1 0.3983、Raw Citation Precision 0.9237（规则型原始引用映射精度，不是语义/人工验证准确率）；另在该小规模离线索引上验证 HNSW recall@5 1.0000。

## 60 秒自我介绍

我专注于把 RAG 从“能检索、能回答”做成可评测、可治理、可排障的 Agent 系统。在 PaperStorm 中，我负责的重点是检索和运行时闭环：上游将论文做结构化解析和切分，中间采用 BM25 与 Dense 双路召回、RRF 融合和重排，下游通过压缩、证据校验和引用约束降低无依据回答。

我不只看单一答案分数，而是把检索、证据、回答、延迟和可观测性拆开度量。例如公开协议下，SciFact recall@10 0.8264（n=300），QASPER retrieval recall@5 0.5526（n=1309）；独立 full 1451 端到端协议的 Answer F1 0.5083、Evidence F1 0.5500。对我来说，这些是固定数据、版本和配置下的评测结果，而不是泛化承诺。我也做了 PIM 私有领域 pilot，规模是 5 篇论文、797 chunks、50 题，用来验证领域链路，不能替代公开基准或线上指标。

## 3 分钟自我介绍

我是面向 RAG 和 Agent 应用工程的候选人，关注的不是把大模型调用拼起来，而是把不确定性显式放进系统设计。PaperStorm 是一个研究型问答系统，我围绕它完成了检索、记忆、证据治理、运行时和评测的工程化连接。

第一部分是检索质量。我将文档解析后的结构、chunk 边界与 parent 上下文一起纳入检索链路，先用 BM25 覆盖术语和精确匹配，再用 Dense 覆盖语义表达，使用 RRF 避免强行比较不同检索器的原始分数，随后用 Cross-Encoder rerank 精排候选。对于回答端，我不把“检索到了”直接等同于“可以作答”：回答需要绑定可展示的原文证据和引用；若证据不足、冲突或引用无法校验，应说明不确定性或拒答。

第二部分是评测与指标解释。公开基准方面，SciFact recall@10 0.8264（n=300），QASPER retrieval recall@5 0.5526（n=1309）。独立 full 1451 端到端协议的 Answer F1 0.5083、Evidence F1 0.5500、claim support 0.9592、unsupported 0.0214。LongMemEval-S cleaned 500/500 retrieval-only：recall@5 0.8003，P95 359.3 ms。这些数字必须和数据集、任务定义、样本量、模型版本、硬件与运行配置一起陈述；它们不代表所有领域、所有负载下的效果。

第三部分是领域验证和可运行性。我做过私有领域 PIM pilot：5 篇论文、797 chunks、50 题。这里 GTE recall@5 0.7200，Answer F1 0.3983，Raw Citation Precision 0.9237（规则型原始引用映射精度，不是语义/人工验证准确率）。该 pilot 用于暴露领域术语、图表数值和多段证据等问题，不应对外表述成生产准确率。HNSW recall@5 1.0000 也是在这个小规模离线治理指标中得到的，说明近似索引在该设置下没有漏掉 Top-5，不证明大规模、不同参数或线上压力下同样成立。

最后，我会把运行时和可观测性视为产品能力的一部分：Agent runtime 的工具调用有超时、幂等和降级边界；多 Agent 协作有状态与责任边界；Langfuse 用于 trace、span、score 和 badcase 分析，但不替代固定 evaluator 或本地审计事实源。我的贡献可以概括为：把 RAG 的质量问题转化成可复现的评测、可定位的 trace 和可讨论的工程决策。

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
