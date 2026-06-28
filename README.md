# PaperStorm Agent

> 基于 Stanford STORM 二次开发的中文论文调研与知识库 Agent，面向 RAG、Memory、Tool Calling、MCP-style tools、Multi-Agent、Runtime Trace、Eval Harness 和 Dashboard 演示。

## 项目一眼看懂

PaperStorm Agent 不是从零重写一个聊天机器人，而是在 Stanford STORM 的 Deep Research / 长文生成框架上做工程化增强，把“论文调研脚本”推进成一个可演示的 Agent 平台原型。

它目前能展示：

- **RAG 调研链路**：arXiv / Local PDF 检索、query 清洗、PIM 领域消歧、引用证据和中文综述生成。
- **Agent Runtime**：ToolRegistry、HookManager、RuntimeEvent、JSONL trace、错误脱敏和上下文压缩。
- **Memory / QA**：线程内 working memory、可治理跨会话长期记忆，以及基于调研产物的 grounded QA。
- **Multi-Agent**：Planner、Retriever、Critic、Memory、Evaluator 分工协作，保留/过滤检索结果并写入 agent trace。
- **Eval / Benchmark**：scorecard 评估任务完成度、检索相关性、跑题风险、文章质量和 runtime 可观测性。
- **Service / Dashboard**：本地 task service、FastAPI 适配器、静态 Dashboard、任务提交/运行/轮询和 release demo。

## 最终能力地图

```text
STORM Workflow
  research -> outline -> article -> polish
        |
        v
PaperStorm RAG
  arXiv / Local PDF -> query sanitizer -> PIM disambiguation -> grounded article
        |
        v
PaperStorm Runtime
  ToolRegistry -> HookManager -> RuntimeEvent -> Context Engine -> Memory Service
        |
        v
Agent Layer
  Planner -> Retriever -> Critic -> Memory -> Evaluator
        |
        v
Service / Dashboard
  TaskService -> FastAPI adapter -> Dashboard -> trace / scorecard / QA
```

## Architecture Map

```text
STORM Workflow -> PaperStorm Runtime -> Service/Dashboard
```

```text
Topic
  -> STORM research / outline / article / polish
  -> PaperStorm retrievers and tools
  -> Runtime trace, memory, context compression
  -> Multi-Agent critic and evaluator
  -> Task service artifacts
  -> Dashboard and scorecard
```

## 最终演示命令

```powershell
D:\SOFTWARE\spyder\envs\storm\python.exe examples\storm_examples\run_paperstorm_release_demo.py `
  --topic "pim 神经网络抑制" `
  --service-root ./results/paperstorm_release_demo `
  --dashboard-dir frontend\paperstorm_dashboard
```

然后打开：

```text
frontend/paperstorm_dashboard/index.html
```

需要演示本地 service 生命周期时，再运行：

```powershell
D:\SOFTWARE\spyder\envs\storm\python.exe examples\storm_examples\start_paperstorm_service.py `
  --service-root ./results/paperstorm_demo_service `
  --host 127.0.0.1 `
  --port 8000
```

面试展示主线：

```text
submit -> queued -> running -> succeeded/failed -> artifacts -> trace/scorecard
```

## v1.2 Final Packaging

v1.2 是本项目第一阶段最终包装版。此后项目进入维护和面试准备阶段，除非有明确岗位需求或真实缺陷，不再继续堆版本。

## v2.0 Research QA Agent

v2.0 是第二阶段 Research QA Agent 演示版。项目从“先调研、再手动问答”升级为“用户直接提问，Agent 判断是否需要检索，基于证据回答”的交互式文献问答链路。

核心入口：

```text
POST /research-agent/ask
```

请求示例：

```json
{
  "question": "PIM 是什么，神经网络如何抑制它？",
  "topic": "pim 神经网络抑制",
  "run_mode": "fake",
  "expected_keywords": ["passive intermodulation", "RF"],
  "forbidden_keywords": ["DRAM", "RAM", "processing-in-memory"]
}
```

返回内容包括：

```text
answer
citations
evidence
grounded
used_task_id
retrieval_triggered
decision
evidence_sufficiency
qa_history
trace
```

Research QA Agent 的阶段能力：

- v1.3：统一 ask 入口，自动创建 research task 或复用已有 task_id。
- v1.4：Evidence Sufficiency，证据不足时 `reject_low_confidence`，避免无关证据硬拼回答。
- v1.5：Dashboard 聊天式问答，页面可直接发送问题并展示 answer、citations、decision 和 sufficiency。
- v1.6：统一 evidence schema，证据包含 `source_type`、`chunk_id`、`score`、`metadata`。
- v1.7：QA history，支持连续追问的轻量会话记忆。
- v1.8：`research_qa` Tool Schema，Research QA 能力可被工具系统发现和调用。
- v1.9：Research QA Benchmark，输出 `research_qa_benchmark_report.json` 和 `research_qa_benchmark_report.md`。
- v2.0：README、版本计划和面试材料收口，形成可演示的文献检索问答 Agent。

本地 service 启动后，可在 Dashboard 的“文献检索问答”区域直接提问；如果没有 Task ID，fake 模式会自动跑一次可复现调研任务。

## v2.1 Research Chat Agent

v2.1 把 Dashboard 从“任务控制台 + 问答输入框”推进为双模式产品：

```text
调研写文章模式：submit task -> run task -> article / trace / scorecard
聊天问答模式：chat message -> context window -> memory -> evidence check -> auto research / answer
```

新增后端入口：

```text
POST /chat/sessions
GET  /chat/sessions/{chat_id}
POST /chat/sessions/{chat_id}/messages
```

聊天问答模式的返回内容包括：

```text
messages
context_window
compressed_context
memory_context
retrieval_triggered
used_task_id
research_answer
```

它解决的是“像豆包一样直接问”的体验问题：用户不用先理解 task_id 和调研产物目录，可以直接创建聊天并发送问题。系统会保留最近对话作为 sliding context window，同时把被滑出窗口的信息压缩进 `compressed_context`，并显示 working / episodic / semantic memory 命中情况。如果当前会话没有可用知识，Agent 会自动创建并运行 research task，再基于证据回答；如果已有 task_id 且证据足够，则复用已有知识库，避免重复检索。

## v3.0 RAG Memory Benchmark

v3.0 将 v2.2 到 v2.5 合并为一版可测试 baseline，补齐标准 RAG 面试主线：

```text
Chunk -> Hash Embedding -> Local JSON Vector Index -> Hybrid Retrieval -> Rerank
  -> ContextCompressionRetriever -> Long-term Memory Recall -> RAG Benchmark
```

新增核心模块：

```text
knowledge_storm/paperstorm_rag.py
knowledge_storm/paperstorm_rag_benchmark.py
```

核心类：

- `PaperStormRAGIndex`：从 PaperStorm run_dir 构建 chunk、metadata、hash embedding 和本地 JSON 索引。
- `ContextCompressionRetriever`：作为 retriever 与 prompt assembly 之间的 wrapper，执行粗过滤、规则压缩和上下文预算分配。
- `PaperStormLongTermMemoryIndex`：把 working / episodic / semantic / preferences 记忆写入本地长期记忆索引，支持跨会话 recall。

当前 v3.0 是无外部依赖 baseline：embedding 使用可复现 hash embedding，向量存储使用本地 JSON，ANN 标记为 `linear_scan_baseline`，并保留 HNSW-ready 参数位。它不是生产级 Qdrant/HNSW/cross-encoder 方案，但已经把 RAG 工程链路和评测接口做通，后续可以替换底层索引和 reranker。

新增 benchmark 输出：

```text
rag_benchmark_report.json
rag_benchmark_report.md
```

## v3.1 Enterprise Intent Router

v3.1 补齐企业 Agent Chat 常见的“四层执行链路”，解决此前聊天模式容易被 topic 绑架的问题，例如用户问“你是什么模型”时不应该硬套 PIM 调研结果。

新增核心模块：

```text
knowledge_storm/paperstorm_intent_router.py
```

四层链路：

```text
Layer 1 Intent Router
  LLM JSON Router / rule fallback -> intent, confidence, reason

Layer 2 Tool & Query Decision
  chat_fallback / research_qa / paper_research / clarify
  follow-up query rewrite

Layer 3 RAG / Memory Execution
  context_window -> compressed_context -> memory_context -> evidence QA / auto research

Layer 4 Trace / UI Observability
  router_decision -> tool_decision -> rewritten_query -> citations / sufficiency / trace
```

`PaperStormIntentRouter` 支持注入 LLM callable，让企业版可以用大模型输出结构化 JSON 决策；本地演示和测试环境使用 deterministic fallback，保证没有 API key 时也能复现。ChatAgent 不再散落写死“聊天还是检索”的判断，而是消费统一的 `router_decision` 和 `tool_decision`，并把这些字段显示到 Dashboard 的聊天调试面板。

面试可讲重点：

- Router 层负责判断用户是闲聊、系统帮助、知识库问答、触发调研还是需要澄清。
- Tool Decision 层负责选择 `chat_fallback`、`research_qa`、`paper_research` 等工具，并输出可审计 reason。
- Query Rewrite 层把“那它为什么不是 DRAM”这类追问改写成带 topic 和上一轮问题的独立 query。
- Trace/UI 层让每次回答都能看到为什么检索、为什么不检索、用了哪个 task、证据是否足够。

## v3.2 Enterprise Knowledge Base Agent

v3.2 把原计划中的“真实 RAG 底座、LLM Router/压缩、企业知识库 UI”合并为一个可演示版本。它不是强制依赖外部服务的生产系统，而是把生产替换点补齐，让项目从论文调研 Agent 进一步靠近企业内部文档知识库 Agent。

新增核心模块：

```text
knowledge_storm/paperstorm_enterprise_kb.py
```

新增核心类：

- `CallableEmbeddingProvider`：允许注入真实 embedding 后端，例如 BGE、OpenAI-compatible embedding 或公司内部 embedding 服务。
- `SentenceTransformerEmbeddingProvider`：可选接入 sentence-transformers / BGE，本地没有依赖时不影响默认运行。
- `EnterpriseKnowledgeBaseService`：支持本地 `.txt / .pdf` 文档路径建库、保存 manifest、构建 RAG index、问答和 citation 返回。

v3.2 链路：

```text
Local docs / PDF paths
  -> document text extraction
  -> chunk + overlap
  -> embedding provider
  -> local RAG index
  -> hybrid retrieval + rerank
  -> ContextCompressionRetriever / optional LLM compressor
  -> enterprise KB answer + citations + retrieval trace
```

新增 API：

```text
POST /enterprise-kbs
GET  /enterprise-kbs
POST /enterprise-kbs/{kb_id}/ask
```

Dashboard 新增“企业知识库 Agent”面板，可输入本地文档路径创建知识库，回填 KB ID 后直接提问，并查看 `KB Manifest`、`KB Retrieval` 和 `KB Citations`。

边界说明：

- 默认 embedding 仍是 hash baseline，保证无外部依赖可测试。
- 真实 embedding 可通过 provider 替换，但当前没有把 Qdrant/Milvus/FAISS 作为强依赖。
- LLM compressor 已支持 callable 注入，默认不偷偷调用外部 API。
- 当前企业知识库是本地文件版，没有 ACL、租户隔离、权限过滤和生产审计。

指标包括：

```text
context_recall
citation_precision
off_topic_rate
avg_latency_ms
p95_latency_ms
qps_estimate
```

## v4.0 RAG Evaluation Baseline

v4.0 先建立评测基线，再进入真实 Hybrid Retrieval 改造。它解决的是“回答很差，但不知道问题发生在哪一层”的排查困难。

新增能力：

- 100 条可审计种子集：80 条 PIM/RAG 可回答问题，20 条无答案问题。
- 检索指标：Recall@K、Precision@K、MRR、nDCG@K。
- 回答指标：required-term recall、citation precision/recall、abstention accuracy。
- 失败归因：retrieval、rerank、compression、generation、citation。
- 报告：`rag_eval_v4_report.json`、`rag_eval_v4_report.md`、`rag_eval_v4_bad_cases.jsonl`。
- Service/API：`POST /evaluations/rag-v4`、`GET /evaluations/rag-v4/latest`。
- Dashboard：RAG 指标、Failure Counts、Dataset Metadata 和 Bad Cases。

运行离线基线：

```powershell
D:\SOFTWARE\spyder\envs\storm\python.exe examples\storm_examples\run_paperstorm_eval_v4.py `
  --output-dir .\results\paperstorm_eval_v4 `
  --top-k 5
```

导出种子集，作为人工扩充模板：

```powershell
D:\SOFTWARE\spyder\envs\storm\python.exe examples\storm_examples\run_paperstorm_eval_v4.py `
  --export-seed-dataset .\results\paperstorm_eval_v4\seed_dataset.json
```

首次 v3.2 Hash/规则基线结果：

```text
pass_rate                0.39
retrieval_recall_at_k    0.3625
MRR                      0.2804
nDCG@K                   0.3006
citation_precision       0.2375
abstention_accuracy      1.0
retrieval_miss           51
generation_miss          10
```

这些分数的价值是作为后续 `BM25 -> Dense -> RRF -> Cross-Encoder` 消融实验的起点，不代表生产效果。内置数据明确标记为 `synthetic_seed`；真实论文 Golden Set 仍需人工审核。没有配置 Judge 时，系统不会用关键词覆盖冒充 faithfulness 分数。

## v4.1 真实 Hybrid Retrieval 与论文消融

v4.1 将 v3.2 的词集合重叠、Hash Embedding 和规则重排替换为可插拔的两阶段检索实验链路：

```text
PDF page/heading metadata
  -> ordinary / structured / contextual chunk
  -> BM25 exact retrieval + multilingual Dense retrieval
  -> Reciprocal Rank Fusion
  -> Cross-Encoder rerank Top-N
  -> Recall / MRR / nDCG / P95 / failure attribution
```

主要实现：

- `rank-bm25` 的 BM25Okapi，分词器保留英文术语、型号、公式以及中文 unigram/bigram。
- `SentenceTransformerProvider`，默认真实模型为 `paraphrase-multilingual-MiniLM-L12-v2`，索引记录模型名、维度和归一化方式。
- RRF 只融合排名，不直接相加 BM25 与 cosine 两种不可校准分数。
- `CrossEncoderReranker` 默认使用多语种 mMARCO MiniLM，只对第一阶段候选 Top-N 联合编码。
- PDF Chunk 保留 `document_id`、`parent_id`、页码、标题层级、策略和 token 数。
- Zotero 只读数据源：解析本地附件、按题名去重、使用哈希文档 ID，报告不保存私人路径或论文全文。
- Dashboard 增加八组检索消融表；网页执行确定性 Smoke，真实模型实验使用 CLI。

真实模型种子集实验（100 Case，CPU）：

| 配置 | Recall@5 | MRR | nDCG@5 | P95 |
| --- | ---: | ---: | ---: | ---: |
| ordinary BM25 | 0.8750 | 0.8348 | 0.8447 | 90 ms |
| ordinary Dense | 0.9750 | 0.9083 | 0.9253 | 128 ms |
| ordinary BM25+Dense+RRF | 0.9875 | 0.8687 | 0.8986 | 69 ms |
| contextual BM25+Dense+RRF+Cross-Encoder | **1.0000** | **0.9938** | **0.9954** | 476 ms |

Zotero 真实论文弱标注实验包含 68 个 Chunk、24 个 Case。BM25 Recall@5 为 `0.7500`；Contextual Hybrid 将 ordinary Hybrid 的 Recall@5 从 `0.7083` 提升到 `0.7500`，nDCG 从 `0.5588` 提升到 `0.6023`。Cross-Encoder 没有提升且 CPU P95 在不同组达到约 `3.26-11.12s`。原因是问题由标题/章节自动构造，天然偏向精确匹配，而且弱标签只接受原章节；该集合用于验证真实 PDF 管线，不能代替领域专家审核的 Golden Set。

运行真实种子集消融：

```powershell
$env:PAPERSTORM_MODEL_CACHE = "C:\Users\<用户名>\Desktop\codex\huggingface"
D:\SOFTWARE\spyder\envs\storm\python.exe examples\storm_examples\run_paperstorm_eval_v41.py `
  --dataset seed `
  --model-cache $env:PAPERSTORM_MODEL_CACHE `
  --output-dir .\results\paperstorm_v41_seed_real
```

运行本地 Zotero 论文实验：

```powershell
$env:PAPERSTORM_ZOTERO_ROOT = "<你的 Zotero 数据目录>"
D:\SOFTWARE\spyder\envs\storm\python.exe examples\storm_examples\run_paperstorm_eval_v41.py `
  --dataset zotero `
  --zotero-root $env:PAPERSTORM_ZOTERO_ROOT `
  --terms 无源互调 "passive intermodulation" `
  --max-papers 5 --max-pages 5 --max-cases 24 `
  --model-cache $env:PAPERSTORM_MODEL_CACHE `
  --output-dir .\results\paperstorm_v41_zotero_real
```

首次运行会下载模型。真实论文集合采用弱监督标签并标记 `domain_review_required=true`；要得到可用于模型选型的结论，下一步应由领域人员复核问题、相关 Chunk 和多相关证据。

## v4.2 可恢复 Context Engine

v4.2 将聊天中的“最近 6 条消息 + 1200 字符截断”替换为 Token 驱动、append-only、可恢复的 Context Engine：

```text
raw message/tool event JSONL
  -> estimate token usage
  -> threshold / high-watermark decision
  -> old tool output -> artifact URI + SHA256
  -> structured handoff summary
  -> dynamic context assembly
  -> model/router input view
  -> restore raw messages by compaction_id
```

核心能力：

- `ContextEventStore` 追加保存原始消息、工具事件和压缩事件，压缩不修改源记录。
- `ContextEngine` 提供 `estimate`、`should_compact`、`compact`、`assemble` 和 `restore`。
- 输入预算动态容纳系统约束、结构化摘要、最近消息、Memory、RAG evidence 和 Tool Schema，并预留输出 token。
- 超限前先将旧工具大输出替换为 `context://message/<id>#<hash>`，再压缩中间历史。
- 交接摘要包含 goal、constraints、completed、in-progress、decisions、entities、sources、errors、todos 和 source message IDs。
- 始终保留系统消息、首轮用户目标和最近完整消息；摘要失败时回退原始消息。
- Chat API 支持查看 Context Meter、强制压缩和按 `compaction_id` 恢复。
- Runtime Trace 记录压缩前后 token、artifact 数、validation 和 compaction ID。

新增 API：

```text
GET  /chat/sessions/{chat_id}/context
POST /chat/sessions/{chat_id}/context/compact
POST /chat/sessions/{chat_id}/context/restore
POST /evaluations/context-v42
```

Context Benchmark 首次结果：

| 指标 | 结果 |
| --- | ---: |
| Before Tokens | 844 |
| After Tokens | 286 |
| Token Savings Rate | 0.6611 |
| Constraint Retention | 1.0000 |
| Entity Retention | 1.0000 |
| Todo Retention | 1.0000 |
| Repeated Compaction Retention | 1.0000 |
| Tool Call Pairing | 1.0000 |
| Exact Restore | 1.0000 |

运行 Benchmark：

```powershell
D:\SOFTWARE\spyder\envs\storm\python.exe -c `
  "from knowledge_storm.paperstorm_context_benchmark_v42 import run_context_benchmark; run_context_benchmark(r'results\paperstorm_context_v42')"
```

当前默认 token counter 是可注入的本地估算器，线上应替换为模型 tokenizer 或 API usage；默认摘要器是确定性结构化摘要，可注入 LLM。Benchmark 验证压缩、约束和恢复契约，不等价于 LLM 答案一致性评测。

## v4.3 可治理的跨会话 Memory Service

v4.3 在 v4.2 Thread Context 之外新增独立长期记忆服务。它不会把所有聊天直接写进向量库，而是经过明确的写入策略和治理链路：

```text
user message
  -> candidate extraction
  -> Pydantic validation
  -> confidence gate / pending queue
  -> exact deduplication
  -> canonical-key conflict detection
  -> append-only upsert / supersede event

query
  -> namespace + enabled + status + time filter
  -> lexical BM25 + hash dense recall
  -> RRF fusion
  -> importance + recency adjustment
  -> top-k memory context
```

数据边界：

- **Thread Context**：v4.2 管理当前会话的 token 工作集、压缩和恢复。
- **Long-Term Memory**：v4.3 保存跨会话稳定事实、偏好、任务经验和操作规则。
- **Document Knowledge**：企业文档与论文 RAG 独立建库，不写入用户 Memory namespace。
- **Raw Archive**：原始聊天和工具事件用于恢复与审计，不自动等价为长期事实。

核心能力：

- Pydantic `MemoryRecordV43` 保存 memory type、subject、namespace、canonical key、来源消息、置信度、重要性、有效期和状态。
- 普通聊天默认跳过；显式偏好、稳定事实和操作规则才进入热路径，低置信度候选进入后台整理队列。
- 同一 canonical key 的冲突事实使旧记录变为 `superseded`，旧事件仍保留，不做静默覆盖。
- namespace 仅允许 `user/<id>`、`team/<id>` 或 `org/<id>`，召回前强制过滤，避免跨用户混用。
- 支持查看、搜索、编辑、软删除、导出、开关记忆和完整 audit events。
- Chat 在回答前召回长期记忆，回答后执行 Memory Policy；Runtime 暴露 `remember` / `recall_memory` 并记录 trace。
- Dashboard 展示召回分项分数、写入决策和 benchmark，可直接执行记忆治理操作。

新增 API：

```text
POST   /memories
GET    /memories
POST   /memories/search
PATCH  /memories/{memory_id}
DELETE /memories/{memory_id}
GET    /memories/export
POST   /memories/settings
POST   /evaluations/memory-v43
```

受控 benchmark 结果：

| 指标 | 结果 |
| --- | ---: |
| Memory Write Precision | 1.0000 |
| Memory Recall@K | 1.0000 |
| Stale Fact Misuse Rate | 0.0000 |
| Cross-Namespace Leakage | 0.0000 |
| Duplicate Rate | 0.0000 |
| Recall P95 | 约 2.43 ms |
| Background Consolidation | 约 431.86 candidates/s |

这些数据来自小规模确定性本地契约测试，证明隔离、冲突、去重和召回链路按设计工作；不代表真实用户流量或 LLM 语义抽取质量。默认 extractor 与 hash embedding 都是可替换 baseline，生产环境仍应接结构化 LLM extraction、真实向量后端、身份认证和人工标注 Memory Eval。

## v4.4 LangGraph Conversation Runtime 与 STORM Tool 化

v4.4 将聊天、Memory、知识检索和深度调研从手写条件链路迁移到真正的 LangGraph `StateGraph`。STORM/DSPy 原有多视角调研 pipeline 不重写，而是作为隔离的 `storm_deep_research` 工具被主图调用。

```text
START -> classify -> memory_recall
  -> casual_chat ------------------------------+
  -> memory_answer ----------------------------+
  -> knowledge_retrieval -> evidence_grade     |
       -> answer_with_citations ---------------+
       -> deep_research(STORM) -> answer -------+-> memory_candidate_write
       -> refuse_or_clarify -------------------+        -> final_trace -> END
```

运行时契约：

- Pydantic 校验每次图调用；`thread_id` 标识会话，`thread_id + request_id` 组成持久化幂等键，避免不同会话复用结果。
- LangGraph `SqliteSaver` 保存线程 checkpoint，服务重启后仍可读取 state 与 history。
- V4.3 Memory Service 作为独立跨线程 Store，按 `user/<id>` 隔离，不复制进 checkpoint 数据库。
- 检索和深度调研节点对 `ConnectionError` / `TimeoutError` 最多尝试 2 次；失败尝试和恢复结果都进入公共 trace，每个节点输出 span、耗时、状态和业务字段。
- STORM 工具只接收结构化 topic/question/options，只返回答案、引用、证据、task ID 和 artifact URI，不把子任务完整会话灌回主图。
- Chat、FastAPI 和 Dashboard 共用同一运行时；网页可查看 executed nodes、graph run、当前 state 和 checkpoint history。

新增 API：

```text
POST /conversation-graph/invoke
GET  /conversation-graph/spec
GET  /conversation-graph/threads/{thread_id}/state
GET  /conversation-graph/threads/{thread_id}/history
POST /evaluations/runtime-v44
GET  /evaluations/runtime-v44/latest
```

受控本机 benchmark：

| 指标 | 结果 |
| --- | ---: |
| Path Accuracy | 1.0000 |
| Idempotency Rate | 1.0000 |
| Checkpoint Restore Rate | 1.0000 |
| Retry Recovery Rate | 1.0000 |
| Cross-User Leakage Rate | 0.0000 |
| Trace Span Coverage | 1.0000 |
| Artifact Contract Rate | 1.0000 |
| Latency P95 | 约 104.95 ms |

运行 benchmark：

```powershell
D:\SOFTWARE\spyder\envs\storm\python.exe -c `
  "from knowledge_storm.paperstorm_langgraph_benchmark_v44 import run_langgraph_benchmark; run_langgraph_benchmark(r'results\paperstorm_runtime_v44')"
```

这些指标使用受控 fake 调研和固定路由案例，验证的是状态图软件契约，不代表真实 arXiv/LLM 延迟或生产 QPS。SQLite 适合本地单进程演示；多进程部署需要数据库 checkpointer、事务型幂等表、异步超时取消和分布式 worker。

## v4.5 生产治理基线

v4.5 不重写 V4.4 LangGraph 业务图，而是在外层加入统一生产控制面。聊天响应的外层运行时为 `paperstorm-production-v4.5`，底层仍明确报告 `graph_runtime=langgraph-v4.4`。

```text
FastAPI / Dashboard
        |
        v
Production Control Plane v4.5
  |- tenant/user/resource ACL + audit
  |- transactional idempotency
  |- TTL cache + tag invalidation
  |- durable jobs + retry + circuit breaker
  `- trace/span store + SLO metrics
        |
        +--> LangGraph Conversation Runtime v4.4
        |      `--> Memory / RAG / STORM Deep Research Tool
        `--> Enterprise KB incremental index worker
```

关键能力：

- 对 conversation thread、trace、knowledge base 和 document 建立 tenant/user 资源策略；业务 manifest、索引和 trace 在读取前先鉴权，知识库列表按身份过滤。
- 以 SQLite 唯一约束实现事务型幂等。并发请求只有一个 owner 执行业务，其余等待并回放结果；同 key 不同 payload 被拒绝。
- 企业文档按 SHA-256 识别变化，通过持久任务增量更新索引；支持幂等入队、失败重试和恢复。
- 问答缓存同时绑定 tenant、KB、query、top_k 与 index version；索引更新按 tag 主动失效并统计命中、未命中和过期。
- provider 故障采用有限重试、持久熔断状态和显式降级；FastAPI 将权限/参数/资源异常映射为 `403/400/404`。
- Trace 统一记录 runtime 与 LangGraph node spans；Dashboard 可以查看控制面状态、当前 trace 和 Production Benchmark。

新增 API：

```text
POST /enterprise-kbs/{kb_id}/index-jobs
POST /production/worker/tick
GET  /production/status
GET  /production/traces/{trace_id}?tenant_id=...&user_id=...
POST /evaluations/production-v45
GET  /evaluations/production-v45/latest
```

500 请求本机治理热路径实验：

| 指标 | 结果 |
| --- | ---: |
| P50 / P95 / P99 | 24.70 / 28.03 / 36.28 ms |
| QPS | 39.63 |
| Error Rate | 0.000 |
| ACL Leakage Rate | 0.000 |
| Idempotency / Job Recovery / Trace Coverage | 1.000 / 1.000 / 1.000 |
| Cache Hit Rate | 0.998 |
| Injected Degradation Rate | 0.002 |

以上数据只验证单进程 SQLite WAL 治理热路径。它不包含真实 LLM、arXiv、Embedding、Reranker 延迟，也不代表分布式线上 QPS；当前未接 OAuth/OIDC、外部 policy service、PostgreSQL/Redis、分布式 worker 或 OpenTelemetry Collector。

本仓库原始项目来自 Stanford STORM。官方 README 已保留在：

```text
README_STORM_OFFICIAL.md
```

官方中文说明：

```text
docs/STORM_OFFICIAL_CN.md
```

当前 GitHub 仓库：

```text
https://github.com/yzy-151/paperstorm-agent
```

当前 README 记录本 fork 的中文项目定位、运行方式、版本计划和求职展示重点。

## 1. 官方 STORM 基础架构

STORM 全称是：

```text
Synthesis of Topic Outlines through Retrieval and Multi-perspective Question Asking
```

它是 Stanford OVAL 提出的长文调研与写作系统，目标是从一个 topic 出发，通过检索、多视角提问、模拟对话、知识整理和写作，生成类似 Wikipedia 风格的长文章。

官方整体架构图：

<p align="center">
  <img src="assets/overview.svg" style="width: 90%; height: auto;">
</p>

官方两阶段流程图：

<p align="center">
  <img src="assets/two_stages.jpg" style="width: 65%; height: auto;">
</p>

STORM 的核心流程：

```text
Pre-writing stage: research -> outline
Writing stage: article -> polish
```

在工程入口中，对应 `STORMWikiRunner.run()` 的四个开关：

```text
do_research
do_generate_outline
do_generate_article
do_polish_article
```

官方 STORM 的关键机制：

- `Perspective-Guided Question Asking`：先发现不同视角，再让不同视角控制问题生成，避免问题浅、散、重复。
- `Simulated Conversation`：模拟 Wikipedia writer 和 topic expert 的多轮对话，让 writer 追问、expert 基于检索结果回答。
- 模块化 `dspy` 实现：retriever、LM、runner、module 解耦，便于替换搜索引擎和语言模型。
- 多 LLM 配置：不同阶段可使用不同模型，在成本、速度、上下文长度和生成质量之间做权衡。

Co-STORM 是官方协作版本，加入 human-AI collaborative knowledge curation、Moderator、LLM experts 和动态 mind map：

<p align="center">
  <img src="assets/co-storm-workflow.jpg" style="width: 65%; height: auto;">
</p>

更详细的官方中文整理见：

```text
docs/STORM_OFFICIAL_CN.md
```

## 2. PaperStorm Agent 项目定位

PaperStorm Agent 的目标不是从零重写 STORM，而是在成熟 Deep Research / RAG 框架上做面向 Agent 开发岗的工程化改造：

- 中文论文调研报告生成。
- arXiv / 本地 PDF 检索。
- LLM query 清洗与领域消歧。
- Runtime Trace / Hook。
- Tool Schema 与 MCP-style 工具入口。
- Memory、Context Compression 与知识库问答。
- Eval Harness 量化评估 Agent 运行质量。
- 后续补 Multi-Agent、知识库服务化、高并发实验和前端展示。

与另一个项目 `nonlinear-nn-agent` 的分工：

- `nonlinear-nn-agent`：从零实现轻量 Agent Harness Runtime，突出 ToolRegistry、Hook、Session、Trace、Async、Retry。
- `PaperStorm Agent`：基于成熟 RAG/Deep Research 框架二次开发，突出 RAG、Memory、MCP、Multi-Agent、Eval、知识库与前端展示。

## 3. 当前已完成能力

### v0.1：PaperStorm MVP

- 支持 DeepSeek / MiniMax LLM 后端。
- 支持 arXiv 论文检索。
- 支持本地 PDF 论文片段检索。
- 支持中文输出。
- 复用 STORM 四阶段流程：

```text
research -> outline -> article -> polish
```

### 运行稳定性

- LLM query sanitizer：过滤空 query、Markdown、JSON、解释性句子。
- PIM 缩写消歧：将“PIM 神经网络抑制”指向 `passive intermodulation suppression`，过滤 `processing-in-memory / RAM / DRAM` 跑题结果。
- arXiv 单 query 失败降级，不中断主流程。
- 空检索结果防护，避免文章生成阶段崩溃。
- Wikipedia 辅助抓取增加 User-Agent、timeout 和结构检查。
- Windows UTF-8、surrogate、输出目录名、run_config 脱敏等修复。

### Agent Runtime 能力

- `paperstorm_trace.jsonl`：记录工具调用、耗时、结果数量、失败原因和产物路径。
- `run_summary.json`：记录一次运行摘要。
- `PaperStormTool`：统一工具 schema。
- `paperstorm_mcp_server.py`：MCP-style stdio server，支持 `tools/list` 和 `tools/call`。
- `paperstorm_eval.py`：规则版 Eval Harness，输出 `scorecard.json` 和 `scorecard.md`。

### v0.2：Memory / Context Compression / QA

- `PaperStormMemoryStore`：提供 working / episodic / semantic 三层记忆和用户偏好保存。
- `compress_context`：把长上下文压缩为结构化摘要，并检查期望关键词和禁止关键词。
- `PaperStormKnowledgeBase`：从一次 PaperStorm 运行目录加载文章和检索结果，支持基于证据的问答。
- `KnowledgeBaseQATool`：将知识库问答封装为标准 `PaperStormTool`，可被 MCP-style server 发现和调用。
- `PaperStormRuntimeSession`：轻量 runtime session，统一管理 tool registry、trace 写入和 working memory。
- `evaluate_qa_artifact`：对 `qa_answer.json` 进行规则版 QA 评估，检查引用、groundedness、关键词覆盖和跑题内容。

### v0.3：Runtime / Hook / Trace 标准化

- `ToolRegistry`：统一注册、查询、列出和校验 PaperStorm 工具。
- `HookManager`：支持 `before_tool_call`、`after_tool_call`、`on_tool_error`、`on_context_compress` 等生命周期 hook。
- `RuntimeEvent`：统一 runtime trace 字段，包括 `run_id`、`task_id`、`stage`、`tool`、`status`、`duration_sec`、`input_summary`、`output_summary`、`error`。
- `PaperStormRuntimeSession` v2：统一 tool registry、hook、trace、memory 和 context compression。
- MCP-style server 已接入 `ToolRegistry`，工具发现和工具调用共享同一套注册模型。

### v0.4：Multi-Agent 论文调研协作

- `PlannerAgent`：根据 topic 和关键词生成带 intent 的 query plan。
- `RetrieverAgent`：通过 `PaperStormRuntimeSession` 调用检索工具，并保留 query 来源。
- `CriticAgent`：基于 expected / forbidden keywords 判断结果保留或过滤，并写明理由。
- `MemoryAgent`：把保留和过滤发现写入 episodic memory。
- `EvaluatorAgent`：给 query plan、critic signal 和 agent trace 打分。
- `PaperStormResearchOrchestrator`：中心化编排多 Agent，输出 `agent_trace.jsonl` 和 `multi_agent_report.json`。

### v0.5：知识库平台化与服务 API

- `PaperStormTaskService`：文件存储版服务核心，管理 task_id、状态、output_dir、trace、summary、scorecard。
- 支持任务状态：`queued / running / succeeded / failed`。
- 支持读取文章、scorecard、trace。
- 支持基于 task artifacts 的知识库 QA。
- 支持 fake runner，不依赖真实 API key 也能测试服务链路。
- 失败任务输出结构化 error，并对 secret / key / token 字段脱敏。
- `paperstorm_service_api.py`：可选 FastAPI 适配器，提供任务提交、运行、状态查询、文章读取、scorecard、trace 和知识库 QA 路由。

### v0.5.1：并发、恢复与压测 baseline

- `PaperStormTaskService(max_concurrent_tasks=...)`：支持可配置并发上限。
- `worker_tick()`：从 queued 任务中按容量启动任务，避免超过并发上限。
- `complete_task()`：手动完成 running 任务，释放后续任务容量。
- `recover_stale_running_tasks()`：将超时 running 任务恢复为 failed，避免任务永久卡住。
- `run_stress_benchmark()`：使用 fake runner 生成压测报告，包含成功数、失败数、失败率、平均延迟、P95 latency、最大观察并发和 retry 次数。
- `benchmark_paperstorm_service.py`：命令行压测入口。

### v0.6：前端 Dashboard 与官方中文文档

- 新增静态 Dashboard：`frontend/paperstorm_dashboard/index.html`。
- Dashboard 展示任务状态、文章、知识库 QA、runtime trace、scorecard、multi-agent 结果和 stress report。
- 新增 `build_paperstorm_demo_bundle.py`，可生成 `sample_data.json` 供前端离线展示。
- 新增 `docs/STORM_OFFICIAL_CN.md`，把官方 STORM 架构和核心机制整理为中文说明。
- README 补充官方 STORM 架构图、两阶段流程图和 Co-STORM 工作流图。

### v0.7：真实 Pipeline Worker 接入

- 新增 `knowledge_storm/paperstorm_pipeline.py`，把真实 STORM pipeline 封装为 service 可调用的 worker。
- `PaperStormTaskService` 支持 `run_mode="paperstorm"`，同一套 task_id、状态、output_dir、article、trace、scorecard 读取接口可承载 fake 和真实任务。
- `PaperStormTaskService(..., pipeline_runner=...)` 支持 runner 注入，单元测试可用本地 runner 验证状态机，不依赖真实 LLM/API/网络。
- 新增 `run_paperstorm_service_task.py`，可从命令行提交并运行单个 service task。
- FastAPI 请求模型补充真实 worker 参数：LLM provider/model、检索器、PDF 目录、阶段开关、trace 开关等。
- 修复领域关键词误脱敏问题：`expected_keywords` / `forbidden_keywords` 不再因为字段名包含 `key` 被错误替换。

### v0.8：Dashboard 读取 Service 产物

- `PaperStormTaskService.get_dashboard_bundle(task_id)`：聚合 task、article、QA、scorecard、trace、pipeline worker 和 service snapshot。
- FastAPI 新增 `GET /research-tasks/{task_id}/dashboard`，前端可一次请求拿到真实任务展示数据。
- Dashboard 增加 Service URL、Task ID、加载真实任务和加载样例数据控件。
- Dashboard 新增 Pipeline Worker 面板，展示真实 worker 的 runner、retriever、LLM 和 score 元数据。
- 静态 `sample_data.json/js` 升级到 v0.8，离线打开 HTML 仍可展示完整样例。
- Demo bundle 递归清理本机临时路径，静态数据中的运行目录统一展示为 `demo://paperstorm_dashboard/...`。

### v0.9：端到端本地 Demo 与任务轮询

- `PaperStormTaskService.list_tasks(status=None)`：支持 Dashboard 获取全部任务或按状态过滤任务。
- FastAPI 新增 `GET /research-tasks`，返回任务列表；同时加入 CORS middleware，方便本地 HTML 访问服务。
- Dashboard 新增任务控制台：topic、run mode、retriever、output language、expected/forbidden keywords。
- Dashboard 支持提交任务、运行选中任务、轮询选中任务、刷新任务列表。
- Dashboard 支持展示结构化任务 error，使失败任务可观察、可复盘。
- 静态 `sample_data.json/js` 升级到 v0.9，离线展示和服务控制台共存。

### v1.0 Release Demo：可投递、可演示的 Agent 平台原型

- 新增 `build_release_demo()`，复用 `PaperStormTaskService` 的 submit -> run -> QA -> dashboard bundle 链路生成稳定演示数据。
- 新增 `run_paperstorm_release_demo.py`，一条命令生成 release summary、文章、trace、scorecard、QA 和前端 `sample_data.json/js`。
- Dashboard 样例数据可升级为 v1.0，展示 RAG、Memory、Runtime Trace、Eval、Task Service 和 Dashboard 的闭环。
- README、版本计划、简历面试文档补充 5 分钟演示路线和面试讲法。
- v1.0 明确为本地可演示平台原型，不夸大为多租户、分布式或生产部署系统。

### v1.1 Demo Runbook：本地演示链路打磨

- 新增 `start_paperstorm_service.py`，把 FastAPI service 启动命令固化为项目入口。
- README 补充本地演示 runbook：启动 service、生成 release demo、打开 Dashboard、提交/运行/轮询任务。
- 演示链路明确为 `submit -> queued -> running -> succeeded/failed -> artifacts -> trace/scorecard`。
- 面试文档补充“演示不是只给静态截图”，而是能展示 Agent 生命周期和可观测链路。

### v1.2 Final Packaging：最终包装与投递收口

- README 首页新增“项目一眼看懂”“最终能力地图”和 `Architecture Map`。
- README 明确最终演示命令和本地 service 生命周期演示主线。
- `docs/VERSION_PLAN.md` 标记项目后续进入维护和面试准备阶段。
- `docs/RESUME_INTERVIEW_PLAN.md` 整理最终简历 bullet 和最终面试 FAQ 精简版。
- 明确当前不建议继续堆版本，后续只围绕真实面试反馈和缺陷修复维护。

## 4. 关键文件

运行入口：

```text
examples/storm_examples/run_paper_storm_minimax.py
examples/storm_examples/paperstorm_mcp_server.py
examples/storm_examples/evaluate_paperstorm_run.py
examples/storm_examples/paperstorm_service_api.py
examples/storm_examples/benchmark_paperstorm_service.py
examples/storm_examples/build_paperstorm_demo_bundle.py
examples/storm_examples/run_paperstorm_service_task.py
examples/storm_examples/run_paperstorm_release_demo.py
examples/storm_examples/start_paperstorm_service.py
```

核心模块：

```text
knowledge_storm/rm.py
knowledge_storm/paperstorm_tools.py
knowledge_storm/paperstorm_eval.py
knowledge_storm/paperstorm_memory.py
knowledge_storm/paperstorm_qa.py
knowledge_storm/paperstorm_runtime.py
knowledge_storm/paperstorm_agents.py
knowledge_storm/paperstorm_service.py
knowledge_storm/paperstorm_demo.py
knowledge_storm/paperstorm_pipeline.py
knowledge_storm/paperstorm_release.py
knowledge_storm/paperstorm_rag.py
knowledge_storm/paperstorm_rag_benchmark.py
```

测试：

```text
tests/test_minimax_runtime_fixes.py
tests/test_paperstorm_retrievers.py
tests/test_paperstorm_logging.py
tests/test_paperstorm_mcp_server.py
tests/test_paperstorm_eval.py
tests/test_paperstorm_memory_qa.py
tests/test_paperstorm_runtime.py
tests/test_paperstorm_multi_agent.py
tests/test_paperstorm_service.py
tests/test_paperstorm_concurrency.py
tests/test_paperstorm_frontend_docs.py
tests/test_paperstorm_pipeline.py
tests/test_paperstorm_service_cli.py
tests/test_paperstorm_release_demo.py
tests/test_paperstorm_release_docs.py
tests/test_paperstorm_demo_runbook.py
tests/test_paperstorm_final_packaging.py
tests/test_paperstorm_rag_v3.py
```

维护文档：

```text
docs/OPERATION_GUIDE.md
docs/VERSION_PLAN.md
docs/RESUME_INTERVIEW_PLAN.md
docs/STORM_OFFICIAL_CN.md
```

前端：

```text
frontend/paperstorm_dashboard/index.html
frontend/paperstorm_dashboard/styles.css
frontend/paperstorm_dashboard/app.js
```

## 5. 环境

当前本地推荐解释器：

```text
D:\SOFTWARE\spyder\envs\storm\python.exe
```

不要用系统默认 `python` 直接运行，容易出现环境不一致。

## 6. 运行 PaperStorm

PowerShell 示例：

```powershell
cd D:\FILEEEEEEEEEEE\projects\storm

D:\SOFTWARE\spyder\envs\storm\python.exe examples\storm_examples\run_paper_storm_minimax.py `
  --topic "pim 神经网络抑制" `
  --retriever arxiv `
  --output-language zh `
  --output-dir ./results/paperstorm_zh `
  --llm-provider deepseek `
  --llm-model flash `
  --do-research `
  --do-generate-outline `
  --do-generate-article `
  --do-polish-article `
  --max-conv-turn 1 `
  --max-perspective 1 `
  --search-top-k 2 `
  --max-thread-num 1
```

常见输出：

```text
storm_gen_outline.txt
storm_gen_article.txt
storm_gen_article_polished.txt
raw_search_results.json
conversation_log.json
url_to_info.json
paperstorm_trace.jsonl
run_summary.json
```

最终可读文章通常是：

```text
storm_gen_article_polished.txt
```

## 7. 运行 Dashboard Demo

生成前端样例数据：

```powershell
D:\SOFTWARE\spyder\envs\storm\python.exe examples\storm_examples\build_paperstorm_demo_bundle.py `
  --output-dir frontend\paperstorm_dashboard
```

然后打开：

```text
frontend/paperstorm_dashboard/index.html
```

Dashboard 展示：

- 任务状态。
- 调研文章。
- 知识库 QA。
- Runtime trace。
- Eval scorecard。
- Multi-Agent 保留/过滤结果。
- Stress benchmark。
- Pipeline Worker 元数据。
- 任务创建、运行和轮询控制。

连接本地 service 查看真实任务：

1. 启动 `paperstorm_service_api.py` 对应的 FastAPI 服务。
2. 在 Dashboard 顶部输入 service URL，例如 `http://127.0.0.1:8000`。
3. 输入 task_id。
4. 点击“加载真实任务”。

本地端到端 fake demo：

1. 启动 FastAPI service。
2. 打开 `frontend/paperstorm_dashboard/index.html`。
3. 保持 Run Mode 为 `fake`。
4. 点击“提交任务”。
5. 点击“运行选中任务”。
6. 点击“轮询选中任务”查看 article、trace、scorecard 和 error。

## 8. v1.0 Release Demo

一条命令生成可演示产物：

```powershell
D:\SOFTWARE\spyder\envs\storm\python.exe examples\storm_examples\run_paperstorm_release_demo.py `
  --topic "pim 神经网络抑制" `
  --service-root ./results/paperstorm_release_demo `
  --dashboard-dir frontend\paperstorm_dashboard
```

它会生成：

```text
results/paperstorm_release_demo/release_demo_summary.json
results/paperstorm_release_demo/results/<task_id>/storm_gen_article_polished.txt
results/paperstorm_release_demo/results/<task_id>/paperstorm_trace.jsonl
results/paperstorm_release_demo/results/<task_id>/scorecard.json
frontend/paperstorm_dashboard/sample_data.json
frontend/paperstorm_dashboard/sample_data.js
```

### 5 分钟演示路线

1. 用 README 的官方 STORM 架构图说明原项目的 research -> outline -> article -> polish。
2. 运行 `run_paperstorm_release_demo.py`，展示同一套 service task 如何生成文章、QA、trace 和 scorecard。
3. 打开 `frontend/paperstorm_dashboard/index.html`，展示 task 状态、文章、知识库 QA、trace timeline 和 eval scorecard。
4. 解释 PIM 消歧：本项目把 PIM 指向 `passive intermodulation`，并过滤 `processing-in-memory / RAM / DRAM` 跑题内容。
5. 讲工程化增强：Tool Schema / MCP-style server、Memory、Context Compression、Multi-Agent、Task Service、Dashboard、Benchmark。
6. 主动说明边界：当前是本地平台原型，生产级多租户、权限、分布式队列和云部署还在后续计划。

## 9. v1.1 Demo Runbook

第一步，启动本地 service：

```powershell
D:\SOFTWARE\spyder\envs\storm\python.exe examples\storm_examples\start_paperstorm_service.py `
  --service-root ./results/paperstorm_demo_service `
  --host 127.0.0.1 `
  --port 8000
```

第二步，生成离线可展示数据：

```powershell
D:\SOFTWARE\spyder\envs\storm\python.exe examples\storm_examples\run_paperstorm_release_demo.py `
  --service-root ./results/paperstorm_release_demo `
  --dashboard-dir frontend\paperstorm_dashboard
```

第三步，打开 Dashboard：

```text
frontend/paperstorm_dashboard/index.html
```

第四步，在 Dashboard 顶部填入：

```text
Service URL: http://127.0.0.1:8000
Run Mode: fake
Topic: pim 神经网络抑制
Expected Keywords: passive intermodulation, RF
Forbidden Keywords: processing-in-memory, DRAM, RAM
```

第五步，按顺序点击：

```text
提交任务 -> 运行选中任务 -> 轮询选中任务 -> 查看 trace / scorecard / article / QA
```

这条链路对应 Agent 平台生命周期：

```text
submit -> queued -> running -> succeeded/failed -> artifacts -> trace/scorecard
```

面试演示重点：

- 不是只给静态截图，而是能展示一次 task 的状态变化和产物生成。
- fake 模式用于稳定演示，不消耗 API key。
- paperstorm 模式可以接真实 DeepSeek/arXiv，但受网络、余额和外部服务稳定性影响。
- Dashboard 展示的是 service 聚合后的 snapshot，前端不直接理解底层文件结构。

## 10. 通过 Service Worker 运行单个任务

fake 模式不需要真实 API key，适合验证 service 状态机：

```powershell
D:\SOFTWARE\spyder\envs\storm\python.exe examples\storm_examples\run_paperstorm_service_task.py `
  --topic "pim 神经网络抑制" `
  --run-mode fake `
  --output-dir ./results/paperstorm_service_demo `
  --expected-keyword "passive intermodulation" `
  --forbidden-keyword DRAM
```

真实 PaperStorm worker 示例：

```powershell
D:\SOFTWARE\spyder\envs\storm\python.exe examples\storm_examples\run_paperstorm_service_task.py `
  --topic "pim 神经网络抑制" `
  --run-mode paperstorm `
  --retriever arxiv `
  --output-language zh `
  --llm-provider deepseek `
  --llm-model flash `
  --output-dir ./results/paperstorm_service_real `
  --max-conv-turn 1 `
  --max-perspective 1 `
  --search-top-k 2 `
  --max-thread-num 1 `
  --expected-keyword "passive intermodulation" `
  --forbidden-keyword DRAM
```

真实 worker 会复用 service 的任务状态、产物目录、trace 和 scorecard 读取接口。真实模式需要可用网络和对应 LLM API key。

## 11. 运行 Eval Harness

示例：

```powershell
D:\SOFTWARE\spyder\envs\storm\python.exe examples\storm_examples\evaluate_paperstorm_run.py `
  --run-dir .\results\paperstorm_zh\PIM `
  --case-file examples\storm_examples\paperstorm_eval_cases.json `
  --topic "pim 神经网络抑制"
```

输出：

```text
scorecard.json
scorecard.md
```

当前评分维度：

- 任务完成度。
- 检索相关性。
- 跑题惩罚。
- 文章质量。
- Runtime 可观测性。

## 12. 运行 MCP-style Server

手工 `tools/list` 验证：

```powershell
'{' + '"jsonrpc":"2.0","id":1,"method":"tools/list"' + '}' |
  D:\SOFTWARE\spyder\envs\storm\python.exe examples\storm_examples\paperstorm_mcp_server.py
```

当前工具：

```text
arxiv_search
local_pdf_search
kb_qa
research_qa
```

其中 `local_pdf_search` 需要传入 `--pdf-dir` 后启用。

## 13. 测试

推荐回归测试：

```powershell
D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest `
  tests.test_paperstorm_frontend_docs `
  tests.test_paperstorm_concurrency `
  tests.test_paperstorm_service `
  tests.test_paperstorm_pipeline `
  tests.test_paperstorm_service_cli `
  tests.test_paperstorm_release_demo `
  tests.test_paperstorm_release_docs `
  tests.test_paperstorm_demo_runbook `
  tests.test_paperstorm_final_packaging `
  tests.test_paperstorm_multi_agent `
  tests.test_paperstorm_runtime `
  tests.test_paperstorm_memory_qa `
  tests.test_paperstorm_langgraph_v44 `
  tests.test_paperstorm_eval `
  tests.test_paperstorm_eval_v4 `
  tests.test_paperstorm_retrieval_v41 `
  tests.test_paperstorm_context_v42 `
  tests.test_paperstorm_mcp_server `
  tests.test_paperstorm_logging `
  tests.test_paperstorm_retrievers `
  tests.test_minimax_runtime_fixes -v
```

最近目标结果以当前回归输出为准，例如：

```text
Ran N tests
OK
```

语法检查：

```powershell
D:\SOFTWARE\spyder\envs\storm\python.exe -m py_compile `
  knowledge_storm\paperstorm_eval.py `
  knowledge_storm\paperstorm_tools.py `
  knowledge_storm\paperstorm_demo.py `
  knowledge_storm\paperstorm_service.py `
  knowledge_storm\paperstorm_pipeline.py `
  knowledge_storm\paperstorm_release.py `
  examples\storm_examples\evaluate_paperstorm_run.py `
  examples\storm_examples\build_paperstorm_demo_bundle.py `
  examples\storm_examples\run_paperstorm_service_task.py `
  examples\storm_examples\run_paperstorm_release_demo.py `
  examples\storm_examples\start_paperstorm_service.py `
  examples\storm_examples\paperstorm_service_api.py `
  examples\storm_examples\paperstorm_mcp_server.py
```

## 14. 后续版本路线

详见：

```text
docs/VERSION_PLAN.md
```

当前建议路线：

- v3.2（已完成）：企业知识库 Agent 本地 baseline，打通文档建库、问答、引用、API 与 Dashboard。
- v4.0（已完成）：建立可审计种子集、坏例工作台和检索/生成分层评测。
- v4.1（已完成）：真实 BM25 + Dense + RRF + Cross-Encoder、Contextual Chunk 和 Zotero 论文弱标注实验。
- v4.2（已完成）：可恢复 Context Engine、Token 动态预算、分层压缩、Context Meter 和恢复 Benchmark。
- v4.3（已完成）：可治理的跨会话 Memory Service、写入策略、冲突失效、Hybrid recall、治理 API 和 Memory Benchmark。
- v4.4（已完成）：LangGraph 状态图、SQLite checkpoint、节点重试、请求幂等、STORM 隔离工具、图调试 API 与 Runtime Benchmark。
- v4.5（已完成）：生产控制面、资源 ACL/审计、事务幂等、增量索引任务、缓存失效、重试/熔断、统一 Trace 和 SLO Benchmark。

第三阶段要求每个版本同时交付代码、测试、Benchmark、Trace 和面试学习记录；未完成的真实 Embedding、向量库、Reranker 和 LangGraph 能力不会提前写入简历。

## 15. 求职与面试材料

详见：

```text
docs/RESUME_INTERVIEW_PLAN.md
```

项目可覆盖的面试关键词：

- Agentic Loop。
- RAG 全流程。
- Tool Calling。
- MCP。
- Memory。
- Multi-Agent。
- Runtime Trace。
- Eval / Benchmark。
- 错误容灾。
- 结构化技术文档。

## 16. 当前边界

已经完成：

- 本地命令行 Agent pipeline。
- RAG 检索与中文报告生成。
- Tool Schema / MCP-style server。
- Runtime Trace。
- Eval Harness v1。
- Memory / Context Compression / QA。
- Multi-Agent 调研编排。
- 文件存储版 Task Service。
- 静态 Dashboard。
- 真实 PaperStorm pipeline worker 接口。
- Dashboard 轻量读取真实 service task 产物。
- Dashboard 提交、运行和轮询本地 service task。
- v1.0 release demo 一键生成本地演示产物和前端样例数据。
- v1.1 demo runbook 固化本地 service、Dashboard 和任务生命周期演示步骤。
- v1.2 final packaging 完成 GitHub 首页、能力地图、最终演示命令和求职材料收口。
- v4.5 本地生产治理基线：资源 ACL/审计、事务幂等、TTL cache、增量索引任务、重试/熔断、统一 Trace 与 SLO Benchmark。

尚未完成：

- 生产级 API 网关。
- 分布式高并发任务队列。
- 企业级监控告警。
- 生产级前端构建、鉴权、自动轮询调度和部署。
- 真实 LLM/API 环境下的大规模压测。
- OAuth/OIDC、细粒度 RBAC/ABAC policy service 与密钥托管。
- PostgreSQL/Redis/分布式 worker、强制异步取消和 OpenTelemetry Collector。

这些内容会按版本计划逐步补齐。
