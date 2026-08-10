# PaperStorm 产物清单（Artifact Inventory）

> 本清单回答两件事：**每个产物是什么、有什么用**，以及**在 service 网页端（Dashboard）点哪个按钮才会产生它**。
> 文档按"网页操作 → 产物"组织；API / CLI 作为备用触发方式标注。路径中的 `<service_root>` = 启动服务时 `--service-root` 指定的目录。
> 第 7 节是追问详解（FAQ），与正文互相引用，会随追问持续补充。

---

## 0. 总览：在网页端点哪个按钮 → 产生什么

> 先决操作：`start_paperstorm_service.py --service-root <service_root>` 启动服务，浏览器打开 `frontend/paperstorm_dashboard/index.html`，顶部填 `http://127.0.0.1:8000`。

| 你在 Dashboard 点什么 | 产生的产物（相对 `<service_root>`） |
| --- | --- |
| 任务控制台：填 Topic / 运行模式 / 关键词 → **提交任务** | `tasks/<task_id>.json`（任务状态元数据） |
| 任务控制台：**运行选中任务** | `results/<task_id>/*`（任务全套产物，含 scorecard，**运行结束自动写**） |
| 任务控制台：**轮询选中任务** | 读取以上产物展示（不产生新文件） |
| 聊天问答：**新建聊天** | `chat_sessions/<chat_id>.json` |
| 聊天问答：输入消息 → **发送** | 更新会话 .json + 追加 `chat_sessions/<chat_id>.context.jsonl` |
| 聊天问答：**强制压缩** / **恢复原始视图** | 追加压缩/恢复事件到 `.context.jsonl` |
| 聊天问答：**查询长期记忆** / **软删除记忆** / **导出记忆** | 读 / 追加 `memory_service_v43/memory_events.jsonl` |
| 聊天问答：发消息自动触发记忆召回与写入 | 同上 |
| 企业知识库：**创建知识库** | `knowledge_bases/<kb_id>/*`（manifest / 索引） |
| 企业知识库：**知识库问答** | 返回带引用答案（写入 KB 目录；任务内 `qa_answer.json` 来自任务知识库问答 `POST /knowledge-bases/{task_id}/query`） |
| 评测区：**运行 v4.0 基线** | `evaluations/rag_v4_latest/*` |
| 评测区：**运行八组 Smoke 实验** | `evaluations/rag_v41_latest/*` |
| 评测区：**运行 Context Benchmark** | `evaluations/context_v42_latest/*` |
| 评测区：**运行 Memory Benchmark** | `evaluations/memory_v43_latest/*` |
| 评测区：**运行 Runtime Benchmark** | `evaluations/runtime_v44_latest/*` |
| 评测区：**运行 Production Benchmark** | `evaluations/production_v45_latest/*` |
| 聊天问答：**刷新图状态与 Checkpoint** | 读 `production_runtime_v45/langgraph_v44/{checkpoints.sqlite, request_results/, traces/}` |
| 生产控制面：**刷新控制面状态** / **加载当前 Trace** | 读 `production_control_v45.sqlite` |
| 每次启动服务 | `server.stdout.log`、`server.stderr.log` |

---

## 1. 任务产物（`results/<task_id>/`）

### 触发方式（网页端为主线）
1. 启动 service → 打开 Dashboard → 切到 **"调研写文章"** 模式
2. 填 **Topic**、选 **运行模式**（`fake`=本地确定性样例 / `paperstorm`=真实调 DeepSeek+arXiv）、填期望/禁止关键词
3. 点 **提交任务** → 点 **运行选中任务** → 点 **轮询选中任务** 查看全部产物

> API 备用：`POST /research-tasks` → `POST /research-tasks/{id}/run`；CLI 备用：`run_paperstorm_service_task.py`。
> 运行模式决定部分产物是否有内容：`fake` 才有 `reflection.txt`；`llm_call_history.jsonl` 只有真实调 LLM 的运行才有内容。

### 产物明细

| 文件 | 作用 | 生成来源 / 时机 |
| --- | --- | --- |
| `storm_gen_outline.txt` | 文章**大纲** | 任务运行时，STORM research 阶段后由 LLM 生成 |
| `storm_gen_article.txt` | 文章**初稿** | article 生成阶段 |
| `storm_gen_article_polished.txt` | **润色后最终文章**（README 说的"最终可读文章"） | polish 阶段 |
| `direct_gen_outline.txt` | 直接生成的大纲（绕过多轮对话，用 LLM 参数知识） | 官方 STORM engine 旁路产物 |
| `conversation_log.json` | **模拟对话记录**：writer ↔ topic expert 多轮问答，文章素材来源 | STORM 核心机制，research 阶段 |
| `raw_search_results.json` | 原始检索结果 | 检索工具执行后 |
| `url_to_info.json` | **引用索引**：`url_to_unified_index`（URL→编号）+ `url_to_info`（URL→title/snippets）。**文章生成阶段 ArticleGen 读它写 `[1][2]` 引用**，demo 用它构造参考文献。即"检索对话的产出、文章引用的输入" | research 阶段汇总对话里的 URL 信息后写 |
| `llm_call_history.jsonl` | **每次真实 LLM 调用的完整日志**。token 数在每行 `usage.prompt_tokens / completion_tokens / total_tokens`，费用在顶层 `cost`（美元）。**注意每行是一个超长 JSON**（JSONL），编辑器里会截断，用命令看（见 7.2） | 每次调 LLM 追加写 |
| `run_config.json` | 本次运行各阶段 LLM 配置（temperature、max_tokens、api_base），key 已脱敏。**temperature 是代码写死的 1.0**（pipeline:206），不是 LLM 决定的，这里只是记录 | 运行开始时由 STORM engine 写 |
| `paperstorm_trace.jsonl` | **Runtime 事件流**。一次运行流程：`run_start` → 每个检索 query（`tool_start → retrieval_start → retrieval_end → tool_end`，有结果时接 `artifact_written`）→ `run_end`。Dashboard trace timeline 逐条渲染 | 运行全程，runtime session 逐事件追加 |
| `run_summary.json` | 运行摘要：success、耗时、事件数、检索统计、产物列表 | 运行结束时 |
| `scorecard.json` / `scorecard.md` | v1 规则打分（total + 各维度分 + notes） | **运行结束自动写**（fake 和 pipeline 都会，不用单独操作） |
| `qa_answer.json` | **最新一轮**问答答案（answer + citations + evidence + grounded），**覆盖式** | 之后对任务做知识库问答 `POST /knowledge-bases/{task_id}/query` |
| `qa_history.json` | **多轮追问累积**记录（保留最近 5 轮），**追加式** | Research QA 流程 `POST /research-agent/ask` |
| `pipeline_worker.json` | **worker 身份标签**：runner / retriever / llm_provider / llm_model / score。Dashboard "Pipeline Worker 面板"读它展示"谁跑的、多少分" | 任务结束（scorecard 算完）时写 |
| `reflection.txt` | Critic 对检索结果的反思（过滤/保留方向），fake 专属**确定性占位** | 仅 fake 模式写（真实 pipeline 不写） |

---

## 2. 评测产物（`evaluations/`）

### 触发方式（网页端为主线）
Dashboard 底部有六个评测区，各自一个"运行"按钮，点一次就把**该类评测最新一次结果**覆盖写进对应的 `xxx_latest` 文件夹（所以是"最近一次快照"）。"加载最近报告"按钮是读取展示，不产生新文件。

| Dashboard 按钮 | 子文件夹 | 里面的文件 |
| --- | --- | --- |
| 运行 v4.0 基线 | `evaluations/rag_v4_latest/` | `rag_eval_v4_report.json/.md`、`rag_eval_v4_bad_cases.jsonl` |
| 运行八组 Smoke 实验 | `evaluations/rag_v41_latest/` | `rag_eval_v41_ablation.json/.md` |
| 运行 Context Benchmark | `evaluations/context_v42_latest/` | `context_benchmark_v42.json/.md` + `runs/<uuid>/context_events.jsonl` |
| 运行 Memory Benchmark | `evaluations/memory_v43_latest/` | `memory_benchmark_v43.json/.md` |
| 运行 Runtime Benchmark | `evaluations/runtime_v44_latest/` | `langgraph_benchmark_v44.json/.md` |
| 运行 Production Benchmark | `evaluations/production_v45_latest/` | `production_benchmark_v45.json/.md` |

> 备用触发：API `POST /evaluations/{rag-v4 | rag-v41 | context-v42 | memory-v43 | runtime-v44 | production-v45}`；或 CLI / `python -c` 直接跑对应 `run_*_benchmark` 写到你自定的目录（与网页端落盘位置相互独立）。
> 注意：网页的 v4.1 按钮是**确定性 smoke**（快速查链路）；真实多语种 Dense / Cross-Encoder / Zotero 实验要跑 V4.1 CLI。
> 每个报告文件含全部指标（recall@K、MRR、nDCG、pass_rate、延迟分位等，指标含义见 README 各版本小节）。

---

## 3. 聊天会话产物（`chat_sessions/`）

### 触发方式（网页端为主线）
切到 **"聊天问答"** 模式：
- 点 **新建聊天** → 写 `chat_sessions/<chat_id>.json`
- 输入消息 → 点 **发送** → 更新 .json + 追加 `<chat_id>.context.jsonl`
- 点 **强制压缩** / **恢复原始视图** → 追加压缩/恢复事件到 `.context.jsonl`
- 点 **刷新上下文状态** → 读取展示

> API 备用：`POST /chat/sessions`、`POST /chat/sessions/{id}/messages`、`POST /chat/sessions/{id}/context/compact`、`POST .../context/restore`。

### 产物明细
| 文件 | 作用 | 生成来源 |
| --- | --- | --- |
| `<chat_id>.json` | 一个会话的**状态快照**（messages、context_config、memory 命中、task_id 等），创建时写、发消息时**覆盖** | 新建聊天 / 发送消息 |
| `<chat_id>.context.jsonl` | 该会话的**追加式上下文事件流**（ContextEventStore）：每条消息、压缩、恢复事件，带 `sequence` 序号 | 每次发消息 / 压缩 / 恢复 |

**FAQ 要点（见 7.8、7.12）**：`<chat_id>.context.jsonl` 是**按会话隔离**的——每个会话有自己的文件，不会把别的会话的数据混进来。目录里"两个 json + 一个 context"是**两个不同会话**：一个会话发过消息所以有 .json + .context.jsonl，另一个只创建没发消息，只有 .json。**没有 context 文件 = 那个会话没发过任何消息**。

---

## 4. 记忆产物（`memory_service_v43/`）

### 触发方式（网页端为主线）
聊天问答模式下：启用"跨会话长期记忆"后，**发送消息本身**就会触发记忆召回（回答前）和记忆写入决策（回答后）。另有操作按钮：
- **查询长期记忆** → 读事件文件做召回展示
- **软删除记忆** → 追加一条状态变更事件
- **导出记忆与审计记录** → 读事件文件导出

> API 备用：`POST/GET /memories`、`POST /memories/search`、`PATCH/DELETE /memories/{id}`。

### 产物明细
| 文件 | 作用 | 生成来源 |
| --- | --- | --- |
| `memory_events.jsonl` | **跨会话长期记忆 + 审计事件**：写入决策、验证、冲突失效（supersede）、设置变更、consolidation 等 | 记忆写入 / 状态变更 / 设置变更时追加 |
| `memory_candidates.jsonl` | 低置信度候选（后台整理队列） | 置信度 < 0.85 的候选排队时写 |

**FAQ 要点（见 7.9）**：`_state()` 在**每次查询时重放整个事件文件**重建记忆状态，所以：
- **跨同一起动 service 的会话**：能跨——记忆存在磁盘上，每次读都从事件文件重建，不需要手动加载
- **跨不同启动（同一个 service-root 重启）**：能跨——数据在磁盘
- **跨不同 service-root**：**不能**——每个 service-root 有自己独立的 `memory_service_v43/`
- 召回是**被动自动**的：聊天时系统自动 `search()`，不用手动加载什么

---

## 5. 图 / 生产运行时产物

### 触发方式（网页端为主线）
聊天问答模式下**发送消息**会调用 LangGraph（`POST /conversation-graph/invoke` 逻辑），点 **刷新图状态与 Checkpoint** 可查看 graph run、executed nodes、state、checkpoint history。生产控制面按钮 **刷新控制面状态** / **加载当前 Trace** 读取治理数据。

### 产物明细
| 文件 | 作用 | 生成来源 |
| --- | --- | --- |
| `production_runtime_v45/langgraph_v44/checkpoints.sqlite` | LangGraph 线程 checkpoint（会话状态 + 历史，**重启可恢复**） | 每次图 invoke 由 SqliteSaver 写 |
| `production_runtime_v45/langgraph_v44/request_results/<hash>.json` | 每次图调用（thread_id + request_id 唯一键）的结果快照，**幂等重放依据** | 每次 invoke 写 |
| `production_runtime_v45/langgraph_v44/traces/<hash>.jsonl` | 每次图调用的节点 span trace | 每次 invoke 写 |
| `production_control_v45.sqlite` | v4.5 生产控制面数据：ACL、事务幂等、TTL 缓存、持久任务、熔断、trace/span 存储 | 生产控制面操作时写 |

**FAQ 要点（见 7.10、7.11）**：
- **request_id**：调用 `/conversation-graph/invoke` 时请求里的字段，和 thread_id 组成幂等键（`request_results/<digest(thread\0request)>.json`）。相同键重放会直接返回缓存并标 `idempotent_replay=true`，不重复执行。
- **span_id**：每个图节点执行时生成一个（uuid4），节点事件 `graph_node_started/failed/succeeded` 带 `span_id`，是"trace span 覆盖率"指标的数据来源。
- **runtime**：节点事件里固定标 `runtime="langgraph-v4.4"`（v4.5 控制面外层运行时仍如实报告底层是 langgraph-v4.4）。
- **checkpoint 恢复原理**：每次 invoke 都 `sqlite3.connect(checkpoints.sqlite)` 重新打开（`_ensure_open`），`thread_id` 相同就能从 checkpoint 续上上次状态。**"重启" = 用同一个 service-root 重启 service**，因为 sqlite 在磁盘上，恢复的是同一个文件。
- **LangGraph 流向**：
```
START → classify（意图分类）
      → memory_recall（召回长期记忆）
          ├→ casual_chat / memory_answer / refuse_or_clarify（不检索直接回）
          └→ knowledge_retrieval → evidence_grade（证据分级）
                ├→ answer_with_citations（证据够，回答+引用）
                ├→ deep_research（STORM 深调研，隔离工具）→ answer_with_citations
                └→ refuse_or_clarify（不够，拒绝/澄清）
      → memory_candidate_write（回答后记忆写入决策）
      → final_trace（统一 trace）→ END
```
`knowledge_retrieval` 和 `deep_research` 两个节点带重试策略（ConnectionError/TimeoutError，最多 2 次）。

---

## 6. 打包与服务产物

| 文件 | 作用 | 生成来源 |
| --- | --- | --- |
| `release_demo_summary.json` | 一键演示汇总（任务、文章、QA、scorecard、bundle 聚合） | CLI：`run_paperstorm_release_demo.py` |
| `frontend/paperstorm_dashboard/sample_data.json` / `.js` | 前端**离线**展示的静态样例数据 | CLI：`run_paperstorm_release_demo.py` 或 `build_paperstorm_demo_bundle.py` |
| `<service_root>/server.stdout.log` / `server.stderr.log` | uvicorn 服务启动日志 | 每次 `start_paperstorm_service.py` 启动 |

---

## 7. 追问详解（FAQ）

> 按 Master 追问持续补充。以下每条都已对照代码核实。

### 7.1 `url_to_info.json` 是谁读取并写进文章的？
两层索引：`url_to_unified_index`（URL→编号，生成 `[1][2]` 引用） + `url_to_info`（URL→title/snippets）。生成于 research 阶段（汇总对话收集的 URL 信息）；**文章生成阶段 ArticleGen** 读取它构造正文引用和参考文献。

### 7.2 `llm_call_history.jsonl` 里 token 和 cost 在哪？
每行是一个超长 JSON（JSONL），编辑器一行会截断。token：`usage.prompt_tokens / completion_tokens / total_tokens`；费用：顶层 `cost`（美元）。查看命令：
```powershell
python -c "import json; d=[json.loads(l) for l in open(r'<路径>\llm_call_history.jsonl',encoding='utf-8')]; print(d[0]['usage']); print(d[0]['cost'])"
```

### 7.3 `run_config.json` 的 temperature 是 LLM 决定的吗？
不是，代码写死的：`paperstorm_pipeline.py:206` 统一 `temperature=1.0`。temperature 是采样随机性参数（0=确定），由调用方设定，模型不参与、运行中不变。该文件只是记录。

### 7.4 `paperstorm_trace.jsonl` 的流程
`run_start` → 每个检索 query：`tool_start → retrieval_start → retrieval_end → tool_end`（有结果接 `artifact_written`）→ `run_end`。字段：`ts` + `event` + 业务上下文，由 `RuntimeEvent` 统一格式写出，Dashboard timeline 渲染。

### 7.5 `qa_answer.json` vs `qa_history.json`
| | `qa_answer.json` | `qa_history.json` |
| --- | --- | --- |
| 内容 | **最新一轮**答案（answer+citations+evidence+grounded） | **多轮追问累积**（保留最近 5 轮） |
| 写入 | 覆盖式 | 追加式 |
| 来源 | 知识库问答 `POST /knowledge-bases/{task_id}/query`（`write_qa_artifact`） | Research QA `POST /research-agent/ask`（`_append_qa_history`） |

### 7.6 `pipeline_worker.json` 的作用
写于 `paperstorm_pipeline.py:261`，任务结束（scorecard 算完）时落盘。是 service 给任务的**身份标签**：runner / retriever / llm_provider / llm_model / score。Dashboard "Pipeline Worker 面板"读它展示"谁跑的、多少分"。

### 7.7 为什么 `reflection.txt` 只有 fake 模式写
是 fake 内置生成器**刻意造的确定性占位**（`paperstorm_service.py:748`），让 fake 演示也有"Critic 反思"味道。真实 pipeline（paperstorm_pipeline.py）**没有写 reflection 的步骤**，真实 Critic 输出在 agent trace / 记忆里。

### 7.8 `chat_sessions/` 三个文件的区别
实际是**两个 `.json`（两个不同会话）+ 一个 `.context.jsonl`**。`.json` 是会话**状态快照**（创建时写、发消息覆盖）；`.context.jsonl` 是**追加式上下文事件流**（带 `sequence` 序号）。没有 context 文件 = 该会话从未发过消息。

### 7.9 长期记忆能跨启动吗？要手动加载吗？
`memory_service_v43` 挂在某个 service-root 下；`_state()` **每次查询时重放 `memory_events.jsonl`** 重建记忆。所以：
- 跨**同 service-root 重启**：能跨（数据在磁盘）
- 跨**不同 service-root**：不能（各自独立目录）
- 召回是**被动自动**的：聊天回答前自动 `search()`，无需手动加载

### 7.10 checkpoint 怎么恢复？"重启"指什么？
每次 invoke 用 `sqlite3.connect(checkpoints.sqlite)` 重开连接（`_ensure_open`），`thread_id` 相同就从 checkpoint 续上。**"重启" = 用同一个 service-root 重启 service**——sqlite 在磁盘上，恢复的是同一份文件。对应网页按钮：**刷新图状态与 Checkpoint**。

### 7.11 LangGraph 在这里如何体现？request_id / span_id / runtime 是什么？
- `request_id`：请求字段，与 thread_id 组成幂等键（`request_results/<digest>`）。重放相同键直接返回缓存（`idempotent_replay=true`）。
- `span_id`：每节点执行生成 uuid4，节点事件带 `span_id`，是 trace span 覆盖率指标的数据来源。
- `runtime`：节点事件固定标 `runtime="langgraph-v4.4"`（v4.5 外层运行时如实报告底层 runtime）。
- 流向图见第 5 节。

### 7.12 `<chat_id>.context.jsonl` 会把不同会话都写进去吗？
不会。每个会话有**自己的** `<chat_id>.context.jsonl`（`ContextEventStore(self.chat_dir / "<chat_id>.context.jsonl")`）。会话数据按 id 隔离，不会混写。

### 7.13 `evaluations/` 里的东西有介绍吗？
有。第 2 节列出了每个 `xxx_latest` 子文件夹、里面的报告文件、对应的网页按钮；指标含义见 README 各版本小节和项目内指标说明。网页只跑确定性 smoke，真实模型实验走 CLI。

---

## 8. 待补充（追问区）

> 此处按 Master 的追问持续补充。
