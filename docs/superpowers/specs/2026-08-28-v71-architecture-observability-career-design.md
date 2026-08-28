# PaperStorm v7.1 架构、可观测性与求职材料设计

## 目标

将 PaperStorm v7.0 的系统能力整理成可编辑架构资产、可运行的 Langfuse Bad Case 演示、
可用于简历和面试的项目叙事，以及可运行的双 Agent RAG 面试模拟器，并发布为 v7.1。

## 交付范围

### 架构资产

- 更新 `paperstorm-agent-system-flow.drawio`，纳入领域评测、Langfuse、异步任务与 SSE。
- 更新 `paperstorm-executive-overview.drawio`，保持适合项目汇报的主流程视图。
- 新增 `paperstorm-async-runtime-sequence.drawio`，描述浏览器、FastAPI、任务队列、Agent Runtime、
  Retriever、LLM、Checkpoint、SSE 和 Langfuse 的异步交互。
- 每张图同时生成 SVG，README 使用 SVG，Draw.io 作为可编辑源文件。

### Langfuse Bad Case 演示

- 新增独立 CLI，从固定示例或 JSON 输入读取一次 RAG 运行。
- 生成根 Trace，并建立 route、retrieve、rerank、context、reader、citation_validate 等 Span。
- 写入 retrieval recall、citation validity、answer groundedness、latency 等 Score。
- 对漏召回、非法引用、证据冲突、拒答错误进行结构化标签，而不是依赖日志全文搜索。
- 未配置 Langfuse 时仍生成本地 JSONL 演示；配置密钥后发送到 Langfuse Cloud。
- README 给出环境变量、运行命令、筛选 Bad Case 和定位根因的步骤。

### 简历材料

- 新增一份正式中文指南，包含 STAR 背景故事、3–5 条可直接使用的简历 bullet、
  60 秒与 3 分钟自我介绍、数字可信边界，以及典型追问的证据链。
- 明确区分公开 Benchmark、私有领域 Pilot、离线治理测试，避免夸大指标。

### 双 Agent 面试模拟器

- Interviewer Agent 根据题库、上一轮回答和覆盖状态提出问题及追问。
- Candidate Agent 结合 PaperStorm 真实架构资料回答，不读取标准答案字段。
- 题库覆盖 RAG 基础、Chunk、Embedding、Hybrid、Rerank、Memory、Context、Runtime、
  Multi-Agent、Langfuse、Benchmark、稳定性、项目难点和成就追问。
- 支持 deterministic 模式用于测试，支持 LLM 模式用于真实模拟，输出 Markdown 会话记录。
- 新增完整面试手册，给出问题、参考答案、追问、评价点和常见失分点。

## 架构边界

- 不新增 Web UI 页面；本版本聚焦可复现 CLI、架构资产和文档。
- 不把私有论文、API Key、Langfuse Key 或本机绝对路径提交到 Git。
- 不把 Langfuse 当作业务数据库；远程上报失败不得阻断 Agent 主链。
- 不用单一总分掩盖 Bad Case，Trace、Span、Score、标签和原始输入输出必须可关联。
- 面试模拟器不承担招聘决策，只用于项目知识复习与模拟追问。

## 验收

1. Draw.io XML 可被解析，所有边连接到存在的节点，SVG 可正常显示。
2. Langfuse 演示在无密钥环境生成本地 Trace，在 mock SDK 下验证 Trace、Span、Score 合同。
3. 双 Agent 模拟器 deterministic 测试稳定，LLM 输出解析失败时有明确降级与错误记录。
4. README 可从架构图进入可编辑源文件、Langfuse 演示、简历指南和面试手册。
5. 全量 Python 测试、文档契约测试、`compileall` 和 `git diff --check` 通过。
6. `main` 与 `version/v7.1` 指向同一发布提交。

## 发布内容

- 包、FastAPI、Trace 和前端发布标识统一为 `7.1.0` / `v7.1`。
- README 只展示最新版本，历史协议保留在对应评测文档中。
- 提交前扫描新增内容中的真实格式凭据。
