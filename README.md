# PaperStorm Agent

> 基于 Stanford STORM 二次开发的中文论文调研 Agent，面向 RAG、Memory、Tool Calling、MCP、Multi-Agent 与 Agent Eval 持续演进。

本仓库原始项目来自 Stanford STORM。官方 README 已保留在：

```text
README_STORM_OFFICIAL.md
```

当前 README 记录本 fork 的中文项目定位、运行方式、版本计划和求职展示重点。

## 1. 项目定位

PaperStorm Agent 的目标不是从零重写 STORM，而是在成熟 Deep Research / RAG 框架上做面向 Agent 开发岗的工程化改造：

- 中文论文调研报告生成。
- arXiv / 本地 PDF 检索。
- LLM query 清洗与领域消歧。
- Runtime Trace / Hook。
- Tool Schema 与 MCP-style 工具入口。
- Eval Harness 量化评估 Agent 运行质量。
- 后续补 Memory、Multi-Agent、知识库服务化和前端展示。

与另一个项目 `nonlinear-nn-agent` 的分工：

- `nonlinear-nn-agent`：从零实现轻量 Agent Harness Runtime，突出 ToolRegistry、Hook、Session、Trace、Async、Retry。
- `PaperStorm Agent`：基于成熟 RAG/Deep Research 框架二次开发，突出 RAG、Memory、MCP、Multi-Agent、Eval、知识库与前端展示。

## 2. 当前已完成能力

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

## 3. 关键文件

运行入口：

```text
examples/storm_examples/run_paper_storm_minimax.py
examples/storm_examples/paperstorm_mcp_server.py
examples/storm_examples/evaluate_paperstorm_run.py
```

核心模块：

```text
knowledge_storm/rm.py
knowledge_storm/paperstorm_tools.py
knowledge_storm/paperstorm_eval.py
```

测试：

```text
tests/test_minimax_runtime_fixes.py
tests/test_paperstorm_retrievers.py
tests/test_paperstorm_logging.py
tests/test_paperstorm_mcp_server.py
tests/test_paperstorm_eval.py
```

维护文档：

```text
docs/OPERATION_GUIDE.md
docs/VERSION_PLAN.md
docs/RESUME_INTERVIEW_PLAN.md
```

## 4. 环境

当前本地推荐解释器：

```text
D:\SOFTWARE\spyder\envs\storm\python.exe
```

不要用系统默认 `python` 直接运行，容易出现环境不一致。

## 5. 运行 PaperStorm

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

## 6. 运行 Eval Harness

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

## 7. 运行 MCP-style Server

手工 `tools/list` 验证：

```powershell
'{' + '"jsonrpc":"2.0","id":1,"method":"tools/list"' + '}' |
  D:\SOFTWARE\spyder\envs\storm\python.exe examples\storm_examples\paperstorm_mcp_server.py
```

当前工具：

```text
arxiv_search
local_pdf_search
```

其中 `local_pdf_search` 需要传入 `--pdf-dir` 后启用。

## 8. 测试

推荐回归测试：

```powershell
D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest `
  tests.test_paperstorm_eval `
  tests.test_paperstorm_mcp_server `
  tests.test_paperstorm_logging `
  tests.test_paperstorm_retrievers `
  tests.test_minimax_runtime_fixes -v
```

最近目标结果：

```text
Ran 38 tests
OK
```

语法检查：

```powershell
D:\SOFTWARE\spyder\envs\storm\python.exe -m py_compile `
  knowledge_storm\paperstorm_eval.py `
  knowledge_storm\paperstorm_tools.py `
  examples\storm_examples\evaluate_paperstorm_run.py `
  examples\storm_examples\paperstorm_mcp_server.py
```

## 9. 后续版本路线

详见：

```text
docs/VERSION_PLAN.md
```

当前建议路线：

- v0.2：RAG 质量与 Memory 模块。
- v0.3：Multi-Agent 论文调研协作。
- v0.4：知识库平台化与 FastAPI 服务。
- v0.5：前端展示 Demo。
- v1.0：可投递、可演示的 Agent 平台化 Demo。

## 10. 求职与面试材料

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

## 11. 当前边界

已经完成：

- 本地命令行 Agent pipeline。
- RAG 检索与中文报告生成。
- Tool Schema / MCP-style server。
- Runtime Trace。
- Eval Harness v1。

尚未完成：

- 生产级 API 网关。
- 多用户和权限系统。
- 高并发任务队列。
- 企业级监控告警。
- 完整 Memory Store。
- 真正 Multi-Agent 编排。
- 前端展示。

这些内容会按版本计划逐步补齐。
