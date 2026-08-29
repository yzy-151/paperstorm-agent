# arXiv Reference Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让真实 Web API 对“Muon 优化器”稳定召回相关 arXiv 论文、生成足量调研内容，并在文章、PDF 与问答中交付可点击原文引用。

**Architecture:** 在 `ArxivRM` 内增加独立查询编译与相关性过滤；在服务交付边界增加统一 Reference Renderer；调研 Profile 由 API 参数贯穿 Pipeline。所有修复先写失败测试，再最小实现。

**Tech Stack:** Python、DSPy/STORM、FastAPI、arXiv Atom API、Markdown/HTML/PDF、Vanilla JS、unittest。

---

### Task 1: arXiv 结构化查询与相关性门禁

**Files:**
- Modify: `knowledge_storm/rm.py`
- Modify: `knowledge_storm/paperstorm_trace.py`
- Test: `tests/test_paperstorm_retrievers.py`

- [ ] 添加失败测试：`muon优化器` 编译出包含 `Muon optimizer`、Newton-Schulz/orthogonalized momentum 的字段查询；μ 子探测器结果被过滤，优化器论文保留；多查询去重且空结果继续回退。
- [ ] 运行 `D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest tests.test_paperstorm_retrievers`，确认 RED。
- [ ] 实现 `compile_arxiv_queries()`、领域锚点相关性过滤与 query diagnostics，保持 PIM 行为兼容。
- [ ] 运行同一测试确认 GREEN，并提交 `fix(retrieval): compile constrained arxiv queries`。

### Task 2: 统一参考文献物化

**Files:**
- Create: `knowledge_storm/paperstorm_references.py`
- Modify: `knowledge_storm/paperstorm_service.py`
- Modify: `knowledge_storm/paperstorm_pipeline.py`
- Test: `tests/test_paperstorm_references.py`
- Test: `tests/test_paperstorm_service.py`

- [ ] 添加失败测试：从 `url_to_info.json` 生成有序标题/作者/arXiv URL，文章末尾追加唯一参考文献，重复调用幂等。
- [ ] 确认 RED 后实现 canonical registry、Markdown renderer 和 artifact materializer。
- [ ] 在真实 Pipeline 完成后物化正文；`get_article()` 返回 `references`；PDF 使用物化后的 Markdown。
- [ ] 运行引用与服务测试确认 GREEN，并提交 `fix(citations): materialize canonical references`。

### Task 3: 问答引用一致性

**Files:**
- Modify: `knowledge_storm/paperstorm_research_qa.py`
- Modify: `knowledge_storm/paperstorm_qa.py`
- Modify: `frontend/paperstorm_dashboard/app.js`
- Test: `tests/test_paperstorm_research_qa.py`
- Test: `tests/test_paperstorm_ui_v57.py`

- [ ] 添加失败测试：grounded 回答的文本末尾包含标题、作者、原文 URL，结构化 citations 不丢失；无引用不追加空标题。
- [ ] 确认 RED 后复用 Reference Renderer 追加答案引用，前端优先显示 canonical URL/title/authors。
- [ ] 运行问答和 UI 测试确认 GREEN，并提交 `fix(qa): append original paper links`。

### Task 4: 恢复调研深度与零结果失败语义

**Files:**
- Modify: `examples/storm_examples/paperstorm_service_api.py`
- Modify: `examples/storm_examples/run_paper_storm_minimax.py`
- Modify: `knowledge_storm/paperstorm_pipeline.py`
- Modify: `frontend/paperstorm_dashboard/index.html`
- Test: `tests/test_paperstorm_service.py`
- Test: `tests/test_paperstorm_pipeline.py`

- [ ] 添加失败测试：API 默认 3 视角/2 轮/Top5/并发3，生成预算达到设计值，零有效来源产生 `empty_retrieval`。
- [ ] 确认 RED 后实现平衡 Profile，并确保请求参数贯穿 CLI/Pipeline；前端默认与后端一致。
- [ ] 运行服务与 Pipeline 测试确认 GREEN，并提交 `fix(research): restore balanced research profile`。

### Task 5: 真实 API 端到端验收

**Files:**
- Modify: `README.md`
- Create/Update: `docs/ARXIV_REFERENCE_RECOVERY.md`

- [ ] 用 `D:\SOFTWARE\spyder\envs\storm\python.exe -m uvicorn examples.storm_examples.paperstorm_service_api:app --host 127.0.0.1 --port 8002` 启动服务。
- [ ] 通过 HTTP API 提交“Muon 优化器”真实调研并生成 PDF；检查相关来源、文章章节/长度、参考文献、URL 与 PDF 文本。
- [ ] 通过问答 API 提问 Muon 更新规则/适用层，检查 grounded answer 与文末原文链接。
- [ ] 运行相关测试与离线全量测试，记录真实任务 ID、产物路径、来源和已知边界；更新 README。
- [ ] 提交 `docs: verify muon research and references`，合并到 main；如用户要求再创建版本分支并推送。
