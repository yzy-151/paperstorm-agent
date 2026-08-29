# arXiv 召回、调研深度与引用交付设计

## 目标

修复 Web API 真实模式下三个连续故障：中文或新术语导致 arXiv 空召回；默认运行预算过小导致文章缩水；正文中的 `[n]` 未被稳定映射为标题、作者和原文链接，进而使 Web、PDF 和问答丢失参考文献。

## 根因证据

现有 `ArxivRM` 把裸查询直接写入 `search_query`。`muon优化器` 实测返回 0 条；`Muon optimizer neural network training` 返回 5 条却以 μ 子物理为主，说明空格词项没有表达“优化器语境必须成立”。Web API 默认 `max_perspective=1`、`max_conv_turn=1`、`search_top_k=2`，文章生成仅 1800 tokens。STORM 虽生成 `url_to_info.json`，但 `get_article()` 只返回正文，PDF 也直接渲染该正文，因此引用 registry 没有进入交付物。

## 设计

### 1. arXiv Query Compiler

`ArxivRM` 接收主题后生成有优先级的 arXiv 查询：标准学术短语、`all:` 字段与 AND 约束、宽松回退。Muon 优化器的领域别名包含 `Muon optimizer`、`MomentUm Orthogonalized by Newton-schulz`、`orthogonalized momentum optimizer`，并要求 optimizer/training/neural-network 语境。每轮结果按原始主题和领域锚点过滤、去重；空结果继续下一条查询，不回退到其他数据源。Trace 保存实际查询与每轮数量。

### 2. Research Quality Profile

Web API 和前端真实调研默认使用平衡 Profile：3 个视角、2 轮对话、每查询 Top5、章节取证 Top5、并发 3。LLM 预算提升为访谈/提问 1200、Outline 2600、Article 5000、Polish 7000。用户仍可在高级设置覆盖。任务成功前增加最低交付门禁：至少一个有效来源；若零来源，返回明确 `empty_retrieval`，不能生成空洞文章。

### 3. Unified Reference Renderer

新增纯函数模块读取 STORM `url_to_info.json` 或问答 citations，规范化 citation index、title、authors、arXiv abstract URL/PDF URL。它负责：

1. 保留正文原有 `[n]`；
2. 在正文末尾追加唯一的 `## 参考文献`；
3. 每条采用 `[n] 标题 — 作者. [原文](URL)`；
4. 缺失作者仍保留标题和 URL，缺 URL 的引用不伪造链接；
5. 重复运行幂等，不重复追加参考文献。

研究任务完成后先物化 Markdown，再交给文章 API 与 PDF；问答 API 在结构化 `citations` 之外，也在答案文本末尾追加同一格式。前端继续渲染结构化引用列表，因此文本、UI、下载 Markdown 和 PDF 一致。

## 错误处理

网络失败、XML 解析失败、单查询无结果分别记录；多查询全部失败时区分 `retrieval_error` 与 `empty_retrieval`。引用 registry 缺失时文章可返回，但任务和 Trace 标记 `references_missing`；正文包含 `[n]` 却找不到对应 registry 时不能伪造来源。

## 验收

1. 单元测试证明中文 Muon 主题会编译为受约束查询，μ 子物理论文被过滤，优化器论文保留。
2. 服务默认 Profile 不再是 1/1/2，且 Pipeline 接收同一配置。
3. 文章 API 内容末尾包含参考文献、标题、作者和 `https://arxiv.org/abs/...`。
4. PDF 文本包含参考文献标题与 URL，公式渲染不回退。
5. 问答答案只要引用论文，就在答案末尾附原文链接，结构化 citations 同时保留。
6. 启动真实 Web API，用 Muon 优化器完成调研与问答；检查任务成功、来源相关、正文长度合理、Web/PDF/问答引用完整。

## 非目标

本次不引入 Crossref/Semantic Scholar，不重写 STORM Multi-Agent，不把 arXiv 全库复制到本地索引，也不以内容长度代替事实质量。
