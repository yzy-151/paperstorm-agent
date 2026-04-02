# PaperStorm 运行可观测性与健壮性学习记录

## 背景

PaperStorm 的主流程已经能调用 DeepSeek、arXiv 和本地 PDF 检索，但实际运行时终端出现大量 warning/error，用户很难判断：

- 程序是否真的失败；
- 哪些只是第三方库噪声；
- 哪些是外部服务暂时失败；
- 结果文件到底在哪里；
- 为什么文章生成阶段会在检索结果为空时崩溃。

这次改动的目标不是单纯“隐藏日志”，而是把运行状态变得可解释、可诊断、可继续迭代。

## 问题 1：第三方库 warning 太吵

### 现象

运行中反复出现：

```text
Pydantic serializer warnings
LiteLLM completion() model=deepseek-chat
Provider List: https://docs.litellm.ai/docs/providers
```

这些信息数量很大，但多数不影响主流程生成结果。

### 根因

- LiteLLM 和 Pydantic 的对象序列化结构不完全匹配，会触发 `UserWarning`；
- LiteLLM 会通过 logging 或 stdout 打印 provider 信息；
- 这些信息不是业务错误，但挤占了用户对真正错误的注意力。

### 改进

- 在 PaperStorm runner 中增加统一日志配置；
- 过滤 Pydantic serializer warning；
- 压低 LiteLLM/httpx/sentence-transformers 等第三方 logger 的等级；
- 屏蔽 LiteLLM 直接写到 stdout 的 Provider List；
- 保留 PaperStorm 自己的关键 INFO。

### 学习点

日志不是越多越好。好的日志应该帮助用户判断系统状态，而不是把用户淹没。

面试表达：

> 我对 PaperStorm 做过 runtime observability 改进。原系统虽然能跑，但第三方库 warning 和 provider 日志刷屏。我把日志按影响面分类，过滤非业务噪声，同时保留模型、检索器、输出目录和关键产物路径，让运行状态更可诊断。

## 问题 2：Wikipedia 辅助页面抓取失败

### 现象

运行中出现：

```text
Error occurs when processing https://en.wikipedia.org/wiki/...:
'NoneType' object has no attribute 'text'
```

### 根因

STORM 的 persona 生成阶段会让 LLM 推荐相关 Wikipedia 页面，再抓取页面标题和目录作为专家视角参考。

原逻辑默认：

```python
soup.find("h1").text
```

但 Wikipedia 对没有 User-Agent 的默认 requests 请求可能返回 403。返回内容不是正常 HTML，自然没有 `<h1>`，于是触发 `NoneType.text`。

### 改进

- 给 Wikipedia 请求加 User-Agent；
- 加 timeout；
- 调用 `raise_for_status()` 明确处理 HTTP 错误；
- 检查 `h1` 是否存在；
- 旧的 Wikipedia 辅助页失败日志不再以 ERROR 干扰主流程。

### 学习点

外部网页结构和服务策略都不可靠，爬取类逻辑必须防御式编程。

面试表达：

> 我修复过 persona generation 里的 Wikipedia 抓取问题。原代码假设页面一定有 h1，但在 403 或页面结构异常时会触发 NoneType.text。我补了 User-Agent、timeout、HTTP 状态检查和结构检查，把隐式异常变成可解释的降级路径。

## 问题 3：LLM 生成的搜索 query 混入结构噪声

### 现象

arXiv 检索里出现了明显不该进入搜索引擎的 query：

```text
```json
queries": [
**Queries:**
以下是根据您的需求...
好的，作为理论基础专家...
```

这些 query 会浪费 arXiv 请求次数，也更容易触发 429。

### 根因

`QuestionToQuery` 期望 LLM 输出多行搜索词，但 DeepSeek 有时会输出 JSON、Markdown 代码块、说明性句子或角色扮演文本。旧的 `clean_search_queries` 只会去掉 `-` 和空行，无法过滤这些结构噪声。

### 改进

- 增加结构噪声过滤；
- 跳过代码块标记、JSON 字段、列表括号；
- 跳过“以下是...”“好的，作为...”这类说明性句子；
- 保留真正可用于检索的搜索词。

### 学习点

LLM 输出不能直接信任。只要 LLM 输出会进入工具调用，就必须做结构校验和清洗。

面试表达：

> 我发现 LLM 生成的 query 会混入 Markdown、JSON 字段和解释性中文句子，导致 arXiv API 收到无效搜索词。我增强了 query sanitizer，并用回归测试覆盖这些脏输出，减少无效外部请求和限流风险。

## 问题 4：arXiv 单条 query 失败不应中断主流程

### 现象

日志里出现：

```text
429 Too Many Requests
Read timed out
```

### 根因

arXiv 是公开 API，可能因为请求频率、网络、代理、服务状态或无效 query 失败。

这类失败通常只影响某一条 query，不代表整个 pipeline 崩溃。

### 改进

- 单条 arXiv query 失败时跳过；
- 日志从 ERROR 降级为 INFO；
- 主流程继续处理其它 query。

### 学习点

在多 query 检索系统里，单个外部请求失败应该被视为部分失败，而不是系统级失败。

面试表达：

> 我把 arXiv 单 query 失败从 ERROR 降为 INFO，因为它是部分外部请求失败，系统可以继续用其它 query 生成结果。这是把日志级别和真实故障严重性对齐。

## 问题 5：检索结果为空时文章生成崩溃

### 现象

文章生成阶段报错：

```text
ValueError: Expected 2D array, got 1D array instead: array=[].
```

### 根因

如果前面所有 arXiv query 都失败或没有有效结果，`StormInformationTable` 中没有 snippet。

旧逻辑仍然执行：

```python
cosine_similarity([encoded_query], self.encoded_snippets)
```

此时 `self.encoded_snippets` 是空数组，sklearn 期望二维矩阵，于是崩溃。

### 改进

- 如果没有 collected snippets，不加载 SentenceTransformer；
- `retrieve_information()` 直接返回空列表；
- 避免空资料表进入向量相似度计算。

### 学习点

RAG/Agent pipeline 的每个阶段都要考虑“上游没有产物”的情况。检索为空是正常边界条件，不应该导致下游崩溃。

面试表达：

> 我修复了文章生成阶段对空检索结果不健壮的问题。当前序阶段没有 snippet 时，原系统仍然做向量相似度计算，导致 sklearn 对空数组报错。我增加了空 information table 的边界处理，使检索为空时下游能优雅降级。

## 本次你应该掌握的工程能力

1. 区分致命错误、部分失败、第三方噪声。
2. 用日志等级表达真实严重性。
3. 对外部 API 和网页抓取做防御式编程。
4. 对 LLM 输出进入工具调用前做清洗。
5. 对 RAG pipeline 的空结果边界做保护。
6. 用回归测试锁住日志行为和异常边界。

## 简历/面试可总结为

> 基于 Stanford STORM 二次开发 PaperStorm 时，我重点改进了运行可观测性和检索链路健壮性。针对 DeepSeek/LiteLLM 运行中的 Pydantic warning、Provider List 刷屏、Wikipedia 辅助抓取失败、arXiv 单 query 失败和空检索结果导致文章生成崩溃等问题，我进行了日志分级、噪声过滤、User-Agent/timeout/HTTP 校验、LLM query sanitizer 和空 information table 防护，并补充单元测试覆盖这些边界。这个改动让项目从“能跑”变成“可诊断、可交付、可复现”。

## 后续维护约定

之后每次 PaperStorm 有实质改动时，都追加一节：

- 改了什么；
- 为什么改；
- 遇到了什么问题；
- 如何验证；
- 你应该学到什么；
- 面试中可以怎么讲。

## 2026-07-21：PIM 歧义消解与 arXiv 检索质量优化

### 现象

当 topic 是“PIM 神经网络抑制”时，用户真实意图是：

```text
Passive Intermodulation，中文是无源互调，属于射频/天线/通信系统问题。
```

但 arXiv 检索可能返回：

```text
Processing-in-Memory
RAM / DRAM system
Product Information Management
```

这些结果虽然也可能缩写为 PIM，但和“无源互调抑制”完全不是一个领域。

### 根因

PIM 是高度歧义缩写。LLM 生成 query 时如果只写：

```text
PIM neural network suppression
PIM system
PIM RAM
```

arXiv 并不知道这里的 PIM 是 passive intermodulation。搜索引擎会按统计相关性召回其它领域的 PIM 论文。

这不是单纯的“模型不聪明”，而是工具调用前缺少领域消歧。

### 改进

在 `ArxivRM` 层增加确定性的 query 规范化和结果过滤：

- 中文关键词英文化：
  - `无源互调` → `passive intermodulation`
  - `神经网络` → `neural network`
  - `抑制` → `suppression`
  - `射频` → `radio frequency`
- 当 query 中出现 PIM，并且上下文包含 suppression、RF、antenna、neural network 等词时，将 PIM 扩展为 `passive intermodulation`；
- 当 query 明确是 RAM、DRAM、processing-in-memory 方向时，直接跳过；
- 当 query 已经消歧为 passive intermodulation 时，过滤掉 title/abstract 中明显是 processing-in-memory、RAM、product information management 的结果。

### 验证

新增回归测试：

- `pim 神经网络抑制` 会被规范化为 `passive intermodulation neural network suppression`；
- 返回结果中保留 passive intermodulation 论文；
- 过滤 processing-in-memory / RAM 论文；
- `PIM RAM processing-in-memory system` 不再请求 arXiv。

测试结果：

```text
27 tests OK
```

### 你应该学到什么

1. 缩写词是检索系统里的高风险输入。
2. LLM 生成 query 后，不能直接丢给外部搜索 API。
3. 对专业领域缩写，要做 query expansion / disambiguation。
4. 检索质量不能只靠 prompt，要在工具层做确定性约束。
5. 结果端也要过滤，因为即使 query 正确，搜索引擎仍可能召回跑题文档。

### 面试可以怎么讲

> 我在 PaperStorm 里处理过专业缩写导致的检索跑题问题。比如 PIM 在通信领域指 passive intermodulation，但在计算机体系结构里也指 processing-in-memory。原系统直接把 LLM 生成的 PIM query 发给 arXiv，容易召回 RAM/DRAM 论文。我在 ArxivRM 层做了 query expansion 和领域消歧：在 RF、suppression、neural network 上下文中把 PIM 扩展为 passive intermodulation，并过滤 processing-in-memory、product information management 等跑题结果，同时用回归测试锁住这个行为。这个改动体现了 Agent 工具调用前的输入规范化和检索质量控制。
