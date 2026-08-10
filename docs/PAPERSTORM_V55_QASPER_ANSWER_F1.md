# PaperStorm v5.5：QASPER 端到端 Answer F1 实验

## 结论

PaperStorm 已从“能否检索到证据”推进到“检索后能否生成正确答案”。冻结协议在 AllenAI QASPER v0.3 官方 test 的全部 1,451 个问题上运行，使用 Hybrid+Rerank Top-5 证据和 DeepSeek Chat，最终得到：

| split | 问题数 | 成功率 | Answer F1 | Exact Match | Evidence F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation | 1,005 | 100% | 0.4722 | 0.2428 | 0.5025 |
| test | 1,451 | 100% | **0.5441** | **0.3274** | **0.5814** |

test 分类型 Answer F1：

| 类型 | F1 |
| --- | ---: |
| Extractive | 0.5690 |
| Abstractive | 0.2651 |
| Boolean | 0.7571 |
| Unanswerable | 0.7174 |

官方 `qasper_evaluator.py --text_evidence_only` 对拍结果为 Answer F1 `0.5441468958749875`、Evidence F1 `0.5814043201421425`、Missing predictions `0`。Answer F1 与项目实现精确一致。

## 冻结协议

- 数据：AllenAI QASPER v0.3；validation 1,005 题，test 1,451 题；
- 检索：问题所属论文内检索，Hybrid+Rerank Top-5；
- Embedding：`sentence-transformers/all-MiniLM-L6-v2`；
- Reranker：`cross-encoder/ms-marco-MiniLM-L-6-v2`；
- 生成模型：`deepseek/deepseek-chat`；
- Prompt：`qasper-grounded-json-v2`；
- 温度：0；最大输出 256 token；
- 输出：最短答案片段、列表或 Yes/No；证据不足输出 `Unanswerable`；
- test 只在 validation 冻结协议后完整运行一次，不再用 test 调参。

Token 用量：validation `972,608`，test `1,371,303`。原始回答、引用、延迟、Token、错误和排名均保存在本地 `results/public_benchmarks/`，不提交数据集或密钥。

## 完整链路

```text
Question
  -> BM25 + Dense
  -> RRF Hybrid
  -> Cross-Encoder Rerank
  -> Top-5 evidence
  -> DeepSeek grounded generation
  -> answer / abstention / evidence IDs
  -> QASPER official evaluator
```

检索和生成分开评测：Recall/MRR/nDCG 判断“有没有找到证据”，Answer F1 判断“找到证据后是否答对”。这样可以区分 retrieval error 与 generation error。

## 实施与排查记录

### 1. 为什么 CI 不调用真实 LLM

CI 使用微型 fixture、Hash Embedding 和 Mock LLM，外部 socket 被显式阻断。这样测试可重复、快速、免费，不受 API 限流和断网影响，也不需要向 Fork PR 暴露密钥。真实模型只在独立 Benchmark 命令中运行。

### 2. 第一版 Prompt 的问题

20 题 validation smoke 的 Answer F1 为 `0.3957`。坏例显示模型经常输出解释性长句，且对可由证据推出的 Yes/No 问题过度拒答。v2 要求最短答案、列表或严格 Yes/No，只有证据完全不涉及问题才拒答。相同 20 题最终 F1 提升到 `0.4826`，EM 从 `0.15` 提升到 `0.30`。

### 3. DeepSeek 返回损坏 JSON

DeepSeek 会稳定地把 `1704.06194::section-5::paragraph-7` 这类 evidence ID 输出为未加引号的 JSON 标识符。重复请求仍产生同样错误，单纯重试会浪费 Token。

修复方式是只针对 `evidence_ids` 数组给裸 ID 加 JSON 引号，不使用 `eval`；其他 JSON 结构仍由标准解析器校验。Runner 还支持 API 指数退避、解析重试、逐题 checkpoint 和失败后续跑。

### 4. 官方 Evidence F1 口径差异

项目首次汇总 Evidence F1 为 `0.5540`，官方脚本为 `0.5814`。排查发现官方 `--text_evidence_only` 会过滤 `FLOAT SELECTED` 图表占位符，本地 gold 最初未过滤。修正文本证据口径后，以官方 evaluator 输出作为最终权威结果。预测内容没有修改。

### 5. Windows 编码问题

官方 evaluator 用系统默认编码读取预测，在 Windows 上把 UTF-8 当成 GBK，触发 `UnicodeDecodeError`。设置 `PYTHONUTF8=1` 后原样对拍，没有转换或修改预测文件。

## 如何理解结果

- `0.5441` 证明系统具备完整 RAG 问答链路，但不是顶尖模型成绩；
- Boolean 和不可答问题表现较好，Abstractive F1 `0.2651` 是主要短板；
- QASPER Evidence Retrieval Recall@5 为 `0.6186`，说明约四成 gold evidence 未进入 Top-5，生成上限仍明显受检索限制；
- Rerank 提升质量但 CPU P95 约 1.32 秒，实时聊天默认应使用 Hybrid，离线调研或低置信问题再启用 Rerank；
- 本实验没有测 LLM-as-Judge Faithfulness，也没有与 Oracle evidence 做生成上限对照，后续不能把 Answer F1 描述成“答案完全可信”。

## 复现

```powershell
$env:PYTHONUTF8='1'
D:\SOFTWARE\spyder\envs\storm\python.exe `
  examples\storm_examples\run_qasper_answer_benchmark.py `
  --split test `
  --retrieval-predictions results\public_benchmarks\v55_qasper_test_real\predictions.jsonl `
  --output-dir results\public_benchmarks\v55_qasper_answer_test_real `
  --cache-dir C:\Users\yzy\Desktop\codex\paperstorm-benchmarks `
  --top-k 5
```

命令可断点续跑，已经成功的 case 不会再次调用 API。

## 面试讲法

> 我把 RAG 评测拆成 retrieval 和 generation 两层。检索侧在 QASPER test 上比较 BM25、Dense、Hybrid 和 Cross-Encoder Rerank；生成侧冻结 validation Prompt 后，在全部 1,451 个 test 问题上用 DeepSeek 基于 Top-5 证据回答，官方 Answer F1 为 0.5441，Evidence F1 为 0.5814，1,451 条全部成功。工程上实现了逐题 checkpoint、失败重试、Token 审计和安全 JSON 修复，并用数据集自带 evaluator 对拍，而不是只报自己实现的指标。结果也暴露出 Abstractive QA 和 Top-5 evidence recall 仍是后续瓶颈。
