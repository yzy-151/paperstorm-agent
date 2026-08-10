# PaperStorm v5.5：公开论文 RAG Benchmark

## 结论先行

v5.5 将 PaperStorm 的评测从 synthetic seed 和本地 Zotero 候选扩展到两个公开、可复现的数据集：

- **BEIR SciFact**：验证科学文献全库检索；
- **AllenAI QASPER**：验证已知论文内部的证据检索、证据段落选择与端到端答案生成。

正式实验均使用真实 `all-MiniLM-L6-v2` Dense embedding、BM25、RRF Hybrid 和 `ms-marco-MiniLM-L-6-v2` Cross-Encoder，保存原始 ranking、坏例、manifest、数据哈希、模型、seed、延迟和 5,000 次 Bootstrap 95% CI。

## SciFact 官方 test

- 语料：5,183 篇科学摘要；
- Query：官方 test 300 条；
- 数据完整性：官方下载压缩包 MD5 `5f7d1de60b170fc8027bb7898e2efca1`；
- Top-K：10；
- Corpus SHA-256：`54e2468b7b03e164cd2a0d87bafe248e00e991cde4f5eab0d5122f540f6731a9`。

| 方法 | Recall@10 | MRR@10 | nDCG@10 | P95 query |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.7592 | 0.6069 | 0.6395 | 42.2 ms |
| Dense | 0.7857 | 0.6071 | 0.6492 | 20.5 ms |
| Hybrid | 0.8114 | 0.6298 | 0.6687 | 67.8 ms |
| Hybrid + Rerank | **0.8379** | **0.6659** | **0.7001** | 2733.5 ms |

Hybrid 相比 BM25 同时提高 Recall、MRR 和 nDCG；Rerank 继续提高排序质量，但 CPU P95 约 2.73 秒。因此质量最优配置是 Hybrid+Rerank，低延迟配置是 Hybrid。

## QASPER 官方 test

QASPER 的标准 evidence-selection 场景已经知道问题属于哪篇论文。因此评测不是在全部论文间找文档，而是在问题所属论文的完整段落中寻找人工 evidence。代码按 `paper_id` 建立 scoped index，保留同论文非证据段落作为 hard negatives；不会只把金标段落送给检索器。

- 官方 test 全部论文段落：19,914；
- 有人工 evidence、进入 Retrieval 分母的问题：1,309；
- Dataset fingerprint：`462b1c1545733a5e`；
- Top-K：5；
- Corpus SHA-256：`a48f71cfc64807cfd665e4f521c31cc8db6a0402464085bb60c278b55fde5c72`。

| 方法 | Evidence Recall@5 | MRR@5 | nDCG@5 | P95 query |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.4279 | 0.3714 | 0.3396 | 0.34 ms |
| Dense | 0.4771 | 0.4527 | 0.4026 | 14.5 ms |
| Hybrid | 0.5057 | 0.4595 | 0.4155 | 15.3 ms |
| Hybrid + Rerank | **0.6186** | **0.5802** | **0.5327** | 1316.7 ms |

validation 上共有 888 条有效问题，趋势与 test 一致：Hybrid+Rerank Recall@5 为 0.6126、nDCG@5 为 0.5024。正式 test 使用相同配置运行一次，结果产生后不再基于 test 调参。

### QASPER 端到端生成

在 20 条 validation smoke 上冻结 `qasper-grounded-json-v2` 后，使用相同 Hybrid+Rerank Top-5 和 `deepseek/deepseek-chat` 完整运行 validation 1,005 题与 test 1,451 题。test 不再用于调参。

| split | Answer F1 | Exact Match | Evidence F1 | 成功率 |
| --- | ---: | ---: | ---: | ---: |
| validation | 0.4722 | 0.2428 | 0.5025 | 100% |
| test | **0.5441** | **0.3274** | **0.5814** | **100%** |

test Answer F1 已与 QASPER v0.3 自带 `qasper_evaluator.py` 精确对拍；Evidence F1 使用官方 `--text_evidence_only` 口径。详细协议与排查过程见 `docs/PAPERSTORM_V55_QASPER_ANSWER_F1.md`。

## v5.5 新增实现

- 统一 `BenchmarkDocument / BenchmarkCase / BenchmarkDataset` 契约；
- BEIR JSONL/TSV 和 Hugging Face QASPER adapter；
- Recall、MRR、graded nDCG、Token F1、Evidence P/R/F1、Abstention P/R；
- SciFact 全库检索和 QASPER `paper_id` scoped retrieval；
- manifest、raw predictions、bad cases、JSON/Markdown 报告；
- 数据 MD5、corpus SHA-256、dataset fingerprint、Git commit 和工作区 dirty 状态；
- 单元测试显式阻断 socket，CI 使用 package-aware discovery；
- BM25 不再计算未使用的 Dense；
- Dense cosine 从 Python pair loop 改为数学等价的 NumPy exact matrix computation。

## 遇到的困难及解决

### 1. 官方 SciFact HTTPS 证书失败

storm Python 环境缺少正确 CA 链，`urllib` 报 `CERTIFICATE_VERIFY_FAILED`。没有关闭 SSL 校验，而是显式使用 `certifi` CA bundle、User-Agent、timeout 和 MD5 校验。

### 2. Windows 模型 ID 被写成本地路径

首次命令使用 `sentence-transformers\all-MiniLM-L6-v2`，被解释为本地目录。改用 Hugging Face 标准 ID `sentence-transformers/all-MiniLM-L6-v2`。

### 3. 精确 Dense 运行超过十分钟

旧 `_dense_search` 对每个 query、每个文档逐元素执行 Python cosine；同时 BM25 模式也无条件计算 Dense。修复后 BM25 只计算词法排名，Dense 使用 NumPy 归一化矩阵乘法。算法仍是 exact cosine，没有换 ANN 或降低候选质量。

### 4. QASPER 的潜在标签泄漏

第一版 smoke subset 只保留 gold evidence 文档，会让任务变得虚假简单。回归测试发现后改为保留所选论文的全部段落，并让每个问题只搜索对应 `paper_id` 的完整段落集合。

### 5. 不能混淆 Evidence Retrieval 与 Answer QA

QASPER Retrieval 只对 1,309 个有文本 evidence 的问题计算 Recall/MRR/nDCG；Answer QA 则覆盖全部 1,451 个 test 问题，包括 79 个不可答问题。两层独立报告，避免把“找到证据”误写成“回答正确”。

### 6. 模型 JSON 与官方口径对拍

DeepSeek 会把带 `::` 的 evidence ID 输出为未加引号的无效 JSON。实现了只修复 `evidence_ids` 裸 ID 的安全解析器，并增加解析重试与断点续跑。官方 Evidence F1 首次与本地不一致，最终定位到 `--text_evidence_only` 对 `FLOAT SELECTED` 的过滤；修正口径后，以官方 evaluator 为权威结果。

## 指标与延迟边界

- Query 延迟是在模型加载和冷建索引完成后测量的 warm-query 延迟；冷建索引时间单独保存在 manifest。
- 本轮正式实验基于未提交工作区执行，因此 `working_tree_dirty=true`；commit 只表示基线版本，完整实现差异仍以当前工作区和本报告为准。
- 延迟来自单机 CPU，只能用于同机同协议比较，不能写成线上 SLA。
- 检索指标由确定性手算 fixture 验证；实验环境未安装可选 BEIR/pytrec_eval，因此不声称完成 BEIR evaluator 自动对拍。QASPER Answer/Evidence F1 已使用数据集自带 evaluator 对拍。
- SciFact/QASPER test 已经用于本版本冻结报告，后续不能依据这些坏例调参后继续称其为未见 test。

## 复现

数据与模型缓存应放在仓库外。下面路径仅为示例：

```powershell
# SciFact：显式下载、校验并运行完整检索
python examples/storm_examples/run_paperstorm_public_benchmark.py `
  --benchmark scifact --download `
  --cache-dir <external-cache> `
  --output-dir results/public_benchmarks/v55_scifact_real `
  --split test --modes bm25 dense hybrid hybrid_rerank `
  --embedding real --model sentence-transformers/all-MiniLM-L6-v2 `
  --reranker --reranker-model cross-encoder/ms-marco-MiniLM-L-6-v2 `
  --top-k 10 --bootstrap-samples 5000 --seed 55

# QASPER：Hugging Face 官方 test，按 paper_id 搜索全文段落
python examples/storm_examples/run_paperstorm_public_benchmark.py `
  --benchmark qasper --cache-dir <external-cache> `
  --output-dir results/public_benchmarks/v55_qasper_test_real `
  --split test --modes bm25 dense hybrid hybrid_rerank `
  --embedding real --model sentence-transformers/all-MiniLM-L6-v2 `
  --reranker --reranker-model cross-encoder/ms-marco-MiniLM-L-6-v2 `
  --top-k 5 --bootstrap-samples 5000 --seed 55

# QASPER：冻结检索结果后的完整生成式 Answer F1
python examples/storm_examples/run_qasper_answer_benchmark.py `
  --split test `
  --retrieval-predictions results/public_benchmarks/v55_qasper_test_real/predictions.jsonl `
  --output-dir results/public_benchmarks/v55_qasper_answer_test_real `
  --cache-dir <external-cache> --top-k 5
```

## 面试推荐讲法

> 我没有继续使用 synthetic 0.99 作为主结果，而是接入了 SciFact 和 QASPER 两个公开论文 Benchmark。SciFact 300 个 test query 上，Hybrid+Rerank 的 nDCG@10 从 BM25 的 0.6395 提升到 0.7001；QASPER 1,309 个 evidence query 上，nDCG@5 从 0.3396 提升到 0.5327。冻结检索和 Prompt 后，我又在全部 1,451 个 QASPER test 问题上得到官方 Answer F1 0.5441、Evidence F1 0.5814，且无缺失预测。与此同时 Rerank 的 CPU P95 达到约 1.32 秒，所以我把质量最优和部署配置分开报告，并实现断点续跑、Token 审计与官方 evaluator 对拍。
