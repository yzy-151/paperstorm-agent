# PaperStorm v7.0 PIM 领域评测

## 目标

该评测验证 PaperStorm 在无源互调（PIM）垂直领域中的完整链路：论文解析、结构化切分、
三种真实 Embedding Profile 检索、Exact/HNSW 对比、证据约束回答和引用校验。它是 50 题的
私有领域 pilot，用于工程选型和 Bad Case 定位，不替代公开 Benchmark 或生产 SLA。

## 数据与协议

- 来源：5 篇 Zotero 本地 PIM 论文，每篇均衡抽取 10 个证据块，共 797 个候选 chunk。
- 题集：50 题；`definition/mechanism/method/experiment/comparison/limitation` 各 8–9 题。
- 生成：Hermes `qwen3.8-flash` 根据指定的唯一证据块生成问题、参考答案和逐字引用。
- 校验：问题和 ID 唯一、六类覆盖、证据 ID 存在、引用能在原文连续定位；近似引用只允许
  高相似度回绑到真实原文，无法回绑则 fail fast。
- 检索：每个 query 搜索全部 797 个 chunk；固定 Top K=5、BM25 + Dense + RRF、Exact oracle，
  关闭 Cross-Encoder，避免混入重排变量。
- Reader：胜出 Profile 的 Top-5 evidence 输入 Hermes `qwen3.8-flash`；标准答案不进入 Prompt。
- 指标：Recall/MRR/nDCG、P50/P95、Answer F1、Evidence/Citation Recall、Citation Precision。

私有论文、chunk、题集和模型原始输出位于
`$env:PAPERSTORM_BENCHMARK_ROOT\domain-pim-v7`，不进入 Git。

## 实测结果

### 三模型检索

| Profile | 模型 | Recall@5 | MRR@5 | nDCG@5 | Build | Query P95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Legacy | `paraphrase-multilingual-MiniLM-L12-v2` | 0.5800 | 0.4533 | 0.4857 | 26.9 s | 48.7 ms |
| CPU-ZH | `BAAI/bge-small-zh-v1.5` | 0.6400 | 0.4507 | 0.4979 | 40.5 s | **42.6 ms** |
| CPU-Multilingual | `Alibaba-NLP/gte-multilingual-base` | **0.7200** | **0.4903** | **0.5466** | 240.4 s | 188.5 ms |

GTE 相对 Legacy 的 Recall@5 提升 0.14，但构建和查询更慢；BGE 是当前中文 CPU 场景的低延迟
备选。默认选择规则是先最大化 Recall@5，再以较低 P95 打破平局。

### 真实向量 ANN

| 向量 | 维度 | Exact P95 | HNSW P95 | HNSW Recall@5 | HNSW index |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 797 | 768 | 2.101 ms | 0.790 ms | 1.0000 | 2,619,072 bytes |

该结果使用真实 GTE 论文向量而非随机矩阵。规模只有 797，说明功能和近似质量正确，但不能据此
推断百万级延迟；大规模结论仍需独立容量实验。

### 50 题端到端回答

| 指标 | 结果 |
| --- | ---: |
| Answer F1 | 0.3983 |
| Evidence Recall | 0.7200 |
| Citation Recall | 0.7200 |
| Raw Citation Precision | 0.9237 |
| Non-empty answer rate | 1.0000 |
| Raw invalid citation IDs | 12 |

Reader 50/50 返回非空答案。12 个原始无效引用 ID 来自模型复制或改写长 chunk ID；产品输出必须
通过 evidence registry 白名单过滤并由可信元数据回填，不能直接信任模型生成的引用字符串。
无引用的回答在本报告中 Citation Precision 记为 0，因此合法拒答也会降低该均值；该指标需与
Non-empty answer rate、Evidence Recall 和拒答率联合解读。
本次 11 个完成的 Hermes 批次共 13 次 API 调用，usage 文件合计 385,871 tokens；服务未提供
可核验单价，因此不报告虚构成本。

## 具体 Bad Case

| 案例 | 现象 | 根因 | 当前处理 | 后续方向 |
| --- | --- | --- | --- | --- |
| `pim-010` 两种数学模型总结 | gold 未进入 Top-5 | 答案跨多个总结句，单 chunk 标签不足 | 保留失败并记录 Evidence Recall | 建立多 chunk gold 与 section parent 标签 |
| `pim-037` 最大残留 ePIM 数值 | gold 未进入 Top-5 | 图表数值与正文术语错配 | Reader 只能基于已召回证据回答 | 图表结构化解析与数值字段索引 |
| `pim-047` 波束赋形计算开销 | gold 未进入 Top-5 | 中英文近义表达和相邻论文竞争 | GTE + RRF 优于 Legacy，但仍漏召回 | 查询扩展与领域术语同义词 |
| `pim-021` 特征值结论 | gold 未进入 Top-5 | 提问引用“多项观测量”，语义目标不够局部 | 失败可追踪，不注入参考答案 | 多向量 section 表示与 query decomposition |
| `pim-019` 模型引用少一个连接词 | 引用不再是逐字原文 | 生成模型做了轻微改写 | 高相似窗口回绑到连续原文 | 继续保持 fail-fast 阈值并人工抽检 |

## 可信边界

1. 问题与参考答案由同一模型生成，虽然标准答案未泄漏给 Reader，仍不等同人工专家题集；
   Answer F1 只能作为链路 pilot。
2. 50 题样本较小，未计算置信区间，不应宣称统计显著或行业领先。
3. 检索比较使用完全相同的题集、语料和协议，可以做该 pilot 内的 Profile 选型。
4. 原始 Citation Precision 反映模型行为；线上必须使用引用白名单，不能用后处理后的零泄漏数字
   替代原始模型指标。

## 复现

数据准备：

```powershell
python examples/storm_examples/prepare_pim_domain_pilot.py `
  --pdf <paper-1.pdf> --pdf <paper-2.pdf> --pdf <paper-3.pdf> `
  --pdf <paper-4.pdf> --pdf <paper-5.pdf> `
  --output-dir "$env:PAPERSTORM_BENCHMARK_ROOT\domain-pim-v7"
```

三模型与 ANN：

```powershell
python examples/storm_examples/run_pim_domain_pilot.py `
  --corpus <domain-pim-v7\corpus.jsonl> `
  --cases <domain-pim-v7\cases.jsonl> `
  --output-dir <domain-pim-v7\runs> `
  --model-cache <paperstorm-benchmarks\models> `
  --top-k 5
```

Reader Prompt 与评分：

```powershell
python examples/storm_examples/answer_pim_domain_pilot.py `
  --corpus <corpus.jsonl> --cases <cases.jsonl> `
  --predictions <runs\retrieval\cpu-multilingual\predictions.jsonl> `
  --output-dir <domain-pim-v7\answers> `
  --responses-dir <domain-pim-v7\answers\responses>
```
