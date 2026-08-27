# PaperStorm 检索栈工程化升级与实验报告

## 1. 改进目标

本轮工作针对五类可直接影响生产可用性的问题：

1. 默认 Embedding 模型较旧，且查询、文档编码规则没有统一冻结；
2. Dense 检索对全部向量线性扫描，规模扩大后延迟随语料量线性增长；
3. 中文 BM25 仅依赖粗粒度字符切分，领域术语容易被拆散；
4. Cross-Encoder 缺少设备、批次和候选上限契约，质量档与 CPU 档边界不清；
5. Parent Context 使用单一全局预算，靠后的高相关 parent 可能完全无法展开。

改进遵循两个原则：公开质量评测与规模性能评测分离；近似索引、模型和重排器均保留
可审计配置，不能把 smoke、随机向量微基准或估算值写成线上质量结论。

## 2. 改进前后对照

| 难点 | 改进前 | 根因 | 改进方案 | 当前结果 |
| --- | --- | --- | --- | --- |
| Embedding 老旧且角色契约分散 | 默认 MiniLM，部分入口可能采用不同 query prompt | 模型名、revision、维度、最大长度和角色编码未形成统一合同 | 建立冻结 Profile：Legacy、BGE、GTE、Qwen；Memory、RAG 和 Benchmark 共用；索引 manifest 校验合同 | 四模型可在同一公开协议下比较；合同变化会要求重建索引 |
| Qwen CPU 编码被系统终止 | 默认大批次触发进程级内存终止，没有 Python traceback | 大模型文档批次未冻结，瞬时工作集过高 | Qwen 文档 batch=2、查询 batch=1，并写入 Profile 指纹 | SciFact 完整完成；构建耗时显著增加，确认其属于质量档而非 CPU 默认档 |
| Dense 全库线性扫描 | Exact 点积复杂度随向量数线性增长 | 没有 ANN 后端和规模切换策略 | 增加 USearch HNSW；小库保留 Exact；大库自动 HNSW；ACL 限定请求 fail-closed 回退 Exact | 10 万随机向量微基准 HNSW P95 `21.591 ms`，Exact P95 `198.504 ms`，Recall@10 `0.9055` |
| 中文术语切分粗糙 | “无源互调”等术语被拆为字符组合，RAM 干扰项可能靠前 | 缺少领域词典与语言感知 analyzer | Jieba 私有 tokenizer + 领域词典 + CJK bigram fallback；索引记录 analyzer revision | 固定 PIM/RAM 用例中 RF 相关文档优先，避免仅靠通用分词 |
| Rerank 尾延迟与部署边界不清 | 单一 Cross-Encoder 配置，候选量可能失控 | 设备、批次、候选上限不属于运行合同 | CPU balanced 与 CUDA quality Profile；最多重排 20 个融合候选；Trace 记录模型、设备和耗时 | 可区分“CPU 可运行”与“GPU 质量档”；无 CUDA 时质量档显式失败或按配置降级 |
| Parent budget starvation | 前几个 parent 可能耗尽预算，后续高相关 parent 无上下文 | 单一贪心预算没有最低配额 | 每个唯一 parent 先分最低配额，剩余按分数加权；围绕命中 child 双向展开 | Trace 可见 allocation、used、truncated 和 reason；高相关 parent 不再因排序位置被完全饿死 |

## 3. Embedding Profile

| Profile | 模型 | 维度 | 角色 | 推荐用途 |
| --- | --- | ---: | --- | --- |
| `legacy-multilingual` | `paraphrase-multilingual-MiniLM-L12-v2` | 384 | 对称语义相似度 | 历史兼容与回归基线 |
| `cpu-zh` | `BAAI/bge-small-zh-v1.5` | 512 | 中文 query instruction / passage | 中文 CPU 检索候选档 |
| `cpu-multilingual` | `Alibaba-NLP/gte-multilingual-base` | 768 | 多语种 query / passage | 当前 CPU 多语种推荐档 |
| `quality-multilingual` | `Qwen/Qwen3-Embedding-0.6B` | 1024 | instruction-aware query / passage | GPU 或离线质量优先档 |

所有官方 Profile 都冻结模型 revision、维度、最大序列长度、query/document 编码规则和归一化策略。
自定义模型仍可显式传入，但会标为 `custom`，不能冒充冻结 Profile 的可复现结果。

## 4. 公开数据集对比协议

- 数据：BEIR SciFact official test、AllenAI QASPER official test；
- 采样：固定 seed `55`，按 `case_id + query` 的 SHA-256 排序抽取 10% 查询；
- SciFact：30 个 query 搜索完整 5,183 篇摘要；
- QASPER：131 个有文本 evidence 的 query，保留抽中问题所属论文的全部 5,265 个段落，
  包含论文内所有 hard negatives；不会只编码金标段落；
- 检索：BM25 + Dense + RRF，Top K=5；Cross-Encoder 关闭，以隔离 Embedding 影响；
- Dense：固定 Exact oracle，避免 ANN 近似误差污染模型比较；
- 指标：Recall@5、MRR@5、nDCG@5、构建时间、查询 P50/P95；
- 证据：每个 Profile 独立保存 manifest、predictions JSONL、metrics JSON 和模型 revision。

这是 10% 查询诊断实验，不是完整 test 的最终模型排名。30 条 SciFact query 尤其容易受样本方差影响，
最终切换默认模型前仍应运行完整 test 或配对 Bootstrap 置信区间。

### 4.1 SciFact 结果

| Profile | Recall@5 | MRR@5 | nDCG@5 | 构建时间 | 查询 P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy MiniLM | 0.6583 | 0.5122 | 0.5398 | 119.3 s | 84.3 ms |
| BGE small zh | 0.6167 | 0.4889 | 0.5099 | 279.1 s | 179.6 ms |
| GTE multilingual | 0.6750 | 0.5583 | 0.5738 | 2326.8 s | 328.1 ms |
| Qwen3 Embedding 0.6B | **0.7250** | **0.5789** | **0.5973** | 8457.8 s | 1769.7 ms |

解释：Qwen 在该样本上取得最高质量，但 CPU 成本远高于其余模型；GTE 提供更实际的多语种
质量与成本折中。BGE 中文模型在英文 SciFact 上弱于 Legacy，不能据此否定其中文场景价值，
也不能把中文模型在英文数据上的结果外推到中文企业语料。

### 4.2 QASPER 结果

| Profile | Recall@5 | MRR@5 | nDCG@5 | 构建时间 | 查询 P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy MiniLM | 0.4673 | 0.3869 | 0.3626 | 133.3 s | 68.0 ms |
| BGE small zh | 0.3746 | 0.3346 | 0.3012 | 154.6 s | 34.5 ms |
| GTE multilingual | **0.5457** | **0.4545** | **0.4351** | 927.2 s | 201.9 ms |
| Qwen3 Embedding 0.6B | 0.5468 | 0.4532 | 0.4348 | 2869.7 s | 345.0 ms |

Qwen 的 Recall@5 仅比 GTE 高 `0.0011`，但构建时间约为 GTE 的 `3.10x`，查询 P95 约为
`1.71x`；GTE 的 MRR 与 nDCG 还略高。因此本机 CPU 多语种默认档选择 GTE，Qwen 保留为
GPU 或离线质量实验档。该结论只适用于当前 131 条 QASPER 样本，不能外推到所有语言和领域。

## 5. 具体 Bad Case

| Query | 基线/旧模型 | 新模型 | 为什么改善或退化 |
| --- | --- | --- | --- |
| `In mouse models, the loss of CSF1R facilitates MOZ-TIF2-induced leuekmogenesis.` | gold 未进 Top 5 | gold 排名 2 | instruction-aware 模型更好地关联实验条件、基因与疾病结论 |
| `CRP is not predictive of postoperative mortality following CABG surgery.` | gold 未进 Top 5 | gold 排名 4 | 对否定关系与缩写上下文的语义匹配更强 |
| `Normal expression of RUNX1 has tumor-promoting effects.` | gold 未进 Top 5 | gold 排名 2 | 对实体、表达状态和肿瘤效应的组合语义匹配更好 |
| `The treatment of cancer patients with co-IR blockade does not cause any adverse autoimmune events.` | gold 排名 5 | gold 退出 Top 5 | 新模型并非逐查询单调提升；复杂否定和治疗条件仍可能误排 |
| `Do they use evolutionary-based optimization algorithms as one of their domain adaptation approaches?` | Legacy gold 未进 Top 5 | GTE 排名 1，Qwen 排名 2 | 新模型能关联 optimization、domain adaptation 与具体方法描述 |
| `Which works better according to human evaluation, the concurrent or the modular system?` | Legacy 仅召回 2 个 gold 中的 1 个 | GTE/Qwen 均召回 2 个 gold | 多证据比较问题的覆盖得到改善 |
| `what training data was used?` | GTE 召回 1 个 gold，排名 5 | Qwen 两个 gold 均未进 Top 5 | 更大模型仍可能在宽泛问题上偏向非证据段，需保留回归集 |

因此不能只展示三个改善案例。当前 30 条 SciFact 样本中，Qwen 相对 Legacy 有 3 条 Recall 改善、
1 条退化；模型升级仍需要回归集和按 query type 的误差分析。

## 6. HNSW 规模实验

本机使用 100,000 个 384 维随机归一化向量、200 个随机 query、Top K=10，对比 Exact 与
USearch HNSW：

| 配置 | 构建 | P50 | P95 | Recall@10 | 索引大小 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact | 0.123 s | 181.779 ms | 198.504 ms | 1.0000 | 原始向量约 153.6 MB |
| HNSW, M=32, efConstruction=400, efSearch=1200 | 140.074 s | 18.805 ms | 21.591 ms | 0.9055 | 181.2 MB |

HNSW 的 P95 约为 Exact 的 `1/9.19`，代价是构建时间、图索引空间和约 9.45% Top-10
近似召回损失。该实验使用随机向量，只证明实现的规模行为，不代表真实论文 Embedding 分布下的
质量。2,000,000 个 384 维 float32 原始向量约 3.072 GB；这里只报告容量估算，不外推延迟。

权限过滤是 ANN 的安全边界：当前实现遇到 ACL scoped request 时回退 Exact 子集搜索，防止先从
全库 ANN 召回再过滤造成敏感候选进入 rerank、trace 或缓存。超大多租户部署应进一步采用支持
原生 metadata filter 的向量库或按租户/策略分片索引。

## 7. 面试表达

可以把本轮工作概括为：

> 我没有直接把旧 Embedding 替换成排行榜更高的模型，而是先统一 query/document 编码合同，
> 再用固定公开数据、查询采样和 Exact oracle 对比 MiniLM、BGE、GTE、Qwen。Qwen 在 10%
> SciFact 上 Recall@5 最高，但 CPU 构建约 2.35 小时、查询 P95 约 1.77 秒；GTE 是更实际的
> 多语种 CPU 档。规模侧我把线性 Dense 抽象为 Exact/HNSW 双后端，10 万向量 P95 从约
> 198.5 ms 降到 21.6 ms，同时如实报告 Recall@10 0.9055 和索引开销。ACL 请求不走无过滤
> ANN，而是 fail-closed 回退授权子集 Exact。中文检索增加领域词典与 Jieba analyzer，Parent
> Context 则用最低配额和分数加权避免预算饥饿。整个过程把质量、延迟、规模、安全和复现合同
> 分开评估，而不是只报一个最好看的 Recall。

## 8. 剩余工作

- 在冻结 validation 上选择默认 Profile，再对完整 test 只运行一次最终配置；
- 使用真实论文向量复测 HNSW Recall/Latency，并给 `efSearch` 绘制 Pareto 曲线；
- 中文内部语料建立独立标注集，避免用英文 SciFact 评价 BGE 中文能力；
- GPU 环境复测 Qwen Embedding 与 BGE Reranker，区分模型上限和 CPU 部署上限；
- 大规模多租户环境接入原生 metadata filter 的向量数据库并复测 ACL、P95 与索引更新成本。
