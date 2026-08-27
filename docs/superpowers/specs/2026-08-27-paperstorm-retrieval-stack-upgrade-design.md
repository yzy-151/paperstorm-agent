# PaperStorm 检索底座升级设计

日期：2026-08-27
状态：已获用户方案确认，待书面复核后实施

## 1. 目标与边界

本轮采用渐进式可插拔改造，解决五个已确认问题：Embedding 默认值分裂、Dense 全库线性扫描、
中文 BM25 术语切分粗糙、CPU Cross-Encoder 尾延迟高、Parent Context 共享预算饥饿。

目标不是用大模型覆盖所有问题，而是在冻结协议下分别测量质量、延迟、内存和索引规模，形成可解释的
Pareto 选择。保留 Exact、旧 MiniLM 和 CJK bigram 作为对照或降级路径；Hash Embedding 仅用于 CI。

本轮不宣称完成 200 万向量线上压测，不迁移到外部 Qdrant/Milvus，也不把无 GPU 的本机结果冒充线上 SLA。

## 2. Embedding Profile 与实验协议

统一产品、Memory、Benchmark 的模型注册表，禁止入口各自硬编码默认模型。

| Profile | 模型 | 主要场景 |
| --- | --- | --- |
| `legacy-multilingual` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 冻结对照与低资源兼容 |
| `cpu-zh` | `BAAI/bge-small-zh-v1.5` | 中文企业文档与 PIM |
| `cpu-multilingual` | `Alibaba-NLP/gte-multilingual-base` | 中英文论文与企业知识库 |
| `quality-multilingual` | `Qwen/Qwen3-Embedding-0.6B` | 质量优先、离线调研 |

Provider 必须声明 query/document 编码规则、归一化方式、维度、最大长度、模型 revision 和可信远程代码策略。
GTE/Qwen 的查询模板不得复用普通对称相似度调用；索引 manifest 必须包含这些参数，变化时强制重建。

### 2.1 十分之一公开数据集对比

- SciFact test：按稳定 `case_id` 哈希抽取约 10% query，保留完整语料库。
- QASPER test：按稳定 question ID 哈希抽取约 10% question，保留每个问题对应的完整论文候选语料。
- PIM 固定中文集只作领域补充，不冒充公开 Benchmark。
- 固定 chunk、BM25、RRF、Top K、候选数和随机种子；Embedding 对比时关闭 Reranker。
- 输出 Recall@K、nDCG@K、MRR、建库时间、查询 P50/P95、峰值 RSS、索引大小和维度。
- 保存抽样 manifest、模型 revision、依赖版本、原始逐案例排名和失败记录。

## 3. Dense Index Backend

新增稳定的 `DenseIndexBackend` 边界：

- `ExactBackend`：NumPy 内积，作为小库与 ANN recall 对照。
- `HnswBackend`：大库近似检索，参数至少包含 `M`、`efConstruction`、`efSearch`。
- `auto`：低于可配置阈值使用 Exact，高于阈值使用 HNSW。

ANN 必须返回稳定 chunk ID 和原始相似度，并把 backend、候选规模与参数写入 Trace/manifest。保存索引时采用
原子替换；Embedding model、维度、归一化或 index revision 不匹配时拒绝加载。

### 3.1 ACL 边界

禁止“全库 ANN 召回后删除越权结果”。无 scope 可使用全局 HNSW；有 tenant/policy scope 时优先查询对应
分区索引。当前进程内实现无法安全满足大范围动态 filter 时，回退到授权子集 Exact，并在 Trace 标记
`acl_exact_fallback`。未来接入支持 payload filter 的向量数据库时保持 Pipeline API 不变。

### 3.2 规模验收

- Exact 与 HNSW 在相同向量上的 ANN Recall@K、P50/P95、建库时间和索引大小对比。
- 至少完成 100k 合成向量本机测试；提供可扩展至 2M 的脚本与内存估算。
- 未实际运行 2M 时只报告估算，不报告伪造实测值。

## 4. 中文词法 Analyzer

将 `multilingual_tokenize` 后面的策略抽象为 Analyzer：

- `cjk-bigram`：兼容与无依赖降级。
- `jieba-domain`：Jieba 精确模式 + 领域用户词典 + Latin 技术词保留 + CJK bigram fallback。

领域词典首批覆盖无源互调、功率放大器、数字预失真、神经网络抑制等术语，并支持配置文件扩展。
索引与 query 必须使用同一 Analyzer revision；词典变化要求重建 BM25。测试需证明完整术语被保留，同时
未知中文词仍能通过 bigram 召回。

## 5. Parent-Child Context

结构化摄取路径中 Parent 定义为 section，document 仅作为 section 的上级节点。兼容 PDF 路径如果无法识别
section，必须明确标记 `page-fallback`，不能把页面 Parent 冒充 section。

Parent 展开保持 Child 排名不变，仅扩充送给生成器的 Context：

1. 按已排序 Child 聚合唯一 Parent。
2. 为每个唯一 Parent 分配最低配额，避免首个长 section 吞掉全部预算。
3. 剩余预算按 Child 相关性和 Parent 新颖性加权分配。
4. 围绕命中 Child 在 Parent 中的位置截取双向窗口，不再只截 section 开头。
5. Trace 记录 requested/allocated/used/truncated token 和 fallback 类型。

预算不足时保留 Child 正文；Parent 是否展开不得反向改写检索排名。

## 6. Reranker Profile

| Profile | 模型 | 默认设备策略 |
| --- | --- | --- |
| `cpu-balanced` | 当前 multilingual MiniLM Cross-Encoder | CPU 默认、选择性触发 |
| `quality-gpu` | `BAAI/bge-reranker-v2-m3` | 仅显式 GPU/质量模式 |

Reranker Trace 必须记录模型、真实 device、候选数、batch size、推理耗时与是否命中策略门限。无 GPU 环境
不把大 Reranker 设为默认；本轮重点验证选择性触发和候选上限，不把模型大小当作唯一质量手段。

## 7. 错误处理与兼容

- ANN/Jieba 作为显式依赖 Profile；缺失时给出可操作错误或按配置回退，禁止静默换算法。
- 旧索引因 manifest 字段不足必须明确要求重建。
- 模型下载失败不得污染已有索引；Benchmark 支持 checkpoint/resume。
- 所有公开指标区分 strict matched、directional 和 synthetic evidence 等级。

## 8. 测试与验收

采用 TDD，分四组交付：

1. 模型注册与 10% Embedding 对比。
2. Exact/HNSW backend、ACL 回退和规模脚本。
3. Jieba 领域 Analyzer 与 Parent 公平预算。
4. Reranker Profile、统一 Trace、文档和开发者控制台。

全量离线测试继续禁止真实网络和真实 LLM。真实模型 Benchmark 单独运行并保存到
`C:\Users\yzy\Desktop\codex\paperstorm-benchmarks\retrieval-stack-upgrade\`，不提交模型权重和大体积结果。

最终报告必须提供：同协议模型差异、具体 Top K Bad Case、质量-延迟-内存 Pareto、已经解决的问题和残余风险。

## 9. 自检结论

- 无待定占位符；模型、数据抽样、指标和安全回退均已明确。
- ANN 与 ACL 不冲突：动态授权无法安全近似检索时显式 Exact 回退。
- Public 10% 与 PIM 领域集分开表述，不混淆证据等级。
- 2M 仅提供可运行脚本与估算，除非实际执行后才写实测结果。
