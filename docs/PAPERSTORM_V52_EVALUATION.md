# PaperStorm v5.2 评测诚信与学习记录

## 1. 为什么要重做 Benchmark

旧简历使用的 `0.3625 / 0.2804 / 0.3006 → 0.9875 / 0.8688 / 0.8986`
来自 100 条 synthetic seed，其中只有 80 条是检索用例。数据与 tokenizer、PIM 消歧规则
高度同分布，适合 smoke test 和消融，不足以证明真实论文检索质量。

另一套 Zotero 弱标注集虽然使用真实 PDF，但 query 直接包含论文标题和章节名，标签是
章节出处而非人工 QA；真实向量下 V4.1 Recall@K 为 0.5755，低于 legacy 的 0.5958。
这说明“用了真实 PDF”不等于“评测就是有效的”。

## 2. v5.2 实验协议

### 数据与任务

- 来源：本地 Zotero，只读解析真实 PDF，不提交私有路径或论文全文。
- 语料：40 篇英文论文，868 chunks。
- 任务：中文领域释义 query 检索对应英文论文，目标粒度为 document retrieval。
- 有效集：23 篇论文、46 条唯一 query；跨文档重复 query 被剔除，避免标签冲突。
- 标注：自动候选，逐条保留标题、页码、证据摘要、内容 SHA-256、hard negatives 和
  `needs_human_review`；不是专家 QA 数据集。

### 防止数据泄漏

- 按 `document_id` 切分，34 dev / 12 frozen test，论文不跨集合。
- BM25 / Dense / Hybrid 只在 dev 上比较；按 nDCG、MRR、Recall、延迟顺序选配置。
- test 不参与参数选择；最终选出 Dense 后只在 test 报告一次。
- 保存 corpus / dataset SHA-256、embedding 模型名和 Git commit。
- 每项主指标同时报告样本数与 2,000 次 bootstrap 95% CI。

## 3. 最终结果

| 指标 | BM25 | Dense（dev 选出） |
| --- | ---: | ---: |
| Recall@5 | 0.0000 | 0.4167 |
| MRR | 0.0000 | 0.2986 |
| nDCG@5 | 0.0000 | 0.2463 |
| P95 单 query 延迟 | 约 226ms | 约 227ms |

- Recall@5 95% CI：`[0.1667, 0.6667]`，`n=12`。
- Embedding：`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`。
- Corpus SHA-256：`e3401799ef05e0665cc3b00f0ffef84695289147df0aea5027d82e4a9a494983`。
- Dataset SHA-256：`d1563d79f43eb51323b32dfedbd12fe06d1e79eb5f5734282c5363850a58d88e`。
- Code commit：`121f4d60b39140d508f9413785582804067d7045`。

结论：真实跨语言场景中，Dense 能召回一部分纯 BM25 无法跨语言匹配的论文，但绝对
Recall 仍低、CI 很宽。这个结果可以证明评测与误差分析能力，不能证明系统已经成熟。

## 4. 排查过程中失败过什么

1. 第一版从 chunk 开头取词，生成了 `transactions/wireless`、`photonics/technol` 等
   期刊页眉和 OCR 断词；test Recall@5 只有 0.05。
2. 第二版使用 TF-IDF 并允许同论文多 evidence，Recall@5 提升到 0.50，但仍出现
   `qingdao/chongqing` 作者单位词，问题质量不合格。
3. 第三版改成标题概念 document retrieval；query 清洗后 BM25 达到 1.0，说明任务只在
   测词面匹配、已经饱和，也不能用来证明 Hybrid。
4. 第四版改为中文释义检索英文论文，发现多篇同主题论文产生相同 query，各自只标自己
   为正确，会制造 false negative；最终增加细粒度领域概念并剔除跨文档重复 query。
5. Cross-Encoder 初始化失败，因此正式结果只比较 BM25 / Dense / Hybrid。失败被保留，
   没有把“计划使用 reranker”写成“已验证 reranker”。

## 5. 面试中应该怎么讲

推荐表述：

> 我接手时 synthetic benchmark 分数接近 0.99，但真实 Zotero 弱标注集没有提升。我没有
> 继续调 synthetic，而是重建了真实论文评测协议：文档级 dev/test 隔离、dev 选型、
> frozen test、bootstrap CI、证据哈希与人工审核清单。最终跨语言 pilot 中 Dense 的
> Recall@5 为 0.4167，BM25 为 0；结果不漂亮，但定位出跨语言召回、问题生成和 reranker
> 兼容性是下一步瓶颈。

不要说：

- “真实业务 Recall 从 0.36 提升到 0.99。”
- “错误率和 ACL 泄漏率都是 0，已经生产可用。”
- “Benchmark 已经完成人工专家标注。”

## 6. 契约指标的正确解释

- Context 66.11%：1 个 8-message 构造场景，844→286 tokens。
- Memory 100%：4 个功能契约案例；同一 query 重放 20 次。
- LangGraph 100%：5 条固定路径；恢复、幂等、重试各 1 个注入案例。
- Production 错误率 0：单进程 SQLite 热路径 100 请求，不含真实 LLM、网络和多机并发。

这些结果可以证明机制存在且回归测试通过，不能外推为线上 SLA。P95 也是单机 CPU
参考值，同协议两次运行约 227-246ms，不应包装成稳定延迟承诺。

## 7. 下一步可执行计划

1. 人工审核 46 条 query，删除不自然问题并补充至少 50 条真实用户查询。
2. 将 test 扩到至少 50 条、每个主题不少于 10 条，再报告窄一些的 CI。
3. 修复 Cross-Encoder 模型兼容与缓存，严格在 dev 选模型和 candidate_k。
4. 增加 answer-level faithfulness、citation precision/recall 和 abstention 指标。
5. 增加冷启动、缓存命中、并发 QPS、网络故障注入；与检索质量报告分开。
6. 逐步格式化 legacy 文件；v5.2 CI 先检查本版本发布面，避免一次提交制造 53 个历史
   文件的无关格式变更。

## 8. 复现

```powershell
D:\SOFTWARE\spyder\envs\storm\python.exe -m knowledge_storm.paperstorm_real_eval_v52 `
  --zotero-root $env:PAPERSTORM_ZOTERO_ROOT `
  --output-dir results\paperstorm_real_eval_v52 `
  --embedding real --max-papers 40 --max-pages 5 --max-cases 60 `
  --top-k 5 --test-ratio 0.25 --cross-lingual-only `
  --modes bm25 dense hybrid
```

本地输出包含 `real_paper_dataset_v52.json`、`real_paper_eval_v52.json`、Markdown 报告和
`real_paper_review_candidates.jsonl`。`results/` 被 gitignore，不会上传 Zotero 论文内容。
