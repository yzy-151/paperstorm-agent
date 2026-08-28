# PaperStorm RAG / Agent 面试题库

## 使用说明与指标口径

题目覆盖 RAG 基础、真实排查、系统设计与项目追问。社区面经只用于归纳高频问题；答案依据项目冻结报告、公开论文、官方仓库和模型卡。所有指标必须连同数据集、样本量、Top K、粒度、evaluator 与硬件口径说明，并能还原项目难点的定位过程。

| 类别 | 固定事实 | 正确边界 |
| --- | --- | --- |
| 公开基准 | SciFact recall@10 0.8264（n=300）；QASPER retrieval recall@5 0.5526（n=1309）；独立 full 1451 端到端协议：Answer F1 0.5083、Evidence F1 0.5500、claim support 0.9592、unsupported 0.0214；LongMemEval-S cleaned 500/500 retrieval-only：recall@5 0.8003，P95 359.3 ms。 | 只对应冻结数据、evaluator 与配置，不外推为线上 SLA。 |
| 私有领域 pilot | PIM pilot：5 篇论文 / 797 chunks / 50 题；GTE recall@5 0.7200、Answer F1 0.3983、Raw Citation Precision 0.9237（规则型原始引用映射精度，不是语义/人工验证准确率）。 | 不代表全部 PIM 资料或生产准确率。 |
| 离线治理指标 | 同一小规模 PIM 索引与固定配置下，HNSW recall@5 1.0000。 | 不证明大规模、高并发或任意参数下无损。 |

禁止夸大：公开基准、私有领域 pilot、离线治理指标必须分开陈述，未解决的回归也要主动说明。

## 基础原理

### 1. 什么场景适合 RAG，什么场景更适合微调？
**参考回答**：RAG 解决知识更新、私有数据、引用和 ACL；微调更适合稳定行为与输出格式，两者可组合。
**项目实例**：PaperStorm 用 RAG 接入 arXiv、本地 PDF 与 Memory，不把论文事实固化进参数。
**排查/设计步骤**：区分知识缺失与行为偏差，再评估更新频率、权限、引用要求和训练成本。
**追问**：什么时候应直接聊天而不检索？
**考察点**：技术边界与需求建模。
**常见失误**：声称 RAG 能消除幻觉。

### 2. 生产 RAG 链路应如何分层？
**参考回答**：解析与 Chunk、索引、查询理解、BM25/Dense、RRF、Rerank、Parent Context、证据门控、生成、引用校验、Trace 与 Benchmark。
**项目实例**：PaperStorm 为每层保存中间产物、版本、耗时和错误。
**排查/设计步骤**：先看 gold 在哪层丢失，再定位召回、排序、预算或生成。
**追问**：为什么最终 Answer F1 无法定位根因？
**考察点**：端到端分层。
**常见失误**：只调 Prompt。

### 3. 为什么不能按固定字符数切分所有文档？
**参考回答**：固定字符会切断标题、定义、表格、公式和论证关系，应优先遵循文档结构和 token 边界。
**项目实例**：本地 PDF 使用结构化 Chunk，child 精确检索，parent 恢复连续上下文。
**排查/设计步骤**：固定其他变量，对比 Recall、nDCG、Evidence F1、token、P95 和跨段 badcase。
**追问**：双栏扫描 PDF 怎么处理？
**考察点**：解析质量与因果实验。
**常见失误**：凭经验固定 500 字。

### 4. overlap 应怎样设定？
**参考回答**：overlap 保留跨边界语义；过大则制造近重复并增加索引与重排成本，应按句长、结构与 tokenizer 调参。
**项目实例**：PaperStorm 用 parent-child 邻域补上下文，不依赖无限增大 overlap。
**排查/设计步骤**：扫描 overlap，观察跨段 Recall、重复候选率、上下文 token 和延迟 Pareto。
**追问**：中文字符数为何不等于 token 数？
**考察点**：边界语义和成本。
**常见失误**：认为 overlap 越大越安全。

### 5. 中文 BM25 为什么需要 Jieba 与领域词典？
**参考回答**：纯 CJK bigram 会把“无源互调”拆碎；Jieba 加领域词典能保留术语，bigram 适合作无依赖回退。
**项目实例**：PaperStorm 为 PIM、DPD 等 RF 术语维护 analyzer profile 与 fingerprint。
**排查/设计步骤**：检查 query/gold token，并对 Jieba、bigram、领域词典做精确术语 Recall 对比。
**追问**：领域新词如何持续更新？
**考察点**：Sparse 检索与中文工程。
**常见失误**：只换 Dense，忽略词法召回。

### 6. Dense 检索为什么不能完全替代 BM25？
**参考回答**：Dense 擅长语义改写，BM25 擅长型号、缩写、数字和专有词，两者错误面互补。
**项目实例**：QASPER 自然语言问题受益于 Dense，PIM 型号和 RF 术语仍依赖 Sparse。
**排查/设计步骤**：按术语型、改写型、数字型分桶，比较 Sparse、Dense、Hybrid 的 Recall 与排名。
**追问**：Hash embedding 能验证什么？
**考察点**：模型能力边界。
**常见失误**：把 Hash fallback 当真实语义检索。

### 7. Embedding 模型如何选择？
**参考回答**：比较语言覆盖、训练目标、上下文长度、维度、指令、吞吐、许可和领域效果，模型 revision 必须进入索引 fingerprint。
**项目实例**：PaperStorm 验证 Legacy MiniLM、BGE、GTE、Qwen；CPU 默认偏向 GTE，Qwen 质量高但更慢。
**排查/设计步骤**：固定语料、Chunk、TopK 与 evaluator，只替换 Profile，报告 Recall、nDCG、P95 与 case movement。
**追问**：为什么不能只看 MTEB 排名？
**考察点**：Embedding 选型。
**常见失误**：不同索引和指令混在一起比较。

### 8. 向量归一化和距离度量如何选择？
**参考回答**：若目标是 cosine，可 L2 归一化后用内积；若模型契约指定其他度量则遵从，索引与查询侧必须一致。
**项目实例**：PaperStorm manifest 保存维度、归一化、metric 与 revision，拒绝静默复用不兼容索引。
**排查/设计步骤**：检查范数分布、metric 和查询实现，用 exact 搜索建立 gold 再核对 ANN。
**追问**：归一化后欧氏距离与 cosine 有什么关系？
**考察点**：向量检索基础。
**常见失误**：索引和评测使用不同度量。

### 9. Hybrid 检索为什么常优于单路检索？
**参考回答**：BM25 保留词法精确性，Dense 提供语义泛化；融合覆盖不同 query，但增加候选量和延迟。
**项目实例**：本地 PDF 与问答使用 BM25 + Dense + RRF；arXiv 首召回仍受上游 API 约束。
**排查/设计步骤**：分别测两路 Recall、union oracle、融合排名和额外延迟，按 query 类型分桶。
**追问**：什么场景只用 BM25 更合理？
**考察点**：互补召回。
**常见失误**：默认 Hybrid 对全部数据都提升。

### 10. RRF 的原理和局限是什么？
**参考回答**：RRF 用 `1/(k+rank)` 融合不同量纲排名，不依赖分数标定；但会丢失原始 margin，不能判断候选是否真的相关。
**项目实例**：PaperStorm 曾重复 RRF 压平 margin 并导致 100% 重排，后改为仅多查询聚合时再次融合。
**排查/设计步骤**：记录各路 rank、贡献和 margin，比较重复融合前后的触发率与 TopK movement。
**追问**：加权 RRF 何时有意义？
**考察点**：排序融合。
**常见失误**：把 RRF 当学习排序或简单倒序。

### 11. Cross-Encoder Rerank 放在哪一层？
**参考回答**：放在高召回候选之后、上下文选择之前，对 query-document 联合编码；质量通常更高，但计算随候选数增加。
**项目实例**：PaperStorm 在 Hybrid/RRF 后选择性触发 Cross-Encoder，再进入 Parent Context。
**排查/设计步骤**：固定候选池，比较前后 Recall/nDCG、改善/回归案例、触发率和 P95。
**追问**：为什么不能对全库做 Cross-Encoder？
**考察点**：两阶段检索与性能。
**常见失误**：把 Rerank 说成倒序排列。

### 12. 选择性 Rerank 门限如何设计？
**参考回答**：可根据 margin、查询歧义、Sparse/Dense 分歧和预算触发，门限必须离线标定并线上观测。
**项目实例**：修复后 SciFact 触发率 36.33%，QASPER 45.68%，避免无差别推理。
**排查/设计步骤**：绘制质量、触发率、P95 与成本曲线，超时则回退融合排序。
**追问**：如何防止门限过拟合单一数据集？
**考察点**：动态策略与降级。
**常见失误**：只看平均延迟。

### 13. HNSW 为什么能加速 Dense 检索？
**参考回答**：HNSW 用多层近邻图导航，避免全库线性扫描；代价是内存、构建时间和近似误差。
**项目实例**：PaperStorm 提供 exact/HNSW 双后端，并用固定小集比较 ANN 与 exact TopK。
**排查/设计步骤**：扫描 M、efConstruction、efSearch，记录 Recall@K、P95、内存和构建时间。
**追问**：两百万 Chunk 时还要考虑什么？
**考察点**：ANN 与容量规划。
**常见失误**：用小库 1.0000 宣称大库无损。

### 14. IVF 与 HNSW 如何取舍？
**参考回答**：HNSW 低延迟高召回但内存大；IVF/PQ 更省内存且适合批量索引，但需训练并调 nprobe。
**项目实例**：PaperStorm 当前本地演示采用 HNSW，百万级扩展再评估分片与 IVF/PQ。
**排查/设计步骤**：同一 exact gold 下比较 Recall/QPS/P95/内存/更新成本。
**追问**：频繁增删文档如何影响选择？
**考察点**：索引系统设计。
**常见失误**：只比单次查询速度。

### 15. Parent-Child 检索为什么合理？
**参考回答**：child 提供高分辨率匹配，parent 恢复论证上下文，兼顾召回、可读性和引用定位。
**项目实例**：PaperStorm 保存 child-parent 映射、邻域、预算和截断原因。
**排查/设计步骤**：检查 gold child、parent 展开、重复 parent 和证据覆盖，对比 child-only。
**追问**：parent 应是 section、page 还是窗口？
**考察点**：多粒度检索。
**常见失误**：无条件展开整篇论文。

### 16. Context Compression 解决什么问题？
**参考回答**：过滤无关 Chunk、压缩冗余并保留可追溯证据，减少 token 与干扰；风险是丢失数字、限定和否定。
**项目实例**：PaperStorm 先粗过滤再结构化压缩，保存 source span 和原文回退路径。
**排查/设计步骤**：比较压缩前后 Evidence F1、Answer F1、token、TTFT 和关键事实丢失率。
**追问**：何时不应调用 LLM 压缩？
**考察点**：上下文工程。
**常见失误**：只报告 token 降低。

### 17. 结构化摘要为什么优于自由摘要？
**参考回答**：结构化字段显式保留实体、数值、单位、时间、否定、结论和来源，便于校验与检索。
**项目实例**：PaperStorm 摘要包含 facts、open_questions、citations 和 temporal validity。
**排查/设计步骤**：用冲突数字、否定和时效事实检查字段完整率与来源回绑。
**追问**：JSON 输出解析失败怎么办？
**考察点**：Prompt 契约与容错。
**常见失误**：只写“简要总结”。

### 18. Memory 与 Evidence 的边界是什么？
**参考回答**：Evidence 支持当前外部事实并可引用；Memory 保存用户偏好、事件和会话事实，不能代替论文证据。
**项目实例**：问“之前聊过的 PIM 论文”先由 Memory 找会话/文档 ID，再从 Evidence 取原文。
**排查/设计步骤**：按来源、生命周期、ACL、可引用性和冲突规则定义 schema。
**追问**：“我更喜欢中文回答”属于哪层？
**考察点**：证据与记忆治理。
**常见失误**：把模型摘要当权威 Evidence。

### 19. 短期、摘要与长期 Memory 如何分层？
**参考回答**：短期窗口保存近期原文；摘要压缩已完成阶段；长期层保存可复用事实、偏好和事件并带 provenance、TTL 与删除能力。
**项目实例**：PaperStorm 组合 Recent、FTS Session 和可选真实 Dense Memory。
**排查/设计步骤**：定义写入、去重、冲突、TTL、TopK 与删除传播，再测跨会话召回。
**追问**：为何不永久保存全部原始聊天？
**考察点**：生命周期与隐私。
**常见失误**：把 Summary 与 Memory 混为一谈。

### 20. Memory 召回需要 Dense 吗？
**参考回答**：不必强制；实体、编号与近期历史用 FTS/BM25 足够，语义改写可选真实 Embedding。
**项目实例**：PaperStorm 可比较 Recent、FTS Session 和 Dense Memory Profile，Hash 仅供测试。
**排查/设计步骤**：按精确实体、语义改写、时间约束分桶，比较 Recall、延迟与误召回。
**追问**：Context 历史为何不必每轮 Dense？
**考察点**：召回信号与成本。
**常见失误**：认为所有文本都必须向量化。

### 21. Agent 路由为何应从内容分类转向动作决策？
**参考回答**：内容类别会无限扩张；动作层只决定回答、查 Memory、查 Evidence、调用 Tool、启动 Research 或澄清，并允许组合。
**项目实例**：PaperStorm 的结构化 LLM decision 输出 actions、confidence、reason、error_type，而非硬编码“故事续写”。
**排查/设计步骤**：用闲聊、续写、事实、歧义和失败集检查动作与误触发 Research 的原因。
**追问**：LLM 路由失败如何回退？
**考察点**：开放意图空间与 Agent Runtime。
**常见失误**：不断增加 if/else 类别。

### 22. 规则分类与 LLM 路由各有什么优缺点？
**参考回答**：规则低延迟、确定、易审计，适合安全边界；LLM 能处理开放语义但有成本和不稳定性。工业系统常规则守底线、LLM 选动作。
**项目实例**：ACL、危险工具和空输入走规则，聊天/检索/Research 由 LLM 决策。
**排查/设计步骤**：记录 rule hit、decision、置信度、最终动作和反馈，按 error_type 维护数据集。
**追问**：为什么不能让 LLM 决定 ACL？
**考察点**：确定性边界。
**常见失误**：全规则或全模型两个极端。

### 23. Tool Calling Schema 为什么重要？
**参考回答**：Schema 约束参数和错误契约；Runtime 还必须处理超时、重试、幂等、权限和审计。
**项目实例**：PaperStorm 将检索、问答、调研封装为结构化 Tool，并提供 MCP 适配。
**排查/设计步骤**：测试缺字段、非法路径、重复请求、超时和副作用，追踪 tool_call_id。
**追问**：写操作为何必须有幂等键？
**考察点**：工具系统可靠性。
**常见失误**：只证明模型能调用函数。

### 24. Multi-Agent 什么时候有价值？
**参考回答**：任务可并行、角色有不同上下文或需要互审时有价值；重复调用同一模型只会增加成本和协调错误。
**项目实例**：STORM 的 Persona Generator、Conv Simulator、WikiWriter 协作完成视角、访谈、大纲和写作。
**排查/设计步骤**：与单 Agent 比覆盖率、证据多样性、质量、token、P95 和失败率。
**追问**：怎样防止讨论循环？
**考察点**：协作边界。
**常见失误**：用 Agent 数量包装复杂度。

### 25. Agent Runtime 与固定 Workflow 有什么区别？
**参考回答**：Workflow 适合已知控制流；Runtime 还管理动态动作、状态、Checkpoint、取消、重试、预算和事件，两者可组合。
**项目实例**：文章调研是可恢复 Workflow，问答支持动态检索和 Research 升级。
**排查/设计步骤**：标出确定步骤与开放决策点，分别设置状态机、预算和终止条件。
**追问**：为什么不全部改成 LangGraph？
**考察点**：架构取舍。
**常见失误**：框架名称代替能力证明。

### 26. Trace 与普通日志有什么区别？
**参考回答**：日志记录事件；Trace 用 trace/span 组织因果链，关联输入输出摘要、模型、token、耗时、错误和评分。
**项目实例**：PaperStorm 保留本地 Trace 与 Langfuse observation，外部平台失败时业务 fail-open。
**排查/设计步骤**：从失败回答沿路由、召回、rerank、压缩、生成和引用逐 span 定位。
**追问**：哪些内容不能上传 Langfuse？
**考察点**：可观测性与隐私。
**常见失误**：上传完整私有文档或密钥。

### 27. Benchmark 如何避免泄漏和刷分？
**参考回答**：冻结数据、split、evaluator、revision 与配置；开发集调参，测试集只作最终验证，报告置信区间和失败案例。
**项目实例**：PaperStorm 用 fingerprint 区分 SciFact、QASPER、LongMemEval-S 与 PIM pilot。
**排查/设计步骤**：审计题源、去重、训练重叠、缓存和人工挑例，保存运行 manifest。
**追问**：LLM 生成的 50 题能否叫公开 Benchmark？
**考察点**：评测可信度。
**常见失误**：把私有 pilot 包装成公开成绩。

### 28. Recall、MRR、nDCG、Answer F1 分别测什么？
**参考回答**：Recall 看 TopK 覆盖；MRR 看首个相关项位置；nDCG 看分级相关和排序；Answer F1 看答案 token 重合，不能互相替代。
**项目实例**：PaperStorm 将 QASPER retrieval 与独立端到端协议分开报告。
**排查/设计步骤**：按链路选择指标并固定 K、粒度、evaluator，下降时分析 query-level movement。
**追问**：Evidence F1 为何不等于引用正确？
**考察点**：指标定义。
**常见失误**：只报最好看的一个分数。

### 29. 为什么 CI 禁止真实网络与 LLM？
**参考回答**：网络和模型输出不确定、昂贵、慢且涉及密钥；CI 用 fixture/fake 验证契约，真实集成放 nightly 或人工任务。
**项目实例**：PaperStorm 离线测试用 Hash embedding/fake LLM，公开 Benchmark 单独运行。
**排查/设计步骤**：强制 offline，mock 外部边界而非内部算法，另设 integration marker 和归档。
**追问**：如何避免 fake 测试掩盖集成故障？
**考察点**：测试金字塔。
**常见失误**：单元测试下载模型或调用 API。

### 30. RAG 发布门禁如何设计？
**参考回答**：同时约束质量、关键回归、P95、成本、安全和恢复；总体提升不能掩盖关键 query 退化。
**项目实例**：PaperStorm 记录 Recall、P95、rerank 触发率、引用治理和 improved/regressed cases。
**排查/设计步骤**：冻结基线，递进实验，审查 badcase 后发布并保留回滚产物。
**追问**：质量提升但 P95 增加十倍怎么办？
**考察点**：多目标工程决策。
**常见失误**：只按平均分发布。

## Bad Case 与排查

### 31. PIM / RAM / DRAM 跑题是怎样发现和修复的？
**参考回答**：输入“PIM 神经网络抑制”却召回 processing-in-memory/DRAM，说明缩写歧义污染首召回；通过 RF 域 SearchPlan、must/negative terms 和查询扩展后，RF 文档升至 Top1、forbidden hit=0。
**项目实例**：真实案例把 PIM 解析为 passive intermodulation，并排除 RAM/DRAM。
**排查/设计步骤**：复现原 query，查看上游 arXiv 结果、token、候选域标签与最终排名；逐步加入域约束并回归非 RF PIM。
**追问**：“PIM 有哪些应用”怎么处理？
**考察点**：歧义消解与边界。
**常见失误**：强制把所有 PIM 都解释成无源互调；无上下文歧义仍应澄清。

### 32. 重复 RRF 为什么导致 100% 重排？
**参考回答**：内部 Hybrid 已有融合分数，外层再次 RRF 把分差压平，margin 门限误判为不确定，导致每条 query 都触发 Cross-Encoder。
**项目实例**：修复后只在多查询合并时 RRF，SciFact/QASPER 触发率分别降至 36.33%/45.68%。
**排查/设计步骤**：对比两次融合前后的 rank、score、margin、trigger_reason 和 P95；写 100% 重排回归测试。
**追问**：为何不能直接提高门限掩盖问题？
**考察点**：分数语义与链路定位。
**常见失误**：只优化模型速度，不查触发逻辑。

### 33. MMR 为什么让 QASPER Recall@5 降到 0.4631？
**参考回答**：旧 MMR 在整个候选池重选 TopK，为追求多样性把已召回 gold 替换掉，优化目标破坏召回门禁。
**项目实例**：改为 recall-safe MMR，只在冻结 TopK 内重排，Recall@5 恢复到 0.5526。
**排查/设计步骤**：比较 MMR 前后 gold rank、被替换候选和相似度；限制候选域后复跑 n=1309。
**追问**：recall-safe 是否牺牲了潜在提升？
**考察点**：多样性与召回约束。
**常见失误**：看到“去重”就默认不会伤质量。

### 34. QASPER 低词面重叠问题如何解决？
**参考回答**：问题与证据表达不同，Sparse 无法把 gold 送入前列；Dense + Hybrid + 选择性 Rerank 增强语义匹配。
**项目实例**：“what evaluation metrics did they use?” 的 gold 从 Top5 外升至 Top1。
**排查/设计步骤**：检查 BM25/Dense 各自排名、union oracle、RRF 贡献和 Cross-Encoder 分数，固定 Chunk 做递进实验。
**追问**：如何证明不是偶然个例？
**考察点**：语义召回和案例归因。
**常见失误**：只展示成功截图，不给全量指标。

### 35. SciFact Vitamin D 案例为什么改善？
**参考回答**：claim 与论文措辞存在改写，单词法排序不足；Dense 补充语义候选，Rerank 识别 claim-evidence 关系。
**项目实例**：该 gold 从 Top10 外升至 Top1，属于 P2 improved case。
**排查/设计步骤**：保存各阶段候选与分数，对比仅 Dense、Hybrid、Hybrid+Rerank，确认提升发生在哪层。
**追问**：怎样避免拿成功案例代替总体结论？
**考察点**：证据链和消融。
**常见失误**：把单例提升写成所有医学 claim 均正确。

### 36. Cross-Encoder 误排 PPM1D 为什么尚未完全解决？
**参考回答**：Cross-Encoder 对局部措辞给错误高分，把原第 3 的 gold 推出 Top10；总体指标提升不能掩盖这一回归。
**项目实例**：PPM1D 被记录为 unresolved regression，并进入 golden badcase。
**排查/设计步骤**：比较融合分数与 rerank logits，检查截断、输入顺序、模型域偏差；候选门控和保序策略需后续验证。
**追问**：是否应该直接关闭 Rerank？
**考察点**：局部回归与诚实边界。
**常见失误**：删掉失败样本或声称全部解决。

### 37. “How significant are improvements” 回归说明什么？
**参考回答**：抽象评价问题需要完整实验上下文，短 child 的词面不足，Reranker 可能偏好含“improvement”的错误段落。
**项目实例**：该 QASPER gold 从 Top1 跌出 Top5，是 P2 的 unresolved case。
**排查/设计步骤**：检查 parent 展开、表格/结果段结构与 hard negatives；加入 query 类型标签后单独评估。
**追问**：Parent Context 能否修复排序阶段的丢失？
**考察点**：结构证据与阶段边界。
**常见失误**：gold 已被丢弃后还指望生成模型恢复。

### 38. Parent Context 预算饥饿怎样发生？
**参考回答**：高分 child 的 parent 较长，前几个 parent 耗尽预算，后续高相关 child 无法展开，最终上下文排序失真。
**项目实例**：PaperStorm 增加 unique-parent 最低配额、分数加权剩余预算与 child 邻域展开。
**排查/设计步骤**：Trace allocation/used/truncated/reason，复现相同 parent 集，比较公平分配前后的 Evidence 覆盖。
**追问**：parent 是整页还是 section？
**考察点**：预算调度与可解释性。
**常见失误**：只按候选顺序贪心塞满。

### 39. 引用映射为何会显示“Generated article paragraph”？
**参考回答**：生成段落只保存内部 placeholder，未稳定回绑原始标题、作者、URL/页码，UI 只能展示合成标签。
**项目实例**：修复后 Citation 对象携带原文 title、authors、source locator，点击定位 Evidence span。
**排查/设计步骤**：从生成 citation ID 反查 evidence registry、URL 与 UI anchor，增加缺失元数据和断链测试。
**追问**：规则型 Raw Citation Precision 测的是什么？
**考察点**：provenance 与前后端契约。
**常见失误**：把能点击等同于语义支持正确。

### 40. PDF 公式为何生成后不显示？
**参考回答**：Markdown/LaTeX、HTML 渲染器和 PDF 引擎的数学支持不一致，公式可能被转义或缺少 MathJax/LaTeX 依赖。
**项目实例**：PaperStorm 统一公式规范化，HTML 预览用数学渲染，PDF 路径用支持公式的转换器并保留源码。
**排查/设计步骤**：分别检查模型原始输出、Markdown AST、HTML、打印 PDF；用行内、块级、编号公式做视觉回归。
**追问**：无 LaTeX 环境怎样降级？
**考察点**：多格式交付链路。
**常见失误**：只检查 txt 文件中存在 `$`。

### 41. 闲聊为何被旧 PIM topic 误导成调研？
**参考回答**：旧路由把持久 topic 当当前意图的强特征，而可见近期上下文太短，导致“写故事”等开放请求被旧主题劫持。
**项目实例**：PaperStorm 改为动作路由，topic 只作可选 evidence hint，普通聊天默认不检索。
**排查/设计步骤**：回放跨轮会话，检查 router 输入、summary、decision 与触发原因；加入 topic 污染回归集。
**追问**：何时应更新 topic？
**考察点**：状态污染与路由。
**常见失误**：继续添加“故事”关键词规则。

### 42. 续写中途截断如何定位？
**参考回答**：先区分 finish_reason=length、网络流中断、客户端渲染丢包或 Runtime 取消；若是 length，应按动作分配更高输出预算并支持 continuation。
**项目实例**：PaperStorm 将输出 token 从固定小值改为 profile 动态预算，保存 finish_reason 和已用 token。
**排查/设计步骤**：检查模型响应、SSE 序号、服务日志和浏览器消息长度；对长故事做端到端回归。
**追问**：动态上限比统一 4000 好在哪里？
**考察点**：生成预算和流式链路。
**常见失误**：只把上限改大，不保留上下文与成本空间。

### 43. “你好，我是 PaperStorm”重复出现的根因是什么？
**参考回答**：错误回退 Prompt 把任意路由/解析异常转成统一自我介绍，掩盖真实 error_type；也可能新建了空会话状态。
**项目实例**：改为结构化错误类型、保留原请求并选择聊天回退或明确失败，不再统一问候。
**排查/设计步骤**：关联 router parse_error、session_id、fallback_reason 和最终文本，统计重复模板来源。
**追问**：用户可见错误应暴露多少内部信息？
**考察点**：错误语义与体验。
**常见失误**：用友好文案吞掉系统错误。

### 44. Memory 跨会话召回失败怎样定位？
**参考回答**：可能是未写入、写错 tenant/user、summary 未入索引、FTS/Dense 未命中、时间过滤或 ACL 拒绝，必须分层检查。
**项目实例**：PaperStorm trace 记录 memory write、index generation、retrieval source、filter 和 selected IDs。
**排查/设计步骤**：先查原始记录与 ACL，再查 analyzer/embedding fingerprint、候选与融合，最后看 Prompt 是否使用。
**追问**：为什么只取最后两条 summary 不是真正召回？
**考察点**：Memory 数据路径。
**常见失误**：直接把 TopK 调大。

### 45. Summary 丢失否定和数字怎样修复？
**参考回答**：自由摘要 Prompt 未规定保真字段，模型会压缩限定词；需结构化事实、数值、单位、否定、来源和不确定性，并做回绑校验。
**项目实例**：PaperStorm 的 Context 摘要增加 schema、source span 与压缩失败回退。
**排查/设计步骤**：用最小对照样本比较原文和摘要字段，统计遗漏类型与下游答案影响。
**追问**：什么时候应保留原文不压缩？
**考察点**：上下文保真。
**常见失误**：用更长 Prompt 但没有可验证字段。

### 46. Evidence 冲突时系统应怎么回答？
**参考回答**：先做实体、时间、条件和研究设计对齐，保留双方来源；能判定时按新鲜度和证据等级解释，不能判定则明确冲突与不确定性。
**项目实例**：PaperStorm 的 evidence set 保存 stance 与 temporal validity，生成层不得静默投票。
**排查/设计步骤**：检查是否同一 claim、是否版本过期、来源独立性和引用覆盖；构造冲突 gold。
**追问**：多数来源一致就一定正确吗？
**考察点**：冲突治理。
**常见失误**：按相似度最高或数量最多直接选边。

### 47. ACL 泄漏应如何定位与修复？
**参考回答**：ACL 必须在候选进入排序和缓存前 fail-closed 过滤；仅在生成前过滤会让分数、摘要或 Trace 泄漏内容。
**项目实例**：PaperStorm 将 tenant/user/document 权限下推到 retrieval adapter，并用不可见文档做负向测试。
**排查/设计步骤**：沿索引、缓存、召回、rerank、trace、citation 检查 document_id；模拟权限撤销与并发请求。
**追问**：删除请求如何传播到向量索引？
**考察点**：多租户安全。
**常见失误**：让 LLM 判断访问权限。

### 48. arXiv 单条检索失败通常是什么？
**参考回答**：可能是限流、超时、解析异常、无结果、PDF 下载失败或单篇元数据损坏；单条失败不应拖垮整批任务。
**项目实例**：PaperStorm 为 arXiv adapter 增加超时、有限重试、错误分类和部分成功汇总。
**排查/设计步骤**：记录 query、HTTP 状态、阶段、重试次数和 paper ID；区分网络、上游和本地解析。
**追问**：哪些错误不应重试？
**考察点**：外部依赖容错。
**常见失误**：捕获所有异常后返回空列表。

### 49. Wikipedia 辅助抓取失败为什么不应导致主流程失败？
**参考回答**：它是可选补充源，不是论文调研的核心证据；应隔离故障并明确降级，而非把异常伪装成正常结果。
**项目实例**：PaperStorm 将可选 source 标记 degraded，继续 arXiv/Local PDF 并在报告中记录缺口。
**排查/设计步骤**：检查 DNS、状态码、解析器与重试；验证降级结果和告警是否可见。
**追问**：什么时候必须 fail-closed？
**考察点**：依赖分级。
**常见失误**：所有外部源使用同一失败策略。

### 50. Pydantic serializer warning 为什么出现？
**参考回答**：SDK 返回对象与声明 schema 版本不匹配，序列化仍可能继续，但产物字段可能缺失；根因常是依赖版本或兼容适配。
**项目实例**：PaperStorm 将第三方 Message 转成内部 DTO，锁定兼容依赖并降低无用告警噪声。
**排查/设计步骤**：记录实际类型和版本，构造最小序列化测试，不用全局忽略掩盖真正错误。
**追问**：Warning 与 Error 的发布策略有何不同？
**考察点**：依赖兼容与日志治理。
**常见失误**：看到流程有输出就完全忽略 warning。

### 51. 检索延迟升高怎样判断是否来自 Cross-Encoder？
**参考回答**：拆分 embedding、Sparse、ANN、fusion、rerank 各 span，查看触发率、候选数、batch、设备和 P50/P95，而非凭总耗时猜测。
**项目实例**：P2 中 SciFact P95 从 70.56 ms 到 1376.54 ms，QASPER 从 27.83 ms 到 1029.34 ms，主要代价与选择性 Rerank 相关。
**排查/设计步骤**：固定 query 重放，关闭/开启 rerank、缩候选、批处理并核对 CPU/GPU provider。
**追问**：上 GPU 一定能改善端到端 P95 吗？
**考察点**：性能剖析。
**常见失误**：把所有 Hybrid 延迟都归因于 BM25。

### 52. Embedding 换成新模型后指标下降怎么办？
**参考回答**：新模型公开排名高不保证领域更好；还可能有错误指令、归一化、语言、截断或旧索引混用。
**项目实例**：PaperStorm 小样本中 BGE 并非所有数据都优于 Legacy，Qwen 质量较高但 CPU 延迟显著。
**排查/设计步骤**：验证 fingerprint 和 query/document 模板，重建索引，按 query 分桶并检查置信区间。
**追问**：何时可保留多 Profile 而非唯一模型？
**考察点**：模型迁移。
**常见失误**：只因为模型年份新就全量替换。

### 53. UI 显示成功但服务实际 500 如何定位？
**参考回答**：模块级 try/except 将 create_app 失败吞成 `app=None`，Uvicorn 启动进程不等于 ASGI 应用可用；应 fail-fast 并加 readiness。
**项目实例**：PaperStorm 删除静默吞异常，缺 FastAPI/Uvicorn 时启动直接给出依赖错误。
**排查/设计步骤**：先请求 `/health/ready`，检查 ASGI app 类型与启动日志，再查首个异常堆栈。
**追问**：liveness 与 readiness 有何区别？
**考察点**：服务诊断。
**常见失误**：看到“startup complete”就认为业务正常。

### 54. 前端样式 404 为什么会呈现无排版页面？
**参考回答**：HTML 能返回，但 CSS/JS 静态目录未挂载或路径相对错误，浏览器只能显示裸 DOM。
**项目实例**：PaperStorm 将 Dashboard 静态资源由同一 FastAPI app 提供，并加 root、CSS、JS smoke test。
**排查/设计步骤**：看浏览器 Network 与服务 404，核对 mount、base path、缓存和 Content-Type。
**追问**：为什么直接双击 HTML 又会遇到 CORS？
**考察点**：前后端资源边界。
**常见失误**：只刷新页面，不查静态请求。

### 55. 如何把 Bad Case 变成可持续改进资产？
**参考回答**：记录输入、预期、实际、错误层、根因、修复、指标影响和残余边界，纳入 golden set、Trace 标签和发布门禁。
**项目实例**：PaperStorm 保存 improved/regressed 案例；SciFact P2 改善 9、回归 1，QASPER 改善 151、回归 52。
**排查/设计步骤**：线上/人工发现后最小复现，分层定位，写测试，递进 Benchmark，版本化数据与决策。
**追问**：如何避免 Bad Case 集过拟合？
**考察点**：质量闭环。
**常见失误**：只修截图中的一句话，不做同类分桶。

## 假设性系统设计

### 56. 如何设计多租户企业知识库 Agent？
**参考回答**：文档接入、解析索引、ACL-aware retrieval、问答 Runtime、Evidence/Citation、审计与评测分层；权限在召回前 fail-closed。
**项目实例**：可复用 PaperStorm 的本地 PDF、Hybrid、Evidence、Trace 与会话层，替换企业身份和存储适配器。
**排查/设计步骤**：先澄清租户、QPS、SLA、更新/删除，再设计索引隔离、缓存键、降级和泄漏测试。
**追问**：共享索引与每租户独立索引如何取舍？
**考察点**：RAG 平台与 ACL。
**常见失误**：只画向量库和 LLM 两个框。

### 57. 如何设计百万级知识库检索？
**参考回答**：离线解析与增量索引，Sparse 倒排加 ANN，metadata/ACL 预过滤，两阶段检索、热点缓存和分片副本。
**项目实例**：PaperStorm 的 exact/HNSW 抽象可扩展到 Milvus/FAISS/pgvector，但需真实容量验证。
**排查/设计步骤**：给出语料规模、更新率、Recall、QPS、P95、内存和恢复目标，做容量压测与 exact 抽样。
**追问**：两百万 Chunk 为什么不能线性扫？
**考察点**：规模化检索。
**常见失误**：拿 797 Chunk pilot 外推百万规模。

### 58. 如何设计高并发 RAG？
**参考回答**：异步 I/O、请求队列、embedding/rerank batch、连接池、缓存、限流、背压、超时与分级降级，生成流式返回。
**项目实例**：PaperStorm Service 的任务状态、SSE、幂等和可取消执行可作为控制面基础。
**排查/设计步骤**：定义 QPS、并发、TTFT/P95、token 和成本；分别压测检索、Rerank、LLM 与端到端。
**追问**：缓存怎样避免 ACL 泄漏？
**考察点**：性能与稳定性。
**常见失误**：只增加线程数。

### 59. 如何设计论文调研 Agent？
**参考回答**：问题澄清、SearchPlan、多视角查询、论文召回、证据治理、大纲、章节并行写作、引用校验和质量评估。
**项目实例**：PaperStorm 在 STORM Multi-Agent 主链上扩展 arXiv、本地 PDF、Runtime、RAG 与 Benchmark。
**排查/设计步骤**：定义来源许可、时间预算、覆盖/可信指标、失败降级和可恢复 Checkpoint。
**追问**：多视角角色是否一定由多个模型实例承担？
**考察点**：Agentic Research 设计。
**常见失误**：直接让一个超长 Prompt 写整篇文章。

### 60. 如何设计可自动补充检索的客服 Agent？
**参考回答**：动作路由先判断可直接回答、查 Memory、查 Evidence、澄清或转人工；低证据时自动检索但受权限、预算和拒答门控。
**项目实例**：PaperStorm 问答支持普通聊天、知识问答与 Research 升级，而非先手工提交任务。
**排查/设计步骤**：建立意图/动作集、证据充分性门限、业务工具、升级策略与会话评测。
**追问**：如何避免每句话都触发检索？
**考察点**：对话与检索融合。
**常见失误**：把所有非“你好”请求当知识问答。

### 61. 如何设计长期记忆 Agent？
**参考回答**：分 Recent、Session Summary、Long-term Fact/Event，支持写入策略、provenance、时间有效性、冲突、ACL、删除和混合召回。
**项目实例**：PaperStorm 用 LongMemEval-S 检查跨会话 retrieval，并区分 Memory 与 Evidence。
**排查/设计步骤**：先定义记忆价值和隐私，再实现候选写入、确认、索引、召回、注入和遗忘闭环。
**追问**：用户纠正旧偏好时怎样更新？
**考察点**：Memory 工程。
**常见失误**：把全量聊天向量化就叫长期记忆。

### 62. 如何设计长上下文预算管理？
**参考回答**：预留 system/tool/output，再按任务动态分配 recent、summary、Memory、Evidence；高相关证据优先，超限先过滤后结构化压缩。
**项目实例**：PaperStorm 提供 128K/256K/512K Profile，而不是把模型 1M 窗口每次塞满。
**排查/设计步骤**：测质量、输入 token、TTFT 与成本 Pareto，记录每层 requested/used/truncated。
**追问**：模型支持 1M，为何仍需压缩？
**考察点**：上下文容量与有效上下文。
**常见失误**：把最大窗口当免费且无干扰。

### 63. 如何设计带工具的 Agent Runtime？
**参考回答**：状态机/图管理动作，Tool Registry 管 schema 与权限，执行器管超时、重试、幂等、取消、预算、Checkpoint 与事件。
**项目实例**：PaperStorm 的调研任务、问答与 MCP Tool 可由同一 Runtime 观测和恢复。
**排查/设计步骤**：先列副作用和失败语义，再设计状态、事件、幂等键、补偿与人工介入。
**追问**：网络超时后写操作是否能直接重试？
**考察点**：生产级 Harness。
**常见失误**：把 while-loop 当完整 Runtime。

### 64. 如何设计离线部署的 RAG Agent？
**参考回答**：本地 Embedding/Reranker/LLM、离线依赖与模型镜像、文档存储、可审计更新包、资源 Profile 和无外网测试。
**项目实例**：PaperStorm 已支持本地 PDF、Hash 测试与本地 Trace，真实离线需替换云 LLM 和 arXiv。
**排查/设计步骤**：盘点所有网络调用、许可证、模型体积、GPU/CPU、更新流程和密钥依赖。
**追问**：如何离线评测新模型？
**考察点**：部署边界。
**常见失误**：只把网页服务绑定 localhost 就称离线。

### 65. 如何设计 RAG 降级策略？
**参考回答**：Dense 失败回退 BM25，Rerank 超时回退 RRF，压缩失败用原文，辅助源失败标 degraded；ACL 与证据安全不能降级放宽。
**项目实例**：PaperStorm 对可选 Wikipedia fail-open，对权限和索引兼容 fail-closed。
**排查/设计步骤**：为每层定义可用性等级、用户可见状态、质量损失和告警，注入故障验证。
**追问**：LLM 不可用时怎样响应？
**考察点**：优雅降级。
**常见失误**：所有异常都返回空答案。

### 66. 如何设计 RAG 可观测性？
**参考回答**：每请求关联路由、检索、排序、压缩、生成、引用与工具 span；记录版本、分数、token、P50/P95、错误和用户反馈。
**项目实例**：PaperStorm 用本地 Trace 保底、Langfuse 做跨运行聚合，并把 badcase 回流离线集。
**排查/设计步骤**：从要回答的故障问题反推字段，做采样、脱敏、保留期和告警设计。
**追问**：怎样用 Langfuse 定位 Rerank 回归？
**考察点**：Observability 闭环。
**常见失误**：只接 SDK，不定义 span 语义。

### 67. 如何设计多源 Evidence 冲突治理？
**参考回答**：规范化 claim、实体、时间和条件，保留 stance、来源类型、质量等级；生成时并列冲突和不确定性。
**项目实例**：PaperStorm 可把 arXiv、本地 PDF 与企业文档统一成 Evidence schema。
**排查/设计步骤**：定义权威性/新鲜度规则、冲突集、引用覆盖和人工升级条件。
**追问**：预印本与正式论文冲突怎么办？
**考察点**：证据治理。
**常见失误**：相似度最高即真相。

### 68. 如何设计增量索引与文档更新？
**参考回答**：内容哈希识别变化，版本化解析/Chunk/Embedding，原子切换索引代次，支持 tombstone、重建和回滚。
**项目实例**：PaperStorm manifest/fingerprint 可阻止模型或 analyzer 变化后误用旧索引。
**排查/设计步骤**：定义文档 ID 稳定性、更新频率、并发读写、删除 SLA 和缓存失效。
**追问**：Chunk 边界变化怎样处理引用？
**考察点**：索引生命周期。
**常见失误**：在原索引上无版本覆盖。

### 69. 如何设计 RAG 成本治理？
**参考回答**：按路由避免无效检索，缓存 Embedding，限制候选，选择性 Rerank，动态上下文/输出预算，并按租户记录 token 与调用成本。
**项目实例**：PaperStorm UI 展示问答 token/耗时，Trace 记录每节点成本。
**排查/设计步骤**：建立单请求成本分解，找出 P95 和成本大户，再做质量-成本 Pareto。
**追问**：缓存答案有哪些风险？
**考察点**：FinOps 与质量。
**常见失误**：只压低 token，不看证据质量。

### 70. 如何设计 RAG 安全防护？
**参考回答**：文档解析隔离、prompt injection 检测、ACL 预过滤、工具最小权限、输出引用校验、PII 脱敏和审计。
**项目实例**：PaperStorm 把检索文本视为不可信数据，不允许文档内容改变系统/工具权限。
**排查/设计步骤**：构造恶意 PDF、越权 query、工具注入和跨租户缓存用例，红队后进入门禁。
**追问**：为什么 Prompt 里写“忽略文档指令”仍不够？
**考察点**：Agent/RAG 安全。
**常见失误**：让模型自己执行安全策略。

### 71. 如何设计可恢复的长时调研任务？
**参考回答**：阶段状态持久化、幂等任务 ID、Checkpoint、可重试节点、部分产物、取消与恢复；事件通过 SSE 推送。
**项目实例**：PaperStorm 调研流程可保存 conversation、outline、article、trace 与失败节点。
**排查/设计步骤**：定义 queued/running/succeeded/failed/cancelled 状态和每节点重放边界，做进程崩溃注入。
**追问**：如何避免恢复后重复调用收费 API？
**考察点**：持久化 Workflow。
**常见失误**：只把异常 catch 后从头重跑。

### 72. 如何设计企业文档接入平台？
**参考回答**：Connector、增量同步、格式解析、质量检测、ACL 映射、Chunk/索引、血缘和删除传播，失败文档进入隔离队列。
**项目实例**：PaperStorm 本地 PDF adapter 是起点，可扩展 SharePoint、Confluence、对象存储。
**排查/设计步骤**：按格式和来源定义契约，监控解析成功率、空文本、重复、索引延迟与权限一致性。
**追问**：表格和图片文档如何处理？
**考察点**：知识库数据工程。
**常见失误**：上传成功就视为可检索。

### 73. 如何设计一个 RAG Eval Harness？
**参考回答**：Dataset adapter、冻结 manifest、runner、retrieval/reader/judge evaluator、case diff、统计区间、报告和发布门禁分层。
**项目实例**：PaperStorm 统一 SciFact、QASPER、LongMemEval-S 与 PIM pilot，但保持任务协议隔离。
**排查/设计步骤**：先定义 claim 与指标，再建立可重复基线、单变量/递进实验和 artifact 校验。
**追问**：LLM-as-a-Judge 怎样校准？
**考察点**：评测平台设计。
**常见失误**：不同数据集数字直接排序。

### 74. 如何设计实时流式问答？
**参考回答**：HTTP 提交或 WebSocket/SSE 流式事件，事件带序号、类型、心跳和终态；服务负责断线重连、取消、背压和最终一致状态。
**项目实例**：PaperStorm 用 SSE 展示路由、检索、节点状态、token 与生成片段。
**排查/设计步骤**：模拟断网、重复事件、乱序、慢客户端和任务失败，核对 UI 与服务终态。
**追问**：SSE 与 WebSocket 如何选择？
**考察点**：实时通信。
**常见失误**：把 Uvicorn 启动等同于前端自动可用。

### 75. 如何设计从 PoC 到生产的演进路线？
**参考回答**：先用真实窄场景和可重复 Benchmark 验证价值，再补 Runtime、ACL、Trace、降级、容量、SLO 与运营闭环。
**项目实例**：PaperStorm 从 STORM 示例演进到论文检索问答、Memory/Context、Eval、Langfuse 与服务 UI。
**排查/设计步骤**：每版本声明风险、验收、回滚和指标，不用功能数量替代成熟度。
**追问**：当前离生产最远的三项是什么？
**考察点**：工程路线与自我判断。
**常见失误**：把本地 Demo 描述为企业生产系统。

## PaperStorm 针对性追问

### 76. 原版 STORM 已有什么，你个人改了什么？
**参考回答**：原版已有多视角角色、访谈式知识整理、大纲、章节写作和润色；个人扩展论文/本地 PDF、统一 RAG、Memory/Context、Runtime、Eval、Langfuse 与服务 UI。
**项目实例**：保留 STORM Agent 协作主链，在外层增加 PaperStorm 工程底座。
**排查/设计步骤**：用 git diff、模块归属和提交记录划边界，面试中分别讲上游能力与个人决策。
**追问**：为何不从零重写 Multi-Agent？
**考察点**：职责诚信与二次开发能力。
**常见失误**：把 Stanford STORM 原始能力全部写成个人实现。

### 77. 为什么优先选 arXiv 而不是 IEEE？
**参考回答**：arXiv API 开放、元数据稳定、易复现；IEEE 常涉及授权和抓取限制。企业方案可通过合法 API/订阅扩源。
**项目实例**：PaperStorm 用 arXiv 做公开论文首召回，本地 PDF/Zotero 补充用户已有材料。
**排查/设计步骤**：比较授权、覆盖、速率、全文可得性、成本和可复现性。
**追问**：arXiv 结果质量有什么局限？
**考察点**：数据源选型。
**常见失误**：绕过版权或反爬限制。

### 78. arXiv 路径现在是否使用自建 Hybrid？
**参考回答**：上游论文首召回主要由 arXiv 查询/API 决定；返回论文进入本地处理后可做 Chunk、Dense/Hybrid 筛选。它不同于本地全库的完整 Hybrid 首召回。
**项目实例**：PaperStorm 对 arXiv 做查询扩展和领域消歧，对本地 PDF/问答用自建 BM25 + Dense + RRF。
**排查/设计步骤**：分别标注 source recall 与 within-document retrieval，避免把两层指标混写。
**追问**：若要改善 arXiv 首召回怎么办？
**考察点**：检索边界。
**常见失误**：声称已经对整个 arXiv 建向量索引。

### 79. 为什么选择 GTE 作为 CPU 默认 Profile？
**参考回答**：选型看本项目质量/延迟 Pareto，不只看榜单；小规模诊断中 GTE 在 QASPER 接近 Qwen 质量但明显更快。
**项目实例**：QASPER 131 条：GTE Recall@5 0.5457、P95 201.9 ms；Qwen 0.5468、345.0 ms。
**排查/设计步骤**：固定数据与检索链，只换 Embedding，结合 SciFact、QASPER 和领域 pilot 决策。
**追问**：为什么不永久锁死 GTE？
**考察点**：模型与硬件 Pareto。
**常见失误**：省略这只是小样本诊断。

### 80. BGE、GTE、Qwen 与 Legacy MiniLM 的差异如何讲？
**参考回答**：Legacy 小且快但语义上限低；BGE/GTE 提升多语与检索训练；Qwen 新、质量潜力高、上下文更强但 CPU 成本高，具体效果依数据而变。
**项目实例**：SciFact 30 条 Recall@5：Legacy 0.6583、BGE 0.6167、GTE 0.6750、Qwen 0.7250。
**排查/设计步骤**：报告 n、K、模型 revision、设备和 P95；分析每条 rank movement，不把小样本当最终结论。
**追问**：为何 BGE 在这组数据反而下降？
**考察点**：多模型实验解读。
**常见失误**：按发布时间推断绝对优劣。

### 81. 为什么模型 Profile 必须带 fingerprint？
**参考回答**：模型、revision、维度、归一化、指令、Chunk 与 analyzer 任一改变，旧向量或倒排索引可能不兼容。
**项目实例**：PaperStorm manifest 在加载时校验 fingerprint，不匹配则重建或拒绝。
**排查/设计步骤**：列出所有影响索引语义的字段，测试错误模型、错误维度和旧 analyzer。
**追问**：metadata 字段变化是否总要重建？
**考察点**：可复现索引。
**常见失误**：只记录模型显示名称。

### 82. 为什么用 SciFact？
**参考回答**：SciFact 有科学 claim 与证据标注，适合测科学文本检索，规模可控，便于快速回归；但领域和问法不代表所有论文 QA。
**项目实例**：PaperStorm 使用 n=300 的固定 retrieval 协议报告 Recall@10。
**排查/设计步骤**：说明 split、corpus、claim-to-evidence 适配、TopK 和 evaluator，并保存 fingerprint。
**追问**：官方 BEIR 数字能直接和本项目比吗？
**考察点**：数据集选择。
**常见失误**：不说明项目协议与 BEIR nDCG@10/Recall@100 的差异。

### 83. 为什么用 QASPER？
**参考回答**：QASPER 是论文全文问答，问题与证据低词面重叠，能测段落检索、Evidence 与答案链路，贴合 PaperStorm。
**项目实例**：retrieval n=1309 与独立 full 1451 端到端协议分开运行。
**排查/设计步骤**：固定 paper-level corpus、证据粒度、不可回答处理、TopK 和 reader/judge。
**追问**：QASPER 官方 LED baseline 能直接横比吗？
**考察点**：论文 QA Benchmark。
**常见失误**：把不同 split 和 reader 的分数排在同一表里。

### 84. 为什么用 LongMemEval-S？
**参考回答**：它针对跨会话长期记忆，覆盖时间、会话定位、信息整合和干扰，能补足论文 retrieval Benchmark 不测 Memory 的缺口。
**项目实例**：PaperStorm 在 cleaned 500/500 retrieval-only 上比较 Recent、FTS Session 与 v5.6 Memory。
**排查/设计步骤**：固定清洗规则、session 粒度、Recall@K 与 reader/judge，分开报告 retrieval 和答案。
**追问**：为什么 retrieval Recall 不能代表长期问答准确率？
**考察点**：Memory 评测协议。
**常见失误**：把 P95 359.3 ms 当线上 SLA。

### 85. PIM 本地数据集是怎么来的？
**参考回答**：从用户合法持有论文中选 5 篇，解析为 797 chunks，再由 Flash 模型基于原文生成 50 题并保留证据映射，用于私有 pilot。
**项目实例**：同一冻结集上 GTE recall@5 0.7200、Answer F1 0.3983。
**排查/设计步骤**：去重问题、审计答案可支持性、冻结论文哈希/Chunk/题目和 evaluator，抽样人工复核。
**追问**：这种题集有什么偏差？
**考察点**：领域数据构造。
**常见失误**：省略模型生成题与仅 5 篇的限制。

### 86. 别人的 Benchmark 成绩是多少，为什么不能直接比较？
**参考回答**：公开报告常使用不同 split、TopK、Chunk 粒度、corpus、reader 与 judge；只有协议完全对齐才可横比，否则只作为量级参考。
**项目实例**：QASPER 官方 LED 端到端结果与 PaperStorm retrieval Recall@5 不是同一任务；BEIR SciFact 常报 nDCG@10。
**排查/设计步骤**：建立 comparison card，逐项核对 dataset version、n、K、unit、evaluator、模型和硬件。
**追问**：面试官追问“行业水平”怎么答？
**考察点**：外部对比诚信。
**常见失误**：摘取一个公开数字直接宣称领先。

### 87. SciFact 与 QASPER 的 P2 总体结果怎么解释？
**参考回答**：Hybrid/选择性 Rerank 提升覆盖，但换来明显延迟和局部回归，应同时报告总体、置信区间、P95 和案例。
**项目实例**：SciFact 0.8114→0.8264，QASPER 0.5057→0.5526；后者增益更明显。
**排查/设计步骤**：配对 query 比较 improved/regressed，计算 CI，检查触发率与每阶段延迟。
**追问**：为什么 QASPER 收益更高？
**考察点**：实验结果归因。
**常见失误**：只报相对提升百分比。

### 88. 如何解释 PIM pilot 的引用映射指标？
**参考回答**：PIM pilot 50 题下，规则型引用 ID 到原始来源的映射精度；它不是语义/人工验证准确率，也不证明 claim 必然被引用支持。
**项目实例**：PaperStorm 将 citation mapping 与 claim support 分成两个 evaluator。
**排查/设计步骤**：核对引用 ID、source locator、重复与断链，再另做语义支持判断。
**追问**：为什么名字里有 Precision 仍不能叫引用准确率？
**考察点**：指标口径。
**常见失误**：删掉“规则型原始引用映射精度”限定。

### 89. HNSW recall@5 1.0000 能写进简历吗？
**参考回答**：可以，但必须写成同一小规模 PIM 索引、固定配置下 ANN 对 exact 的离线治理结果，不能写大规模零损失。
**项目实例**：它验证后端切换没有改变该测试集 Top5，而非验证答案或生产容量。
**排查/设计步骤**：保存 exact gold、参数、库规模、query 数和硬件；扩大规模后重测。
**追问**：ANN Recall 和文档相关性 Recall 有何区别？
**考察点**：离线指标边界。
**常见失误**：把两种 Recall 混为一谈。

### 90. 为什么 Cross-Encoder 不默认全量开启？
**参考回答**：逐 query-document 联合推理昂贵且会误排，收益依 query；选择性触发能控制 P95 和回归。
**项目实例**：PaperStorm 修复 100% 重排并限制候选最多 20，提供超时回退。
**排查/设计步骤**：分析 margin、分歧、候选数、设备与 query 类型，做触发曲线和 badcase 门禁。
**追问**：GPU 会改变默认策略吗？
**考察点**：精度-延迟权衡。
**常见失误**：认为模型越复杂越应全开。

### 91. 为什么采用 recall-safe MMR？
**参考回答**：检索门禁优先保证 gold 不被多样性策略移除；因此先冻结 TopK，再在内部去重，而不是从全池重选。
**项目实例**：旧策略 QASPER Recall@5 0.4631，新策略恢复 0.5526。
**排查/设计步骤**：追踪每个被替换候选、gold 保留率和冗余度，再评估答案端收益。
**追问**：如何在不伤 Recall 的前提下扩大多样性？
**考察点**：受约束排序。
**常见失误**：只看候选更“多样”。

### 92. Parent Context 的 parent 粒度如何选？
**参考回答**：优先语义 section，过长时用页/邻域窗口；粒度应保留论证又受 token 预算约束，并保持 child 精确引用。
**项目实例**：PaperStorm 记录 parent_id、邻域、allocation 和 truncation reason。
**排查/设计步骤**：比较 section/page/window 的 Evidence 覆盖、重复、token、Answer F1 与 P95。
**追问**：表格跨页怎么办？
**考察点**：结构化上下文。
**常见失误**：把 parent 固定成整页且不验证。

### 93. 为什么不直接复用 LangChain/LangGraph 全部重写？
**参考回答**：成熟框架适合状态图、Checkpoint 和生态集成，但迁移有行为回归与依赖成本；应复用稳定组件而保留 STORM 专有调研逻辑。
**项目实例**：PaperStorm 采用适配器和可选 LangGraph，而非一次替换全部模块。
**排查/设计步骤**：列能力 gap，做最小垂直切片，对比状态恢复、Trace、测试与性能再决定迁移。
**追问**：哪些模块最值得框架化？
**考察点**：避免重复造轮子与迁移判断。
**常见失误**：把“用了框架”当成熟度。

### 94. Langfuse 在 PaperStorm 中怎样定位 Bad Case？
**参考回答**：按 dataset/version/router/retriever 标签筛异常分数或高延迟 trace，展开路由、召回、rerank、压缩、生成 span，回放输入和配置。
**项目实例**：本地 trace 是事实源，Langfuse 用于聚合、对比与人工 score；不可用时不阻塞业务。
**排查/设计步骤**：从失败答案找到 trace_id，判断 gold 丢失阶段，导出 case 加入离线集，再验证修复。
**追问**：怎样避免上传敏感 Evidence？
**考察点**：Langfuse 实战。
**常见失误**：只展示 Dashboard 截图。

### 95. PaperStorm 的 Agent Runtime 成熟度如何评价？
**参考回答**：已具备任务状态、SSE、幂等、Trace、工具契约、部分 Checkpoint 与失败分类；离生产还需分布式队列、容量压测、灾备和更严格多租户治理。
**项目实例**：文章 Workflow 与问答动态动作共用服务控制面。
**排查/设计步骤**：按状态、执行、恢复、预算、权限、观测、部署七维打分，避免只数功能。
**追问**：最优先补哪一项？
**考察点**：项目自省与 Runtime。
**常见失误**：把本地单进程称量产级。

### 96. PaperStorm 最大技术难点是什么？
**参考回答**：不是接入一个模型，而是把答案错误拆成可定位的召回、重排、上下文、生成、引用和状态问题，并建立可重复闭环。
**项目实例**：从重复 RRF、MMR 回归、PIM 歧义到 Parent 预算都通过 trace + case benchmark 定位。
**排查/设计步骤**：现象最小化、冻结输入、逐层比中间态、写回归、递进评测、记录残余风险。
**追问**：哪次修复最能体现工程判断？
**考察点**：难题与方法论。
**常见失误**：只说“调 Prompt 很困难”。

### 97. PaperStorm 最有说服力的项目成就是什么？
**参考回答**：将开源调研 Demo 扩为可追踪、可评测的论文 RAG Agent，并用公开基准、领域 pilot 和真实 badcase 约束结论。
**项目实例**：QASPER retrieval 提升、LongMemEval-S Memory 评测、引用治理与 Langfuse 均有冻结产物。
**排查/设计步骤**：成就按个人负责范围、难点、技术行动、指标口径和局限表达。
**追问**：如果只能在简历保留三条写什么？
**考察点**：价值提炼。
**常见失误**：堆功能名而没有验证。

### 98. 如果面试官问“结果为何仍有回归”，怎么回答？
**参考回答**：检索是多目标排序，新模型/Rerank 会改善多数语义问题也可能放大 hard negative；用 improved/regressed 分析和关键门禁管理，而非承诺单调提升。
**项目实例**：P2 SciFact 改善 9/回归 1，QASPER 改善 151/回归 52；PPM1D 保留为 unresolved。
**排查/设计步骤**：报告总体与分桶，找共同根因，提出保序/校准/模型升级实验，不篡改基线。
**追问**：什么回归会阻止发布？
**考察点**：实验诚信与风险判断。
**常见失误**：用平均提升掩盖关键错误。

### 99. 如何用 STAR 在三分钟内介绍 PaperStorm？
**参考回答**：S：原 STORM 缺论文私域检索和生产治理；T：负责 RAG/Memory/Runtime/Eval 扩展；A：Hybrid、Parent-Child、选择性 Rerank、Trace/Langfuse；R：给出冻结指标及限制。
**项目实例**：可选 QASPER 0.5057→0.5526 与相应 P95 代价作为一条主线。
**排查/设计步骤**：先划贡献边界，再选一个难点案例、一个量化结果、一个未解决问题，控制术语数量。
**追问**：一分钟版本如何压缩？
**考察点**：结构化表达。
**常见失误**：从仓库历史开始流水账。

### 100. 下一步最有价值的改进是什么？
**参考回答**：扩大真实领域评测与人工标注，校准选择性 Rerank/冲突证据，做百万级与高并发验证，并完善 Memory reader/judge 和多租户安全。
**项目实例**：当前公开 retrieval、端到端 QASPER、LongMemEval-S 和 PIM pilot 已覆盖不同层，但尚非生产流量验证。
**排查/设计步骤**：按岗位价值和风险排序，定义每项基线、验收、预算、回滚与简历可陈述边界。
**追问**：如果只有一周先做什么？
**考察点**：路线规划与优先级。
**常见失误**：继续加框架和 UI，而不补真实数据与门禁。

## 参考资料

- [Nowcoder Agent 面经汇总](https://www.nowcoder.com/discuss/891332286421929984)：只用于归纳 Hybrid、Rerank、Embedding 与评测等高频追问。
- [Kimi Agent 面试记录](https://www.nowcoder.com/discuss/922643334898647040)：只用于归纳模型选型、向量库、Chunk 与系统设计追问。
- [QASPER 官方 baseline](https://github.com/allenai/qasper-led-baseline)：用于理解官方任务与端到端协议，不与本项目 retrieval 指标直接横比。
- [LongMemEval](https://arxiv.org/abs/2410.10813)：长期记忆评测的任务来源。
- [Qwen3 Embedding](https://qwenlm.github.io/blog/qwen3-embedding/)：模型能力与使用契约来源。
- 项目内 `RAG_BADCASE_PROGRESSIVE_RESULTS.md`、`RAG_BAD_CASES_AND_ROADMAP.md`、`PAPERSTORM_RETRIEVAL_STACK_UPGRADE.md`、`PAPERSTORM_DOMAIN_PILOT.md`：项目数字与案例的事实源。
