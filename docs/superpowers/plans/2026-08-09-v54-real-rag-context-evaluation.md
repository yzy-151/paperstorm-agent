# PaperStorm v5.4 真实 RAG 与上下文评测实施计划

> **供 Agent 执行：** 必须使用 `subagent-driven-development`（推荐）或 `executing-plans`，严格按任务逐项执行。所有步骤使用复选框记录进度。

**目标：** 在真实 Zotero 论文上建立带人工审核门禁的检索、重排和上下文压缩评测，并在网页开发者控制台清楚展示方法、指标、证据等级与局限。

**架构：** 新增一个独立的 v5.4 评测模块，负责标注存储、统计、上下文场景和脱敏报告；现有检索索引与 ContextEngine 作为被测对象，不复制其实现。FastAPI 只负责参数校验和调用，前端只消费脱敏摘要与标注 API。

**技术栈：** Python 3.10+、unittest、FastAPI、BM25、sentence-transformers、CrossEncoder、Bootstrap、原生 HTML/CSS/JavaScript、Zotero PDF。

---

## 文件结构

- 新建 `knowledge_storm/paperstorm_eval_v54.py`：标注契约、审核存储、评测门禁、统计、Context 对照实验与脱敏投影。
- 修改 `knowledge_storm/paperstorm_service.py`：管理 v5.4 评测目录并暴露核心调用。
- 修改 `examples/storm_examples/paperstorm_service_api.py`：增加标注和评测 HTTP 接口。
- 新建 `tests/test_paperstorm_eval_v54.py`：核心行为的 TDD 测试。
- 修改 `tests/test_paperstorm_service_api.py`：API 契约测试。
- 修改 `frontend/paperstorm_dashboard/index.html`：可信度、检索、Context、标注四区。
- 修改 `frontend/paperstorm_dashboard/app.js`：数据加载、标注保存和可解释指标渲染。
- 修改 `frontend/paperstorm_dashboard/styles.css`：状态、表格、标注工作台与响应式布局。
- 修改 `setup.py`、`knowledge_storm/__init__.py`：版本升级至 `5.4.0`。
- 修改 `README.md` 并新建 `docs/PAPERSTORM_V54_EVALUATION.md`：记录实测结果与面试边界。

### 任务 1：人工标注契约与发布门禁

**文件：**
- 新建：`knowledge_storm/paperstorm_eval_v54.py`
- 新建：`tests/test_paperstorm_eval_v54.py`

- [ ] **步骤 1：编写失败测试**

覆盖以下行为：完整审核可保存；`needs_edit` 必须填写修改后问题；相关论文不能为空；审核记录按 `case_id` 合并；0 条审核为 `candidate`，不足 50 条为 `pilot`，达到数量和领域门槛后为 `release_ready`；数据哈希变化使旧结果失效。

```python
def test_unreviewed_dataset_is_not_release_ready(self):
    store = AnnotationStore(self.root, self.dataset)
    progress = store.progress()
    self.assertEqual(progress["trust_level"], "candidate")
    self.assertFalse(progress["frozen_test_allowed"])

def test_review_requires_relevant_documents(self):
    with self.assertRaisesRegex(ValueError, "相关论文"):
        validate_review({"query_validity": "valid", "relevant_document_ids": []})
```

- [ ] **步骤 2：确认测试因模块缺失而失败**

运行：

```powershell
D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest tests.test_paperstorm_eval_v54 -v
```

预期：导入 `paperstorm_eval_v54` 失败。

- [ ] **步骤 3：实现最小标注模块**

实现 `validate_review()`、`AnnotationStore.list_cases()`、`save_review()`、`progress()`、`export_reviewed_dataset()` 和稳定哈希。审核文件使用 JSONL 原子替换写入，原始候选数据只读。

- [ ] **步骤 4：运行测试并确认通过**

运行同一步骤 2，预期所有任务 1 测试通过。

- [ ] **步骤 5：提交**

```powershell
git add knowledge_storm/paperstorm_eval_v54.py tests/test_paperstorm_eval_v54.py
git commit -m "feat(eval): add v5.4 annotation gate"
```

### 任务 2：检索、重排统计与防泄漏实验

**文件：**
- 修改：`knowledge_storm/paperstorm_eval_v54.py`
- 修改：`tests/test_paperstorm_eval_v54.py`

- [ ] **步骤 1：编写失败测试**

测试 Recall@5/10、Precision@5、MRR、nDCG@5、P50/P95、Bootstrap CI、配对差值和重排胜平负；测试配置只能根据 dev 排名，test 变化不能影响选型；测试未通过门禁时只能以 `pilot` 运行，不能生成 `release` 报告。

```python
def test_selection_ignores_test_scores(self):
    selected = select_dev_configuration({
        "dense": {"dev": {"ndcg_at_5": 0.4}, "test": {"ndcg_at_5": 0.9}},
        "hybrid": {"dev": {"ndcg_at_5": 0.6}, "test": {"ndcg_at_5": 0.1}},
    })
    self.assertEqual(selected, "hybrid")

def test_rerank_delta_counts_wins_ties_and_losses(self):
    delta = paired_rank_delta([1, 0, 2], [0, 0, 3])
    self.assertEqual(delta["wins"], 1)
    self.assertEqual(delta["ties"], 1)
    self.assertEqual(delta["losses"], 1)
```

- [ ] **步骤 2：运行定向测试并观察预期失败**

- [ ] **步骤 3：实现统计与实验编排**

复用 `HybridPaperIndex`；支持 BM25、Dense、Hybrid 和 Hybrid+Rerank；记录 `candidate_k`、RRF 配置、模型、样本量、失败原因和逐问题排名。重排初始化失败时写入 `skipped_reason`，不伪造结果。

- [ ] **步骤 4：运行测试并确认通过**

- [ ] **步骤 5：提交**

```powershell
git commit -am "feat(eval): add retrieval comparison metrics"
```

### 任务 3：真实论文 Context Engineering 对照实验

**文件：**
- 修改：`knowledge_storm/paperstorm_eval_v54.py`
- 修改：`tests/test_paperstorm_eval_v54.py`

- [ ] **步骤 1：编写失败测试**

构造真实论文元数据形态的多轮场景，比较 `full_history`、`fixed_window`、`structured_compaction`。验证 Token 数、约束、实体、来源、工具调用配对、可恢复性、重复压缩漂移和相关文档保持率均有明确分母。

```python
def test_context_report_compares_three_strategies(self):
    report = evaluate_context_scenarios([reviewed_context_case()])
    self.assertEqual(set(report["strategies"]), {
        "full_history", "fixed_window", "structured_compaction"
    })
    self.assertIn("source_retention_rate", report["strategies"]["structured_compaction"])
```

- [ ] **步骤 2：运行定向测试并观察预期失败**

- [ ] **步骤 3：实现 Context 场景与评测**

使用 `ContextEngine` 进行结构化压缩；完整历史作为参考；固定窗口作为基线。确定性评测不调用 LLM，回答级 LLM 裁判保持可选并标记 `model_judged`。

- [ ] **步骤 4：运行测试并确认通过**

- [ ] **步骤 5：提交**

```powershell
git commit -am "feat(context): add real-paper v5.4 benchmark"
```

### 任务 4：服务 API 与脱敏报告

**文件：**
- 修改：`knowledge_storm/paperstorm_service.py`
- 修改：`examples/storm_examples/paperstorm_service_api.py`
- 修改：`tests/test_paperstorm_service_api.py`
- 修改：`tests/test_paperstorm_eval_v54.py`

- [ ] **步骤 1：编写失败测试**

覆盖以下接口：

```text
GET  /evaluations/v54/status
GET  /evaluations/v54/annotations
PUT  /evaluations/v54/annotations/{case_id}
POST /evaluations/v54/retrieval
POST /evaluations/v54/context
GET  /evaluations/v54/latest
```

验证报告不包含 Zotero 根路径、PDF 路径、完整证据摘要和逐问题私有内容。

- [ ] **步骤 2：运行 API 测试并观察 404 或方法缺失**

- [ ] **步骤 3：实现服务与 API**

默认从 `service_root/evaluations/v54/` 读取数据。若当前服务目录没有候选集，则允许从命令参数指定数据集导入；不存在时返回明确提示。API 版本更新为 `5.4`。

- [ ] **步骤 4：运行 API 和核心测试**

- [ ] **步骤 5：提交**

```powershell
git commit -am "feat(api): expose v5.4 evaluation workbench"
```

### 任务 5：开发者 Benchmark 控制台与标注台

**文件：**
- 修改：`frontend/paperstorm_dashboard/index.html`
- 修改：`frontend/paperstorm_dashboard/app.js`
- 修改：`frontend/paperstorm_dashboard/styles.css`
- 新建或修改：`tests/test_paperstorm_dashboard_v54.py`

- [ ] **步骤 1：编写失败的静态契约测试**

验证 v5.4 版本、四个区域、关键元素 ID、指标定义、证据徽标、标注字段、保存按钮和错误状态均存在。

- [ ] **步骤 2：运行测试并确认失败**

- [ ] **步骤 3：实现界面**

可信度卡片显示门禁；方法表显示 Recall、Precision、MRR、nDCG、CI 和延迟；Context 表显示三策略；标注台支持上一条、下一条、保存、进度和导出。按钮有运行中、完成和失败状态。原始 JSON 放入折叠区。

- [ ] **步骤 4：运行静态测试，启动服务并用浏览器检查**

运行：

```powershell
D:\SOFTWARE\spyder\envs\storm\python.exe examples\storm_examples\start_paperstorm_service.py --service-root ./results/paperstorm_demo_service --host 127.0.0.1 --port 8002
```

检查桌面和移动视口无文本重叠，空数据、候选、小规模实验和可发布状态均可理解。

- [ ] **步骤 5：提交**

```powershell
git commit -am "feat(frontend): add v5.4 benchmark console"
```

### 任务 6：真实 Zotero 小规模实验、版本与文档

**文件：**
- 修改：`setup.py`
- 修改：`knowledge_storm/__init__.py`
- 修改：`README.md`
- 新建：`docs/PAPERSTORM_V54_EVALUATION.md`
- 新建：`docs/benchmarks/paperstorm_real_eval_v54_summary.json`
- 修改：`tests/test_paperstorm_release_integrity_v52.py` 或新建 v5.4 发布测试

- [ ] **步骤 1：编写版本和脱敏报告失败测试**

要求包版本为 `5.4.0`、网页为 `v5.4`，报告必须包含证据状态、样本量、模型和局限，且不能包含本地绝对路径。

- [ ] **步骤 2：运行真实 Zotero 开发集和候选小规模实验**

使用 `local_zotero_root.txt` 指向的只读 Zotero 数据，复用 40 篇论文和 v5.2 候选集。若本机存在兼容的多语言 Cross-Encoder，则纳入 dev；否则记录初始化失败并保留 Dense/Hybrid 结果。

- [ ] **步骤 3：生成脱敏摘要并更新中文文档**

文档必须报告 v5.2 基线、v5.4 新协议、实测数字、负结果、人工审核待办和面试可说/不可说内容。

- [ ] **步骤 4：运行完整验证**

```powershell
D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest discover -s tests -v
git diff --check
```

- [ ] **步骤 5：提交但不推送**

```powershell
git commit -am "release: prepare PaperStorm v5.4"
```

## 计划自检

- 覆盖规格中的人工标注、门禁、防泄漏、检索与重排、Context 对照、网页解释、隐私和版本要求。
- 真实结果与自动候选结果分开，不预设重排一定提升。
- 用户唯一必须参与的步骤是审核冻结测试问题；实现完成后由网页标注台承接。
- 不包含 GitHub 推送。
