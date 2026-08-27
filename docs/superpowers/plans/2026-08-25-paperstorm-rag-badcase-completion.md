# PaperStorm RAG Bad Case Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 递进完成 P1-P4，并只对每个里程碑实际影响的 Benchmark 做对比，同时形成可复现的具体 Bad Case 证据链。

**Architecture:** 在 P0 `RetrievalPipeline` 稳定契约上直接演进，不复制版本模块、不添加旧算法产品回退开关。P1 扩展查询规划与结构化召回，P2 增加选择性重排和证据治理，P3 建立 claim-citation 闭环，P4 增加检索前 ACL、运行治理和发布门禁。

**Tech Stack:** Python 3.10/3.11、rank-bm25、SentenceTransformers、Cross-Encoder、SQLite WAL、FastAPI、Langfuse、SciFact、QASPER、unittest。

## 执行状态

| 阶段 | 状态 | 证据 |
| --- | --- | --- |
| Task 1-4 / P1 | 已完成 | `119 tests OK`；规格、质量评审通过；PIM/SciFact/QASPER Retrieval 已运行 |
| Task 5-6 / P2 | 已完成 | SciFact/QASPER 配对评测完成；选择性重排、recall-safe Coverage、冲突/拒答治理已验收 |
| Task 7-8 / P3 | 已完成 | Claim-Citation 与离线测试完成；QASPER test 1451/1451 成功，Answer F1 0.5083，Claim support 0.9592 |
| Task 9-11 / P4 | 已完成 | 检索前 ACL、策略缓存、超时/熔断/恢复、Trace 脱敏、离线 Replay 与 Release Gate 已验收 |
| Task 12-13 | 已完成 | 累积结果、案例面板、全量验证与交付边界已完成 |

P1/P2 结果见 `docs/RAG_BADCASE_PROGRESSIVE_RESULTS.md`。P1 对冻结 v5.5 基线不可比；P2 对指纹一致的 P1 运行做 2000 次配对 Bootstrap 比较。

---

## File Map

- Create `knowledge_storm/search_planning.py`: `SearchPlan` schema、确定性 planner、LLM JSON adapter。
- Create `knowledge_storm/evidence_governance.py`: `RerankPolicy`、MMR、EvidenceAssessment、conflict groups。
- Create `knowledge_storm/answer_validation.py`: `AnswerDraft`、claim-citation validator、局部修复策略。
- Create `knowledge_storm/badcase_reporting.py`: Case Dossier、阶段对比、Bootstrap CI、失败分类。
- Modify `knowledge_storm/document_ingestion.py`: Section/Passage/Table/Formula 和 parent-child lineage。
- Modify `knowledge_storm/retrieval.py`: metadata filter、parent expansion、阶段候选信息。
- Modify `knowledge_storm/retrieval_pipeline.py`: SearchPlan、选择性 rerank、coverage、assessment 和纠错。
- Modify `knowledge_storm/paperstorm_qa.py`: AnswerDraft 与 citation validation。
- Modify `knowledge_storm/paperstorm_enterprise_kb.py`: chunk ACL 与新版索引 schema。
- Modify `knowledge_storm/control_plane.py`: tenant cache、timeout/circuit metrics 和 release gate。
- Modify `knowledge_storm/paperstorm_observability.py`: 检索策略、候选、Token、成本和失败阶段。
- Modify `knowledge_storm/evaluation/public_benchmarks/runner.py`: 递进 milestone manifest 与 case export。
- Modify `knowledge_storm/evaluation/public_benchmarks/qasper_generation.py`: typed answer 和 citation 指标。
- Create `examples/storm_examples/run_paperstorm_milestone.py`: 只运行受影响 Benchmark 的统一入口。
- Create `tests/fixtures/pim_badcases.json`: PIM/RAM/DRAM 固定案例。
- Create `tests/fixtures/evidence_governance_badcases.json`: 多证据、冲突、无答案案例。
- Create `tests/test_search_planning.py`, `tests/test_evidence_governance.py`, `tests/test_answer_validation.py`, `tests/test_badcase_reporting.py`。

## Task 1: Benchmark Manifest 与 Case Dossier 基础设施

**Files:**
- Create: `knowledge_storm/badcase_reporting.py`
- Create: `tests/test_badcase_reporting.py`
- Modify: `knowledge_storm/evaluation/public_benchmarks/runner.py`

- [ ] **Step 1: 写失败测试，固定 Case Dossier 与 milestone manifest schema**

```python
def test_case_dossier_records_before_root_cause_and_after():
    dossier = CaseDossier(
        case_id="pim-domain-01",
        milestone="P1",
        question="PIM 神经网络抑制",
        before={"top_ids": ["dram-1"]},
        root_cause={"stage": "planning", "failure_type": "domain_ambiguity"},
        change={"component": "SearchPlan", "reason": "resolve RF domain"},
        after={"top_ids": ["rf-pim-1"], "resolved": True},
    )
    assert dossier.to_dict()["after"]["resolved"] is True
```

- [ ] **Step 2: 运行测试并确认因模块不存在而失败**

Run: `D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest tests.test_badcase_reporting -q`

- [ ] **Step 3: 实现稳定 schema、JSONL writer、git/data/model manifest 和 Bootstrap CI**

```python
@dataclass(frozen=True)
class CaseDossier:
    case_id: str
    milestone: str
    question: str
    before: dict
    root_cause: dict
    change: dict
    after: dict
    residual_risk: str = ""
```

Manifest 必须保存 `git_sha`、dataset path/digest、split、models、Top K、seed、command、started_at、
finished_at、API usage 和 host profile；禁止保存 API Key。

- [ ] **Step 4: 运行测试并提交**

Run: `D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest tests.test_badcase_reporting tests.test_public_benchmark_contracts -q`

Commit: `feat: add milestone and bad case reporting`

## Task 2: P1 SearchPlan 与查询改写

**Files:**
- Create: `knowledge_storm/search_planning.py`
- Create: `tests/test_search_planning.py`
- Create: `tests/fixtures/pim_badcases.json`

- [ ] **Step 1: 写 PIM 歧义、standalone rewrite 和 JSON 校验失败测试**

```python
def test_planner_resolves_rf_pim_domain():
    plan = SearchPlanner().plan("PIM 神经网络抑制")
    assert plan.domain == "rf-passive-intermodulation"
    assert "passive intermodulation" in plan.must_terms
    assert {"dram", "processing-in-memory"} <= set(plan.negative_terms)
    assert len(plan.subqueries) <= 3
```

- [ ] **Step 2: 确认测试失败**

Run: `D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest tests.test_search_planning -q`

- [ ] **Step 3: 实现 SearchPlan 与 planner**

```python
@dataclass(frozen=True)
class SearchPlan:
    original_query: str
    standalone_query: str
    domain: str = ""
    entities: tuple = ()
    must_terms: tuple = ()
    negative_terms: tuple = ()
    filters: dict = field(default_factory=dict)
    subqueries: tuple = ()
    answer_type: str = "factoid"
```

确定性 planner 只处理可验证的缩写词典与多轮省略；真实 LLM adapter 使用严格 JSON schema，最多重试一次，
失败时返回 `PlanningError(error_type="invalid_structured_output")`，不静默改用旧 topic。

- [ ] **Step 4: 运行 PIM 固定案例并提交**

Run: `D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest tests.test_search_planning tests.test_paperstorm_router_llm -q`

Commit: `feat: add structured search planning`

## Task 3: P1 结构化 Chunk 与 Parent-Child Retrieval

**Files:**
- Modify: `knowledge_storm/document_ingestion.py`
- Modify: `knowledge_storm/retrieval.py`
- Create: `tests/test_structured_ingestion.py`

- [ ] **Step 1: 写章节、表格、公式、页码和 parent expansion 测试**

```python
def test_parent_child_retrieval_preserves_formula_and_section():
    nodes = ingest_document(
        document_id="paper-1",
        pages=[{"page_number": 3, "text": "2 Method\nPIM power is $P_3\\propto P_1^3$."}],
    )
    passage = next(node for node in nodes if node["node_type"] == "passage")
    assert passage["parent_id"]
    assert passage["metadata"]["page_number"] == 3
    assert "$P_3\\propto P_1^3$" in passage["content"]
```

- [ ] **Step 2: 确认测试失败**

Run: `D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest tests.test_structured_ingestion -q`

- [ ] **Step 3: 实现结构节点和 parent expansion**

节点 schema 固定为 `document/section/passage/table/formula`；标题切分优先，超长 section 再按 token
切分。检索命中 passage 后按 `parent_budget_tokens` 补回 section，不改变命中 passage 的 gold ID。

- [ ] **Step 4: 升级索引 schema revision，旧索引明确要求重建**

Run: `D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest tests.test_structured_ingestion tests.test_retrieval -q`

Commit: `feat: add structured parent child retrieval`

## Task 4: P1 集成与受影响 Benchmark

**Files:**
- Modify: `knowledge_storm/retrieval_pipeline.py`
- Modify: `knowledge_storm/paperstorm_qa.py`
- Modify: `knowledge_storm/evaluation/public_benchmarks/runner.py`
- Create: `examples/storm_examples/run_paperstorm_milestone.py`
- Modify: `tests/test_retrieval_pipeline.py`

- [ ] **Step 1: 写 pipeline 接收 SearchPlan 的失败测试**

```python
result = pipeline.search(RetrievalRequest(query="PIM 神经网络抑制", search_plan=plan))
assert result["search_plan"]["domain"] == "rf-passive-intermodulation"
assert result["stages"][0]["name"] == "plan"
```

- [ ] **Step 2: 集成 plan、subquery fusion、metadata filter 和 parent expansion**

Pipeline stage 固定为 `plan/retrieve/fuse/parent_expand/gate`，每阶段记录输入数、输出数、延迟和原因。

- [ ] **Step 3: 运行 P1 离线回归**

Run: `D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest tests.test_search_planning tests.test_structured_ingestion tests.test_retrieval_pipeline tests.test_retrieval_runtime -q`

- [ ] **Step 4: 运行 P1 受影响真实评测**

Preflight:

```powershell
$env:PAPERSTORM_BENCHMARK_ROOT="C:\Users\yzy\Desktop\codex\paperstorm-benchmarks"
$env:PAPERSTORM_MODEL_CACHE="$env:PAPERSTORM_BENCHMARK_ROOT\models"
```

只运行 SciFact Retrieval、QASPER Retrieval 和 PIM 固定集；不运行 QASPER Answer 或 LongMemEval-S。
若 SciFact 仍无读取权限，记录 `blocked: dataset_permission_denied`，先完成 QASPER/PIM，不伪造 SciFact。

- [ ] **Step 5: 输出 P1 Case Dossier 并提交**

至少输出 PIM 歧义、QASPER 词汇不一致和一个仍未解决案例。

Commit: `feat: complete P1 retrieval improvements`

## Task 5: P2 RerankPolicy 与 Coverage Selector

**Files:**
- Create: `knowledge_storm/evidence_governance.py`
- Create: `tests/test_evidence_governance.py`
- Create: `tests/fixtures/evidence_governance_badcases.json`

- [x] **Step 1: 写选择性 rerank 与 MMR 测试**

```python
decision = RerankPolicy(max_p95_ms=800).decide(features)
assert decision.enabled is False
assert decision.reason == "latency_budget_exceeded"

selected = select_evidence(candidates, top_k=3, lambda_mmr=0.65)
assert len({item["parent_id"] for item in selected}) >= 2
```

- [x] **Step 2: 确认测试失败后实现 policy、MMR 和 coverage score**

Policy 输入 answer risk、BM25/Dense overlap、RRF margin、候选数、缓存状态和延迟预算；输出
`enabled/reason/candidate_count/model/latency_budget_ms`。不增加 never/always 产品开关。

- [x] **Step 3: 集成 Pipeline 并提交**

Run: `D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest tests.test_evidence_governance tests.test_retrieval_pipeline -q`

Commit: `feat: add selective rerank and evidence coverage`

## Task 6: P2 EvidenceAssessment、冲突与有限纠错

**Files:**
- Modify: `knowledge_storm/evidence_governance.py`
- Modify: `knowledge_storm/retrieval_pipeline.py`
- Modify: `tests/test_evidence_governance.py`

- [x] **Step 1: 写无答案、条件冲突和最大纠错轮数测试**

```python
assessment = assessor.assess(query, evidence)
assert assessment.conflicts[0].relation == "contradicted"
assert assessment.next_action == "present_conflict"
assert assessment.max_corrections == 1
```

- [x] **Step 2: 实现 EvidenceAssessment schema**

字段为 `relevance/coverage/answerability/conflicts/confidence/failure_type/next_action`。纠错动作只允许
`rewrite/expand_candidates/switch_source/abstain/present_conflict`，且最多一轮。

- [x] **Step 3: 运行 P2 回归与受影响评测**

运行 SciFact Retrieval、QASPER Retrieval、无答案/冲突治理集；不运行 QASPER Answer 和
LongMemEval-S。输出 `P1+P2` 对 P1 的质量-P95 delta。

- [x] **Step 4: 输出 Case Dossier 并提交**

至少输出 QASPER 多证据、文献条件冲突和无答案误答案例。

完成证据：最终运行目录为 `C:\Users\yzy\Desktop\codex\paperstorm-benchmarks\p2\runs\2026-08-26-final-recall-safe`。SciFact Recall@10 较 P1 +0.0149，QASPER Recall@5 +0.0469；配对 95% CI 均不跨 0；治理固定集 3/3 通过。两次失败候选及具体改善/退化 case 已记录于 `docs/RAG_BADCASE_PROGRESSIVE_RESULTS.md`。

Commit: `feat: complete P2 evidence governance`

## Task 7: P3 AnswerDraft 与 Claim-Citation Validator

**Files:**
- Create: `knowledge_storm/answer_validation.py`
- Create: `tests/test_answer_validation.py`
- Modify: `knowledge_storm/paperstorm_qa.py`

- [x] **Step 1: 写支持、部分支持、矛盾和无支持 claim 测试**

```python
draft = AnswerDraft.from_payload(payload)
verdict = validator.validate(draft, evidence)
assert verdict.claims[0].status == "entailed"
assert verdict.claims[1].status == "unsupported"
```

- [x] **Step 2: 实现严格 schema 与 span 对齐**

`AnswerDraft` 包含 `answer_type/claims/citation_ids/uncertainty/abstain_reason`；每个 citation 保存
原始标题、作者、页码/章节、URL、evidence span。LLM 输出解析失败最多重试一次。

- [x] **Step 3: 实现一次局部修复**

只把失败 claim 和其候选 evidence 交给修复器。修复后仍 unsupported 时删除 claim、降低措辞或拒答，
不能整篇无界重写。

- [x] **Step 4: 运行 P3 离线测试并提交代码**

Run: `D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest tests.test_answer_validation tests.test_paperstorm_research_qa -q`

Commit: `feat: add claim citation validation`

## Task 8: P3 QASPER Answer 递进评测

**Files:**
- Modify: `knowledge_storm/evaluation/public_benchmarks/qasper_generation.py`
- Modify: `knowledge_storm/evaluation/public_benchmarks/qasper.py`
- Create: `tests/test_qasper_answer_validation.py`

- [x] **Step 1: 扩展指标测试**

新增 citation precision/recall、claim support rate、unsupported-claim rate、abstention precision/recall，
并按 extractive/abstractive/boolean/list/comparison 分组。

- [x] **Step 2: API 预检**

Run: `D:\SOFTWARE\spyder\envs\storm\python.exe -c "import os; assert os.getenv('DEEPSEEK_API_KEY'), 'DEEPSEEK_API_KEY is required for P3'"`

Key 缺失时停止付费评测并报告，不把历史聊天中的 Key 写入命令或文件。

- [x] **Step 3: 只运行 QASPER Answer/Evidence（1451/1451 完成）**

不重跑 SciFact 或 LongMemEval-S。保存 raw response、解析失败、usage、cost、latency 和 checkpoint；
断点续跑不得重复计费已成功 case。

- [x] **Step 4: 输出引用不支持与无答案 Case Dossier 并提交**

Commit: `feat: complete P3 grounded answering`

## Task 9: P4 检索前 ACL 与 Tenant Cache

**Files:**
- Modify: `knowledge_storm/retrieval_pipeline.py`
- Modify: `knowledge_storm/paperstorm_enterprise_kb.py`
- Modify: `knowledge_storm/control_plane.py`
- Create: `tests/test_retrieval_governance.py`

- [x] **Step 1: 写检索前 ACL 和跨租户缓存泄漏测试**

```python
result = pipeline.search(request_for_user("tenant-a", "mallory"))
assert "private-chunk" not in {item["chunk_id"] for item in result["results"]}
assert cache_key("tenant-a", "q") != cache_key("tenant-b", "q")
```

- [x] **Step 2: 实现 pre-retrieval candidate scope**

索引层按 tenant/document/chunk metadata 建立允许集合，BM25 与 Dense 只在允许集合上排名；禁止召回后
再从最终结果删除。缓存 namespace 必须包含 tenant、user policy digest、index revision 和 SearchPlan digest。

- [x] **Step 3: 运行 ACL/缓存测试并提交**

Run: `D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest tests.test_retrieval_governance tests.test_control_plane -q`

Commit: `feat: enforce retrieval governance boundaries`

## Task 10: P4 超时、熔断、批量与可观测性

**Files:**
- Modify: `knowledge_storm/control_plane.py`
- Modify: `knowledge_storm/paperstorm_observability.py`
- Modify: `knowledge_storm/retrieval.py`
- Create: `tests/test_retrieval_resilience.py`

- [x] **Step 1: 写 timeout、circuit、batch 和脱敏测试**

确保 Reranker 超时触发明确 failure type，熔断打开后不继续调用模型，批量输出顺序稳定，Trace 不含
API Key、完整私有文档或用户 ID 明文。

- [x] **Step 2: 实现运行治理**

模型保持进程内延迟加载单例；Cross-Encoder 批量评分；缓存命中、模型、候选数、policy reason、
P50/P95、Token 和成本进入本地 JSONL 与 Langfuse span。

- [x] **Step 3: 运行 resilience/observability 测试并提交**

Run: `D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest tests.test_retrieval_resilience tests.test_paperstorm_observability_v58 -q`

Commit: `feat: add retrieval resilience telemetry`

## Task 11: P4 Release Gate 与离线 Replay

**Files:**
- Modify: `knowledge_storm/paperstorm_benchmarks.py`
- Modify: `knowledge_storm/badcase_reporting.py`
- Modify: `examples/storm_examples/run_paperstorm_milestone.py`
- Create: `tests/test_rag_release_gate.py`

- [x] **Step 1: 写发布门禁测试**

```python
decision = ReleaseGate().evaluate(baseline, candidate, policy)
assert decision.allowed is False
assert "p95_regression" in decision.reasons
```

- [x] **Step 2: 实现门禁和 replay**

门禁比较受影响指标、Bootstrap CI、P95、unsupported claim、ACL leak 和失败率。离线 replay 使用冻结
manifest/predictions，不伪装线上 canary。

- [x] **Step 3: 运行 P4 受影响评测**

只运行离线 replay、并发/P95、ACL 泄漏、恢复、Langfuse 本地 fallback 和 release gate；质量算法未
变化，不重跑 SciFact、QASPER 或 LongMemEval-S。

- [x] **Step 4: 输出 ACL/缓存 Case Dossier 并提交**

Commit: `feat: complete P4 production governance`

## Task 12: 最终文档、UI 与全量验证

**Files:**
- Modify: `README.md`
- Modify: `docs/RAG_BAD_CASES_AND_ROADMAP.md`
- Modify: `docs/RAG_P0_IMPLEMENTATION_REPORT.md`
- Create: `docs/RAG_BADCASE_PROGRESSIVE_RESULTS.md`
- Modify: `frontend/paperstorm_dashboard/app.js`
- Test: `tests/test_paperstorm_final_packaging.py`

- [x] **Step 1: 汇总四个累积里程碑**

表格只展示 `P0 baseline -> P1 -> P1+P2 -> P1+P2+P3 -> P1+P2+P3+P4`，不展示单能力开关。
每个指标注明是否受影响、是否运行、数据 split、模型、样本量、CI、P95 和 API 成本。

- [x] **Step 2: 写七类 Case Dossier 中文复盘**

按“难点 → 改进前 → 根因 → 方案 → 改进后 → 是否解决 → 残余风险”展示；至少保留一个未完全解决案例。

- [x] **Step 3: 更新开发者控制台**

只展示四个里程碑和受影响 Benchmark；Case 面板可查看 before/after Top K、失败阶段、Trace 和引用。

- [x] **Step 4: 运行全量离线验证**

```powershell
$env:PAPERSTORM_OFFLINE_TESTS="1"
$env:PAPERSTORM_RETRIEVAL_EMBEDDING="hash"
D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest discover -s tests -p "test_*.py" -q
D:\SOFTWARE\spyder\envs\storm\python.exe -m compileall -q knowledge_storm examples tests
git diff --check
```

- [x] **Step 5: 独立代码审查，修复 Critical/Important 后提交**

Commit: `docs: publish progressive RAG bad case results`

## Task 13: 完成与推送边界

- [x] **Step 1: 确认工作区仅剩用户原有无关文件**

Run: `git status --short`

不得暂存 `docs/DESIGN_SOURCES.md`、`.codex-temp/` 或现有 patch/js 临时文件。

- [x] **Step 2: 报告提交、测试、真实评测和未解决风险**

仅在用户明确要求时执行 `git push`；本计划本身不授予推送权限。
