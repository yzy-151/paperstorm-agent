# PaperStorm P0 RAG 主链统一与版本化代码清理 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task with review checkpoints.

**Goal:** 将问答、企业知识库与公开 Benchmark 统一到同一套 Hybrid Retrieval Pipeline，移除生产代码中的内部版本号和已失效 toy 评测，同时保持论文调研、聊天、知识库、公开评测和 Web API 的主要行为可用。

**Architecture:** 保留 Stanford STORM/Co-STORM 原始模块；PaperStorm 扩展收敛为 `document_ingestion -> retrieval -> retrieval_pipeline -> evidence/context -> reader`。Context、Memory、Conversation Runtime 和 Control Plane 分别成为稳定命名模块。旧索引只做显式检测和重建提示，不再静默回退到 hash/字符重叠检索。公开 Benchmark 通过依赖注入使用同一个 `RetrievalPipeline`，测试环境显式注入确定性 embedding，真实服务默认使用真实 embedding。

**Tech Stack:** Python 3.10/3.11、unittest、Pydantic、FastAPI、sentence-transformers、rank-bm25/项目内 BM25、RRF、Cross-Encoder、LangGraph、SQLite。

---

## 范围与完成定义

本计划只执行已批准路线图中的 P0：代码收敛、统一检索主链、删除旧入口和建立回归保护。查询改写、多跳检索、冲突证据治理、动态 rerank、CRAG/Self-RAG 等 P1-P4 能力不在本轮伪装实现，继续以 [RAG_BAD_CASES_AND_ROADMAP.md](../../RAG_BAD_CASES_AND_ROADMAP.md) 为后续依据。

完成时必须同时满足：

1. `knowledge_storm` 的生产模块不再 import `paperstorm_*_vNN`。
2. 问答、企业知识库和公开检索 Benchmark 都调用 `RetrievalPipeline.search()`。
3. 相同 corpus、query、provider、mode 和 Top-K 在三个入口得到相同 chunk ID 排名与统一 stage schema。
4. 真实服务默认 embedding 不是 hash；hash 只允许测试或显式 smoke profile 使用。
5. 旧索引不能被新代码静默误读，必须返回可执行的重建错误。
6. `/evaluations/v54/*`、`/evaluations/runtime-v44*`、`/evaluations/production-v45*` 不再注册。
7. SciFact、QASPER、LongMemEval 和 Context Pareto 等当前公开 Benchmark 仍可从统一目录发现。
8. 全量测试通过，README 不再把内部实现版本号当架构名。

## Task 1: 建立模块边界红线测试

**Files:**
- Create: `tests/test_paperstorm_module_boundaries.py`
- Modify: `tests/test_paperstorm_release_integrity_v52.py`

**Step 1: 写失败测试**

新增测试扫描 `knowledge_storm/*.py`，允许版本号只出现在迁移器、历史数据说明和发布元数据中，禁止生产 Python import 版本化模块：

```python
import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "knowledge_storm"


class PaperStormModuleBoundaryTest(unittest.TestCase):
    def test_production_modules_do_not_import_versioned_modules(self):
        violations = []
        for path in PACKAGE.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.ImportFrom) and node.module:
                    names.append(node.module)
                elif isinstance(node, ast.Import):
                    names.extend(alias.name for alias in node.names)
                for name in names:
                    if "paperstorm_" in name and any(
                        marker in name for marker in ("_v3", "_v4", "_v5", "_v6")
                    ):
                        violations.append(f"{path.name}:{node.lineno}:{name}")
        self.assertEqual([], violations)

    def test_stable_runtime_modules_are_importable(self):
        from knowledge_storm import context_engine
        from knowledge_storm import control_plane
        from knowledge_storm import conversation_runtime
        from knowledge_storm import document_ingestion
        from knowledge_storm import memory_store
        from knowledge_storm import retrieval
        from knowledge_storm import retrieval_pipeline

        self.assertTrue(context_engine.ContextEngine)
        self.assertTrue(control_plane.ProductionControlPlane)
        self.assertTrue(conversation_runtime.PaperStormConversationRuntime)
        self.assertTrue(document_ingestion.chunk_pdf_pages)
        self.assertTrue(memory_store.LongTermMemoryService)
        self.assertTrue(retrieval.HybridPaperIndex)
        self.assertTrue(retrieval_pipeline.RetrievalPipeline)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: 运行并确认失败原因正确**

Run:

```powershell
D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest tests.test_paperstorm_module_boundaries -v
```

Expected: `ModuleNotFoundError` 和现有版本化 imports 导致失败，而不是语法或环境错误。

**Step 3: 更新 release integrity 测试口径**

把 `test_paperstorm_release_integrity_v52.py` 中对旧文件名、旧版本目录和旧 API 的正向断言改成：稳定模块必须存在；旧生产入口最终必须不存在；公开 Benchmark 目录必须存在。

**Step 4: 提交测试红线**

```powershell
git add tests/test_paperstorm_module_boundaries.py tests/test_paperstorm_release_integrity_v52.py
git commit -m "test: define stable PaperStorm module boundaries"
```

## Task 2: 收敛文档解析与 Retrieval 基础能力

**Files:**
- Create: `knowledge_storm/document_ingestion.py`
- Create: `knowledge_storm/retrieval.py`
- Modify: `knowledge_storm/paperstorm_retrieval_runtime.py`
- Modify: `knowledge_storm/paperstorm_zotero.py`
- Create: `tests/test_document_ingestion.py`
- Create: `tests/test_retrieval.py`
- Delete after migration: `knowledge_storm/paperstorm_document_v41.py`
- Delete after migration: `knowledge_storm/paperstorm_retrieval_v41.py`
- Delete after migration: `tests/test_paperstorm_retrieval_v41.py`

**Step 1: 先迁移测试并保持失败**

将 `test_paperstorm_retrieval_v41.py` 的行为断言迁移到 `test_retrieval.py`，import 改为：

```python
from knowledge_storm.retrieval import (
    CrossEncoderReranker,
    HashEmbeddingProvider,
    HybridPaperIndex,
    SentenceTransformerProvider,
    reciprocal_rank_fusion,
)
```

新增以下契约：

```python
def test_hash_embedding_must_be_explicit():
    from knowledge_storm.retrieval import build_embedding_provider

    provider = build_embedding_provider("hash")
    assert provider.name == "hash"


def test_real_embedding_is_default():
    from knowledge_storm.retrieval import build_embedding_provider

    with mock.patch.dict(os.environ, {}, clear=True):
        provider = build_embedding_provider()
    assert provider.name != "hash"
```

测试文件 import `os` 和 `from unittest import mock`，不引入 pytest 依赖。

**Step 2: 运行目标测试并确认 import 失败**

```powershell
D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest tests.test_retrieval tests.test_document_ingestion -v
```

**Step 3: 创建稳定模块并迁移实现**

`document_ingestion.py` 接收 `paperstorm_document_v41.py` 的 PDF page extraction、heading-aware chunking 和 slug 工具；`retrieval.py` 接收 `paperstorm_retrieval_v41.py` 的 multilingual tokenizer、RRF、SentenceTransformer provider、CrossEncoder 和 `HybridPaperIndex`。

将 `paperstorm_rag.py` 中仍有价值的 `HashEmbeddingProvider` 移入 `retrieval.py`，但要求调用端显式传入 `"hash"`。默认 provider 解析规则固定为：

```python
def build_embedding_provider(provider=None, model_name=None, cache_folder=None):
    selected = provider or os.getenv(
        "PAPERSTORM_EMBEDDING_PROVIDER", "sentence-transformer"
    )
    if selected == "hash":
        return HashEmbeddingProvider()
    if selected in {"real", "sentence-transformer"}:
        return SentenceTransformerProvider(
            model_name=model_name, cache_folder=cache_folder
        )
    raise ValueError(f"unsupported embedding provider: {selected}")
```

**Step 4: 消除反向依赖**

把 `HybridPaperIndex.from_run_dir()` 对 `paperstorm_rag.chunk_text` 的依赖改为 `document_ingestion.chunk_text`。`paperstorm_retrieval_runtime.py` 和 `paperstorm_zotero.py` 改用稳定模块。

**Step 5: 验证并删除旧文件**

```powershell
D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest tests.test_retrieval tests.test_document_ingestion tests.test_paperstorm_retrieval_runtime tests.test_paperstorm_retrievers -v
Select-String -Path knowledge_storm\*.py,tests\*.py -Pattern 'paperstorm_retrieval_v41|paperstorm_document_v41'
```

Expected: 测试通过，搜索无有效引用后删除旧模块和旧测试。

**Step 6: 提交**

```powershell
git add knowledge_storm/document_ingestion.py knowledge_storm/retrieval.py knowledge_storm/paperstorm_retrieval_runtime.py knowledge_storm/paperstorm_zotero.py tests/test_document_ingestion.py tests/test_retrieval.py tests/test_paperstorm_retrieval_runtime.py
git add -u knowledge_storm/paperstorm_document_v41.py knowledge_storm/paperstorm_retrieval_v41.py tests/test_paperstorm_retrieval_v41.py
git commit -m "refactor: stabilize retrieval and ingestion modules"
```

## Task 3: 建立唯一 RetrievalPipeline 契约

**Files:**
- Create: `knowledge_storm/retrieval_pipeline.py`
- Create: `tests/test_retrieval_pipeline.py`
- Modify: `knowledge_storm/paperstorm_qa.py`
- Modify: `knowledge_storm/paperstorm_enterprise_kb.py`
- Modify: `knowledge_storm/paperstorm_retrieval_runtime.py`

**Step 1: 写跨入口一致性失败测试**

测试构造三篇固定文档并显式注入 `HashEmbeddingProvider`，分别经 Pipeline、`PaperStormKnowledgeBase`、`EnterpriseKnowledgeBaseService` 检索，断言：

```python
self.assertEqual(pipeline_ids, qa_ids)
self.assertEqual(pipeline_ids, enterprise_ids)
self.assertEqual(
    ["retrieve", "fuse", "rerank", "gate"],
    [stage["name"] for stage in pipeline_result["stages"]],
)
```

即使 reranker 未启用，`rerank` stage 也必须存在并标记 `status="skipped"`，避免前端和 trace schema 随配置漂移。

**Step 2: 定义稳定请求与结果模型**

`retrieval_pipeline.py` 提供：

```python
@dataclass(frozen=True)
class RetrievalRequest:
    query: str
    top_k: int = 5
    candidate_k: int = 20
    mode: str = "hybrid"
    expected_keywords: tuple[str, ...] = ()
    forbidden_keywords: tuple[str, ...] = ()
    enable_reranker: bool = False


class RetrievalPipeline:
    def __init__(self, index, reranker=None, relevance_gate=None):
        self.index = index
        self.reranker = reranker
        self.relevance_gate = relevance_gate

    def search(self, request: RetrievalRequest) -> dict:
        if not request.query.strip():
            raise ValueError("query is required")
        # 实现按 retrieve -> fuse -> rerank -> gate 记录统一 stage，
        # 返回 query、results、stages、models、latency_ms 和 schema_revision。
```

实际实现不得把 `results` 复制为三份；每个 stage 只记录输入数量、输出数量、耗时、状态和模型名，最终 chunk 只在顶层 `results` 返回。

**Step 3: 迁移问答入口**

`PaperStormKnowledgeBase` 构造函数接收可选 `retrieval_pipeline`。`from_run_dir()` 使用 `build_runtime_index()` 构造 Pipeline。删除 `search()` 中吞掉所有异常后回退到 set-overlap 的逻辑；仅对明确的 `IndexMigrationRequiredError` 返回重建提示，其他异常向上抛给服务 trace。

**Step 4: 迁移企业知识库**

企业知识库 manifest 改为：

```json
{
  "index_schema": "paperstorm-hybrid-index",
  "schema_revision": 2,
  "embedding_provider": "sentence-transformer",
  "retrieval_mode": "hybrid"
}
```

创建知识库默认 `embedding_provider="sentence-transformer"`；单元测试显式传 `"hash"`。加载 revision 1 的 `rag_index.json` 时抛：

```python
class IndexMigrationRequiredError(RuntimeError):
    pass
```

错误消息必须包含 KB ID、旧 revision 和重新入队 index job 的操作说明。

**Step 5: 迁移 runtime helper**

把 `paperstorm_retrieval_runtime.py` 改名为 `retrieval_runtime.py`，其 `search_runtime_index()` 成为构造 `RetrievalRequest` 并调用 Pipeline 的薄适配器；删除 `legacy_index` 对比和 `paperstorm_eval_v4.build_seed_dataset` 依赖。

**Step 6: 验证唯一主链**

```powershell
D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest tests.test_retrieval_pipeline tests.test_paperstorm_enterprise_v32 tests.test_paperstorm_retrieval_runtime -v
Select-String -Path knowledge_storm\*.py -Pattern 'PaperStormRAGIndex|legacy_fallback|set_overlap'
```

Expected: 只有待删除旧模块可命中；生产问答和企业知识库无命中。

**Step 7: 提交**

```powershell
git add knowledge_storm/retrieval_pipeline.py knowledge_storm/retrieval_runtime.py knowledge_storm/paperstorm_qa.py knowledge_storm/paperstorm_enterprise_kb.py tests/test_retrieval_pipeline.py tests/test_paperstorm_enterprise_v32.py tests/test_paperstorm_retrieval_runtime.py
git add -u knowledge_storm/paperstorm_retrieval_runtime.py
git commit -m "refactor: unify PaperStorm retrieval entrypoints"
```

## Task 4: 合并 Context Engine

**Files:**
- Create: `knowledge_storm/context_engine.py`
- Create: `tests/test_context_engine.py`
- Modify: `knowledge_storm/paperstorm_chat_agent.py`
- Modify: `knowledge_storm/paperstorm_runtime.py`
- Modify: `knowledge_storm/evaluation/public_benchmarks/qasper.py`
- Modify: `knowledge_storm/evaluation/public_benchmarks/longbench_context.py`
- Delete: `knowledge_storm/paperstorm_context_v42.py`
- Delete: `knowledge_storm/paperstorm_context_v56.py`
- Delete: `tests/test_paperstorm_context_v42.py`
- Delete: `tests/test_paperstorm_context_v56.py`

**Step 1: 迁移行为测试**

把两份旧测试合并为 `test_context_engine.py`，保留 token 预算、tool group 原子性、pinned message、structured summary、ledger、compact/restore 和 evidence retention 测试。稳定 import 为：

```python
from knowledge_storm.context_engine import (
    ContextEngine,
    ContextEngineConfig,
    ContextEventStore,
    ContextLedger,
    build_structured_summary_prompt,
    estimate_tokens,
    truncate_to_tokens,
)
```

**Step 2: 运行并确认稳定模块尚不存在**

```powershell
D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest tests.test_context_engine -v
```

**Step 3: 合并实现并去版本类名**

以 v5.6 Context Engine 为行为主体，内联 v4.2 中仍使用的 token/event helpers。类名固定为 `ContextEngineConfig`、`ContextLedger`、`ContextEventStore`、`ContextEngine`；持久化 payload 使用：

```python
{"schema": "paperstorm-context-ledger", "schema_revision": 1}
```

旧 JSON 中缺失 `schema_revision` 时按 revision 1 读取，不要求用户迁移。

**Step 4: 更新所有生产和公开评测 import**

用稳定模块替换 chat agent、runtime、QASPER context 和 LongBench context 的 import；公开结果中的 `engine` 标签改为 `paperstorm-context`，发布版本只写在报告 `release` 字段。

**Step 5: 验证和删除旧文件**

```powershell
D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest tests.test_context_engine tests.test_memory_context_public_benchmarks_v56 tests.test_paperstorm_service -v
Select-String -Path knowledge_storm\*.py,tests\*.py -Pattern 'paperstorm_context_v42|paperstorm_context_v56|ContextEngineV56|ContextEngineConfigV56'
```

**Step 6: 提交**

```powershell
git add knowledge_storm/context_engine.py knowledge_storm/paperstorm_chat_agent.py knowledge_storm/paperstorm_runtime.py knowledge_storm/evaluation/public_benchmarks/qasper.py knowledge_storm/evaluation/public_benchmarks/longbench_context.py tests/test_context_engine.py tests/test_memory_context_public_benchmarks_v56.py tests/test_paperstorm_service.py
git add -u knowledge_storm/paperstorm_context_v42.py knowledge_storm/paperstorm_context_v56.py tests/test_paperstorm_context_v42.py tests/test_paperstorm_context_v56.py
git commit -m "refactor: consolidate context engine"
```

## Task 5: 合并长期 Memory Store

**Files:**
- Create: `knowledge_storm/memory_store.py`
- Create: `tests/test_memory_store.py`
- Modify: `knowledge_storm/paperstorm_chat_agent.py`
- Modify: `knowledge_storm/paperstorm_runtime.py`
- Modify: `knowledge_storm/evaluation/public_benchmarks/longmemeval.py`
- Modify: `knowledge_storm/evaluation/public_benchmarks/longmemeval_runner.py`
- Delete: `knowledge_storm/paperstorm_memory_v43.py`
- Delete: `knowledge_storm/paperstorm_memory_v56.py`
- Delete: `tests/test_paperstorm_memory_v43.py`
- Delete: `tests/test_paperstorm_memory_v56.py`

**Step 1: 合并测试并明确真实/测试 embedding 边界**

`test_memory_store.py` 保留写入策略、去重、编辑/删除、时间有效性、BM25+dense+MMR、namespace 隔离和跨会话召回测试。增加：

```python
def test_default_memory_embedding_is_real(self):
    provider = build_memory_embedding_provider()
    self.assertNotEqual("hash", provider.name)


def test_hash_memory_embedding_requires_explicit_injection(self):
    service = LongTermMemoryService(
        self.temp_dir, embedding_provider=HashEmbeddingProvider()
    )
    self.assertEqual("hash", service.embedding_provider.name)
```

第一条只构造 provider，不下载模型；模型应延迟到首次 encode 时加载。

**Step 2: 创建稳定模块**

以 v5.6 service 为主体，把 v4.3 的 `MemoryRecord`、`MemoryCandidate`、`MemoryWritePolicy` 和 Pydantic dump helper 合并进同一文件。去掉类名版本号；数据库 schema 继续兼容原 SQLite 表，不改列名。

**Step 3: 更新生产与 Benchmark import**

chat/runtime/LongMemEval 统一 import `LongTermMemoryService`。结果报告中用 `memory_engine="paperstorm-memory"` 和独立 `release` 字段，不再输出 `v5.6 Memory` 作为算法名；展示层可保留 “PaperStorm Memory” 对比标签。

**Step 4: 验证和删除旧模块**

```powershell
D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest tests.test_memory_store tests.test_longmemeval_answer_v56 tests.test_memory_context_public_benchmarks_v56 tests.test_paperstorm_service -v
Select-String -Path knowledge_storm\*.py,tests\*.py -Pattern 'paperstorm_memory_v43|paperstorm_memory_v56|MemoryCandidateV43|LongTermMemoryServiceV56'
```

**Step 5: 提交**

```powershell
git add knowledge_storm/memory_store.py knowledge_storm/paperstorm_chat_agent.py knowledge_storm/paperstorm_runtime.py knowledge_storm/evaluation/public_benchmarks/longmemeval.py knowledge_storm/evaluation/public_benchmarks/longmemeval_runner.py tests/test_memory_store.py tests/test_longmemeval_answer_v56.py tests/test_memory_context_public_benchmarks_v56.py tests/test_paperstorm_service.py
git add -u knowledge_storm/paperstorm_memory_v43.py knowledge_storm/paperstorm_memory_v56.py tests/test_paperstorm_memory_v43.py tests/test_paperstorm_memory_v56.py
git commit -m "refactor: consolidate long-term memory store"
```

## Task 6: 稳定 Conversation Runtime 与 Control Plane

**Files:**
- Create: `knowledge_storm/conversation_runtime.py`
- Create: `knowledge_storm/control_plane.py`
- Create: `tests/test_conversation_runtime.py`
- Create: `tests/test_control_plane.py`
- Modify: `knowledge_storm/paperstorm_service.py`
- Modify: `knowledge_storm/paperstorm_enterprise_kb.py`
- Modify: `examples/storm_examples/paperstorm_service_api.py`
- Delete: `knowledge_storm/paperstorm_langgraph_v44.py`
- Delete: `knowledge_storm/paperstorm_production_v45.py`
- Delete: `tests/test_paperstorm_langgraph_v44.py`
- Delete: `tests/test_paperstorm_production_v45.py`

**Step 1: 迁移现有行为测试到稳定名称**

稳定 public classes：

```python
from knowledge_storm.conversation_runtime import (
    ConversationRequest,
    ConversationState,
    PaperStormConversationRuntime,
    StormDeepResearchTool,
)
from knowledge_storm.control_plane import (
    ProductionControlPlane,
    ProductionRuntime,
)
```

保留 LangGraph checkpoint/replay、LLM action routing、research fallback、ACL、idempotency、retry、trace 和 worker tests。trace component 名改为稳定 stage 名；数据库路径继续读取已有的 `production_control_v45.sqlite`，新安装创建 `production_control.sqlite`。

**Step 2: 创建稳定模块并更新 service helper 名**

`paperstorm_service.py` 内 helper 改为：

```python
def _conversation_runtime(self):
    return PaperStormConversationRuntime(
        root_dir=self.root_dir / "conversation_runtime",
        task_service=self,
        memory_service=self._memory_service(),
    )

def _production_runtime(self):
    return ProductionRuntime(
        root_dir=self.root_dir / "production_runtime",
        task_service=self,
        control_plane=self._production_control(),
    )

def _production_control(self):
    current = self.root_dir / "production_control.sqlite"
    legacy = self.root_dir / "production_control_v45.sqlite"
    return ProductionControlPlane(legacy if legacy.exists() else current)

def _memory_service(self):
    return LongTermMemoryService(self.root_dir / "memory_service")
```

不得保留 `_v43/_v44/_v45` alias，以便边界测试真正约束生产代码。

**Step 3: 保持 API 路径稳定，删除版本化评测路径**

保留 `/conversation-graph/*`、`/production/status`、`/production/traces/*`、`/memories/*`。删除：

```text
/evaluations/v54/*
/evaluations/runtime-v44
/evaluations/runtime-v44/latest
/evaluations/production-v45
/evaluations/production-v45/latest
```

公开评测统一通过现有 `/benchmarks` catalog/start/status 接口运行。

**Step 4: 测试 API 不再暴露旧路径**

在 service API 测试中读取 OpenAPI schema：

```python
paths = create_app(service=fake_service).openapi()["paths"]
self.assertIn("/conversation-graph/invoke", paths)
self.assertIn("/benchmarks", paths)
self.assertFalse(any("v54" in path or "v44" in path or "v45" in path for path in paths))
```

**Step 5: 验证并删除旧模块**

```powershell
D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest tests.test_conversation_runtime tests.test_control_plane tests.test_paperstorm_service tests.test_paperstorm_service_cli -v
Select-String -Path knowledge_storm\*.py,examples\storm_examples\*.py,tests\*.py -Pattern 'paperstorm_langgraph_v44|paperstorm_production_v45|runtime-v44|production-v45'
```

**Step 6: 提交**

```powershell
git add knowledge_storm/conversation_runtime.py knowledge_storm/control_plane.py knowledge_storm/paperstorm_service.py knowledge_storm/paperstorm_enterprise_kb.py examples/storm_examples/paperstorm_service_api.py tests/test_conversation_runtime.py tests/test_control_plane.py tests/test_paperstorm_service.py tests/test_paperstorm_service_cli.py
git add -u knowledge_storm/paperstorm_langgraph_v44.py knowledge_storm/paperstorm_production_v45.py tests/test_paperstorm_langgraph_v44.py tests/test_paperstorm_production_v45.py
git commit -m "refactor: stabilize conversation runtime and control plane"
```

## Task 7: 统一公开 Benchmark 命名和检索依赖

**Files:**
- Rename: `knowledge_storm/evaluation/public_benchmarks/v60_harness.py` -> `knowledge_storm/evaluation/public_benchmarks/harness.py`
- Rename: `knowledge_storm/evaluation/public_benchmarks/v60_llm.py` -> `knowledge_storm/evaluation/public_benchmarks/llm_reader.py`
- Modify: `knowledge_storm/evaluation/public_benchmarks/runner.py`
- Modify: `knowledge_storm/evaluation/public_benchmarks/beir_scifact.py`
- Modify: `knowledge_storm/evaluation/public_benchmarks/qasper.py`
- Modify: `knowledge_storm/paperstorm_benchmarks.py`
- Modify: `examples/storm_examples/run_paperstorm_public_benchmark.py`
- Modify: `examples/storm_examples/run_context_profile_pareto.py`
- Modify: `examples/storm_examples/run_longmemeval_e2e_v60.py`
- Create: `tests/test_public_benchmark_contracts.py`
- Delete: `tests/test_paperstorm_v60_benchmarks.py`

**Step 1: 写统一 Pipeline 注入测试**

构造 `RecordingPipeline` 记录 `RetrievalRequest`，分别执行 SciFact 和 QASPER adapter，断言 adapter 没有自行实例化 `HybridPaperIndex`，并且公开输出保留官方 metric 名称。

**Step 2: 重命名 harness 和 reader**

所有 import 改用：

```python
from knowledge_storm.evaluation.public_benchmarks.harness import BenchmarkHarness
from knowledge_storm.evaluation.public_benchmarks.llm_reader import LLMReader
```

版本号只保留在 release metadata，不再出现在 Python 文件名、类名、benchmark ID。现有结果目录继续可读，registry 的 `legacy_ids` 映射旧 ID 到稳定 ID：

```python
LEGACY_BENCHMARK_IDS = {
    "scifact-retrieval-v55": "scifact-retrieval",
    "qasper-retrieval-v55": "qasper-retrieval",
    "qasper-answer-v55": "qasper-answer",
    "longmemeval-retrieval-v56": "longmemeval-retrieval",
    "qasper-context-v56": "qasper-context",
    "context-pareto-v60": "context-pareto",
    "longmemeval-e2e-v60": "longmemeval-e2e",
}
```

API 接收旧 ID 时映射并返回 `deprecated_id`，保证已有前端缓存不会突然 404；catalog 只展示稳定 ID。

**Step 3: 让 Retrieval Benchmark 使用统一 Pipeline**

SciFact/QASPER runner 构造 corpus-specific `HybridPaperIndex` 后统一注入 `RetrievalPipeline`。Smoke profile 显式 `HashEmbeddingProvider`；quality profile 使用真实 `SentenceTransformerProvider` 和可选 CrossEncoder。

**Step 4: 验证**

```powershell
D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest tests.test_public_benchmark_contracts tests.test_public_benchmarks_v55 tests.test_qasper_generation_v55 tests.test_memory_context_public_benchmarks_v56 -v
D:\SOFTWARE\spyder\envs\storm\python.exe examples\storm_examples\run_paperstorm_public_benchmark.py --help
```

**Step 5: 提交**

```powershell
git add knowledge_storm/evaluation/public_benchmarks knowledge_storm/paperstorm_benchmarks.py examples/storm_examples/run_paperstorm_public_benchmark.py examples/storm_examples/run_context_profile_pareto.py examples/storm_examples/run_longmemeval_e2e_v60.py tests/test_public_benchmark_contracts.py tests/test_public_benchmarks_v55.py tests/test_qasper_generation_v55.py tests/test_memory_context_public_benchmarks_v56.py
git add -u tests/test_paperstorm_v60_benchmarks.py
git commit -m "refactor: unify public benchmark retrieval contracts"
```

## Task 8: 删除 toy 评测与旧 RAG 实现

**Files:**
- Delete: `knowledge_storm/paperstorm_eval_v4.py`
- Delete: `knowledge_storm/paperstorm_ablation_v41.py`
- Delete: `knowledge_storm/paperstorm_real_eval_v52.py`
- Delete: `knowledge_storm/paperstorm_eval_v54.py`
- Delete: `knowledge_storm/paperstorm_rag.py`
- Delete: `knowledge_storm/paperstorm_rag_benchmark.py`
- Delete: `knowledge_storm/paperstorm_multi_task_benchmark.py`
- Delete: `knowledge_storm/paperstorm_release.py`
- Delete corresponding obsolete tests after behavior coverage is moved to Tasks 2-7.
- Modify: `knowledge_storm/paperstorm_service.py`
- Modify: `examples/storm_examples/paperstorm_service_api.py`
- Modify: `frontend/paperstorm_dashboard/app.js`
- Modify: `frontend/paperstorm_dashboard/index.html`

**Step 1: 建立删除清单测试**

在 `test_paperstorm_module_boundaries.py` 增加：

```python
def test_removed_legacy_modules_do_not_exist(self):
    removed = {
        "paperstorm_eval_v4.py",
        "paperstorm_ablation_v41.py",
        "paperstorm_real_eval_v52.py",
        "paperstorm_eval_v54.py",
        "paperstorm_rag.py",
        "paperstorm_rag_benchmark.py",
        "paperstorm_multi_task_benchmark.py",
        "paperstorm_release.py",
    }
    existing = {path.name for path in PACKAGE.glob("*.py")}
    self.assertFalse(removed & existing)
```

**Step 2: 迁移唯一仍有价值的 helper**

删除前确认：embedding/chunk/compression helper 已分别进入 `retrieval.py`、`document_ingestion.py`、`context_engine.py`；公开 metrics 已进入 `evaluation/public_benchmarks/metrics.py`；production controls 已进入 `control_plane.py`。

**Step 3: 删除旧 service methods 和前端调用**

彻底移除 `_evaluation_v54_*`、`run_langgraph_benchmark_v44()`、`run_production_benchmark_v45()` 及其 getter。开发者控制台只展示 Benchmark Registry 中当前公开 Benchmark，不再展示旧 v4/v5 内部按钮。

**Step 4: 删除旧测试**

删除只证明 toy 数值或旧文件名存在的测试：

```text
tests/test_paperstorm_eval_v4.py
tests/test_paperstorm_eval_v54.py
tests/test_paperstorm_real_eval_v52.py
tests/test_paperstorm_rag_v3.py
tests/test_paperstorm_multi_task_benchmark.py
tests/test_paperstorm_release_demo.py
```

若其中有 ACL、引用、缓存或恢复行为断言，先迁入对应稳定模块测试再删除；不得仅为减少测试数量而丢失行为覆盖。

**Step 5: 验证无引用**

```powershell
Select-String -Path knowledge_storm\*.py,examples\storm_examples\*.py,frontend\paperstorm_dashboard\*.js,tests\*.py -Pattern 'paperstorm_eval_v4|paperstorm_eval_v54|paperstorm_real_eval_v52|paperstorm_rag_benchmark|paperstorm_multi_task_benchmark|PaperStormRAGIndex'
D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest tests.test_paperstorm_module_boundaries tests.test_paperstorm_service tests.test_paperstorm_final_packaging -v
```

Expected: 搜索无命中，测试通过。

**Step 6: 提交**

```powershell
git add knowledge_storm/paperstorm_service.py examples/storm_examples/paperstorm_service_api.py frontend/paperstorm_dashboard/app.js frontend/paperstorm_dashboard/index.html tests/test_paperstorm_module_boundaries.py tests/test_paperstorm_service.py tests/test_paperstorm_final_packaging.py
git add -u knowledge_storm tests
git commit -m "refactor: remove legacy RAG and toy evaluations"
```

## Task 9: 更新专业文档与改进前后证据

**Files:**
- Modify: `README.md`
- Modify: `docs/RAG_BAD_CASES_AND_ROADMAP.md`
- Create: `docs/RAG_P0_IMPLEMENTATION_REPORT.md`
- Modify: `docs/UPDATE_PLAN.md` if present; otherwise modify the repository's current public update-plan document.

**Step 1: 写实现报告**

报告每项必须按以下结构：

```markdown
## 难点：多条检索链导致线上与 Benchmark 行为不一致

**真实案例：** 同一 query 在论文问答使用 Hybrid，在企业 KB 静默回退 hash/set-overlap，公开 Benchmark 又直接构造 Index。

**根因：** 三个入口分别拥有索引构造、默认 provider、rerank 和 relevance gate 决策。

**改进方案：** 三个入口统一注入 RetrievalPipeline，hash 仅限显式测试配置，stage schema 固定。

**改进结果：** 填入自动化一致性测试的实际 Top-K、stage schema、测试数量和耗时；没有测得的效果不得写“提升”。

**剩余 Gap：** 链路统一不等于召回质量完成，查询改写、多跳、冲突治理和动态 rerank 进入 P1-P4。
```

至少覆盖：PIM 消歧、多入口漂移、静默 fallback、CrossEncoder 延迟、上下文保真、Memory/Evidence 边界、旧 toy 指标误导七项。

**Step 2: README 更新**

README 架构图和模块表只使用稳定模块名；Benchmark 只列 SciFact、QASPER、LongMemEval、Context Pareto 当前入口；明确 smoke 使用 hash 仅为离线确定性验证，quality 才能形成可发布结果。

**Step 3: 路线图勾选 P0**

在 `RAG_BAD_CASES_AND_ROADMAP.md` 标记实际完成项，未实现 P1-P4 保持未完成，不使用“假装已完成”的表述。

**Step 4: 文档测试**

```powershell
D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest tests.test_paperstorm_release_docs tests.test_paperstorm_final_packaging -v
Select-String -Path README.md,docs\RAG_P0_IMPLEMENTATION_REPORT.md -Pattern 'paperstorm_.*_v[0-9]+'
```

Expected: 无内部版本化模块名；历史 benchmark 数值可出现发布版本，但须标注数据集、profile 和证据等级。

**Step 5: 提交**

```powershell
git add README.md docs/RAG_BAD_CASES_AND_ROADMAP.md docs/RAG_P0_IMPLEMENTATION_REPORT.md docs/UPDATE_PLAN.md tests/test_paperstorm_release_docs.py tests/test_paperstorm_final_packaging.py
git commit -m "docs: document unified RAG architecture and gaps"
```

## Task 10: 全量回归、服务冒烟与发布检查

**Files:**
- Modify only if failures prove a product regression in files changed by Tasks 1-9.

**Step 1: 静态边界检查**

```powershell
Select-String -Path knowledge_storm\*.py -Pattern 'paperstorm_.*_v[0-9]+'
Select-String -Path knowledge_storm\*.py,examples\storm_examples\*.py -Pattern 'PaperStormRAGIndex|legacy_fallback|set_overlap'
```

Expected: 无命中。

**Step 2: 核心测试组**

```powershell
D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest tests.test_retrieval tests.test_retrieval_pipeline tests.test_context_engine tests.test_memory_store tests.test_conversation_runtime tests.test_control_plane tests.test_public_benchmark_contracts -v
```

**Step 3: 全量离线测试**

```powershell
$env:PAPERSTORM_OFFLINE_TESTS='1'
D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

Expected: 不下载模型、不访问 arXiv、不调用真实 LLM；需要真实网络的测试必须以显式 integration marker/环境变量跳过。

**Step 4: FastAPI 和 CLI 冒烟**

```powershell
D:\SOFTWARE\spyder\envs\storm\python.exe -c "from examples.storm_examples.paperstorm_service_api import app; assert app is not None; print(len(app.routes))"
D:\SOFTWARE\spyder\envs\storm\python.exe examples\storm_examples\start_paperstorm_service.py --help
D:\SOFTWARE\spyder\envs\storm\python.exe examples\storm_examples\run_paperstorm_public_benchmark.py --help
```

**Step 5: 运行确定性跨入口契约并记录结果**

```powershell
D:\SOFTWARE\spyder\envs\storm\python.exe -m unittest tests.test_retrieval_pipeline.RetrievalEntryPointContractTest -v
```

把实际排名、stage schema 和耗时写入 `docs/RAG_P0_IMPLEMENTATION_REPORT.md`。该测试只证明实现一致性，不包装为公开质量分数。

**Step 6: 检查工作树，保护用户文件**

```powershell
git status --short
git diff --check
git diff --stat HEAD~10..HEAD
```

不得 stage 或删除用户现有的 `.codex-temp/`、patch 文件以及 `docs/DESIGN_SOURCES.md` 的删除状态。

**Step 7: 最终提交**

```powershell
git add docs/RAG_P0_IMPLEMENTATION_REPORT.md
git commit -m "test: verify unified PaperStorm RAG pipeline"
```

## 最终验收表

| 维度 | 改进前 | P0 验收目标 | 验证方式 |
|---|---|---|---|
| 检索实现 | QA、企业 KB、Benchmark 多条链 | 单一 `RetrievalPipeline` | 跨入口 Top-K 契约测试 |
| 默认 embedding | 企业 KB 可默认 hash | 真实服务默认 sentence-transformer | provider contract test |
| fallback | 异常可静默降级 set-overlap | 索引错误显式失败并可追踪 | migration/error test |
| stage trace | 各入口字段不同 | retrieve/fuse/rerank/gate 固定 schema | stage schema test |
| 内部命名 | v4.1/v5.6 类名与文件名 | 稳定模块名 + schema revision | AST import boundary test |
| Benchmark | toy 与公开评测混杂 | 只保留公开/诊断性 Benchmark | catalog/API test |
| API | 暴露 v44/v45/v54 路径 | 稳定生产 API + `/benchmarks` | OpenAPI path test |
| 兼容性 | 旧索引可能被错误读取 | 明确重建提示；SQLite 会话/记忆兼容 | migration + persistence tests |
| 文档可信度 | 旧版本名和 toy 数字干扰 | 真实案例、证据等级、剩余 gap 明示 | docs/package tests |

## 实施纪律

1. 每个 Task 必须先看到新增测试因预期原因失败，再写最小实现使其通过。
2. 不在重命名提交里夹带 P1-P4 算法增强，便于定位行为回归。
3. 不使用 broad `except Exception: pass` 保护旧 fallback；可恢复错误必须分类并进入 trace。
4. Smoke 测试中的 hash embedding 只能通过显式注入出现，不能成为生产默认值。
5. 删除测试前必须指出其行为由哪个新测试接管；只验证旧版本号存在的测试可直接删除。
6. 不修改 Stanford STORM/Co-STORM 原始算法目录，除非统一入口所需的适配 bug 已由回归测试证明。
7. 每个 commit 后执行对应目标测试；Task 10 才声明整体完成。
