let sseSource = null;
const DASHBOARD_VERSION = "v4.5";
let activeProductMode = "chat";

async function loadDashboard() {
  try {
    setDashboardMode(initialModeFromUrl());
    initializeServiceUrl();
    const data = window.PAPERSTORM_SAMPLE_DATA || await fetchSampleData();
    renderDashboard(data);
    setStatus("sample data", "idle");
  } catch (error) {
    document.querySelector("#task-list").innerHTML =
      `<div class="item">请先运行 <code>python examples/storm_examples/build_paperstorm_demo_bundle.py --output-dir frontend/paperstorm_dashboard</code></div>`;
    document.querySelector("#project-version").textContent = "no data";
    setStatus(error.message, "error");
  }
}

function renderDashboard(data) {
  renderProject(data.project || {});
  renderTasks(data.tasks || []);
  renderScorecard(data.scorecard || {});
  renderArticle(data.article || {});
  renderQA(data.qa || {});
  renderTrace(data.trace || {});
  renderMultiAgent(data.multi_agent || {}, data.agent_trace || []);
  renderProcessDetails(data.process || {});
  renderPipelineWorker(data.pipeline_worker || {}, data.service_snapshot || {});
  renderTaskError((data.tasks || [])[0] || {});
  renderStress(data.stress_report || {});
  renderRAGEvaluationV4(data.rag_evaluation_v4 || {});
  renderRAGEvaluationV41(data.rag_evaluation_v41 || {});
}

async function runRAGEvaluationV4() {
  try {
    setStatus("running RAG evaluation v4.0", "running");
    setButtonBusy("run-rag-eval-v4", true, "评测中");
    const report = await fetchJson("/evaluations/rag-v4", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({top_k: 5}),
    });
    renderRAGEvaluationV4(report);
    setStatus(`RAG eval completed: ${report.metrics?.passed_cases || 0}/${report.metrics?.total_cases || 0}`, "success");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setButtonBusy("run-rag-eval-v4", false);
  }
}

async function loadRAGEvaluationV4() {
  try {
    setStatus("loading latest RAG evaluation", "loading");
    setButtonBusy("load-rag-eval-v4", true, "加载中");
    const report = await fetchJson("/evaluations/rag-v4/latest");
    renderRAGEvaluationV4(report);
    setStatus("latest RAG evaluation loaded", Object.keys(report).length ? "success" : "idle");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setButtonBusy("load-rag-eval-v4", false);
  }
}

async function runRAGEvaluationV41() {
  try {
    setStatus("running v4.1 retrieval ablation", "running");
    setButtonBusy("run-rag-eval-v41", true, "八组实验运行中");
    const report = await fetchJson("/evaluations/rag-v41", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({top_k: 5, backend: "deterministic"}),
    });
    renderRAGEvaluationV41(report);
    setStatus(`v4.1 ablation completed: ${report.experiments?.length || 0} experiments`, "success");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setButtonBusy("run-rag-eval-v41", false);
  }
}

async function loadRAGEvaluationV41() {
  try {
    setStatus("loading latest v4.1 ablation", "loading");
    setButtonBusy("load-rag-eval-v41", true, "加载中");
    const report = await fetchJson("/evaluations/rag-v41/latest");
    renderRAGEvaluationV41(report);
    setStatus("latest v4.1 ablation loaded", Object.keys(report).length ? "success" : "idle");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setButtonBusy("load-rag-eval-v41", false);
  }
}

async function fetchSampleData() {
  const response = await fetch("sample_data.json");
  if (!response.ok) {
    throw new Error("sample_data.json not found");
  }
  return response.json();
}

async function loadServiceTask() {
  const taskId = getSelectedTaskId();
  if (!taskId) {
    setStatus("请输入 task_id");
    return;
  }
  setStatus("loading service task", "loading");
  try {
    setButtonBusy("load-service-task", true, "加载中");
    connectSSE(taskId);
    const data = await fetchJson(`/research-tasks/${encodeURIComponent(taskId)}/dashboard`);
    renderDashboard(data);
    setStatus(`service task ${taskId}`, statusTone((data.tasks || [])[0]?.status));
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setButtonBusy("load-service-task", false);
  }
}

function setDashboardMode(mode) {
  if (mode === "developer") {
    document.body.dataset.mode = "developer";
    document.querySelector("#show-research-mode").classList.toggle("active", false);
    document.querySelector("#show-chat-mode").classList.toggle("active", false);
    document.querySelector("#show-developer-mode").classList.add("active");
    document.querySelector("#show-developer-mode").textContent = "返回产品界面";
    setStatus("developer console", "idle");
    return;
  }
  activeProductMode = mode;
  const isChat = mode === "chat";
  document.body.dataset.mode = isChat ? "chat" : "research";
  document.querySelector("#show-research-mode").classList.toggle("active", !isChat);
  document.querySelector("#show-chat-mode").classList.toggle("active", isChat);
  const developerButton = document.querySelector("#show-developer-mode");
  developerButton.classList.remove("active");
  developerButton.textContent = "开发者控制台";
  setStatus(isChat ? "chat mode ready" : "research workflow ready", "idle");
}

function initialModeFromUrl() {
  const mode = new URLSearchParams(window.location.search).get("mode");
  return mode === "developer" || mode === "research" || mode === "chat" ? mode : "chat";
}

function toggleDeveloperMode() {
  if (document.body.dataset.mode === "developer") {
    setDashboardMode(activeProductMode);
    return;
  }
  activeProductMode = document.body.dataset.mode === "chat" ? "chat" : "research";
  setDashboardMode("developer");
}

async function loadSampleData() {
  try {
    setButtonBusy("load-sample-data", true, "加载中");
    const data = window.PAPERSTORM_SAMPLE_DATA || await fetchSampleData();
    renderDashboard(data);
    setStatus("sample data ready", "success");
  } finally {
    setButtonBusy("load-sample-data", false);
  }
}

async function submitTask() {
  const payload = {
    topic: document.querySelector("#task-topic").value.trim(),
    run_mode: document.querySelector("#task-run-mode").value,
    retriever: document.querySelector("#task-retriever").value,
    output_language: document.querySelector("#task-output-language").value,
    expected_keywords: splitKeywords(document.querySelector("#task-expected-keyword").value),
    forbidden_keywords: splitKeywords(document.querySelector("#task-forbidden-keyword").value),
    max_conv_turn: 1,
    max_perspective: 1,
    search_top_k: 2,
    max_thread_num: 1,
  };
  if (!payload.topic) {
    setStatus("请输入 topic", "error");
    return;
  }
  try {
    setStatus("submitting task", "loading");
    setButtonBusy("submit-task", true, "提交中");
    const task = await fetchJson("/research-tasks", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    document.querySelector("#service-task-id").value = task.task_id;
    connectSSE(task.task_id);
    await fetchTaskList();
    setStatus(`created ${task.task_id}`, statusTone(task.status));
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setButtonBusy("submit-task", false);
  }
}

async function runSelectedTask() {
  const taskId = getSelectedTaskId();
  if (!taskId) {
    setStatus("请输入 task_id", "error");
    return;
  }
  try {
    connectSSE(taskId);
    setStatus(`running ${taskId}`, "running");
    setButtonBusy("run-selected-task", true, "运行中");
    const task = await fetchJson(`/research-tasks/${encodeURIComponent(taskId)}/run`, {
      method: "POST",
    });
    setStatus(`run ${task.status}`, statusTone(task.status));
    await pollSelectedTask();
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setButtonBusy("run-selected-task", false);
  }
}

async function pollSelectedTask() {
  const taskId = getSelectedTaskId();
  if (!taskId) {
    setStatus("请输入 task_id", "error");
    return;
  }
  try {
    connectSSE(taskId);
    setStatus(`polling ${taskId}`, "loading");
    setButtonBusy("poll-selected-task", true, "轮询中");
    const data = await fetchJson(`/research-tasks/${encodeURIComponent(taskId)}/dashboard`);
    renderDashboard(data);
    const taskStatus = (data.tasks || [])[0]?.status || "unknown";
    setStatus(`polled ${taskStatus}`, statusTone(taskStatus));
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setButtonBusy("poll-selected-task", false);
  }
}

async function fetchTaskList() {
  try {
    setStatus("refreshing tasks", "loading");
    setButtonBusy("refresh-task-list", true, "刷新中");
    const data = await fetchJson("/research-tasks");
    renderTasks(data.tasks || []);
    setStatus(`tasks ${(data.tasks || []).length}`, "success");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setButtonBusy("refresh-task-list", false);
  }
}

async function askResearchAgent() {
  const question = document.querySelector("#research-question").value.trim();
  if (!question) {
    setStatus("请输入问题", "error");
    return;
  }
  const payload = {
    question,
    topic: document.querySelector("#task-topic").value.trim() || question,
    task_id: getSelectedTaskId() || undefined,
    run_mode: document.querySelector("#task-run-mode").value,
    retriever: document.querySelector("#task-retriever").value,
    output_language: document.querySelector("#task-output-language").value,
    expected_keywords: splitKeywords(document.querySelector("#task-expected-keyword").value),
    forbidden_keywords: splitKeywords(document.querySelector("#task-forbidden-keyword").value),
  };
  try {
    setStatus("research qa asking", "loading");
    setButtonBusy("ask-research-agent", true, "回答中");
    appendSSEEvent("planning", JSON.stringify({question}));
    const answer = await fetchJson("/research-agent/ask", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    if (answer.used_task_id) {
      document.querySelector("#service-task-id").value = answer.used_task_id;
      connectSSE(answer.used_task_id);
    }
    renderResearchQA(answer);
    setStatus(`qa ${answer.decision?.action || "answered"}`, answer.grounded ? "success" : "error");
  } catch (error) {
    setStatus(error.message, "error");
    renderResearchQA({answer: error.message, citations: [], decision: {action: "failed"}, evidence_sufficiency: {}});
  } finally {
    setButtonBusy("ask-research-agent", false);
  }
}

async function createChatSession() {
  const runMode = document.querySelector("#chat-run-mode")?.value
    || document.querySelector("#task-run-mode").value;
  const retriever = document.querySelector("#chat-retriever")?.value
    || document.querySelector("#task-retriever").value;
  document.querySelector("#task-run-mode").value = runMode;
  document.querySelector("#task-retriever").value = retriever;
  const payload = {
    title: document.querySelector("#task-topic").value.trim() || "PaperStorm Chat",
    topic: document.querySelector("#task-topic").value.trim(),
    run_mode: runMode,
    retriever: retriever,
    output_language: document.querySelector("#task-output-language").value,
    expected_keywords: splitKeywords(document.querySelector("#task-expected-keyword").value),
    forbidden_keywords: splitKeywords(document.querySelector("#task-forbidden-keyword").value),
    context_window_size: 6,
    context_token_limit: Number(document.querySelector("#chat-context-token-limit").value) || 4096,
    user_id: document.querySelector("#chat-user-id").value.trim() || "local-user",
    tenant_id: document.querySelector("#chat-tenant-id").value.trim() || "local",
    memory_enabled: document.querySelector("#chat-memory-enabled").checked,
  };
  try {
    setDashboardMode("chat");
    setStatus("creating chat session", "loading");
    setButtonBusy("create-chat-session", true, "创建中");
    const session = await fetchJson("/chat/sessions", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    document.querySelector("#chat-session-id").value = session.chat_id;
    renderChatSession(session);
    setStatus(`chat ${session.chat_id}`, "success");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setButtonBusy("create-chat-session", false);
  }
}

async function sendChatMessage() {
  let chatId = document.querySelector("#chat-session-id").value.trim();
  const message = document.querySelector("#chat-message-input").value.trim();
  if (!message) {
    setStatus("请输入聊天问题", "error");
    return;
  }
  try {
    setDashboardMode("chat");
    setStatus("chat agent thinking", "loading");
    setButtonBusy("send-chat-message", true, "思考中");
    if (!chatId) {
      await createChatSession();
      chatId = document.querySelector("#chat-session-id").value.trim();
    }
    appendChatMessage({role: "user", content: message});
    const reply = await fetchJson(`/chat/sessions/${encodeURIComponent(chatId)}/messages`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({message}),
    });
    if (reply.used_task_id) {
      document.querySelector("#service-task-id").value = reply.used_task_id;
      connectSSE(reply.used_task_id);
    }
    renderChatSession(reply);
    setStatus(
      reply.retrieval_triggered ? "chat answered after research" : "chat answered from memory/kb",
      reply.research_answer?.grounded ? "success" : "running",
    );
  } catch (error) {
    setStatus(error.message, "error");
    appendChatMessage({role: "assistant", content: error.message});
  } finally {
    setButtonBusy("send-chat-message", false);
  }
}

async function loadChatContext() {
  const chatId = document.querySelector("#chat-session-id").value.trim();
  if (!chatId) {
    setStatus("请先创建聊天", "error");
    return;
  }
  try {
    setButtonBusy("refresh-chat-context", true, "刷新中");
    const context = await fetchJson(`/chat/sessions/${encodeURIComponent(chatId)}/context`);
    renderContextState(context);
    setStatus("context state loaded", "success");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setButtonBusy("refresh-chat-context", false);
  }
}

async function compactChatContext() {
  const chatId = document.querySelector("#chat-session-id").value.trim();
  if (!chatId) {
    setStatus("请先创建聊天", "error");
    return;
  }
  try {
    setButtonBusy("compact-chat-context", true, "压缩中");
    const result = await fetchJson(`/chat/sessions/${encodeURIComponent(chatId)}/context/compact`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({force: true}),
    });
    document.querySelector("#chat-compaction-id").value = result.compaction_id || "";
    renderContextState({
      context_meter: result.context_meter,
      compressed_context: {
        status: result.status,
        compaction_id: result.compaction_id,
        summary: result.summary_text,
        handoff: result.summary,
        artifact_refs: result.artifact_refs,
      },
      context_view: result.messages,
      events: [],
    });
    setStatus(`context ${result.status}`, result.status === "fallback_original" ? "error" : "success");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setButtonBusy("compact-chat-context", false);
  }
}

async function restoreChatContext() {
  const chatId = document.querySelector("#chat-session-id").value.trim();
  const compactionId = document.querySelector("#chat-compaction-id").value.trim();
  if (!chatId || !compactionId) {
    setStatus("需要 Chat ID 和 Compaction ID", "error");
    return;
  }
  try {
    setButtonBusy("restore-chat-context", true, "恢复中");
    const result = await fetchJson(`/chat/sessions/${encodeURIComponent(chatId)}/context/restore`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({compaction_id: compactionId}),
    });
    document.querySelector("#chat-context-window").textContent = JSON.stringify(result.messages || [], null, 2);
    document.querySelector("#chat-compressed-context").textContent = JSON.stringify({
      status: "restored",
      compaction_id: compactionId,
      raw_messages_unchanged: result.raw_messages_unchanged,
    }, null, 2);
    setStatus(`restored ${result.messages?.length || 0} raw messages`, "success");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setButtonBusy("restore-chat-context", false);
  }
}

async function runContextBenchmarkV42() {
  try {
    setButtonBusy("run-context-v42-benchmark", true, "运行中");
    const report = await fetchJson("/evaluations/context-v42", {method: "POST"});
    renderContextBenchmarkV42(report);
    setStatus("context benchmark completed", "success");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setButtonBusy("run-context-v42-benchmark", false);
  }
}

function selectedMemoryNamespace() {
  const raw = (document.querySelector("#chat-user-id").value.trim() || "local-user").toLowerCase();
  const safe = raw.replace(/[^a-z0-9._-]+/g, "-").replace(/^[-.]+|[-.]+$/g, "") || "local-user";
  return `user/${safe.slice(0, 128)}`;
}

async function searchChatMemory() {
  const namespace = selectedMemoryNamespace();
  const query = document.querySelector("#chat-memory-query").value.trim() || "偏好";
  try {
    setButtonBusy("search-chat-memory", true, "查询中");
    const result = await fetchJson("/memories/search", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({namespace, query, top_k: 8}),
    });
    renderLongTermMemory(result);
    setStatus(`memory recall ${result.results?.length || 0} hits`, "success");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setButtonBusy("search-chat-memory", false);
  }
}

async function updateMemorySetting() {
  const namespace = selectedMemoryNamespace();
  const enabled = document.querySelector("#chat-memory-enabled").checked;
  try {
    const result = await fetchJson("/memories/settings", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({namespace, enabled}),
    });
    document.querySelector("#chat-memory-write").textContent = JSON.stringify(result, null, 2);
    setStatus(`memory ${enabled ? "enabled" : "disabled"}`, "success");
  } catch (error) {
    setStatus(error.message, "error");
  }
}

async function exportChatMemory() {
  const namespace = selectedMemoryNamespace();
  try {
    setButtonBusy("export-chat-memory", true, "导出中");
    const result = await fetchJson(`/memories/export?namespace=${encodeURIComponent(namespace)}`);
    renderLongTermMemory(result);
    setStatus(`exported ${result.memories?.length || 0} memory records`, "success");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setButtonBusy("export-chat-memory", false);
  }
}

async function deleteChatMemory() {
  const namespace = selectedMemoryNamespace();
  const memoryId = document.querySelector("#chat-memory-id").value.trim();
  if (!memoryId) {
    setStatus("请输入 Memory ID", "error");
    return;
  }
  try {
    setButtonBusy("delete-chat-memory", true, "删除中");
    const result = await fetchJson(
      `/memories/${encodeURIComponent(memoryId)}?namespace=${encodeURIComponent(namespace)}&reason=user_request`,
      {method: "DELETE"},
    );
    document.querySelector("#chat-memory-write").textContent = JSON.stringify(result, null, 2);
    setStatus(`memory ${memoryId} soft deleted`, "success");
    await searchChatMemory();
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setButtonBusy("delete-chat-memory", false);
  }
}

async function runMemoryBenchmarkV43() {
  try {
    setButtonBusy("run-memory-v43-benchmark", true, "运行中");
    const report = await fetchJson("/evaluations/memory-v43", {method: "POST"});
    renderMemoryBenchmarkV43(report);
    setStatus("memory benchmark completed", "success");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setButtonBusy("run-memory-v43-benchmark", false);
  }
}

async function refreshChatGraph() {
  const chatId = document.querySelector("#chat-session-id").value.trim();
  if (!chatId) {
    setStatus("请先创建聊天", "error");
    return;
  }
  try {
    setButtonBusy("refresh-chat-graph", true, "刷新中");
    const encoded = encodeURIComponent(chatId);
    const tenant = encodeURIComponent(document.querySelector("#chat-tenant-id").value.trim() || "local");
    const user = encodeURIComponent(document.querySelector("#chat-user-id").value.trim() || "local-user");
    const [state, history] = await Promise.all([
      fetchJson(`/conversation-graph/threads/${encoded}/state?tenant_id=${tenant}&user_id=${user}`),
      fetchJson(`/conversation-graph/threads/${encoded}/history?limit=30&tenant_id=${tenant}&user_id=${user}`),
    ]);
    document.querySelector("#chat-graph-run").textContent = JSON.stringify(state, null, 2);
    document.querySelector("#chat-checkpoint-history").textContent = JSON.stringify(history, null, 2);
    setStatus(`checkpoint ${history.checkpoints?.length || 0}`, "success");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setButtonBusy("refresh-chat-graph", false);
  }
}

async function runRuntimeBenchmarkV44() {
  try {
    setButtonBusy("run-runtime-v44-benchmark", true, "运行中");
    const report = await fetchJson("/evaluations/runtime-v44", {method: "POST"});
    renderRuntimeBenchmarkV44(report);
    setStatus("runtime v4.4 benchmark completed", "success");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setButtonBusy("run-runtime-v44-benchmark", false);
  }
}

async function loadRuntimeBenchmarkV44() {
  try {
    setButtonBusy("load-runtime-v44-benchmark", true, "加载中");
    const report = await fetchJson("/evaluations/runtime-v44/latest");
    renderRuntimeBenchmarkV44(report);
    setStatus("latest runtime v4.4 benchmark loaded", Object.keys(report).length ? "success" : "idle");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setButtonBusy("load-runtime-v44-benchmark", false);
  }
}

async function runProductionBenchmarkV45() {
  try {
    setButtonBusy("run-production-v45-benchmark", true, "运行中");
    const report = await fetchJson("/evaluations/production-v45", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({request_count: 100}),
    });
    renderProductionBenchmarkV45(report);
    setStatus("production v4.5 benchmark completed", "success");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setButtonBusy("run-production-v45-benchmark", false);
  }
}

async function loadProductionBenchmarkV45() {
  try {
    setButtonBusy("load-production-v45-benchmark", true, "加载中");
    const report = await fetchJson("/evaluations/production-v45/latest");
    renderProductionBenchmarkV45(report);
    setStatus("latest production v4.5 benchmark loaded", Object.keys(report).length ? "success" : "idle");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setButtonBusy("load-production-v45-benchmark", false);
  }
}

async function loadProductionStatusV45() {
  try {
    const status = await fetchJson("/production/status");
    document.querySelector("#production-v45-status").textContent = JSON.stringify(status, null, 2);
    setStatus("production control plane loaded", "success");
  } catch (error) {
    setStatus(error.message, "error");
  }
}

async function loadProductionTraceV45() {
  const traceId = document.querySelector("#production-v45-trace-id").value.trim();
  if (!traceId) {
    setStatus("请先发送聊天消息或输入 Trace ID", "error");
    return;
  }
  const tenant = encodeURIComponent(document.querySelector("#chat-tenant-id").value.trim() || "local");
  const user = encodeURIComponent(document.querySelector("#chat-user-id").value.trim() || "local-user");
  try {
    const trace = await fetchJson(`/production/traces/${encodeURIComponent(traceId)}?tenant_id=${tenant}&user_id=${user}`);
    document.querySelector("#production-v45-trace").textContent = JSON.stringify(trace, null, 2);
    setStatus(`trace spans ${trace.spans?.length || 0}`, "success");
  } catch (error) {
    setStatus(error.message, "error");
  }
}

async function createEnterpriseKB() {
  const payload = {
    name: document.querySelector("#enterprise-kb-name").value.trim() || "Enterprise Knowledge Base",
    source_paths: splitLines(document.querySelector("#enterprise-kb-source-paths").value),
    expected_keywords: splitKeywords(document.querySelector("#task-expected-keyword").value),
    forbidden_keywords: splitKeywords(document.querySelector("#task-forbidden-keyword").value),
    chunk_size: 500,
    chunk_overlap: 100,
    embedding_provider: "hash",
  };
  if (!payload.source_paths.length) {
    setStatus("请输入本地文档路径", "error");
    return;
  }
  try {
    setStatus("building enterprise kb", "loading");
    setButtonBusy("create-enterprise-kb", true, "建库中");
    const kb = await fetchJson("/enterprise-kbs", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    document.querySelector("#enterprise-kb-id").value = kb.kb_id || "";
    renderEnterpriseKBManifest(kb);
    setStatus(`kb ${kb.kb_id}`, "success");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setButtonBusy("create-enterprise-kb", false);
  }
}

async function createZoteroKB() {
  const termsValue = document.querySelector("#zotero-kb-terms").value.trim();
  const payload = {
    name: document.querySelector("#enterprise-kb-name").value.trim() || "Zotero 论文知识库",
    query_terms: termsValue
      ? termsValue.split(/[,，\s]+/).filter(Boolean)
      : [],
    max_papers: Number(document.querySelector("#zotero-kb-max-papers").value) || 8,
    zotero_root: document.querySelector("#zotero-kb-root").value.trim() || undefined,
    expected_keywords: splitKeywords(document.querySelector("#task-expected-keyword").value),
    forbidden_keywords: splitKeywords(document.querySelector("#task-forbidden-keyword").value),
    embedding_provider: "hash",
  };
  try {
    setStatus("importing papers from Zotero", "loading");
    setButtonBusy("create-zotero-kb", true, "导入中");
    const kb = await fetchJson("/enterprise-kbs/from-zotero", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    document.querySelector("#enterprise-kb-id").value = kb.kb_id || "";
    renderEnterpriseKBManifest(kb);
    const count = (kb.source_papers || []).length;
    setStatus(`kb ${kb.kb_id} · ${count} 篇论文 · root=${kb.zotero_root}`, "success");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setButtonBusy("create-zotero-kb", false);
  }
}

async function listEnterpriseKB() {
  try {
    setStatus("loading enterprise kbs", "loading");
    setButtonBusy("list-enterprise-kb", true, "刷新中");
    const result = await fetchJson("/enterprise-kbs");
    const latest = (result.knowledge_bases || []).slice(-1)[0] || {};
    if (latest.kb_id) {
      document.querySelector("#enterprise-kb-id").value = latest.kb_id;
    }
    renderEnterpriseKBManifest(result);
    setStatus(`knowledge bases ${(result.knowledge_bases || []).length}`, "success");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setButtonBusy("list-enterprise-kb", false);
  }
}

async function askEnterpriseKB() {
  const kbId = document.querySelector("#enterprise-kb-id").value.trim();
  const question = document.querySelector("#enterprise-kb-question").value.trim();
  if (!kbId || !question) {
    setStatus("请输入 KB ID 和问题", "error");
    return;
  }
  try {
    setStatus("enterprise kb asking", "loading");
    setButtonBusy("ask-enterprise-kb", true, "回答中");
    const answer = await fetchJson(`/enterprise-kbs/${encodeURIComponent(kbId)}/ask`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({question, top_k: 4}),
    });
    renderEnterpriseKBAnswer(answer);
    setStatus(answer.grounded ? "enterprise kb answered" : "enterprise kb no evidence", answer.grounded ? "success" : "error");
  } catch (error) {
    setStatus(error.message, "error");
    renderEnterpriseKBAnswer({answer: error.message, citations: [], retrieval: {}});
  } finally {
    setButtonBusy("ask-enterprise-kb", false);
  }
}

async function fetchJson(path, options) {
  const baseUrl = getServiceBaseUrl();
  const response = await fetch(`${baseUrl}${path}`, options);
  if (!response.ok) {
    throw new Error(`service ${response.status}`);
  }
  return response.json();
}

function connectSSE(taskId = "") {
  const baseUrl = getServiceBaseUrl();
  if (!baseUrl) {
    appendSSEEvent("error", "missing service url");
    return;
  }
  if (sseSource) {
    sseSource.close();
  }
  const query = taskId ? `?task_id=${encodeURIComponent(taskId)}` : "";
  sseSource = new EventSource(`${baseUrl}/events${query}`);
  appendSSEEvent("connect", `${baseUrl}/events${query}`);
  setStatus("SSE connecting", "loading");

  ["service", "heartbeat", "task_status"].forEach(eventName => {
    sseSource.addEventListener(eventName, event => {
      appendSSEEvent(eventName, event.data);
      if (eventName === "task_status") {
        try {
          const payload = JSON.parse(event.data);
          setStatus(`SSE ${payload.task_status || "unknown"}`, statusTone(payload.task_status));
        } catch {
          setStatus("SSE task update", "running");
        }
      }
    });
  });

  sseSource.onerror = () => {
    appendSSEEvent("error", "SSE disconnected or service unavailable");
    setStatus("SSE disconnected", "error");
  };
}

function initializeServiceUrl() {
  const input = document.querySelector("#service-url");
  if (!input.value.trim()) {
    input.value = window.location.origin;
  }
}

function getServiceBaseUrl() {
  const input = document.querySelector("#service-url");
  const value = input.value.trim().replace(/\/+$/, "");
  return value || window.location.origin;
}

function getSelectedTaskId() {
  return document.querySelector("#service-task-id").value.trim();
}

function splitKeywords(value) {
  return value.split(",").map(item => item.trim()).filter(Boolean);
}

function splitLines(value) {
  return value.split(/\r?\n/).map(item => item.trim()).filter(Boolean);
}

function renderProject(project) {
  document.querySelector("#project-version").textContent = DASHBOARD_VERSION;
}

function renderTasks(tasks) {
  document.querySelector("#task-list").innerHTML = tasks.map(task => `
    <div class="item task-row" data-task-id="${escapeHtml(task.task_id || "")}">
      <div class="label">${escapeHtml(task.task_id || "")}</div>
      <div class="value">${escapeHtml(task.status || "")}</div>
      <p>${escapeHtml(task.topic || "")}</p>
      <div class="label">created: ${escapeHtml(formatTimestamp(task.created_at || ""))}</div>
      <div class="label">updated: ${escapeHtml(formatTimestamp(task.updated_at || ""))}</div>
      <button type="button" data-select-task="${escapeHtml(task.task_id || "")}">选择</button>
    </div>
  `).join("");
  document.querySelectorAll("[data-select-task]").forEach(button => {
    button.addEventListener("click", () => {
      document.querySelector("#service-task-id").value = button.dataset.selectTask;
      setStatus(`selected ${button.dataset.selectTask}`, "idle");
      connectSSE(button.dataset.selectTask);
    });
  });
}

function renderScorecard(scorecard) {
  const scores = scorecard.scores || {};
  document.querySelector("#scorecard").innerHTML = Object.entries(scores).map(([name, value]) => metric(name, value)).join("");
}

function renderArticle(article) {
  document.querySelector("#article-content").textContent = article.content || "";
}

function renderQA(qa) {
  const citations = qa.citations || [];
  document.querySelector("#qa-answer").innerHTML = `
    <div class="item">
      <div class="label">${escapeHtml(qa.question || "知识库问题")}</div>
      <p>${escapeHtml(qa.answer || "")}</p>
      <div class="label">grounded: ${Boolean(qa.grounded)}</div>
    </div>
    ${citations.map(c => `<div class="item">${escapeHtml(c.title || c.url || "")}</div>`).join("")}
  `;
}

function renderRAGEvaluationV4(report) {
  const metrics = report.metrics || {};
  const metricNames = [
    "total_cases",
    "pass_rate",
    "retrieval_recall_at_k",
    "retrieval_precision_at_k",
    "mrr",
    "ndcg_at_k",
    "citation_precision",
    "citation_recall",
    "abstention_accuracy",
    "p95_latency_ms",
  ];
  document.querySelector("#rag-eval-v4-metrics").innerHTML = metricNames
    .map(name => metric(name, metrics[name]))
    .join("");
  document.querySelector("#rag-eval-v4-failures").textContent =
    JSON.stringify(metrics.failure_counts || {}, null, 2);
  document.querySelector("#rag-eval-v4-dataset").textContent =
    JSON.stringify(report.dataset || {}, null, 2);
  const badCases = report.bad_cases || [];
  document.querySelector("#rag-eval-v4-bad-cases").innerHTML = badCases.length
    ? badCases.slice(0, 30).map(item => `
      <div class="item rejected">
        <strong>${escapeHtml(item.case_id || "")}</strong>
        <span class="soft-badge">${escapeHtml(item.failure_stage || "unknown")}</span>
        <p>${escapeHtml(item.query || "")}</p>
        <div class="label">recall@k ${escapeHtml(item.retrieval?.recall_at_k ?? "")} · MRR ${escapeHtml(item.retrieval?.mrr ?? "")} · citations ${escapeHtml(item.answer?.citation_precision ?? "")}</div>
      </div>
    `).join("")
    : `<div class="item">尚未运行评测，或当前报告没有坏例。</div>`;
}

function renderRAGEvaluationV41(report) {
  const experiments = report.experiments || [];
  document.querySelector("#rag-eval-v41-table").innerHTML = experiments.length
    ? experiments.map(item => `
      <tr>
        <td>${escapeHtml(item.experiment_id || "")}</td>
        <td>${escapeHtml(item.metrics?.retrieval_recall_at_k ?? "")}</td>
        <td>${escapeHtml(item.metrics?.mrr ?? "")}</td>
        <td>${escapeHtml(item.metrics?.ndcg_at_k ?? "")}</td>
        <td>${escapeHtml(item.metrics?.p95_latency_ms ?? "")}</td>
      </tr>
    `).join("")
    : `<tr><td colspan="5">尚未运行 V4.1 消融实验。</td></tr>`;
  document.querySelector("#rag-eval-v41-best").textContent = JSON.stringify({
    best_by_recall: report.best_by_recall || null,
    best_by_ndcg: report.best_by_ndcg || null,
    dataset_version: report.dataset_version || null,
  }, null, 2);
  document.querySelector("#rag-eval-v41-notes").textContent =
    JSON.stringify(report.notes || [], null, 2);
}

function renderResearchQA(answer) {
  document.querySelector("#research-answer").innerHTML = `
    <strong>Agent</strong>
    <p>${escapeHtml(answer.answer || "暂无回答。")}</p>
    <div class="label">grounded: ${Boolean(answer.grounded)} · task_id: ${escapeHtml(answer.used_task_id || "")}${answer.retrieval_stack ? ` · stack: ${escapeHtml(answer.retrieval_stack)}` : ""}</div>
  `;
  document.querySelector("#research-decision").textContent =
    JSON.stringify(answer.decision || {}, null, 2);
  document.querySelector("#research-sufficiency").textContent =
    JSON.stringify(answer.evidence_sufficiency || {}, null, 2);
  const citations = answer.citations || [];
  document.querySelector("#research-citations").innerHTML = citations.length
    ? citations.map(citation => `
      <div class="item">
        <strong>[${escapeHtml(citation.id || "")}] ${escapeHtml(citation.title || citation.document_id || "")}</strong>
        <div class="label">${escapeHtml(citation.source || "")} ${escapeHtml(citation.url || "")}</div>
      </div>
    `).join("")
    : `<div class="item">暂无引用。证据不足时 Agent 会拒答。</div>`;
  (answer.trace || []).forEach(event => {
    appendSSEEvent(event.event || "research_qa", JSON.stringify(event.payload || {}));
  });
}

function renderEnterpriseKBManifest(manifest) {
  document.querySelector("#enterprise-kb-manifest").textContent =
    JSON.stringify(manifest || {}, null, 2);
}

function renderEnterpriseKBAnswer(answer) {
  document.querySelector("#enterprise-kb-answer").innerHTML = `
    <strong>Enterprise KB Agent</strong>
    <p>${escapeHtml(answer.answer || "暂无回答。")}</p>
    <div class="label">grounded: ${Boolean(answer.grounded)} · kb_id: ${escapeHtml(answer.kb_id || "")}</div>
  `;
  document.querySelector("#enterprise-kb-retrieval").textContent =
    JSON.stringify(answer.retrieval || {}, null, 2);
  document.querySelector("#enterprise-kb-citations").innerHTML = (answer.citations || []).length
    ? answer.citations.map(citation => `
      <div class="item">
        <strong>[${escapeHtml(citation.id || "")}] ${escapeHtml(citation.title || citation.document_id || "")}</strong>
        <div class="label">${escapeHtml(citation.source_type || citation.source || "")} ${escapeHtml(citation.url || "")}</div>
      </div>
    `).join("")
    : `<div class="item">暂无引用。</div>`;
  if (answer.manifest) {
    renderEnterpriseKBManifest(answer.manifest);
  }
}

function renderChatSession(session) {
  const messages = session.messages || [];
  const list = document.querySelector("#chat-message-list");
  if (messages.length) {
    list.innerHTML = messages.map(message => chatBubble(message)).join("");
    list.scrollTop = list.scrollHeight;
  }
  document.querySelector("#chat-context-window").textContent =
    JSON.stringify(session.context_window || [], null, 2);
  document.querySelector("#chat-compressed-context").textContent =
    JSON.stringify(session.compressed_context || {}, null, 2);
  document.querySelector("#chat-memory-context").textContent =
    JSON.stringify(session.memory_context || {}, null, 2);
  renderLongTermMemory(session.long_term_memory || {});
  document.querySelector("#chat-memory-write").textContent =
    JSON.stringify(session.memory_write || {}, null, 2);
  document.querySelector("#chat-graph-run").textContent =
    JSON.stringify(session.graph_run || {}, null, 2);
  const graphRun = session.graph_run || {};
  if (graphRun.trace_id) {
    document.querySelector("#production-v45-trace-id").value = graphRun.trace_id;
  }
  renderContextState(session);
  document.querySelector("#chat-router-decision").textContent =
    JSON.stringify(session.router_decision || {}, null, 2);
  document.querySelector("#chat-tool-decision").textContent =
    JSON.stringify(session.tool_decision || {}, null, 2);
  document.querySelector("#chat-rewritten-query").textContent =
    (session.router_decision || {}).rewritten_query || "";
  if (session.chat_id) {
    document.querySelector("#chat-session-id").value = session.chat_id;
    document.querySelector("#chat-session-id-label").textContent = session.chat_id;
  }
  if (session.user_id) {
    document.querySelector("#chat-user-id").value = session.user_id;
  }
  if (session.tenant_id) {
    document.querySelector("#chat-tenant-id").value = session.tenant_id;
  }
  if (session.run_mode) {
    document.querySelector("#chat-run-mode").value = session.run_mode;
    document.querySelector("#task-run-mode").value = session.run_mode;
  }
  if (session.retriever) {
    document.querySelector("#chat-retriever").value = session.retriever;
    document.querySelector("#task-retriever").value = session.retriever;
  }
  if (typeof session.memory_enabled === "boolean") {
    document.querySelector("#chat-memory-enabled").checked = session.memory_enabled;
  }
  const answer = session.research_answer || {};
  document.querySelector("#research-decision").textContent =
    JSON.stringify(answer.decision || {}, null, 2);
  document.querySelector("#research-sufficiency").textContent =
    JSON.stringify(answer.evidence_sufficiency || {}, null, 2);
  if (answer.citations) {
    renderResearchQA(answer);
  }
}

function renderContextState(state) {
  const meter = state.context_meter || {};
  const ratio = Math.max(0, Math.min(1, Number(meter.usage_ratio || 0)));
  document.querySelector("#context-meter-fill").style.width = `${Math.round(ratio * 100)}%`;
  document.querySelector("#context-meter-fill").dataset.tone = meter.high_watermark
    ? "error"
    : meter.should_compact
      ? "warning"
      : "normal";
  document.querySelector("#chat-context-meter").textContent = JSON.stringify(meter, null, 2);
  document.querySelector("#chat-context-events").textContent = JSON.stringify(
    state.context_events || state.events || [], null, 2
  );
  if (state.context_view) {
    document.querySelector("#chat-context-window").textContent = JSON.stringify(state.context_view, null, 2);
  }
  if (state.compressed_context) {
    document.querySelector("#chat-compressed-context").textContent =
      JSON.stringify(state.compressed_context, null, 2);
    const compactionId = state.compressed_context.compaction_id || state.active_compaction_id;
    if (compactionId) {
      document.querySelector("#chat-compaction-id").value = compactionId;
    }
  }
}

function renderContextBenchmarkV42(report) {
  const metrics = report.metrics || {};
  document.querySelector("#context-v42-metrics").innerHTML = Object.entries(metrics)
    .map(([name, value]) => metric(name, value))
    .join("");
  document.querySelector("#context-v42-summary").textContent =
    JSON.stringify({summary: report.summary || {}, limitations: report.limitations || []}, null, 2);
}

function renderLongTermMemory(payload) {
  document.querySelector("#chat-long-term-memory").textContent = JSON.stringify(payload || {}, null, 2);
  const records = payload.results || payload.memories || [];
  if (records.length && !document.querySelector("#chat-memory-id").value) {
    document.querySelector("#chat-memory-id").value = records[0].id || "";
  }
}

function renderMemoryBenchmarkV43(report) {
  const metrics = report.metrics || {};
  document.querySelector("#memory-v43-metrics").innerHTML = Object.entries(metrics)
    .map(([name, value]) => metric(name, value))
    .join("");
  document.querySelector("#memory-v43-summary").textContent = JSON.stringify({
    architecture: report.architecture || {},
    counts: report.counts || {},
    limitations: report.limitations || [],
  }, null, 2);
}

function renderRuntimeBenchmarkV44(report) {
  const metrics = report.metrics || {};
  document.querySelector("#runtime-v44-metrics").innerHTML = Object.entries(metrics)
    .map(([name, value]) => metric(name, value))
    .join("");
  document.querySelector("#runtime-v44-summary").textContent = JSON.stringify({
    runtime: report.runtime || {},
    paths: report.paths || {},
    limitations: report.limitations || [],
  }, null, 2);
}

function renderProductionBenchmarkV45(report) {
  const metrics = report.metrics || {};
  document.querySelector("#production-v45-metrics").innerHTML = Object.entries(metrics)
    .map(([name, value]) => metric(name, value))
    .join("");
  document.querySelector("#production-v45-status").textContent = JSON.stringify(
    report.control_plane || {}, null, 2
  );
  document.querySelector("#production-v45-summary").textContent = JSON.stringify({
    slo: report.slo || {},
    degradation: report.degradation || {},
    limitations: report.limitations || [],
  }, null, 2);
}

async function runRetrievalRuntimeBenchmark() {
  try {
    setStatus("running retrieval runtime benchmark", "running");
    setButtonBusy("run-retrieval-runtime", true, "对比运行中");
    const report = await fetchJson("/evaluations/retrieval-runtime", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({embedding: "hash", top_k: 5}),
    });
    renderRetrievalRuntimeBenchmark(report);
    const delta = report.deltas?.recall_at_k ?? 0;
    setStatus(
      `retrieval benchmark: v4.1 recall ${report.v41?.recall_at_k} vs legacy ${report.legacy?.recall_at_k} (${delta >= 0 ? "+" : ""}${delta})`,
      "success",
    );
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setButtonBusy("run-retrieval-runtime", false);
  }
}

async function loadContextBenchmarkV42() {
  try {
    const report = await fetchJson("/evaluations/context-v42/latest");
    renderContextBenchmarkV42(report);
  } catch (error) {
    setStatus(error.message, "error");
  }
}

async function loadMemoryBenchmarkV43() {
  try {
    const report = await fetchJson("/evaluations/memory-v43/latest");
    renderMemoryBenchmarkV43(report);
  } catch (error) {
    setStatus(error.message, "error");
  }
}

async function loadLatestBenchmarks() {
  await Promise.allSettled([
    loadRetrievalRuntimeBenchmark(),
    loadContextBenchmarkV42(),
    loadMemoryBenchmarkV43(),
    loadRuntimeBenchmarkV44(),
    loadProductionBenchmarkV45(),
    loadRAGEvaluationV4(),
    loadRAGEvaluationV41(),
  ]);
}

async function loadDeepLink() {
  const params = new URLSearchParams(window.location.search);
  const target = params.get("load");
  if (!target) {
    return;
  }
  try {
    if (target === "bench") {
      await loadLatestBenchmarks();
      setStatus("benchmark results loaded", "success");
    } else if (target.startsWith("chat:")) {
      const chatId = target.slice(5);
      const session = await fetchJson(`/chat/sessions/${encodeURIComponent(chatId)}`);
      renderChatSession(session);
      setStatus(`loaded chat ${chatId}`, "success");
    } else if (target.startsWith("task:")) {
      const taskId = target.slice(5);
      const data = await fetchJson(`/research-tasks/${encodeURIComponent(taskId)}/dashboard`);
      renderDashboard(data);
      setStatus(`loaded task ${taskId}`, "success");
    } else if (target.startsWith("kb:")) {
      const kbId = target.slice(3);
      document.querySelector("#enterprise-kb-id").value = kbId;
      await askEnterpriseKB();
    }
  } catch (error) {
    setStatus(error.message, "error");
  }
}

async function loadRetrievalRuntimeBenchmark() {
  try {
    setStatus("loading latest retrieval benchmark", "loading");
    setButtonBusy("load-retrieval-runtime", true, "加载中");
    const report = await fetchJson("/evaluations/retrieval-runtime/latest");
    renderRetrievalRuntimeBenchmark(report);
    setStatus("latest retrieval benchmark loaded", Object.keys(report).length ? "success" : "idle");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setButtonBusy("load-retrieval-runtime", false);
  }
}

function renderRetrievalRuntimeBenchmark(report) {
  const rows = [
    ["Recall@K", "recall_at_k"],
    ["MRR", "mrr"],
    ["nDCG@K", "ndcg_at_k"],
    ["P95 延迟(ms)", "p95_latency_ms"],
  ];
  const hasReport = Boolean(report && report.deltas);
  document.querySelector("#retrieval-runtime-table").innerHTML = hasReport
    ? rows.map(([label, key]) => `
      <tr>
        <td>${escapeHtml(label)}</td>
        <td>${escapeHtml(report.legacy?.[key] ?? "")}</td>
        <td>${escapeHtml(report.v41?.[key] ?? "")}</td>
        <td>${escapeHtml(report.deltas?.[key] ?? "")}</td>
      </tr>
    `).join("")
    : `<tr><td colspan="4">尚未运行对比，或最近报告为空。</td></tr>`;
  document.querySelector("#retrieval-runtime-summary").textContent = hasReport
    ? JSON.stringify({
        dataset: report.dataset,
        embedding: report.embedding,
        case_count: report.legacy?.case_count,
        relative_recall_gain_pct: report.deltas?.relative_recall_gain_pct,
        stack_meta: report.stack_meta,
      }, null, 2)
    : "尚未运行对比。点击“运行检索对比 Benchmark”生成 legacy vs V4.1 报告；或设置 PAPERSTORM_RETRIEVAL_EMBEDDING=real 后再跑真实向量对比。";
}

function appendChatMessage(message) {
  const list = document.querySelector("#chat-message-list");
  list.insertAdjacentHTML("beforeend", chatBubble(message));
  list.scrollTop = list.scrollHeight;
}

function chatBubble(message) {
  const role = message.role === "user" ? "user" : "assistant";
  const label = role === "user" ? "你" : "PaperStorm";
  const metadata = message.metadata || {};
  const stack = metadata.retrieval_stack
    ? ` · stack: ${escapeHtml(metadata.retrieval_stack)}`
    : "";
  const meta = metadata.used_task_id
    ? `<div class="label">task: ${escapeHtml(metadata.used_task_id)} · retrieval: ${Boolean(metadata.retrieval_triggered)}${stack}</div>`
    : "";
  return `
    <div class="chat-message ${role}">
      <strong>${label}</strong>
      <p>${escapeHtml(message.content || "")}</p>
      ${meta}
    </div>
  `;
}

function renderTrace(trace) {
  const events = trace.events || [];
  document.querySelector("#trace-list").innerHTML = events.map(event => `
    <li class="log-event-${escapeHtml(event.event || "unknown")}">
      <strong>${escapeHtml(event.event || "")}</strong>
      <span>${escapeHtml(formatTimestamp(event.timestamp || event.time || event.created_at || ""))}</span>
      <br>${escapeHtml(event.tool || event.status || event.task_id || "")}
    </li>
  `).join("");
}

function appendSSEEvent(eventName, payload) {
  const list = document.querySelector("#sse-event-list");
  if (!list) {
    return;
  }
  const item = document.createElement("li");
  const parsed = parsePayload(payload);
  const time = formatTimestamp(parsed.timestamp || Date.now());
  item.className = `log-event-${eventName}`;
  item.innerHTML = `<strong>${escapeHtml(eventName)}</strong> <span>${escapeHtml(time)}</span><br>${escapeHtml(formatPayloadForLog(parsed, payload))}`;
  list.prepend(item);
  while (list.children.length > 40) {
    list.removeChild(list.lastChild);
  }
}

function parsePayload(payload) {
  if (typeof payload !== "string") {
    return payload || {};
  }
  try {
    return JSON.parse(payload);
  } catch {
    return {};
  }
}

function formatPayloadForLog(parsed, fallback) {
  if (!parsed || !Object.keys(parsed).length) {
    return String(fallback || "");
  }
  const compact = {
    status: parsed.status,
    task_status: parsed.task_status,
    task_id: parsed.task_id,
    task_count: parsed.task_count,
    topic: parsed.topic,
    message: parsed.message,
  };
  return JSON.stringify(
    Object.fromEntries(Object.entries(compact).filter(([, value]) => value !== undefined && value !== "")),
    null,
    2,
  );
}

function renderMultiAgent(report, agentTrace) {
  const kept = report.kept_results || [];
  const rejected = report.rejected_results || [];
  document.querySelector("#multi-agent").innerHTML = `
    <div class="metric-grid">
      ${metric("queries", (report.query_plan || []).length)}
      ${metric("kept", kept.length)}
      ${metric("rejected", rejected.length)}
      ${metric("agent events", agentTrace.length)}
    </div>
    <div class="stack">
      ${kept.map(item => `<div class="item kept"><strong>保留</strong> ${escapeHtml(item.title || "")}</div>`).join("")}
      ${rejected.map(item => `<div class="item rejected"><strong>过滤</strong> ${escapeHtml(item.title || "")}<br>${escapeHtml(item.reason || "")}</div>`).join("")}
    </div>
  `;
}

function renderProcessDetails(process) {
  document.querySelector("#outline-content").textContent =
    process.outline || "暂无 outline。真实 paperstorm 任务完成后会读取 storm_gen_outline.txt。";
  document.querySelector("#reflection-content").textContent =
    process.reflection || process.run_summary || "暂无 reflection/run_summary。";
  document.querySelector("#plan-content").textContent =
    process.plan || process.raw_search_results || "暂无 plan/search 结果。";
  document.querySelector("#conversation-content").textContent =
    process.conversation || "暂无 conversation_log。真实 research 阶段完成后会显示访谈式调研对话。";
}

function renderPipelineWorker(worker, snapshot) {
  const data = Object.keys(worker).length ? worker : snapshot;
  document.querySelector("#pipeline-worker").innerHTML = [
    "runner",
    "run_mode",
    "retriever",
    "llm_provider",
    "llm_model",
    "status",
    "score",
  ].map(name => metric(name, data[name])).join("");
}

function renderTaskError(task) {
  document.querySelector("#task-error").innerHTML = task.error
    ? `<div class="item rejected">${escapeHtml(task.error)}</div>`
    : `<div class="item">当前任务没有结构化错误。</div>`;
}

function renderStress(report) {
  document.querySelector("#stress-report").innerHTML = [
    "total_tasks",
    "succeeded",
    "failed",
    "failure_rate",
    "avg_latency_sec",
    "p95_latency_sec",
    "max_observed_running",
  ].map(name => metric(name, report[name])).join("");
}

function setStatus(message, tone = "idle") {
  const dot = document.querySelector("#runtime-status-dot");
  document.querySelector("#runtime-status-text").textContent = message;
  dot.className = `status-dot ${tone}`;
}

function statusTone(status) {
  if (status === "succeeded") {
    return "success";
  }
  if (status === "failed") {
    return "error";
  }
  if (status === "running") {
    return "running";
  }
  if (status === "queued") {
    return "loading";
  }
  return "idle";
}

function setButtonBusy(buttonId, busy, busyText = "") {
  const button = document.querySelector(`#${buttonId}`);
  if (!button) {
    return;
  }
  if (!button.dataset.idleText) {
    button.dataset.idleText = button.textContent;
  }
  button.disabled = Boolean(busy);
  button.textContent = busy ? busyText || button.dataset.idleText : button.dataset.idleText;
}

function formatTimestamp(value) {
  if (!value) {
    return "";
  }
  let date;
  if (typeof value === "number") {
    date = new Date(value > 100000000000 ? value : value * 1000);
  } else {
    date = new Date(value);
  }
  if (Number.isNaN(date.getTime())) {
    return String(value);
  }
  return date.toLocaleString("zh-CN", {hour12: false});
}

function metric(name, value) {
  return `
    <div class="metric">
      <div class="label">${escapeHtml(name)}</div>
      <div class="value">${escapeHtml(String(value ?? ""))}</div>
    </div>
  `;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

document.querySelector("#load-service-task").addEventListener("click", loadServiceTask);
document.querySelector("#load-sample-data").addEventListener("click", loadSampleData);
document.querySelector("#submit-task").addEventListener("click", submitTask);
document.querySelector("#run-selected-task").addEventListener("click", runSelectedTask);
document.querySelector("#poll-selected-task").addEventListener("click", pollSelectedTask);
document.querySelector("#refresh-task-list").addEventListener("click", fetchTaskList);
document.querySelector("#ask-research-agent").addEventListener("click", askResearchAgent);
document.querySelector("#service-url").addEventListener("change", () => connectSSE(getSelectedTaskId()));
document.querySelector("#show-research-mode").addEventListener("click", () => setDashboardMode("research"));
document.querySelector("#show-chat-mode").addEventListener("click", () => setDashboardMode("chat"));
document.querySelector("#show-developer-mode").addEventListener("click", toggleDeveloperMode);
document.querySelector("#create-chat-session").addEventListener("click", createChatSession);
document.querySelector("#new-chat-button").addEventListener("click", createChatSession);
document.querySelector("#send-chat-message").addEventListener("click", sendChatMessage);
document.querySelector("#refresh-chat-context").addEventListener("click", loadChatContext);
document.querySelector("#refresh-chat-graph").addEventListener("click", refreshChatGraph);
document.querySelector("#compact-chat-context").addEventListener("click", compactChatContext);
document.querySelector("#restore-chat-context").addEventListener("click", restoreChatContext);
document.querySelector("#run-context-v42-benchmark").addEventListener("click", runContextBenchmarkV42);
document.querySelector("#search-chat-memory").addEventListener("click", searchChatMemory);
document.querySelector("#export-chat-memory").addEventListener("click", exportChatMemory);
document.querySelector("#delete-chat-memory").addEventListener("click", deleteChatMemory);
document.querySelector("#chat-memory-enabled").addEventListener("change", updateMemorySetting);
document.querySelector("#run-memory-v43-benchmark").addEventListener("click", runMemoryBenchmarkV43);
document.querySelector("#run-runtime-v44-benchmark").addEventListener("click", runRuntimeBenchmarkV44);
document.querySelector("#load-runtime-v44-benchmark").addEventListener("click", loadRuntimeBenchmarkV44);
document.querySelector("#run-production-v45-benchmark").addEventListener("click", runProductionBenchmarkV45);
document.querySelector("#load-production-v45-benchmark").addEventListener("click", loadProductionBenchmarkV45);
document.querySelector("#load-production-v45-status").addEventListener("click", loadProductionStatusV45);
document.querySelector("#load-production-v45-trace").addEventListener("click", loadProductionTraceV45);
document.querySelector("#run-retrieval-runtime").addEventListener("click", runRetrievalRuntimeBenchmark);
document.querySelector("#load-retrieval-runtime").addEventListener("click", loadRetrievalRuntimeBenchmark);
document.querySelector("#create-enterprise-kb").addEventListener("click", createEnterpriseKB);
document.querySelector("#create-zotero-kb").addEventListener("click", createZoteroKB);
document.querySelector("#list-enterprise-kb").addEventListener("click", listEnterpriseKB);
document.querySelector("#ask-enterprise-kb").addEventListener("click", askEnterpriseKB);
document.querySelector("#run-rag-eval-v4").addEventListener("click", runRAGEvaluationV4);
document.querySelector("#load-rag-eval-v4").addEventListener("click", loadRAGEvaluationV4);
document.querySelector("#run-rag-eval-v41").addEventListener("click", runRAGEvaluationV41);
document.querySelector("#load-rag-eval-v41").addEventListener("click", loadRAGEvaluationV41);

loadDashboard();
loadDeepLink();
