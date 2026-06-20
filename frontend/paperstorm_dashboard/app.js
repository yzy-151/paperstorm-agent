let sseSource = null;
const DASHBOARD_VERSION = "v3.2";

async function loadDashboard() {
  try {
    setDashboardMode("research");
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
  const isChat = mode === "chat";
  document.body.dataset.mode = isChat ? "chat" : "research";
  document.querySelector("#show-research-mode").classList.toggle("active", !isChat);
  document.querySelector("#show-chat-mode").classList.toggle("active", isChat);
  setStatus(isChat ? "chat mode ready" : "research workflow ready", "idle");
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
  const payload = {
    title: document.querySelector("#task-topic").value.trim() || "PaperStorm Chat",
    topic: document.querySelector("#task-topic").value.trim(),
    run_mode: document.querySelector("#task-run-mode").value,
    retriever: document.querySelector("#task-retriever").value,
    output_language: document.querySelector("#task-output-language").value,
    expected_keywords: splitKeywords(document.querySelector("#task-expected-keyword").value),
    forbidden_keywords: splitKeywords(document.querySelector("#task-forbidden-keyword").value),
    context_window_size: 6,
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

function renderResearchQA(answer) {
  document.querySelector("#research-answer").innerHTML = `
    <strong>Agent</strong>
    <p>${escapeHtml(answer.answer || "暂无回答。")}</p>
    <div class="label">grounded: ${Boolean(answer.grounded)} · task_id: ${escapeHtml(answer.used_task_id || "")}</div>
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
  document.querySelector("#chat-router-decision").textContent =
    JSON.stringify(session.router_decision || {}, null, 2);
  document.querySelector("#chat-tool-decision").textContent =
    JSON.stringify(session.tool_decision || {}, null, 2);
  document.querySelector("#chat-rewritten-query").textContent =
    (session.router_decision || {}).rewritten_query || "";
  if (session.chat_id) {
    document.querySelector("#chat-session-id").value = session.chat_id;
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

function appendChatMessage(message) {
  const list = document.querySelector("#chat-message-list");
  list.insertAdjacentHTML("beforeend", chatBubble(message));
  list.scrollTop = list.scrollHeight;
}

function chatBubble(message) {
  const role = message.role === "user" ? "user" : "assistant";
  const label = role === "user" ? "你" : "PaperStorm";
  const metadata = message.metadata || {};
  const meta = metadata.used_task_id
    ? `<div class="label">task: ${escapeHtml(metadata.used_task_id)} · retrieval: ${Boolean(metadata.retrieval_triggered)}</div>`
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
document.querySelector("#create-chat-session").addEventListener("click", createChatSession);
document.querySelector("#send-chat-message").addEventListener("click", sendChatMessage);
document.querySelector("#create-enterprise-kb").addEventListener("click", createEnterpriseKB);
document.querySelector("#list-enterprise-kb").addEventListener("click", listEnterpriseKB);
document.querySelector("#ask-enterprise-kb").addEventListener("click", askEnterpriseKB);

loadDashboard();
