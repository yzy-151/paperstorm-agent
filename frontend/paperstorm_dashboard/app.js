const state = {
  mode: "research",
  catalog: null,
  selectedBenchmark: null,
  benchmarkRun: null,
  benchmarkPoll: null,
  researchTaskId: "",
  chatId: "",
  chatSessions: [],
  chatController: null,
  lastArticle: "",
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

function serviceBase() {
  const configured = $("#service-url")?.value.trim();
  return (configured || window.location.origin).replace(/\/$/, "");
}

async function fetchJson(path, options = {}) {
  const response = await fetch(`${serviceBase()}${path}`, {
    headers: {"Content-Type": "application/json", ...(options.headers || {})},
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || payload.message || `HTTP ${response.status}`);
  }
  return payload;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function toast(message, tone = "info") {
  const node = $("#toast");
  node.textContent = message;
  node.className = `toast ${tone === "error" ? "error" : ""}`;
  window.clearTimeout(node._hideTimer);
  node._hideTimer = window.setTimeout(() => node.classList.add("hidden"), 3600);
}

function setServiceState(status, detail) {
  const node = $("#service-state");
  node.className = `service-pill ${status}`;
  node.querySelector("span").textContent = detail;
}

function setMode(mode) {
  state.mode = mode;
  const developer = mode === "developer";
  $("#developer-view").classList.toggle("hidden", !developer);
  $$(".product-only").forEach((node) => node.classList.toggle("hidden", developer));
  if (!developer) {
    $("#research-view").classList.toggle("hidden", mode !== "research");
    $("#chat-view").classList.toggle("hidden", mode !== "chat");
    $("#show-research-mode").classList.toggle("active", mode === "research");
    $("#show-chat-mode").classList.toggle("active", mode === "chat");
  }
  document.body.dataset.mode = mode;
  const labels = {research: "论文调研", chat: "智能问答", developer: "开发者控制台"};
  if ($("#workspace-title")) $("#workspace-title").textContent = labels[mode] || "工作空间";
  $$(".rail-item").forEach((node) => node.classList.remove("active"));
  const activeNav = mode === "research" ? $("#show-research-mode") : mode === "chat" ? $("#show-chat-mode") : $("#show-developer-mode");
  activeNav?.classList.add("active");
  if (developer) loadBenchmarkCatalog();
  if (mode === "chat") loadChatSessions();
}

function researchPayload(demo = false) {
  return {
    topic: demo ? "无源互调的神经网络抑制方法" : $("#task-topic").value.trim(),
    retriever: $("#task-retriever").value,
    output_language: $("#task-output-language").value,
    run_mode: $("#task-run-mode").value,
    expected_keywords: demo ? [] : [$("#task-expected-keyword").value.trim()].filter(Boolean),
    forbidden_keywords: demo ? [] : [$("#task-forbidden-keyword").value.trim()].filter(Boolean),
  };
}

function renderResearchProgress(stage, failed = false) {
  const order = ["created", "retrieval", "outline", "writing", "completed"];
  const current = order.indexOf(stage);
  $$("#research-progress > div").forEach((node, index) => {
    node.classList.remove("complete", "active", "failed");
    if (failed && index === Math.max(0, current)) node.classList.add("failed");
    else if (index < current || stage === "completed") node.classList.add("complete");
    else if (index === current) node.classList.add("active");
  });
}

async function runResearchWorkflow(demo = false) {
  const payload = researchPayload(demo);
  if (!payload.topic) return toast("请输入调研主题", "error");
  const button = demo ? $("#start-research-demo") : $("#start-research-workflow");
  button.disabled = true;
  button.textContent = "正在调研";
  $("#research-current-activity").textContent = "正在创建调研任务...";
  renderResearchProgress("created");
  try {
    const task = await fetchJson("/research-tasks", {method: "POST", body: JSON.stringify(payload)});
    state.researchTaskId = task.task_id;
    $("#research-task-id").textContent = task.task_id;
    renderResearchProgress("retrieval");
    $("#research-current-activity").textContent = "Agent 正在检索证据并组织文章...";
    const completed = await fetchJson(`/research-tasks/${encodeURIComponent(task.task_id)}/run`, {method: "POST"});
    if (completed.status !== "succeeded") throw new Error(completed.error || `任务状态：${completed.status}`);
    renderResearchProgress("completed");
    $("#research-current-activity").textContent = "调研完成，文章、评分与引用已更新。";
    await loadResearchResult(task.task_id);
    toast("调研任务已完成");
  } catch (error) {
    renderResearchProgress("retrieval", true);
    $("#research-current-activity").textContent = error.message;
    toast(error.message, "error");
  } finally {
    button.disabled = false;
    button.textContent = demo ? "运行示例主题" : "开始调研";
  }
}

async function loadResearchResult(taskId) {
  const [article, scorecard] = await Promise.all([
    fetchJson(`/research-tasks/${encodeURIComponent(taskId)}/article`),
    fetchJson(`/research-tasks/${encodeURIComponent(taskId)}/scorecard`),
  ]);
  $("#article-content").textContent = article.content || "任务完成，但没有生成文章。";
  state.lastArticle = article.content || "";
  $("#download-article-md").disabled = !state.lastArticle;
  renderMetricGrid($("#scorecard"), scorecard, 8);
  $("#research-score-section").classList.toggle("hidden", !Object.keys(scorecard).length);
}

async function loadChatSessions() {
  try {
    state.chatSessions = await fetchJson("/chat/sessions?limit=50");
    renderChatSessions();
  } catch (error) {
    toast(error.message, "error");
  }
}

function renderChatSessions() {
  const container = $("#chat-session-list");
  if (!state.chatSessions.length) {
    container.innerHTML = '<p class="empty-state">暂无历史会话。</p>';
    return;
  }
  container.innerHTML = state.chatSessions.map((session) => `
    <button class="session-item ${state.chatId === session.chat_id ? "active" : ""}" data-chat-id="${escapeHtml(session.chat_id)}" type="button">
      <strong>${escapeHtml(session.title || "PaperStorm Chat")}</strong>
      <span>${session.message_count} 条消息 · ${escapeHtml(session.run_mode)} / ${escapeHtml(session.retriever)}</span>
      <small>${escapeHtml(session.last_preview || "无消息")}</small>
    </button>
  `).join("");
  $$(".session-item").forEach((item) => {
    item.addEventListener("click", () => loadChatSession(item.dataset.chatId));
  });
}

async function loadChatSession(chatId) {
  try {
    const session = await fetchJson(`/chat/sessions/${encodeURIComponent(chatId)}`);
    state.chatId = session.chat_id;
    $("#chat-session-id").textContent = session.chat_id;
    $("#chat-context-summary").textContent = "等待消息";
    renderChatMessages(session.messages || []);
    renderChatSessions();
  } catch (error) {
    toast(error.message, "error");
  }
}

async function createChat() {
  const button = $("#create-chat");
  button.disabled = true;
  try {
    const session = await fetchJson("/chat/sessions", {
      method: "POST",
      body: JSON.stringify({
        title: "PaperStorm chat",
        topic: $("#task-topic").value.trim(),
        run_mode: $("#chat-run-mode").value,
        retriever: $("#chat-retriever").value,
        output_language: "zh",
        memory_enabled: true,
      }),
    });
    state.chatId = session.chat_id;
    $("#chat-session-id").textContent = session.chat_id;
    $("#chat-messages").innerHTML = "";
    renderChatMessages(session.messages || []);
    loadChatSessions();
    toast("新会话已创建");
  } catch (error) {
    toast(error.message, "error");
  } finally {
    button.disabled = false;
  }
}

async function sendChat(event) {
  event.preventDefault();
  const message = $("#chat-input").value.trim();
  if (!message) return;
  if (!state.chatId) await createChat();
  if (!state.chatId) return;
  $("#chat-input").value = "";
  appendMessage("user", message);
  const button = $("#send-chat");
  button.disabled = true;
  button.textContent = "思考中";
  $("#stop-chat").classList.remove("hidden");
  $("#regenerate-chat").disabled = true;
  state.chatController = new AbortController();
  try {
    const result = await fetchJson(`/chat/sessions/${encodeURIComponent(state.chatId)}/messages`, {
      method: "POST",
      body: JSON.stringify({message}),
      signal: state.chatController.signal,
    });
    renderChatMessages(result.messages || []);
    const context = result.context || result.context_meter || {};
    $("#chat-context-summary").textContent = context.input_tokens
      ? `${context.input_tokens} tokens · ${context.compacted ? "已压缩" : "未压缩"}`
      : `${(result.messages || []).length} 条消息`;
    $("#regenerate-chat").disabled = !((result.messages || []).at(-1)?.role === "assistant");
    loadChatSessions();
  } catch (error) {
    if (error.name === "AbortError") {
      appendMessage("assistant", "已停止生成。");
    } else {
      appendMessage("assistant", `请求失败：${error.message}`);
    }
  } finally {
    state.chatController = null;
    $("#stop-chat").classList.add("hidden");
    button.disabled = false;
    button.textContent = "发送";
  }
}

async function stopChat() {
  if (state.chatController) state.chatController.abort();
  if (state.chatId) {
    try {
      await fetchJson(`/chat/sessions/${encodeURIComponent(state.chatId)}/stop`, {method: "POST"});
    } catch (_error) {
      // stop is best-effort; the UI already aborted the request
    }
  }
  $("#stop-chat").classList.add("hidden");
  toast("已停止生成");
}

async function regenerateChat() {
  if (!state.chatId) return;
  const button = $("#regenerate-chat");
  button.disabled = true;
  try {
    const result = await fetchJson(`/chat/sessions/${encodeURIComponent(state.chatId)}/regenerate`, {method: "POST"});
    renderChatMessages(result.messages || []);
    toast(result.regenerated ? "已生成新版本回答" : "重新生成完成");
    loadChatSessions();
  } catch (error) {
    toast(error.message, "error");
  } finally {
    const messages = $("#chat-messages").querySelectorAll(".message");
    button.disabled = !(messages.length && messages[messages.length - 1].classList.contains("assistant"));
  }
}

function renderMessageNode(message) {
  const node = document.createElement("div");
  const role = message.role === "user" ? "user" : "assistant";
  node.className = `message ${role}`;
  const metadata = message.metadata || {};
  const version = metadata.version ? ` <span class="version-badge">v${escapeHtml(metadata.version)}</span>` : "";
  const regenerated = metadata.regenerated ? " <span class=\"version-badge\">重新生成</span>" : "";
  let citationsHtml = "";
  const citations = Array.isArray(metadata.citations) ? metadata.citations : [];
  if (citations.length) {
    const items = citations.map((citation) => {
      const title = escapeHtml(citation.title || citation.name || "未命名来源");
      const url = citation.url || "";
      const page = citation.page ? ` · 第 ${escapeHtml(citation.page)} 页` : "";
      const chunk = citation.chunk ? ` · ${escapeHtml(citation.chunk)}` : "";
      const source = url
        ? `<a href="${escapeHtml(url)}" target="_blank" rel="noopener">来源</a>`
        : '<b class="missing-badge">失效</b>';
      return `<li>${title}${source}${page}${chunk}</li>`;
    }).join("");
    citationsHtml = `<details class="citations"><summary>引用 ${citations.length} 条</summary><ul>${items}</ul></details>`;
  }
  node.innerHTML = `<strong>${role === "user" ? "你" : "PaperStorm"}${version}${regenerated}</strong><p>${escapeHtml(message.content)}</p>${citationsHtml}`;
  return node;
}

function renderChatMessages(messages) {
  const container = $("#chat-messages");
  container.innerHTML = "";
  if (!messages.length) {
    appendMessage("assistant", "会话已创建。你可以直接聊天，也可以提出需要论文证据的问题。");
    return;
  }
  messages.forEach((message) => {
    container.appendChild(renderMessageNode(message));
  });
  container.scrollTop = container.scrollHeight;
}

function appendMessage(role, content) {
  const container = $("#chat-messages");
  container.appendChild(renderMessageNode({role, content, metadata: {}}));
  container.scrollTop = container.scrollHeight;
}

function downloadArticleMarkdown() {
  if (!state.lastArticle) return;
  const blob = new Blob([state.lastArticle], {type: "text/markdown;charset=utf-8"});
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `paperstorm-${state.researchTaskId || "research"}.md`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
  toast("Markdown 已下载");
}

async function loadBenchmarkCatalog() {
  setServiceState("", "连接中");
  try {
    const [catalog, observability] = await Promise.all([
      fetchJson("/benchmarks/catalog"),
      fetchJson("/observability/status"),
    ]);
    state.catalog = catalog;
    setServiceState("online", "服务在线");
    renderBenchmarkReadiness(state.catalog);
    renderObservabilityStatus(observability);
    renderBenchmarkCatalog(state.catalog.benchmarks || []);
    if (state.selectedBenchmark) {
      const updated = state.catalog.benchmarks.find((item) => item.id === state.selectedBenchmark.id);
      if (updated) selectBenchmark(updated.id);
    }
  } catch (error) {
    setServiceState("offline", "服务不可用");
    $("#ready-service").textContent = "连接失败";
    $("#ready-service-detail").textContent = error.message;
    $("#benchmark-catalog").innerHTML = `<p class="empty-state">${escapeHtml(error.message)}</p>`;
  }
}

function renderObservabilityStatus(status) {
  const labels = {
    configured: "已配置",
    degraded: "降级",
    unavailable: "SDK 缺失",
    "local-only": "本地模式",
  };
  $("#ready-langfuse").textContent = labels[status.status] || status.status || "未知";
  $("#ready-langfuse-detail").textContent = status.remote_enabled
    ? `${status.environment} · failures ${status.export_failures || 0}`
    : "设置 PAPERSTORM_OBSERVABILITY=langfuse";
}

function renderBenchmarkReadiness(catalog) {
  const benchmarks = catalog.benchmarks || [];
  const ready = benchmarks.filter((item) => item.ready).length;
  $("#ready-service").textContent = "API 正常";
  $("#ready-service-detail").textContent = `${benchmarks.length} 个 Registry 条目`;
  $("#ready-root").textContent = catalog.benchmark_root ? "已发现" : "未发现";
  $("#ready-root-detail").textContent = catalog.benchmark_root || "设置 PAPERSTORM_BENCHMARK_ROOT";
  $("#ready-datasets").textContent = `${ready} / ${benchmarks.length}`;
  $("#ready-python").textContent = (catalog.python || "未知").split(/[\\/]/).pop();
  $("#ready-model-cache").textContent = catalog.model_cache || "模型缓存未配置";
}

function renderBenchmarkCatalog(benchmarks) {
  $("#benchmark-catalog").innerHTML = benchmarks.map((item) => `
    <article class="benchmark-card ${item.ready ? "" : "blocked"} ${state.selectedBenchmark?.id === item.id ? "selected" : ""}" data-benchmark-id="${escapeHtml(item.id)}" data-kind="${benchmarkKind(item)}" tabindex="0">
      <div class="card-meta"><span>${escapeHtml(item.version)} · ${escapeHtml(item.category)}</span><span class="readiness-badge ${item.ready ? "" : "blocked"}">${item.ready ? "READY" : "BLOCKED"}</span></div>
      <h3>${escapeHtml(item.name)}</h3>
      <p>${escapeHtml(item.description)}</p>
      <footer><span>${escapeHtml(item.evidence_tier)}</span><span>${item.latest_result_path ? "已有正式结果" : "暂无结果"}</span></footer>
    </article>
  `).join("");
  $$(".benchmark-card").forEach((card) => {
    card.addEventListener("click", () => selectBenchmark(card.dataset.benchmarkId));
    card.addEventListener("keydown", (event) => { if (event.key === "Enter") selectBenchmark(card.dataset.benchmarkId); });
  });
}

function benchmarkKind(item) {
  const value = `${item.id} ${item.category}`.toLowerCase();
  if (value.includes("memory") || value.includes("longmem")) return "MEM";
  if (value.includes("context") || value.includes("longbench")) return "CTX";
  if (value.includes("answer")) return "F1";
  return "RAG";
}

function selectBenchmark(benchmarkId) {
  const item = state.catalog?.benchmarks?.find((entry) => entry.id === benchmarkId);
  if (!item) return;
  state.selectedBenchmark = item;
  renderBenchmarkCatalog(state.catalog.benchmarks);
  $("#benchmark-selected-name").textContent = item.name;
  $("#benchmark-selected-description").textContent = `${item.description} ${item.estimated_time}`;
  $("#benchmark-evidence-tier").textContent = item.evidence_tier;
  $("#paid-confirm-row").classList.toggle("hidden", !item.requires_llm);
  $("#benchmark-input-manifest").innerHTML = item.inputs.map((input) => `
    <div class="input-row"><span>${escapeHtml(input.key)}</span><code title="${escapeHtml(input.path)}">${escapeHtml(input.path || "未发现")}</code><b class="${input.available ? "" : "missing"}">${input.available ? "READY" : "MISSING"}</b></div>
  `).join("");
  $("#benchmark-command-preview").textContent = buildBenchmarkPreview(item, $("#benchmark-profile").value);
  $("#start-benchmark-run").disabled = !item.ready;
  $("#benchmark-limitations").innerHTML = item.blocker
    ? `<li>${escapeHtml(item.blocker)}</li>`
    : `<li>${escapeHtml(item.estimated_time)}</li><li>延迟是本机参考值，不等同于线上 SLA。</li><li>${item.requires_llm ? "该实验会产生真实 LLM API 成本。" : "该实验默认不调用真实 LLM。"}</li>`;
  renderBenchmarkResult(item.latest_result || {}, item.latest_result_path || "");
}

function buildBenchmarkPreview(item, profile) {
  const paths = Object.fromEntries(item.inputs.map((input) => [input.key, input.path || `<${input.key}>`]));
  const output = "<service-root>/benchmark_runs/<run-id>/artifacts";
  const commands = {
    "scifact-retrieval-v55": `python examples/storm_examples/run_paperstorm_public_benchmark.py --benchmark scifact --dataset-dir "${paths.scifact_dir}" --output-dir "${output}" --embedding ${profile === "smoke" ? "hash --smoke-limit 20" : "real --reranker"}`,
    "qasper-retrieval-v55": `python examples/storm_examples/run_paperstorm_public_benchmark.py --benchmark qasper --dataset-dir "${paths.qasper_json}" --output-dir "${output}" --embedding ${profile === "smoke" ? "hash --smoke-limit 20" : "real --reranker"}`,
    "qasper-answer-v55": `python examples/storm_examples/run_qasper_answer_benchmark.py --split test --retrieval-predictions "${paths.qasper_rankings}" --output-dir "${output}"${profile === "smoke" ? " --smoke-limit 10" : ""}`,
    "longmemeval-retrieval-v56": `python examples/storm_examples/run_longmemeval_benchmark.py --dataset "${paths.longmemeval_json}" --output-dir "${output}" --embedding ${profile === "smoke" ? "hash --limit 10" : "sentence-transformer"}`,
    "qasper-context-v56": `python examples/storm_examples/run_qasper_context_benchmark.py --dataset "${paths.qasper_json}" --rankings "${paths.qasper_rankings}" --output-dir "${output}"`,
    "longbench-context-v56": "Blocked: 先生成同模型 full/fixed/v5.6 配对预测。",
  };
  return commands[item.id] || "该任务不可运行";
}

async function startBenchmarkRun() {
  const item = state.selectedBenchmark;
  if (!item) return toast("请先选择 Benchmark", "error");
  const allowPaid = $("#benchmark-allow-paid").checked;
  if (item.requires_llm && !allowPaid) return toast("请先确认付费 LLM 成本", "error");
  const button = $("#start-benchmark-run");
  button.disabled = true;
  try {
    const run = await fetchJson("/benchmarks/runs", {
      method: "POST",
      body: JSON.stringify({benchmark_id: item.id, profile: $("#benchmark-profile").value, allow_paid_llm: allowPaid}),
    });
    state.benchmarkRun = run;
    renderBenchmarkRun(run);
    scheduleBenchmarkPoll();
  } catch (error) {
    button.disabled = !item.ready;
    toast(error.message, "error");
  }
}

function scheduleBenchmarkPoll() {
  window.clearTimeout(state.benchmarkPoll);
  state.benchmarkPoll = window.setTimeout(pollBenchmarkRun, 900);
}

async function pollBenchmarkRun() {
  if (!state.benchmarkRun) return;
  try {
    const run = await fetchJson(`/benchmarks/runs/${encodeURIComponent(state.benchmarkRun.run_id)}`);
    state.benchmarkRun = run;
    renderBenchmarkRun(run);
    if (run.status === "running") scheduleBenchmarkPoll();
  } catch (error) {
    toast(error.message, "error");
  }
}

function renderBenchmarkRun(run) {
  const status = run.status || "idle";
  $("#benchmark-run-status").className = `run-status ${status}`;
  $("#benchmark-run-status").textContent = status;
  const progress = $("#benchmark-progress-dot").parentElement;
  progress.className = `run-progress ${status}`;
  $("#benchmark-progress-title").textContent = status === "running" ? "Benchmark 正在运行" : `运行${status === "succeeded" ? "完成" : "结束"}`;
  $("#benchmark-progress-detail").textContent = `run ${run.run_id} · PID ${run.pid || "-"} · ${run.output_dir || ""}`;
  $("#benchmark-log-tail").textContent = run.log_tail || "进程已启动，等待日志...";
  $("#benchmark-command-preview").textContent = run.command_preview || $("#benchmark-command-preview").textContent;
  $("#cancel-benchmark-run").disabled = status !== "running";
  $("#start-benchmark-run").disabled = status === "running" || !state.selectedBenchmark?.ready;
  if (run.result && Object.keys(run.result).length) renderBenchmarkResult(run.result, run.result_path);
}

async function cancelBenchmarkRun() {
  if (!state.benchmarkRun) return;
  try {
    const run = await fetchJson(`/benchmarks/runs/${encodeURIComponent(state.benchmarkRun.run_id)}/cancel`, {method: "POST"});
    state.benchmarkRun = run;
    renderBenchmarkRun(run);
    toast("Benchmark 已停止");
  } catch (error) {
    toast(error.message, "error");
  }
}

function renderBenchmarkResult(result, resultPath = "") {
  renderBenchmarkMetrics($("#benchmark-result-metrics"), result);
  $("#benchmark-raw-result").textContent = JSON.stringify(result || {}, null, 2);
  $("#benchmark-artifacts").innerHTML = resultPath
    ? `<code>${escapeHtml(resultPath)}</code>`
    : "尚无运行产物。";
}

function renderBenchmarkMetrics(container, result) {
  const id = state.selectedBenchmark?.id || "";
  let metrics = [];
  if (id.endsWith("retrieval-v55")) {
    const suffix = id.startsWith("scifact") ? "10" : "5";
    const labels = {bm25: "BM25", dense: "Dense", hybrid: "Hybrid", hybrid_rerank: "Hybrid + Rerank"};
    Object.entries(result.modes || {}).forEach(([mode, values]) => {
      metrics.push([`${labels[mode] || mode} Recall@${suffix}`, values[`recall_at_${suffix}`]]);
      metrics.push([`${labels[mode] || mode} nDCG@${suffix}`, values[`ndcg_at_${suffix}`]]);
      metrics.push([`${labels[mode] || mode} P95`, values.p95_latency_ms, "ms"]);
    });
  } else if (id === "qasper-answer-v55") {
    const values = result.metrics || result;
    metrics = [
      ["Answer F1", values.answer_f1], ["Evidence F1", values.evidence_f1],
      ["Exact Match", values.answer_exact_match], ["样本数", values.case_count || result.case_count],
      ["成功预测", result.successful_predictions], ["失败预测", result.failed_predictions],
    ];
  } else if (id === "longmemeval-retrieval-v56") {
    const recent = result.modes?.recent_window || {};
    const memory = result.modes?.v56_memory || {};
    metrics = [
      ["样本数", result.case_count], ["Top K", result.top_k],
      ["Recent Recall@5", recent.retrieval_recall_at_5],
      ["v5.6 Memory Recall@5", memory.retrieval_recall_at_5],
      ["v5.6 P50", memory.p50_latency_ms, "ms"], ["v5.6 P95", memory.p95_latency_ms, "ms"],
    ];
  } else if (id === "qasper-context-v56") {
    metrics = [
      ["样本数", result.case_count], ["输入上限", result.input_limit_tokens, "tokens"],
      ["证据保留率", result.retrieved_evidence_retention],
      ["压缩前 Gold Recall", result.gold_evidence_recall_before_context],
      ["压缩后 Gold Recall", result.gold_evidence_recall_after_context],
      ["Context / 全文", result.mean_context_to_full_document_ratio],
      ["超预算率", result.over_budget_rate], ["结构校验通过率", result.validation_pass_rate],
    ];
  }
  metrics = metrics.filter(([, value]) => typeof value === "number" && Number.isFinite(value));
  if (!metrics.length) return renderMetricGrid(container, result, 12);
  container.innerHTML = metrics.slice(0, 12).map(([label, value, unit = ""]) => `
    <div class="metric"><small>${escapeHtml(label)}</small><strong>${escapeHtml(formatMetric(value))}${unit ? `<em>${escapeHtml(unit)}</em>` : ""}</strong></div>
  `).join("");
}

function renderMetricGrid(container, value, limit = 10) {
  const metrics = [];
  collectMetrics(value, "", metrics);
  container.innerHTML = metrics.slice(0, limit).map(([key, number]) => `
    <div class="metric"><small>${escapeHtml(key)}</small><strong>${escapeHtml(formatMetric(number))}</strong></div>
  `).join("") || '<p class="empty-state">暂无指标。</p>';
}

function collectMetrics(value, prefix, output) {
  if (typeof value === "number" && Number.isFinite(value)) {
    output.push([prefix || "value", value]);
    return;
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) return;
  Object.entries(value).forEach(([key, child]) => {
    if (["predictions", "bad_cases", "rows"].includes(key)) return;
    collectMetrics(child, prefix ? `${prefix}.${key}` : key, output);
  });
}

function formatMetric(value) {
  if (Number.isInteger(value)) return String(value);
  return Math.abs(value) >= 100 ? value.toFixed(1) : value.toFixed(4);
}

async function refreshRuntimeDiagnostics() {
  try {
    const [production, tasks, observability] = await Promise.all([
      fetchJson("/production/status"),
      fetchJson("/research-tasks"),
      fetchJson("/observability/status"),
    ]);
    $("#runtime-production-status").textContent = JSON.stringify(
      {production, observability}, null, 2
    );
    const latest = (tasks.tasks || []).at(-1) || {};
    $("#runtime-task-status").textContent = JSON.stringify(latest, null, 2);
    if (latest.task_id) {
      const trace = await fetchJson(`/research-tasks/${encodeURIComponent(latest.task_id)}/trace`);
      $("#runtime-trace-status").textContent = JSON.stringify(trace, null, 2);
    }
  } catch (error) {
    toast(error.message, "error");
  }
}

async function copyBenchmarkCommand() {
  try {
    await navigator.clipboard.writeText($("#benchmark-command-preview").textContent);
    toast("命令已复制");
  } catch (_error) {
    toast("浏览器未授予剪贴板权限", "error");
  }
}

$("#show-developer-mode").addEventListener("click", () => setMode("developer"));
$("#leave-developer-mode").addEventListener("click", () => setMode("research"));
$("#show-research-mode").addEventListener("click", () => setMode("research"));
$("#show-chat-mode").addEventListener("click", () => setMode("chat"));
$("#start-research-demo").addEventListener("click", () => runResearchWorkflow(true));
$("#start-research-workflow").addEventListener("click", () => runResearchWorkflow(false));
$("#create-chat").addEventListener("click", createChat);
$("#chat-form").addEventListener("submit", sendChat);
$("#refresh-chat-sessions").addEventListener("click", loadChatSessions);
$("#regenerate-chat").addEventListener("click", regenerateChat);
$("#stop-chat").addEventListener("click", stopChat);
$("#download-article-md").addEventListener("click", downloadArticleMarkdown);
$("#refresh-benchmark-catalog").addEventListener("click", loadBenchmarkCatalog);
$("#benchmark-profile").addEventListener("change", () => {
  if (state.selectedBenchmark) selectBenchmark(state.selectedBenchmark.id);
});
$("#start-benchmark-run").addEventListener("click", startBenchmarkRun);
$("#cancel-benchmark-run").addEventListener("click", cancelBenchmarkRun);
$("#copy-benchmark-command").addEventListener("click", copyBenchmarkCommand);
$("#refresh-runtime-diagnostics").addEventListener("click", refreshRuntimeDiagnostics);

renderResearchProgress("created");
loadBenchmarkCatalog();
