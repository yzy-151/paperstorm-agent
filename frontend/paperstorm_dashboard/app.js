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
  lastPdfUrl: "",
  researchEvents: null,
  pipelineStatus: {},
  hasStageTrace: false,
  activeInvocations: {},
  artifactStatus: {},
  runFinished: false,
};

const pipelineNodes = {
  request: {title: "任务编排", description: "规范化主题、运行模式、检索源和领域约束。", input: "用户主题、语言、检索器", output: "Research Task + 运行配置"},
  persona: {title: "Persona Generator", description: "从不同知识视角生成调研角色，扩大问题覆盖面。", input: "主题与背景资料", output: "多视角研究角色"},
  dialogue: {title: "Conv Simulator", description: "WikiWriter 与 Topic Expert 迭代提问、检索和反思。", input: "角色、对话状态", output: "Conversation Log + 信息表"},
  query: {title: "查询规划", description: "生成独立检索式，并进行领域消歧和空查询过滤。", input: "问题、角色、历史", output: "规范化查询集合"},
  retrieval: {title: "论文检索", description: "从 arXiv、本地 PDF 或 Zotero 获取候选论文与段落。", input: "查询集合", output: "原始检索结果"},
  evidence: {title: "证据治理", description: "对文献 Chunk 执行 BM25 + Dense 融合、RRF、Rerank 与引用映射。", input: "论文与段落", output: "可追溯 Evidence"},
  outline: {title: "大纲生成", description: "先生成草案，再依据调研信息细化章节结构。", input: "信息表、证据", output: "Refined Outline"},
  writer: {title: "章节写作", description: "各章节独立检索相关证据并并发生成带引用内容。", input: "大纲、Evidence Index", output: "Draft Article"},
  polish: {title: "文章润色", description: "去重、统一结构和表达，同时保留引用标记。", input: "Draft Article", output: "Polished Article"},
  evaluate: {title: "质量评估", description: "检查领域一致性、证据覆盖、引用与运行完整性。", input: "文章、Trace、Evidence", output: "Scorecard"},
  deliver: {title: "交付产物", description: "汇总 Markdown、Trace、检索结果和评估报告。", input: "全部流水线产物", output: "Article + Trace + Score"},
};

const pipelineExecutionEdges = [
  ["request", "persona"], ["persona", "dialogue"], ["dialogue", "query"],
  ["query", "retrieval"], ["retrieval", "evidence"], ["evidence", "outline"],
  ["outline", "writer"], ["writer", "polish"], ["polish", "evaluate"],
  ["evaluate", "deliver"],
];

const pipelineArtifactEdges = [
  {id: "task-persona", from: "request", to: "persona", port: "research_task", label: "research_task.json"},
  {id: "task-query", from: "request", to: "query", port: "research_task", label: "research_task.json"},
  {id: "personas-dialogue", from: "persona", to: "dialogue", port: "personas", label: "personas.json"},
  {id: "conversation-outline", from: "dialogue", to: "outline", sourcePort: "conversation", targetPort: "conversation", label: "conversation_log.json"},
  {id: "queries-retrieval", from: "query", to: "retrieval", port: "queries", label: "queries.json"},
  {id: "results-evidence", from: "retrieval", to: "evidence", port: "raw_results", label: "raw_search_results.json"},
  {id: "evidence-outline", from: "evidence", to: "outline", port: "evidence", label: "evidence_index.json"},
  {id: "evidence-writer", from: "evidence", to: "writer", sourcePort: "evidence", targetPort: "evidence", label: "evidence_index.json"},
  {id: "outline-writer", from: "outline", to: "writer", port: "outline", label: "storm_gen_outline.txt"},
  {id: "draft-polish", from: "writer", to: "polish", port: "draft", label: "storm_gen_article.txt"},
  {id: "article-evaluate", from: "polish", to: "evaluate", port: "article", label: "article_polished.txt"},
  {id: "article-deliver", from: "polish", to: "deliver", sourcePort: "article", targetPort: "article", label: "article_polished.txt"},
  {id: "score-deliver", from: "evaluate", to: "deliver", port: "scorecard", label: "scorecard.json"},
];

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
    generate_pdf: $("#task-generate-pdf").checked,
    expected_keywords: demo ? [] : [$("#task-expected-keyword").value.trim()].filter(Boolean),
    forbidden_keywords: demo ? [] : [$("#task-forbidden-keyword").value.trim()].filter(Boolean),
  };
}

function renderResearchProgress(stage, failed = false) {
  if (stage !== "created") return;
  resetPipelineGraph();
  setPipelineNodeStatus(
    "request",
    failed ? "failed" : "active",
    failed ? "任务创建失败" : "正在提交并冻结运行配置",
  );
}

function resetPipelineGraph() {
  state.pipelineStatus = {};
  state.activeInvocations = {};
  state.artifactStatus = {};
  state.runFinished = false;
  state.hasStageTrace = false;
  state.lastPdfUrl = "";
  $("#open-article-pdf").disabled = true;
  $("#pipeline-open-pdf").classList.add("hidden");
  Object.keys(pipelineNodes).forEach((nodeId) => setPipelineNodeStatus(nodeId, "waiting", "尚未运行"));
}

function formatDuration(milliseconds) {
  if (!Number.isFinite(milliseconds) || milliseconds < 0) return "-";
  return milliseconds < 1000 ? `${Math.round(milliseconds)} ms` : `${(milliseconds / 1000).toFixed(2)} s`;
}

function formatPipelineTelemetry(trace = {}) {
  const details = trace.details && typeof trace.details === "object" ? trace.details : {};
  const usage = trace.usage || trace.token_usage || details.usage || {};
  const durationMs = Number(trace.duration_ms ?? trace.latency_ms ?? details.duration_ms ?? details.latency_ms);
  const promptTokens = Number(usage.prompt_tokens ?? usage.input_tokens ?? trace.prompt_tokens ?? details.prompt_tokens ?? 0);
  const completionTokens = Number(usage.completion_tokens ?? usage.output_tokens ?? trace.completion_tokens ?? details.completion_tokens ?? 0);
  const totalTokens = Number(usage.total_tokens ?? trace.total_tokens ?? details.total_tokens ?? (promptTokens + completionTokens));
  const costUsd = Number(trace.cost_usd ?? details.cost_usd);
  return {
    input: trace.input ?? details.input,
    activity: trace.operation ?? trace.activity ?? details.activity ?? trace.message,
    output: trace.output_summary ?? trace.output ?? details.output,
    durationMs: Number.isFinite(durationMs) ? durationMs : undefined,
    promptTokens,
    completionTokens,
    totalTokens,
    costUsd: Number.isFinite(Number(trace.estimated_cost)) ? Number(trace.estimated_cost) : Number.isFinite(costUsd) ? costUsd : undefined,
    finishReason: trace.finish_reason ?? details.finish_reason,
    errorType: trace.error_type ?? details.error_type,
    error: trace.error?.message ?? trace.error_message ?? details.error?.message ?? details.error,
  };
}

function setPipelineNodeStatus(nodeId, status, detail = "", telemetry = {}) {
  const previous = state.pipelineStatus[nodeId] || {};
  const now = performance.now();
  const startedAt = status === "active" && previous.status !== "active" ? now : previous.startedAt;
  const measuredDuration = status === "complete" || status === "failed"
    ? telemetry.durationMs ?? (startedAt ? now - startedAt : previous.durationMs)
    : telemetry.durationMs ?? previous.durationMs;
  state.pipelineStatus[nodeId] = {
    ...previous,
    ...telemetry,
    status,
    startedAt,
    durationMs: measuredDuration,
    detail: detail || telemetry.activity || previous.detail || "",
  };
  const node = $(`.pipeline-node[data-node="${nodeId}"]`);
  if (!node) return;
  node.classList.remove("waiting", "active", "complete", "failed", "skipped");
  node.classList.add(status);
  node.querySelector("em").textContent = {waiting: "WAIT", active: "RUN", complete: "DONE", failed: "ERR", skipped: "SKIP"}[status] || status;
  let time = node.querySelector(".node-time");
  if ((status === "complete" || status === "failed") && Number.isFinite(measuredDuration)) {
    if (!time) {
      time = document.createElement("time");
      time.className = "node-time";
      node.appendChild(time);
    }
    time.textContent = formatDuration(measuredDuration);
  } else {
    time?.remove();
  }
  updatePipelineWires();
  if (node.classList.contains("selected")) showPipelineNode(nodeId);
}

function initializePipelineGraph() {
  $$(".pipeline-node").forEach((node) => node.addEventListener("click", () => showPipelineNode(node.dataset.node)));
  drawPipelineWires();
  resetPipelineGraph();
  window.addEventListener("resize", drawPipelineWires);
}

function drawPipelineWires() {
  const canvas = $("#pipeline-canvas");
  const executionSvg = $("#pipeline-execution-wires");
  const artifactSvg = $("#pipeline-artifact-wires");
  if (!canvas || !executionSvg || !artifactSvg) return;
  const canvasRect = canvas.getBoundingClientRect();
  [executionSvg, artifactSvg].forEach((svg) => svg.setAttribute("viewBox", `0 0 ${canvasRect.width} ${canvasRect.height}`));
  executionSvg.innerHTML = pipelineExecutionEdges.map(([from, to], index) => {
    const source = $(`.pipeline-node[data-node="${from}"] .flow-out`);
    const target = $(`.pipeline-node[data-node="${to}"] .flow-in`);
    return source && target ? `<path class="execution-wire" data-from="${from}" data-to="${to}" d="${pipelineExecutionPath(source, target, canvasRect)}" />` : "";
  }).join("");
  artifactSvg.innerHTML = pipelineArtifactEdges.map((edge, index) => {
    const sourcePort = edge.sourcePort || edge.port;
    const targetPort = edge.targetPort || edge.port;
    const source = $(`.pipeline-node[data-node="${edge.from}"] .output-port[data-port="${sourcePort}"] i`);
    const target = $(`.pipeline-node[data-node="${edge.to}"] .input-port[data-port="${targetPort}"] i`);
    if (!source || !target) return "";
    const pathId = `artifact-path-${edge.id}`;
    const route = pipelineArtifactRoute(source, target, canvasRect, index);
    return `<path id="${pathId}" class="artifact-wire" data-edge-id="${edge.id}" data-from="${edge.from}" data-to="${edge.to}" d="${route.path}" /><text class="artifact-label" data-label-x="${route.labelX}" data-label-y="${route.labelY}">${escapeHtml(edge.label)}</text>`;
  }).join("");
  positionArtifactLabels();
  updatePipelineWires();
}

function pipelinePortPoints(sourceNode, targetNode, canvasRect) {
  const source = sourceNode?.getBoundingClientRect();
  const target = targetNode?.getBoundingClientRect();
  if (!source || !target) return null;
  return {
    x1: source.left + source.width / 2 - canvasRect.left,
    y1: source.top + source.height / 2 - canvasRect.top,
    x2: target.left + target.width / 2 - canvasRect.left,
    y2: target.top + target.height / 2 - canvasRect.top,
  };
}

function pipelineExecutionPath(source, target, canvasRect) {
  const points = pipelinePortPoints(source, target, canvasRect);
  if (!points) return "";
  if (Math.abs(points.y1 - points.y2) < 24 && points.x2 > points.x1) {
    const bend = Math.max(24, Math.min(48, (points.x2 - points.x1) * .42));
    return `M ${points.x1} ${points.y1} C ${points.x1 + bend} ${points.y1}, ${points.x2 - bend} ${points.y2}, ${points.x2} ${points.y2}`;
  }
  return pipelineRowWrapPath(points, canvasRect);
}

function pipelineRowWrapPath({x1, y1, x2, y2}, canvasRect) {
  const rightGutter = canvasRect.width - 14;
  const leftGutter = 14;
  const laneY = (y1 + y2) / 2;
  const radius = 12;
  return [
    `M ${x1} ${y1}`,
    `L ${rightGutter - radius} ${y1}`,
    `Q ${rightGutter} ${y1} ${rightGutter} ${y1 + radius}`,
    `L ${rightGutter} ${laneY - radius}`,
    `Q ${rightGutter} ${laneY} ${rightGutter - radius} ${laneY}`,
    `L ${leftGutter + radius} ${laneY}`,
    `Q ${leftGutter} ${laneY} ${leftGutter} ${laneY + radius}`,
    `L ${leftGutter} ${y2 - radius}`,
    `Q ${leftGutter} ${y2} ${leftGutter + radius} ${y2}`,
    `L ${x2} ${y2}`,
  ].join(" ");
}

function pipelineArtifactPath(source, target, canvasRect, offsetSeed = 0) {
  return pipelineArtifactRoute(source, target, canvasRect, offsetSeed).path;
}

function pipelineArtifactRoute(source, target, canvasRect, offsetSeed = 0) {
  const points = pipelinePortPoints(source, target, canvasRect);
  if (!points) return {path: "", labelX: 0, labelY: 0};
  const sourceNode = source.closest(".pipeline-node")?.getBoundingClientRect();
  const targetNode = target.closest(".pipeline-node")?.getBoundingClientRect();
  if (!sourceNode || !targetNode) return {path: "", labelX: 0, labelY: 0};
  const sameRow = Math.abs(sourceNode.top - targetNode.top) < 30;
  const adjacent = sameRow && points.x2 > points.x1 && points.x2 - points.x1 < 180;
  if (adjacent) {
    const bend = Math.max(20, (points.x2 - points.x1) * .44);
    return {
      path: `M ${points.x1} ${points.y1} C ${points.x1 + bend} ${points.y1}, ${points.x2 - bend} ${points.y2}, ${points.x2} ${points.y2}`,
      labelX: (points.x1 + points.x2) / 2,
      labelY: Math.min(points.y1, points.y2) - 7,
    };
  }
  const sourceBottom = sourceNode.bottom - canvasRect.top;
  const targetTop = targetNode.top - canvasRect.top;
  const laneOffset = artifactLaneOffset(offsetSeed);
  const laneY = sameRow
    ? Math.min(canvasRect.height - 12, sourceBottom + 16 + laneOffset)
    : Math.min(targetTop - 14, sourceBottom + 16 + laneOffset);
  const curve = 28;
  const exitX = points.x1 + curve * 1.7;
  const entryX = points.x2 - curve * 1.7;
  const reverse = entryX < exitX;
  const exitControlX = exitX + (reverse ? 18 : -18);
  return {
    path: `M ${points.x1} ${points.y1} C ${points.x1 + curve} ${points.y1}, ${exitControlX} ${laneY}, ${exitX} ${laneY} C ${exitX + (reverse ? -18 : 18)} ${laneY}, ${entryX - 18} ${laneY}, ${entryX} ${laneY} C ${entryX + 18} ${laneY}, ${points.x2 - curve} ${points.y2}, ${points.x2} ${points.y2}`,
    labelX: (points.x1 + points.x2) / 2,
    labelY: laneY - 7,
  };
}

function artifactLaneOffset(offsetSeed) {
  return Number(offsetSeed || 0) * 6;
}

function positionArtifactLabels() {
  $$("#pipeline-artifact-wires .artifact-label").forEach((label) => {
    label.setAttribute("x", label.dataset.labelX || "0");
    label.setAttribute("y", label.dataset.labelY || "0");
    label.setAttribute("text-anchor", "middle");
  });
}

function updatePipelineWires() {
  $$("#pipeline-execution-wires path").forEach((path) => {
    const source = state.pipelineStatus[path.dataset.from]?.status || "waiting";
    const target = state.pipelineStatus[path.dataset.to]?.status || "waiting";
    path.className.baseVal = `execution-wire ${target === "active" ? "active" : source === "complete" && target === "complete" ? "complete" : target === "failed" ? "failed" : "waiting"}`;
  });
  $$("#pipeline-artifact-wires path").forEach((path) => {
    const status = state.artifactStatus[path.dataset.edgeId] || "waiting";
    path.className.baseVal = `artifact-wire ${status}`;
  });
}

function markArtifactsReady(stage, trace = {}) {
  if (state.runFinished) return;
  const artifactName = String(trace.artifact_name || trace.path || "").toLowerCase();
  pipelineArtifactEdges.forEach((edge) => {
    if (edge.from !== stage) return;
    if (artifactName && !artifactName.includes(edge.label.toLowerCase().replace("storm_gen_", ""))) return;
    if (state.artifactStatus[edge.id] !== "complete") state.artifactStatus[edge.id] = "active";
  });
  updatePipelineWires();
}

function markArtifactsConsumed(stage) {
  pipelineArtifactEdges.forEach((edge) => {
    if (edge.to === stage && state.artifactStatus[edge.id] === "active") state.artifactStatus[edge.id] = "complete";
  });
  updatePipelineWires();
}

function settleArtifactStatuses() {
  Object.keys(state.artifactStatus).forEach((edgeId) => {
    if (state.artifactStatus[edgeId] === "active") state.artifactStatus[edgeId] = "complete";
  });
  updatePipelineWires();
}

function failActiveArtifactStatuses() {
  state.runFinished = true;
  Object.keys(state.artifactStatus).forEach((edgeId) => {
    if (state.artifactStatus[edgeId] === "active") state.artifactStatus[edgeId] = "failed";
  });
  updatePipelineWires();
}

function showPipelineNode(nodeId) {
  const definition = pipelineNodes[nodeId];
  if (!definition) return;
  $$(".pipeline-node").forEach((node) => node.classList.toggle("selected", node.dataset.node === nodeId));
  const runtime = state.pipelineStatus[nodeId] || {status: "waiting", detail: "尚未运行"};
  $("#pipeline-node-title").textContent = definition.title;
  $("#pipeline-node-description").textContent = definition.description;
  $("#pipeline-node-input").textContent = runtime.input ? JSON.stringify(runtime.input, null, 2) : definition.input;
  $("#pipeline-node-activity").textContent = runtime.activity || runtime.detail || "等待运行";
  $("#pipeline-node-output").textContent = runtime.output ? JSON.stringify(runtime.output, null, 2) : definition.output;
  $("#pipeline-node-detail").textContent = runtime.detail || "等待 Trace 事件";
  $("#pipeline-node-duration").textContent = formatDuration(runtime.durationMs);
  $("#pipeline-node-tokens").textContent = runtime.totalTokens
    ? `${runtime.totalTokens.toLocaleString()} (${runtime.promptTokens || 0} in / ${runtime.completionTokens || 0} out)`
    : "-";
  $("#pipeline-node-cost").textContent = Number.isFinite(runtime.costUsd) ? `$${runtime.costUsd.toFixed(6)}` : "-";
  $("#pipeline-node-finish").textContent = runtime.finishReason || "-";
  $("#pipeline-node-error").textContent = runtime.error
    ? `${runtime.errorType ? `[${runtime.errorType}] ` : ""}${runtime.error}`
    : "-";
  $("#pipeline-node-status").textContent = runtime.status.toUpperCase();
  $("#pipeline-node-status").className = runtime.status;
  $("#pipeline-pdf-actions").classList.toggle("hidden", nodeId !== "deliver");
  $("#pipeline-open-pdf").classList.toggle("hidden", !state.lastPdfUrl);
  $("#pipeline-open-pdf").href = state.lastPdfUrl || "#";
}

function openResearchEventStream(taskId) {
  state.researchEvents?.close();
  const source = new EventSource(`${serviceBase()}/events?task_id=${encodeURIComponent(taskId)}`);
  state.researchEvents = source;
  source.addEventListener("task_status", (event) => {
    const payload = JSON.parse(event.data || "{}");
    if (payload.task_status === "running") {
      if ((state.pipelineStatus.request?.status || "waiting") === "waiting") {
        setPipelineNodeStatus("request", "active", `task ${taskId} 等待阶段 Trace`);
      }
    } else if (payload.task_status === "failed") {
      failActiveArtifactStatuses();
      const existingFailedNode = Object.keys(state.pipelineStatus).find(
        (key) => state.pipelineStatus[key].status === "failed",
      );
      const active = Object.keys(state.pipelineStatus).find(
        (key) => state.pipelineStatus[key].status === "active",
      ) || "request";
      if (!existingFailedNode) {
        setPipelineNodeStatus(active, "failed", payload.error || "任务执行失败，请查看 Trace", {
          errorType: "runtime_error",
          error: payload.error || "任务执行失败",
        });
      }
      source.close();
      state.researchEvents = null;
    } else if (payload.task_status === "succeeded") {
      state.runFinished = true;
      settleArtifactStatuses();
      source.close();
      state.researchEvents = null;
    }
  });
  source.addEventListener("trace", (event) => applyPipelineTrace((JSON.parse(event.data || "{}")).trace || {}));
}

function applyPipelineTrace(trace) {
  const eventName = String(trace.event || trace.node || "").toLowerCase();
  const stage = String(trace.stage || "").toLowerCase();
  const path = String(trace.path || "").toLowerCase();
  const detail = trace.tool_name || trace.tool || trace.retriever || trace.path || eventName;
  const telemetry = formatPipelineTelemetry(trace);
  const invocationId = String(trace.invocation_id || "");
  if (eventName.startsWith("stage_")) state.hasStageTrace = true;
  if (eventName === "stage_start" && pipelineNodes[stage]) {
    markArtifactsConsumed(stage);
    if (invocationId) {
      const active = state.activeInvocations[stage] || new Set();
      active.add(invocationId);
      state.activeInvocations[stage] = active;
    }
    setPipelineNodeStatus(stage, "active", trace.operation || detail, telemetry);
    $("#research-current-activity").textContent = trace.operation || pipelineNodes[stage].title;
  } else if (eventName === "stage_progress" && pipelineNodes[stage]) {
    setPipelineNodeStatus(stage, "active", trace.operation || detail, telemetry);
    $("#research-current-activity").textContent = trace.operation || pipelineNodes[stage].title;
  } else if (eventName === "stage_end" && pipelineNodes[stage]) {
    if (invocationId) state.activeInvocations[stage]?.delete(invocationId);
    const stillActive = (state.activeInvocations[stage]?.size || 0) > 0;
    setPipelineNodeStatus(
      stage,
      stillActive ? "active" : "complete",
      stillActive ? "仍有并发调用正在执行" : (trace.operation || "阶段完成"),
      telemetry,
    );
    if (!stillActive) markArtifactsReady(stage, trace);
  } else if (eventName === "stage_error" && pipelineNodes[stage]) {
    if (invocationId) state.activeInvocations[stage]?.delete(invocationId);
    setPipelineNodeStatus(stage, "failed", trace.operation || trace.error_message || "阶段失败", telemetry);
    markDownstreamSkipped(stage);
    $("#research-current-activity").textContent = trace.error_message || "阶段执行失败";
  } else if (eventName === "stage_usage" && pipelineNodes[stage]) {
    const status = state.pipelineStatus[stage]?.status || "complete";
    setPipelineNodeStatus(stage, status, trace.operation || "用量已汇总", telemetry);
  } else if (eventName === "artifact_ready" || eventName === "artifact_written") {
    if (pipelineNodes[stage]) markArtifactsReady(stage, trace);
  } else if (eventName === "run_start") {
    setPipelineNodeStatus("request", "active", "运行配置已冻结，正在初始化", telemetry);
  } else if (!state.hasStageTrace && (eventName.includes("retrieval_start") || eventName === "tool_start")) {
    if (!state.pipelineStatus.retrieval || state.pipelineStatus.retrieval.status === "waiting") {
      setPipelineNodeStatus("retrieval", "active", detail, telemetry);
    }
  } else if (!state.hasStageTrace && (eventName.includes("retrieval_end") || eventName === "tool_end")) {
    setPipelineNodeStatus("retrieval", "complete", detail, telemetry);
  } else if (!state.hasStageTrace && eventName === "artifact_written" && path.includes("outline")) {
    setPipelineNodeStatus("evidence", "complete", "证据索引已建立");
    setPipelineNodeStatus("outline", "complete", detail, telemetry);
    setPipelineNodeStatus("writer", "active", "按章节生成内容");
  } else if (!state.hasStageTrace && eventName === "artifact_written" && path.includes("article_polished")) {
    setPipelineNodeStatus("writer", "complete", "文章草稿已生成");
    setPipelineNodeStatus("polish", "complete", detail, telemetry);
    setPipelineNodeStatus("evaluate", "active", "检查引用与质量指标");
  } else if (!state.hasStageTrace && eventName === "artifact_written" && path.includes("article")) {
    setPipelineNodeStatus("writer", "complete", detail, telemetry);
    setPipelineNodeStatus("polish", "active", "去重并统一结构");
  } else if (eventName === "run_end") {
    if (trace.success === false) {
      failActiveArtifactStatuses();
      const existingFailedNode = Object.keys(state.pipelineStatus).find(
        (nodeId) => state.pipelineStatus[nodeId]?.status === "failed",
      );
      if (existingFailedNode) return;
      const activeNode = Object.keys(state.pipelineStatus).find((nodeId) => state.pipelineStatus[nodeId]?.status === "active") || "request";
      setPipelineNodeStatus(activeNode, "failed", trace.error || "任务失败", telemetry);
      markDownstreamSkipped(activeNode);
    } else {
      state.runFinished = true;
      settleArtifactStatuses();
    }
  }
}

function markDownstreamSkipped(failedStage) {
  const order = ["request", "persona", "dialogue", "query", "retrieval", "evidence", "outline", "writer", "polish", "evaluate", "deliver"];
  const failedIndex = order.indexOf(failedStage);
  order.slice(failedIndex + 1).forEach((nodeId) => {
    if ((state.pipelineStatus[nodeId]?.status || "waiting") === "waiting") {
      setPipelineNodeStatus(nodeId, "skipped", "上游阶段失败，未执行");
    }
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
    openResearchEventStream(task.task_id);
    $("#research-current-activity").textContent = "任务已提交，等待后端阶段 Trace...";
    const completed = await fetchJson(`/research-tasks/${encodeURIComponent(task.task_id)}/run`, {method: "POST"});
    if (completed.status !== "succeeded") throw new Error(completed.error || `任务状态：${completed.status}`);
    $("#research-current-activity").textContent = "调研完成，文章、评分与引用已更新。";
    await loadResearchResult(task.task_id);
    toast("调研任务已完成");
  } catch (error) {
    const active = Object.keys(state.pipelineStatus).find((nodeId) => state.pipelineStatus[nodeId]?.status === "active");
    if (active && state.pipelineStatus[active]?.status !== "failed") {
      setPipelineNodeStatus(active, "failed", error.message, {errorType: "request_error", error: error.message});
      markDownstreamSkipped(active);
    }
    $("#research-current-activity").textContent = error.message;
    toast(error.message, "error");
  } finally {
    button.disabled = false;
    button.textContent = demo ? "运行示例主题" : "开始调研";
  }
}

async function loadResearchResult(taskId) {
  const [article, scorecard, dashboard] = await Promise.all([
    fetchJson(`/research-tasks/${encodeURIComponent(taskId)}/article`),
    fetchJson(`/research-tasks/${encodeURIComponent(taskId)}/scorecard`),
    fetchJson(`/research-tasks/${encodeURIComponent(taskId)}/dashboard`),
  ]);
  renderResearchArticle(article.content || "");
  state.lastArticle = article.content || "";
  $("#download-article-md").disabled = !state.lastArticle;
  renderMetricGrid($("#scorecard"), scorecard, 8);
  $("#research-score-section").classList.toggle("hidden", !Object.keys(scorecard).length);
  updatePdfArtifact(dashboard.artifacts?.pdf || {});
}

function updatePdfArtifact(pdf = {}) {
  state.lastPdfUrl = pdf.status === "ready" && pdf.url
    ? `${serviceBase()}${pdf.url}`
    : "";
  $("#open-article-pdf").disabled = !state.lastPdfUrl;
  $("#pipeline-open-pdf").classList.toggle("hidden", !state.lastPdfUrl);
  $("#pipeline-open-pdf").href = state.lastPdfUrl || "#";
  if (pdf.status === "failed") {
    setPipelineNodeStatus("deliver", "failed", pdf.error_message || "PDF 生成失败", {
      errorType: pdf.error_type || "pdf_render_error",
      error: pdf.error_message || "PDF 生成失败",
      output: {markdown: "ready", pdf: "failed"},
    });
  } else if (pdf.status === "ready") {
    setPipelineNodeStatus("deliver", "complete", `PDF 已生成 · ${pdf.page_count || "-"} 页`, {
      output: {
        markdown: "ready",
        pdf: "paperstorm_report.pdf",
        page_count: pdf.page_count,
        size_bytes: pdf.size_bytes,
      },
    });
  }
}

function renderResearchArticle(content) {
  const container = $("#article-content");
  container.innerHTML = "";
  if (!content.trim()) {
    container.textContent = "任务完成，但没有生成文章。";
    return;
  }
  let paragraphIndex = 0;
  content.split(/\n\s*\n/).map((block) => block.trim()).filter(Boolean).forEach((block) => {
    if (block.startsWith("#")) {
      const level = Math.min(4, Math.max(2, (block.match(/^#+/) || ["##"])[0].length + 1));
      const heading = document.createElement(`h${level}`);
      heading.textContent = block.replace(/^#+\s*/, "");
      container.appendChild(heading);
      return;
    }
    paragraphIndex += 1;
    const paragraph = document.createElement("p");
    paragraph.id = `article-paragraph-${paragraphIndex}`;
    paragraph.dataset.articleAnchor = paragraph.id;
    paragraph.textContent = block;
    container.appendChild(paragraph);
  });
}

async function focusArticleCitation(anchor, taskId) {
  if (taskId && state.researchTaskId !== taskId) {
    state.researchTaskId = taskId;
    await loadResearchResult(taskId);
  }
  setMode("research");
  const target = document.getElementById(anchor);
  if (!target) return toast("未找到对应文章段落", "error");
  $$("#article-content .citation-target").forEach((node) => node.classList.remove("citation-target"));
  target.classList.add("citation-target");
  target.scrollIntoView({behavior: "smooth", block: "center"});
  window.setTimeout(() => target.classList.remove("citation-target"), 3200);
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
        memory_retrieval_mode: $("#chat-memory-mode").value,
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
  const telemetry = metadata.telemetry || {};
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
      const authors = Array.isArray(citation.authors) ? citation.authors.filter(Boolean) : [];
      const authorText = authors.length ? `<small class="citation-authors">${escapeHtml(authors.join(", "))}</small>` : "";
      const originalSources = Array.isArray(citation.original_sources) ? citation.original_sources : [];
      const articleLink = citation.article_anchor
        ? `<button class="citation-locator" type="button" data-article-anchor="${escapeHtml(citation.article_anchor)}" data-task-id="${escapeHtml(metadata.used_task_id || "")}">定位文章</button>`
        : "";
      const source = url
        ? `<span class="citation-source"><a href="${escapeHtml(url)}" target="_blank" rel="noopener">${title}</a>${authorText}</span>`
        : articleLink ? `<span class="citation-title">${title}</span>` : `<span class="citation-title">${title}</span><b class="missing-badge">无可用链接</b>`;
      const originals = originalSources.length ? `<ul class="original-sources">${originalSources.map((item) => {
        const sourceTitle = escapeHtml(item.title || `来源 ${item.citation_index || ""}`);
        const sourceUrl = item.url || "";
        const sourceAuthors = Array.isArray(item.authors) && item.authors.length ? `<small>${escapeHtml(item.authors.join(", "))}</small>` : "";
        return `<li><span>[${escapeHtml(item.citation_index || "-")}]</span><span>${sourceUrl ? `<a href="${escapeHtml(sourceUrl)}" target="_blank" rel="noopener">${sourceTitle}</a>` : sourceTitle}${sourceAuthors}</span></li>`;
      }).join("")}</ul>` : "";
      return `<li><div class="citation-row">${source}${articleLink}${page}${chunk}</div>${originals}</li>`;
    }).join("");
    citationsHtml = `<details class="citations"><summary>引用 ${citations.length} 条</summary><ul>${items}</ul></details>`;
  }
  const metricPrefix = telemetry.estimated ? "≈" : "";
  const usageKind = telemetry.estimated ? "估算用量" : "真实用量";
  const durationText = telemetry.duration_ms === null || telemetry.duration_ms === undefined
    ? "耗时未记录"
    : formatDuration(Number(telemetry.duration_ms));
  const metricHtml = role === "user"
    ? (telemetry.message_tokens ? `<span class="message-usage message-telemetry">输入 ${metricPrefix}${escapeHtml(telemetry.message_tokens)} tokens · ${usageKind}</span>` : "")
    : ((telemetry.total_tokens || telemetry.duration_ms !== undefined)
      ? `<span class="message-usage message-telemetry">输入 ${metricPrefix}${escapeHtml(telemetry.prompt_tokens || 0)} · 输出 ${metricPrefix}${escapeHtml(telemetry.completion_tokens || 0)} · 总计 ${metricPrefix}${escapeHtml(telemetry.total_tokens || (Number(telemetry.prompt_tokens || 0) + Number(telemetry.completion_tokens || 0)))} tokens · ${durationText} · ${usageKind}</span>`
      : "");
  const avatar = role === "user" ? "/dashboard/assets/avatar-user.svg" : "/dashboard/assets/avatar-paperstorm.svg";
  node.innerHTML = `<img class="message-avatar" src="${avatar}" alt="" /><div class="message-body"><div class="message-head"><strong>${role === "user" ? "你" : "PaperStorm"}${version}${regenerated}</strong></div><p>${escapeHtml(message.content)}</p>${citationsHtml}${metricHtml}</div>`;
  node.querySelectorAll("[data-article-anchor]").forEach((button) => {
    button.addEventListener("click", () => focusArticleCitation(button.dataset.articleAnchor, button.dataset.taskId));
  });
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

function openArticlePdf() {
  if (!state.lastPdfUrl) return toast("当前任务没有可用的 PDF", "error");
  window.open(state.lastPdfUrl, "_blank", "noopener");
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
$("#open-article-pdf").addEventListener("click", openArticlePdf);
$("#refresh-benchmark-catalog").addEventListener("click", loadBenchmarkCatalog);
$("#benchmark-profile").addEventListener("change", () => {
  if (state.selectedBenchmark) selectBenchmark(state.selectedBenchmark.id);
});
$("#start-benchmark-run").addEventListener("click", startBenchmarkRun);
$("#cancel-benchmark-run").addEventListener("click", cancelBenchmarkRun);
$("#copy-benchmark-command").addEventListener("click", copyBenchmarkCommand);
$("#refresh-runtime-diagnostics").addEventListener("click", refreshRuntimeDiagnostics);

initializePipelineGraph();
loadBenchmarkCatalog();
