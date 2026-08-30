"""Generate editable Draw.io and SVG architecture diagrams for PaperStorm."""

from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent

COLORS = {
    "ink": "#172033",
    "muted": "#526078",
    "line": "#CBD5E1",
    "canvas": "#F7F9FC",
    "white": "#FFFFFF",
    "blue": "#2F6FED",
    "blue_soft": "#EDF3FF",
    "teal": "#0F9D8A",
    "teal_soft": "#E8F7F4",
    "gold": "#D58A16",
    "gold_soft": "#FFF5DF",
    "purple": "#7567A8",
    "purple_soft": "#F1EFFA",
    "red": "#D94A4A",
    "red_soft": "#FFF0EE",
    "slate": "#435A73",
    "slate_soft": "#EEF2F6",
}

KINDS = {
    "primary": (COLORS["blue"], COLORS["blue_soft"]),
    "control": (COLORS["teal"], COLORS["teal_soft"]),
    "external": (COLORS["gold"], COLORS["gold_soft"]),
    "storage": (COLORS["purple"], COLORS["purple_soft"]),
    "danger": (COLORS["red"], COLORS["red_soft"]),
    "storm": (COLORS["slate"], COLORS["slate_soft"]),
    "neutral": (COLORS["line"], COLORS["white"]),
}


@dataclass
class Node:
    id: str
    x: int
    y: int
    w: int
    h: int
    title: str
    subtitle: str = ""
    kind: str = "neutral"
    shape: str = "rounded"
    font_size: int = 16
    layer: str = "node"


@dataclass
class Edge:
    source: str
    target: str
    label: str = ""
    kind: str = "primary"
    dashed: bool = False
    direction: str = "auto"
    points: list[tuple[int, int]] = field(default_factory=list)
    source_offset: Optional[float] = None
    target_offset: Optional[float] = None


@dataclass(frozen=True)
class SequenceMessage:
    source: str
    target: str
    label: str
    source_offset: float
    target_offset: float
    kind: str = "control"


@dataclass
class Diagram:
    name: str
    width: int
    height: int
    title: str
    subtitle: str
    nodes: list[Node]
    edges: list[Edge]


def panel(node_id, x, y, w, h, title, kind="neutral"):
    return Node(node_id, x, y, w, h, title, kind=kind, shape="panel", font_size=17, layer="panel")


def executive_diagram():
    nodes = [
        panel("business_panel", 70, 105, 1780, 120, "业务入口与需求", "neutral"),
        panel("flow_panel", 70, 245, 1780, 490, "核心业务流程", "primary"),
        panel("base_panel", 70, 755, 1780, 135, "核心能力底座", "control"),
        panel("value_panel", 70, 910, 1780, 115, "工程保障与业务价值", "storage"),
        Node("business", 720, 132, 480, 66, "业务需求", "论文调研 · 知识问答 · 内部知识服务", "external", font_size=22),
        Node("platform", 720, 275, 480, 70, "PaperStorm Agent 平台", "统一入口 · 任务服务 · 会话服务", "primary", font_size=23),
        Node("route", 720, 385, 480, 72, "意图识别与任务编排", "区分闲聊、知识问答与深度调研", "control", font_size=20),
        Node("chat", 300, 510, 420, 82, "智能问答链路", "上下文装配 → 记忆召回 → RAG 证据门控", "control", font_size=20),
        Node("research", 1200, 510, 420, 82, "深度调研链路", "Multi-Agent 协作 → STORM 长文生成", "storm", font_size=20),
        Node("answer", 300, 635, 420, 68, "有引用的可信回答", "连续对话 · 跨会话记忆 · 证据可展开", "primary", font_size=19),
        Node("article", 1200, 635, 420, 68, "结构化调研文章", "知识策展 · 大纲 · 章节写作 · 润色", "primary", font_size=19),
        Node("rag", 145, 790, 365, 70, "RAG", "混合召回 · 重排 · 引用", "primary", font_size=19),
        Node("memory", 565, 790, 365, 70, "Memory", "跨会话长期记忆 · 时间事实", "storage", font_size=19),
        Node("context", 985, 790, 365, 70, "Context", "分层 Token 预算 · 自动压缩恢复", "control", font_size=19),
        Node("multi", 1405, 790, 365, 70, "Multi-Agent", "规划 · 检索 · 审核 · 生成", "storm", font_size=19),
        Node("runtime", 145, 943, 480, 56, "异步运行", "Async Queue · LangGraph · Checkpoint · SSE", "storage", font_size=17),
        Node("evaluation", 720, 943, 480, 56, "可观测与领域评测", "Langfuse Score · PIM Domain Pilot", "primary", font_size=17),
        Node("value", 1295, 943, 480, 56, "业务价值", "可追溯 · 可恢复 · 可迭代 · 可演示", "control", font_size=17),
    ]
    edges = [
        Edge("business", "platform", direction="down"),
        Edge("platform", "route", direction="down"),
        Edge("route", "chat", "问答", "control", direction="down"),
        Edge("route", "research", "调研", "storm", direction="down"),
        Edge("chat", "answer", direction="down"),
        Edge("research", "article", direction="down"),
        Edge("answer", "rag", kind="primary", direction="down", points=[(510, 730), (327, 730)]),
        Edge("answer", "memory", kind="storage", direction="down", points=[(510, 735), (747, 735)]),
        Edge("answer", "context", kind="control", direction="down", points=[(510, 740), (1167, 740)]),
        Edge("article", "multi", kind="storm", direction="down"),
        Edge("rag", "runtime", kind="storage", direction="down"),
        Edge("memory", "runtime", kind="storage", direction="down"),
        Edge("context", "evaluation", kind="primary", direction="down"),
        Edge("multi", "evaluation", kind="primary", direction="down"),
        Edge("runtime", "value", kind="control", direction="right"),
        Edge("evaluation", "value", kind="control", direction="right"),
    ]
    return Diagram(
        "paperstorm-executive-overview",
        1920,
        1080,
        "PaperStorm：论文调研与知识问答 Agent 平台",
        "从业务需求到可信回答与调研文章，形成可追溯、可恢复、可评测的 Agent 工程闭环",
        nodes,
        edges,
    )


def detailed_diagram():
    nodes = [
        panel("external_panel", 30, 105, 280, 930, "外部资源与模型", "external"),
        panel("chat_panel", 330, 105, 575, 930, "智能问答链路", "control"),
        panel("research_panel", 925, 105, 605, 930, "STORM 深度调研链路", "storm"),
        panel("algo_panel", 1550, 105, 620, 930, "关键能力底座与算法", "control"),
        panel("bottom_panel", 30, 1055, 2140, 215, "工程保障、公开评测与反馈闭环", "storage"),
        Node("web", 65, 155, 210, 66, "用户与系统入口", "Web · REST API · SSE · MCP · CLI", "control", font_size=17),
        Node("papers", 65, 255, 210, 86, "论文与知识源", "arXiv\n本地 PDF\nZotero\n企业文档", "external", font_size=17),
        Node("llm", 65, 385, 210, 76, "LLM 服务", "DeepSeek · MiniMax\n意图 / 对话 / 写作 / 裁判", "external", font_size=17),
        Node("embedding", 65, 505, 210, 76, "检索模型", "GTE / BGE / Qwen3\nMiniLM / Cross-Encoder", "external", font_size=17),
        Node("tools", 65, 625, 210, 100, "Tool System", "ArxivRM\nLocalPDFRM\nKnowledgeBaseQA\nResearchQA", "primary", font_size=17),
        Node("public_data", 65, 770, 210, 96, "公开评测数据", "SciFact\nQASPER\nLongMemEval-S\nPIM Pilot", "external", font_size=17),
        Node("artifacts", 65, 910, 210, 76, "交付与运行产物", "引用回答 · 对话 · 大纲 · 文章\nTrace · Metrics · PDF", "storage", font_size=16),

        Node("chat_entry", 370, 145, 495, 62, "Chat Service 与动作路由", "普通回复 / Memory / Evidence / Deep Research", "control", font_size=18),
        Node("context", 370, 245, 150, 66, "Context Assemble", "Pinned / Active / Summary", "control", font_size=14),
        Node("memory", 542, 245, 150, 66, "Memory Recall", "跨会话 episode / fact", "storage", font_size=14),
        Node("session", 714, 245, 150, 66, "Session Recall", "历史消息与旧话题", "storage", font_size=14),
        Node("unified_rag", 445, 365, 345, 82, "统一 Hybrid RAG", "BM25 + Dense → RRF → 可选 Cross-Encoder", "primary", font_size=17),
        Node("grade", 445, 495, 345, 72, "Evidence Gate", "来源过滤 · 充分性判断 · Claim-Citation", "control", font_size=16),
        Node("cited", 370, 620, 205, 66, "有引用的回答", "原始标题 · 作者 · 链接", "primary", font_size=16),
        Node("deep", 600, 620, 205, 66, "升级深度调研", "现有证据不足", "storm", font_size=16),
        Node("memory_write", 445, 745, 345, 66, "Memory Policy 与写回", "仅稳定用户事实、偏好和决策", "storage", font_size=16),
        Node("chat_trace", 445, 870, 345, 62, "回答、引用与 Trace", "Token · Latency · Span · Session", "control", font_size=16),

        Node("research_task", 965, 145, 525, 62, "Research Task", "主题 · arXiv / Local PDF · 运行参数", "storm", font_size=18),
        Node("persona", 965, 245, 150, 66, "Persona Generator", "生成多研究视角", "storm", font_size=14),
        Node("conv", 1135, 245, 180, 66, "Conv Simulator", "WikiWriter ↔ TopicExpert", "storm", font_size=14),
        Node("source_rm", 1335, 245, 155, 66, "论文源检索", "arXiv API / Local PDF", "external", font_size=14),
        Node("curation", 1030, 365, 390, 76, "Knowledge Curation", "多视角对话 + 查询循环 → StormInformationTable", "storm", font_size=17),
        Node("research_files", 1030, 480, 390, 68, "研究阶段产物", "conversation_log.json · raw_search_results.json", "storage", font_size=15),
        Node("outline", 965, 610, 230, 70, "两阶段 Outline", "直接草案 → 对话增强", "storm", font_size=16),
        Node("semantic_topk", 1220, 610, 270, 70, "MiniLM 语义 Top-K", "按章节余弦筛选已收集 snippet", "primary", font_size=16),
        Node("writing", 965, 735, 230, 70, "章节并发写作", "Outline + StormInformationTable", "storm", font_size=16),
        Node("polish", 1220, 735, 270, 70, "Article Polish", "整篇去重 · 摘要 · 风格统一", "storm", font_size=16),
        Node("article", 1030, 870, 390, 70, "结构化调研文章", "Markdown · 原始论文引用 · Scorecard · PDF", "primary", font_size=18),

        Node("rag_algo", 1580, 150, 560, 132, "问答 / 企业知识库 RAG", "结构化 Chunk / Overlap\nBM25 + Dense 双路召回 → RRF\n可选 Cross-Encoder 重排\nEvidence Gate → Citation", "primary", font_size=18),
        Node("storm_retrieval", 1580, 305, 560, 132, "调研专用检索", "arXiv：API relevance + 领域过滤\nLocal PDF：固定 Chunk + 词项交集\n写作：MiniLM cosine Top-K\n不经过统一 Hybrid RAG", "storm", font_size=18),
        Node("memory_algo", 1580, 460, 560, 112, "Memory 算法", "Episode → Fact → Provenance / Entity\nSQLite WAL + 时间有效事实\nBM25 + Embedding + Entity + Time → RRF / MMR", "storage", font_size=18),
        Node("context_algo", 1580, 595, 560, 132, "Context 工程", "Pinned / Active / Summary\nMemory / Evidence / Artifact 分层预算\nWatermark 压缩\nCompaction DAG + Restore", "control", font_size=18),
        Node("runtime_algo", 1580, 750, 560, 132, "Agent Runtime", "Async Queue · LangGraph StateGraph\nSQLite Checkpoint · Retry · Cancel\nSSE Event · Span Trace\nLangfuse Score", "storm", font_size=18),
        Node("governance_algo", 1580, 905, 560, 92, "生产治理", "ACL · 幂等 · Cache · 熔断 · 脱敏 · 失败恢复", "danger", font_size=18),
        Node("checkpoint", 80, 1115, 320, 92, "可恢复 Runtime", "LangGraph Checkpoint · Retry · Cancel\nSession / Task 持久化", "storage", font_size=17),
        Node("observability", 445, 1115, 320, 92, "Langfuse Score", "SSE · Runtime Event · Agent Trace\nSpan Tree · Token / Latency", "control", font_size=17),
        Node("bench", 810, 1115, 470, 92, "PIM Domain Pilot", "PIM 专项评测 · SciFact · QASPER\nSmoke / Quality · Metrics / CI / Bad Cases", "primary", font_size=17),
        Node("feedback", 1325, 1115, 360, 92, "数据驱动优化", "检索质量 · Memory 召回\nContext 保真 · 延迟与成本", "control", font_size=17),
        Node("outcome", 1730, 1115, 390, 92, "项目价值", "可信回答 · 深度文章\n可追溯 · 可恢复 · 可量化", "external", font_size=17),
    ]
    edges = [
        Edge("web", "chat_entry", kind="control", direction="right"),
        Edge("chat_entry", "context", kind="control", direction="down"),
        Edge("chat_entry", "memory", kind="storage", direction="down"),
        Edge("chat_entry", "session", kind="storage", direction="down"),
        Edge("context", "unified_rag", kind="primary", direction="down"),
        Edge("memory", "unified_rag", kind="storage", direction="down"),
        Edge("session", "unified_rag", kind="storage", direction="down"),
        Edge("unified_rag", "grade", kind="primary", direction="down"),
        Edge("grade", "cited", "sufficient", "primary", direction="down"),
        Edge("grade", "deep", "insufficient", "storm", direction="down"),
        Edge("cited", "memory_write", kind="storage", direction="down"),
        Edge("memory_write", "chat_trace", kind="control", direction="down"),
        Edge("deep", "research_task", kind="storm", direction="right"),
        Edge("research_task", "persona", kind="storm", direction="down"),
        Edge("persona", "conv", kind="storm", direction="right"),
        Edge("conv", "source_rm", "query", "external", direction="right"),
        Edge("conv", "curation", "dialogue", "storm", direction="down"),
        Edge("source_rm", "curation", "evidence", "primary", direction="down"),
        Edge("curation", "research_files", kind="storage", direction="down"),
        Edge("research_files", "outline", "conversation_log", "storm", direction="down"),
        Edge("curation", "semantic_topk", "StormInformationTable", "primary", direction="down"),
        Edge("outline", "writing", kind="storm", direction="down"),
        Edge("semantic_topk", "writing", "Top-K snippets", "primary", direction="down"),
        Edge("writing", "polish", kind="storm", direction="right"),
        Edge("polish", "article", kind="primary", direction="down"),
        Edge("papers", "tools", kind="external", direction="down"),
        Edge("llm", "chat_entry", kind="external", direction="right"),
        Edge("public_data", "bench", kind="external", direction="down"),
        Edge("chat_trace", "checkpoint", kind="storage", direction="down"),
        Edge("article", "observability", kind="control", direction="down"),
        Edge("checkpoint", "observability", kind="storage", direction="right"),
        Edge("observability", "bench", kind="primary", direction="right"),
        Edge("bench", "feedback", kind="primary", direction="right"),
        Edge("feedback", "outcome", kind="control", direction="right"),
    ]
    return Diagram(
        "paperstorm-agent-system-flow",
        2200,
        1300,
        "PaperStorm Agent 系统流程与关键算法",
        "问答使用统一 Hybrid RAG；深度调研使用 STORM Multi-Agent、论文源召回与原生语义筛选",
        nodes,
        edges,
    )


def async_runtime_sequence_diagram():
    participants = [
        ("browser", "Browser", "submit topic"),
        ("fastapi", "FastAPI", "task API"),
        ("queue", "Async Queue", "durable jobs"),
        ("runtime", "Agent Runtime", "LangGraph"),
        ("retriever", "Retriever", "hybrid RAG"),
        ("llm", "LLM", "route and generate"),
        ("checkpoint", "Checkpoint", "resume state"),
        ("sse", "SSE", "event stream"),
        ("langfuse", "Langfuse", "trace and score"),
    ]
    nodes = []
    control_participants = {"fastapi", "runtime", "sse", "langfuse"}
    for index, (node_id, title, subtitle) in enumerate(participants):
        kind = "control" if node_id in control_participants else "primary"
        nodes.append(
            Node(
                node_id,
                55 + index * 205,
                130,
                165,
                760,
                title,
                subtitle,
                kind,
                shape="participant",
                font_size=15,
            )
        )
    messages = [
        SequenceMessage("browser", "fastapi", "POST /research", 0.15, 0.15),
        SequenceMessage("fastapi", "queue", "enqueue task", 0.20, 0.20),
        SequenceMessage("fastapi", "browser", "202 Accepted + task_id", 0.25, 0.25),
        SequenceMessage("queue", "runtime", "claim job", 0.32, 0.32),
        SequenceMessage("runtime", "checkpoint", "create checkpoint", 0.37, 0.37),
        SequenceMessage("runtime", "langfuse", "start trace", 0.42, 0.42),
        SequenceMessage("runtime", "retriever", "retrieve evidence", 0.49, 0.49),
        SequenceMessage("retriever", "llm", "ranked context", 0.54, 0.54),
        SequenceMessage("llm", "runtime", "grounded draft", 0.59, 0.59),
        SequenceMessage("runtime", "checkpoint", "persist progress", 0.66, 0.66),
        SequenceMessage("runtime", "sse", "emit stage event", 0.72, 0.72),
        SequenceMessage("sse", "browser", "stream progress", 0.78, 0.78),
        SequenceMessage("runtime", "langfuse", "record spans + score", 0.84, 0.84),
        SequenceMessage("runtime", "sse", "emit completed result", 0.90, 0.90),
        SequenceMessage("sse", "browser", "final article + citations", 0.95, 0.95),
    ]
    edges = [
        Edge(
            message.source,
            message.target,
            message.label,
            message.kind,
            direction="sequence",
            source_offset=message.source_offset,
            target_offset=message.target_offset,
        )
        for message in messages
    ]
    return Diagram(
        "paperstorm-async-runtime-sequence",
        1900,
        920,
        "PaperStorm 异步 Agent Runtime 时序",
        "请求快速确认，后台检索与生成可恢复执行，并通过 SSE、Langfuse 持续反馈进度与质量",
        nodes,
        edges,
    )


def drawio_style(node):
    stroke, fill = KINDS[node.kind]
    if node.shape == "panel":
        return (
            "rounded=1;whiteSpace=wrap;html=1;verticalAlign=top;align=left;"
            f"spacingTop=12;spacingLeft=14;fontSize={node.font_size};fontStyle=1;"
            f"strokeColor={stroke};fillColor={fill};strokeWidth=2;arcSize=8;"
        )
    if node.shape == "participant":
        stroke, fill = KINDS[node.kind]
        return (
            "shape=umlLifeline;perimeter=lifelinePerimeter;whiteSpace=wrap;html=1;"
            f"fontSize={node.font_size};fontColor={COLORS['ink']};fontStyle=1;"
            f"strokeColor={stroke};fillColor={fill};strokeWidth=1.5;"
        )
    return (
        "rounded=0;whiteSpace=wrap;html=1;align=center;verticalAlign=middle;"
        f"fontSize={node.font_size};fontColor={COLORS['ink']};fontStyle=1;"
        f"strokeColor={stroke};fillColor={fill};strokeWidth=1.5;spacing=6;shadow=1;"
    )


def drawio_label(node):
    value = f"<b>{escape(node.title)}</b>"
    if node.subtitle:
        subtitle = escape(node.subtitle).replace("\n", "<br>")
        value += f"<br><font style=\"font-size:10px;color:{COLORS['muted']}\">{subtitle}</font>"
    return value


def validate_sequence_offsets(diagram):
    for edge in diagram.edges:
        if edge.direction != "sequence":
            continue
        for attribute in ("source_offset", "target_offset"):
            offset = getattr(edge, attribute)
            if not isinstance(offset, (int, float)) or not 0 <= offset <= 1:
                raise ValueError(
                    f"sequence edge {edge.source}->{edge.target} {attribute} "
                    "must be within 0..1"
                )


def write_drawio(diagram):
    validate_sequence_offsets(diagram)
    mxfile = ET.Element("mxfile", {"host": "app.diagrams.net", "version": "24.7.17"})
    page = ET.SubElement(mxfile, "diagram", {"id": diagram.name, "name": "Page-1"})
    model = ET.SubElement(page, "mxGraphModel", {
        "dx": "2400", "dy": "1500", "grid": "1", "gridSize": "10",
        "guides": "1", "tooltips": "1", "connect": "1", "arrows": "1",
        "fold": "1", "page": "1", "pageScale": "1",
        "pageWidth": str(diagram.width), "pageHeight": str(diagram.height),
        "math": "0", "shadow": "0",
    })
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})
    for node in diagram.nodes:
        cell = ET.SubElement(root, "mxCell", {
            "id": node.id, "value": drawio_label(node), "style": drawio_style(node),
            "vertex": "1", "parent": "1",
        })
        ET.SubElement(cell, "mxGeometry", {
            "x": str(node.x), "y": str(node.y), "width": str(node.w),
            "height": str(node.h), "as": "geometry",
        })
    for index, edge in enumerate(diagram.edges, 1):
        color = KINDS[edge.kind][0]
        style = (
            "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;"
            f"html=1;strokeColor={color};strokeWidth=2;endArrow=block;endFill=1;"
            f"dashed={1 if edge.dashed else 0};fontSize=10;fontColor={COLORS['muted']};"
        )
        if edge.direction == "sequence":
            source = next(node for node in diagram.nodes if node.id == edge.source)
            target = next(node for node in diagram.nodes if node.id == edge.target)
            style += (
                f"exitX=0.5;exitY={edge.source_offset};exitPerimeter=0;"
                f"entryX=0.5;entryY={edge.target_offset};entryPerimeter=0;"
            )
        cell = ET.SubElement(root, "mxCell", {
            "id": f"edge-{index}", "value": escape(edge.label), "style": style,
            "edge": "1", "parent": "1", "source": edge.source, "target": edge.target,
        })
        geometry = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
        if edge.points:
            array = ET.SubElement(geometry, "Array", {"as": "points"})
            for x, y in edge.points:
                ET.SubElement(array, "mxPoint", {"x": str(x), "y": str(y)})
    ET.indent(mxfile, space="  ")
    ET.ElementTree(mxfile).write(ROOT / f"{diagram.name}.drawio", encoding="utf-8", xml_declaration=True)


def svg_text(parent, node):
    title_y = node.y + (28 if node.shape == "panel" else 24)
    text = ET.SubElement(parent, "text", {
        "x": str(node.x + (16 if node.shape == "panel" else node.w / 2)),
        "y": str(title_y),
        "class": "panel-title" if node.shape == "panel" else "node-title",
        "text-anchor": "start" if node.shape == "panel" else "middle",
        "font-size": str(node.font_size),
    })
    text.text = node.title
    if node.subtitle:
        lines = node.subtitle.split("\n")
        start_y = title_y + (20 if node.font_size >= 18 else 17)
        for index, line in enumerate(lines):
            sub = ET.SubElement(parent, "text", {
                "x": str(node.x + node.w / 2), "y": str(start_y + index * 15),
                "class": "node-subtitle", "text-anchor": "middle", "font-size": "11",
            })
            sub.text = line


def edge_path(source, target, edge):
    if edge.direction == "sequence":
        sx, sy = source.x + source.w / 2, source.y + source.h * edge.source_offset
        tx, ty = target.x + target.w / 2, target.y + target.h * edge.target_offset
        return f"M {sx} {sy} L {tx} {ty}"
    if edge.points:
        start = (source.x + source.w, source.y + source.h / 2)
        end = (target.x, target.y + target.h / 2)
        values = [start, *edge.points, end]
        return "M " + " L ".join(f"{x} {y}" for x, y in values)
    if edge.direction == "down" or (edge.direction == "auto" and target.y > source.y + source.h):
        sx, sy = source.x + source.w / 2, source.y + source.h
        tx, ty = target.x + target.w / 2, target.y
        mid = (sy + ty) / 2
        return f"M {sx} {sy} L {sx} {mid} L {tx} {mid} L {tx} {ty}"
    sx, sy = source.x + source.w, source.y + source.h / 2
    tx, ty = target.x, target.y + target.h / 2
    mid = (sx + tx) / 2
    return f"M {sx} {sy} L {mid} {sy} L {mid} {ty} L {tx} {ty}"


def write_svg(diagram):
    validate_sequence_offsets(diagram)
    svg = ET.Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg", "width": str(diagram.width),
        "height": str(diagram.height), "viewBox": f"0 0 {diagram.width} {diagram.height}",
        "role": "img", "aria-label": diagram.title,
    })
    defs = ET.SubElement(svg, "defs")
    shadow = ET.SubElement(defs, "filter", {"id": "node-shadow", "x": "-20%", "y": "-20%", "width": "140%", "height": "150%"})
    ET.SubElement(shadow, "feDropShadow", {"dx": "0", "dy": "3", "stdDeviation": "3", "flood-color": "#24364A", "flood-opacity": ".14"})
    pattern = ET.SubElement(defs, "pattern", {"id": "grid", "width": "20", "height": "20", "patternUnits": "userSpaceOnUse"})
    ET.SubElement(pattern, "path", {"d": "M 20 0 L 0 0 0 20", "fill": "none", "stroke": "#E6EBF2", "stroke-width": ".6"})
    for kind, (color, _) in KINDS.items():
        marker = ET.SubElement(defs, "marker", {
            "id": f"arrow-{kind}", "viewBox": "0 0 10 10", "refX": "9", "refY": "5",
            "markerWidth": "7", "markerHeight": "7", "orient": "auto-start-reverse",
        })
        ET.SubElement(marker, "path", {"d": "M 0 0 L 10 5 L 0 10 z", "fill": color})
    style = ET.SubElement(defs, "style")
    style.text = (
        "text{font-family:'Segoe UI','Microsoft YaHei',Arial,sans-serif;fill:#172033;}"
        ".header{font-weight:750;fill:#FFFFFF;}.subtitle{fill:#C8D2DE;}"
        ".legend{fill:#D8E0E8;font-size:9px;}"
        ".panel-title,.node-title{font-weight:700;}.node-subtitle{fill:#526078;font-weight:400;}"
        ".edge-label{fill:#526078;font-size:10px;font-weight:600;}"
    )
    ET.SubElement(svg, "rect", {"width": "100%", "height": "100%", "fill": COLORS["canvas"]})
    ET.SubElement(svg, "rect", {"width": "100%", "height": "100%", "fill": "url(#grid)"})
    ET.SubElement(svg, "rect", {"x": "20", "y": "18", "width": str(diagram.width - 40), "height": "70", "fill": "#172033", "stroke": "#172033"})
    ET.SubElement(svg, "rect", {"x": "20", "y": "18", "width": "10", "height": "70", "fill": COLORS["teal"]})
    title = ET.SubElement(svg, "text", {"x": "48", "y": "49", "class": "header", "font-size": "28"})
    title.text = diagram.title
    subtitle = ET.SubElement(svg, "text", {"x": "48", "y": "73", "class": "subtitle", "font-size": "13"})
    subtitle.text = diagram.subtitle
    for node in (item for item in diagram.nodes if item.layer == "panel"):
        stroke, fill = KINDS[node.kind]
        ET.SubElement(svg, "rect", {
            "x": str(node.x), "y": str(node.y), "width": str(node.w), "height": str(node.h),
            "fill": COLORS["white"], "fill-opacity": ".82", "stroke": COLORS["line"], "stroke-width": "1.5",
        })
        ET.SubElement(svg, "rect", {
            "x": str(node.x), "y": str(node.y), "width": str(node.w), "height": "42",
            "fill": stroke, "fill-opacity": ".11", "stroke": "none",
        })
        ET.SubElement(svg, "rect", {"x": str(node.x), "y": str(node.y), "width": "6", "height": "42", "fill": stroke})
        svg_text(svg, node)

    lookup = {node.id: node for node in diagram.nodes}
    for edge in diagram.edges:
        source, target = lookup[edge.source], lookup[edge.target]
        color = KINDS[edge.kind][0]
        ET.SubElement(svg, "path", {
            "d": edge_path(source, target, edge), "fill": "none", "stroke": color,
            "stroke-width": "2.2", "stroke-dasharray": "7 6" if edge.dashed else "none",
            "marker-end": f"url(#arrow-{edge.kind})",
        })
        if edge.label:
            if edge.direction == "sequence":
                lx = (source.x + source.w / 2 + target.x + target.w / 2) / 2
                ly = source.y + source.h * edge.source_offset - 6
            else:
                lx = (source.x + source.w / 2 + target.x + target.w / 2) / 2
                ly = (source.y + source.h / 2 + target.y + target.h / 2) / 2 - 6
            label = ET.SubElement(svg, "text", {"x": str(lx), "y": str(ly), "class": "edge-label", "text-anchor": "middle"})
            label.text = edge.label
    node_index = 0
    for node in (item for item in diagram.nodes if item.layer == "node"):
        node_index += 1
        stroke, fill = KINDS[node.kind]
        if node.shape == "participant":
            ET.SubElement(svg, "rect", {
                "x": str(node.x), "y": str(node.y), "width": str(node.w), "height": "66",
                "fill": COLORS["white"], "stroke": stroke, "stroke-width": "1.5", "filter": "url(#node-shadow)",
            })
            ET.SubElement(svg, "line", {
                "x1": str(node.x + node.w / 2), "y1": str(node.y + 66),
                "x2": str(node.x + node.w / 2), "y2": str(node.y + node.h),
                "stroke": stroke, "stroke-width": "1.5", "stroke-dasharray": "6 6",
            })
        else:
            ET.SubElement(svg, "rect", {
                "x": str(node.x), "y": str(node.y), "width": str(node.w), "height": str(node.h),
                "fill": COLORS["white"], "stroke": stroke, "stroke-width": "1.5", "filter": "url(#node-shadow)",
            })
            ET.SubElement(svg, "rect", {"x": str(node.x), "y": str(node.y), "width": "7", "height": str(node.h), "fill": stroke})
        if node.w >= 180 and node.shape != "participant":
            ET.SubElement(svg, "rect", {"x": str(node.x + 14), "y": str(node.y + 12), "width": "24", "height": "18", "fill": stroke})
            number = ET.SubElement(svg, "text", {"x": str(node.x + 26), "y": str(node.y + 25), "text-anchor": "middle", "font-size": "9", "font-weight": "800", "fill": "#FFFFFF"})
            number.text = f"{node_index:02d}"
        svg_text(svg, node)
    legend_x = diagram.width - 545
    legend_y = 40
    for offset, (kind, label) in enumerate((("primary", "检索数据"), ("control", "控制流"), ("storm", "STORM"), ("external", "外部依赖"), ("storage", "记忆 / 存储"))):
        color = KINDS[kind][0]
        x = legend_x + offset * 104
        ET.SubElement(svg, "rect", {"x": str(x), "y": str(legend_y), "width": "8", "height": "8", "fill": color})
        item = ET.SubElement(svg, "text", {"x": str(x + 13), "y": str(legend_y + 8), "class": "legend"})
        item.text = label
    note = ET.SubElement(svg, "text", {"x": str(diagram.width - 545), "y": "70", "class": "legend"})
    note.text = "实线箭头表示执行或数据方向；颜色表示链路类型，不表示执行先后"
    ET.indent(svg, space="  ")
    ET.ElementTree(svg).write(ROOT / f"{diagram.name}.svg", encoding="utf-8", xml_declaration=True)


def main():
    for diagram in (executive_diagram(), detailed_diagram(), async_runtime_sequence_diagram()):
        write_drawio(diagram)
        write_svg(diagram)
        print(f"generated {diagram.name}: {diagram.width}x{diagram.height}")


if __name__ == "__main__":
    main()
