import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from . import STORMWikiLMConfigs, STORMWikiRunner, STORMWikiRunnerArguments
from .lm import LitellmModel
from .paperstorm_eval import EvalCase, evaluate_run, write_scorecards
from .paperstorm_references import materialize_article_references
from .rm import ArxivRM, LocalPDFRM
from .utils import load_api_key

from examples.storm_examples.run_paper_storm_minimax import (
    build_artifact_paths,
    build_lm_settings,
    build_lm_token_limits,
    configure_paperstorm_logging,
)
from .paperstorm_trace import (
    PaperStormStageCallback,
    PaperStormTraceRecorder,
    TracedRetrievalModel,
)
from examples.storm_examples.run_storm_wiki_minimax import get_topic_for_storm


class EmptyRetrievalError(RuntimeError):
    pass


@dataclass
class PaperStormPipelineConfig:
    topic: str
    topic_for_storm: str
    output_root: str
    output_dir_name: str
    article_dir: str
    output_language: str = "zh"
    llm_provider: str = "deepseek"
    llm_model: str = "flash"
    retriever: str = "arxiv"
    pdf_dir: str = ""
    max_thread_num: int = 3
    max_conv_turn: int = 2
    max_perspective: int = 3
    search_top_k: int = 5
    retrieve_top_k: int = 5
    do_research: bool = True
    do_generate_outline: bool = True
    do_generate_article: bool = True
    do_polish_article: bool = True
    remove_duplicate: bool = False
    disable_trace: bool = False
    verbose: bool = False
    expected_keywords: tuple = ()
    forbidden_keywords: tuple = ()


def build_pipeline_config_from_task_state(state):
    options = state.get("options") or {}
    article_dir = Path(state["output_dir"])
    topic = state["topic"]
    output_language = state.get("output_language", "zh")
    return PaperStormPipelineConfig(
        topic=topic,
        topic_for_storm=get_topic_for_storm(topic, output_language=output_language),
        output_root=str(article_dir.parent),
        output_dir_name=article_dir.name,
        article_dir=str(article_dir),
        output_language=output_language,
        llm_provider=options.get("llm_provider", "deepseek"),
        llm_model=options.get("llm_model", "flash"),
        retriever=state.get("retriever", "arxiv"),
        pdf_dir=options.get("pdf_dir") or "",
        max_thread_num=int(options.get("max_thread_num", 3)),
        max_conv_turn=int(options.get("max_conv_turn", 2)),
        max_perspective=int(options.get("max_perspective", 3)),
        search_top_k=int(options.get("search_top_k", 5)),
        retrieve_top_k=int(options.get("retrieve_top_k", 5)),
        do_research=bool(options.get("do_research", True)),
        do_generate_outline=bool(options.get("do_generate_outline", True)),
        do_generate_article=bool(options.get("do_generate_article", True)),
        do_polish_article=bool(options.get("do_polish_article", True)),
        remove_duplicate=bool(options.get("remove_duplicate", False)),
        disable_trace=bool(options.get("disable_trace", False)),
        verbose=bool(options.get("verbose", False)),
        expected_keywords=tuple(state.get("expected_keywords") or ()),
        forbidden_keywords=tuple(state.get("forbidden_keywords") or ()),
    )


def run_paperstorm_pipeline_task(state):
    config = build_pipeline_config_from_task_state(state)
    return run_paperstorm_pipeline(config)


def run_paperstorm_pipeline(config: PaperStormPipelineConfig):
    configure_paperstorm_logging(verbose=config.verbose)
    if os.path.exists("secrets.toml"):
        load_api_key(toml_file_path="secrets.toml")

    article_dir = Path(config.article_dir)
    article_dir.mkdir(parents=True, exist_ok=True)
    args = _config_to_namespace(config)
    settings = build_lm_settings(args)
    trace = PaperStormTraceRecorder(str(article_dir), enabled=not config.disable_trace)
    trace.emit(
        "run_start",
        topic=config.topic,
        topic_for_storm=config.topic_for_storm,
        output_dir=str(article_dir),
        llm_provider=config.llm_provider,
        llm_model=settings["model"],
        retriever=config.retriever,
        run_mode="paperstorm",
        do_research=config.do_research,
        do_generate_outline=config.do_generate_outline,
        do_generate_article=config.do_generate_article,
        do_polish_article=config.do_polish_article,
    )
    trace.start_stage(
        "request",
        "初始化模型、检索器与 STORM Runner",
        input={
            "topic": config.topic,
            "llm_provider": config.llm_provider,
            "llm_model": settings["model"],
            "retriever": config.retriever,
        },
    )

    try:
        lm_configs = _build_lm_configs(args)
        engine_args = STORMWikiRunnerArguments(
            output_dir=config.output_root,
            max_conv_turn=config.max_conv_turn,
            max_perspective=config.max_perspective,
            search_top_k=config.search_top_k,
            max_thread_num=config.max_thread_num,
        )
        rm = _build_paper_retriever(args)
        if not config.disable_trace:
            rm = TracedRetrievalModel(
                rm,
                trace_recorder=trace,
                retriever_name=type(rm).__name__,
            )
        runner = STORMWikiRunner(engine_args, lm_configs, rm)
        _instrument_runner_stages(runner, trace)
        callback_handler = PaperStormStageCallback(trace, topic=config.topic)
        trace.end_stage(
            output_summary={
                "runner": "STORMWikiRunner",
                "retriever": config.retriever,
                "status": "ready",
            }
        )
        trace.emit(
            "artifact_ready",
            stage="request",
            artifact_name="research_task.json",
        )
        runner.run(
            topic=config.topic_for_storm,
            output_dir_name=config.output_dir_name,
            do_research=config.do_research,
            do_generate_outline=config.do_generate_outline,
            do_generate_article=config.do_generate_article,
            do_polish_article=config.do_polish_article,
            remove_duplicate=config.remove_duplicate,
            callback_handler=callback_handler,
        )
        ensure_research_sources(
            article_dir,
            retriever=config.retriever,
            enabled=config.do_research,
        )
        trace.start_stage(
            "evaluate",
            "汇总运行日志并检查文章与引用完整性",
            input={"article_dir": str(article_dir)},
        )
        runner.post_run()
        runner.summary()
        reference_result = materialize_article_references(article_dir)
        trace.emit(
            "artifact_ready",
            stage="evaluate",
            artifact_name="references",
            reference_count=reference_result["reference_count"],
        )
        _write_pipeline_scorecard(config)
        scorecard_path = article_dir / "scorecard.json"
        scorecard = (
            json.loads(scorecard_path.read_text(encoding="utf-8"))
            if scorecard_path.exists()
            else {}
        )
        trace.end_stage(output_summary={"scorecard": scorecard.get("scores", {})})
        trace.emit(
            "artifact_ready", stage="evaluate", artifact_name="scorecard.json"
        )
        trace.start_stage(
            "deliver",
            "登记文章、Trace 与评估产物",
            input={"article_dir": str(article_dir)},
        )
        artifact_paths = build_artifact_paths(str(article_dir))
        existing_artifacts = [
            path for path in artifact_paths.values() if os.path.exists(path)
        ]
        for artifact in existing_artifacts:
            trace.emit("artifact_written", path=artifact)
        trace.end_stage(output_summary={"artifacts": existing_artifacts})
        trace.emit("run_end", success=True)
        trace.write_summary(
            success=True,
            artifacts=existing_artifacts,
            extra={
                "topic": config.topic,
                "retriever": config.retriever,
                "llm_model": settings["model"],
                "output_dir": str(article_dir),
            },
        )
        return {
            "success": True,
            "article_dir": str(article_dir),
            "artifacts": existing_artifacts,
        }
    except Exception as error:
        if not trace.events or trace.events[-1].get("event") != "stage_error":
            trace.fail_current_stage(error)
        trace.emit(
            "run_end",
            success=False,
            error_type=type(error).__name__,
            error=str(error),
        )
        trace.write_summary(
            success=False,
            error={"type": type(error).__name__, "message": str(error)},
            extra={
                "topic": config.topic,
                "retriever": config.retriever,
                "llm_model": settings["model"],
                "output_dir": str(article_dir),
            },
        )
        raise


def ensure_research_sources(article_dir, retriever, enabled=True):
    """Reject a zero-source research run instead of publishing a hollow report."""
    if not enabled or retriever != "arxiv":
        return 0
    article_dir = Path(article_dir)
    source_count = _count_saved_sources(article_dir / "url_to_info.json", registry=True)
    if source_count == 0:
        # A research-only run persists retrieval results before STORM builds the
        # article-stage URL registry. Treat those results as authoritative too.
        source_count = _count_saved_sources(article_dir / "raw_search_results.json")
    if source_count == 0:
        raise EmptyRetrievalError(
            "empty_retrieval: arXiv 未返回可用论文，请调整主题或领域约束后重试。"
        )
    return source_count


def _count_saved_sources(path, registry=False):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    if registry and isinstance(payload, dict):
        payload = payload.get("url_to_info") or {}
    if isinstance(payload, dict):
        return sum(
            1
            for key, value in payload.items()
            if str(key).startswith(("http://", "https://"))
            or (isinstance(value, dict) and value.get("url"))
        )
    if isinstance(payload, list):
        return sum(1 for value in payload if isinstance(value, dict) and value.get("url"))
    return 0


def _instrument_runner_stages(runner, trace):
    _wrap_runner_usage(
        runner,
        "run_knowledge_curation_module",
        trace,
        "dialogue",
    )
    _wrap_runner_usage(
        runner,
        "run_outline_generation_module",
        trace,
        "outline",
    )
    _wrap_runner_method(
        runner,
        "run_article_generation_module",
        trace,
        "writer",
        "按大纲与证据生成文章章节",
    )
    _wrap_runner_method(
        runner,
        "run_article_polishing_module",
        trace,
        "polish",
        "去重并统一文章结构与表达",
    )


def _wrap_runner_method(runner, method_name, trace, stage, operation):
    original = getattr(runner, method_name, None)
    if original is None:
        return

    def traced_method(*args, **kwargs):
        before = _snapshot_lm_telemetry(getattr(runner, "lm_configs", None))
        trace.start_stage(stage, operation)
        try:
            result = original(*args, **kwargs)
        except Exception as error:
            trace.fail_current_stage(error)
            raise
        after = _snapshot_lm_telemetry(getattr(runner, "lm_configs", None))
        trace.end_stage(
            output_summary={"status": "completed"},
            **_telemetry_delta(before, after),
        )
        artifact_name = {
            "writer": "storm_gen_article.txt",
            "polish": "article_polished.txt",
        }.get(stage)
        if artifact_name:
            trace.emit(
                "artifact_ready", stage=stage, artifact_name=artifact_name
            )
        return result

    setattr(runner, method_name, traced_method)


def _wrap_runner_usage(runner, method_name, trace, stage):
    original = getattr(runner, method_name, None)
    if original is None:
        return

    def traced_method(*args, **kwargs):
        before = _snapshot_lm_telemetry(getattr(runner, "lm_configs", None))
        result = original(*args, **kwargs)
        after = _snapshot_lm_telemetry(getattr(runner, "lm_configs", None))
        trace.emit(
            "stage_usage",
            stage=stage,
            operation="语言模型用量已汇总",
            **_telemetry_delta(before, after),
        )
        return result

    setattr(runner, method_name, traced_method)


def _snapshot_lm_telemetry(lm_configs):
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "estimated_cost": 0.0,
        "cost_events": 0,
    }
    if lm_configs is None:
        return totals
    seen = set()
    for attr_name, lm in vars(lm_configs).items():
        if "_lm" not in attr_name or lm is None or id(lm) in seen:
            continue
        seen.add(id(lm))
        for entry in list(getattr(lm, "history", []) or []):
            usage = entry.get("usage") or {}
            prompt = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
            completion = int(
                usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
            )
            totals["prompt_tokens"] += prompt
            totals["completion_tokens"] += completion
            totals["total_tokens"] += int(usage.get("total_tokens", prompt + completion) or 0)
            cost = entry.get("cost")
            if isinstance(cost, (int, float)):
                totals["estimated_cost"] += float(cost)
                totals["cost_events"] += 1
    totals["estimated_cost"] = round(totals["estimated_cost"], 10)
    return totals


def _telemetry_delta(before, after):
    delta = {
        "prompt_tokens": max(0, after["prompt_tokens"] - before["prompt_tokens"]),
        "completion_tokens": max(
            0, after["completion_tokens"] - before["completion_tokens"]
        ),
        "total_tokens": max(0, after["total_tokens"] - before["total_tokens"]),
    }
    if after.get("cost_events", 0) > before.get("cost_events", 0):
        delta["estimated_cost"] = round(
            max(0.0, after["estimated_cost"] - before["estimated_cost"]), 10
        )
    return delta


def _config_to_namespace(config):
    return SimpleNamespace(
        llm_provider=config.llm_provider,
        llm_model=config.llm_model,
        retriever=config.retriever,
        pdf_dir=config.pdf_dir or None,
        search_top_k=config.search_top_k,
    )


def _build_lm_configs(args):
    settings = build_lm_settings(args)
    token_limits = build_lm_token_limits()
    api_key = os.getenv(settings["api_env"])
    if not api_key:
        raise ValueError(f"Please set {settings['api_env']} in secrets.toml.")

    llm_kwargs = {
        "api_key": api_key,
        "api_base": settings["api_base"],
        "temperature": 1.0,
        "top_p": 0.9,
    }
    model_name = settings["model"]

    lm_configs = STORMWikiLMConfigs()
    lm_configs.set_conv_simulator_lm(
        LitellmModel(
            model=model_name, max_tokens=token_limits["conv_simulator"], **llm_kwargs
        )
    )
    lm_configs.set_question_asker_lm(
        LitellmModel(
            model=model_name, max_tokens=token_limits["question_asker"], **llm_kwargs
        )
    )
    lm_configs.set_outline_gen_lm(
        LitellmModel(
            model=model_name, max_tokens=token_limits["outline_gen"], **llm_kwargs
        )
    )
    lm_configs.set_article_gen_lm(
        LitellmModel(
            model=model_name, max_tokens=token_limits["article_gen"], **llm_kwargs
        )
    )
    lm_configs.set_article_polish_lm(
        LitellmModel(
            model=model_name, max_tokens=token_limits["article_polish"], **llm_kwargs
        )
    )
    return lm_configs


def _build_paper_retriever(args):
    if args.retriever == "arxiv":
        return ArxivRM(k=args.search_top_k)
    if args.retriever == "local-pdf":
        if not args.pdf_dir:
            raise ValueError("--pdf-dir is required when retriever is local-pdf")
        return LocalPDFRM(pdf_dir=args.pdf_dir, k=args.search_top_k)
    raise ValueError(f"Unsupported retriever: {args.retriever}")


def _write_pipeline_scorecard(config):
    case = EvalCase(
        topic=config.topic,
        expected_keywords=list(config.expected_keywords),
        forbidden_keywords=list(config.forbidden_keywords),
        expected_language=config.output_language,
        min_sources=1,
    )
    output_dir = Path(config.article_dir)
    scorecard = evaluate_run(output_dir, case)
    write_scorecards(output_dir, scorecard)
    (output_dir / "pipeline_worker.json").write_text(
        json.dumps(
            {
                "runner": "paperstorm",
                "retriever": config.retriever,
                "llm_provider": config.llm_provider,
                "llm_model": config.llm_model,
                "score": scorecard["scores"]["total"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
