"""
PaperStorm pipeline powered by MiniMax M3 with paper-focused retrievers.

Example:
    python examples/storm_examples/run_paper_storm_minimax.py \
        --topic "retrieval augmented generation evaluation" \
        --retriever arxiv \
        --output-language zh \
        --output-dir ./results/paperstorm_zh \
        --do-research \
        --do-generate-outline \
        --do-generate-article \
        --do-polish-article
"""

import logging
import os
import sys
import warnings
from argparse import ArgumentParser

from knowledge_storm import (
    STORMWikiRunner,
    STORMWikiRunnerArguments,
    STORMWikiLMConfigs,
)
from knowledge_storm.lm import LitellmModel
from knowledge_storm.rm import ArxivRM, LocalPDFRM
from knowledge_storm.utils import load_api_key

from examples.storm_examples.run_storm_wiki_minimax import (
    get_output_dir_name,
    get_topic_for_storm,
)


LOGGER = logging.getLogger("paperstorm")


class PaperStormStdoutFilter:
    def __init__(self, stream):
        self.stream = stream

    def write(self, text):
        if "Provider List: https://docs.litellm.ai/docs/providers" in text:
            return len(text)
        return self.stream.write(text)

    def flush(self):
        return self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


class PaperStormNoiseFilter(logging.Filter):
    def filter(self, record):
        message = record.getMessage()
        if (
            "Error occurs when processing https://en.wikipedia.org/wiki/" in message
            and "'NoneType' object has no attribute 'text'" in message
        ):
            return False
        return True


def configure_paperstorm_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(name)s : %(levelname)-8s : %(message)s")

    noise_filter = PaperStormNoiseFilter()
    root_logger = logging.getLogger()
    if not any(isinstance(item, PaperStormNoiseFilter) for item in root_logger.filters):
        root_logger.addFilter(noise_filter)
    for handler in root_logger.handlers:
        if not any(isinstance(item, PaperStormNoiseFilter) for item in handler.filters):
            handler.addFilter(noise_filter)

    for logger_name in ("LiteLLM", "litellm", "httpx", "httpcore"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    if not verbose and not isinstance(sys.stdout, PaperStormStdoutFilter):
        sys.stdout = PaperStormStdoutFilter(sys.stdout)

    warnings.filterwarnings(
        "ignore",
        message=r".*Pydantic serializer warnings.*",
        category=UserWarning,
    )


def build_lm_settings(args):
    if args.llm_provider == "minimax":
        return {
            "model": args.llm_model or "openai/MiniMax-M3",
            "api_env": "MINIMAX_API_KEY",
            "api_base": os.getenv("MINIMAX_API_BASE", "https://api.minimax.chat/v1"),
        }
    if args.llm_provider == "deepseek":
        model = args.llm_model or "deepseek/deepseek-chat"
        if model == "flash":
            model = "deepseek/deepseek-chat"
        return {
            "model": model,
            "api_env": "DEEPSEEK_API_KEY",
            "api_base": os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
        }
    raise ValueError(f"Unsupported llm provider: {args.llm_provider}")


def build_lm_token_limits():
    return {
        "conv_simulator": 700,
        "question_asker": 700,
        "outline_gen": 1800,
        "article_gen": 1800,
        "article_polish": 4000,
    }


def build_lm_configs(args):
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


def build_paper_retriever(args):
    if args.retriever == "arxiv":
        return ArxivRM(k=args.search_top_k)
    if args.retriever == "local-pdf":
        if not args.pdf_dir:
            raise ValueError("--pdf-dir is required when --retriever local-pdf")
        return LocalPDFRM(pdf_dir=args.pdf_dir, k=args.search_top_k)
    raise ValueError(f"Unsupported retriever: {args.retriever}")


def main(args):
    configure_paperstorm_logging(verbose=args.verbose)
    if os.path.exists("secrets.toml"):
        load_api_key(toml_file_path="secrets.toml")

    settings = build_lm_settings(args)
    topic = args.topic or input("Topic: ")
    topic_for_storm = get_topic_for_storm(
        topic, output_language=args.output_language
    )
    output_dir_name = get_output_dir_name(topic)
    article_dir = os.path.join(args.output_dir, output_dir_name)

    LOGGER.info("Starting PaperStorm run")
    LOGGER.info("Topic: %s", topic)
    LOGGER.info("LLM: %s (%s)", settings["model"], args.llm_provider)
    LOGGER.info("Retriever: %s", args.retriever)
    LOGGER.info("Output directory: %s", article_dir)

    lm_configs = build_lm_configs(args)
    engine_args = STORMWikiRunnerArguments(
        output_dir=args.output_dir,
        max_conv_turn=args.max_conv_turn,
        max_perspective=args.max_perspective,
        search_top_k=args.search_top_k,
        max_thread_num=args.max_thread_num,
    )
    rm = build_paper_retriever(args)
    runner = STORMWikiRunner(engine_args, lm_configs, rm)

    try:
        runner.run(
            topic=topic_for_storm,
            output_dir_name=output_dir_name,
            do_research=args.do_research,
            do_generate_outline=args.do_generate_outline,
            do_generate_article=args.do_generate_article,
            do_polish_article=args.do_polish_article,
            remove_duplicate=args.remove_duplicate,
        )
        runner.post_run()
        runner.summary()
        LOGGER.info("Key outputs:")
        LOGGER.info("  conversation: %s", os.path.join(article_dir, "conversation_log.json"))
        LOGGER.info("  raw search:   %s", os.path.join(article_dir, "raw_search_results.json"))
        LOGGER.info("  outline:      %s", os.path.join(article_dir, "storm_gen_outline.txt"))
    except Exception as e:
        logging.exception(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--topic", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./results/paperstorm")
    parser.add_argument("--output-language", choices=["original", "zh"], default="zh")
    parser.add_argument(
        "--llm-provider", choices=["minimax", "deepseek"], default="deepseek"
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="flash",
        help="Model name. For DeepSeek, flash maps to deepseek/deepseek-chat.",
    )
    parser.add_argument(
        "--retriever", choices=["arxiv", "local-pdf"], default="arxiv"
    )
    parser.add_argument("--pdf-dir", type=str, default=None)
    parser.add_argument("--max-thread-num", type=int, default=3)
    parser.add_argument("--max-conv-turn", type=int, default=2)
    parser.add_argument("--max-perspective", type=int, default=2)
    parser.add_argument("--search-top-k", type=int, default=3)
    parser.add_argument("--retrieve-top-k", type=int, default=3)
    parser.add_argument("--do-research", action="store_true")
    parser.add_argument("--do-generate-outline", action="store_true")
    parser.add_argument("--do-generate-article", action="store_true")
    parser.add_argument("--do-polish-article", action="store_true")
    parser.add_argument("--remove-duplicate", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    main(parser.parse_args())
