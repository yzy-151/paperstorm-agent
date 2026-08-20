"""LLM-backed intent routing for the conversation runtime.

Policy:
    - run_mode == "paperstorm" (real research mode) routes through the LLM
      whenever a provider key is configured; rules remain the safety net.
    - run_mode == "fake" (local demo) stays rule-based unless the operator
      explicitly sets PAPERSTORM_ROUTER_LLM=1.

Env knobs:
    PAPERSTORM_ROUTER_LLM         1 to force LLM routing in fake mode
    PAPERSTORM_ROUTER_PROVIDER    deepseek (default)
    PAPERSTORM_ROUTER_MODEL       deepseek-v4-flash (default)
    PAPERSTORM_ROUTER_API_KEY     falls back to DEEPSEEK_API_KEY
    PAPERSTORM_ROUTER_API_BASE    falls back to DEEPSEEK_API_BASE
"""

import functools
import os
import re
import time
from pathlib import Path
from typing import Callable, Dict, Optional


DEFAULT_PROVIDER = "deepseek"
DEFAULT_MODEL = "deepseek-v4-flash"
DEEPSEEK_V4_CONTEXT_TOKENS = 1_000_000
DEEPSEEK_V4_MAX_OUTPUT_TOKENS = 384_000


def _router_cache_size() -> int:
    try:
        return max(0, int(os.getenv("PAPERSTORM_ROUTER_CACHE_SIZE", "512")))
    except ValueError:
        return 512


@functools.lru_cache(maxsize=_router_cache_size())
def _cached_router_completion(
    model_name: str, prompt: str, api_key: str, api_base: str
) -> Dict:
    """LRU-cached router completion.

    The cache key is the full prompt plus model/config, so identical messages
    in identical context reuse the decision instead of paying another API call.
    Set PAPERSTORM_ROUTER_CACHE_SIZE=0 to disable (fresh decision every turn).
    """
    import litellm

    started = time.perf_counter()
    try:
        response = litellm.completion(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            api_key=api_key,
            api_base=api_base,
            temperature=0.0,
            max_tokens=512,
            timeout=20,
            response_format={"type": "json_object"},
            cache={"no-cache": True, "no-store": True},
        )
        result = _completion_result(response, model_name)
        result["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        return result
    except Exception as error:
        return _failed_result(error, time.perf_counter() - started)


def _load_flat_toml_env(path: str = "secrets.toml"):
    """Load flat KEY = \"value\" TOML into the environment (no-op if absent)."""
    candidates = [Path(path)]
    if os.getenv("PAPERSTORM_SECRETS_PATH"):
        candidates.append(Path(os.getenv("PAPERSTORM_SECRETS_PATH")))
    candidates.append(Path(__file__).resolve().parents[1] / "secrets.toml")
    seen = set()
    for candidate in candidates:
        try:
            text = candidate.read_text(encoding="utf-8")
        except OSError:
            continue
        resolved = str(candidate.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        for key, value in re.findall(
            r'^\s*([A-Za-z0-9_]+)\s*=\s*"([^"]*)"', text, flags=re.MULTILINE
        ):
            os.environ.setdefault(key, value)


def build_router_llm_callable(
    enabled: Optional[bool] = None,
) -> Optional[Callable[[str], str]]:
    """Return a prompt->text callable for intent routing, or None when off."""
    if enabled is None:
        flag = str(os.getenv("PAPERSTORM_ROUTER_LLM", "")).strip().lower()
        enabled = flag in {"1", "true", "yes", "on"}
    if not enabled:
        return None
    config = _resolve_provider_config()
    if config is None:
        return None
    model_name, api_key, api_base = config

    def router_llm(prompt: str) -> Dict:
        return _cached_router_completion(model_name, prompt, api_key, api_base)

    return router_llm


def build_chat_llm_callable(
    enabled: Optional[bool] = None,
) -> Optional[Callable[[str], str]]:
    """Return a prompt->text callable that generates casual chat replies.

    Policy:
        - PAPERSTORM_CHAT_LLM=1 enables the provider, =0 disables it.
        - An unset flag is offline by default. Callers in ``paperstorm`` mode
          must opt in with ``enabled=True``. Merely having a key in the shell
          must never make tests or fake demos call a paid API.
    """
    if enabled is None:
        flag = str(os.getenv("PAPERSTORM_CHAT_LLM", "")).strip().lower()
        enabled = flag in {"1", "true", "yes", "on"}
    if not enabled:
        return None
    config = _resolve_provider_config()
    if config is None:
        return None
    model_name, api_key, api_base = config

    def chat_llm(
        prompt: str,
        response_contract: Optional[Dict] = None,
        user_message: str = "",
        output_budget: Optional[int] = None,
    ) -> Dict:
        import litellm

        budget = int(
            output_budget
            or select_output_budget(user_message, response_contract or {})
        )
        return complete_chat_with_telemetry(
            completion=litellm.completion,
            model=model_name,
            prompt=prompt,
            api_key=api_key,
            api_base=api_base,
            output_budget=budget,
            timeout=25,
        )

    return chat_llm


def build_judge_llm_callable(
    enabled: Optional[bool] = None,
) -> Optional[Callable[[str], str]]:
    """Return a prompt->text callable used as an LLM evidence judge.

    Frontier agents (Claude Code / Hermes) do not rely on keyword-overlap
    thresholds: the model itself reads the question plus retrieved evidence and
    decides whether it can answer. This callable powers that step; it is
    explicitly enabled by a real runtime or PAPERSTORM_JUDGE_LLM=1. An unset
    flag falls back to the deterministic local grader.
    """
    if enabled is None:
        flag = str(os.getenv("PAPERSTORM_JUDGE_LLM", "")).strip().lower()
        enabled = flag in {"1", "true", "yes", "on"}
    if not enabled:
        return None
    config = _resolve_provider_config()
    if config is None:
        return None
    model_name, api_key, api_base = config

    def judge_llm(prompt: str) -> str:
        import litellm

        response = litellm.completion(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            api_key=api_key,
            api_base=api_base,
            temperature=0.0,
            max_tokens=30,
            timeout=20,
            cache={"no-cache": True, "no-store": True},
        )
        choice = response["choices"][0]
        message = choice.get("message") or {}
        return str(message.get("content") or "")

    return judge_llm


def build_memory_extractor_callable(enabled: Optional[bool] = None):
    """Return a structured durable-memory candidate extractor."""
    if enabled is None:
        enabled = str(os.getenv("PAPERSTORM_MEMORY_LLM", "")).lower() in {
            "1", "true", "yes", "on"
        }
    if not enabled:
        return None
    config = _resolve_provider_config()
    if config is None:
        return None
    model_name, api_key, api_base = config

    def extract(prompt: str):
        import litellm

        response = litellm.completion(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            api_key=api_key,
            api_base=api_base,
            temperature=0.0,
            max_tokens=500,
            timeout=25,
            response_format={"type": "json_object"},
            cache={"no-cache": True, "no-store": True},
        )
        return str((response["choices"][0].get("message") or {}).get("content") or "")

    return extract


def build_context_summarizer_callable(enabled: Optional[bool] = None):
    """Return the LLM compressor used only after the context watermark."""
    if enabled is None:
        enabled = str(os.getenv("PAPERSTORM_SUMMARY_LLM", "")).lower() in {
            "1", "true", "yes", "on"
        }
    if not enabled:
        return None
    config = _resolve_provider_config()
    if config is None:
        return None
    model_name, api_key, api_base = config

    def summarize(prompt: str):
        import litellm

        response = litellm.completion(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            api_key=api_key,
            api_base=api_base,
            temperature=0.0,
            max_tokens=4000,
            timeout=60,
            response_format={"type": "json_object"},
            cache={"no-cache": True, "no-store": True},
        )
        return str((response["choices"][0].get("message") or {}).get("content") or "")

    return summarize


def _resolve_provider_config():
    """Return (model_name, api_key, api_base) or None when not configured."""
    _load_flat_toml_env()
    provider = os.getenv("PAPERSTORM_ROUTER_PROVIDER") or DEFAULT_PROVIDER
    model = os.getenv("PAPERSTORM_ROUTER_MODEL") or DEFAULT_MODEL
    api_key = (
        os.getenv("PAPERSTORM_ROUTER_API_KEY")
        or os.getenv("{0}_API_KEY".format(provider.upper()))
        or os.getenv("DEEPSEEK_API_KEY")
    )
    api_base = (
        os.getenv("PAPERSTORM_ROUTER_API_BASE")
        or os.getenv("{0}_API_BASE".format(provider.upper()))
        or os.getenv("DEEPSEEK_API_BASE")
    )
    if not api_key:
        return None
    if provider == "deepseek" and model.startswith("deepseek-v4-"):
        # Older LiteLLM releases do not map DeepSeek V4 yet. Its official API
        # is OpenAI-compatible, so use the generic provider without relying on
        # LiteLLM's model metadata table.
        return "openai/{0}".format(model), api_key, api_base or "https://api.deepseek.com"
    return "{0}/{1}".format(provider, model), api_key, api_base


def model_capabilities(model=DEFAULT_MODEL):
    if str(model).startswith("deepseek-v4-"):
        return {
            "model": str(model),
            "context_tokens": DEEPSEEK_V4_CONTEXT_TOKENS,
            "max_output_tokens": DEEPSEEK_V4_MAX_OUTPUT_TOKENS,
            "provider_mode": "openai_compatible",
        }
    return {
        "model": str(model),
        "context_tokens": 64_000,
        "max_output_tokens": 8_000,
        "provider_mode": "litellm_mapped",
    }


def select_output_budget(
    message: str,
    response_contract: Optional[Dict] = None,
    max_output_tokens: int = 65_536,
) -> int:
    """Choose enough output room without reserving a huge budget for every turn."""
    contract = response_contract or {}
    requested = _positive_int(contract.get("requested_output_tokens"))
    text = "{0} {1}".format(message or "", contract.get("task") or "").lower()
    explicit = _requested_length_tokens(text)
    if requested or explicit:
        return min(max_output_tokens, max(2048, requested, explicit))
    if contract.get("continue_previous") or any(
        marker in text for marker in ("续写", "继续写", "小说", "故事", "长篇", "creative")
    ):
        return min(max_output_tokens, 16_384)
    if contract.get("requires_citations") or any(
        marker in text for marker in ("详细", "报告", "分析", "代码", "方案", "论文")
    ):
        return min(max_output_tokens, 8192)
    if len(str(message or "").strip()) <= 20:
        return 2048
    return min(max_output_tokens, 4096)


def complete_chat_with_telemetry(
    completion,
    model: str,
    prompt: str,
    api_key: str,
    api_base: str,
    output_budget: int,
    timeout: int,
) -> Dict:
    """Run chat generation and continue once when the provider stops on length."""
    started = time.perf_counter()
    content_parts = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    finish_reason = ""
    try:
        for segment in range(2):
            segment_prompt = prompt
            if segment:
                partial = "".join(content_parts)
                segment_prompt = (
                    "继续上一段回答，从中断处直接接上。不要重复开头，不要总结，不要"
                    "自我介绍。\n\n原始任务：\n{0}\n\n已生成内容：\n{1}\n\n续写："
                ).format(prompt, partial[-12000:])
            response = completion(
                model=model,
                messages=[{"role": "user", "content": segment_prompt}],
                api_key=api_key,
                api_base=api_base,
                temperature=0.7,
                max_tokens=max(1, int(output_budget)),
                timeout=timeout,
                cache={"no-cache": True, "no-store": True},
            )
            normalized = _completion_result(response, model)
            content_parts.append(normalized["content"])
            for key in usage:
                usage[key] += int((normalized.get("usage") or {}).get(key) or 0)
            finish_reason = normalized.get("finish_reason") or ""
            if finish_reason != "length":
                break
        return {
            "content": "".join(content_parts),
            "finish_reason": finish_reason or "unknown",
            "usage": usage,
            "cost_usd": _estimate_cost(model, usage),
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "output_budget": int(output_budget),
            "segments": len(content_parts),
            "truncated": finish_reason == "length",
            "error": None,
        }
    except Exception as error:
        failed = _failed_result(error, time.perf_counter() - started)
        failed.update(
            {
                "output_budget": int(output_budget),
                "segments": len(content_parts),
                "truncated": False,
                "usage": usage,
            }
        )
        return failed


def classify_llm_error(error) -> str:
    name = type(error).__name__.lower()
    message = str(error or "").lower()
    if isinstance(error, TimeoutError) or "timeout" in name or "timed out" in message:
        return "timeout"
    if "rate" in name or "rate limit" in message or "429" in message:
        return "rate_limit"
    if "auth" in name or "api key" in message or "401" in message or "403" in message:
        return "authentication"
    if "connection" in name or "unavailable" in message or "dns" in message:
        return "provider_unavailable"
    return "provider_error"


def _completion_result(response, model):
    choice = response["choices"][0]
    message = choice.get("message") or {}
    usage = _usage_dict(response.get("usage") or {})
    return {
        "content": str(message.get("content") or ""),
        "finish_reason": str(choice.get("finish_reason") or "unknown"),
        "usage": usage,
        "cost_usd": _estimate_cost(model, usage),
        "error": None,
    }


def _failed_result(error, elapsed_seconds):
    return {
        "content": "",
        "finish_reason": "error",
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "cost_usd": 0.0,
        "latency_ms": round(float(elapsed_seconds) * 1000.0, 3),
        "error": {
            "type": classify_llm_error(error),
            "message": str(error),
            "recoverable": classify_llm_error(error) not in {"authentication"},
        },
    }


def _usage_dict(value):
    return {
        key: int(value.get(key) or 0)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def _estimate_cost(model, usage):
    # DeepSeek public API list prices used by the existing benchmark reports.
    if "deepseek" not in str(model).lower():
        return 0.0
    return round(
        int(usage.get("prompt_tokens") or 0) / 1_000_000 * 0.27
        + int(usage.get("completion_tokens") or 0) / 1_000_000 * 1.10,
        8,
    )


def _requested_length_tokens(text):
    match = re.search(r"(\d{1,7})\s*(字|词|words?|tokens?)", str(text), re.I)
    if match:
        amount = int(match.group(1))
        unit = match.group(2).lower()
        multiplier = 2.0 if unit == "字" else 1.5 if unit in {"词", "word", "words"} else 1.0
        return int(amount * multiplier)
    if "十万字" in text:
        return 65_536
    if "一万字" in text or "万字" in text:
        return 20_000
    return 0


def _positive_int(value):
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def build_intent_router(
    run_mode: str = "fake",
    llm_router: Optional[Callable[[str], str]] = None,
):
    """Construct the conversation intent router with LLM wiring applied."""
    from .paperstorm_intent_router import PaperStormIntentRouter

    if llm_router is not None:
        return PaperStormIntentRouter(llm_router=llm_router)
    real_mode = str(run_mode or "").strip().lower() == "paperstorm"
    return PaperStormIntentRouter(
        llm_router=build_router_llm_callable(enabled=real_mode)
    )
