"""LLM-backed intent routing for the conversation runtime.

Policy:
    - run_mode == "paperstorm" (real research mode) routes through the LLM
      whenever a provider key is configured; rules remain the safety net.
    - run_mode == "fake" (local demo) stays rule-based unless the operator
      explicitly sets PAPERSTORM_ROUTER_LLM=1.

Env knobs:
    PAPERSTORM_ROUTER_LLM         1 to force LLM routing in fake mode
    PAPERSTORM_ROUTER_PROVIDER    deepseek (default)
    PAPERSTORM_ROUTER_MODEL       deepseek-chat (default)
    PAPERSTORM_ROUTER_API_KEY     falls back to DEEPSEEK_API_KEY
    PAPERSTORM_ROUTER_API_BASE    falls back to DEEPSEEK_API_BASE
"""

import functools
import os
import re
from pathlib import Path
from typing import Callable, Optional


DEFAULT_PROVIDER = "deepseek"
DEFAULT_MODEL = "deepseek-chat"


def _router_cache_size() -> int:
    try:
        return max(0, int(os.getenv("PAPERSTORM_ROUTER_CACHE_SIZE", "512")))
    except ValueError:
        return 512


@functools.lru_cache(maxsize=_router_cache_size())
def _cached_router_completion(
    model_name: str, prompt: str, api_key: str, api_base: str
) -> str:
    """LRU-cached router completion.

    The cache key is the full prompt plus model/config, so identical messages
    in identical context reuse the decision instead of paying another API call.
    Set PAPERSTORM_ROUTER_CACHE_SIZE=0 to disable (fresh decision every turn).
    """
    import litellm

    response = litellm.completion(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        api_key=api_key,
        api_base=api_base,
        temperature=0.0,
        max_tokens=180,
        timeout=20,
        cache={"no-cache": True, "no-store": True},
    )
    choice = response["choices"][0]
    message = choice.get("message") or {}
    return str(message.get("content") or "")


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

    def router_llm(prompt: str) -> str:
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

    def chat_llm(prompt: str) -> str:
        import litellm

        response = litellm.completion(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            api_key=api_key,
            api_base=api_base,
            temperature=0.7,
            max_tokens=400,
            timeout=25,
            cache={"no-cache": True, "no-store": True},
        )
        choice = response["choices"][0]
        message = choice.get("message") or {}
        return str(message.get("content") or "")

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
    return "{0}/{1}".format(provider, model), api_key, api_base


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
