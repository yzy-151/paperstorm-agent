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

import os
import re
from pathlib import Path
from typing import Callable, Optional


DEFAULT_PROVIDER = "deepseek"
DEFAULT_MODEL = "deepseek-chat"


def _load_flat_toml_env(path: str = "secrets.toml"):
    """Load flat KEY = \"value\" TOML into the environment (no-op if absent)."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return
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
    model_name = "{0}/{1}".format(provider, model)

    def router_llm(prompt: str) -> str:
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

    return router_llm


def build_intent_router(
    run_mode: str = "fake",
    llm_router: Optional[Callable[[str], str]] = None,
):
    """Construct the conversation intent router with LLM wiring applied."""
    from .paperstorm_intent_router import PaperStormIntentRouter

    if llm_router is not None:
        return PaperStormIntentRouter(llm_router=llm_router)
    real_mode = str(run_mode or "").strip().lower() == "paperstorm"
    return PaperStormIntentRouter(llm_router=build_router_llm_callable(enabled=real_mode))
