"""Provider resolution.

Resolution priority:
  1. Explicit `provider=` argument passed by the caller
  2. LAB_LLM_PROVIDER env var
  3. Auto-detect: if exactly one of ANTHROPIC_API_KEY / OPENAI_API_KEY is
     set, use that provider
  4. Tiebreak when both keys are set: anthropic (preserves original behavior)
  5. Error if neither key is set
"""

from __future__ import annotations

import logging
import os

from lab.llm.provider import LLMProvider

logger = logging.getLogger(__name__)


SUPPORTED_PROVIDERS: tuple[str, ...] = ("anthropic", "openai")


def get_provider(
    provider: str | None = None,
    *,
    api_key: str | None = None,
) -> LLMProvider:
    """Resolve and construct a provider instance.

    `api_key` is provider-specific — passing it forces that provider's
    auth path regardless of env vars. If you need to be explicit about
    which provider an api_key belongs to, also pass `provider=`.
    """
    chosen = _resolve_name(provider)
    if chosen == "anthropic":
        from lab.llm.anthropic_provider import AnthropicProvider
        return AnthropicProvider(api_key=api_key)
    if chosen == "openai":
        from lab.llm.openai_provider import OpenAIProvider
        return OpenAIProvider(api_key=api_key)
    raise ValueError(
        f"unsupported provider {chosen!r}; expected one of {SUPPORTED_PROVIDERS}"
    )


def _resolve_name(explicit: str | None) -> str:
    if explicit:
        if explicit not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"unsupported provider {explicit!r}; "
                f"expected one of {SUPPORTED_PROVIDERS}"
            )
        return explicit

    env_choice = os.environ.get("LAB_LLM_PROVIDER", "").strip().lower()
    if env_choice:
        if env_choice not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"LAB_LLM_PROVIDER={env_choice!r} is not a known provider; "
                f"expected one of {SUPPORTED_PROVIDERS}"
            )
        return env_choice

    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    if has_anthropic and not has_openai:
        return "anthropic"
    if has_openai and not has_anthropic:
        return "openai"
    if has_anthropic and has_openai:
        logger.info("both ANTHROPIC_API_KEY and OPENAI_API_KEY set; defaulting to anthropic")
        return "anthropic"
    raise RuntimeError(
        "No LLM provider configured. Set one of:\n"
        "  - ANTHROPIC_API_KEY (recommended for --agent)\n"
        "  - OPENAI_API_KEY\n"
        "Or pass provider= explicitly."
    )


def list_available_providers() -> list[str]:
    """For diagnostics — which providers have working credentials right now."""
    out: list[str] = []
    if os.environ.get("ANTHROPIC_API_KEY"):
        out.append("anthropic")
    if os.environ.get("OPENAI_API_KEY"):
        out.append("openai")
    return out
