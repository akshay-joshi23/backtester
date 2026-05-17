"""Anthropic API integration for Strategy Lab.

Takes a natural-language strategy description and returns Python code defining
a Strategy subclass that can be executed by `lab.runner`.

Uses prompt caching for the system prompt (which contains the Strategy ABC
docs and few-shot examples — ~3k tokens, reused on every call).
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL = "claude-opus-4-7"
SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "system.md"


@dataclass
class GenerationResult:
    code: str                       # the extracted Python source
    raw_response: str               # the model's full text response
    universe: list[str]             # parsed from `# Defaults:` comment or detected
    train_end: str                  # parsed from defaults or fallback
    rebalance_freq: int             # parsed from defaults or fallback
    usage: dict | None              # token usage from the API response


DEFAULT_UNIVERSE = ["SPY", "TLT"]
DEFAULT_TRAIN_END = "2015-01-01"
DEFAULT_REBALANCE_FREQ = 21


def load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text()


def generate_strategy(
    prompt: str,
    *,
    model: str = MODEL,
    max_tokens: int = 4096,
    temperature: float = 0.2,
    api_key: str | None = None,
) -> GenerationResult:
    """Call Anthropic with the system prompt + user description, parse out code.

    Raises if ANTHROPIC_API_KEY is not set and `api_key` not passed.
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. "
            "Set it via `export ANTHROPIC_API_KEY=...` or pass api_key=."
        )

    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError("anthropic SDK not installed. `pip install anthropic`") from e

    client = anthropic.Anthropic(api_key=api_key)
    system_prompt = load_system_prompt()

    logger.info("calling %s for strategy generation (system=%d chars)",
                model, len(system_prompt))
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": prompt}],
    )
    raw = "".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    )
    code = _extract_python_block(raw)
    universe, train_end, rebalance_freq = _parse_defaults(code)
    usage = None
    if response.usage:
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cache_creation_input_tokens": getattr(
                response.usage, "cache_creation_input_tokens", 0,
            ),
            "cache_read_input_tokens": getattr(
                response.usage, "cache_read_input_tokens", 0,
            ),
        }
    return GenerationResult(
        code=code,
        raw_response=raw,
        universe=universe,
        train_end=train_end,
        rebalance_freq=rebalance_freq,
        usage=usage,
    )


def _extract_python_block(text: str) -> str:
    """Pull the first ```python ... ``` fenced block. Fallback to whole text."""
    m = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


_DEFAULTS_PAT = re.compile(
    r"#\s*Defaults?:\s*"
    r"universe\s*=\s*\[(?P<universe>[^\]]+)\]\s*,\s*"
    r"train_end\s*=\s*\"(?P<train_end>[\d\-]+)\"\s*,\s*"
    r"rebalance_freq\s*=\s*(?P<rebalance_freq>\d+)",
)


def _parse_defaults(code: str) -> tuple[list[str], str, int]:
    """Pull universe/train_end/rebalance_freq from the top-of-file comment.

    Falls back to sensible defaults if the comment is missing or malformed.
    """
    m = _DEFAULTS_PAT.search(code)
    if not m:
        return DEFAULT_UNIVERSE, DEFAULT_TRAIN_END, DEFAULT_REBALANCE_FREQ
    tickers_raw = m.group("universe")
    universe = [t.strip().strip("'\"").upper() for t in tickers_raw.split(",")]
    universe = [t for t in universe if t]
    if not universe:
        universe = DEFAULT_UNIVERSE
    return (
        universe,
        m.group("train_end"),
        int(m.group("rebalance_freq")),
    )
