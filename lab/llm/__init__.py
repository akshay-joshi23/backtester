"""Strategy Lab LLM layer (provider-agnostic).

Public API preserved across providers:
  - generate_strategy(prompt, ...) — natural language → validated Strategy code
  - refine_strategy(prior_code, prior_metrics, user_prompt, ...) — one-shot self-critique
  - build_universe_brief(tickers, ...) — markdown summary for --data-aware
  - GenerationResult — return type for both above

Provider selection:
  - Explicit `provider="anthropic"|"openai"` keyword arg (highest priority)
  - LAB_LLM_PROVIDER env var
  - Auto-detect from which API key is set
  - If both set: default to anthropic (preserves original behavior)
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from lab.llm.provider import (
    LLMMessage,
    LLMProvider,
    LLMResponse,
    LLMToolCall,
    LLMToolSpec,
)
from lab.llm.selection import get_provider

logger = logging.getLogger(__name__)

# Backward-compat constants — kept at the module surface so any external code
# (`from lab.llm import MODEL`) continues to work. MODEL is the Anthropic
# default; the per-provider default lives on each provider class.
MODEL = "claude-opus-4-7"
SYSTEM_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "system.md"

DEFAULT_UNIVERSE = ["SPY", "TLT"]
DEFAULT_TRAIN_END = "2015-01-01"
DEFAULT_REBALANCE_FREQ = 21


@dataclass
class GenerationResult:
    code: str
    raw_response: str
    universe: list[str]
    train_end: str
    rebalance_freq: int
    usage: dict | None
    attempts: int = 1
    retry_reasons: list[str] | None = None
    provider: str = "anthropic"   # which backend produced this result


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #


def load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text()


def generate_strategy(
    prompt: str,
    *,
    model: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.2,
    api_key: str | None = None,
    validate: bool = True,
    max_retries: int = 2,
    universe_brief: str | None = None,
    provider: str | None = None,
) -> GenerationResult:
    """Generate + (optionally) validate a Strategy subclass from natural language.

    `provider` may be 'anthropic' or 'openai'. If None, resolves via
    LAB_LLM_PROVIDER env var or auto-detects from available API keys.
    """
    p = get_provider(provider, api_key=api_key)
    system_prompt = load_system_prompt()

    user_content = prompt
    if universe_brief:
        user_content = (
            "Strategy request:\n\n" + prompt + "\n\n" +
            "Here is recent context on the universe you'll be trading:\n\n" +
            universe_brief
        )
    messages: list[LLMMessage] = [LLMMessage(role="user", content=user_content)]
    retry_reasons: list[str] = []
    usage_total: dict = {
        "input_tokens": 0, "output_tokens": 0,
        "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
        "cache_read_tokens": 0,
    }

    for attempt in range(1, max_retries + 2):
        logger.info("calling %s/%s (attempt %d/%d)",
                    p.name, model or p.default_model, attempt, max_retries + 1)
        response = p.generate(
            system=system_prompt, messages=messages, model=model,
            max_tokens=max_tokens, temperature=temperature,
        )
        raw = response.text
        for k, v in response.usage.items():
            usage_total[k] = usage_total.get(k, 0) + (v or 0)

        code = _extract_python_block(raw)
        universe, train_end, rebalance_freq = _parse_defaults(code)

        if not validate:
            return _build_result(p, code, raw, universe, train_end,
                                  rebalance_freq, usage_total, attempt, retry_reasons)

        problem = _validate_generated_code(code)
        if problem is None:
            return _build_result(p, code, raw, universe, train_end,
                                  rebalance_freq, usage_total, attempt, retry_reasons)

        retry_reasons.append(problem)
        logger.warning("validation failed on attempt %d: %s", attempt, problem)
        if attempt > max_retries:
            break

        # Build retry conversation.
        messages.append(LLMMessage(role="assistant", content=raw))
        messages.append(LLMMessage(role="user", content=(
            f"The strategy you wrote failed validation:\n\n{problem}\n\n"
            "Please fix the issue and emit ONLY the corrected Python code "
            "inside a single ```python ... ``` block, no prose. "
            "Keep the same overall intent."
        )))

    raise RuntimeError(
        f"strategy generation failed validation after {max_retries + 1} attempts. "
        f"Reasons: {retry_reasons}"
    )


def refine_strategy(
    prior_code: str,
    prior_metrics: dict,
    user_prompt: str,
    *,
    model: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.2,
    api_key: str | None = None,
    provider: str | None = None,
) -> GenerationResult:
    """One-shot self-critique. Hand the model its prior code + observed
    metrics, ask if anything looks off, return either the same code or a fix."""
    p = get_provider(provider, api_key=api_key)
    system_prompt = load_system_prompt()

    metrics_summary = "\n".join(
        f"  {k}: {v:.4f}" if isinstance(v, (int, float)) else f"  {k}: {v}"
        for k, v in prior_metrics.items()
    )
    refinement_prompt = (
        f"Original user request:\n\n{user_prompt}\n\n"
        f"You wrote this strategy:\n\n```python\n{prior_code}\n```\n\n"
        f"A dry backtest produced these metrics:\n\n{metrics_summary}\n\n"
        "Review the code AND the metrics. If the strategy is doing what the "
        "user asked AND the metrics look reasonable, return the SAME code "
        "unchanged in a python code block. If you spot a bug (wrong sign, "
        "missing edge case, suspicious all-zero weights, etc.), return the "
        "corrected version. Either way: code only, no prose."
    )
    response = p.generate(
        system=system_prompt,
        messages=[LLMMessage(role="user", content=refinement_prompt)],
        model=model, max_tokens=max_tokens, temperature=temperature,
    )
    code = _extract_python_block(response.text)
    universe, train_end, rebalance_freq = _parse_defaults(code)
    return GenerationResult(
        code=code, raw_response=response.text,
        universe=universe, train_end=train_end, rebalance_freq=rebalance_freq,
        usage=response.usage, attempts=1,
        retry_reasons=["refinement pass"], provider=p.name,
    )


def build_universe_brief(
    tickers: list[str],
    *,
    end_date: str | None = None,
    sample_days: int = 30,
) -> str:
    """30-day market-context brief for the LLM prompt. Provider-agnostic."""
    from lab.data import load_universe

    try:
        bundle = load_universe(tickers)
    except Exception as e:
        return f"_universe brief unavailable: {type(e).__name__}: {e}_"
    rets = bundle.returns
    if end_date is not None:
        rets = rets.loc[rets.index <= end_date]
    window = rets.iloc[-sample_days:] if len(rets) >= sample_days else rets
    if window.empty:
        return "_universe brief unavailable: no data in window_"

    lines: list[str] = []
    lines.append(f"### Universe data brief (last {len(window)} bars, "
                 f"through {window.index.max().date()})")
    lines.append("")
    lines.append("Per-asset summary:")
    lines.append("")
    lines.append("| ticker | last_close | ret_1d | ret_5d | ret_21d | ann_vol |")
    lines.append("|--------|-----------|--------|--------|---------|---------|")
    prices = bundle.prices.loc[window.index]
    for t in tickers:
        last = prices[t].iloc[-1]
        r1 = window[t].iloc[-1]
        r5 = window[t].iloc[-5:].sum() if len(window) >= 5 else float("nan")
        r21 = window[t].sum()
        vol = window[t].std() * (252 ** 0.5)
        lines.append(
            f"| {t} | {last:.2f} | {r1*100:+.2f}% | {r5*100:+.2f}% | "
            f"{r21*100:+.2f}% | {vol*100:.1f}% |"
        )
    lines.append("")
    if len(tickers) >= 2 and len(window) >= 10:
        corr = window.corr().round(2)
        lines.append("Correlation matrix:")
        lines.append("")
        header = "| | " + " | ".join(tickers) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(tickers) + 1))
        for t in tickers:
            row = f"| **{t}** | " + " | ".join(f"{corr.loc[t, c]:+.2f}" for c in tickers) + " |"
            lines.append(row)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Validation pipeline (shared across providers)
# --------------------------------------------------------------------------- #


def _ruff_check(code: str) -> str | None:
    """Run ruff with F821 + E9 rules. Silent no-op if ruff missing."""
    import shutil
    import subprocess

    ruff = shutil.which("ruff")
    if not ruff:
        return None
    try:
        result = subprocess.run(
            [ruff, "check", "--select", "F821,E9", "--quiet",
             "--output-format", "concise", "-"],
            input=code, capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.warning("ruff invocation failed: %s", e)
        return None
    if result.returncode == 0:
        return None
    msg = result.stdout.strip() or result.stderr.strip()
    lines = msg.splitlines()
    if len(lines) > 5:
        lines = lines[:5] + [f"... ({len(msg.splitlines()) - 5} more)"]
    return "lint errors:\n" + "\n".join(lines)


def _validate_generated_code(code: str) -> str | None:
    """Returns None if the code passes; else a short problem string."""
    try:
        compile(code, "<generated_strategy>", "exec")
    except SyntaxError as e:
        return f"SyntaxError: {e.msg} (line {e.lineno})"

    lint_problem = _ruff_check(code)
    if lint_problem:
        return lint_problem

    try:
        from lab.runner import execute_strategy_code
        cls = execute_strategy_code(code)
    except Exception as e:
        return f"{type(e).__name__}: {e}"

    try:
        import numpy as np
        import pandas as pd
        idx = pd.bdate_range("2020-01-06", periods=300)
        rng = np.random.default_rng(0)
        synthetic = pd.DataFrame(
            rng.normal(0.0003, 0.01, size=(300, 3)),
            index=idx, columns=["SPY", "TLT", "GLD"],
        )
        strat = cls()
        strat.fit(synthetic.iloc[:200])
        out = strat.rebalance(idx[200], synthetic.iloc[:200])
        if not isinstance(out, pd.Series):
            return f"rebalance returned {type(out).__name__}, expected pd.Series"
        if not all(np.isfinite(out.dropna().to_numpy())):
            return "rebalance returned non-finite weights"
    except Exception as e:
        return f"runtime error during smoke test: {type(e).__name__}: {e}"

    return None


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #


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
    """Pull universe/train_end/rebalance_freq from the top-of-file comment."""
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


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _build_result(provider, code, raw, universe, train_end, rebalance_freq,
                  usage, attempt, retry_reasons) -> GenerationResult:
    return GenerationResult(
        code=code, raw_response=raw, universe=universe,
        train_end=train_end, rebalance_freq=rebalance_freq,
        usage=usage, attempts=attempt, retry_reasons=retry_reasons,
        provider=provider.name,
    )


# Backward-compat: some external code may import these. Keep them re-exported.
__all__ = [
    "GenerationResult",
    "MODEL",
    "DEFAULT_UNIVERSE",
    "DEFAULT_TRAIN_END",
    "DEFAULT_REBALANCE_FREQ",
    "SYSTEM_PROMPT_PATH",
    "generate_strategy",
    "refine_strategy",
    "build_universe_brief",
    "load_system_prompt",
    "get_provider",
    "LLMMessage",
    "LLMResponse",
    "LLMToolCall",
    "LLMToolSpec",
    "LLMProvider",
    # Private but tested by name in the test suite — keep importable.
    "_validate_generated_code",
    "_extract_python_block",
    "_parse_defaults",
    "_ruff_check",
]
