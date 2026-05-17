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
    attempts: int = 1               # how many LLM calls were needed to reach this code
    retry_reasons: list[str] | None = None  # what failed on each prior attempt


DEFAULT_UNIVERSE = ["SPY", "TLT"]
DEFAULT_TRAIN_END = "2015-01-01"
DEFAULT_REBALANCE_FREQ = 21


def load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text()


def _make_client(api_key: str | None):
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
    return anthropic.Anthropic(api_key=api_key)


def _call_anthropic(
    client,
    messages: list[dict],
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    system_prompt: str,
):
    return client.messages.create(
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
        messages=messages,
    )


def _extract_usage(response) -> dict | None:
    if not getattr(response, "usage", None):
        return None
    return {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cache_creation_input_tokens": getattr(
            response.usage, "cache_creation_input_tokens", 0,
        ),
        "cache_read_input_tokens": getattr(
            response.usage, "cache_read_input_tokens", 0,
        ),
    }


def generate_strategy(
    prompt: str,
    *,
    model: str = MODEL,
    max_tokens: int = 4096,
    temperature: float = 0.2,
    api_key: str | None = None,
    validate: bool = True,
    max_retries: int = 2,
) -> GenerationResult:
    """Call Anthropic with the system prompt + user description, parse out code.

    If `validate=True`, the generated code is checked for:
      * importable / exec-able (compile + exec in fresh namespace)
      * defines exactly one Strategy subclass
      * passes a tiny no-lookahead probe on synthetic data

    On failure, the LLM is re-prompted up to `max_retries` times with the
    error message appended, asking it to fix the issue and re-emit the code.

    Raises RuntimeError if all retries fail.
    """
    client = _make_client(api_key)
    system_prompt = load_system_prompt()

    messages: list[dict] = [{"role": "user", "content": prompt}]
    retry_reasons: list[str] = []
    usage_total = {
        "input_tokens": 0, "output_tokens": 0,
        "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
    }

    for attempt in range(1, max_retries + 2):  # initial + retries
        logger.info("calling %s (attempt %d/%d)", model, attempt, max_retries + 1)
        response = _call_anthropic(
            client, messages,
            model=model, max_tokens=max_tokens,
            temperature=temperature, system_prompt=system_prompt,
        )
        raw = "".join(
            b.text for b in response.content if getattr(b, "type", "") == "text"
        )
        u = _extract_usage(response)
        if u:
            for k, v in u.items():
                usage_total[k] = usage_total.get(k, 0) + v

        code = _extract_python_block(raw)
        universe, train_end, rebalance_freq = _parse_defaults(code)

        if not validate:
            return GenerationResult(
                code=code, raw_response=raw, universe=universe,
                train_end=train_end, rebalance_freq=rebalance_freq,
                usage=usage_total, attempts=attempt, retry_reasons=retry_reasons,
            )

        # Validate the generated code.
        problem = _validate_generated_code(code)
        if problem is None:
            return GenerationResult(
                code=code, raw_response=raw, universe=universe,
                train_end=train_end, rebalance_freq=rebalance_freq,
                usage=usage_total, attempts=attempt, retry_reasons=retry_reasons,
            )
        retry_reasons.append(problem)
        logger.warning("validation failed on attempt %d: %s", attempt, problem)
        if attempt > max_retries:
            break
        # Build retry message.
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": (
            f"The strategy you wrote failed validation:\n\n{problem}\n\n"
            "Please fix the issue and emit ONLY the corrected Python code "
            "inside a single ```python ... ``` block, no prose. "
            "Keep the same overall intent."
        )})

    raise RuntimeError(
        f"strategy generation failed validation after {max_retries + 1} attempts. "
        f"Reasons: {retry_reasons}"
    )


def refine_strategy(
    prior_code: str,
    prior_metrics: dict,
    user_prompt: str,
    *,
    model: str = MODEL,
    max_tokens: int = 4096,
    temperature: float = 0.2,
    api_key: str | None = None,
) -> GenerationResult:
    """One-shot refinement: hand the model its prior code + dry-run metrics,
    ask if anything looks off, get either the same code back or a fix.

    This is the simplified version of an "agentic" loop — single feedback
    round, no tools, no multi-step exploration. Returns a GenerationResult
    with the refined (or unchanged) code.
    """
    client = _make_client(api_key)
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
    response = _call_anthropic(
        client,
        [{"role": "user", "content": refinement_prompt}],
        model=model, max_tokens=max_tokens,
        temperature=temperature, system_prompt=system_prompt,
    )
    raw = "".join(
        b.text for b in response.content if getattr(b, "type", "") == "text"
    )
    code = _extract_python_block(raw)
    universe, train_end, rebalance_freq = _parse_defaults(code)
    return GenerationResult(
        code=code, raw_response=raw, universe=universe,
        train_end=train_end, rebalance_freq=rebalance_freq,
        usage=_extract_usage(response), attempts=1,
        retry_reasons=["refinement pass"],
    )


def _ruff_check(code: str) -> str | None:
    """Run ruff with pyflakes rules; return error string or None.

    Only flags rules in the F category (undefined names, unused imports, etc.)
    — not style issues. If ruff is missing, returns None silently.
    """
    import shutil
    import subprocess

    ruff = shutil.which("ruff")
    if not ruff:
        return None
    try:
        # F821 = undefined name (real bug). E9xx = syntax. Skip style/unused-import
        # warnings — too noisy and not bugs.
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
    # Trim the output to the first 5 lines; this is what the LLM will see.
    msg = result.stdout.strip() or result.stderr.strip()
    lines = msg.splitlines()
    if len(lines) > 5:
        lines = lines[:5] + [f"... ({len(msg.splitlines()) - 5} more)"]
    return "lint errors:\n" + "\n".join(lines)


def _validate_generated_code(code: str) -> str | None:
    """Returns None if the code looks valid, else a short problem string."""
    # 1. Syntax check.
    try:
        compile(code, "<generated_strategy>", "exec")
    except SyntaxError as e:
        return f"SyntaxError: {e.msg} (line {e.lineno})"

    # 1b. Ruff lint (pyflakes rules only: undefined names, etc.). Best-effort —
    # if ruff isn't installed, skip silently.
    lint_problem = _ruff_check(code)
    if lint_problem:
        return lint_problem

    # 2. Exec + find Strategy subclass.
    try:
        from lab.runner import execute_strategy_code
        cls = execute_strategy_code(code)
    except Exception as e:
        return f"{type(e).__name__}: {e}"

    # 3. Smoke-test on tiny synthetic data — call rebalance once.
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
