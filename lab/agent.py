"""Multi-tool agent loop for strategy generation.

The simpler `generate_strategy()` path generates code in one shot then
optionally `refine_strategy()` does a single self-critique. This module
goes further: it gives the LLM a small set of tools so it can iterate
toward a working, evaluated strategy before submitting.

Tools available to the model:

  - run_dry_backtest(code: str) -> metrics dict
      Quickly backtests on a small synthetic universe to surface bugs
      (NaN weights, all-zero strategies, etc.) without spending real
      data-load time.

  - fetch_history(ticker: str, lookback_days: int) -> summary
      Returns recent OHLC-ish summary for one ticker so the model can
      check liquidity / volatility / autocorrelation.

  - compute_metric(equity: list[float], metric: str) -> float
      Compute a named metric (sharpe, cagr, sortino, max_drawdown)
      from an arbitrary equity series.

  - search_examples(pattern: str) -> str
      Returns relevant snippets from the reference strategies that
      match the pattern (e.g., "momentum", "vol target").

Loop:
  1. Send the prompt + tools.
  2. Model either emits a tool_use block or a `FINAL` code block.
  3. Execute the tool, return result, continue.
  4. Halt on FINAL sentinel, max iterations, or token budget.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lab.llm import (
    GenerationResult, _call_anthropic, _extract_python_block, _extract_usage,
    _make_client, _parse_defaults, load_system_prompt,
)

logger = logging.getLogger(__name__)


TOOL_SCHEMAS: list[dict] = [
    {
        "name": "run_dry_backtest",
        "description": (
            "Run a quick backtest of a candidate strategy on synthetic data "
            "(3 tickers, 400 days of random returns with mild drift). Returns "
            "the metrics dict (sharpe, cagr, max_drawdown, etc.) or an error "
            "string if the code fails. Useful for catching bugs (NaN weights, "
            "always-zero output, divide-by-zero) before submitting the final."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Full Python source for a Strategy subclass.",
                },
            },
            "required": ["code"],
        },
    },
    {
        "name": "fetch_history",
        "description": (
            "Return recent summary statistics for one ticker: last 30-day "
            "annualized vol, recent 1d / 5d / 21d returns, last close. Useful "
            "to check that the universe the user requested is reasonable."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "lookback_days": {"type": "integer", "default": 60},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "compute_metric",
        "description": (
            "Compute a named metric on an equity series. Available metrics: "
            "sharpe, sortino, cagr, max_drawdown, calmar, ann_vol."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "equity": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Sequence of NAV values, indexed by trading day.",
                },
                "metric": {"type": "string"},
            },
            "required": ["equity", "metric"],
        },
    },
    {
        "name": "search_examples",
        "description": (
            "Search the system-prompt example library for snippets matching "
            "a keyword pattern (e.g. 'momentum', 'vol target', 'mean rev'). "
            "Returns matching examples as a string."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
            },
            "required": ["pattern"],
        },
    },
]


# --------------------------------------------------------------------------- #
# Tool implementations (executed by us, not the model)
# --------------------------------------------------------------------------- #


def _tool_run_dry_backtest(code: str) -> dict | str:
    """Quick synthetic-data backtest for a candidate strategy."""
    try:
        from lab.backtest import BacktestConfig, walk_forward_backtest
        from lab.costs import ZeroCost
        from lab.metrics import compute_metrics
        from lab.runner import execute_strategy_code

        cls = execute_strategy_code(code)
        strat = cls()
        rng = np.random.default_rng(0)
        idx = pd.bdate_range("2010-01-04", periods=400)
        rets = pd.DataFrame(
            rng.normal(0.0003, 0.012, size=(400, 3)),
            index=idx, columns=["SPY", "TLT", "GLD"],
        )
        cfg = BacktestConfig(train_end=idx[150].strftime("%Y-%m-%d"),
                             rebalance_freq=21)
        res = walk_forward_backtest(strat, rets, cfg=cfg, cost_model=ZeroCost())
        m = compute_metrics(res.equity)
        return {
            "sharpe": m.get("sharpe"),
            "cagr": m.get("cagr"),
            "max_drawdown": m.get("max_drawdown"),
            "final_nav": m.get("final_nav"),
            "n_obs": m.get("n_obs"),
        }
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"


def _tool_fetch_history(ticker: str, lookback_days: int = 60) -> dict | str:
    try:
        from lab.data import load_universe

        bundle = load_universe([ticker])
        rets = bundle.returns[ticker].iloc[-lookback_days:]
        if rets.empty:
            return {"ticker": ticker, "error": "no data in window"}
        prices = bundle.prices[ticker].iloc[-lookback_days:]
        ann_vol = float(rets.std() * (252 ** 0.5))
        return {
            "ticker": ticker,
            "last_close": float(prices.iloc[-1]),
            "ret_1d": float(rets.iloc[-1]),
            "ret_5d": float(rets.iloc[-5:].sum()) if len(rets) >= 5 else None,
            "ret_21d": float(rets.iloc[-21:].sum()) if len(rets) >= 21 else None,
            "ann_vol": ann_vol,
            "n_obs": int(len(rets)),
        }
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"


def _tool_compute_metric(equity: list[float], metric: str) -> float | str:
    try:
        from lab.metrics import compute_metrics

        s = pd.Series(equity)
        if len(s) < 2:
            return "error: equity must have >=2 points"
        m = compute_metrics(s)
        v = m.get(metric)
        if v is None:
            return f"error: unknown metric '{metric}'"
        return float(v)
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"


def _tool_search_examples(pattern: str) -> str:
    sys_prompt = load_system_prompt()
    blocks = re.findall(r"### Example.*?```python\n(.*?)```",
                        sys_prompt, flags=re.DOTALL)
    hits = [b for b in blocks if pattern.lower() in b.lower()]
    if not hits:
        return f"no examples matched pattern {pattern!r}"
    return "\n\n---\n\n".join(hits[:3])


TOOL_IMPLS = {
    "run_dry_backtest": _tool_run_dry_backtest,
    "fetch_history": _tool_fetch_history,
    "compute_metric": _tool_compute_metric,
    "search_examples": _tool_search_examples,
}


# --------------------------------------------------------------------------- #
# Agent loop
# --------------------------------------------------------------------------- #

AGENT_SYSTEM_SUFFIX = """

## Agent mode instructions

You have access to tools (run_dry_backtest, fetch_history, compute_metric,
search_examples). Use them ONLY if you genuinely need them — most strategies
don't. When you're confident the strategy is correct and ready to backtest,
emit the FINAL Python code in a ```python ... ``` block with the literal
word `FINAL` at the very top as a comment, like:

```python
# FINAL
from lab.strategy import Strategy
...
```

Halt conditions: emit FINAL, or 5 tool-use rounds reached. Do not spend tool
calls exploring unrelated paths.
"""


def run_agent(
    prompt: str,
    *,
    model: str = "claude-opus-4-7",
    max_tokens: int = 4096,
    temperature: float = 0.2,
    api_key: str | None = None,
    max_iterations: int = 5,
) -> GenerationResult:
    """Run the tool-using agent until it emits FINAL or exhausts iterations.

    Returns the same GenerationResult shape as `generate_strategy`. The
    `attempts` field is the number of agent rounds taken; `retry_reasons`
    records what the agent did each round.
    """
    client = _make_client(api_key)
    system = load_system_prompt() + AGENT_SYSTEM_SUFFIX

    messages: list[dict] = [{"role": "user", "content": prompt}]
    log: list[str] = []
    usage_total = {"input_tokens": 0, "output_tokens": 0,
                   "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}

    for it in range(1, max_iterations + 1):
        logger.info("agent iteration %d/%d", it, max_iterations)
        response = client.messages.create(
            model=model, max_tokens=max_tokens, temperature=temperature,
            system=[{"type": "text", "text": system,
                     "cache_control": {"type": "ephemeral"}}],
            messages=messages, tools=TOOL_SCHEMAS,
        )
        u = _extract_usage(response)
        if u:
            for k, v in u.items():
                usage_total[k] = usage_total.get(k, 0) + v

        # Walk the response content blocks looking for tool_use and text.
        tool_calls: list[Any] = []
        text_parts: list[str] = []
        for block in response.content:
            btype = getattr(block, "type", "")
            if btype == "tool_use":
                tool_calls.append(block)
            elif btype == "text":
                text_parts.append(block.text)
        raw_text = "".join(text_parts)

        # Check for FINAL sentinel in the assistant's text.
        if "# FINAL" in raw_text or "#FINAL" in raw_text:
            code = _extract_python_block(raw_text)
            universe, train_end, rebalance_freq = _parse_defaults(code)
            log.append(f"iter {it}: FINAL emitted")
            return GenerationResult(
                code=code, raw_response=raw_text, universe=universe,
                train_end=train_end, rebalance_freq=rebalance_freq,
                usage=usage_total, attempts=it, retry_reasons=log,
            )

        # Append assistant message and process tool calls.
        messages.append({"role": "assistant", "content": response.content})
        if not tool_calls:
            # No tool call and no FINAL — model returned plain text. Treat as
            # final attempt and try to extract a code block; if there isn't
            # one, raise.
            code = _extract_python_block(raw_text)
            if "class " in code and "Strategy" in code:
                log.append(f"iter {it}: model emitted bare code (no FINAL marker)")
                universe, train_end, rebalance_freq = _parse_defaults(code)
                return GenerationResult(
                    code=code, raw_response=raw_text, universe=universe,
                    train_end=train_end, rebalance_freq=rebalance_freq,
                    usage=usage_total, attempts=it, retry_reasons=log,
                )
            log.append(f"iter {it}: model returned no tools and no code; nudging")
            messages.append({
                "role": "user",
                "content": "You didn't call any tools or emit code. Please "
                           "either call a tool to gather more info, or emit "
                           "FINAL with the strategy code.",
            })
            continue

        # Execute each tool call and append the result.
        tool_results: list[dict] = []
        for tc in tool_calls:
            name = tc.name
            input_args = dict(tc.input) if hasattr(tc, "input") else {}
            impl = TOOL_IMPLS.get(name)
            if impl is None:
                output = f"error: unknown tool {name!r}"
            else:
                try:
                    output = impl(**input_args)
                except TypeError as e:
                    output = f"error: bad tool args: {e}"
            log.append(f"iter {it}: tool {name}({input_args}) -> {str(output)[:80]}")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tc.id,
                "content": json.dumps(output, default=str) if not isinstance(output, str) else output,
            })
        messages.append({"role": "user", "content": tool_results})

    # Exhausted iterations without FINAL.
    raise RuntimeError(
        f"agent exhausted {max_iterations} iterations without emitting FINAL. "
        f"Log:\n  " + "\n  ".join(log)
    )
