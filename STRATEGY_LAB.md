# Strategy Lab

Natural-language → walk-forward backtest. Type a strategy in plain English; Claude writes the code; the framework runs it; you get Sharpe, drawdown, equity curve, and the source code side by side.

## Quick start

```bash
# 1. Install deps (uv recommended)
source .venv/bin/activate
uv pip install -e .

# 2. Set your Anthropic API key
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Sanity check the framework with no LLM call:
python -m lab.cli run-reference 60_40 --universe SPY TLT

# 4. Ask for a strategy in natural language:
python -m lab.cli backtest "Equal-weight SPY, QQQ, IWM with a 200-day SMA filter on each. Rebalance monthly."

# 5. See all runs:
python -m lab.cli list

# 6. Compare two runs side-by-side:
python -m lab.cli compare <run_id_a> <run_id_b>

# 7. Show one run's details (prompt, code, metrics):
python -m lab.cli show <run_id>
```

## How it works

```
natural-language prompt
        │
        ▼
lab.llm.generate_strategy   ───►  Anthropic API (Opus 4.7, prompt-cached system prompt)
        │                          • parses out the ```python code block
        │                          • reads `# Defaults: universe=..., train_end=...` comment
        ▼
lab.runner.execute_strategy_code
        │   exec() in fresh namespace, find unique Strategy subclass, instantiate
        ▼
lab.data.load_universe          yfinance + parquet cache
        │
        ▼
lab.backtest.walk_forward_backtest
        │   fit() once on training window
        │   walk daily, rebalance every N days
        │   charge transaction costs on |Δw|
        │   drift weights between rebalances
        ▼
lab.metrics.compute_metrics     Sharpe, Sortino, Calmar, max DD, CAGR, turnover
        │
        ▼
runs/<timestamp>/               prompt.txt, strategy.py, equity.csv, metrics.json, …
```

## Strategy interface (what the LLM has to write)

```python
from lab.strategy import Strategy
import pandas as pd

class MyStrategy(Strategy):
    name = "My strategy"

    def fit(self, history: pd.DataFrame) -> None:
        # Optional one-time training-window setup.
        pass

    def rebalance(self, date: pd.Timestamp, history: pd.DataFrame) -> pd.Series:
        # `history`: log returns indexed by date, strictly before `date` (no lookahead).
        # Return: pandas Series of target weights indexed by ticker.
        # Long-only by default, sum in [0, 1] (1 - sum is implicit cash).
        return pd.Series({"SPY": 0.6, "TLT": 0.4})
```

The system prompt in `lab/prompts/system.md` documents this interface plus three few-shot examples (buy-and-hold, MA-filtered EW, top-k momentum). Prompt caching keeps the system prompt cheap on every call.

## LLM provider — Anthropic or OpenAI

Strategy Lab speaks to both backends through a single provider abstraction.
By default it picks the one you have credentials for; if both, it defaults
to Anthropic.

```bash
# Use whichever key is set:
export ANTHROPIC_API_KEY="sk-ant-..."     # picks anthropic
# or
export OPENAI_API_KEY="sk-..."            # picks openai
# or both — picks anthropic, override per-command

# Explicit per command:
lab backtest "..." --provider openai
lab backtest "..." --provider anthropic --model claude-sonnet-4-6

# Session-wide default:
export LAB_LLM_PROVIDER=openai

# Inside `lab chat`:
:provider openai
:provider anthropic
:provider auto       # back to auto-detect
```

Per-provider defaults:

| Provider | Default model | Notes |
|---|---|---|
| anthropic | `claude-opus-4-7` | Explicit prompt caching; `--agent` tested most here |
| openai | `gpt-4o` | Automatic prompt caching (≥1024 token prompts); cheaper input/output |

Everything else (`--refine`, `--agent`, `--data-aware`, validation retries,
chat REPL, fork) works identically across both. Tool schemas, response
parsing, and tool-result messages are translated per-provider inside
`lab.llm.{anthropic,openai}_provider`.

## Live & paper trading

For executing a saved strategy against a real broker (paper-only in v1),
see **[LIVE_TRADING.md](LIVE_TRADING.md)** for the setup walkthrough,
operational runbook, kill-switch reference, and the explicit multi-step
path required before enabling real-money trading.

CLI quick reference:
```bash
lab paper-trade <run_id> --dry-run        # safe preview
lab paper-trade <run_id> --broker alpaca  # one real paper cycle
lab status <run_id>                       # state + positions + last events
lab halt <run_id> --reason "..."          # stop next cycle
lab resume <run_id>                       # un-halt
lab reconcile <run_id>                    # diff intended vs actual
lab history <run_id>                      # tail the trade log
```

## What's in the box

- `lab/strategy.py`     — Strategy ABC + sanity helpers
- `lab/data.py`         — generic yfinance loader, parquet cache, lookahead validator
- `lab/backtest.py`     — walk-forward engine (any Strategy, any universe)
- `lab/costs.py`        — `FlatBpsPerLeg`, `ZeroCost`
- `lab/metrics.py`      — Sharpe / Sortino / Calmar / max DD / CAGR / turnover
- `lab/llm.py`          — Anthropic call, response parsing
- `lab/runner.py`       — code execution, backtest orchestration, run persistence
- `lab/cli.py`          — argparse CLI
- `lab/strategies/`     — built-in reference strategies (60/40, equal-weight, momentum)

## Tests

```bash
python -m pytest tests/test_strategy_lab.py -q
```

19 tests covering: backtest math (known-answer test against hand-computed equity), cost models, metrics, lookahead validator, reference strategies, edge cases (negative weights, over-leverage, wrong return types).

## What v1 *won't* do

- No live trading or broker integration
- No fundamentals, options, intraday, alt-data
- No multi-strategy portfolios
- No agentic iteration loop (just discrete CLI invocations)
- No sandboxed execution — generated code runs in the same process

All of those are reasonable v2+ extensions.

## Cost notes

Each `lab backtest` call is one Anthropic message. With prompt caching the system prompt (~3k tokens) is billed at the cached-read rate after the first call. Each call's expected cost is on the order of $0.05–$0.20 depending on response length.

## Iteration UX (the killer feature, when it's wired up)

For now: each run is its own dir, you `lab compare` them side-by-side. A future version will accept follow-ups ("now try the same with 100-day MA") and diff the generated code automatically.
