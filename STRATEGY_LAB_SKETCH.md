# Strategy Lab — Natural-Language Backtester

**Status:** Design sketch, not implemented. Captured 2026-05-17.
**Parent branch:** `Akshay-Joshi/main` at the regime-model state.

---

## Product idea

User types a strategy in natural language:

> "Hold 60% SPY, 40% TLT, rebalance monthly when drift exceeds 5%"

System: writes a `Strategy` subclass, runs it through the walk-forward backtester from the regime-model repo, returns metrics + equity curve + charts.

Then:

> "Now try the same with a 100-day MA filter on SPY"

System: edits the strategy in place (or versions it), reruns, shows side-by-side comparison.

---

## Why this is interesting

- **Plays directly to LLM strengths**: code generation, iteration, natural-language understanding.
- **Uses existing rigor as substrate**: the walk-forward loop, no-lookahead validator, transaction-cost modeling already exist in the regime-model code. This product wraps them with a natural-language frontend.
- **Genuine product gap**: ChatGPT can write strategy code, but doesn't actually run it through a serious backtester. "LLM-quant-bench" doesn't exist as a clean tool.
- **The rigor IS the differentiator**: walk-forward + lookahead validation makes the output trustworthy in a way that "Jupyter + GPT" isn't.

## Non-goals (v1)

- Not a trading product. Backtest only. No live broker, no PnL.
- Not multi-agent. Single LLM call → single Strategy → one backtest run.
- No fundamentals / options / alternatives. ETFs + equities, daily bars, yfinance.
- No portfolio of strategies. One strategy at a time.

---

## Design choices to nail down (these change the architecture)

### 1. Interface

| Option | Pros | Cons |
|--------|------|------|
| CLI: `lab backtest "buy SPY when..."` | Fast iteration, scriptable | Hard to show charts inline |
| Jupyter notebook | Native chart rendering, exploratory | Heavier setup, not the model's-best-fit for natural-language |
| Tiny web UI (FastAPI + HTML) | Best UX for charts + iteration | More code to maintain |
| Plain Python function | Trivial, composable | No user surface |

**Lean:** CLI first (smallest surface), with a `--show` flag that opens a results HTML page. Defer the web UI until the CLI is solid.

### 2. LLM access

| Option | Pros | Cons |
|--------|------|------|
| Anthropic API direct in-tool | Self-contained product | Requires API key plumbing |
| Run inside Claude Code (you prompt Claude Code, which calls the tool) | Zero new infra, leverages existing assistant | Tool is just a backtest function, not a product |
| MCP server | Reusable across clients | Spec overhead for v1 |

**Lean:** Run inside Claude Code for v0 (i.e., this assistant becomes the natural-language frontend, and the tool is just `run_strategy(code: str) -> Results`). API integration is a follow-up if it gains legs.

### 3. Strategy interface

The contract the LLM has to fit into. Two viable shapes:

```python
# Option A — per-bar weights (clean, uniform)
class Strategy(ABC):
    @abstractmethod
    def rebalance(self, date: pd.Timestamp, history: pd.DataFrame) -> pd.Series:
        """Return target weights indexed by ticker. History contains only data <= date - 1."""

    def fit(self, history: pd.DataFrame) -> None:
        """Optional one-time fit on training data."""
```

```python
# Option B — event-driven (closer to natural language)
class Strategy(ABC):
    @abstractmethod
    def on_bar(self, date, history, state) -> Action:
        """Action is BUY/SELL/HOLD with quantities; framework tracks positions."""
```

**Lean:** Option A. Per-bar target weights are simpler, deterministic, and natural-language strategies usually decompose to "what's the target portfolio today?" cleanly. Event-driven is more flexible but invites bugs and is harder for the LLM to generate correctly.

### 4. Code-gen trust model

- **Direct execution** (run LLM-generated Python in the same process): simple, fast, lets the user eyeball.
- **Sandboxed subprocess**: harder to weaponize but a lot more plumbing.

**Lean:** Direct execution for personal use. If this ever ships to others, sandboxing becomes mandatory.

### 5. Iteration UX

- New strategy each run, save versioned to `runs/2026-05-17-v1.py` etc.
- Each run produces a results bundle (metrics JSON, equity curve PNG, weights heatmap PNG)
- A `compare` command takes two run IDs and renders side-by-side

**Killer feature**: `lab compare v1 v3` showing "v1 Sharpe 0.7, max DD -18% / v3 Sharpe 0.9, max DD -11%" side by side. Sells the product.

---

## Architecture sketch

```
strategy-lab/
├── pyproject.toml
├── lab/
│   ├── __init__.py
│   ├── strategy.py           # ABC: Strategy.rebalance(date, history) -> weights
│   ├── backtest.py           # walk-forward loop (port from regime_model/allocation/backtest.py)
│   ├── data.py               # yfinance loader (port from regime_model/data/loaders.py)
│   ├── features.py           # standardized features + no-lookahead validator (port)
│   ├── costs.py              # FlatBpsPerLeg
│   ├── metrics.py            # Sharpe, Sortino, Calmar, max DD, turnover
│   ├── runner.py             # exec generated code, capture Strategy subclass, run backtest
│   ├── llm.py                # natural-language → strategy code (Anthropic API or Claude Code tool)
│   └── compare.py            # side-by-side metrics + plots
├── runs/                     # versioned strategy code + results per run
├── strategies/               # canonical reference strategies (60/40, momentum, etc.)
└── cli.py                    # entrypoint: `lab backtest "..."`, `lab compare a b`
```

## What gets ported from regime_model

| From | To | Notes |
|------|-----|------|
| `regime_model/data/loaders.py` | `lab/data.py` | Generalize ALLOCATION_TICKERS → user-config universe |
| `regime_model/data/features.py` | `lab/features.py` | Keep no-lookahead validator |
| `regime_model/allocation/backtest.py` | `lab/backtest.py` | Strip regime-specific year-loop body, drive any Strategy |
| `regime_model/allocation/baselines.py` | `lab/strategies/baselines.py` | 60/40, equal-weight |
| (nothing else) | | regime model stays in its own dir; it'd become one of many strategies |

---

## Open questions for the next session

1. CLI vs notebook vs web UI for v1
2. LLM access path: API call vs Claude Code as the frontend
3. Whether to keep the regime model wired in as the "complex example" strategy from day one
4. What "iteration UX" looks like in practice — is it a chat loop, or discrete CLI invocations?

---

## What this branch is NOT yet

- No code written
- No tests
- No CI
- Just the idea, parked

Next step when this is picked back up: invoke the brainstorming skill, lock down a spec, then writing-plans → implement.
