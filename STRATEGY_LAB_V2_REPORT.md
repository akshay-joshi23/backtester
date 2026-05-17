# Strategy Lab — v2 Improvement Session Report

**Branch:** `strategy-lab`
**Session date:** 2026-05-17 (continuation)
**Outcome:** All 10 remaining items shipped — 5 "skipped" built, 5 "attempt"
simplifications deepened to their proper versions. 142 tests passing
(86 lab + 56 regime_model).

---

## Why this session existed

The earlier session shipped 15/20 items in two buckets:
- 10 "confident" items in full
- 5 "attempt" items with documented simplifications
- 5 items "skipped" with rationale because they needed design / data /
  vendor decisions

The user explicitly delegated those decisions ("I trust your UI capabilities
and everything else"). This session converted all 10 remaining items into
shipped features. Each one's commit message documents the decisions I made.

---

## What shipped

### Five previously-skipped items now built

| # | ID | Commit | What | Decision I made |
|---|----|--------|------|------------------|
| 16 | A-4 | `bed4628` | `--data-aware`: 30-day universe sample + correlation matrix injected as markdown into the LLM prompt | Send 30 daily bars of stats (mean, vol, 1d/5d/21d returns) + pairwise corr matrix. Capped at ~3kB so we don't blow the cache. |
| 17 | C-10 | `4f0d2de` | Hansen 2005 SPA test (`lab spa <bm> <alt1> <alt2> ...`) | Used Hansen's "SPA-c" recentering, stationary block bootstrap (Politis-Romano), HAC variance estimator with Bartlett weights, default block length T^(1/3). |
| 18 | C-11 | `b309413` | `BidAskSpread` + `SquareRootImpact` + `realistic_cost_model()` builder | Per-asset bps dictionaries with hardcoded conservative defaults for major US ETFs (1-2bps for SPY/QQQ, 4-5bps for FX, 10bps fallback). Square-root impact calibrated so coef=10bps ≈ 5bps on a 25% leg. |
| 19 | E-15 | `91856ea` | Daily / weekly / monthly bars (`--frequency D|W|M`) | Resample from daily prices using W-FRI / month-end. Auto-adjust annualization (D=252, W=52, M=12). No intraday (different data source). |
| 20 | E-16 | `fba6ec4` | `lab/fundamentals.py` with `FundamentalsSnapshot` + universe loader | Use yfinance `Ticker.info` for current snapshot. Honestly document that this is NOT point-in-time and link to Sharadar/Norgate for serious work. |

### Five "attempt" simplifications deepened

| # | ID | v1 simplified | v2 deepened | v2 commit |
|---|----|---------------|-------------|-----------|
| 11 | A-1 | `--refine` (one critique round) | `--agent`: real multi-tool loop with `run_dry_backtest` / `fetch_history` / `compute_metric` / `search_examples`, FINAL sentinel, 5 max iterations | `f03f963` |
| 12 | B-5 | Plain `input()` REPL | prompt_toolkit history/completion + rich-formatted output (colored metric tables, syntax-highlighted code) + `:save` / `:load` named sessions + `:universe` command | `9877753` |
| 13 | E-14 | `--long-short` flag (no borrow modeled) | `BorrowCost` model with per-asset bps table + `CompositeCostModel` chaining + per-period `holding_cost` charged in backtest loop | `6c0ca8b` |
| 14 | C-9 | Grid sweep (`lab sweep`) | Real walk-forward HP selection: rolling tuning window, retune frequency, in-sample objective, stitched OOS equity, per-window winners (`lab sweep --walk-forward`) | `5efb2ce` |
| 15 | F-18 | AST audit + subprocess preflight | Docker container preflight: `--network none`, `--read-only`, memory/CPU caps, `python -I -S`, falls back gracefully if no Docker (`--container` flag) | `a10eb28` |

---

## Decisions I made on your behalf (where I had latitude)

These are the personal-choice items. None are load-bearing — they're all
overridable, and the documentation explicitly flags where they are
approximations. We'll polish these when you come back.

| Choice | What I picked | Why this default |
|--------|---------------|------------------|
| `--data-aware` brief format | Markdown table with mean/vol/returns + corr matrix | LLM-friendly, ~2kB, includes the most decision-relevant info |
| LLM agent tool inventory | `run_dry_backtest`, `fetch_history`, `compute_metric`, `search_examples` | Smallest set that covers "test my idea" / "check the universe" / "look up an example" |
| Agent halt condition | `# FINAL` marker in code block, max 5 iterations | Clear, sentinel-based; bounded cost |
| Borrow rates default | 50bps for major ETFs (SPY, QQQ, etc.); 75-100bps for ETFs like TLT/GLD; 300bps fallback | Conservative — real IBKR borrow on SPY is <0.5% but better to err high in a backtest |
| Bid-ask defaults | 1-2bps for major ETFs, 4-5bps for FX, 10bps fallback | Matches median spreads on Interactive Brokers' liquid tier |
| SquareRootImpact coef | 10bps (≈5bps on a 25% leg) | Institutional rule-of-thumb for liquid ETFs |
| Walk-forward tuning defaults | 3-year tuning window, retune yearly, 3-year initial train, optimize Sharpe | Standard practice; user can override |
| SPA defaults | 2000 bootstrap resamples, block_length = T^(1/3), seed=0 | Standard SPA-c specs from Hansen 2005 |
| `lab chat` libraries | `prompt_toolkit` + `rich` | Battle-tested REPL stack; widely available |
| Container image | `python:3.11-slim` | Smallest official Python that supports stdlib |
| Container caps | `--memory 512m --cpus 1.0` | Generous enough for any pure-Python strategy, restrictive enough to bound abuse |
| Fundamentals source | yfinance (free, ships with the project) | Free; honest about non-point-in-time limitation |

---

## Two honest caveats

1. **None of the LLM-touching paths have been verified end-to-end against
   a real API key.** That includes `--agent`, `--refine`, `--data-aware`,
   the chat REPL's streaming/saving paths, and `lab spa` (the SPA itself
   is fully tested with synthetic data — only the LLM-driven wrapper
   paths are unverified). All have parsing/validation tests with mocked
   responses. Verifying these with your real key takes maybe 15 minutes.

2. **Container sandbox needs Docker.** I tested the no-Docker-found path
   (clean error message) and the audit-runs-first path (synthetic
   blocklisted code is rejected without invoking Docker). The actual
   container execution path is not exercised in CI because we don't have
   a Docker daemon in this session — when you run it locally with Docker
   available, the JSON contract on stdin/stdout is straightforward, but
   first-time invocations may surface issues I haven't seen.

---

## Stats

- **17 new commits** since v1 baseline (`43beb19`)
- **86 lab tests** + **56 regime_model tests** = **142 total, all green**
- **+30 new tests** during this session
- New modules: `lab/agent.py`, `lab/spa.py`, `lab/fundamentals.py`
- Rewritten: `lab/chat.py` (rich+prompt_toolkit), `lab/costs.py` (composite model)
- Updated CLI flags: `--data-aware`, `--agent`, `--frequency`, `--realistic-costs`, `--container`, `--long-short` (already shipped), `--max-leverage` (already shipped)
- New CLI commands: `lab spa`, `lab sweep --walk-forward`

---

## End-to-end recipe

```bash
cd /Users/akshayjoshi/.superset/worktrees/backtester/Akshay-Joshi/main
source .venv/bin/activate

# === Without API key — verify framework still works ===
python -m lab.cli run-reference 60_40 --universe SPY TLT
python -m lab.cli run-reference 60_40 --universe SPY TLT --frequency M  # monthly bars
python -m lab.cli run-reference long_short_momentum --universe SPY QQQ IWM TLT GLD
python -m lab.cli show <run_id> --open                                  # HTML report
python -m lab.cli show <run_id> --bootstrap                              # block-bootstrap CIs
python -m lab.cli compare <a> <b> --plot --diff --open                   # side-by-side

# === Real-cost realism check ===
python -m lab.cli sweep <run_id> --param lookback --values 30,60,120,250 \
    --walk-forward --tuning-years 3 --retune-years 1

# === Reality check on multiple alternatives ===
python -m lab.cli spa <benchmark_run_id> <alt1> <alt2> <alt3>

# === With ANTHROPIC_API_KEY set ===
export ANTHROPIC_API_KEY=sk-ant-...

# Generate a strategy from natural language:
python -m lab.cli backtest "Equal-weight 5 ETFs with vol target 8%" \
    --universe SPY QQQ IWM TLT GLD \
    --realistic-costs                    # bid-ask + impact + borrow

# Data-aware prompt (sends 30d of stats to the LLM):
python -m lab.cli backtest "Find a strategy that works on this universe" \
    --universe SPY TLT GLD UUP HYG --data-aware

# Multi-tool agent that iterates with run_dry_backtest etc.:
python -m lab.cli backtest "Cross-asset momentum with risk parity overlay" \
    --universe SPY TLT GLD UUP HYG --agent

# Forked iteration:
python -m lab.cli fork <run_id> "now try with 100-day MA instead of 200"

# Interactive REPL with named sessions:
python -m lab.cli chat
# Inside the REPL:
#   :universe SPY TLT GLD
#   Equal weight with monthly rebalance
#   :save my_first_session
#   :show 1
#   :compare
#   :load my_first_session

# Container-isolated backtest (requires Docker):
python -m lab.cli backtest "..." --container
```

---

## What's still genuinely on the table for v3

- **Streaming output in `lab chat`.** I built rich formatting but kept the
  per-turn LLM call non-streaming because the Anthropic SDK tool-use
  protocol makes streaming tool calls non-trivial. Worth doing if you
  spend a lot of time staring at the REPL.
- **Full walk-forward in the agent.** The agent tools use synthetic
  data for `run_dry_backtest`. A v3 could let it dry-run on the actual
  user universe in a controlled subset.
- **Hosted distribution.** Web UI / cloud-hosted version. Big shift in
  scope; only worth it if you decide the product is the goal.
- **Point-in-time fundamentals.** Requires picking a paid vendor.
- **Multi-agent search (e.g. 5 LLMs propose strategies, SPA test picks
  winner).** Natural extension once `lab spa` and `--agent` work.
