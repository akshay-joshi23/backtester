# Strategy Lab — Improvement Set Tracker

Started 2026-05-17. This file is updated as I go through the 20 items agreed in the conversation. Each "shipped" item has its own commit. "Attempt" items ship with simplifications documented here. "Skipped" items have rationale.

| # | ID | Item | Class | Status | Commit | Notes |
|---|----|------|-------|--------|--------|-------|
| 1 | D-12 | HTML report per run | confident | **DONE** | e314619 | matplotlib charts + self-contained HTML |
| 2 | A-2 | Auto-validate generated strategy + LLM retry | confident | **DONE** | 1c61877 | compile / class-count / smoke-test / Series-type check, with feedback retry loop |
| 3 | F-20 | Ruff lint pre-exec | confident | **DONE** | afa2dcf | F821 + E9 only; ruff missing is silent skip |
| 4 | F-19 | Per-run timeout | confident | **DONE** | 979f04a | SIGALRM/setitimer, 120s default, --timeout CLI flag |
| 5 | D-13 | Plotting in `lab compare` | confident | **DONE** | 0c49a60 | --plot flag, renders overlaid equity + drawdown PNG |
| 6 | B-7 | Auto-diff strategy code in compare | confident | **DONE** | 0c49a60 | --diff flag, unified_diff between consecutive runs |
| 7 | C-8 | Block-bootstrap CIs | confident | **DONE** | dd5d6f6 | metrics.block_bootstrap_metrics; `lab show --bootstrap` |
| 8 | B-6 | Strategy family tree | confident | **DONE** | c88bc81 | `lab tree`, `lab fork`, --parent flag, ASCII tree |
| 9 | A-3 | Richer few-shot library | confident | **DONE** | c04e4fc | 7 examples now (added vol-target, inv-vol, z-score MR, SMA cross) + gotchas |
| 10 | E-17 | Regime model as reference strategy | confident | **DONE** | b46b892 | `bayesian_regime` reference; fit-once + per-rebalance smoother |
| 11 | A-1 | Tool-use agent loop (simplified) | attempt | **DONE (simplified)** | ae6bad0 | shipped as `--refine` flag = one self-critique pass with metrics in context; multi-tool agent NOT built (see commit msg) |
| 12 | B-5 | `lab chat` REPL (minimal) | attempt | **DONE (simplified)** | 89e7456 | input loop, :help/:runs/:show/:compare/:reset/:exit; no streaming/rich UI |
| 13 | E-14 | Short + leverage support | attempt | **DONE (simplified)** | 58c6213 | --long-short, --max-leverage CLI; LongShortMomentum ref; NO borrow costs (documented in commit) |
| 14 | C-9 | Walk-forward HP sweep (opt-in) | attempt | **DONE (simplified)** | e682b65 | `lab sweep`; grid sweep, NOT full walk-forward HP selection (documented) |
| 15 | F-18 | Sandboxed exec | attempt | **DONE (simplified)** | 965d509 | AST audit + opt-in `python -I` subprocess preflight; honest "not a security boundary" caveat |
| 16 | A-4 | Pass universe sample to model | ~~skipped~~ | **DONE (v2)** | bed4628 | `--data-aware`: 30d OHLC + corr matrix in prompt |
| 17 | C-10 | SPA / White's reality check | ~~skipped~~ | **DONE (v2)** | 4f0d2de | Hansen 2005 SPA-c; `lab spa benchmark alt1 alt2 ...` |
| 18 | C-11 | Better cost model (bid-ask/slippage) | ~~skipped~~ | **DONE (v2)** | b309413 | BidAskSpread + SquareRootImpact + realistic_cost_model; `--realistic-costs` |
| 19 | E-15 | Multi-frequency bars | ~~skipped~~ | **DONE (v2)** | 91856ea | D/W/M via resampling; auto ann_factor; `--frequency` |
| 20 | E-16 | Fundamentals data | ~~skipped~~ | **DONE (v2)** | fba6ec4 | yfinance current-snapshot; honest "NOT point-in-time" caveat |

## v2 deepening session (2026-05-17 continued)

The 5 "skipped" items got built. The 5 "attempt" items got their proper
versions in parallel.

| v2 ID | Item | v2 commit | Replaces / supplements |
|-------|------|-----------|------------------------|
| A-1 v2 | Multi-tool agent loop | f03f963 | supplements `--refine`; `--agent` flag |
| B-5 v2 | Rich `lab chat` UI | 9877753 | replaces minimal input() loop; prompt_toolkit + rich + named sessions |
| E-14 v2 | Borrow-cost model | 6c0ca8b | supplements `--long-short`; `BorrowCost`, `CompositeCostModel`, holding-cost loop |
| C-9 v2 | Real walk-forward HP selection | 5efb2ce | supplements grid sweep; `lab sweep --walk-forward` |
| F-18 v2 | Docker container sandbox | a10eb28 | supplements AST audit; `--container` flag |

## Definitions

- **confident** — design space narrow; I'll just build it.
- **attempt** — bigger surface area; I'll ship a documented simplification.
- **skipped** — needs decisions I shouldn't make alone; rationale above.

## Constraints I'm operating under

- `ANTHROPIC_API_KEY` is not set in this shell — LLM-touching features are mock-tested only. Real LLM verification deferred to user.
- All existing tests must continue to pass. New tests per feature.
- Each shippable item gets its own commit.
- If I hit a genuine blocker, I write it up here and move on.

## Blocker log

No hard blockers. The only deviations are the five "skipped" items (16–20),
each deferred deliberately for reasons that are decision-level, not
implementation-level. Detailed rationale and proposed next steps are in
`STRATEGY_LAB_SESSION_REPORT.md`.

## End-of-session summary

Done: **15 / 20** items. Confident bucket complete. Attempt bucket complete
with documented simplifications (see commit messages and the session report).
Skipped bucket needs decisions before implementing.

Final commit count: 17 since v1 baseline. Net test delta: +32 tests, all green.
See `STRATEGY_LAB_SESSION_REPORT.md` for the full debrief.
