# Strategy Lab — Improvement Set Tracker

Started 2026-05-17. This file is updated as I go through the 20 items agreed in the conversation. Each "shipped" item has its own commit. "Attempt" items ship with simplifications documented here. "Skipped" items have rationale.

| # | ID | Item | Class | Status | Commit | Notes |
|---|----|------|-------|--------|--------|-------|
| 1 | D-12 | HTML report per run | confident | pending | | matplotlib charts + self-contained HTML |
| 2 | A-2 | Auto lookahead validator + retry | confident | pending | | run validator before backtest, retry LLM if fails |
| 3 | F-20 | Ruff lint pre-exec | confident | pending | | catch syntax / undefined names before exec |
| 4 | F-19 | Per-run timeout | confident | pending | | SIGALRM, 60s default |
| 5 | D-13 | Plotting in `lab compare` | confident | pending | | overlaid equity curves PNG |
| 6 | B-7 | Auto-diff strategy code in compare | confident | pending | | difflib.unified_diff |
| 7 | C-8 | Block-bootstrap CIs | confident | pending | | Sharpe/CAGR/maxDD with mean ± 95% CI |
| 8 | B-6 | Strategy family tree | confident | pending | | parent_run_id; `lab tree`, `lab fork` |
| 9 | A-3 | Richer few-shot library | confident | pending | | vol-target, inverse-vol, z-score MR, regime |
| 10 | E-17 | Regime model as reference strategy | confident | pending | | wire `regime_model` as one of `lab/strategies` |
| 11 | A-1 | Tool-use agent loop (simplified) | attempt | pending | | single retry-with-error, not multi-tool |
| 12 | B-5 | `lab chat` REPL (minimal) | attempt | pending | | input loop, regen each turn |
| 13 | E-14 | Short + leverage support | attempt | pending | | long_only=False, max_leverage configurable |
| 14 | C-9 | Walk-forward HP sweep (opt-in) | attempt | pending | | `lab sweep` CLI command |
| 15 | F-18 | Sandboxed exec | attempt | pending | | subprocess + import denylist |
| 16 | A-4 | Pass universe sample to model | skipped | — | — | adds non-trivial latency/cost; need design call |
| 17 | C-10 | SPA / White's reality check | skipped | — | — | statistical correctness needs review |
| 18 | C-11 | Better cost model (bid-ask/slippage) | skipped | — | — | needs calibration-data choices |
| 19 | E-15 | Multi-frequency bars | skipped | — | — | architectural change to data layer |
| 20 | E-16 | Fundamentals data | skipped | — | — | vendor choice |

## Definitions

- **confident** — design space narrow; I'll just build it.
- **attempt** — bigger surface area; I'll ship a documented simplification.
- **skipped** — needs decisions I shouldn't make alone; rationale above.

## Constraints I'm operating under

- `ANTHROPIC_API_KEY` is not set in this shell — LLM-touching features are mock-tested only. Real LLM verification deferred to user.
- All existing tests must continue to pass. New tests per feature.
- Each shippable item gets its own commit.
- If I hit a genuine blocker, I write it up here and move on.

## Blocker log (if any)

(Populated as I go.)
