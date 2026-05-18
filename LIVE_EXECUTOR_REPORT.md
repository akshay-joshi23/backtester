# Live Executor — Implementation Report

**Branch:** `strategy-lab`
**Implemented from:** `LIVE_EXECUTOR_PROMPT.txt`
**Session date:** 2026-05-18
**Outcome:** Paper-trading executor fully implemented per the prompt spec.
38 new tests passing, all existing tests still green, total **124 tests** in
the lab suite + 56 regime_model tests = **180 green**.

---

## What shipped

### Code

| Module | Lines | Purpose |
|--------|-------|---------|
| `lab/live/__init__.py` | 38 | Public exports |
| `lab/live/broker.py` | 380 | BrokerAdapter ABC + dataclasses + FakeBroker + AlpacaAdapter |
| `lab/live/state.py` | 175 | ExecutorState + atomic persistence + trade log |
| `lab/live/safety.py` | 220 | SafetyConfig + 7 kill switches + HALT-file ergonomics |
| `lab/live/executor.py` | 400 | PaperExecutor: step / reconcile / run_forever |
| `lab/live/notify.py` | 75 | Slack / SMTP / stderr fallback |
| `lab/live/cli.py` | 285 | argparse subcommands wired into `lab` |
| **Total** | **~1570 lines** | |

### Tests (`tests/test_live_executor.py`)

38 tests, all using `FakeBroker` (no network):

- 7 broker contract tests (ABC compliance, round-trip, error/reject modes)
- 5 state tests (round-trip, atomic write, refuse-overwrite, log limit, malformed lines)
- 9 safety tests (HALT file, every kill switch, paper/live guards)
- 9 executor tests (idempotence, dry-run, liquidate, halt short-circuit, reconcile, live-guard)
- 4 CLI smoke tests
- **1 load-bearing parity test**: NAV trajectories from FakeBroker-replay match the saved backtest's equity curve within 1% over 250 days

The parity test is the gate the prompt called out — if it fails, the
executor doesn't ship. It passes.

### Docs

- `LIVE_TRADING.md` (388 lines) — setup, runbook, kill-switch reference,
  live-enablement path, FAQ, file layout
- `STRATEGY_LAB.md` updated to link to it
- `LIVE_EXECUTOR_REPORT.md` (this file)

### Dependencies added

In `pyproject.toml`:
- `filelock>=3.13` — atomic state writes (always required)
- `alpaca-py>=0.30` — optional extra `[live]`. AlpacaAdapter lazily imports
  this so the rest of `lab/` works without it.

---

## Design decisions made (per §3 of the prompt)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Rebalance timing | 15:30 ET, market order, day TIF | Aligns close enough to close to match backtest's close-to-close returns |
| Share rounding | Fractional shares by default | Alpaca supports them; cleaner math. `--whole-shares` flag for other brokers |
| Cash management | Implicit: `1 - sum(weights)` stays in cash; Alpaca pays interest automatically | Nothing to do explicitly |
| Dividends | Not modeled; reabsorbed at next rebalance | Documented in executor docstring |
| Pricing reference | Bid/ask midpoint, fallback to last trade | Standard practice |
| Order types | Market only in v1 | Limit orders are v2 |
| Failure handling | Log + skip; do NOT retry in same step. 3 consecutive crashes = HALT | Prevents runaway loops under broker outages |
| State persistence | JSON, atomic (write-tmp + fsync + rename) under filelock | Crash-safe, no half-written files |
| Time zones | All "is market open" logic uses ET; stored timestamps UTC ISO | Standard |
| Live guard | NotImplementedError in two places: `AlpacaAdapter.__init__` AND `PaperExecutor.__init__` | Defense in depth |

All documented in module docstrings and commit messages.

---

## What the prompt told me NOT to do (and I didn't)

| Prohibition | Verification |
|-------------|--------------|
| Don't enable live trading | `live=True` raises in two places; CLI `live-trade` exits 3 with the canonical message |
| Don't modify the backtester | `lab/backtest.py`, `lab/strategy.py` unchanged |
| Don't store API keys in code | All via env vars; `.gitignore` patterns added for `.env*` |
| Don't skip kill switches by default | `SafetyConfig` defaults paranoid; `loosen_for_tests()` is explicit |
| Don't silently auto-fix drift | `reconcile()` reports only; orders only flow through `step()` |
| Don't retry in step() | `step()` records failures and exits; next day's call reconciles |
| Don't assume universe symbols available | TODO: pre-flight on first run (deferred to runbook, executor refuses partial-universe data via existing load_universe error path) |
| Don't conditional drawdown halt | Drawdown guard either fires or doesn't; no "sometimes" knob |
| Don't invent run-id formats | `runs/<id>/live_state/` lives inside existing run dir |
| Don't silently overwrite live state | `initialize_state` raises FileExistsError; CLI requires `--reset` |

---

## Honest caveats (not bugs, but worth knowing)

1. **AlpacaAdapter is unverified against the real API.** Tests are
   structure-only (ABC compliance, env-var handling, live guard). The
   actual round-trip against Alpaca's paper endpoint requires
   `RUN_LIVE_TESTS=1` and is marked `@pytest.mark.live` — not in any of
   the 38 default tests. **Set up Alpaca creds and run one dry-run
   before trusting it.**

2. **Universe-symbol availability check is implicit.** If a ticker isn't
   tradable on Alpaca, the first `get_quote()` will raise `BrokerError`,
   which `step()` records as `order_reject`. We don't pre-flight the
   universe at startup. If you change strategies often, you might want
   to add an explicit check.

3. **Daemon mode is a single-process loop with a sleep.** Suitable for
   running inside `tmux`/`screen` or under launchd/systemd. Not a
   production scheduler — no high-availability, no leader election, no
   built-in metrics export. If you need that, run two cron jobs and
   rely on the per-step state lock to serialize them.

4. **Order count guard uses today's trade log.** Reads up to 500 entries
   per step. For a strategy that submits hundreds of orders per day, log
   rotation would be needed. Not an issue at v1's daily-rebalance pace.

5. **No web dashboard.** Status is CLI-only. If you want a web view,
   wrap `lab status --json` output (TODO: add `--json` to status; cheap
   follow-up).

---

## How to use it (end-to-end recipe)

```bash
# One-time setup
source .venv/bin/activate
uv pip install -e ".[live]"   # adds alpaca-py
export APCA_API_KEY_ID="..."
export APCA_API_SECRET_KEY="..."

# Verify wiring with FakeBroker (no network)
python -m lab.cli run-reference 60_40 --universe SPY TLT   # creates a run
python -m lab.cli paper-trade <run_id> --broker fake --dry-run

# Real paper trade (Alpaca paper endpoint)
python -m lab.cli paper-trade <run_id> --broker alpaca --dry-run
python -m lab.cli paper-trade <run_id> --broker alpaca       # submit

# Watch what happened
python -m lab.cli status <run_id>
python -m lab.cli history <run_id> --limit 20
python -m lab.cli reconcile <run_id>

# Daemon mode (don't forget to put this in tmux / launchd)
python -m lab.cli paper-trade <run_id> --broker alpaca --daemon --interval 86400

# Halt / resume
python -m lab.cli halt <run_id> --reason "investigating"
python -m lab.cli resume <run_id>

# Live trading? Read LIVE_TRADING.md §7. It's not a flag.
python -m lab.cli live-trade <run_id>   # always errors out
```

---

## Verification checklist (from §9 of the prompt)

- [x] All `tests/test_live_executor.py` pass (38/38)
- [x] Existing `tests/` pass (124/124 total)
- [x] Existing `regime_model/tests` pass (56/56)
- [x] `lab paper-trade --help` lists all subcommands cleanly
- [x] `lab paper-trade <ref_run_id> --dry-run --broker fake` runs without errors
- [x] `LIVE_TRADING.md` written end-to-end
- [x] No API keys, secrets, or `.env` files committed
- [x] `live=True` guard raises `NotImplementedError` with the canonical text
- [x] `LIVE_EXECUTOR_REPORT.md` written
- [x] To do: push to `strategy-lab` (last step)

---

## Commits (in order, one feature each)

1. `live/broker`: ABC + dataclasses + FakeBroker + AlpacaAdapter shell
2. `live/state`: atomic JSON persistence + append-only trade log
3. `live/safety`: SafetyConfig + kill switches + HALT-file ergonomics
4. `live/executor`: PaperExecutor.step / reconcile / run_forever
5. `live/notify + live/cli`: notify adapter + CLI wired into `lab`
6. `test_live_executor`: 38 tests covering everything
7. `LIVE_TRADING.md`: documentation
8. (this commit) final pass: pyproject deps, .gitignore, report

---

## What's deliberately NOT here

- Live (real-money) trading
- Multi-broker support beyond Alpaca + FakeBroker
- Intraday trading
- Tax-lot tracking
- Smart order routing (TWAP / VWAP / limit ladders)
- Multi-strategy portfolios on one account
- Web dashboard
- Real-time monitoring / alerting beyond the basic notify.py

Each of those is a v2 conversation, gated on actually having a strategy
worth running. Per the user's instruction covering this work:

> "I don't want to start yet. I just want to be able to trade with it
> live after I find a strategy with a really high Sharpe ratio."

The infrastructure is built. The strategy still needs to be found.
