# Strategy Lab — Live & Paper Trading

This documents the `lab/live/` executor: how it works, how to set it up, how
to operate it safely, and the explicit multi-step path you must follow before
turning on real-money trading.

**Status: paper-only.** Real-money trading is guarded by a `NotImplementedError`
that requires editing source code to bypass — not a config flag, not a CLI
argument, not an env var. That's deliberate. See §7.

---

## 1. What this is

`lab/live/` reads a saved backtest run (a `runs/<id>/` directory) and trades
that strategy daily against a broker. It uses the **same Python code** for
strategy evaluation as the backtester, so behavior in paper trading matches
the backtest by construction (this property is verified by a load-bearing
parity test — see §6).

In scope:
- Daily rebalance via market orders against Alpaca's paper endpoint.
- Drift threshold to suppress noise trades.
- Persistent state on disk (`runs/<id>/live_state/`).
- Append-only trade log of every event.
- Kill switches: daily loss, drawdown, position size, order size, order count.
- Manual halt via `lab halt` (or `touch runs/<id>/HALT`).
- Dry-run mode that prints orders without submitting.
- CLI: `paper-trade`, `halt`, `resume`, `status`, `reconcile`, `history`.

Explicitly out of scope:
- Live (real-money) trading. Behind the canonical guard.
- Intraday trading. End-of-day rebalance only.
- Multi-broker support. Only Alpaca is implemented (FakeBroker for tests).
- Tax-lot tracking. Defer to broker's accounting.
- Smart order routing. Market orders for v1.
- Multi-strategy portfolios. One saved run = one paper account.

---

## 2. Setup

### Install dependencies

```bash
source .venv/bin/activate
uv pip install -e .
uv pip install alpaca-py
```

### Get Alpaca credentials

1. Create a free Alpaca account at https://alpaca.markets.
2. Go to the "Paper Trading" section. Generate API key + secret.
3. Export them:

```bash
export APCA_API_KEY_ID="..."
export APCA_API_SECRET_KEY="..."
# Optional — defaults to paper. Setting this to a non-paper URL is REFUSED.
# export APCA_BASE_URL="https://paper-api.alpaca.markets"
```

Never commit these to git. The repo's `.gitignore` already covers `.env` and
related patterns.

### Optional notifications

```bash
# Slack webhook for alerts:
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."

# Or SMTP email:
export SMTP_HOST="smtp.gmail.com"
export SMTP_PORT="587"
export SMTP_USER="you@gmail.com"
export SMTP_PASSWORD="app-specific-password"
export SMTP_FROM="you@gmail.com"
export SMTP_TO="alerts@you.com"
```

If neither is configured, notifications fall back to stderr.

---

## 3. First-time walkthrough

### Generate a strategy and back-test it

```bash
python -m lab.cli backtest "60/40 SPY-TLT, monthly rebalance"
# Note the run_id; e.g. 20260518-040000
```

Or use a built-in reference:

```bash
python -m lab.cli run-reference 60_40 --universe SPY TLT
```

### Dry-run paper trading

Always start with `--dry-run`. This loads the strategy, computes target
weights, prints what it WOULD trade, and exits without submitting anything.

```bash
python -m lab.cli paper-trade <run_id> --broker fake --dry-run
```

`--broker fake` uses the in-memory `FakeBroker` — no Alpaca call, no network.
This is the safest way to verify the pipeline.

### One real paper-trading cycle

Once the dry-run looks reasonable:

```bash
python -m lab.cli paper-trade <run_id> --broker alpaca
```

This will:
1. Check for a `HALT` sentinel file (`runs/<run_id>/HALT`).
2. Check market hours (skip if closed unless `--ignore-market-hours`).
3. Get account NAV from Alpaca.
4. Run the strategy's `rebalance()` for today.
5. Compute target shares (using bid/ask midpoint).
6. Diff vs current positions; drop trades below the drift threshold.
7. Run every safety check (raises if any guard fires; halts on raise).
8. Submit each order as a market order, day-time-in-force.
9. Log every submit / fill / failure to `runs/<run_id>/live_state/trades.jsonl`.
10. Persist `state.json` with the updated tallies.

### Status check

```bash
python -m lab.cli status <run_id>
```

Shows current state: NAV trajectory peak, order count, halt status,
intended positions, last 5 trade-log events.

### Reconcile broker vs intended positions

```bash
python -m lab.cli reconcile <run_id>
```

Reports any drift between what we think we own and what the broker says
we own. Does not submit orders — diagnostic only.

### Daemon mode (run continuously)

```bash
python -m lab.cli paper-trade <run_id> --broker alpaca --daemon \
  --interval 86400
```

Calls `step()` once every 24 hours. On 3 consecutive crashes, writes the
`HALT` file and exits. Not a substitute for real process supervision — pair
with `launchd` (macOS), `systemd` (Linux), or run inside a tmux session.

---

## 4. Operational runbook

### "How do I stop everything right now?"

```bash
python -m lab.cli halt <run_id> --reason "ad-hoc stop"
```

This writes `runs/<run_id>/HALT`. The next `step()` short-circuits before
any broker call. The daemon detects it and exits.

Equivalent shell shortcut:

```bash
touch runs/<run_id>/HALT
```

### "How do I resume after halting?"

```bash
python -m lab.cli resume <run_id>
```

Removes the `HALT` file and clears `state.halt_reason`. The next `step()`
runs normally. **Read `lab history <run_id>` first to understand why it
halted.**

### "I see an error in the trade log."

```bash
python -m lab.cli history <run_id> --limit 50
```

Tails the trade-log JSONL. Look for `kind: order_reject` or `kind:
safety_halt` entries. Each has a `payload.reason` or `payload.error`.

### "My broker positions don't match what I expected."

```bash
python -m lab.cli reconcile <run_id>
```

The output shows per-ticker intended vs actual + drift as a percentage of
NAV. If `max_drift_pct` is small (< 1%), the next normal rebalance will
absorb it. If it's large, **stop and investigate** — something is off.
The executor will NOT auto-correct large drift; that's a feature.

### "I want to wipe live state and start fresh."

```bash
python -m lab.cli paper-trade <run_id> --reset
```

Removes `live_state/state.json` and `live_state/trades.jsonl`. Re-initializes
with current broker NAV as the new starting NAV. **History is gone.**

---

## 5. Kill-switch reference

Every safety check runs **before** any order is submitted in `step()`. Any
violation raises `KillSwitchTriggered`, which `step()` catches by calling
`halt()` — that sets `state.halt_reason`, writes the `HALT` file, and logs
a `safety_halt` event.

| Guard | Default | What it checks |
|-------|---------|----------------|
| `max_daily_loss_pct` | 0.05 (5%) | Today's NAV vs `state.starting_nav`. Fires when NAV is below start by more than the threshold. |
| `max_total_drawdown_pct` | 0.20 (20%) | Today's NAV vs `state.peak_nav` (all-time high). Fires below peak by more than threshold. |
| `max_position_pct` | 0.50 (50%) | Pro-forma post-trade `abs(shares * price) / NAV` for each ticker. |
| `max_order_size_pct` | 0.25 (25%) | `abs(shares) * price / NAV` for each individual order. |
| `max_orders_per_day` | 20 | Count of `order_submit` events today in the trade log + the new batch. |
| `require_market_open` | True | Calls `broker.market_is_open()`. Skips (does not halt) when False. |
| `require_paper` | True | If `live=True` is somehow requested with this set, halt. |

All thresholds are overridable per-invocation via CLI flags. **Don't loosen
the defaults in production**. The `SafetyConfig.loosen_for_tests()` helper is
explicitly named to discourage misuse.

---

## 6. Backtest-live parity

The most important property of this system is that **the equity curve from
paper trading matches the equity curve from backtesting on the same data**.
If they diverge, something is wrong in the executor and you don't want it
trading real money.

`tests/test_live_executor.py::test_backtest_live_parity_within_one_percent`
verifies this:

1. Run a 60/40 backtest on synthetic returns; save the run.
2. Step the executor through every day with a `FakeBroker` configured to
   fill at the backtest's daily prices.
3. Assert that the executor's NAV trajectory matches the backtest's
   equity curve within 1% over the full run.

If you change anything in `lab/backtest.py`, `lab/live/executor.py`, or the
strategy interface — **rerun this test**. If it fails, fix it before merging.

For real validation against Alpaca's paper endpoint, run the executor for
3–6 months of actual paper trading and compare the resulting NAV trajectory
to a backtest over the same period. They should match within ~1–3% allowing
for slippage, half-spread, and dividend timing differences.

---

## 7. Enabling real-money trading

There is no flag. There is no CLI argument. There is no env var.

To enable live trading, someone with code-write access must do all four of
the following:

1. **Verify paper trading has run cleanly for 3+ months.** No unexplained
   halts. No reconciliation drift > 1%. Realized P&L within 2% of the
   backtest's predicted P&L over the same period.

2. **Manually remove the live guards.** Two places:
   - `lab/live/broker.py`, `AlpacaAdapter.__init__` — the `if live:` check.
   - `lab/live/executor.py`, `PaperExecutor.__init__` — the `if cfg.live:`
     check.
   Removing these guards must be done in a commit that's reviewed by
   another human. No "drive-by" changes.

3. **Add per-trade size caps.** Live mode must enforce a per-order dollar
   cap and a per-day notional cap independent of the percentage thresholds.
   E.g., "$5,000 per order, $25,000 per day, regardless of NAV."

4. **Set up real-time monitoring.** Slack/email alerts on every order, every
   fill, and every halt. A dashboard that pages you on anomalies. A daily
   reconciliation report.

If any of those four steps haven't been done, you do not have the right to
flip the switch — even if you wrote the code.

---

## 8. FAQ

**Q: Can I run multiple strategies at once?**
A: Not in v1. Each `paper-trade <run_id>` invocation runs one strategy
against one broker account. If you want to run two, you need two Alpaca
accounts (or two run_ids on the same account, which will trade against the
same NAV and conflict).

**Q: What happens at market open vs close?**
A: The executor uses market orders with `time_in_force="day"`. At Alpaca,
that means the order fills at the next available execution. If you submit
during market hours, it's basically immediate. If you submit after hours,
the order queues for the next open. Default `rebalance_time` is 15:30 ET
(30 min before close) — close enough to close to align with the backtest's
close-to-close return assumption.

**Q: What about dividends and splits?**
A: Alpaca handles these in the account. Dividends credit cash; the next
rebalance treats that cash as part of NAV and reallocates accordingly.
Splits adjust share counts automatically. The executor doesn't model these
explicitly — it just reads `broker.get_positions()` and `broker.get_account()`
each cycle and trusts them.

**Q: What if Alpaca is down?**
A: Order submission errors are logged as `order_reject` events and the
step continues. The strategy doesn't retry within the cycle — the next
day's rebalance will reconcile naturally. If three consecutive `step()`
calls crash, the daemon writes `HALT` and exits.

**Q: Can I see what the executor would do tomorrow without trading?**
A: Yes. `lab paper-trade <run_id> --dry-run`. It runs every step except
the actual order submit — prints what it would have sent.

**Q: Why fractional shares by default?**
A: Alpaca supports them, and they make share-count rounding cleaner. Set
`--whole-shares` if your broker doesn't (or if you want simpler tax-lot
math).

**Q: What's the smallest possible trade?**
A: The drift threshold (default 2% of NAV). Below that, the executor
doesn't trade — it would cost more in spread/fees than the rebalance
gains.

**Q: How do I know if my strategy is working?**
A: After at least one month of paper trading, run `lab status <run_id>`
and compare `cumulative_pnl` to what the backtest predicted for the same
period. They should be close. Big divergence = bug or unmodeled cost.

---

## 9. Where things live

```
lab/live/
├── __init__.py        # public exports
├── broker.py          # BrokerAdapter ABC + FakeBroker + AlpacaAdapter
├── state.py           # ExecutorState + atomic persistence + trade log
├── safety.py          # SafetyConfig + check_safety + HALT helpers
├── executor.py        # PaperExecutor.step / reconcile / run_forever
├── notify.py          # Slack / SMTP / stderr fallback
└── cli.py             # CLI subcommands

runs/<id>/live_state/
├── state.json         # ExecutorState snapshot
├── trades.jsonl       # append-only event log
└── state.json.lock    # filelock guard

runs/<id>/HALT         # sentinel: presence halts the executor
```
