"""CLI subcommands for the live executor.

Mounted from lab.cli.main via add_live_subcommands(). Subcommands:
  paper-trade <run_id>  — one step (default) or --daemon
  live-trade  <run_id>  — raises NotImplementedError with the guard message
  halt        <run_id>  — write the HALT file
  resume      <run_id>  — remove the HALT file
  status      <run_id>  — print state + positions + last trade
  reconcile   <run_id>  — broker vs intended; no orders submitted
  history     <run_id>  — tail the trade log
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from tabulate import tabulate

logger = logging.getLogger(__name__)


def _resolve_broker(broker_name: str):
    """Construct a broker adapter by name.

    `fake` is always available (for testing). `alpaca` requires
    APCA_API_KEY_ID + APCA_API_SECRET_KEY in the environment.
    `LAB_USE_FAKE_BROKER=1` env var overrides any broker_name to 'fake'.
    """
    if os.environ.get("LAB_USE_FAKE_BROKER") == "1":
        from lab.live.broker import FakeBroker
        return FakeBroker()
    if broker_name == "fake":
        from lab.live.broker import FakeBroker
        return FakeBroker()
    if broker_name == "alpaca":
        from lab.live.broker import AlpacaAdapter
        return AlpacaAdapter()
    raise ValueError(f"unknown broker: {broker_name}")


def _resolve_run_dir(run_id: str) -> Path:
    from lab.runner import RUNS_DIR
    return RUNS_DIR / run_id


# --------------------------------------------------------------------------- #
# paper-trade
# --------------------------------------------------------------------------- #


def cmd_paper_trade(args: argparse.Namespace) -> int:
    from lab.live.executor import ExecutorConfig, PaperExecutor
    from lab.live.safety import SafetyConfig
    from lab.live.state import reset_state

    run_dir = _resolve_run_dir(args.run_id)
    if not run_dir.exists():
        print(f"no such run: {run_dir}", file=sys.stderr)
        return 2
    if args.reset:
        reset_state(run_dir)

    broker = _resolve_broker(args.broker)
    cfg = ExecutorConfig(
        run_id=args.run_id,
        broker_name=args.broker,
        drift_threshold_pct=args.drift_threshold,
        dry_run=args.dry_run,
        live=False,
        safety=SafetyConfig(
            max_daily_loss_pct=args.max_daily_loss,
            max_total_drawdown_pct=args.max_drawdown,
            max_position_pct=args.max_position,
            max_order_size_pct=args.max_order_size,
            max_orders_per_day=args.max_orders_per_day,
            require_market_open=not args.ignore_market_hours,
        ),
        allow_fractional=not args.whole_shares,
    )
    executor = PaperExecutor(cfg, broker, run_dir=run_dir)

    if args.daemon:
        executor.run_forever(interval_seconds=args.interval)
        return 0

    result = executor.step()
    _print_step_result(result)
    return 0 if not result.halted else 1


def cmd_live_trade(args: argparse.Namespace) -> int:
    from lab.live.broker import AlpacaAdapter
    print(
        "live-trade is not enabled.\n\n"
        f"{AlpacaAdapter.LIVE_GUARD_MESSAGE}",
        file=sys.stderr,
    )
    return 3


# --------------------------------------------------------------------------- #
# halt / resume
# --------------------------------------------------------------------------- #


def cmd_halt(args: argparse.Namespace) -> int:
    from lab.live.safety import write_halt_file
    run_dir = _resolve_run_dir(args.run_id)
    if not run_dir.exists():
        print(f"no such run: {run_dir}", file=sys.stderr)
        return 2
    path = write_halt_file(run_dir, args.reason)
    print(f"HALT written: {path}")
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    from lab.live.safety import remove_halt_file
    from lab.live.state import load_state, save_state, state_exists

    run_dir = _resolve_run_dir(args.run_id)
    if not run_dir.exists():
        print(f"no such run: {run_dir}", file=sys.stderr)
        return 2
    remove_halt_file(run_dir)
    if state_exists(run_dir):
        state = load_state(run_dir)
        state.halt_reason = None
        save_state(run_dir, state)
    print("resumed (HALT file cleared, state.halt_reason cleared)")
    return 0


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #


def cmd_status(args: argparse.Namespace) -> int:
    from lab.live.safety import read_halt_file
    from lab.live.state import load_state, read_trade_log, state_exists

    run_dir = _resolve_run_dir(args.run_id)
    if not run_dir.exists():
        print(f"no such run: {run_dir}", file=sys.stderr)
        return 2
    if not state_exists(run_dir):
        print("no live state yet — paper-trade has never been run for this run_id")
        return 0
    state = load_state(run_dir)
    halt = read_halt_file(run_dir)

    print(f"# Run {state.run_id} live status\n")
    print(f"  broker:           {state.broker_name}")
    print(f"  is_live:          {state.is_live}")
    print(f"  started_at:       {state.started_at}")
    print(f"  last_trade_at:    {state.last_trade_at or '—'}")
    print(f"  starting_nav:     {state.starting_nav:.2f}")
    print(f"  peak_nav:         {state.peak_nav:.2f}")
    print(f"  cumulative_pnl:   {state.cumulative_pnl:.2f}")
    print(f"  total_orders:     {state.total_orders}")
    print(f"  total_fills:      {state.total_fills}")
    print(f"  halt_reason:      {state.halt_reason or '—'}")
    print(f"  HALT file:        {'PRESENT — ' + halt if halt else '(not set)'}")

    if state.intended_positions:
        print("\nIntended positions:")
        rows = [(t, f"{s:.4f}") for t, s in state.intended_positions.items()]
        print(tabulate(rows, headers=["ticker", "shares"]))

    log = read_trade_log(run_dir, limit=5)
    if log:
        print("\nLast 5 trade-log events:")
        for entry in log:
            print(f"  {entry.get('ts')}  {entry.get('kind')}  "
                  f"{json.dumps(entry.get('payload', {}))[:80]}")
    return 0


# --------------------------------------------------------------------------- #
# reconcile
# --------------------------------------------------------------------------- #


def cmd_reconcile(args: argparse.Namespace) -> int:
    from lab.live.executor import ExecutorConfig, PaperExecutor

    run_dir = _resolve_run_dir(args.run_id)
    if not run_dir.exists():
        print(f"no such run: {run_dir}", file=sys.stderr)
        return 2
    broker = _resolve_broker(args.broker)
    cfg = ExecutorConfig(run_id=args.run_id, broker_name=args.broker)
    executor = PaperExecutor(cfg, broker, run_dir=run_dir)
    result = executor.reconcile()
    print(f"# Reconcile @ {result.timestamp.isoformat()}\n")
    print(f"max drift as fraction of NAV: {result.max_drift_pct*100:.3f}%")
    rows = []
    tickers = sorted(set(result.intended) | set(result.actual))
    for t in tickers:
        rows.append([
            t,
            f"{result.intended.get(t, 0.0):.4f}",
            f"{result.actual.get(t, 0.0):.4f}",
            f"{result.drift.get(t, 0.0):.4f}",
            f"{result.drift_pct.get(t, 0.0)*100:.3f}%",
        ])
    print(tabulate(rows, headers=["ticker", "intended", "actual", "drift",
                                  "drift_pct"]))
    return 0


# --------------------------------------------------------------------------- #
# history
# --------------------------------------------------------------------------- #


def cmd_history(args: argparse.Namespace) -> int:
    from lab.live.state import read_trade_log

    run_dir = _resolve_run_dir(args.run_id)
    if not run_dir.exists():
        print(f"no such run: {run_dir}", file=sys.stderr)
        return 2
    log = read_trade_log(run_dir, limit=args.limit)
    if not log:
        print("(no trade-log entries)")
        return 0
    for entry in log:
        print(f"{entry.get('ts')}  {entry.get('kind')}  "
              f"{json.dumps(entry.get('payload', {}))}")
    return 0


# --------------------------------------------------------------------------- #
# Pretty-printing
# --------------------------------------------------------------------------- #


def _print_step_result(result) -> None:
    print(f"# Step @ {result.timestamp.isoformat()}\n")
    if result.halted:
        print(f"  HALTED: {result.halt_reason}")
        return
    if result.skipped_market_closed:
        print("  market closed — skipped")
        return
    print(f"  pre-trade NAV:     {result.pre_nav:.2f}")
    print(f"  post-trade NAV:    {result.post_nav:.2f}")
    print(f"  orders submitted:  {len(result.orders_submitted)}")
    print(f"  orders failed:     {len(result.orders_failed)}")
    if result.target_weights:
        print("\nTarget weights:")
        rows = [(t, f"{w:.4f}") for t, w in result.target_weights.items()]
        print(tabulate(rows, headers=["ticker", "weight"]))
    if result.orders_submitted:
        print("\nOrders submitted:")
        rows = [(o.get("ticker"), o.get("side"), f"{o.get('shares', 0):.4f}",
                 o.get("order_id", "dry-run"))
                for o in result.orders_submitted]
        print(tabulate(rows, headers=["ticker", "side", "shares", "order_id"]))
    if result.orders_failed:
        print("\nOrders FAILED:")
        rows = [(o.get("ticker"), o.get("side"), f"{o.get('shares', 0):.4f}",
                 o.get("error", ""))
                for o in result.orders_failed]
        print(tabulate(rows, headers=["ticker", "side", "shares", "error"]))


# --------------------------------------------------------------------------- #
# Subcommand registration (called from lab.cli.main)
# --------------------------------------------------------------------------- #


def add_live_subcommands(sub) -> None:
    """Attach all live subcommands to the given argparse subparsers object."""
    pt = sub.add_parser("paper-trade",
                        help="run one paper-trading rebalance against a saved run")
    pt.add_argument("run_id")
    pt.add_argument("--broker", default="fake",
                    help="alpaca | fake (default fake; set LAB_USE_FAKE_BROKER=1 to force)")
    pt.add_argument("--drift-threshold", type=float, default=0.02,
                    help="don't trade unless drift > N percent of NAV (default 0.02)")
    pt.add_argument("--dry-run", action="store_true",
                    help="print orders, don't submit")
    pt.add_argument("--daemon", action="store_true",
                    help="loop forever with --interval seconds between cycles")
    pt.add_argument("--interval", type=int, default=86400,
                    help="seconds between daemon cycles (default 86400)")
    pt.add_argument("--reset", action="store_true",
                    help="wipe existing live state before starting")
    pt.add_argument("--max-daily-loss", type=float, default=0.05)
    pt.add_argument("--max-drawdown", type=float, default=0.20)
    pt.add_argument("--max-position", type=float, default=0.50)
    pt.add_argument("--max-order-size", type=float, default=0.25)
    pt.add_argument("--max-orders-per-day", type=int, default=20)
    pt.add_argument("--ignore-market-hours", action="store_true",
                    help="trade even when market is closed (testing only)")
    pt.add_argument("--whole-shares", action="store_true",
                    help="disallow fractional shares (default allow)")
    pt.set_defaults(func=cmd_paper_trade)

    lt = sub.add_parser("live-trade",
                        help="(guarded — always errors out for safety)")
    lt.add_argument("run_id")
    lt.set_defaults(func=cmd_live_trade)

    h = sub.add_parser("halt", help="write HALT sentinel — stops next step()")
    h.add_argument("run_id")
    h.add_argument("--reason", default="manual halt")
    h.set_defaults(func=cmd_halt)

    r = sub.add_parser("resume", help="remove HALT sentinel + clear halt_reason")
    r.add_argument("run_id")
    r.set_defaults(func=cmd_resume)

    s = sub.add_parser("status", help="print live state for a run")
    s.add_argument("run_id")
    s.set_defaults(func=cmd_status)

    rc = sub.add_parser("reconcile",
                        help="diff broker positions vs intended; no orders")
    rc.add_argument("run_id")
    rc.add_argument("--broker", default="fake")
    rc.set_defaults(func=cmd_reconcile)

    hi = sub.add_parser("history", help="tail the trade log")
    hi.add_argument("run_id")
    hi.add_argument("--limit", type=int, default=20)
    hi.set_defaults(func=cmd_history)
