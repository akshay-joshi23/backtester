"""PaperExecutor — the daily rebalance loop.

This is the main module of lab.live. Reads a saved backtest run, loads
its Strategy code, computes today's target weights, diffs against the
broker's current positions, runs every safety check, and submits the
required orders.

Key invariants:
  - step() is idempotent on the same day. If positions match targets
    within the drift threshold, no orders are submitted.
  - check_safety() runs BEFORE any submit_order call. A KillSwitch
    failure halts the executor; subsequent step()s short-circuit on
    the HALT sentinel.
  - On any broker error during submit, the step records the failure
    and proceeds — it does NOT retry within the same cycle. The next
    daily run reconciles naturally.

The executor is NOT a daemon by itself — schedule it with cron,
launchd, or `lab paper-trade --daemon` (a thin run_forever loop with
sleep + error handling).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from lab.live.broker import (
    BrokerAdapter,
    BrokerError,
    OrderRequest,
    OrderSide,
    OrderType,
    Position,
)
from lab.live.safety import (
    KillSwitchTriggered,
    SafetyConfig,
    check_safety,
    halt,
    is_halted,
    read_halt_file,
)
from lab.live.state import (
    ExecutorState,
    append_trade_log,
    initialize_state,
    live_state_dir,
    load_state,
    save_state,
    state_exists,
)
from lab.runner import RUNS_DIR, execute_strategy_code, load_run

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Configs and result containers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ExecutorConfig:
    run_id: str
    broker_name: str = "alpaca"
    rebalance_time: str = "15:30"          # ET; informational only in v1
    drift_threshold_pct: float = 0.02      # 2% of NAV
    max_capital: float | None = None
    dry_run: bool = False
    live: bool = False                     # GUARDED — raises
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    allow_fractional: bool = True


@dataclass
class StepResult:
    timestamp: datetime
    halted: bool = False
    halt_reason: str | None = None
    skipped_market_closed: bool = False
    target_weights: dict[str, float] = field(default_factory=dict)
    target_shares: dict[str, float] = field(default_factory=dict)
    orders_submitted: list[dict] = field(default_factory=list)
    orders_failed: list[dict] = field(default_factory=list)
    pre_nav: float = 0.0
    post_nav: float = 0.0


@dataclass
class ReconcileResult:
    timestamp: datetime
    intended: dict[str, float]
    actual: dict[str, float]
    drift: dict[str, float]               # ticker -> |intended - actual| shares
    drift_pct: dict[str, float]           # ticker -> drift / NAV
    max_drift_pct: float


# --------------------------------------------------------------------------- #
# PaperExecutor
# --------------------------------------------------------------------------- #


class PaperExecutor:
    """Drives one saved run via a BrokerAdapter."""

    def __init__(
        self,
        cfg: ExecutorConfig,
        broker: BrokerAdapter,
        run_dir: Path | None = None,
    ):
        if cfg.live:
            from lab.live.broker import AlpacaAdapter
            raise NotImplementedError(AlpacaAdapter.LIVE_GUARD_MESSAGE)
        self.cfg = cfg
        self.broker = broker
        self.run_dir = Path(run_dir) if run_dir else RUNS_DIR / cfg.run_id
        if not self.run_dir.exists():
            raise FileNotFoundError(f"run not found: {self.run_dir}")
        # Load strategy artifacts up front so we fail loudly if missing.
        self._run = load_run(cfg.run_id, runs_dir=self.run_dir.parent)
        # Load or initialize state.
        if state_exists(self.run_dir):
            self.state = load_state(self.run_dir)
        else:
            account = self.broker.get_account()
            self.state = initialize_state(
                self.run_dir, run_id=cfg.run_id,
                broker_name=broker.name, starting_nav=account.equity,
            )

    # -- core loop ------------------------------------------------------- #

    def step(self) -> StepResult:
        """Execute one rebalance cycle.

        Returns a StepResult describing what happened. The state file is
        always updated even on early returns (so peak_nav and other
        observability fields stay current).
        """
        now = datetime.now(timezone.utc)
        result = StepResult(timestamp=now)

        # 1. HALT check (before any broker call).
        if is_halted(self.run_dir, self.state):
            reason = read_halt_file(self.run_dir) or self.state.halt_reason
            result.halted = True
            result.halt_reason = reason
            append_trade_log(self.run_dir, "skip_halted", {"reason": reason})
            return result

        # 2. Market-open check.
        try:
            market_open = self.broker.market_is_open()
        except BrokerError as e:
            logger.warning("market_is_open() failed: %s", e)
            market_open = True  # fail open — we still run safety checks below
        if self.cfg.safety.require_market_open and not market_open:
            result.skipped_market_closed = True
            append_trade_log(self.run_dir, "skip_market_closed", {})
            return result

        # 3. Account snapshot + peak update.
        account = self.broker.get_account()
        result.pre_nav = account.equity
        if account.equity > self.state.peak_nav:
            self.state.peak_nav = account.equity

        # 4. Compute today's target weights via the saved strategy.
        target_weights = self._compute_target_weights(now)
        result.target_weights = target_weights

        # 5. Translate weights → target shares using broker quotes.
        nav = account.equity if self.cfg.max_capital is None else min(
            account.equity, self.cfg.max_capital,
        )
        target_shares = self._weights_to_shares(target_weights, nav)
        result.target_shares = target_shares
        self.state.intended_positions = target_shares

        # 6. Diff against current positions + drift filter.
        current_positions = self.broker.get_positions()
        orders = self._build_orders(target_shares, current_positions, nav)

        # 7. Safety check on the candidate orders.
        try:
            check_safety(
                broker=self.broker, state=self.state, target_orders=orders,
                cfg=self.cfg.safety, is_live_requested=self.cfg.live,
                orders_today=self._orders_today(now),
            )
        except KillSwitchTriggered as e:
            halt(self.run_dir, self.state, str(e))
            result.halted = True
            result.halt_reason = str(e)
            return result

        # 8. Dry-run: print, don't submit.
        if self.cfg.dry_run:
            for o in orders:
                payload = {
                    "ticker": o.ticker, "shares": o.shares,
                    "side": o.side.value, "dry_run": True,
                }
                result.orders_submitted.append(payload)
                append_trade_log(self.run_dir, "dry_run_order", payload)
            save_state(self.run_dir, self.state)
            return result

        # 9. Submit orders.
        for o in orders:
            try:
                or_ = self.broker.submit_order(o)
                self.state.total_orders += 1
                payload = {
                    "order_id": or_.order_id, "ticker": o.ticker,
                    "shares": o.shares, "side": o.side.value,
                    "submitted_at": str(or_.submitted_at),
                }
                result.orders_submitted.append(payload)
                append_trade_log(self.run_dir, "order_submit", payload)
                # Eagerly check fill (FakeBroker fills instantly; live polls).
                try:
                    status = self.broker.get_order(or_.order_id)
                    if status.fills:
                        self.state.total_fills += 1
                        for fill in status.fills:
                            append_trade_log(self.run_dir, "order_fill", {
                                "order_id": or_.order_id,
                                "ticker": o.ticker, "shares": fill.shares,
                                "price": fill.price,
                                "filled_at": str(fill.timestamp),
                            })
                except BrokerError as e:
                    logger.warning("post-submit get_order failed: %s", e)
            except BrokerError as e:
                payload = {
                    "ticker": o.ticker, "shares": o.shares,
                    "side": o.side.value, "error": str(e),
                }
                result.orders_failed.append(payload)
                append_trade_log(self.run_dir, "order_reject", payload)

        # 10. Refresh NAV + persist state.
        try:
            post = self.broker.get_account()
            result.post_nav = post.equity
            if post.equity > self.state.peak_nav:
                self.state.peak_nav = post.equity
        except BrokerError:
            result.post_nav = result.pre_nav
        self.state.last_trade_at = now.isoformat()
        save_state(self.run_dir, self.state)
        return result

    def run_forever(self, interval_seconds: int = 86400) -> None:
        """Daemon loop. step() once per interval. Halts on 3 consecutive
        unhandled exceptions. Designed to be invoked from cron / launchd
        / systemd — NOT a substitute for real process supervision.
        """
        import time
        consecutive_failures = 0
        while True:
            try:
                result = self.step()
                if result.halted:
                    logger.error("Executor halted: %s", result.halt_reason)
                    return
                consecutive_failures = 0
            except Exception:
                logger.exception("step() crashed")
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    halt(self.run_dir, self.state,
                          "3 consecutive step() failures")
                    return
            time.sleep(interval_seconds)

    def reconcile(self) -> ReconcileResult:
        """Compare intended positions to broker positions. No orders submitted."""
        now = datetime.now(timezone.utc)
        intended = dict(self.state.intended_positions)
        actual_positions = self.broker.get_positions()
        actual = {t: p.shares for t, p in actual_positions.items()}
        all_tickers = set(intended) | set(actual)
        drift: dict[str, float] = {}
        drift_pct: dict[str, float] = {}
        try:
            nav = self.broker.get_account().equity
        except BrokerError:
            nav = 0.0
        for t in all_tickers:
            d = abs(intended.get(t, 0.0) - actual.get(t, 0.0))
            drift[t] = d
            if nav > 0:
                try:
                    q = self.broker.get_quote(t)
                    drift_pct[t] = d * q.mid / nav
                except BrokerError:
                    drift_pct[t] = 0.0
        self.state.last_reconcile_at = now.isoformat()
        self.state.last_reconcile_drift = drift
        save_state(self.run_dir, self.state)
        append_trade_log(self.run_dir, "reconcile", {
            "intended": intended, "actual": actual, "drift": drift,
        })
        return ReconcileResult(
            timestamp=now, intended=intended, actual=actual,
            drift=drift, drift_pct=drift_pct,
            max_drift_pct=max(drift_pct.values()) if drift_pct else 0.0,
        )

    # -- internals ------------------------------------------------------- #

    def _compute_target_weights(self, today: datetime) -> dict[str, float]:
        """Run the saved strategy's rebalance() against history we have.

        We use the equity.csv index as a proxy for trading-day cadence and
        the saved run's universe + start date. For historical alignment
        we re-load from yfinance via lab.data — same data path as the
        backtest, so weights match.
        """
        from lab.data import load_universe

        universe = self._run["config"]["universe"]
        bundle = load_universe(universe, start="2010-01-01")
        returns = bundle.returns[universe]

        # Exec the strategy code in a fresh namespace.
        cls = execute_strategy_code(self._run["strategy_code"])
        strategy = cls()
        # Fit on the same training window used in backtest.
        train_end = pd.Timestamp(self._run["config"]["train_end"])
        train_history = returns.loc[returns.index < train_end]
        strategy.fit(train_history)
        # History through yesterday.
        today_ts = pd.Timestamp(today.date())
        history = returns.loc[returns.index < today_ts]
        if len(history) == 0:
            logger.warning("no history before today (%s)", today_ts)
            return {t: 0.0 for t in universe}
        raw = strategy.rebalance(today_ts, history)
        if not isinstance(raw, pd.Series):
            logger.error("strategy returned %s, not Series", type(raw))
            return {t: 0.0 for t in universe}
        # Normalize index.
        raw = raw.reindex(universe).fillna(0.0)
        return {t: float(raw[t]) for t in universe}

    def _weights_to_shares(self, weights: dict[str, float], nav: float) -> dict[str, float]:
        out: dict[str, float] = {}
        for t, w in weights.items():
            if abs(w) < 1e-9 or nav <= 0:
                out[t] = 0.0
                continue
            try:
                q = self.broker.get_quote(t)
            except BrokerError as e:
                logger.warning("quote for %s failed: %s", t, e)
                out[t] = 0.0
                continue
            target_dollars = w * nav
            shares = target_dollars / q.mid if q.mid > 0 else 0.0
            if not self.cfg.allow_fractional:
                # Round TOWARD ZERO so we don't over-buy.
                shares = math.copysign(math.floor(abs(shares)), shares) if shares != 0 else 0.0
            out[t] = shares
        return out

    def _build_orders(
        self,
        target_shares: dict[str, float],
        current_positions: dict[str, Position],
        nav: float,
    ) -> list[OrderRequest]:
        threshold_dollars = nav * self.cfg.drift_threshold_pct
        orders: list[OrderRequest] = []
        all_tickers = set(target_shares) | set(current_positions)
        for t in all_tickers:
            tgt = target_shares.get(t, 0.0)
            cur = current_positions[t].shares if t in current_positions else 0.0
            delta = tgt - cur
            if abs(delta) < 1e-9:
                continue
            # Drift filter.
            try:
                q = self.broker.get_quote(t)
            except BrokerError:
                continue
            delta_dollars = abs(delta) * q.mid
            if delta_dollars < threshold_dollars:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(OrderRequest(
                ticker=t, shares=abs(delta), side=side,
                order_type=OrderType.MARKET, time_in_force="day",
            ))
        return orders

    def _orders_today(self, now: datetime) -> int:
        """Count orders submitted today by walking the trade log.

        Cheap unless the log is gigantic, in which case we should rotate it.
        """
        from lab.live.state import read_trade_log
        today_date = now.date().isoformat()
        n = 0
        for entry in read_trade_log(self.run_dir, limit=500):
            if entry.get("kind") == "order_submit" and entry.get("ts", "").startswith(today_date):
                n += 1
        return n
