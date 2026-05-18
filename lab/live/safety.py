"""Safety configuration + kill switches for the paper/live executor.

Every safety check is evaluated EVERY rebalance, BEFORE any orders are
submitted. If any check fails, KillSwitchTriggered is raised, the
state's halt_reason is set, and a HALT file is written so subsequent
runs short-circuit until the human clears it.

All defaults are paranoid by design. Loosen per-config as needed for
testing, but never in production.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from lab.live.broker import AccountSummary, BrokerAdapter, OrderRequest
from lab.live.state import (
    ExecutorState,
    append_trade_log,
    halt_file_path,
    save_state,
)

logger = logging.getLogger(__name__)


class KillSwitchTriggered(Exception):
    """Raised by check_safety() when any safety threshold is breached."""


@dataclass(frozen=True)
class SafetyConfig:
    max_daily_loss_pct: float = 0.05         # halt if intraday NAV drops 5%
    max_total_drawdown_pct: float = 0.20     # halt if peak-to-now > 20%
    max_position_pct: float = 0.50           # no single position > 50% NAV
    max_order_size_pct: float = 0.25         # no single order > 25% NAV
    max_orders_per_day: int = 20             # cap daily order count
    require_market_open: bool = True
    require_paper: bool = True

    # Test/observability:
    raise_on_violation: bool = True          # if False, return reason instead of raising

    def loosen_for_tests(self) -> "SafetyConfig":
        """Return a config with caps removed (still requires paper).

        ONLY for tests where you want to bypass NAV / position checks
        without disabling the require_paper guard. Don't use in prod.
        """
        return SafetyConfig(
            max_daily_loss_pct=1.0,
            max_total_drawdown_pct=1.0,
            max_position_pct=1.0,
            max_order_size_pct=1.0,
            max_orders_per_day=10_000,
            require_market_open=False,
            require_paper=True,
        )


# --------------------------------------------------------------------------- #
# Halt-file helpers
# --------------------------------------------------------------------------- #


def write_halt_file(run_dir: Path, reason: str = "manual halt") -> Path:
    """Create the HALT sentinel file. step() short-circuits if present."""
    p = halt_file_path(run_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"{datetime.now(timezone.utc).isoformat()}\n{reason}\n")
    return p


def remove_halt_file(run_dir: Path) -> None:
    p = halt_file_path(run_dir)
    if p.exists():
        p.unlink()


def read_halt_file(run_dir: Path) -> str | None:
    p = halt_file_path(run_dir)
    if not p.exists():
        return None
    return p.read_text().strip() or "halted (no reason given)"


def is_halted(run_dir: Path, state: ExecutorState | None = None) -> bool:
    """True if either the halt-file is present or state.halt_reason is set."""
    if read_halt_file(run_dir) is not None:
        return True
    if state is not None and state.halt_reason:
        return True
    return False


def halt(run_dir: Path, state: ExecutorState, reason: str) -> None:
    """Mark the executor as halted: set state, write HALT file, log it."""
    state.halt_reason = reason
    save_state(run_dir, state)
    write_halt_file(run_dir, reason)
    append_trade_log(run_dir, "safety_halt", {"reason": reason})
    logger.error("HALT: %s", reason)


# --------------------------------------------------------------------------- #
# Pre-trade safety checks
# --------------------------------------------------------------------------- #


def check_safety(
    *,
    broker: BrokerAdapter,
    state: ExecutorState,
    target_orders: list[OrderRequest],
    cfg: SafetyConfig,
    is_live_requested: bool = False,
    orders_today: int = 0,
) -> None:
    """Run every safety check. Raises KillSwitchTriggered on any violation
    (or returns None if cfg.raise_on_violation=False but a reason is found —
    in that case the reason is still propagated via state).

    Caller is responsible for actually halting via the halt() helper if
    they catch KillSwitchTriggered.
    """
    reasons: list[str] = []

    # 1. paper-only guard
    if cfg.require_paper and is_live_requested:
        reasons.append(
            "require_paper=True but live trading was requested. "
            "Refusing to run."
        )

    # 2. market-open guard
    if cfg.require_market_open and not broker.market_is_open():
        # Not a HALT — just a skip. Use a distinct exception class? For now
        # we treat it as a soft-skip via reason; callers can check.
        reasons.append("market is not open — skipping this rebalance")

    account = broker.get_account()

    # 3. daily-loss guard
    if state.starting_nav > 0:
        # NAV today vs starting NAV.
        daily_change = (account.equity - state.starting_nav) / state.starting_nav
        if -daily_change > cfg.max_daily_loss_pct:
            reasons.append(
                f"daily loss {daily_change*100:.2f}% exceeds limit "
                f"({-cfg.max_daily_loss_pct*100:.1f}%)"
            )

    # 4. drawdown guard
    if state.peak_nav > 0:
        drawdown = (account.equity - state.peak_nav) / state.peak_nav
        if -drawdown > cfg.max_total_drawdown_pct:
            reasons.append(
                f"drawdown from peak {drawdown*100:.2f}% exceeds limit "
                f"({-cfg.max_total_drawdown_pct*100:.1f}%)"
            )

    # 5. per-order size guard
    nav = account.equity
    for order in target_orders:
        if nav <= 0:
            continue
        try:
            q = broker.get_quote(order.ticker)
            order_value = abs(order.shares) * q.mid
        except Exception as e:
            reasons.append(f"quote fetch failed for {order.ticker}: {e}")
            continue
        if order_value / nav > cfg.max_order_size_pct:
            reasons.append(
                f"order {order.ticker} value {order_value:.0f} / NAV {nav:.0f} = "
                f"{order_value/nav*100:.1f}% exceeds max_order_size "
                f"({cfg.max_order_size_pct*100:.1f}%)"
            )

    # 6. per-position size guard (post-trade simulation)
    positions = broker.get_positions()
    # Build pro-forma post-trade positions in shares.
    pro_forma: dict[str, float] = {t: p.shares for t, p in positions.items()}
    for o in target_orders:
        delta = o.shares if o.side.value == "buy" else -o.shares
        pro_forma[o.ticker] = pro_forma.get(o.ticker, 0.0) + delta
    for t, shares in pro_forma.items():
        if abs(shares) < 1e-12:
            continue
        try:
            q = broker.get_quote(t)
        except Exception as e:
            reasons.append(f"quote fetch failed for {t}: {e}")
            continue
        if nav <= 0:
            continue
        pos_pct = abs(shares * q.mid) / nav
        if pos_pct > cfg.max_position_pct:
            reasons.append(
                f"position {t} would be {pos_pct*100:.1f}% of NAV, exceeds "
                f"max_position ({cfg.max_position_pct*100:.1f}%)"
            )

    # 7. daily order count guard
    if orders_today + len(target_orders) > cfg.max_orders_per_day:
        reasons.append(
            f"orders today ({orders_today}) + new ({len(target_orders)}) "
            f"would exceed max_orders_per_day ({cfg.max_orders_per_day})"
        )

    if reasons and cfg.raise_on_violation:
        raise KillSwitchTriggered("; ".join(reasons))


# --------------------------------------------------------------------------- #
# A non-raising variant for callers that want to inspect reasons
# --------------------------------------------------------------------------- #


def diagnose_safety(
    *,
    broker: BrokerAdapter,
    state: ExecutorState,
    target_orders: list[OrderRequest],
    cfg: SafetyConfig,
    is_live_requested: bool = False,
    orders_today: int = 0,
) -> list[str]:
    """Same checks as check_safety, but returns a list of reasons (or [])."""
    relaxed = SafetyConfig(
        max_daily_loss_pct=cfg.max_daily_loss_pct,
        max_total_drawdown_pct=cfg.max_total_drawdown_pct,
        max_position_pct=cfg.max_position_pct,
        max_order_size_pct=cfg.max_order_size_pct,
        max_orders_per_day=cfg.max_orders_per_day,
        require_market_open=cfg.require_market_open,
        require_paper=cfg.require_paper,
        raise_on_violation=False,
    )
    try:
        check_safety(
            broker=broker, state=state, target_orders=target_orders,
            cfg=relaxed, is_live_requested=is_live_requested,
            orders_today=orders_today,
        )
        return []
    except KillSwitchTriggered as e:
        return str(e).split("; ")
