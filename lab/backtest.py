"""Walk-forward backtest engine.

Drives an arbitrary `Strategy` against a returns DataFrame:

  1. Call `strategy.fit(history)` once on the training-window prefix.
  2. Walk day-by-day through the out-of-sample window.
  3. On rebalance days, call `strategy.rebalance(date, history_through_yesterday)`
     to get target weights. Charge transaction cost on |Δw| via `cost_model`.
  4. Between rebalances, weights drift with realized returns.
  5. Apply day-t simple returns to the drifted portfolio; mark NAV.

No lookahead by construction: `history_through_yesterday` strictly excludes
date `t` or later.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from lab.costs import CostModel, FlatBpsPerLeg
from lab.strategy import Strategy, StrategyError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestConfig:
    train_end: str = "2015-01-01"
    rebalance_freq: int = 21         # trading days between rebalances (monthly default)
    initial_wealth: float = 1.0
    long_only: bool = True
    max_leverage: float = 1.0
    progress: bool = False


@dataclass
class BacktestResult:
    equity: pd.Series                 # daily NAV, OOS only
    target_weights: pd.DataFrame      # post-trade weights on rebalance days, else carried
    realized_weights: pd.DataFrame    # end-of-day drifted weights
    turnover: pd.Series               # |Δw| sum on rebalance days; 0 elsewhere
    transaction_costs: pd.Series      # daily $ cost (in NAV units)
    rebalance_dates: list[pd.Timestamp] = field(default_factory=list)
    strategy_name: str = ""
    config: BacktestConfig | None = None


def walk_forward_backtest(
    strategy: Strategy,
    returns: pd.DataFrame,
    cfg: BacktestConfig | None = None,
    cost_model: CostModel | None = None,
) -> BacktestResult:
    """Run a walk-forward backtest of `strategy` against `returns`.

    Parameters
    ----------
    strategy : Strategy
        A concrete Strategy instance (already constructed; the engine calls
        `fit` and `rebalance` on it).
    returns : pd.DataFrame
        Daily log returns indexed by trading date, columns are tickers.
    cfg : BacktestConfig | None
        Backtest config. Default = monthly rebalance, train_end=2015-01-01.
    cost_model : CostModel | None
        Defaults to FlatBpsPerLeg(5.0).

    Returns
    -------
    BacktestResult
    """
    cfg = cfg or BacktestConfig()
    cost_model = cost_model or FlatBpsPerLeg(5.0)

    returns = returns.dropna(how="any").sort_index()
    if returns.empty:
        raise ValueError("returns is empty after dropna")

    train_end_ts = pd.Timestamp(cfg.train_end)
    if returns.index.min() >= train_end_ts:
        raise ValueError(
            f"training window has no data before {cfg.train_end}; "
            f"data starts at {returns.index.min().date()}"
        )
    if returns.index.max() < train_end_ts:
        raise ValueError(
            f"all data is before train_end={cfg.train_end}; no OOS to backtest"
        )

    train_history = returns.loc[returns.index < train_end_ts]
    oos_dates = returns.index[returns.index >= train_end_ts]
    tickers = list(returns.columns)
    A = len(tickers)

    # 1. Fit on training window.
    logger.info("fit() on %d training obs (%s..%s)",
                len(train_history),
                train_history.index.min().date(),
                train_history.index.max().date())
    strategy.fit(train_history)

    # 2. Walk-forward loop.
    equity = pd.Series(0.0, index=oos_dates, name="nav")
    target_w = pd.DataFrame(0.0, index=oos_dates, columns=tickers)
    realized_w = pd.DataFrame(0.0, index=oos_dates, columns=tickers)
    turnover = pd.Series(0.0, index=oos_dates)
    tc_series = pd.Series(0.0, index=oos_dates)
    rebalance_dates: list[pd.Timestamp] = []

    nav = cfg.initial_wealth
    weights_prev = np.zeros(A)
    days_since_rebalance = 0
    iterator = oos_dates
    if cfg.progress:
        from tqdm import tqdm  # local import to keep module light
        iterator = tqdm(oos_dates, desc=f"backtest:{strategy.name}", unit="d")

    for t in iterator:
        history = returns.loc[returns.index < t]

        if days_since_rebalance == 0:
            raw = strategy.rebalance(t, history)
            w_target = _coerce_weights(raw, tickers, cfg)
            cost = cost_model.apply(weights_prev, w_target)
            tc_series.loc[t] = nav * cost
            turnover.loc[t] = float(np.abs(w_target - weights_prev).sum())
            nav_after_trade = nav * (1.0 - cost)
            weights_after_trade = w_target
            rebalance_dates.append(t)
        else:
            weights_after_trade = weights_prev
            nav_after_trade = nav
        target_w.loc[t] = weights_after_trade

        # Apply day-t returns.
        r_log = returns.loc[t].values
        r_simple = np.expm1(r_log)
        port_simple = float(weights_after_trade @ r_simple)
        nav_new = nav_after_trade * (1.0 + port_simple)
        # Drift weights with realized returns.
        invested = weights_after_trade.sum()
        if invested > 0 and (1.0 + port_simple) > 0:
            drifted = weights_after_trade * (1.0 + r_simple) / (1.0 + port_simple)
        else:
            drifted = weights_after_trade.copy()
        realized_w.loc[t] = drifted

        weights_prev = drifted
        nav = nav_new
        equity.loc[t] = nav

        days_since_rebalance += 1
        if days_since_rebalance >= cfg.rebalance_freq:
            days_since_rebalance = 0

    return BacktestResult(
        equity=equity,
        target_weights=target_w,
        realized_weights=realized_w,
        turnover=turnover,
        transaction_costs=tc_series,
        rebalance_dates=rebalance_dates,
        strategy_name=strategy.name,
        config=cfg,
    )


def _coerce_weights(
    raw: pd.Series, tickers: list[str], cfg: BacktestConfig
) -> np.ndarray:
    """Validate and sanitize the user's rebalance() output."""
    if not isinstance(raw, pd.Series):
        raise StrategyError(
            f"rebalance must return a pandas Series, got {type(raw).__name__}"
        )
    aligned = raw.reindex(tickers).fillna(0.0)
    aligned = aligned.replace([np.inf, -np.inf], 0.0)
    arr = aligned.to_numpy(dtype=float)
    if cfg.long_only and (arr < -1e-9).any():
        bad = aligned[aligned < -1e-9].to_dict()
        raise StrategyError(f"long_only=True but got negative weights: {bad}")
    # Renormalize by gross exposure if it exceeds the leverage cap. This
    # works for both long-only (sum |w| = sum w) and long/short (sum |w| >
    # net exposure) cases.
    gross = float(np.abs(arr).sum())
    if gross > cfg.max_leverage + 1e-9:
        arr = arr * (cfg.max_leverage / gross)
    if cfg.long_only:
        arr = np.clip(arr, 0.0, None)
    if not np.all(np.isfinite(arr)):
        raise StrategyError(f"rebalance returned non-finite weights: {aligned.to_dict()}")
    return arr
