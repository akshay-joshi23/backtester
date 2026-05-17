"""Performance metrics for a backtest equity curve."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(
    equity: pd.Series,
    ann_factor: float = 252.0,
    rf_rate: float = 0.0,
) -> dict[str, float]:
    """Standard headline metrics for a daily NAV series.

    Parameters
    ----------
    equity : pd.Series
        Daily portfolio NAV, indexed by trading date. First entry is the
        starting wealth (usually 1.0).
    ann_factor : float
        Trading periods per year. 252 for daily bars.
    rf_rate : float
        Annualized risk-free rate (decimal, e.g. 0.05 for 5%) to subtract in
        Sharpe/Sortino. Default 0.

    Returns
    -------
    dict[str, float]
        Keys: sharpe, sortino, ann_return, ann_vol, max_drawdown, calmar,
        cagr, final_nav, n_obs.
    """
    if len(equity) < 2:
        return {}
    rets = equity.pct_change().dropna()
    if rets.empty:
        return {}
    daily_rf = rf_rate / ann_factor
    excess = rets - daily_rf

    mean_ann = float(excess.mean() * ann_factor)
    vol_ann = float(rets.std(ddof=1) * np.sqrt(ann_factor))
    sharpe = mean_ann / vol_ann if vol_ann > 1e-12 else float("nan")

    downside = rets[rets < daily_rf].std(ddof=1) * np.sqrt(ann_factor)
    sortino = mean_ann / float(downside) if downside > 1e-12 else float("nan")

    cum = equity / equity.iloc[0]
    cummax = cum.cummax()
    dd = cum / cummax - 1.0
    max_dd = float(dd.min())

    n_years = len(rets) / ann_factor
    cagr = float(cum.iloc[-1] ** (1.0 / n_years) - 1.0) if n_years > 0 else float("nan")
    calmar = cagr / abs(max_dd) if max_dd < -1e-12 else float("nan")

    return {
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "ann_return": float(rets.mean() * ann_factor),
        "ann_vol": float(vol_ann),
        "max_drawdown": float(max_dd),
        "calmar": float(calmar),
        "cagr": float(cagr),
        "final_nav": float(cum.iloc[-1]),
        "n_obs": int(len(rets)),
    }


def annual_turnover(turnover: pd.Series, ann_factor: float = 252.0) -> float:
    """Average annualized turnover from per-day |Δw| series.

    Turnover is the sum of |Δw| on each rebalance, summed over a year. A pure
    buy-and-hold has ~0; daily rebalancing of a 60/40 mix has ~0.1; an active
    daily momentum strategy can have 5+.
    """
    if turnover.empty:
        return 0.0
    return float(turnover.sum() * ann_factor / len(turnover))
