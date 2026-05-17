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


def block_bootstrap_metrics(
    equity: pd.Series,
    *,
    block_size: int = 20,
    n_resamples: int = 1000,
    ann_factor: float = 252.0,
    seed: int = 0,
    metrics: tuple[str, ...] = ("sharpe", "cagr", "max_drawdown"),
) -> dict[str, dict[str, float]]:
    """Block-bootstrap confidence intervals on backtest metrics.

    Method
    ------
    1. Convert NAV to daily *simple* returns (one number per trading day).
    2. Draw N resamples of the same length: each resample is built by
       concatenating randomly chosen blocks of `block_size` consecutive
       returns (sampled with replacement, wrap-around). This preserves
       short-horizon autocorrelation that an IID bootstrap would destroy.
    3. Rebuild a NAV path for each resample and compute the requested
       metrics.
    4. Return mean / std / 2.5%-97.5% quantiles per metric.

    For Sharpe and CAGR this gives a meaningful spread; for max_drawdown
    block bootstrap underestimates the true distribution because the worst
    drawdowns often involve longer-than-block-size runs of bad days. Still
    informative as a sanity check.
    """
    rng = np.random.default_rng(seed)
    rets = equity.pct_change().dropna().to_numpy()
    n = len(rets)
    if n < block_size * 2:
        raise ValueError(
            f"need at least {block_size * 2} returns for block bootstrap, got {n}"
        )
    n_blocks = (n + block_size - 1) // block_size

    out: dict[str, list[float]] = {m: [] for m in metrics}
    for _ in range(n_resamples):
        # Sample block start indices with replacement; wrap-around lookup.
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block_size)[None, :]) % n
        sample = rets[idx].ravel()[:n]
        sample_nav = pd.Series(np.cumprod(1.0 + sample))
        m = compute_metrics(sample_nav, ann_factor=ann_factor)
        for key in metrics:
            v = m.get(key)
            if v is not None and np.isfinite(v):
                out[key].append(float(v))

    result: dict[str, dict[str, float]] = {}
    for key, vals in out.items():
        if not vals:
            continue
        arr = np.asarray(vals)
        result[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)),
            "ci_low_95": float(np.quantile(arr, 0.025)),
            "ci_high_95": float(np.quantile(arr, 0.975)),
            "n_resamples": int(len(arr)),
        }
    return result


def annual_turnover(turnover: pd.Series, ann_factor: float = 252.0) -> float:
    """Average annualized turnover from per-day |Δw| series.

    Turnover is the sum of |Δw| on each rebalance, summed over a year. A pure
    buy-and-hold has ~0; daily rebalancing of a 60/40 mix has ~0.1; an active
    daily momentum strategy can have 5+.
    """
    if turnover.empty:
        return 0.0
    return float(turnover.sum() * ann_factor / len(turnover))
