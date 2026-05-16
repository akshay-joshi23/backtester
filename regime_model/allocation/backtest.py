"""Walk-forward backtest framework for the regime allocation strategy.

Loop semantics:
  - Training windows are expanding: VI is refit at the end of each calendar
    year on all data through that year.
  - The PF runs continuously: particles persist across year boundaries; only
    the parameters used for propagation/weighting change.
  - Rebalancing happens every `rebalance_freq` trading days. On a rebalance
    day we use the PF posterior at end of t-1 and the gamma history through
    t-1 to compute target weights, then trade to those weights at the close
    of t-1 (so day-t returns flow into the new weights).
  - Transaction cost: 5bps per leg, charged on |w_new - w_drifted|.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from regime_model.allocation.strategy import (
    StrategyConfig,
    regime_target_weights,
)
from regime_model.data.loaders import ALLOCATION_TICKERS
from regime_model.inference.particle_filter import (
    init_pf_state,
    run_pf_segment,
)
from regime_model.inference.variational import (
    PosteriorPointEstimate,
    SVIConfig,
    fit_svi,
)
from regime_model.models.bayesian_regime import ModelConfig

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Config + result containers
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class BacktestConfig:
    train_end: str = "2011-01-01"
    rebalance_freq: int = 5            # trading days between rebalances
    transaction_cost_bps: float = 5.0  # 5bps per leg
    n_particles: int = 10_000
    refit_calendar_year: bool = True
    initial_wealth: float = 1.0
    seed: int = 0


@dataclass
class BacktestResult:
    equity_curve: pd.Series             # daily portfolio NAV
    target_weights: pd.DataFrame        # (T_oos, A) target weights set on rebalance days
    realized_weights: pd.DataFrame      # (T_oos, A) drifted weights at end of each day
    turnover: pd.Series                 # (T_oos,) total |Δw| on rebalance days, else 0
    transaction_costs: pd.Series        # daily cost in NAV units
    regime_probs: pd.DataFrame          # (T_oos, K) PF posterior
    diagnostics: dict[str, pd.Series] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _expanding_window_years(
    feature_index: pd.DatetimeIndex,
    train_end: str,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Yield (year_start, year_end, vi_train_cutoff) per OOS calendar year.

    vi_train_cutoff is the (exclusive) last date used to fit VI for that year.
    """
    train_end_ts = pd.Timestamp(train_end)
    last_year = feature_index.max().year
    out = []
    for y in range(train_end_ts.year, last_year + 1):
        year_start = pd.Timestamp(f"{y}-01-01")
        year_end = pd.Timestamp(f"{y + 1}-01-01")
        cutoff = year_start  # VI uses data strictly before year_start
        out.append((year_start, year_end, cutoff))
    return out


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #

def walk_forward_backtest(
    returns: pd.DataFrame,
    features: pd.DataFrame,
    cfg: BacktestConfig | None = None,
    strategy_cfg: StrategyConfig | None = None,
    model_cfg: ModelConfig | None = None,
    svi_cfg: SVIConfig | None = None,
    fit_callback: Callable[[int, PosteriorPointEstimate], None] | None = None,
) -> BacktestResult:
    """Run the walk-forward backtest end to end.

    Parameters
    ----------
    returns : DataFrame of daily log returns for the allocation universe.
              Must contain ALLOCATION_TICKERS as columns.
    features : DataFrame of standardized features for the model. Must align
               with `returns` on dates.
    cfg : BacktestConfig.

    Returns
    -------
    BacktestResult
    """
    cfg = cfg or BacktestConfig()
    strategy_cfg = strategy_cfg or StrategyConfig()
    model_cfg = model_cfg or ModelConfig()
    svi_cfg = svi_cfg or SVIConfig()

    if not all(t in returns.columns for t in ALLOCATION_TICKERS):
        missing = set(ALLOCATION_TICKERS) - set(returns.columns)
        raise ValueError(f"returns missing allocation tickers: {missing}")
    alloc_ret = returns[list(ALLOCATION_TICKERS)].copy()

    # Align features and returns; drop the warmup period (NaNs in features).
    aligned = features.dropna().join(alloc_ret, how="inner", lsuffix="_f", rsuffix="")
    feature_cols = [c for c in features.columns]
    feat_aligned = aligned[[f"{c}_f" if f"{c}_f" in aligned.columns else c for c in feature_cols]]
    feat_aligned.columns = feature_cols  # restore original names
    ret_aligned = aligned[list(ALLOCATION_TICKERS)]

    # Setup walk-forward years.
    years = _expanding_window_years(feat_aligned.index, cfg.train_end)
    train_end_ts = pd.Timestamp(cfg.train_end)
    if feat_aligned.index.min() >= train_end_ts:
        raise ValueError("training window has no data before cfg.train_end")

    # Storage for OOS results.
    oos_mask = feat_aligned.index >= train_end_ts
    oos_idx = feat_aligned.index[oos_mask]
    A = len(ALLOCATION_TICKERS)
    K = model_cfg.K

    target_w = pd.DataFrame(0.0, index=oos_idx, columns=ALLOCATION_TICKERS)
    realized_w = pd.DataFrame(0.0, index=oos_idx, columns=ALLOCATION_TICKERS)
    turnover = pd.Series(0.0, index=oos_idx)
    tc_series = pd.Series(0.0, index=oos_idx)
    gamma_oos = pd.DataFrame(0.0, index=oos_idx, columns=[f"P_s{k}" for k in range(K)])
    nav = pd.Series(0.0, index=oos_idx, name="nav")
    diag_scale = pd.Series(0.0, index=oos_idx)
    diag_pre_lev = pd.Series(0.0, index=oos_idx)

    # Initialize PF from a uniform prior (real init happens after first VI fit).
    pf_state = None
    nav_prev = cfg.initial_wealth
    weights_prev = np.zeros(A)
    pe: PosteriorPointEstimate | None = None
    gamma_history = pd.DataFrame(columns=[f"P_s{k}" for k in range(K)])

    days_since_rebalance = 0
    cost_per_leg = cfg.transaction_cost_bps / 1e4

    for year_idx, (year_start, year_end, cutoff) in enumerate(years):
        # 1. Refit VI on data strictly before `cutoff`.
        train_feat = feat_aligned.loc[feat_aligned.index < cutoff]
        if len(train_feat) < 252:
            raise RuntimeError(f"training window too short for year {year_start.year}")
        logger.info("=== year %d  refit VI on %d obs (%s..%s) ===",
                    year_start.year, len(train_feat),
                    train_feat.index.min().date(), train_feat.index.max().date())
        svi_result = fit_svi(
            train_feat.to_numpy(),
            model_cfg=model_cfg,
            svi_cfg=svi_cfg,
            n_posterior_samples=100,
        )
        pe = svi_result.point_estimate
        if fit_callback is not None:
            fit_callback(year_start.year, pe)

        # On the first year, initialize PF state and run it on the training
        # window so the gamma_history is populated for moment estimation.
        if pf_state is None:
            pf_state = init_pf_state(pe.pi, n_particles=cfg.n_particles, seed=cfg.seed)
            res_train, pf_state = run_pf_segment(
                pf_state, train_feat.to_numpy(),
                pi=pe.pi, A=pe.A, mu=pe.mu, scale_tril=pe.scale_tril,
            )
            gamma_history = pd.DataFrame(
                res_train.state_probs, index=train_feat.index,
                columns=[f"P_s{k}" for k in range(K)],
            )

        # 2. Run PF on the year's data.
        year_mask = (feat_aligned.index >= year_start) & (feat_aligned.index < year_end)
        year_dates = feat_aligned.index[year_mask]
        if len(year_dates) == 0:
            continue
        year_feat = feat_aligned.loc[year_dates]
        year_ret = ret_aligned.loc[year_dates]
        res_year, pf_state = run_pf_segment(
            pf_state, year_feat.to_numpy(),
            pi=pe.pi, A=pe.A, mu=pe.mu, scale_tril=pe.scale_tril,
        )
        gamma_year = pd.DataFrame(
            res_year.state_probs, index=year_dates,
            columns=[f"P_s{k}" for k in range(K)],
        )
        gamma_oos.loc[year_dates] = gamma_year.values

        # Append to history for moment estimation in subsequent rebalances.
        gamma_history = pd.concat([gamma_history, gamma_year])

        # 3. Walk through year's days and rebalance weekly.
        for t_idx, t in enumerate(year_dates):
            # Compute target weights on rebalance day (using info through previous day).
            if days_since_rebalance == 0:
                # Use history strictly before today, intersected across both
                # gamma_history (regime posteriors) and aligned (returns).
                hist_dates = gamma_history.index[gamma_history.index < t]
                hist_dates = hist_dates.intersection(aligned.index)
                ret_hist = aligned.loc[hist_dates, list(ALLOCATION_TICKERS)]
                gam_hist = gamma_history.loc[hist_dates]
                current_probs = gam_hist.iloc[-1].values  # latest available posterior
                w_target, diag = regime_target_weights(
                    ret_hist, gam_hist, current_probs, strategy_cfg
                )
                # Charge transaction cost on the change from drifted to target.
                trade = w_target - weights_prev
                cost = cost_per_leg * np.abs(trade).sum()
                nav_post_trade = nav_prev * (1.0 - cost)
                tc_series.loc[t] = nav_prev * cost
                turnover.loc[t] = float(np.abs(trade).sum())
                weights_after_trade = w_target.copy()
                diag_scale.loc[t] = diag["vol_target_scale"]
                diag_pre_lev.loc[t] = diag["pre_scale_leverage"]
            else:
                weights_after_trade = weights_prev.copy()
                nav_post_trade = nav_prev
            target_w.loc[t] = weights_after_trade

            # Apply day t's returns to weights_after_trade.
            r_t = year_ret.loc[t].values  # log returns
            simple_r = np.expm1(r_t)
            port_simple_r = float(weights_after_trade @ simple_r)
            new_nav = nav_post_trade * (1.0 + port_simple_r)
            # Drift weights: w_i scales by (1 + r_i) / (1 + port_r), then re-allocates the cash arm.
            invested = weights_after_trade.sum()
            if invested > 0 and (1.0 + port_simple_r) > 0:
                drifted_invested = weights_after_trade * (1.0 + simple_r) / (1.0 + port_simple_r)
            else:
                drifted_invested = weights_after_trade.copy()
            realized_w.loc[t] = drifted_invested
            weights_prev = drifted_invested
            nav_prev = new_nav
            nav.loc[t] = new_nav

            days_since_rebalance += 1
            if days_since_rebalance >= cfg.rebalance_freq:
                days_since_rebalance = 0

    return BacktestResult(
        equity_curve=nav,
        target_weights=target_w,
        realized_weights=realized_w,
        turnover=turnover,
        transaction_costs=tc_series,
        regime_probs=gamma_oos,
        diagnostics={
            "vol_target_scale": diag_scale,
            "pre_scale_leverage": diag_pre_lev,
        },
    )


# --------------------------------------------------------------------------- #
# Performance metrics
# --------------------------------------------------------------------------- #

def compute_metrics(equity: pd.Series, ann_factor: float = 252.0) -> dict[str, float]:
    """Standard headline metrics. equity is daily NAV, indexed by trading date."""
    rets = equity.pct_change().dropna()
    if len(rets) == 0:
        return {}
    mean_ann = rets.mean() * ann_factor
    vol_ann = rets.std() * np.sqrt(ann_factor)
    sharpe = mean_ann / vol_ann if vol_ann > 0 else float("nan")
    downside = rets[rets < 0].std() * np.sqrt(ann_factor)
    sortino = mean_ann / downside if downside > 0 else float("nan")
    cum = equity / equity.iloc[0]
    cummax = cum.cummax()
    dd = cum / cummax - 1.0
    max_dd = float(dd.min())
    calmar = mean_ann / abs(max_dd) if max_dd < 0 else float("nan")
    return {
        "ann_return": float(mean_ann),
        "ann_vol": float(vol_ann),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": max_dd,
        "calmar": float(calmar),
        "n_obs": int(len(rets)),
    }
