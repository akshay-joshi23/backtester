"""Baseline strategies required by the spec for performance comparison.

Implemented:
    - 60/40 static SPY/TLT
    - Risk parity (inverse-vol) across the 5 allocation ETFs
    - HMM-EM regime model + same regime-conditional allocation pipeline
    - Random regime assignment (negative control — should lose vs informed)

Not implemented (out of scope for V1):
    - Markov-switching GARCH on SPY (sanity check; complex enough to deserve
      its own module). See spec Phase 5 baselines.

Each baseline returns a `BacktestResult` so it plugs into the same metrics
and plotting code as the main regime strategy.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

from regime_model.allocation.backtest import (
    BacktestConfig,
    BacktestResult,
)
from regime_model.allocation.strategy import (
    StrategyConfig,
    regime_target_weights,
)
from regime_model.data.loaders import ALLOCATION_TICKERS
from regime_model.models.bayesian_regime import ModelConfig
from regime_model.models.hmm_baseline import (
    HMMConfig,
    fit_hmm,
    filtered_state_logprobs,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Generic loop: takes a function (date, history) -> target weights
# --------------------------------------------------------------------------- #

def _run_loop(
    returns: pd.DataFrame,
    target_weights_fn: Callable[[pd.Timestamp], np.ndarray | None],
    cfg: BacktestConfig,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Run the daily allocation/PnL loop.

    `target_weights_fn(t)` returns the target weights to enter on day t,
    or None if no rebalance is forced. The caller decides rebalance cadence
    (typically every cfg.rebalance_freq days).

    Returns (nav, target_w, realized_w, turnover, transaction_costs).
    """
    A = len(ALLOCATION_TICKERS)
    cost_per_leg = cfg.transaction_cost_bps / 1e4
    nav = pd.Series(0.0, index=returns.index, name="nav")
    target_w = pd.DataFrame(0.0, index=returns.index, columns=ALLOCATION_TICKERS)
    realized_w = pd.DataFrame(0.0, index=returns.index, columns=ALLOCATION_TICKERS)
    turnover = pd.Series(0.0, index=returns.index)
    tc_series = pd.Series(0.0, index=returns.index)

    days_since_rebalance = 0
    weights_prev = np.zeros(A)
    nav_prev = cfg.initial_wealth

    for t in returns.index:
        forced_w = target_weights_fn(t) if days_since_rebalance == 0 else None
        if forced_w is not None:
            trade = forced_w - weights_prev
            cost = cost_per_leg * np.abs(trade).sum()
            nav_post_trade = nav_prev * (1.0 - cost)
            tc_series.loc[t] = nav_prev * cost
            turnover.loc[t] = float(np.abs(trade).sum())
            weights_after_trade = forced_w.copy()
        else:
            weights_after_trade = weights_prev.copy()
            nav_post_trade = nav_prev
        target_w.loc[t] = weights_after_trade

        r_t = returns.loc[t].values
        simple_r = np.expm1(r_t)
        port_simple_r = float(weights_after_trade @ simple_r)
        new_nav = nav_post_trade * (1.0 + port_simple_r)
        invested = weights_after_trade.sum()
        if invested > 0 and (1.0 + port_simple_r) > 0:
            drifted = weights_after_trade * (1.0 + simple_r) / (1.0 + port_simple_r)
        else:
            drifted = weights_after_trade.copy()
        realized_w.loc[t] = drifted
        weights_prev = drifted
        nav_prev = new_nav
        nav.loc[t] = new_nav

        days_since_rebalance += 1
        if days_since_rebalance >= cfg.rebalance_freq:
            days_since_rebalance = 0

    return nav, target_w, realized_w, turnover, tc_series


# --------------------------------------------------------------------------- #
# 60/40 static SPY/TLT
# --------------------------------------------------------------------------- #

def static_60_40(
    returns: pd.DataFrame,
    cfg: BacktestConfig | None = None,
) -> BacktestResult:
    """Buy and hold (with weekly rebalancing) 60% SPY, 40% TLT."""
    cfg = cfg or BacktestConfig()
    target = np.zeros(len(ALLOCATION_TICKERS))
    target[ALLOCATION_TICKERS.index("SPY")] = 0.6
    target[ALLOCATION_TICKERS.index("TLT")] = 0.4
    oos_idx = returns.index[returns.index >= pd.Timestamp(cfg.train_end)]
    ret = returns.loc[oos_idx, list(ALLOCATION_TICKERS)]

    def fn(_t):
        return target

    nav, tw, rw, to, tc = _run_loop(ret, fn, cfg)
    return BacktestResult(
        equity_curve=nav, target_weights=tw, realized_weights=rw,
        turnover=to, transaction_costs=tc,
        regime_probs=pd.DataFrame(index=oos_idx),
    )


# --------------------------------------------------------------------------- #
# Risk parity (inverse-vol)
# --------------------------------------------------------------------------- #

def inverse_vol_weights(
    returns_history: pd.DataFrame,
    lookback: int = 252,
) -> np.ndarray:
    """w_i = (1/σ_i) / sum_j (1/σ_j), σ from trailing window."""
    if len(returns_history) < lookback:
        return np.full(returns_history.shape[1], 1.0 / returns_history.shape[1])
    vols = returns_history.iloc[-lookback:].std().to_numpy()
    inv = 1.0 / np.where(vols > 1e-9, vols, 1e-9)
    return inv / inv.sum()


def risk_parity(
    returns: pd.DataFrame,
    cfg: BacktestConfig | None = None,
    lookback: int = 252,
) -> BacktestResult:
    """Inverse-vol weighted across the 5 allocation ETFs, weekly rebalanced."""
    cfg = cfg or BacktestConfig()
    full_ret = returns[list(ALLOCATION_TICKERS)]
    oos_idx = returns.index[returns.index >= pd.Timestamp(cfg.train_end)]
    ret = full_ret.loc[oos_idx]

    def fn(t):
        hist = full_ret.loc[full_ret.index < t]
        return inverse_vol_weights(hist, lookback=lookback)

    nav, tw, rw, to, tc = _run_loop(ret, fn, cfg)
    return BacktestResult(
        equity_curve=nav, target_weights=tw, realized_weights=rw,
        turnover=to, transaction_costs=tc,
        regime_probs=pd.DataFrame(index=oos_idx),
    )


# --------------------------------------------------------------------------- #
# HMM regime model + same allocation pipeline
# --------------------------------------------------------------------------- #

def hmm_regime_strategy(
    returns: pd.DataFrame,
    features: pd.DataFrame,
    cfg: BacktestConfig | None = None,
    strategy_cfg: StrategyConfig | None = None,
    K: int = 3,
    hmm_cfg: HMMConfig | None = None,
) -> BacktestResult:
    """HMM-EM as the regime model, same allocation logic as the main strategy.

    Annual refit on expanding window (mirrors the Bayesian backtest). The
    HMM forward-pass provides exact filtered marginals, so no PF needed.
    """
    cfg = cfg or BacktestConfig()
    strategy_cfg = strategy_cfg or StrategyConfig()
    hmm_cfg = hmm_cfg or HMMConfig(K=K, max_iter=200, tol=1e-5, seed=0, cov_ridge=1e-3)

    feat = features.dropna()
    full_ret = returns[list(ALLOCATION_TICKERS)]
    aligned = feat.join(full_ret, how="inner", lsuffix="_f", rsuffix="")
    feat_aligned = aligned[[f"{c}_f" if f"{c}_f" in aligned.columns else c
                            for c in feat.columns]]
    feat_aligned.columns = list(feat.columns)
    ret_aligned = aligned[list(ALLOCATION_TICKERS)]

    train_end_ts = pd.Timestamp(cfg.train_end)
    last_year = feat_aligned.index.max().year
    gamma_history = pd.DataFrame(
        columns=[f"P_s{k}" for k in range(K)], dtype=float
    )

    # Pre-compute gamma per OOS year using HMM refit.
    for y in range(train_end_ts.year, last_year + 1):
        cutoff = pd.Timestamp(f"{y}-01-01")
        next_cutoff = pd.Timestamp(f"{y + 1}-01-01")
        train_feat = feat_aligned.loc[feat_aligned.index < cutoff]
        if len(train_feat) < 252:
            raise RuntimeError(f"HMM training window too short for year {y}")
        logger.info("HMM refit year %d  on %d obs", y, len(train_feat))
        params = fit_hmm(train_feat.to_numpy(), hmm_cfg)

        # Compute filtered marginals on training data the first time, OOS slice every year.
        if gamma_history.empty:
            log_gamma_train = np.asarray(filtered_state_logprobs(
                train_feat.to_numpy(), params
            ))
            gamma_history = pd.DataFrame(
                np.exp(log_gamma_train), index=train_feat.index,
                columns=[f"P_s{k}" for k in range(K)],
            )
        # Year slice. We compute filtered probs on (training + year) and
        # extract just the year (so the recursion has correct prefix).
        all_feat = feat_aligned.loc[feat_aligned.index < next_cutoff]
        log_gamma_all = np.asarray(filtered_state_logprobs(all_feat.to_numpy(), params))
        gamma_all = pd.DataFrame(np.exp(log_gamma_all), index=all_feat.index,
                                 columns=[f"P_s{k}" for k in range(K)])
        year_mask = (all_feat.index >= cutoff) & (all_feat.index < next_cutoff)
        year_dates = all_feat.index[year_mask]
        gamma_history = pd.concat(
            [gamma_history.loc[gamma_history.index < cutoff], gamma_all.loc[year_dates]]
        )

    # Now run the allocation loop on OOS dates.
    oos_idx = ret_aligned.index[ret_aligned.index >= train_end_ts]
    ret_oos = ret_aligned.loc[oos_idx]

    def fn(t):
        # Use only history before t for moments and current posterior.
        hist_dates = gamma_history.index[gamma_history.index < t]
        hist_dates = hist_dates.intersection(ret_aligned.index)
        if len(hist_dates) < 252:
            return np.full(len(ALLOCATION_TICKERS), 1.0 / len(ALLOCATION_TICKERS))
        ret_hist = ret_aligned.loc[hist_dates]
        gam_hist = gamma_history.loc[hist_dates]
        current = gam_hist.iloc[-1].values
        w, _ = regime_target_weights(ret_hist, gam_hist, current, strategy_cfg)
        return w

    nav, tw, rw, to, tc = _run_loop(ret_oos, fn, cfg)
    return BacktestResult(
        equity_curve=nav, target_weights=tw, realized_weights=rw,
        turnover=to, transaction_costs=tc,
        regime_probs=gamma_history.loc[oos_idx],
    )


# --------------------------------------------------------------------------- #
# Random regime assignment (negative control)
# --------------------------------------------------------------------------- #

def random_regime_strategy(
    returns: pd.DataFrame,
    features: pd.DataFrame,
    cfg: BacktestConfig | None = None,
    strategy_cfg: StrategyConfig | None = None,
    K: int = 3,
    seed: int = 0,
) -> BacktestResult:
    """Each day, regime probabilities are a random simplex point.

    Per spec, this should LOSE. If it doesn't, the regime model isn't adding
    information — that's a red flag.
    """
    cfg = cfg or BacktestConfig()
    strategy_cfg = strategy_cfg or StrategyConfig()
    rng = np.random.default_rng(seed)

    feat = features.dropna()
    full_ret = returns[list(ALLOCATION_TICKERS)]
    aligned = feat.join(full_ret, how="inner", lsuffix="_f", rsuffix="")
    ret_aligned = aligned[list(ALLOCATION_TICKERS)]

    train_end_ts = pd.Timestamp(cfg.train_end)
    all_dates = ret_aligned.index
    gamma_random = pd.DataFrame(
        rng.dirichlet(np.ones(K), size=len(all_dates)),
        index=all_dates,
        columns=[f"P_s{k}" for k in range(K)],
    )

    oos_idx = all_dates[all_dates >= train_end_ts]
    ret_oos = ret_aligned.loc[oos_idx]

    def fn(t):
        hist_dates = gamma_random.index[gamma_random.index < t]
        hist_dates = hist_dates.intersection(ret_aligned.index)
        if len(hist_dates) < 252:
            return np.full(len(ALLOCATION_TICKERS), 1.0 / len(ALLOCATION_TICKERS))
        ret_hist = ret_aligned.loc[hist_dates]
        gam_hist = gamma_random.loc[hist_dates]
        current = gam_hist.iloc[-1].values
        w, _ = regime_target_weights(ret_hist, gam_hist, current, strategy_cfg)
        return w

    nav, tw, rw, to, tc = _run_loop(ret_oos, fn, cfg)
    return BacktestResult(
        equity_curve=nav, target_weights=tw, realized_weights=rw,
        turnover=to, transaction_costs=tc,
        regime_probs=gamma_random.loc[oos_idx],
    )
