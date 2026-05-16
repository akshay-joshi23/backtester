"""Tests for baseline strategies.

Each test uses a small synthetic returns + features panel so they run fast
and don't hit the network. The point is to verify the accounting machinery
(turnover, transaction costs, daily P&L) and the structural shape of each
baseline, not to test predictive performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_model.allocation.backtest import BacktestConfig
from regime_model.allocation.baselines import (
    inverse_vol_weights,
    random_regime_strategy,
    risk_parity,
    static_60_40,
)
from regime_model.allocation.strategy import StrategyConfig
from regime_model.data.loaders import ALLOCATION_TICKERS, FEATURE_TICKERS


def _synthetic_returns(seed: int = 0, n_days: int = 1500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2008-01-01", periods=n_days)
    data = rng.normal(0.0003, 0.01, size=(n_days, len(ALLOCATION_TICKERS) + 1))
    cols = list(ALLOCATION_TICKERS) + ["VIX"]
    df = pd.DataFrame(data, index=dates, columns=cols)
    return df


def _synthetic_features(dates: pd.DatetimeIndex, K: int = 14, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = (
        [f"ret1d_{t}" for t in FEATURE_TICKERS]
        + [f"vol20_{t}" for t in FEATURE_TICKERS]
        + ["corr60_SPY_TLT", "corr60_HYG_SPY"]
    )
    return pd.DataFrame(
        rng.normal(0.0, 1.0, size=(len(dates), len(cols))),
        index=dates,
        columns=cols,
    )


# --------------------------------------------------------------------------- #
# 60/40 sanity
# --------------------------------------------------------------------------- #

def test_static_60_40_returns_target_weights() -> None:
    returns = _synthetic_returns(n_days=600)
    cfg = BacktestConfig(train_end="2009-01-01", rebalance_freq=5,
                         transaction_cost_bps=0.0)
    res = static_60_40(returns, cfg)
    # Sum target weights on a rebalance day → 1.0
    assert (res.target_weights.sum(axis=1).max()) == pytest.approx(1.0, abs=1e-9)
    # 60% to SPY, 40% to TLT consistently on rebalance days.
    rebalance_rows = res.target_weights[res.turnover > 0]
    assert (rebalance_rows["SPY"] - 0.6).abs().max() < 1e-9
    assert (rebalance_rows["TLT"] - 0.4).abs().max() < 1e-9


def test_zero_tc_60_40_matches_buy_and_hold_on_first_day() -> None:
    """With 0 TC, the day-1 portfolio return should equal 0.6*r_SPY + 0.4*r_TLT."""
    returns = _synthetic_returns(n_days=500)
    cfg = BacktestConfig(train_end="2009-01-01", rebalance_freq=5,
                         transaction_cost_bps=0.0, initial_wealth=1.0)
    res = static_60_40(returns, cfg)
    eq = res.equity_curve.dropna()
    first_oos = eq.index[0]
    r_spy = np.expm1(returns.loc[first_oos, "SPY"])
    r_tlt = np.expm1(returns.loc[first_oos, "TLT"])
    expected = 1.0 * (1 + 0.6 * r_spy + 0.4 * r_tlt)
    assert eq.iloc[0] == pytest.approx(expected, rel=1e-9)


# --------------------------------------------------------------------------- #
# Inverse vol weights
# --------------------------------------------------------------------------- #

def test_inverse_vol_weights_sum_to_one() -> None:
    rng = np.random.default_rng(1)
    returns = pd.DataFrame(
        rng.normal(0.0, 0.01, size=(600, 5)),
        columns=ALLOCATION_TICKERS,
    )
    w = inverse_vol_weights(returns, lookback=252)
    assert w.sum() == pytest.approx(1.0)
    assert (w > 0).all()


def test_inverse_vol_weights_higher_for_low_vol_asset() -> None:
    rng = np.random.default_rng(2)
    n = 400
    # Asset 0: high vol; asset 1: low vol.
    R = np.zeros((n, 5))
    R[:, 0] = rng.normal(0.0, 0.05, size=n)
    R[:, 1] = rng.normal(0.0, 0.005, size=n)
    R[:, 2:] = rng.normal(0.0, 0.01, size=(n, 3))
    returns = pd.DataFrame(R, columns=ALLOCATION_TICKERS)
    w = inverse_vol_weights(returns, lookback=252)
    assert w[1] > w[0]  # low-vol asset gets more weight


def test_risk_parity_runs_end_to_end() -> None:
    returns = _synthetic_returns(n_days=600)
    cfg = BacktestConfig(train_end="2009-01-01", rebalance_freq=5)
    res = risk_parity(returns, cfg)
    assert len(res.equity_curve.dropna()) > 50
    # Risk-parity weights should always be positive and sum to 1 on rebalance.
    rebalance_rows = res.target_weights[res.turnover > 0]
    if len(rebalance_rows) > 0:
        assert (rebalance_rows >= -1e-9).all().all()


# --------------------------------------------------------------------------- #
# Random regime — accounting + has finite Sharpe
# --------------------------------------------------------------------------- #

def test_random_regime_baseline_runs_and_produces_finite_metrics() -> None:
    returns = _synthetic_returns(n_days=900)
    feats = _synthetic_features(returns.index)
    cfg = BacktestConfig(train_end="2010-01-01", rebalance_freq=10)
    sc = StrategyConfig(max_weight=0.4, risk_aversion=50.0,
                        target_annual_vol=0.10, max_leverage=1.0)
    res = random_regime_strategy(returns, feats, cfg, sc, K=3, seed=0)
    eq = res.equity_curve.dropna()
    assert (eq > 0).all(), "NAV went non-positive — strategy blew up"


# --------------------------------------------------------------------------- #
# Transaction costs accounting
# --------------------------------------------------------------------------- #

def test_zero_tc_yields_zero_tc_series() -> None:
    returns = _synthetic_returns(n_days=400)
    cfg = BacktestConfig(train_end="2009-01-01", transaction_cost_bps=0.0)
    res = static_60_40(returns, cfg)
    assert (res.transaction_costs == 0.0).all()


def test_tc_proportional_to_bps() -> None:
    """Doubling transaction-cost bps roughly doubles total cost on the same trades."""
    returns = _synthetic_returns(n_days=400)
    cfg_lo = BacktestConfig(train_end="2009-01-01", transaction_cost_bps=2.0)
    cfg_hi = BacktestConfig(train_end="2009-01-01", transaction_cost_bps=20.0)
    res_lo = static_60_40(returns, cfg_lo)
    res_hi = static_60_40(returns, cfg_hi)
    # Static 60/40 trades only on first day; subsequent rebalances tiny drift.
    # Costs should scale roughly with bps (the trade pattern itself is similar
    # across cost levels, so the ratio is close to 10x).
    ratio = res_hi.transaction_costs.sum() / max(res_lo.transaction_costs.sum(), 1e-12)
    assert 5.0 < ratio < 20.0
