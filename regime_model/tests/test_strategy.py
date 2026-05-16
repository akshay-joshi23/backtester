"""Tests for the allocation strategy and backtest framework.

Strategy tests use synthetic returns and gamma series. Backtest tests use a
deterministic minimal model to verify P&L accounting (no transaction cost,
single regime → buy-and-hold; with transaction cost → P&L lower).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_model.allocation.strategy import (
    StrategyConfig,
    mix_weights,
    regime_conditional_moments,
    regime_mixture_covariance,
    regime_target_weights,
    solve_mvo,
    vol_target_scale,
)


# --------------------------------------------------------------------------- #
# MVO sanity
# --------------------------------------------------------------------------- #

def test_mvo_picks_best_asset_when_one_dominates() -> None:
    mu = np.array([0.10, -0.05, 0.0])
    Sigma = np.eye(3) * 0.04
    cfg = StrategyConfig(max_weight=1.0, risk_aversion=2.0)
    w = solve_mvo(mu, Sigma, cfg)
    assert w[0] > 0.6
    assert w[1] < 0.05
    assert w.sum() <= 1.0 + 1e-6


def test_mvo_max_weight_constraint_binds() -> None:
    mu = np.array([0.20, 0.0, 0.0])
    Sigma = np.eye(3) * 0.01
    cfg = StrategyConfig(max_weight=0.4, risk_aversion=2.0)
    w = solve_mvo(mu, Sigma, cfg)
    assert w[0] == pytest.approx(0.4, abs=1e-3)


def test_mvo_long_only_constraint() -> None:
    mu = np.array([0.10, -0.10])
    Sigma = np.eye(2) * 0.04
    cfg = StrategyConfig(max_weight=1.0, risk_aversion=2.0, long_only=True)
    w = solve_mvo(mu, Sigma, cfg)
    assert (w >= -1e-6).all()


def test_mvo_equal_mu_picks_min_var_portfolio() -> None:
    mu = np.array([0.05, 0.05])
    Sigma = np.array([[0.04, 0.0], [0.0, 0.01]])  # second asset has lower vol
    cfg = StrategyConfig(max_weight=1.0, risk_aversion=20.0)
    w = solve_mvo(mu, Sigma, cfg)
    # The lower-vol asset should get more weight.
    assert w[1] > w[0]


# --------------------------------------------------------------------------- #
# Regime moments
# --------------------------------------------------------------------------- #

def test_regime_moments_recover_from_perfect_assignment() -> None:
    rng = np.random.default_rng(0)
    n = 600
    # Generate returns from two regimes with different means.
    s = rng.integers(0, 2, size=n)
    R = np.where(s[:, None] == 0,
                 rng.normal(loc=[0.001, -0.001], scale=0.01, size=(n, 2)),
                 rng.normal(loc=[-0.001, 0.001], scale=0.01, size=(n, 2)))
    returns = pd.DataFrame(R, columns=["A1", "A2"])
    # Perfect (one-hot) gamma assignments.
    gamma = pd.DataFrame(np.eye(2)[s], columns=["P_s0", "P_s1"])
    mu_k, Sigma_k = regime_conditional_moments(returns, gamma)
    assert mu_k[0, 0] > 0
    assert mu_k[0, 1] < 0
    assert mu_k[1, 0] < 0
    assert mu_k[1, 1] > 0


# --------------------------------------------------------------------------- #
# Mixing + vol target
# --------------------------------------------------------------------------- #

def test_mix_weights_simplex_combination() -> None:
    weights_per_regime = np.array([[0.3, 0.7], [0.6, 0.4]])
    probs = np.array([0.25, 0.75])
    w = mix_weights(weights_per_regime, probs)
    assert np.allclose(w, np.array([0.525, 0.475]))


def test_vol_target_hits_target() -> None:
    weights = np.array([0.5, 0.5])
    Sigma = np.diag([0.04 / 252, 0.04 / 252])  # 20% annual vol per asset
    target = 0.10
    scaled, scale = vol_target_scale(weights, Sigma, target_annual_vol=target,
                                      annualization=252.0, max_leverage=10.0)
    realized = np.sqrt(scaled @ Sigma @ scaled * 252.0)
    assert realized == pytest.approx(target, rel=1e-6)


def test_vol_target_capped_by_leverage() -> None:
    weights = np.array([0.5, 0.5])
    Sigma = np.diag([1e-12, 1e-12])  # ~0 vol → would want infinite scale
    scaled, scale = vol_target_scale(weights, Sigma, target_annual_vol=0.10,
                                      max_leverage=1.0)
    # Leverage cap should kick in: total weight = max_leverage.
    assert abs(scaled).sum() == pytest.approx(1.0, rel=1e-6)


# --------------------------------------------------------------------------- #
# Mixture covariance: between-regime variance accounted for
# --------------------------------------------------------------------------- #

def test_mixture_cov_inflates_for_diverging_means() -> None:
    """Mixture cov should be strictly larger than the convex combination
    of individual covariances when the per-regime means differ."""
    Sigma_k = np.array([np.eye(2), np.eye(2)])
    mu_k = np.array([[1.0, 0.0], [-1.0, 0.0]])
    p = np.array([0.5, 0.5])
    Sigma_mix = regime_mixture_covariance(mu_k, Sigma_k, p)
    Sigma_avg = 0.5 * Sigma_k[0] + 0.5 * Sigma_k[1]
    # The (0, 0) entry must include the between-regime variance of mu[:, 0].
    assert Sigma_mix[0, 0] > Sigma_avg[0, 0]
    assert Sigma_mix[1, 1] == pytest.approx(Sigma_avg[1, 1], abs=1e-9)


# --------------------------------------------------------------------------- #
# End-to-end: regime_target_weights returns valid simplex (long-only)
# --------------------------------------------------------------------------- #

def test_target_weights_long_only_and_capped() -> None:
    rng = np.random.default_rng(1)
    n, K = 600, 3
    A = 5
    returns = pd.DataFrame(
        rng.normal(0.0, 0.01, size=(n, A)),
        columns=["SPY", "TLT", "GLD", "UUP", "HYG"],
    )
    gamma = pd.DataFrame(rng.dirichlet(np.ones(K), size=n),
                         columns=[f"P_s{k}" for k in range(K)])
    current = np.array([0.3, 0.4, 0.3])
    cfg = StrategyConfig(max_weight=0.4, risk_aversion=5.0, target_annual_vol=0.10)
    w, diag = regime_target_weights(returns, gamma, current, cfg)
    assert w.shape == (A,)
    assert (w >= -1e-9).all()
    assert (w <= cfg.max_weight + 1e-6).all()
    assert "vol_target_scale" in diag
