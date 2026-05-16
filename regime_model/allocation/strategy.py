"""Regime-conditional allocation strategy.

At each rebalance the allocator does:

    1. Estimate per-regime expected return mu_k and covariance Sigma_k of the
       allocation universe (5 ETFs) from the gamma-weighted history.
    2. Solve a constrained mean-variance problem per regime:
            max_w  mu_k^T w - (lambda/2) w^T Sigma_k w
            s.t.   w >= 0, sum(w) <= 1, w_i <= max_weight
    3. Mix the per-regime weights by the current filtered posterior:
            w_t = sum_k P(s_t = k | data) * w_k*
    4. Vol-target by scaling so that sqrt(252) * sqrt(w^T Sigma_mix w)
       = target_annual_vol. Sigma_mix is the regime-mixture covariance,
       inflated for between-regime variance:
            Sigma_mix = sum_k p_k Sigma_k + sum_k p_k (mu_k - mu_bar)(...)^T
    5. Cap leverage at 1.0 (long-only, no borrowing in V1).

The vol-targeting step is also why we keep `risk_aversion` mostly nominal:
the final scale is set by the vol target, not by the mean-variance lambda.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# --------------------------------------------------------------------------- #
# Strategy config
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class StrategyConfig:
    max_weight: float = 0.4               # max weight per asset
    # MVO lambda: large by default to push toward minimum-variance, since the
    # per-regime mean estimates are noisy (gamma-weighted historical means
    # have wide standard errors when any regime has limited effective sample
    # size). Sensitivity sweep in the writeup justifies the choice.
    risk_aversion: float = 50.0
    target_annual_vol: float = 0.10       # 10% annualized portfolio vol
    long_only: bool = True
    max_leverage: float = 1.0             # cap on sum of weights post-vol-target
    cov_ridge: float = 1e-5               # ridge on per-regime covariance


# --------------------------------------------------------------------------- #
# Per-regime mu/Sigma from gamma-weighted history
# --------------------------------------------------------------------------- #

def regime_conditional_moments(
    returns: pd.DataFrame,
    gamma: pd.DataFrame,
    cfg: StrategyConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-regime mean and covariance of allocation-asset returns.

    Parameters
    ----------
    returns : (T, A) DataFrame of daily log returns for the A allocation assets.
    gamma   : (T, K) DataFrame of regime posterior probabilities aligned to returns.
    cfg     : strategy config (used for cov_ridge).

    Returns
    -------
    mu_k    : (K, A)
    Sigma_k : (K, A, A)
    """
    cfg = cfg or StrategyConfig()
    R = returns.to_numpy()
    G = gamma.to_numpy()
    T, A = R.shape
    K = G.shape[1]
    if G.shape[0] != T:
        raise ValueError(f"gamma and returns row count mismatch: {G.shape[0]} vs {T}")

    mu_k = np.zeros((K, A))
    Sigma_k = np.zeros((K, A, A))
    eye = np.eye(A) * cfg.cov_ridge

    for k in range(K):
        w = G[:, k]
        ws = w.sum()
        if ws < 1e-9:
            mu_k[k] = R.mean(axis=0)
            Sigma_k[k] = np.cov(R.T) + eye
            continue
        mu = (w[:, None] * R).sum(axis=0) / ws
        diff = R - mu
        Sigma = (w[:, None] * diff).T @ diff / ws
        mu_k[k] = mu
        Sigma_k[k] = Sigma + eye
    return mu_k, Sigma_k


# --------------------------------------------------------------------------- #
# Constrained mean-variance optimization
# --------------------------------------------------------------------------- #

def solve_mvo(
    mu: np.ndarray,
    Sigma: np.ndarray,
    cfg: StrategyConfig | None = None,
) -> np.ndarray:
    """Solve a single-regime constrained MVO.

    Maximize mu^T w - (lambda/2) w^T Sigma w subject to:
        - w >= 0 if long_only
        - sum(w) <= 1
        - w_i <= max_weight for each i

    Returns weights (A,). If the QP is infeasible or fails, returns equal-weight
    capped at max_weight.
    """
    cfg = cfg or StrategyConfig()
    A = mu.shape[0]

    def neg_utility(w: np.ndarray) -> float:
        return -(mu @ w) + 0.5 * cfg.risk_aversion * w @ Sigma @ w

    def neg_utility_grad(w: np.ndarray) -> np.ndarray:
        return -mu + cfg.risk_aversion * Sigma @ w

    bounds = [(0.0 if cfg.long_only else -cfg.max_weight, cfg.max_weight) for _ in range(A)]
    constraints = [{"type": "ineq", "fun": lambda w: 1.0 - w.sum(), "jac": lambda w: -np.ones(A)}]
    x0 = np.full(A, min(1.0 / A, cfg.max_weight))

    try:
        res = minimize(
            neg_utility, x0, jac=neg_utility_grad, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 100, "ftol": 1e-9},
        )
        if not res.success:
            return _fallback_equal_weight(A, cfg.max_weight)
        w = np.clip(res.x, 0.0 if cfg.long_only else -cfg.max_weight, cfg.max_weight)
        if w.sum() > 1.0 + 1e-6:
            w = w / w.sum()
        return w
    except Exception:
        return _fallback_equal_weight(A, cfg.max_weight)


def _fallback_equal_weight(A: int, max_weight: float) -> np.ndarray:
    w = np.full(A, min(1.0 / A, max_weight))
    if w.sum() > 1.0:
        w = w / w.sum()
    return w


# --------------------------------------------------------------------------- #
# Mixing + vol target
# --------------------------------------------------------------------------- #

def mix_weights(weights_per_regime: np.ndarray, regime_probs: np.ndarray) -> np.ndarray:
    """w_t = sum_k p_k w_k*. weights_per_regime: (K, A); regime_probs: (K,)."""
    return regime_probs @ weights_per_regime


def regime_mixture_covariance(
    mu_k: np.ndarray, Sigma_k: np.ndarray, regime_probs: np.ndarray
) -> np.ndarray:
    """Mixture covariance: sum_k p_k Sigma_k + sum_k p_k (mu_k - mu_bar)(mu_k - mu_bar)^T."""
    mu_bar = regime_probs @ mu_k
    within = np.einsum("k,kij->ij", regime_probs, Sigma_k)
    diff = mu_k - mu_bar
    between = np.einsum("k,ki,kj->ij", regime_probs, diff, diff)
    return within + between


def vol_target_scale(
    weights: np.ndarray,
    Sigma_mix: np.ndarray,
    target_annual_vol: float,
    annualization: float = 252.0,
    max_leverage: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Scale weights so that annualized portfolio vol = target.

    Returns (scaled_weights, scale_factor). Cap total leverage at max_leverage.
    """
    daily_var = float(weights @ Sigma_mix @ weights)
    if daily_var <= 1e-16:
        return weights, 1.0
    annual_vol = np.sqrt(daily_var * annualization)
    scale = target_annual_vol / annual_vol
    # Cap so total leverage <= max_leverage.
    total = abs(weights).sum()
    if total > 1e-12:
        scale = min(scale, max_leverage / total)
    return weights * scale, float(scale)


# --------------------------------------------------------------------------- #
# Top-level decision: returns daily target weights
# --------------------------------------------------------------------------- #

def regime_target_weights(
    returns_history: pd.DataFrame,
    gamma_history: pd.DataFrame,
    current_regime_probs: np.ndarray,
    cfg: StrategyConfig | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute the day-t target weights given history through t-1.

    `returns_history` and `gamma_history` must be aligned and contain only data
    available BEFORE the day we're trading on (caller guarantees no look-ahead).

    Returns (weights, diagnostics) where diagnostics carries the vol-target
    scale and pre-scale leverage.
    """
    cfg = cfg or StrategyConfig()
    mu_k, Sigma_k = regime_conditional_moments(returns_history, gamma_history, cfg)
    K, A = mu_k.shape

    weights_per_regime = np.zeros((K, A))
    for k in range(K):
        weights_per_regime[k] = solve_mvo(mu_k[k], Sigma_k[k], cfg)

    raw_w = mix_weights(weights_per_regime, current_regime_probs)
    Sigma_mix = regime_mixture_covariance(mu_k, Sigma_k, current_regime_probs)
    scaled_w, scale = vol_target_scale(
        raw_w, Sigma_mix,
        target_annual_vol=cfg.target_annual_vol,
        max_leverage=cfg.max_leverage,
    )
    # Per-spec, max-weight is a hard constraint. Vol-target scaling can push a
    # single asset over the cap; clip and re-allocate within the cap.
    scaled_w = _enforce_max_weight(scaled_w, cfg.max_weight)

    diag = {
        "vol_target_scale": scale,
        "pre_scale_leverage": float(raw_w.sum()),
        "post_scale_leverage": float(scaled_w.sum()),
        "ann_pre_scale_vol": float(np.sqrt(raw_w @ Sigma_mix @ raw_w * 252.0)),
    }
    return scaled_w, diag


def _enforce_max_weight(w: np.ndarray, max_weight: float) -> np.ndarray:
    """Cap each weight at max_weight; redistribute the trimmed mass among
    uncapped assets up to their own caps. Iterates until no asset is over."""
    w = w.copy()
    A = len(w)
    for _ in range(A):
        over = w > max_weight + 1e-12
        if not over.any():
            break
        excess = (w[over] - max_weight).sum()
        w[over] = max_weight
        free = ~over & (w > 0)
        if not free.any() or excess <= 0:
            break
        room = max_weight - w[free]
        if room.sum() <= 1e-12:
            break
        w[free] += excess * (room / room.sum())
    return w
