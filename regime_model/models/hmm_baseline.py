"""Gaussian-emission HMM with EM fitting.

This is the spec's required baseline ("Standard HMM with EM fitting, no
Bayesian regularization"). It serves two purposes:
  1. Sanity check that distinct regimes exist in the feature data.
  2. A reusable forward-backward implementation for the Bayesian model's
     E-step on q(s_{1:T}).

All forward / backward / E-step computations are done in log space using
log-sum-exp to avoid underflow on multi-thousand-step sequences. Emission
log-likelihoods use a single Cholesky decomposition per state per iteration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp

logger = logging.getLogger(__name__)

LOG_TWO_PI = np.log(2.0 * np.pi)


# --------------------------------------------------------------------------- #
# Config / fitted parameter container
# --------------------------------------------------------------------------- #

@dataclass
class HMMConfig:
    K: int = 3                    # number of latent states
    max_iter: int = 100
    tol: float = 1e-4             # log-likelihood improvement threshold
    cov_ridge: float = 1e-4       # diagonal added to each Σ_k after M-step
    seed: int = 0
    verbose: bool = False


@dataclass
class HMMParams:
    pi: np.ndarray   # (K,)         initial state distribution
    A: np.ndarray    # (K, K)       row-stochastic transition matrix
    mu: np.ndarray   # (K, D)       state means
    Sigma: np.ndarray  # (K, D, D)  state covariances
    log_likelihood_history: list[float] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Emission log-likelihoods (vectorized over T)
# --------------------------------------------------------------------------- #

def _gaussian_log_pdf(X: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """log N(x_t | mu, Sigma) for all t. Vectorized via Cholesky.

    X: (T, D); mu: (D,); Sigma: (D, D). Returns (T,).
    """
    D = X.shape[1]
    L = np.linalg.cholesky(Sigma)
    diff = X - mu  # (T, D)
    # Solve L y = diff^T  -> y is (D, T); norm squared by column.
    y = solve_triangular(L, diff.T, lower=True, check_finite=False)
    quad = np.einsum("dt,dt->t", y, y)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (D * LOG_TWO_PI + log_det + quad)


def _all_emission_logp(X: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """Stack emission log-likelihoods across K states. Returns (T, K)."""
    K = mu.shape[0]
    out = np.empty((X.shape[0], K))
    for k in range(K):
        out[:, k] = _gaussian_log_pdf(X, mu[k], Sigma[k])
    return out


# --------------------------------------------------------------------------- #
# Forward-backward in log space
# --------------------------------------------------------------------------- #

def _forward(log_pi: np.ndarray, log_A: np.ndarray, log_b: np.ndarray) -> tuple[np.ndarray, float]:
    """Log alpha (T, K) and total log-likelihood log P(X | params)."""
    T, K = log_b.shape
    log_alpha = np.empty((T, K))
    log_alpha[0] = log_pi + log_b[0]
    for t in range(1, T):
        # log_alpha[t, j] = logsumexp_i (log_alpha[t-1, i] + log_A[i, j]) + log_b[t, j]
        log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0) + log_b[t]
    log_lik = logsumexp(log_alpha[-1])
    return log_alpha, float(log_lik)


def _backward(log_A: np.ndarray, log_b: np.ndarray) -> np.ndarray:
    """Log beta (T, K)."""
    T, K = log_b.shape
    log_beta = np.zeros((T, K))  # log(1) = 0 at the final step
    for t in range(T - 2, -1, -1):
        # log_beta[t, i] = logsumexp_j (log_A[i, j] + log_b[t+1, j] + log_beta[t+1, j])
        log_beta[t] = logsumexp(log_A + (log_b[t + 1] + log_beta[t + 1])[None, :], axis=1)
    return log_beta


# --------------------------------------------------------------------------- #
# EM (Baum-Welch)
# --------------------------------------------------------------------------- #

def _kmeans_init(X: np.ndarray, K: int, seed: int) -> np.ndarray:
    """K-means++ style init for state means. Light-touch: 20 Lloyd iterations."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    # k-means++ seeding
    idx = [rng.integers(0, n)]
    for _ in range(K - 1):
        dists = np.min(
            np.linalg.norm(X[:, None, :] - X[idx][None, :, :], axis=-1), axis=1
        ) ** 2
        probs = dists / dists.sum()
        idx.append(int(rng.choice(n, p=probs)))
    centers = X[idx].copy()
    for _ in range(20):
        d = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=-1)
        labels = d.argmin(axis=1)
        for k in range(K):
            mask = labels == k
            if mask.any():
                centers[k] = X[mask].mean(axis=0)
    return centers


def _init_params(X: np.ndarray, cfg: HMMConfig) -> HMMParams:
    K, D = cfg.K, X.shape[1]
    mu = _kmeans_init(X, K, cfg.seed)
    # Covariance: data covariance for each state initially.
    cov = np.cov(X.T) + cfg.cov_ridge * np.eye(D)
    Sigma = np.tile(cov[None, :, :], (K, 1, 1))
    pi = np.full(K, 1.0 / K)
    A = np.full((K, K), 1.0 / K)
    return HMMParams(pi=pi, A=A, mu=mu, Sigma=Sigma)


def fit_hmm(X: np.ndarray, cfg: HMMConfig | None = None) -> HMMParams:
    """Fit a Gaussian-emission HMM via EM. Returns the final HMMParams.

    `X`: (T, D) observation matrix. Caller is responsible for dropping NaN rows.
    """
    cfg = cfg or HMMConfig()
    if np.any(~np.isfinite(X)):
        raise ValueError("X contains non-finite values; drop NaNs before fitting")
    T, D = X.shape
    params = _init_params(X, cfg)
    log_pi = np.log(params.pi)
    log_A = np.log(params.A)

    prev_ll = -np.inf
    for it in range(cfg.max_iter):
        # E-step
        log_b = _all_emission_logp(X, params.mu, params.Sigma)
        log_alpha, log_lik = _forward(log_pi, log_A, log_b)
        log_beta = _backward(log_A, log_b)

        log_gamma = log_alpha + log_beta - log_lik       # (T, K) log P(s_t = k | X)
        gamma = np.exp(log_gamma)

        # log_xi[t, i, j] = log_alpha[t, i] + log_A[i, j] + log_b[t+1, j] + log_beta[t+1, j] - log_lik
        log_xi = (
            log_alpha[:-1, :, None]
            + log_A[None, :, :]
            + log_b[1:, None, :]
            + log_beta[1:, None, :]
            - log_lik
        )
        xi = np.exp(log_xi)  # (T-1, K, K)

        # M-step
        new_pi = gamma[0] / gamma[0].sum()
        denom = xi.sum(axis=(0, 2))                       # (K,)
        # Guard against empty states.
        denom = np.where(denom > 0, denom, 1.0)
        new_A = xi.sum(axis=0) / denom[:, None]
        new_A = new_A / new_A.sum(axis=1, keepdims=True)

        gamma_sum = gamma.sum(axis=0)                     # (K,)
        gamma_sum = np.where(gamma_sum > 0, gamma_sum, 1.0)
        new_mu = (gamma.T @ X) / gamma_sum[:, None]       # (K, D)

        new_Sigma = np.empty((cfg.K, D, D))
        for k in range(cfg.K):
            diff = X - new_mu[k]
            weighted = (gamma[:, k][:, None] * diff)
            new_Sigma[k] = weighted.T @ diff / gamma_sum[k]
            new_Sigma[k] += cfg.cov_ridge * np.eye(D)

        params = HMMParams(
            pi=new_pi,
            A=new_A,
            mu=new_mu,
            Sigma=new_Sigma,
            log_likelihood_history=params.log_likelihood_history + [log_lik],
        )
        log_pi = np.log(np.clip(new_pi, 1e-300, None))
        log_A = np.log(np.clip(new_A, 1e-300, None))

        improvement = log_lik - prev_ll
        if cfg.verbose:
            logger.info("iter %3d  log-lik %.4f  delta %.2e", it, log_lik, improvement)
        if 0 < improvement < cfg.tol:
            break
        prev_ll = log_lik

    return params


# --------------------------------------------------------------------------- #
# Inference utilities (used after fit)
# --------------------------------------------------------------------------- #

def filtered_state_logprobs(X: np.ndarray, params: HMMParams) -> np.ndarray:
    """log P(s_t | X_{1:t}) — the online "filtered" marginal at each step.

    This is what the particle filter will be compared to in Phase 4.
    """
    log_b = _all_emission_logp(X, params.mu, params.Sigma)
    log_pi = np.log(np.clip(params.pi, 1e-300, None))
    log_A = np.log(np.clip(params.A, 1e-300, None))
    T, K = log_b.shape
    log_alpha = np.empty((T, K))
    log_alpha[0] = log_pi + log_b[0]
    for t in range(1, T):
        log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0) + log_b[t]
    return log_alpha - logsumexp(log_alpha, axis=1, keepdims=True)


def smoothed_state_logprobs(X: np.ndarray, params: HMMParams) -> tuple[np.ndarray, float]:
    """log P(s_t | X_{1:T}) via forward-backward, plus total log-likelihood."""
    log_b = _all_emission_logp(X, params.mu, params.Sigma)
    log_pi = np.log(np.clip(params.pi, 1e-300, None))
    log_A = np.log(np.clip(params.A, 1e-300, None))
    log_alpha, log_lik = _forward(log_pi, log_A, log_b)
    log_beta = _backward(log_A, log_b)
    log_gamma = log_alpha + log_beta - log_lik
    return log_gamma, log_lik


def viterbi(X: np.ndarray, params: HMMParams) -> np.ndarray:
    """Most likely state sequence (T,) by Viterbi decoding in log space."""
    log_b = _all_emission_logp(X, params.mu, params.Sigma)
    log_pi = np.log(np.clip(params.pi, 1e-300, None))
    log_A = np.log(np.clip(params.A, 1e-300, None))
    T, K = log_b.shape
    delta = np.empty((T, K))
    psi = np.empty((T, K), dtype=int)
    delta[0] = log_pi + log_b[0]
    for t in range(1, T):
        scores = delta[t - 1][:, None] + log_A
        psi[t] = scores.argmax(axis=0)
        delta[t] = scores.max(axis=0) + log_b[t]
    states = np.empty(T, dtype=int)
    states[-1] = int(delta[-1].argmax())
    for t in range(T - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]
    return states


def bic(X: np.ndarray, params: HMMParams, log_lik: float) -> float:
    """Bayesian Information Criterion. Lower is better."""
    T, D = X.shape
    K = params.mu.shape[0]
    # Free parameters: K-1 for pi, K*(K-1) for A, K*D for means, K*D*(D+1)/2 for covs.
    n_params = (K - 1) + K * (K - 1) + K * D + K * D * (D + 1) // 2
    return n_params * np.log(T) - 2.0 * log_lik
