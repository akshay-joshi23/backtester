"""Tests for the Gaussian HMM baseline.

The hardest test is "given samples from a known HMM, can the EM fit recover
the parameters?" — we use a 2-state, 2-D HMM with well-separated means so the
test is deterministic and not just measuring local-optimum luck.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import logsumexp

from regime_model.models.hmm_baseline import (
    HMMConfig,
    HMMParams,
    _all_emission_logp,
    _backward,
    _forward,
    bic,
    filtered_state_logprobs,
    fit_hmm,
    smoothed_state_logprobs,
    viterbi,
)


def _simulate_hmm(
    pi: np.ndarray,
    A: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    T: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample (X, S) from a known HMM."""
    rng = np.random.default_rng(seed)
    K = len(pi)
    states = np.empty(T, dtype=int)
    states[0] = rng.choice(K, p=pi)
    for t in range(1, T):
        states[t] = rng.choice(K, p=A[states[t - 1]])
    X = np.empty((T, mu.shape[1]))
    for t in range(T):
        X[t] = rng.multivariate_normal(mu[states[t]], Sigma[states[t]])
    return X, states


# --------------------------------------------------------------------------- #
# Forward-backward sanity
# --------------------------------------------------------------------------- #

def test_forward_likelihood_matches_brute_force_for_short_sequence() -> None:
    """For T=3, K=2 we can brute-force enumerate all 8 state paths and sum."""
    pi = np.array([0.6, 0.4])
    A = np.array([[0.7, 0.3], [0.2, 0.8]])
    mu = np.array([[0.0], [3.0]])
    Sigma = np.array([[[1.0]], [[1.0]]])
    X = np.array([[0.5], [1.5], [2.5]])

    log_b = _all_emission_logp(X, mu, Sigma)
    _, log_lik = _forward(np.log(pi), np.log(A), log_b)

    # Brute-force: sum over all state sequences
    K, T = 2, 3
    total = 0.0
    for s0 in range(K):
        for s1 in range(K):
            for s2 in range(K):
                p = pi[s0] * A[s0, s1] * A[s1, s2]
                p *= np.exp(log_b[0, s0] + log_b[1, s1] + log_b[2, s2])
                total += p
    assert log_lik == pytest.approx(np.log(total), abs=1e-10)


def test_smoothed_marginals_sum_to_one() -> None:
    pi = np.array([0.5, 0.5])
    A = np.array([[0.9, 0.1], [0.1, 0.9]])
    mu = np.array([[-1.0, 0.0], [1.0, 0.0]])
    Sigma = np.tile(np.eye(2)[None, :, :], (2, 1, 1))
    X, _ = _simulate_hmm(pi, A, mu, Sigma, T=200, seed=7)
    params = HMMParams(pi=pi, A=A, mu=mu, Sigma=Sigma)
    log_gamma, _ = smoothed_state_logprobs(X, params)
    assert np.allclose(logsumexp(log_gamma, axis=1), 0.0, atol=1e-9)


def test_log_space_handles_long_sequence_without_underflow() -> None:
    """T=10000 should not underflow because we work in log-space throughout."""
    pi = np.array([0.5, 0.5])
    A = np.array([[0.95, 0.05], [0.05, 0.95]])
    mu = np.array([[-2.0], [2.0]])
    Sigma = np.array([[[1.0]], [[1.0]]])
    X, _ = _simulate_hmm(pi, A, mu, Sigma, T=10000, seed=11)
    params = HMMParams(pi=pi, A=A, mu=mu, Sigma=Sigma)
    log_gamma, log_lik = smoothed_state_logprobs(X, params)
    assert np.isfinite(log_lik)
    assert np.all(np.isfinite(log_gamma))


# --------------------------------------------------------------------------- #
# EM recovery on synthetic data
# --------------------------------------------------------------------------- #

def test_em_recovers_known_2state_hmm() -> None:
    """2-state HMM with well-separated means; EM should recover params."""
    pi_true = np.array([0.5, 0.5])
    A_true = np.array([[0.95, 0.05], [0.10, 0.90]])
    mu_true = np.array([[-3.0, 0.0], [3.0, 0.0]])
    Sigma_true = np.tile(0.5 * np.eye(2)[None, :, :], (2, 1, 1))

    X, _ = _simulate_hmm(pi_true, A_true, mu_true, Sigma_true, T=4000, seed=42)
    params = fit_hmm(X, HMMConfig(K=2, max_iter=200, tol=1e-6, seed=0))

    # Match fitted states to true states by mean (EM has label-switching).
    perm = []
    for true_mu in mu_true:
        idx = int(np.argmin(np.linalg.norm(params.mu - true_mu, axis=1)))
        perm.append(idx)
    perm = np.array(perm)

    fitted_mu = params.mu[perm]
    fitted_A = params.A[perm][:, perm]

    assert np.allclose(fitted_mu, mu_true, atol=0.15)
    # Diagonal of A is the persistence probability — most important to recover.
    assert np.allclose(np.diag(fitted_A), np.diag(A_true), atol=0.05)


def test_em_log_likelihood_is_monotone_increasing() -> None:
    pi = np.array([0.5, 0.5])
    A = np.array([[0.9, 0.1], [0.1, 0.9]])
    mu = np.array([[-2.0, 0.0], [2.0, 0.0]])
    Sigma = np.tile(np.eye(2)[None, :, :], (2, 1, 1))
    X, _ = _simulate_hmm(pi, A, mu, Sigma, T=1500, seed=1)
    params = fit_hmm(X, HMMConfig(K=2, max_iter=80, tol=0.0, seed=0))
    history = np.array(params.log_likelihood_history)
    diffs = np.diff(history)
    # EM is guaranteed to increase the data log-likelihood at every step
    # (up to numerical noise).
    assert (diffs >= -1e-6).all(), f"EM not monotone: min diff {diffs.min():.3e}"


# --------------------------------------------------------------------------- #
# Inference utilities
# --------------------------------------------------------------------------- #

def test_viterbi_matches_majority_state_in_strong_separation_regime() -> None:
    pi = np.array([1.0, 0.0])
    A = np.array([[1.0, 0.0], [0.0, 1.0]])  # absorbing — stays in state 0
    mu = np.array([[-5.0], [5.0]])
    Sigma = np.array([[[0.1]], [[0.1]]])
    X, true_states = _simulate_hmm(pi, A, mu, Sigma, T=100, seed=3)
    params = HMMParams(pi=pi, A=A, mu=mu, Sigma=Sigma)
    decoded = viterbi(X, params)
    assert (decoded == true_states).mean() > 0.99


def test_filtered_logprobs_normalize_to_one() -> None:
    pi = np.array([0.5, 0.5])
    A = np.array([[0.9, 0.1], [0.1, 0.9]])
    mu = np.array([[-1.0], [1.0]])
    Sigma = np.array([[[1.0]], [[1.0]]])
    X, _ = _simulate_hmm(pi, A, mu, Sigma, T=300, seed=5)
    params = HMMParams(pi=pi, A=A, mu=mu, Sigma=Sigma)
    flp = filtered_state_logprobs(X, params)
    assert np.allclose(logsumexp(flp, axis=1), 0.0, atol=1e-9)


def test_bic_penalizes_extra_states() -> None:
    """K=3 fit on data generated from a 2-state HMM should have higher BIC than K=2."""
    pi = np.array([0.5, 0.5])
    A = np.array([[0.9, 0.1], [0.1, 0.9]])
    mu = np.array([[-3.0, 0.0], [3.0, 0.0]])
    Sigma = np.tile(0.5 * np.eye(2)[None, :, :], (2, 1, 1))
    X, _ = _simulate_hmm(pi, A, mu, Sigma, T=2000, seed=9)

    p2 = fit_hmm(X, HMMConfig(K=2, max_iter=100, tol=1e-6, seed=0))
    p3 = fit_hmm(X, HMMConfig(K=3, max_iter=100, tol=1e-6, seed=0))

    bic2 = bic(X, p2, p2.log_likelihood_history[-1])
    bic3 = bic(X, p3, p3.log_likelihood_history[-1])
    assert bic2 < bic3, f"BIC failed to favor true K=2: BIC(K=2)={bic2:.1f}, BIC(K=3)={bic3:.1f}"
