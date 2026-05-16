"""Tests for the bootstrap particle filter.

The most important test is "with N large, the PF's filtered marginals match
the exact HMM forward-pass marginals." That validates both the propagation
and the weighting.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from regime_model.inference.particle_filter import (
    _systematic_resample,
    particle_filter,
)
from regime_model.models.hmm_baseline import (
    HMMParams,
    filtered_state_logprobs,
    smoothed_state_logprobs,
)


def _simulate_hmm(
    pi: np.ndarray,
    A: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    T: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
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
# Systematic resampling unit test
# --------------------------------------------------------------------------- #

def test_systematic_resample_concentrates_on_high_weight_particle() -> None:
    """If one particle has weight ~1 and others 0, all resampled indices land on it."""
    import jax.random as jr
    weights = np.zeros(100)
    weights[42] = 1.0
    indices = np.asarray(_systematic_resample(jr.PRNGKey(0), jnp.asarray(weights)))
    assert (indices == 42).all()


def test_systematic_resample_uniform_weights_stratifies() -> None:
    """With uniform weights, indices should be a stratified sample of [0, N)."""
    import jax.random as jr
    weights = np.full(20, 1.0 / 20)
    indices = np.asarray(_systematic_resample(jr.PRNGKey(7), jnp.asarray(weights)))
    # Should hit each particle at most twice in stratified resampling.
    assert indices.min() >= 0 and indices.max() < 20
    assert len(indices) == 20


# --------------------------------------------------------------------------- #
# PF marginals match exact HMM filtered marginals
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("N", [5_000])
def test_pf_filtered_matches_hmm_forward(N: int) -> None:
    """Bootstrap PF with N large should match exact HMM forward pass marginals."""
    pi = np.array([0.3, 0.4, 0.3])
    A = np.array([[0.95, 0.03, 0.02],
                  [0.04, 0.93, 0.03],
                  [0.02, 0.05, 0.93]])
    mu = np.array([[-2.0, 0.0],
                   [0.0, 0.0],
                   [2.0, 0.5]])
    Sigma = np.tile(np.eye(2)[None, :, :], (3, 1, 1))
    scale_tril = np.linalg.cholesky(Sigma)

    T = 800
    X, _ = _simulate_hmm(pi, A, mu, Sigma, T=T, seed=0)

    # Exact HMM filtered marginals.
    params = HMMParams(pi=pi, A=A, mu=mu, Sigma=Sigma)
    log_filt_exact = np.asarray(filtered_state_logprobs(X, params))
    filt_exact = np.exp(log_filt_exact)

    # PF filtered marginals.
    pf = particle_filter(X, pi, A, mu, scale_tril, n_particles=N, seed=0)

    # Mean absolute deviation per state should be small (Monte Carlo error ~ 1/sqrt(N)).
    mad = np.mean(np.abs(filt_exact - pf.state_probs))
    assert mad < 0.02, f"PF filtered marginals deviate too much from HMM exact: MAD={mad:.4f}"


def test_pf_log_lik_matches_hmm_log_lik() -> None:
    """The PF's running log-lik estimate should be close to the exact HMM log-lik."""
    pi = np.array([0.5, 0.5])
    A = np.array([[0.95, 0.05], [0.05, 0.95]])
    mu = np.array([[-2.0], [2.0]])
    Sigma = np.array([[[1.0]], [[1.0]]])
    scale_tril = np.linalg.cholesky(Sigma)
    T = 400
    X, _ = _simulate_hmm(pi, A, mu, Sigma, T=T, seed=2)

    params = HMMParams(pi=pi, A=A, mu=mu, Sigma=Sigma)
    _, ll_exact = smoothed_state_logprobs(X, params)

    pf = particle_filter(X, pi, A, mu, scale_tril, n_particles=10_000, seed=0)

    # PF log-lik should be within ~1% of the exact HMM log-lik.
    rel_err = abs(pf.log_lik - ll_exact) / abs(ll_exact)
    assert rel_err < 0.01, f"PF log-lik {pf.log_lik:.2f} vs exact {ll_exact:.2f}, rel err {rel_err:.4f}"


def test_pf_resamples_at_least_sometimes_and_keeps_ess_finite() -> None:
    pi = np.array([0.5, 0.5])
    A = np.array([[0.95, 0.05], [0.05, 0.95]])
    mu = np.array([[-3.0, 0.0], [3.0, 0.0]])  # well separated → tight likelihood → ESS drops
    Sigma = np.tile(0.5 * np.eye(2)[None, :, :], (2, 1, 1))
    scale_tril = np.linalg.cholesky(Sigma)
    T = 300
    X, _ = _simulate_hmm(pi, A, mu, Sigma, T=T, seed=3)

    pf = particle_filter(X, pi, A, mu, scale_tril, n_particles=2_000, seed=0)
    # ESS should always be positive (no degeneracy).
    assert (pf.ess > 0).all()
    # We expect at least a few resample events on this configuration.
    assert pf.resampled.sum() > 0
    # ESS recorded is post-weight, pre-resample, so it is bounded above by N
    # but typically lower (re-weighting always reduces ESS). On a persistent
    # regime the high-water mark should still be close to N.
    assert pf.ess.max() <= 2_000 * (1 + 1e-9)
    assert pf.ess.max() >= 0.85 * 2_000


def test_pf_handles_single_state_degenerate() -> None:
    """K=1 case: one state, all probability mass should always be on it."""
    pi = np.array([1.0])
    A = np.array([[1.0]])
    mu = np.array([[0.0]])
    Sigma = np.array([[[1.0]]])
    scale_tril = np.linalg.cholesky(Sigma)
    rng = np.random.default_rng(5)
    X = rng.normal(size=(50, 1))
    pf = particle_filter(X, pi, A, mu, scale_tril, n_particles=200, seed=0)
    assert np.allclose(pf.state_probs, 1.0)


def test_pf_marginals_sum_to_one_at_every_step() -> None:
    pi = np.array([0.3, 0.4, 0.3])
    A = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
    mu = np.array([[-2.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
    Sigma = np.tile(np.eye(2)[None, :, :], (3, 1, 1))
    scale_tril = np.linalg.cholesky(Sigma)
    X, _ = _simulate_hmm(pi, A, mu, Sigma, T=200, seed=1)
    pf = particle_filter(X, pi, A, mu, scale_tril, n_particles=1_000, seed=0)
    assert np.allclose(pf.state_probs.sum(axis=1), 1.0, atol=1e-9)
