"""Tests for the JAX Bayesian regime model and SVI inference.

These tests are independent of yfinance / network. The SVI test fits on a
small synthetic 2-state HMM dataset; with a tight ELBO budget it still runs
in a few seconds.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from regime_model.inference.variational import SVIConfig, fit_svi, smoothed_posterior
from regime_model.models.bayesian_regime import (
    ModelConfig,
    forward_backward,
    forward_marginal_loglik,
    gauss_logp,
    persistence_alpha,
)
from regime_model.models.hmm_baseline import (
    HMMConfig,
    HMMParams,
    _all_emission_logp,
    _forward,
    fit_hmm,
)


def _simulate_2state(T: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pi = np.array([0.5, 0.5])
    A = np.array([[0.97, 0.03], [0.05, 0.95]])
    mu = np.array([[-3.0, 0.0], [3.0, 0.0]])
    Sigma = np.tile(0.5 * np.eye(2)[None, :, :], (2, 1, 1))
    K = 2
    states = np.empty(T, dtype=int)
    states[0] = rng.choice(K, p=pi)
    for t in range(1, T):
        states[t] = rng.choice(K, p=A[states[t - 1]])
    X = np.empty((T, 2))
    for t in range(T):
        X[t] = rng.multivariate_normal(mu[states[t]], Sigma[states[t]])
    return X


# --------------------------------------------------------------------------- #
# Forward-pass parity with HMM baseline
# --------------------------------------------------------------------------- #

def test_jax_forward_loglik_matches_numpy_hmm() -> None:
    """The JAX forward-marginal-loglik must agree with the numpy HMM forward
    pass on the same parameters."""
    rng = np.random.default_rng(0)
    K, D, T = 3, 4, 200

    pi = np.full(K, 1.0 / K)
    A = 0.85 * np.eye(K) + 0.05  # row-sum 1
    A /= A.sum(axis=1, keepdims=True)
    mu = rng.normal(size=(K, D))
    L = np.tile(np.eye(D), (K, 1, 1)) * 0.5 + rng.normal(size=(K, D, D)) * 0.05
    # Make L lower-triangular and PD
    for k in range(K):
        L[k] = np.linalg.cholesky(L[k] @ L[k].T + np.eye(D))
    Sigma = L @ L.transpose(0, 2, 1)

    X = rng.normal(size=(T, D))

    # numpy HMM forward
    log_b_np = _all_emission_logp(X, mu, Sigma)
    _, ll_np = _forward(np.log(pi), np.log(A), log_b_np)

    # jax forward
    ll_jax = float(forward_marginal_loglik(
        jnp.asarray(X), jnp.log(jnp.asarray(pi)), jnp.log(jnp.asarray(A)),
        jnp.asarray(mu), jnp.asarray(L),
    ))

    assert ll_jax == pytest.approx(ll_np, rel=1e-5)


def test_jax_forward_backward_matches_numpy_smoothed() -> None:
    rng = np.random.default_rng(1)
    K, D, T = 3, 3, 150
    pi = np.full(K, 1.0 / K)
    A = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
    mu = rng.normal(size=(K, D)) * 2.0
    Sigma = np.tile(np.eye(D), (K, 1, 1))
    L = np.linalg.cholesky(Sigma)

    X = rng.normal(size=(T, D))

    # numpy smoothed
    from regime_model.models.hmm_baseline import smoothed_state_logprobs
    params = HMMParams(pi=pi, A=A, mu=mu, Sigma=Sigma)
    log_gamma_np, ll_np = smoothed_state_logprobs(X, params)

    # jax smoothed
    log_gamma_jax, ll_jax = forward_backward(
        jnp.asarray(X), jnp.log(jnp.asarray(pi)), jnp.log(jnp.asarray(A)),
        jnp.asarray(mu), jnp.asarray(L),
    )

    assert float(ll_jax) == pytest.approx(ll_np, rel=1e-5)
    assert np.allclose(np.asarray(log_gamma_jax), log_gamma_np, atol=1e-5)


# --------------------------------------------------------------------------- #
# Persistence prior shape
# --------------------------------------------------------------------------- #

def test_persistence_alpha_has_high_diagonal() -> None:
    alpha = persistence_alpha(K=3, alpha_diag=10.0, alpha_off=1.0)
    diag = jnp.diag(alpha)
    off = alpha[~jnp.eye(3, dtype=bool)]
    assert (diag == 10.0).all()
    assert (off == 1.0).all()


# --------------------------------------------------------------------------- #
# SVI smoke + recovery
# --------------------------------------------------------------------------- #

def test_svi_runs_and_decreases_loss() -> None:
    """A 1500-step SVI run on synthetic 2-state data should reduce the loss
    by a clearly significant amount."""
    X = _simulate_2state(T=600, seed=2)
    result = fit_svi(
        X,
        model_cfg=ModelConfig(K=2, persistence_diag=10.0, persistence_off=1.0),
        svi_cfg=SVIConfig(n_steps=1500, learning_rate=5e-3, seed=0, log_every=10_000, progress=False),
        n_posterior_samples=50,
    )
    assert result.losses.shape == (1500,)
    assert np.isfinite(result.losses).all()
    early = result.losses[:50].mean()
    late = result.losses[-50:].mean()
    assert late < early - 50.0, f"loss did not improve: early={early:.1f}, late={late:.1f}"


def test_svi_recovers_state_means_on_synthetic_data() -> None:
    X = _simulate_2state(T=1000, seed=3)
    result = fit_svi(
        X,
        model_cfg=ModelConfig(K=2, persistence_diag=10.0, persistence_off=1.0),
        svi_cfg=SVIConfig(n_steps=2500, learning_rate=5e-3, seed=0, log_every=10_000, progress=False),
        n_posterior_samples=100,
    )
    pe = result.point_estimate
    # True means are ±3 in the first dim, 0 in the second.
    # Permute: closest fit-state to each true state.
    true_mu = np.array([[-3.0, 0.0], [3.0, 0.0]])
    perm = []
    for tm in true_mu:
        idx = int(np.argmin(np.linalg.norm(pe.mu - tm, axis=1)))
        perm.append(idx)
    perm = np.array(perm)
    fitted_mu = pe.mu[perm]
    assert np.allclose(fitted_mu, true_mu, atol=0.4), f"recovered means: {fitted_mu}"


def test_smoothed_posterior_runs_after_svi() -> None:
    X = _simulate_2state(T=300, seed=4)
    result = fit_svi(
        X,
        model_cfg=ModelConfig(K=2),
        svi_cfg=SVIConfig(n_steps=400, learning_rate=5e-3, seed=0, log_every=10_000, progress=False),
        n_posterior_samples=40,
    )
    gamma, ll = smoothed_posterior(X, result.point_estimate)
    assert gamma.shape == (300, 2)
    assert np.allclose(gamma.sum(axis=1), 1.0, atol=1e-5)
    assert np.isfinite(ll)
