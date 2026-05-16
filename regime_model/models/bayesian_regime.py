"""Hierarchical Bayesian regime-switching model in NumPyro.

Generative process (matches the spec):

    pi      ~ Dirichlet(1)                    # initial state distribution
    A_i,:   ~ Dirichlet(alpha_i)              # one row per source state, with
                                              # persistence prior alpha_diag=10,
                                              # alpha_off=1
    mu_0    ~ Normal(0, 1)^D                  # global mean hyperprior
    tau     ~ HalfCauchy(1)                   # global scale hyperprior
    mu_k    ~ Normal(mu_0, tau)^D             # per-state mean
    sigma_k ~ HalfCauchy(2.5)^D               # per-state marginal scale
    L_k     ~ LKJCholesky(D, eta=2)           # per-state correlation factor
    Sigma_k = diag(sigma_k) L_k L_k^T diag(sigma_k)

    s_t | s_{t-1} ~ Categorical(A[s_{t-1}, :])
    y_t | s_t = k ~ MultivariateNormal(mu_k, Sigma_k)

The discrete chain s_{1:T} is marginalized analytically using a JAX-vectorized
forward pass. SVI (in inference/variational.py) only learns the continuous
parameters; the regime posterior is recovered post-hoc via forward-backward.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import logsumexp


# --------------------------------------------------------------------------- #
# Forward marginal log-likelihood (vectorized, JAX)
# --------------------------------------------------------------------------- #

def gauss_logp(y: jnp.ndarray, mu: jnp.ndarray, scale_tril: jnp.ndarray) -> jnp.ndarray:
    """log N(y_t | mu_k, Sigma_k = scale_tril_k scale_tril_k^T) for each (t, k).

    y           (T, D)
    mu          (K, D)
    scale_tril  (K, D, D) lower-triangular Cholesky factors of Sigma
    Returns     (T, K)
    """
    T, D = y.shape
    K = mu.shape[0]
    # diff[t, k, d] = y[t, d] - mu[k, d]
    diff = y[:, None, :] - mu[None, :, :]
    # Solve scale_tril_k z = diff[t, k] for each (t, k). Vmap over both.
    def solve_one(L, d):  # L: (D, D), d: (D,)
        return solve_triangular(L, d, lower=True)
    z = jax.vmap(jax.vmap(solve_one, in_axes=(None, 0)), in_axes=(0, 1), out_axes=1)(
        scale_tril, diff
    )  # (T, K, D)
    quad = jnp.einsum("tkd,tkd->tk", z, z)
    log_det = 2.0 * jnp.log(jnp.diagonal(scale_tril, axis1=-2, axis2=-1)).sum(-1)  # (K,)
    return -0.5 * (D * jnp.log(2.0 * jnp.pi) + log_det[None, :] + quad)


def forward_marginal_loglik(
    y: jnp.ndarray,
    log_pi: jnp.ndarray,
    log_A: jnp.ndarray,
    mu: jnp.ndarray,
    scale_tril: jnp.ndarray,
) -> jnp.ndarray:
    """log p(y_{1:T} | params), marginalizing over states.

    Implements the standard forward recursion in log space using lax.scan.
    """
    log_b = gauss_logp(y, mu, scale_tril)  # (T, K)

    def step(log_alpha_prev, log_b_t):
        log_alpha = logsumexp(log_alpha_prev[:, None] + log_A, axis=0) + log_b_t
        return log_alpha, None

    log_alpha_init = log_pi + log_b[0]
    log_alpha_final, _ = jax.lax.scan(step, log_alpha_init, log_b[1:])
    return logsumexp(log_alpha_final)


def forward_backward(
    y: jnp.ndarray,
    log_pi: jnp.ndarray,
    log_A: jnp.ndarray,
    mu: jnp.ndarray,
    scale_tril: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Smoothed log P(s_t = k | y_{1:T}) and total log-likelihood. Returns (log_gamma, log_lik)."""
    log_b = gauss_logp(y, mu, scale_tril)
    T, K = log_b.shape

    def fwd_step(log_alpha_prev, log_b_t):
        log_alpha = logsumexp(log_alpha_prev[:, None] + log_A, axis=0) + log_b_t
        return log_alpha, log_alpha

    log_alpha_init = log_pi + log_b[0]
    log_alpha_final, log_alphas_rest = jax.lax.scan(fwd_step, log_alpha_init, log_b[1:])
    log_alphas = jnp.concatenate([log_alpha_init[None, :], log_alphas_rest])  # (T, K)
    log_lik = logsumexp(log_alphas[-1])

    def bwd_step(log_beta_next, log_b_next):
        log_beta = logsumexp(log_A + (log_b_next + log_beta_next)[None, :], axis=1)
        return log_beta, log_beta

    log_beta_init = jnp.zeros(K)
    _, log_betas_rest = jax.lax.scan(
        bwd_step, log_beta_init, log_b[1:][::-1]
    )
    log_betas_rest = log_betas_rest[::-1]  # back to forward time order
    log_betas = jnp.concatenate([log_betas_rest, log_beta_init[None, :]])  # (T, K)

    log_gamma = log_alphas + log_betas - log_lik
    return log_gamma, log_lik


# --------------------------------------------------------------------------- #
# Persistence prior
# --------------------------------------------------------------------------- #

def persistence_alpha(K: int, alpha_diag: float = 10.0, alpha_off: float = 1.0) -> jnp.ndarray:
    """Per-row Dirichlet concentration with high mass on the diagonal.

    Returns an array of shape (K, K) where row i is alpha_off everywhere with
    alpha_diag on column i. Each row is a valid Dirichlet concentration.
    """
    eye = jnp.eye(K)
    return alpha_off * (1.0 - eye) + alpha_diag * eye


# --------------------------------------------------------------------------- #
# NumPyro model
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ModelConfig:
    K: int = 3
    persistence_diag: float = 10.0
    persistence_off: float = 1.0
    mu_hyper_scale: float = 1.0    # τ_0 for the hyperprior on μ_0
    sigma_scale_prior: float = 2.5 # half-Cauchy scale for per-state σ
    lkj_concentration: float = 2.0


def regime_model(y: jnp.ndarray, cfg: ModelConfig | None = None) -> None:
    """NumPyro model. Call inside an SVI run or NUTS kernel."""
    cfg = cfg or ModelConfig()
    T, D = y.shape
    K = cfg.K

    # Initial state distribution.
    pi = numpyro.sample("pi", dist.Dirichlet(jnp.ones(K)))

    # Transition matrix. One Dirichlet per row, each with a persistence prior.
    alpha_rows = persistence_alpha(K, cfg.persistence_diag, cfg.persistence_off)
    # Use a plate so each row is its own Dirichlet draw.
    with numpyro.plate("rows", K):
        A = numpyro.sample("A", dist.Dirichlet(alpha_rows))

    # Hierarchical mean.
    mu_0 = numpyro.sample("mu_0", dist.Normal(0.0, cfg.mu_hyper_scale).expand([D]).to_event(1))
    tau = numpyro.sample("tau", dist.HalfCauchy(cfg.mu_hyper_scale))
    with numpyro.plate("states_mu", K):
        mu = numpyro.sample("mu", dist.Normal(mu_0, tau).to_event(1))

    # Per-state covariance: half-Cauchy scale + LKJ correlation.
    with numpyro.plate("states_cov", K):
        sigma = numpyro.sample("sigma", dist.HalfCauchy(cfg.sigma_scale_prior).expand([D]).to_event(1))
        L_corr = numpyro.sample("L_corr", dist.LKJCholesky(D, concentration=cfg.lkj_concentration))

    # Build scale_tril per state: scale_tril_k = diag(sigma_k) @ L_corr_k.
    scale_tril = sigma[..., None] * L_corr

    # Marginal log-likelihood factor (forward pass).
    log_pi = jnp.log(pi)
    log_A = jnp.log(A)
    log_lik = forward_marginal_loglik(y, log_pi, log_A, mu, scale_tril)
    numpyro.factor("y_likelihood", log_lik)


# --------------------------------------------------------------------------- #
# Posterior parameter -> scale_tril helper (used after SVI fit)
# --------------------------------------------------------------------------- #

def build_scale_tril(sigma: jnp.ndarray, L_corr: jnp.ndarray) -> jnp.ndarray:
    """scale_tril_k = diag(sigma_k) @ L_corr_k, batched over k."""
    return sigma[..., None] * L_corr
