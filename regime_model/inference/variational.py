"""Stochastic Variational Inference for the Bayesian regime model.

Uses NumPyro's `AutoNormal` guide (mean-field on the unconstrained parameter
space) and Adam with the spec's learning rate of 1e-3 for 10k steps. The
trainer also exposes posterior parameter samples and a numpy-friendly
`PosteriorPointEstimate` for downstream code (allocation, particle filter)
that wants a single set of regime parameters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.initialization import init_to_median

from regime_model.models.bayesian_regime import (
    ModelConfig,
    build_scale_tril,
    forward_backward,
    regime_model,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Config + result containers
# --------------------------------------------------------------------------- #

@dataclass
class SVIConfig:
    n_steps: int = 10_000
    learning_rate: float = 1e-3
    seed: int = 0
    init_scale: float = 0.1   # AutoNormal init scale on unconstrained params
    log_every: int = 500
    progress: bool = True


@dataclass
class PosteriorPointEstimate:
    """Posterior-mean parameters in numpy. Plug into HMM utilities directly."""
    pi: np.ndarray         # (K,)
    A: np.ndarray          # (K, K)
    mu: np.ndarray         # (K, D)
    Sigma: np.ndarray      # (K, D, D)
    scale_tril: np.ndarray # (K, D, D)


@dataclass
class SVIResult:
    losses: np.ndarray                  # (n_steps,) ELBO loss history
    params: dict[str, Any]              # Raw guide params (jax arrays)
    samples: dict[str, np.ndarray]      # Posterior samples in constrained space
    point_estimate: PosteriorPointEstimate
    cfg: SVIConfig = field(default_factory=SVIConfig)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def fit_svi(
    y: np.ndarray | jnp.ndarray,
    model_cfg: ModelConfig | None = None,
    svi_cfg: SVIConfig | None = None,
    n_posterior_samples: int = 200,
) -> SVIResult:
    """Fit the Bayesian regime model with SVI + AutoNormal guide.

    Returns an SVIResult with ELBO trace, posterior samples, and a
    PosteriorPointEstimate suitable for downstream HMM utilities.
    """
    model_cfg = model_cfg or ModelConfig()
    svi_cfg = svi_cfg or SVIConfig()
    y_jax = jnp.asarray(y)

    # init_to_median is critical for the LKJCholesky prior: AutoNormal's
    # default init_to_uniform(radius=2) maps through CorrCholeskyTransform to
    # near-singular Cholesky factors (diagonal entries ~1e-4 in 14 dims),
    # which makes the initial gauss_logp explode by ~10 orders of magnitude.
    guide = AutoNormal(
        regime_model,
        init_loc_fn=init_to_median(num_samples=20),
        init_scale=svi_cfg.init_scale,
    )
    optimizer = numpyro.optim.Adam(svi_cfg.learning_rate)
    svi = SVI(regime_model, guide, optimizer, loss=Trace_ELBO())

    rng = jax.random.PRNGKey(svi_cfg.seed)
    rng, init_rng = jax.random.split(rng)
    state = svi.init(init_rng, y_jax, model_cfg)

    losses = np.empty(svi_cfg.n_steps, dtype=np.float64)

    @jax.jit
    def train_step(state):
        return svi.update(state, y_jax, model_cfg)

    for step in range(svi_cfg.n_steps):
        state, loss = train_step(state)
        losses[step] = float(loss)
        if svi_cfg.progress and step % svi_cfg.log_every == 0:
            logger.info("SVI step %5d  loss=%12.2f", step, losses[step])

    # Final params + posterior samples.
    params = svi.get_params(state)
    rng, sample_rng = jax.random.split(rng)
    samples = guide.sample_posterior(sample_rng, params, sample_shape=(n_posterior_samples,))
    samples_np = {k: np.asarray(v) for k, v in samples.items()}

    point_estimate = _posterior_mean_estimate(samples_np)
    return SVIResult(
        losses=losses,
        params=params,
        samples=samples_np,
        point_estimate=point_estimate,
        cfg=svi_cfg,
    )


# --------------------------------------------------------------------------- #
# Posterior summarization
# --------------------------------------------------------------------------- #

def _posterior_mean_estimate(samples: dict[str, np.ndarray]) -> PosteriorPointEstimate:
    pi = samples["pi"].mean(axis=0)
    A = samples["A"].mean(axis=0)
    # A may end up not exactly stochastic after averaging — renormalize rows.
    A = A / A.sum(axis=1, keepdims=True)

    mu = samples["mu"].mean(axis=0)        # (K, D)
    sigma = samples["sigma"].mean(axis=0)  # (K, D)
    L_corr = samples["L_corr"].mean(axis=0)  # (K, D, D)
    # The mean of LKJ Cholesky samples is not itself a valid Cholesky factor,
    # but for downstream point-estimate use it's close enough; we re-build a
    # PSD covariance from sigma and L_corr.
    scale_tril = np.array(build_scale_tril(jnp.asarray(sigma), jnp.asarray(L_corr)))
    Sigma = scale_tril @ scale_tril.transpose(0, 2, 1)
    return PosteriorPointEstimate(
        pi=pi,
        A=A,
        mu=mu,
        Sigma=Sigma,
        scale_tril=scale_tril,
    )


def smoothed_posterior(
    y: np.ndarray | jnp.ndarray,
    pe: PosteriorPointEstimate,
) -> tuple[np.ndarray, float]:
    """Forward-backward smoothed P(s_t | y_{1:T}) using the point estimate.

    Returns (gamma in probability space (T, K), total log-likelihood).
    """
    log_pi = jnp.log(jnp.asarray(pe.pi))
    log_A = jnp.log(jnp.asarray(pe.A))
    log_gamma, log_lik = forward_backward(
        jnp.asarray(y),
        log_pi,
        log_A,
        jnp.asarray(pe.mu),
        jnp.asarray(pe.scale_tril),
    )
    return np.asarray(jnp.exp(log_gamma)), float(log_lik)
