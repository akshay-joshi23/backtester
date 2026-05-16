"""Bootstrap particle filter for the discrete-state regime HMM.

This is the spec's "online inference" centerpiece. The filter assumes the
regime parameters (pi, A, mu, Sigma) are fixed — they come from the VI fit on
the training window — and only the regime sequence s_{1:T} is uncertain.

Algorithm (per step t):

    1. Propagate:  s_t^(i) ~ Categorical(A[s_{t-1}^(i), :])     for i=1..N
    2. Weight:     log w_t^(i) = log w_{t-1}^(i) + log p(y_t | s_t^(i))
                   (then normalize so sum_i w_t^(i) = 1)
    3. ESS check:  if 1 / sum_i w_t^(i)^2 < ess_threshold * N → resample
                   (systematic resampling, weights reset to 1/N)

Implementation notes:
    - Per-step emission likelihood is computed once per state (K, not N) and
      gathered to particles, so cost is O(N + K * D^2) per step.
    - The whole filter is wrapped in lax.scan and JIT-compiled.
    - ESS is computed in log space to avoid underflow:
          log ESS = -logsumexp(2 * log_w_normalized)
    - Resample/no-resample is a lax.cond over each step.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import logsumexp


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #

@dataclass
class PFResult:
    state_probs: np.ndarray      # (T, K) filtered marginal P(s_t | y_{1:t})
    log_lik: float               # log P(y_{1:T})
    ess: np.ndarray              # (T,) ESS at each step (post-weight, pre-resample)
    resampled: np.ndarray        # (T,) bool — was this step resampled


@dataclass
class PFState:
    """Mutable state carried between segments of a walk-forward run."""
    particles: np.ndarray   # (N,) int regime indices
    log_weights: np.ndarray # (N,) normalized log weights
    rng_seed: int           # next seed to use


# --------------------------------------------------------------------------- #
# Per-step kernels
# --------------------------------------------------------------------------- #

def _emission_logp(y_t: jnp.ndarray, mu: jnp.ndarray, scale_tril: jnp.ndarray) -> jnp.ndarray:
    """log N(y_t | mu_k, Sigma_k) for k=0..K-1. Returns (K,)."""
    diff = y_t - mu  # (K, D)
    z = jax.vmap(lambda L, d: solve_triangular(L, d, lower=True))(scale_tril, diff)
    quad = (z ** 2).sum(-1)  # (K,)
    log_det = 2.0 * jnp.log(jnp.diagonal(scale_tril, axis1=-2, axis2=-1)).sum(-1)
    D = y_t.shape[0]
    return -0.5 * (D * jnp.log(2.0 * jnp.pi) + log_det + quad)


def _systematic_resample(rng: jax.Array, weights: jnp.ndarray) -> jnp.ndarray:
    """Systematic resampling. Returns N indices into the original particle array.

    Standard procedure: draw u_0 ~ Uniform(0, 1/N), then u_n = u_0 + n/N for
    n in [0, N). The n-th resampled particle is the smallest index j with
    cumsum(weights)[j] >= u_n.
    """
    N = weights.shape[0]
    u0 = random.uniform(rng) / N
    u = u0 + jnp.arange(N) / N  # (N,)
    cumsum = jnp.cumsum(weights)
    return jnp.searchsorted(cumsum, u)


def _log_ess(log_w: jnp.ndarray) -> jnp.ndarray:
    """log of ESS = 1 / sum_i w_i^2, given normalized log weights."""
    return -logsumexp(2.0 * log_w)


# --------------------------------------------------------------------------- #
# Filter
# --------------------------------------------------------------------------- #

def particle_filter(
    y: np.ndarray | jnp.ndarray,
    pi: np.ndarray | jnp.ndarray,
    A: np.ndarray | jnp.ndarray,
    mu: np.ndarray | jnp.ndarray,
    scale_tril: np.ndarray | jnp.ndarray,
    n_particles: int = 10_000,
    seed: int = 0,
    ess_threshold_frac: float = 0.5,
) -> PFResult:
    """Bootstrap particle filter for a discrete-state regime HMM.

    Parameters
    ----------
    y : (T, D) observations
    pi : (K,) initial state distribution
    A : (K, K) transition matrix (rows sum to 1)
    mu : (K, D) state means
    scale_tril : (K, D, D) lower-triangular Cholesky factors of state covariances
    n_particles : N (default 10_000 per spec)
    seed : PRNG seed
    ess_threshold_frac : resample when ESS < threshold * N (default 0.5 per spec)
    """
    y = jnp.asarray(y)
    pi = jnp.asarray(pi)
    A = jnp.asarray(A)
    mu = jnp.asarray(mu)
    scale_tril = jnp.asarray(scale_tril)

    T, D = y.shape
    K = pi.shape[0]
    N = n_particles
    log_pi = jnp.log(jnp.clip(pi, 1e-300, None))
    log_A = jnp.log(jnp.clip(A, 1e-300, None))
    log_thresh = jnp.log(ess_threshold_frac * N)

    state_probs, log_lik, ess_arr, resampled_arr = _run_filter_jit(
        y, log_pi, log_A, mu, scale_tril, N, K, log_thresh, seed
    )
    return PFResult(
        state_probs=np.asarray(state_probs),
        log_lik=float(log_lik),
        ess=np.asarray(ess_arr),
        resampled=np.asarray(resampled_arr).astype(bool),
    )


def _run_filter_jit(
    y: jnp.ndarray,
    log_pi: jnp.ndarray,
    log_A: jnp.ndarray,
    mu: jnp.ndarray,
    scale_tril: jnp.ndarray,
    N: int,
    K: int,
    log_thresh: jnp.ndarray,
    seed: int,
):
    """Inner JIT-able filter routine."""
    rng = random.PRNGKey(seed)

    # Initial particles ~ pi, then weight by p(y_0 | s_0).
    rng, k_init = random.split(rng)
    particles_0 = random.categorical(k_init, log_pi, shape=(N,))
    log_w_init = -jnp.log(N) * jnp.ones(N)
    log_b_0 = _emission_logp(y[0], mu, scale_tril)         # (K,)
    log_w_0 = log_w_init + log_b_0[particles_0]            # (N,)
    log_norm_0 = logsumexp(log_w_0)
    log_w_0 = log_w_0 - log_norm_0                         # normalize
    log_lik_0 = log_norm_0
    state_probs_0 = _aggregate_state_probs(particles_0, log_w_0, K)
    log_ess_0 = _log_ess(log_w_0)

    def step(carry, x):
        particles_prev, log_w_prev, log_lik_acc, key = carry
        y_t = x

        key, k_prop, k_res = random.split(key, 3)
        # 1. Propagate.
        log_A_per = log_A[particles_prev]  # (N, K)
        particles_new = random.categorical(k_prop, log_A_per)  # (N,)

        # 2. Reweight.
        log_b = _emission_logp(y_t, mu, scale_tril)        # (K,)
        log_w_unnorm = log_w_prev + log_b[particles_new]
        log_norm = logsumexp(log_w_unnorm)
        log_w = log_w_unnorm - log_norm
        log_lik_acc = log_lik_acc + log_norm

        # 3. ESS-based resampling.
        log_ess = _log_ess(log_w)
        do_resample = log_ess < log_thresh

        def resample_branch(_):
            indices = _systematic_resample(k_res, jnp.exp(log_w))
            new_particles = particles_new[indices]
            new_log_w = -jnp.log(N) * jnp.ones(N)
            return new_particles, new_log_w

        def keep_branch(_):
            return particles_new, log_w

        particles_out, log_w_out = jax.lax.cond(
            do_resample, resample_branch, keep_branch, operand=None
        )

        state_probs = _aggregate_state_probs(particles_out, log_w_out, K)
        return (particles_out, log_w_out, log_lik_acc, key), (state_probs, log_ess, do_resample)

    init_carry = (particles_0, log_w_0, log_lik_0, rng)
    final_carry, scan_out = jax.lax.scan(step, init_carry, y[1:])
    state_probs_rest, log_ess_rest, resampled_rest = scan_out

    state_probs = jnp.concatenate([state_probs_0[None, :], state_probs_rest])
    log_ess = jnp.concatenate([jnp.array([log_ess_0]), log_ess_rest])
    resampled = jnp.concatenate([jnp.array([False]), resampled_rest])
    log_lik = final_carry[2]
    return state_probs, log_lik, jnp.exp(log_ess), resampled


def _aggregate_state_probs(particles: jnp.ndarray, log_w: jnp.ndarray, K: int) -> jnp.ndarray:
    """sum_i w_i I[s_i = k], computed without exp where possible."""
    one_hot = jax.nn.one_hot(particles, K)  # (N, K)
    w = jnp.exp(log_w)
    return (w[:, None] * one_hot).sum(axis=0)


# --------------------------------------------------------------------------- #
# Segment runner: carries PF state across segments with potentially-different params
# --------------------------------------------------------------------------- #

def init_pf_state(
    pi: np.ndarray | jnp.ndarray,
    n_particles: int = 10_000,
    seed: int = 0,
) -> PFState:
    """Sample N particles from the initial state distribution. Equal weights."""
    pi = jnp.asarray(pi)
    log_pi = jnp.log(jnp.clip(pi, 1e-300, None))
    rng = random.PRNGKey(seed)
    particles = np.asarray(random.categorical(rng, log_pi, shape=(n_particles,)))
    log_w = np.full(n_particles, -np.log(n_particles))
    return PFState(particles=particles, log_weights=log_w, rng_seed=seed + 1)


def run_pf_segment(
    state: PFState,
    y: np.ndarray | jnp.ndarray,
    pi: np.ndarray | jnp.ndarray,
    A: np.ndarray | jnp.ndarray,
    mu: np.ndarray | jnp.ndarray,
    scale_tril: np.ndarray | jnp.ndarray,
    ess_threshold_frac: float = 0.5,
) -> tuple[PFResult, PFState]:
    """Run the bootstrap PF on a segment of observations starting from `state`.

    Unlike `particle_filter` (which initializes from pi), this consumes an
    existing particle distribution (carried from a previous segment) and
    applies the segment's params. `pi` here is unused for state init; it's
    accepted for API symmetry but ignored. (We keep `init_pf_state` separate
    for clarity.)
    """
    y = jnp.asarray(y)
    A = jnp.asarray(A)
    mu = jnp.asarray(mu)
    scale_tril = jnp.asarray(scale_tril)
    K = A.shape[0]
    N = state.particles.shape[0]
    log_A = jnp.log(jnp.clip(A, 1e-300, None))
    log_thresh = jnp.log(ess_threshold_frac * N)

    state_probs, final_particles, final_log_w, ess_arr, resampled_arr, log_lik = \
        _run_segment_jit(
            y,
            jnp.asarray(state.particles),
            jnp.asarray(state.log_weights),
            log_A, mu, scale_tril, N, K, log_thresh, state.rng_seed,
        )
    new_state = PFState(
        particles=np.asarray(final_particles),
        log_weights=np.asarray(final_log_w),
        rng_seed=state.rng_seed + len(y) + 1,
    )
    result = PFResult(
        state_probs=np.asarray(state_probs),
        log_lik=float(log_lik),
        ess=np.asarray(ess_arr),
        resampled=np.asarray(resampled_arr).astype(bool),
    )
    return result, new_state


def _run_segment_jit(
    y: jnp.ndarray,
    particles_init: jnp.ndarray,
    log_w_init: jnp.ndarray,
    log_A: jnp.ndarray,
    mu: jnp.ndarray,
    scale_tril: jnp.ndarray,
    N: int,
    K: int,
    log_thresh: jnp.ndarray,
    seed: int,
):
    """JIT-able segment runner. Carries (particles, log_w, rng) across timesteps."""
    rng = random.PRNGKey(seed)

    def step(carry, x):
        particles_prev, log_w_prev, log_lik_acc, key = carry
        y_t = x

        key, k_prop, k_res = random.split(key, 3)
        particles_new = random.categorical(k_prop, log_A[particles_prev])
        log_b = _emission_logp(y_t, mu, scale_tril)
        log_w_unnorm = log_w_prev + log_b[particles_new]
        log_norm = logsumexp(log_w_unnorm)
        log_w = log_w_unnorm - log_norm
        log_lik_acc = log_lik_acc + log_norm

        log_ess = _log_ess(log_w)
        do_resample = log_ess < log_thresh

        def resample_branch(_):
            indices = _systematic_resample(k_res, jnp.exp(log_w))
            return particles_new[indices], -jnp.log(N) * jnp.ones(N)

        def keep_branch(_):
            return particles_new, log_w

        particles_out, log_w_out = jax.lax.cond(
            do_resample, resample_branch, keep_branch, operand=None
        )
        state_probs = _aggregate_state_probs(particles_out, log_w_out, K)
        return (particles_out, log_w_out, log_lik_acc, key), \
               (state_probs, log_ess, do_resample)

    init_carry = (particles_init, log_w_init, jnp.float64(0.0), rng)
    final_carry, scan_out = jax.lax.scan(step, init_carry, y)
    state_probs, log_ess, resampled = scan_out
    final_particles = final_carry[0]
    final_log_w = final_carry[1]
    log_lik = final_carry[2]
    return state_probs, final_particles, final_log_w, jnp.exp(log_ess), resampled, log_lik
