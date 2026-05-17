"""Hansen's Superior Predictive Ability (SPA) test.

Reference: Hansen, "A Test for Superior Predictive Ability" (2005),
Journal of Business & Economic Statistics 23(4): 365–380.

Use case: you have a benchmark strategy and N candidate alternatives.
You want to know whether ANY of the N is genuinely better than the
benchmark, adjusting for the fact that the best of N looks great by
chance even if all are noise.

Mechanic:
  1. Daily performance series f_i,t = alt_i_return_t − benchmark_return_t
     for i = 1..N.
  2. Studentized statistic T_SPA = max_i sqrt(T) * mean(f_i) / omega_i
     where omega_i is HAC-style std of f_i.
  3. Resample under the null (no alt is better than benchmark) using
     stationary block bootstrap. Recenter resampled series at zero so the
     bootstrap distribution reflects "no superiority."
  4. p-value = P(T_SPA_bootstrap >= T_SPA_observed).

Notes:
  - We use the "consistent" recentering (Hansen's SPA-c) — alts with
    sample mean below -A_T are NOT recentered (their bootstrap stays at
    zero), reflecting that they're clearly inferior. Threshold
    A_T = -sqrt((omega_i^2 * log(log(T))) / T).
  - Block length default = T^(1/3) per common practice; user-tunable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SPAResult:
    p_value: float
    t_stat: float
    bootstrap_quantiles: dict[str, float]
    per_alt: pd.DataFrame   # name, mean_excess, t_score, recentered
    n_resamples: int
    block_length: int


def stationary_block_bootstrap_indices(
    n: int, block_length: int, rng: np.random.Generator,
) -> np.ndarray:
    """Politis & Romano stationary bootstrap indices.

    Returns a length-`n` array of indices in [0, n). Each next index
    advances by 1 with probability 1 - 1/block_length, else jumps to a
    uniformly random index. This makes block lengths geometrically
    distributed with mean = block_length.
    """
    p_jump = 1.0 / block_length
    idx = np.empty(n, dtype=np.int64)
    idx[0] = rng.integers(0, n)
    jumps = rng.random(n) < p_jump
    for t in range(1, n):
        if jumps[t]:
            idx[t] = rng.integers(0, n)
        else:
            idx[t] = (idx[t - 1] + 1) % n
    return idx


def _hac_std(x: np.ndarray, max_lag: int | None = None) -> float:
    """Newey-West-style long-run std with Bartlett weights.

    Returns omega = sqrt(gamma_0 + 2 * sum_{l=1..L} k(l) * gamma_l) where
    k(l) = 1 - l/(L+1). If returns are i.i.d., this collapses to std(x).
    """
    n = len(x)
    if n < 2:
        return 0.0
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    if max_lag is None:
        max_lag = max(1, int(n ** (1.0 / 3.0)))
    gamma_0 = float(x @ x) / n
    var = gamma_0
    for lag in range(1, max_lag + 1):
        if lag >= n:
            break
        gamma_l = float(x[:-lag] @ x[lag:]) / n
        weight = 1.0 - lag / (max_lag + 1.0)
        var += 2.0 * weight * gamma_l
    return float(np.sqrt(max(var, 1e-18)))


def spa_test(
    benchmark_returns: pd.Series,
    alt_returns: dict[str, pd.Series],
    *,
    n_resamples: int = 2000,
    block_length: int | None = None,
    seed: int = 0,
) -> SPAResult:
    """Hansen 2005 SPA test for "any alt strictly better than benchmark."

    Parameters
    ----------
    benchmark_returns : pd.Series
        Daily simple returns of the benchmark strategy.
    alt_returns : dict[name -> pd.Series]
        Daily simple returns of each alternative (must share the
        benchmark's date index).
    n_resamples : int
        Bootstrap iterations. 1000–5000 typical.
    block_length : int | None
        Stationary-bootstrap mean block length. Default = T^(1/3).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    SPAResult
        p_value : probability under H0 of observing a T_SPA at least as
                  large. Low p (<0.05) → strong evidence some alt beats
                  the benchmark beyond chance.
        t_stat  : the observed studentized max.
        per_alt : dataframe of each alt's mean excess return, t-score,
                  and whether SPA-c recentered it (True = competitive).
    """
    if not alt_returns:
        raise ValueError("at least one alternative is required")

    # Align all series on the benchmark index.
    df = pd.DataFrame({"_bm": benchmark_returns.copy()})
    for name, s in alt_returns.items():
        df[name] = s
    df = df.dropna(how="any")
    if len(df) < 60:
        raise ValueError(f"need >=60 aligned daily observations, got {len(df)}")
    bm = df["_bm"].to_numpy()
    alt_names = [c for c in df.columns if c != "_bm"]
    alts = df[alt_names].to_numpy()

    T, M = alts.shape
    # Excess returns f_i = alt_i - bm.
    f = alts - bm[:, None]   # shape (T, M)

    means = f.mean(axis=0)
    omegas = np.array([_hac_std(f[:, i]) for i in range(M)])
    omegas = np.where(omegas > 1e-12, omegas, 1e-12)
    t_scores = np.sqrt(T) * means / omegas
    t_stat = float(t_scores.max())

    # SPA-c recentering threshold per Hansen 2005 eq 8.
    # A_T_i = -sqrt( (omega_i^2 / T) * log(log(T)) )
    A_T = -np.sqrt((omegas ** 2 / T) * np.log(np.log(max(T, 3))))
    # Recenter mean to 0 if mean_i > A_T_i (alt is "competitive"), else keep
    # the alt's bootstrap recentered to mean(f) - A_T (strictly bounded away
    # from improving). For the consistent SPA-c, the formal rule is:
    #   recenter g_i = f_i - mean_i * 1(mean_i / omega_i > -sqrt(2*log(log(T))/T))
    competitive = means / omegas > -np.sqrt(2.0 * np.log(max(np.log(max(T, 3)), 1e-12)) / T)
    # For competitive alts, subtract sample mean → bootstrap under H0 mean=0.
    # For non-competitive, leave them as is (bootstrap reflects their actual
    # negative mean — they don't contribute to extra tail mass).
    centered = f.copy()
    centered[:, competitive] = f[:, competitive] - means[competitive]

    if block_length is None:
        block_length = max(2, int(T ** (1.0 / 3.0)))
    rng = np.random.default_rng(seed)

    bootstrap_t = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = stationary_block_bootstrap_indices(T, block_length, rng)
        sample = centered[idx]
        sample_means = sample.mean(axis=0)
        # Use the SAME omegas (original) per Hansen; cheaper + closer to spec
        # than re-estimating per bootstrap.
        t_b = (np.sqrt(T) * sample_means / omegas).max()
        bootstrap_t[b] = t_b

    p_value = float((bootstrap_t >= t_stat).mean())
    quantiles = {
        "q50": float(np.quantile(bootstrap_t, 0.50)),
        "q90": float(np.quantile(bootstrap_t, 0.90)),
        "q95": float(np.quantile(bootstrap_t, 0.95)),
        "q99": float(np.quantile(bootstrap_t, 0.99)),
    }

    per_alt = pd.DataFrame({
        "name": alt_names,
        "mean_excess_per_period": means,
        "hac_std": omegas,
        "t_score": t_scores,
        "competitive": competitive,
    }).set_index("name")

    return SPAResult(
        p_value=p_value,
        t_stat=t_stat,
        bootstrap_quantiles=quantiles,
        per_alt=per_alt,
        n_resamples=n_resamples,
        block_length=block_length,
    )
