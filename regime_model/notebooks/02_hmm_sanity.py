"""Phase-2 HMM sanity check.

Fits a Gaussian HMM with K=2,3,4 to the standardized cross-asset features
and answers two questions the spec calls out:

  1. Do regimes exist? (BIC selection, distinctness of state means/vols)
  2. Does the K=3 fit look like risk-on / risk-off / crisis?
     We overlay smoothed regime probabilities on the SPY drawdown so the user
     can eyeball whether the "crisis" state lights up around 2008, 2020, 2022.

Outputs:
  outputs/phase2_hmm_K3_states.png   smoothed P(s_t | y_{1:T}) vs SPY drawdown
  outputs/phase2_hmm_bic_table.csv   BIC across K, plus per-state summaries
  outputs/phase2_hmm_K3_summary.csv  state means, vols, and dwell times
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from regime_model.data.features import build_features
from regime_model.data.loaders import FEATURE_TICKERS, load_universe
from regime_model.models.hmm_baseline import (
    HMMConfig,
    HMMParams,
    bic,
    fit_hmm,
    smoothed_state_logprobs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(name)s  %(message)s")
log = logging.getLogger("phase2")
OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def _drawdown(prices: pd.Series) -> pd.Series:
    cummax = prices.cummax()
    return prices / cummax - 1.0


def _state_summary(params: HMMParams, feature_names: list[str]) -> pd.DataFrame:
    """One row per state, columns = each feature's mean and the diagonal vol."""
    K, D = params.mu.shape
    rows = []
    for k in range(K):
        row = {"state": k}
        for d, name in enumerate(feature_names):
            row[f"mu_{name}"] = params.mu[k, d]
            row[f"sd_{name}"] = float(np.sqrt(params.Sigma[k, d, d]))
        rows.append(row)
    return pd.DataFrame(rows).set_index("state")


def _expected_dwell_time(A: np.ndarray) -> np.ndarray:
    """For a Markov chain, expected time in state k = 1 / (1 - A[k, k])."""
    diag = np.diag(A)
    return 1.0 / np.clip(1.0 - diag, 1e-12, None)


def _label_states_by_vol(params: HMMParams, vol_col_indices: list[int]) -> dict[int, str]:
    """Heuristic: rank states by mean vol across the per-asset vol features.
    Lowest vol -> "calm", middle -> "normal" (or "neutral"), highest -> "stress"."""
    K = params.mu.shape[0]
    avg_vol = params.mu[:, vol_col_indices].mean(axis=1)
    order = np.argsort(avg_vol)
    if K == 2:
        names = ["calm", "stress"]
    elif K == 3:
        names = ["calm", "normal", "stress"]
    else:
        names = [f"q{i}" for i in range(K)]
    labels: dict[int, str] = {}
    for rank, state in enumerate(order):
        labels[int(state)] = names[rank]
    return labels


def main() -> None:
    log.info("=== Phase 2: HMM sanity ===")
    bundle = load_universe(start="2005-01-01")
    bundle_feat = build_features(bundle.returns, min_zscore_periods=252)

    z = bundle_feat.standardized.dropna()
    feature_names = list(z.columns)
    vol_col_indices = [i for i, c in enumerate(feature_names) if c.startswith("vol20_")]
    log.info("fitting on %d obs × %d features (%s..%s)", len(z), z.shape[1],
             z.index.min().date(), z.index.max().date())

    X = z.to_numpy()

    # 1. BIC sweep.
    bic_rows = []
    fits: dict[int, HMMParams] = {}
    for K in (2, 3, 4):
        log.info("fitting HMM(K=%d)...", K)
        params = fit_hmm(X, HMMConfig(K=K, max_iter=200, tol=1e-5, seed=0, cov_ridge=1e-3))
        ll = params.log_likelihood_history[-1]
        bic_rows.append({"K": K, "log_lik": ll, "bic": bic(X, params, ll)})
        fits[K] = params
        log.info("  K=%d  log-lik=%.1f  BIC=%.1f", K, ll, bic_rows[-1]["bic"])

    bic_df = pd.DataFrame(bic_rows).set_index("K")
    print("\n--- BIC sweep ---")
    print(bic_df.to_string(float_format=lambda x: f"{x:12.2f}"))
    bic_df.to_csv(OUT / "phase2_hmm_bic_table.csv")

    # 2. K=3 deep-dive.
    K = 3
    params = fits[K]
    labels = _label_states_by_vol(params, vol_col_indices)
    log.info("K=3 state labels (by mean vol): %s", labels)

    summary = _state_summary(params, feature_names)
    summary["dwell_time_days"] = _expected_dwell_time(params.A)
    summary["label"] = [labels[k] for k in summary.index]
    summary["unconditional_p"] = _stationary_dist(params.A)
    print(f"\n--- K={K} state summary (means + diagonal sd) ---")
    cols_to_show = ["label", "dwell_time_days", "unconditional_p"] + \
        [c for c in summary.columns if c.startswith("mu_")]
    print(summary[cols_to_show].to_string(float_format=lambda x: f"{x:7.3f}"))

    # Identifiability check: max pairwise mean-distance.
    K_ = params.mu.shape[0]
    pairwise = np.zeros((K_, K_))
    for i in range(K_):
        for j in range(K_):
            pairwise[i, j] = np.linalg.norm(params.mu[i] - params.mu[j])
    print("\n--- Pairwise state mean distances ---")
    print(pd.DataFrame(pairwise).round(3).to_string())
    min_offdiag = pairwise[~np.eye(K_, dtype=bool)].min()
    log.info("min off-diag pairwise mean distance: %.3f (must be > 0 for identifiability)",
             min_offdiag)
    if min_offdiag < 0.5:
        log.warning("STATES MAY HAVE COLLAPSED — pairwise distance is small")

    summary.to_csv(OUT / "phase2_hmm_K3_summary.csv")

    # 3. Smoothed posterior over time + SPY drawdown.
    log_gamma, _ = smoothed_state_logprobs(X, params)
    gamma = np.exp(log_gamma)
    gamma_df = pd.DataFrame(gamma, index=z.index, columns=[f"P(s={k}, {labels[k]})" for k in range(K)])

    spy_prices = bundle.prices["SPY"].reindex(z.index)
    dd = _drawdown(spy_prices)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    gamma_df.plot.area(ax=ax1, alpha=0.55, linewidth=0)
    ax1.set_ylabel("P(state | data, K=3)")
    ax1.set_title("Smoothed regime posterior")
    ax1.set_ylim(0, 1)
    ax1.legend(loc="upper left", fontsize=8)

    ax2.fill_between(dd.index, dd.values, 0, color="firebrick", alpha=0.5)
    ax2.set_ylabel("SPY drawdown")
    ax2.set_title("SPY drawdown (overlay)")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out_png = OUT / "phase2_hmm_K3_states.png"
    fig.savefig(out_png, dpi=130)
    log.info("saved %s", out_png)

    # 4. Regime probability at named crisis dates.
    print("\n--- Regime posterior at named crisis dates ---")
    probe_dates = ["2008-10-15", "2011-08-08", "2018-12-24", "2020-03-23", "2022-06-13"]
    rows = []
    for d in probe_dates:
        ts = pd.Timestamp(d)
        loc = z.index.get_indexer([ts], method="nearest")[0]
        actual_date = z.index[loc]
        rows.append({
            "asked": d,
            "actual": actual_date.date(),
            **{f"P({labels[k]})": float(gamma[loc, k]) for k in range(K)},
        })
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:5.3f}"))


def _stationary_dist(A: np.ndarray) -> np.ndarray:
    """Stationary distribution: solution to π A = π, π·1 = 1."""
    K = A.shape[0]
    # Solve (A^T - I) π = 0 with sum(π) = 1.
    M = np.vstack([A.T - np.eye(K), np.ones(K)])
    rhs = np.zeros(K + 1)
    rhs[-1] = 1.0
    pi, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    return pi / pi.sum()


if __name__ == "__main__":
    main()
