"""Phase-3 VI fit on the training window.

Fits the Bayesian regime model with SVI on the spec's training window
(2005-2010), produces ELBO and parameter-posterior diagnostics, and
side-by-side compares the smoothed regime posterior to the HMM baseline.

Outputs:
  outputs/phase3_vi_elbo.png            ELBO trace
  outputs/phase3_vi_state_summary.csv   per-state means + dwell times
  outputs/phase3_vi_vs_hmm.png          smoothed P(state) overlay
  outputs/phase3_vi_pairwise_dist.csv   identifiability check
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
from regime_model.data.loaders import load_universe
from regime_model.inference.variational import (
    SVIConfig,
    fit_svi,
    smoothed_posterior,
)
from regime_model.models.bayesian_regime import ModelConfig
from regime_model.models.hmm_baseline import (
    HMMConfig,
    fit_hmm,
    smoothed_state_logprobs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(name)s  %(message)s")
log = logging.getLogger("phase3")
OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_END = "2011-01-01"
K = 3


def _label_states_by_vol(mu: np.ndarray, vol_idx: list[int]) -> dict[int, str]:
    avg_vol = mu[:, vol_idx].mean(axis=1)
    order = np.argsort(avg_vol)
    names = ["calm", "normal", "stress"] if mu.shape[0] == 3 else \
        [f"q{i}" for i in range(mu.shape[0])]
    return {int(s): names[r] for r, s in enumerate(order)}


def _match_states(mu_a: np.ndarray, mu_b: np.ndarray) -> np.ndarray:
    """Return a permutation perm such that mu_a[perm] best matches mu_b."""
    K = mu_a.shape[0]
    perm = []
    used = set()
    for k in range(K):
        candidates = [j for j in range(K) if j not in used]
        dists = [np.linalg.norm(mu_a[j] - mu_b[k]) for j in candidates]
        choice = candidates[int(np.argmin(dists))]
        perm.append(choice)
        used.add(choice)
    return np.array(perm)


def main() -> None:
    log.info("=== Phase 3: Bayesian VI fit ===")
    bundle = load_universe(start="2005-01-01")
    bundle_feat = build_features(bundle.returns, min_zscore_periods=252)
    z_all = bundle_feat.standardized.dropna()

    z_train = z_all.loc[z_all.index < TRAIN_END]
    log.info("training window: %s..%s  n=%d  D=%d",
             z_train.index.min().date(), z_train.index.max().date(),
             len(z_train), z_train.shape[1])

    feature_names = list(z_train.columns)
    vol_idx = [i for i, c in enumerate(feature_names) if c.startswith("vol20_")]

    X_train = z_train.to_numpy()

    # 1. SVI fit.
    log.info("running SVI...")
    svi_result = fit_svi(
        X_train,
        model_cfg=ModelConfig(
            K=K,
            persistence_diag=10.0,
            persistence_off=1.0,
            mu_hyper_scale=1.0,
            sigma_scale_prior=2.5,
            lkj_concentration=2.0,
        ),
        svi_cfg=SVIConfig(n_steps=10_000, learning_rate=1e-3, seed=0,
                          init_scale=0.1, log_every=2000, progress=True),
        n_posterior_samples=300,
    )
    log.info("SVI done. final loss=%.2f  initial loss=%.2f",
             svi_result.losses[-1], svi_result.losses[0])

    pe = svi_result.point_estimate

    # 2. ELBO trace.
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(svi_result.losses, linewidth=0.8)
    ax.set_xlabel("SVI step")
    ax.set_ylabel("Negative ELBO (loss)")
    ax.set_title("SVI loss curve")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "phase3_vi_elbo.png", dpi=130)

    # 3. State summary + identifiability.
    labels = _label_states_by_vol(pe.mu, vol_idx)
    print("\n--- VI state labels (by mean vol) ---")
    print(labels)

    summary_rows = []
    for k in range(K):
        row = {"state": k, "label": labels[k],
               "dwell_days": float(1.0 / max(1.0 - pe.A[k, k], 1e-12))}
        for d, name in enumerate(feature_names):
            row[f"mu_{name}"] = pe.mu[k, d]
            row[f"sd_{name}"] = float(np.sqrt(pe.Sigma[k, d, d]))
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows).set_index("state")
    summary_df.to_csv(OUT / "phase3_vi_state_summary.csv")

    # Pairwise mean distance (collapse check).
    pairwise = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            pairwise[i, j] = np.linalg.norm(pe.mu[i] - pe.mu[j])
    pairwise_df = pd.DataFrame(pairwise, index=range(K), columns=range(K)).round(3)
    pairwise_df.to_csv(OUT / "phase3_vi_pairwise_dist.csv")
    print("\n--- Pairwise state mean distances ---")
    print(pairwise_df.to_string())
    min_off = pairwise[~np.eye(K, dtype=bool)].min()
    log.info("min off-diagonal mean distance = %.3f", min_off)
    if min_off < 0.5:
        log.warning("WARNING: states may have collapsed (min distance < 0.5)")

    print("\n--- VI state summary (label, dwell, means) ---")
    cols = ["label", "dwell_days"] + [c for c in summary_df.columns if c.startswith("mu_")]
    print(summary_df[cols].to_string(float_format=lambda x: f"{x:7.3f}"))

    # 4. Smoothed posterior on training window.
    gamma_vi, ll_vi = smoothed_posterior(X_train, pe)
    log.info("VI training-window smoothed log-lik: %.2f", ll_vi)

    # 5. Compare to HMM baseline fit on the same window.
    log.info("fitting HMM on same training window for comparison...")
    hmm = fit_hmm(X_train, HMMConfig(K=K, max_iter=200, tol=1e-6, seed=0, cov_ridge=1e-3))
    log_gamma_hmm, ll_hmm = smoothed_state_logprobs(X_train, hmm)
    gamma_hmm = np.exp(log_gamma_hmm)
    log.info("HMM training-window smoothed log-lik: %.2f", ll_hmm)

    # Permute HMM states to match VI labels.
    perm = _match_states(hmm.mu, pe.mu)
    gamma_hmm_aligned = gamma_hmm[:, perm]

    # 6. Plot smoothed posteriors side-by-side.
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    legend_names = [f"P(s={k}, {labels[k]})" for k in range(K)]

    pd.DataFrame(gamma_vi, index=z_train.index, columns=legend_names).plot.area(
        ax=axes[0], alpha=0.55, linewidth=0
    )
    axes[0].set_title(f"Bayesian VI smoothed posterior (training {z_train.index.min().date()}..{z_train.index.max().date()})")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("P(state | y)")
    axes[0].legend(loc="upper left", fontsize=8)

    pd.DataFrame(gamma_hmm_aligned, index=z_train.index, columns=legend_names).plot.area(
        ax=axes[1], alpha=0.55, linewidth=0
    )
    axes[1].set_title("HMM (EM) smoothed posterior — same window, states permuted to match VI")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("P(state | y)")
    axes[1].legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "phase3_vi_vs_hmm.png", dpi=130)
    log.info("saved comparison plot")

    # 7. Crisis date probe.
    print("\n--- VI regime posterior at GFC dates ---")
    probe_dates = ["2007-08-15", "2008-09-29", "2008-10-15", "2009-03-09", "2010-05-06"]
    rows = []
    for d in probe_dates:
        ts = pd.Timestamp(d)
        if ts < z_train.index.min() or ts > z_train.index.max():
            continue
        loc = z_train.index.get_indexer([ts], method="nearest")[0]
        actual = z_train.index[loc]
        rows.append({"asked": d, "actual": actual.date(),
                     **{f"P({labels[k]})": float(gamma_vi[loc, k]) for k in range(K)}})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:5.3f}"))


if __name__ == "__main__":
    main()
