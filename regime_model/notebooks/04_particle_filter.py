"""Phase-4 particle filter on the OOS window using VI-fit parameters.

Workflow:
    1. Fit VI on the spec's training window (2005-2010).
    2. Run the bootstrap particle filter on the full series with N=10,000.
    3. Compare PF filtered marginals to the exact HMM forward-pass marginals
       (both using the same VI parameters) — they should agree closely.
    4. Plot ESS over time and report any degeneracy.

This is the spec's particle-filter sanity check: "If particle filter has
degeneracy issues (ESS always low), pause and reconsider."
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
from regime_model.inference.particle_filter import particle_filter
from regime_model.inference.variational import SVIConfig, fit_svi
from regime_model.models.bayesian_regime import (
    ModelConfig,
    forward_backward,
)
from regime_model.models.hmm_baseline import (
    HMMConfig,
    HMMParams,
    fit_hmm,
    filtered_state_logprobs,
)

import jax.numpy as jnp

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(name)s  %(message)s")
log = logging.getLogger("phase4")
OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_END = "2011-01-01"
OOS_START = "2011-01-01"
K = 3
N_PARTICLES = 10_000


def _label_states(mu: np.ndarray, vol_idx: list[int]) -> dict[int, str]:
    avg_vol = mu[:, vol_idx].mean(axis=1)
    order = np.argsort(avg_vol)
    names = ["calm", "normal", "stress"]
    return {int(s): names[r] for r, s in enumerate(order)}


def main() -> None:
    log.info("=== Phase 4: Particle filter on OOS data ===")

    bundle = load_universe(start="2005-01-01")
    bundle_feat = build_features(bundle.returns, min_zscore_periods=252)
    z = bundle_feat.standardized.dropna()
    feature_names = list(z.columns)
    vol_idx = [i for i, c in enumerate(feature_names) if c.startswith("vol20_")]

    z_train = z.loc[z.index < TRAIN_END]
    log.info("training window: %s..%s  n=%d", z_train.index.min().date(),
             z_train.index.max().date(), len(z_train))

    # 1. VI fit (params come from training window only — no look-ahead).
    log.info("fitting VI on training window (10k steps)...")
    svi_result = fit_svi(
        z_train.to_numpy(),
        model_cfg=ModelConfig(K=K, persistence_diag=10.0, persistence_off=1.0),
        svi_cfg=SVIConfig(n_steps=10_000, learning_rate=1e-3, seed=0,
                          init_scale=0.1, log_every=2500, progress=True),
        n_posterior_samples=200,
    )
    pe = svi_result.point_estimate
    labels = _label_states(pe.mu, vol_idx)
    log.info("VI fit done. Labels: %s", labels)

    # 2. Particle filter on the FULL series (so we can validate on the training
    # portion too) using VI-fit params.
    log.info("running bootstrap PF with N=%d on full series (T=%d)...",
             N_PARTICLES, len(z))
    pf = particle_filter(
        z.to_numpy(),
        pi=pe.pi, A=pe.A, mu=pe.mu, scale_tril=pe.scale_tril,
        n_particles=N_PARTICLES, seed=0, ess_threshold_frac=0.5,
    )
    log.info("PF done. log_lik=%.2f  resampled %d / %d steps  ESS min=%.0f mean=%.0f",
             pf.log_lik, int(pf.resampled.sum()), len(z),
             pf.ess.min(), pf.ess.mean())

    # 3. Exact forward pass with same params (this is the ground truth to compare against).
    log_pi = jnp.log(jnp.asarray(pe.pi))
    log_A = jnp.log(jnp.asarray(pe.A))
    # Use forward-backward then take only the forward part. Easier: reuse HMM filtered.
    hmm_params_from_vi = HMMParams(pi=pe.pi, A=pe.A, mu=pe.mu, Sigma=pe.Sigma)
    log_filt_exact = np.asarray(filtered_state_logprobs(z.to_numpy(), hmm_params_from_vi))
    filt_exact = np.exp(log_filt_exact)

    # Per-state mean absolute deviation between PF and exact filtered.
    mad_per_state = np.mean(np.abs(filt_exact - pf.state_probs), axis=0)
    log.info("PF vs exact filtered MAD per state: %s",
             {labels[k]: f"{mad_per_state[k]:.4f}" for k in range(K)})

    # 4. Plots.
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    legend = [f"P(s={k}, {labels[k]})" for k in range(K)]

    pd.DataFrame(pf.state_probs, index=z.index, columns=legend).plot.area(
        ax=axes[0], alpha=0.55, linewidth=0
    )
    axes[0].set_title(f"Particle filter (N={N_PARTICLES}) filtered posterior")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("P(state | y_{1:t})")
    axes[0].axvline(pd.Timestamp(OOS_START), color="red", linestyle="--",
                    linewidth=1, label="train→OOS")
    axes[0].legend(loc="upper left", fontsize=7)

    pd.DataFrame(filt_exact, index=z.index, columns=legend).plot.area(
        ax=axes[1], alpha=0.55, linewidth=0
    )
    axes[1].set_title("Exact HMM forward-pass filtered posterior (same VI params)")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("P(state | y_{1:t})")
    axes[1].axvline(pd.Timestamp(OOS_START), color="red", linestyle="--", linewidth=1)
    axes[1].legend(loc="upper left", fontsize=7)

    axes[2].plot(z.index, pf.ess, linewidth=0.6, color="navy")
    axes[2].axhline(N_PARTICLES * 0.5, color="orange", linestyle="--",
                    linewidth=0.8, label=f"resample threshold (N/2={N_PARTICLES//2})")
    axes[2].set_title("Effective sample size over time")
    axes[2].set_ylabel("ESS")
    axes[2].set_yscale("log")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    out_png = OUT / "phase4_pf_vs_exact.png"
    fig.savefig(out_png, dpi=130)
    log.info("saved %s", out_png)

    # 5. OOS-only diagnostic.
    oos_idx = z.index >= OOS_START
    print("\n--- OOS PF diagnostics ---")
    print(f"  OOS rows                       : {oos_idx.sum()}")
    print(f"  ESS (OOS) min/mean/median      : {pf.ess[oos_idx].min():.0f} / {pf.ess[oos_idx].mean():.0f} / {np.median(pf.ess[oos_idx]):.0f}")
    print(f"  Resample fraction (OOS)        : {pf.resampled[oos_idx].mean():.2%}")
    print(f"  PF vs exact MAD (OOS)          : {np.mean(np.abs(filt_exact[oos_idx] - pf.state_probs[oos_idx])):.4f}")

    # Crisis-date probe (OOS only).
    print("\n--- PF regime call at OOS named events ---")
    probe_dates = ["2011-08-08", "2015-08-24", "2018-12-24", "2020-03-23",
                   "2022-06-13", "2023-03-13"]
    rows = []
    for d in probe_dates:
        ts = pd.Timestamp(d)
        if ts < z.index.min() or ts > z.index.max():
            continue
        loc = z.index.get_indexer([ts], method="nearest")[0]
        actual = z.index[loc]
        rows.append({
            "asked": d, "actual": actual.date(),
            **{f"P({labels[k]})": float(pf.state_probs[loc, k]) for k in range(K)},
            "ESS": float(pf.ess[loc]),
        })
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:6.3f}"))

    # 6. Pass/fail flags from spec.
    print("\n--- Spec sanity gates ---")
    pf_oos_ess_min = pf.ess[oos_idx].min()
    pf_oos_resample_rate = pf.resampled[oos_idx].mean()
    if pf_oos_ess_min < 100:
        log.warning("WARNING: ESS dropped below 100 on OOS (min=%.0f)", pf_oos_ess_min)
    else:
        log.info("OK: ESS stays >= %.0f on OOS — no degeneracy", pf_oos_ess_min)
    if pf_oos_resample_rate > 0.95:
        log.warning("WARNING: resampling on >95%% of OOS steps — high variance")
    else:
        log.info("OK: resampling rate %.1f%% (healthy)", pf_oos_resample_rate * 100)


if __name__ == "__main__":
    main()
