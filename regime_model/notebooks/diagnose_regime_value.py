"""Diagnose why regime-conditional MVO doesn't outperform RandomRegime.

Hypothesis we are testing: regimes are well-separated in feature space, but
once we translate them to per-regime asset-return moments and run them through
MVO with risk_aversion=50, the resulting weights converge to nearly the same
allocation across regimes — so regime detection has nothing to do.

Outputs to stdout + saves a markdown summary in outputs/diag_regime_value.md.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Project imports
from regime_model.allocation.strategy import (
    StrategyConfig,
    regime_conditional_moments,
    solve_mvo,
)
from regime_model.data.features import build_features
from regime_model.data.loaders import ALLOCATION_TICKERS, load_universe
from regime_model.inference.variational import (
    SVIConfig,
    fit_svi,
    smoothed_posterior,
)
from regime_model.models.bayesian_regime import ModelConfig

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("diag")

TRAIN_END = "2011-01-01"
ANN = 252.0
OUT_PATH = Path(__file__).resolve().parents[1] / "outputs" / "diag_regime_value.md"


def annualize_mean(daily_log_mean: np.ndarray) -> np.ndarray:
    """Convert daily log-return mean to approximate annualized arithmetic %."""
    return (np.exp(daily_log_mean * ANN) - 1.0) * 100.0


def annualize_vol(daily_log_std: np.ndarray) -> np.ndarray:
    return daily_log_std * np.sqrt(ANN) * 100.0


def regime_label(mu_k_features: np.ndarray) -> list[str]:
    """Heuristic labeling by SPY mean and vol — same logic the project uses."""
    # mu_k_features is (K, D); we don't have feature names here, so we use the
    # crude proxy: the regime with highest 'first-feature' value... unused.
    K = mu_k_features.shape[0]
    return [f"s{k}" for k in range(K)]


def main() -> int:
    log.info("=== Regime-Value Diagnostic ===")
    log.info("Train window cutoff: %s", TRAIN_END)

    # 1. Load data + build features (cached parquet exists).
    bundle = load_universe()
    feat = build_features(bundle.returns).standardized
    feat = feat.dropna()
    log.info("Feature rows: %d  Feature dims: %d", len(feat), feat.shape[1])

    train_end_ts = pd.Timestamp(TRAIN_END)
    feat_train = feat.loc[feat.index < train_end_ts]
    log.info(
        "Training window: %s..%s  (%d obs)",
        feat_train.index.min().date(),
        feat_train.index.max().date(),
        len(feat_train),
    )

    # 2. Fit VI on the training window.
    log.info("\nFitting SVI (10k steps)...")
    svi_cfg = SVIConfig(n_steps=10_000, learning_rate=1e-3, seed=0, progress=False)
    model_cfg = ModelConfig()
    svi_res = fit_svi(feat_train.to_numpy(), model_cfg=model_cfg, svi_cfg=svi_cfg,
                     n_posterior_samples=100)
    pe = svi_res.point_estimate
    log.info("Final ELBO loss: %.2f", svi_res.losses[-1])
    K = pe.A.shape[0]
    log.info("K=%d regimes", K)

    # 3. Compute smoothed posterior on training window.
    gamma_np, ll = smoothed_posterior(feat_train.to_numpy(), pe)
    gamma_train = pd.DataFrame(
        gamma_np, index=feat_train.index, columns=[f"s{k}" for k in range(K)]
    )
    log.info("Training log-likelihood: %.2f", ll)

    # 4. Compute per-regime asset moments. Align returns to feature index.
    alloc_ret = bundle.returns[list(ALLOCATION_TICKERS)].loc[feat_train.index].dropna()
    gamma_aligned = gamma_train.loc[alloc_ret.index]
    cfg = StrategyConfig()  # risk_aversion=50 default
    mu_k, Sigma_k = regime_conditional_moments(alloc_ret, gamma_aligned, cfg)
    log.info("\nmu_k shape: %s  Sigma_k shape: %s", mu_k.shape, Sigma_k.shape)

    # Effective per-regime sample size.
    gamma_sum = gamma_aligned.sum(axis=0).values
    log.info("Effective per-regime sample size (sum of gamma): %s",
             np.round(gamma_sum, 1).tolist())

    # 5. Label regimes by annualized SPY mean (descending = "calm" → "stress").
    spy_idx = list(ALLOCATION_TICKERS).index("SPY")
    spy_means_ann = annualize_mean(mu_k[:, spy_idx])
    order = np.argsort(-spy_means_ann)  # highest SPY mean first
    labels = ["calm", "normal", "stress"][:K]
    label_map = {order[i]: labels[i] for i in range(K)}

    # 6. Per-regime annualized return / vol table.
    rows = []
    for k in range(K):
        annual_ret = annualize_mean(mu_k[k])
        annual_vol = annualize_vol(np.sqrt(np.diag(Sigma_k[k])))
        rows.append({
            "regime": label_map[k],
            "id": k,
            "p_uncond": float(gamma_sum[k] / gamma_sum.sum()),
            **{f"{t}_ret%": float(annual_ret[i]) for i, t in enumerate(ALLOCATION_TICKERS)},
            **{f"{t}_vol%": float(annual_vol[i]) for i, t in enumerate(ALLOCATION_TICKERS)},
        })
    moments_df = pd.DataFrame(rows).set_index("regime")
    log.info("\n--- Per-regime annualized RETURNS (%) ---")
    log.info(moments_df[[f"{t}_ret%" for t in ALLOCATION_TICKERS]].round(2).to_string())
    log.info("\n--- Per-regime annualized VOLS (%) ---")
    log.info(moments_df[[f"{t}_vol%" for t in ALLOCATION_TICKERS]].round(2).to_string())
    log.info("\nUnconditional regime probabilities: %s",
             {label_map[k]: f"{gamma_sum[k]/gamma_sum.sum():.2%}" for k in range(K)})

    # 7. Solve per-regime MVO weights — THE PUNCHLINE.
    w_per_regime = np.zeros((K, len(ALLOCATION_TICKERS)))
    for k in range(K):
        w_per_regime[k] = solve_mvo(mu_k[k], Sigma_k[k], cfg)
    weights_df = pd.DataFrame(
        w_per_regime,
        index=[label_map[k] for k in range(K)],
        columns=list(ALLOCATION_TICKERS),
    )
    weights_df["sum"] = weights_df.sum(axis=1)
    log.info("\n--- Per-regime MVO WEIGHTS (lambda=%g, max_weight=%g) ---",
             cfg.risk_aversion, cfg.max_weight)
    log.info(weights_df.round(3).to_string())

    # Pairwise weight distance.
    log.info("\n--- Pairwise weight L1 distance ---")
    dists = pd.DataFrame(
        index=weights_df.index[:-0] if False else weights_df.index,
        columns=weights_df.index, dtype=float,
    )
    for i, ri in enumerate(weights_df.index):
        for j, rj in enumerate(weights_df.index):
            dists.loc[ri, rj] = float(
                np.abs(w_per_regime[i] - w_per_regime[j]).sum()
            )
    log.info(dists.round(3).to_string())

    # 8. Transition matrix.
    A_df = pd.DataFrame(
        pe.A,
        index=[f"from_{label_map[k]}" for k in range(K)],
        columns=[f"to_{label_map[k]}" for k in range(K)],
    )
    log.info("\n--- Transition matrix A ---")
    log.info(A_df.round(4).to_string())
    stay_probs = np.diag(pe.A)
    dwell = 1.0 / (1.0 - stay_probs)
    log.info("Implied dwell times (days): %s",
             {label_map[k]: f"{dwell[k]:.1f}" for k in range(K)})

    # 9. Posterior decisiveness over time.
    max_prob = gamma_aligned.max(axis=1)
    entropy = -(gamma_aligned * np.log(gamma_aligned + 1e-12)).sum(axis=1)
    decisive_share = float((max_prob > 0.9).mean())
    log.info("\n--- Posterior decisiveness on training window ---")
    log.info("Mean max-prob:      %.3f", max_prob.mean())
    log.info("Median max-prob:    %.3f", max_prob.median())
    log.info("Share with max_p>0.9: %.1f%%", decisive_share * 100.0)
    log.info("Mean entropy (bits): %.3f", entropy.mean() / np.log(2))

    # 9b. PREDICTIVE vs DESCRIPTIVE test.
    #
    # mu_k above is the average return on days the SMOOTHED posterior assigned
    # to regime k. That's "descriptive" — it conditions on contemporaneous
    # state. The strategy needs PREDICTIVE: given today's filter label, what
    # do the NEXT h days look like? Compute realized forward returns by
    # filter-argmax label, using only causal (filtered, not smoothed) probs.
    log.info("\n--- PREDICTIVE TEST: forward returns by argmax(gamma_t) ---")
    log.info("(Uses smoothed posterior as label proxy on training data;")
    log.info(" walk-forward OOS test would use filter only.)")
    fwd_horizons = [1, 5, 21]
    argmax_label = gamma_aligned.values.argmax(axis=1)
    fwd_table_rows = []
    for h in fwd_horizons:
        # Sum log returns over t+1 ... t+h. We label by state at t and look at
        # cumulative returns over the NEXT h days.
        fwd_logret = alloc_ret.rolling(h).sum().shift(-h).values  # (T, A) future
        for k in range(K):
            mask = (argmax_label == k) & (~np.isnan(fwd_logret).any(axis=1))
            if mask.sum() < 10:
                continue
            mean_fwd = fwd_logret[mask].mean(axis=0)
            mean_fwd_ann = (np.exp(mean_fwd * (ANN / h)) - 1.0) * 100.0
            row = {"horizon_d": h, "regime": label_map[k], "n": int(mask.sum())}
            for i, t in enumerate(ALLOCATION_TICKERS):
                row[f"{t}_fwd_ann%"] = float(mean_fwd_ann[i])
            fwd_table_rows.append(row)
    fwd_df = pd.DataFrame(fwd_table_rows)
    log.info(fwd_df.round(2).to_string(index=False))

    # Compare in-sample mu_k vs forward returns for SPY only (most diagnostic).
    log.info("\nIn-sample vs forward annualized SPY return (%):")
    in_sample_spy = annualize_mean(mu_k[:, spy_idx])
    for k in range(K):
        in_s = float(in_sample_spy[k])
        fwd_1 = fwd_df[(fwd_df["horizon_d"] == 1) & (fwd_df["regime"] == label_map[k])]["SPY_fwd_ann%"]
        fwd_5 = fwd_df[(fwd_df["horizon_d"] == 5) & (fwd_df["regime"] == label_map[k])]["SPY_fwd_ann%"]
        fwd_21 = fwd_df[(fwd_df["horizon_d"] == 21) & (fwd_df["regime"] == label_map[k])]["SPY_fwd_ann%"]
        log.info(
            "  %-7s  in_sample=%7.2f  fwd_1d=%7.2f  fwd_5d=%7.2f  fwd_21d=%7.2f",
            label_map[k],
            in_s,
            float(fwd_1.iloc[0]) if len(fwd_1) else float("nan"),
            float(fwd_5.iloc[0]) if len(fwd_5) else float("nan"),
            float(fwd_21.iloc[0]) if len(fwd_21) else float("nan"),
        )

    # 9c. OOS PREDICTIVE TEST.
    #
    # Now apply the train-period VI fit to the OOS feature window and run the
    # smoothed posterior. The smoother is hindsight-biased within the OOS
    # window but trained PARAMS are causal (frozen before OOS starts). This is
    # the upper bound on predictive power without retraining.
    log.info("\n--- OOS PREDICTIVE TEST (params frozen at train_end) ---")
    feat_oos = feat.loc[feat.index >= train_end_ts]
    if len(feat_oos) < 100:
        log.info("not enough OOS data, skipping")
    else:
        gamma_oos_np, ll_oos = smoothed_posterior(feat_oos.to_numpy(), pe)
        gamma_oos_df = pd.DataFrame(
            gamma_oos_np, index=feat_oos.index, columns=[f"s{k}" for k in range(K)]
        )
        alloc_ret_oos = bundle.returns[list(ALLOCATION_TICKERS)].loc[feat_oos.index].dropna()
        gamma_oos_aligned = gamma_oos_df.loc[alloc_ret_oos.index]
        argmax_oos = gamma_oos_aligned.values.argmax(axis=1)

        oos_counts = pd.Series(
            [int((argmax_oos == k).sum()) for k in range(K)],
            index=[label_map[k] for k in range(K)],
            name="oos_n",
        )
        log.info("OOS day count by regime label: %s", oos_counts.to_dict())
        log.info("OOS share: %s",
                 {k: f"{v/oos_counts.sum():.1%}" for k, v in oos_counts.items()})

        oos_rows = []
        for h in fwd_horizons:
            fwd_oos = alloc_ret_oos.rolling(h).sum().shift(-h).values
            for k in range(K):
                mask = (argmax_oos == k) & (~np.isnan(fwd_oos).any(axis=1))
                if mask.sum() < 10:
                    continue
                mean_fwd = fwd_oos[mask].mean(axis=0)
                mean_fwd_ann = (np.exp(mean_fwd * (ANN / h)) - 1.0) * 100.0
                row = {"horizon_d": h, "regime": label_map[k], "n": int(mask.sum())}
                for i, t in enumerate(ALLOCATION_TICKERS):
                    row[f"{t}_fwd_ann%"] = float(mean_fwd_ann[i])
                oos_rows.append(row)
        oos_fwd_df = pd.DataFrame(oos_rows)
        log.info("OOS forward returns by regime label:")
        log.info(oos_fwd_df.round(2).to_string(index=False))

        # Direct comparison of in-sample vs OOS regime-conditional SPY return.
        log.info("\nIn-sample vs OOS  SPY annualized return (%) by regime:")
        for k in range(K):
            in_s = float(in_sample_spy[k])
            oos_h21 = oos_fwd_df[
                (oos_fwd_df["horizon_d"] == 21) & (oos_fwd_df["regime"] == label_map[k])
            ]["SPY_fwd_ann%"]
            log.info(
                "  %-7s  in_sample=%7.2f  oos_fwd_21d=%7.2f  shift=%+7.2f",
                label_map[k], in_s,
                float(oos_h21.iloc[0]) if len(oos_h21) else float("nan"),
                float(oos_h21.iloc[0]) - in_s if len(oos_h21) else float("nan"),
            )

        # 9d. Spread test: how much do regime-conditional means differ on OOS
        # vs an unconditional baseline? If they collapse OOS, the model is
        # fitting one episode (2008) and not generalizing.
        log.info("\nOOS unconditional 21d-forward annualized return (%):")
        fwd_oos_21 = alloc_ret_oos.rolling(21).sum().shift(-21)
        uncond = (np.exp(fwd_oos_21.mean() * (ANN / 21)) - 1.0) * 100.0
        log.info("  %s",
                 {t: f"{uncond[t]:.2f}" for t in ALLOCATION_TICKERS})

    # 10. Mixed weight under typical posterior vs corner-case posteriors.
    log.info("\n--- Mixed weights under representative posteriors ---")
    representative = {
        "all-calm (1,0,0)": np.eye(K)[order[0]],
        "all-normal (0,1,0)": np.eye(K)[order[1]] if K >= 2 else None,
        "all-stress (0,0,1)": np.eye(K)[order[-1]],
        "uniform (1/3,1/3,1/3)": np.ones(K) / K,
        "unconditional (steady-state)": gamma_sum / gamma_sum.sum(),
    }
    mix_rows = {}
    for label, probs in representative.items():
        if probs is None:
            continue
        mixed = probs @ w_per_regime
        mix_rows[label] = mixed
    mix_df = pd.DataFrame(mix_rows, index=list(ALLOCATION_TICKERS)).T
    mix_df["sum"] = mix_df.sum(axis=1)
    log.info(mix_df.round(3).to_string())

    # 11. Save markdown summary.
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        f.write("# Regime-Value Diagnostic\n\n")
        f.write(f"Training window: {feat_train.index.min().date()} .. "
                f"{feat_train.index.max().date()} ({len(feat_train)} obs)\n\n")
        f.write(f"Strategy config: risk_aversion={cfg.risk_aversion}, "
                f"max_weight={cfg.max_weight}, target_vol={cfg.target_annual_vol}\n\n")
        f.write("## Per-regime annualized returns (%)\n\n")
        f.write(moments_df[[f"{t}_ret%" for t in ALLOCATION_TICKERS]].round(2).to_markdown())
        f.write("\n\n## Per-regime annualized vols (%)\n\n")
        f.write(moments_df[[f"{t}_vol%" for t in ALLOCATION_TICKERS]].round(2).to_markdown())
        f.write("\n\n## Per-regime MVO weights\n\n")
        f.write(weights_df.round(3).to_markdown())
        f.write("\n\n## Pairwise weight L1 distance\n\n")
        f.write(dists.round(3).to_markdown())
        f.write("\n\n## Transition matrix\n\n")
        f.write(A_df.round(4).to_markdown())
        f.write(f"\n\nImplied dwell times (days): "
                f"{ {label_map[k]: round(float(dwell[k]),1) for k in range(K)} }\n")
        f.write("\n## Posterior decisiveness\n\n")
        f.write(f"- Mean max-prob: {max_prob.mean():.3f}\n")
        f.write(f"- Median max-prob: {max_prob.median():.3f}\n")
        f.write(f"- Share decisive (max_p>0.9): {decisive_share:.1%}\n")
        f.write(f"- Mean entropy (bits): {entropy.mean()/np.log(2):.3f}\n")
        f.write("\n## Mixed weights under representative posteriors\n\n")
        f.write(mix_df.round(3).to_markdown())
    log.info("\nWrote summary: %s", OUT_PATH)

    return 0


if __name__ == "__main__":
    sys.exit(main())
