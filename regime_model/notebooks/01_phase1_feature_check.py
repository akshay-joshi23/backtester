"""Phase 1 sanity-check driver.

Loads the cross-asset universe from 2005-present, builds the 14-dim feature
matrix, runs the look-ahead validator, and emits diagnostic summaries plus a
plot of the standardized features. Intended to be run after the pipeline is
implemented to verify everything wires up end-to-end.

Run:
    PYTHONPATH=. .venv/bin/python regime_model/notebooks/01_phase1_feature_check.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from regime_model.data.features import (
    FEATURE_DIM,
    build_features,
    validate_no_lookahead,
)
from regime_model.data.loaders import (
    ALLOCATION_TICKERS,
    FEATURE_TICKERS,
    load_universe,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(name)s  %(message)s")
log = logging.getLogger("phase1")

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _coverage_table(returns: pd.DataFrame, spliced: dict[str, pd.Timestamp]) -> pd.DataFrame:
    rows = []
    for sym in returns.columns:
        s = returns[sym].dropna()
        rows.append(
            {
                "symbol": sym,
                "first_date": s.index.min().date() if not s.empty else None,
                "last_date": s.index.max().date() if not s.empty else None,
                "n_obs": len(s),
                "spliced_at": spliced.get(sym, pd.NaT),
                "ann_mean_pct": s.mean() * 252 * 100,
                "ann_vol_pct": s.std() * np.sqrt(252) * 100,
            }
        )
    return pd.DataFrame(rows).set_index("symbol")


def _feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    desc = df.describe(percentiles=[0.05, 0.5, 0.95]).T
    desc["nan_share"] = df.isna().mean()
    return desc[["count", "mean", "std", "min", "5%", "50%", "95%", "max", "nan_share"]]


def main() -> None:
    log.info("=== Phase 1 feature check ===")

    # 1. Load.
    bundle = load_universe(start="2005-01-01")
    log.info(
        "loaded prices: %d rows, %s..%s, %d cols",
        len(bundle.prices),
        bundle.prices.index.min().date(),
        bundle.prices.index.max().date(),
        bundle.prices.shape[1],
    )

    print("\n--- Asset coverage ---")
    print(_coverage_table(bundle.returns, bundle.spliced).to_string(float_format=lambda x: f"{x:8.3f}"))

    # 2. Features.
    bundle_feat = build_features(bundle.returns, min_zscore_periods=252)
    raw, z = bundle_feat.raw, bundle_feat.standardized

    log.info("raw features:  %s, dim=%d", raw.shape, FEATURE_DIM)
    log.info("standardized:  %s", z.shape)
    log.info("groups: %s", {k: len(v) for k, v in bundle_feat.column_groups.items()})

    # 3. Look-ahead validator.
    log.info("running look-ahead validator...")
    validate_no_lookahead(bundle.returns, probe_dates=8, seed=0)
    log.info("look-ahead validator passed.")

    # 4. Distribution summaries.
    print("\n--- Raw feature summary ---")
    print(_feature_summary(raw).to_string(float_format=lambda x: f"{x:9.4f}"))

    print("\n--- Standardized feature summary (after warmup) ---")
    z_after = z.dropna(how="all")
    print(_feature_summary(z_after.loc[z_after.index >= "2007-01-01"]).to_string(
        float_format=lambda x: f"{x:9.4f}"
    ))

    # 5. NaN diagnostic by year.
    nan_by_year = raw.isna().mean().to_frame("overall")
    print("\n--- Raw NaN share by year (first 4 cols) ---")
    nan_yearly = raw.isna().groupby(raw.index.year).mean()
    print(nan_yearly.iloc[:, :4].to_string(float_format=lambda x: f"{x:6.3f}"))

    # 6. Allocation-universe sanity.
    print("\n--- Allocation universe (excludes VIX) ---")
    print(f"  {ALLOCATION_TICKERS}")
    print(f"  feature universe: {FEATURE_TICKERS}")

    # 7. Plot standardized features (one panel per group).
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for ax, (group, cols) in zip(axes, bundle_feat.column_groups.items()):
        z[cols].plot(ax=ax, linewidth=0.7)
        ax.set_title(f"Standardized features: {group}")
        ax.axhline(0, color="k", linewidth=0.4, linestyle="--")
        ax.legend(loc="upper left", fontsize=7, ncol=3)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path = OUT_DIR / "phase1_standardized_features.png"
    fig.savefig(out_path, dpi=130)
    log.info("saved plot to %s", out_path)

    # 8. Save a CSV slice for inspection.
    csv_path = OUT_DIR / "phase1_features_sample.csv"
    sample = z.dropna().head(10).round(3)
    sample.to_csv(csv_path)
    log.info("saved 10-row sample to %s", csv_path)

    print("\n--- First 10 standardized rows (after warmup) ---")
    print(sample.to_string(float_format=lambda x: f"{x:6.2f}"))


if __name__ == "__main__":
    main()
