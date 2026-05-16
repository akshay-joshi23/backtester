"""Phase-5 walk-forward backtest with the regime allocation strategy.

Runs the full pipeline (annual VI refit, continuous PF, weekly rebalance)
end-to-end from 2011-01 through today, and reports headline performance plus
plots of the equity curve, regime probabilities, and allocation weights.

Outputs:
  outputs/phase5_backtest_equity.png
  outputs/phase5_backtest_metrics.csv
  outputs/phase5_backtest_weights.png
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from regime_model.allocation.backtest import (
    BacktestConfig,
    compute_metrics,
    walk_forward_backtest,
)
from regime_model.allocation.strategy import StrategyConfig
from regime_model.data.features import build_features
from regime_model.data.loaders import ALLOCATION_TICKERS, load_universe
from regime_model.inference.variational import SVIConfig
from regime_model.models.bayesian_regime import ModelConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(name)s  %(message)s")
log = logging.getLogger("phase5")
OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    log.info("=== Phase 5: walk-forward backtest ===")
    bundle = load_universe(start="2005-01-01")
    bf = build_features(bundle.returns, min_zscore_periods=252)

    features = bf.standardized
    returns = bundle.returns

    cfg = BacktestConfig(
        train_end="2011-01-01",
        rebalance_freq=5,                # weekly
        transaction_cost_bps=5.0,
        n_particles=10_000,
        seed=0,
    )
    strategy_cfg = StrategyConfig(
        max_weight=0.4,
        risk_aversion=50.0,
        target_annual_vol=0.10,
        long_only=True,
        max_leverage=1.0,
    )
    model_cfg = ModelConfig(K=3, persistence_diag=10.0, persistence_off=1.0)
    # Use 5k SVI steps for the annual refits — Phase-3 diagnostics showed
    # ELBO converged within 0.1% by step 5000 vs the spec's 10k.
    svi_cfg = SVIConfig(n_steps=5_000, learning_rate=1e-3, seed=0,
                        init_scale=0.1, log_every=10_000, progress=False)

    t0 = time.time()
    result = walk_forward_backtest(
        returns=returns,
        features=features,
        cfg=cfg,
        strategy_cfg=strategy_cfg,
        model_cfg=model_cfg,
        svi_cfg=svi_cfg,
    )
    log.info("backtest finished in %.1fs", time.time() - t0)

    eq = result.equity_curve.dropna()
    eq = eq[eq > 0]
    log.info("equity curve: %d obs, %s..%s, final NAV %.4f",
             len(eq), eq.index.min().date(), eq.index.max().date(), eq.iloc[-1])

    # Headline metrics.
    metrics = compute_metrics(eq)
    log.info("Headline: Sharpe=%.2f  AnnRet=%.2f%%  AnnVol=%.2f%%  MaxDD=%.2f%%  Calmar=%.2f",
             metrics["sharpe"], metrics["ann_return"] * 100, metrics["ann_vol"] * 100,
             metrics["max_drawdown"] * 100, metrics["calmar"])
    metrics_df = pd.Series(metrics).to_frame("regime_strategy").T
    metrics_df.to_csv(OUT / "phase5_backtest_metrics.csv")

    # Turnover summary.
    annual_turnover = result.turnover.resample("YE").sum()
    print("\n--- Annual turnover (sum of |Δw|) ---")
    print(annual_turnover.to_string(float_format=lambda x: f"{x:.2f}"))

    # 1. Equity curve vs SPY buy-and-hold (bare reference).
    spy = bundle.prices["SPY"].reindex(eq.index)
    spy_norm = spy / spy.iloc[0] * eq.iloc[0]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(eq.index, eq.values, label="Regime strategy", linewidth=1.4)
    axes[0].plot(spy_norm.index, spy_norm.values, label="SPY (reference)",
                 linewidth=1.0, alpha=0.7)
    axes[0].set_title("Equity curve (initial NAV = 1.0)")
    axes[0].set_yscale("log")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.3)

    # 2. Regime probabilities.
    K = result.regime_probs.shape[1]
    result.regime_probs.plot.area(
        ax=axes[1], alpha=0.55, linewidth=0
    )
    axes[1].set_title("PF filtered regime posterior (OOS)")
    axes[1].set_ylim(0, 1)
    axes[1].legend(loc="upper left", fontsize=8)

    # 3. Allocation weights over time.
    result.target_weights.plot.area(
        ax=axes[2], alpha=0.7, linewidth=0
    )
    axes[2].set_title("Target allocation weights")
    axes[2].set_ylabel("Weight")
    axes[2].set_ylim(0, 1.05)
    axes[2].legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "phase5_backtest_equity.png", dpi=130)

    # Separate weights detail plot.
    fig2, ax = plt.subplots(figsize=(11, 5))
    result.realized_weights.plot.area(ax=ax, alpha=0.7, linewidth=0)
    ax.set_title("Realized (drifted) weights, daily")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    fig2.tight_layout()
    fig2.savefig(OUT / "phase5_backtest_weights.png", dpi=130)

    print("\n--- Final allocation breakdown (last day) ---")
    last_day = result.realized_weights.index.max()
    print(result.realized_weights.loc[last_day].round(3).to_string())

    print(f"\n--- Total transaction cost paid: ${result.transaction_costs.sum():.4f} (vs initial NAV 1.0) ---")


if __name__ == "__main__":
    main()
