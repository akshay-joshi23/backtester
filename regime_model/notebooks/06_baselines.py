"""Phase-6 baseline comparison.

Runs the regime strategy plus the spec's baselines and compares headline
metrics + equity curves on a single OOS window.

Outputs:
  outputs/phase6_baseline_metrics.csv
  outputs/phase6_baseline_equity.png
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
from regime_model.allocation.baselines import (
    hmm_regime_strategy,
    random_regime_strategy,
    risk_parity,
    static_60_40,
)
from regime_model.allocation.strategy import StrategyConfig
from regime_model.data.features import build_features
from regime_model.data.loaders import load_universe
from regime_model.inference.variational import SVIConfig
from regime_model.models.bayesian_regime import ModelConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(name)s  %(message)s")
log = logging.getLogger("phase6")
OUT = Path(__file__).resolve().parents[1] / "outputs"


def main() -> None:
    log.info("=== Phase 6: baseline comparison ===")
    bundle = load_universe(start="2005-01-01")
    bf = build_features(bundle.returns, min_zscore_periods=252)
    features = bf.standardized
    returns = bundle.returns

    cfg = BacktestConfig(train_end="2011-01-01", rebalance_freq=5,
                         transaction_cost_bps=5.0, n_particles=10_000, seed=0)
    strategy_cfg = StrategyConfig(max_weight=0.4, risk_aversion=50.0,
                                  target_annual_vol=0.10, long_only=True,
                                  max_leverage=1.0)
    model_cfg = ModelConfig(K=3, persistence_diag=10.0, persistence_off=1.0)
    svi_cfg = SVIConfig(n_steps=5_000, learning_rate=1e-3, seed=0,
                        init_scale=0.1, log_every=100_000, progress=False)

    results = {}

    log.info("running 60/40...")
    t0 = time.time()
    results["60/40"] = static_60_40(returns, cfg)
    log.info("  done in %.1fs", time.time() - t0)

    log.info("running risk parity (5 ETFs)...")
    t0 = time.time()
    results["RiskParity"] = risk_parity(returns, cfg)
    log.info("  done in %.1fs", time.time() - t0)

    log.info("running random regime (negative control)...")
    t0 = time.time()
    results["RandomRegime"] = random_regime_strategy(
        returns, features, cfg, strategy_cfg, K=3, seed=0
    )
    log.info("  done in %.1fs", time.time() - t0)

    log.info("running HMM regime strategy (annual EM refit)...")
    t0 = time.time()
    results["HMM-EM"] = hmm_regime_strategy(
        returns, features, cfg, strategy_cfg, K=3
    )
    log.info("  done in %.1fs", time.time() - t0)

    log.info("running Bayesian VI regime strategy (main strategy)...")
    t0 = time.time()
    results["BayesVI+PF"] = walk_forward_backtest(
        returns, features, cfg, strategy_cfg, model_cfg, svi_cfg
    )
    log.info("  done in %.1fs", time.time() - t0)

    # Compute metrics.
    metrics = {}
    for name, res in results.items():
        eq = res.equity_curve.dropna()
        eq = eq[eq > 0]
        if len(eq) < 50:
            continue
        m = compute_metrics(eq)
        m["final_nav"] = float(eq.iloc[-1])
        m["total_turnover"] = float(res.turnover.sum())
        m["total_tc"] = float(res.transaction_costs.sum())
        metrics[name] = m
    metrics_df = pd.DataFrame(metrics).T
    metrics_df = metrics_df[["sharpe", "ann_return", "ann_vol", "max_drawdown",
                              "calmar", "sortino", "final_nav",
                              "total_turnover", "total_tc", "n_obs"]]
    print("\n=== Headline metrics ===")
    print(metrics_df.to_string(float_format=lambda x: f"{x:8.4f}"))
    metrics_df.to_csv(OUT / "phase6_baseline_metrics.csv")

    # Spec gate checks.
    print("\n=== Spec gate checks ===")
    bayes_sharpe = metrics["BayesVI+PF"]["sharpe"]
    six40_sharpe = metrics["60/40"]["sharpe"]
    rand_sharpe = metrics["RandomRegime"]["sharpe"]
    print(f"  Bayes Sharpe vs 60/40 Sharpe: {bayes_sharpe:.3f} vs {six40_sharpe:.3f}  "
          f"({'PASS' if bayes_sharpe >= six40_sharpe else 'FAIL — debug per spec'})")
    print(f"  Random Sharpe should be < Bayes Sharpe: "
          f"{rand_sharpe:.3f} vs {bayes_sharpe:.3f}  "
          f"({'PASS' if rand_sharpe < bayes_sharpe else 'WARNING — model not adding info'})")

    # Equity curves.
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, res in results.items():
        eq = res.equity_curve.dropna()
        eq = eq[eq > 0]
        ax.plot(eq.index, eq.values, label=name, linewidth=1.4)
    ax.set_yscale("log")
    ax.set_title("Equity curves (initial NAV = 1.0)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = OUT / "phase6_baseline_equity.png"
    fig.savefig(out_png, dpi=130)
    log.info("saved %s", out_png)


if __name__ == "__main__":
    main()
