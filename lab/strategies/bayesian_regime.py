"""Wire the regime-switching Bayesian model as a `lab` Strategy.

This is intentionally a thin wrapper around the existing `regime_model`
package — same VI fit, same regime-conditional MVO, same vol target. The
purpose is to demonstrate that the generic backtester can drive a
non-trivial stateful strategy (one that does Bayesian inference under the
hood) cleanly via the `Strategy.fit() + Strategy.rebalance()` interface.

Simplifications versus the full `regime_model` pipeline:
  - No yearly VI refit. We fit once in `fit()` on the training-window
    history and freeze parameters. (The full pipeline refits each year.)
  - No incremental particle filter. We run the forward-backward smoother
    on history-through-`date-1` on each rebalance. Slower than online
    filtering, but correctness-equivalent for the moments we care about.
  - Uses raw log returns as the feature vector (not the engineered
    feature set from `regime_model.data.features`). Simpler, and stays
    self-contained — the strategy doesn't need an external VIX series.

Default risk-aversion (λ=50), max_weight (0.4), vol target (10%) match
the values found to be best in the regime-model sweep.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from lab.strategy import Strategy

logger = logging.getLogger(__name__)


class BayesianRegime(Strategy):
    """Bayesian regime-switching cross-asset allocator.

    Pipeline at fit-time:
      1. Standardize the training-window returns (expanding-z).
      2. SVI fit a K-state Gaussian HMM (NumPyro).

    Pipeline at each rebalance:
      1. Standardize history-through-yesterday with the *frozen* mean/std
         from training (so OOS standardization is causal).
      2. Run forward-backward; take the last-row filtered posterior.
      3. Compute per-regime gamma-weighted μ_k, Σ_k on the same training
         + OOS-so-far data.
      4. Solve MVO per regime, mix by current posterior, vol-target.
    """

    name = "BayesianRegime"

    def __init__(
        self,
        K: int = 3,
        risk_aversion: float = 50.0,
        target_annual_vol: float = 0.10,
        max_weight: float = 0.4,
        vi_steps: int = 5000,
        seed: int = 0,
    ):
        self.K = K
        self.risk_aversion = risk_aversion
        self.target_annual_vol = target_annual_vol
        self.max_weight = max_weight
        self.vi_steps = vi_steps
        self.seed = seed
        # Fit-time state.
        self._pe = None         # PosteriorPointEstimate from VI
        self._mean = None       # training-window per-column mean (for OOS z-score)
        self._std = None        # training-window per-column std
        self._tickers: list[str] | None = None
        self._train_returns: pd.DataFrame | None = None

    def fit(self, history: pd.DataFrame) -> None:
        # Defer heavy imports until actually used so import-time stays cheap.
        from regime_model.inference.variational import SVIConfig, fit_svi
        from regime_model.models.bayesian_regime import ModelConfig

        self._tickers = list(history.columns)
        # Standardize features = standardized log returns over the training window.
        self._mean = history.mean()
        self._std = history.std().replace(0.0, 1.0)
        z = (history - self._mean) / self._std
        z = z.dropna()
        if len(z) < 252:
            raise ValueError(
                f"BayesianRegime needs >=252 training obs, got {len(z)}"
            )
        model_cfg = ModelConfig(K=self.K)
        svi_cfg = SVIConfig(
            n_steps=self.vi_steps, learning_rate=1e-3,
            seed=self.seed, progress=False,
        )
        logger.info("BayesianRegime.fit: SVI on %d obs, K=%d, %d steps",
                    len(z), self.K, self.vi_steps)
        result = fit_svi(z.to_numpy(), model_cfg=model_cfg, svi_cfg=svi_cfg,
                         n_posterior_samples=50)
        self._pe = result.point_estimate
        self._train_returns = history.copy()

    def rebalance(self, date: pd.Timestamp, history: pd.DataFrame) -> pd.Series:
        from regime_model.allocation.strategy import (
            StrategyConfig, regime_target_weights,
        )
        from regime_model.inference.variational import smoothed_posterior

        if self._pe is None:
            return pd.Series(0.0, index=history.columns)

        # OOS-causal z-score using frozen training-window mean/std.
        z = (history - self._mean) / self._std
        z = z.dropna()
        if len(z) < 60:
            return pd.Series(0.0, index=history.columns)

        # Run forward-backward; current regime probs = last row of gamma.
        gamma_np, _ = smoothed_posterior(z.to_numpy(), self._pe)
        gamma = pd.DataFrame(
            gamma_np, index=z.index,
            columns=[f"s{k}" for k in range(self.K)],
        )
        current_probs = gamma.iloc[-1].to_numpy()

        # Per-regime moments use the full history-so-far (train + OOS).
        cfg = StrategyConfig(
            max_weight=self.max_weight,
            risk_aversion=self.risk_aversion,
            target_annual_vol=self.target_annual_vol,
        )
        # regime_target_weights wants history aligned to gamma — use the same z index.
        ret_aligned = history.loc[z.index]
        weights_arr, _diag = regime_target_weights(
            ret_aligned, gamma, current_probs, cfg,
        )
        return pd.Series(weights_arr, index=history.columns)
