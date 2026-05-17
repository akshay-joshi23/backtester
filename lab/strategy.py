"""Strategy interface for Strategy Lab.

A `Strategy` is anything that, on a given trading date, looks at the history of
asset returns available so far and returns target portfolio weights for that
date. The walk-forward backtester guarantees no future information is in the
`history` argument.

Contract:
    * `rebalance(date, history)` returns a pandas Series indexed by ticker.
      Weights must be finite and sum to at most 1.0 (long-only by default).
      A weight of 0 means "no position." Missing tickers default to 0.
    * `fit(history)` is called once at the start of the backtest with the
      training-window history. Most strategies don't need it.
    * State is held on the strategy instance (`self.*`) — stateful strategies
      (Kalman filters, online ML, particle filters) just keep their state as
      instance attributes.

The backtester calls `rebalance` only on rebalance dates (per
`BacktestConfig.rebalance_freq`); between rebalances, the previous targets
drift with realized returns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class StrategyError(Exception):
    """Raised when a Strategy returns malformed weights."""


class Strategy(ABC):
    """Base class for backtestable strategies.

    Subclasses MUST implement `rebalance`. Optional: override `fit` for
    one-time training-window setup.
    """

    #: Human-readable name. Override on the subclass.
    name: str = "UnnamedStrategy"

    def fit(self, history: pd.DataFrame) -> None:
        """One-time setup on the training window. Default is no-op.

        `history` is a DataFrame of log returns indexed by trading date,
        with one column per ticker. Strategies that need to learn parameters
        from historical data should do so here.
        """
        return None

    @abstractmethod
    def rebalance(self, date: pd.Timestamp, history: pd.DataFrame) -> pd.Series:
        """Return target weights for `date`.

        Parameters
        ----------
        date : pd.Timestamp
            The trading date for which target weights are being requested.
        history : pd.DataFrame
            Log returns indexed by trading date, with one column per ticker.
            History contains rows strictly BEFORE `date` (no lookahead).

        Returns
        -------
        pd.Series
            Target weights indexed by ticker. Long-only: 0 <= w_i.
            Sum of weights must be in [0, 1] (cash is implied by 1 - sum).
            Tickers not present in the Series default to 0.
        """
        raise NotImplementedError

    # ----------------------------------------------------------------- #
    # Helpers for subclass authors
    # ----------------------------------------------------------------- #

    def validate_weights(
        self,
        weights: pd.Series,
        universe: list[str],
        long_only: bool = True,
        max_leverage: float = 1.0,
    ) -> pd.Series:
        """Sanitize a weights Series before returning from `rebalance`.

        - Reindexes to the full universe with 0-fill.
        - Replaces NaN/Inf with 0.
        - If `long_only`, clips negatives to 0.
        - Renormalizes if sum exceeds `max_leverage`.

        Returns the cleaned Series. Call this from your `rebalance`
        implementation if you want a defensive layer.
        """
        w = weights.reindex(universe).fillna(0.0)
        w = w.replace([np.inf, -np.inf], 0.0)
        if long_only:
            w = w.clip(lower=0.0)
        total = float(w.sum())
        if total > max_leverage:
            w = w * (max_leverage / total)
        return w
