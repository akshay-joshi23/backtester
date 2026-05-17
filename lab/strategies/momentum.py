"""Cross-sectional momentum strategy.

Buys the top-k tickers by trailing N-day return, equal-weight, with a cash
reserve. A simple, well-known baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lab.strategy import Strategy


class LongShortMomentum(Strategy):
    """Long top-k, short bottom-k by trailing return — dollar-neutral.

    Requires `long_only=False` in BacktestConfig (cfg.long_only=False).
    Gross exposure is `2 * target_leg_size * top_k` (capped by max_leverage).
    """

    def __init__(
        self,
        lookback: int = 126,
        top_k: int = 2,
        target_leg_size: float = 0.25,
        name: str | None = None,
    ):
        self.lookback = int(lookback)
        self.top_k = int(top_k)
        self.target_leg_size = float(target_leg_size)
        self.name = name or f"LSMomentum(lookback={lookback},top_k={top_k})"

    def rebalance(self, date: pd.Timestamp, history: pd.DataFrame) -> pd.Series:
        if len(history) < self.lookback or len(history.columns) < 2 * self.top_k:
            return pd.Series(0.0, index=history.columns)
        window = history.iloc[-self.lookback:]
        scores = window.sum(axis=0)
        longs = scores.nlargest(self.top_k).index
        shorts = scores.nsmallest(self.top_k).index
        w = pd.Series(0.0, index=history.columns)
        w.loc[longs] = self.target_leg_size
        w.loc[shorts] = -self.target_leg_size
        return w


class CrossSectionalMomentum(Strategy):
    """Long the top-k tickers by trailing return, equal-weight.

    Parameters
    ----------
    lookback : int
        Days of trailing returns to score. 126 ≈ 6 months.
    top_k : int
        How many tickers to hold. None → all positive-momentum tickers.
    target_leverage : float
        Sum of weights across the held names. 1.0 = fully invested.
    """

    def __init__(
        self,
        lookback: int = 126,
        top_k: int | None = 3,
        target_leverage: float = 1.0,
        name: str | None = None,
    ):
        self.lookback = int(lookback)
        self.top_k = top_k
        self.target_leverage = float(target_leverage)
        self.name = name or f"Momentum(lookback={lookback},top_k={top_k})"

    def rebalance(self, date: pd.Timestamp, history: pd.DataFrame) -> pd.Series:
        if len(history) < self.lookback:
            return pd.Series(0.0, index=history.columns)
        window = history.iloc[-self.lookback :]
        scores = window.sum(axis=0)  # cumulative log return over window
        if self.top_k is not None:
            chosen = scores.nlargest(self.top_k)
        else:
            chosen = scores[scores > 0]
        chosen = chosen[chosen > 0]  # require positive momentum
        if chosen.empty:
            return pd.Series(0.0, index=history.columns)
        w = pd.Series(0.0, index=history.columns)
        w.loc[chosen.index] = self.target_leverage / len(chosen)
        return w
