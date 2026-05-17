"""Static baseline strategies."""

from __future__ import annotations

import pandas as pd

from lab.strategy import Strategy


class FixedMix(Strategy):
    """Hold a fixed weight vector, rebalance whenever the engine asks.

    Example: `FixedMix({"SPY": 0.6, "TLT": 0.4})` → classic 60/40.
    """

    def __init__(self, weights: dict[str, float], name: str | None = None):
        if any(v < 0 for v in weights.values()):
            raise ValueError("FixedMix weights must be non-negative")
        total = sum(weights.values())
        if total > 1.0 + 1e-9:
            raise ValueError(f"FixedMix weights sum to {total:.4f} > 1.0")
        self._weights = pd.Series(weights, dtype=float)
        self.name = name or f"FixedMix({dict(weights)})"

    def rebalance(self, date: pd.Timestamp, history: pd.DataFrame) -> pd.Series:
        return self._weights.copy()


class EqualWeight(Strategy):
    """Equal-weight across all tickers in the universe."""

    name = "EqualWeight"

    def __init__(self):
        self._tickers: list[str] | None = None

    def fit(self, history: pd.DataFrame) -> None:
        self._tickers = list(history.columns)

    def rebalance(self, date: pd.Timestamp, history: pd.DataFrame) -> pd.Series:
        tickers = self._tickers or list(history.columns)
        n = len(tickers)
        return pd.Series(1.0 / n, index=tickers)
