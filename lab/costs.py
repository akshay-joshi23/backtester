"""Transaction- and holding-cost models for Strategy Lab.

A cost model returns NAV haircuts for:
  - `trade_cost(prev, target)` — charged on rebalance, proportional to |Δw|
  - `holding_cost(weights, dt)` — charged every period, e.g. for shorts

Both default to 0. Subclasses can implement one or both. `CompositeCostModel`
chains several models so you can stack a per-leg trade cost + slippage +
borrow cost.

The `apply()` method is preserved for backward compatibility — it now calls
`trade_cost`, so any existing CostModel subclass keeps working.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field

import numpy as np


class CostModel(ABC):
    """Abstract cost model. Override `trade_cost` and/or `holding_cost`."""

    def trade_cost(
        self,
        weights_prev: np.ndarray,
        weights_target: np.ndarray,
        tickers: list[str] | None = None,
    ) -> float:
        """Proportional cost of trading from prev to target weights."""
        return 0.0

    def holding_cost(
        self,
        weights: np.ndarray,
        tickers: list[str] | None = None,
        dt_years: float = 1.0 / 252.0,
    ) -> float:
        """Proportional cost of holding the given weights for one period.

        `dt_years` defaults to one trading day (1/252). Multiply your
        annualized rates by `dt_years` to get the per-period charge.
        """
        return 0.0

    def apply(
        self,
        weights_prev: np.ndarray,
        weights_target: np.ndarray,
    ) -> float:
        """Backward-compat shim — returns trade_cost only."""
        return self.trade_cost(weights_prev, weights_target)


@dataclass(frozen=True)
class FlatBpsPerLeg(CostModel):
    """Flat basis-points charge on |Δw| summed across legs.

    `bps=5.0` charges 0.05% of NAV per unit of |Δw|. For example, going from
    {SPY: 0.6, TLT: 0.4} to {SPY: 0.5, TLT: 0.5} has total |Δw|=0.2, so cost
    is 0.2 * 5/1e4 = 0.01% of NAV.
    """

    bps: float = 5.0

    def trade_cost(
        self,
        weights_prev: np.ndarray,
        weights_target: np.ndarray,
        tickers: list[str] | None = None,
    ) -> float:
        trade = float(np.abs(weights_target - weights_prev).sum())
        return trade * (self.bps / 1e4)


@dataclass(frozen=True)
class ZeroCost(CostModel):
    """Pretend trading is free — useful for sanity tests."""


# --------------------------------------------------------------------------- #
# E-14 v2: borrow-cost model for shorts
# --------------------------------------------------------------------------- #

#: Conservative defaults for major US ETFs (annualized borrow %). Real IBKR
#: borrow rates for SPY/QQQ/IWM are typically <0.5%; here we use 0.5% to
#: avoid making strategies look better than they are.
DEFAULT_BORROW_RATES_BPS: dict[str, float] = {
    "SPY": 50, "QQQ": 50, "IWM": 50, "DIA": 50, "VOO": 50, "VTI": 50,
    "TLT": 75, "IEF": 75, "SHY": 50,
    "GLD": 75, "SLV": 100,
    "HYG": 150, "LQD": 100,
    "UUP": 100, "FXE": 100, "FXY": 100,
    "EFA": 75, "EEM": 100, "VWO": 100,
    "VNQ": 100,
    "XLE": 100, "XLF": 100, "XLK": 75, "XLY": 100, "XLP": 100, "XLV": 100,
    "XLI": 100, "XLB": 100, "XLU": 100, "XLRE": 100,
}
#: Annualized borrow rate (bps) applied to any ticker not in the table.
DEFAULT_BORROW_FALLBACK_BPS: float = 300.0


@dataclass(frozen=True)
class BorrowCost(CostModel):
    """Annualized borrow cost on the short side, charged daily.

    Conservative approximation — real borrow rates vary by name, by lender,
    and by date. The defaults are tuned for major US ETFs and lean
    pessimistic. For single-name shorts, override `rates` with the actual
    IBKR/your-broker schedule.
    """

    rates_bps: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_BORROW_RATES_BPS))
    fallback_bps: float = DEFAULT_BORROW_FALLBACK_BPS

    def holding_cost(
        self,
        weights: np.ndarray,
        tickers: list[str] | None = None,
        dt_years: float = 1.0 / 252.0,
    ) -> float:
        if tickers is None or len(tickers) != len(weights):
            return 0.0
        cost = 0.0
        for w, t in zip(weights, tickers):
            if w < -1e-12:  # only shorts pay borrow
                rate = self.rates_bps.get(t.upper(), self.fallback_bps) / 1e4
                cost += abs(float(w)) * rate * dt_years
        return cost


@dataclass(frozen=True)
class CompositeCostModel(CostModel):
    """Sum of multiple cost models. trade_cost and holding_cost stack additively."""

    models: tuple[CostModel, ...] = ()

    def trade_cost(
        self,
        weights_prev: np.ndarray,
        weights_target: np.ndarray,
        tickers: list[str] | None = None,
    ) -> float:
        return sum(m.trade_cost(weights_prev, weights_target, tickers) for m in self.models)

    def holding_cost(
        self,
        weights: np.ndarray,
        tickers: list[str] | None = None,
        dt_years: float = 1.0 / 252.0,
    ) -> float:
        return sum(m.holding_cost(weights, tickers, dt_years) for m in self.models)
