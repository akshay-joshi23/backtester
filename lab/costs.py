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


# --------------------------------------------------------------------------- #
# C-11 v2: bid-ask spread and square-root impact slippage
# --------------------------------------------------------------------------- #

#: Approximate effective bid-ask spread in bps for major ETFs. These are
#: round-trip-equivalent — the spread cost on one leg is half this number.
#: Values are typical median spreads during US market hours; tighter during
#: high-vol regimes, wider in pre/post-market.
DEFAULT_SPREAD_BPS: dict[str, float] = {
    "SPY": 1.0, "QQQ": 1.0, "IWM": 2.0, "DIA": 2.0,
    "VOO": 1.0, "VTI": 1.0,
    "TLT": 2.0, "IEF": 2.0, "SHY": 2.0,
    "GLD": 1.5, "SLV": 3.0,
    "HYG": 2.0, "LQD": 2.0,
    "UUP": 4.0, "FXE": 5.0, "FXY": 5.0,
    "EFA": 1.5, "EEM": 2.0, "VWO": 2.0,
    "VNQ": 2.0,
    "XLE": 1.5, "XLF": 1.5, "XLK": 1.5, "XLY": 1.5, "XLP": 1.5, "XLV": 1.5,
    "XLI": 1.5, "XLB": 2.0, "XLU": 2.0, "XLRE": 2.0,
}
DEFAULT_SPREAD_FALLBACK_BPS: float = 10.0


@dataclass(frozen=True)
class BidAskSpread(CostModel):
    """Half-spread per leg (paid on every trade).

    Cost = sum_i |Δw_i| * (spread_bps[i] / 2) / 1e4

    Half because each round-trip crosses the spread once at entry and once
    at exit; this model charges only the entry side, so each leg
    incurs half the round-trip cost.
    """

    spread_bps: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SPREAD_BPS))
    fallback_bps: float = DEFAULT_SPREAD_FALLBACK_BPS

    def trade_cost(
        self,
        weights_prev: np.ndarray,
        weights_target: np.ndarray,
        tickers: list[str] | None = None,
    ) -> float:
        if tickers is None or len(tickers) != len(weights_prev):
            # Fallback: assume fallback bps for every leg.
            trade = float(np.abs(weights_target - weights_prev).sum())
            return trade * (self.fallback_bps / 2.0) / 1e4
        delta = np.abs(weights_target - weights_prev)
        cost = 0.0
        for d, t in zip(delta, tickers):
            spread = self.spread_bps.get(t.upper(), self.fallback_bps)
            cost += float(d) * (spread / 2.0) / 1e4
        return cost


@dataclass(frozen=True)
class SquareRootImpact(CostModel):
    """Square-root market-impact slippage.

    Models the empirical observation that market impact grows as
    sqrt(trade_size / ADV). Charged on |Δw| for each leg, scaled by the
    impact_coefficient.

    For ETF rebalances at small AUM the trade size is tiny vs ADV so
    impact is negligible. This becomes meaningful when sizing > $1M per
    leg for less-liquid names.

    Approximation:
        impact_bps = impact_coef * sqrt(|Δw| * portfolio_aum / adv_dollars)

    SIMPLIFICATION: we don't know portfolio_aum or per-asset ADV in this
    framework. Instead we use a simple |Δw|^0.5 scaling — calibrated so
    impact_coef=10 produces ~5bps slippage on a 25% rebalance leg, which
    matches institutional rules of thumb for liquid ETFs.
    """

    impact_coef_bps: float = 10.0

    def trade_cost(
        self,
        weights_prev: np.ndarray,
        weights_target: np.ndarray,
        tickers: list[str] | None = None,
    ) -> float:
        delta = np.abs(weights_target - weights_prev)
        # impact per leg = coef * sqrt(|Δw|) in bps.
        cost = float((self.impact_coef_bps * np.sqrt(delta)).sum()) / 1e4
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


def realistic_cost_model(
    *,
    bps_per_leg: float = 0.0,
    use_spread: bool = True,
    use_impact: bool = True,
    use_borrow: bool = True,
    impact_coef_bps: float = 10.0,
) -> CompositeCostModel:
    """Convenience builder: chain spread + impact + borrow into a single model.

    Default leaves out FlatBpsPerLeg (BidAskSpread already covers per-leg cost).
    Pass `bps_per_leg > 0` to add a flat commission on top.
    """
    parts: list[CostModel] = []
    if bps_per_leg > 0:
        parts.append(FlatBpsPerLeg(bps=bps_per_leg))
    if use_spread:
        parts.append(BidAskSpread())
    if use_impact:
        parts.append(SquareRootImpact(impact_coef_bps=impact_coef_bps))
    if use_borrow:
        parts.append(BorrowCost())
    return CompositeCostModel(models=tuple(parts))
