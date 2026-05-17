"""Transaction-cost models for Strategy Lab.

A cost model converts a trade (delta in weights) into a NAV haircut. v1 ships
with `FlatBpsPerLeg` — a configurable basis-points charge on the absolute
change in each leg, which is realistic for liquid ETFs.

Adding a new model: subclass `CostModel` and implement `apply`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class CostModel(ABC):
    """Abstract transaction-cost model."""

    @abstractmethod
    def apply(self, weights_prev: np.ndarray, weights_target: np.ndarray) -> float:
        """Return the proportional cost (fraction of NAV) of going from
        `weights_prev` to `weights_target`. Should be in [0, 1].

        Both inputs are 1-D numpy arrays aligned to the same universe.
        """


@dataclass(frozen=True)
class FlatBpsPerLeg(CostModel):
    """Flat basis-points charge on |Δw| summed across legs.

    `bps=5.0` charges 0.05% of NAV per unit of |Δw|. For example, going from
    {SPY: 0.6, TLT: 0.4} to {SPY: 0.5, TLT: 0.5} has total |Δw|=0.2, so cost
    is 0.2 * 5/1e4 = 0.01% of NAV.
    """

    bps: float = 5.0

    def apply(self, weights_prev: np.ndarray, weights_target: np.ndarray) -> float:
        trade = float(np.abs(weights_target - weights_prev).sum())
        return trade * (self.bps / 1e4)


@dataclass(frozen=True)
class ZeroCost(CostModel):
    """Pretend trading is free — useful for sanity tests."""

    def apply(self, weights_prev: np.ndarray, weights_target: np.ndarray) -> float:
        return 0.0
