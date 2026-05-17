"""Fundamentals data via yfinance.

HONEST LIMITATIONS:
  - yfinance's `Ticker.info` is a current snapshot, not a historical
    time series. We can read it, but we cannot say "what was AAPL's PE
    ratio on 2018-06-15" — only "what is it today."
  - `Ticker.quarterly_financials` and `.quarterly_balance_sheet` DO
    have history (~5 years), but the rows present vary by company and
    by year, and yfinance occasionally returns garbled / missing rows.
  - For point-in-time historical fundamentals at decision-time (so a
    backtest doesn't peek at restated numbers), you need a paid source
    like Sharadar, Norgate, or FactSet. yfinance is good enough for
    rough cross-sectional ranking on a forward-looking basis.

This module gives strategies a `FundamentalsSnapshot` per ticker for
*current* characteristics (use case: rank a basket by PE, dividend
yield, etc. when generating today's weights — no lookahead concern
because we're using current data, not historical).

For walk-forward use that requires point-in-time, this is NOT the
right tool — document this loudly so the user doesn't accidentally
hand themselves false alpha.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FundamentalsSnapshot:
    """Current-snapshot fundamentals for a single ticker.

    All values may be None when yfinance doesn't have them.
    """
    ticker: str
    market_cap: float | None
    trailing_pe: float | None
    forward_pe: float | None
    dividend_yield: float | None
    price_to_book: float | None
    profit_margins: float | None
    return_on_equity: float | None
    revenue_growth: float | None
    free_cash_flow: float | None
    beta: float | None
    sector: str | None
    industry: str | None


_INFO_KEYS = {
    "market_cap": "marketCap",
    "trailing_pe": "trailingPE",
    "forward_pe": "forwardPE",
    "dividend_yield": "dividendYield",
    "price_to_book": "priceToBook",
    "profit_margins": "profitMargins",
    "return_on_equity": "returnOnEquity",
    "revenue_growth": "revenueGrowth",
    "free_cash_flow": "freeCashflow",
    "beta": "beta",
    "sector": "sector",
    "industry": "industry",
}


def load_fundamentals_snapshot(ticker: str) -> FundamentalsSnapshot:
    """Pull current-snapshot fundamentals for one ticker.

    Returns a snapshot where missing values are None. Network call; not cached.
    """
    import yfinance as yf

    t = yf.Ticker(ticker)
    try:
        info = t.info or {}
    except Exception as e:
        logger.warning("yfinance .info failed for %s: %s", ticker, e)
        info = {}

    fields: dict[str, Any] = {"ticker": ticker.upper()}
    for our_key, yf_key in _INFO_KEYS.items():
        v = info.get(yf_key)
        if v is None or (isinstance(v, float) and not _isfinite(v)):
            fields[our_key] = None
        else:
            fields[our_key] = v
    return FundamentalsSnapshot(**fields)


def load_fundamentals_for_universe(tickers: list[str]) -> pd.DataFrame:
    """Return a DataFrame of current snapshots, one row per ticker."""
    rows = []
    for t in tickers:
        try:
            snap = load_fundamentals_snapshot(t)
        except Exception as e:
            logger.warning("fundamentals fetch failed for %s: %s", t, e)
            continue
        rows.append({
            "ticker": snap.ticker,
            "market_cap": snap.market_cap,
            "trailing_pe": snap.trailing_pe,
            "forward_pe": snap.forward_pe,
            "dividend_yield": snap.dividend_yield,
            "price_to_book": snap.price_to_book,
            "profit_margins": snap.profit_margins,
            "return_on_equity": snap.return_on_equity,
            "revenue_growth": snap.revenue_growth,
            "free_cash_flow": snap.free_cash_flow,
            "beta": snap.beta,
            "sector": snap.sector,
            "industry": snap.industry,
        })
    return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()


def _isfinite(x: float) -> bool:
    return x == x and x not in (float("inf"), float("-inf"))


# --------------------------------------------------------------------------- #
# Example strategy
# --------------------------------------------------------------------------- #

DIVIDEND_YIELD_EXAMPLE = '''
# Example: dividend-yield tilt (current snapshot, no point-in-time)

from lab.strategy import Strategy
from lab.fundamentals import load_fundamentals_for_universe
import pandas as pd

class DividendYieldTilt(Strategy):
    """Weight assets proportional to current dividend yield.

    WARNING: uses current snapshot — fundamentals are loaded ONCE in fit()
    and held fixed for the whole backtest. Not point-in-time. Real walk-
    forward research needs historical fundamentals (Sharadar / Norgate).
    """

    name = "Dividend yield tilt"

    def __init__(self, default_weight: float = 0.1):
        self.default_weight = default_weight
        self._weights = None

    def fit(self, history: pd.DataFrame) -> None:
        funds = load_fundamentals_for_universe(list(history.columns))
        if funds.empty or "dividend_yield" not in funds.columns:
            self._weights = pd.Series(
                self.default_weight, index=history.columns,
            )
            return
        y = funds["dividend_yield"].fillna(0.0).clip(lower=0.0)
        total = float(y.sum())
        if total <= 0:
            self._weights = pd.Series(
                self.default_weight, index=history.columns,
            )
            return
        self._weights = (y / total).reindex(history.columns).fillna(0.0)

    def rebalance(self, date, history) -> pd.Series:
        if self._weights is None:
            return pd.Series(0.0, index=history.columns)
        return self._weights.copy()
'''
