"""Data ingestion for the regime-switching model.

Pulls daily Close prices for the cross-asset universe from Yahoo Finance,
stitches in pre-listing proxies for the two ETFs that don't reach 2005,
and caches the result to parquet to avoid repeated network calls.

The canonical output is a wide DataFrame of daily log returns indexed by
trading date, with one column per asset symbol.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Universe definition
# --------------------------------------------------------------------------- #

# Allocation universe: tradable ETFs only. ^VIX is excluded (not investable).
ALLOCATION_TICKERS: tuple[str, ...] = ("SPY", "TLT", "GLD", "UUP", "HYG")

# Feature universe: ALLOCATION_TICKERS plus VIX (informative regime signal).
FEATURE_TICKERS: tuple[str, ...] = ALLOCATION_TICKERS + ("VIX",)

# Symbol -> (primary yfinance ticker, optional pre-listing proxy ticker, handover date).
# Returns from the proxy are spliced in for dates strictly before the handover.
TICKER_SPECS: dict[str, tuple[str, str | None, str | None]] = {
    "SPY": ("SPY", None, None),
    "TLT": ("TLT", None, None),
    "GLD": ("GLD", None, None),
    "UUP": ("UUP", "DX-Y.NYB", "2007-03-01"),  # UUP listed 2007-02-20; first clean day 2007-03-01
    "HYG": ("HYG", "VWEHX", "2007-04-11"),     # HYG listed 2007-04-11
    "VIX": ("^VIX", None, None),
}

DEFAULT_START = "2005-01-01"


@dataclass(frozen=True)
class DataBundle:
    """Returned by `load_universe`. Holds aligned price and log-return frames."""

    prices: pd.DataFrame   # adjusted close levels (synthetic for backfilled segments)
    returns: pd.DataFrame  # daily log returns
    spliced: dict[str, pd.Timestamp]  # symbol -> handover date for stitched series


# --------------------------------------------------------------------------- #
# Raw download
# --------------------------------------------------------------------------- #

def _download_close(ticker: str, start: str, end: str | None) -> pd.Series:
    """Download a single ticker's adjusted Close as a Series.

    auto_adjust=True folds dividends/splits into the price series, which is what
    we want for total-return analysis. yfinance returns a MultiIndex column when
    given one ticker; we flatten it.
    """
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker!r}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Close" not in df.columns:
        raise RuntimeError(f"No Close column in {ticker!r} download (got {list(df.columns)})")
    s = df["Close"].astype(float).dropna()
    s.name = ticker
    s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    return s


# --------------------------------------------------------------------------- #
# Backfill stitching
# --------------------------------------------------------------------------- #

def _stitch_returns(
    primary: pd.Series,
    proxy: pd.Series,
    handover: pd.Timestamp,
) -> tuple[pd.Series, pd.Timestamp]:
    """Splice proxy returns onto primary returns, keyed off the handover date.

    For dates strictly before `handover` we use proxy log returns; on/after we
    use primary log returns. The two series may have slightly different trading
    calendars; we keep the union and forward-fill nothing (NaNs propagate so the
    feature layer can decide how to handle them — typically by dropping until
    all features are populated).
    """
    primary_ret = np.log(primary / primary.shift(1))
    proxy_ret = np.log(proxy / proxy.shift(1))

    pre = proxy_ret.loc[proxy_ret.index < handover]
    post = primary_ret.loc[primary_ret.index >= handover]

    # Drop overlap on the seam (post wins by construction).
    pre = pre.loc[~pre.index.isin(post.index)]

    stitched = pd.concat([pre, post]).sort_index()
    stitched.name = primary.name
    return stitched, handover


def _reconstruct_level(returns: pd.Series, base: float = 1.0) -> pd.Series:
    """Build a level series from log returns, anchored at `base` on the first day.

    The first observation is always the anchor (level = base) regardless of
    whether returns.iloc[0] is finite or NaN. Subsequent levels follow the
    cumulative log-return path, so log(levels / levels.shift(1)) recovers the
    input returns from index 1 onward.
    """
    if returns.empty:
        return pd.Series(index=returns.index, dtype=float, name=returns.name)
    cum = returns.fillna(0.0).cumsum()
    cum = cum - cum.iloc[0]  # anchor: level at first index = base
    levels = base * np.exp(cum)
    levels.name = returns.name
    return levels


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def load_universe(
    start: str = DEFAULT_START,
    end: str | None = None,
    cache_dir: Path | str | None = None,
    refresh: bool = False,
) -> DataBundle:
    """Load the cross-asset feature universe.

    Parameters
    ----------
    start, end : str | None
        Date range (inclusive of `start`, exclusive of `end` per yfinance).
        `end=None` means "through the latest trading day."
    cache_dir : Path | str | None
        Directory for parquet caches. Defaults to `regime_model/cache/`.
        Pass `None`-equivalent by setting `refresh=True` to force a re-download.
    refresh : bool
        If True, ignore any cached data and re-pull from yfinance.

    Returns
    -------
    DataBundle with `prices`, `returns`, and `spliced` (handover dates).
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else Path(__file__).resolve().parents[1] / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_key = f"universe_{start}_{end or 'latest'}.parquet"
    cache_path = cache_dir / cache_key

    if cache_path.exists() and not refresh:
        logger.info("loading cached universe from %s", cache_path)
        prices = pd.read_parquet(cache_path)
        prices.index = pd.DatetimeIndex(prices.index)
        returns = np.log(prices / prices.shift(1))
        spliced = {
            sym: pd.Timestamp(spec[2])
            for sym, spec in TICKER_SPECS.items()
            if spec[2] is not None
        }
        return DataBundle(prices=prices, returns=returns, spliced=spliced)

    # Download every primary and proxy series we need.
    needed: set[str] = set()
    for sym in FEATURE_TICKERS:
        primary, proxy, _ = TICKER_SPECS[sym]
        needed.add(primary)
        if proxy is not None:
            needed.add(proxy)

    raw: dict[str, pd.Series] = {}
    for tkr in sorted(needed):
        logger.info("downloading %s", tkr)
        raw[tkr] = _download_close(tkr, start=start, end=end)

    # Build the per-symbol stitched return series.
    returns_by_sym: dict[str, pd.Series] = {}
    spliced: dict[str, pd.Timestamp] = {}
    for sym in FEATURE_TICKERS:
        primary_tkr, proxy_tkr, handover = TICKER_SPECS[sym]
        primary = raw[primary_tkr]
        if proxy_tkr is None:
            ret = np.log(primary / primary.shift(1))
            ret.name = sym
        else:
            proxy = raw[proxy_tkr]
            ret, hand_ts = _stitch_returns(primary, proxy, pd.Timestamp(handover))
            spliced[sym] = hand_ts
        returns_by_sym[sym] = ret

    returns = pd.concat(returns_by_sym.values(), axis=1).sort_index()
    returns.columns = list(returns_by_sym.keys())

    # Reconstruct a level series per symbol so plots/diagnostics are continuous.
    prices = pd.concat(
        [_reconstruct_level(returns[sym]) for sym in FEATURE_TICKERS],
        axis=1,
    )
    prices.columns = list(FEATURE_TICKERS)
    prices = prices.sort_index()

    # Trim to the date range the user actually asked for.
    prices = prices.loc[(prices.index >= pd.Timestamp(start))]
    if end is not None:
        prices = prices.loc[prices.index < pd.Timestamp(end)]

    prices.to_parquet(cache_path)
    logger.info("cached universe to %s (%d rows, %d cols)", cache_path, len(prices), prices.shape[1])

    returns = np.log(prices / prices.shift(1))

    return DataBundle(prices=prices, returns=returns, spliced=spliced)
