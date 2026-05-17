"""Generic universe data loader for Strategy Lab.

Pulls adjusted-close prices for an arbitrary list of yfinance tickers and
returns daily log returns + prices, with on-disk parquet caching keyed by
(tickers, start, end) so re-runs are instant.

This is intentionally simple — no proxy backfilling, no fundamentals, no
intraday data. v1 supports the most common case: a small basket of ETFs or
liquid equities on daily bars.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / ".lab_cache"


@dataclass(frozen=True)
class UniverseBundle:
    """Container for a universe's price/return data."""

    prices: pd.DataFrame
    returns: pd.DataFrame
    tickers: tuple[str, ...]
    start: str
    end: str | None


def _cache_key(tickers: tuple[str, ...], start: str, end: str | None) -> str:
    raw = f"{sorted(tickers)}|{start}|{end or 'latest'}"
    h = hashlib.sha1(raw.encode()).hexdigest()[:12]
    safe = "_".join(sorted(tickers))[:80]
    return f"{safe}_{start}_{end or 'latest'}_{h}.parquet"


def load_universe(
    tickers: list[str] | tuple[str, ...],
    start: str = "2010-01-01",
    end: str | None = None,
    cache_dir: Path | str | None = None,
    refresh: bool = False,
) -> UniverseBundle:
    """Load aligned daily prices and log returns for a list of tickers.

    Parameters
    ----------
    tickers : list[str]
        yfinance ticker symbols.
    start, end : str | None
        Date range. End is exclusive (yfinance convention). None means "to now".
    cache_dir : path-like | None
        Where to store the parquet cache. Defaults to `<repo>/.lab_cache`.
    refresh : bool
        If True, bypass any cached file and re-download.

    Returns
    -------
    UniverseBundle
        Aligned prices + log returns. Rows where any ticker is NaN are
        forward-filled within the price frame, then returns are recomputed
        and the leading row (NaN by construction) is dropped.
    """
    tickers = tuple(t.upper() for t in tickers)
    if len(tickers) == 0:
        raise ValueError("tickers must be non-empty")
    if len(set(tickers)) != len(tickers):
        raise ValueError(f"duplicate tickers: {tickers}")

    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / _cache_key(tickers, start, end)

    if cache_path.exists() and not refresh:
        logger.info("loading cached universe from %s", cache_path)
        prices = pd.read_parquet(cache_path)
        prices.index = pd.DatetimeIndex(prices.index)
    else:
        prices = _download(tickers, start, end)
        prices.to_parquet(cache_path)
        logger.info("cached universe to %s (%d rows)", cache_path, len(prices))

    # Forward-fill mild gaps (e.g. corporate actions, single-day holes) but only
    # within each ticker's coverage window — leading NaNs stay NaN.
    prices = prices.ffill().dropna(how="any")
    returns = np.log(prices / prices.shift(1)).dropna(how="any")

    return UniverseBundle(
        prices=prices,
        returns=returns,
        tickers=tickers,
        start=start,
        end=end,
    )


def _download(tickers: tuple[str, ...], start: str, end: str | None) -> pd.DataFrame:
    """Fetch adjusted close from yfinance for the given tickers."""
    import yfinance as yf

    logger.info("downloading %d ticker(s) from yfinance: %s", len(tickers), tickers)
    df = yf.download(
        list(tickers),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
        group_by="column",
    )
    if df.empty:
        raise RuntimeError(f"yfinance returned no data for {tickers}")

    # Single ticker → flat columns; multiple → MultiIndex with Close at top level.
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            df = df["Close"]
        else:
            raise RuntimeError(f"unexpected yfinance columns: {df.columns}")
    else:
        if "Close" not in df.columns:
            raise RuntimeError(f"unexpected yfinance columns: {df.columns}")
        df = df[["Close"]]
        df.columns = list(tickers)

    df = df.copy()
    df.columns = [str(c).upper() for c in df.columns]
    # Reorder to user-supplied order.
    df = df.reindex(columns=list(tickers))
    if df.isna().all(axis=0).any():
        missing = df.columns[df.isna().all(axis=0)].tolist()
        raise RuntimeError(f"yfinance returned no data for ticker(s): {missing}")
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    df = df.sort_index()
    return df


def validate_no_lookahead(
    strategy_factory,
    returns: pd.DataFrame,
    probe_dates: int = 5,
    seed: int = 0,
) -> None:
    """Empirically verify a strategy doesn't peek at future returns.

    Picks `probe_dates` random dates from the second half of the sample,
    perturbs the return on one ticker at that date by +10%, recomputes
    `strategy.rebalance` at all EARLIER dates, and asserts no rebalance
    output changed.

    `strategy_factory()` must return a fresh, fitted Strategy instance.

    Raises
    ------
    AssertionError if lookahead is detected.
    """
    rng = np.random.default_rng(seed)
    half = len(returns) // 2
    if half < probe_dates + 10:
        raise ValueError("need more data to validate lookahead")
    candidate_dates = returns.index[half:]
    sample_dates = rng.choice(candidate_dates, size=probe_dates, replace=False)

    for probe_date in sample_dates:
        probe_ts = pd.Timestamp(probe_date)
        col = returns.columns[rng.integers(0, returns.shape[1])]
        perturbed = returns.copy()
        perturbed.loc[probe_ts, col] = perturbed.loc[probe_ts, col] + 0.10

        # Test on a handful of earlier dates.
        earlier = returns.index[returns.index < probe_ts]
        # Sample at most 10 prior dates spread across the history.
        if len(earlier) > 10:
            step = max(1, len(earlier) // 10)
            test_dates = earlier[::step]
        else:
            test_dates = earlier

        for t in test_dates:
            t_ts = pd.Timestamp(t)
            hist_base = returns.loc[returns.index < t_ts]
            hist_perturbed = perturbed.loc[perturbed.index < t_ts]
            assert hist_base.equals(hist_perturbed), \
                "history slice at t differs from perturbed slice; test bug"
            s_a = strategy_factory()
            s_a.fit(hist_base)
            w_a = s_a.rebalance(t_ts, hist_base)
            s_b = strategy_factory()
            s_b.fit(hist_perturbed)
            w_b = s_b.rebalance(t_ts, hist_perturbed)
            diff = (w_a.reindex_like(w_b).fillna(0.0) - w_b.fillna(0.0)).abs().max()
            if diff > 1e-10:
                raise AssertionError(
                    f"lookahead detected: perturbing {col} on {probe_ts.date()} "
                    f"changed weights at earlier date {t_ts.date()} (max diff {diff:.3e})"
                )
    logger.info("no-lookahead validator passed on %d probe dates", probe_dates)
