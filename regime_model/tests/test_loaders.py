"""Tests for the data loader.

Most tests are unit tests that operate on synthetic Series so we don't depend
on yfinance availability. One slow integration test pulls real data; mark it
with `slow` so it can be deselected in CI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_model.data.loaders import (
    ALLOCATION_TICKERS,
    DataBundle,
    FEATURE_TICKERS,
    TICKER_SPECS,
    _reconstruct_level,
    _stitch_returns,
    load_universe,
)


# --------------------------------------------------------------------------- #
# Stitch unit tests
# --------------------------------------------------------------------------- #

def _level_from_returns(rng: np.random.Generator, n: int, start: str) -> pd.Series:
    rets = rng.normal(0.0, 0.01, size=n)
    levels = 100.0 * np.exp(np.cumsum(rets))
    idx = pd.bdate_range(start, periods=n)
    return pd.Series(levels, index=idx, name="x")


def test_stitch_uses_proxy_before_handover() -> None:
    rng = np.random.default_rng(0)
    proxy = _level_from_returns(rng, 600, "2005-01-03")
    primary = _level_from_returns(rng, 400, "2007-04-11")
    primary.name = "HYG"
    handover = pd.Timestamp("2007-04-11")

    stitched, h = _stitch_returns(primary, proxy, handover)
    assert h == handover

    # Pre-handover days should equal proxy log returns.
    proxy_ret = np.log(proxy / proxy.shift(1))
    pre_dates = stitched.index[stitched.index < handover]
    pd.testing.assert_series_equal(
        stitched.loc[pre_dates], proxy_ret.loc[pre_dates], check_names=False
    )

    # Post-handover days should equal primary log returns.
    primary_ret = np.log(primary / primary.shift(1))
    post_dates = stitched.index[stitched.index >= handover]
    pd.testing.assert_series_equal(
        stitched.loc[post_dates], primary_ret.loc[post_dates], check_names=False
    )


def test_stitch_no_overlap() -> None:
    rng = np.random.default_rng(1)
    proxy = _level_from_returns(rng, 500, "2005-01-03")
    primary = _level_from_returns(rng, 300, "2007-04-11")
    handover = pd.Timestamp("2007-04-11")
    stitched, _ = _stitch_returns(primary, proxy, handover)
    # Index must be unique (no day appears twice from proxy + primary).
    assert stitched.index.is_unique


def test_reconstruct_level_round_trip() -> None:
    rng = np.random.default_rng(2)
    rets = pd.Series(
        rng.normal(0.0, 0.01, size=100),
        index=pd.bdate_range("2010-01-01", periods=100),
        name="x",
    )
    levels = _reconstruct_level(rets, base=1.0)
    # Check that log returns of the reconstructed level match the input
    # (after the first day, which carries the base).
    rebuilt = np.log(levels / levels.shift(1)).dropna()
    pd.testing.assert_series_equal(rebuilt, rets.iloc[1:], check_names=False)


# --------------------------------------------------------------------------- #
# Spec sanity
# --------------------------------------------------------------------------- #

def test_allocation_universe_excludes_vix() -> None:
    assert "VIX" not in ALLOCATION_TICKERS
    assert "VIX" in FEATURE_TICKERS


def test_specs_define_proxy_for_uup_and_hyg() -> None:
    assert TICKER_SPECS["UUP"][1] == "DX-Y.NYB"
    assert TICKER_SPECS["HYG"][1] == "VWEHX"
    # Other assets do not need a proxy.
    for sym in ("SPY", "TLT", "GLD", "VIX"):
        assert TICKER_SPECS[sym][1] is None


# --------------------------------------------------------------------------- #
# Integration test: hits yfinance, slow.
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_load_universe_smoke(tmp_path) -> None:
    bundle = load_universe(start="2010-01-01", end="2010-06-01", cache_dir=tmp_path)
    assert isinstance(bundle, DataBundle)
    assert list(bundle.prices.columns) == list(FEATURE_TICKERS)
    assert len(bundle.prices) > 50
    # Returns should be roughly mean-zero on this short window.
    daily_means = bundle.returns.mean().abs()
    assert (daily_means < 0.05).all()
