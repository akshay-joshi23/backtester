"""Tests for feature engineering. Uses synthetic data — no network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_model.data.features import (
    CORR_PAIRS,
    FEATURE_DIM,
    build_features,
    build_raw_features,
    expanding_zscore,
    validate_no_lookahead,
)
from regime_model.data.loaders import FEATURE_TICKERS


# --------------------------------------------------------------------------- #
# Synthetic data fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def synthetic_returns() -> pd.DataFrame:
    """4 years of independent N(0, 0.01) daily log returns for the 6 assets."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2010-01-01", "2014-01-01")
    data = rng.normal(loc=0.0, scale=0.01, size=(len(dates), len(FEATURE_TICKERS)))
    return pd.DataFrame(data, index=dates, columns=list(FEATURE_TICKERS))


# --------------------------------------------------------------------------- #
# Shape / column tests
# --------------------------------------------------------------------------- #

def test_feature_dim_is_14(synthetic_returns: pd.DataFrame) -> None:
    raw, groups = build_raw_features(synthetic_returns)
    assert raw.shape[1] == FEATURE_DIM == 14
    assert len(groups["returns"]) == 6
    assert len(groups["vols"]) == 6
    assert len(groups["corrs"]) == 2


def test_feature_columns_have_expected_names(synthetic_returns: pd.DataFrame) -> None:
    raw, _ = build_raw_features(synthetic_returns)
    expected_returns = {f"ret1d_{t}" for t in FEATURE_TICKERS}
    expected_vols = {f"vol20_{t}" for t in FEATURE_TICKERS}
    expected_corrs = {f"corr60_{a}_{b}" for a, b in CORR_PAIRS}
    cols = set(raw.columns)
    assert expected_returns.issubset(cols)
    assert expected_vols.issubset(cols)
    assert expected_corrs.issubset(cols)


def test_feature_index_matches_returns_index(synthetic_returns: pd.DataFrame) -> None:
    raw, _ = build_raw_features(synthetic_returns)
    assert raw.index.equals(synthetic_returns.index)


def test_features_reject_missing_columns(synthetic_returns: pd.DataFrame) -> None:
    bad = synthetic_returns.drop(columns=["VIX"])
    with pytest.raises(ValueError, match="missing required columns"):
        build_raw_features(bad)


# --------------------------------------------------------------------------- #
# Lag / no-look-ahead semantics
# --------------------------------------------------------------------------- #

def test_first_row_features_are_nan(synthetic_returns: pd.DataFrame) -> None:
    """All features are shifted by 1, so row 0 must be entirely NaN."""
    raw, _ = build_raw_features(synthetic_returns)
    assert raw.iloc[0].isna().all()


def test_return_feature_at_t_equals_return_at_tminus1(synthetic_returns: pd.DataFrame) -> None:
    raw, _ = build_raw_features(synthetic_returns)
    # ret1d_SPY at row 5 should equal SPY return at row 4.
    assert raw["ret1d_SPY"].iloc[5] == pytest.approx(synthetic_returns["SPY"].iloc[4])


def test_validator_passes_on_correct_features(synthetic_returns: pd.DataFrame) -> None:
    # Should not raise.
    validate_no_lookahead(synthetic_returns, probe_dates=4, seed=1)


def test_validator_catches_leaky_builder(synthetic_returns: pd.DataFrame) -> None:
    """A deliberately leaky builder (no shift on the return block) must trip
    the validator."""
    def leaky_builder(returns: pd.DataFrame):
        # Same as build_raw_features but without the .shift(1) on ret1d.
        ret_block = returns.copy()
        ret_block.columns = [f"ret1d_{c}" for c in returns.columns]
        # vol and corr blocks: shift(1) so they're clean — only the return block leaks.
        vol_block = (returns.rolling(20).std() * np.sqrt(252)).shift(1)
        vol_block.columns = [f"vol20_{c}" for c in returns.columns]
        corr_block = pd.concat(
            [
                returns[a].rolling(60).corr(returns[b]).shift(1).rename(f"corr60_{a}_{b}")
                for a, b in CORR_PAIRS
            ],
            axis=1,
        )
        out = pd.concat([ret_block, vol_block, corr_block], axis=1)
        return out, {"returns": list(ret_block.columns), "vols": list(vol_block.columns), "corrs": list(corr_block.columns)}

    with pytest.raises(AssertionError, match="look-ahead detected"):
        validate_no_lookahead(synthetic_returns, builder=leaky_builder, probe_dates=4, seed=1)


# --------------------------------------------------------------------------- #
# Standardization
# --------------------------------------------------------------------------- #

def test_zscore_uses_only_past_data(synthetic_returns: pd.DataFrame) -> None:
    """If we change the LAST feature value, no earlier zscored value should move."""
    raw, _ = build_raw_features(synthetic_returns)
    z = expanding_zscore(raw, min_periods=60)

    raw_perturbed = raw.copy()
    raw_perturbed.iloc[-1, 0] += 100.0  # huge perturbation to last cell

    z_perturbed = expanding_zscore(raw_perturbed, min_periods=60)
    earlier = z.index[:-1]
    diff = (z_perturbed.loc[earlier] - z.loc[earlier]).abs().max().max()
    assert pd.isna(diff) or diff < 1e-9


def test_zscore_warmup_is_nan(synthetic_returns: pd.DataFrame) -> None:
    bundle = build_features(synthetic_returns, min_zscore_periods=252)
    early = bundle.standardized.iloc[:200]
    assert early.isna().all().all()


def test_zscore_is_finite_after_warmup(synthetic_returns: pd.DataFrame) -> None:
    bundle = build_features(synthetic_returns, min_zscore_periods=252)
    # Skip the first warmup + an extra buffer for the rolling-window features.
    after = bundle.standardized.iloc[300:]
    finite_share = np.isfinite(after.to_numpy()).mean()
    assert finite_share > 0.95, f"too many NaNs/Infs after warmup: finite share {finite_share:.3f}"
