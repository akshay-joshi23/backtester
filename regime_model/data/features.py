"""Feature engineering for the regime-switching model.

Builds the 14-dimensional cross-asset feature vector consumed by the model:
    - 6 daily log returns       (per asset in FEATURE_TICKERS)
    - 6 realized vols (20d)     (per asset)
    - SPY-TLT 60d correlation
    - HYG-SPY 60d correlation

All features at time t are constructed from data strictly through t-1, so the
feature available at the *start* of trading day t is what the model conditions
on. Standardization uses an expanding-window mean/std also computed through
t-1, eliminating any in-sample standardization bias.

`validate_no_lookahead` is the runtime guard: it perturbs a single day's price
and asserts no earlier feature value moves.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from regime_model.data.loaders import ALLOCATION_TICKERS, FEATURE_TICKERS

logger = logging.getLogger(__name__)

# Window sizes (calendar / trading days)
VOL_WINDOW = 20
SKEW_WINDOW = 60
CORR_WINDOW = 60
RET_5D_WINDOW = 5

# Asset pairs used for cross-asset correlation features.
CORR_PAIRS: tuple[tuple[str, str], ...] = (
    ("SPY", "TLT"),
    ("HYG", "SPY"),
)

FEATURE_DIM = len(FEATURE_TICKERS) * 2 + len(CORR_PAIRS)  # 6 + 6 + 2 = 14


@dataclass(frozen=True)
class FeatureBundle:
    raw: pd.DataFrame          # unstandardized, lagged-by-1 features
    standardized: pd.DataFrame # expanding z-score standardized version
    column_groups: dict[str, list[str]]  # category -> list of column names


# --------------------------------------------------------------------------- #
# Per-asset feature blocks
# --------------------------------------------------------------------------- #

def _daily_log_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Log returns shifted by one day so feature[t] = r_{t-1}."""
    out = returns.shift(1).copy()
    out.columns = [f"ret1d_{c}" for c in out.columns]
    return out


def _realized_vol(returns: pd.DataFrame, window: int = VOL_WINDOW) -> pd.DataFrame:
    """Annualized realized vol over a trailing window, shifted by one day.

    The std is computed over the window ending at t-1 (after the .shift(1)),
    so the value at row t uses returns from [t-window, t-1].
    """
    vol = returns.rolling(window=window, min_periods=window).std() * np.sqrt(252)
    vol = vol.shift(1)
    vol.columns = [f"vol{window}_{c}" for c in returns.columns]
    return vol


# --------------------------------------------------------------------------- #
# Cross-asset features
# --------------------------------------------------------------------------- #

def _rolling_correlation(
    returns: pd.DataFrame,
    a: str,
    b: str,
    window: int = CORR_WINDOW,
) -> pd.Series:
    """Pearson correlation of two return series over a trailing window, lagged."""
    corr = returns[a].rolling(window=window, min_periods=window).corr(returns[b])
    corr = corr.shift(1)
    corr.name = f"corr{window}_{a}_{b}"
    return corr


# --------------------------------------------------------------------------- #
# Public builders
# --------------------------------------------------------------------------- #

def build_raw_features(returns: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Build the full 14-dim raw feature DataFrame plus a column-group index.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns indexed by date with one column per FEATURE_TICKERS
        symbol. Caller is responsible for ensuring ordering matches FEATURE_TICKERS.

    Returns
    -------
    (features, groups) where:
        features   : DataFrame with FEATURE_DIM columns, indexed by date.
        groups     : Dict mapping group name ("returns", "vols", "corrs") to
                     the list of column names in that group.
    """
    if list(returns.columns) != list(FEATURE_TICKERS):
        # Reorder defensively. Missing columns are an error.
        missing = set(FEATURE_TICKERS) - set(returns.columns)
        if missing:
            raise ValueError(f"returns is missing required columns: {sorted(missing)}")
        returns = returns[list(FEATURE_TICKERS)]

    blocks: list[pd.DataFrame | pd.Series] = []

    ret_block = _daily_log_returns(returns)
    blocks.append(ret_block)

    vol_block = _realized_vol(returns, window=VOL_WINDOW)
    blocks.append(vol_block)

    corr_cols: list[pd.Series] = []
    for a, b in CORR_PAIRS:
        corr_cols.append(_rolling_correlation(returns, a, b, window=CORR_WINDOW))
    corr_block = pd.concat(corr_cols, axis=1)
    blocks.append(corr_block)

    features = pd.concat(blocks, axis=1)
    assert features.shape[1] == FEATURE_DIM, (
        f"expected {FEATURE_DIM} feature columns, got {features.shape[1]}"
    )

    groups = {
        "returns": list(ret_block.columns),
        "vols": list(vol_block.columns),
        "corrs": list(corr_block.columns),
    }
    return features, groups


def expanding_zscore(
    df: pd.DataFrame,
    min_periods: int = 252,
) -> pd.DataFrame:
    """Standardize each column using an expanding mean and std.

    The mean and std at time t are computed over rows [0, t-1] (i.e., shifted
    by one), so the standardized value at t does not include itself.

    Rows with fewer than `min_periods` of history are returned as NaN; the
    caller drops the warmup window.
    """
    mean = df.expanding(min_periods=min_periods).mean().shift(1)
    std = df.expanding(min_periods=min_periods).std().shift(1)
    z = (df - mean) / std.replace(0.0, np.nan)
    return z


def build_features(
    returns: pd.DataFrame,
    min_zscore_periods: int = 252,
) -> FeatureBundle:
    """End-to-end builder. Returns raw + standardized features."""
    raw, groups = build_raw_features(returns)
    standardized = expanding_zscore(raw, min_periods=min_zscore_periods)
    return FeatureBundle(raw=raw, standardized=standardized, column_groups=groups)


# --------------------------------------------------------------------------- #
# Look-ahead validator
# --------------------------------------------------------------------------- #

def validate_no_lookahead(
    returns: pd.DataFrame,
    builder=build_raw_features,
    probe_dates: int = 8,
    seed: int = 0,
) -> None:
    """Empirically verify that perturbing returns at date t' does not change
    features at any date t <= t'.

    The test picks `probe_dates` random dates from the second half of the
    sample (avoiding the warmup window where features are NaN), perturbs the
    return for one date, recomputes features, and asserts that no earlier
    feature value moved.

    Raises
    ------
    AssertionError if look-ahead is detected.
    """
    rng = np.random.default_rng(seed)
    base, _ = builder(returns)

    valid_dates = base.dropna().index
    if len(valid_dates) < probe_dates * 2:
        raise ValueError(
            f"need at least {probe_dates * 2} valid feature dates to validate, "
            f"got {len(valid_dates)}"
        )

    candidate_dates = valid_dates[len(valid_dates) // 2 :]
    sample_dates = rng.choice(candidate_dates, size=probe_dates, replace=False)

    for probe_date in sample_dates:
        perturbed = returns.copy()
        # Bump one cell by a meaningful amount.
        col = returns.columns[rng.integers(0, returns.shape[1])]
        perturbed.loc[probe_date, col] += 0.1  # +10% log return shock

        new_features, _ = builder(perturbed)
        # The "no look-ahead" rule: feature[t] must not depend on returns[s]
        # for s >= t. So perturbing returns at probe_date must leave feature[t]
        # unchanged for all t <= probe_date (note <=, not <).
        earlier = base.index <= probe_date
        diff = (new_features.loc[earlier] - base.loc[earlier]).abs()
        max_diff = diff.max().max()
        if pd.notna(max_diff) and max_diff > 1e-12:
            offenders = diff.stack().sort_values(ascending=False).head(5)
            raise AssertionError(
                f"look-ahead detected: perturbing {col} on "
                f"{pd.Timestamp(probe_date).date()} changed earlier feature "
                f"values (max diff {max_diff:.3e}). top offenders:\n{offenders}"
            )

    logger.info("look-ahead validator passed on %d probe dates", probe_dates)


def allocation_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Subset of returns restricted to the tradable allocation universe."""
    return returns[list(ALLOCATION_TICKERS)].copy()
