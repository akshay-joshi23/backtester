"""Hyperparameter sweep for a fixed strategy code template.

Takes a Python file that defines a Strategy subclass with a single
hyperparameter (e.g. `lookback`, `target_k`), runs the same backtest with
multiple values, and tabulates results so you can compare.

SIMPLIFICATION VERSUS THE ORIGINAL PROPOSAL:
  - This is a *grid* sweep, not walk-forward parameter selection.
    Walk-forward HP selection would tune the parameter on a rolling
    in-sample window and apply it forward — much more work to wire in.
  - The parameter must be a class-init kwarg, and the strategy class must
    accept it via constructor. (No magic source-rewriting.)
  - The sweep returns the full per-value metrics table — you decide
    which is best. No automatic selection / multiple-testing correction.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from lab.backtest import BacktestConfig, walk_forward_backtest
from lab.costs import FlatBpsPerLeg
from lab.data import load_universe
from lab.metrics import compute_metrics
from lab.runner import execute_strategy_code

logger = logging.getLogger(__name__)


def sweep_hyperparameter(
    code: str,
    *,
    param_name: str,
    param_values: list,
    universe: list[str],
    start: str = "2010-01-01",
    end: str | None = None,
    train_end: str = "2015-01-01",
    rebalance_freq: int = 21,
    cost_bps: float = 5.0,
    long_only: bool = True,
    max_leverage: float = 1.0,
) -> pd.DataFrame:
    """Run the same strategy code with different values for one hyperparameter.

    The strategy class is constructed as `cls(**{param_name: value})`. All
    other parameters use the class defaults. Returns a DataFrame indexed by
    parameter value, columns = metrics.

    `param_values` is a flat list of values to try (ints, floats, strings —
    whatever the constructor accepts).
    """
    cls = execute_strategy_code(code)
    bundle = load_universe(universe, start=start, end=end)
    returns = bundle.returns[universe]

    rows: list[dict] = []
    for value in param_values:
        logger.info("sweep: %s=%s", param_name, value)
        try:
            strategy = cls(**{param_name: value})
        except TypeError as e:
            raise RuntimeError(
                f"strategy class {cls.__name__} does not accept "
                f"keyword '{param_name}': {e}"
            ) from e
        cfg = BacktestConfig(
            train_end=train_end, rebalance_freq=rebalance_freq,
            long_only=long_only, max_leverage=max_leverage,
        )
        try:
            res = walk_forward_backtest(
                strategy, returns, cfg=cfg, cost_model=FlatBpsPerLeg(cost_bps),
            )
            m = compute_metrics(res.equity)
        except Exception as e:
            logger.warning("sweep value %s=%s failed: %s", param_name, value, e)
            rows.append({param_name: value, "error": str(e)})
            continue
        rows.append({
            param_name: value,
            "sharpe": m.get("sharpe"),
            "cagr": m.get("cagr"),
            "ann_vol": m.get("ann_vol"),
            "max_drawdown": m.get("max_drawdown"),
            "calmar": m.get("calmar"),
            "final_nav": m.get("final_nav"),
            "n_obs": m.get("n_obs"),
        })
    df = pd.DataFrame(rows)
    return df.set_index(param_name) if param_name in df.columns else df


def walk_forward_sweep(
    code: str,
    *,
    param_name: str,
    param_values: list,
    universe: list[str],
    start: str = "2010-01-01",
    end: str | None = None,
    tuning_window_years: float = 3.0,
    retune_freq_years: float = 1.0,
    initial_train_years: float = 3.0,
    rebalance_freq: int = 21,
    cost_bps: float = 5.0,
    long_only: bool = True,
    max_leverage: float = 1.0,
    objective: str = "sharpe",
) -> dict:
    """Real walk-forward hyperparameter selection.

    For each retune window:
      1. Take the trailing `tuning_window_years` of data as the in-sample tune.
      2. For each value in `param_values`, run an in-sample backtest, score
         it by `objective` (sharpe / cagr / sortino / calmar).
      3. The winning value is applied forward for `retune_freq_years` of OOS.
      4. Slide the window and repeat.

    Returns
    -------
    dict with keys:
      'per_window': pd.DataFrame indexed by retune-window start with columns
                    [winning_value, in_sample_sharpe, oos_sharpe, ...]
      'aggregated': dict of aggregated OOS metrics across the full walk-forward
      'oos_equity': pd.Series — stitched OOS equity curve
    """
    cls = execute_strategy_code(code)
    bundle = load_universe(universe, start=start, end=end)
    returns = bundle.returns[universe]

    # Build the schedule of (tune_start, tune_end, oos_start, oos_end) windows.
    schedule = _build_walk_forward_schedule(
        returns.index,
        initial_train_years=initial_train_years,
        tuning_window_years=tuning_window_years,
        retune_freq_years=retune_freq_years,
    )
    if not schedule:
        raise ValueError("not enough data for the requested walk-forward windows")

    per_window_rows = []
    stitched_equity = pd.Series(dtype=float)
    running_nav = 1.0

    for i, (tune_start, tune_end, oos_start, oos_end) in enumerate(schedule):
        # 1. In-sample sweep over param_values.
        is_returns = returns.loc[tune_start:tune_end]
        if len(is_returns) < 252:
            continue
        # Use the second-half of is_returns as the in-sample "OOS" for tuning
        # so we don't fit-and-evaluate on the same period.
        tune_split = is_returns.index[len(is_returns) // 2]

        best_score = -np.inf
        best_value = param_values[0]
        is_scores: dict = {}
        for value in param_values:
            try:
                strat = cls(**{param_name: value})
            except TypeError as e:
                raise RuntimeError(
                    f"strategy class {cls.__name__} does not accept '{param_name}'"
                ) from e
            cfg = BacktestConfig(
                train_end=tune_split.strftime("%Y-%m-%d"),
                rebalance_freq=rebalance_freq,
                long_only=long_only, max_leverage=max_leverage,
            )
            try:
                res = walk_forward_backtest(
                    strat, is_returns, cfg=cfg, cost_model=FlatBpsPerLeg(cost_bps),
                )
                m = compute_metrics(res.equity)
            except Exception as e:
                logger.warning("tune %s=%s window %s: %s", param_name, value, i, e)
                is_scores[value] = float("-inf")
                continue
            score = m.get(objective, float("-inf"))
            is_scores[value] = score
            if score is not None and score > best_score:
                best_score = score
                best_value = value

        # 2. OOS evaluation with the winning value.
        oos_full = returns.loc[(returns.index >= tune_start) & (returns.index <= oos_end)]
        if len(oos_full) < 252:
            continue
        strat_oos = cls(**{param_name: best_value})
        cfg_oos = BacktestConfig(
            train_end=oos_start.strftime("%Y-%m-%d"),
            rebalance_freq=rebalance_freq,
            initial_wealth=running_nav,
            long_only=long_only, max_leverage=max_leverage,
        )
        try:
            res_oos = walk_forward_backtest(
                strat_oos, oos_full, cfg=cfg_oos,
                cost_model=FlatBpsPerLeg(cost_bps),
            )
            m_oos = compute_metrics(res_oos.equity)
        except Exception as e:
            logger.warning("OOS window %s with %s=%s failed: %s",
                           i, param_name, best_value, e)
            continue
        # Stitch: append OOS equity to the running curve.
        stitched_equity = pd.concat([stitched_equity, res_oos.equity])
        running_nav = float(res_oos.equity.iloc[-1])

        per_window_rows.append({
            "window": i,
            "tune_start": tune_start.date(),
            "tune_end": tune_end.date(),
            "oos_start": oos_start.date(),
            "oos_end": oos_end.date(),
            f"winning_{param_name}": best_value,
            "in_sample_score": best_score,
            "oos_sharpe": m_oos.get("sharpe"),
            "oos_cagr": m_oos.get("cagr"),
            "oos_max_dd": m_oos.get("max_drawdown"),
            "oos_n_obs": m_oos.get("n_obs"),
        })

    per_window = pd.DataFrame(per_window_rows).set_index("window")
    aggregated = compute_metrics(stitched_equity) if not stitched_equity.empty else {}
    return {
        "per_window": per_window,
        "aggregated": aggregated,
        "oos_equity": stitched_equity,
    }


def _build_walk_forward_schedule(
    dates: pd.DatetimeIndex,
    initial_train_years: float,
    tuning_window_years: float,
    retune_freq_years: float,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Returns [(tune_start, tune_end, oos_start, oos_end), ...]."""
    if len(dates) < 252:
        return []
    first = dates.min()
    last = dates.max()
    initial_oos_start = first + pd.DateOffset(days=int(initial_train_years * 365.25))
    retune_step = pd.DateOffset(days=int(retune_freq_years * 365.25))
    tune_offset = pd.DateOffset(days=int(tuning_window_years * 365.25))

    schedule = []
    oos_start = initial_oos_start
    while oos_start < last:
        tune_end = oos_start  # exclusive — tuning data is strictly before OOS
        tune_start = max(first, tune_end - tune_offset)
        oos_end = min(last, oos_start + retune_step)
        schedule.append((tune_start, tune_end, oos_start, oos_end))
        oos_start = oos_end
    return schedule


def parse_value_list(s: str) -> list:
    """CLI helper: parse '60,120,250' or '0.05,0.1,0.2' into a typed list."""
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError:
            try:
                out.append(float(p))
            except ValueError:
                out.append(p)
    return out
