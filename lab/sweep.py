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
