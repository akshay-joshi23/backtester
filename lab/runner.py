"""Execute a generated Strategy and run a backtest against it.

This module connects the LLM-generated code to the backtest engine:

  1. Take a code string from `lab.llm.generate_strategy`
  2. Exec it in a fresh namespace, find the Strategy subclass
  3. Instantiate it with no args (the generator is responsible for sensible defaults)
  4. Pull data for the user-requested universe via `lab.data`
  5. Run `walk_forward_backtest`
  6. Compute metrics, package into a run directory

A "run" is a timestamped directory under `runs/` containing:
  - prompt.txt        (the user's natural-language description)
  - strategy.py       (the executed code)
  - config.json       (universe, train_end, rebalance_freq, cost model)
  - equity.csv        (daily NAV)
  - weights.csv       (daily realized weights)
  - metrics.json      (Sharpe etc.)
  - llm_usage.json    (token usage, if available)
"""

from __future__ import annotations

import json
import logging
import textwrap
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from lab.backtest import BacktestConfig, walk_forward_backtest
from lab.costs import FlatBpsPerLeg
from lab.data import UniverseBundle, load_universe
from lab.llm import GenerationResult
from lab.metrics import annual_turnover, compute_metrics
from lab.strategy import Strategy

logger = logging.getLogger(__name__)

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


@dataclass
class RunArtifacts:
    run_id: str
    run_dir: Path
    metrics: dict[str, float]
    strategy_name: str
    universe: list[str]
    config: BacktestConfig


def execute_strategy_code(code: str) -> type[Strategy]:
    """Exec `code` in a fresh namespace and return the unique Strategy subclass.

    Raises RuntimeError if zero or multiple Strategy subclasses are defined.
    """
    namespace: dict = {}
    try:
        exec(compile(code, "<generated_strategy>", "exec"), namespace)
    except Exception as e:
        raise RuntimeError(
            f"generated strategy code failed to execute:\n"
            f"  {type(e).__name__}: {e}\n\n"
            f"--- code ---\n{textwrap.indent(code, '  ')}"
        ) from e

    classes = [
        v for v in namespace.values()
        if isinstance(v, type)
        and issubclass(v, Strategy)
        and v is not Strategy
    ]
    if not classes:
        raise RuntimeError(
            "generated code defines no Strategy subclass\n\n"
            f"--- code ---\n{textwrap.indent(code, '  ')}"
        )
    if len(classes) > 1:
        names = [c.__name__ for c in classes]
        raise RuntimeError(
            f"generated code defines {len(classes)} Strategy subclasses: {names}. "
            "Generate exactly one."
        )
    return classes[0]


def run_backtest(
    code: str,
    *,
    universe: list[str],
    train_end: str,
    rebalance_freq: int,
    start: str = "2010-01-01",
    end: str | None = None,
    cost_bps: float = 5.0,
    initial_wealth: float = 1.0,
    bundle: UniverseBundle | None = None,
) -> tuple[pd.Series, pd.DataFrame, dict, BacktestConfig, str]:
    """Execute strategy code and run the backtest. Returns the core artifacts.

    Returns
    -------
    (equity, realized_weights, metrics, config, strategy_name)
    """
    cls = execute_strategy_code(code)
    strategy = cls()  # type: ignore[call-arg]
    if not isinstance(strategy, Strategy):
        raise RuntimeError("instantiated object is not a Strategy")
    if bundle is None:
        bundle = load_universe(universe, start=start, end=end)
    returns = bundle.returns
    missing = [t for t in universe if t not in returns.columns]
    if missing:
        raise RuntimeError(f"universe tickers missing from loaded data: {missing}")
    returns = returns[universe]
    cfg = BacktestConfig(
        train_end=train_end,
        rebalance_freq=rebalance_freq,
        initial_wealth=initial_wealth,
    )
    result = walk_forward_backtest(
        strategy, returns, cfg=cfg, cost_model=FlatBpsPerLeg(bps=cost_bps),
    )
    metrics = compute_metrics(result.equity)
    metrics["annual_turnover"] = annual_turnover(result.turnover)
    metrics["total_transaction_cost"] = float(result.transaction_costs.sum())
    return result.equity, result.realized_weights, metrics, cfg, strategy.name


def save_run(
    *,
    prompt: str,
    generation: GenerationResult,
    equity: pd.Series,
    weights: pd.DataFrame,
    metrics: dict,
    cfg: BacktestConfig,
    strategy_name: str,
    universe: list[str],
    runs_dir: Path | None = None,
) -> RunArtifacts:
    """Persist all artifacts of one run to disk, return RunArtifacts."""
    runs_dir = runs_dir or RUNS_DIR
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    # Disambiguate within the same second.
    candidate = runs_dir / run_id
    suffix = 1
    while candidate.exists():
        candidate = runs_dir / f"{run_id}-{suffix}"
        suffix += 1
    run_dir = candidate
    run_dir.mkdir()

    (run_dir / "prompt.txt").write_text(prompt + "\n")
    (run_dir / "strategy.py").write_text(generation.code + "\n")
    (run_dir / "config.json").write_text(json.dumps({
        "universe": universe,
        "train_end": cfg.train_end,
        "rebalance_freq": cfg.rebalance_freq,
        "initial_wealth": cfg.initial_wealth,
        "long_only": cfg.long_only,
        "max_leverage": cfg.max_leverage,
        "strategy_name": strategy_name,
    }, indent=2))
    equity.to_csv(run_dir / "equity.csv", header=["nav"])
    weights.to_csv(run_dir / "weights.csv")
    metrics_clean = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                     for k, v in metrics.items()}
    (run_dir / "metrics.json").write_text(json.dumps(metrics_clean, indent=2))
    if generation.usage:
        (run_dir / "llm_usage.json").write_text(json.dumps(generation.usage, indent=2))
    (run_dir / "raw_response.txt").write_text(generation.raw_response)

    # Auto-render the HTML report. Best-effort; don't fail the run if matplotlib
    # blows up (e.g. headless backend issue).
    try:
        from lab.report import render_run_report
        render_run_report({
            "run_id": run_dir.name,
            "run_dir": run_dir,
            "config": {
                "universe": universe,
                "train_end": cfg.train_end,
                "rebalance_freq": cfg.rebalance_freq,
                "strategy_name": strategy_name,
            },
            "metrics": metrics_clean,
            "equity": equity,
            "weights": weights,
            "prompt": prompt,
            "strategy_code": generation.code,
        })
    except Exception as e:
        logger.warning("HTML report failed: %s", e)

    return RunArtifacts(
        run_id=run_dir.name,
        run_dir=run_dir,
        metrics=metrics_clean,
        strategy_name=strategy_name,
        universe=universe,
        config=cfg,
    )


def load_run(run_id: str, runs_dir: Path | None = None) -> dict:
    """Load a saved run's artifacts as a dict. Used by `lab show` and `lab compare`."""
    runs_dir = runs_dir or RUNS_DIR
    run_dir = runs_dir / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"no such run: {run_dir}")
    config = json.loads((run_dir / "config.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    equity = pd.read_csv(run_dir / "equity.csv", index_col=0, parse_dates=True).iloc[:, 0]
    prompt = (run_dir / "prompt.txt").read_text().strip()
    strategy_code = (run_dir / "strategy.py").read_text()
    out = {
        "run_id": run_id,
        "run_dir": run_dir,
        "config": config,
        "metrics": metrics,
        "equity": equity,
        "prompt": prompt,
        "strategy_code": strategy_code,
    }
    weights_path = run_dir / "weights.csv"
    if weights_path.exists():
        out["weights"] = pd.read_csv(weights_path, index_col=0, parse_dates=True)
    return out


def list_runs(runs_dir: Path | None = None) -> list[dict]:
    """List all saved runs with minimal summary info."""
    runs_dir = runs_dir or RUNS_DIR
    if not runs_dir.exists():
        return []
    out = []
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        cfg_path = d / "config.json"
        met_path = d / "metrics.json"
        if not (cfg_path.exists() and met_path.exists()):
            continue
        try:
            cfg = json.loads(cfg_path.read_text())
            met = json.loads(met_path.read_text())
        except Exception:
            continue
        out.append({
            "run_id": d.name,
            "strategy_name": cfg.get("strategy_name", "?"),
            "universe": cfg.get("universe", []),
            "sharpe": met.get("sharpe"),
            "cagr": met.get("cagr"),
            "max_drawdown": met.get("max_drawdown"),
        })
    return out
