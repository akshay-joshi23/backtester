"""Strategy Lab CLI entrypoint.

Commands:
  lab backtest "<natural-language prompt>"   — generate + run
  lab list                                    — show saved runs
  lab show <run_id>                           — print one run's details
  lab compare <run_id_a> <run_id_b>           — side-by-side metrics
  lab run-reference <strategy>                — run a built-in reference strategy
                                                (no LLM call; useful for sanity)

Environment:
  ANTHROPIC_API_KEY must be set to use `backtest`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from tabulate import tabulate

from lab.backtest import BacktestConfig, walk_forward_backtest
from lab.costs import FlatBpsPerLeg
from lab.data import load_universe
from lab.llm import generate_strategy
from lab.metrics import annual_turnover, compute_metrics
from lab.runner import RUNS_DIR, list_runs, load_run, run_backtest, save_run
from lab.strategies import CrossSectionalMomentum, EqualWeight, FixedMix

logger = logging.getLogger(__name__)


def cmd_backtest(args: argparse.Namespace) -> int:
    prompt = args.prompt
    logger.info("Generating strategy code...")
    generation = generate_strategy(
        prompt,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    print("--- generated strategy ---")
    print(generation.code)
    print("--- end strategy ---\n")

    universe = args.universe or generation.universe
    train_end = args.train_end or generation.train_end
    rebalance_freq = args.rebalance_freq or generation.rebalance_freq
    logger.info(
        "Backtesting on universe=%s, train_end=%s, rebalance_freq=%d",
        universe, train_end, rebalance_freq,
    )
    equity, weights, metrics, cfg, strategy_name = run_backtest(
        generation.code,
        universe=universe,
        train_end=train_end,
        rebalance_freq=rebalance_freq,
        start=args.start,
        end=args.end,
        cost_bps=args.cost_bps,
        timeout_seconds=args.timeout,
    )
    artifacts = save_run(
        prompt=prompt,
        generation=generation,
        equity=equity,
        weights=weights,
        metrics=metrics,
        cfg=cfg,
        strategy_name=strategy_name,
        universe=universe,
    )
    print(f"\nRun saved: {artifacts.run_dir}")
    _print_metrics_table([(artifacts.run_id, strategy_name, metrics)])
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    runs = list_runs()
    if not runs:
        print("no runs yet — try `lab backtest \"...\"`")
        return 0
    rows = [(
        r["run_id"],
        r["strategy_name"],
        ",".join(r["universe"]),
        f"{r['sharpe']:.3f}" if r["sharpe"] is not None else "—",
        f"{r['cagr']*100:.2f}%" if r["cagr"] is not None else "—",
        f"{r['max_drawdown']*100:.2f}%" if r["max_drawdown"] is not None else "—",
    ) for r in runs]
    print(tabulate(
        rows, headers=["run_id", "strategy", "universe", "Sharpe", "CAGR", "MaxDD"],
    ))
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    run = load_run(args.run_id)
    if args.open:
        from lab.report import render_run_report
        html_path = render_run_report(run)
        import webbrowser
        webbrowser.open(f"file://{html_path}")
        print(f"Opened report: {html_path}")
        return 0
    print(f"# Run: {run['run_id']}")
    print(f"# Strategy: {run['config'].get('strategy_name', '?')}")
    print(f"# Universe: {run['config']['universe']}")
    print(f"# Train end: {run['config']['train_end']}, rebalance every {run['config']['rebalance_freq']} days")
    print()
    print("## Prompt")
    print(textwrap.indent(run["prompt"], "  "))
    print()
    print("## Metrics")
    _print_metrics_table([(run["run_id"], run["config"].get("strategy_name", "?"), run["metrics"])])
    print()
    print("## Strategy code")
    print(textwrap.indent(run["strategy_code"], "  "))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    runs = [load_run(rid) for rid in args.run_ids]
    print("## Side-by-side metrics\n")
    rows = []
    keys = [
        "sharpe", "sortino", "cagr", "ann_vol", "max_drawdown", "calmar",
        "final_nav", "annual_turnover", "total_transaction_cost", "n_obs",
    ]
    headers = ["metric"] + [f"{r['run_id']}\n{r['config'].get('strategy_name','?')}" for r in runs]
    for k in keys:
        row = [k]
        for r in runs:
            v = r["metrics"].get(k)
            if v is None:
                row.append("—")
            elif k in ("cagr", "ann_vol", "max_drawdown"):
                row.append(f"{v*100:.2f}%")
            elif k in ("n_obs", "annual_turnover"):
                row.append(f"{v:.2f}" if isinstance(v, float) else str(v))
            else:
                row.append(f"{v:.4f}")
        rows.append(row)
    print(tabulate(rows, headers=headers))
    return 0


REFERENCE_STRATEGIES = {
    "60_40": lambda: FixedMix({"SPY": 0.6, "TLT": 0.4}, name="60/40 SPY-TLT"),
    "equal_weight": lambda: EqualWeight(),
    "momentum_top2_6m": lambda: CrossSectionalMomentum(lookback=126, top_k=2),
}


def cmd_run_reference(args: argparse.Namespace) -> int:
    factory = REFERENCE_STRATEGIES.get(args.strategy)
    if not factory:
        print(f"unknown reference: {args.strategy}. options: {list(REFERENCE_STRATEGIES)}")
        return 2
    strategy = factory()
    universe = args.universe or ["SPY", "TLT", "QQQ", "GLD"]
    bundle = load_universe(universe, start=args.start, end=args.end)
    cfg = BacktestConfig(train_end=args.train_end, rebalance_freq=args.rebalance_freq)
    res = walk_forward_backtest(
        strategy, bundle.returns[universe], cfg=cfg,
        cost_model=FlatBpsPerLeg(args.cost_bps),
    )
    metrics = compute_metrics(res.equity)
    metrics["annual_turnover"] = annual_turnover(res.turnover)
    metrics["total_transaction_cost"] = float(res.transaction_costs.sum())

    # Mimic save_run for parity with `lab list`.
    from lab.llm import GenerationResult
    fake_generation = GenerationResult(
        code=f"# Reference strategy: {args.strategy}\n",
        raw_response="reference strategy — not LLM-generated\n",
        universe=universe,
        train_end=args.train_end,
        rebalance_freq=args.rebalance_freq,
        usage=None,
    )
    artifacts = save_run(
        prompt=f"[reference] {args.strategy}",
        generation=fake_generation,
        equity=res.equity,
        weights=res.realized_weights,
        metrics=metrics,
        cfg=cfg,
        strategy_name=strategy.name,
        universe=universe,
    )
    print(f"\nRun saved: {artifacts.run_dir}")
    _print_metrics_table([(artifacts.run_id, strategy.name, metrics)])
    return 0


def _print_metrics_table(rows: list[tuple[str, str, dict]]):
    keys = ["sharpe", "sortino", "cagr", "ann_vol", "max_drawdown", "calmar", "final_nav"]
    pct_keys = {"cagr", "ann_vol", "max_drawdown"}
    headers = ["run_id", "strategy"] + keys
    out = []
    for run_id, name, m in rows:
        row = [run_id, name]
        for k in keys:
            v = m.get(k)
            if v is None:
                row.append("—")
            elif k in pct_keys:
                row.append(f"{v*100:.2f}%")
            else:
                row.append(f"{v:.4f}")
        out.append(row)
    print(tabulate(out, headers=headers))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    p = argparse.ArgumentParser(prog="lab", description="Strategy Lab CLI")
    sub = p.add_subparsers(dest="command", required=True)

    p_bt = sub.add_parser("backtest", help="generate a strategy from natural language and backtest it")
    p_bt.add_argument("prompt", help="natural-language strategy description")
    p_bt.add_argument("--universe", nargs="+", help="override ticker universe")
    p_bt.add_argument("--train-end", default=None, help="OOS start date, e.g. 2015-01-01")
    p_bt.add_argument("--rebalance-freq", type=int, default=None, help="trading days between rebalances")
    p_bt.add_argument("--start", default="2010-01-01", help="data fetch start")
    p_bt.add_argument("--end", default=None, help="data fetch end (exclusive)")
    p_bt.add_argument("--cost-bps", type=float, default=5.0)
    p_bt.add_argument("--timeout", type=float, default=120.0,
                      help="seconds; aborts runaway backtests. 0 to disable.")
    p_bt.add_argument("--model", default="claude-opus-4-7")
    p_bt.add_argument("--temperature", type=float, default=0.2)
    p_bt.add_argument("--max-tokens", type=int, default=4096)
    p_bt.set_defaults(func=cmd_backtest)

    p_ls = sub.add_parser("list", help="list saved runs")
    p_ls.set_defaults(func=cmd_list)

    p_sh = sub.add_parser("show", help="show one run's details")
    p_sh.add_argument("run_id")
    p_sh.add_argument("--open", action="store_true",
                      help="render HTML report and open in browser")
    p_sh.set_defaults(func=cmd_show)

    p_cmp = sub.add_parser("compare", help="side-by-side compare two or more runs")
    p_cmp.add_argument("run_ids", nargs="+")
    p_cmp.set_defaults(func=cmd_compare)

    p_ref = sub.add_parser("run-reference", help="run a built-in reference strategy (no LLM)")
    p_ref.add_argument("strategy", choices=sorted(REFERENCE_STRATEGIES))
    p_ref.add_argument("--universe", nargs="+", default=None)
    p_ref.add_argument("--start", default="2010-01-01")
    p_ref.add_argument("--end", default=None)
    p_ref.add_argument("--train-end", default="2015-01-01")
    p_ref.add_argument("--rebalance-freq", type=int, default=21)
    p_ref.add_argument("--cost-bps", type=float, default=5.0)
    p_ref.set_defaults(func=cmd_run_reference)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
