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
from lab.llm import build_universe_brief, generate_strategy, refine_strategy
from lab.metrics import annual_turnover, block_bootstrap_metrics, compute_metrics
from lab.runner import (
    RUNS_DIR, format_run_tree, list_runs, load_run, run_backtest, save_run,
)
from lab.strategies import (
    BayesianRegime, CrossSectionalMomentum, EqualWeight, FixedMix,
    LongShortMomentum,
)

logger = logging.getLogger(__name__)


def cmd_backtest(args: argparse.Namespace) -> int:
    prompt = args.prompt
    parent_run_id = getattr(args, "parent", None)
    if parent_run_id:
        # Compose: prepend parent's prompt + previous code as context.
        parent = load_run(parent_run_id)
        full_prompt = (
            f"Previously you wrote this strategy (run {parent_run_id}, "
            f"named '{parent['config'].get('strategy_name','?')}'):\n\n"
            "```python\n" + parent["strategy_code"] + "```\n\n"
            f"Now revise it for the following follow-up:\n\n{prompt}"
        )
    else:
        full_prompt = prompt
    universe_brief = None
    if args.data_aware:
        # Snap to whatever universe is going to be used.
        sniff_universe = args.universe or ["SPY", "TLT"]
        universe_brief = build_universe_brief(sniff_universe, sample_days=30)
    if args.agent:
        from lab.agent import run_agent
        logger.info("Running agent loop (max 5 iterations)...")
        generation = run_agent(
            (full_prompt + ("\n\n" + universe_brief if universe_brief else "")),
            model=args.model, max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    else:
        logger.info("Generating strategy code...")
        generation = generate_strategy(
            full_prompt,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            universe_brief=universe_brief,
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
        long_only=not args.long_short,
        max_leverage=args.max_leverage,
        sandbox=args.sandbox,
        frequency=args.frequency,
        realistic_costs=args.realistic_costs,
    )

    if args.refine:
        # A-1 simplified: single self-critique round w/ metrics feedback.
        logger.info("Running refinement pass...")
        refined = refine_strategy(
            generation.code, metrics, full_prompt,
            model=args.model, max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        if refined.code.strip() != generation.code.strip():
            logger.info("Refinement changed the code — re-running backtest")
            print("--- refined strategy ---")
            print(refined.code)
            print("--- end refined strategy ---\n")
            generation = refined
            equity, weights, metrics, cfg, strategy_name = run_backtest(
                generation.code,
                universe=universe,
                train_end=train_end,
                rebalance_freq=rebalance_freq,
                start=args.start,
                end=args.end,
                cost_bps=args.cost_bps,
                timeout_seconds=args.timeout,
                long_only=not args.long_short,
                max_leverage=args.max_leverage,
                sandbox=args.sandbox,
            )
        else:
            logger.info("Refinement pass returned the same code — no rerun")
    artifacts = save_run(
        prompt=prompt,
        generation=generation,
        equity=equity,
        weights=weights,
        metrics=metrics,
        cfg=cfg,
        strategy_name=strategy_name,
        universe=universe,
        parent_run_id=parent_run_id,
    )
    print(f"\nRun saved: {artifacts.run_dir}")
    if parent_run_id:
        print(f"  Forked from: {parent_run_id}")
    _print_metrics_table([(artifacts.run_id, strategy_name, metrics)])
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    from lab.chat import run_chat
    return run_chat()


def cmd_sweep(args: argparse.Namespace) -> int:
    """Sweep one hyperparameter across a list of values for a saved run's code."""
    from lab.sweep import parse_value_list, sweep_hyperparameter, walk_forward_sweep
    run = load_run(args.run_id)
    values = parse_value_list(args.values)
    if args.walk_forward:
        result = walk_forward_sweep(
            run["strategy_code"],
            param_name=args.param,
            param_values=values,
            universe=args.universe or run["config"]["universe"],
            start=args.start,
            end=args.end,
            tuning_window_years=args.tuning_years,
            retune_freq_years=args.retune_years,
            initial_train_years=args.initial_train_years,
            rebalance_freq=args.rebalance_freq or run["config"]["rebalance_freq"],
            cost_bps=args.cost_bps,
            long_only=run["config"].get("long_only", True),
            max_leverage=run["config"].get("max_leverage", 1.0),
            objective=args.objective,
        )
        print(f"## Walk-forward HP selection: {args.param} ∈ {values}\n")
        print(f"Tuning window: {args.tuning_years}y, retune every {args.retune_years}y, "
              f"objective: {args.objective}\n")
        print("### Per-window winners")
        if not result["per_window"].empty:
            print(tabulate(result["per_window"], headers="keys", tablefmt="simple"))
        print(f"\n### Aggregated OOS metrics")
        agg = result["aggregated"]
        for k in ("sharpe", "cagr", "max_drawdown", "calmar", "final_nav"):
            v = agg.get(k)
            if v is None:
                continue
            if k in ("cagr", "max_drawdown"):
                print(f"  {k}: {v*100:.2f}%")
            else:
                print(f"  {k}: {v:.4f}")
        if args.output:
            out_path = Path(args.output)
            result["per_window"].to_csv(out_path)
            print(f"\nWalk-forward results saved: {out_path}")
        return 0
    df = sweep_hyperparameter(
        run["strategy_code"],
        param_name=args.param,
        param_values=values,
        universe=args.universe or run["config"]["universe"],
        start=args.start,
        end=args.end,
        train_end=args.train_end or run["config"]["train_end"],
        rebalance_freq=args.rebalance_freq or run["config"]["rebalance_freq"],
        cost_bps=args.cost_bps,
        long_only=run["config"].get("long_only", True),
        max_leverage=run["config"].get("max_leverage", 1.0),
    )
    print(f"## Sweep: {args.param} ∈ {values}\n")
    # Format columns nicely.
    display = df.copy()
    for col in ("cagr", "ann_vol", "max_drawdown"):
        if col in display.columns:
            display[col] = display[col].map(
                lambda v: f"{v*100:.2f}%" if pd.notna(v) else "—"
            )
    for col in ("sharpe", "calmar", "final_nav"):
        if col in display.columns:
            display[col] = display[col].map(
                lambda v: f"{v:.3f}" if pd.notna(v) else "—"
            )
    print(tabulate(display.reset_index(), headers="keys", tablefmt="simple",
                   showindex=False))
    # Optional save.
    if args.output:
        out_path = Path(args.output)
        df.to_csv(out_path)
        print(f"\nSweep results saved: {out_path}")
    return 0


def cmd_spa(args: argparse.Namespace) -> int:
    """Run Hansen's SPA on a benchmark vs N alternatives (each by run_id)."""
    from lab.spa import spa_test
    bm = load_run(args.benchmark)
    alts = {rid: load_run(rid) for rid in args.alternatives}
    bm_rets = bm["equity"].pct_change().dropna()
    alt_rets = {
        f"{rid} ({a['config'].get('strategy_name','?')})":
            a["equity"].pct_change().dropna()
        for rid, a in alts.items()
    }
    result = spa_test(
        bm_rets, alt_rets,
        n_resamples=args.n_resamples,
        block_length=args.block_length,
        seed=args.seed,
    )
    print(f"## SPA test: benchmark = {args.benchmark} "
          f"({bm['config'].get('strategy_name', '?')})\n")
    print(f"  T_SPA = {result.t_stat:.3f}")
    print(f"  p-value = {result.p_value:.4f}  "
          f"(prob no alt is genuinely better, after multiple-testing adjustment)")
    print(f"  block_length = {result.block_length}, "
          f"n_resamples = {result.n_resamples}\n")
    print("### Per-alternative diagnostics\n")
    display = result.per_alt.copy()
    display["mean_excess_per_period"] = display["mean_excess_per_period"].map(
        lambda v: f"{v*100:.4f}%"
    )
    display["hac_std"] = display["hac_std"].map(lambda v: f"{v:.5f}")
    display["t_score"] = display["t_score"].map(lambda v: f"{v:.3f}")
    print(tabulate(display, headers="keys", tablefmt="simple"))
    print(f"\n### Interpretation")
    if result.p_value < 0.05:
        print("  p < 0.05 → strong evidence at least one alternative genuinely "
              "beats the benchmark, not just by chance.")
    elif result.p_value < 0.10:
        print("  p < 0.10 → weak/marginal evidence; results sensitive to specs.")
    else:
        print("  p ≥ 0.10 → no statistically significant alternative; what looks "
              "best may be just lucky on this sample.")
    return 0


def cmd_tree(args: argparse.Namespace) -> int:
    print(format_run_tree())
    return 0


def cmd_fork(args: argparse.Namespace) -> int:
    """Shortcut for backtest with --parent."""
    args.parent = args.run_id
    args.prompt = args.followup
    return cmd_backtest(args)


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
    if args.bootstrap:
        boot = block_bootstrap_metrics(
            run["equity"], block_size=args.block_size,
            n_resamples=args.n_resamples, seed=0,
        )
        print(f"# Bootstrap CIs (block_size={args.block_size}, "
              f"n_resamples={args.n_resamples}):")
        rows = []
        for key, stats in boot.items():
            label = key.replace("_", " ").title()
            if key in ("cagr", "max_drawdown"):
                rows.append([
                    label,
                    f"{stats['mean']*100:.2f}%",
                    f"±{stats['std']*100:.2f}%",
                    f"[{stats['ci_low_95']*100:.2f}%, {stats['ci_high_95']*100:.2f}%]",
                ])
            else:
                rows.append([
                    label,
                    f"{stats['mean']:.3f}",
                    f"±{stats['std']:.3f}",
                    f"[{stats['ci_low_95']:.3f}, {stats['ci_high_95']:.3f}]",
                ])
        print(tabulate(rows, headers=["metric", "mean", "std", "95% CI"]))
        return 0
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

    # B-7: Code diff between consecutive runs.
    if args.diff and len(runs) >= 2:
        import difflib
        print("\n## Strategy code diff\n")
        for i in range(len(runs) - 1):
            a, b = runs[i], runs[i + 1]
            diff = difflib.unified_diff(
                a["strategy_code"].splitlines(keepends=True),
                b["strategy_code"].splitlines(keepends=True),
                fromfile=f"{a['run_id']} ({a['config'].get('strategy_name','?')})",
                tofile=f"{b['run_id']} ({b['config'].get('strategy_name','?')})",
                lineterm="",
            )
            d = "".join(diff)
            if d.strip():
                print(d)
            else:
                print(f"  (no code difference between {a['run_id']} and {b['run_id']})")
            print()

    # D-13: Overlaid equity-curve plot.
    if args.plot:
        from lab.report import render_comparison_plot
        out = Path("runs") / f"compare_{'_'.join(args.run_ids)}.png"
        plot_path = render_comparison_plot(runs, out)
        print(f"\nComparison plot saved: {plot_path}")
        if args.open:
            import webbrowser
            webbrowser.open(f"file://{plot_path.resolve()}")
    return 0


REFERENCE_STRATEGIES = {
    "60_40": lambda: FixedMix({"SPY": 0.6, "TLT": 0.4}, name="60/40 SPY-TLT"),
    "equal_weight": lambda: EqualWeight(),
    "momentum_top2_6m": lambda: CrossSectionalMomentum(lookback=126, top_k=2),
    "long_short_momentum": lambda: LongShortMomentum(lookback=126, top_k=2),
    "bayesian_regime": lambda: BayesianRegime(K=3, vi_steps=3000),
}

# Strategies that require long_only=False.
LONG_SHORT_REFERENCES = {"long_short_momentum"}


def cmd_run_reference(args: argparse.Namespace) -> int:
    factory = REFERENCE_STRATEGIES.get(args.strategy)
    if not factory:
        print(f"unknown reference: {args.strategy}. options: {list(REFERENCE_STRATEGIES)}")
        return 2
    strategy = factory()
    universe = args.universe or ["SPY", "TLT", "QQQ", "GLD"]
    bundle = load_universe(universe, start=args.start, end=args.end)
    long_only = args.strategy not in LONG_SHORT_REFERENCES
    cfg = BacktestConfig(
        train_end=args.train_end, rebalance_freq=args.rebalance_freq,
        long_only=long_only, max_leverage=args.max_leverage,
    )
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
    p_bt.add_argument("--parent", default=None,
                      help="fork from this run_id; prompt becomes a follow-up")
    p_bt.add_argument("--universe", nargs="+", help="override ticker universe")
    p_bt.add_argument("--train-end", default=None, help="OOS start date, e.g. 2015-01-01")
    p_bt.add_argument("--rebalance-freq", type=int, default=None, help="trading days between rebalances")
    p_bt.add_argument("--start", default="2010-01-01", help="data fetch start")
    p_bt.add_argument("--end", default=None, help="data fetch end (exclusive)")
    p_bt.add_argument("--cost-bps", type=float, default=5.0)
    p_bt.add_argument("--long-short", action="store_true",
                      help="allow negative weights (shorts). Default long-only.")
    p_bt.add_argument("--max-leverage", type=float, default=1.0,
                      help="gross leverage cap (sum |w| ≤ N). Default 1.0.")
    p_bt.add_argument("--timeout", type=float, default=120.0,
                      help="seconds; aborts runaway backtests. 0 to disable.")
    p_bt.add_argument("--model", default="claude-opus-4-7")
    p_bt.add_argument("--temperature", type=float, default=0.2)
    p_bt.add_argument("--max-tokens", type=int, default=4096)
    p_bt.add_argument("--refine", action="store_true",
                      help="after the initial run, ask the LLM to review the "
                           "code + metrics and propose a fix (single round)")
    p_bt.add_argument("--agent", action="store_true",
                      help="use the multi-tool agent loop: model can call "
                           "run_dry_backtest / fetch_history / compute_metric "
                           "/ search_examples and iterate up to 5 rounds")
    p_bt.add_argument("--sandbox", action="store_true",
                      help="pre-flight code in an isolated subprocess "
                           "(static audit + subprocess exec). Defense-in-depth, "
                           "not a security boundary.")
    p_bt.add_argument("--frequency", default="D", choices=["D", "W", "M"],
                      help="bar frequency: D=daily, W=weekly, M=monthly")
    p_bt.add_argument("--realistic-costs", action="store_true",
                      help="use BidAskSpread + SquareRootImpact + BorrowCost; "
                           "more realistic than flat bps")
    p_bt.add_argument("--data-aware", action="store_true",
                      help="include 30-day universe sample + correlation matrix "
                           "in the prompt (adds ~1-2k tokens per call)")
    p_bt.set_defaults(func=cmd_backtest)

    p_ls = sub.add_parser("list", help="list saved runs")
    p_ls.set_defaults(func=cmd_list)

    p_sh = sub.add_parser("show", help="show one run's details")
    p_sh.add_argument("run_id")
    p_sh.add_argument("--open", action="store_true",
                      help="render HTML report and open in browser")
    p_sh.add_argument("--bootstrap", action="store_true",
                      help="compute block-bootstrap CIs on key metrics")
    p_sh.add_argument("--block-size", type=int, default=20)
    p_sh.add_argument("--n-resamples", type=int, default=1000)
    p_sh.set_defaults(func=cmd_show)

    p_cmp = sub.add_parser("compare", help="side-by-side compare two or more runs")
    p_cmp.add_argument("run_ids", nargs="+")
    p_cmp.add_argument("--plot", action="store_true",
                       help="render an overlaid equity-curve PNG")
    p_cmp.add_argument("--diff", action="store_true",
                       help="show unified diff of strategy code between runs")
    p_cmp.add_argument("--open", action="store_true",
                       help="open the plot in the default browser")
    p_cmp.set_defaults(func=cmd_compare)

    p_chat = sub.add_parser("chat", help="interactive REPL: each turn generates+runs a strategy")
    p_chat.set_defaults(func=lambda args: _cmd_chat(args))

    p_spa = sub.add_parser("spa", help="Hansen SPA test: benchmark vs alternatives")
    p_spa.add_argument("benchmark", help="run_id of the benchmark strategy")
    p_spa.add_argument("alternatives", nargs="+", help="run_ids of alternative strategies")
    p_spa.add_argument("--n-resamples", type=int, default=2000)
    p_spa.add_argument("--block-length", type=int, default=None,
                       help="stationary bootstrap mean block length; default = T^(1/3)")
    p_spa.add_argument("--seed", type=int, default=0)
    p_spa.set_defaults(func=cmd_spa)

    p_tree = sub.add_parser("tree", help="show the strategy-family tree across runs")
    p_tree.set_defaults(func=cmd_tree)

    p_fk = sub.add_parser("fork", help="fork from a prior run: re-generate strategy with follow-up")
    p_fk.add_argument("run_id", help="parent run id")
    p_fk.add_argument("followup", help="follow-up natural-language description")
    # Mirror the relevant backtest args so cmd_fork can dispatch.
    p_fk.add_argument("--universe", nargs="+", default=None)
    p_fk.add_argument("--train-end", default=None)
    p_fk.add_argument("--rebalance-freq", type=int, default=None)
    p_fk.add_argument("--start", default="2010-01-01")
    p_fk.add_argument("--end", default=None)
    p_fk.add_argument("--cost-bps", type=float, default=5.0)
    p_fk.add_argument("--timeout", type=float, default=120.0)
    p_fk.add_argument("--model", default="claude-opus-4-7")
    p_fk.add_argument("--temperature", type=float, default=0.2)
    p_fk.add_argument("--max-tokens", type=int, default=4096)
    p_fk.set_defaults(func=cmd_fork)

    p_ref = sub.add_parser("run-reference", help="run a built-in reference strategy (no LLM)")
    p_ref.add_argument("strategy", choices=sorted(REFERENCE_STRATEGIES))
    p_ref.add_argument("--universe", nargs="+", default=None)
    p_ref.add_argument("--start", default="2010-01-01")
    p_ref.add_argument("--end", default=None)
    p_ref.add_argument("--train-end", default="2015-01-01")
    p_ref.add_argument("--rebalance-freq", type=int, default=21)
    p_ref.add_argument("--cost-bps", type=float, default=5.0)
    p_ref.add_argument("--max-leverage", type=float, default=1.0)
    p_ref.set_defaults(func=cmd_run_reference)

    p_sw = sub.add_parser("sweep", help="sweep one hyperparameter for a saved run's strategy")
    p_sw.add_argument("run_id", help="base run id; the strategy code is taken from here")
    p_sw.add_argument("--param", required=True, help="constructor kwarg to vary")
    p_sw.add_argument("--values", required=True,
                      help="comma-separated values, e.g. '60,120,250' or '0.05,0.1,0.2'")
    p_sw.add_argument("--universe", nargs="+", default=None)
    p_sw.add_argument("--start", default="2010-01-01")
    p_sw.add_argument("--end", default=None)
    p_sw.add_argument("--train-end", default=None)
    p_sw.add_argument("--rebalance-freq", type=int, default=None)
    p_sw.add_argument("--cost-bps", type=float, default=5.0)
    p_sw.add_argument("--output", default=None, help="path to write CSV of results")
    p_sw.add_argument("--walk-forward", action="store_true",
                      help="run real walk-forward HP selection (retune + apply OOS) "
                           "instead of a single grid sweep")
    p_sw.add_argument("--tuning-years", type=float, default=3.0,
                      help="(walk-forward only) tuning-window length in years")
    p_sw.add_argument("--retune-years", type=float, default=1.0,
                      help="(walk-forward only) retune frequency in years")
    p_sw.add_argument("--initial-train-years", type=float, default=3.0,
                      help="(walk-forward only) data reserved before first OOS")
    p_sw.add_argument("--objective", default="sharpe",
                      choices=["sharpe", "sortino", "cagr", "calmar"],
                      help="(walk-forward only) in-sample tuning objective")
    p_sw.set_defaults(func=cmd_sweep)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
