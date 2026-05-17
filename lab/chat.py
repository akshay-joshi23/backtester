"""Minimal lab chat REPL.

User types prompts at a `>>` line. Each turn:
  1. Compose the prompt — if there's a prior run, prepend it as parent context.
  2. Generate strategy (validated + retry as usual).
  3. Run backtest.
  4. Save the run, link to prior turn as parent.
  5. Print metrics + ask for next turn.

Built-in commands inside the REPL:
  :help         show commands
  :exit / :q    leave
  :runs         list runs in this session
  :show <n>     show metrics + code for the n-th run this session
  :compare      compare all runs in this session side-by-side
  :reset        forget the parent chain (next turn starts fresh)

Deliberate simplifications versus a polished UI:
  - No streaming output; the LLM response comes back as one chunk.
  - No rich formatting / colors (plain text).
  - No history-search / arrow-key recall (use up-arrow at your shell).
  - No mid-turn cancellation.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field

from tabulate import tabulate

from lab.llm import generate_strategy
from lab.runner import load_run, run_backtest, save_run

logger = logging.getLogger(__name__)

HELP = """\
Commands:
  :help               show this help
  :exit, :q, :quit    leave the REPL
  :runs               list runs in this session
  :show <n>           show details for run n (1-indexed in this session)
  :compare            side-by-side metrics for all runs in this session
  :reset              forget the parent chain; next turn starts fresh
Anything else is treated as a strategy prompt.
"""


@dataclass
class SessionState:
    run_ids: list[str] = field(default_factory=list)
    parent_run_id: str | None = None
    universe: list[str] | None = None
    train_end: str | None = None
    rebalance_freq: int | None = None
    start: str = "2010-01-01"
    end: str | None = None
    cost_bps: float = 5.0
    timeout_seconds: float = 120.0
    model: str = "claude-opus-4-7"
    temperature: float = 0.2
    max_tokens: int = 4096


def _print_metrics(run_id: str, strategy_name: str, metrics: dict) -> None:
    keys = ["sharpe", "sortino", "cagr", "ann_vol", "max_drawdown", "calmar", "final_nav"]
    pct_keys = {"cagr", "ann_vol", "max_drawdown"}
    row = [run_id, strategy_name]
    for k in keys:
        v = metrics.get(k)
        if v is None:
            row.append("—")
        elif k in pct_keys:
            row.append(f"{v*100:.2f}%")
        else:
            row.append(f"{v:.4f}")
    print(tabulate([row], headers=["run_id", "strategy"] + keys))


def _do_turn(state: SessionState, user_prompt: str) -> None:
    """Generate + backtest + save for one user prompt."""
    if state.parent_run_id:
        parent = load_run(state.parent_run_id)
        full_prompt = (
            f"Previously you wrote this strategy (run {state.parent_run_id}, "
            f"named '{parent['config'].get('strategy_name','?')}'):\n\n"
            "```python\n" + parent["strategy_code"] + "```\n\n"
            f"Now revise it for the following follow-up:\n\n{user_prompt}"
        )
    else:
        full_prompt = user_prompt

    logger.info("turn %d: generating strategy...", len(state.run_ids) + 1)
    generation = generate_strategy(
        full_prompt, model=state.model, max_tokens=state.max_tokens,
        temperature=state.temperature,
    )
    print("--- generated strategy ---")
    print(generation.code)
    print("--- end strategy ---\n")
    universe = state.universe or generation.universe
    train_end = state.train_end or generation.train_end
    rebalance_freq = state.rebalance_freq or generation.rebalance_freq
    equity, weights, metrics, cfg, strategy_name = run_backtest(
        generation.code,
        universe=universe, train_end=train_end, rebalance_freq=rebalance_freq,
        start=state.start, end=state.end,
        cost_bps=state.cost_bps, timeout_seconds=state.timeout_seconds,
    )
    artifacts = save_run(
        prompt=user_prompt, generation=generation,
        equity=equity, weights=weights, metrics=metrics,
        cfg=cfg, strategy_name=strategy_name, universe=universe,
        parent_run_id=state.parent_run_id,
    )
    state.run_ids.append(artifacts.run_id)
    state.parent_run_id = artifacts.run_id  # chain to next turn
    state.universe = universe  # lock the universe after the first turn
    state.train_end = train_end
    state.rebalance_freq = rebalance_freq

    print(f"\nRun saved: {artifacts.run_dir}")
    _print_metrics(artifacts.run_id, strategy_name, metrics)


def _do_command(state: SessionState, line: str) -> bool:
    """Returns True if the REPL should keep going, False to exit."""
    parts = line[1:].split()
    if not parts:
        return True
    cmd, rest = parts[0], parts[1:]
    if cmd in ("exit", "q", "quit"):
        print("bye")
        return False
    if cmd == "help":
        print(HELP)
        return True
    if cmd == "runs":
        for i, rid in enumerate(state.run_ids, 1):
            r = load_run(rid)
            print(f"  {i}. {rid}  {r['config'].get('strategy_name','?')}  "
                  f"Sharpe={r['metrics'].get('sharpe', 'NA'):.3f}")
        return True
    if cmd == "show":
        if not rest:
            print("usage: :show <n>")
            return True
        try:
            idx = int(rest[0]) - 1
            rid = state.run_ids[idx]
        except (ValueError, IndexError):
            print(f"invalid run index: {rest[0]}")
            return True
        r = load_run(rid)
        _print_metrics(rid, r["config"].get("strategy_name", "?"), r["metrics"])
        print("\nstrategy code:")
        print(r["strategy_code"])
        return True
    if cmd == "compare":
        if len(state.run_ids) < 2:
            print("need >= 2 runs to compare")
            return True
        # Defer to the CLI compare logic for consistency.
        import argparse
        from lab.cli import cmd_compare
        ns = argparse.Namespace(
            run_ids=state.run_ids, plot=False, diff=False, open=False,
        )
        cmd_compare(ns)
        return True
    if cmd == "reset":
        state.parent_run_id = None
        print("parent chain cleared; next turn starts fresh")
        return True
    print(f"unknown command: :{cmd}. Type :help for the list.")
    return True


def run_chat(*, initial_state: SessionState | None = None) -> int:
    state = initial_state or SessionState()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print("lab chat — type :help for commands, :exit to leave. Each prompt "
          "regenerates a strategy and runs it; the previous run becomes the "
          "parent for the next turn.")
    while True:
        try:
            line = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line.startswith(":"):
            if not _do_command(state, line):
                return 0
            continue
        try:
            _do_turn(state, line)
        except KeyboardInterrupt:
            print("(interrupted; previous turn discarded)")
            continue
        except Exception as e:
            print(f"error: {type(e).__name__}: {e}")
            logger.exception("turn failed")
    return 0


if __name__ == "__main__":
    sys.exit(run_chat())
