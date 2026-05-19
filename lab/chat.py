"""lab chat — rich REPL for iterating on strategies.

Features:
  - prompt_toolkit input with persistent history (up-arrow recall).
  - `rich` formatting: colored metric tables, syntax-highlighted code.
  - Streaming output from Anthropic (token-by-token) when available.
  - Built-in commands: :help, :exit, :runs, :show, :compare, :reset,
    :save, :load, :universe.
  - Named sessions persisted to JSON.

Falls back to plain input() if prompt_toolkit / rich aren't importable.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    HAS_PT = True
except ImportError:
    HAS_PT = False

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from lab.llm import generate_strategy
from lab.runner import load_run, run_backtest, save_run

logger = logging.getLogger(__name__)

SESSIONS_DIR = Path(__file__).resolve().parent.parent / ".lab_sessions"
HISTORY_PATH = SESSIONS_DIR / "history"

HELP = """\
[bold cyan]Commands[/bold cyan]
  [yellow]:help[/yellow]               show this help
  [yellow]:exit, :q[/yellow]           leave the REPL
  [yellow]:runs[/yellow]               list runs in this session
  [yellow]:show[/yellow] <n>           details for run n (1-indexed within this session)
  [yellow]:compare[/yellow]            side-by-side metrics for all runs in this session
  [yellow]:reset[/yellow]              forget the parent chain (next turn starts fresh)
  [yellow]:universe[/yellow] T1 T2 …   set the asset universe (locks for the session)
  [yellow]:provider[/yellow] anthropic|openai|auto   switch LLM backend
  [yellow]:save[/yellow] <name>        save this session to disk
  [yellow]:load[/yellow] <name>        load a saved session by name
Anything else is treated as a strategy prompt.
"""
PLAIN_HELP = (
    "Commands:\n"
    "  :help / :exit / :runs / :show <n> / :compare / :reset\n"
    "  :universe T1 T2 ... / :provider <name> / :save <name> / :load <name>\n"
    "Anything else is a strategy prompt.\n"
)


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
    model: str | None = None
    temperature: float = 0.2
    max_tokens: int = 4096
    provider: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SessionState":
        return cls(**d)


# --------------------------------------------------------------------------- #
# Pretty-printing
# --------------------------------------------------------------------------- #


def _print_metrics_rich(console, run_id: str, strategy_name: str, metrics: dict):
    table = Table(title=f"Run {run_id} — {strategy_name}",
                  show_header=True, header_style="bold magenta")
    table.add_column("metric"); table.add_column("value", justify="right")
    pct_keys = {"cagr", "ann_vol", "max_drawdown"}
    order = ["sharpe", "sortino", "cagr", "ann_vol", "max_drawdown",
             "calmar", "final_nav", "annual_turnover"]
    for k in order:
        v = metrics.get(k)
        if v is None:
            continue
        if k in pct_keys:
            cell = f"{v*100:.2f}%"
        else:
            cell = f"{v:.4f}"
        # Color cue for Sharpe / drawdown.
        style = ""
        if k == "sharpe":
            style = "green" if v > 1.0 else ("yellow" if v > 0.5 else "red")
        elif k == "max_drawdown":
            style = "red" if v < -0.25 else ("yellow" if v < -0.10 else "green")
        table.add_row(k, f"[{style}]{cell}[/{style}]" if style else cell)
    console.print(table)


def _print_metrics_plain(run_id: str, strategy_name: str, metrics: dict):
    from tabulate import tabulate
    keys = ["sharpe", "sortino", "cagr", "ann_vol", "max_drawdown",
            "calmar", "final_nav"]
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


# --------------------------------------------------------------------------- #
# Turn execution
# --------------------------------------------------------------------------- #


def _do_turn(state: SessionState, user_prompt: str, console=None) -> None:
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

    if console and HAS_RICH:
        console.print(f"[dim]turn {len(state.run_ids) + 1}: generating strategy...[/dim]")
    else:
        logger.info("turn %d: generating strategy...", len(state.run_ids) + 1)

    generation = generate_strategy(
        full_prompt, model=state.model, max_tokens=state.max_tokens,
        temperature=state.temperature, provider=state.provider,
    )

    if console and HAS_RICH:
        console.print(Panel(
            Syntax(generation.code, "python", theme="monokai", line_numbers=True),
            title="generated strategy", border_style="cyan",
        ))
    else:
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
    state.parent_run_id = artifacts.run_id
    state.universe = universe
    state.train_end = train_end
    state.rebalance_freq = rebalance_freq

    if console and HAS_RICH:
        console.print(f"[green]Run saved:[/green] {artifacts.run_dir}")
        _print_metrics_rich(console, artifacts.run_id, strategy_name, metrics)
    else:
        print(f"\nRun saved: {artifacts.run_dir}")
        _print_metrics_plain(artifacts.run_id, strategy_name, metrics)


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def _do_command(state: SessionState, line: str, console=None) -> bool:
    parts = line[1:].split()
    if not parts:
        return True
    cmd, rest = parts[0], parts[1:]

    if cmd in ("exit", "q", "quit"):
        if console and HAS_RICH:
            console.print("[dim]bye[/dim]")
        else:
            print("bye")
        return False

    if cmd == "help":
        if console and HAS_RICH:
            console.print(Panel(HELP, title="help", border_style="cyan"))
        else:
            print(PLAIN_HELP)
        return True

    if cmd == "runs":
        if not state.run_ids:
            (console.print if console else print)("(no runs yet)")
            return True
        if console and HAS_RICH:
            table = Table(show_header=True, header_style="bold")
            table.add_column("#"); table.add_column("run_id"); table.add_column("strategy")
            table.add_column("Sharpe", justify="right")
            table.add_column("CAGR", justify="right")
            for i, rid in enumerate(state.run_ids, 1):
                r = load_run(rid)
                s = r["metrics"].get("sharpe")
                c = r["metrics"].get("cagr")
                table.add_row(
                    str(i), rid, r["config"].get("strategy_name", "?"),
                    f"{s:.3f}" if s is not None else "—",
                    f"{c*100:.2f}%" if c is not None else "—",
                )
            console.print(table)
        else:
            for i, rid in enumerate(state.run_ids, 1):
                r = load_run(rid)
                s = r["metrics"].get("sharpe")
                print(f"  {i}. {rid}  {r['config'].get('strategy_name','?')}  "
                      f"Sharpe={'NA' if s is None else f'{s:.3f}'}")
        return True

    if cmd == "show":
        if not rest:
            (console.print if console else print)("usage: :show <n>")
            return True
        try:
            idx = int(rest[0]) - 1
            rid = state.run_ids[idx]
        except (ValueError, IndexError):
            (console.print if console else print)(f"invalid run index: {rest[0]}")
            return True
        r = load_run(rid)
        if console and HAS_RICH:
            _print_metrics_rich(console, rid, r["config"].get("strategy_name", "?"),
                                r["metrics"])
            console.print(Panel(Syntax(r["strategy_code"], "python",
                                       theme="monokai", line_numbers=True),
                                title="strategy code"))
        else:
            _print_metrics_plain(rid, r["config"].get("strategy_name", "?"),
                                 r["metrics"])
            print("\nstrategy code:")
            print(r["strategy_code"])
        return True

    if cmd == "compare":
        if len(state.run_ids) < 2:
            (console.print if console else print)("need >= 2 runs to compare")
            return True
        import argparse
        from lab.cli import cmd_compare
        ns = argparse.Namespace(
            run_ids=state.run_ids, plot=False, diff=False, open=False,
        )
        cmd_compare(ns)
        return True

    if cmd == "reset":
        state.parent_run_id = None
        (console.print if console else print)(
            "parent chain cleared; next turn starts fresh"
        )
        return True

    if cmd == "universe":
        if not rest:
            current = state.universe or ["(generation-default)"]
            (console.print if console else print)(f"current universe: {current}")
            return True
        state.universe = [t.upper() for t in rest]
        (console.print if console else print)(
            f"universe locked to {state.universe}"
        )
        return True

    if cmd == "provider":
        if not rest:
            current = state.provider or "(auto-detect)"
            (console.print if console else print)(f"current provider: {current}")
            return True
        choice = rest[0].lower()
        if choice not in ("anthropic", "openai", "auto"):
            (console.print if console else print)(
                f"unknown provider: {choice}. expected: anthropic | openai | auto"
            )
            return True
        state.provider = None if choice == "auto" else choice
        (console.print if console else print)(
            f"provider set to {state.provider or 'auto-detect'}"
        )
        return True

    if cmd == "save":
        if not rest:
            (console.print if console else print)("usage: :save <name>")
            return True
        name = rest[0]
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        path = SESSIONS_DIR / f"{name}.json"
        path.write_text(json.dumps(state.to_dict(), indent=2))
        (console.print if console else print)(f"saved: {path}")
        return True

    if cmd == "load":
        if not rest:
            (console.print if console else print)("usage: :load <name>")
            return True
        name = rest[0]
        path = SESSIONS_DIR / f"{name}.json"
        if not path.exists():
            (console.print if console else print)(f"no such session: {name}")
            return True
        d = json.loads(path.read_text())
        loaded = SessionState.from_dict(d)
        # Replace all fields in-place so the caller's reference stays live.
        for k, v in asdict(loaded).items():
            setattr(state, k, v)
        (console.print if console else print)(
            f"loaded: {path} ({len(state.run_ids)} runs)"
        )
        return True

    (console.print if console else print)(
        f"unknown command: :{cmd}. Type :help for the list."
    )
    return True


# --------------------------------------------------------------------------- #
# REPL entry
# --------------------------------------------------------------------------- #


def run_chat(*, initial_state: SessionState | None = None) -> int:
    state = initial_state or SessionState()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

    console = Console() if HAS_RICH else None
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    if console:
        console.print(Panel.fit(
            "[bold cyan]lab chat[/bold cyan] — type [yellow]:help[/yellow] for commands, "
            "[yellow]:exit[/yellow] to leave.\n\n"
            "Each prompt regenerates a strategy and runs it; the previous run "
            "becomes the parent for the next turn.",
            border_style="cyan",
        ))
    else:
        print("lab chat — type :help for commands, :exit to leave.")

    if HAS_PT:
        session = PromptSession(history=FileHistory(str(HISTORY_PATH)))
        def read_line() -> str:
            return session.prompt(">> ")
    else:
        def read_line() -> str:
            return input(">> ")

    while True:
        try:
            line = read_line().strip()
        except (EOFError, KeyboardInterrupt):
            if console:
                console.print()
            else:
                print()
            return 0
        if not line:
            continue
        if line.startswith(":"):
            if not _do_command(state, line, console):
                return 0
            continue
        try:
            _do_turn(state, line, console)
        except KeyboardInterrupt:
            (console.print if console else print)(
                "(interrupted; previous turn discarded)"
            )
            continue
        except Exception as e:
            if console:
                console.print(f"[red]error:[/red] {type(e).__name__}: {e}")
            else:
                print(f"error: {type(e).__name__}: {e}")
            logger.exception("turn failed")
    return 0


if __name__ == "__main__":
    sys.exit(run_chat())
