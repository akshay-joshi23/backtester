"""Persistent state for the paper/live executor.

State per run lives under runs/<run_id>/live_state/:
  state.json       — ExecutorState (JSON-serialized)
  trades.jsonl     — append-only log of every order + fill + reconcile event
  state.json.lock  — file lock guarding atomic writes

Writes are atomic (write-to-tmp, fsync, rename) and guarded by a file
lock so concurrent invocations don't corrupt files.

Trade log lines are JSON dicts with at least:
  ts       — UTC ISO timestamp
  kind     — order_submit | order_fill | order_reject | reconcile |
             safety_halt | startup | shutdown
  payload  — kind-specific dict
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock

logger = logging.getLogger(__name__)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ExecutorState:
    run_id: str
    broker_name: str = "fake"
    is_live: bool = False
    started_at: str = field(default_factory=_iso_now)
    last_trade_at: str | None = None
    total_orders: int = 0
    total_fills: int = 0
    cumulative_pnl: float = 0.0
    peak_nav: float = 0.0
    starting_nav: float = 0.0
    halt_reason: str | None = None
    intended_positions: dict[str, float] = field(default_factory=dict)
    last_reconcile_at: str | None = None
    last_reconcile_drift: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    @classmethod
    def from_json(cls, text: str) -> "ExecutorState":
        d = json.loads(text)
        return cls(**d)


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #


def live_state_dir(run_dir: Path) -> Path:
    """Directory containing live-state artifacts for a given run."""
    return Path(run_dir) / "live_state"


def state_path(run_dir: Path) -> Path:
    return live_state_dir(run_dir) / "state.json"


def trade_log_path(run_dir: Path) -> Path:
    return live_state_dir(run_dir) / "trades.jsonl"


def halt_file_path(run_dir: Path) -> Path:
    """The HALT sentinel — created by `lab halt`, checked by step()."""
    return Path(run_dir) / "HALT"


def lock_path(run_dir: Path) -> Path:
    return live_state_dir(run_dir) / "state.json.lock"


# --------------------------------------------------------------------------- #
# State I/O
# --------------------------------------------------------------------------- #


def save_state(run_dir: Path, state: ExecutorState) -> None:
    """Atomic write under a file lock. Won't leave half-written files."""
    live_dir = live_state_dir(run_dir)
    live_dir.mkdir(parents=True, exist_ok=True)
    target = state_path(run_dir)
    lock = FileLock(str(lock_path(run_dir)), timeout=5.0)
    with lock:
        # write -> fsync -> rename for atomicity.
        fd, tmp = tempfile.mkstemp(prefix="state-", suffix=".json", dir=str(live_dir))
        try:
            with os.fdopen(fd, "w") as f:
                f.write(state.to_json())
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, target)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise


def load_state(run_dir: Path) -> ExecutorState:
    """Read state from disk. Raises FileNotFoundError if state doesn't exist."""
    sp = state_path(run_dir)
    if not sp.exists():
        raise FileNotFoundError(f"no live state at {sp}")
    lock = FileLock(str(lock_path(run_dir)), timeout=5.0)
    with lock:
        text = sp.read_text()
    return ExecutorState.from_json(text)


def state_exists(run_dir: Path) -> bool:
    return state_path(run_dir).exists()


# --------------------------------------------------------------------------- #
# Trade log
# --------------------------------------------------------------------------- #


def append_trade_log(run_dir: Path, kind: str, payload: dict) -> None:
    """Append one JSON line to the trade log."""
    live_dir = live_state_dir(run_dir)
    live_dir.mkdir(parents=True, exist_ok=True)
    entry = {"ts": _iso_now(), "kind": kind, "payload": payload}
    path = trade_log_path(run_dir)
    # File-append is already atomic for small writes on POSIX; no lock needed.
    with open(path, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def read_trade_log(run_dir: Path, limit: int | None = None) -> list[dict]:
    """Read the trade log. `limit=None` reads everything; otherwise last N lines."""
    path = trade_log_path(run_dir)
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    if limit is not None:
        lines = lines[-limit:]
    out: list[dict] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            logger.warning("bad trade-log line skipped: %s (%s)", line[:80], e)
    return out


# --------------------------------------------------------------------------- #
# State helpers
# --------------------------------------------------------------------------- #


def initialize_state(run_dir: Path, *, run_id: str, broker_name: str,
                      starting_nav: float) -> ExecutorState:
    """Create + persist an initial state record. Refuses to overwrite."""
    if state_exists(run_dir):
        raise FileExistsError(
            f"state already exists at {state_path(run_dir)}; "
            "use --reset to wipe and restart"
        )
    state = ExecutorState(
        run_id=run_id, broker_name=broker_name,
        starting_nav=starting_nav, peak_nav=starting_nav,
    )
    save_state(run_dir, state)
    append_trade_log(run_dir, "startup", {
        "run_id": run_id, "broker": broker_name,
        "starting_nav": starting_nav,
    })
    return state


def reset_state(run_dir: Path) -> None:
    """Delete state.json and trades.jsonl. Used by `--reset`."""
    for p in (state_path(run_dir), trade_log_path(run_dir), lock_path(run_dir)):
        if p.exists():
            p.unlink()
