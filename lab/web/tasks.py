"""In-memory background task tracker for the web server.

Backtests take 10–60 seconds; we don't want to block the HTTP request.
Each /api/run call enqueues a background thread, returns a task_id
immediately, and the frontend polls /api/tasks/{task_id} for status.

This is intentionally simple — single-process, in-memory dict, threading.
Not a real job queue. For single-user local use that's fine.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class TaskState:
    task_id: str
    status: str = "pending"           # pending | running | done | error
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    result: Any = None                # task-defined payload on success
    error: str | None = None          # exception repr on failure
    log: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result": self.result,
            "error": self.error,
            "log": list(self.log),
            "elapsed_seconds": (
                (self.finished_at or time.time()) - self.started_at
            ),
        }


class TaskRegistry:
    """Thread-safe in-memory task store."""

    def __init__(self) -> None:
        self._tasks: dict[str, TaskState] = {}
        self._lock = threading.Lock()

    def submit(self, fn: Callable[[TaskState], Any]) -> TaskState:
        """Run `fn(state)` in a background thread. Returns the TaskState
        immediately (status='pending')."""
        task_id = uuid.uuid4().hex[:12]
        state = TaskState(task_id=task_id)
        with self._lock:
            self._tasks[task_id] = state

        def _runner():
            state.status = "running"
            try:
                state.result = fn(state)
                state.status = "done"
            except Exception as e:
                logger.exception("task %s failed", task_id)
                state.status = "error"
                state.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}"
            finally:
                state.finished_at = time.time()

        t = threading.Thread(target=_runner, name=f"task-{task_id}", daemon=True)
        t.start()
        return state

    def get(self, task_id: str) -> TaskState | None:
        with self._lock:
            return self._tasks.get(task_id)

    def list_recent(self, limit: int = 50) -> list[TaskState]:
        with self._lock:
            items = sorted(self._tasks.values(),
                            key=lambda s: s.started_at, reverse=True)
            return items[:limit]


# Module-level registry — the server uses this. Single instance per process.
registry = TaskRegistry()
