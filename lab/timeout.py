"""Per-call timeout helper using SIGALRM (POSIX-only).

Used by the backtest runner to kill runaway strategies. Not perfect — a
strategy busy in C code or waiting on a syscall may not be interruptible
exactly at the deadline — but good enough for the common case where a
strategy loop in pure Python hangs.

Usage:
    with timeout(seconds=30):
        do_something()
"""

from __future__ import annotations

import contextlib
import logging
import signal
import threading

logger = logging.getLogger(__name__)


class TimeoutError(Exception):
    """Raised when a `with timeout(...)` block exceeds its deadline."""


@contextlib.contextmanager
def timeout(seconds: float | int):
    """Raise TimeoutError if the wrapped block runs longer than `seconds`.

    `seconds <= 0` disables the timeout (acts as a no-op). On non-POSIX
    systems (Windows), this is also a no-op — signal.SIGALRM is unavailable.
    From non-main threads (e.g., the web server's task workers) this is
    also a no-op — signal.signal() raises ValueError outside the main
    thread.
    """
    if seconds is None or seconds <= 0:
        yield
        return
    if not hasattr(signal, "SIGALRM"):
        # Windows: best we can do is no-op. Document upstream.
        yield
        return
    if threading.current_thread() is not threading.main_thread():
        # signal.signal() only works in the main thread. The web server
        # runs backtests in background threads — skip the timeout there
        # rather than crash. Worth a one-time warning so it's visible in
        # logs, but it doesn't block the work.
        logger.warning(
            "timeout requested in non-main thread (%s); skipping. "
            "Set timeout_seconds=0 in the caller to silence.",
            threading.current_thread().name,
        )
        yield
        return

    def _handler(signum, frame):
        raise TimeoutError(f"operation exceeded {seconds:g}s deadline")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    # setitimer supports float seconds. signal.alarm only takes ints.
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)
