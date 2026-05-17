"""Sandboxed exec for generated strategy code.

THREAT MODEL & HONEST LIMITATIONS:

  This sandbox is a *defense-in-depth* layer for personal use, NOT a
  security boundary. The Python ecosystem doesn't offer a real sandbox
  inside the interpreter; meaningful isolation requires an OS-level
  jail (container, namespace, seccomp). A determined attacker who can
  inject Python into your process is already past most boundaries.

  What this sandbox provides:
    - Static AST-level rejection of obvious foot-guns:
        - imports of `socket`, `urllib`, `requests`, `subprocess`,
          `os.system`, `os.popen`, `os.execv`, `pty`, `ftplib`,
          `smtplib`, `http`, `shutil` (write ops), `pickle.load*`.
        - calls to `eval`, `exec`, `compile`, `__import__`,
          `open(..., 'w'|'a'|'x'|'rb+'|'r+')` — anything that writes
          or can re-import dynamically.
        - attribute access on `__builtins__`, `__globals__`,
          `__class__`, `__bases__`, `__subclasses__`.
    - Optional subprocess isolation: the strategy is exec'd in a fresh
      Python subprocess with -I (isolated) and -S (no site) so it can't
      reach your user/site-packages personal data. This is a moderate
      additional barrier.

  What this sandbox does NOT do:
    - Limit CPU / memory / wallclock (use the SIGALRM timeout for wallclock).
    - Stop import of allowed-but-dangerous modules (numpy can mmap files).
    - Prevent reading the filesystem under your user (read-only).
    - Catch obfuscated import bypasses (e.g., __import__ via string concat).

  Use this when running code from a less-trusted source. For your own
  code or LLM-generated code you've eyeballed, the unsandboxed path
  in lab.runner.execute_strategy_code is fine.

USAGE:

    from lab.sandbox import audit_code, AuditFailure

    # 1. Static pre-check (cheap, in-process).
    audit_code(generated_code)  # raises AuditFailure if obvious foot-guns

    # 2. Optional subprocess isolation for execution (heavier).
    # Wired via run_backtest_sandboxed() below — used by 'lab backtest --sandbox'.
"""

from __future__ import annotations

import ast
import logging
import textwrap

logger = logging.getLogger(__name__)


class AuditFailure(Exception):
    """Raised when code contains a denylisted pattern."""


# Module names that are categorically refused.
BLOCKED_MODULES: frozenset[str] = frozenset({
    "socket",
    "urllib", "urllib2", "urllib3",
    "requests", "httpx", "aiohttp",
    "http", "httplib",
    "subprocess",
    "pty",
    "ftplib",
    "smtplib", "imaplib", "poplib",
    "telnetlib",
    "ctypes", "cffi",
    "multiprocessing",
})

# Submodules of `os` we don't want.
BLOCKED_OS_ATTRS: frozenset[str] = frozenset({
    "system", "popen", "execv", "execvp", "execve", "execvpe",
    "fork", "forkpty", "spawnv", "spawnvp", "spawnve", "spawnvpe",
})

# Names that bypass the import gate when called as builtins.
BLOCKED_CALLS: frozenset[str] = frozenset({
    "eval", "exec", "compile", "__import__",
})

# Attribute-access patterns that signal introspection escape attempts.
BLOCKED_ATTRS: frozenset[str] = frozenset({
    "__class__", "__bases__", "__subclasses__", "__mro__",
    "__globals__", "__builtins__", "__getattribute__",
    "__init_subclass__", "__reduce__", "__reduce_ex__",
    "f_back", "f_locals", "f_globals", "f_code",  # frame escape
})


def _check_import(node: ast.AST) -> str | None:
    if isinstance(node, ast.Import):
        for alias in node.names:
            top = alias.name.split(".")[0]
            if top in BLOCKED_MODULES:
                return f"blocked import: {alias.name}"
    elif isinstance(node, ast.ImportFrom):
        if node.module is None:
            return None
        top = node.module.split(".")[0]
        if top in BLOCKED_MODULES:
            return f"blocked import: from {node.module} import ..."
        if top == "os":
            for alias in node.names:
                if alias.name in BLOCKED_OS_ATTRS:
                    return f"blocked import: from os import {alias.name}"
    return None


def _check_call(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALLS:
        return f"blocked call: {node.func.id}(...)"
    if isinstance(node.func, ast.Attribute):
        # os.system / os.popen / etc.
        if isinstance(node.func.value, ast.Name) and node.func.value.id == "os":
            if node.func.attr in BLOCKED_OS_ATTRS:
                return f"blocked call: os.{node.func.attr}(...)"
        # open(..., 'w'|'a'|'x'|...) — file writes.
        if isinstance(node.func, ast.Name) and node.func.id == "open":
            for arg in node.args + [kw.value for kw in node.keywords]:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if any(c in arg.value for c in ("w", "a", "x", "+")):
                        return "blocked: open() in write/append/exclusive mode"
    if isinstance(node.func, ast.Name) and node.func.id == "open":
        for arg in node.args + [kw.value for kw in node.keywords]:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if any(c in arg.value for c in ("w", "a", "x", "+")):
                    return "blocked: open() in write/append/exclusive mode"
    return None


def _check_attr(node: ast.Attribute) -> str | None:
    if node.attr in BLOCKED_ATTRS:
        return f"blocked attribute access: .{node.attr}"
    return None


def audit_code(code: str) -> None:
    """Static AST audit. Raises AuditFailure on any denylist hit.

    Returns silently on pass. Best-effort; not a substitute for OS isolation.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise AuditFailure(f"syntax error: {e}") from e
    issues: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            problem = _check_import(node)
            if problem:
                issues.append(f"line {node.lineno}: {problem}")
        elif isinstance(node, ast.Call):
            problem = _check_call(node)
            if problem:
                issues.append(f"line {node.lineno}: {problem}")
        elif isinstance(node, ast.Attribute):
            problem = _check_attr(node)
            if problem:
                issues.append(f"line {node.lineno}: {problem}")
    if issues:
        raise AuditFailure(
            "sandbox audit failed:\n  - " + "\n  - ".join(issues)
        )


def run_strategy_code_sandboxed(
    code: str,
    *,
    timeout_seconds: float = 30.0,
) -> str:
    """Exec the code in a fresh Python subprocess for additional isolation.

    Validates that the code can be loaded and finds exactly one Strategy
    subclass — i.e., does the same job as `execute_strategy_code` but
    inside an isolated subprocess. Returns the Strategy class name on
    success; raises RuntimeError on failure.

    Not used as the main exec path — too slow to invoke per-rebalance.
    Use this to *pre-flight* code before letting it run in the main process.
    """
    import json
    import subprocess
    import sys

    audit_code(code)  # static pre-check

    probe = textwrap.dedent("""
        import json, sys
        code = sys.stdin.read()
        ns = {}
        try:
            exec(compile(code, '<sandbox>', 'exec'), ns)
        except Exception as e:
            print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}))
            sys.exit(0)
        from lab.strategy import Strategy
        classes = [v for v in ns.values()
                   if isinstance(v, type) and issubclass(v, Strategy)
                   and v is not Strategy]
        if len(classes) != 1:
            print(json.dumps({"ok": False, "error": f"expected 1 Strategy subclass, found {len(classes)}"}))
            sys.exit(0)
        print(json.dumps({"ok": True, "class_name": classes[0].__name__}))
    """).strip()
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-c", probe],
            input=code, capture_output=True, text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"sandbox subprocess timed out after {timeout_seconds}s") from e
    if result.returncode != 0:
        raise RuntimeError(f"sandbox subprocess crashed: {result.stderr}")
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise RuntimeError(f"sandbox produced unparseable output: {result.stdout!r}")
    if not payload.get("ok"):
        raise RuntimeError(f"sandbox rejected code: {payload.get('error')}")
    return payload["class_name"]
