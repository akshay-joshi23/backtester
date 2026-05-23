"""Strategy Lab web UI.

Local-only browser frontend to the lab. Launches via `python -m lab.cli web`
or `python -m lab.web.server`. Binds to 127.0.0.1 by default — never
exposed to the network unless the user explicitly overrides.
"""

from lab.web.server import create_app

__all__ = ["create_app"]
