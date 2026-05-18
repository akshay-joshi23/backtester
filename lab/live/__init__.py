"""Strategy Lab live/paper trading executor.

Public API is intentionally small. The executor consumes a saved
backtest run (runs/<id>/) and trades the strategy via a BrokerAdapter.

Default and ONLY-shipped mode is paper trading via Alpaca. The
`live=True` flag on the AlpacaAdapter raises NotImplementedError with
a canonical message — see broker.AlpacaAdapter.LIVE_GUARD_MESSAGE.
"""

from lab.live.broker import (
    AccountSummary,
    AlpacaAdapter,
    BrokerAdapter,
    BrokerError,
    FakeBroker,
    FakeBrokerConfig,
    Fill,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderState,
    OrderStatus,
    OrderType,
    Position,
    Quote,
)

__all__ = [
    "AccountSummary",
    "AlpacaAdapter",
    "BrokerAdapter",
    "BrokerError",
    "FakeBroker",
    "FakeBrokerConfig",
    "Fill",
    "OrderRequest",
    "OrderResult",
    "OrderSide",
    "OrderState",
    "OrderStatus",
    "OrderType",
    "Position",
    "Quote",
]
