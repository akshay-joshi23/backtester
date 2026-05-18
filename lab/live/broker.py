"""Broker abstraction layer for live/paper trading.

This module defines the BrokerAdapter ABC + a deterministic FakeBroker
implementation used across the test suite. The real Alpaca implementation
lives in this same file but is loaded lazily so the module can be imported
even when the `alpaca-py` package isn't installed.

Design contract for any BrokerAdapter:
  - All methods are synchronous; the executor calls them once per rebalance.
  - Timestamps are timezone-aware (UTC).
  - submit_order returns an OrderResult immediately; the order may still
    be pending. Use get_order(order_id) to poll fill status.
  - Implementations should NOT retry internally — the executor handles
    failure recovery on the next rebalance cycle.
"""

from __future__ import annotations

import logging
import os
import random
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderState(str, Enum):
    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class AccountSummary:
    cash: float
    equity: float
    buying_power: float
    day_trade_count: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class Position:
    ticker: str
    shares: float                # signed (negative = short)
    avg_entry_price: float
    market_value: float
    unrealized_pl: float = 0.0


@dataclass(frozen=True)
class Quote:
    ticker: str
    bid: float
    ask: float
    last: float
    timestamp: datetime

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.last


@dataclass(frozen=True)
class OrderRequest:
    ticker: str
    shares: float                # always positive; side determines direction
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    time_in_force: str = "day"
    client_order_id: str | None = None


@dataclass(frozen=True)
class Fill:
    shares: float
    price: float
    timestamp: datetime


@dataclass(frozen=True)
class OrderResult:
    order_id: str
    submitted_at: datetime
    request: OrderRequest


@dataclass(frozen=True)
class OrderStatus:
    order_id: str
    state: OrderState
    request: OrderRequest
    fills: tuple[Fill, ...] = ()
    rejected_reason: str | None = None

    @property
    def filled_shares(self) -> float:
        return sum(f.shares for f in self.fills)

    @property
    def average_fill_price(self) -> float:
        if not self.fills:
            return 0.0
        total_qty = self.filled_shares
        if total_qty == 0:
            return 0.0
        return sum(f.shares * f.price for f in self.fills) / total_qty


class BrokerError(Exception):
    """Raised when a broker call fails in a way the executor should record but
    not retry within the same step()."""


# --------------------------------------------------------------------------- #
# BrokerAdapter ABC
# --------------------------------------------------------------------------- #


class BrokerAdapter(ABC):
    """Synchronous broker interface. Implementations: AlpacaAdapter, FakeBroker."""

    name: str = "abstract"

    @abstractmethod
    def get_account(self) -> AccountSummary: ...

    @abstractmethod
    def get_positions(self) -> dict[str, Position]: ...

    @abstractmethod
    def get_quote(self, ticker: str) -> Quote: ...

    @abstractmethod
    def submit_order(self, order: OrderRequest) -> OrderResult: ...

    @abstractmethod
    def get_order(self, order_id: str) -> OrderStatus: ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> None: ...

    @abstractmethod
    def market_is_open(self) -> bool: ...


# --------------------------------------------------------------------------- #
# FakeBroker — deterministic in-memory adapter for tests
# --------------------------------------------------------------------------- #


@dataclass
class FakeBrokerConfig:
    starting_cash: float = 100_000.0
    fill_mode: str = "instant"          # instant | delayed | partial | reject | error
    fill_delay_seconds: float = 0.0
    partial_fill_fraction: float = 0.5
    rejection_reason: str = "simulated rejection"
    error_message: str = "simulated broker error"
    fixed_quotes: dict[str, float] | None = None
    spread_bps: float = 2.0
    market_open: bool = True
    allow_fractional: bool = True


class FakeBroker(BrokerAdapter):
    """Fully deterministic broker used in tests.

    Holds positions / cash in memory. Quotes either come from a fixed dict
    or are generated from a simple geometric-random-walk anchored at $100.
    Fill behavior is configurable per-instance: see FakeBrokerConfig.
    """

    name = "fake"

    def __init__(self, cfg: FakeBrokerConfig | None = None):
        self.cfg = cfg or FakeBrokerConfig()
        self._cash = self.cfg.starting_cash
        self._positions: dict[str, Position] = {}
        self._orders: dict[str, OrderStatus] = {}
        self._submitted_at: dict[str, float] = {}
        self._rng = random.Random(0)
        # Stable per-ticker reference price; updated by submit_order to drift.
        self._ref_prices: dict[str, float] = {}

    # -- account / positions --------------------------------------------- #

    def get_account(self) -> AccountSummary:
        # NAV = cash + market value of all positions at current quotes.
        positions = self.get_positions()
        equity = self._cash + sum(p.market_value for p in positions.values())
        return AccountSummary(
            cash=self._cash, equity=equity, buying_power=self._cash * 2.0,
        )

    def get_positions(self) -> dict[str, Position]:
        out: dict[str, Position] = {}
        for t, p in self._positions.items():
            if abs(p.shares) < 1e-12:
                continue
            q = self.get_quote(t)
            mv = p.shares * q.mid
            unr = mv - (p.shares * p.avg_entry_price)
            out[t] = Position(
                ticker=t, shares=p.shares,
                avg_entry_price=p.avg_entry_price,
                market_value=mv, unrealized_pl=unr,
            )
        return out

    # -- quotes ---------------------------------------------------------- #

    def set_quote(self, ticker: str, price: float) -> None:
        """Test helper — pin a specific price for a ticker."""
        if self.cfg.fixed_quotes is None:
            self.cfg = FakeBrokerConfig(
                **{**self.cfg.__dict__, "fixed_quotes": {}}
            )
        assert self.cfg.fixed_quotes is not None
        self.cfg.fixed_quotes[ticker] = price
        self._ref_prices[ticker] = price

    def get_quote(self, ticker: str) -> Quote:
        if self.cfg.fixed_quotes and ticker in self.cfg.fixed_quotes:
            mid = float(self.cfg.fixed_quotes[ticker])
        else:
            mid = self._ref_prices.setdefault(ticker, 100.0)
        spread = mid * self.cfg.spread_bps / 1e4
        return Quote(
            ticker=ticker, bid=mid - spread / 2, ask=mid + spread / 2,
            last=mid, timestamp=datetime.now(timezone.utc),
        )

    # -- orders ---------------------------------------------------------- #

    def submit_order(self, order: OrderRequest) -> OrderResult:
        if self.cfg.fill_mode == "error":
            raise BrokerError(self.cfg.error_message)
        order_id = f"fake-{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)
        self._orders[order_id] = OrderStatus(
            order_id=order_id, state=OrderState.PENDING,
            request=order, fills=(),
        )
        self._submitted_at[order_id] = time.time()
        if self.cfg.fill_mode == "instant":
            self._fill_order(order_id, full=True)
        elif self.cfg.fill_mode == "partial":
            self._fill_order(order_id, full=False)
        elif self.cfg.fill_mode == "reject":
            self._orders[order_id] = OrderStatus(
                order_id=order_id, state=OrderState.REJECTED,
                request=order, fills=(),
                rejected_reason=self.cfg.rejection_reason,
            )
        # 'delayed' keeps it pending until poll arrives after fill_delay.
        return OrderResult(order_id=order_id, submitted_at=now, request=order)

    def _fill_order(self, order_id: str, full: bool) -> None:
        status = self._orders[order_id]
        req = status.request
        q = self.get_quote(req.ticker)
        fill_price = q.ask if req.side == OrderSide.BUY else q.bid
        fill_qty = req.shares if full else req.shares * self.cfg.partial_fill_fraction
        if not self.cfg.allow_fractional:
            fill_qty = float(int(fill_qty))
        fill = Fill(shares=fill_qty, price=fill_price,
                    timestamp=datetime.now(timezone.utc))
        state = OrderState.FILLED if full else OrderState.PARTIAL
        self._orders[order_id] = OrderStatus(
            order_id=order_id, state=state, request=req, fills=(fill,),
        )
        # Apply position + cash change.
        sign = 1 if req.side == OrderSide.BUY else -1
        delta_shares = sign * fill_qty
        cost = sign * fill_qty * fill_price
        self._cash -= cost
        prev = self._positions.get(req.ticker)
        if prev is None or abs(prev.shares) < 1e-12:
            new_shares = delta_shares
            new_avg = fill_price if new_shares != 0 else 0.0
        else:
            new_shares = prev.shares + delta_shares
            if abs(new_shares) < 1e-12:
                new_avg = 0.0
            elif (prev.shares > 0 and delta_shares > 0) or (prev.shares < 0 and delta_shares < 0):
                # Adding to position — weighted average entry price.
                new_avg = (prev.shares * prev.avg_entry_price + delta_shares * fill_price) / new_shares
            else:
                # Reducing or flipping — keep prior entry price for the remainder.
                new_avg = prev.avg_entry_price
        self._positions[req.ticker] = Position(
            ticker=req.ticker, shares=new_shares,
            avg_entry_price=new_avg, market_value=new_shares * fill_price,
        )

    def get_order(self, order_id: str) -> OrderStatus:
        status = self._orders.get(order_id)
        if status is None:
            raise BrokerError(f"unknown order_id: {order_id}")
        # Honor delayed-fill: if we're past the delay, transition pending to filled.
        if status.state == OrderState.PENDING and self.cfg.fill_mode == "delayed":
            submitted = self._submitted_at.get(order_id, 0)
            if time.time() - submitted >= self.cfg.fill_delay_seconds:
                self._fill_order(order_id, full=True)
                status = self._orders[order_id]
        return status

    def cancel_order(self, order_id: str) -> None:
        status = self._orders.get(order_id)
        if status is None:
            raise BrokerError(f"unknown order_id: {order_id}")
        if status.state in (OrderState.FILLED, OrderState.REJECTED,
                             OrderState.CANCELLED):
            return
        self._orders[order_id] = OrderStatus(
            order_id=order_id, state=OrderState.CANCELLED,
            request=status.request, fills=status.fills,
        )

    def market_is_open(self) -> bool:
        return self.cfg.market_open

    # -- test helpers ---------------------------------------------------- #

    def force_fill_pending(self) -> int:
        """Force-fill any pending orders. Returns count filled. Test only."""
        n = 0
        for oid, status in list(self._orders.items()):
            if status.state == OrderState.PENDING:
                self._fill_order(oid, full=True)
                n += 1
        return n


# --------------------------------------------------------------------------- #
# AlpacaAdapter — placeholder; populated in a follow-up commit
# --------------------------------------------------------------------------- #


class AlpacaAdapter(BrokerAdapter):
    """Alpaca paper / live adapter. Requires `alpaca-py` package.

    Auth via env: APCA_API_KEY_ID, APCA_API_SECRET_KEY.

    Default endpoint = paper. `live=True` is GUARDED — it raises
    NotImplementedError with the canonical safety message. The user has to
    edit code (not pass a flag) to enable real-money trading.
    """

    name = "alpaca"

    LIVE_GUARD_MESSAGE = (
        "Live trading is not enabled in this build. To enable it, "
        "someone must: (1) verify paper trading has run cleanly for "
        "3+ months, (2) explicitly remove this guard, (3) add per-"
        "trade size caps, (4) set up real-time monitoring. None of "
        "that has been done."
    )

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_secret: str | None = None,
        live: bool = False,
        base_url: str | None = None,
    ):
        if live:
            raise NotImplementedError(self.LIVE_GUARD_MESSAGE)

        # If the user has APCA_BASE_URL set to a live URL, refuse loudly.
        env_url = base_url or os.environ.get("APCA_BASE_URL", "")
        if env_url and "paper" not in env_url.lower():
            raise NotImplementedError(
                f"APCA_BASE_URL appears to point at a non-paper endpoint "
                f"({env_url}). Refusing. " + self.LIVE_GUARD_MESSAGE
            )

        self._api_key = api_key or os.environ.get("APCA_API_KEY_ID")
        self._api_secret = api_secret or os.environ.get("APCA_API_SECRET_KEY")
        if not (self._api_key and self._api_secret):
            raise RuntimeError(
                "Alpaca credentials not found. Set APCA_API_KEY_ID and "
                "APCA_API_SECRET_KEY in the environment."
            )
        try:
            import alpaca  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "AlpacaAdapter requires the 'alpaca-py' package. "
                "Install with: pip install alpaca-py"
            ) from e
        # Lazy-import the actual clients so module-load is cheap.
        from alpaca.trading.client import TradingClient
        from alpaca.data.historical import StockHistoricalDataClient

        self._trading = TradingClient(
            api_key=self._api_key, secret_key=self._api_secret, paper=True,
        )
        self._data = StockHistoricalDataClient(
            api_key=self._api_key, secret_key=self._api_secret,
        )

    def get_account(self) -> AccountSummary:
        a = self._trading.get_account()
        return AccountSummary(
            cash=float(a.cash), equity=float(a.equity),
            buying_power=float(a.buying_power),
            day_trade_count=int(getattr(a, "daytrade_count", 0)),
        )

    def get_positions(self) -> dict[str, Position]:
        try:
            ps = self._trading.get_all_positions()
        except Exception as e:
            raise BrokerError(f"get_positions failed: {e}") from e
        out: dict[str, Position] = {}
        for p in ps:
            out[p.symbol.upper()] = Position(
                ticker=p.symbol.upper(),
                shares=float(p.qty),
                avg_entry_price=float(p.avg_entry_price),
                market_value=float(p.market_value),
                unrealized_pl=float(p.unrealized_pl or 0.0),
            )
        return out

    def get_quote(self, ticker: str) -> Quote:
        from alpaca.data.requests import StockLatestQuoteRequest
        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=ticker)
            res = self._data.get_stock_latest_quote(req)
            q = res[ticker]
        except Exception as e:
            raise BrokerError(f"get_quote failed for {ticker}: {e}") from e
        return Quote(
            ticker=ticker, bid=float(q.bid_price), ask=float(q.ask_price),
            last=float((q.bid_price + q.ask_price) / 2),
            timestamp=q.timestamp,
        )

    def submit_order(self, order: OrderRequest) -> OrderResult:
        from alpaca.trading.enums import OrderSide as AOS
        from alpaca.trading.enums import TimeInForce
        from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest

        side = AOS.BUY if order.side == OrderSide.BUY else AOS.SELL
        tif = TimeInForce.DAY if order.time_in_force == "day" else TimeInForce.GTC

        if order.order_type == OrderType.MARKET:
            req = MarketOrderRequest(
                symbol=order.ticker, qty=order.shares,
                side=side, time_in_force=tif,
                client_order_id=order.client_order_id,
            )
        else:
            if order.limit_price is None:
                raise ValueError("limit order requires limit_price")
            req = LimitOrderRequest(
                symbol=order.ticker, qty=order.shares,
                side=side, time_in_force=tif,
                limit_price=order.limit_price,
                client_order_id=order.client_order_id,
            )
        try:
            response = self._trading.submit_order(req)
        except Exception as e:
            raise BrokerError(f"submit_order failed: {e}") from e
        return OrderResult(
            order_id=str(response.id),
            submitted_at=response.submitted_at or datetime.now(timezone.utc),
            request=order,
        )

    def get_order(self, order_id: str) -> OrderStatus:
        try:
            o = self._trading.get_order_by_id(order_id)
        except Exception as e:
            raise BrokerError(f"get_order failed for {order_id}: {e}") from e
        state_map = {
            "new": OrderState.PENDING,
            "accepted": OrderState.PENDING,
            "pending_new": OrderState.PENDING,
            "partially_filled": OrderState.PARTIAL,
            "filled": OrderState.FILLED,
            "rejected": OrderState.REJECTED,
            "canceled": OrderState.CANCELLED,
            "cancelled": OrderState.CANCELLED,
        }
        state = state_map.get(str(o.status).lower(), OrderState.PENDING)
        fills: tuple[Fill, ...] = ()
        if o.filled_qty and float(o.filled_qty) > 0:
            fills = (Fill(
                shares=float(o.filled_qty),
                price=float(o.filled_avg_price or 0.0),
                timestamp=o.filled_at or datetime.now(timezone.utc),
            ),)
        # We don't have OrderRequest stored on Alpaca's side, reconstruct minimally.
        request = OrderRequest(
            ticker=str(o.symbol).upper(),
            shares=float(o.qty), side=OrderSide(str(o.side).lower()),
            order_type=OrderType(str(o.order_type).lower()),
            limit_price=float(o.limit_price) if o.limit_price else None,
        )
        return OrderStatus(
            order_id=order_id, state=state, request=request, fills=fills,
            rejected_reason=getattr(o, "rejected_reason", None),
        )

    def cancel_order(self, order_id: str) -> None:
        try:
            self._trading.cancel_order_by_id(order_id)
        except Exception as e:
            raise BrokerError(f"cancel_order failed for {order_id}: {e}") from e

    def market_is_open(self) -> bool:
        try:
            clock = self._trading.get_clock()
        except Exception as e:
            raise BrokerError(f"get_clock failed: {e}") from e
        return bool(clock.is_open)
