"""BrokerAdapter ABC + Order/Position/AccountMetrics dataclasses.

Phase A1.13 per internal design notes.

## Architecture

The `BrokerAdapter` ABC is the unique seam between strategy / engine code and
ANY broker (IBKR paper, IBKR live, TopstepX funded, MockBroker for backtest).
BACKTEST-IS-LIVE: identical code paths in backtest vs live; only the
concrete adapter changes at the seam.

## Contracts

- **Mode-agnostic** (backtest-is-live): engine never imports a concrete adapter; always
  receives one via `RuntimeContext` injection.
- **No silent downgrades** (fail-loud policy): adapter rejects unsupported
  order types loudly via `OrderTypeUnsupportedError`. Strategy layer must
  declare order-type requirements up-front + query `supported_order_types()`
  at startup.
- **Pre-trade hard-gate**: `is_trade_allowed()` is called
  by every strategy BEFORE `place_order()`. Topstep-aware adapters check
  daily-loss + trailing-DD + consistency-rule + max-position; IBKR adapter
  checks margin headroom only.
- **Provenance**: every `OrderEvent` carries `raw_broker_payload_sha`
  (SHA256 of the upstream broker's response bytes) for replay-diff + audit.

## Order types (13 — Phase A1 in-scope)

Per internal design notes matrix. Excluded as Phase A2:
Iceberg + Algo (size too small + IBALGO complexity).

References:
- internal design notes (Python signatures locked)
- internal design notes (IBKR 15-type matrix)
- internal design notes (ProjectX 7-type subset)
- internal design notes (TopstepX downgrade contract)
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from typing import Literal

from futur3.data.types import SHA256_HEX_LENGTH, ContractSymbol

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class OrderType(Enum):
    """13 order types in Phase A1 scope (per internal design notes).

    IBKR supports 11 (all except `JOIN_BID` / `JOIN_ASK` which are TopstepX-native).
    TopstepX supports 7 (`MKT` / `LMT` / `STP` / `STP_LMT` / `TRAIL` /
    `JOIN_BID` / `JOIN_ASK`). MockBroker simulates all 13.

    See `BrokerAdapter.supported_order_types()` for per-adapter capability.
    """

    MKT = "market"
    LMT = "limit"
    STP = "stop"
    STP_LMT = "stop_limit"
    STP_PRT = "stop_with_protection"  # CME-native; IBKR pass-through; TopstepX downgrade to STP
    MIT = "market_if_touched"  # IBKR only
    LIT = "limit_if_touched"  # IBKR only
    MOC = "market_on_close"  # IBKR only — strategies that require MOC
    MOO = "market_on_open"  # IBKR only — gap-fill strategies
    TRAIL = "trailing_stop"  # both IBKR + TopstepX
    TRAIL_LMT = "trailing_stop_limit"  # IBKR only
    JOIN_BID = "join_bid"  # TopstepX only — passive entry
    JOIN_ASK = "join_ask"  # TopstepX only — passive entry


class Side(Enum):
    BUY = "buy"
    SELL = "sell"


class Duration(Enum):
    DAY = "day"
    GTC = "good_till_cancelled"
    GTD = "good_till_date"


OrderEventType = Literal[
    "submitted",
    "ack",
    "filled",
    "partial_fill",
    "cancelled",
    "rejected",
    "modified",
]


ConsistencyRuleStatus = Literal["compliant", "warning", "violation"]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class BrokerError(Exception):
    """Base for all broker-layer errors. Caught at strategy boundary."""


class OrderTypeUnsupportedError(BrokerError):
    """The adapter does not natively support the requested OrderType.

    Per the fail-loud no-silent-fallback policy: strategy layer must catch
    + skip the strategy on that adapter (or query `supported_order_types`
    upfront). Raised by `BrokerAdapter.place_order` when:
    - TopstepX receives MOC / MOO / Iceberg / Algo (REJECT per internal notes)
    - Any adapter receives an OrderType not in its `supported_order_types()`
    """


class BrokerNotConnectedError(BrokerError):
    """The adapter has no active session (e.g., IBKR Gateway offline).

    Strategy layer must NOT retry blindly — surface to operator + check
    `healthcheck()` before reconnecting. Distinct from `OrderTypeUnsupportedError`
    (which is permanent for that adapter) — this one is transient.
    """


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Order:
    """Pre-submission order spec. Adapter-agnostic.

    `client_order_id` is the futur3-side correlation key (UUID or strategy-id-tag);
    the broker assigns its own `broker_order_id` on submission and that gets
    returned by `place_order()`. Track both for the audit chain.
    """

    contract: ContractSymbol
    side: Side
    quantity: int  # always > 0; direction is in `side`
    order_type: OrderType
    limit_price: Decimal | None = None  # required for LMT / STP_LMT / TRAIL_LMT / LIT
    stop_price: Decimal | None = None  # required for STP / STP_LMT / STP_PRT / MIT
    duration: Duration = Duration.DAY
    parent_order_id: str | None = None  # bracket-leg pointer
    oco_group_id: str | None = None  # OCO siblings group
    client_order_id: str | None = None  # futur3-side correlation

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(
                f"Order.quantity must be > 0 (direction in `side`); got {self.quantity}"
            )
        # OrderType -> required price field invariants (no-silent-fallback).
        _requires_limit = {
            OrderType.LMT,
            OrderType.STP_LMT,
            OrderType.LIT,
            OrderType.TRAIL_LMT,
        }
        _requires_stop = {
            OrderType.STP,
            OrderType.STP_LMT,
            OrderType.STP_PRT,
            OrderType.MIT,
        }
        if self.order_type in _requires_limit and self.limit_price is None:
            raise ValueError(f"Order.limit_price required for {self.order_type.name}; got None")
        if self.order_type in _requires_stop and self.stop_price is None:
            raise ValueError(f"Order.stop_price required for {self.order_type.name}; got None")


@dataclass(frozen=True)
class OrderEvent:
    """Single state-transition event for an order. Streamed by
    `BrokerAdapter.stream_order_events()`.

    `raw_broker_payload_sha` is SHA256 of the upstream broker's response bytes
    for that event — required for the audit chain + replay-diff.
    """

    broker_order_id: str
    client_order_id: str
    contract: ContractSymbol
    event_type: OrderEventType
    fill_price: Decimal | None
    fill_quantity: int | None
    cumulative_filled: int  # >= 0; tracks total filled across partials
    remaining_quantity: int  # >= 0
    ts_utc: datetime
    raw_broker_payload_sha: str

    def __post_init__(self) -> None:
        if self.ts_utc.tzinfo is None:
            raise ValueError(f"OrderEvent.ts_utc must be TZ-aware; got naive {self.ts_utc!r}")
        if self.ts_utc.tzinfo.utcoffset(self.ts_utc) != UTC.utcoffset(self.ts_utc):
            # Defensive: future sources might emit non-UTC datetimes; coerce upstream.
            raise ValueError(f"OrderEvent.ts_utc must be UTC; got tzinfo {self.ts_utc.tzinfo!r}")
        if self.cumulative_filled < 0:
            raise ValueError(
                f"OrderEvent.cumulative_filled must be >= 0; got {self.cumulative_filled}"
            )
        if self.remaining_quantity < 0:
            raise ValueError(
                f"OrderEvent.remaining_quantity must be >= 0; got {self.remaining_quantity}"
            )
        if self.fill_quantity is not None and self.fill_quantity < 0:
            raise ValueError(
                f"OrderEvent.fill_quantity must be >= 0 if set; got {self.fill_quantity}"
            )
        if len(self.raw_broker_payload_sha) != SHA256_HEX_LENGTH:
            raise ValueError(
                f"OrderEvent.raw_broker_payload_sha must be hex-SHA256 "
                f"({SHA256_HEX_LENGTH} chars); got len={len(self.raw_broker_payload_sha)}"
            )


@dataclass(frozen=True)
class Position:
    """Current open position on a single contract.

    `quantity` is SIGNED: positive = long, negative = short, 0 = flat.
    """

    contract: ContractSymbol
    quantity: int
    avg_entry_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl_today: Decimal
    margin_used: Decimal

    def __post_init__(self) -> None:
        if self.margin_used < 0:
            raise ValueError(f"Position.margin_used must be >= 0; got {self.margin_used}")
        if self.quantity != 0 and self.avg_entry_price <= 0:
            raise ValueError(
                f"Position.avg_entry_price must be > 0 when quantity != 0; "
                f"got {self.avg_entry_price} with quantity {self.quantity}"
            )


@dataclass(frozen=True)
class AccountMetrics:
    """Aggregate account state. Topstep-specific fields are optional —
    IBKRBrokerAdapter populates them as None; TopstepXBrokerAdapter populates them.
    """

    account_id: str
    equity: Decimal
    cash: Decimal
    margin_used: Decimal
    margin_available: Decimal
    leverage_used: Decimal  # equity / margin_used (or 0 if no positions)
    realized_pnl_today: Decimal
    unrealized_pnl: Decimal
    # Topstep-specific (None on IBKR-side; populated by TopstepXBrokerAdapter)
    trailing_drawdown_remaining: Decimal | None = None
    consistency_rule_status: ConsistencyRuleStatus | None = None
    max_position_size_allowed: int | None = None

    def __post_init__(self) -> None:
        if self.margin_used < 0:
            raise ValueError(f"AccountMetrics.margin_used must be >= 0; got {self.margin_used}")
        if self.margin_available < 0:
            raise ValueError(
                f"AccountMetrics.margin_available must be >= 0; got {self.margin_available}"
            )
        if self.leverage_used < 0:
            raise ValueError(f"AccountMetrics.leverage_used must be >= 0; got {self.leverage_used}")
        if self.max_position_size_allowed is not None and self.max_position_size_allowed < 0:
            raise ValueError(
                f"AccountMetrics.max_position_size_allowed must be >= 0 if set; "
                f"got {self.max_position_size_allowed}"
            )


# ---------------------------------------------------------------------------
# BrokerAdapter ABC
# ---------------------------------------------------------------------------


class BrokerAdapter(abc.ABC):
    """Abstract base for ALL broker integrations (Mock + IBKR + TopstepX).

    Methods are async because real broker SDKs (`ib_async`, ProjectX REST/WS)
    are coroutine-based. Subclasses MUST honor the cost contract per
    internal design notes:
    - No silent fallback on unsupported order types (raise OrderTypeUnsupportedError)
    - Pre-trade hard-gate via `is_trade_allowed` (synchronous; no I/O)
    - `supported_order_types()` returns the set known at construction time
    - `stream_order_events()` is the SOLE event source — never read order state
      via repeated `get_positions()` polls (race conditions)
    """

    @property
    @abc.abstractmethod
    def broker_id(self) -> str:
        """Stable per-adapter identifier (e.g., `"ibkr_tws"`, `"topstepx"`,
        `"mock"`). Used as `source_id`-equivalent for execution-side provenance.
        """

    @abc.abstractmethod
    async def place_order(self, order: Order) -> str:
        """Submit a new order. Returns broker-assigned `broker_order_id`.

        Raises:
            OrderTypeUnsupportedError: order.order_type not in supported_order_types().
            BrokerNotConnectedError: adapter session offline.
            BrokerError: any other broker-side rejection.
        """

    @abc.abstractmethod
    async def cancel_order(self, broker_order_id: str) -> None:
        """Cancel an open order by broker-assigned ID.

        Raises:
            BrokerNotConnectedError: adapter session offline.
            BrokerError: order already filled / not found / etc.
        """

    @abc.abstractmethod
    async def modify_order(
        self,
        broker_order_id: str,
        new_limit_price: Decimal | None = None,
        new_stop_price: Decimal | None = None,
        new_quantity: int | None = None,
    ) -> None:
        """Modify limit price / stop price / remaining quantity in-place.

        At least one of `new_limit_price`, `new_stop_price`, `new_quantity`
        must be non-None. Adapter validates which fields are modifiable for
        the underlying order_type.

        Raises:
            ValueError: all 3 new_* args are None.
            BrokerNotConnectedError: adapter session offline.
            BrokerError: order already filled / modification rejected.
        """

    @abc.abstractmethod
    async def get_positions(self) -> list[Position]:
        """Snapshot of all open positions. Read-only; not the event source.

        Use `stream_order_events()` for state transitions; this method is for
        position-aware decisions (e.g., flatten-all-on-EOD, sizing checks).
        """

    @abc.abstractmethod
    async def get_account_metrics(self) -> AccountMetrics:
        """Snapshot of account-level state (equity / margin / trailing-DD / etc.)."""

    @abc.abstractmethod
    def stream_order_events(self) -> AsyncIterator[OrderEvent]:
        """Stream order-state-transition events from the broker.

        The SOLE source-of-truth for order state — never derive state from
        repeated `get_positions()` polls (race conditions + missed events).
        Subclasses should handle reconnect transparently (e.g., IBKR 23:45 CT
        daily disconnect — log gap + resume).

        Note: declared as `def` returning AsyncIterator (not `async def` —
        avoids "called but not awaited" trap). Subclass implementations are
        async generators.
        """

    @abc.abstractmethod
    def is_trade_allowed(
        self,
        contract: ContractSymbol,
        side: Side,
        quantity: int,
    ) -> tuple[bool, str | None]:
        """Pre-trade hard-gate. Returns `(allowed, reason_if_blocked)`.

        Synchronous + no I/O — strategy layer calls this BEFORE every
        `place_order`. Topstep-aware adapters check daily-loss + trailing-DD +
        consistency-rule + max-position. IBKR adapter checks margin headroom.
        MockBroker returns `(True, None)` unless a test fixture sets a blocker.
        """

    @abc.abstractmethod
    def supported_order_types(self) -> set[OrderType]:
        """Set of OrderType values this adapter supports natively.

        Strategy layer queries at startup; strategies with unsupported types
        get SKIPPED on the affected adapter with explicit log. NO silent
        fallbacks (fail-loud policy).
        """


__all__: list[str] = [
    "AccountMetrics",
    "BrokerAdapter",
    "BrokerError",
    "BrokerNotConnectedError",
    "ConsistencyRuleStatus",
    "Duration",
    "Order",
    "OrderEvent",
    "OrderEventType",
    "OrderType",
    "OrderTypeUnsupportedError",
    "Position",
    "Side",
]
