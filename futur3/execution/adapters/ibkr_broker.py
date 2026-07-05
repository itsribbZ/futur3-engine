"""IBKRBrokerAdapter — Phase A1.13 SHELL.

Per internal design notes + internal design notes.

## Shell scope (A1.13)

This is a **shell-only** implementation. The adapter:
- Exposes the full `BrokerAdapter` interface so engine + strategy code can be
  written against it.
- Declares the IBKR `supported_order_types()` set correctly (11 of 13 — all
  except `JOIN_BID` / `JOIN_ASK` which are TopstepX-native).
- `is_trade_allowed()` returns `(True, None)` as a basic stub — IBKR is not
  Topstep so margin-aware checks come with live wire (A1.13.b).
- **All async live methods raise `IBKRBrokerNotImplemented`** with a clear
  message pointing to A1.13.b for live wiring.

## Live wiring (deferred to A1.13.b)

A1.13.b will wire `ib_async` (v2.0.1) to connect to IB Gateway on port 4002
(paper) / 4001 (live). Per internal design notes:
- 50-200ms residential latency budget.
- 23:45 CT daily auth disconnect; IBC auto-reconnect handles; stream must
  tolerate ~5min reconnect gap.
- STP-PRT exposed via IBKR's protective-stop helper.

A1.13.b is **gated on operator setup**: IB Gateway install + paper account
signup + market-data subscription per
`tests/fixtures/ibkr/INTEGRATION_SMOKE.md` Procedure A.

## Contracts honored even in shell

- **No-silent-fallback**: shell methods raise loudly, not silently.
- **BACKTEST-IS-LIVE**: adapter is constructible without RuntimeContext;
  RuntimeContext can inject it at engine wire-up time without changing the
  interface.
- **Per-step quality bar**: shell + tests triple-green on first
  commit; live wiring is its own ship.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from decimal import Decimal
from typing import Final

from futur3.data.types import ContractSymbol
from futur3.execution.broker import (
    AccountMetrics,
    BrokerAdapter,
    BrokerError,
    Order,
    OrderEvent,
    OrderType,
    Position,
    Side,
)

# IBKR port defaults per internal notes + IB Gateway docs.
# Subclasses / live config can override at construction.
IBKR_PAPER_PORT: Final[int] = 4002
IBKR_LIVE_PORT: Final[int] = 4001
IBKR_DEFAULT_HOST: Final[str] = "127.0.0.1"

# Bounds for custom (non-default) port. 1024 = first non-privileged TCP port;
# 65535 = max valid TCP port.
_MIN_USER_PORT: Final[int] = 1024
_MAX_USER_PORT: Final[int] = 65535

# IBKR `supported_order_types()` — 11 of 13 per internal design notes.
# Excludes `JOIN_BID` / `JOIN_ASK` (TopstepX-native). Includes the full IBKR-supported
# matrix: MKT, LMT, STP, STP_LMT, STP_PRT, MIT, LIT, MOC, MOO, TRAIL, TRAIL_LMT.
_IBKR_SUPPORTED_ORDER_TYPES: Final[frozenset[OrderType]] = frozenset(
    {
        OrderType.MKT,
        OrderType.LMT,
        OrderType.STP,
        OrderType.STP_LMT,
        OrderType.STP_PRT,
        OrderType.MIT,
        OrderType.LIT,
        OrderType.MOC,
        OrderType.MOO,
        OrderType.TRAIL,
        OrderType.TRAIL_LMT,
    }
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class IBKRBrokerNotImplemented(BrokerError):
    """The IBKRBrokerAdapter shell does not implement this method.

    A1.13 ships the shell only; live wiring lands in A1.13.b once operator
    setup is done (IB Gateway install + paper account + market-data
    subscription per `INTEGRATION_SMOKE.md` Procedure A).

    Distinct from `BrokerNotConnectedError` (which is a transient session
    state once live) — this is PERMANENT for shell builds.
    """


# ---------------------------------------------------------------------------
# IBKRBrokerAdapter shell
# ---------------------------------------------------------------------------


class IBKRBrokerAdapter(BrokerAdapter):
    """Phase A1.13 SHELL — interface complete, live methods raise.

    Constructor accepts `host` + `port` for future live wiring; defaults
    target IB Gateway paper on `127.0.0.1:4002`. `client_id` is the standard
    ib_async correlation key; futur3 reserves `1` for the engine's primary
    session per internal design notes.
    """

    BROKER_ID: str = "ibkr_tws"

    def __init__(
        self,
        *,
        host: str = IBKR_DEFAULT_HOST,
        port: int = IBKR_PAPER_PORT,
        client_id: int = 1,
        account_id: str | None = None,
    ) -> None:
        if not host:
            raise ValueError("IBKRBrokerAdapter.host must be non-empty")
        if port not in (IBKR_PAPER_PORT, IBKR_LIVE_PORT) and not (
            _MIN_USER_PORT <= port <= _MAX_USER_PORT
        ):
            raise ValueError(
                f"IBKRBrokerAdapter.port must be {IBKR_PAPER_PORT} (paper), "
                f"{IBKR_LIVE_PORT} (live), or a custom port in "
                f"[{_MIN_USER_PORT}, {_MAX_USER_PORT}]; got {port}"
            )
        if client_id < 0:
            raise ValueError(f"IBKRBrokerAdapter.client_id must be >= 0; got {client_id}")
        self._host = host
        self._port = port
        self._client_id = client_id
        self._account_id = account_id

    # ----- Sync interface (works in shell) -----------------------------

    @property
    def broker_id(self) -> str:
        return self.BROKER_ID

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    @property
    def client_id(self) -> int:
        return self._client_id

    @property
    def is_paper(self) -> bool:
        """True iff configured against IB Gateway paper port (4002)."""
        return self._port == IBKR_PAPER_PORT

    def __repr__(self) -> str:
        return (
            f"IBKRBrokerAdapter("
            f"host={self._host!r}, port={self._port}, "
            f"client_id={self._client_id}, paper={self.is_paper})"
        )

    def supported_order_types(self) -> set[OrderType]:
        """All 11 IBKR-native types per internal design notes."""
        return set(_IBKR_SUPPORTED_ORDER_TYPES)

    def is_trade_allowed(
        self,
        contract: ContractSymbol,
        side: Side,
        quantity: int,
    ) -> tuple[bool, str | None]:
        """Shell stub: returns `(True, None)`. Margin-aware checks come with A1.13.b.

        Validates `quantity > 0` (caller invariant — Order dataclass already enforces).
        """
        if quantity <= 0:
            return (
                False,
                f"is_trade_allowed: quantity must be > 0; got {quantity} (caller invariant)",
            )
        # A1.13.b will add: margin headroom check via get_account_metrics + initial-margin lookup
        return True, None

    # ----- Async interface (shell raises) ------------------------------

    async def place_order(self, order: Order) -> str:
        """Shell: raises `IBKRBrokerNotImplemented`. Live wire A1.13.b."""
        raise IBKRBrokerNotImplemented(
            f"IBKRBrokerAdapter.place_order is shell-only (A1.13). "
            f"Live ib_async wiring deferred to A1.13.b — gated on operator setup "
            f"(IB Gateway install + paper account + market-data subscription). "
            f"Order received: {order.order_type.name} {order.side.name} "
            f"{order.quantity} {order.contract}."
        )

    async def cancel_order(self, broker_order_id: str) -> None:
        raise IBKRBrokerNotImplemented(
            f"IBKRBrokerAdapter.cancel_order is shell-only (A1.13). "
            f"Live wiring A1.13.b. Received cancel for broker_order_id={broker_order_id!r}."
        )

    async def modify_order(
        self,
        broker_order_id: str,
        new_limit_price: Decimal | None = None,
        new_stop_price: Decimal | None = None,
        new_quantity: int | None = None,
    ) -> None:
        if new_limit_price is None and new_stop_price is None and new_quantity is None:
            raise ValueError(
                "IBKRBrokerAdapter.modify_order: at least one of "
                "new_limit_price / new_stop_price / new_quantity must be non-None"
            )
        raise IBKRBrokerNotImplemented(
            f"IBKRBrokerAdapter.modify_order is shell-only (A1.13). "
            f"Live wiring A1.13.b. Received modify for broker_order_id={broker_order_id!r}."
        )

    async def get_positions(self) -> list[Position]:
        raise IBKRBrokerNotImplemented(
            "IBKRBrokerAdapter.get_positions is shell-only (A1.13). "
            "Live ib_async positions wiring A1.13.b."
        )

    async def get_account_metrics(self) -> AccountMetrics:
        raise IBKRBrokerNotImplemented(
            "IBKRBrokerAdapter.get_account_metrics is shell-only (A1.13). "
            "Live ib_async accountSummary wiring A1.13.b."
        )

    async def stream_order_events(self) -> AsyncIterator[OrderEvent]:
        """Shell: raises before any event is yielded.

        Note: declared as async generator (with unreachable `yield`) so the
        function-type matches the ABC's `AsyncIterator[OrderEvent]` return
        type. The `yield` is unreachable at runtime — mypy needs it to
        recognize the function as an async generator.
        """
        raise IBKRBrokerNotImplemented(
            "IBKRBrokerAdapter.stream_order_events is shell-only (A1.13). "
            "Live ib_async event-stream wiring A1.13.b (with 23:45 CT "
            "disconnect-tolerant reconnect logic)."
        )
        # Unreachable; the `yield` is required so Python treats this as an
        # async generator (matching the ABC's AsyncIterator[OrderEvent] return).
        yield  # pragma: no cover


__all__: list[str] = [
    "IBKR_DEFAULT_HOST",
    "IBKR_LIVE_PORT",
    "IBKR_PAPER_PORT",
    "IBKRBrokerAdapter",
    "IBKRBrokerNotImplemented",
]
