"""TopstepXBrokerAdapter — Phase A1.14 STUB (paper-first pivot honored).

Per internal design notes + internal design notes.

## STUB scope (A1.14)

This is a **STUB** implementation per the paper-first plan. The adapter:
- Exposes the `BrokerAdapter` interface so engine + strategy code can be written
  against it.
- Declares the TopstepX `supported_order_types()` set correctly (7 ProjectX
  types only — MKT/LMT/STP/STP_LMT/TRAIL/JOIN_BID/JOIN_ASK).
- Exposes the `downgrade_order_type()` helper that implements the downgrade
  matrix loudly (STP_PRT->STP + HALT-LOW log; MIT->STP + HALT-MEDIUM log;
  LIT->LMT + HALT-MEDIUM log; MOC/MOO/TRAIL_LMT->raise OrderTypeUnsupportedError
  with HALT-HIGH log) (diagnostics are features by design).
- `is_trade_allowed()` returns `(True, None)` as a STUB — 3-state MLL trailing
  logic (Combine intraday / XFA EOD / LFA static $0) lands with full integration.
- **All async live methods raise `TopstepXBrokerNotImplemented`** with explicit
  "STUB-only per paper-first pivot; full integration Phase B post-Combine"
  messages.

## Full integration (deferred to Phase B)

Phase B will wire ProjectX REST + WebSocket:
- REST: POST `/orders`, GET `/positions`, GET `/account/metrics`.
- WebSocket: `/orders` + `/positions` topics with exponential-backoff reconnect.
- 3-state MLL trailing (Combine / XFA / LFA) per internal design notes.
- Macro-event embargo windows (NFP/FOMC/CPI) — planned enhancement.
- Calendar-spread / BAG combo support (planned).

Phase B is **gated on**:
1. The operator signs up for a Topstep Combine + funds it.
2. Broker-side ticket resolved (BAG combo support + macro embargo windows).
3. Phase A1 paper trading shows positive Sharpe.

## Contracts honored even in STUB

- **No-silent-fallback**: STUB raises loudly; downgrade matrix logs every
  HALT-FLAG event for audit trail.
- **BACKTEST-IS-LIVE**: STUB is constructible without RuntimeContext;
  injection at engine wire-up time.
- **Diagnostics are features**: downgrade logging is a feature, not optional — present
  even in STUB so audit trail starts day one.
- **Topstep-respect**: per-state MLL logic + consistency rule documented
  in the code even though live impl deferred.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from decimal import Decimal
from typing import Final, Literal

from futur3.data.types import ContractSymbol
from futur3.execution.broker import (
    AccountMetrics,
    BrokerAdapter,
    BrokerError,
    Order,
    OrderEvent,
    OrderType,
    OrderTypeUnsupportedError,
    Position,
    Side,
)

logger = logging.getLogger(__name__)

# Topstep account state — controls MLL trailing semantics
TopstepAccountState = Literal["Combine", "XFA", "LFA"]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# 7 ProjectX-native order types per internal design notes
# + internal design notes. Strategy layer queries this
# at startup; strategies needing types in the REJECT tier are SKIPPED.
_TOPSTEPX_NATIVE_ORDER_TYPES: Final[frozenset[OrderType]] = frozenset(
    {
        OrderType.MKT,
        OrderType.LMT,
        OrderType.STP,
        OrderType.STP_LMT,
        OrderType.TRAIL,
        OrderType.JOIN_BID,
        OrderType.JOIN_ASK,
    }
)


HaltSeverity = Literal["LOW", "MEDIUM", "HIGH"]


@dataclass(frozen=True)
class TopstepXDowngradeRule:
    """Single row of the downgrade matrix.

    `target_order_type=None` means REJECT (raise OrderTypeUnsupportedError).
    Otherwise, the adapter rewrites the order to `target_order_type` before
    submission and logs a HALT-FLAG event at `severity` for audit trail.
    """

    target_order_type: OrderType | None
    severity: HaltSeverity
    reason: str


# Downgrade matrix per internal design notes. Native types
# (in _TOPSTEPX_NATIVE_ORDER_TYPES) are passthrough — they're NOT in this dict.
# Anything not in `native | downgrade` lookups raises OrderTypeUnsupportedError
# (defensive: future OrderType additions must explicitly opt in).
_DOWNGRADE_MATRIX: Final[dict[OrderType, TopstepXDowngradeRule]] = {
    OrderType.STP_PRT: TopstepXDowngradeRule(
        target_order_type=OrderType.STP,
        severity="LOW",
        reason=(
            "TopstepX has no STP_PRT (CME-native protective stop); "
            "downgrade to STP - futur3 default safety stop affected"
        ),
    ),
    OrderType.MIT: TopstepXDowngradeRule(
        target_order_type=OrderType.STP,
        severity="MEDIUM",
        reason=(
            "TopstepX has no MIT (market-if-touched); simulate via STP "
            "with price watcher - semantic gap: touch vs cross"
        ),
    ),
    OrderType.LIT: TopstepXDowngradeRule(
        target_order_type=OrderType.LMT,
        severity="MEDIUM",
        reason=(
            "TopstepX has no LIT (limit-if-touched); simulate via LMT "
            "with price watcher - semantic gap: touch vs cross"
        ),
    ),
    OrderType.MOC: TopstepXDowngradeRule(
        target_order_type=None,
        severity="HIGH",
        reason=(
            "TopstepX REJECTS MOC (market-on-close); MOC-dependent "
            "strategies cannot run on TopstepX funded path"
        ),
    ),
    OrderType.MOO: TopstepXDowngradeRule(
        target_order_type=None,
        severity="HIGH",
        reason=(
            "TopstepX REJECTS MOO (market-on-open); gap-fill strategies cannot run on TopstepX"
        ),
    ),
    OrderType.TRAIL_LMT: TopstepXDowngradeRule(
        target_order_type=None,
        severity="HIGH",
        reason="TopstepX REJECTS TRAIL_LMT (trailing-stop-limit)",
    ),
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class TopstepXBrokerNotImplemented(BrokerError):
    """The TopstepXBrokerAdapter STUB does not implement this method.

    A1.14 ships STUB only per the paper-first plan. Full integration
    lands Phase B (post-Combine pass). See module docstring for the gating
    conditions.
    """


# ---------------------------------------------------------------------------
# TopstepXBrokerAdapter STUB
# ---------------------------------------------------------------------------


class TopstepXBrokerAdapter(BrokerAdapter):
    """Phase A1.14 STUB - interface complete, live methods raise.

    Constructor accepts the operator-local-only config; per internal notes the
    TopstepX local-execution invariant (per Topstep T&C) is enforced by
    using this adapter from the operator's local machine ONLY - never VPS/cloud/co-lo.

    `account_state` is required at construction to wire the correct MLL trailing
    semantics once live integration ships. STUB defaults to `Combine` (most
    restrictive - intraday-trailing).
    """

    BROKER_ID: str = "topstepx"

    def __init__(
        self,
        *,
        account_id: str | None = None,
        account_state: TopstepAccountState = "Combine",
        api_key_env_var: str = "TOPSTEPX_API_KEY",
    ) -> None:
        if account_state not in ("Combine", "XFA", "LFA"):
            raise ValueError(
                f"TopstepXBrokerAdapter.account_state must be one of "
                f"Combine|XFA|LFA; got {account_state!r}"
            )
        if not api_key_env_var:
            raise ValueError("TopstepXBrokerAdapter.api_key_env_var must be non-empty")
        self._account_id = account_id
        self._account_state: TopstepAccountState = account_state
        self._api_key_env_var = api_key_env_var

    # ----- Sync interface (works in STUB) ------------------------------

    @property
    def broker_id(self) -> str:
        return self.BROKER_ID

    @property
    def account_state(self) -> TopstepAccountState:
        return self._account_state

    def __repr__(self) -> str:
        return (
            f"TopstepXBrokerAdapter("
            f"account_state={self._account_state!r}, "
            f"account_id={self._account_id!r})"
        )

    def supported_order_types(self) -> set[OrderType]:
        """7 ProjectX-native types. Strategies needing REJECT-tier types
        (MOC/MOO/TRAIL_LMT) are SKIPPED at engine startup.
        """
        return set(_TOPSTEPX_NATIVE_ORDER_TYPES)

    def downgrade_order_type(self, requested: OrderType) -> OrderType:
        """Apply the TopstepX downgrade matrix to `requested`.

        Returns the OrderType to actually submit to TopstepX. Logs a HALT-FLAG
        event at the appropriate severity (LOW/MEDIUM/HIGH) for audit trail.
        Native types passthrough unchanged.

        Raises:
            OrderTypeUnsupportedError: REJECT-tier types (MOC/MOO/TRAIL_LMT)
                + any OrderType not in native|downgrade lookup (defensive
                for future enum additions).
        """
        if requested in _TOPSTEPX_NATIVE_ORDER_TYPES:
            return requested
        rule = _DOWNGRADE_MATRIX.get(requested)
        if rule is None:
            raise OrderTypeUnsupportedError(
                f"TopstepX: {requested.name} is not native and has no documented "
                f"downgrade rule. Defensive REJECT per the fail-loud policy (no silent fallback)."
            )
        if rule.target_order_type is None:
            logger.warning(
                "TopstepX REJECT %s (severity=%s): %s",
                requested.name,
                rule.severity,
                rule.reason,
            )
            raise OrderTypeUnsupportedError(
                f"TopstepX REJECT {requested.name} ({rule.severity}): {rule.reason}"
            )
        logger.info(
            "TopstepX downgrade %s -> %s (severity=%s): %s",
            requested.name,
            rule.target_order_type.name,
            rule.severity,
            rule.reason,
        )
        return rule.target_order_type

    def is_trade_allowed(
        self,
        contract: ContractSymbol,
        side: Side,
        quantity: int,
    ) -> tuple[bool, str | None]:
        """STUB: returns `(True, None)`. Per-state MLL logic lands Phase B.

        Real impl will dispatch on `self.account_state` per internal notes:
        - Combine: intraday-trailing MLL check (realized + unrealized + risk)
        - XFA: EOD-trailing MLL check (realized only during session)
        - LFA: static $0 floor (equity - risk >= initial_funded)

        Plus consistency_rule status + position_size limit + macro embargo.
        """
        if quantity <= 0:
            return (
                False,
                f"is_trade_allowed: quantity must be > 0; got {quantity}",
            )
        # Phase B will add: MLL trailing check per account_state +
        # consistency_rule_status check + max_position_size check + macro embargo.
        return True, None

    # ----- Async interface (STUB raises) -------------------------------

    async def place_order(self, order: Order) -> str:
        """STUB: raises `TopstepXBrokerNotImplemented`. Phase B live wire.

        Phase B impl will: (1) call `downgrade_order_type(order.order_type)` -
        possibly raising OrderTypeUnsupportedError for REJECT-tier; (2) submit
        downgraded order via ProjectX REST `/orders`; (3) return broker_order_id.
        """
        raise TopstepXBrokerNotImplemented(
            f"TopstepXBrokerAdapter.place_order is STUB-only (A1.14). "
            f"Full integration Phase B (post-Combine). "
            f"Order received: {order.order_type.name} {order.side.name} "
            f"{order.quantity} {order.contract}."
        )

    async def cancel_order(self, broker_order_id: str) -> None:
        raise TopstepXBrokerNotImplemented(
            f"TopstepXBrokerAdapter.cancel_order is STUB-only (A1.14). "
            f"Phase B. Received cancel for broker_order_id={broker_order_id!r}."
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
                "TopstepXBrokerAdapter.modify_order: at least one of "
                "new_limit_price / new_stop_price / new_quantity must be non-None"
            )
        raise TopstepXBrokerNotImplemented(
            f"TopstepXBrokerAdapter.modify_order is STUB-only (A1.14). "
            f"Phase B. Received modify for broker_order_id={broker_order_id!r}."
        )

    async def get_positions(self) -> list[Position]:
        raise TopstepXBrokerNotImplemented(
            "TopstepXBrokerAdapter.get_positions is STUB-only (A1.14). "
            "Phase B will wire ProjectX REST `/positions`."
        )

    async def get_account_metrics(self) -> AccountMetrics:
        raise TopstepXBrokerNotImplemented(
            "TopstepXBrokerAdapter.get_account_metrics is STUB-only (A1.14). "
            "Phase B will wire ProjectX REST `/account/metrics` + populate "
            "Topstep-specific fields (trailing_drawdown_remaining + "
            "consistency_rule_status + max_position_size_allowed)."
        )

    async def stream_order_events(self) -> AsyncIterator[OrderEvent]:
        """STUB: raises before any event is yielded.

        Phase B will subscribe to ProjectX WS `/orders` + `/positions` topics
        with exponential-backoff reconnect + bit-repro raw_broker_payload_sha.
        """
        raise TopstepXBrokerNotImplemented(
            "TopstepXBrokerAdapter.stream_order_events is STUB-only (A1.14). "
            "Phase B will wire ProjectX WebSocket + reconnect logic."
        )
        # Unreachable; required so Python treats this as an async generator
        # matching the ABC's AsyncIterator[OrderEvent] return type.
        yield  # pragma: no cover


__all__: list[str] = [
    "TopstepAccountState",
    "TopstepXBrokerAdapter",
    "TopstepXBrokerNotImplemented",
    "TopstepXDowngradeRule",
]
