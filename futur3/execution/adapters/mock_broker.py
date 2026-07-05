"""MockBroker — Phase A1.18 backtest + test-harness BrokerAdapter.

Per internal design notes.

## Scope (A1.18 v1)

In-memory, deterministic, no-network broker for:
- pytest fixtures verifying strategy code submits correct orders.
- bit-reproducibility tests (deterministic broker_order_id sequence).
- ReplayDataSource + verifier integration without IB Gateway running.

This v1 implementation:
- Supports ALL 13 OrderType values (simulator has no exchange-side limits).
- Stores submitted orders by broker_order_id + emits "submitted" + "ack"
  events on `place_order`.
- Test-helper `fill_order()` injects fills synthetically — updates positions
  + emits "filled" / "partial_fill" event.
- Tracks SIGNED position quantity (BUY adds, SELL subtracts).
- Maintains synthetic AccountMetrics; can be configured via constructor.
- `stream_order_events()` is a snapshot async generator — yields all currently
  buffered events then returns. (Real-time event streaming with awaitable
  back-pressure is A1.18.b future.)

## Deferred to A1.18.b / A1.19

- **Auto-fill against VerifiedBar time series**: A1.18.b will add an `advance(bar)`
  method that processes outstanding orders against a `VerifiedBar` (limit/stop
  trigger checks + mid-fill of MKT orders).
- **SlippageModel integration**: A1.19 introduces SlippageModel; A1.18.c wires
  it into MockBroker's auto-fill so backtest cost model matches research spec
  (internal design notes).
- **Macro-event blocks + Topstep-style account limits**: not in scope for MockBroker;
  those are TopstepXBrokerAdapter responsibilities (A1.14 + Phase B).

## Contracts

- **No-silent-fallback**: invalid order_type raises OrderTypeUnsupportedError
  even though MockBroker "supports everything" — the check fires when the test
  fixture intentionally restricts the supported set.
- **BACKTEST-IS-LIVE**: same BrokerAdapter interface; engine code that
  works against MockBroker also works against IBKR/TopstepX adapters.
- **Determinism**: broker_order_id is monotonically incremented from 1
  (`mock-000001`, `mock-000002`, ...) so test runs are byte-equal across
  invocations.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal
from typing import Final

from futur3.data.types import ContractSymbol
from futur3.data.verifier import VerifiedBar
from futur3.execution.broker import (
    AccountMetrics,
    BrokerAdapter,
    BrokerError,
    Order,
    OrderEvent,
    OrderEventType,
    OrderType,
    OrderTypeUnsupportedError,
    Position,
    Side,
)
from futur3.execution.risk_manager import multiplier_for
from futur3.execution.slippage import SlippageModel

# Default initial state per internal design notes
# (Combine $50K headroom analysis).
_DEFAULT_STARTING_EQUITY: Final[Decimal] = Decimal("50000")
_DEFAULT_ACCOUNT_ID: Final[str] = "MOCK-ACCT-001"


# ---------------------------------------------------------------------------
# MockBroker
# ---------------------------------------------------------------------------


class MockBroker(BrokerAdapter):
    """In-memory broker for pytest + backtest harness.

    Supports all 13 OrderType values by default. Test fixtures can restrict
    via `restrict_supported_to=` kwarg to verify strategy code handles
    OrderTypeUnsupportedError correctly.
    """

    BROKER_ID: str = "mock"

    def __init__(
        self,
        *,
        starting_equity: Decimal = _DEFAULT_STARTING_EQUITY,
        account_id: str = _DEFAULT_ACCOUNT_ID,
        block_all_trades: bool = False,
        restrict_supported_to: frozenset[OrderType] | None = None,
        clock_now_fn: object | None = None,
    ) -> None:
        if starting_equity <= 0:
            raise ValueError(f"MockBroker.starting_equity must be > 0; got {starting_equity}")
        if not account_id:
            raise ValueError("MockBroker.account_id must be non-empty")
        self._starting_equity = starting_equity
        self._account_id = account_id
        self._block_all_trades = block_all_trades
        self._restrict_supported_to = restrict_supported_to
        self._clock_now_fn = clock_now_fn  # callable returning TZ-aware UTC dt, or None

        # State
        self._orders: dict[str, Order] = {}
        self._cancelled: set[str] = set()
        self._filled_qty: dict[str, int] = {}  # broker_order_id -> cumulative filled
        self._events: list[OrderEvent] = []
        self._positions: dict[ContractSymbol, Position] = {}
        self._realized_pnl_today: Decimal = Decimal("0")
        # Per-trade ledger: `_trade_pnls` is the PnL of each CLOSED economic bet (flat-to-flat),
        # in close order; `_open_trade_pnl` is the realized-so-far tally for each still-open bet
        # (keyed by contract, carried front->back across rolls). See `trade_pnls` + `roll_position`.
        self._trade_pnls: list[Decimal] = []
        self._open_trade_pnl: dict[ContractSymbol, Decimal] = {}
        self._next_order_id_seq: int = 0

    # ----- Sync interface --------------------------------------------------

    @property
    def broker_id(self) -> str:
        return self.BROKER_ID

    @property
    def starting_equity(self) -> Decimal:
        return self._starting_equity

    @property
    def order_count(self) -> int:
        """Submitted orders (NOT counting cancelled)."""
        return len(self._orders) - len(self._cancelled)

    @property
    def events_buffer(self) -> tuple[OrderEvent, ...]:
        """Read-only snapshot of all emitted OrderEvents (test introspection)."""
        return tuple(self._events)

    def __repr__(self) -> str:
        return (
            f"MockBroker(account_id={self._account_id!r}, "
            f"equity={self._starting_equity}, orders={self.order_count}, "
            f"events={len(self._events)})"
        )

    def supported_order_types(self) -> set[OrderType]:
        """Default: all 13 types. Restricted set when `restrict_supported_to=...`."""
        if self._restrict_supported_to is not None:
            return set(self._restrict_supported_to)
        return set(OrderType)

    def is_trade_allowed(
        self,
        contract: ContractSymbol,
        side: Side,
        quantity: int,
    ) -> tuple[bool, str | None]:
        if quantity <= 0:
            return False, f"MockBroker: quantity must be > 0; got {quantity}"
        if self._block_all_trades:
            return False, "MockBroker: block_all_trades=True (test config)"
        return True, None

    # ----- Async interface (in-memory works in test) -----------------------

    async def place_order(self, order: Order) -> str:
        """Assign broker_order_id, store, emit 'submitted' + 'ack' events."""
        if order.order_type not in self.supported_order_types():
            raise OrderTypeUnsupportedError(
                f"MockBroker: {order.order_type.name} not in restricted "
                f"supported_order_types: {sorted(t.name for t in self.supported_order_types())}"
            )
        self._next_order_id_seq += 1
        broker_order_id = f"mock-{self._next_order_id_seq:06d}"
        self._orders[broker_order_id] = order
        self._filled_qty[broker_order_id] = 0
        client_id = order.client_order_id or f"anon-{self._next_order_id_seq:06d}"

        for event_type in ("submitted", "ack"):
            self._emit_event(
                broker_order_id=broker_order_id,
                client_order_id=client_id,
                contract=order.contract,
                event_type=event_type,
                fill_price=None,
                fill_quantity=None,
                cumulative_filled=0,
                remaining_quantity=order.quantity,
            )
        return broker_order_id

    async def cancel_order(self, broker_order_id: str) -> None:
        if broker_order_id not in self._orders:
            raise BrokerError(f"MockBroker: order {broker_order_id!r} not found")
        if broker_order_id in self._cancelled:
            raise BrokerError(f"MockBroker: order {broker_order_id!r} already cancelled")
        order = self._orders[broker_order_id]
        client_id = order.client_order_id or "anon"
        self._cancelled.add(broker_order_id)
        filled = self._filled_qty[broker_order_id]
        self._emit_event(
            broker_order_id=broker_order_id,
            client_order_id=client_id,
            contract=order.contract,
            event_type="cancelled",
            fill_price=None,
            fill_quantity=None,
            cumulative_filled=filled,
            remaining_quantity=max(0, order.quantity - filled),
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
                "MockBroker.modify_order: at least one of new_limit_price / "
                "new_stop_price / new_quantity must be non-None"
            )
        if broker_order_id not in self._orders:
            raise BrokerError(f"MockBroker: order {broker_order_id!r} not found")
        if broker_order_id in self._cancelled:
            raise BrokerError(f"MockBroker: cannot modify cancelled order {broker_order_id!r}")
        order = self._orders[broker_order_id]
        # Build a NEW frozen Order with the updates (Order is frozen).
        new_order = Order(
            contract=order.contract,
            side=order.side,
            quantity=new_quantity if new_quantity is not None else order.quantity,
            order_type=order.order_type,
            limit_price=new_limit_price if new_limit_price is not None else order.limit_price,
            stop_price=new_stop_price if new_stop_price is not None else order.stop_price,
            duration=order.duration,
            parent_order_id=order.parent_order_id,
            oco_group_id=order.oco_group_id,
            client_order_id=order.client_order_id,
        )
        self._orders[broker_order_id] = new_order
        filled = self._filled_qty[broker_order_id]
        client_id = order.client_order_id or "anon"
        self._emit_event(
            broker_order_id=broker_order_id,
            client_order_id=client_id,
            contract=order.contract,
            event_type="modified",
            fill_price=None,
            fill_quantity=None,
            cumulative_filled=filled,
            remaining_quantity=max(0, new_order.quantity - filled),
        )

    async def get_positions(self) -> list[Position]:
        return sorted(
            self._positions.values(),
            key=lambda p: str(p.contract),
        )

    async def get_account_metrics(self) -> AccountMetrics:
        unrealized = sum(
            (p.unrealized_pnl for p in self._positions.values()),
            Decimal("0"),
        )
        margin_used = sum(
            (p.margin_used for p in self._positions.values()),
            Decimal("0"),
        )
        equity = self._starting_equity + self._realized_pnl_today + unrealized
        margin_available = max(Decimal("0"), equity - margin_used)
        leverage_used = margin_used / equity if equity > 0 else Decimal("0")
        return AccountMetrics(
            account_id=self._account_id,
            equity=equity,
            cash=self._starting_equity + self._realized_pnl_today,
            margin_used=margin_used,
            margin_available=margin_available,
            leverage_used=leverage_used,
            realized_pnl_today=self._realized_pnl_today,
            unrealized_pnl=unrealized,
        )

    async def stream_order_events(self) -> AsyncIterator[OrderEvent]:
        """Snapshot stream: yields all currently buffered events then returns.

        Tests can call this multiple times; the same events will be re-yielded
        (read-only snapshot, not a consume-once stream).
        """
        for event in self._events:
            yield event

    # ----- Backtest auto-fill (A1.18.b) ------------------------------------

    @property
    def trade_pnls(self) -> tuple[Decimal, ...]:
        """Realized PnL of each CLOSED economic bet (flat-to-flat), in close order -- the per-trade
        ledger behind win rate / payoff / expectancy (`stats.performance.compute_trade_metrics`).
        Rolls are folded into the bet they continue (not separate trades); a position still OPEN at
        run end is excluded (no realized outcome yet -- its unrealized PnL still marks equity).
        """
        return tuple(self._trade_pnls)

    def advance(
        self,
        bar: VerifiedBar,
        slippage_model: SlippageModel | None = None,
    ) -> list[OrderEvent]:
        """Fill outstanding MKT / LMT / STP orders for `bar.contract` against `bar`'s OHLC.

        Look-ahead-free by construction: an order outstanding BEFORE this bar fills AGAINST
        this bar, never at a price the strategy already observed to decide. The engine loop
        calls `advance(bar)` at the TOP of each iteration (fill prior-bar orders at this bar's
        open), THEN lets the strategy decide on this bar's close — so an order placed on bar t
        fills at bar t+1's open.

        Fill model:
        - MKT: fills at `bar.open` (+ adverse slippage if a model is given).
        - LMT: BUY fills iff `bar.low <= limit_price`; SELL iff `bar.high >= limit_price`; fills
          AT `limit_price` (capped — no slippage).
        - STP: BUY triggers iff `bar.high >= stop_price`; SELL iff `bar.low <= stop_price`; on
          trigger it becomes market and fills AT `stop_price` (+ adverse slippage).
        - Any other OrderType is left OPEN (documented MVP scope; trigger handling for
          STP_LMT / MIT / LIT / TRAIL / MOC / MOO is a later extension).
        Order Duration (DAY/GTC) is not yet enforced (all open orders are eligible).

        Fills the FULL remaining quantity per order, in deterministic order-submission order
       , and delegates to `fill_order` so position/PnL/event logic is shared verbatim.

        Returns the list of emitted fill OrderEvents (empty if nothing filled).
        """
        events: list[OrderEvent] = []
        for broker_order_id in list(self._orders):  # snapshot keys; fill_order mutates state
            if broker_order_id in self._cancelled:
                continue
            order = self._orders[broker_order_id]
            if order.contract != bar.contract:
                continue  # this bar prices a different contract
            if self._filled_qty[broker_order_id] >= order.quantity:
                continue  # already fully filled
            fill_price = self._backtest_fill_price(order, bar)
            if fill_price is None:
                continue  # does not fill against this bar
            if slippage_model is not None and order.order_type in (OrderType.MKT, OrderType.STP):
                fill_price = slippage_model.apply_slippage(
                    fill_price, order.side, order.quantity, order.contract, bar.ts
                )
            events.append(self.fill_order(broker_order_id, fill_price))
        return events

    def _backtest_fill_price(self, order: Order, bar: VerifiedBar) -> Decimal | None:
        """Resolve the executed price for `order` against `bar`, or None if it does not fill.

        LMT/STP price presence is guaranteed by `Order.__post_init__` (broker.py); the asserts
        narrow the Optional for the type checker + document that upstream invariant.
        """
        if order.order_type == OrderType.MKT:
            return bar.open
        if order.order_type == OrderType.LMT:
            assert order.limit_price is not None  # Order.__post_init__ guarantees this for LMT
            crossed = (order.side == Side.BUY and bar.low <= order.limit_price) or (
                order.side == Side.SELL and bar.high >= order.limit_price
            )
            return order.limit_price if crossed else None
        if order.order_type == OrderType.STP:
            assert order.stop_price is not None  # Order.__post_init__ guarantees this for STP
            triggered = (order.side == Side.BUY and bar.high >= order.stop_price) or (
                order.side == Side.SELL and bar.low <= order.stop_price
            )
            return order.stop_price if triggered else None
        return None  # other order types: not auto-filled in this MVP

    def roll_position(
        self,
        front: ContractSymbol,
        back: ContractSymbol,
        front_fill: Decimal,
        back_fill: Decimal,
    ) -> int:
        """Backtest roll: atomically close any net position in `front` at `front_fill`
        (realizing PnL) and re-open the same signed quantity in `back` at `back_fill`. Returns the
        signed quantity rolled (0 if flat in `front`).

        The backtest analog of `RollExecutor.execute_roll` (backtest-is-live: a live roll routes through the
        broker's real calendar-spread / sequential path - W1.7). Reuses `_apply_fill_to_position`
        so the front leg realizes its PnL and the back leg re-establishes the EXACT net exposure at
        the back price - account equity is continuous across the roll (the raw front-vs-back price
        gap becomes a real paired trade, never a phantom mark-to-market jump)."""
        existing = self._positions.get(front)
        if existing is None or existing.quantity == 0:
            return 0
        qty = existing.quantity  # signed: long > 0, short < 0
        close_side = Side.SELL if qty > 0 else Side.BUY
        open_side = Side.BUY if qty > 0 else Side.SELL
        self._apply_fill_to_position(front, close_side, abs(qty), front_fill, is_roll=True)
        self._apply_fill_to_position(back, open_side, abs(qty), back_fill, is_roll=True)
        # The economic bet continues across the roll, so carry the front leg's accumulated
        # trade PnL into the back contract; the back-leg close then books ONE flat-to-flat
        # trade for the whole bet. A roll is not a separate trade and must not skew the win rate.
        carried = self._open_trade_pnl.pop(front, Decimal("0"))
        if carried != Decimal("0"):
            self._open_trade_pnl[back] = self._open_trade_pnl.get(back, Decimal("0")) + carried
        return qty

    # ----- Test helpers ----------------------------------------------------

    def fill_order(
        self,
        broker_order_id: str,
        fill_price: Decimal,
        fill_quantity: int | None = None,
    ) -> OrderEvent:
        """Test-harness: inject a fill against `broker_order_id`.

        If `fill_quantity` is None, fills the full remaining quantity.
        Updates position tracking + realized_pnl + emits 'filled' (full)
        or 'partial_fill' (partial) event. Returns the emitted event.

        Raises:
            BrokerError: order not found, already cancelled, or
                         fill_quantity exceeds remaining.
            ValueError: fill_price <= 0 or fill_quantity <= 0.
        """
        if broker_order_id not in self._orders:
            raise BrokerError(f"MockBroker: order {broker_order_id!r} not found")
        if broker_order_id in self._cancelled:
            raise BrokerError(f"MockBroker: cannot fill cancelled order {broker_order_id!r}")
        if fill_price <= 0:
            raise ValueError(f"fill_price must be > 0; got {fill_price}")

        order = self._orders[broker_order_id]
        already_filled = self._filled_qty[broker_order_id]
        remaining = order.quantity - already_filled
        if remaining <= 0:
            raise BrokerError(
                f"MockBroker: order {broker_order_id!r} fully filled "
                f"({already_filled}/{order.quantity})"
            )

        actual_fill_qty = fill_quantity if fill_quantity is not None else remaining
        if actual_fill_qty <= 0:
            raise ValueError(f"fill_quantity must be > 0; got {actual_fill_qty}")
        if actual_fill_qty > remaining:
            raise BrokerError(
                f"MockBroker: fill_quantity {actual_fill_qty} > remaining {remaining} "
                f"on order {broker_order_id!r}"
            )

        new_filled = already_filled + actual_fill_qty
        self._filled_qty[broker_order_id] = new_filled
        new_remaining = order.quantity - new_filled

        # Update position
        self._apply_fill_to_position(
            contract=order.contract,
            side=order.side,
            quantity=actual_fill_qty,
            fill_price=fill_price,
        )

        # Emit event
        client_id = order.client_order_id or "anon"
        event_type: OrderEventType = "filled" if new_remaining == 0 else "partial_fill"
        return self._emit_event(
            broker_order_id=broker_order_id,
            client_order_id=client_id,
            contract=order.contract,
            event_type=event_type,
            fill_price=fill_price,
            fill_quantity=actual_fill_qty,
            cumulative_filled=new_filled,
            remaining_quantity=new_remaining,
        )

    def reset(self) -> None:
        """Test-harness: clear all state (orders, positions, events) and
        reset broker_order_id sequence. Useful for setup() / teardown()
        in pytest fixtures.
        """
        self._orders.clear()
        self._cancelled.clear()
        self._filled_qty.clear()
        self._events.clear()
        self._positions.clear()
        self._realized_pnl_today = Decimal("0")
        self._trade_pnls.clear()
        self._open_trade_pnl.clear()
        self._next_order_id_seq = 0

    # ----- Private helpers -------------------------------------------------

    def _now(self) -> datetime:
        """TZ-aware UTC timestamp for event emission. Injectable via constructor."""
        if self._clock_now_fn is not None:
            ts: datetime = self._clock_now_fn()  # type: ignore[operator]
            return ts
        return datetime.now(UTC)

    def _emit_event(
        self,
        *,
        broker_order_id: str,
        client_order_id: str,
        contract: ContractSymbol,
        event_type: OrderEventType,
        fill_price: Decimal | None,
        fill_quantity: int | None,
        cumulative_filled: int,
        remaining_quantity: int,
    ) -> OrderEvent:
        ts = self._now()
        payload = (
            f"{broker_order_id}||{event_type}||{fill_price}||{fill_quantity}||{ts.isoformat()}"
        )
        sha = hashlib.sha256(payload.encode()).hexdigest()
        event = OrderEvent(
            broker_order_id=broker_order_id,
            client_order_id=client_order_id,
            contract=contract,
            event_type=event_type,
            fill_price=fill_price,
            fill_quantity=fill_quantity,
            cumulative_filled=cumulative_filled,
            remaining_quantity=remaining_quantity,
            ts_utc=ts,
            raw_broker_payload_sha=sha,
        )
        self._events.append(event)
        return event

    def _apply_fill_to_position(
        self,
        contract: ContractSymbol,
        side: Side,
        quantity: int,
        fill_price: Decimal,
        is_roll: bool = False,
    ) -> None:
        """Update signed position quantity + recompute avg_entry on increases.

        BUY adds, SELL subtracts. Flipping sign or closing realizes PnL.

        `is_roll`: True only for a contract roll's two legs (see `roll_position`). A roll's
        front-leg close realizes PnL but is NOT a discretionary trade, so it is excluded from the
        per-trade win-rate ledger -- its PnL is carried into the back contract's open bet instead.
        """
        delta = quantity if side == Side.BUY else -quantity
        existing = self._positions.get(contract)
        if existing is None or existing.quantity == 0:
            # New position, OR a RE-OPEN from a flat slot. A fully-closed position is retained at
            # quantity=0 / avg_entry_price=0 (to carry the day's realized PnL), so a fill into that
            # flat slot must open fresh here — never fall through to the close/partial-close math
            # below, which would copy the flat avg_entry_price=0 onto a nonzero position and trip
            # the qty!=0 -> price>0 invariant (the bug the 2018+ full-grid C.1 surfaced). Preserve
            # day's realized PnL when re-entering an existing flat contract slot.
            self._positions[contract] = Position(
                contract=contract,
                quantity=delta,
                avg_entry_price=fill_price,
                unrealized_pnl=Decimal("0"),  # at-entry, unrealized is 0
                realized_pnl_today=(
                    existing.realized_pnl_today if existing is not None else Decimal("0")
                ),
                margin_used=Decimal("0"),  # MockBroker doesn't model margin
            )
            return

        old_qty = existing.quantity
        new_qty = old_qty + delta

        # Direction-preserving increase: weighted-avg the entry price.
        # Direction-reversing or partial-close: realize PnL on closed portion.
        if (old_qty > 0 and delta > 0) or (old_qty < 0 and delta < 0):
            # Same direction: WAC entry
            old_notional = existing.avg_entry_price * abs(old_qty)
            new_notional = fill_price * abs(delta)
            new_avg = (old_notional + new_notional) / abs(new_qty)
            self._positions[contract] = Position(
                contract=contract,
                quantity=new_qty,
                avg_entry_price=new_avg,
                unrealized_pnl=Decimal("0"),
                realized_pnl_today=existing.realized_pnl_today,
                margin_used=existing.margin_used,
            )
        else:
            # Opposite direction: realize PnL on the closed portion
            closing_qty = min(abs(old_qty), abs(delta))
            # Realized PnL in ACCOUNT CURRENCY: price move * closing_qty * contract multiplier.
            # The multiplier is load-bearing - without it the equity curve mixes units
            # (realized in price-points vs the engine MTM's dollar unrealized) and jumps spuriously
            # on every close. Matches _mark_to_market + a real broker's dollar accounting.
            mult = multiplier_for(contract)
            if old_qty > 0:
                realized_delta = (fill_price - existing.avg_entry_price) * closing_qty * mult
            else:
                realized_delta = (existing.avg_entry_price - fill_price) * closing_qty * mult
            self._realized_pnl_today += realized_delta
            cumulative_realized = existing.realized_pnl_today + realized_delta
            # Per-trade ledger: accumulate this realization onto the OPEN economic bet in
            # `contract`. The win/loss is BOOKED only when the bet actually ends (full close or
            # flip) -- never on a partial close (the bet continues) and never on a roll's front-leg
            # close (is_roll -> the bet continues in the back contract; roll_position carries it).
            self._open_trade_pnl[contract] = (
                self._open_trade_pnl.get(contract, Decimal("0")) + realized_delta
            )

            if new_qty == 0:
                # Fully closed: the bet ends -> book the trade, UNLESS this is a roll (the bet
                # continues in the back contract; roll_position carries the accumulated PnL there).
                if not is_roll:
                    self._trade_pnls.append(self._open_trade_pnl.pop(contract, Decimal("0")))
                self._positions[contract] = Position(
                    contract=contract,
                    quantity=0,
                    avg_entry_price=Decimal("0"),
                    unrealized_pnl=Decimal("0"),
                    realized_pnl_today=cumulative_realized,
                    margin_used=Decimal("0"),
                )
            elif (old_qty > 0) != (new_qty > 0):
                # Flipped sign: the old bet fully closed -> book it; a new reverse bet opens fresh
                # at fill_price (its future realizations accumulate from zero). A roll never flips.
                self._trade_pnls.append(self._open_trade_pnl.pop(contract, Decimal("0")))
                self._positions[contract] = Position(
                    contract=contract,
                    quantity=new_qty,
                    avg_entry_price=fill_price,
                    unrealized_pnl=Decimal("0"),
                    realized_pnl_today=cumulative_realized,
                    margin_used=existing.margin_used,
                )
            else:
                # Partial close: the bet continues -> PnL stays accumulated, no trade booked yet.
                self._positions[contract] = Position(
                    contract=contract,
                    quantity=new_qty,
                    avg_entry_price=existing.avg_entry_price,
                    unrealized_pnl=Decimal("0"),
                    realized_pnl_today=cumulative_realized,
                    margin_used=existing.margin_used,
                )


__all__: list[str] = [
    "MockBroker",
]
