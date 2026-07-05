"""A1.18 MockBroker test suite.

Test discipline:
- Fixture-only (no network, no IB Gateway, no TopstepX session).
- Verify the broker adapter shape end-to-end: construction, order lifecycle,
  position tracking (signed quantity + WAC entry + realized PnL on closes),
  event emission, account metrics aggregation.

References:
- `futur3/execution/adapters/mock_broker.py` (implementation)
- internal design notes (MockBroker spec)
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from futur3.data.types import ContractSymbol
from futur3.execution.adapters.mock_broker import MockBroker
from futur3.execution.broker import (
    BrokerAdapter,
    BrokerError,
    Order,
    OrderType,
    OrderTypeUnsupportedError,
    Side,
)

# ============================================================================
# Helpers
# ============================================================================


def _mkt_order(
    *,
    contract: str = "ESM26",
    side: Side = Side.BUY,
    quantity: int = 1,
    client_order_id: str | None = None,
) -> Order:
    return Order(
        contract=ContractSymbol(contract),
        side=side,
        quantity=quantity,
        order_type=OrderType.MKT,
        client_order_id=client_order_id,
    )


def _lmt_order(
    *,
    limit_price: str = "5260.00",
    side: Side = Side.BUY,
    quantity: int = 1,
) -> Order:
    return Order(
        contract=ContractSymbol("ESM26"),
        side=side,
        quantity=quantity,
        order_type=OrderType.LMT,
        limit_price=Decimal(limit_price),
    )


# ============================================================================
# TestA1_18_Construction
# ============================================================================


class TestA1_18_Construction:
    def test_default_construction(self) -> None:
        b = MockBroker()
        assert b.broker_id == "mock"
        assert b.starting_equity == Decimal("50000")
        assert b.order_count == 0
        assert b.events_buffer == ()

    def test_custom_equity(self) -> None:
        b = MockBroker(starting_equity=Decimal("100000"))
        assert b.starting_equity == Decimal("100000")

    def test_zero_equity_raises(self) -> None:
        with pytest.raises(ValueError, match="starting_equity must be > 0"):
            MockBroker(starting_equity=Decimal("0"))

    def test_negative_equity_raises(self) -> None:
        with pytest.raises(ValueError, match="starting_equity must be > 0"):
            MockBroker(starting_equity=Decimal("-1000"))

    def test_empty_account_id_raises(self) -> None:
        with pytest.raises(ValueError, match="account_id must be non-empty"):
            MockBroker(account_id="")

    def test_is_subclass_of_broker_adapter(self) -> None:
        assert issubclass(MockBroker, BrokerAdapter)

    def test_repr(self) -> None:
        b = MockBroker()
        rep = repr(b)
        assert "MOCK-ACCT-001" in rep


# ============================================================================
# TestA1_18_SupportedOrderTypes
# ============================================================================


class TestA1_18_SupportedOrderTypes:
    def test_default_supports_all_13(self) -> None:
        """MockBroker default supports every OrderType — sim has no limits."""
        b = MockBroker()
        assert len(b.supported_order_types()) == 13

    def test_restricted_subset(self) -> None:
        b = MockBroker(restrict_supported_to=frozenset({OrderType.MKT, OrderType.LMT}))
        assert b.supported_order_types() == {OrderType.MKT, OrderType.LMT}

    def test_returns_copy_not_reference(self) -> None:
        """Caller mutation must NOT affect adapter."""
        b = MockBroker()
        s = b.supported_order_types()
        s.clear()
        assert len(b.supported_order_types()) == 13


# ============================================================================
# TestA1_18_IsTradeAllowed
# ============================================================================


class TestA1_18_IsTradeAllowed:
    def test_basic_allow(self) -> None:
        b = MockBroker()
        ok, reason = b.is_trade_allowed(ContractSymbol("ESM26"), Side.BUY, 1)
        assert ok is True
        assert reason is None

    def test_zero_quantity_blocked(self) -> None:
        b = MockBroker()
        ok, reason = b.is_trade_allowed(ContractSymbol("ESM26"), Side.BUY, 0)
        assert ok is False
        assert reason is not None

    def test_block_all_trades_config(self) -> None:
        """Test fixture: globally block all trades."""
        b = MockBroker(block_all_trades=True)
        ok, reason = b.is_trade_allowed(ContractSymbol("ESM26"), Side.BUY, 1)
        assert ok is False
        assert reason is not None
        assert "block_all_trades" in reason


# ============================================================================
# TestA1_18_PlaceOrder
# ============================================================================


class TestA1_18_PlaceOrder:
    def test_place_market_order_returns_broker_order_id(self) -> None:
        b = MockBroker()
        broker_order_id = asyncio.run(b.place_order(_mkt_order()))
        assert broker_order_id.startswith("mock-")
        assert b.order_count == 1

    def test_sequential_order_ids_deterministic(self) -> None:
        """Determinism: broker_order_id sequence is byte-equal across MockBroker instances."""
        b1 = MockBroker()
        b2 = MockBroker()
        id1 = asyncio.run(b1.place_order(_mkt_order()))
        id2 = asyncio.run(b2.place_order(_mkt_order()))
        assert id1 == id2 == "mock-000001"

    def test_two_orders_increment_ids(self) -> None:
        b = MockBroker()
        id1 = asyncio.run(b.place_order(_mkt_order()))
        id2 = asyncio.run(b.place_order(_mkt_order()))
        assert id1 == "mock-000001"
        assert id2 == "mock-000002"

    def test_submitted_and_ack_events_emitted(self) -> None:
        """Each place_order emits submitted + ack events."""
        b = MockBroker()
        asyncio.run(b.place_order(_mkt_order()))
        types = [e.event_type for e in b.events_buffer]
        assert types == ["submitted", "ack"]

    def test_client_order_id_propagated(self) -> None:
        b = MockBroker()
        asyncio.run(b.place_order(_mkt_order(client_order_id="strat_x_001")))
        assert b.events_buffer[0].client_order_id == "strat_x_001"

    def test_anon_client_order_id_assigned_when_none(self) -> None:
        b = MockBroker()
        asyncio.run(b.place_order(_mkt_order()))
        assert b.events_buffer[0].client_order_id.startswith("anon-")

    def test_unsupported_order_type_raises(self) -> None:
        """Restricted supported set: requesting outside-set raises."""
        b = MockBroker(restrict_supported_to=frozenset({OrderType.MKT}))
        with pytest.raises(OrderTypeUnsupportedError, match="LMT"):
            asyncio.run(b.place_order(_lmt_order()))


# ============================================================================
# TestA1_18_CancelOrder
# ============================================================================


class TestA1_18_CancelOrder:
    def test_cancel_emits_cancelled_event(self) -> None:
        b = MockBroker()
        oid = asyncio.run(b.place_order(_mkt_order()))
        asyncio.run(b.cancel_order(oid))
        assert b.events_buffer[-1].event_type == "cancelled"

    def test_cancel_reduces_order_count(self) -> None:
        b = MockBroker()
        oid = asyncio.run(b.place_order(_mkt_order()))
        assert b.order_count == 1
        asyncio.run(b.cancel_order(oid))
        assert b.order_count == 0

    def test_cancel_unknown_raises(self) -> None:
        b = MockBroker()
        with pytest.raises(BrokerError, match="not found"):
            asyncio.run(b.cancel_order("mock-999999"))

    def test_cancel_twice_raises(self) -> None:
        b = MockBroker()
        oid = asyncio.run(b.place_order(_mkt_order()))
        asyncio.run(b.cancel_order(oid))
        with pytest.raises(BrokerError, match="already cancelled"):
            asyncio.run(b.cancel_order(oid))


# ============================================================================
# TestA1_18_ModifyOrder
# ============================================================================


class TestA1_18_ModifyOrder:
    def test_modify_limit_price(self) -> None:
        b = MockBroker()
        oid = asyncio.run(b.place_order(_lmt_order(limit_price="5260.00")))
        asyncio.run(b.modify_order(oid, new_limit_price=Decimal("5265.00")))
        assert b.events_buffer[-1].event_type == "modified"

    def test_modify_quantity(self) -> None:
        b = MockBroker()
        oid = asyncio.run(b.place_order(_mkt_order(quantity=1)))
        asyncio.run(b.modify_order(oid, new_quantity=3))
        # Last event is modified; remaining_quantity reflects new size
        last = b.events_buffer[-1]
        assert last.event_type == "modified"
        assert last.remaining_quantity == 3

    def test_modify_all_none_raises(self) -> None:
        b = MockBroker()
        oid = asyncio.run(b.place_order(_mkt_order()))
        with pytest.raises(ValueError, match="must be non-None"):
            asyncio.run(b.modify_order(oid))

    def test_modify_unknown_raises(self) -> None:
        b = MockBroker()
        with pytest.raises(BrokerError, match="not found"):
            asyncio.run(b.modify_order("mock-999999", new_quantity=2))

    def test_modify_cancelled_raises(self) -> None:
        b = MockBroker()
        oid = asyncio.run(b.place_order(_mkt_order()))
        asyncio.run(b.cancel_order(oid))
        with pytest.raises(BrokerError, match="cannot modify cancelled"):
            asyncio.run(b.modify_order(oid, new_quantity=2))


# ============================================================================
# TestA1_18_FillOrder — test-harness fill injection + position arithmetic
# ============================================================================


class TestA1_18_FillOrder:
    def test_full_fill_emits_filled_event(self) -> None:
        b = MockBroker()
        oid = asyncio.run(b.place_order(_mkt_order(quantity=1)))
        b.fill_order(oid, Decimal("5260.00"))
        last = b.events_buffer[-1]
        assert last.event_type == "filled"
        assert last.cumulative_filled == 1
        assert last.remaining_quantity == 0
        assert last.fill_price == Decimal("5260.00")

    def test_partial_fill_emits_partial_fill_event(self) -> None:
        b = MockBroker()
        oid = asyncio.run(b.place_order(_mkt_order(quantity=5)))
        b.fill_order(oid, Decimal("5260.00"), fill_quantity=2)
        last = b.events_buffer[-1]
        assert last.event_type == "partial_fill"
        assert last.cumulative_filled == 2
        assert last.remaining_quantity == 3

    def test_two_partials_complete(self) -> None:
        b = MockBroker()
        oid = asyncio.run(b.place_order(_mkt_order(quantity=5)))
        b.fill_order(oid, Decimal("5260.00"), fill_quantity=3)
        b.fill_order(oid, Decimal("5261.00"), fill_quantity=2)
        last = b.events_buffer[-1]
        assert last.event_type == "filled"
        assert last.cumulative_filled == 5
        assert last.remaining_quantity == 0

    def test_overfill_raises(self) -> None:
        b = MockBroker()
        oid = asyncio.run(b.place_order(_mkt_order(quantity=2)))
        with pytest.raises(BrokerError, match="fill_quantity 3 > remaining 2"):
            b.fill_order(oid, Decimal("5260.00"), fill_quantity=3)

    def test_zero_fill_price_raises(self) -> None:
        b = MockBroker()
        oid = asyncio.run(b.place_order(_mkt_order()))
        with pytest.raises(ValueError, match="fill_price must be > 0"):
            b.fill_order(oid, Decimal("0"))

    def test_zero_fill_quantity_raises(self) -> None:
        b = MockBroker()
        oid = asyncio.run(b.place_order(_mkt_order(quantity=1)))
        with pytest.raises(ValueError, match="fill_quantity must be > 0"):
            b.fill_order(oid, Decimal("5260.00"), fill_quantity=0)

    def test_fill_cancelled_raises(self) -> None:
        b = MockBroker()
        oid = asyncio.run(b.place_order(_mkt_order()))
        asyncio.run(b.cancel_order(oid))
        with pytest.raises(BrokerError, match="cannot fill cancelled"):
            b.fill_order(oid, Decimal("5260.00"))

    def test_fill_unknown_raises(self) -> None:
        b = MockBroker()
        with pytest.raises(BrokerError, match="not found"):
            b.fill_order("mock-999999", Decimal("5260.00"))


# ============================================================================
# TestA1_18_Positions — signed quantity + WAC entry + realized PnL
# ============================================================================


class TestA1_18_Positions:
    def test_buy_creates_long_position(self) -> None:
        b = MockBroker()
        oid = asyncio.run(b.place_order(_mkt_order(quantity=2)))
        b.fill_order(oid, Decimal("5260.00"))
        positions = asyncio.run(b.get_positions())
        assert len(positions) == 1
        assert positions[0].quantity == 2
        assert positions[0].avg_entry_price == Decimal("5260.00")

    def test_sell_creates_short_position(self) -> None:
        b = MockBroker()
        oid = asyncio.run(b.place_order(_mkt_order(side=Side.SELL, quantity=2)))
        b.fill_order(oid, Decimal("5260.00"))
        positions = asyncio.run(b.get_positions())
        assert positions[0].quantity == -2

    def test_two_buys_wac_entry(self) -> None:
        """Same-direction adds: weighted-avg-cost entry price."""
        b = MockBroker()
        oid1 = asyncio.run(b.place_order(_mkt_order(quantity=2)))
        b.fill_order(oid1, Decimal("5260.00"))
        oid2 = asyncio.run(b.place_order(_mkt_order(quantity=2)))
        b.fill_order(oid2, Decimal("5280.00"))
        positions = asyncio.run(b.get_positions())
        # WAC: (2*5260 + 2*5280) / 4 = 5270
        assert positions[0].quantity == 4
        assert positions[0].avg_entry_price == Decimal("5270.00")

    def test_buy_then_sell_realizes_pnl(self) -> None:
        """Long 1 @ 5260, sell 1 @ 5265 -> +5 pts x ES mult 50 = +250 realized, flat."""
        b = MockBroker()
        oid1 = asyncio.run(b.place_order(_mkt_order(quantity=1)))
        b.fill_order(oid1, Decimal("5260.00"))
        oid2 = asyncio.run(b.place_order(_mkt_order(side=Side.SELL, quantity=1)))
        b.fill_order(oid2, Decimal("5265.00"))

        positions = asyncio.run(b.get_positions())
        assert positions[0].quantity == 0
        assert positions[0].realized_pnl_today == Decimal("250.00")  # +5 pts x ES mult 50

    def test_buy_then_sell_with_loss(self) -> None:
        b = MockBroker()
        oid1 = asyncio.run(b.place_order(_mkt_order(quantity=1)))
        b.fill_order(oid1, Decimal("5260.00"))
        oid2 = asyncio.run(b.place_order(_mkt_order(side=Side.SELL, quantity=1)))
        b.fill_order(oid2, Decimal("5255.00"))

        positions = asyncio.run(b.get_positions())
        assert positions[0].realized_pnl_today == Decimal("-250.00")  # -5 pts x ES mult 50

    def test_partial_close_keeps_entry_price(self) -> None:
        """Long 4 @ 5260; sell 2 @ 5270 -> 20 pts x 50 = 1000 realized, position 2 @ 5260."""
        b = MockBroker()
        oid1 = asyncio.run(b.place_order(_mkt_order(quantity=4)))
        b.fill_order(oid1, Decimal("5260.00"))
        oid2 = asyncio.run(b.place_order(_mkt_order(side=Side.SELL, quantity=2)))
        b.fill_order(oid2, Decimal("5270.00"))

        positions = asyncio.run(b.get_positions())
        assert positions[0].quantity == 2
        assert positions[0].avg_entry_price == Decimal("5260.00")  # unchanged
        assert positions[0].realized_pnl_today == Decimal("1000.00")  # 20 pts x ES mult 50

    def test_flip_long_to_short_new_entry(self) -> None:
        """Long 1 @ 5260; sell 3 @ 5270 -> 10 pts x 50 = 500 realized + short 2 @ 5270."""
        b = MockBroker()
        oid1 = asyncio.run(b.place_order(_mkt_order(quantity=1)))
        b.fill_order(oid1, Decimal("5260.00"))
        oid2 = asyncio.run(b.place_order(_mkt_order(side=Side.SELL, quantity=3)))
        b.fill_order(oid2, Decimal("5270.00"))

        positions = asyncio.run(b.get_positions())
        assert positions[0].quantity == -2
        assert positions[0].avg_entry_price == Decimal("5270.00")
        assert positions[0].realized_pnl_today == Decimal("500.00")  # 10 pts x ES mult 50


# ============================================================================
# TestA1_18_AccountMetrics
# ============================================================================


class TestA1_18_AccountMetrics:
    def test_initial_metrics(self) -> None:
        b = MockBroker(starting_equity=Decimal("50000"))
        m = asyncio.run(b.get_account_metrics())
        assert m.equity == Decimal("50000")
        assert m.cash == Decimal("50000")
        assert m.realized_pnl_today == Decimal("0")
        assert m.unrealized_pnl == Decimal("0")
        assert m.leverage_used == Decimal("0")

    def test_realized_pnl_aggregates(self) -> None:
        b = MockBroker(starting_equity=Decimal("50000"))
        oid1 = asyncio.run(b.place_order(_mkt_order(quantity=1)))
        b.fill_order(oid1, Decimal("5260.00"))
        oid2 = asyncio.run(b.place_order(_mkt_order(side=Side.SELL, quantity=1)))
        b.fill_order(oid2, Decimal("5265.00"))
        m = asyncio.run(b.get_account_metrics())
        assert m.realized_pnl_today == Decimal("250.00")  # +5 pts x ES mult 50
        assert m.cash == Decimal("50250.00")
        # No open position -> equity = cash
        assert m.equity == Decimal("50250.00")

    def test_topstep_fields_none_on_mock(self) -> None:
        """MockBroker doesn't populate Topstep-specific fields."""
        b = MockBroker()
        m = asyncio.run(b.get_account_metrics())
        assert m.trailing_drawdown_remaining is None
        assert m.consistency_rule_status is None
        assert m.max_position_size_allowed is None

    def test_account_id_propagated(self) -> None:
        b = MockBroker(account_id="DU-TEST-99")
        m = asyncio.run(b.get_account_metrics())
        assert m.account_id == "DU-TEST-99"


# ============================================================================
# TestA1_18_StreamOrderEvents
# ============================================================================


class TestA1_18_StreamOrderEvents:
    def test_stream_yields_all_events(self) -> None:
        b = MockBroker()
        oid = asyncio.run(b.place_order(_mkt_order()))
        b.fill_order(oid, Decimal("5260.00"))

        async def collect() -> list[str]:
            events = []
            async for e in b.stream_order_events():
                events.append(e.event_type)
            return events

        types = asyncio.run(collect())
        assert types == ["submitted", "ack", "filled"]

    def test_stream_can_iterate_twice(self) -> None:
        """Snapshot semantics: re-iterating yields the same events."""
        b = MockBroker()
        asyncio.run(b.place_order(_mkt_order()))

        async def collect() -> int:
            n = 0
            async for _ in b.stream_order_events():
                n += 1
            return n

        first = asyncio.run(collect())
        second = asyncio.run(collect())
        assert first == second == 2


# ============================================================================
# TestA1_18_Reset
# ============================================================================


class TestA1_18_Reset:
    def test_reset_clears_state(self) -> None:
        b = MockBroker()
        oid = asyncio.run(b.place_order(_mkt_order()))
        b.fill_order(oid, Decimal("5260.00"))
        assert b.order_count == 1
        assert len(b.events_buffer) > 0

        b.reset()
        assert b.order_count == 0
        assert b.events_buffer == ()
        # broker_order_id sequence also reset
        new_oid = asyncio.run(b.place_order(_mkt_order()))
        assert new_oid == "mock-000001"


# ============================================================================
# TestA1_18_ClockInjection
# ============================================================================


class TestA1_18_ClockInjection:
    def test_clock_fn_used_for_event_ts(self) -> None:
        """Injectable clock fn for deterministic ts_utc in tests."""
        fixed = datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC)
        b = MockBroker(clock_now_fn=lambda: fixed)
        asyncio.run(b.place_order(_mkt_order()))
        assert b.events_buffer[0].ts_utc == fixed
