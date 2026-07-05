"""A1.18.b MockBroker.advance() backtest auto-fill test suite.

Covers the look-ahead-free fill model: MKT fills at bar.open (+ adverse slippage); LMT fills at its
capped limit price on a range-cross (no slippage); STP triggers on a range-cross and fills at the
stop price (+ adverse slippage); non-MKT/LMT/STP types + cancelled + already-filled + un-triggered
orders are left alone; fills are deterministic (submission order) and update position/PnL via
the shared fill_order path.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal

from futur3.data import SourceTier
from futur3.data.types import BarResolution, ContractSymbol, content_sha256
from futur3.data.verifier import VerifiedBar
from futur3.execution.adapters.mock_broker import MockBroker
from futur3.execution.broker import Order, OrderType, Position, Side
from futur3.execution.slippage import SlippageModel

_TS = datetime(2026, 5, 21, 14, 0, tzinfo=UTC)


class _FixedSlippage(SlippageModel):
    """Adverse fixed-amount slippage double (decouples advance tests from TickHaircut specifics)."""

    def __init__(self, amount: Decimal) -> None:
        self._amount = amount

    def apply_slippage(
        self,
        intended_price: Decimal,
        side: Side,
        quantity: int,
        contract: ContractSymbol,
        ts: datetime,
    ) -> Decimal:
        return intended_price + self._amount if side == Side.BUY else intended_price - self._amount


def _bar(
    *, o: str, h: str, low: str, c: str, contract: str = "ESM26", ts: datetime = _TS
) -> VerifiedBar:
    return VerifiedBar(
        contract=ContractSymbol(contract),
        ts=ts,
        resolution=BarResolution.MIN_5,
        open=Decimal(o),
        high=Decimal(h),
        low=Decimal(low),
        close=Decimal(c),
        volume=100,
        n_sources_agreed=2,
        source_provenance_hashes=(content_sha256(b"s1"),),
        verifier_run_hash=content_sha256(f"{contract}|{ts}|{c}".encode()),
        prev_bar_hash=None,
        tier_used=SourceTier.T2_EXCHANGE,
        policy_id="PHASE_A1",
    )


def _order(
    *,
    side: Side = Side.BUY,
    order_type: OrderType = OrderType.MKT,
    quantity: int = 1,
    limit_price: Decimal | None = None,
    stop_price: Decimal | None = None,
    contract: str = "ESM26",
) -> Order:
    return Order(
        contract=ContractSymbol(contract),
        side=side,
        quantity=quantity,
        order_type=order_type,
        limit_price=limit_price,
        stop_price=stop_price,
    )


def _place(broker: MockBroker, order: Order) -> str:
    return asyncio.run(broker.place_order(order))


def _positions(broker: MockBroker) -> list[Position]:
    return asyncio.run(broker.get_positions())


# ============================================================================
# TestAdvanceMarket
# ============================================================================


class TestAdvanceMarket:
    def test_mkt_buy_fills_at_open(self) -> None:
        b = MockBroker()
        _place(b, _order(side=Side.BUY))
        events = b.advance(_bar(o="5260.00", h="5265.00", low="5255.00", c="5262.00"))
        assert len(events) == 1
        assert events[0].event_type == "filled"
        assert events[0].fill_price == Decimal("5260.00")  # bar.open, no slippage
        pos = _positions(b)
        assert len(pos) == 1
        assert pos[0].quantity == 1
        assert pos[0].avg_entry_price == Decimal("5260.00")

    def test_mkt_sell_fills_at_open(self) -> None:
        b = MockBroker()
        _place(b, _order(side=Side.SELL))
        b.advance(_bar(o="5260.00", h="5265.00", low="5255.00", c="5262.00"))
        assert _positions(b)[0].quantity == -1

    def test_mkt_buy_takes_adverse_slippage(self) -> None:
        b = MockBroker()
        _place(b, _order(side=Side.BUY))
        events = b.advance(
            _bar(o="5260.00", h="5265.00", low="5255.00", c="5262.00"),
            _FixedSlippage(Decimal("0.50")),
        )
        assert events[0].fill_price == Decimal("5260.50")  # open + 0.50 (adverse for BUY)

    def test_mkt_sell_takes_adverse_slippage(self) -> None:
        b = MockBroker()
        _place(b, _order(side=Side.SELL))
        events = b.advance(
            _bar(o="5260.00", h="5265.00", low="5255.00", c="5262.00"),
            _FixedSlippage(Decimal("0.50")),
        )
        assert events[0].fill_price == Decimal("5259.50")  # open - 0.50 (adverse for SELL)


# ============================================================================
# TestAdvanceLimit
# ============================================================================


class TestAdvanceLimit:
    def test_buy_limit_fills_when_low_crosses(self) -> None:
        b = MockBroker()
        _place(b, _order(side=Side.BUY, order_type=OrderType.LMT, limit_price=Decimal("5256.00")))
        events = b.advance(_bar(o="5260.00", h="5265.00", low="5255.00", c="5262.00"))
        assert len(events) == 1
        assert events[0].fill_price == Decimal("5256.00")  # at the limit, not the open

    def test_buy_limit_no_fill_when_low_above_limit(self) -> None:
        b = MockBroker()
        _place(b, _order(side=Side.BUY, order_type=OrderType.LMT, limit_price=Decimal("5250.00")))
        events = b.advance(_bar(o="5260.00", h="5265.00", low="5255.00", c="5262.00"))
        assert events == []
        assert _positions(b) == []

    def test_sell_limit_fills_when_high_crosses(self) -> None:
        b = MockBroker()
        _place(b, _order(side=Side.SELL, order_type=OrderType.LMT, limit_price=Decimal("5263.00")))
        events = b.advance(_bar(o="5260.00", h="5265.00", low="5255.00", c="5262.00"))
        assert events[0].fill_price == Decimal("5263.00")

    def test_limit_ignores_slippage(self) -> None:
        b = MockBroker()
        _place(b, _order(side=Side.BUY, order_type=OrderType.LMT, limit_price=Decimal("5256.00")))
        events = b.advance(
            _bar(o="5260.00", h="5265.00", low="5255.00", c="5262.00"),
            _FixedSlippage(Decimal("0.50")),
        )
        assert events[0].fill_price == Decimal("5256.00")  # capped at limit, no slippage


# ============================================================================
# TestAdvanceStop
# ============================================================================


class TestAdvanceStop:
    def test_buy_stop_triggers_when_high_crosses(self) -> None:
        b = MockBroker()
        _place(b, _order(side=Side.BUY, order_type=OrderType.STP, stop_price=Decimal("5263.00")))
        events = b.advance(_bar(o="5260.00", h="5265.00", low="5255.00", c="5262.00"))
        assert events[0].fill_price == Decimal("5263.00")  # at stop, no slippage model

    def test_buy_stop_takes_slippage_on_trigger(self) -> None:
        b = MockBroker()
        _place(b, _order(side=Side.BUY, order_type=OrderType.STP, stop_price=Decimal("5263.00")))
        events = b.advance(
            _bar(o="5260.00", h="5265.00", low="5255.00", c="5262.00"),
            _FixedSlippage(Decimal("0.50")),
        )
        assert events[0].fill_price == Decimal("5263.50")  # stop + 0.50 (becomes market, adverse)

    def test_sell_stop_triggers_when_low_crosses(self) -> None:
        b = MockBroker()
        _place(b, _order(side=Side.SELL, order_type=OrderType.STP, stop_price=Decimal("5257.00")))
        events = b.advance(_bar(o="5260.00", h="5265.00", low="5255.00", c="5262.00"))
        assert events[0].fill_price == Decimal("5257.00")

    def test_stop_no_fill_when_not_triggered(self) -> None:
        b = MockBroker()
        _place(b, _order(side=Side.BUY, order_type=OrderType.STP, stop_price=Decimal("5270.00")))
        assert b.advance(_bar(o="5260.00", h="5265.00", low="5255.00", c="5262.00")) == []


# ============================================================================
# TestAdvanceSkips
# ============================================================================


class TestAdvanceSkips:
    def test_other_order_type_not_autofilled(self) -> None:
        b = MockBroker()
        _place(b, _order(order_type=OrderType.MOC))  # market-on-close: not in MVP advance scope
        assert b.advance(_bar(o="5260.00", h="5265.00", low="5255.00", c="5262.00")) == []
        assert _positions(b) == []

    def test_cancelled_order_not_filled(self) -> None:
        b = MockBroker()
        oid = _place(b, _order())
        asyncio.run(b.cancel_order(oid))
        assert b.advance(_bar(o="5260.00", h="5265.00", low="5255.00", c="5262.00")) == []

    def test_already_filled_not_refilled(self) -> None:
        b = MockBroker()
        oid = _place(b, _order())
        b.fill_order(oid, Decimal("5260.00"))  # fill it manually first
        assert b.advance(_bar(o="5260.00", h="5265.00", low="5255.00", c="5262.00")) == []

    def test_no_orders_returns_empty(self) -> None:
        b = MockBroker()
        assert b.advance(_bar(o="5260.00", h="5265.00", low="5255.00", c="5262.00")) == []


# ============================================================================
# TestAdvanceDeterminism
# ============================================================================


class TestAdvanceDeterminism:
    def test_fills_in_submission_order(self) -> None:
        b = MockBroker()
        id1 = _place(b, _order(contract="ESM26"))
        id2 = _place(b, _order(contract="ESM26"))
        events = b.advance(_bar(o="5260.00", h="5265.00", low="5255.00", c="5262.00"))
        assert [e.broker_order_id for e in events] == [id1, id2]  # submission order

    def test_only_fills_bar_contract(self) -> None:
        # advance(ESM26 bar) must fill only ESM26 orders; an NQM26 order stays open.
        b = MockBroker()
        es = _place(b, _order(contract="ESM26"))
        _place(b, _order(contract="NQM26"))
        bar = _bar(o="5260.00", h="5265.00", low="5255.00", c="5262.00", contract="ESM26")
        events = b.advance(bar)
        assert [e.broker_order_id for e in events] == [es]
        assert [p.contract for p in _positions(b)] == [ContractSymbol("ESM26")]


# ============================================================================
# TestReopenFromFlat — regression: re-open after a full close must not build a price-0 position
# ============================================================================


class TestReopenFromFlat:
    """A fully-closed position is retained at quantity=0 / avg_entry_price=0 (to carry the day's
    realized PnL). A later fill into that flat slot must open FRESH at the new fill price — not fall
    through the close math and copy avg_entry_price=0 onto a nonzero position (which trips the
    qty!=0 -> price>0 invariant). The 2018+ full-grid C.1 surfaced this when a candidate closed a
    short to flat and re-shorted on a later bar."""

    def test_close_then_reopen_short(self) -> None:
        b = MockBroker()
        _place(b, _order(side=Side.BUY))  # long 1 @ 100
        b.advance(_bar(o="100", h="101", low="99", c="100"))
        _place(b, _order(side=Side.SELL))  # fully close @ 110 -> flat slot retained
        b.advance(_bar(o="110", h="111", low="109", c="110"))
        flat = _positions(b)[0]
        assert flat.quantity == 0
        realized = flat.realized_pnl_today  # (110-100)*1*mult, kept on the flat slot
        _place(b, _order(side=Side.SELL, quantity=2))  # RE-OPEN short from flat (old crash path)
        b.advance(_bar(o="120", h="121", low="119", c="120"))
        pos = _positions(b)[0]
        assert pos.quantity == -2
        assert pos.avg_entry_price == Decimal("120")  # fresh entry at the new bar.open, never 0
        assert pos.realized_pnl_today == realized  # same-day realized PnL preserved across re-open

    def test_close_then_reopen_long(self) -> None:
        b = MockBroker()
        _place(b, _order(side=Side.SELL))  # short 1 @ 100
        b.advance(_bar(o="100", h="101", low="99", c="100"))
        _place(b, _order(side=Side.BUY))  # close -> flat
        b.advance(_bar(o="90", h="91", low="89", c="90"))
        _place(b, _order(side=Side.BUY, quantity=3))  # re-open long from flat
        b.advance(_bar(o="95", h="96", low="94", c="95"))
        pos = _positions(b)[0]
        assert pos.quantity == 3
        assert pos.avg_entry_price == Decimal("95")  # fresh entry, never the flat-slot 0
