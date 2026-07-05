"""A1.13 BrokerAdapter ABC + dataclasses + enums test suite.

Test discipline:
- ALL tests fixture-only (zero network, zero broker session).
- Boundary-invariant tests (dataclass validation, frozen immutability, ABC enforcement).
- Enum stability (values + count locked).

References:
- `futur3/execution/broker.py` (implementation)
- internal design notes (spec)
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from futur3.data.types import ContractSymbol
from futur3.execution.broker import (
    AccountMetrics,
    BrokerAdapter,
    BrokerError,
    BrokerNotConnectedError,
    Duration,
    Order,
    OrderEvent,
    OrderType,
    OrderTypeUnsupportedError,
    Position,
    Side,
)

# ============================================================================
# TestA1_13_Imports
# ============================================================================


class TestA1_13_Imports:
    def test_broker_adapter_is_abstract(self) -> None:
        assert issubclass(BrokerAdapter, abc.ABC)

    def test_all_dataclasses_importable(self) -> None:
        assert Order is not None
        assert OrderEvent is not None
        assert Position is not None
        assert AccountMetrics is not None

    def test_all_enums_importable(self) -> None:
        assert OrderType is not None
        assert Side is not None
        assert Duration is not None

    def test_all_exceptions_importable(self) -> None:
        assert BrokerError is not None
        assert OrderTypeUnsupportedError is not None
        assert BrokerNotConnectedError is not None

    def test_exception_hierarchy(self) -> None:
        assert issubclass(OrderTypeUnsupportedError, BrokerError)
        assert issubclass(BrokerNotConnectedError, BrokerError)
        assert issubclass(BrokerError, Exception)


# ============================================================================
# TestA1_13_OrderTypeEnum
# ============================================================================


class TestA1_13_OrderTypeEnum:
    def test_13_order_types(self) -> None:
        """Phase A1 scope: 13 types (Iceberg + Algo deferred to Phase A2)."""
        assert len(OrderType) == 13

    def test_market_value(self) -> None:
        assert OrderType.MKT.value == "market"

    def test_limit_value(self) -> None:
        assert OrderType.LMT.value == "limit"

    def test_stop_with_protection_present(self) -> None:
        """CME-native STP_PRT — futur3's default safety stop."""
        assert OrderType.STP_PRT.value == "stop_with_protection"

    def test_topstep_only_join_types_present(self) -> None:
        assert OrderType.JOIN_BID.value == "join_bid"
        assert OrderType.JOIN_ASK.value == "join_ask"

    def test_ibkr_only_moc_moo_present(self) -> None:
        """MOC-dependent strategies + gap-fill strategies (IBKR-only)."""
        assert OrderType.MOC.value == "market_on_close"
        assert OrderType.MOO.value == "market_on_open"

    def test_iceberg_not_in_phase_a1(self) -> None:
        """Iceberg deferred to Phase A2 (size too small for Phase A1 universe)."""
        assert not any(t.name == "ICEBERG" for t in OrderType)

    def test_algo_not_in_phase_a1(self) -> None:
        """Algo / IBALGO deferred to Phase A2."""
        assert not any(t.name == "ALGO" for t in OrderType)


# ============================================================================
# TestA1_13_SideEnum
# ============================================================================


class TestA1_13_SideEnum:
    def test_2_sides(self) -> None:
        assert len(Side) == 2

    def test_buy_value(self) -> None:
        assert Side.BUY.value == "buy"

    def test_sell_value(self) -> None:
        assert Side.SELL.value == "sell"


# ============================================================================
# TestA1_13_DurationEnum
# ============================================================================


class TestA1_13_DurationEnum:
    def test_3_durations(self) -> None:
        assert len(Duration) == 3

    def test_day_value(self) -> None:
        assert Duration.DAY.value == "day"

    def test_gtc_value(self) -> None:
        assert Duration.GTC.value == "good_till_cancelled"


# ============================================================================
# TestA1_13_OrderDataclass
# ============================================================================


class TestA1_13_OrderDataclass:
    def _market_order(self) -> Order:
        return Order(
            contract=ContractSymbol("ESM26"),
            side=Side.BUY,
            quantity=1,
            order_type=OrderType.MKT,
        )

    def test_market_order_valid(self) -> None:
        o = self._market_order()
        assert o.quantity == 1
        assert o.duration == Duration.DAY  # default

    def test_limit_order_requires_limit_price(self) -> None:
        with pytest.raises(ValueError, match="limit_price required for LMT"):
            Order(
                contract=ContractSymbol("ESM26"),
                side=Side.BUY,
                quantity=1,
                order_type=OrderType.LMT,
                limit_price=None,
            )

    def test_stop_order_requires_stop_price(self) -> None:
        with pytest.raises(ValueError, match="stop_price required for STP"):
            Order(
                contract=ContractSymbol("ESM26"),
                side=Side.SELL,
                quantity=1,
                order_type=OrderType.STP,
                stop_price=None,
            )

    def test_stop_limit_requires_both(self) -> None:
        with pytest.raises(ValueError, match="limit_price required for STP_LMT"):
            Order(
                contract=ContractSymbol("ESM26"),
                side=Side.BUY,
                quantity=1,
                order_type=OrderType.STP_LMT,
                stop_price=Decimal("5260.00"),
                limit_price=None,
            )

    def test_stop_with_protection_requires_stop(self) -> None:
        """STP_PRT is futur3's default safety stop — must have stop_price."""
        with pytest.raises(ValueError, match="stop_price required for STP_PRT"):
            Order(
                contract=ContractSymbol("ESM26"),
                side=Side.SELL,
                quantity=1,
                order_type=OrderType.STP_PRT,
                stop_price=None,
            )

    def test_negative_quantity_raises(self) -> None:
        with pytest.raises(ValueError, match="quantity must be > 0"):
            Order(
                contract=ContractSymbol("ESM26"),
                side=Side.BUY,
                quantity=-1,
                order_type=OrderType.MKT,
            )

    def test_zero_quantity_raises(self) -> None:
        with pytest.raises(ValueError, match="quantity must be > 0"):
            Order(
                contract=ContractSymbol("ESM26"),
                side=Side.BUY,
                quantity=0,
                order_type=OrderType.MKT,
            )

    def test_frozen_immutability(self) -> None:
        o = self._market_order()
        with pytest.raises(AttributeError):
            o.quantity = 2  # type: ignore[misc]


# ============================================================================
# TestA1_13_OrderEventDataclass
# ============================================================================


class TestA1_13_OrderEventDataclass:
    def _valid_kwargs(self) -> dict[str, object]:
        return {
            "broker_order_id": "ib-12345",
            "client_order_id": "futur3-ord-001",
            "contract": ContractSymbol("ESM26"),
            "event_type": "filled",
            "fill_price": Decimal("5260.25"),
            "fill_quantity": 1,
            "cumulative_filled": 1,
            "remaining_quantity": 0,
            "ts_utc": datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
            "raw_broker_payload_sha": "a" * 64,
        }

    def test_valid_event(self) -> None:
        OrderEvent(**self._valid_kwargs())  # type: ignore[arg-type]

    def test_naive_ts_raises(self) -> None:
        kw = self._valid_kwargs()
        kw["ts_utc"] = datetime(2026, 5, 21, 22, 0, 0)  # no tzinfo
        with pytest.raises(ValueError, match="must be TZ-aware"):
            OrderEvent(**kw)  # type: ignore[arg-type]

    def test_non_utc_ts_raises(self) -> None:
        kw = self._valid_kwargs()
        kw["ts_utc"] = datetime(2026, 5, 21, 17, 0, 0, tzinfo=timezone(timedelta(hours=-5)))
        with pytest.raises(ValueError, match="must be UTC"):
            OrderEvent(**kw)  # type: ignore[arg-type]

    def test_negative_cumulative_raises(self) -> None:
        kw = self._valid_kwargs()
        kw["cumulative_filled"] = -1
        with pytest.raises(ValueError, match="cumulative_filled must be >= 0"):
            OrderEvent(**kw)  # type: ignore[arg-type]

    def test_negative_remaining_raises(self) -> None:
        kw = self._valid_kwargs()
        kw["remaining_quantity"] = -1
        with pytest.raises(ValueError, match="remaining_quantity must be >= 0"):
            OrderEvent(**kw)  # type: ignore[arg-type]

    def test_bad_sha_length_raises(self) -> None:
        kw = self._valid_kwargs()
        kw["raw_broker_payload_sha"] = "abc"
        with pytest.raises(ValueError, match="must be hex-SHA256"):
            OrderEvent(**kw)  # type: ignore[arg-type]

    def test_partial_fill_allowed(self) -> None:
        kw = self._valid_kwargs()
        kw["event_type"] = "partial_fill"
        kw["fill_quantity"] = 2
        kw["cumulative_filled"] = 2
        kw["remaining_quantity"] = 3
        OrderEvent(**kw)  # type: ignore[arg-type]


# ============================================================================
# TestA1_13_PositionDataclass
# ============================================================================


class TestA1_13_PositionDataclass:
    def test_long_position(self) -> None:
        p = Position(
            contract=ContractSymbol("ESM26"),
            quantity=2,
            avg_entry_price=Decimal("5260.00"),
            unrealized_pnl=Decimal("125.50"),
            realized_pnl_today=Decimal("0"),
            margin_used=Decimal("28000"),
        )
        assert p.quantity == 2

    def test_short_position(self) -> None:
        p = Position(
            contract=ContractSymbol("ESM26"),
            quantity=-1,
            avg_entry_price=Decimal("5260.00"),
            unrealized_pnl=Decimal("-50"),
            realized_pnl_today=Decimal("0"),
            margin_used=Decimal("14000"),
        )
        assert p.quantity == -1

    def test_flat_position_allowed_no_entry_price(self) -> None:
        """When quantity == 0, avg_entry_price can be 0."""
        Position(
            contract=ContractSymbol("ESM26"),
            quantity=0,
            avg_entry_price=Decimal("0"),
            unrealized_pnl=Decimal("0"),
            realized_pnl_today=Decimal("125"),
            margin_used=Decimal("0"),
        )

    def test_non_flat_position_requires_positive_entry(self) -> None:
        with pytest.raises(ValueError, match="avg_entry_price must be > 0"):
            Position(
                contract=ContractSymbol("ESM26"),
                quantity=2,
                avg_entry_price=Decimal("0"),
                unrealized_pnl=Decimal("0"),
                realized_pnl_today=Decimal("0"),
                margin_used=Decimal("28000"),
            )

    def test_negative_margin_raises(self) -> None:
        with pytest.raises(ValueError, match="margin_used must be >= 0"):
            Position(
                contract=ContractSymbol("ESM26"),
                quantity=0,
                avg_entry_price=Decimal("0"),
                unrealized_pnl=Decimal("0"),
                realized_pnl_today=Decimal("0"),
                margin_used=Decimal("-1"),
            )


# ============================================================================
# TestA1_13_AccountMetricsDataclass
# ============================================================================


class TestA1_13_AccountMetricsDataclass:
    def _valid_kwargs(self) -> dict[str, object]:
        return {
            "account_id": "DU1234567",
            "equity": Decimal("50000"),
            "cash": Decimal("48000"),
            "margin_used": Decimal("2000"),
            "margin_available": Decimal("46000"),
            "leverage_used": Decimal("0.04"),
            "realized_pnl_today": Decimal("250"),
            "unrealized_pnl": Decimal("-50"),
        }

    def test_ibkr_style_no_topstep_fields(self) -> None:
        """IBKR adapter populates trailing_drawdown_remaining/etc. as None."""
        m = AccountMetrics(**self._valid_kwargs())  # type: ignore[arg-type]
        assert m.trailing_drawdown_remaining is None
        assert m.consistency_rule_status is None
        assert m.max_position_size_allowed is None

    def test_topstep_style_with_fields(self) -> None:
        kw = self._valid_kwargs()
        kw["trailing_drawdown_remaining"] = Decimal("1500")
        kw["consistency_rule_status"] = "compliant"
        kw["max_position_size_allowed"] = 5
        m = AccountMetrics(**kw)  # type: ignore[arg-type]
        assert m.trailing_drawdown_remaining == Decimal("1500")
        assert m.consistency_rule_status == "compliant"
        assert m.max_position_size_allowed == 5

    def test_negative_margin_raises(self) -> None:
        kw = self._valid_kwargs()
        kw["margin_used"] = Decimal("-1")
        with pytest.raises(ValueError, match="margin_used must be >= 0"):
            AccountMetrics(**kw)  # type: ignore[arg-type]

    def test_negative_margin_available_raises(self) -> None:
        kw = self._valid_kwargs()
        kw["margin_available"] = Decimal("-1")
        with pytest.raises(ValueError, match="margin_available must be >= 0"):
            AccountMetrics(**kw)  # type: ignore[arg-type]

    def test_negative_leverage_raises(self) -> None:
        kw = self._valid_kwargs()
        kw["leverage_used"] = Decimal("-1")
        with pytest.raises(ValueError, match="leverage_used must be >= 0"):
            AccountMetrics(**kw)  # type: ignore[arg-type]

    def test_negative_max_position_raises(self) -> None:
        kw = self._valid_kwargs()
        kw["max_position_size_allowed"] = -1
        with pytest.raises(ValueError, match="max_position_size_allowed must be >= 0"):
            AccountMetrics(**kw)  # type: ignore[arg-type]


# ============================================================================
# TestA1_13_ABCEnforcement
# ============================================================================


class TestA1_13_ABCEnforcement:
    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            BrokerAdapter()  # type: ignore[abstract]

    def test_subclass_must_implement_all(self) -> None:
        """Partial subclass missing abstractmethods → still uninstantiable."""

        class _Partial(BrokerAdapter):
            @property
            def broker_id(self) -> str:
                return "partial"

            # missing all other abstract methods

        with pytest.raises(TypeError, match="abstract"):
            _Partial()  # type: ignore[abstract]

    def test_concrete_subclass_works(self) -> None:
        """Full concrete subclass → instantiable (smoke check for ABC contract)."""

        class _Full(BrokerAdapter):
            @property
            def broker_id(self) -> str:
                return "full"

            async def place_order(self, order: Order) -> str:
                return "order-1"

            async def cancel_order(self, broker_order_id: str) -> None:
                return None

            async def modify_order(
                self,
                broker_order_id: str,
                new_limit_price: Decimal | None = None,
                new_stop_price: Decimal | None = None,
                new_quantity: int | None = None,
            ) -> None:
                return None

            async def get_positions(self) -> list[Position]:
                return []

            async def get_account_metrics(self) -> AccountMetrics:
                return AccountMetrics(
                    account_id="x",
                    equity=Decimal("0"),
                    cash=Decimal("0"),
                    margin_used=Decimal("0"),
                    margin_available=Decimal("0"),
                    leverage_used=Decimal("0"),
                    realized_pnl_today=Decimal("0"),
                    unrealized_pnl=Decimal("0"),
                )

            def stream_order_events(self) -> AsyncIterator[OrderEvent]:
                async def _empty() -> AsyncIterator[OrderEvent]:
                    return
                    yield  # type: ignore[unreachable]

                return _empty()

            def is_trade_allowed(
                self,
                contract: ContractSymbol,
                side: Side,
                quantity: int,
            ) -> tuple[bool, str | None]:
                return True, None

            def supported_order_types(self) -> set[OrderType]:
                return {OrderType.MKT}

        adapter = _Full()
        assert adapter.broker_id == "full"
        assert adapter.supported_order_types() == {OrderType.MKT}
