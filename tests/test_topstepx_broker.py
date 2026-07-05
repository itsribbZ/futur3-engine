"""A1.14 TopstepXBrokerAdapter STUB test suite.

Test discipline (paper-first plan):
- ALL tests fixture-only (zero TopstepX REST/WS session).
- STUB tests verify the downgrade matrix works end-to-end (testable BEFORE
  live wiring) + each async live method raises TopstepXBrokerNotImplemented.
- Sync interface (broker_id, supported_order_types, is_trade_allowed,
  downgrade_order_type, construction validation) works fully in STUB.

References:
- `futur3/execution/adapters/topstepx_broker.py` (implementation)
- internal design notes (STUB spec)
- internal design notes (STUB scope)
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from futur3.data.types import ContractSymbol
from futur3.execution.adapters.topstepx_broker import (
    TopstepXBrokerAdapter,
    TopstepXBrokerNotImplemented,
)
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


def _mkt_order() -> Order:
    return Order(
        contract=ContractSymbol("MESM26"),  # micro for TopstepX-typical
        side=Side.BUY,
        quantity=1,
        order_type=OrderType.MKT,
    )


# ============================================================================
# TestA1_14_TopstepXConstruction
# ============================================================================


class TestA1_14_TopstepXConstruction:
    def test_default_construction(self) -> None:
        a = TopstepXBrokerAdapter()
        assert a.broker_id == "topstepx"
        assert a.account_state == "Combine"  # most restrictive default

    def test_xfa_state(self) -> None:
        a = TopstepXBrokerAdapter(account_state="XFA")
        assert a.account_state == "XFA"

    def test_lfa_state(self) -> None:
        a = TopstepXBrokerAdapter(account_state="LFA")
        assert a.account_state == "LFA"

    def test_invalid_account_state_raises(self) -> None:
        with pytest.raises(ValueError, match="account_state must be"):
            TopstepXBrokerAdapter(account_state="invalid")  # type: ignore[arg-type]

    def test_empty_api_key_env_var_raises(self) -> None:
        with pytest.raises(ValueError, match="api_key_env_var must be non-empty"):
            TopstepXBrokerAdapter(api_key_env_var="")

    def test_with_account_id(self) -> None:
        a = TopstepXBrokerAdapter(account_id="TS-12345-Combine")
        # account_id stored internally — not crashed
        assert a is not None

    def test_is_subclass_of_broker_adapter(self) -> None:
        assert issubclass(TopstepXBrokerAdapter, BrokerAdapter)

    def test_repr_includes_state(self) -> None:
        a = TopstepXBrokerAdapter(account_state="XFA")
        rep = repr(a)
        assert "XFA" in rep


# ============================================================================
# TestA1_14_TopstepXSupportedOrderTypes
# ============================================================================


class TestA1_14_TopstepXSupportedOrderTypes:
    def test_7_native_types(self) -> None:
        """ProjectX OrderType enum is NARROW — exactly 7 types."""
        a = TopstepXBrokerAdapter()
        assert len(a.supported_order_types()) == 7

    def test_mkt_supported(self) -> None:
        a = TopstepXBrokerAdapter()
        assert OrderType.MKT in a.supported_order_types()

    def test_lmt_supported(self) -> None:
        a = TopstepXBrokerAdapter()
        assert OrderType.LMT in a.supported_order_types()

    def test_trail_supported(self) -> None:
        a = TopstepXBrokerAdapter()
        assert OrderType.TRAIL in a.supported_order_types()

    def test_join_bid_supported(self) -> None:
        """JOIN_BID is TopstepX-native passive entry — IBKR doesn't have it."""
        a = TopstepXBrokerAdapter()
        assert OrderType.JOIN_BID in a.supported_order_types()

    def test_join_ask_supported(self) -> None:
        a = TopstepXBrokerAdapter()
        assert OrderType.JOIN_ASK in a.supported_order_types()

    def test_stp_prt_not_native(self) -> None:
        """STP_PRT downgrades to STP — NOT in supported_order_types() set."""
        a = TopstepXBrokerAdapter()
        assert OrderType.STP_PRT not in a.supported_order_types()

    def test_moc_not_supported(self) -> None:
        """MOC is REJECT-tier on TopstepX — MOC-dependent strategies skip."""
        a = TopstepXBrokerAdapter()
        assert OrderType.MOC not in a.supported_order_types()

    def test_trail_lmt_not_supported(self) -> None:
        a = TopstepXBrokerAdapter()
        assert OrderType.TRAIL_LMT not in a.supported_order_types()


# ============================================================================
# TestA1_14_TopstepXDowngradeMatrix
# ============================================================================


class TestA1_14_TopstepXDowngradeMatrix:
    def test_native_mkt_passthrough(self) -> None:
        a = TopstepXBrokerAdapter()
        assert a.downgrade_order_type(OrderType.MKT) == OrderType.MKT

    def test_native_lmt_passthrough(self) -> None:
        a = TopstepXBrokerAdapter()
        assert a.downgrade_order_type(OrderType.LMT) == OrderType.LMT

    def test_native_trail_passthrough(self) -> None:
        a = TopstepXBrokerAdapter()
        assert a.downgrade_order_type(OrderType.TRAIL) == OrderType.TRAIL

    def test_stp_prt_downgrades_to_stp(self) -> None:
        """LOW severity downgrade — futur3 default safety stop affected."""
        a = TopstepXBrokerAdapter()
        assert a.downgrade_order_type(OrderType.STP_PRT) == OrderType.STP

    def test_mit_downgrades_to_stp(self) -> None:
        """MEDIUM severity downgrade — semantic gap (touch vs cross)."""
        a = TopstepXBrokerAdapter()
        assert a.downgrade_order_type(OrderType.MIT) == OrderType.STP

    def test_lit_downgrades_to_lmt(self) -> None:
        """MEDIUM severity downgrade — semantic gap (touch vs cross)."""
        a = TopstepXBrokerAdapter()
        assert a.downgrade_order_type(OrderType.LIT) == OrderType.LMT

    def test_moc_raises_reject(self) -> None:
        """HIGH severity REJECT — MOC-dependent strategies cannot run."""
        a = TopstepXBrokerAdapter()
        with pytest.raises(OrderTypeUnsupportedError, match="MOC"):
            a.downgrade_order_type(OrderType.MOC)

    def test_moo_raises_reject(self) -> None:
        a = TopstepXBrokerAdapter()
        with pytest.raises(OrderTypeUnsupportedError, match="MOO"):
            a.downgrade_order_type(OrderType.MOO)

    def test_trail_lmt_raises_reject(self) -> None:
        a = TopstepXBrokerAdapter()
        with pytest.raises(OrderTypeUnsupportedError, match="TRAIL_LMT"):
            a.downgrade_order_type(OrderType.TRAIL_LMT)


# ============================================================================
# TestA1_14_TopstepXIsTradeAllowed
# ============================================================================


class TestA1_14_TopstepXIsTradeAllowed:
    def test_basic_allow_combine(self) -> None:
        a = TopstepXBrokerAdapter(account_state="Combine")
        ok, reason = a.is_trade_allowed(ContractSymbol("MESM26"), Side.BUY, 1)
        assert ok is True
        assert reason is None

    def test_basic_allow_xfa(self) -> None:
        a = TopstepXBrokerAdapter(account_state="XFA")
        ok, _reason = a.is_trade_allowed(ContractSymbol("MESM26"), Side.SELL, 2)
        assert ok is True

    def test_basic_allow_lfa(self) -> None:
        a = TopstepXBrokerAdapter(account_state="LFA")
        ok, _reason = a.is_trade_allowed(ContractSymbol("MESM26"), Side.BUY, 1)
        assert ok is True

    def test_zero_quantity_blocked(self) -> None:
        a = TopstepXBrokerAdapter()
        ok, reason = a.is_trade_allowed(ContractSymbol("MESM26"), Side.BUY, 0)
        assert ok is False
        assert reason is not None
        assert "quantity must be > 0" in reason

    def test_negative_quantity_blocked(self) -> None:
        a = TopstepXBrokerAdapter()
        ok, _reason = a.is_trade_allowed(ContractSymbol("MESM26"), Side.SELL, -1)
        assert ok is False


# ============================================================================
# TestA1_14_TopstepXShellRaises
# ============================================================================


class TestA1_14_TopstepXShellRaises:
    def test_place_order_raises(self) -> None:
        a = TopstepXBrokerAdapter()
        with pytest.raises(TopstepXBrokerNotImplemented, match="STUB-only"):
            asyncio.run(a.place_order(_mkt_order()))

    def test_cancel_order_raises(self) -> None:
        a = TopstepXBrokerAdapter()
        with pytest.raises(TopstepXBrokerNotImplemented, match="STUB-only"):
            asyncio.run(a.cancel_order("ts-xyz"))

    def test_modify_order_raises(self) -> None:
        a = TopstepXBrokerAdapter()
        with pytest.raises(TopstepXBrokerNotImplemented, match="STUB-only"):
            asyncio.run(a.modify_order("ts-xyz", new_limit_price=Decimal("5260")))

    def test_modify_order_all_none_args_raises_value_error(self) -> None:
        a = TopstepXBrokerAdapter()
        with pytest.raises(ValueError, match="must be non-None"):
            asyncio.run(a.modify_order("ts-xyz"))

    def test_get_positions_raises(self) -> None:
        a = TopstepXBrokerAdapter()
        with pytest.raises(TopstepXBrokerNotImplemented, match="STUB-only"):
            asyncio.run(a.get_positions())

    def test_get_account_metrics_raises(self) -> None:
        a = TopstepXBrokerAdapter()
        with pytest.raises(TopstepXBrokerNotImplemented, match="STUB-only"):
            asyncio.run(a.get_account_metrics())

    def test_stream_order_events_raises_on_iteration(self) -> None:
        a = TopstepXBrokerAdapter()

        async def consume() -> None:
            async for _ in a.stream_order_events():  # pragma: no cover
                pass

        with pytest.raises(TopstepXBrokerNotImplemented, match="STUB-only"):
            asyncio.run(consume())


# ============================================================================
# TestA1_14_TopstepXExceptionHierarchy
# ============================================================================


class TestA1_14_TopstepXExceptionHierarchy:
    def test_extends_broker_error(self) -> None:
        assert issubclass(TopstepXBrokerNotImplemented, BrokerError)

    def test_is_exception(self) -> None:
        assert issubclass(TopstepXBrokerNotImplemented, Exception)
