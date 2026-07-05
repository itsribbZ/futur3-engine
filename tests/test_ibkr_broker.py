"""A1.13 IBKRBrokerAdapter shell test suite.

Test discipline:
- ALL tests fixture-only (zero network, zero IB Gateway session).
- Shell tests verify the interface is wired correctly + each live method
  raises `IBKRBrokerNotImplemented` (NOT a silent fallback).
- Sync interface (broker_id, supported_order_types, is_trade_allowed,
  construction validation) works fully in shell.

References:
- `futur3/execution/adapters/ibkr_broker.py` (implementation)
- internal design notes (IBKR shell spec)
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from futur3.data.types import ContractSymbol
from futur3.execution.adapters.ibkr_broker import (
    IBKR_DEFAULT_HOST,
    IBKR_LIVE_PORT,
    IBKR_PAPER_PORT,
    IBKRBrokerAdapter,
    IBKRBrokerNotImplemented,
)
from futur3.execution.broker import (
    BrokerAdapter,
    BrokerError,
    Order,
    OrderType,
    Side,
)

# ============================================================================
# Helpers
# ============================================================================


def _mkt_order() -> Order:
    return Order(
        contract=ContractSymbol("ESM26"),
        side=Side.BUY,
        quantity=1,
        order_type=OrderType.MKT,
    )


# ============================================================================
# TestA1_13_IBKRConstants
# ============================================================================


class TestA1_13_IBKRConstants:
    def test_paper_port_4002(self) -> None:
        assert IBKR_PAPER_PORT == 4002

    def test_live_port_4001(self) -> None:
        assert IBKR_LIVE_PORT == 4001

    def test_default_host_loopback(self) -> None:
        """Local-only invariant — never bind 0.0.0.0."""
        assert IBKR_DEFAULT_HOST == "127.0.0.1"


# ============================================================================
# TestA1_13_IBKRConstruction
# ============================================================================


class TestA1_13_IBKRConstruction:
    def test_default_construction(self) -> None:
        a = IBKRBrokerAdapter()
        assert a.host == IBKR_DEFAULT_HOST
        assert a.port == IBKR_PAPER_PORT
        assert a.client_id == 1
        assert a.is_paper is True

    def test_live_port_construction(self) -> None:
        a = IBKRBrokerAdapter(port=IBKR_LIVE_PORT)
        assert a.port == IBKR_LIVE_PORT
        assert a.is_paper is False

    def test_custom_port_in_valid_range(self) -> None:
        a = IBKRBrokerAdapter(port=7497)
        assert a.port == 7497

    def test_empty_host_raises(self) -> None:
        with pytest.raises(ValueError, match="host must be non-empty"):
            IBKRBrokerAdapter(host="")

    def test_port_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="port must be"):
            IBKRBrokerAdapter(port=80)  # privileged port < 1024

    def test_port_too_high_raises(self) -> None:
        with pytest.raises(ValueError, match="port must be"):
            IBKRBrokerAdapter(port=99999)

    def test_negative_client_id_raises(self) -> None:
        with pytest.raises(ValueError, match="client_id must be >= 0"):
            IBKRBrokerAdapter(client_id=-1)

    def test_custom_account_id(self) -> None:
        a = IBKRBrokerAdapter(account_id="DU1234567")
        # account_id is internal; not exposed as property in shell — verify no crash
        assert a is not None

    def test_is_subclass_of_broker_adapter(self) -> None:
        assert issubclass(IBKRBrokerAdapter, BrokerAdapter)


# ============================================================================
# TestA1_13_IBKRBrokerID
# ============================================================================


class TestA1_13_IBKRBrokerID:
    def test_broker_id_is_ibkr_tws(self) -> None:
        a = IBKRBrokerAdapter()
        assert a.broker_id == "ibkr_tws"

    def test_repr_includes_host_port(self) -> None:
        a = IBKRBrokerAdapter()
        rep = repr(a)
        assert "127.0.0.1" in rep
        assert "4002" in rep
        assert "paper=True" in rep


# ============================================================================
# TestA1_13_IBKRSupportedOrderTypes
# ============================================================================


class TestA1_13_IBKRSupportedOrderTypes:
    def test_11_supported_types(self) -> None:
        """IBKR supports 11 of 13 (all except JOIN_BID + JOIN_ASK)."""
        a = IBKRBrokerAdapter()
        assert len(a.supported_order_types()) == 11

    def test_join_bid_not_supported(self) -> None:
        """JOIN_BID is TopstepX-native; IBKR rejects."""
        a = IBKRBrokerAdapter()
        assert OrderType.JOIN_BID not in a.supported_order_types()

    def test_join_ask_not_supported(self) -> None:
        """JOIN_ASK is TopstepX-native; IBKR rejects."""
        a = IBKRBrokerAdapter()
        assert OrderType.JOIN_ASK not in a.supported_order_types()

    def test_mkt_supported(self) -> None:
        a = IBKRBrokerAdapter()
        assert OrderType.MKT in a.supported_order_types()

    def test_stp_prt_supported(self) -> None:
        """CME-native protective stop — futur3 default safety stop."""
        a = IBKRBrokerAdapter()
        assert OrderType.STP_PRT in a.supported_order_types()

    def test_moc_supported(self) -> None:
        """MOC-dependent strategies run on IBKR only."""
        a = IBKRBrokerAdapter()
        assert OrderType.MOC in a.supported_order_types()

    def test_supported_returns_set_copy(self) -> None:
        """Caller-mutation of returned set must NOT affect adapter."""
        a = IBKRBrokerAdapter()
        s = a.supported_order_types()
        s.clear()
        # Adapter still returns full set
        assert len(a.supported_order_types()) == 11


# ============================================================================
# TestA1_13_IBKRIsTradeAllowed
# ============================================================================


class TestA1_13_IBKRIsTradeAllowed:
    def test_basic_allow(self) -> None:
        """Shell stub: returns (True, None) for valid input."""
        a = IBKRBrokerAdapter()
        ok, reason = a.is_trade_allowed(ContractSymbol("ESM26"), Side.BUY, 1)
        assert ok is True
        assert reason is None

    def test_zero_quantity_blocked(self) -> None:
        a = IBKRBrokerAdapter()
        ok, reason = a.is_trade_allowed(ContractSymbol("ESM26"), Side.BUY, 0)
        assert ok is False
        assert reason is not None
        assert "quantity must be > 0" in reason

    def test_negative_quantity_blocked(self) -> None:
        a = IBKRBrokerAdapter()
        ok, reason = a.is_trade_allowed(ContractSymbol("ESM26"), Side.SELL, -1)
        assert ok is False
        assert reason is not None


# ============================================================================
# TestA1_13_IBKRShellRaises — async live methods raise IBKRBrokerNotImplemented
# ============================================================================


class TestA1_13_IBKRShellRaises:
    def test_place_order_raises(self) -> None:
        a = IBKRBrokerAdapter()
        with pytest.raises(IBKRBrokerNotImplemented, match="shell-only"):
            asyncio.run(a.place_order(_mkt_order()))

    def test_cancel_order_raises(self) -> None:
        a = IBKRBrokerAdapter()
        with pytest.raises(IBKRBrokerNotImplemented, match="shell-only"):
            asyncio.run(a.cancel_order("ib-xyz"))

    def test_modify_order_raises(self) -> None:
        a = IBKRBrokerAdapter()
        with pytest.raises(IBKRBrokerNotImplemented, match="shell-only"):
            asyncio.run(a.modify_order("ib-xyz", new_limit_price=Decimal("5260")))

    def test_modify_order_all_none_args_raises_value_error(self) -> None:
        """ValueError takes precedence over IBKRBrokerNotImplemented when args invalid."""
        a = IBKRBrokerAdapter()
        with pytest.raises(ValueError, match="must be non-None"):
            asyncio.run(a.modify_order("ib-xyz"))

    def test_get_positions_raises(self) -> None:
        a = IBKRBrokerAdapter()
        with pytest.raises(IBKRBrokerNotImplemented, match="shell-only"):
            asyncio.run(a.get_positions())

    def test_get_account_metrics_raises(self) -> None:
        a = IBKRBrokerAdapter()
        with pytest.raises(IBKRBrokerNotImplemented, match="shell-only"):
            asyncio.run(a.get_account_metrics())

    def test_stream_order_events_raises_on_iteration(self) -> None:
        a = IBKRBrokerAdapter()

        async def consume() -> None:
            async for _ in a.stream_order_events():  # pragma: no cover - never yields
                pass

        with pytest.raises(IBKRBrokerNotImplemented, match="shell-only"):
            asyncio.run(consume())


# ============================================================================
# TestA1_13_IBKRExceptionHierarchy
# ============================================================================


class TestA1_13_IBKRExceptionHierarchy:
    def test_ibkr_not_implemented_extends_broker_error(self) -> None:
        assert issubclass(IBKRBrokerNotImplemented, BrokerError)

    def test_ibkr_not_implemented_is_exception(self) -> None:
        assert issubclass(IBKRBrokerNotImplemented, Exception)
