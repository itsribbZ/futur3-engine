"""PropAwareMockBroker test suite.

Covers the prop-account risk overlay end-to-end: construction validation, the
three gates (position cap / DLL / trailing MLL), the always-allow-closes rule,
halt latching, reset, and preservation of the base MockBroker contract.

Fixture-only (no network). Dollar amounts are derived from `multiplier_for` so
the assertions are correct regardless of the contract's exact point multiplier.

References:
- `futur3/execution/adapters/prop_aware_mock_broker.py` (implementation)
- `tests/test_mock_broker.py` (base-broker conventions mirrored here)
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from futur3.data.types import ContractSymbol
from futur3.execution.adapters.mock_broker import MockBroker
from futur3.execution.adapters.prop_aware_mock_broker import PropAwareMockBroker
from futur3.execution.broker import Order, OrderType, Side
from futur3.execution.risk_manager import multiplier_for

_CONTRACT = "ESM26"
_MULT = multiplier_for(ContractSymbol(_CONTRACT))  # $/point for ESM26
_HUGE = Decimal("100000000")  # effectively-disabled limit


# ============================================================================
# Helpers
# ============================================================================


def _broker(
    *,
    dll: Decimal = _HUGE,
    mll: Decimal = _HUGE,
    max_contracts: int = 10,
    starting_equity: Decimal = Decimal("50000"),
    initial_equity_peak: Decimal | None = None,
) -> PropAwareMockBroker:
    return PropAwareMockBroker(
        daily_loss_limit=dll,
        trailing_max_loss_limit=mll,
        max_contracts=max_contracts,
        starting_equity=starting_equity,
        initial_equity_peak=initial_equity_peak,
    )


def _fill(b: PropAwareMockBroker, *, side: Side, quantity: int, price: str) -> None:
    """Place + fully fill a MKT order (drives positions + realized PnL)."""
    order = Order(
        contract=ContractSymbol(_CONTRACT),
        side=side,
        quantity=quantity,
        order_type=OrderType.MKT,
    )
    oid = asyncio.run(b.place_order(order))
    b.fill_order(oid, Decimal(price))


def _allowed(b: PropAwareMockBroker, *, side: Side, quantity: int) -> tuple[bool, str | None]:
    return b.is_trade_allowed(ContractSymbol(_CONTRACT), side, quantity)


# ============================================================================
# Construction
# ============================================================================


class TestConstruction:
    def test_valid_defaults(self) -> None:
        b = _broker(dll=Decimal("1000"), mll=Decimal("2000"), max_contracts=1)
        assert b.broker_id == "prop-mock"
        assert b.max_contracts == 1
        assert b.daily_loss_limit == Decimal("1000")
        assert b.trailing_max_loss_limit == Decimal("2000")
        assert b.equity_peak == b.starting_equity
        assert b.halt_reason is None

    def test_is_subclass_of_mock_broker(self) -> None:
        assert issubclass(PropAwareMockBroker, MockBroker)

    @pytest.mark.parametrize("bad", [Decimal("0"), Decimal("-1")])
    def test_nonpositive_dll_raises(self, bad: Decimal) -> None:
        with pytest.raises(ValueError, match="daily_loss_limit must be > 0"):
            _broker(dll=bad)

    @pytest.mark.parametrize("bad", [Decimal("0"), Decimal("-1")])
    def test_nonpositive_mll_raises(self, bad: Decimal) -> None:
        with pytest.raises(ValueError, match="trailing_max_loss_limit must be > 0"):
            _broker(mll=bad)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_max_contracts_below_one_raises(self, bad: int) -> None:
        with pytest.raises(ValueError, match="max_contracts must be >= 1"):
            _broker(max_contracts=bad)

    def test_initial_peak_below_starting_raises(self) -> None:
        with pytest.raises(ValueError, match="initial_equity_peak"):
            _broker(starting_equity=Decimal("50000"), initial_equity_peak=Decimal("49999"))

    def test_initial_peak_seeds_trailing_high_water(self) -> None:
        b = _broker(starting_equity=Decimal("50000"), initial_equity_peak=Decimal("55000"))
        assert b.equity_peak == Decimal("55000")

    def test_base_starting_equity_validation_still_fires(self) -> None:
        with pytest.raises(ValueError, match="starting_equity must be > 0"):
            _broker(starting_equity=Decimal("0"))


# ============================================================================
# Position cap (on the RESULTING position)
# ============================================================================


class TestPositionCap:
    def test_fresh_entry_within_cap_allowed(self) -> None:
        b = _broker(max_contracts=1)
        assert _allowed(b, side=Side.BUY, quantity=1) == (True, None)

    def test_fresh_entry_over_cap_blocked(self) -> None:
        b = _broker(max_contracts=1)
        ok, reason = _allowed(b, side=Side.BUY, quantity=2)
        assert ok is False
        assert reason is not None and "max_contracts" in reason

    def test_short_entry_over_cap_blocked(self) -> None:
        b = _broker(max_contracts=1)
        ok, reason = _allowed(b, side=Side.SELL, quantity=2)
        assert ok is False
        assert reason is not None and "max_contracts" in reason

    def test_add_that_breaches_cap_blocked(self) -> None:
        b = _broker(max_contracts=1)
        _fill(b, side=Side.BUY, quantity=1, price="5000")  # now long 1 (at cap)
        ok, reason = _allowed(b, side=Side.BUY, quantity=1)  # would make 2
        assert ok is False
        assert reason is not None and "max_contracts" in reason


# ============================================================================
# Daily Loss Limit
# ============================================================================


class TestDailyLossLimit:
    def test_breach_blocks_new_entry_and_latches(self) -> None:
        # 50-point loss on 1 lot => realized = -50 * mult. Set DLL just under it.
        dll = Decimal(50) * _MULT - Decimal("1")
        b = _broker(dll=dll, max_contracts=1)
        _fill(b, side=Side.BUY, quantity=1, price="5000")
        _fill(b, side=Side.SELL, quantity=1, price="4950")  # realize -50 * mult, now flat
        ok, reason = _allowed(b, side=Side.BUY, quantity=1)
        assert ok is False
        assert reason is not None and "DLL" in reason
        assert b.halt_reason is not None and "DLL" in b.halt_reason

    def test_small_loss_within_dll_allowed(self) -> None:
        b = _broker(dll=_HUGE, max_contracts=1)
        _fill(b, side=Side.BUY, quantity=1, price="5000")
        _fill(b, side=Side.SELL, quantity=1, price="4999")  # tiny realized loss
        assert _allowed(b, side=Side.BUY, quantity=1) == (True, None)

    def test_reducing_order_allowed_after_dll_halt(self) -> None:
        dll = Decimal("1")  # any real loss trips it
        b = _broker(dll=dll, max_contracts=10)
        _fill(b, side=Side.BUY, quantity=2, price="5000")  # long 2
        _fill(b, side=Side.SELL, quantity=1, price="4000")  # realize big loss, hold 1
        # The first increasing order detects the breach + latches the halt.
        inc_ok, inc_reason = _allowed(b, side=Side.BUY, quantity=1)
        assert inc_ok is False
        assert inc_reason is not None and "DLL" in inc_reason
        assert b.halt_reason is not None  # now latched
        # ...but closing the remaining long is ALWAYS permitted, even when halted.
        assert _allowed(b, side=Side.SELL, quantity=1) == (True, None)


# ============================================================================
# Trailing Max Loss Limit
# ============================================================================


class TestTrailingMLL:
    def test_seeded_peak_drawdown_blocks(self) -> None:
        # Peaked at 55000, now sitting at 50000 starting => 5000 in drawdown.
        b = _broker(
            mll=Decimal("4000"),
            starting_equity=Decimal("50000"),
            initial_equity_peak=Decimal("55000"),
        )
        ok, reason = _allowed(b, side=Side.BUY, quantity=1)
        assert ok is False
        assert reason is not None and "MLL" in reason

    def test_seeded_peak_within_mll_allowed(self) -> None:
        b = _broker(
            mll=Decimal("6000"),  # 5000 drawdown < 6000
            starting_equity=Decimal("50000"),
            initial_equity_peak=Decimal("55000"),
        )
        assert _allowed(b, side=Side.BUY, quantity=1) == (True, None)

    def test_peak_rises_via_account_metrics_then_drawdown_blocks(self) -> None:
        gain = Decimal(100) * _MULT  # 100-point round trip on 1 lot
        b = _broker(mll=gain - Decimal("1"), max_contracts=10, starting_equity=Decimal("50000"))
        # Win: realized +gain, equity 50000+gain.
        _fill(b, side=Side.BUY, quantity=1, price="5000")
        _fill(b, side=Side.SELL, quantity=1, price="5100")
        # Engine polls metrics each bar -> registers the new high-water mark.
        metrics = asyncio.run(b.get_account_metrics())
        assert b.equity_peak == metrics.equity == Decimal("50000") + gain
        # Give it all back: realized back to 0, equity 50000 => drawdown == gain.
        _fill(b, side=Side.BUY, quantity=1, price="5100")
        _fill(b, side=Side.SELL, quantity=1, price="5000")
        ok, reason = _allowed(b, side=Side.BUY, quantity=1)
        assert ok is False
        assert reason is not None and "MLL" in reason


# ============================================================================
# Latch + reset
# ============================================================================


class TestLatchAndReset:
    def test_halt_persists_across_calls(self) -> None:
        b = _broker(dll=Decimal("1"), max_contracts=1)
        _fill(b, side=Side.BUY, quantity=1, price="5000")
        _fill(b, side=Side.SELL, quantity=1, price="4000")  # breach
        assert _allowed(b, side=Side.BUY, quantity=1)[0] is False
        # Second attempt still blocked (latched, not re-evaluated).
        ok, reason = _allowed(b, side=Side.BUY, quantity=1)
        assert ok is False
        assert reason is not None and "halted" in reason

    def test_reset_clears_halt_and_peak(self) -> None:
        b = _broker(
            dll=Decimal("1"),
            max_contracts=1,
            starting_equity=Decimal("50000"),
            initial_equity_peak=Decimal("55000"),
        )
        _fill(b, side=Side.BUY, quantity=1, price="5000")
        _fill(b, side=Side.SELL, quantity=1, price="4000")  # breach + latch
        _allowed(b, side=Side.BUY, quantity=1)  # trip the latch
        assert b.halt_reason is not None
        b.reset()
        assert b.halt_reason is None
        assert b.equity_peak == Decimal("55000")  # restored to seeded peak


# ============================================================================
# Base-contract preservation
# ============================================================================


class TestBaseContractPreserved:
    def test_loose_limits_allow_normal_trade(self) -> None:
        b = _broker(dll=_HUGE, mll=_HUGE, max_contracts=1000)
        assert _allowed(b, side=Side.BUY, quantity=5) == (True, None)

    def test_base_quantity_validation_still_fires(self) -> None:
        b = _broker()
        ok, reason = _allowed(b, side=Side.BUY, quantity=0)
        assert ok is False
        assert reason is not None and "quantity must be > 0" in reason
