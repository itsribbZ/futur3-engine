"""FixedSizeRiskManager test suite.

Verifies the fixed-deployable-size sizer: it pins position size to a constant
contract count (capped by affordability), preserves no-edge -> 0, keeps the base
diagnostics, and crucially is INVARIANT to the price level (the bug it fixes: the
base sizer wanders 1-3 MNQ across the cache's NQ price range).

Fixture-only. Uses MNQ (mult 2, margin $2,280) for realistic prop-scale numbers.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from futur3.data.types import ContractSymbol
from futur3.execution.fixed_size_risk_manager import FixedSizeRiskManager
from futur3.execution.risk_manager import RiskManager

_MNQ = ContractSymbol("MNQM26")
_EDGE = Decimal("100")  # the OvernightDriftStrategy full-Kelly magnitude
_EQUITY = Decimal("50000")  # a $50k Combine-scale account
_PRICE = Decimal("21000")


class TestConstruction:
    def test_is_subclass_of_risk_manager(self) -> None:
        assert issubclass(FixedSizeRiskManager, RiskManager)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_below_one_raises(self, bad: int) -> None:
        with pytest.raises(ValueError, match="contracts must be >= 1"):
            FixedSizeRiskManager(contracts=bad)

    def test_exposes_fixed_contracts(self) -> None:
        assert FixedSizeRiskManager(contracts=1).fixed_contracts == 1


class TestFixedSizing:
    def test_pins_to_one_when_base_wants_more(self) -> None:
        base = RiskManager().size_position(_MNQ, _EDGE, _EQUITY, _PRICE)
        assert base.contracts > 1  # the un-fixed sizer wants several MNQ
        fixed = FixedSizeRiskManager(contracts=1).size_position(_MNQ, _EDGE, _EQUITY, _PRICE)
        assert fixed.contracts == 1
        # base diagnostics preserved (audit trail intact)
        assert fixed.leverage_contracts == base.leverage_contracts
        assert fixed.margin_contracts == base.margin_contracts
        assert fixed.kelly_contracts == base.kelly_contracts

    def test_no_edge_sizes_zero(self) -> None:
        d = FixedSizeRiskManager(contracts=1).size_position(_MNQ, Decimal("0"), _EQUITY, _PRICE)
        assert d.contracts == 0

    def test_capped_by_affordability(self) -> None:
        # Ask for an absurd 100 contracts; margin/leverage cap well below that.
        base = RiskManager().size_position(_MNQ, _EDGE, _EQUITY, _PRICE)
        d = FixedSizeRiskManager(contracts=100).size_position(_MNQ, _EDGE, _EQUITY, _PRICE)
        assert d.contracts == min(base.margin_contracts, base.leverage_contracts)
        assert d.contracts < 100

    def test_price_invariance_pins_one(self) -> None:
        # The bug FixedSize fixes: the base sizer wanders with price; fixed stays 1.
        sizer = FixedSizeRiskManager(contracts=1)
        base_counts = set()
        for px in ("12000", "15000", "18000", "22000"):
            base_counts.add(RiskManager().size_position(_MNQ, _EDGE, _EQUITY, Decimal(px)).contracts)
            assert sizer.size_position(_MNQ, _EDGE, _EQUITY, Decimal(px)).contracts == 1
        assert len(base_counts) > 1  # confirm the base actually does wander across price
