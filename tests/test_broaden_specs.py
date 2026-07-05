"""Broaden wave 1: the broaden-universe roots (equity YM/RTY, FX 6E/6A, rates ZN/ZB) are fully wired.

Verifies each new root has multiplier + tick + margin registry entries (so RiskManager / slippage
don't raise), a quarterly roll cycle + spec, and a buildable roll calendar whose dates are sane
(tick value = tick x multiplier reconciles vs the CME-verified spec; FX LTD lands in the contract
month with no FND, rates LTD in-month with FND in the prior month).
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from futur3.data.types import ContractSymbol
from futur3.execution.risk_manager import (
    CONTRACT_INITIAL_MARGIN,
    CONTRACT_MULTIPLIER,
    multiplier_for,
)
from futur3.execution.roll_executor import ROOT_CYCLE, RollCalendarBuilder
from futur3.execution.slippage import CONTRACT_TICK_SIZE, tick_size_for

_NEW_ROOTS = ("YM", "RTY", "6E", "6A", "ZN", "ZB")
_TICK_VALUE = {  # CME-verified tick value ($) = tick size x point multiplier
    "YM": "5",
    "RTY": "5",
    "6E": "6.25",
    "6A": "5",
    "ZN": "15.625",
    "ZB": "31.25",
}


class TestBroadenSpecsWired:
    @pytest.mark.parametrize("root", _NEW_ROOTS)
    def test_present_in_all_three_registries(self, root: str) -> None:
        assert root in CONTRACT_MULTIPLIER
        assert root in CONTRACT_TICK_SIZE
        assert root in CONTRACT_INITIAL_MARGIN

    @pytest.mark.parametrize("root", _NEW_ROOTS)
    def test_tick_value_reconciles(self, root: str) -> None:
        assert CONTRACT_TICK_SIZE[root] * CONTRACT_MULTIPLIER[root] == Decimal(_TICK_VALUE[root])

    @pytest.mark.parametrize("root", _NEW_ROOTS)
    def test_lookup_helpers_resolve(self, root: str) -> None:
        contract = ContractSymbol(f"{root}H26")
        assert multiplier_for(contract) > 0
        assert tick_size_for(contract) > 0

    @pytest.mark.parametrize("root", _NEW_ROOTS)
    def test_quarterly_cycle(self, root: str) -> None:
        assert ROOT_CYCLE[root] == ("H", "M", "U", "Z")

    @pytest.mark.parametrize("root", _NEW_ROOTS)
    def test_roll_calendar_builds_with_sane_ordering(self, root: str) -> None:
        entries = RollCalendarBuilder().build_cycle(root, 2024, 2025)
        assert len(entries) == 2 * 4  # 4 quarterly months x 2 years
        for e in entries:
            assert e.roll_target <= e.roll_deadline <= e.ltd_date  # roll before deadline before LTD

    def test_fx_ltd_in_contract_month_no_fnd(self) -> None:
        h26 = RollCalendarBuilder().build_cycle("6E", 2026, 2026)[0]  # 6EH26 (March)
        assert h26.ltd_date.year == 2026 and h26.ltd_date.month == 3
        assert h26.fnd_date is None  # FX cash-rolled on volume -> no FND splice

    def test_rates_ltd_in_month_fnd_in_prior_month(self) -> None:
        h26 = RollCalendarBuilder().build_cycle("ZN", 2026, 2026)[0]  # ZNH26 (March)
        assert h26.ltd_date.month == 3  # 7 trading days before end of the delivery month
        assert h26.fnd_date is not None and h26.fnd_date.month == 2  # last biz day of February
