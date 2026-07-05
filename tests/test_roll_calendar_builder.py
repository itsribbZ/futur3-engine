"""RollCalendarBuilder test suite.

Per the build plan — derives RollCalendarEntry roll windows from CME
expiry rules using CMETradingCalendar. Worked examples verified against the
CME rulebook (NYMEX Ch.200 / COMEX Ch.113 / CME Ch.350):
- ES/NQ: 3rd Friday LTD, cash-settled (FND None), T-5/T-2 windows.
- CL: 25th-anchored LTD — both the non-business branch (CLF26: Dec 25 = Christmas) and the
  plain-weekday branch (CLH26: Feb 25); FND = LTD + 2 BD; deadline clamped <= FND-1.
- GC: FND-anchored (last BD of preceding month), LTD = 3rd-to-last BD of delivery month.
- MBT/MET: last-Friday LTD, RTH -> 24/7 regime flip at the 2026-05-29 launch.
- Builder: unknown-root raise, build_cycle / build_static_calendar, inverted-range raise,
  determinism, and the RollCalendarEntry invariant holding across a multi-year sweep.
"""

from __future__ import annotations

from datetime import date

import pytest

from futur3.contracts import CMETradingCalendar
from futur3.data.types import ContractSymbol
from futur3.execution import RollCalendarBuilder as _PkgRollCalendarBuilder
from futur3.execution.roll_executor import (
    RollCalendarBuilder,
    RollExecutorError,
)


def _b() -> RollCalendarBuilder:
    return RollCalendarBuilder()


# ============================================================================
# TestW1_2_EquityIndex (ES / NQ — 3rd Friday, cash-settled)
# ============================================================================


class TestW1_2_EquityIndex:
    def test_esh26(self) -> None:
        e = _b().build("ESH26")
        assert e.ltd_date == date(2026, 3, 20)  # 3rd Friday of March 2026
        assert e.fnd_date is None  # cash-settled (SOQ)
        assert e.roll_target == date(2026, 3, 13)  # LTD - 5 trading days
        assert e.roll_deadline == date(2026, 3, 18)  # LTD - 2 trading days
        assert e.regime == "RTH"
        assert e.back_symbol == "ESM26"  # quarterly H(Mar) -> M(Jun)

    def test_nqh26_same_third_friday(self) -> None:
        e = _b().build("NQH26")
        assert e.ltd_date == date(2026, 3, 20)
        assert e.back_symbol == "NQM26"


# ============================================================================
# TestW1_2_CrudeOil (CL — 25th-anchored, physical, FND = LTD+2)
# ============================================================================


class TestW1_2_CrudeOil:
    def test_clf26_nonbusiness_25th_branch(self) -> None:
        # Anchor = 25 Dec 2025 (Christmas, non-business) -> last BD before = Dec 24 -> -3 BD.
        e = _b().build("CLF26")
        assert e.ltd_date == date(2025, 12, 19)
        assert e.fnd_date == date(2025, 12, 23)  # LTD + 2 trading days
        assert e.roll_target == date(2025, 12, 12)  # LTD - 5
        assert e.roll_deadline == date(2025, 12, 16)  # min(LTD-3, FND-1)
        assert e.regime == "RTH"
        assert e.back_symbol == "CLG26"

    def test_clh26_business_25th_branch(self) -> None:
        # Anchor = 25 Feb 2026 (Wed, a normal business day) -> exercises the happy-path branch.
        e = _b().build("CLH26")
        assert e.ltd_date == date(2026, 2, 20)  # Feb 25 - 3 trading days
        assert e.fnd_date == date(2026, 2, 24)  # LTD + 2 trading days
        assert e.back_symbol == "CLJ26"

    def test_fnd_is_two_trading_days_after_ltd(self) -> None:
        e = _b().build("CLM26")
        assert e.fnd_date is not None
        assert CMETradingCalendar().add_trading_days(e.ltd_date, 2) == e.fnd_date


# ============================================================================
# TestW1_2_Gold (GC — FND-anchored)
# ============================================================================


class TestW1_2_Gold:
    def test_gcg26(self) -> None:
        e = _b().build("GCG26")
        assert e.ltd_date == date(2026, 2, 25)  # 3rd-to-last BD of Feb 2026
        assert e.fnd_date == date(2026, 1, 30)  # last BD of Jan 2026 (Jan 31 = Sat)
        assert e.roll_target == date(2026, 1, 21)  # FND - 7 trading days
        assert e.roll_deadline == date(2026, 1, 27)  # FND - 3 trading days
        assert e.regime == "RTH"
        assert e.back_symbol == "GCJ26"

    def test_roll_anchors_on_fnd_before_ltd(self) -> None:
        # Gold's FND (prior month) precedes its LTD (delivery month) — the window sits before FND.
        e = _b().build("GCG26")
        assert e.fnd_date is not None
        assert e.roll_deadline < e.fnd_date <= e.ltd_date


# ============================================================================
# TestW1_2_Crypto (MBT / MET — last Friday, 24/7 regime flip)
# ============================================================================


class TestW1_2_Crypto:
    def test_mbtf26_rth(self) -> None:
        e = _b().build("MBTF26")
        assert e.ltd_date == date(2026, 1, 30)  # last Friday of Jan 2026
        assert e.fnd_date is None  # cash-settled (BRR)
        assert e.roll_target == date(2026, 1, 21)  # LTD - 7
        assert e.roll_deadline == date(2026, 1, 27)  # LTD - 3
        assert e.regime == "RTH"  # before 2026-05-29
        assert e.back_symbol == "MBTG26"

    def test_mbtm26_after_24_7_launch(self) -> None:
        e = _b().build("MBTM26")
        assert e.ltd_date == date(2026, 6, 26)  # last Friday of June 2026
        assert e.regime == "24/7"  # on/after 2026-05-29
        assert e.back_symbol == "MBTN26"

    def test_metf26(self) -> None:
        e = _b().build("METF26")
        assert e.ltd_date == date(2026, 1, 30)
        assert e.fnd_date is None
        assert e.back_symbol == "METG26"


# ============================================================================
# TestW1_2_Builder (cycle / determinism / guards)
# ============================================================================


class TestW1_2_Builder:
    def test_unknown_root_raises(self) -> None:
        with pytest.raises(RollExecutorError, match="no roll spec"):
            _b().build("ZZF26")

    def test_build_cycle_count(self) -> None:
        # GC bi-monthly = 6 codes per year.
        assert len(_b().build_cycle("GC", 2026, 2026)) == 6
        # CL monthly = 12 per year, 2 years = 24.
        assert len(_b().build_cycle("CL", 2026, 2027)) == 24

    def test_build_static_calendar_lookup(self) -> None:
        cal = _b().build_static_calendar("CL", 2026, 2026)
        assert cal.lookup(ContractSymbol("CLM26")) is not None
        assert cal.lookup(ContractSymbol("ESM26")) is None  # wrong root not present

    def test_inverted_year_range_raises(self) -> None:
        with pytest.raises(RollExecutorError, match="start_year"):
            _b().build_cycle("ES", 2027, 2026)

    @pytest.mark.bitrepro
    def test_deterministic(self) -> None:
        assert _b().build("GCG26") == _b().build("GCG26")
        assert _b().build_cycle("CL", 2026, 2026) == _b().build_cycle("CL", 2026, 2026)

    def test_entry_invariant_holds_across_universe(self) -> None:
        # RollCalendarEntry enforces roll_target <= roll_deadline <= ltd_date; if any derived
        # window violated it, build() would raise. Sweep the whole universe to prove it never does.
        b = _b()
        for root in ("ES", "NQ", "CL", "GC", "MBT", "MET"):
            for e in b.build_cycle(root, 2024, 2028):
                assert e.roll_target <= e.roll_deadline <= e.ltd_date


def test_exported_from_execution_package() -> None:
    assert _PkgRollCalendarBuilder is RollCalendarBuilder
