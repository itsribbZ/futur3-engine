"""CMETradingCalendar test suite.

Per the build plan — the CME full-closure trading-day calendar that
unblocks RollCalendarBuilder's T-N business-day math:
- Holiday computation: the 10 computed 2026 closures (pinned, deterministic); floating
  holidays (MLK/Presidents/Memorial/Labor/Thanksgiving); Good Friday (Easter-2); observed-date
  shifting (Sat->Fri, Sun->Mon) incl. the New-Year backward spill onto prior Dec 31; the
  Juneteenth 2022 first-observance boundary.
- Trading-day predicate: weekdays open, weekends closed, holidays closed.
- Business-day arithmetic: add_trading_days forward/backward over weekends + holidays, the
  n==0 identity, next/prev, trading_days_between, inverted-range raise.
- Overrides: extra_holidays, extra_trading_days (incl. forcing a weekend open), contradictory
  override raise.
"""

from __future__ import annotations

from datetime import date

import pytest

from futur3.contracts import CMETradingCalendar as _PkgCMETradingCalendar
from futur3.contracts.trading_calendar import (
    CMETradingCalendar,
    TradingCalendarError,
    _holidays_in_year,
)

# The full computed CME closure set for 2026 (observed dates). July 4 is a Saturday -> Jul 3.
_HOLIDAYS_2026 = frozenset(
    {
        date(2026, 1, 1),  # New Year's Day (Thu)
        date(2026, 1, 19),  # MLK Jr. Day (3rd Mon)
        date(2026, 2, 16),  # Washington's Birthday (3rd Mon)
        date(2026, 4, 3),  # Good Friday (Easter Sun = Apr 5)
        date(2026, 5, 25),  # Memorial Day (last Mon)
        date(2026, 6, 19),  # Juneteenth (Fri)
        date(2026, 7, 3),  # Independence Day observed (Jul 4 is Sat)
        date(2026, 9, 7),  # Labor Day (1st Mon)
        date(2026, 11, 26),  # Thanksgiving (4th Thu)
        date(2026, 12, 25),  # Christmas Day (Fri)
    }
)


def _cal(**kwargs: object) -> CMETradingCalendar:
    return CMETradingCalendar(**kwargs)  # type: ignore[arg-type]


# ============================================================================
# TestW1_1_HolidayComputation
# ============================================================================


class TestW1_1_HolidayComputation:
    @pytest.mark.bitrepro
    def test_full_2026_set_pinned(self) -> None:
        # Pins the computed output bit-for-bit. If the algorithm drifts, this breaks.
        assert _holidays_in_year(2026) == _HOLIDAYS_2026

    def test_floating_holidays(self) -> None:
        cal = _cal()
        assert cal.is_holiday(date(2026, 1, 19))  # MLK 3rd Mon Jan
        assert cal.is_holiday(date(2026, 2, 16))  # Presidents 3rd Mon Feb
        assert cal.is_holiday(date(2026, 5, 25))  # Memorial last Mon May
        assert cal.is_holiday(date(2026, 9, 7))  # Labor 1st Mon Sep
        assert cal.is_holiday(date(2026, 11, 26))  # Thanksgiving 4th Thu Nov

    def test_good_friday(self) -> None:
        # Easter Sunday 2026 = Apr 5 -> Good Friday Apr 3.
        assert _cal().is_holiday(date(2026, 4, 3))

    def test_independence_day_saturday_observed_friday(self) -> None:
        cal = _cal()
        assert cal.is_holiday(date(2026, 7, 3))  # observed (Jul 4 is Sat)
        assert not cal.is_holiday(date(2026, 7, 4))  # the Saturday itself is a weekend
        assert not cal.is_trading_day(date(2026, 7, 3))  # observed closure
        assert not cal.is_trading_day(date(2026, 7, 4))  # weekend

    def test_new_year_saturday_spills_to_prior_dec_31(self) -> None:
        # Jan 1 2022 is a Saturday -> observed Fri Dec 31 2021 (a cross-year spill).
        cal = _cal()
        assert cal.is_holiday(date(2021, 12, 31))
        assert not cal.is_trading_day(date(2021, 12, 31))

    def test_christmas_sunday_observed_monday(self) -> None:
        # Dec 25 2022 is a Sunday -> observed Mon Dec 26 2022.
        assert _cal().is_holiday(date(2022, 12, 26))

    def test_juneteenth_first_observed_2022_not_2021(self) -> None:
        cal = _cal()
        # 2021: Jun 19 is Sat -> observed Fri Jun 18, but CME did NOT close -> trading day.
        assert cal.is_trading_day(date(2021, 6, 18))
        assert not cal.is_holiday(date(2021, 6, 18))
        # 2022: Jun 19 is Sun -> observed Mon Jun 20, CME's first Juneteenth closure.
        assert cal.is_holiday(date(2022, 6, 20))


# ============================================================================
# TestW1_1_TradingDay
# ============================================================================


class TestW1_1_TradingDay:
    def test_ordinary_weekday_is_trading(self) -> None:
        assert _cal().is_trading_day(date(2026, 1, 6))  # Tuesday, no holiday

    def test_weekend_is_not_trading(self) -> None:
        cal = _cal()
        assert not cal.is_trading_day(date(2026, 1, 3))  # Saturday
        assert not cal.is_trading_day(date(2026, 1, 4))  # Sunday

    def test_holiday_weekday_is_not_trading(self) -> None:
        assert not _cal().is_trading_day(date(2026, 1, 19))  # MLK Monday


# ============================================================================
# TestW1_1_Arithmetic
# ============================================================================


class TestW1_1_Arithmetic:
    def test_forward_over_weekend(self) -> None:
        # Fri Jan 9 2026 + 1 trading day -> Mon Jan 12 (skip Sat/Sun).
        assert _cal().add_trading_days(date(2026, 1, 9), 1) == date(2026, 1, 12)

    def test_backward_over_holiday_and_weekend(self) -> None:
        # Tue Jan 20 2026 - 1 -> Fri Jan 16 (skip MLK Mon Jan 19 + weekend).
        assert _cal().add_trading_days(date(2026, 1, 20), -1) == date(2026, 1, 16)

    def test_t_minus_5_trading_days(self) -> None:
        # Fri Jan 16 2026 - 5 trading days -> Fri Jan 9 (Thu15,Wed14,Tue13,Mon12,Fri9).
        assert _cal().add_trading_days(date(2026, 1, 16), -5) == date(2026, 1, 9)

    def test_zero_is_identity(self) -> None:
        assert _cal().add_trading_days(date(2026, 1, 19), 0) == date(2026, 1, 19)

    def test_next_and_prev(self) -> None:
        cal = _cal()
        assert cal.next_trading_day(date(2026, 1, 9)) == date(2026, 1, 12)
        assert cal.prev_trading_day(date(2026, 1, 12)) == date(2026, 1, 9)

    def test_between_full_week(self) -> None:
        # Mon Jan 12 .. Fri Jan 16 2026, no holiday -> 5 trading days (inclusive).
        assert _cal().trading_days_between(date(2026, 1, 12), date(2026, 1, 16)) == 5

    def test_between_week_with_holiday(self) -> None:
        # Mon Jan 19 (MLK) .. Fri Jan 23 2026 -> 4 trading days.
        assert _cal().trading_days_between(date(2026, 1, 19), date(2026, 1, 23)) == 4

    def test_between_inverted_raises(self) -> None:
        with pytest.raises(TradingCalendarError, match="after end"):
            _cal().trading_days_between(date(2026, 1, 16), date(2026, 1, 12))


# ============================================================================
# TestW1_1_Overrides
# ============================================================================


class TestW1_1_Overrides:
    def test_extra_holiday_closes_a_weekday(self) -> None:
        cal = _cal(extra_holidays=[date(2026, 1, 6)])  # force a normal Tuesday closed
        assert cal.is_holiday(date(2026, 1, 6))
        assert not cal.is_trading_day(date(2026, 1, 6))

    def test_extra_trading_day_opens_a_holiday(self) -> None:
        cal = _cal(extra_trading_days=[date(2026, 1, 19)])  # force MLK open
        assert cal.is_trading_day(date(2026, 1, 19))
        assert not cal.is_holiday(date(2026, 1, 19))

    def test_extra_trading_day_opens_a_weekend(self) -> None:
        # 24/7 crypto-regime: force a Saturday open.
        cal = _cal(extra_trading_days=[date(2026, 1, 3)])
        assert cal.is_trading_day(date(2026, 1, 3))

    def test_contradictory_override_raises(self) -> None:
        with pytest.raises(TradingCalendarError, match="BOTH"):
            _cal(
                extra_holidays=[date(2026, 1, 6)],
                extra_trading_days=[date(2026, 1, 6)],
            )


def test_exported_from_contracts_package() -> None:
    assert _PkgCMETradingCalendar is CMETradingCalendar
