"""futur3.contracts.trading_calendar — CME trading-day calendar.

The CME-holiday calendar `RollCalendarBuilder` needs —
deriving `roll_target` / `roll_deadline` as T-N *trading* days from LTD/FND requires
business-day arithmetic that skips weekends AND exchange holidays. Faking business-day
math without holidays would be a silent-bug risk, so this is the real calendar.

Pure stdlib, deterministic (no external data dependency), bit-reproducible: the
full-closure holiday set is COMPUTED algorithmically (the US federal holidays CME observes
plus Good Friday), correct for any year with zero maintenance and no giant date literal.

Models the CME Globex FULL-CLOSURE holidays — the days with no settlement, which is exactly
what roll-date T-N arithmetic needs. NOT modeled (documented limitation — stated honestly, never
silently approximate as exact):
  - Early-close / shortened-session days (day after Thanksgiving, Christmas Eve, ...): treated
    as full trading days, because a settlement price still prints — correct for roll-date math.
  - Product-specific holiday variation (energy vs equity-index vs metals occasionally differ):
    the common full-closure set is used; inject `extra_holidays` for any product-specific
    closure, or `extra_trading_days` to force a computed holiday (or a weekend) back to open.
"""

from __future__ import annotations

import calendar
from collections.abc import Iterable
from datetime import date, timedelta
from functools import cache
from typing import Final

# date.weekday(): Mon=0 .. Sat=5, Sun=6.
_SATURDAY: Final[int] = calendar.SATURDAY
_JUNETEENTH_FIRST_CME_YEAR: Final[int] = 2022  # CME's first observed Juneteenth closure


class TradingCalendarError(Exception):
    """Trading-calendar contract violation (inverted date range, contradictory overrides, ...)."""


def _observed(holiday: date) -> date:
    """US federal observed-date rule: Sat -> prior Fri, Sun -> following Mon, else unchanged."""
    weekday = holiday.weekday()
    if weekday == calendar.SATURDAY:
        return holiday - timedelta(days=1)
    if weekday == calendar.SUNDAY:
        return holiday + timedelta(days=1)
    return holiday


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The n-th (1-based) `weekday` in `month` — e.g. 3rd Monday of January."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return date(year, month, 1 + offset + (n - 1) * 7)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """The last `weekday` in `month` — e.g. last Monday of May (Memorial Day)."""
    last_dom = calendar.monthrange(year, month)[1]
    last = date(year, month, last_dom)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _easter_sunday(year: int) -> date:
    """Gregorian Easter Sunday — Anonymous Gregorian algorithm (Meeus/Jones/Butcher)."""
    golden = year % 19
    century, year_of_century = divmod(year, 100)
    century_div4, century_mod4 = divmod(century, 4)
    correction = (century + 8) // 25
    sunday_letter = (century - correction + 1) // 3
    epact = (19 * golden + century - century_div4 - sunday_letter + 15) % 30
    yoc_div4, yoc_mod4 = divmod(year_of_century, 4)
    offset = (32 + 2 * century_mod4 + 2 * yoc_div4 - epact - yoc_mod4) % 7
    month_seed = (golden + 11 * epact + 22 * offset) // 451
    month = (epact + offset - 7 * month_seed + 114) // 31
    day = ((epact + offset - 7 * month_seed + 114) % 31) + 1
    return date(year, month, day)


@cache
def _anchored_holidays(year: int) -> frozenset[date]:
    """Observed full-closure holiday dates *belonging to* `year`.

    A New Year observed backward onto the prior Dec 31 carries a different calendar year, so
    callers union with the neighbouring year (see `_holidays_in_year`)."""
    days: set[date] = {
        _observed(date(year, 1, 1)),  # New Year's Day
        _nth_weekday(year, 1, calendar.MONDAY, 3),  # MLK Jr. Day (3rd Mon Jan)
        _nth_weekday(year, 2, calendar.MONDAY, 3),  # Washington's Birthday (3rd Mon Feb)
        _easter_sunday(year) - timedelta(days=2),  # Good Friday
        _last_weekday(year, 5, calendar.MONDAY),  # Memorial Day (last Mon May)
        _observed(date(year, 7, 4)),  # Independence Day
        _nth_weekday(year, 9, calendar.MONDAY, 1),  # Labor Day (1st Mon Sep)
        _nth_weekday(year, 11, calendar.THURSDAY, 4),  # Thanksgiving (4th Thu Nov)
        _observed(date(year, 12, 25)),  # Christmas Day
    }
    if year >= _JUNETEENTH_FIRST_CME_YEAR:
        days.add(_observed(date(year, 6, 19)))  # Juneteenth (CME observes from 2022)
    return frozenset(days)


@cache
def _holidays_in_year(year: int) -> frozenset[date]:
    """All observed full-closure holidays whose *calendar year* is `year`.

    Unions this year's anchored holidays with next year's (to catch a New Year's Day observed
    backward onto Dec 31 of `year`), then filters to dates that actually land in `year`."""
    candidates = _anchored_holidays(year) | _anchored_holidays(year + 1)
    return frozenset(d for d in candidates if d.year == year)


class CMETradingCalendar:
    """CME Globex trading-day calendar for roll-date business-day arithmetic.

    Weekends are closed; computed full-closure holidays are closed. `extra_holidays` adds
    product-specific closures; `extra_trading_days` forces a date open — overriding both a
    computed holiday and a weekend (e.g. a post-2026-05-29 24/7 crypto-regime weekend day).
    Both override sets win over the computed calendar (fail-loud: explicit operator intent is never
    silently ignored). A date declared in both override sets raises at construction."""

    def __init__(
        self,
        *,
        extra_holidays: Iterable[date] = (),
        extra_trading_days: Iterable[date] = (),
    ) -> None:
        self._extra_holidays: frozenset[date] = frozenset(extra_holidays)
        self._extra_trading: frozenset[date] = frozenset(extra_trading_days)
        contradictory = self._extra_holidays & self._extra_trading
        if contradictory:
            raise TradingCalendarError(
                "dates declared as BOTH extra_holiday and extra_trading_day: "
                f"{sorted(contradictory)}"
            )

    def is_holiday(self, day: date) -> bool:
        """True iff `day` is a *weekday* CME full-closure holiday (weekends are not holidays)."""
        if day in self._extra_trading:
            return False
        if day in self._extra_holidays:
            return True
        if day.weekday() >= _SATURDAY:
            return False
        return day in _holidays_in_year(day.year)

    def is_trading_day(self, day: date) -> bool:
        """True iff CME settles on `day` (not a weekend, not a holiday; overrides applied)."""
        if day in self._extra_trading:
            return True
        if day in self._extra_holidays:
            return False
        if day.weekday() >= _SATURDAY:
            return False
        return day not in _holidays_in_year(day.year)

    def add_trading_days(self, start: date, n: int) -> date:
        """`start` shifted by `n` trading days (n>0 forward, n<0 backward, n==0 -> `start`).

        Counts trading days *landed on*; `start` itself is never counted. So a T-5 roll target
        five trading days before the last trading day is `add_trading_days(ltd, -5)`."""
        if n == 0:
            return start
        step = 1 if n > 0 else -1
        remaining = abs(n)
        # Fail-loud: a pathologically over-broad `extra_holidays` must fail loud, never hang.
        max_calendar_steps = abs(n) * 7 + 30
        steps = 0
        current = start
        while remaining > 0:
            current += timedelta(days=step)
            steps += 1
            if steps > max_calendar_steps:
                raise TradingCalendarError(
                    f"could not find {n} trading days from {start} within "
                    f"{max_calendar_steps} calendar days — over-broad extra_holidays?"
                )
            if self.is_trading_day(current):
                remaining -= 1
        return current

    def next_trading_day(self, day: date) -> date:
        """The first trading day strictly after `day`."""
        return self.add_trading_days(day, 1)

    def prev_trading_day(self, day: date) -> date:
        """The last trading day strictly before `day`."""
        return self.add_trading_days(day, -1)

    def trading_days_between(self, start: date, end: date) -> int:
        """Count of trading days in the closed interval [start, end]. Raises if start > end."""
        if start > end:
            raise TradingCalendarError(f"start {start} is after end {end}")
        count = 0
        current = start
        while current <= end:
            if self.is_trading_day(current):
                count += 1
            current += timedelta(days=1)
        return count
