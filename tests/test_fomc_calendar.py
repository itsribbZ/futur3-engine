"""Structural validation of the committed FOMC statement-date artifact.

Acquired from the Fed's published calendar. A small-model parse can
silently misread a date, so these checks make any such error a LOUD failure: scheduled
FOMC statements fall on a weekday (Tue/Wed/Thu), 8 per year (2020 = 7, its March meeting
preempted by the COVID emergency), sorted + unique, every year 2010-2027 covered. A stray
date only adds noise to the ~143-event mask, but it is a real-money input, so it is locked.
"""

from __future__ import annotations

import json
import pathlib
from datetime import date

_ARTIFACT = pathlib.Path(__file__).resolve().parents[1] / "research" / "fomc_statement_dates.json"
_RAW = json.loads(_ARTIFACT.read_text(encoding="utf-8"))["statement_dates"]
_DATES = [date.fromisoformat(s) for s in _RAW]

_TUE, _WED, _THU = 1, 2, 3  # date.weekday(): Mon=0 .. Sun=6
_FIRST_YEAR, _LAST_YEAR = 2010, 2027
_EXPECTED_TOTAL = 143  # 8/yr x 18yr (2010-2027) minus 2020's preempted March meeting


class TestFomcCalendarArtifact:
    def test_total_count(self) -> None:
        assert len(_DATES) == _EXPECTED_TOTAL

    def test_all_dates_in_range(self) -> None:
        assert min(_DATES) >= date(_FIRST_YEAR, 1, 1)
        assert max(_DATES) <= date(_LAST_YEAR, 12, 31)

    def test_sorted_and_unique(self) -> None:
        assert sorted(_DATES) == _DATES
        assert len(set(_DATES)) == len(_DATES)

    def test_all_statements_are_tue_wed_or_thu(self) -> None:
        # statements release on the meeting's last day = Tue/Wed, or Thu when the meeting is Wed-Thu
        # (Sep-2012 QE3; the Nov 2018/2020/2024 meetings shifted a day to clear Election Day).
        # A Mon/Fri or weekend would be a parse error. (Emergency actions are excluded.)
        offenders = [d.isoformat() for d in _DATES if d.weekday() not in (_TUE, _WED, _THU)]
        assert offenders == []

    def test_every_year_covered(self) -> None:
        years = {d.year for d in _DATES}
        assert years == set(range(_FIRST_YEAR, _LAST_YEAR + 1))

    def test_per_year_counts(self) -> None:
        # 8 scheduled meetings/year is the Fed standard; 2020 = 7 (March preempted).
        for year in range(_FIRST_YEAR, _LAST_YEAR + 1):
            n = sum(1 for d in _DATES if d.year == year)
            expected = 7 if year == 2020 else 8
            assert n == expected, f"{year}: expected {expected} scheduled statements, got {n}"
