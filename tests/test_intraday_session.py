"""ET cash-session time-of-day test suite (futur3.intraday_session).

Locks the UTC->ET conversion across BOTH DST regimes (the same UTC hour maps to a different ET
wall-clock time in summer vs winter -- the classic bug), the RTH open/close boundaries (open
inclusive, close exclusive), the midnight-wrapping overnight window, the ET-date gotcha (UTC date
!= ET date across midnight), and the naive-datetime guard.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time

import pytest

from futur3.intraday_session import (
    RTH_CLOSE,
    RTH_OPEN,
    IntradaySessionError,
    et_time,
    in_window,
    is_rth,
    session_date,
    to_et,
)

# ============================================================================
# TestConstants - lock the cash-session definition
# ============================================================================


class TestConstants:
    def test_rth_open_close(self) -> None:
        assert time(9, 30) == RTH_OPEN
        assert time(16, 0) == RTH_CLOSE


# ============================================================================
# TestToEt - DST-aware conversion preserving the instant
# ============================================================================


class TestToEt:
    def test_summer_edt_close(self) -> None:
        # EDT (UTC-4): 2023-07-03 20:00 UTC == 16:00 ET
        et = to_et(datetime(2023, 7, 3, 20, 0, tzinfo=UTC))
        assert (et.hour, et.minute) == (16, 0)

    def test_winter_est_close(self) -> None:
        # EST (UTC-5): 2023-01-17 21:00 UTC == 16:00 ET
        et = to_et(datetime(2023, 1, 17, 21, 0, tzinfo=UTC))
        assert (et.hour, et.minute) == (16, 0)

    def test_preserves_instant(self) -> None:
        src = datetime(2023, 6, 15, 13, 45, tzinfo=UTC)
        assert to_et(src).astimezone(UTC) == src

    def test_naive_raises(self) -> None:
        with pytest.raises(IntradaySessionError, match="naive"):
            to_et(datetime(2023, 1, 17, 21, 0))


# ============================================================================
# TestIsRth - boundaries + the DST trap
# ============================================================================


class TestIsRth:
    def test_open_inclusive_summer(self) -> None:
        # 13:30 UTC == 09:30 EDT (open, inclusive)
        assert is_rth(datetime(2023, 7, 3, 13, 30, tzinfo=UTC))

    def test_before_open_excluded(self) -> None:
        # 13:00 UTC == 09:00 EDT
        assert not is_rth(datetime(2023, 7, 3, 13, 0, tzinfo=UTC))

    def test_close_exclusive(self) -> None:
        # 20:00 UTC == 16:00 EDT (close, exclusive)
        assert not is_rth(datetime(2023, 7, 3, 20, 0, tzinfo=UTC))

    def test_just_before_close_included(self) -> None:
        # 19:59 UTC == 15:59 EDT
        assert is_rth(datetime(2023, 7, 3, 19, 59, tzinfo=UTC))

    def test_same_utc_hour_differs_across_dst(self) -> None:
        # 20:00 UTC: summer -> 16:00 ET (closed, exclusive); winter -> 15:00 ET (open)
        assert not is_rth(datetime(2023, 7, 3, 20, 0, tzinfo=UTC))
        assert is_rth(datetime(2023, 1, 17, 20, 0, tzinfo=UTC))


# ============================================================================
# TestInWindow - same-day + midnight-wrapping (overnight)
# ============================================================================


class TestInWindow:
    _OVERNIGHT = (time(22, 0), time(4, 0))  # wraps midnight

    def test_wrapping_window_includes_early_morning(self) -> None:
        # 07:00 UTC == 02:00 EST -> inside 22:00-04:00
        assert in_window(datetime(2023, 1, 17, 7, 0, tzinfo=UTC), *self._OVERNIGHT)

    def test_wrapping_window_includes_late_evening(self) -> None:
        # 2023-01-18 04:00 UTC == 2023-01-17 23:00 EST -> inside 22:00-04:00
        assert in_window(datetime(2023, 1, 18, 4, 0, tzinfo=UTC), *self._OVERNIGHT)

    def test_wrapping_window_excludes_midday(self) -> None:
        # 17:00 UTC == 12:00 EST -> outside
        assert not in_window(datetime(2023, 1, 17, 17, 0, tzinfo=UTC), *self._OVERNIGHT)

    def test_wrapping_window_end_exclusive(self) -> None:
        # 09:00 UTC == 04:00 EST -> end exclusive
        assert not in_window(datetime(2023, 1, 17, 9, 0, tzinfo=UTC), *self._OVERNIGHT)

    def test_same_day_window(self) -> None:
        # 18:00 UTC == 13:00 EST inside RTH 09:30-16:00
        assert in_window(datetime(2023, 1, 17, 18, 0, tzinfo=UTC), RTH_OPEN, RTH_CLOSE)


# ============================================================================
# TestSessionDateAndTime
# ============================================================================


class TestSessionDateAndTime:
    def test_session_date_crosses_midnight(self) -> None:
        # 2023-01-18 02:00 UTC == 2023-01-17 21:00 ET -> session date is the 17th, not the 18th
        assert session_date(datetime(2023, 1, 18, 2, 0, tzinfo=UTC)) == date(2023, 1, 17)

    def test_et_time(self) -> None:
        # 20:00 UTC == 15:00 EST
        assert et_time(datetime(2023, 1, 17, 20, 0, tzinfo=UTC)) == time(15, 0)
