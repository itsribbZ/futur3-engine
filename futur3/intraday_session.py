"""futur3.intraday_session - ET cash-session time-of-day primitives for the intraday rebuild.

Intraday session logic is anchored to the US EQUITY cash session, not the 23h Globex clock:
strategies key off wall-clock windows within the ET session. Bars arrive in UTC
(Databento), so these primitives convert to America/New_York -- DST-aware via `zoneinfo`, the
canonical ET that `cot_source` / `bls_macro` / `fred_macro` already use -- and test wall-clock
windows.

No holiday calendar is needed: a bar's mere existence means the market traded, so `session_date()`
and `in_window()` operate on the bars Databento actually returned. Pure stdlib, deterministic.
"""

from __future__ import annotations

from datetime import date, datetime, time
from typing import Final
from zoneinfo import ZoneInfo

ET: Final[ZoneInfo] = ZoneInfo("America/New_York")  # canonical ET; DST-aware

# US equity cash session (NYSE / Nasdaq regular trading hours), ET wall clock.
RTH_OPEN: Final[time] = time(9, 30)
RTH_CLOSE: Final[time] = time(16, 0)


class IntradaySessionError(Exception):
    """Time-of-day primitive misuse (e.g. a naive datetime at the boundary)."""


def to_et(ts: datetime) -> datetime:
    """Convert a TZ-aware datetime (UTC bars, or any zone) to America/New_York.

    Raises `IntradaySessionError` on a naive datetime (bug-class guard: TZ confusion).
    """
    if ts.tzinfo is None:
        raise IntradaySessionError(f"to_et requires a TZ-aware datetime; got naive {ts!r}")
    return ts.astimezone(ET)


def et_time(ts: datetime) -> time:
    """The ET wall-clock time-of-day of `ts`."""
    return to_et(ts).time()


def session_date(ts: datetime) -> date:
    """The ET calendar date of `ts` (the cash-session day for an RTH bar)."""
    return to_et(ts).date()


def in_window(ts: datetime, start: time, end: time) -> bool:
    """True iff `ts`'s ET wall-clock time is in the half-open window [start, end).

    A same-day window has start <= end (e.g. RTH 09:30-16:00). A window that WRAPS MIDNIGHT has
    start > end (e.g. the overnight 22:00-04:00 ET): then the test is `time >= start OR time < end`.
    """
    t = et_time(ts)
    if start <= end:
        return start <= t < end
    return t >= start or t < end


def is_rth(ts: datetime) -> bool:
    """True iff `ts` falls in the 09:30-16:00 ET equity cash session (close exclusive)."""
    return in_window(ts, RTH_OPEN, RTH_CLOSE)


__all__ = [
    "ET",
    "RTH_CLOSE",
    "RTH_OPEN",
    "IntradaySessionError",
    "et_time",
    "in_window",
    "is_rth",
    "session_date",
    "to_et",
]
