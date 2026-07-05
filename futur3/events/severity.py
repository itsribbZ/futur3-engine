"""futur3.events.severity - ex-ante macro-event SEVERITY + the risk-gate size multiplier.

A capital-preservation overlay, NOT alpha (the pre-registration). The
severity of a scheduled macro series is CALENDAR-ONLY (FOMC/CPI/NFP are known months in advance), so
the size multiplier is a PURE function of (the forward-known release schedule, a reference date) --
it consults NO realized values and NO sentiment, which makes it look-ahead-clean BY CONSTRUCTION
(the analog of the engine's `enforce_pit_gate`: there is simply nothing future to leak).

The multiplier is in [0, 1]: 1.0 = trade normally, 0.5 = halve notional into a Tier-1 window, 0.0
would flat (not used in v1 -- the literature warns a full cut leaves edge on the table). The frozen,
pre-registered maps below are the ONLY tunables and are deliberately a singleton (one tier map, one
multiplier) so there is nothing to p-hack over.

Pure stdlib, float-domain (applied to the float return series), frozen dataclasses, deterministic.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Final

from futur3.data.macro_types import MacroSeries


class EventSeverity(StrEnum):
    """Ex-ante severity tier of a scheduled macro release (calendar-only, look-ahead-clean)."""

    RELEASED = "RELEASED"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# Pre-registered ex-ante series->severity. Default LOW for anything unlisted. CRITICAL/HIGH =
# the "Tier-1" set that triggers the derisk; MEDIUM/LOW are untouched in v1.
SERIES_SEVERITY: Final[dict[MacroSeries, EventSeverity]] = {
    MacroSeries.FOMC_STATEMENT: EventSeverity.CRITICAL,
    MacroSeries.CPI: EventSeverity.HIGH,
    MacroSeries.NFP: EventSeverity.HIGH,
    MacroSeries.JOBLESS_CLAIMS: EventSeverity.MEDIUM,
}

# Pre-registered Tier-1 derisk: CRITICAL/HIGH -> halve notional; below Tier-1 -> untouched.
TIER_MULTIPLIER: Final[dict[EventSeverity, float]] = {
    EventSeverity.CRITICAL: 0.5,
    EventSeverity.HIGH: 0.5,
    EventSeverity.MEDIUM: 1.0,
    EventSeverity.LOW: 1.0,
    EventSeverity.RELEASED: 1.0,
}


def calendar_severity(series: MacroSeries) -> EventSeverity:
    """Ex-ante severity of a scheduled macro series (default LOW). Pure lookup, look-ahead-clean --
    the severity of 'an FOMC meeting' / 'a CPI print' is known forever in advance."""
    return SERIES_SEVERITY.get(series, EventSeverity.LOW)


def severity_multiplier(severity: EventSeverity) -> float:
    """The pre-registered position-size multiplier in [0, 1] for a given severity."""
    return TIER_MULTIPLIER[severity]


@dataclass(frozen=True)
class CalendarEvent:
    """One scheduled macro release: series + forward-known release DATE. No realized value (so it
    cannot carry look-ahead)."""

    series: MacroSeries
    release_date: date


@dataclass(frozen=True)
class EventCalendar:
    """An immutable Tier-1 scheduled-event calendar; the multiplier is a PURE function of it."""

    events: tuple[CalendarEvent, ...]

    def multiplier(
        self,
        ref_date: date,
        *,
        days_lo: int,
        days_hi: int,
        exclude: frozenset[MacroSeries] = frozenset(),
    ) -> float:
        """Min size multiplier over events `e` with `e.series not in exclude` and
        `days_lo <= (e.release_date - ref_date).days <= days_hi`. Returns 1.0 if no such event.

        The (days_lo, days_hi) window frames the gate relative to `ref_date`:
          - (0, 2)  = an event today / tomorrow / in 2 days -> a 48h-ahead guard window.
          - (-1, 0) = an event yesterday / today            -> a backward-looking guard window.
        Look-ahead-clean: consults only forward-known release DATES, never realized values.
        """
        mult = 1.0
        for e in self.events:
            if e.series in exclude:
                continue
            delta = (e.release_date - ref_date).days
            if days_lo <= delta <= days_hi:
                mult = min(mult, severity_multiplier(calendar_severity(e.series)))
        return mult


def apply_gate(
    returns: Mapping[date, float],
    calendar: EventCalendar,
    *,
    days_lo: int,
    days_hi: int,
    exclude: frozenset[MacroSeries] = frozenset(),
) -> dict[date, float]:
    """Scale each return by its date's gate multiplier -> the gated return series. Pure; equals the
    ungated series exactly when the calendar triggers no event in any date's window (mult 1.0)."""
    return {
        d: r * calendar.multiplier(d, days_lo=days_lo, days_hi=days_hi, exclude=exclude)
        for d, r in returns.items()
    }


__all__ = [
    "SERIES_SEVERITY",
    "TIER_MULTIPLIER",
    "CalendarEvent",
    "EventCalendar",
    "EventSeverity",
    "apply_gate",
    "calendar_severity",
    "severity_multiplier",
]
