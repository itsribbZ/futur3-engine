"""Event-severity risk-gate test suite (fixture-only; no network/cache).

Verifies the pure, look-ahead-clean core of the capital-preservation event gate:
- ex-ante series->severity mapping (FOMC CRITICAL, CPI/NFP HIGH, jobless MEDIUM, unknown LOW);
- the pre-registered Tier-1 multiplier (CRITICAL/HIGH -> 0.5x, else 1.0x);
- EventCalendar.multiplier window logic (inclusive bounds, exclude set, min-pick over events);
- apply_gate: gated == ungated when no event triggers (reconciliation), and a PLANTED
  drawdown on a Tier-1 day is halved -> the gate cuts the left tail (the mechanism, front-to-back).
"""

from __future__ import annotations

from datetime import date

from futur3.data.macro_types import MacroSeries
from futur3.events.severity import (
    CalendarEvent,
    EventCalendar,
    EventSeverity,
    apply_gate,
    calendar_severity,
    severity_multiplier,
)


class TestCalendarSeverity:
    def test_tier1_series(self) -> None:
        assert calendar_severity(MacroSeries.FOMC_STATEMENT) is EventSeverity.CRITICAL
        assert calendar_severity(MacroSeries.CPI) is EventSeverity.HIGH
        assert calendar_severity(MacroSeries.NFP) is EventSeverity.HIGH

    def test_lower_tiers_and_default(self) -> None:
        assert calendar_severity(MacroSeries.JOBLESS_CLAIMS) is EventSeverity.MEDIUM
        # an unlisted series defaults to LOW (untouched by the gate)
        assert calendar_severity(MacroSeries.BEIGE_BOOK) is EventSeverity.LOW


class TestSeverityMultiplier:
    def test_tier1_halves(self) -> None:
        assert severity_multiplier(EventSeverity.CRITICAL) == 0.5
        assert severity_multiplier(EventSeverity.HIGH) == 0.5

    def test_below_tier1_untouched(self) -> None:
        assert severity_multiplier(EventSeverity.MEDIUM) == 1.0
        assert severity_multiplier(EventSeverity.LOW) == 1.0
        assert severity_multiplier(EventSeverity.RELEASED) == 1.0


class TestEventCalendarMultiplier:
    def _cal(self) -> EventCalendar:
        return EventCalendar(
            (
                CalendarEvent(MacroSeries.FOMC_STATEMENT, date(2023, 3, 22)),
                CalendarEvent(MacroSeries.CPI, date(2023, 3, 14)),
                CalendarEvent(MacroSeries.JOBLESS_CLAIMS, date(2023, 3, 16)),
            )
        )

    def test_no_event_in_window_is_neutral(self) -> None:
        # 2023-03-01: nearest Tier-1 (CPI 03-14) is 13 days out -> outside a (0,2) window.
        assert self._cal().multiplier(date(2023, 3, 1), days_lo=0, days_hi=2) == 1.0

    def test_event_ahead_triggers_halving(self) -> None:
        # the FOMC is 1 day ahead of 2023-03-21 -> within (0,2) -> CRITICAL -> 0.5.
        assert self._cal().multiplier(date(2023, 3, 21), days_lo=0, days_hi=2) == 0.5

    def test_window_upper_bound_is_inclusive(self) -> None:
        cal = self._cal()
        # CPI on 03-14 is 2 days ahead of 03-12 -> inside (0,2); 3 days ahead of 03-11 -> out.
        assert cal.multiplier(date(2023, 3, 12), days_lo=0, days_hi=2) == 0.5
        assert cal.multiplier(date(2023, 3, 11), days_lo=0, days_hi=2) == 1.0

    def test_exclude_set_skips_series(self) -> None:
        cal = self._cal()
        # excluding FOMC, the day before the FOMC has no other Tier-1 in (0,2) -> neutral.
        m = cal.multiplier(
            date(2023, 3, 21), days_lo=0, days_hi=2, exclude=frozenset({MacroSeries.FOMC_STATEMENT})
        )
        assert m == 1.0

    def test_lower_tier_does_not_halve(self) -> None:
        # only the MEDIUM jobless claim (03-16) is within (0,2) of 03-15 -> stays 1.0.
        assert self._cal().multiplier(date(2023, 3, 15), days_lo=0, days_hi=2) == 1.0

    def test_backward_window_for_prefomc(self) -> None:
        # a CPI (03-14) one day BEFORE 03-15 -> inside a (-1, 0) backward-looking window -> 0.5.
        assert self._cal().multiplier(date(2023, 3, 15), days_lo=-1, days_hi=0) == 0.5


class TestApplyGateFrontToBack:
    def test_no_trigger_is_identity(self) -> None:
        cal = EventCalendar((CalendarEvent(MacroSeries.FOMC_STATEMENT, date(2030, 1, 1)),))
        rets = {date(2023, 1, 3): 0.01, date(2023, 1, 4): -0.02}
        gated = apply_gate(rets, cal, days_lo=0, days_hi=2)
        assert gated == rets  # no event near these dates -> byte-identical (reconciliation)

    def test_planted_drawdown_is_halved(self) -> None:
        # Plant a big loss the day before an FOMC; the gate must halve it -> the left tail shrinks.
        fomc = date(2023, 3, 22)
        cal = EventCalendar((CalendarEvent(MacroSeries.FOMC_STATEMENT, fomc),))
        rets = {
            date(2023, 3, 20): 0.005,
            date(2023, 3, 21): -0.10,  # the gated catastrophe (1 day before the FOMC)
            date(2023, 3, 28): 0.004,  # well after the window -> untouched
        }
        gated = apply_gate(rets, cal, days_lo=0, days_hi=2)
        assert gated[date(2023, 3, 21)] == -0.05  # halved
        assert gated[date(2023, 3, 28)] == 0.004  # untouched
        # the worst gated return is strictly less severe than the worst ungated return
        assert min(gated.values()) > min(rets.values())

    def test_determinism(self) -> None:
        cal = EventCalendar((CalendarEvent(MacroSeries.CPI, date(2023, 6, 13)),))
        rets = {date(2023, 6, 12): -0.03}
        a = apply_gate(rets, cal, days_lo=0, days_hi=2)
        b = apply_gate(rets, cal, days_lo=0, days_hi=2)
        assert a == b  # pure/deterministic
