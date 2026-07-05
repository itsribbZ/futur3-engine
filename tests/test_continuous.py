"""ContinuousSeriesBuilder (NUL stitcher) test suite.

Per the build plan. Builds a 2-contract CL continuous series (CLF26 ->
CLG26) across the real Dec-12-2025 roll (roll_target from the W1.2 RollCalendarBuilder) with a
deliberate +$0.70 same-day roll spread, and asserts:
- the NUL splice selects the right contract per active window + carries RAW prices (no adjustment);
- roll_flags mark exactly the contract transition;
- the RollEvent records front/back same-day closes + roll_gap = back - front;
- ratio_adjusted_closes() is the research view (last segment unadjusted; earlier scaled by ratio);
- current_contract / roll_event_at lookups;
- Fail-loud raises: untracked contract, empty input; determinism.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from futur3.data.continuous import (
    ContinuousSeries,
    ContinuousSeriesBuilder,
    ContinuousSeriesError,
)
from futur3.data.types import BarResolution, ContractSymbol, RawBar, content_sha256
from futur3.execution.roll_executor import RollCalendarBuilder

_CL_CALENDAR = RollCalendarBuilder().build_static_calendar("CL", 2025, 2026)
_ROLL_DATE = date(2025, 12, 12)  # CLF26 roll_target (verified in W1.2)


def _bar(contract: str, day: date, close: str) -> RawBar:
    ts = datetime(day.year, day.month, day.day, tzinfo=UTC)
    price = Decimal(close)
    return RawBar(
        contract=ContractSymbol(contract),
        ts=ts,
        resolution=BarResolution.DAY_1,
        open=price,
        high=price,
        low=price,
        close=price,
        volume=1000,
        oi=None,
        source_id="test",
        as_of_iso=ts,
        content_bytes_sha=content_sha256(f"{contract}{day}{close}".encode()),
    )


def _scenario() -> dict[ContractSymbol, list[RawBar]]:
    # CLF26 front (~75) trades through the roll; CLG26 back (~76) trades just before+after.
    # roll_target(CLF26) = 2025-12-12 -> CLF26 active for dates < 12; CLG26 active for dates >= 12.
    return {
        ContractSymbol("CLF26"): [
            _bar("CLF26", date(2025, 12, 8), "75.00"),
            _bar("CLF26", date(2025, 12, 9), "75.10"),
            _bar("CLF26", date(2025, 12, 10), "75.20"),
            _bar("CLF26", date(2025, 12, 11), "75.30"),
            _bar("CLF26", date(2025, 12, 12), "75.40"),  # exists on roll date (front_settle)
        ],
        ContractSymbol("CLG26"): [
            _bar("CLG26", date(2025, 12, 11), "76.00"),  # trades before it goes active (excluded)
            _bar("CLG26", date(2025, 12, 12), "76.10"),  # first active bar (back_settle)
            _bar("CLG26", date(2025, 12, 15), "76.20"),
            _bar("CLG26", date(2025, 12, 16), "76.30"),
        ],
    }


def _build() -> ContinuousSeries:
    return ContinuousSeriesBuilder().build(_scenario(), _CL_CALENDAR, "CL")


# ============================================================================
# TestW1_4_NulSplice
# ============================================================================


class TestW1_4_NulSplice:
    def test_splice_selects_active_contract_per_window(self) -> None:
        series = _build()
        # CLF26 contributes Dec 8-11 (date < 12); CLG26 contributes Dec 12,15,16 (date >= 12).
        assert [str(b.contract) for b in series.bars] == [
            "CLF26",
            "CLF26",
            "CLF26",
            "CLF26",
            "CLG26",
            "CLG26",
            "CLG26",
        ]
        assert [b.ts.date() for b in series.bars] == [
            date(2025, 12, 8),
            date(2025, 12, 9),
            date(2025, 12, 10),
            date(2025, 12, 11),
            date(2025, 12, 12),
            date(2025, 12, 15),
            date(2025, 12, 16),
        ]

    def test_prices_are_raw_unadjusted(self) -> None:
        series = _build()
        assert series.bars[3].close == Decimal("75.30")  # last CLF26 (Dec 11) — raw
        assert series.bars[4].close == Decimal("76.10")  # first CLG26 (Dec 12) — raw, NOT adjusted

    def test_roll_flags_mark_only_the_transition(self) -> None:
        series = _build()
        assert series.roll_flags == (False, False, False, False, True, False, False)


# ============================================================================
# TestW1_4_RollEvent
# ============================================================================


class TestW1_4_RollEvent:
    def test_single_roll_event_same_day_settles(self) -> None:
        series = _build()
        assert len(series.roll_events) == 1
        event = series.roll_events[0]
        assert event.roll_date == _ROLL_DATE
        assert event.old_contract == "CLF26"
        assert event.new_contract == "CLG26"
        assert event.front_settle == Decimal("75.40")  # CLF26 close on the roll date
        assert event.back_settle == Decimal("76.10")  # CLG26 close on the roll date
        assert event.roll_gap == Decimal("0.70")  # back - front (the phantom-jump magnitude)

    def test_roll_event_at_lookup(self) -> None:
        series = _build()
        assert series.roll_event_at(_ROLL_DATE) is not None
        assert series.roll_event_at(date(2025, 12, 10)) is None


# ============================================================================
# TestW1_4_ResearchView (ratio-adjusted — never an engine input)
# ============================================================================


class TestW1_4_ResearchView:
    def test_last_segment_unadjusted(self) -> None:
        series = _build()
        adjusted = series.ratio_adjusted_closes()
        assert len(adjusted) == len(series.bars)
        # CLG26 is the final segment -> factor 1.0 -> raw closes preserved.
        assert adjusted[4] == Decimal("76.10")
        assert adjusted[5] == Decimal("76.20")
        assert adjusted[6] == Decimal("76.30")

    def test_earlier_segment_scaled_by_roll_ratio(self) -> None:
        series = _build()
        adjusted = series.ratio_adjusted_closes()
        ratio = Decimal("76.10") / Decimal("75.40")  # back/front
        assert adjusted[0] == Decimal("75.00") * ratio
        # within-segment percentage step preserved (factor cancels): 75.10/75.00.
        assert adjusted[1] == Decimal("75.10") * ratio


# ============================================================================
# TestW1_4_Lookups / Guards
# ============================================================================


class TestW1_4_Queries:
    def test_current_contract(self) -> None:
        series = _build()
        assert series.current_contract(date(2025, 12, 10)) == "CLF26"
        assert series.current_contract(date(2025, 12, 12)) == "CLG26"
        assert series.current_contract(date(2025, 12, 16)) == "CLG26"

    def test_current_contract_before_first_bar_raises(self) -> None:
        series = _build()
        with pytest.raises(ContinuousSeriesError, match="precedes the first bar"):
            series.current_contract(date(2025, 12, 1))


class TestW1_4_Guards:
    def test_empty_input_raises(self) -> None:
        with pytest.raises(ContinuousSeriesError, match="empty"):
            ContinuousSeriesBuilder().build({}, _CL_CALENDAR, "CL")

    def test_untracked_contract_raises(self) -> None:
        # A GC calendar does not track CL contracts.
        gc_cal = RollCalendarBuilder().build_static_calendar("GC", 2025, 2026)
        with pytest.raises(ContinuousSeriesError, match="not in the roll calendar"):
            ContinuousSeriesBuilder().build(_scenario(), gc_cal, "CL")

    def test_wrong_root_raises(self) -> None:
        with pytest.raises(ContinuousSeriesError, match="does not belong to root"):
            ContinuousSeriesBuilder().build(_scenario(), _CL_CALENDAR, "GC")

    @pytest.mark.bitrepro
    def test_deterministic(self) -> None:
        assert _build() == _build()
