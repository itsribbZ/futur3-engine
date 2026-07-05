"""futur3.data.term_structure (StaticTermStructure) test suite — front/next index.

Builds a 3-contract CL term structure (CLF26 -> CLG26 -> CLH26) on the REAL CL roll calendar
(W1.2 RollCalendarBuilder) and asserts: the per-date front/next pairing (front = the active
contract, back = its roll-cycle successor), the on/before back-close lookup, the PIT / degenerate
None paths (missing back leg, out-of-range, non-positive close, naive as_of), the construction
guards (empty / wrong-root / untracked), the CarryInputs invariants, and determinism.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from futur3.data.term_structure import (
    CarryInputs,
    StaticTermStructure,
    TermStructureError,
)
from futur3.data.types import BarResolution, ContractSymbol, RawBar, content_sha256
from futur3.execution.roll_executor import RollCalendarBuilder, RollCalendarEntry

_CAL = RollCalendarBuilder().build_static_calendar("CL", 2025, 2026)


def _entry(symbol: str) -> RollCalendarEntry:
    found = _CAL.lookup(ContractSymbol(symbol))
    assert found is not None
    return found


_F = _entry("CLF26")
_G = _entry("CLG26")
_H = _entry("CLH26")

_D0 = _F.roll_target - timedelta(days=30)  # CLF26 window, before any back bar -> None
_D1 = _F.roll_target - timedelta(days=3)  # CLF26 window, back present
_D1B = _F.roll_target - timedelta(days=2)  # CLF26 window, back bar only on _D1 (on/before)
_D2 = _F.roll_target + timedelta(days=3)  # CLG26 window (still < CLG26 roll_target)


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
    # Backwardation: front (CLF26) richer than back (CLG26) on _D1; CLG26 richer than CLH26 on _D2.
    return {
        ContractSymbol("CLF26"): [
            _bar("CLF26", _D0, "75.00"),  # no CLG26 bar on/before -> None
            _bar("CLF26", _D1, "75.40"),  # pairs with CLG26@_D1
            _bar("CLF26", _D1B, "75.50"),  # CLG26 has no bar on _D1B -> uses on/before (_D1)
        ],
        ContractSymbol("CLG26"): [
            _bar("CLG26", _D1, "75.00"),  # back leg for the CLF26 window
            _bar("CLG26", _D2, "74.50"),  # front leg in the CLG26 window
        ],
        ContractSymbol("CLH26"): [
            _bar("CLH26", _D2, "74.00"),  # back leg for the CLG26 window
        ],
    }


def _build() -> StaticTermStructure:
    return StaticTermStructure(_scenario(), _CAL, "CL")


def _at(day: date) -> CarryInputs | None:
    return _build().carry_inputs(datetime(day.year, day.month, day.day, tzinfo=UTC))


class TestCarryPairing:
    def test_front_window_pairs_front_with_next(self) -> None:
        ci = _at(_D1)
        assert ci is not None
        assert ci.front == "CLF26"
        assert ci.back == "CLG26"
        assert ci.front_close == Decimal("75.40")
        assert ci.back_close == Decimal("75.00")
        assert ci.days_front == (_F.ltd_date - _D1).days
        assert ci.days_back == (_G.ltd_date - _D1).days

    def test_second_window_pairs_next_with_its_successor(self) -> None:
        ci = _at(_D2)
        assert ci is not None
        assert ci.front == "CLG26"
        assert ci.back == "CLH26"
        assert ci.front_close == Decimal("74.50")
        assert ci.back_close == Decimal("74.00")
        assert ci.days_front == (_G.ltd_date - _D2).days
        assert ci.days_back == (_H.ltd_date - _D2).days

    def test_back_close_uses_on_or_before(self) -> None:
        ci = _at(_D1B)
        assert ci is not None
        assert ci.front == "CLF26"
        assert ci.front_close == Decimal("75.50")
        assert ci.back_close == Decimal("75.00")  # CLG26 last close on/before _D1B (the _D1 bar)


class TestNonePaths:
    def test_missing_back_leg_returns_none(self) -> None:
        assert _at(_D0) is None  # CLG26 has no bar on/before _D0

    def test_out_of_range_date_returns_none(self) -> None:
        assert _at(_H.roll_target + timedelta(days=60)) is None

    def test_unplaced_date_returns_none(self) -> None:
        assert _at(_F.roll_target - timedelta(days=1)) is None  # in window but no front bar placed

    def test_nonpositive_front_close_returns_none(self) -> None:
        bars: dict[ContractSymbol, list[RawBar]] = {
            ContractSymbol("CLF26"): [_bar("CLF26", _D1, "0")],
            ContractSymbol("CLG26"): [_bar("CLG26", _D1, "75.00")],
        }
        ts = StaticTermStructure(bars, _CAL, "CL")
        assert ts.carry_inputs(datetime(_D1.year, _D1.month, _D1.day, tzinfo=UTC)) is None

    def test_naive_as_of_raises(self) -> None:
        with pytest.raises(ValueError, match="TZ-aware"):
            _build().carry_inputs(datetime(_D1.year, _D1.month, _D1.day))


class TestGuards:
    def test_empty_input_raises(self) -> None:
        with pytest.raises(TermStructureError, match="empty"):
            StaticTermStructure({}, _CAL, "CL")

    def test_wrong_root_raises(self) -> None:
        with pytest.raises(TermStructureError, match="does not belong to root"):
            StaticTermStructure(_scenario(), _CAL, "GC")

    def test_untracked_contract_raises(self) -> None:
        gc_cal = RollCalendarBuilder().build_static_calendar("GC", 2025, 2026)
        with pytest.raises(TermStructureError, match="not in the roll calendar"):
            StaticTermStructure(_scenario(), gc_cal, "CL")


class TestCarryInputsInvariants:
    def test_nonpositive_close_raises(self) -> None:
        with pytest.raises(TermStructureError, match="closes must be > 0"):
            CarryInputs(
                as_of=_D1,
                front=ContractSymbol("CLF26"),
                back=ContractSymbol("CLG26"),
                front_close=Decimal("0"),
                back_close=Decimal("75"),
                days_front=10,
                days_back=40,
            )

    def test_bad_days_ordering_raises(self) -> None:
        with pytest.raises(TermStructureError, match="0 < days_front < days_back"):
            CarryInputs(
                as_of=_D1,
                front=ContractSymbol("CLF26"),
                back=ContractSymbol("CLG26"),
                front_close=Decimal("75.4"),
                back_close=Decimal("75"),
                days_front=40,
                days_back=10,
            )


class TestDeterminism:
    @pytest.mark.bitrepro
    def test_index_is_deterministic(self) -> None:
        assert _build().available_dates() == _build().available_dates()
        assert _at(_D1) == _at(_D1)

    def test_available_dates_sorted_and_complete(self) -> None:
        dates = _build().available_dates()
        assert dates == sorted(dates)
        assert _D1 in dates
        assert _D1B in dates
        assert _D2 in dates
        assert _D0 not in dates
