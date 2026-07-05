"""futur3.data.term_structure — front/next term-structure source for carry.

Term structure carry needs, at each date, the FRONT and the NEXT
contract's price + their days-to-expiry — the two-leg term structure the NUL continuous series
(`data/continuous.py`) collapses away (it keeps one price/day: the active front). This module
rebuilds the front/next pair from the SAME per-contract bars + roll calendar the continuous
stitcher consumes, so a carry strategy reads a real two-leg observation on ANY trading date the
active contract has a bar — not only on roll dates, where `RollEvent` already carries both legs.

`front` = the active contract the continuous series (and the engine) is on for that date; `back` =
its roll-cycle successor (`RollCalendarEntry.back_symbol`). The active contract per date is the one
whose `[prev_roll_target, roll_target)` window contains the date — exactly the splice rule
`ContinuousSeriesBuilder` uses, so carry's `front` is always the contract being traded.

PIT (bug class 5 / look-ahead): each date's observation uses only the front bar's close on that
date, the back contract's close on/before that date, and expiry dates fixed in advance by the roll
calendar — no future information. The full per-date index is precomputed once at construction.

Source-agnostic (any `RawBar` sequences: Databento individual contracts at race day; fixtures in
tests). Imports `RollCalendar` under TYPE_CHECKING only (mirrors `continuous.py`) to avoid the
`data` ↔ `execution` import cycle. Pure stdlib, Decimal, frozen dataclasses, deterministic.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from futur3.data.types import ContractSymbol, RawBar, _assert_tz_aware

if TYPE_CHECKING:
    from futur3.execution.roll_executor import RollCalendar


class TermStructureError(Exception):
    """Term-structure contract violation (untracked root, empty input, malformed observation)."""


@dataclass(frozen=True)
class CarryInputs:
    """The two-leg term-structure observation a carry strategy needs on one date.

    `front` / `back` are the active contract and its roll-cycle successor; `front_close` is the
    front bar's close on `as_of`, `back_close` the back contract's close on/before `as_of`;
    `days_front` / `days_back` are calendar days to each contract's last trading day.

    Invariants (fail-loud — a CONSTRUCTED observation is always valid; the source returns None rather
    than building a degenerate one): both closes > 0 and 0 < days_front < days_back (the back
    contract expires strictly later, so `days_back - days_front` is the positive inter-expiry gap).
    """

    as_of: date
    front: ContractSymbol
    back: ContractSymbol
    front_close: Decimal
    back_close: Decimal
    days_front: int
    days_back: int

    def __post_init__(self) -> None:
        if self.front_close <= 0 or self.back_close <= 0:
            raise TermStructureError(
                f"CarryInputs closes must be > 0; got front={self.front_close} "
                f"back={self.back_close} for {self.front}->{self.back} @ {self.as_of}"
            )
        if not (0 < self.days_front < self.days_back):
            raise TermStructureError(
                f"CarryInputs needs 0 < days_front < days_back; got {self.days_front} / "
                f"{self.days_back} for {self.front}->{self.back} @ {self.as_of}"
            )


class TermStructureSource(abc.ABC):
    """Seam between futur3 and a front/next term-structure view for carry.

    The single accessor `carry_inputs(as_of)` returns the PIT-safe two-leg observation for a date,
    or None when a leg's price is unavailable / the date is outside the tracked range — the carry
    strategy then emits no signal (FLAT) rather than fabricating a spread (fail-loud)."""

    @abc.abstractmethod
    def carry_inputs(self, as_of: datetime) -> CarryInputs | None:
        """The front/next observation known at `as_of`, or None if unavailable."""
        ...


def _close_on_or_before(bars: Sequence[RawBar], target: date) -> Decimal | None:
    """Close of the latest bar on/before `target` (no look-ahead), or None if there is none.
    Mirrors `ContinuousSeriesBuilder._close_on_or_before`; kept local to avoid a cross-module
    private import."""
    best: RawBar | None = None
    for bar in bars:
        day = bar.ts.date()
        if day <= target and (best is None or day > best.ts.date()):
            best = bar
    return best.close if best is not None else None


def _in_window(day: date, start: date | None, end: date) -> bool:
    """A bar's date falls in a contract's active window `[start, end)` (start None = open-ended)."""
    return (start is None or day >= start) and day < end


@dataclass(frozen=True)
class _Plan:
    """Resolved roll metadata for the present contracts: the roll-ordered `chain` plus each
    contract's roll_target window-end, last-trading-day, and roll-cycle successor."""

    chain: tuple[ContractSymbol, ...]
    roll_target: Mapping[ContractSymbol, date]
    ltd: Mapping[ContractSymbol, date]
    back_of: Mapping[ContractSymbol, ContractSymbol]


class StaticTermStructure(TermStructureSource):
    """In-memory term-structure source built from per-contract bars + a roll calendar.

    Precomputes, for every date the active front contract has a bar, the `CarryInputs` pairing it
    with its roll-cycle successor — so `carry_inputs` is an O(1) dict lookup and the PIT discipline
    is applied exactly once, at construction. Dates where the back leg has no price / is untracked
    are simply absent (carry then returns None — fail-loud: no fabricated leg)."""

    def __init__(
        self,
        per_contract_bars: Mapping[ContractSymbol, Sequence[RawBar]],
        roll_calendar: RollCalendar,
        root: str,
    ) -> None:
        present = [contract for contract, bars in per_contract_bars.items() if bars]
        if not present:
            raise TermStructureError("per_contract_bars is empty (no contract has any bars)")
        plan = self._resolve(present, roll_calendar, root)
        self._root = root
        self._by_date = self._precompute(plan, per_contract_bars, roll_calendar)

    @property
    def root(self) -> str:
        return self._root

    def carry_inputs(self, as_of: datetime) -> CarryInputs | None:
        _assert_tz_aware(as_of, "StaticTermStructure.carry_inputs.as_of")
        return self._by_date.get(as_of.date())

    def available_dates(self) -> list[date]:
        """Sorted dates that have a valid two-leg observation (diagnostics / tests)."""
        return sorted(self._by_date)

    @staticmethod
    def _resolve(present: list[ContractSymbol], roll_calendar: RollCalendar, root: str) -> _Plan:
        roll_target: dict[ContractSymbol, date] = {}
        ltd: dict[ContractSymbol, date] = {}
        back_of: dict[ContractSymbol, ContractSymbol] = {}
        for contract in present:
            if not str(contract).startswith(root):
                raise TermStructureError(f"contract {contract!r} does not belong to root {root!r}")
            entry = roll_calendar.lookup(contract)
            if entry is None:
                raise TermStructureError(f"contract {contract!r} is not in the roll calendar")
            roll_target[contract] = entry.roll_target
            ltd[contract] = entry.ltd_date
            back_of[contract] = ContractSymbol(entry.back_symbol)
        chain = tuple(sorted(present, key=lambda contract: roll_target[contract]))
        return _Plan(chain=chain, roll_target=roll_target, ltd=ltd, back_of=back_of)

    @staticmethod
    def _precompute(
        plan: _Plan,
        per_contract_bars: Mapping[ContractSymbol, Sequence[RawBar]],
        roll_calendar: RollCalendar,
    ) -> dict[date, CarryInputs]:
        index: dict[date, CarryInputs] = {}
        prev_roll: date | None = None
        for front in plan.chain:
            window_end = plan.roll_target[front]
            back = plan.back_of[front]
            back_bars = per_contract_bars.get(back)
            back_ltd = plan.ltd.get(back)
            if back_ltd is None:  # back beyond the loaded window: read its LTD from the calendar
                entry = roll_calendar.lookup(back)
                back_ltd = entry.ltd_date if entry is not None else None
            front_ltd = plan.ltd[front]
            for bar in per_contract_bars[front]:
                day = bar.ts.date()
                if not _in_window(day, prev_roll, window_end):
                    continue
                if back_bars is None or back_ltd is None:
                    continue
                back_close = _close_on_or_before(back_bars, day)
                if back_close is None or bar.close <= 0 or back_close <= 0:
                    continue
                days_front = (front_ltd - day).days
                days_back = (back_ltd - day).days
                if not (0 < days_front < days_back):
                    continue
                index[day] = CarryInputs(
                    as_of=day,
                    front=front,
                    back=back,
                    front_close=bar.close,
                    back_close=back_close,
                    days_front=days_front,
                    days_back=days_back,
                )
            prev_roll = window_end
        return index


__all__: list[str] = [
    "CarryInputs",
    "StaticTermStructure",
    "TermStructureError",
    "TermStructureSource",
]
