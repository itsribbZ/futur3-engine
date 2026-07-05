"""futur3.data.continuous — NUL continuous-contract stitcher.

Per the build plan + internal microstructure notes (the
GOVERNING "NUL primary" decision). Fixes the
roll-contamination: a raw Databento `c.0` splice carries a price-level discontinuity at every
roll, which `_mark_to_market` turned into phantom unrealized PnL -> phantom returns -> a
contaminated gauntlet verdict.

NUL (unadjusted splice + roll-as-paired-trade): keep RAW per-contract prices — each spliced bar
carries its REAL underlying contract, never a synthetic `CL.c.0` — and record each roll as an
explicit `RollEvent` (front/back close on the roll date) so the engine (W1.5) books the roll as a
real paired trade rather than seeing a phantom price jump. This is the BACKTEST-IS-LIVE +
fail-loud design choice: zero look-ahead surface, EXACT round-trip (no float-epsilon adjustment).

`ratio_adjusted_closes()` is a DERIVED RESEARCH VIEW only — it bakes in future
roll ratios (look-ahead) so it must NEVER feed the engine; the engine consumes the NUL `bars`.

Source-agnostic: consumes any `RawBar` sequences (Databento individual contracts at race day;
fixtures in tests). Pure stdlib, Decimal, frozen dataclasses, deterministic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING

from futur3.data.types import ContractSymbol, RawBar

if TYPE_CHECKING:
    from futur3.execution.roll_executor import RollCalendar


class ContinuousSeriesError(Exception):
    """Continuous-series contract violation (untracked contract, empty input, gap at a roll)."""


@dataclass(frozen=True)
class RollEvent:
    """One roll in a NUL continuous series: on `roll_date` the active contract switches from
    `old_contract` to `new_contract`. `front_settle` / `back_settle` are the two contracts'
    closes on the roll date; `roll_gap = back_settle - front_settle` is the same-day roll spread
    the engine (W1.5) books as a paired trade — NOT a phantom price jump."""

    roll_date: date
    old_contract: ContractSymbol
    new_contract: ContractSymbol
    front_settle: Decimal
    back_settle: Decimal
    roll_gap: Decimal

    def __post_init__(self) -> None:
        expected = self.back_settle - self.front_settle
        if self.roll_gap != expected:
            raise ContinuousSeriesError(
                f"RollEvent.roll_gap {self.roll_gap} != back-front {expected} for "
                f"{self.old_contract}->{self.new_contract} @ {self.roll_date}"
            )


@dataclass(frozen=True)
class ContinuousSeries:
    """A NUL (unadjusted) continuous contract: raw per-contract bars spliced in roll order, each
    carrying its REAL underlying contract, plus the roll metadata the engine needs to book rolls
    as paired trades. `roll_flags[i]` is True iff `bars[i]` is the first bar of a new contract;
    `roll_events` has one entry per such transition.

    The engine consumes `bars` (raw prices — BACKTEST-IS-LIVE). `ratio_adjusted_closes()` is a
    DERIVED RESEARCH VIEW only and must never be fed to the engine."""

    root: str
    bars: tuple[RawBar, ...]
    roll_events: tuple[RollEvent, ...]
    roll_flags: tuple[bool, ...]

    def __post_init__(self) -> None:
        if not self.bars:
            raise ContinuousSeriesError("ContinuousSeries has no bars")
        if len(self.bars) != len(self.roll_flags):
            raise ContinuousSeriesError(
                f"bars/roll_flags length mismatch: {len(self.bars)} != {len(self.roll_flags)}"
            )
        if self.roll_flags[0]:
            raise ContinuousSeriesError("roll_flags[0] must be False (first bar starts no roll)")
        if sum(self.roll_flags) != len(self.roll_events):
            raise ContinuousSeriesError(
                f"roll_flags True={sum(self.roll_flags)} != roll_events={len(self.roll_events)}"
            )
        for prev, cur in zip(self.bars, self.bars[1:], strict=False):
            if cur.ts <= prev.ts:
                raise ContinuousSeriesError(
                    f"bars must be strictly increasing in ts; got {prev.ts} >= {cur.ts}"
                )

    def current_contract(self, asof: date) -> ContractSymbol:
        """The active underlying contract as of `asof` (the contract of the latest bar on/before
        `asof`). Raises if `asof` precedes the first bar."""
        active: ContractSymbol | None = None
        for bar in self.bars:
            if bar.ts.date() > asof:
                break
            active = bar.contract
        if active is None:
            raise ContinuousSeriesError(
                f"asof {asof} precedes the first bar {self.bars[0].ts.date()}"
            )
        return active

    def roll_event_at(self, asof: date) -> RollEvent | None:
        """The roll occurring on `asof`, or None."""
        for event in self.roll_events:
            if event.roll_date == asof:
                return event
        return None

    def ratio_adjusted_closes(self) -> tuple[Decimal, ...]:
        """DERIVED RESEARCH VIEW (`MATH` section 22) — the ratio-adjusted close series. NOT for
        engine use: it preserves within-contract percentage returns and is continuous across
        rolls, but bakes in FUTURE roll ratios (look-ahead), so it must never feed sizing/PnL.
        The engine uses `bars` (raw NUL prices)."""
        n_segments = len(self.roll_events) + 1
        ratios = [event.back_settle / event.front_settle for event in self.roll_events]
        factors = [Decimal(1)] * n_segments
        for segment in range(n_segments - 2, -1, -1):
            factors[segment] = factors[segment + 1] * ratios[segment]
        adjusted: list[Decimal] = []
        segment = 0
        for index, bar in enumerate(self.bars):
            if self.roll_flags[index]:
                segment += 1
            adjusted.append(bar.close * factors[segment])
        return tuple(adjusted)


class ContinuousSeriesBuilder:
    """Builds a NUL `ContinuousSeries` from raw per-contract bars + a roll calendar.

    Each contract's active window is `[previous contract's roll_target, its own roll_target)`; the
    builder splices each contract's in-window bars in roll order and records a `RollEvent` (with
    both contracts' same-day closes) at every contract transition in the spliced stream."""

    def build(
        self,
        per_contract_bars: Mapping[ContractSymbol, Sequence[RawBar]],
        roll_calendar: RollCalendar,
        root: str,
    ) -> ContinuousSeries:
        """Splice `per_contract_bars` into one NUL continuous series for `root`."""
        present = [contract for contract, bars in per_contract_bars.items() if bars]
        if not present:
            raise ContinuousSeriesError("per_contract_bars is empty (no contract has any bars)")
        for contract in present:
            if not str(contract).startswith(root):
                raise ContinuousSeriesError(
                    f"contract {contract!r} does not belong to root {root!r}"
                )

        roll_target: dict[ContractSymbol, date] = {}
        for contract in present:
            entry = roll_calendar.lookup(contract)
            if entry is None:
                raise ContinuousSeriesError(f"contract {contract!r} is not in the roll calendar")
            roll_target[contract] = entry.roll_target

        chain = sorted(present, key=lambda contract: roll_target[contract])

        spliced: list[RawBar] = []
        prev_roll: date | None = None
        for contract in chain:
            window_end = roll_target[contract]
            window = [
                bar
                for bar in per_contract_bars[contract]
                if self._in_window(bar.ts.date(), prev_roll, window_end)
            ]
            window.sort(key=lambda bar: bar.ts)
            spliced.extend(window)
            prev_roll = window_end
        if not spliced:
            raise ContinuousSeriesError(
                f"no bars fell inside any contract's active window for root {root!r}"
            )

        roll_flags: list[bool] = [False]
        roll_events: list[RollEvent] = []
        for index in range(1, len(spliced)):
            changed = spliced[index].contract != spliced[index - 1].contract
            roll_flags.append(changed)
            if changed:
                roll_events.append(
                    self._make_roll_event(
                        spliced[index - 1].contract,
                        spliced[index].contract,
                        roll_target,
                        per_contract_bars,
                    )
                )

        return ContinuousSeries(
            root=root,
            bars=tuple(spliced),
            roll_events=tuple(roll_events),
            roll_flags=tuple(roll_flags),
        )

    def _make_roll_event(
        self,
        front: ContractSymbol,
        back: ContractSymbol,
        roll_target: Mapping[ContractSymbol, date],
        per_contract_bars: Mapping[ContractSymbol, Sequence[RawBar]],
    ) -> RollEvent:
        roll_date = roll_target[front]
        front_settle = self._close_on_or_before(per_contract_bars[front], roll_date)
        back_settle = self._close_on_or_after(per_contract_bars[back], roll_date)
        if front_settle is None:
            raise ContinuousSeriesError(f"{front!r} has no bar on/before roll date {roll_date}")
        if back_settle is None:
            raise ContinuousSeriesError(f"{back!r} has no bar on/after roll date {roll_date}")
        return RollEvent(
            roll_date=roll_date,
            old_contract=front,
            new_contract=back,
            front_settle=front_settle,
            back_settle=back_settle,
            roll_gap=back_settle - front_settle,
        )

    @staticmethod
    def _in_window(day: date, start: date | None, end: date) -> bool:
        return (start is None or day >= start) and day < end

    @staticmethod
    def _close_on_or_before(bars: Sequence[RawBar], target: date) -> Decimal | None:
        best: RawBar | None = None
        for bar in bars:
            day = bar.ts.date()
            if day <= target and (best is None or day > best.ts.date()):
                best = bar
        return best.close if best is not None else None

    @staticmethod
    def _close_on_or_after(bars: Sequence[RawBar], target: date) -> Decimal | None:
        best: RawBar | None = None
        for bar in bars:
            day = bar.ts.date()
            if day >= target and (best is None or day < best.ts.date()):
                best = bar
        return best.close if best is not None else None
