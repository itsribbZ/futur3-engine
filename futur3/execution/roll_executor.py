"""futur3.execution.roll_executor - RollExecutor decision/policy engine (A1.20).

Per internal design notes(item 3.8). This ship is the
deterministic ROLL DECISION layer + contract-cycle math:

- contract-cycle math: `parse_contract` / `next_contract` over the CME month-code cycles
  (quarterly H/M/U/Z for ES/NQ; monthly for CL/MBT/MET; bi-monthly G/J/M/Q/V/Z for GC);
- `RollCalendar` ABC + `StaticRollCalendar` (in-memory precomputed entries);
- `RollDivergenceCheck` protocol + `StaticRollDivergenceCheck` (explicit divergent-pair config);
- `RollExecutor.decide_rolls(today, positions)` - calendar-spread-first method policy with
  sequential fallback (no broker bag support OR verifier divergence) and past-deadline emergency
  detection (fail-loud: never sugar-coat a missed roll).

DEFERRED to a later ship (documented, NOT stubbed here per the A1.19 no-hypothetical-stubs rule):
- `execute_roll` order DISPATCH - needs `MockBroker.place_bag_order` (BAG/combo simulation) + slippage wiring + `RuntimeContext` injection (backtest-is-live); the spec's
  `_execute_*` bodies are `...` for exactly this reason.
- `ParquetRollCalendar` PERSISTENCE - the `roll_calendar.parquet` loader/writer. The roll-window
  DERIVATION it depended on is now built: `RollCalendarBuilder` computes
  LTD/FND/roll_target/roll_deadline from CME expiry rules using `CMETradingCalendar` (the holiday
  calendar whose absence blocked this). In-memory `build_static_calendar` suffices for the
  backtest re-verdict; on-disk caching is the remaining sub-ship.
"""

from __future__ import annotations

import abc
import calendar
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import Final, Literal, Protocol

from futur3.contracts import CMETradingCalendar
from futur3.data.types import ContractSymbol

_SYMBOL_SUFFIX_LEN: Final[int] = 3  # <month_code><year_2digit>, e.g. "M26"
_YEAR_2DIGIT_PIVOT: Final[int] = 49  # 00-49 -> 2000+; 50-99 -> 1900+ (CME 2-digit-year convention)

# CME month codes -> calendar month number (1-12).
_MONTH_CODE_TO_NUM: Final[dict[str, int]] = {
    "F": 1,
    "G": 2,
    "H": 3,
    "J": 4,
    "K": 5,
    "M": 6,
    "N": 7,
    "Q": 8,
    "U": 9,
    "V": 10,
    "X": 11,
    "Z": 12,
}

# Roll cycles as ordered month-code tuples (per the CME listing-cadence table).
_QUARTERLY: Final[tuple[str, ...]] = ("H", "M", "U", "Z")  # Mar Jun Sep Dec
_MONTHLY: Final[tuple[str, ...]] = ("F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z")
_BI_MONTHLY: Final[tuple[str, ...]] = ("G", "J", "M", "Q", "V", "Z")  # Feb Apr Jun Aug Oct Dec
_HKNUZ: Final[tuple[str, ...]] = (
    "H",
    "K",
    "N",
    "U",
    "Z",
)  # Mar May Jul Sep Dec (COMEX Cu/Ag active)

# Per-root roll cycle for the 6-contract universe + micros (matches the risk_manager registries).
ROOT_CYCLE: Final[dict[str, tuple[str, ...]]] = {
    "ES": _QUARTERLY,
    "MES": _QUARTERLY,
    "NQ": _QUARTERLY,
    "MNQ": _QUARTERLY,
    "CL": _MONTHLY,
    "MCL": _MONTHLY,
    "GC": _BI_MONTHLY,
    "MGC": _BI_MONTHLY,
    "MBT": _MONTHLY,
    "MET": _MONTHLY,
    # Broaden wave 1 (all quarterly H,M,U,Z): equity YM/RTY, FX 6E/6A, rates ZN/ZB.
    "YM": _QUARTERLY,
    "RTY": _QUARTERLY,
    "6E": _QUARTERLY,
    "6A": _QUARTERLY,
    "ZN": _QUARTERLY,
    "ZB": _QUARTERLY,
    # Broaden wave 2 (cycle = LIQUID months, verified against cached volume 2026-05-24):
    "6J": _QUARTERLY,
    "6B": _QUARTERLY,
    "6C": _QUARTERLY,  # serial months list but >99% volume is quarterly (H,M,U,Z)
    "6S": _QUARTERLY,
    "HG": _HKNUZ,  # all 12 list but >99% volume in H,K,N,U,Z (NOT _MONTHLY — empirical)
    "SI": _HKNUZ,
    "ZT": _QUARTERLY,
    "ZF": _QUARTERLY,
    "UB": _QUARTERLY,
    "NKD": _QUARTERLY,
}

# Decision method. SEQUENTIAL_EMERGENCY = past-deadline fail-loud alert path.
RollMethod = Literal["CALENDAR_SPREAD", "SEQUENTIAL", "SEQUENTIAL_EMERGENCY"]


class RollExecutorError(Exception):
    """Roll-executor error (unparseable symbol, unknown root, month code outside its cycle)."""


def _expand_year(yy: int) -> int:
    """2-digit -> 4-digit year (00-49 -> 2000+; 50-99 -> 1900+)."""
    return 2000 + yy if yy <= _YEAR_2DIGIT_PIVOT else 1900 + yy


def parse_contract(symbol: ContractSymbol | str) -> tuple[str, str, int]:
    """Split `ESM26` -> ("ES", "M", 2026): (root, CME month code, 4-digit year)."""
    s = str(symbol)
    if len(s) <= _SYMBOL_SUFFIX_LEN:
        raise RollExecutorError(f"contract symbol too short to parse: {s!r}")
    root = s[:-_SYMBOL_SUFFIX_LEN]
    code = s[-_SYMBOL_SUFFIX_LEN]
    yy = s[-(_SYMBOL_SUFFIX_LEN - 1) :]
    if code not in _MONTH_CODE_TO_NUM:
        raise RollExecutorError(f"invalid CME month code {code!r} in {s!r}")
    if not yy.isdigit():
        raise RollExecutorError(f"invalid 2-digit year {yy!r} in {s!r}")
    return root, code, _expand_year(int(yy))


def next_contract(symbol: ContractSymbol) -> ContractSymbol:
    """Next contract in the root's roll cycle (front -> back), wrapping the year past Z.

    ESM26 -> ESU26 (quarterly); ESZ26 -> ESH27 (year wrap); CLZ26 -> CLF27 (monthly);
    GCZ26 -> GCG27 (bi-monthly). Raises if the root is unknown or the month code is not a valid
    member of that root's cycle (fail-loud rather than silently pick a wrong next contract).
    """
    root, code, year = parse_contract(symbol)
    cycle = ROOT_CYCLE.get(root)
    if cycle is None:
        raise RollExecutorError(
            f"no roll cycle configured for root {root!r}; known: {sorted(ROOT_CYCLE)}"
        )
    if code not in cycle:
        raise RollExecutorError(f"month code {code!r} is not in the {root!r} cycle {cycle}")
    i = cycle.index(code)
    if i + 1 < len(cycle):
        next_code, next_year = cycle[i + 1], year
    else:
        next_code, next_year = cycle[0], year + 1
    return ContractSymbol(f"{root}{next_code}{next_year % 100:02d}")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpenPosition:
    """A held futures position eligible for rolling. `qty` is signed (long > 0, short < 0)."""

    contract: ContractSymbol
    qty: int
    avg_price: Decimal


@dataclass(frozen=True)
class RollCalendarEntry:
    """Precomputed roll window for one front contract (a row of roll_calendar.parquet).

    `regime` is "RTH" or "24/7" (crypto post-2026-05-29). `fnd_date` is set only for
    physically-deliverable contracts (CL, GC). Invariants enforced: roll_target <= roll_deadline
    <= ltd_date - you cannot roll after the contract stops trading.
    """

    front_symbol: str
    back_symbol: str
    ltd_date: date
    roll_target: date
    roll_deadline: date
    regime: str = "RTH"
    fnd_date: date | None = None

    def __post_init__(self) -> None:
        if self.roll_target > self.roll_deadline:
            raise RollExecutorError(
                f"roll_target {self.roll_target} > roll_deadline {self.roll_deadline} "
                f"for {self.front_symbol}"
            )
        if self.roll_deadline > self.ltd_date:
            raise RollExecutorError(
                f"roll_deadline {self.roll_deadline} > ltd_date {self.ltd_date} "
                f"for {self.front_symbol} (cannot roll after last trading day)"
            )


@dataclass(frozen=True)
class RollDecision:
    """A roll the engine should execute on `target_date`."""

    front: str
    back: str
    qty: int
    method: RollMethod
    target_date: date
    deadline_date: date
    reason: str


@dataclass(frozen=True)
class RollDivergenceResult:
    """Outcome of a front-vs-back roll-day cross-check (raw-vs-adjusted)."""

    divergence_flag: bool
    detail: str = ""


# ---------------------------------------------------------------------------
# Seams: roll calendar + divergence check
# ---------------------------------------------------------------------------


class RollCalendar(abc.ABC):
    """Roll-window provider. The parquet-backed loader is a later ship; this ABC lets
    `decide_rolls` be written + tested against any source."""

    @abc.abstractmethod
    def lookup(self, contract: ContractSymbol) -> RollCalendarEntry | None:
        """Return the roll entry whose front_symbol == contract, or None if not tracked."""


class StaticRollCalendar(RollCalendar):
    """In-memory roll calendar (Phase A1 + tests). Keyed by front contract symbol."""

    def __init__(self, entries: Iterable[RollCalendarEntry]) -> None:
        self._by_front: dict[str, RollCalendarEntry] = {e.front_symbol: e for e in entries}

    def lookup(self, contract: ContractSymbol) -> RollCalendarEntry | None:
        return self._by_front.get(str(contract))


class RollDivergenceCheck(Protocol):
    """Roll-day front-vs-back price divergence check. The live MultiSourceVerifier roll
    cross-check will satisfy this protocol structurally."""

    def cross_check(self, front: str, back: str, today: date) -> RollDivergenceResult: ...


class StaticRollDivergenceCheck:
    """Explicit, deterministic divergence check (Phase A1). Flags ONLY the (front, back) pairs
    configured as divergent - e.g. the CL April-2020 negative-price archetype (stress scenario 2). Divergence must be configured explicitly rather than silently assumed absent
    until the live verifier cross-check is wired (fail-loud)."""

    def __init__(self, divergent_pairs: Iterable[tuple[str, str]] = ()) -> None:
        self._divergent: set[tuple[str, str]] = set(divergent_pairs)

    def cross_check(self, front: str, back: str, today: date) -> RollDivergenceResult:
        if (front, back) in self._divergent:
            return RollDivergenceResult(True, f"configured divergence for {front}->{back}")
        return RollDivergenceResult(False)


# ---------------------------------------------------------------------------
# RollExecutor (decision layer)
# ---------------------------------------------------------------------------


class RollExecutor:
    """Roll DECISION engine (A1.20). BACKTEST-IS-LIVE: the same decisions apply in backtest
    and live - `decide_rolls` is broker-agnostic; `supports_bag_orders` is the only broker
    capability it consults (passed in, sourced from the active adapter at engine wire-up).

    Order DISPATCH (`execute_roll`) is a later ship - it needs broker BAG-order support +
    slippage + RuntimeContext; it is intentionally absent rather than
    stubbed (A1.19 no-hypothetical-stubs rule)."""

    def __init__(
        self,
        calendar: RollCalendar,
        divergence_check: RollDivergenceCheck,
        *,
        supports_bag_orders: bool,
    ) -> None:
        self._calendar = calendar
        self._divergence = divergence_check
        self._supports_bag = supports_bag_orders

    def decide_rolls(self, today: date, positions: list[OpenPosition]) -> list[RollDecision]:
        """Return the rolls that should execute today.

        For each non-zero position with a tracked roll entry:
        - inside [roll_target, roll_deadline]: SEQUENTIAL if the verifier flags front-vs-back
          divergence (don't combo on possibly-wrong prices); else CALENDAR_SPREAD when the broker
          supports BAG orders, SEQUENTIAL otherwise;
        - past roll_deadline: SEQUENTIAL_EMERGENCY with a PAST_DEADLINE reason (fail-loud alert);
        - before roll_target: no decision (not yet time to roll).
        """
        decisions: list[RollDecision] = []
        for p in positions:
            if p.qty == 0:  # nothing to roll
                continue
            entry = self._calendar.lookup(p.contract)
            if entry is None:
                continue

            if entry.roll_target <= today <= entry.roll_deadline:
                verdict = self._divergence.cross_check(entry.front_symbol, entry.back_symbol, today)
                if verdict.divergence_flag:
                    reason = f"VerifierDivergence: {verdict.detail}; SEQUENTIAL with manual review"
                    decisions.append(
                        RollDecision(
                            front=entry.front_symbol,
                            back=entry.back_symbol,
                            qty=p.qty,
                            method="SEQUENTIAL",
                            target_date=today,
                            deadline_date=entry.roll_deadline,
                            reason=reason,
                        )
                    )
                else:
                    method: RollMethod = "CALENDAR_SPREAD" if self._supports_bag else "SEQUENTIAL"
                    decisions.append(
                        RollDecision(
                            front=entry.front_symbol,
                            back=entry.back_symbol,
                            qty=p.qty,
                            method=method,
                            target_date=today,
                            deadline_date=entry.roll_deadline,
                            reason="OnSchedule",
                        )
                    )
            elif today > entry.roll_deadline:
                decisions.append(
                    RollDecision(
                        front=entry.front_symbol,
                        back=entry.back_symbol,
                        qty=p.qty,
                        method="SEQUENTIAL_EMERGENCY",
                        target_date=today,
                        deadline_date=entry.roll_deadline,
                        reason="PAST_DEADLINE — deadline invariant violated; manual review required",
                    )
                )
        return decisions


# ---------------------------------------------------------------------------
# RollCalendarBuilder — derive roll windows from CME expiry rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _RollSpec:
    """Per-root roll offsets (verified against the CME rulebook). `anchor` is LTD or FND;
    `target_offset` / `deadline_offset` are TRADING-day counts before the anchor."""

    anchor: Literal["LTD", "FND"]
    target_offset: int
    deadline_offset: int


# Verified 2026-05-23 against the CME rulebook (NYMEX/COMEX/CME).
_ROLL_SPECS: Final[dict[str, _RollSpec]] = {
    "ES": _RollSpec("LTD", 5, 2),
    "NQ": _RollSpec("LTD", 5, 2),
    "CL": _RollSpec("LTD", 5, 3),  # deadline also clamped <= FND-1 (flat before notice)
    "GC": _RollSpec("FND", 7, 3),
    "MBT": _RollSpec("LTD", 7, 3),
    "MET": _RollSpec("LTD", 7, 3),
    # Broaden wave 1. Equity/FX roll on volume before LTD (cash / no delivery splice); rates
    # anchor on FND (the late-prior-month Treasury roll, ahead of delivery).
    "YM": _RollSpec("LTD", 5, 2),
    "RTY": _RollSpec("LTD", 5, 2),
    "6E": _RollSpec("LTD", 5, 2),
    "6A": _RollSpec("LTD", 5, 2),
    "ZN": _RollSpec("FND", 5, 2),
    "ZB": _RollSpec("FND", 5, 2),
    # Broaden wave 2: FX rolls on volume before LTD (like 6E/6A); rates anchor FND (like ZN/ZB);
    # COMEX metals anchor FND like GC (last biz day of preceding month); NKD cash-settled -> LTD.
    "6J": _RollSpec("LTD", 5, 2),
    "6B": _RollSpec("LTD", 5, 2),
    "6C": _RollSpec("LTD", 5, 2),
    "6S": _RollSpec("LTD", 5, 2),
    "HG": _RollSpec("FND", 7, 3),
    "SI": _RollSpec("FND", 7, 3),
    "ZT": _RollSpec("FND", 5, 2),
    "ZF": _RollSpec("FND", 5, 2),
    "UB": _RollSpec("FND", 5, 2),
    "NKD": _RollSpec("LTD", 5, 2),
}

_CRYPTO_24_7_LAUNCH: Final[date] = date(2026, 5, 29)  # CME crypto 24/7 launch


class RollCalendarBuilder:
    """Derives `RollCalendarEntry` rows from CME expiry rules.

    This is the auto-computation the module header (lines 14-20) deferred until a CME holiday
    calendar existed: `CMETradingCalendar` now provides the T-N *trading*-day math, so roll
    windows are DERIVED rather than hand-injected. Per-root LTD/FND algorithms are verified
    against the CME rulebook (fail-loud — no faked business-day math):

      ES / NQ : LTD = 3rd Friday of the contract month (cash/SOQ); FND = None.
      CL      : LTD = 3 trading days before the 25th of the month preceding delivery (or before
                the last business day preceding the 25th if the 25th is non-business); FND = LTD+2.
      GC      : FND = last trading day of the month preceding delivery; LTD = 3rd-to-last trading
                day of the delivery month. Roll anchors on FND (it precedes LTD for gold).
      MBT/MET : LTD = last Friday of the contract month (CME Ch.350 "business day in either UK or
                US"; for a month's LAST Friday the only US closures possible are Good Friday and
                Christmas, both also UK closures, so the US calendar implements the both-closed
                test exactly — fail-loud: a future US-only holiday on a last Friday would need a UK
                calendar). FND = None. regime flips RTH -> 24/7 at the 2026-05-29 launch.

    Roll offsets (trading days before the anchor) per `_ROLL_SPECS`; CL's deadline is additionally
    clamped to <= FND-1 so the position is flat before first notice."""

    def __init__(self, calendar_: CMETradingCalendar | None = None) -> None:
        self._cal = calendar_ if calendar_ is not None else CMETradingCalendar()

    def build(self, front: ContractSymbol | str) -> RollCalendarEntry:
        """Build the roll entry for one front contract (e.g. "CLF26")."""
        root, code, year = parse_contract(front)
        spec = _ROLL_SPECS.get(root)
        if spec is None:
            raise RollExecutorError(f"no roll spec for root {root!r}; known: {sorted(_ROLL_SPECS)}")
        month = _MONTH_CODE_TO_NUM[code]
        ltd, fnd, regime = self._derive(root, month, year)
        anchor = ltd if spec.anchor == "LTD" else fnd
        if anchor is None:  # FND-anchored root with no FND date is a contract violation (fail-loud)
            raise RollExecutorError(f"{root!r} anchors on FND but derived no FND date")
        roll_target = self._cal.add_trading_days(anchor, -spec.target_offset)
        roll_deadline = self._cal.add_trading_days(anchor, -spec.deadline_offset)
        if root == "CL" and fnd is not None:
            roll_deadline = min(roll_deadline, self._cal.add_trading_days(fnd, -1))
        return RollCalendarEntry(
            front_symbol=str(front),
            back_symbol=str(next_contract(ContractSymbol(str(front)))),
            ltd_date=ltd,
            roll_target=roll_target,
            roll_deadline=roll_deadline,
            regime=regime,
            fnd_date=fnd,
        )

    def build_cycle(self, root: str, start_year: int, end_year: int) -> list[RollCalendarEntry]:
        """All roll entries for `root` across [start_year, end_year] inclusive (cycle order)."""
        cycle = ROOT_CYCLE.get(root)
        if cycle is None:
            raise RollExecutorError(f"no roll cycle for root {root!r}; known: {sorted(ROOT_CYCLE)}")
        if start_year > end_year:
            raise RollExecutorError(f"start_year {start_year} > end_year {end_year}")
        return [
            self.build(ContractSymbol(f"{root}{code}{year % 100:02d}"))
            for year in range(start_year, end_year + 1)
            for code in cycle
        ]

    def build_static_calendar(
        self, root: str, start_year: int, end_year: int
    ) -> StaticRollCalendar:
        """A `StaticRollCalendar` populated for `root` over [start_year, end_year]."""
        return StaticRollCalendar(self.build_cycle(root, start_year, end_year))

    # -- per-root derivation (returns (ltd, fnd, regime)) ----------------------

    def _derive(self, root: str, month: int, year: int) -> tuple[date, date | None, str]:
        if root in ("ES", "NQ", "YM", "RTY", "NKD"):  # cash-settled equity index
            return self._derive_equity(root, month, year)
        if root in ("6E", "6A", "6J", "6B", "6C", "6S"):  # all G10 FX share the IMM rule
            return self._derive_fx(month, year)
        if root in ("ZN", "ZB", "ZT", "ZF", "UB"):  # CBOT Treasuries share the FND roll
            return self._derive_rates(month, year)
        if root == "CL":
            return self._derive_cl(month, year)
        if root in ("GC", "HG", "SI"):  # COMEX metals share GC's FND rule (verified)
            return self._derive_gc(month, year)
        if root in ("MBT", "MET"):
            return self._derive_crypto(month, year)
        raise RollExecutorError(f"no derivation for root {root!r}")

    def _derive_cl(self, month: int, year: int) -> tuple[date, date, str]:
        anchor_year, anchor_month = (year - 1, 12) if month == 1 else (year, month - 1)
        d25 = date(anchor_year, anchor_month, 25)
        base = d25 if self._cal.is_trading_day(d25) else self._cal.prev_trading_day(d25)
        ltd = self._cal.add_trading_days(base, -3)
        fnd = self._cal.add_trading_days(ltd, 2)
        return ltd, fnd, "RTH"

    def _derive_gc(self, month: int, year: int) -> tuple[date, date, str]:
        prev_year, prev_month = (year - 1, 12) if month == 1 else (year, month - 1)
        fnd = self._last_trading_day_of_month(prev_year, prev_month)
        ltd = self._nth_to_last_trading_day_of_month(year, month, 3)
        return ltd, fnd, "RTH"

    def _derive_crypto(self, month: int, year: int) -> tuple[date, None, str]:
        ltd = self._guard(self._last_friday(year, month))
        regime = "RTH" if ltd < _CRYPTO_24_7_LAUNCH else "24/7"
        return ltd, None, regime

    def _derive_equity(self, root: str, month: int, year: int) -> tuple[date, None, str]:
        """Cash-settled equity index. ES/NQ/YM/RTY settle to the 3rd-Friday SOQ (LTD = 3rd Friday);
        NKD (Nikkei 225 USD) settles to the 2nd-Friday SQ (LTD = the trading day before)."""
        if root == "NKD":
            return self._cal.prev_trading_day(self._second_friday(year, month)), None, "RTH"
        return self._guard(self._third_friday(year, month)), None, "RTH"

    def _derive_fx(self, month: int, year: int) -> tuple[date, None, str]:
        """CME G10 FX (6E/6A/6J/6B/6C/6S): LTD = 2 trading days before the 3rd Wednesday, rolled
        on volume ahead of any delivery, so no FND is needed for the NUL splice."""
        ltd = self._cal.add_trading_days(self._guard(self._third_wednesday(year, month)), -2)
        return ltd, None, "RTH"

    def _derive_rates(self, month: int, year: int) -> tuple[date, date, str]:
        """CBOT Treasuries (ZN/ZB/ZT/ZF/UB): LTD = 7 trading days before the last trading day of
        the delivery month; FND = last trading day of the PRIOR month, so the roll anchors on FND
        (the standard late-prior-month Treasury roll). ZT/ZF actually trade to month-end, but the
        roll is FND-anchored so the LTD field is immaterial; the NUL splice is robust to a few days
        either way (live should rulebook-verify the exact FND)."""
        ltd = self._cal.add_trading_days(self._last_trading_day_of_month(year, month), -7)
        prev_year, prev_month = (year - 1, 12) if month == 1 else (year, month - 1)
        fnd = self._last_trading_day_of_month(prev_year, prev_month)
        return ltd, fnd, "RTH"

    # -- calendar helpers ------------------------------------------------------

    def _guard(self, day: date) -> date:
        """Roll a computed anchor back to the prior trading day if it is closed."""
        return day if self._cal.is_trading_day(day) else self._cal.prev_trading_day(day)

    def _third_friday(self, year: int, month: int) -> date:
        first = date(year, month, 1)
        offset = (calendar.FRIDAY - first.weekday()) % 7
        return date(year, month, 1 + offset + 14)

    def _second_friday(self, year: int, month: int) -> date:
        first = date(year, month, 1)
        offset = (calendar.FRIDAY - first.weekday()) % 7
        return date(year, month, 1 + offset + 7)

    def _third_wednesday(self, year: int, month: int) -> date:
        first = date(year, month, 1)
        offset = (calendar.WEDNESDAY - first.weekday()) % 7
        return date(year, month, 1 + offset + 14)

    def _last_friday(self, year: int, month: int) -> date:
        last = date(year, month, calendar.monthrange(year, month)[1])
        return last - timedelta(days=(last.weekday() - calendar.FRIDAY) % 7)

    def _last_trading_day_of_month(self, year: int, month: int) -> date:
        return self._guard(date(year, month, calendar.monthrange(year, month)[1]))

    def _nth_to_last_trading_day_of_month(self, year: int, month: int, n: int) -> date:
        day = self._last_trading_day_of_month(year, month)
        for _ in range(n - 1):
            day = self._cal.prev_trading_day(day)
        return day
