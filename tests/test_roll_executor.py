"""A1.20 RollExecutor decision-engine test suite.

Per internal design notes:
- Contract-cycle math: parse_contract (valid / short / bad code / bad year / year disambiguation);
  next_contract across quarterly (ES/NQ), monthly (CL/MBT), bi-monthly (GC) incl. year wrap;
  fail-loud on unknown root + month-outside-cycle.
- RollCalendarEntry invariants; StaticRollCalendar lookup; StaticRollDivergenceCheck.
- decide_rolls every branch: before window (no-op), on-schedule CALENDAR_SPREAD / SEQUENTIAL
  (no bag support), verifier-divergence SEQUENTIAL, past-deadline SEQUENTIAL_EMERGENCY; no-entry
  skip; qty=0 skip; inclusive target/deadline boundaries; multi-position; signed short qty.
- Stress scenarios: 2 (CL-2020 negative-price divergence) + 5 (past-deadline alert).
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from futur3.data.types import ContractSymbol
from futur3.execution import RollExecutor as _PkgRollExecutor
from futur3.execution.roll_executor import (
    OpenPosition,
    RollCalendarEntry,
    RollDecision,
    RollDivergenceResult,
    RollExecutor,
    RollExecutorError,
    StaticRollCalendar,
    StaticRollDivergenceCheck,
    next_contract,
    parse_contract,
)


def _entry(
    *,
    front: str = "ESM26",
    back: str = "ESU26",
    ltd: date = date(2026, 6, 19),
    target: date = date(2026, 6, 12),
    deadline: date = date(2026, 6, 17),
    regime: str = "RTH",
    fnd: date | None = None,
) -> RollCalendarEntry:
    return RollCalendarEntry(
        front_symbol=front,
        back_symbol=back,
        ltd_date=ltd,
        roll_target=target,
        roll_deadline=deadline,
        regime=regime,
        fnd_date=fnd,
    )


def _pos(contract: str = "ESM26", qty: int = 2) -> OpenPosition:
    return OpenPosition(contract=ContractSymbol(contract), qty=qty, avg_price=Decimal("5260.00"))


def _executor(
    entries: list[RollCalendarEntry] | None = None,
    *,
    divergent: list[tuple[str, str]] | None = None,
    supports_bag: bool = True,
) -> RollExecutor:
    return RollExecutor(
        StaticRollCalendar(entries if entries is not None else [_entry()]),
        StaticRollDivergenceCheck(divergent or []),
        supports_bag_orders=supports_bag,
    )


# ============================================================================
# TestA1_20_ParseContract
# ============================================================================


class TestA1_20_ParseContract:
    def test_basic(self) -> None:
        assert parse_contract(ContractSymbol("ESM26")) == ("ES", "M", 2026)

    def test_micro_root(self) -> None:
        assert parse_contract("MESU26") == ("MES", "U", 2026)

    def test_year_disambiguation_2000s(self) -> None:
        assert parse_contract("ESM49")[2] == 2049

    def test_year_disambiguation_1900s(self) -> None:
        assert parse_contract("ESM50")[2] == 1950

    def test_too_short_raises(self) -> None:
        with pytest.raises(RollExecutorError, match="too short"):
            parse_contract("M26")

    def test_bad_month_code_raises(self) -> None:
        with pytest.raises(RollExecutorError, match="invalid CME month code"):
            parse_contract("ESB26")  # B is not a CME month code

    def test_bad_year_raises(self) -> None:
        with pytest.raises(RollExecutorError, match="invalid 2-digit year"):
            parse_contract("ESMZ9")  # "Z9" is not all-digit


# ============================================================================
# TestA1_20_NextContract
# ============================================================================


class TestA1_20_NextContract:
    def test_quarterly_mid_cycle(self) -> None:
        assert next_contract(ContractSymbol("ESM26")) == "ESU26"  # Jun -> Sep

    def test_quarterly_year_wrap(self) -> None:
        assert next_contract(ContractSymbol("ESZ26")) == "ESH27"  # Dec -> Mar next yr

    def test_quarterly_nq(self) -> None:
        assert next_contract(ContractSymbol("NQH26")) == "NQM26"  # Mar -> Jun

    def test_monthly_mid_cycle(self) -> None:
        assert next_contract(ContractSymbol("CLM26")) == "CLN26"  # Jun -> Jul

    def test_monthly_year_wrap(self) -> None:
        assert next_contract(ContractSymbol("CLZ26")) == "CLF27"  # Dec -> Jan next yr

    def test_bi_monthly(self) -> None:
        assert next_contract(ContractSymbol("GCM26")) == "GCQ26"  # Jun -> Aug

    def test_bi_monthly_year_wrap(self) -> None:
        assert next_contract(ContractSymbol("GCZ26")) == "GCG27"  # Dec -> Feb next yr

    def test_crypto_monthly(self) -> None:
        assert next_contract(ContractSymbol("MBTK26")) == "MBTM26"  # May -> Jun

    def test_unknown_root_raises(self) -> None:
        with pytest.raises(RollExecutorError, match="no roll cycle configured"):
            next_contract(ContractSymbol("ZZM26"))

    def test_month_outside_cycle_raises(self) -> None:
        # F (Jan) is a valid CME code but NOT in the ES quarterly cycle.
        with pytest.raises(RollExecutorError, match="not in the 'ES' cycle"):
            next_contract(ContractSymbol("ESF26"))


# ============================================================================
# TestA1_20_CalendarEntry / Calendar / DivergenceCheck
# ============================================================================


class TestA1_20_CalendarEntry:
    def test_valid_entry(self) -> None:
        e = _entry()
        assert e.front_symbol == "ESM26"
        assert e.roll_target <= e.roll_deadline <= e.ltd_date

    def test_target_after_deadline_raises(self) -> None:
        with pytest.raises(RollExecutorError, match=r"roll_target .* > roll_deadline"):
            _entry(target=date(2026, 6, 18), deadline=date(2026, 6, 17))

    def test_deadline_after_ltd_raises(self) -> None:
        with pytest.raises(RollExecutorError, match="cannot roll after last trading day"):
            _entry(deadline=date(2026, 6, 20), ltd=date(2026, 6, 19))


class TestA1_20_StaticCalendar:
    def test_lookup_hit(self) -> None:
        cal = StaticRollCalendar([_entry()])
        assert cal.lookup(ContractSymbol("ESM26")) is not None

    def test_lookup_miss(self) -> None:
        cal = StaticRollCalendar([_entry()])
        assert cal.lookup(ContractSymbol("NQM26")) is None


class TestA1_20_DivergenceCheck:
    def test_configured_pair_flags(self) -> None:
        chk = StaticRollDivergenceCheck([("CLK26", "CLM26")])
        res = chk.cross_check("CLK26", "CLM26", date(2026, 4, 15))
        assert res.divergence_flag is True
        assert "CLK26->CLM26" in res.detail

    def test_unconfigured_pair_clears(self) -> None:
        chk = StaticRollDivergenceCheck([])
        assert chk.cross_check("ESM26", "ESU26", date(2026, 6, 12)) == RollDivergenceResult(False)


# ============================================================================
# TestA1_20_DecideRolls
# ============================================================================


class TestA1_20_DecideRolls:
    def test_before_window_no_decision(self) -> None:
        out = _executor().decide_rolls(date(2026, 6, 10), [_pos()])  # target is 06-12
        assert out == []

    def test_on_schedule_calendar_spread(self) -> None:
        out = _executor(supports_bag=True).decide_rolls(date(2026, 6, 12), [_pos()])
        assert len(out) == 1
        d = out[0]
        assert d.method == "CALENDAR_SPREAD"
        assert d.reason == "OnSchedule"
        assert d.front == "ESM26" and d.back == "ESU26"
        assert d.qty == 2
        assert d.target_date == date(2026, 6, 12)
        assert d.deadline_date == date(2026, 6, 17)

    def test_on_schedule_sequential_when_no_bag_support(self) -> None:
        out = _executor(supports_bag=False).decide_rolls(date(2026, 6, 12), [_pos()])
        assert out[0].method == "SEQUENTIAL"
        assert out[0].reason == "OnSchedule"

    def test_divergence_forces_sequential(self) -> None:
        out = _executor(divergent=[("ESM26", "ESU26")]).decide_rolls(date(2026, 6, 12), [_pos()])
        assert out[0].method == "SEQUENTIAL"
        assert out[0].reason.startswith("VerifierDivergence")

    def test_past_deadline_emergency(self) -> None:
        out = _executor().decide_rolls(date(2026, 6, 18), [_pos()])  # deadline is 06-17
        assert out[0].method == "SEQUENTIAL_EMERGENCY"
        assert "PAST_DEADLINE" in out[0].reason

    def test_no_calendar_entry_skipped(self) -> None:
        out = _executor().decide_rolls(date(2026, 6, 12), [_pos("NQM26")])  # only ESM26 tracked
        assert out == []

    def test_zero_qty_skipped(self) -> None:
        out = _executor().decide_rolls(date(2026, 6, 12), [_pos(qty=0)])
        assert out == []

    def test_target_boundary_inclusive(self) -> None:
        out = _executor().decide_rolls(date(2026, 6, 12), [_pos()])  # today == roll_target
        assert len(out) == 1 and out[0].reason == "OnSchedule"

    def test_deadline_boundary_inclusive(self) -> None:
        out = _executor().decide_rolls(date(2026, 6, 17), [_pos()])  # today == roll_deadline
        assert len(out) == 1 and out[0].method == "CALENDAR_SPREAD"

    def test_signed_short_qty_preserved(self) -> None:
        out = _executor().decide_rolls(date(2026, 6, 12), [_pos(qty=-3)])
        assert out[0].qty == -3

    def test_multi_position(self) -> None:
        entries = [
            _entry(front="ESM26", back="ESU26"),
            _entry(
                front="CLN26",
                back="CLQ26",
                ltd=date(2026, 6, 22),
                target=date(2026, 6, 15),
                deadline=date(2026, 6, 19),
                fnd=date(2026, 6, 23),
            ),
        ]
        ex = _executor(entries)
        out = ex.decide_rolls(date(2026, 6, 16), [_pos("ESM26"), _pos("CLN26")])
        # ESM26: 06-16 in [06-12, 06-17] -> decision; CLN26: 06-16 in [06-15, 06-19] -> decision
        assert {d.front for d in out} == {"ESM26", "CLN26"}


# ============================================================================
# TestA1_20_StressScenarios
# ============================================================================


class TestA1_20_StressScenarios:
    def test_scenario2_cl_2020_negative_price_divergence(self) -> None:
        # CL super-contango: verifier flags raw-vs-adjusted divergence -> SEQUENTIAL, NOT combo.
        entry = _entry(
            front="CLK20",
            back="CLM20",
            ltd=date(2020, 4, 21),
            target=date(2020, 4, 14),
            deadline=date(2020, 4, 17),
            fnd=date(2020, 4, 22),
        )
        ex = _executor([entry], divergent=[("CLK20", "CLM20")])
        out = ex.decide_rolls(date(2020, 4, 15), [_pos("CLK20")])
        assert out[0].method == "SEQUENTIAL"
        assert "manual review" in out[0].reason

    def test_scenario5_past_deadline_alert(self) -> None:
        # Held past deadline (engine downtime) -> emergency sequential + fail-loud alert.
        out = _executor().decide_rolls(date(2026, 6, 25), [_pos()])
        decision: RollDecision = out[0]
        assert decision.method == "SEQUENTIAL_EMERGENCY"
        assert "PAST_DEADLINE" in decision.reason


def test_exported_from_execution_package() -> None:
    assert _PkgRollExecutor is RollExecutor
