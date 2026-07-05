"""Roll-exclusion test suite (futur3.data.intraday_continuous).

The $0 exploration-tier roll handling for Databento `.c.0` continuous intraday data: roll boundaries
(underlying-contract changes) are flagged and the contaminated cross-contract return is excluded.
These tests lock the core guarantee -- a roll-boundary jump NEVER enters the return series -- plus
the within-contract returns, the validation guards, and the no-roll degenerate case.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from futur3.data.intraday_continuous import (
    IntradayContinuousError,
    RollExcludedSeries,
    build_roll_excluded_series,
)
from futur3.data.types import BarResolution, ContractSymbol, RawBar, content_sha256

_T0 = datetime(2023, 1, 2, 23, 0, tzinfo=UTC)


def _bar(contract: str, i: int, close: str) -> RawBar:
    """A valid hourly RawBar at hour-offset `i` with O=H=L=C=`close` (flat bar; high>=low holds)."""
    px = Decimal(close)
    return RawBar(
        contract=ContractSymbol(contract),
        ts=_T0 + timedelta(hours=i),
        resolution=BarResolution.HOUR_1,
        open=px,
        high=px,
        low=px,
        close=px,
        volume=1,
        oi=None,
        source_id="test",
        as_of_iso=_T0,
        content_bytes_sha=content_sha256(f"{contract}|{i}".encode()),
    )


# ============================================================================
# TestRollFlags - boundaries mark underlying-contract changes
# ============================================================================


class TestRollFlags:
    def test_flags_mark_contract_changes(self) -> None:
        bars = [
            _bar("ESH3", 0, "100"),
            _bar("ESH3", 1, "101"),
            _bar("ESM3", 2, "110"),
            _bar("ESM3", 3, "111"),
            _bar("ESU3", 4, "120"),
        ]
        s = build_roll_excluded_series("ES", bars)
        assert s.roll_flags == (False, False, True, False, True)
        assert s.n_rolls == 2

    def test_single_contract_has_no_rolls(self) -> None:
        bars = [_bar("ESH3", i, str(100 + i)) for i in range(5)]
        s = build_roll_excluded_series("ES", bars)
        assert s.roll_flags == (False,) * 5
        assert s.n_rolls == 0


# ============================================================================
# TestRollCleanReturns - the contaminated boundary return is excluded
# ============================================================================


class TestRollCleanReturns:
    def test_first_return_is_none(self) -> None:
        s = build_roll_excluded_series("ES", [_bar("ESH3", 0, "100"), _bar("ESH3", 1, "101")])
        r = s.roll_clean_returns()
        assert r[0] is None
        assert r[1] == pytest.approx(0.01)

    def test_roll_boundary_excluded_no_phantom(self) -> None:
        # ESH3 100 -> 101 (+1%), roll to ESM3 at 110 (a +8.9% PHANTOM), then ESM3 110 -> 112.
        bars = [
            _bar("ESH3", 0, "100"),
            _bar("ESH3", 1, "101"),
            _bar("ESM3", 2, "110"),
            _bar("ESM3", 3, "112"),
        ]
        r = build_roll_excluded_series("ES", bars).roll_clean_returns()
        assert r[0] is None
        assert r[1] == pytest.approx(0.01)  # within ESH3
        assert r[2] is None  # ESH3 -> ESM3 boundary: the +8.9% phantom is EXCLUDED
        assert r[3] == pytest.approx(112 / 110 - 1)  # within ESM3
        # the phantom (110/101 - 1 ~= +8.9%) appears nowhere
        assert all(x is None or abs(x) < 0.05 for x in r)

    def test_length_matches_bars(self) -> None:
        bars = [_bar("ESH3", i, str(100 + i)) for i in range(6)]
        assert len(build_roll_excluded_series("ES", bars).roll_clean_returns()) == 6

    def test_zero_prev_close_excluded(self) -> None:
        # a zero base price yields no division-by-zero -- it is excluded as None
        s = build_roll_excluded_series("ES", [_bar("ESH3", 0, "0"), _bar("ESH3", 1, "100")])
        assert s.roll_clean_returns()[1] is None


# ============================================================================
# TestValidation - fail-loud on contract violations (fail-loud: never a silent skip)
# ============================================================================


class TestValidation:
    def test_empty_raises(self) -> None:
        with pytest.raises(IntradayContinuousError, match="no bars"):
            build_roll_excluded_series("ES", [])

    def test_nonincreasing_ts_raises(self) -> None:
        dup = [_bar("ESH3", 0, "100"), _bar("ESH3", 0, "101")]  # same ts
        with pytest.raises(IntradayContinuousError, match="increasing"):
            build_roll_excluded_series("ES", dup)

    def test_roll_flag_zero_true_raises(self) -> None:
        bars = (_bar("ESH3", 0, "100"), _bar("ESH3", 1, "101"))
        with pytest.raises(IntradayContinuousError, match=r"roll_flags\[0\]"):
            RollExcludedSeries(root="ES", bars=bars, roll_flags=(True, False))

    def test_length_mismatch_raises(self) -> None:
        bars = (_bar("ESH3", 0, "100"), _bar("ESH3", 1, "101"))
        with pytest.raises(IntradayContinuousError, match="length mismatch"):
            RollExcludedSeries(root="ES", bars=bars, roll_flags=(False,))
