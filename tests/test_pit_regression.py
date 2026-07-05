"""A1.17 — Point-In-Time (PIT) regression suite.

Pins bug class 5 (look-ahead leak) — one of the highest-stakes invariants in
the apparatus. PIT honesty must hold at EVERY layer:

1. DataSource.latest_settle(contract, as_of): never returns a record whose
   as_of_date > as_of.date().
2. ReplayDataSource: future-dated archive rows EXCLUDED from PIT replay.
3. MultiSourceVerifier: verifier_run_hash includes settle.as_of_iso so
   replayed-at-different-time queries with identical settle data produce
   identical hash (bit-repro corrective fix).
4. StoreQuery: half-open interval [ts_start, ts_end) + TZ-aware required
   at construction.
5. FutureDatedSourceError: exception infrastructure for sources that
   detect a future-dated record at fetch time.

These tests are the REGRESSION GATE — when any of them fails, look-ahead
leak has been introduced somewhere in the pipeline. They live in their own
suite (separate from per-component tests) so the regression intent is
explicit in the test layout.

PIT honesty is LOAD-BEARING: all strategy work depends on this gate holding.

References:
- the verifier design (revision preservation contract)
- the data-layer design (provenance hash chain)
"""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import cast

import pytest

from futur3.data.source import (
    BarsNotSupported,
    ContractStillActiveError,
    DataSourceError,
    FutureDatedSourceError,
    GeoBlockedError,
)
from futur3.data.sources.cme_eod import CMEEODDataSource
from futur3.data.sources.replay import ReplayDataSource
from futur3.data.types import (
    BarResolution,
    ContractSymbol,
    RawBar,
    Settle,
    SettleState,
)
from futur3.storage.abcs import StoreQuery

# ============================================================================
# Helpers
# ============================================================================


def _sha(seed: str) -> str:
    """Return 64-char hex SHA256 of `seed` — meets Settle.content_bytes_sha invariant."""
    return hashlib.sha256(seed.encode()).hexdigest()


def _seed_archive_row(
    archive_root: Path,
    *,
    contract: str = "ESM26",
    contract_root: str = "ES",
    as_of_date: date,
    settle: str = "5260.00",
    settle_state: SettleState = "final",
    as_of_iso: datetime,
    content_bytes_sha: str | None = None,
) -> None:
    """Seed a single Settle row using CMEEODDataSource._write_archive
    (the canonical seed path that ReplayDataSource scans for).
    """
    sha = content_bytes_sha or _sha(f"{contract}_{as_of_date}_{settle_state}")
    s = Settle(
        contract=ContractSymbol(contract),
        as_of_date=as_of_date,
        settle=Decimal(settle),
        settle_state=cast(SettleState, settle_state),
        open=Decimal("5252.25"),
        high=Decimal("5263.50"),
        low=Decimal("5248.75"),
        last=Decimal(settle),
        change=Decimal("8.00"),
        volume_est=1_280_000,
        oi_prior=1_500_000,
        source_id="cme_public_settlements",
        as_of_iso=as_of_iso,
        content_bytes_sha=sha,
        cme_month_code="M",
    )
    cme = CMEEODDataSource(archive_root=archive_root)
    cme._write_archive(contract_root, [s])


# ============================================================================
# TestA1_17_PITSettleReplay — boundary scenarios for ReplayDataSource
# ============================================================================


class TestA1_17_PITSettleReplay:
    """Canonical PIT scenarios for the BACKTEST-IS-LIVE foundation."""

    def test_future_year_row_excluded(self, tmp_path: Path) -> None:
        """2027 row in archive must NOT be emitted when as_of is 2026."""
        root = tmp_path / "cme_eod_archive"
        _seed_archive_row(
            root,
            as_of_date=date(2027, 1, 15),
            as_of_iso=datetime(2027, 1, 15, 22, 0, 0, tzinfo=UTC),
        )
        # Also seed a valid 2026 row so replay finds something
        _seed_archive_row(
            root,
            as_of_date=date(2026, 5, 21),
            as_of_iso=datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        )
        src = ReplayDataSource(archive_root=root)
        result = src.latest_settle(
            ContractSymbol("ESM26"),
            datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        )
        # PIT cutoff: only 2026 row is eligible. 2027 row is FUTURE.
        assert result is not None
        assert result.as_of_date == date(2026, 5, 21)

    def test_same_day_row_included(self, tmp_path: Path) -> None:
        """Row's as_of_date == query as_of.date() IS included (PIT-cutoff is <=)."""
        root = tmp_path / "cme_eod_archive"
        _seed_archive_row(
            root,
            as_of_date=date(2026, 5, 21),
            as_of_iso=datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        )
        src = ReplayDataSource(archive_root=root)
        result = src.latest_settle(
            ContractSymbol("ESM26"),
            datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        )
        assert result is not None
        assert result.as_of_date == date(2026, 5, 21)

    def test_one_day_past_as_of_excluded(self, tmp_path: Path) -> None:
        """Row at as_of_date = as_of.date()+1 MUST NOT leak."""
        root = tmp_path / "cme_eod_archive"
        _seed_archive_row(
            root,
            as_of_date=date(2026, 5, 22),
            as_of_iso=datetime(2026, 5, 22, 22, 0, 0, tzinfo=UTC),
        )
        # No prior row in archive
        src = ReplayDataSource(archive_root=root)
        result = src.latest_settle(
            ContractSymbol("ESM26"),
            datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        )
        # PIT: future-dated row excluded → no eligible Settles → None
        assert result is None

    def test_prior_day_row_included(self, tmp_path: Path) -> None:
        """Row at as_of_date = as_of.date()-1 IS included."""
        root = tmp_path / "cme_eod_archive"
        _seed_archive_row(
            root,
            as_of_date=date(2026, 5, 20),
            as_of_iso=datetime(2026, 5, 20, 22, 0, 0, tzinfo=UTC),
        )
        src = ReplayDataSource(archive_root=root)
        result = src.latest_settle(
            ContractSymbol("ESM26"),
            datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        )
        assert result is not None
        assert result.as_of_date == date(2026, 5, 20)

    def test_multi_row_picks_latest_pit_eligible(self, tmp_path: Path) -> None:
        """Multiple rows; query picks the LATEST that's PIT-eligible."""
        root = tmp_path / "cme_eod_archive"
        # Seed 5-21, 5-22, 5-23
        for d in (date(2026, 5, 21), date(2026, 5, 22), date(2026, 5, 23)):
            _seed_archive_row(
                root,
                as_of_date=d,
                as_of_iso=datetime(d.year, d.month, d.day, 22, 0, 0, tzinfo=UTC),
            )
        src = ReplayDataSource(archive_root=root)
        # Query as_of = 5-22 → picks 5-22 (latest <= cutoff)
        result = src.latest_settle(
            ContractSymbol("ESM26"),
            datetime(2026, 5, 22, 23, 59, 59, tzinfo=UTC),
        )
        assert result is not None
        assert result.as_of_date == date(2026, 5, 22)

    def test_preliminary_then_final_pit_aware(self, tmp_path: Path) -> None:
        """Same as_of_date with prelim+final coexisting; query before final-publish
        must NOT see final state (final's as_of_iso is in the future relative to query).

        Note: ReplayDataSource's PIT filter is on as_of_date (not as_of_iso). So
        both preliminary AND final rows for the same date are eligible whenever
        date is in the past — the tie-break picks the highest-rank settle_state.
        This is INTENTIONAL: replay represents 'I have access to everything
        published by EOD of as_of_date'. The bit-repro hash chain captures the
        final, since that's the canonical record.

        This test pins that behavior so future changes (e.g., adding an
        as_of_iso-based PIT filter for intraday replay) are intentional.
        """
        root = tmp_path / "cme_eod_archive"
        d = date(2026, 5, 21)
        _seed_archive_row(
            root,
            as_of_date=d,
            settle_state="preliminary",
            as_of_iso=datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
            content_bytes_sha=_sha("prelim"),
        )
        _seed_archive_row(
            root,
            as_of_date=d,
            settle_state="final",
            as_of_iso=datetime(2026, 5, 22, 14, 0, 0, tzinfo=UTC),
            content_bytes_sha=_sha("final"),
        )
        src = ReplayDataSource(archive_root=root)
        result = src.latest_settle(
            ContractSymbol("ESM26"),
            datetime(2026, 5, 22, 23, 59, 59, tzinfo=UTC),
        )
        # When BOTH rows are PIT-eligible (date <= as_of), tie-break picks final
        # (higher settle_state rank).
        assert result is not None
        assert result.settle_state == "final"


# ============================================================================
# TestA1_17_PITStorageQuery — StoreQuery half-open + TZ-aware invariants
# ============================================================================


class TestA1_17_PITStorageQuery:
    def test_half_open_interval_default(self) -> None:
        """StoreQuery default is half-open [ts_start, ts_end) — PIT-friendly."""
        q = StoreQuery(
            contract=ContractSymbol("ESM26"),
            ts_start=datetime(2026, 5, 21, 14, 0, 0, tzinfo=UTC),
            ts_end=datetime(2026, 5, 21, 15, 0, 0, tzinfo=UTC),
        )
        assert q.inclusive_end is False

    def test_naive_ts_start_rejected(self) -> None:
        """PIT requires absolute time. Naive datetime = ambiguous = bug class 7."""
        with pytest.raises(ValueError, match="ts_start must be TZ-aware"):
            StoreQuery(
                contract=ContractSymbol("ESM26"),
                ts_start=datetime(2026, 5, 21, 14, 0, 0),  # naive
                ts_end=datetime(2026, 5, 21, 15, 0, 0, tzinfo=UTC),
            )

    def test_naive_ts_end_rejected(self) -> None:
        with pytest.raises(ValueError, match="ts_end must be TZ-aware"):
            StoreQuery(
                contract=ContractSymbol("ESM26"),
                ts_start=datetime(2026, 5, 21, 14, 0, 0, tzinfo=UTC),
                ts_end=datetime(2026, 5, 21, 15, 0, 0),  # naive
            )

    def test_inverted_interval_rejected(self) -> None:
        """ts_end <= ts_start is non-sensical for PIT queries."""
        with pytest.raises(ValueError, match="ts_end must be > ts_start"):
            StoreQuery(
                contract=ContractSymbol("ESM26"),
                ts_start=datetime(2026, 5, 21, 15, 0, 0, tzinfo=UTC),
                ts_end=datetime(2026, 5, 21, 14, 0, 0, tzinfo=UTC),
            )


# ============================================================================
# TestA1_17_PITRawBarTimestamp — RawBar TZ-aware invariant
# ============================================================================


class TestA1_17_PITRawBarTimestamp:
    def test_raw_bar_ts_must_be_tz_aware(self) -> None:
        """RawBar enforces TZ-aware ts at construction — Bug class 7 guard."""
        with pytest.raises(ValueError, match="ts must be IANA-TZ-aware"):
            RawBar(
                contract=ContractSymbol("ESM26"),
                ts=datetime(2026, 5, 21, 14, 0, 0),  # naive
                resolution=BarResolution.MIN_5,
                open=Decimal("5260"),
                high=Decimal("5260"),
                low=Decimal("5260"),
                close=Decimal("5260"),
                volume=100,
                oi=None,
                source_id="test",
                as_of_iso=datetime(2026, 5, 21, 14, 0, 0, tzinfo=UTC),
                content_bytes_sha=_sha("x"),
            )

    def test_raw_bar_as_of_iso_must_be_tz_aware(self) -> None:
        """RawBar as_of_iso (capture time) must be TZ-aware — PIT-honest."""
        with pytest.raises(ValueError, match="as_of_iso must be IANA-TZ-aware"):
            RawBar(
                contract=ContractSymbol("ESM26"),
                ts=datetime(2026, 5, 21, 14, 0, 0, tzinfo=UTC),
                resolution=BarResolution.MIN_5,
                open=Decimal("5260"),
                high=Decimal("5260"),
                low=Decimal("5260"),
                close=Decimal("5260"),
                volume=100,
                oi=None,
                source_id="test",
                as_of_iso=datetime(2026, 5, 21, 14, 0, 0),  # naive
                content_bytes_sha=_sha("x"),
            )


# ============================================================================
# TestA1_17_PITSettleDataclass — Settle TZ-aware as_of_iso invariant
# ============================================================================


class TestA1_17_PITSettleDataclass:
    def test_settle_as_of_iso_tz_aware_when_set(self) -> None:
        """Settle.as_of_iso is Optional but when SET must be TZ-aware."""
        with pytest.raises(ValueError, match="as_of_iso must be IANA-TZ-aware"):
            Settle(
                contract=ContractSymbol("ESM26"),
                as_of_date=date(2026, 5, 21),
                settle=Decimal("5260"),
                settle_state="final",
                open=Decimal("5260"),
                high=Decimal("5260"),
                low=Decimal("5260"),
                last=Decimal("5260"),
                change=Decimal("0"),
                volume_est=100,
                oi_prior=0,
                source_id="test",
                as_of_iso=datetime(2026, 5, 21, 14, 0, 0),  # naive
                content_bytes_sha=_sha("x"),
            )

    def test_settle_with_none_as_of_iso_allowed_at_construction(self) -> None:
        """Settle dataclass allows None as_of_iso at construction.

        Verifier-side fix: if a source produces None as_of_iso, the
        verifier raises DataSourceError before computing hash. This pins the
        dataclass-level allowance + flags the seam where M4 enforcement happens.
        """
        # Should NOT raise
        s = Settle(
            contract=ContractSymbol("ESM26"),
            as_of_date=date(2026, 5, 21),
            settle=Decimal("5260"),
            settle_state="final",
            open=Decimal("5260"),
            high=Decimal("5260"),
            low=Decimal("5260"),
            last=Decimal("5260"),
            change=Decimal("0"),
            volume_est=100,
            oi_prior=0,
            source_id="test",
            as_of_iso=None,
            content_bytes_sha=_sha("x"),
        )
        assert s.as_of_iso is None


# ============================================================================
# TestA1_17_PITFutureDatedException — bug class 5 exception infrastructure
# ============================================================================


class TestA1_17_PITFutureDatedException:
    def test_future_dated_source_error_extends_data_source_error(self) -> None:
        assert issubclass(FutureDatedSourceError, DataSourceError)

    def test_future_dated_source_error_is_exception(self) -> None:
        assert issubclass(FutureDatedSourceError, Exception)

    def test_can_raise_with_message(self) -> None:
        """Smoke: exception is raisable + carries message."""
        with pytest.raises(FutureDatedSourceError, match="future-dated bar"):
            raise FutureDatedSourceError("source X returned a future-dated bar ts=2027-01-01")

    def test_distinct_from_other_data_source_errors(self) -> None:
        """FutureDatedSourceError is NOT confused with siblings (sanity)."""
        assert FutureDatedSourceError is not GeoBlockedError
        assert FutureDatedSourceError is not BarsNotSupported
        assert FutureDatedSourceError is not ContractStillActiveError


# ============================================================================
# TestA1_17_PITDeterminism — same archive + same query -> same result
# ============================================================================


class TestA1_17_PITDeterminism:
    def test_replay_same_query_returns_same_settle(self, tmp_path: Path) -> None:
        """Bit-repro + PIT: same (archive, as_of) -> bit-equal Settle across replays."""
        root = tmp_path / "cme_eod_archive"
        _seed_archive_row(
            root,
            as_of_date=date(2026, 5, 21),
            as_of_iso=datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        )
        src1 = ReplayDataSource(archive_root=root)
        src2 = ReplayDataSource(archive_root=root)
        result1 = src1.latest_settle(
            ContractSymbol("ESM26"),
            datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        )
        result2 = src2.latest_settle(
            ContractSymbol("ESM26"),
            datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        )
        assert result1 == result2
        assert result1 is not None
        assert result1.content_bytes_sha == result2.content_bytes_sha  # type: ignore[union-attr]
