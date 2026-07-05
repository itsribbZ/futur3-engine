"""A1.7 ReplayDataSource test suite — fixture-based coverage.

Test discipline:
- Round-trip realism: archives written via CMEEODDataSource._write_archive
  (the exact code path that production CME archive writes go through),
  then read back via ReplayDataSource.
- PIT-honesty verified: future-dated as_of cannot leak future records.
- Decimal precision preserved exactly through Polars Parquet round-trip.
- Determinism: same archive + same query → same Settle (bit-repro anchor).
- ABC contract: get_bars raises BarsNotSupported (A1.16 deferral);
  get_ticks raises TicksNotSupported (default).

References:
- futur3/data/sources/replay.py (implementation)
- futur3/data/sources/cme_eod.py::_write_archive (archive writer)
- the data-layer design (A1.7 spec)
- the verifier spec (verifier-side usage)
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest

from futur3.data.source import (
    BarsNotSupported,
    ContractNotConfigured,
    DataSource,
    DataSourceError,
    TicksNotSupported,
)
from futur3.data.sources.cme_eod import CMEEODDataSource
from futur3.data.sources.replay import (
    DEFAULT_CME_ARCHIVE_ROOT,
    ReplayDataSource,
)
from futur3.data.types import (
    BarResolution,
    ContractSymbol,
    Settle,
    SourceTier,
)

# ============================================================================
# Helpers + fixtures
# ============================================================================


def _make_settle(
    *,
    contract: str = "ESM26",
    as_of_date: date,
    settle: str,
    settle_state: str = "preliminary",
    open_: str = "5230.00",
    high: str = "5260.00",
    low: str = "5220.00",
    last: str = "5240.00",
    change: str = "0.00",
    volume_est: int = 1_000_000,
    oi_prior: int = 1_500_000,
    source_id: str = "cme_public_settlements",
    as_of_iso: datetime | None = None,
    content_bytes_sha: str = "a" * 64,
    cme_month_code: str = "M",
) -> Settle:
    """Construct a Settle with sensible defaults for archive seeding."""
    return Settle(
        contract=ContractSymbol(contract),
        as_of_date=as_of_date,
        settle=Decimal(settle),
        settle_state=settle_state,  # type: ignore[arg-type]
        open=Decimal(open_),
        high=Decimal(high),
        low=Decimal(low),
        last=Decimal(last),
        change=Decimal(change),
        volume_est=volume_est,
        oi_prior=oi_prior,
        source_id=source_id,
        as_of_iso=as_of_iso or datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        content_bytes_sha=content_bytes_sha,
        cme_month_code=cme_month_code,
    )


def _seed_archive(archive_root: Path, contract_root: str, settles: list[Settle]) -> None:
    """Use CMEEODDataSource._write_archive to seed an archive in the locked schema."""
    cme = CMEEODDataSource(archive_root=archive_root)
    cme._write_archive(contract_root, settles)


@pytest.fixture
def es_archive(tmp_path: Path) -> Path:
    """Temp archive seeded with 3 ESM26 settles across May 19/20/21 2026.

    Includes a preliminary→final transition on May 21 (two rows same date).
    """
    archive_root = tmp_path / "cme_eod_archive"
    settles = [
        _make_settle(
            as_of_date=date(2026, 5, 19),
            settle="5240.00",
            settle_state="final",
            content_bytes_sha="11" * 32,
        ),
        _make_settle(
            as_of_date=date(2026, 5, 20),
            settle="5252.00",
            settle_state="final",
            content_bytes_sha="22" * 32,
        ),
        _make_settle(
            as_of_date=date(2026, 5, 21),
            settle="5260.00",
            settle_state="preliminary",
            content_bytes_sha="33" * 32,
        ),
        _make_settle(
            as_of_date=date(2026, 5, 21),
            settle="5260.25",
            settle_state="final",
            content_bytes_sha="44" * 32,
        ),
    ]
    _seed_archive(archive_root, "ES", settles)
    return archive_root


@pytest.fixture
def empty_tmp_archive(tmp_path: Path) -> Path:
    """Existing-but-empty archive root."""
    archive_root = tmp_path / "cme_eod_archive"
    archive_root.mkdir()
    return archive_root


# ============================================================================
# TestA1_7_Imports — module + ABC compliance
# ============================================================================


class TestA1_7_Imports:
    def test_replay_is_datasource(self) -> None:
        assert issubclass(ReplayDataSource, DataSource)

    def test_default_constants_match_cme_layout(self) -> None:
        assert Path("data/cme_eod_archive") == DEFAULT_CME_ARCHIVE_ROOT

    def test_default_source_id_is_replay(self) -> None:
        src = ReplayDataSource(archive_root=Path("/tmp/x"))
        assert src.source_id == "replay"

    def test_custom_source_id_param(self) -> None:
        src = ReplayDataSource(archive_root=Path("/tmp/x"), source_id="replay@cme")
        assert src.source_id == "replay@cme"

    def test_default_tier_is_t4_derived(self) -> None:
        src = ReplayDataSource(archive_root=Path("/tmp/x"))
        assert src.tier == SourceTier.T4_DERIVED

    def test_custom_tier_param(self) -> None:
        src = ReplayDataSource(archive_root=Path("/tmp/x"), tier=SourceTier.T2_EXCHANGE)
        assert src.tier == SourceTier.T2_EXCHANGE

    def test_repr_includes_source_id_and_tier(self) -> None:
        src = ReplayDataSource(archive_root=Path("/tmp/x"), source_id="replay@cme")
        rep = repr(src)
        assert "replay@cme" in rep
        assert "T4_DERIVED" in rep


# ============================================================================
# TestA1_7_Healthcheck — archive root existence
# ============================================================================


class TestA1_7_Healthcheck:
    def test_healthcheck_true_when_archive_exists(self, empty_tmp_archive: Path) -> None:
        src = ReplayDataSource(archive_root=empty_tmp_archive)
        assert src.healthcheck() is True

    def test_healthcheck_false_when_archive_missing(self, tmp_path: Path) -> None:
        src = ReplayDataSource(archive_root=tmp_path / "does_not_exist")
        assert src.healthcheck() is False


# ============================================================================
# TestA1_7_LatestSettle — happy paths + filtering + PIT-honesty
# ============================================================================


class TestA1_7_LatestSettle:
    def test_latest_settle_round_trip_returns_settle(self, es_archive: Path) -> None:
        """ESM26, as_of = May 21 2026 → returns the latest archived Settle (final wins)."""
        src = ReplayDataSource(archive_root=es_archive)
        settle = src.latest_settle(
            ContractSymbol("ESM26"),
            datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        )
        assert settle is not None
        # May 21 has prelim+final; final ranks higher
        assert settle.as_of_date == date(2026, 5, 21)
        assert settle.settle_state == "final"
        assert settle.settle == Decimal("5260.25")

    def test_picks_latest_date_below_as_of_filter(self, es_archive: Path) -> None:
        """as_of = May 20 → picks May 20, not May 21 (PIT-honest)."""
        src = ReplayDataSource(archive_root=es_archive)
        settle = src.latest_settle(
            ContractSymbol("ESM26"),
            datetime(2026, 5, 20, 22, 0, 0, tzinfo=UTC),
        )
        assert settle is not None
        assert settle.as_of_date == date(2026, 5, 20)
        assert settle.settle == Decimal("5252.00")

    def test_final_wins_over_preliminary_on_same_date(self, es_archive: Path) -> None:
        """May 21 has both prelim (5260.00) and final (5260.25) → final picked."""
        src = ReplayDataSource(archive_root=es_archive)
        settle = src.latest_settle(
            ContractSymbol("ESM26"),
            datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        )
        assert settle is not None
        assert settle.settle_state == "final"

    def test_returns_none_when_archive_missing(self, tmp_path: Path) -> None:
        """No archive on disk → None (not an exception)."""
        src = ReplayDataSource(archive_root=tmp_path / "missing")
        assert (
            src.latest_settle(
                ContractSymbol("ESM26"),
                datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
            )
            is None
        )

    def test_returns_none_when_contract_not_archived(self, empty_tmp_archive: Path) -> None:
        """Archive exists but contract dir doesn't → None."""
        src = ReplayDataSource(archive_root=empty_tmp_archive)
        assert (
            src.latest_settle(
                ContractSymbol("ESM26"),
                datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
            )
            is None
        )

    def test_returns_none_when_all_rows_after_as_of(self, es_archive: Path) -> None:
        """as_of before all archived dates → None (PIT-honest)."""
        src = ReplayDataSource(archive_root=es_archive)
        assert (
            src.latest_settle(
                ContractSymbol("ESM26"),
                datetime(2026, 5, 1, 0, 0, 0, tzinfo=UTC),
            )
            is None
        )

    def test_pit_honesty_future_row_not_emitted(self, tmp_path: Path) -> None:
        """A 2027 row in archive must NOT be emitted when as_of is in 2026."""
        archive_root = tmp_path / "cme_eod_archive"
        settles = [
            _make_settle(
                as_of_date=date(2027, 1, 5),
                settle="5500.00",
                settle_state="final",
                content_bytes_sha="aa" * 32,
            ),
        ]
        _seed_archive(archive_root, "ES", settles)
        src = ReplayDataSource(archive_root=archive_root)
        assert (
            src.latest_settle(
                ContractSymbol("ESM26"),
                datetime(2026, 12, 31, 23, 59, 59, tzinfo=UTC),
            )
            is None
        )

    def test_picks_latest_across_year_partitions(self, tmp_path: Path) -> None:
        """Archive has 2025 + 2026 rows → picks the latest 2026 within as_of filter."""
        archive_root = tmp_path / "cme_eod_archive"
        settles = [
            _make_settle(
                as_of_date=date(2025, 12, 31),
                settle="5000.00",
                settle_state="final",
                content_bytes_sha="bb" * 32,
            ),
            _make_settle(
                as_of_date=date(2026, 1, 2),
                settle="5050.00",
                settle_state="final",
                content_bytes_sha="cc" * 32,
            ),
        ]
        _seed_archive(archive_root, "ES", settles)
        src = ReplayDataSource(archive_root=archive_root)
        settle = src.latest_settle(
            ContractSymbol("ESM26"),
            datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        )
        assert settle is not None
        assert settle.as_of_date == date(2026, 1, 2)
        assert settle.settle == Decimal("5050.00")

    def test_naive_as_of_accepted(self, es_archive: Path) -> None:
        """ABC convention (matches CMEEODDataSource): naive as_of accepted via .date()."""
        src = ReplayDataSource(archive_root=es_archive)
        settle = src.latest_settle(
            ContractSymbol("ESM26"),
            datetime(2026, 5, 21, 22, 0, 0),  # naive
        )
        assert settle is not None
        assert settle.as_of_date == date(2026, 5, 21)


# ============================================================================
# TestA1_7_FieldRoundTrip — exact preservation of all Settle fields
# ============================================================================


class TestA1_7_FieldRoundTrip:
    def test_decimal_precision_exact_round_trip(self, tmp_path: Path) -> None:
        """Decimal('5260.12345') survives Parquet round-trip exactly."""
        archive_root = tmp_path / "cme_eod_archive"
        settles = [
            _make_settle(
                as_of_date=date(2026, 5, 21),
                settle="5260.12345",
                settle_state="final",
                open_="5252.987654321",
                high="5263.50",
                low="5248.75",
                last="5260.0",
                change="7.99999",
                content_bytes_sha="dd" * 32,
            ),
        ]
        _seed_archive(archive_root, "ES", settles)
        src = ReplayDataSource(archive_root=archive_root)
        settle = src.latest_settle(
            ContractSymbol("ESM26"),
            datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        )
        assert settle is not None
        assert settle.settle == Decimal("5260.12345")
        assert settle.open == Decimal("5252.987654321")
        assert settle.change == Decimal("7.99999")

    def test_source_id_preserved_in_record(self, tmp_path: Path) -> None:
        """Original capturer's source_id (e.g., 'cme_public_settlements') preserved."""
        archive_root = tmp_path / "cme_eod_archive"
        settles = [
            _make_settle(
                as_of_date=date(2026, 5, 21),
                settle="5260.00",
                source_id="cme_public_settlements",
                content_bytes_sha="ee" * 32,
            ),
        ]
        _seed_archive(archive_root, "ES", settles)
        # ReplayDataSource's container source_id differs from emitted record's source_id
        src = ReplayDataSource(archive_root=archive_root, source_id="replay@cme")
        settle = src.latest_settle(
            ContractSymbol("ESM26"),
            datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        )
        assert settle is not None
        # Container ID
        assert src.source_id == "replay@cme"
        # Original source_id preserved in record (backtest-is-live: verifier cross-source consensus works)
        assert settle.source_id == "cme_public_settlements"

    def test_content_bytes_sha_preserved(self, tmp_path: Path) -> None:
        """Provenance hash round-trips byte-exact (bit-repro anchor)."""
        archive_root = tmp_path / "cme_eod_archive"
        sha = "9f" * 32
        settles = [
            _make_settle(
                as_of_date=date(2026, 5, 21),
                settle="5260.00",
                content_bytes_sha=sha,
            ),
        ]
        _seed_archive(archive_root, "ES", settles)
        src = ReplayDataSource(archive_root=archive_root)
        settle = src.latest_settle(
            ContractSymbol("ESM26"),
            datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        )
        assert settle is not None
        assert settle.content_bytes_sha == sha
        assert len(settle.content_bytes_sha) == 64

    def test_as_of_iso_tz_preserved_as_utc(self, tmp_path: Path) -> None:
        """as_of_iso stored TZ-aware → read back TZ-aware UTC."""
        archive_root = tmp_path / "cme_eod_archive"
        capture_time = datetime(2026, 5, 22, 9, 30, 0, tzinfo=UTC)
        settles = [
            _make_settle(
                as_of_date=date(2026, 5, 21),
                settle="5260.00",
                as_of_iso=capture_time,
                content_bytes_sha="ff" * 32,
            ),
        ]
        _seed_archive(archive_root, "ES", settles)
        src = ReplayDataSource(archive_root=archive_root)
        settle = src.latest_settle(
            ContractSymbol("ESM26"),
            datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        )
        assert settle is not None
        assert settle.as_of_iso is not None
        assert settle.as_of_iso.tzinfo is not None
        assert settle.as_of_iso == capture_time

    def test_oi_prior_and_volume_est_round_trip(self, tmp_path: Path) -> None:
        """Int fields preserved exactly."""
        archive_root = tmp_path / "cme_eod_archive"
        settles = [
            _make_settle(
                as_of_date=date(2026, 5, 21),
                settle="5260.00",
                volume_est=1_280_000,
                oi_prior=1_750_000,
                content_bytes_sha="ab" * 32,
            ),
        ]
        _seed_archive(archive_root, "ES", settles)
        src = ReplayDataSource(archive_root=archive_root)
        settle = src.latest_settle(
            ContractSymbol("ESM26"),
            datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        )
        assert settle is not None
        assert settle.volume_est == 1_280_000
        assert settle.oi_prior == 1_750_000

    def test_cme_month_code_preserved(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "cme_eod_archive"
        settles = [
            _make_settle(
                as_of_date=date(2026, 5, 21),
                settle="5260.00",
                cme_month_code="M",
                content_bytes_sha="cd" * 32,
            ),
        ]
        _seed_archive(archive_root, "ES", settles)
        src = ReplayDataSource(archive_root=archive_root)
        settle = src.latest_settle(
            ContractSymbol("ESM26"),
            datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        )
        assert settle is not None
        assert settle.cme_month_code == "M"


# ============================================================================
# TestA1_7_Determinism — same archive + same query → same Settle
# ============================================================================


class TestA1_7_Determinism:
    def test_repeated_calls_return_identical_settle(self, es_archive: Path) -> None:
        """Two replays of the same archive + query produce byte-equal Settles."""
        src1 = ReplayDataSource(archive_root=es_archive)
        src2 = ReplayDataSource(archive_root=es_archive)
        as_of = datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC)
        s1 = src1.latest_settle(ContractSymbol("ESM26"), as_of)
        s2 = src2.latest_settle(ContractSymbol("ESM26"), as_of)
        assert s1 is not None and s2 is not None
        assert s1 == s2
        assert s1.content_bytes_sha == s2.content_bytes_sha

    def test_tie_break_by_content_sha_when_state_and_date_equal(self, tmp_path: Path) -> None:
        """Two final settles same date → tie-break by content_bytes_sha lexicographic."""
        archive_root = tmp_path / "cme_eod_archive"
        # Both rows: same as_of_date, same settle_state=final, different content_sha
        settles = [
            _make_settle(
                as_of_date=date(2026, 5, 21),
                settle="5260.00",
                settle_state="final",
                content_bytes_sha="11" * 32,
            ),
            _make_settle(
                as_of_date=date(2026, 5, 21),
                settle="5260.50",
                settle_state="final",
                content_bytes_sha="ff" * 32,
            ),
        ]
        _seed_archive(archive_root, "ES", settles)
        src = ReplayDataSource(archive_root=archive_root)
        s = src.latest_settle(
            ContractSymbol("ESM26"),
            datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        )
        assert s is not None
        # "ff" > "11" lexicographic → final from the ff row wins
        assert s.settle == Decimal("5260.50")
        assert s.content_bytes_sha == "ff" * 32


# ============================================================================
# TestA1_7_NotSupported — get_bars + get_ticks ABC contract
# ============================================================================


class TestA1_7_NotSupported:
    def test_get_bars_raises_bars_not_supported(self, es_archive: Path) -> None:
        src = ReplayDataSource(archive_root=es_archive)
        with pytest.raises(BarsNotSupported, match=r"A1\.16"):
            list(
                src.get_bars(
                    ContractSymbol("ESM26"),
                    datetime(2026, 5, 19, 0, 0, 0, tzinfo=UTC),
                    datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
                    BarResolution.DAY_1,
                )
            )

    def test_get_ticks_raises_default_not_supported(self, es_archive: Path) -> None:
        src = ReplayDataSource(archive_root=es_archive)
        as_of = datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC)
        with pytest.raises(TicksNotSupported):
            list(src.get_ticks(ContractSymbol("ESM26"), as_of, as_of))


# ============================================================================
# TestA1_7_Errors — defensive validation
# ============================================================================


class TestA1_7_Errors:
    def test_too_short_contract_raises_not_configured(self, es_archive: Path) -> None:
        src = ReplayDataSource(archive_root=es_archive)
        with pytest.raises(ContractNotConfigured, match="too short"):
            src.latest_settle(
                ContractSymbol("ES"),
                datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
            )

    def test_unknown_settle_state_in_archive_raises_data_source_error(self, tmp_path: Path) -> None:
        """A row with an out-of-spec settle_state surfaces as DataSourceError."""
        archive_root = tmp_path / "cme_eod_archive"
        contract_dir = archive_root / "contract=ES" / "year=2026"
        contract_dir.mkdir(parents=True)
        df = pl.DataFrame(
            [
                {
                    "contract": "ESM26",
                    "as_of_date": date(2026, 5, 21),
                    "settle": "5260.00",
                    "settle_state": "garbage",  # invalid
                    "open": "5252.25",
                    "high": "5263.50",
                    "low": "5248.75",
                    "last": "5260.00",
                    "change": "0.00",
                    "volume_est": 1_000_000,
                    "oi_prior": 1_500_000,
                    "source_id": "cme_public_settlements",
                    "as_of_iso": datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
                    "content_bytes_sha": "ab" * 32,
                    "cme_month_code": "M",
                }
            ]
        )
        df.write_parquet(contract_dir / "data.parquet")
        src = ReplayDataSource(archive_root=archive_root)
        with pytest.raises(DataSourceError, match="unknown settle_state"):
            src.latest_settle(
                ContractSymbol("ESM26"),
                datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
            )

    def test_malformed_year_dir_skipped(self, tmp_path: Path) -> None:
        """Non-`year=NNNN` sub-dirs ignored, not crashed."""
        archive_root = tmp_path / "cme_eod_archive"
        (archive_root / "contract=ES" / "year=garbage").mkdir(parents=True)
        # Also seed a valid year so we have a positive control
        settles = [
            _make_settle(
                as_of_date=date(2026, 5, 21),
                settle="5260.00",
                settle_state="final",
                content_bytes_sha="ef" * 32,
            ),
        ]
        _seed_archive(archive_root, "ES", settles)
        src = ReplayDataSource(archive_root=archive_root)
        settle = src.latest_settle(
            ContractSymbol("ESM26"),
            datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        )
        assert settle is not None
        assert settle.settle == Decimal("5260.00")

    def test_m2_naive_as_of_iso_in_archive_raises_data_source_error(self, tmp_path: Path) -> None:
        """Naive as_of_iso in archive row → DataSourceError.

        Prior behavior silently coerced naive datetime → UTC, masking write-side
        bugs where TZ got stripped during Parquet write. Fail-loud: refuse silent
        coercion; force write-side to populate TZ.
        """
        archive_root = tmp_path / "cme_eod_archive"
        contract_dir = archive_root / "contract=ES" / "year=2026"
        contract_dir.mkdir(parents=True)
        df = pl.DataFrame(
            [
                {
                    "contract": "ESM26",
                    "as_of_date": date(2026, 5, 21),
                    "settle": "5260.00",
                    "settle_state": "final",
                    "open": "5252.25",
                    "high": "5263.50",
                    "low": "5248.75",
                    "last": "5260.00",
                    "change": "0.00",
                    "volume_est": 1_000_000,
                    "oi_prior": 1_500_000,
                    "source_id": "cme_public_settlements",
                    # Intentionally naive (no tzinfo) — the M2 fix MUST refuse this
                    "as_of_iso": datetime(2026, 5, 21, 22, 0, 0),
                    "content_bytes_sha": "ab" * 32,
                    "cme_month_code": "M",
                }
            ]
        )
        df.write_parquet(contract_dir / "data.parquet")
        src = ReplayDataSource(archive_root=archive_root)
        with pytest.raises(DataSourceError, match="as_of_iso is naive"):
            src.latest_settle(
                ContractSymbol("ESM26"),
                datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
            )
