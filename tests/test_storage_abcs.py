"""A1.16a BarStore + SettleStore ABCs test suite.

Test discipline:
- Fixture-only (no Parquet, no DuckDB — concrete impls in A1.16b).
- ABCs enforce abstractness (can't instantiate directly; partial subclasses
  fail; full concrete subclass works).
- Dataclass validation (frozen, bounds checks, TZ-aware enforcement).
- Exception hierarchy correct.

References:
- `futur3/storage/abcs.py` (implementation)
- internal design notes (spec)
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from futur3.data.types import (
    BarResolution,
    ContractSymbol,
    RawBar,
    Settle,
)
from futur3.storage.abcs import (
    BarStore,
    IdempotentDuplicateError,
    PartitionNotFound,
    SettleStore,
    StorageError,
    StorageNotInitialized,
    StoreQuery,
    StoreWriteResult,
)

# ============================================================================
# Helpers
# ============================================================================


def _sha(seed: str) -> str:
    return (seed.encode().hex() * 16)[:64]


# ============================================================================
# TestA1_16a_Imports
# ============================================================================


class TestA1_16a_Imports:
    def test_bar_store_importable(self) -> None:
        assert BarStore is not None

    def test_settle_store_importable(self) -> None:
        assert SettleStore is not None

    def test_dataclasses_importable(self) -> None:
        assert StoreQuery is not None
        assert StoreWriteResult is not None

    def test_exceptions_importable(self) -> None:
        assert StorageError is not None
        assert StorageNotInitialized is not None
        assert IdempotentDuplicateError is not None
        assert PartitionNotFound is not None


# ============================================================================
# TestA1_16a_ExceptionHierarchy
# ============================================================================


class TestA1_16a_ExceptionHierarchy:
    def test_storage_error_is_exception(self) -> None:
        assert issubclass(StorageError, Exception)

    def test_storage_not_initialized_extends_storage_error(self) -> None:
        assert issubclass(StorageNotInitialized, StorageError)

    def test_idempotent_duplicate_extends_storage_error(self) -> None:
        assert issubclass(IdempotentDuplicateError, StorageError)

    def test_partition_not_found_extends_storage_error(self) -> None:
        assert issubclass(PartitionNotFound, StorageError)


# ============================================================================
# TestA1_16a_StoreWriteResult
# ============================================================================


class TestA1_16a_StoreWriteResult:
    def test_valid_construction(self) -> None:
        r = StoreWriteResult(
            rows_written=10,
            dedupe_count=2,
            path=Path("/data/x.parquet"),
        )
        assert r.rows_written == 10
        assert r.dedupe_count == 2
        assert r.total_submitted == 12

    def test_zero_result_allowed(self) -> None:
        """Empty write (no-op) returns 0/0 result successfully."""
        r = StoreWriteResult(
            rows_written=0,
            dedupe_count=0,
            path=Path("/data/x.parquet"),
        )
        assert r.total_submitted == 0

    def test_negative_rows_written_raises(self) -> None:
        with pytest.raises(ValueError, match="rows_written must be >= 0"):
            StoreWriteResult(
                rows_written=-1,
                dedupe_count=0,
                path=Path("/data/x.parquet"),
            )

    def test_negative_dedupe_count_raises(self) -> None:
        with pytest.raises(ValueError, match="dedupe_count must be >= 0"):
            StoreWriteResult(
                rows_written=10,
                dedupe_count=-1,
                path=Path("/data/x.parquet"),
            )

    def test_frozen_immutable(self) -> None:
        r = StoreWriteResult(rows_written=1, dedupe_count=0, path=Path("/x"))
        with pytest.raises(AttributeError):
            r.rows_written = 2  # type: ignore[misc]


# ============================================================================
# TestA1_16a_StoreQuery
# ============================================================================


class TestA1_16a_StoreQuery:
    def _valid_query(self) -> StoreQuery:
        return StoreQuery(
            contract=ContractSymbol("ESM26"),
            ts_start=datetime(2026, 5, 21, 14, 0, 0, tzinfo=UTC),
            ts_end=datetime(2026, 5, 21, 15, 0, 0, tzinfo=UTC),
            resolution=BarResolution.MIN_5,
        )

    def test_valid_query(self) -> None:
        q = self._valid_query()
        assert q.inclusive_end is False  # default half-open

    def test_naive_ts_start_raises(self) -> None:
        with pytest.raises(ValueError, match="ts_start must be TZ-aware"):
            StoreQuery(
                contract=ContractSymbol("ESM26"),
                ts_start=datetime(2026, 5, 21, 14, 0, 0),  # naive
                ts_end=datetime(2026, 5, 21, 15, 0, 0, tzinfo=UTC),
            )

    def test_naive_ts_end_raises(self) -> None:
        with pytest.raises(ValueError, match="ts_end must be TZ-aware"):
            StoreQuery(
                contract=ContractSymbol("ESM26"),
                ts_start=datetime(2026, 5, 21, 14, 0, 0, tzinfo=UTC),
                ts_end=datetime(2026, 5, 21, 15, 0, 0),  # naive
            )

    def test_ts_end_before_start_raises(self) -> None:
        with pytest.raises(ValueError, match="ts_end must be > ts_start"):
            StoreQuery(
                contract=ContractSymbol("ESM26"),
                ts_start=datetime(2026, 5, 21, 15, 0, 0, tzinfo=UTC),
                ts_end=datetime(2026, 5, 21, 14, 0, 0, tzinfo=UTC),
            )

    def test_ts_end_equals_start_raises(self) -> None:
        ts = datetime(2026, 5, 21, 14, 0, 0, tzinfo=UTC)
        with pytest.raises(ValueError, match="ts_end must be > ts_start"):
            StoreQuery(
                contract=ContractSymbol("ESM26"),
                ts_start=ts,
                ts_end=ts,
            )

    def test_settle_query_no_resolution(self) -> None:
        """Settle queries omit resolution — None is allowed."""
        q = StoreQuery(
            contract=ContractSymbol("ESM26"),
            ts_start=datetime(2026, 5, 21, 14, 0, 0, tzinfo=UTC),
            ts_end=datetime(2026, 5, 21, 15, 0, 0, tzinfo=UTC),
        )
        assert q.resolution is None

    def test_frozen_immutable(self) -> None:
        q = self._valid_query()
        with pytest.raises(AttributeError):
            q.inclusive_end = True  # type: ignore[misc]


# ============================================================================
# TestA1_16a_BarStoreABC
# ============================================================================


class TestA1_16a_BarStoreABC:
    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            BarStore()  # type: ignore[abstract]

    def test_partial_subclass_still_abstract(self) -> None:
        class _Partial(BarStore):
            @property
            def backend_id(self) -> str:
                return "partial"

            # missing write_bars + read_bars + healthcheck

        with pytest.raises(TypeError, match="abstract"):
            _Partial()  # type: ignore[abstract]

    def test_full_concrete_subclass_works(self) -> None:
        class _Full(BarStore):
            @property
            def backend_id(self) -> str:
                return "memory_test"

            def write_bars(self, bars: list[RawBar]) -> StoreWriteResult:
                return StoreWriteResult(
                    rows_written=len(bars),
                    dedupe_count=0,
                    path=Path("/dev/null"),
                )

            def read_bars(self, query: StoreQuery) -> Iterable[RawBar]:
                return []

            def healthcheck(self) -> bool:
                return True

        store = _Full()
        assert store.backend_id == "memory_test"
        assert store.healthcheck() is True

        # Smoke: write empty list returns zero result
        result = store.write_bars([])
        assert result.rows_written == 0


# ============================================================================
# TestA1_16a_SettleStoreABC
# ============================================================================


class TestA1_16a_SettleStoreABC:
    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            SettleStore()  # type: ignore[abstract]

    def test_partial_subclass_still_abstract(self) -> None:
        class _Partial(SettleStore):
            @property
            def backend_id(self) -> str:
                return "partial"

            # missing write_settles + read_settles + healthcheck

        with pytest.raises(TypeError, match="abstract"):
            _Partial()  # type: ignore[abstract]

    def test_full_concrete_subclass_works(self) -> None:
        class _Full(SettleStore):
            @property
            def backend_id(self) -> str:
                return "memory_test"

            def write_settles(self, settles: list[Settle]) -> StoreWriteResult:
                return StoreWriteResult(
                    rows_written=len(settles),
                    dedupe_count=0,
                    path=Path("/dev/null"),
                )

            def read_settles(
                self,
                contract: ContractSymbol,
                as_of_date_start: date,
                as_of_date_end: date,
            ) -> Iterable[Settle]:
                return []

            def healthcheck(self) -> bool:
                return True

        store = _Full()
        assert store.backend_id == "memory_test"
        assert store.healthcheck() is True

        # Smoke: empty write succeeds + smoke read empty returns empty
        write_result = store.write_settles([])
        assert write_result.rows_written == 0
        read_result = list(
            store.read_settles(
                ContractSymbol("ESM26"),
                date(2026, 5, 1),
                date(2026, 5, 31),
            )
        )
        assert read_result == []


# ============================================================================
# TestA1_16a_RoundTripSmoke
# ============================================================================


class TestA1_16a_RoundTripSmoke:
    """In-memory concrete subclass round-trips RawBars through the ABC contract."""

    def test_bar_round_trip(self) -> None:
        stored: list[RawBar] = []

        class _MemBarStore(BarStore):
            @property
            def backend_id(self) -> str:
                return "mem"

            def write_bars(self, bars: list[RawBar]) -> StoreWriteResult:
                stored.extend(bars)
                return StoreWriteResult(
                    rows_written=len(bars),
                    dedupe_count=0,
                    path=Path("/mem"),
                )

            def read_bars(self, query: StoreQuery) -> Iterable[RawBar]:
                return [
                    b
                    for b in stored
                    if b.contract == query.contract and query.ts_start <= b.ts < query.ts_end
                ]

            def healthcheck(self) -> bool:
                return True

        store = _MemBarStore()

        bar = RawBar(
            contract=ContractSymbol("ESM26"),
            ts=datetime(2026, 5, 21, 14, 5, 0, tzinfo=UTC),
            resolution=BarResolution.MIN_5,
            open=Decimal("5260.00"),
            high=Decimal("5261.50"),
            low=Decimal("5259.75"),
            close=Decimal("5260.50"),
            volume=100,
            oi=None,
            source_id="mem",
            as_of_iso=datetime(2026, 5, 21, 14, 5, 0, tzinfo=UTC),
            content_bytes_sha=_sha("test"),
        )
        write = store.write_bars([bar])
        assert write.rows_written == 1

        read_bars = list(
            store.read_bars(
                StoreQuery(
                    contract=ContractSymbol("ESM26"),
                    ts_start=datetime(2026, 5, 21, 14, 0, 0, tzinfo=UTC),
                    ts_end=datetime(2026, 5, 21, 15, 0, 0, tzinfo=UTC),
                    resolution=BarResolution.MIN_5,
                )
            )
        )
        assert len(read_bars) == 1
        assert read_bars[0].close == Decimal("5260.50")
