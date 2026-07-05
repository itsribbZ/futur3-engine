"""A1.16b ParquetBarStore test suite (real Parquet round-trips in tmp_path).

Per the `BarStore` ABC contract:
- Round-trip exactness: Decimal (lossless via string), tz-aware datetimes, oi nullable.
- Idempotent dedupe on (contract, ts, resolution, content_bytes_sha); revisions (distinct sha)
  preserved; dedupe_count / rows_written accurate.
- Half-open vs inclusive window; partition fan-out across months/years; multi-contract filter.
- Deterministic ascending order; resolution required for bar reads; empty/missing -> [].
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from futur3.data.types import BarResolution, ContractSymbol, RawBar, content_sha256
from futur3.storage import ParquetBarStore as _PkgParquetBarStore
from futur3.storage.abcs import BarStore, StorageError, StoreQuery
from futur3.storage.parquet_store import ParquetBarStore


def _bar(
    *,
    contract: str = "ESM26",
    ts: datetime | None = None,
    close: str = "5260.50",
    sha: str | None = None,
    oi: int | None = None,
    volume: int = 100,
    resolution: BarResolution = BarResolution.MIN_5,
) -> RawBar:
    ts = ts or datetime(2026, 5, 21, 14, 5, tzinfo=UTC)
    return RawBar(
        contract=ContractSymbol(contract),
        ts=ts,
        resolution=resolution,
        open=Decimal("5260.00"),
        high=Decimal("5261.50"),
        low=Decimal("5259.75"),
        close=Decimal(close),
        volume=volume,
        oi=oi,
        source_id="test",
        as_of_iso=datetime(2026, 5, 21, 14, 5, tzinfo=UTC),
        content_bytes_sha=sha or content_sha256(f"{contract}|{ts}|{close}".encode()),
    )


def _query(contract: str = "ESM26", *, inclusive: bool = False) -> StoreQuery:
    return StoreQuery(
        contract=ContractSymbol(contract),
        ts_start=datetime(2026, 5, 21, 13, 0, tzinfo=UTC),
        ts_end=datetime(2026, 5, 21, 16, 0, tzinfo=UTC),
        resolution=BarResolution.MIN_5,
        inclusive_end=inclusive,
    )


# ============================================================================
# TestA1_16b_Imports / Construction
# ============================================================================


class TestA1_16b_Imports:
    def test_importable(self) -> None:
        assert ParquetBarStore is not None

    def test_exported_from_storage_package(self) -> None:
        assert _PkgParquetBarStore is ParquetBarStore

    def test_is_bar_store(self, tmp_path: Path) -> None:
        assert isinstance(ParquetBarStore(tmp_path), BarStore)


class TestA1_16b_Construction:
    def test_backend_id(self, tmp_path: Path) -> None:
        assert ParquetBarStore(tmp_path).backend_id == "parquet_local"

    def test_healthcheck_creates_dir(self, tmp_path: Path) -> None:
        base = tmp_path / "store"
        assert ParquetBarStore(base).healthcheck() is True
        assert base.is_dir()


# ============================================================================
# TestA1_16b_RoundTrip
# ============================================================================


class TestA1_16b_RoundTrip:
    def test_write_then_read_exact(self, tmp_path: Path) -> None:
        store = ParquetBarStore(tmp_path)
        bar = _bar(close="5260.50", oi=12345)
        result = store.write_bars([bar])
        assert result.rows_written == 1
        assert result.dedupe_count == 0

        out = list(store.read_bars(_query()))
        assert len(out) == 1
        got = out[0]
        assert got.contract == ContractSymbol("ESM26")
        assert got.close == Decimal("5260.50")
        assert got.open == Decimal("5260.00")
        assert got.volume == 100
        assert got.oi == 12345
        assert got.source_id == "test"
        assert got.content_bytes_sha == bar.content_bytes_sha

    def test_ts_round_trips_tz_aware(self, tmp_path: Path) -> None:
        store = ParquetBarStore(tmp_path)
        bar = _bar()
        store.write_bars([bar])
        got = next(iter(store.read_bars(_query())))
        assert got.ts.tzinfo is not None  # never naive (bug class 7)
        assert got.ts == bar.ts  # same instant
        assert got.as_of_iso.tzinfo is not None

    def test_oi_none_round_trips(self, tmp_path: Path) -> None:
        store = ParquetBarStore(tmp_path)
        store.write_bars([_bar(oi=None)])
        got = next(iter(store.read_bars(_query())))
        assert got.oi is None

    def test_high_precision_decimal_exact(self, tmp_path: Path) -> None:
        store = ParquetBarStore(tmp_path)
        store.write_bars([_bar(close="5260.123456789")])
        got = next(iter(store.read_bars(_query())))
        assert got.close == Decimal("5260.123456789")


# ============================================================================
# TestA1_16b_Dedupe / Revision
# ============================================================================


class TestA1_16b_Dedupe:
    def test_rewrite_same_is_idempotent(self, tmp_path: Path) -> None:
        store = ParquetBarStore(tmp_path)
        bar = _bar()
        first = store.write_bars([bar])
        assert first.rows_written == 1 and first.dedupe_count == 0
        second = store.write_bars([bar])
        assert second.rows_written == 0
        assert second.dedupe_count == 1
        assert len(list(store.read_bars(_query()))) == 1  # not duplicated

    def test_internal_batch_dedupe(self, tmp_path: Path) -> None:
        store = ParquetBarStore(tmp_path)
        bar = _bar()
        result = store.write_bars([bar, bar, bar])  # 3 identical
        assert result.rows_written == 1
        assert result.dedupe_count == 2

    def test_revision_distinct_sha_both_kept(self, tmp_path: Path) -> None:
        store = ParquetBarStore(tmp_path)
        ts = datetime(2026, 5, 21, 14, 5, tzinfo=UTC)
        v1 = _bar(ts=ts, close="5260.50", sha=content_sha256(b"v1"))
        v2 = _bar(ts=ts, close="5260.75", sha=content_sha256(b"v2"))  # revision
        store.write_bars([v1])
        store.write_bars([v2])
        out = list(store.read_bars(_query()))
        assert len(out) == 2  # same (contract, ts, res) but distinct sha -> both kept
        assert {b.close for b in out} == {Decimal("5260.50"), Decimal("5260.75")}


# ============================================================================
# TestA1_16b_Window
# ============================================================================


class TestA1_16b_Window:
    def _three_bars(self, store: ParquetBarStore) -> None:
        store.write_bars(
            [
                _bar(
                    ts=datetime(2026, 5, 21, 14, 0, tzinfo=UTC), close="1", sha=content_sha256(b"a")
                ),
                _bar(
                    ts=datetime(2026, 5, 21, 14, 5, tzinfo=UTC), close="2", sha=content_sha256(b"b")
                ),
                _bar(
                    ts=datetime(2026, 5, 21, 14, 10, tzinfo=UTC),
                    close="3",
                    sha=content_sha256(b"c"),
                ),
            ]
        )

    def test_half_open_excludes_end(self, tmp_path: Path) -> None:
        store = ParquetBarStore(tmp_path)
        self._three_bars(store)
        q = StoreQuery(
            contract=ContractSymbol("ESM26"),
            ts_start=datetime(2026, 5, 21, 14, 0, tzinfo=UTC),
            ts_end=datetime(2026, 5, 21, 14, 10, tzinfo=UTC),
            resolution=BarResolution.MIN_5,
        )
        out = list(store.read_bars(q))
        assert [b.close for b in out] == [Decimal("1"), Decimal("2")]  # 14:10 excluded

    def test_inclusive_end_includes_end(self, tmp_path: Path) -> None:
        store = ParquetBarStore(tmp_path)
        self._three_bars(store)
        q = StoreQuery(
            contract=ContractSymbol("ESM26"),
            ts_start=datetime(2026, 5, 21, 14, 0, tzinfo=UTC),
            ts_end=datetime(2026, 5, 21, 14, 10, tzinfo=UTC),
            resolution=BarResolution.MIN_5,
            inclusive_end=True,
        )
        out = list(store.read_bars(q))
        assert [b.close for b in out] == [Decimal("1"), Decimal("2"), Decimal("3")]

    def test_ascending_order(self, tmp_path: Path) -> None:
        store = ParquetBarStore(tmp_path)
        # write out of order
        store.write_bars(
            [
                _bar(
                    ts=datetime(2026, 5, 21, 14, 10, tzinfo=UTC),
                    close="3",
                    sha=content_sha256(b"c"),
                ),
                _bar(
                    ts=datetime(2026, 5, 21, 14, 0, tzinfo=UTC), close="1", sha=content_sha256(b"a")
                ),
                _bar(
                    ts=datetime(2026, 5, 21, 14, 5, tzinfo=UTC), close="2", sha=content_sha256(b"b")
                ),
            ]
        )
        out = list(store.read_bars(_query(inclusive=True)))
        assert [b.close for b in out] == [Decimal("1"), Decimal("2"), Decimal("3")]


# ============================================================================
# TestA1_16b_Partitioning / MultiContract
# ============================================================================


class TestA1_16b_Partitioning:
    def test_spans_months_and_years(self, tmp_path: Path) -> None:
        store = ParquetBarStore(tmp_path)
        store.write_bars(
            [
                _bar(
                    ts=datetime(2026, 5, 21, 14, 0, tzinfo=UTC), close="1", sha=content_sha256(b"a")
                ),
                _bar(
                    ts=datetime(2026, 6, 21, 14, 0, tzinfo=UTC), close="2", sha=content_sha256(b"b")
                ),
                _bar(
                    ts=datetime(2027, 1, 5, 14, 0, tzinfo=UTC), close="3", sha=content_sha256(b"c")
                ),
            ]
        )
        # distinct partition files exist
        base = tmp_path / "futures" / "bars" / "ES" / "5m"
        assert (base / "year=2026" / "month=05" / "data.parquet").exists()
        assert (base / "year=2026" / "month=06" / "data.parquet").exists()
        assert (base / "year=2027" / "month=01" / "data.parquet").exists()
        # query spanning all
        q = StoreQuery(
            contract=ContractSymbol("ESM26"),
            ts_start=datetime(2026, 1, 1, tzinfo=UTC),
            ts_end=datetime(2028, 1, 1, tzinfo=UTC),
            resolution=BarResolution.MIN_5,
        )
        assert [b.close for b in store.read_bars(q)] == [Decimal("1"), Decimal("2"), Decimal("3")]

    def test_multi_contract_same_partition_filtered(self, tmp_path: Path) -> None:
        store = ParquetBarStore(tmp_path)
        ts = datetime(2026, 5, 21, 14, 0, tzinfo=UTC)
        store.write_bars(
            [
                _bar(contract="ESM26", ts=ts, close="5260", sha=content_sha256(b"m")),
                _bar(contract="ESU26", ts=ts, close="5280", sha=content_sha256(b"u")),
            ]
        )
        out = list(store.read_bars(_query("ESM26")))
        assert len(out) == 1
        assert out[0].contract == ContractSymbol("ESM26")
        assert out[0].close == Decimal("5260")


# ============================================================================
# TestA1_16b_EdgeCases
# ============================================================================


class TestA1_16b_EdgeCases:
    def test_empty_write_zero_result(self, tmp_path: Path) -> None:
        result = ParquetBarStore(tmp_path).write_bars([])
        assert result.rows_written == 0
        assert result.dedupe_count == 0

    def test_read_missing_partition_empty(self, tmp_path: Path) -> None:
        assert list(ParquetBarStore(tmp_path).read_bars(_query("NQM26"))) == []

    def test_read_requires_resolution(self, tmp_path: Path) -> None:
        store = ParquetBarStore(tmp_path)
        q = StoreQuery(
            contract=ContractSymbol("ESM26"),
            ts_start=datetime(2026, 5, 21, 13, 0, tzinfo=UTC),
            ts_end=datetime(2026, 5, 21, 16, 0, tzinfo=UTC),
        )  # resolution defaults None
        with pytest.raises(ValueError, match=r"requires query\.resolution"):
            list(store.read_bars(q))

    def test_short_contract_root_raises(self, tmp_path: Path) -> None:
        store = ParquetBarStore(tmp_path)
        with pytest.raises(StorageError, match="too short to parse root"):
            store.write_bars([_bar(contract="ES")])


# ============================================================================
# TestA1_16b_Determinism
# ============================================================================


class TestA1_16b_Determinism:
    def test_reread_identical(self, tmp_path: Path) -> None:
        store = ParquetBarStore(tmp_path)
        self_bars = [
            _bar(ts=datetime(2026, 5, 21, 14, 5, tzinfo=UTC), close="2", sha=content_sha256(b"b")),
            _bar(ts=datetime(2026, 5, 21, 14, 0, tzinfo=UTC), close="1", sha=content_sha256(b"a")),
        ]
        store.write_bars(self_bars)
        first = [(b.ts, b.close) for b in store.read_bars(_query(inclusive=True))]
        second = [(b.ts, b.close) for b in store.read_bars(_query(inclusive=True))]
        assert first == second  # deterministic across reads
        assert first == [
            (datetime(2026, 5, 21, 14, 0, tzinfo=UTC), Decimal("1")),
            (datetime(2026, 5, 21, 14, 5, tzinfo=UTC), Decimal("2")),
        ]
