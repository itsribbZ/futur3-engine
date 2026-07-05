"""A1.16c DuckDbQueryEngine test suite (SQL-over-Parquet, real archives in tmp_path).

Per internal design notes(DuckDB additive query layer):
- Cross-component integration: query EXACTLY what ParquetBarStore (A1.16b) + ParquetSettleStore
  (A1.16c) wrote (validates glob/layout alignment - the load-bearing seam).
- Decimal-as-string CAST aggregation; hive partition columns (year/month) surfaced from path.
- Empty archive -> empty DataFrame (no read_parquet "no files" crash).
- Read-only by construction: operator SQL (incl. CREATE/INSERT) never mutates the on-disk archive.
- where + `?` params; raw `query()` escape hatch; context manager; bad SQL -> StorageError.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from futur3.data.types import (
    BarResolution,
    ContractSymbol,
    RawBar,
    Settle,
    content_sha256,
)
from futur3.storage import DuckDbQueryEngine as _PkgDuckDbQueryEngine
from futur3.storage.abcs import StorageError
from futur3.storage.duckdb_query import DuckDbQueryEngine
from futur3.storage.parquet_settle_store import ParquetSettleStore
from futur3.storage.parquet_store import ParquetBarStore


def _bar(*, contract: str = "ESM26", ts: datetime, close: str, sha: bytes) -> RawBar:
    return RawBar(
        contract=ContractSymbol(contract),
        ts=ts,
        resolution=BarResolution.MIN_5,
        open=Decimal("5260.00"),
        high=Decimal("5261.50"),
        low=Decimal("5259.75"),
        close=Decimal(close),
        volume=100,
        oi=12345,
        source_id="test",
        as_of_iso=ts,
        content_bytes_sha=content_sha256(sha),
    )


def _settle(*, contract: str = "ESM26", d: date, settle: str, sha: bytes) -> Settle:
    return Settle(
        contract=ContractSymbol(contract),
        as_of_date=d,
        settle=Decimal(settle),
        settle_state="final",
        open=Decimal("5248.25"),
        high=Decimal("5263.00"),
        low=Decimal("5247.75"),
        last=Decimal("5260.25"),
        change=Decimal("12.25"),
        volume_est=1_350_000,
        oi_prior=2_100_000,
        as_of_iso=datetime(2026, 5, 21, 23, 0, tzinfo=UTC),
        content_bytes_sha=content_sha256(sha),
        cme_month_code="M",
    )


def _seed_bars(base: Path) -> ParquetBarStore:
    store = ParquetBarStore(base)
    store.write_bars(
        [
            _bar(ts=datetime(2026, 5, 21, 14, 0, tzinfo=UTC), close="5260.00", sha=b"a"),
            _bar(ts=datetime(2026, 5, 21, 14, 5, tzinfo=UTC), close="5262.50", sha=b"b"),
            _bar(
                contract="NQM26",
                ts=datetime(2026, 5, 21, 14, 0, tzinfo=UTC),
                close="18000.00",
                sha=b"c",
            ),
        ]
    )
    return store


def _seed_settles(base: Path) -> ParquetSettleStore:
    store = ParquetSettleStore(base)
    store.write_settles(
        [
            _settle(d=date(2026, 5, 20), settle="5248.25", sha=b"s1"),
            _settle(d=date(2026, 5, 21), settle="5260.50", sha=b"s2"),
        ]
    )
    return store


# ============================================================================
# TestA1_16c_Duck_Imports / Construction
# ============================================================================


class TestA1_16c_Duck_Imports:
    def test_importable(self) -> None:
        assert DuckDbQueryEngine is not None

    def test_exported_from_storage_package(self) -> None:
        assert _PkgDuckDbQueryEngine is DuckDbQueryEngine

    def test_backend_id(self, tmp_path: Path) -> None:
        with DuckDbQueryEngine(tmp_path) as eng:
            assert eng.backend_id == "duckdb_query"

    def test_healthcheck(self, tmp_path: Path) -> None:
        with DuckDbQueryEngine(tmp_path) as eng:
            assert eng.healthcheck() is True


# ============================================================================
# TestA1_16c_Duck_EmptyArchive
# ============================================================================


class TestA1_16c_Duck_EmptyArchive:
    def test_query_bars_empty(self, tmp_path: Path) -> None:
        with DuckDbQueryEngine(tmp_path) as eng:
            assert eng.query_bars().is_empty()

    def test_query_settles_empty(self, tmp_path: Path) -> None:
        with DuckDbQueryEngine(tmp_path) as eng:
            assert eng.query_settles().is_empty()

    def test_file_enumerators_empty(self, tmp_path: Path) -> None:
        with DuckDbQueryEngine(tmp_path) as eng:
            assert eng.bar_files() == []
            assert eng.settle_files() == []


# ============================================================================
# TestA1_16c_Duck_BarsIntegration  (the load-bearing glob/layout seam)
# ============================================================================


class TestA1_16c_Duck_BarsIntegration:
    def test_reads_what_bar_store_wrote(self, tmp_path: Path) -> None:
        _seed_bars(tmp_path)
        with DuckDbQueryEngine(tmp_path) as eng:
            assert len(eng.bar_files()) >= 1
            df = eng.query_bars()
            assert df.height == 3  # 2 ES + 1 NQ

    def test_count_via_select(self, tmp_path: Path) -> None:
        _seed_bars(tmp_path)
        with DuckDbQueryEngine(tmp_path) as eng:
            df = eng.query_bars(select="COUNT(*) AS n")
            assert df["n"][0] == 3

    def test_cast_aggregation_on_string_decimal(self, tmp_path: Path) -> None:
        _seed_bars(tmp_path)
        with DuckDbQueryEngine(tmp_path) as eng:
            df = eng.query_bars(
                select="MAX(CAST(close AS DECIMAL(38,12))) AS mx",
                where="contract = 'ESM26'",
            )
            assert df["mx"][0] == Decimal("5262.500000000000")

    def test_hive_year_month_columns_present(self, tmp_path: Path) -> None:
        _seed_bars(tmp_path)
        with DuckDbQueryEngine(tmp_path) as eng:
            df = eng.query_bars(select="DISTINCT year, month")
            assert df["year"][0] == 2026  # parses as int
            assert df["month"][0] == "05"  # zero-padded -> DuckDB hive-types it as VARCHAR

    def test_where_with_params(self, tmp_path: Path) -> None:
        _seed_bars(tmp_path)
        with DuckDbQueryEngine(tmp_path) as eng:
            df = eng.query_bars(where="contract = ?", params=["NQM26"])
            assert df.height == 1
            assert df["contract"][0] == "NQM26"


# ============================================================================
# TestA1_16c_Duck_SettlesIntegration
# ============================================================================


class TestA1_16c_Duck_SettlesIntegration:
    def test_reads_what_settle_store_wrote(self, tmp_path: Path) -> None:
        _seed_settles(tmp_path)
        with DuckDbQueryEngine(tmp_path) as eng:
            assert len(eng.settle_files()) >= 1
            df = eng.query_settles()
            assert df.height == 2

    def test_settle_cast_aggregation(self, tmp_path: Path) -> None:
        _seed_settles(tmp_path)
        with DuckDbQueryEngine(tmp_path) as eng:
            df = eng.query_settles(select="MIN(CAST(settle AS DECIMAL(38,12))) AS lo")
            assert df["lo"][0] == Decimal("5248.250000000000")


# ============================================================================
# TestA1_16c_Duck_RawQuery / ReadOnly / Errors
# ============================================================================


class TestA1_16c_Duck_RawQuery:
    def test_raw_query_escape_hatch(self, tmp_path: Path) -> None:
        with DuckDbQueryEngine(tmp_path) as eng:
            df = eng.query("SELECT 1 + 1 AS two")
            assert df["two"][0] == 2

    def test_bad_sql_raises_storage_error(self, tmp_path: Path) -> None:
        with DuckDbQueryEngine(tmp_path) as eng, pytest.raises(StorageError, match="query failed"):
            eng.query("SELECT * FROM nonexistent_table_xyz")

    def test_read_only_archive_not_mutated(self, tmp_path: Path) -> None:
        # An operator CREATE/INSERT lands only in the ephemeral in-memory catalog; the on-disk
        # Parquet archive must be byte-identical before and after.
        _seed_bars(tmp_path)
        files = sorted((tmp_path / "futures" / "bars").glob("**/data.parquet"))
        before = {f: f.read_bytes() for f in files}
        with DuckDbQueryEngine(tmp_path) as eng:
            eng.query("CREATE TABLE scratch AS SELECT 42 AS x")
            eng.query("INSERT INTO scratch VALUES (43)")
            assert eng.query("SELECT COUNT(*) AS n FROM scratch")["n"][0] == 2
        after = {f: f.read_bytes() for f in files}
        assert before == after  # archive untouched

    def test_context_manager_closes(self, tmp_path: Path) -> None:
        eng = DuckDbQueryEngine(tmp_path)
        with eng:
            assert eng.healthcheck() is True
        # after close, the connection is no longer usable -> healthcheck reports False
        assert eng.healthcheck() is False
