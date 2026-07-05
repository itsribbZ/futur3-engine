"""futur3.storage.duckdb_query - DuckDbQueryEngine (A1.16c SQL-over-Parquet).

Additive analyst/operator query layer over the hive-partitioned Parquet archive that
ParquetBarStore (A1.16b) + ParquetSettleStore (A1.16c) write. This is NOT a BarStore/SettleStore
implementation: the ABC windowed-read contract is fully served by the polars stores
(internal design notes"DuckDB is additive for scale").
This engine exists for ad-hoc aggregation / cross-contract scans that do not fit the
get_bars / read_settles window contract.

Read-only by construction: the connection is an in-memory DuckDB (`:memory:`) that scans the
on-disk Parquet via `read_parquet()`. The archive is NEVER mutated - even a stray CREATE/INSERT in
operator SQL lands only in the ephemeral in-memory catalog, so it cannot corrupt the source of
truth on disk.

Precision note: Decimal price columns are persisted as strings (lossless, never float). In SQL,
CAST them for arithmetic, e.g. `AVG(CAST(close AS DOUBLE))`. The raw string survives untouched for
exact-precision needs. With hive partitioning (default), the partition columns (`year`, and `month`
for bars) are surfaced from the path for pruned filters.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Final

import duckdb
import polars as pl

from futur3.storage.abcs import StorageError


def _files_sql(files: list[Path]) -> str:
    """Render a list of Parquet paths as a DuckDB list literal.

    Forward-slash (`as_posix`) is DuckDB-portable on Windows; single quotes are SQL-escaped by
    doubling. Inputs are filesystem paths this process enumerated (not user input), but the escape
    keeps the literal well-formed regardless.
    """
    inner = ", ".join("'" + f.as_posix().replace("'", "''") + "'" for f in files)
    return f"[{inner}]"


class DuckDbQueryEngine:
    """SQL-over-Parquet ad-hoc query engine (A1.16c). Read-only by construction (in-memory
    connection scanning the on-disk archive); additive to the polars-backed ABC read path."""

    BACKEND_ID: Final[str] = "duckdb_query"

    def __init__(self, base_path: Path) -> None:
        self._base = Path(base_path)
        self._conn = duckdb.connect(":memory:")

    @property
    def backend_id(self) -> str:
        return self.BACKEND_ID

    def healthcheck(self) -> bool:
        try:
            self._conn.execute("SELECT 1").fetchone()
        except duckdb.Error:
            return False
        return True

    def bar_files(self) -> list[Path]:
        """Sorted Parquet files under the bar archive (empty if none written yet)."""
        return sorted((self._base / "futures" / "bars").glob("**/data.parquet"))

    def settle_files(self) -> list[Path]:
        """Sorted Parquet files under the settle archive (empty if none written yet)."""
        return sorted((self._base / "futures" / "settles").glob("**/data.parquet"))

    def query(self, sql: str, params: list[object] | None = None) -> pl.DataFrame:
        """Execute arbitrary read SQL, returning a polars DataFrame (operator escape hatch).

        `params` binds `?` placeholders. DuckDB errors are re-raised as StorageError so callers
        catch one storage-layer exception type at the engine boundary (no-silent-fallback).
        """
        try:
            result: pl.DataFrame = self._conn.execute(
                sql, params if params is not None else []
            ).pl()
        except duckdb.Error as exc:
            raise StorageError(f"DuckDbQueryEngine query failed: {exc}") from exc
        return result

    def query_bars(
        self,
        *,
        select: str = "*",
        where: str | None = None,
        params: list[object] | None = None,
    ) -> pl.DataFrame:
        """Query the bar archive without hardcoding the glob. Empty archive -> empty DataFrame."""
        return self._query_archive(self.bar_files(), select=select, where=where, params=params)

    def query_settles(
        self,
        *,
        select: str = "*",
        where: str | None = None,
        params: list[object] | None = None,
    ) -> pl.DataFrame:
        """Query the settle archive without hardcoding the glob. Empty -> empty DataFrame."""
        return self._query_archive(self.settle_files(), select=select, where=where, params=params)

    def _query_archive(
        self,
        files: list[Path],
        *,
        select: str,
        where: str | None,
        params: list[object] | None,
    ) -> pl.DataFrame:
        # No files yet: return an empty frame rather than letting read_parquet raise on an empty
        # glob (mirrors ParquetBarStore/ParquetSettleStore "missing partition -> empty" contract).
        if not files:
            return pl.DataFrame()
        sql = f"SELECT {select} FROM read_parquet({_files_sql(files)}, hive_partitioning = true)"
        if where:
            sql += f" WHERE {where}"
        return self.query(sql, params=params)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> DuckDbQueryEngine:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()
