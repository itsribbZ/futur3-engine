"""futur3.storage.parquet_store - ParquetBarStore (A1.16b concrete BarStore).

Per internal design notes, implementing the A1.16a
`BarStore` ABC. Uses polars (the proven pattern from `cme_eod._write_archive`): Decimal stored
as string (lossless precision), tz-aware datetimes coerced to UTC, idempotent dedupe via
`unique`/anti-join, deterministic sort.

Partition layout (hive-style, DuckDB-glob-readable per spec section 1.4):
    <base>/futures/bars/<root>/<resolution>/year=<YYYY>/month=<MM>/data.parquet

Append-only + idempotent dedupe on (contract, ts, resolution, content_bytes_sha): a repeated
fetch (same sha) is skipped; a revision (different sha) is a distinct key and BOTH are kept
(preserves revision history for the verifier). Rows are sorted (ts asc, content_bytes_sha) before
write so re-derivation from identical inputs yields identical logical content + order.

A1.16c will add ParquetSettleStore (same pattern, daily partitioning) + the DuckDbQueryEngine
(SQL-over-Parquet for ad-hoc/operator queries). The ABC read path here uses polars filtering,
which fully satisfies the windowed-query contract without DuckDB.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Final, Literal

import polars as pl

from futur3.data.types import BarResolution, ContractSymbol, RawBar
from futur3.storage.abcs import BarStore, StorageError, StoreQuery, StoreWriteResult

ParquetCompression = Literal["snappy", "zstd", "gzip", "lz4", "brotli", "uncompressed"]

_SYMBOL_SUFFIX_LEN: Final[int] = 3  # <month_code><year_2digit>, e.g. "M26"
_DEDUPE_SUBSET: Final[list[str]] = ["contract", "ts", "resolution", "content_bytes_sha"]
_SORT_KEYS: Final[list[str]] = ["ts", "content_bytes_sha"]
# Decimal price columns stored as string for lossless round-trip (never float).
_DECIMAL_COLS: Final[tuple[str, ...]] = ("open", "high", "low", "close")


def _bar_root(contract: ContractSymbol) -> str:
    s = str(contract)
    if len(s) <= _SYMBOL_SUFFIX_LEN:
        raise StorageError(f"contract symbol too short to parse root: {s!r}")
    return s[:-_SYMBOL_SUFFIX_LEN]


def _bars_to_df(bars: list[RawBar]) -> pl.DataFrame:
    rows = [
        {
            "contract": str(b.contract),
            "ts": b.ts.astimezone(UTC),
            "resolution": b.resolution.value,
            "open": str(b.open),
            "high": str(b.high),
            "low": str(b.low),
            "close": str(b.close),
            "volume": b.volume,
            "oi": b.oi,
            "source_id": b.source_id,
            "as_of_iso": b.as_of_iso.astimezone(UTC),
            "content_bytes_sha": b.content_bytes_sha,
        }
        for b in bars
    ]
    # Force nullable Int64 so an all-None oi batch does not infer a Null-typed column that
    # would clash with an existing Int64 column on concat.
    return pl.DataFrame(rows, schema_overrides={"volume": pl.Int64, "oi": pl.Int64})


def _row_to_bar(row: dict[str, object]) -> RawBar:
    ts = row["ts"]
    as_of = row["as_of_iso"]
    volume = row["volume"]
    oi = row["oi"]
    assert isinstance(ts, datetime)
    assert isinstance(as_of, datetime)
    assert isinstance(volume, int)
    assert oi is None or isinstance(oi, int)
    return RawBar(
        contract=ContractSymbol(str(row["contract"])),
        ts=ts,
        resolution=BarResolution(str(row["resolution"])),
        open=Decimal(str(row["open"])),
        high=Decimal(str(row["high"])),
        low=Decimal(str(row["low"])),
        close=Decimal(str(row["close"])),
        volume=volume,
        oi=oi,
        source_id=str(row["source_id"]),
        as_of_iso=as_of,
        content_bytes_sha=str(row["content_bytes_sha"]),
    )


class ParquetBarStore(BarStore):
    """Concrete polars-backed Parquet BarStore (A1.16b). Bit-reproducible via
    deterministic sort; append-only with idempotent dedupe."""

    BACKEND_ID: Final[str] = "parquet_local"

    def __init__(self, base_path: Path, compression: ParquetCompression = "snappy") -> None:
        self._base = Path(base_path)
        self._compression = compression

    @property
    def backend_id(self) -> str:
        return self.BACKEND_ID

    def healthcheck(self) -> bool:
        try:
            self._base.mkdir(parents=True, exist_ok=True)
            return self._base.is_dir()
        except OSError:
            return False

    def _partition_path(self, root: str, resolution: str, year: int, month: int) -> Path:
        return (
            self._base
            / "futures"
            / "bars"
            / root
            / resolution
            / f"year={year}"
            / f"month={month:02d}"
            / "data.parquet"
        )

    def write_bars(self, bars: list[RawBar]) -> StoreWriteResult:
        if not bars:
            return StoreWriteResult(rows_written=0, dedupe_count=0, path=self._base)

        groups: dict[tuple[str, str, int, int], list[RawBar]] = {}
        for b in bars:
            ts_utc = b.ts.astimezone(UTC)
            key = (_bar_root(b.contract), b.resolution.value, ts_utc.year, ts_utc.month)
            groups.setdefault(key, []).append(b)

        written = 0
        deduped = 0
        try:
            for (root, resolution, year, month), part_bars in groups.items():
                path = self._partition_path(root, resolution, year, month)
                path.parent.mkdir(parents=True, exist_ok=True)
                submitted = _bars_to_df(part_bars).unique(
                    subset=_DEDUPE_SUBSET, keep="last", maintain_order=True
                )
                internal_dupes = len(part_bars) - submitted.height
                if path.exists():
                    existing = pl.read_parquet(path)
                    new_only = submitted.join(
                        existing.select(_DEDUPE_SUBSET), on=_DEDUPE_SUBSET, how="anti"
                    )
                    collisions = submitted.height - new_only.height
                    final = pl.concat([existing, new_only], how="vertical_relaxed").sort(_SORT_KEYS)
                    final.write_parquet(path, compression=self._compression)
                    written += new_only.height
                    deduped += internal_dupes + collisions
                else:
                    submitted.sort(_SORT_KEYS).write_parquet(path, compression=self._compression)
                    written += submitted.height
                    deduped += internal_dupes
        except OSError as exc:
            raise StorageError(f"ParquetBarStore write failed: {exc}") from exc

        return StoreWriteResult(rows_written=written, dedupe_count=deduped, path=self._base)

    def read_bars(self, query: StoreQuery) -> Iterable[RawBar]:
        if query.resolution is None:
            raise ValueError("ParquetBarStore.read_bars requires query.resolution (bar read)")
        root = _bar_root(query.contract)
        partition_dir = self._base / "futures" / "bars" / root / query.resolution.value
        if not partition_dir.is_dir():
            return []
        files = sorted(partition_dir.glob("year=*/month=*/data.parquet"))
        if not files:
            return []

        ts_start = query.ts_start.astimezone(UTC)
        ts_end = query.ts_end.astimezone(UTC)
        end_predicate = pl.col("ts") <= ts_end if query.inclusive_end else pl.col("ts") < ts_end
        try:
            df = (
                pl.read_parquet([str(f) for f in files])
                .filter(pl.col("contract") == str(query.contract))
                .filter(pl.col("ts") >= ts_start)
                .filter(end_predicate)
                .sort(_SORT_KEYS)
            )
        except OSError as exc:
            raise StorageError(f"ParquetBarStore read failed: {exc}") from exc
        return [_row_to_bar(row) for row in df.iter_rows(named=True)]
