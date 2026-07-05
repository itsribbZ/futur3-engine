"""futur3.storage.parquet_settle_store - ParquetSettleStore (A1.16c concrete SettleStore).

Per internal design notes, implementing the A1.16a
`SettleStore` ABC. The daily-cadence twin of `ParquetBarStore` (A1.16b): same proven polars
pattern (Decimal stored as string for lossless precision, tz-aware datetimes coerced to UTC,
idempotent dedupe via anti-join, deterministic sort), differing only in partition cadence
(daily settles -> year partitions, no intraday month split) and the dedupe key.

Partition layout (hive-style, DuckDB-glob-readable per spec section 1.4):
    <base>/futures/settles/<root>/year=<YYYY>/data.parquet

Append-only + idempotent dedupe on (contract, as_of_date, settle_state, content_bytes_sha):
- identical key (same fetch repeated) -> skipped (idempotent; counted in dedupe_count);
- preliminary->final transition (same date, distinct settle_state) -> BOTH kept (required for
  BACKTEST-IS-LIVE backtest-vs-replay equivalence);
- retroactive revision (same date+state, distinct content_bytes_sha) -> BOTH kept (preserves
  revision history for the verifier's RetroactiveRevision detection).
Rows are sorted (as_of_date asc, settle_state-rank asc, content_bytes_sha) before write so
re-derivation from identical inputs yields identical logical content + order. The
settle_state rank reflects the chronological progression live < preliminary < final.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Final, Literal, cast

import polars as pl

from futur3.data.types import ContractSymbol, Settle, SettleState
from futur3.storage.abcs import SettleStore, StorageError, StoreWriteResult

ParquetCompression = Literal["snappy", "zstd", "gzip", "lz4", "brotli", "uncompressed"]

# <month_code><year_2digit>, e.g. "M26"; mirrors ParquetBarStore._bar_root, kept local so this
# module stays self-contained (no cross-import of a private helper from parquet_store).
_SYMBOL_SUFFIX_LEN: Final[int] = 3
_DEDUPE_SUBSET: Final[list[str]] = ["contract", "as_of_date", "settle_state", "content_bytes_sha"]
# Chronological settle progression rank -> deterministic ordering on write + read.
_SETTLE_STATE_RANK: Final[dict[str, int]] = {"live": 0, "preliminary": 1, "final": 2}


def _settle_root(contract: ContractSymbol) -> str:
    s = str(contract)
    if len(s) <= _SYMBOL_SUFFIX_LEN:
        raise StorageError(f"contract symbol too short to parse root: {s!r}")
    return s[:-_SYMBOL_SUFFIX_LEN]


def _sort_exprs() -> list[pl.Expr]:
    """Deterministic sort: as_of_date asc, settle_state chronological rank asc, sha tiebreak."""
    return [
        pl.col("as_of_date"),
        pl.col("settle_state").replace_strict(_SETTLE_STATE_RANK, return_dtype=pl.Int8),
        pl.col("content_bytes_sha"),
    ]


def _settles_to_df(settles: list[Settle]) -> pl.DataFrame:
    rows = [
        {
            "contract": str(s.contract),
            "as_of_date": s.as_of_date,
            "settle": str(s.settle),
            "settle_state": s.settle_state,
            "open": str(s.open),
            "high": str(s.high),
            "low": str(s.low),
            "last": str(s.last),
            "change": str(s.change),  # may be negative (settle - prior_settle)
            "volume_est": s.volume_est,
            "oi_prior": s.oi_prior,
            "source_id": s.source_id,
            "as_of_iso": s.as_of_iso.astimezone(UTC) if s.as_of_iso is not None else None,
            "content_bytes_sha": s.content_bytes_sha,
            "cme_month_code": s.cme_month_code,
        }
        for s in settles
    ]
    # Force typed/nullable columns so an all-None as_of_iso batch does not infer a Null-typed
    # column that would clash with an existing typed column on concat (the bars `oi` lesson).
    return pl.DataFrame(
        rows,
        schema_overrides={
            "volume_est": pl.Int64,
            "oi_prior": pl.Int64,
            "as_of_iso": pl.Datetime("us", time_zone="UTC"),
        },
    )


def _row_to_settle(row: dict[str, object]) -> Settle:
    as_of_date = row["as_of_date"]
    as_of_iso = row["as_of_iso"]
    volume_est = row["volume_est"]
    oi_prior = row["oi_prior"]
    state_str = str(row["settle_state"])
    assert isinstance(as_of_date, date)
    assert as_of_iso is None or isinstance(as_of_iso, datetime)
    assert isinstance(volume_est, int)
    assert isinstance(oi_prior, int)
    if state_str not in _SETTLE_STATE_RANK:  # no silent fallback: corrupt state fails loud
        raise StorageError(f"corrupt settle_state in storage: {state_str!r}")
    return Settle(
        contract=ContractSymbol(str(row["contract"])),
        as_of_date=as_of_date,
        settle=Decimal(str(row["settle"])),
        settle_state=cast(SettleState, state_str),
        open=Decimal(str(row["open"])),
        high=Decimal(str(row["high"])),
        low=Decimal(str(row["low"])),
        last=Decimal(str(row["last"])),
        change=Decimal(str(row["change"])),
        volume_est=volume_est,
        oi_prior=oi_prior,
        source_id=str(row["source_id"]),
        as_of_iso=as_of_iso,
        content_bytes_sha=str(row["content_bytes_sha"]),
        cme_month_code=str(row["cme_month_code"]),
    )


class ParquetSettleStore(SettleStore):
    """Concrete polars-backed Parquet SettleStore (A1.16c). The daily-cadence twin of
    ParquetBarStore: bit-reproducible via deterministic sort; append-only with idempotent
    dedupe; preliminary->final + retroactive revisions preserved."""

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

    def _partition_path(self, root: str, year: int) -> Path:
        return self._base / "futures" / "settles" / root / f"year={year}" / "data.parquet"

    def write_settles(self, settles: list[Settle]) -> StoreWriteResult:
        if not settles:
            return StoreWriteResult(rows_written=0, dedupe_count=0, path=self._base)

        groups: dict[tuple[str, int], list[Settle]] = {}
        for s in settles:
            groups.setdefault((_settle_root(s.contract), s.as_of_date.year), []).append(s)

        written = 0
        deduped = 0
        try:
            for (root, year), part in groups.items():
                path = self._partition_path(root, year)
                path.parent.mkdir(parents=True, exist_ok=True)
                submitted = _settles_to_df(part).unique(
                    subset=_DEDUPE_SUBSET, keep="last", maintain_order=True
                )
                internal_dupes = len(part) - submitted.height
                if path.exists():
                    existing = pl.read_parquet(path)
                    new_only = submitted.join(
                        existing.select(_DEDUPE_SUBSET), on=_DEDUPE_SUBSET, how="anti"
                    )
                    collisions = submitted.height - new_only.height
                    final = pl.concat([existing, new_only], how="vertical_relaxed").sort(
                        _sort_exprs()
                    )
                    final.write_parquet(path, compression=self._compression)
                    written += new_only.height
                    deduped += internal_dupes + collisions
                else:
                    submitted.sort(_sort_exprs()).write_parquet(path, compression=self._compression)
                    written += submitted.height
                    deduped += internal_dupes
        except OSError as exc:
            raise StorageError(f"ParquetSettleStore write failed: {exc}") from exc

        return StoreWriteResult(rows_written=written, dedupe_count=deduped, path=self._base)

    def read_settles(
        self,
        contract: ContractSymbol,
        as_of_date_start: date,
        as_of_date_end: date,
    ) -> Iterable[Settle]:
        if as_of_date_end < as_of_date_start:
            raise ValueError(
                f"ParquetSettleStore.read_settles: as_of_date_end {as_of_date_end} < "
                f"as_of_date_start {as_of_date_start}"
            )
        root = _settle_root(contract)
        partition_dir = self._base / "futures" / "settles" / root
        if not partition_dir.is_dir():
            return []
        files = sorted(partition_dir.glob("year=*/data.parquet"))
        if not files:
            return []
        try:
            df = (
                pl.read_parquet([str(f) for f in files])
                .filter(pl.col("contract") == str(contract))
                .filter(pl.col("as_of_date") >= as_of_date_start)
                .filter(pl.col("as_of_date") <= as_of_date_end)  # INCLUSIVE end (daily cadence)
                .sort(_sort_exprs())
            )
        except OSError as exc:
            raise StorageError(f"ParquetSettleStore read failed: {exc}") from exc
        return [_row_to_settle(row) for row in df.iter_rows(named=True)]
