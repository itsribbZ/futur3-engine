"""BarStore + SettleStore ABCs + storage layer dataclasses + exceptions.

Phase A1.16a per internal design notes.

## Scope (A1.16a interface only)

This module defines the abstract write/read seam between the data layer and
the persistent storage backend. The seam is **dataclass-typed** — `BarStore`
takes `RawBar` lists, `SettleStore` takes `Settle` lists — so backend
implementations cannot accidentally write wrong-shaped records.

Concrete Parquet+DuckDB implementations land in A1.16b. This file ships:
- 2 ABCs (BarStore + SettleStore) with the persist + read seam
- 2 frozen dataclasses (StoreQuery + StoreWriteResult)
- 4-member exception hierarchy
- 0 concrete implementations

## Storage semantics (locked per research)

- **Append-only**: never delete rows. Preliminary->final + retroactive
  revisions are preserved with distinct `content_bytes_sha` for verifier diff.
- **Idempotent dedupe**: writing the same (key, content_bytes_sha) tuple
  twice is a no-op (write_result.dedupe_count counts skipped rows).
- **Bit-reproducible**: sort rows deterministically before write
  (ts ascending + content_bytes_sha tie-breaker); same input -> byte-equal
  file on disk (modulo backend-specific framing).
- **Read returns frozen dataclasses** (RawBar / Settle) — never raw rows
  or vendor types leak past the seam.

## Contracts honored even in ABC

- **BACKTEST-IS-LIVE**: same storage interface in backtest vs live;
  ReplayDataSource can write to the same backend the live system reads from.
- **No-silent-fallback**: write errors raise StorageError subclasses;
  empty reads return empty iterables but never silently swallow IO failures.
- **Per-step quality bar**: ABC shell + tests triple-green on
  first commit; concrete impl is its own ship (A1.16b).

References:
- internal design notes (architectural integration)
- the data-layer design (dataclass schema lock)
- the verifier design (revision preservation contract)
"""

from __future__ import annotations

import abc
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Final

from futur3.data.types import (
    BarResolution,
    ContractSymbol,
    RawBar,
    Settle,
)

# Minimum number of rows in a sortable write batch. Single-row writes are
# allowed (used by tests) but multi-row writes MUST be sortable for bit-reproducibility.
_MIN_WRITE_BATCH_SIZE: Final[int] = 0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class StorageError(Exception):
    """Base for all storage-layer errors. Caught at engine + verifier boundary."""


class StorageNotInitialized(StorageError):
    """Store accessed before its backend (directory, connection, etc.) is ready.

    Concrete impls raise this on operations BEFORE first successful `healthcheck()`
    or BEFORE the operator has materialized the backend root directory.
    """


class IdempotentDuplicateError(StorageError):
    """Write attempted a (key, content_bytes_sha) tuple that ALREADY exists.

    NOTE: this is an EXCEPTIONAL signal — backends SHOULD silently skip
    duplicate rows during normal `write_bars()` (counting them in
    `dedupe_count`). This error is reserved for STRICT mode where the
    engine wants to know the duplicate happened (e.g., revision-detection
    tests).
    """


class PartitionNotFound(StorageError):
    """Requested partition (contract / year / month) does not exist in storage.

    Read operations on a missing partition return empty iterables by default;
    backends raise this only when explicit "must exist" semantics are needed
    (e.g., PIT replay must fail-loudly if the requested archive is absent).
    """


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StoreWriteResult:
    """Outcome of a single `write_*` call.

    `rows_written` = count of NEW rows persisted (post-dedupe).
    `dedupe_count` = count of submitted rows that were already on disk and
    therefore skipped (idempotent dedupe contract).
    `path` = backend-specific location written to (Parquet file path for
    Parquet backends; SQL identifier for relational backends).
    """

    rows_written: int
    dedupe_count: int
    path: Path

    def __post_init__(self) -> None:
        if self.rows_written < 0:
            raise ValueError(f"StoreWriteResult.rows_written must be >= 0; got {self.rows_written}")
        if self.dedupe_count < 0:
            raise ValueError(f"StoreWriteResult.dedupe_count must be >= 0; got {self.dedupe_count}")

    @property
    def total_submitted(self) -> int:
        return self.rows_written + self.dedupe_count


@dataclass(frozen=True)
class StoreQuery:
    """Read-side query spec. Bar-and-Settle-shape-agnostic.

    Backends use `(contract, ts_start, ts_end)` as the primary filter.
    `resolution` is required for bar queries, ignored for settle queries.
    `inclusive_end` defaults to False (half-open interval `[ts_start, ts_end)`)
    to match `DataSource.get_bars()` convention.
    """

    contract: ContractSymbol
    ts_start: datetime
    ts_end: datetime
    resolution: BarResolution | None = None
    inclusive_end: bool = False

    def __post_init__(self) -> None:
        if self.ts_start.tzinfo is None:
            raise ValueError(f"StoreQuery.ts_start must be TZ-aware; got naive {self.ts_start!r}")
        if self.ts_end.tzinfo is None:
            raise ValueError(f"StoreQuery.ts_end must be TZ-aware; got naive {self.ts_end!r}")
        if self.ts_end <= self.ts_start:
            raise ValueError(
                f"StoreQuery: ts_end must be > ts_start; "
                f"got ts_start={self.ts_start} ts_end={self.ts_end}"
            )


# ---------------------------------------------------------------------------
# BarStore ABC
# ---------------------------------------------------------------------------


class BarStore(abc.ABC):
    """Abstract bar storage backend.

    Concrete impls (A1.16b ParquetBarStore + DuckDbQueryEngine) persist
    `RawBar` / `VerifiedBar` lists via deterministic file layout
    + idempotent dedupe + bit-reproducible serialization.

    Engine + verifier code is written against THIS interface, never the
    concrete backend.
    """

    @property
    @abc.abstractmethod
    def backend_id(self) -> str:
        """Stable per-backend identifier (e.g., `"parquet_local"`,
        `"duckdb_inmem"`). Used for telemetry + provenance tagging.
        """

    @abc.abstractmethod
    def write_bars(self, bars: list[RawBar]) -> StoreWriteResult:
        """Persist a batch of RawBars.

        Append-only + idempotent dedupe on (contract, ts, resolution,
        content_bytes_sha) — if any submitted row collides with an existing
        row by that 4-tuple, it is silently skipped (counted in
        `dedupe_count`). Backends MUST sort the input deterministically
        before persistence so the output file is byte-equal across runs
        with identical inputs.

        Args:
            bars: List of RawBar records. Empty list returns a zero-result
                  successfully (no-op). Mixing different contracts /
                  resolutions in one call is allowed — backends fan out
                  into per-partition files internally.

        Returns:
            StoreWriteResult with `rows_written` + `dedupe_count` + `path`.

        Raises:
            StorageNotInitialized: backend not ready (directory absent / etc.).
            StorageError: any other backend-side write failure (disk full,
                          schema drift, permission denied, etc.).
        """

    @abc.abstractmethod
    def read_bars(self, query: StoreQuery) -> Iterable[RawBar]:
        """Stream RawBars matching `query`.

        Backends MUST return RawBars in deterministic (ts ascending) order so
        downstream consumers (verifier, replay) see consistent state.

        Returns an empty iterable if the partition is empty OR does not
        exist (no PartitionNotFound by default). To force-fail on missing
        partitions, use `require_partition_exists(query)` first.

        Args:
            query: StoreQuery with at least `contract` + `ts_start` +
                   `ts_end` set. `resolution` is REQUIRED for bar reads;
                   if None, backends raise ValueError.

        Returns:
            Iterable[RawBar] in ascending-ts order.

        Raises:
            ValueError: query.resolution is None.
            StorageNotInitialized: backend not ready.
            StorageError: backend-side read failure (schema mismatch, etc.).
        """

    @abc.abstractmethod
    def healthcheck(self) -> bool:
        """Return True iff backend is ready for write + read operations.

        Concrete impls verify directory existence (Parquet) or connection
        liveness (DuckDB). Returns False rather than raising so callers
        can decide policy.
        """


# ---------------------------------------------------------------------------
# SettleStore ABC
# ---------------------------------------------------------------------------


class SettleStore(abc.ABC):
    """Abstract daily-settle storage backend.

    Mirrors BarStore but typed over `Settle` records — daily-business-date
    cadence (not intraday-ts), preliminary->final transition preservation,
    retroactive revision preservation per internal design notes.
    """

    @property
    @abc.abstractmethod
    def backend_id(self) -> str:
        """Stable per-backend identifier."""

    @abc.abstractmethod
    def write_settles(self, settles: list[Settle]) -> StoreWriteResult:
        """Persist a batch of Settles.

        Append-only + idempotent dedupe on (contract, as_of_date,
        settle_state, content_bytes_sha) — preserving preliminary->final
        transitions + retroactive revisions per internal notes

        Empty list returns a zero-result successfully.

        Raises:
            StorageNotInitialized / StorageError per BarStore semantics.
        """

    @abc.abstractmethod
    def read_settles(
        self,
        contract: ContractSymbol,
        as_of_date_start: date,
        as_of_date_end: date,
    ) -> Iterable[Settle]:
        """Stream Settles for `contract` with `as_of_date` in
        `[as_of_date_start, as_of_date_end]` (INCLUSIVE end — daily cadence).

        Multiple Settles per `as_of_date` are possible (preliminary +
        final + revisions); backends return ALL of them in ascending
        (as_of_date, settle_state-rank) order so the consumer can pick
        per its own policy.

        Returns:
            Iterable[Settle] in ascending order.

        Raises:
            ValueError: as_of_date_end < as_of_date_start.
            StorageNotInitialized / StorageError per BarStore semantics.
        """

    @abc.abstractmethod
    def healthcheck(self) -> bool:
        """True iff backend ready."""


__all__: list[str] = [
    "BarStore",
    "IdempotentDuplicateError",
    "PartitionNotFound",
    "SettleStore",
    "StorageError",
    "StorageNotInitialized",
    "StoreQuery",
    "StoreWriteResult",
]
