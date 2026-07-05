"""futur3.storage — Persistent storage layer (Phase A1.16+).

Per internal design notes +
the data-layer design.

A1.16a SHIPS THE ABCs only (this module). Concrete Parquet+DuckDB
implementations land in A1.16b. Engine + verifier code is written against
these abstract interfaces so the persistence backend can be swapped without
touching business logic (BACKTEST-IS-LIVE compatible).
"""

from __future__ import annotations

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
from futur3.storage.duckdb_query import DuckDbQueryEngine
from futur3.storage.parquet_settle_store import ParquetSettleStore
from futur3.storage.parquet_store import ParquetBarStore

__all__: list[str] = [
    "BarStore",
    "DuckDbQueryEngine",
    "IdempotentDuplicateError",
    "ParquetBarStore",
    "ParquetSettleStore",
    "PartitionNotFound",
    "SettleStore",
    "StorageError",
    "StorageNotInitialized",
    "StoreQuery",
    "StoreWriteResult",
]
