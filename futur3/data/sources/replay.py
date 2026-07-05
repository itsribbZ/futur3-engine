"""ReplayDataSource — BACKTEST-IS-LIVE foundation.

Phase A1.7 per the data-layer design + the verifier design.

Reads archived `Settle` records from the CME Parquet archive layout (already
written by `CMEEODDataSource._write_archive`), reconstructs the original
records (preserving every field including `source_id` + `as_of_iso` + the
provenance `content_bytes_sha`), and emits them through the `DataSource` ABC.

The engine cannot distinguish a `ReplayDataSource` from the original live source
beyond `DataSource.source_id` — by design (BACKTEST-IS-LIVE). The records
themselves still carry the ORIGINAL source's `source_id`, so the
`MultiSourceVerifier` (A1.8) cross-source consensus logic behaves identically
in backtest mode.

## Scope (A1.7 v1)

- **Settle replay** from `data/cme_eod_archive/contract={root}/year={YYYY}/data.parquet`
  — the layout locked by `CMEEODDataSource._write_archive`.
- **`latest_settle`** returns the freshest Settle ≤ `as_of` for a contract,
  reconstructed from the archived row.
- **PIT-honest** — the original `as_of_iso` (when the record was captured) is
  preserved. No look-ahead leak: records emit only if their capture-time was
  before the replay's `as_of` (bug class 5 prevention).
- **Idempotent + deterministic** — same archive + same query → same Settle.

## Deferred to later steps

- **Bar replay** (`get_bars`) — deferred to A1.16 storage layer + A1.7-v2.
  Raises `BarsNotSupported` with the deferral noted.
- **Tick replay** (`get_ticks`) — deferred to A1.6 streaming + storage. Raises
  `TicksNotSupported`.
- **Multi-archive merge** — A1.7 v1 reads one archive root. A1.7-v2 (after
  A1.16 storage abstraction) can merge across sources for cross-source replay.

## Invariants

- **PIT-honest by construction**: `_filter_settle_rows` selects only rows whose
  `as_of_date` ≤ `as_of.date()`. A future `as_of_iso` capture-time is impossible
  in replay since we always emit records that existed at archive-write-time.
- **Decimal-via-str on read** — archive stores Decimal as str (per
  `CMEEODDataSource._settles_to_dataframe`); reconstruction goes through
  `Decimal(row["..."])` directly (str → Decimal is exact). No float intermediary.
- **TZ-aware preserved** — Polars stores datetime with TZ; read-back returns
  TZ-aware datetime; we coerce to UTC defensively.
- **No vendor types leak** — Polars is a sources-layer internal; ABC surface
  emits `Settle` only.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, ClassVar, Final, cast

import polars as pl

from futur3.data.source import (
    BarsNotSupported,
    ContractNotConfigured,
    DataSource,
    DataSourceError,
)
from futur3.data.types import (
    BarResolution,
    ContractSymbol,
    RawBar,
    Settle,
    SettleState,
    SourceTier,
)

logger = logging.getLogger(__name__)

# Settle-state preference order for tie-breaking (latest 'final' wins over latest 'preliminary'
# on the same as_of_date — the CME archive preserves both via the dedupe key).
_SETTLE_STATE_RANK: Final[dict[SettleState, int]] = {
    "live": 0,
    "preliminary": 1,
    "final": 2,
}
# Min ContractSymbol length: <ROOT><MONTH_CODE><YY> = 4+ chars.
MIN_CONTRACT_SYMBOL_LENGTH: Final[int] = 4
# Default archive root matches CMEEODDataSource convention.
DEFAULT_CME_ARCHIVE_ROOT: Final[Path] = Path("data/cme_eod_archive")


class ReplayDataSource(DataSource):
    """Replays archived `Settle` records via the `DataSource` ABC.

    Reads the Parquet archive layout written by `CMEEODDataSource._write_archive`:
    `<archive_root>/contract={root}/year={YYYY}/data.parquet`. Reconstructs
    `Settle` objects preserving every captured field, including the original
    source's `source_id` (so cross-source verifier consensus works identically
    in backtest mode — BACKTEST-IS-LIVE).

    Args:
        archive_root: Directory containing `contract={root}/year={YYYY}/data.parquet`
            sub-tree. Default is `data/cme_eod_archive`.
        source_id: Identifier for THIS DataSource container (used by the verifier
            to distinguish ReplayDataSource instances when running multiple in
            parallel — e.g., one per archived source). Default: `"replay"`.
        tier: SourceTier for THIS container. Default `T4_DERIVED` per internal notes
    """

    DEFAULT_SOURCE_ID: ClassVar[str] = "replay"

    def __init__(
        self,
        *,
        archive_root: Path | None = None,
        source_id: str | None = None,
        tier: SourceTier = SourceTier.T4_DERIVED,
    ) -> None:
        self._archive_root: Path = archive_root or DEFAULT_CME_ARCHIVE_ROOT
        self._source_id: str = source_id or self.DEFAULT_SOURCE_ID
        self._tier: SourceTier = tier

    # ------------------------------------------------------------------------
    # DataSource ABC contract
    # ------------------------------------------------------------------------

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def tier(self) -> SourceTier:
        return self._tier

    def get_bars(
        self,
        contract: ContractSymbol,
        ts_start: datetime,
        ts_end: datetime,
        resolution: BarResolution,
    ) -> Iterable[RawBar]:
        """Bar replay is deferred to A1.16 storage layer + A1.7-v2.

        Raises `BarsNotSupported`. A1.7 v1 supports settle replay only — bar
        replay requires the BarStore/SettleStore ABCs (Phase A1.16) which
        externalize the on-disk schema beyond what CMEEODDataSource writes today.
        """
        raise BarsNotSupported(
            f"{self.source_id}: bar replay deferred to A1.16 storage layer + A1.7-v2"
        )

    def latest_settle(
        self,
        contract: ContractSymbol,
        as_of: datetime,
    ) -> Settle | None:
        """Return the freshest Settle ≤ `as_of` for `contract` from the archive.

        Reads the contract's Parquet archive, filters by `as_of_date <= as_of.date()`,
        and selects the row with the latest `as_of_date` (tie-break: prefer 'final'
        over 'preliminary' over 'live', then content_bytes_sha lexicographic for
        full determinism).

        Args:
            contract: ContractSymbol (e.g., ESM26). The root is parsed to locate
                the archive sub-directory.
            as_of: Reference time. Naive OR TZ-aware accepted (matches
                CMEEODDataSource ABC convention) — `.date()` is used for the
                PIT filter.

        Returns:
            Settle for the most recent archived business-date ≤ `as_of.date()`
            matching `contract`; or None if no row matches (archive missing,
            contract not present in archive, or all rows newer than `as_of`).

        Raises:
            ContractNotConfigured: `contract` symbol shorter than 4 chars (cannot
                parse root).
            DataSourceError: Parquet read failure or row reconstruction failure.
        """
        root = self._parse_contract_root(contract)
        contract_dir = self._archive_root / f"contract={root}"
        if not contract_dir.exists():
            return None

        as_of_date = as_of.date()
        candidates: list[Settle] = []

        # Iterate year sub-dirs whose year ≤ as_of's year (skip future years).
        for year_dir in sorted(contract_dir.glob("year=*")):
            year = self._parse_year_dir(year_dir)
            if year is None or year > as_of_date.year:
                continue
            parquet_path = year_dir / "data.parquet"
            if not parquet_path.exists():
                continue
            try:
                df = pl.read_parquet(parquet_path)
            except Exception as e:  # pragma: no cover - polars internal failure
                raise DataSourceError(
                    f"{self.source_id}: failed to read archive {parquet_path}: "
                    f"{type(e).__name__}: {e}"
                ) from e

            matching = df.filter(
                (pl.col("contract") == str(contract)) & (pl.col("as_of_date") <= as_of_date)
            )
            if matching.is_empty():
                continue
            for row in matching.to_dicts():
                candidates.append(self._row_to_settle(row))

        if not candidates:
            return None

        # Latest by as_of_date; tie-break by settle_state preference (final > preliminary > live),
        # then by content_bytes_sha lexicographic for total determinism.
        candidates.sort(
            key=lambda s: (
                s.as_of_date,
                _SETTLE_STATE_RANK[s.settle_state],
                s.content_bytes_sha,
            )
        )
        return candidates[-1]

    def healthcheck(self) -> bool:
        """Archive root must exist for healthcheck to pass."""
        return self._archive_root.exists()

    # ------------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------------

    def _parse_contract_root(self, contract: ContractSymbol) -> str:
        """ESM26 → 'ES'. Strict: requires ≥4 chars + last 3 are <code><yy>."""
        s = str(contract)
        if len(s) < MIN_CONTRACT_SYMBOL_LENGTH:
            raise ContractNotConfigured(
                f"{self.source_id}: contract {s!r} too short for root parse "
                f"(expected <ROOT><MONTH_CODE><YY>)"
            )
        return s[:-3]

    @staticmethod
    def _parse_year_dir(year_dir: Path) -> int | None:
        """`year=2026` → 2026. Returns None on malformed name (logs at debug)."""
        name = year_dir.name
        prefix = "year="
        if not name.startswith(prefix):
            return None
        try:
            return int(name[len(prefix) :])
        except ValueError:
            logger.debug("ReplayDataSource: skipping malformed year dir %s", year_dir)
            return None

    def _row_to_settle(self, row: dict[str, Any]) -> Settle:
        """Reconstruct a `Settle` from a polars row dict.

        Per `_settles_to_dataframe` schema:
        - Decimal fields stored as str → `Decimal(row[...])`
        - settle_state stored as str → cast to `SettleState` Literal
        - as_of_date stored as Python date
        - as_of_iso stored as TZ-aware datetime → coerced to UTC defensively
        - oi_prior + volume_est stored as int
        - contract/source_id/cme_month_code/content_bytes_sha stored as str
        """
        settle_state_raw = row["settle_state"]
        if settle_state_raw not in _SETTLE_STATE_RANK:
            raise DataSourceError(
                f"{self.source_id}: archive row has unknown settle_state "
                f"{settle_state_raw!r}; valid: {sorted(_SETTLE_STATE_RANK)}"
            )
        settle_state: SettleState = cast(SettleState, settle_state_raw)

        as_of_iso_raw = row.get("as_of_iso")
        if as_of_iso_raw is None:
            as_of_iso_val: datetime | None = None
        elif isinstance(as_of_iso_raw, datetime):
            # Refuse naive datetime (fail-loud).
            # Polars stores datetime[us, UTC] when the source had TZ, but if the
            # Parquet was written by a different tool or the schema evolved, TZ
            # could be stripped. Silent .replace(tzinfo=UTC) would let look-ahead
            # bugs propagate; raise so the write-side is forced to populate TZ.
            if as_of_iso_raw.tzinfo is None:
                raise DataSourceError(
                    f"{self.source_id}: archive row as_of_iso is naive "
                    f"{as_of_iso_raw!r}; refusing silent UTC assumption "
                    f"(fail-loud — write-side must populate TZ)"
                )
            as_of_iso_val = as_of_iso_raw.astimezone(UTC)
        else:
            raise DataSourceError(
                f"{self.source_id}: as_of_iso type {type(as_of_iso_raw).__name__} "
                f"not coercible to datetime (value: {as_of_iso_raw!r})"
            )

        as_of_date_raw = row["as_of_date"]
        if not isinstance(as_of_date_raw, date):
            raise DataSourceError(
                f"{self.source_id}: as_of_date type {type(as_of_date_raw).__name__} "
                f"is not a date (value: {as_of_date_raw!r})"
            )

        return Settle(
            contract=ContractSymbol(str(row["contract"])),
            as_of_date=as_of_date_raw,
            settle=Decimal(str(row["settle"])),
            settle_state=settle_state,
            open=Decimal(str(row["open"])),
            high=Decimal(str(row["high"])),
            low=Decimal(str(row["low"])),
            last=Decimal(str(row["last"])),
            change=Decimal(str(row["change"])),
            volume_est=int(row["volume_est"]),
            oi_prior=int(row["oi_prior"]),
            source_id=str(row["source_id"]),
            as_of_iso=as_of_iso_val,
            content_bytes_sha=str(row["content_bytes_sha"]),
            cme_month_code=str(row["cme_month_code"]),
        )
