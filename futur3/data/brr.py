"""BRRReconstructor — Phase A1.11 free-first BRR / ETHUSD_RR reconstruction.

Per internal crypto-data notes (4-of-7 reconstruction architecture).

Reconstructs the CF Benchmarks Bitcoin Reference Rate (BRR) — and the analogous
ETHUSD_RR — from 4-of-7 BRR-constituent venues (Coinbase + Kraken + Bitstamp +
Gemini). Each fix is computed from the 15:00-16:00 London hour, partitioned
into 12 x 5-minute windows.

## Algorithm (per §7.4)

1. For the given `fix_date`, compute the 15:00:00.000-16:00:00.000 Europe/London
   window in UTC (DST-aware via `zoneinfo`).
2. Fetch 5-min OHLCV bars per venue via `DataSource.get_bars()`.
3. For each of the 12 partitions `Pi` (chronological, 5-min stride):
   a. Collect the close price from each venue's bar landing in `[Pi_start, Pi_end)`.
   b. Drop any venue with zero bars in `Pi` (graceful degradation).
   c. Require `>=2` venues per partition (else `Pi` is marked invalid).
   d. `Pi_aggregate` = arithmetic mean of remaining venue closes.
4. `BRR_reconstructed` = arithmetic mean of valid `Pi_aggregate` values.
5. Content hash = SHA256 of canonical-JSON of `(date, contract, per-venue partition
   tuples sorted by venue_id)`.

## v1 simplifications vs full BRR methodology

- Uses 5-min BAR CLOSE as the per-partition price (NOT tick-level VWMed). Full
  VWMed precision requires WebSocket tick streams (deferred to A1.6+ realtime
  layer). Drift envelope vs published BRR will be wider than the 1-5 bp target
  until WS lands.
- No outlier exclusion (10% top/bottom requires N=10+ samples; with N=4 the
  exclusion mask is empty per spec §7.4).
- No `cfbenchmarks.com` EOD calibration scrape (A1.11.b future ship).
- No Parquet archive write (defer to A1.16 storage layer).

## Contracts

- **Mode-agnostic** (BACKTEST-IS-LIVE): same code path live vs replay; a
  ReplayDataSource for crypto can replace any source transparently.
- **Deterministic**: same source inputs → byte-equal `content_bytes_sha`.
  Source order independence: venues sorted by `source_id` at construction.
- **Apparatus-pure** (fail-loud): no I/O beyond `DataSource.get_bars` calls; no
  logging of values (only event counts + warnings on degraded partitions).
- **DST-aware**: 15:00-16:00 London window honors BST↔GMT transitions via
  `ZoneInfo("Europe/London")`.

References:
- internal crypto-data notes (4-of-7 architecture)
- internal crypto-data notes (algorithm)
- internal crypto-data notes (calibration cadence — v2)
- the data-layer design (Phase A1 step location)
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from typing import Final
from zoneinfo import ZoneInfo

from futur3.data.source import DataSource, DataSourceError
from futur3.data.types import (
    SHA256_HEX_LENGTH,
    BarResolution,
    ContractSymbol,
    RawBar,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LONDON_TZ: Final[ZoneInfo] = ZoneInfo("Europe/London")
_BRR_WINDOW_START_LONDON: Final[time] = time(15, 0, 0)  # 15:00:00 London
_BRR_WINDOW_END_LONDON: Final[time] = time(16, 0, 0)  # 16:00:00 London
_NUM_PARTITIONS: Final[int] = 12  # 12 x 5-min in 1hr window
_PARTITION_MINUTES: Final[int] = 5
_MIN_VENUES_PER_PARTITION: Final[int] = 2  # graceful-degradation floor

# BRR / ETHUSD_RR supported contracts (the only two CF Benchmarks publishes).
_SUPPORTED_CONTRACTS: Final[frozenset[ContractSymbol]] = frozenset(
    {
        ContractSymbol("BTCUSD"),
        ContractSymbol("ETHUSD"),
    }
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class BRRError(DataSourceError):
    """Base error for BRR reconstruction failures."""


class BRRWindowEmpty(BRRError):
    """No partition reached MIN_VENUES_PER_PARTITION — cannot reconstruct."""


class InsufficientVenues(BRRError):
    """Construction-time: < MIN_VENUES passed to BRRReconstructor."""


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BRRReconstruction:
    """Single-day reconstructed BRR / ETHUSD_RR rate.

    Fields:
    - `contract`: `BTCUSD` or `ETHUSD`
    - `fix_date`: London business date the fix applies to
    - `reconstructed_value`: final reconstructed rate (Decimal-via-str)
    - `partition_values`: 12 partition aggregate values (None for degraded
      partitions where < MIN_VENUES_PER_PARTITION responded)
    - `num_partitions_valid`: count of partitions that aggregated successfully
    - `venue_ids`: tuple of `source_id` strings (sorted for determinism)
    - `content_bytes_sha`: hex-SHA256 of canonical-JSON inputs (bit-repro anchor)
    """

    contract: ContractSymbol
    fix_date: date
    reconstructed_value: Decimal
    partition_values: tuple[Decimal | None, ...]
    num_partitions_valid: int
    venue_ids: tuple[str, ...]
    content_bytes_sha: str

    def __post_init__(self) -> None:
        if len(self.partition_values) != _NUM_PARTITIONS:
            raise ValueError(
                f"BRRReconstruction.partition_values must have "
                f"{_NUM_PARTITIONS} entries; got {len(self.partition_values)}"
            )
        if self.num_partitions_valid < 1 or self.num_partitions_valid > _NUM_PARTITIONS:
            raise ValueError(
                f"BRRReconstruction.num_partitions_valid must be in "
                f"[1, {_NUM_PARTITIONS}]; got {self.num_partitions_valid}"
            )
        if self.reconstructed_value <= 0:
            raise ValueError(
                f"BRRReconstruction.reconstructed_value must be > 0; got {self.reconstructed_value}"
            )
        if len(self.content_bytes_sha) != SHA256_HEX_LENGTH:
            raise ValueError(
                f"BRRReconstruction.content_bytes_sha must be hex-SHA256 "
                f"({SHA256_HEX_LENGTH} chars); got {len(self.content_bytes_sha)}"
            )
        if len(self.venue_ids) < _MIN_VENUES_PER_PARTITION:
            raise ValueError(
                f"BRRReconstruction.venue_ids must have at least "
                f"{_MIN_VENUES_PER_PARTITION}; got {len(self.venue_ids)}"
            )


# ---------------------------------------------------------------------------
# BRRReconstructor
# ---------------------------------------------------------------------------


class BRRReconstructor:
    """4-of-7 BRR reconstructor consuming N >= 2 crypto DataSources.

    Constructor takes a list of `DataSource` instances (typically
    Coinbase + Kraken + Bitstamp + Gemini per BRR constituent spec). Sources
    are sorted by `source_id` at construction for deterministic ordering
    (network call order is non-deterministic; canonical sort fixes the content hash).
    """

    def __init__(self, sources: list[DataSource]) -> None:
        if len(sources) < _MIN_VENUES_PER_PARTITION:
            raise InsufficientVenues(
                f"BRRReconstructor requires at least "
                f"{_MIN_VENUES_PER_PARTITION} sources; got {len(sources)}"
            )
        self._sources: tuple[DataSource, ...] = tuple(sorted(sources, key=lambda s: s.source_id))

    @property
    def venue_ids(self) -> tuple[str, ...]:
        """Sorted tuple of source_ids (deterministic; matches reconstruction output)."""
        return tuple(s.source_id for s in self._sources)

    def __repr__(self) -> str:
        return f"BRRReconstructor(venues={list(self.venue_ids)})"

    def reconstruct(
        self,
        contract: ContractSymbol,
        fix_date: date,
    ) -> BRRReconstruction:
        """Reconstruct BRR for `contract` on `fix_date`.

        Args:
            contract: `BTCUSD` or `ETHUSD` (per `_SUPPORTED_CONTRACTS`).
            fix_date: London business date the BRR fix applies to.

        Raises:
            DataSourceError: contract not supported.
            BRRWindowEmpty: 0 of 12 partitions reached `MIN_VENUES_PER_PARTITION`.
        """
        if contract not in _SUPPORTED_CONTRACTS:
            raise DataSourceError(
                f"BRRReconstructor: contract {contract!r} not supported. "
                f"Supported: {sorted(_SUPPORTED_CONTRACTS)}"
            )

        ts_start_utc, ts_end_utc = self._london_window_utc(fix_date)

        # Fetch + window-filter bars per venue. Defensive filter: enforces
        # half-open [ts_start, ts_end) in case a venue returns boundary-leaky
        # bars. Same filtered set drives BOTH partitioning AND hash (so the
        # hash captures only bars that contributed to reconstruction).
        bars_per_venue: dict[str, list[RawBar]] = {}
        for src in self._sources:
            raw_bars = list(
                src.get_bars(
                    contract,
                    ts_start_utc,
                    ts_end_utc,
                    BarResolution.MIN_5,
                )
            )
            bars_per_venue[src.source_id] = [
                b for b in raw_bars if ts_start_utc <= b.ts < ts_end_utc
            ]

        partition_aggregates: list[Decimal | None] = []
        for i in range(_NUM_PARTITIONS):
            p_start = ts_start_utc + timedelta(minutes=i * _PARTITION_MINUTES)
            p_end = p_start + timedelta(minutes=_PARTITION_MINUTES)
            venue_closes: list[Decimal] = []
            for src in self._sources:
                hits = [b for b in bars_per_venue[src.source_id] if p_start <= b.ts < p_end]
                if not hits:
                    continue  # graceful: venue missed this partition
                # v1: use bar CLOSE (tick-level VWMed deferred to A1.6).
                # Multiple bars in one 5-min partition shouldn't happen with
                # BarResolution.MIN_5, but if it does, take the LAST close.
                last_close = max(hits, key=lambda b: b.ts).close
                venue_closes.append(last_close)
            if len(venue_closes) < _MIN_VENUES_PER_PARTITION:
                partition_aggregates.append(None)
                logger.debug(
                    "BRR partition %d/%d for %s @ %s degraded: only %d venue(s)",
                    i + 1,
                    _NUM_PARTITIONS,
                    contract,
                    fix_date.isoformat(),
                    len(venue_closes),
                )
                continue
            partition_aggregates.append(self._arithmetic_mean(venue_closes))

        valid_partitions = [v for v in partition_aggregates if v is not None]
        if not valid_partitions:
            raise BRRWindowEmpty(
                f"BRRReconstructor: 0 of {_NUM_PARTITIONS} partitions reached "
                f"min_venues={_MIN_VENUES_PER_PARTITION} for {contract} "
                f"@ {fix_date.isoformat()} (window {ts_start_utc.isoformat()} "
                f"- {ts_end_utc.isoformat()})"
            )
        brr_value = self._arithmetic_mean(valid_partitions)

        content_sha = self._compute_content_sha(
            contract=contract,
            fix_date=fix_date,
            bars_per_venue=bars_per_venue,
        )

        return BRRReconstruction(
            contract=contract,
            fix_date=fix_date,
            reconstructed_value=brr_value,
            partition_values=tuple(partition_aggregates),
            num_partitions_valid=len(valid_partitions),
            venue_ids=self.venue_ids,
            content_bytes_sha=content_sha,
        )

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _london_window_utc(fix_date: date) -> tuple[datetime, datetime]:
        """Return `(ts_start_utc, ts_end_utc)` for the 15:00-16:00 London
        window on `fix_date`. DST-aware via `ZoneInfo("Europe/London")` —
        in BST (summer) the window is 14:00-15:00 UTC; in GMT (winter) it
        is 15:00-16:00 UTC.
        """
        start_local = datetime.combine(fix_date, _BRR_WINDOW_START_LONDON, tzinfo=_LONDON_TZ)
        end_local = datetime.combine(fix_date, _BRR_WINDOW_END_LONDON, tzinfo=_LONDON_TZ)
        return start_local.astimezone(UTC), end_local.astimezone(UTC)

    @staticmethod
    def _arithmetic_mean(values: list[Decimal]) -> Decimal:
        """Decimal arithmetic mean. Empty list raises (caller invariant)."""
        if not values:
            raise AssertionError(
                "_arithmetic_mean called with empty list; "
                "caller must guard via partition validity check first"
            )
        total = sum(values, Decimal("0"))
        return total / Decimal(len(values))

    @staticmethod
    def _compute_content_sha(
        contract: ContractSymbol,
        fix_date: date,
        bars_per_venue: dict[str, list[RawBar]],
    ) -> str:
        """Content hash: canonical-JSON of sorted per-venue bar tuples.

        Per internal notes: SHA256 of (sorted concat of trade-tuples per venue).
        v1 approximation: 5-min bar tuples instead of ticks (precision degrades
        but determinism preserved).
        """
        venues_payload: dict[str, list[dict[str, str | int]]] = {}
        for venue_id in sorted(bars_per_venue):
            venues_payload[venue_id] = [
                {
                    "ts": b.ts.isoformat(),
                    "close": str(b.close),
                    "volume": b.volume,
                }
                for b in sorted(bars_per_venue[venue_id], key=lambda x: x.ts)
            ]
        payload: dict[str, object] = {
            "contract": str(contract),
            "fix_date": fix_date.isoformat(),
            "venues": venues_payload,
        }
        json_bytes = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(json_bytes).hexdigest()


__all__: list[str] = [
    "BRRError",
    "BRRReconstruction",
    "BRRReconstructor",
    "BRRWindowEmpty",
    "InsufficientVenues",
]
