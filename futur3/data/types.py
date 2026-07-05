"""futur3.data.types — frozen dataclasses for the data layer.

Schema LOCKED per the data-layer design + the verifier spec.
**Field-additive only forever** — never rename/remove fields; only add optional fields.

Core invariants:
- Decimal for all prices/qty/volumes (never float — bug class avoidance)
- IANA-TZ-aware datetimes at the boundary (naive datetimes raise)
- frozen=True for immutability + hashability
- SHA256 provenance (`content_bytes_sha`) on every record for bit-repro
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum, IntEnum
from typing import Final, Literal

# Hex-encoded SHA256 digest is always 64 characters (32 bytes × 2 hex)
SHA256_HEX_LENGTH: Final[int] = 64

# ----------------------------------------------------------------------------
# Enums
# ----------------------------------------------------------------------------


class SourceTier(IntEnum):
    """Lower rank = higher trust. Used by `highest_tier_wins` verifier policy.

    See the data-layer design for tier definitions + when to use.
    """

    T1_EXCHANGE_HISTORICAL = 1  # CME DataMine (excluded by cost policy but tier reserved)
    T2_EXCHANGE = 2  # CME public settlements scraper
    T2_BROKER = 3  # IBKR realtime/historical
    T2_MACRO = 4  # FRED (cash-vs-futures basis only)
    T3_AGGREGATOR = 5  # Yahoo, Barchart, Quandl/NDL
    T4_DERIVED = 6  # ReplayDataSource, scraper-archive


class BarResolution(Enum):
    """Bar boundary granularity. Stored as canonical string for stable hashing."""

    SEC_1 = "1s"
    SEC_5 = "5s"
    MIN_1 = "1m"
    MIN_5 = "5m"
    MIN_15 = "15m"
    HOUR_1 = "1h"
    DAY_1 = "1d"
    SETTLE = "settle"


class Side(Enum):
    BUY = "buy"
    SELL = "sell"


SettleState = Literal["live", "preliminary", "final"]
"""Settle progression: live (intraday); preliminary (post-CME 18:00 CT); final (next morning)."""


# ----------------------------------------------------------------------------
# Identifiers
# ----------------------------------------------------------------------------


class ContractSymbol(str):
    """Canonical futures contract symbol.

    Format: `<root><month_code><year_2digit>` (e.g., `ESM26` = ES June 2026).
    2-digit-year disambiguation (internal notes): 00-49 → 2000+; 50-99 → 1900+;
    rotates 2050 (no contracts that old; safe through ~2049).

    CME month codes: F=Jan G=Feb H=Mar J=Apr K=May M=Jun N=Jul Q=Aug U=Sep V=Oct X=Nov Z=Dec.
    """

    __slots__ = ()


# ----------------------------------------------------------------------------
# Bar / tick / settle records
# ----------------------------------------------------------------------------


def _assert_tz_aware(ts: datetime, field_name: str) -> None:
    """A7 bug-class invariant: naive datetime is a bug class 7 (timezone confusion) red flag.

    Verifier rejects any source emitting naive datetime at the DataSource boundary.
    """
    if ts.tzinfo is None:
        raise ValueError(f"{field_name} must be IANA-TZ-aware datetime; got naive {ts!r}")


@dataclass(frozen=True)
class RawBar:
    """Single bar from a single source PRE-verification.

    The `MultiSourceVerifier` consumes N RawBars (one per source) and emits a `VerifiedBar`
    (defined in `futur3.data.verifier`) after cross-source consensus.

    Fields locked — field-additive only.
    """

    contract: ContractSymbol
    ts: datetime  # IANA-TZ-aware (UTC canonical at boundary)
    resolution: BarResolution
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    oi: int | None  # source may not publish; CME publishes prior-day OI only
    source_id: str  # stable per source (e.g., "ibkr_tws_v10.30")
    as_of_iso: datetime  # when fetched (UTC)
    content_bytes_sha: str  # SHA256 of raw upstream bytes pre-parse

    def __post_init__(self) -> None:
        _assert_tz_aware(self.ts, "RawBar.ts")
        _assert_tz_aware(self.as_of_iso, "RawBar.as_of_iso")
        if self.high < self.low:
            raise ValueError(
                f"RawBar.high {self.high} < low {self.low} for {self.contract} @ {self.ts}"
            )
        if self.volume < 0:
            raise ValueError(f"RawBar.volume must be >= 0; got {self.volume}")
        if self.oi is not None and self.oi < 0:
            raise ValueError(f"RawBar.oi must be >= 0 if set; got {self.oi}")
        if len(self.content_bytes_sha) != SHA256_HEX_LENGTH:
            raise ValueError(
                f"RawBar.content_bytes_sha must be hex-SHA256 ({SHA256_HEX_LENGTH} chars); "
                f"got len={len(self.content_bytes_sha)}"
            )


@dataclass(frozen=True)
class Settle:
    """Daily settlement record from a single source.

    Per internal notes, schema mirrors CME public settlement-page columns.

    Critical: `oi_prior` is PRIOR-DAY OI (CME publishes prior-day only). Mis-binding to
    `oi_today` is bug class 4 (look-ahead). Field name makes this explicit.
    """

    contract: ContractSymbol
    as_of_date: date  # business date this settle applies to
    settle: Decimal  # PRIMARY anchor
    settle_state: SettleState
    open: Decimal
    high: Decimal
    low: Decimal
    last: Decimal  # last-traded price; NOT necessarily == settle
    change: Decimal  # settle - prior_day_settle
    volume_est: int  # preliminary until next-morning final
    oi_prior: int  # PRIOR-day OI — load-bearing naming
    source_id: str = "cme_public_settlements"
    as_of_iso: datetime | None = None
    content_bytes_sha: str = ""  # SHA256 of raw HTML/CSV pre-parse
    cme_month_code: str = ""  # M=Jun, U=Sep, Z=Dec, H=Mar etc.

    def __post_init__(self) -> None:
        if self.as_of_iso is not None:
            _assert_tz_aware(self.as_of_iso, "Settle.as_of_iso")
        if self.high < self.low:
            raise ValueError(
                f"Settle.high {self.high} < low {self.low} for {self.contract} @ {self.as_of_date}"
            )
        if self.volume_est < 0:
            raise ValueError(f"Settle.volume_est must be >= 0; got {self.volume_est}")
        if self.oi_prior < 0:
            raise ValueError(f"Settle.oi_prior must be >= 0; got {self.oi_prior}")
        if self.content_bytes_sha and len(self.content_bytes_sha) != SHA256_HEX_LENGTH:
            raise ValueError(
                f"Settle.content_bytes_sha must be hex-SHA256 ({SHA256_HEX_LENGTH} chars) if set; "
                f"got len={len(self.content_bytes_sha)}"
            )


@dataclass(frozen=True)
class RawTick:
    """Single tick from a single source PRE-verification.

    Tick-level data only for sources that publish it (per-trade granularity).
    Most futures sources only publish bars; ticks come from broker direct feeds or
    crypto WS streams.
    """

    contract: ContractSymbol
    ts: datetime
    price: Decimal
    size: int
    side: Literal["bid", "ask", "trade"] | None
    source_id: str
    as_of_iso: datetime
    content_bytes_sha: str

    def __post_init__(self) -> None:
        _assert_tz_aware(self.ts, "RawTick.ts")
        _assert_tz_aware(self.as_of_iso, "RawTick.as_of_iso")
        if self.size <= 0:
            raise ValueError(f"RawTick.size must be > 0; got {self.size}")
        if self.price <= 0:
            raise ValueError(f"RawTick.price must be > 0; got {self.price}")
        if len(self.content_bytes_sha) != SHA256_HEX_LENGTH:
            raise ValueError(
                f"RawTick.content_bytes_sha must be hex-SHA256 ({SHA256_HEX_LENGTH} chars); "
                f"got len={len(self.content_bytes_sha)}"
            )


# ----------------------------------------------------------------------------
# Provenance helpers (internal notes)
# ----------------------------------------------------------------------------


def content_sha256(payload: bytes) -> str:
    """Hex-SHA256 of raw upstream bytes. Used for content_bytes_sha + provenance hash chain.

    The hash is the verifier's apparatus for detecting bug class 9 (silent revision).
    Stable across Python versions + platforms (cryptographic property of SHA256).
    """
    return hashlib.sha256(payload).hexdigest()


def source_provenance_hash(source_id: str, as_of_iso: datetime, content_bytes_sha: str) -> str:
    """SHA256(source_id || as_of_iso || content_bytes_sha) per internal notes

    The verifier composes N source_provenance_hashes into the bar's verifier_run_hash.
    Deterministic ordering required (network call order non-deterministic).
    """
    payload = f"{source_id}||{as_of_iso.isoformat()}||{content_bytes_sha}".encode()
    return hashlib.sha256(payload).hexdigest()
