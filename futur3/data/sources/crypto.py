"""CcxtCryptoDataSource — Phase A1.10 base + concrete venue subclasses.

Spot crypto exchanges (Coinbase + Kraken + Bitstamp + Gemini + Binance.US oracle)
that publish public market data via REST. ccxt v4.x is the unified abstraction:
each venue's quirks (BTC vs XBT, "BTC/USD" vs "btcusd") are hidden behind ccxt's
standardized Exchange interface.

Per internal crypto-data notes:
- All 4 BRR-load-bearing venues (Coinbase + Kraken + Bitstamp + Gemini) are
  free-tier public REST + WebSocket — $0/mo (free-first).
- Binance.US serves as a 5th oracle for drift sanity-check (per
  internal design notes).
- BRR reconstruction (4-of-7 constituent average) is a separate step (A1.11),
  consumes these DataSources via the verifier.

## Architecture

```
ContractSymbol("BTCUSD")
    │
    ▼  per-subclass SYMBOL_MAP
CcxtCryptoDataSource.get_bars(...)
    │
    ├─► _resolve_ccxt_symbol     → "BTC/USD"  (ccxt canonical)
    ├─► _resolve_ccxt_timeframe  → "1m" / "5m" / "1d"
    ├─► _compute_since_ms        → ts_start UTC → epoch ms
    └─► CcxtExchange.fetch_ohlcv(...) → [[ts_ms, o, h, l, c, v], ...]
        └─► _ohlcv_row_to_raw_bar → RawBar (Decimal-via-str, TZ-aware UTC,
                                            content_bytes_sha = canonical JSON)
```

## Hard-seam pattern

`CcxtExchange` Protocol abstracts the (large, vendor-typed) `ccxt.Exchange`
class — futur3 code outside this module never sees `ccxt.coinbase()` instances
directly. `_DefaultCcxtExchange` wraps a real ccxt exchange with lazy import;
tests inject `FixtureCcxtExchange` instead. Per internal design notes +
the same seam pattern shipped at A1.4 for `ib_async`.

## Scope (A1.10a — base + Coinbase)

- `CcxtCryptoDataSource` base class.
- `CoinbaseCryptoDataSource` concrete subclass — BTC/USD + ETH/USD.
- `get_bars` works fully against fixture ccxt.
- `get_ticks` raises TicksNotSupported (A1.6+ WS streaming).
- `latest_settle` raises SettlesNotSupported (crypto spot has no settlement
  in the futures sense; BRR reconstruction is A1.11 and uses get_bars).

## Deferred to later steps

- **A1.10b/c/d/e**: Kraken + Bitstamp + Gemini + Binance.US subclasses
  (each ~30-50 LOC; same pattern).
- **A1.11**: BRR / ETHUSD_RR free-first reconstruction (4-of-7 constituent
  consensus at 16:00 London EOD).
- **WS tick streaming**: deferred to A1.6+ realtime layer.

References:
- internal crypto-data notes (per-venue API spec, 14 dimensions each)
- the data-layer design (Phase A1 step)
- the verifier spec (verifier consumes these sources)
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, ClassVar, Final, Protocol, runtime_checkable

from futur3.data.source import (
    ContractNotConfigured,
    DataSource,
    DataSourceError,
)
from futur3.data.types import (
    BarResolution,
    ContractSymbol,
    RawBar,
    SourceTier,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MILLISECONDS_PER_SECOND: Final[int] = 1_000

# OHLCV row index per ccxt convention: [timestamp_ms, open, high, low, close, volume]
_OHLCV_TS_IDX: Final[int] = 0
_OHLCV_OPEN_IDX: Final[int] = 1
_OHLCV_HIGH_IDX: Final[int] = 2
_OHLCV_LOW_IDX: Final[int] = 3
_OHLCV_CLOSE_IDX: Final[int] = 4
_OHLCV_VOLUME_IDX: Final[int] = 5
_OHLCV_EXPECTED_FIELDS: Final[int] = 6


# ---------------------------------------------------------------------------
# Protocol — CcxtExchange (hard seam)
# ---------------------------------------------------------------------------


@runtime_checkable
class CcxtExchange(Protocol):
    """Structural Protocol over ccxt's `Exchange` class.

    We declare ONLY the methods the futur3 crypto data layer calls. Real
    `ccxt.coinbase()` instances satisfy this via duck typing; tests inject
    `FixtureCcxtExchange` that satisfies the same shape with canned OHLCV
    payloads + configurable error injection.
    """

    @property
    def id(self) -> str:
        """Lower-case venue name (e.g., "coinbase", "kraken"). Used as
        SOURCE_ID + log tag."""
        ...

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        since: int | None = None,
        limit: int | None = None,
    ) -> list[list[float]]:
        """Fetch OHLCV bars.

        Returns a list of `[ts_ms, open, high, low, close, volume]` rows.
        - `symbol`: ccxt canonical (e.g., "BTC/USD").
        - `timeframe`: ccxt string ("1m", "5m", "15m", "1h", "1d").
        - `since`: epoch milliseconds (start of range; ccxt convention).
        - `limit`: max rows to return.
        """
        ...


# ---------------------------------------------------------------------------
# Default ccxt wrapper (lazy import)
# ---------------------------------------------------------------------------


class _DefaultCcxtExchange:
    """Default CcxtExchange wrapping a real `ccxt.{venue}()` instance.

    Lazy-imports ccxt on first construction. Engine code never instantiates
    this directly; subclasses of CcxtCryptoDataSource construct it via the
    `_build_default_exchange` factory.
    """

    def __init__(self, venue_name: str) -> None:
        # PLC0415 lazy import is intentional — defer ccxt cost until first use.
        # ccxt ships no py.typed marker; structural Protocol enforces shape downstream.
        import ccxt  # type: ignore[import-untyped]  # noqa: PLC0415

        if not hasattr(ccxt, venue_name):
            raise DataSourceError(
                f"ccxt has no exchange named {venue_name!r}; "
                f"available: {sorted(ccxt.exchanges)[:20]}..."
            )
        self._exchange: Any = getattr(ccxt, venue_name)()

    @property
    def id(self) -> str:
        return str(self._exchange.id)

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        since: int | None = None,
        limit: int | None = None,
    ) -> list[list[float]]:
        result: list[list[float]] = self._exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        return result


# ---------------------------------------------------------------------------
# CcxtCryptoDataSource base
# ---------------------------------------------------------------------------


class CcxtCryptoDataSource(DataSource):
    """Base for crypto spot venue DataSources backed by ccxt v4.x.

    Subclasses MUST override:
    - `VENUE_NAME`: ccxt module-level name (e.g., "coinbase").
    - `SOURCE_ID`: stable identifier for verifier-side provenance.
    - `SYMBOL_MAP`: ContractSymbol → ccxt canonical symbol (e.g.,
      `{"BTCUSD": "BTC/USD", "ETHUSD": "ETH/USD"}`).

    Inherited from this base:
    - `get_bars` REST OHLCV fetch via ccxt with Decimal precision + provenance.
    - `get_ticks` raises TicksNotSupported (A1.6 WS streaming layer).
    - `latest_settle` raises SettlesNotSupported (crypto spot has no settle).
    - `healthcheck` returns True (REST endpoints are stateless; we don't
      probe on each call — verifier policy will quarantine if fetches fail).
    """

    # Subclasses override these class-level constants
    VENUE_NAME: ClassVar[str] = ""
    SOURCE_ID: ClassVar[str] = ""
    SYMBOL_MAP: ClassVar[dict[str, str]] = {}

    # Per the data-layer design: BarResolution → ccxt timeframe string
    TIMEFRAME_MAP: ClassVar[dict[BarResolution, str]] = {
        BarResolution.MIN_1: "1m",
        BarResolution.MIN_5: "5m",
        BarResolution.MIN_15: "15m",
        BarResolution.HOUR_1: "1h",
        BarResolution.DAY_1: "1d",
    }

    def __init__(
        self,
        exchange: CcxtExchange | None = None,
    ) -> None:
        """Construct with an optional injected CcxtExchange.

        Tests inject FixtureCcxtExchange. Production code passes None →
        `_DefaultCcxtExchange(self.VENUE_NAME)` wraps the real ccxt instance.

        Subclasses MUST set VENUE_NAME + SOURCE_ID + SYMBOL_MAP before super().__init__.
        """
        if not self.VENUE_NAME:
            raise ContractNotConfigured(
                f"{type(self).__name__}: VENUE_NAME class attr must be set "
                f"(e.g., 'coinbase') before instantiation"
            )
        if not self.SOURCE_ID:
            raise ContractNotConfigured(f"{type(self).__name__}: SOURCE_ID class attr must be set")
        if not self.SYMBOL_MAP:
            raise ContractNotConfigured(
                f"{type(self).__name__}: SYMBOL_MAP class attr must be set "
                f"(ContractSymbol → ccxt symbol mapping)"
            )
        self._exchange: CcxtExchange = exchange or _DefaultCcxtExchange(self.VENUE_NAME)

    # ------------------------------------------------------------------------
    # DataSource ABC
    # ------------------------------------------------------------------------

    @property
    def source_id(self) -> str:
        return self.SOURCE_ID

    @property
    def tier(self) -> SourceTier:
        # Crypto spot venues are T2_EXCHANGE per internal design notes
        return SourceTier.T2_EXCHANGE

    def get_bars(
        self,
        contract: ContractSymbol,
        ts_start: datetime,
        ts_end: datetime,
        resolution: BarResolution,
    ) -> Iterable[RawBar]:
        """Fetch OHLCV bars for `contract` in [ts_start, ts_end) at `resolution`.

        Per DataSource ABC: ts_start <= bar.ts < ts_end (half-open). All emitted
        bars carry IANA-TZ-aware datetimes (UTC canonical).

        Raises:
            ContractNotConfigured: contract not in SYMBOL_MAP.
            ValueError: ts_start/ts_end naive or ts_end <= ts_start or
                resolution not in TIMEFRAME_MAP.
            DataSourceError: ccxt fetch raised (wrapped); malformed OHLCV row.
        """
        if ts_start.tzinfo is None:
            raise ValueError(f"{self.source_id}: ts_start must be TZ-aware; got naive {ts_start!r}")
        if ts_end.tzinfo is None:
            raise ValueError(f"{self.source_id}: ts_end must be TZ-aware; got naive {ts_end!r}")
        if ts_end <= ts_start:
            raise ValueError(
                f"{self.source_id}: ts_end {ts_end!r} must be after ts_start {ts_start!r}"
            )
        ccxt_symbol = self._resolve_ccxt_symbol(contract)
        timeframe = self._resolve_ccxt_timeframe(resolution)
        since_ms = self._compute_since_ms(ts_start)
        as_of_iso = datetime.now(UTC)

        try:
            ohlcv_rows = self._exchange.fetch_ohlcv(
                symbol=ccxt_symbol,
                timeframe=timeframe,
                since=since_ms,
                limit=None,
            )
        except DataSourceError:
            raise
        except Exception as e:
            raise DataSourceError(
                f"{self.source_id}: fetch_ohlcv failed for {contract} "
                f"[{ts_start} → {ts_end} @ {resolution.name}]: "
                f"{type(e).__name__}: {e}"
            ) from e

        return [
            self._ohlcv_row_to_raw_bar(
                row=row,
                contract=contract,
                resolution=resolution,
                as_of_iso=as_of_iso,
            )
            for row in ohlcv_rows
            if self._row_in_window(row, ts_start, ts_end)
        ]

    # get_ticks + latest_settle inherit defaults from DataSource ABC:
    # - get_ticks → TicksNotSupported
    # - latest_settle → SettlesNotSupported (crypto spot has no daily settlement)

    def healthcheck(self) -> bool:
        """Stateless: REST endpoints don't carry session state. True by default.

        Verifier policy quarantines the source if subsequent fetches fail
        (DataSourceError propagation). A dedicated probe ping would add latency
        without information gain at the shell level.
        """
        return True

    # ------------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------------

    def _resolve_ccxt_symbol(self, contract: ContractSymbol) -> str:
        s = str(contract)
        if s not in self.SYMBOL_MAP:
            raise ContractNotConfigured(
                f"{self.source_id}: contract {s!r} not in SYMBOL_MAP. "
                f"Configured: {sorted(self.SYMBOL_MAP)}"
            )
        return self.SYMBOL_MAP[s]

    def _resolve_ccxt_timeframe(self, resolution: BarResolution) -> str:
        if resolution not in self.TIMEFRAME_MAP:
            raise ValueError(
                f"{self.source_id}: BarResolution {resolution.name} not supported "
                f"by ccxt crypto venues; supported: {[r.name for r in self.TIMEFRAME_MAP]}"
            )
        return self.TIMEFRAME_MAP[resolution]

    @staticmethod
    def _compute_since_ms(ts_start: datetime) -> int:
        """TZ-aware datetime → ccxt `since` (epoch ms)."""
        return int(ts_start.astimezone(UTC).timestamp() * MILLISECONDS_PER_SECOND)

    @staticmethod
    def _row_in_window(
        row: list[float],
        ts_start: datetime,
        ts_end: datetime,
    ) -> bool:
        """ABC contract: ts_start <= bar.ts < ts_end (half-open)."""
        ts_ms = row[_OHLCV_TS_IDX]
        bar_ts = datetime.fromtimestamp(ts_ms / MILLISECONDS_PER_SECOND, tz=UTC)
        return ts_start <= bar_ts < ts_end

    def _ohlcv_row_to_raw_bar(
        self,
        *,
        row: list[float],
        contract: ContractSymbol,
        resolution: BarResolution,
        as_of_iso: datetime,
    ) -> RawBar:
        """Convert ccxt OHLCV row → RawBar with provenance hash.

        Per the TZ rules (bug class 7 + 14): TZ-aware UTC; Decimal-via-str (no IEEE-754
        leak); content_bytes_sha = SHA256 of canonical JSON (no raw wire bytes
        from ccxt — vendor returns structured list, not bytes).
        """
        if len(row) < _OHLCV_EXPECTED_FIELDS:
            raise DataSourceError(
                f"{self.source_id}: malformed OHLCV row (expected ≥6 fields, "
                f"got {len(row)}): {row!r}"
            )
        ts_ms = row[_OHLCV_TS_IDX]
        bar_ts = datetime.fromtimestamp(ts_ms / MILLISECONDS_PER_SECOND, tz=UTC)
        open_d = Decimal(str(row[_OHLCV_OPEN_IDX]))
        high_d = Decimal(str(row[_OHLCV_HIGH_IDX]))
        low_d = Decimal(str(row[_OHLCV_LOW_IDX]))
        close_d = Decimal(str(row[_OHLCV_CLOSE_IDX]))
        # ccxt volume is float (base-asset units). Convert to int for RawBar
        # via str round-trip (no float→int truncation surprises).
        volume_d = Decimal(str(row[_OHLCV_VOLUME_IDX]))
        volume = int(volume_d)

        payload = json.dumps(
            {
                "contract": str(contract),
                "resolution": resolution.value,
                "ts_ms": ts_ms,
                "open": str(open_d),
                "high": str(high_d),
                "low": str(low_d),
                "close": str(close_d),
                "volume_decimal": str(volume_d),
                "venue": self.VENUE_NAME,
            },
            sort_keys=True,
        ).encode("utf-8")
        content_sha = hashlib.sha256(payload).hexdigest()

        return RawBar(
            contract=contract,
            ts=bar_ts,
            resolution=resolution,
            open=open_d,
            high=high_d,
            low=low_d,
            close=close_d,
            volume=volume,
            oi=None,  # spot crypto has no OI
            source_id=self.SOURCE_ID,
            as_of_iso=as_of_iso,
            content_bytes_sha=content_sha,
        )


# ---------------------------------------------------------------------------
# Concrete venue subclass — Coinbase Advanced Trade
# ---------------------------------------------------------------------------


class CoinbaseCryptoDataSource(CcxtCryptoDataSource):
    """Coinbase Advanced Trade public market data via ccxt.

    Per internal design notes:
    - Public REST + WebSocket; no auth for market data.
    - 24/7 since 2015 BTC, 2016 ETH; minute data effectively unbounded.
    - "BTC-USD" / "ETH-USD" product IDs map to ccxt canonical "BTC/USD" / "ETH/USD".
    - Tier T2_EXCHANGE; BRR-load-bearing constituent (1-of-4).
    """

    VENUE_NAME: ClassVar[str] = "coinbase"
    SOURCE_ID: ClassVar[str] = "coinbase_advanced"
    SYMBOL_MAP: ClassVar[dict[str, str]] = {
        "BTCUSD": "BTC/USD",
        "ETHUSD": "ETH/USD",
    }


class KrakenCryptoDataSource(CcxtCryptoDataSource):
    """Kraken Spot public market data via ccxt.

    Per internal design notes:
    - Public REST + WebSocket; no auth for market data.
    - Raw Kraken pair codes use "XBT" (not "BTC") for Bitcoin — historical
      XBT/CHF convention. ccxt v4.x normalizes to canonical "BTC/USD".
    - Free quarterly OHLCVT CSV bundles available for deep history (since
      2013 BTC, 2015 ETH). Phase A1 uses REST polling; bulk backfill is
      A1.11+ enhancement.
    - Tier T2_EXCHANGE; BRR-load-bearing constituent (2-of-4).
    """

    VENUE_NAME: ClassVar[str] = "kraken"
    SOURCE_ID: ClassVar[str] = "kraken_spot"
    SYMBOL_MAP: ClassVar[dict[str, str]] = {
        "BTCUSD": "BTC/USD",
        "ETHUSD": "ETH/USD",
    }


class BitstampCryptoDataSource(CcxtCryptoDataSource):
    """Bitstamp public market data via ccxt.

    Per internal design notes:
    - Public REST + WebSocket; no auth for market data.
    - Microsecond trade timestamp resolution (highest among 4 BRR venues —
      load-bearing for BRR precision).
    - Raw Bitstamp pair codes are lowercase ("btcusd"); ccxt normalizes to
      canonical "BTC/USD".
    - 400 req/sec quota per client — most generous of the 4 venues here.
    - Tier T2_EXCHANGE; BRR-load-bearing constituent (3-of-4).
    """

    VENUE_NAME: ClassVar[str] = "bitstamp"
    SOURCE_ID: ClassVar[str] = "bitstamp_spot"
    SYMBOL_MAP: ClassVar[dict[str, str]] = {
        "BTCUSD": "BTC/USD",
        "ETHUSD": "ETH/USD",
    }


class GeminiCryptoDataSource(CcxtCryptoDataSource):
    """Gemini public market data via ccxt.

    Per internal design notes (Gemini section):
    - Public REST + WebSocket; no auth for market data.
    - US-regulated (NYDFS); BRR-load-bearing constituent.
    - Trade history capped at recent ~7 days via REST — WS persistent capture
      is the long-term-history path (A1.6+).
    - Tier T2_EXCHANGE; BRR-load-bearing constituent (4-of-4).
    """

    VENUE_NAME: ClassVar[str] = "gemini"
    SOURCE_ID: ClassVar[str] = "gemini_spot"
    SYMBOL_MAP: ClassVar[dict[str, str]] = {
        "BTCUSD": "BTC/USD",
        "ETHUSD": "ETH/USD",
    }


class BinanceUSCryptoDataSource(CcxtCryptoDataSource):
    """Binance.US public market data via ccxt.

    Per internal design notes (Binance section):
    - Binance.com is HTTP-451 geo-blocked from US IPs (NOT usable — geo-blocked).
    - Binance.US (the US-licensed entity) IS accessible from US IPs and
      serves as a **drift sanity-check oracle**, NOT a BRR constituent.
    - Tier T2_EXCHANGE; oracle role for cross-venue drift monitoring (A1.11+).

    Use case: verifier consumes Coinbase + Kraken + Bitstamp + Gemini as the
    4 BRR constituents and Binance.US as a 5th oracle source that should
    track within the BRR drift envelope (1-5 bp typical).
    """

    VENUE_NAME: ClassVar[str] = "binanceus"
    SOURCE_ID: ClassVar[str] = "binanceus_oracle"
    SYMBOL_MAP: ClassVar[dict[str, str]] = {
        "BTCUSD": "BTC/USD",
        "ETHUSD": "ETH/USD",
    }
