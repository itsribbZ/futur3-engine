"""futur3.data.source — DataSource ABC + exception hierarchy.

Per the data-layer design

**Hard seam** (CI-grep enforced): no code outside `futur3/data/sources/` imports
vendor SDKs directly. All vendor calls (`ib_async`, `ccxt`, `curl_cffi`, etc.) wrapped behind
DataSource subclasses. Engine accesses everything through this seam.

This eliminates bug class 2 (monkey-patching in tests) by construction — the verifier consumes
the ABC; nothing to monkey-patch at the engine layer.
"""

from __future__ import annotations

import abc
from collections.abc import Iterable
from datetime import datetime

from futur3.data.types import (
    BarResolution,
    ContractSymbol,
    RawBar,
    RawTick,
    Settle,
    SourceTier,
)

# ----------------------------------------------------------------------------
# Exception hierarchy
# ----------------------------------------------------------------------------


class DataSourceError(Exception):
    """Base exception for any DataSource subclass."""


class BarsNotSupported(DataSourceError):
    """Source does not publish bar data at this resolution.

    e.g., CMEEODDataSource has no intraday — only daily settle.
    """


class TicksNotSupported(DataSourceError):
    """Source does not publish tick-level data (most sources; Phase A1 default)."""


class SettlesNotSupported(DataSourceError):
    """Source does not publish settlement data (crypto venues don't; CME does)."""


class GeoBlockedError(DataSourceError):
    """Source is geo-blocked from current location.

    Per internal notes: Binance Global returns HTTP 451 from US IPs.
    """


class SchemaMismatch(DataSourceError):
    """Schema-header hash diverged from cached expected — bug class 9 detection."""


class FutureDatedSourceError(DataSourceError):
    """Source emitted record with as_of_iso > now() — bug class 5 (data-snooping) detection."""


class ContractStillActiveError(DataSourceError):
    """Final-settle queried before contract LTD passed — bug class 4 prevention (internal notes)."""


class ContractNotConfigured(DataSourceError):
    """Caller queried a contract whose URL/spec is not configured for this source.

    Catches typos + premature querying of contracts not yet in the 10-contract universe
    (scope-locked universe). Bug class 19 (configuration drift) prevention.
    """


# ----------------------------------------------------------------------------
# CME public settlements scraper (A1.3) exceptions
# ----------------------------------------------------------------------------


class CMEScrapeError(DataSourceError):
    """Parent for CME public-settlements-page scrape errors.

    Subclasses cover: WAF block (Cloudflare 403), malformed HTML, schema drift,
    rate-limit triggered, network failure. Reserved for the `CMEEODDataSource`
    fetch pipeline (internal notes).
    """


class WAFBlockedError(CMEScrapeError):
    """Cloudflare WAF challenge returned in place of settlement HTML.

    Detected via Cloudflare-specific markers in response (challenge page title,
    `cf-please-wait` element, 403 status). Mitigation: `curl_cffi` browser-
    fingerprint TLS (plain `requests` gets blocked).
    """


class MalformedSettlementPage(CMEScrapeError):
    """HTML response is structurally invalid for settlement parsing.

    Causes:
    - No `<table>` element (CME maintenance page or 404 served as 200)
    - `<tbody>` empty (no settlement rows — page rendered but data unavailable)
    - Required `<th>` headers missing entirely
    - Row cells don't match column count

    Distinguished from `SchemaMismatch` (headers PRESENT but RENAMED — bug class 9).
    """


# ----------------------------------------------------------------------------
# IBKR (A1.4) exceptions — historical-data via ib_async + IB Gateway
# ----------------------------------------------------------------------------


class IBKRError(DataSourceError):
    """Parent for IBKR-specific errors (gateway, rate-limit, contract, response)."""


class IBKRConnectionError(IBKRError):
    """IB Gateway unreachable or refused connection.

    Causes: Gateway not running, wrong port (4001 live vs 4002 paper), client ID conflict,
    daily 23:45 CT reset window, network failure.
    """


class IBKRReqError(IBKRError):
    """`reqHistoricalData` returned an error code or empty response.

    Distinguishes vendor-side failures (no permission, invalid contract spec, rate limit
    soft-throttle) from connection-level issues. Wraps the underlying ib_async exception.
    """


class IBKRContractAmbiguity(IBKRError):
    """ContractSymbol cannot be resolved to a unique `ib_async.Future` contract.

    Triggered when the futur3 contract symbol (e.g., ESM26) maps to multiple IBKR
    contracts (different exchanges, expirations, multipliers) and no disambiguator
    is configured. Bug class 19 prevention (config drift).
    """


# ----------------------------------------------------------------------------
# DataSource ABC
# ----------------------------------------------------------------------------


class DataSource(abc.ABC):
    """The single seam between futur3 and external data.

    Hard rule (CI-grep enforced): no code outside `futur3/data/sources/` may import
    vendor SDKs directly. All vendor calls (`ib_async`, `ccxt`, `requests-to-cmegroup`, etc.)
    wrapped behind a DataSource subclass.

    Subclasses MUST implement:
    - `source_id` property (stable string used in provenance hash)
    - `tier` property (SourceTier classification)
    - `get_bars()` (raise BarsNotSupported if N/A)

    Subclasses MAY override:
    - `get_ticks()` (default raises TicksNotSupported)
    - `latest_settle()` (default raises SettlesNotSupported)
    - `healthcheck()` (default returns True)
    """

    @property
    @abc.abstractmethod
    def source_id(self) -> str:
        """Stable string per source. Used as provenance-hash input.

        Examples:
        - "ibkr_tws_v10.30"
        - "cme_public_settlements"
        - "coinbase_advanced_v3"
        - "kraken_spot_rest"
        - "bullish_trading_api_v1"
        """
        ...

    @property
    @abc.abstractmethod
    def tier(self) -> SourceTier:
        """SourceTier classification (internal notes). Determines highest_tier_wins ordering."""
        ...

    @abc.abstractmethod
    def get_bars(
        self,
        contract: ContractSymbol,
        ts_start: datetime,
        ts_end: datetime,
        resolution: BarResolution,
    ) -> Iterable[RawBar]:
        """Yield bars in chronological order.

        Raises `BarsNotSupported` if source can't provide bars at this resolution
        (e.g., `CMEEODDataSource` only publishes daily settle, not intraday bars).

        Raises `SchemaMismatch` if upstream schema-header hash diverged.
        Raises `FutureDatedSourceError` if any returned bar has as_of_iso > now.
        Raises `GeoBlockedError` if source unreachable from current location.

        Implementation contract:
        - `ts_start` <= bar.ts < `ts_end` for every emitted bar (half-open interval)
        - bars emitted in strictly-increasing `ts` order
        - all bars carry IANA-TZ-aware datetimes (enforced in RawBar.__post_init__)
        - `content_bytes_sha` = SHA256 of the raw upstream bytes (for revision detection)
        """
        ...

    def get_ticks(
        self,
        contract: ContractSymbol,
        ts_start: datetime,
        ts_end: datetime,
    ) -> Iterable[RawTick]:
        """Yield ticks in chronological order. Default: tick data unsupported.

        Tick-publishing subclasses (crypto venue WS, IBKR realtime) override.
        """
        raise TicksNotSupported(self.source_id)

    def latest_settle(
        self,
        contract: ContractSymbol,
        as_of: datetime,
    ) -> Settle | None:
        """Return the settle for the business-day containing `as_of`.

        Returns `None` if settle is not yet published (e.g., querying mid-session for
        same-day settle that posts at 18:00 CT preliminary).

        Raises `SettlesNotSupported` if source doesn't publish settles (crypto venues, etc.).

        Subclasses that DO publish settles (CMEEODDataSource, IBKR daily bars) override.
        """
        raise SettlesNotSupported(self.source_id)

    def healthcheck(self) -> bool:
        """Lightweight liveness check.

        Used by `MultiSourceVerifier` to skip dark sources before bar-completion timeout fires.
        Default: returns True (always-alive). Subclasses with network state override.
        """
        return True

    def __repr__(self) -> str:
        return f"<{type(self).__name__} source_id={self.source_id!r} tier={self.tier.name}>"
