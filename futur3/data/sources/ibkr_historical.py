"""IBKRHistoricalDataSource — historical bar data via ib_async + IB Gateway.

Phase A1.4 SKELETON per internal design notes. Provides
historical futures bars across the 10-contract futur3 universe (CME / NYMEX / COMEX)
through the ib_async 2.0+ sync API + BackoffQueue rate-limit enforcement.

## Architecture

```
ContractSymbol("ESM26")
    │
    ▼
IBKRHistoricalDataSource.get_bars(...)
    │
    ├─► _build_ib_contract  → ib_async.Future(symbol="ES", lastTradeDate="202606", exchange="CME")
    ├─► _map_bar_size       → "1 min" / "5 mins" / "1 day"
    ├─► _compute_duration   → "300 S" / "1 D"
    ├─► BackoffQueue.acquire(contract_key) — enforces 60/10min + 5/2s + 50-concurrent
    └─► IBClient.req_historical_data(...) → list[BarData] → list[RawBar]
```

## Critical invariants

- **Connection lifecycle**: `_ensure_connected` lazily connects on first use; idempotent
  thereafter. Tolerates the daily 23:45 CT IBKR session reset by surfacing
  `IBKRConnectionError` for caller-level retry (A1.7 will add automatic reconnect).
- **`useRTH=0` default** locked by operator decision (capture overnight
  ETH; strategies needing RTH-only filter at engine layer).
- **Decimal coercion via str()**: floats from ib_async go through `Decimal(str(float))`
  to preserve precision (never direct `Decimal(float)` — IEEE-754 leak).
- **TZ-aware always**: every emitted RawBar carries IANA-TZ-aware datetime (RawBar
  __post_init__ enforces; we coerce IB's date to UTC explicitly).
- **No vendor types leak**: `IBClient` protocol abstracts ib_async types — engine + verifier
  never see `ib_async.BarData` or `ib_async.Future`. Hard seam per the data-layer design.
- **BackoffQueue enforces rate caps BEFORE every request** — 60/10min + 5/2s + 50-concurrent.

## Out-of-scope for A1.4/A1.5 (deferred to later steps)

- **Multi-year chunked backfill** (A1.6+): durationStr has per-bar-size caps per
  internal notes; multi-year stitching needs orchestration logic.
- **Tick data** (`reqHistoricalTicks`) — `get_ticks` raises `TicksNotSupported`; A1.6 streaming.
- **Realtime streaming** (`keepUpToDate=True`) — A1.6.
- **Daily 23:45 CT auto-reconnect** — A1.7.
- **Continuous-contract `CONTFUT`** — Phase A1.20+ ContinuousContract construction layer.

## A1.5 — `latest_settle` from daily-bar close (SHIPPED)

`latest_settle(contract, as_of)` requests `durationStr="2 D"` + `barSize="1 day"`
and converts the most-recent daily bar (≤ `as_of`) into a `Settle`. Settlement
state is heuristically derived from elapsed-since-session-close in CT:
- elapsed < 18h (Mon-Thu) or < 42h (Fri) → `preliminary`
- elapsed ≥ threshold → `final`

The Friday-extended lag reflects the documented IBKR behavior: "Friday settlement price will
sometimes not be available until Saturday." `change` field is filled from the
prior daily bar (`latest.close - prior.close`); `oi_prior=0` because IBKR daily
bars don't publish OI (verifier policy in A1.9 will mark as structural-zero).

References:
- internal design notes (IBKR primary spec)
- the data-layer design (Phase A1 priority order)
- futur3/data/sources/backoff_queue.py (rate-limit enforcer)
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any, ClassVar, Final, Protocol, runtime_checkable
from zoneinfo import ZoneInfo

from futur3.data.source import (
    BarsNotSupported,
    ContractNotConfigured,
    DataSource,
    IBKRConnectionError,
    IBKRReqError,
    TicksNotSupported,
)
from futur3.data.sources.backoff_queue import BackoffQueue
from futur3.data.types import (
    BarResolution,
    ContractSymbol,
    RawBar,
    RawTick,
    Settle,
    SettleState,
    SourceTier,
)

logger = logging.getLogger(__name__)

# Module-level constants
MIN_CONTRACT_SYMBOL_LENGTH: Final[int] = 4
SECONDS_PER_DAY: Final[int] = 86_400
SECONDS_PER_HOUR: Final[int] = 3_600
DEFAULT_HOST: Final[str] = "127.0.0.1"
DEFAULT_PORT_PAPER: Final[int] = 4002  # IB Gateway paper-trading port
DEFAULT_PORT_LIVE: Final[int] = 4001
DEFAULT_CLIENT_ID: Final[int] = 1
# Y2K-style 2-digit-year pivot per ContractSymbol convention (types.py):
# 00-49 → 2000+; 50-99 → 1900+. Rotates at 2050 — safe through ~2049.
Y2K_CENTURY_PIVOT_YEAR: Final[int] = 50

# A1.5 — settlement-publication lag (heuristic for settle_state classification).
# Per internal notes: 'The Friday settlement price will sometimes not be available
# until Saturday.' We translate IBKR's lack-of-explicit-marker into a conservative
# time-since-session-close test:
#   - Mon-Thu session closes ~16:00 CT; CME publishes final ~next morning  → 18h conservative
#   - Fri session closes ~16:00 CT; CME publishes final ~Sat AM            → 42h conservative
# Inside the lag window → settle_state='preliminary'; after → 'final'.
PRELIMINARY_PUBLISH_LAG_HOURS_WEEKDAY: Final[int] = 18
PRELIMINARY_PUBLISH_LAG_HOURS_FRIDAY: Final[int] = 42
# Globex sessions close at 16:00 CT. We use IANA "America/Chicago" so the lag
# heuristic survives DST transitions (CME is in CT).
CME_SESSION_TZ: Final[ZoneInfo] = ZoneInfo("America/Chicago")
# Python weekday() index for Friday (Mon=0..Sun=6).
FRIDAY_WEEKDAY_INDEX: Final[int] = 4
# Min bars-list length to populate the Settle's `change` field (latest - prior).
LATEST_AND_PRIOR_BAR_COUNT: Final[int] = 2


# ============================================================================
# IB client protocol (injectable for testability)
# ============================================================================


@runtime_checkable
class IBClient(Protocol):
    """Lightweight protocol over ib_async — hides vendor SDK from engine.

    Production impl wraps `ib_async.IB`. Test impl returns canned bar lists.
    No code outside `futur3.data.sources.*` should reference `ib_async` directly.
    """

    def connect(self, host: str, port: int, client_id: int) -> None:
        """Connect to IB Gateway. Raises if Gateway unreachable."""
        ...

    def disconnect(self) -> None:
        """Disconnect cleanly. No-op if already disconnected."""
        ...

    def is_connected(self) -> bool:
        """True if connected to IB Gateway."""
        ...

    def build_future_contract(
        self,
        symbol: str,
        last_trade_date: str,
        exchange: str,
    ) -> Any:
        """Construct an opaque IB Future contract object.

        Args:
            symbol: Root symbol (e.g., "ES", "CL", "MBT").
            last_trade_date: YYYYMM string (e.g., "202606").
            exchange: IBKR exchange code ("CME" / "NYMEX" / "COMEX").

        Returns:
            Opaque object passed back to `req_historical_data`. In production this is
            `ib_async.Future`. In tests this is a fixture-defined dataclass.
        """
        ...

    def req_historical_data(
        self,
        contract: Any,
        end_datetime: str,
        duration: str,
        bar_size: str,
        what_to_show: str,
        use_rth: int,
    ) -> list[Any]:
        """Fetch historical bars. Returns a list of opaque BarData objects.

        Raises an arbitrary exception on vendor-side failure (we wrap as IBKRReqError).
        """
        ...


@runtime_checkable
class ClockProtocol(Protocol):
    """UTC clock — injected for deterministic `as_of_iso` in tests."""

    def now_utc(self) -> datetime: ...


class _DefaultIBClient:
    """Default IBClient wrapping `ib_async.IB()` with lazy import."""

    def __init__(self) -> None:
        self._ib: Any = None  # ib_async.IB; lazy

    def connect(self, host: str, port: int, client_id: int) -> None:
        if self._ib is None:
            from ib_async import IB, util

            # ib_async's SYNC API drives the event loop internally; called from inside the runner's
            # asyncio.run loop it raises "event loop is already running". patchAsyncio() applies
            # nest_asyncio so the sync API is re-entrant-safe (the standard ib_async mechanism for a
            # running loop; idempotent; never reached on the fake-client test path).
            util.patchAsyncio()  # type: ignore[no-untyped-call]  # ib_async ships no stub here
            self._ib = IB()
        try:
            self._ib.connect(host, port, clientId=client_id)
        except Exception as e:
            raise IBKRConnectionError(
                f"IB Gateway connect failed at {host}:{port} clientId={client_id}: "
                f"{type(e).__name__}: {e}"
            ) from e

    def disconnect(self) -> None:
        if self._ib is not None:
            try:
                self._ib.disconnect()
            except Exception as e:
                logger.debug("IB disconnect raised (non-fatal): %s", e)

    def is_connected(self) -> bool:
        if self._ib is None:
            return False
        try:
            return bool(self._ib.isConnected())
        except Exception:
            return False

    def build_future_contract(
        self,
        symbol: str,
        last_trade_date: str,
        exchange: str,
    ) -> Any:
        from ib_async import Future

        return Future(
            symbol=symbol,
            lastTradeDateOrContractMonth=last_trade_date,
            exchange=exchange,
        )

    def req_historical_data(
        self,
        contract: Any,
        end_datetime: str,
        duration: str,
        bar_size: str,
        what_to_show: str,
        use_rth: int,
    ) -> list[Any]:
        if self._ib is None:
            raise IBKRConnectionError(
                "IBClient not connected — call connect() before req_historical_data"
            )
        result: list[Any] = self._ib.reqHistoricalData(
            contract,
            endDateTime=end_datetime,
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=use_rth,
            formatDate=2,  # epoch UTC; eliminates TZ parsing ambiguity
        )
        return result


class _SystemClock:
    """Default UTC clock (TZ-aware)."""

    def now_utc(self) -> datetime:
        return datetime.now(UTC)


# ============================================================================
# Main class
# ============================================================================


class IBKRHistoricalDataSource(DataSource):
    """IBKR historical-data scraper (10-contract universe, T2_BROKER tier).

    See module docstring for architecture overview and invariants.
    """

    SOURCE_ID: ClassVar[str] = "ibkr_tws_historical"

    # Per internal notes — exchange routing for each contract root
    EXCHANGE_BY_ROOT: ClassVar[dict[str, str]] = {
        "ES": "CME",
        "MES": "CME",
        "NQ": "CME",
        "MNQ": "CME",
        "CL": "NYMEX",
        "MCL": "NYMEX",
        "GC": "COMEX",
        "MGC": "COMEX",
        "MBT": "CME",
        "MET": "CME",
        # CBOT 30y T-bond (ZB) -- the one CBOT-exchange contract in the pull universe.
        "ZB": "CBOT",
    }

    # CME month code → numeric month (matches CMEEODDataSource.MONTH_NAME_TO_CODE)
    MONTH_CODE_TO_NUMBER: ClassVar[dict[str, int]] = {
        "F": 1,
        "G": 2,
        "H": 3,
        "J": 4,
        "K": 5,
        "M": 6,
        "N": 7,
        "Q": 8,
        "U": 9,
        "V": 10,
        "X": 11,
        "Z": 12,
    }
    VALID_MONTH_CODES: ClassVar[frozenset[str]] = frozenset(MONTH_CODE_TO_NUMBER)

    # Per internal notes — futur3 BarResolution → IBKR barSizeSetting string
    BAR_SIZE_MAP: ClassVar[dict[BarResolution, str]] = {
        BarResolution.SEC_1: "1 secs",
        BarResolution.SEC_5: "5 secs",
        BarResolution.MIN_1: "1 min",
        BarResolution.MIN_5: "5 mins",
        BarResolution.MIN_15: "15 mins",
        BarResolution.HOUR_1: "1 hour",
        BarResolution.DAY_1: "1 day",
        # SETTLE is not a historical-bar size — raises BarsNotSupported
    }

    def __init__(
        self,
        ib_client: IBClient | None = None,
        backoff_queue: BackoffQueue | None = None,
        clock: ClockProtocol | None = None,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT_PAPER,
        client_id: int = DEFAULT_CLIENT_ID,
        use_rth: bool = False,  # default: capture overnight ETH
    ) -> None:
        self._ib: IBClient = ib_client or _DefaultIBClient()
        self._queue: BackoffQueue = backoff_queue or BackoffQueue()
        self._clock: ClockProtocol = clock or _SystemClock()
        self._host = host
        self._port = port
        self._client_id = client_id
        self._use_rth = use_rth

    # ------------------------------------------------------------------------
    # DataSource ABC contract
    # ------------------------------------------------------------------------

    @property
    def source_id(self) -> str:
        return self.SOURCE_ID

    @property
    def tier(self) -> SourceTier:
        return SourceTier.T2_BROKER

    def get_bars(
        self,
        contract: ContractSymbol,
        ts_start: datetime,
        ts_end: datetime,
        resolution: BarResolution,
    ) -> Iterable[RawBar]:
        """Fetch historical bars for `contract` in [ts_start, ts_end) at `resolution`.

        Raises:
            ContractNotConfigured: contract root not in EXCHANGE_BY_ROOT or bad format
            BarsNotSupported: BarResolution not in BAR_SIZE_MAP (e.g., SETTLE)
            IBKRConnectionError: IB Gateway unreachable
            IBKRReqError: req_historical_data raised or returned empty
            ValueError: ts_end <= ts_start
        """
        if ts_end <= ts_start:
            raise ValueError(
                f"{self.source_id}: ts_end {ts_end!r} must be after ts_start {ts_start!r}"
            )
        bar_size = self._map_bar_size(resolution)
        ib_contract = self._build_ib_contract(contract)
        duration = self._compute_duration_str(ts_start, ts_end)
        end_datetime = self._format_end_datetime(ts_end)

        self._ensure_connected()

        contract_key = self._contract_key(contract, "TRADES")
        as_of_iso = self._clock.now_utc()

        with self._queue.acquire(contract_key):
            try:
                ib_bars = self._ib.req_historical_data(
                    contract=ib_contract,
                    end_datetime=end_datetime,
                    duration=duration,
                    bar_size=bar_size,
                    what_to_show="TRADES",
                    use_rth=int(self._use_rth),
                )
            except IBKRConnectionError:
                raise
            except Exception as e:
                raise IBKRReqError(
                    f"{self.source_id}: reqHistoricalData failed for {contract} "
                    f"[{ts_start} → {ts_end} @ {resolution.name}]: "
                    f"{type(e).__name__}: {e}"
                ) from e

        return [self._ib_bar_to_raw(b, contract, resolution, as_of_iso) for b in ib_bars]

    def get_ticks(
        self,
        contract: ContractSymbol,
        ts_start: datetime,
        ts_end: datetime,
    ) -> Iterable[RawTick]:
        """Not supported in A1.4 — tick streaming deferred to A1.6."""
        raise TicksNotSupported(
            f"{self.source_id}: reqHistoricalTicks integration deferred to A1.6 streaming layer"
        )

    def latest_settle(
        self,
        contract: ContractSymbol,
        as_of: datetime,
    ) -> Settle | None:
        """Return the latest daily-bar-derived settle for `contract` at or before `as_of`.

        Per internal notes: IBKR's daily-bar close IS the official settlement once
        published. IBKR exposes no preliminary/final marker, so settle_state is
        derived heuristically from elapsed time since the session-close in CT
        (`_compute_settle_state`).

        We request `durationStr="2 D"` so a prior-day bar is also available; the
        Settle's `change` field is `latest.close - prior.close`. If only one bar
        is available (first session of contract life, etc.), `change = Decimal(0)`.

        Args:
            contract: ContractSymbol (e.g., ESM26).
            as_of: TZ-aware reference time. The latest daily bar with timestamp
                ≤ as_of is selected. Naive datetimes raise (fail-loud).

        Returns:
            Settle for the most recent session at or before `as_of`; or None if
            IBKR has no daily bars for the contract on/before that date (e.g.,
            pre-contract-listing, contract expired without `includeExpired=True`).

        Raises:
            ContractNotConfigured: contract root not in EXCHANGE_BY_ROOT, or
                malformed contract symbol.
            IBKRConnectionError: IB Gateway unreachable.
            IBKRReqError: reqHistoricalData raised on vendor side.
            ValueError: `as_of` is naive (must be TZ-aware).
        """
        if as_of.tzinfo is None:
            raise ValueError(
                f"{self.source_id}: latest_settle as_of must be TZ-aware; got naive {as_of!r}"
            )
        as_of_utc = as_of.astimezone(UTC)

        ib_contract = self._build_ib_contract(contract)
        end_dt_str = self._format_end_datetime(as_of_utc)

        self._ensure_connected()

        contract_key = self._contract_key(contract, "TRADES")

        with self._queue.acquire(contract_key):
            try:
                ib_bars = self._ib.req_historical_data(
                    contract=ib_contract,
                    end_datetime=end_dt_str,
                    duration="2 D",  # 2 days → latest + prior for change diff
                    bar_size="1 day",
                    what_to_show="TRADES",
                    use_rth=int(self._use_rth),
                )
            except IBKRConnectionError:
                raise
            except Exception as e:
                raise IBKRReqError(
                    f"{self.source_id}: latest_settle reqHistoricalData failed for "
                    f"{contract} as_of={as_of_utc.isoformat()}: "
                    f"{type(e).__name__}: {e}"
                ) from e

        if not ib_bars:
            return None

        # Sort ascending by date (IBKR convention is ascending; defensive sort).
        ib_bars_sorted = sorted(ib_bars, key=lambda b: self._coerce_ts_to_utc(b.date))
        latest = ib_bars_sorted[-1]
        prior = ib_bars_sorted[-2] if len(ib_bars_sorted) >= LATEST_AND_PRIOR_BAR_COUNT else None

        return self._ib_bar_to_settle(
            ib_bar=latest,
            prior_bar=prior,
            contract=contract,
            as_of_utc=as_of_utc,
        )

    def healthcheck(self) -> bool:
        """Returns True if IB Gateway connection is live."""
        try:
            return self._ib.is_connected()
        except Exception as e:
            logger.debug("IBKR healthcheck error: %s", e)
            return False

    # ------------------------------------------------------------------------
    # Public API beyond ABC
    # ------------------------------------------------------------------------

    def disconnect(self) -> None:
        """Explicit disconnect — callers should invoke at shutdown.

        Connection lifecycle is otherwise lazy (auto-connect on first get_bars).
        """
        self._ib.disconnect()

    # ------------------------------------------------------------------------
    # Private: contract translation
    # ------------------------------------------------------------------------

    def _parse_contract_symbol(self, contract: ContractSymbol) -> tuple[str, str, int]:
        """ESM26 → ('ES', 'M', 26). Validates root in EXCHANGE_BY_ROOT."""
        s = str(contract)
        if len(s) < MIN_CONTRACT_SYMBOL_LENGTH:
            raise ContractNotConfigured(
                f"{self.source_id}: contract {s!r} too short; expected <ROOT><MONTH_CODE><YY>"
            )
        year_str = s[-2:]
        if not year_str.isdigit():
            raise ContractNotConfigured(
                f"{self.source_id}: contract {s!r} year-suffix {year_str!r} not 2 digits"
            )
        year_2dig = int(year_str)
        month_code = s[-3]
        if month_code not in self.VALID_MONTH_CODES:
            raise ContractNotConfigured(
                f"{self.source_id}: contract {s!r} invalid month code {month_code!r}; "
                f"valid CME codes: {sorted(self.VALID_MONTH_CODES)}"
            )
        root = s[:-3]
        if root not in self.EXCHANGE_BY_ROOT:
            raise ContractNotConfigured(
                f"{self.source_id}: contract {s!r} root {root!r} not in EXCHANGE_BY_ROOT. "
                f"Configured: {sorted(self.EXCHANGE_BY_ROOT)}"
            )
        return root, month_code, year_2dig

    def _build_ib_contract(self, contract: ContractSymbol) -> Any:
        """ContractSymbol → ib_async.Future via IBClient."""
        root, month_code, year_2dig = self._parse_contract_symbol(contract)
        month_num = self.MONTH_CODE_TO_NUMBER[month_code]
        # Year disambiguation per types.py ContractSymbol: 00-49 → 2000+; 50-99 → 1900+
        century = 2000 if year_2dig < Y2K_CENTURY_PIVOT_YEAR else 1900
        year_full = century + year_2dig
        last_trade_date = f"{year_full:04d}{month_num:02d}"  # "202606"
        exchange = self.EXCHANGE_BY_ROOT[root]
        return self._ib.build_future_contract(
            symbol=root,
            last_trade_date=last_trade_date,
            exchange=exchange,
        )

    def _contract_key(self, contract: ContractSymbol, what_to_show: str) -> str:
        """BackoffQueue per-contract key: `<symbol>@<exchange>@<what_to_show>`."""
        root, _, _ = self._parse_contract_symbol(contract)
        exchange = self.EXCHANGE_BY_ROOT[root]
        return f"{contract}@{exchange}@{what_to_show}"

    # ------------------------------------------------------------------------
    # Private: bar size + duration mapping
    # ------------------------------------------------------------------------

    def _map_bar_size(self, resolution: BarResolution) -> str:
        """BarResolution → IBKR barSizeSetting string. Raises if unsupported."""
        if resolution not in self.BAR_SIZE_MAP:
            raise BarsNotSupported(
                f"{self.source_id}: BarResolution {resolution.name} not supported by "
                f"IBKR historical-bars API; supported: {[r.name for r in self.BAR_SIZE_MAP]}"
            )
        return self.BAR_SIZE_MAP[resolution]

    def _compute_duration_str(self, ts_start: datetime, ts_end: datetime) -> str:
        """Build IBKR `durationStr` covering [ts_start, ts_end].

        Format: `"<N> <unit>"` where unit ∈ {S, D, W, M, Y}. Per internal notes, the
        valid unit depends on bar size — here we keep it simple: choose seconds for
        short ranges, days otherwise. Multi-year chunked-request orchestration is
        out of scope for A1.4 skeleton (deferred to A1.5+).
        """
        delta = ts_end - ts_start
        delta_seconds = int(delta.total_seconds())
        if delta_seconds < SECONDS_PER_DAY:
            # Short range: express in seconds (IBKR allows up to 86400 S)
            return f"{delta_seconds} S"
        # Long range: express in whole days, rounding up
        delta_days = (delta_seconds + SECONDS_PER_DAY - 1) // SECONDS_PER_DAY
        return f"{delta_days} D"

    def _format_end_datetime(self, ts_end: datetime) -> str:
        """Format `ts_end` per IBKR `endDateTime` convention.

        IBKR accepts `"YYYYMMDD HH:MM:SS TZ"` (e.g., `"20260521 22:00:00 UTC"`) or empty
        string for "now". We always pass an explicit UTC timestamp for determinism.
        """
        if ts_end.tzinfo is None:
            raise ValueError(f"{self.source_id}: ts_end must be TZ-aware; got naive {ts_end!r}")
        utc_dt = ts_end.astimezone(UTC)
        return utc_dt.strftime("%Y%m%d %H:%M:%S UTC")

    # ------------------------------------------------------------------------
    # Private: connection lifecycle
    # ------------------------------------------------------------------------

    def _ensure_connected(self) -> None:
        """Lazy connect on first use; idempotent thereafter."""
        if self._ib.is_connected():
            return
        try:
            self._ib.connect(self._host, self._port, self._client_id)
        except IBKRConnectionError:
            raise
        except Exception as e:
            raise IBKRConnectionError(
                f"{self.source_id}: failed to connect to IB Gateway at "
                f"{self._host}:{self._port} clientId={self._client_id}: "
                f"{type(e).__name__}: {e}"
            ) from e

    # ------------------------------------------------------------------------
    # Private: bar conversion
    # ------------------------------------------------------------------------

    def _ib_bar_to_raw(
        self,
        ib_bar: Any,
        contract: ContractSymbol,
        resolution: BarResolution,
        as_of_iso: datetime,
    ) -> RawBar:
        """Convert ib_async.BarData → RawBar with provenance hash.

        Per internal notes: daily-bar `date` for futures is session-close-day. IBKR's
        `formatDate=2` returns epoch UTC; we coerce to TZ-aware UTC datetime.

        Per the TZ-aware-always rule + bug class 7 (timezone confusion): every emitted
        bar carries IANA-TZ-aware ts. Float prices go through `Decimal(str(...))` to
        avoid IEEE-754 precision leak.
        """
        # Date handling: ib_async BarData.date is either int (epoch with formatDate=2)
        # or datetime/str depending on version. We accept both for robustness.
        ts = self._coerce_ts_to_utc(ib_bar.date)

        # Decimal coercion via str() to avoid float precision loss
        open_d = Decimal(str(ib_bar.open))
        high_d = Decimal(str(ib_bar.high))
        low_d = Decimal(str(ib_bar.low))
        close_d = Decimal(str(ib_bar.close))
        volume = int(ib_bar.volume) if ib_bar.volume is not None else 0
        # IBKR daily bars don't carry OI; expose as None for verifier downstream
        oi: int | None = None

        # Provenance hash: canonical JSON of bar fields (no raw upstream bytes available
        # from ib_async — vendor SDK returns structured object, not wire-format bytes).
        # Hash matches content_bytes_sha contract (64-char SHA256 hex).
        payload = json.dumps(
            {
                "open": str(open_d),
                "high": str(high_d),
                "low": str(low_d),
                "close": str(close_d),
                "volume": volume,
                "ts": ts.isoformat(),
                "contract": str(contract),
                "resolution": resolution.value,
            },
            sort_keys=True,
        ).encode("utf-8")
        content_sha = hashlib.sha256(payload).hexdigest()

        return RawBar(
            contract=contract,
            ts=ts,
            resolution=resolution,
            open=open_d,
            high=high_d,
            low=low_d,
            close=close_d,
            volume=volume,
            oi=oi,
            source_id=self.SOURCE_ID,
            as_of_iso=as_of_iso,
            content_bytes_sha=content_sha,
        )

    def _coerce_ts_to_utc(self, raw_ts: Any) -> datetime:
        """Coerce ib_async BarData.date (int / datetime / str) → TZ-aware UTC datetime.

        Refuses naive datetime per the fail-loud policy. `formatDate=2`
        should produce epoch UTC or TZ-aware datetime; a naive datetime means
        IB Gateway is leaking account-TZ (default CT) and silent UTC coercion
        would shift bars 5-6h. Raise IBKRReqError so caller surfaces the bug.
        """
        if isinstance(raw_ts, datetime):
            if raw_ts.tzinfo is None:
                raise IBKRReqError(
                    f"{self.SOURCE_ID}: IB returned naive datetime {raw_ts!r} "
                    f"despite formatDate=2; refusing silent TZ assumption (fail-loud). "
                    f"Verify IB Gateway TWS API config 'Send instrument-specific "
                    f"attributes' + formatDate=2 are both active."
                )
            return raw_ts.astimezone(UTC)
        if isinstance(raw_ts, date):
            # Daily bars: ib_async returns a plain `date` (formatDate=2 gives an epoch int only for
            # intraday). Anchor at noon UTC so the date is stable in both UTC and CME tz; daily
            # engine logic keys on ts.date(). After the datetime branch above (a datetime is-a
            # date), so intraday bars are unaffected.
            return datetime(raw_ts.year, raw_ts.month, raw_ts.day, 12, 0, tzinfo=UTC)
        if isinstance(raw_ts, int | float):
            # Epoch seconds (formatDate=2)
            return datetime.fromtimestamp(float(raw_ts), tz=UTC)
        if isinstance(raw_ts, str):
            # ISO-8601 (TZ-aware expected; naive raises per M1 fix)
            parsed = datetime.fromisoformat(raw_ts)
            if parsed.tzinfo is None:
                raise IBKRReqError(
                    f"{self.SOURCE_ID}: IB returned naive ISO datetime string "
                    f"{raw_ts!r}; refusing silent TZ assumption (fail-loud)"
                )
            return parsed.astimezone(UTC)
        raise IBKRReqError(
            f"{self.SOURCE_ID}: cannot coerce IB bar date {raw_ts!r} "
            f"(type {type(raw_ts).__name__}) to UTC datetime"
        )

    # ------------------------------------------------------------------------
    # Private: A1.5 settle helpers
    # ------------------------------------------------------------------------

    def _compute_settle_state(
        self,
        bar_ts_utc: datetime,
        as_of_utc: datetime,
    ) -> SettleState:
        """Determine settle_state heuristically from publication-lag window.

        Per internal notes: 'Generally the official settlement price is not available
        until several hours after a trading session closes. The Friday settlement
        price will sometimes not be available until Saturday.'

        IBKR daily bar's date = session-close timestamp. We convert to CT
        (CME's home TZ) to detect Mon-Thu vs Fri sessions; the Friday session
        has an extended lag (~Saturday morning publish, vs. next-day for Mon-Thu).

        Returns 'preliminary' inside the lag window; 'final' after.

        Note: 'live' is NOT emitted by this method. 'live' is CMEEODDataSource's
        intraday-page-tape state; IBKR daily bars only materialize after session
        close, so we never observe intraday tape via this code path.
        """
        bar_ct = bar_ts_utc.astimezone(CME_SESSION_TZ)
        is_friday_close = bar_ct.weekday() == FRIDAY_WEEKDAY_INDEX
        lag_hours = (
            PRELIMINARY_PUBLISH_LAG_HOURS_FRIDAY
            if is_friday_close
            else PRELIMINARY_PUBLISH_LAG_HOURS_WEEKDAY
        )
        elapsed_hours = (as_of_utc - bar_ts_utc).total_seconds() / SECONDS_PER_HOUR
        return "preliminary" if elapsed_hours < lag_hours else "final"

    def _ib_bar_to_settle(
        self,
        *,
        ib_bar: Any,
        prior_bar: Any | None,
        contract: ContractSymbol,
        as_of_utc: datetime,
    ) -> Settle:
        """Convert ib_async.BarData (daily) → Settle with provenance hash.

        Per internal notes: daily-bar close = official settlement once published.
        IBKR does not distinguish 'preliminary' vs 'final' in the API response;
        settle_state is derived heuristically via `_compute_settle_state`.

        Field decisions (documented for verifier reconciliation in A1.9):
        - `settle` = bar.close (Decimal via str — no float→Decimal IEEE-754 leak).
        - `last`  = settle (IBKR daily bar exposes no separate last-trade price).
        - `change` = latest.close - prior.close; or Decimal(0) if no prior bar.
        - `volume_est` = bar.volume.
        - `oi_prior` = 0 (IBKR daily bars don't publish OI — verifier policy in
          A1.9 marks this as a structural-zero, not a value-disagreement).
        - `as_of_date` = bar's session-close day in CT (matches CME business-date).
        - `cme_month_code` = month code parsed from ContractSymbol.

        `content_bytes_sha` is SHA256 of canonical-JSON-encoded fields (no raw
        upstream bytes available from ib_async; the structured payload IS our
        provenance trail).
        """
        bar_ts_utc = self._coerce_ts_to_utc(ib_bar.date)

        # Decimal coercion via str() (bug class 7 prevention)
        settle_d = Decimal(str(ib_bar.close))
        open_d = Decimal(str(ib_bar.open))
        high_d = Decimal(str(ib_bar.high))
        low_d = Decimal(str(ib_bar.low))
        last_d = settle_d  # IBKR daily bar: no separate last-trade

        if prior_bar is not None:
            prior_close_d = Decimal(str(prior_bar.close))
            change_d = settle_d - prior_close_d
        else:
            change_d = Decimal("0")

        volume_est = int(ib_bar.volume) if ib_bar.volume is not None else 0

        settle_state: SettleState = self._compute_settle_state(bar_ts_utc, as_of_utc)

        bar_ct = bar_ts_utc.astimezone(CME_SESSION_TZ)
        as_of_date_val = bar_ct.date()

        _, month_code, _ = self._parse_contract_symbol(contract)

        # Provenance hash: canonical JSON of fields → SHA256. Deterministic across
        # platforms (no float, no nondeterministic dict ordering).
        payload = json.dumps(
            {
                "contract": str(contract),
                "as_of_date": as_of_date_val.isoformat(),
                "settle": str(settle_d),
                "settle_state": settle_state,
                "open": str(open_d),
                "high": str(high_d),
                "low": str(low_d),
                "last": str(last_d),
                "change": str(change_d),
                "volume_est": volume_est,
                "bar_ts_utc": bar_ts_utc.isoformat(),
            },
            sort_keys=True,
        ).encode("utf-8")
        content_sha = hashlib.sha256(payload).hexdigest()

        return Settle(
            contract=contract,
            as_of_date=as_of_date_val,
            settle=settle_d,
            settle_state=settle_state,
            open=open_d,
            high=high_d,
            low=low_d,
            last=last_d,
            change=change_d,
            volume_est=volume_est,
            oi_prior=0,  # IBKR daily bars don't publish OI
            source_id=self.SOURCE_ID,
            as_of_iso=self._clock.now_utc(),
            content_bytes_sha=content_sha,
            cme_month_code=month_code,
        )
