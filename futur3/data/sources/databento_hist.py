"""futur3.data.sources.databento_hist - DatabentoHistoricalDataSource (CME GLBX.MDP3 daily bars).

The institutional historical price tier: Databento is a licensed CME redistributor (Globex MDP 3.0),
so it is `T1_EXCHANGE_HISTORICAL` - PIT-correct, survivorship-bias-free, true per-contract +
continuous-front-month daily OHLCV for the CME universe (ES/NQ/CL/GC). This is what unblocks
deep-history research on the physically-delivered markets (CL/GC) - the public
`cme_eod` scraper only has recent settlements, not deep history.

Cost discipline (free-first, small prepaid PAYG credit): this source pulls ONLY `ohlcv-1d`
(daily bars) - tiny data, a few dollars of credit for the whole universe over a decade. NEVER
pull tick/MBO here (GB-scale, burns the credit). The recurring $179/mo Databento LIVE subscription
is NOT used - paper trading runs on the free live/delayed path.

Hard seam: the `databento` SDK (+ its pandas/zstandard deps) lives ONLY inside
`_DefaultDatabentoClient`, behind the `DatabentoClient` Protocol. The source maps plain
`DatabentoBar` records -> `RawBar`; the engine + verifier never see a databento type. Tests inject
a fixture client (ZERO network, no `databento` install, no API key).

Symbology: futur3 ContractSymbol root -> Databento continuous front-month (e.g. "CL" -> "CL.c.0",
`stype_in='continuous'`) - the standard single-series continuous for a systematic backtest.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Final, Protocol

from futur3.data.source import (
    BarsNotSupported,
    ContractNotConfigured,
    DataSource,
    FutureDatedSourceError,
)
from futur3.data.types import (
    BarResolution,
    ContractSymbol,
    RawBar,
    SourceTier,
    _assert_tz_aware,
    content_sha256,
)
from futur3.data.verifier import ClockProtocol

_DATASET: Final[str] = "GLBX.MDP3"  # CME Globex MDP 3.0
_OHLCV_1D_SCHEMA: Final[str] = "ohlcv-1d"
_STYPE_CONTINUOUS: Final[str] = "continuous"
_ROOT_SUFFIX_LEN: Final[int] = 3  # ContractSymbol = <root><month_code(1)><year(2)>

# futur3 contract root -> Databento continuous front-month symbol (rank 0). CME-only: crypto
# (MBT/MET) price comes from the free crypto venues, not Databento.
_CONTINUOUS_SYMBOLS: Final[dict[str, str]] = {
    "ES": "ES.c.0",
    "NQ": "NQ.c.0",
    "CL": "CL.c.0",
    "GC": "GC.c.0",
}


@dataclass(frozen=True)
class DatabentoBar:
    """A plain daily OHLCV record from the vendor seam (no databento/pandas types leak past it)."""

    ts: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int


class DatabentoClient(Protocol):
    """Transport seam over the Databento historical API. Returns plain `DatabentoBar`s so the
    source (and its tests) never touch the `databento`/pandas SDK."""

    def fetch_ohlcv_1d(
        self, dataset: str, symbol: str, stype_in: str, start: datetime, end: datetime
    ) -> list[DatabentoBar]:
        """Daily OHLCV bars for `symbol` in [start, end]. Raises on transport / auth failure."""
        ...


class _DefaultDatabentoClient:
    """Live transport via the `databento` SDK (lazy-imported per the hard seam).

    Reads `DATABENTO_API_KEY` from the environment when no key is passed (the SDK's own default).
    Prices come back as Decimal via `to_df(price_type='decimal')` - no float leak.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key

    def fetch_ohlcv_1d(
        self, dataset: str, symbol: str, stype_in: str, start: datetime, end: datetime
    ) -> list[DatabentoBar]:
        import databento as db  # lazy: only the live path needs the SDK + pandas

        client = db.Historical(self._api_key) if self._api_key else db.Historical()
        data = client.timeseries.get_range(
            dataset=dataset,
            symbols=[symbol],
            schema=_OHLCV_1D_SCHEMA,
            stype_in=stype_in,
            start=start.isoformat(),
            end=end.isoformat(),
        )
        df = data.to_df(price_type="decimal", pretty_ts=True, tz="UTC")
        bars: list[DatabentoBar] = []
        for row in df.itertuples():
            bars.append(
                DatabentoBar(
                    ts=row.Index.to_pydatetime(),
                    open=Decimal(str(row.open)),
                    high=Decimal(str(row.high)),
                    low=Decimal(str(row.low)),
                    close=Decimal(str(row.close)),
                    volume=int(row.volume),
                )
            )
        return bars


class DatabentoHistoricalDataSource(DataSource):
    """CME daily-bar history via Databento GLBX.MDP3 (T1 exchange-historical, PIT-correct)."""

    SOURCE_ID: Final[str] = "databento_glbx_mdp3"

    def __init__(
        self,
        client: DatabentoClient | None = None,
        clock: ClockProtocol | None = None,
        *,
        symbols: dict[str, str] | None = None,
    ) -> None:
        self._client: DatabentoClient = client or _DefaultDatabentoClient()
        self._clock = clock
        self._symbols = dict(symbols) if symbols is not None else dict(_CONTINUOUS_SYMBOLS)

    @classmethod
    def from_api_key(
        cls, api_key: str | None = None, clock: ClockProtocol | None = None
    ) -> DatabentoHistoricalDataSource:
        """Build a live source. `api_key` falls back to the DATABENTO_API_KEY env var."""
        return cls(_DefaultDatabentoClient(api_key), clock)

    @property
    def source_id(self) -> str:
        return self.SOURCE_ID

    @property
    def tier(self) -> SourceTier:
        return SourceTier.T1_EXCHANGE_HISTORICAL  # licensed CME MDP3 redistributor

    def _now(self) -> datetime:
        return self._clock.now_utc() if self._clock is not None else datetime.now(UTC)

    def _resolve_symbol(self, contract: ContractSymbol) -> str:
        key = str(contract)
        root = key if key in self._symbols else key[:-_ROOT_SUFFIX_LEN]
        symbol = self._symbols.get(root)
        if symbol is None:
            raise ContractNotConfigured(
                f"{self.SOURCE_ID}: no Databento symbol for {contract!r} (root {root!r}); "
                f"configured: {sorted(self._symbols)}"
            )
        return symbol

    def get_bars(
        self,
        contract: ContractSymbol,
        ts_start: datetime,
        ts_end: datetime,
        resolution: BarResolution,
    ) -> Iterable[RawBar]:
        if resolution is not BarResolution.DAY_1:
            raise BarsNotSupported(
                f"{self.SOURCE_ID} only serves DAY_1 (ohlcv-1d); got {resolution.value}"
            )
        _assert_tz_aware(ts_start, "DatabentoHistoricalDataSource.get_bars ts_start")
        _assert_tz_aware(ts_end, "DatabentoHistoricalDataSource.get_bars ts_end")
        symbol = self._resolve_symbol(contract)
        now = self._now()
        records = self._client.fetch_ohlcv_1d(_DATASET, symbol, _STYPE_CONTINUOUS, ts_start, ts_end)
        bars: list[RawBar] = []
        for rec in records:
            if rec.ts > now:
                raise FutureDatedSourceError(
                    f"{self.SOURCE_ID}: bar ts {rec.ts.isoformat()} > now {now.isoformat()}"
                )
            if not (ts_start <= rec.ts < ts_end):  # half-open interval per the ABC contract
                continue
            bars.append(
                RawBar(
                    contract=contract,
                    ts=rec.ts,
                    resolution=BarResolution.DAY_1,
                    open=rec.open,
                    high=rec.high,
                    low=rec.low,
                    close=rec.close,
                    volume=rec.volume,
                    oi=None,  # MDP3 ohlcv-1d carries no OI
                    source_id=self.SOURCE_ID,
                    as_of_iso=now,
                    content_bytes_sha=content_sha256(
                        f"{symbol}|{rec.ts.isoformat()}|{rec.open}|{rec.high}|"
                        f"{rec.low}|{rec.close}|{rec.volume}".encode()
                    ),
                )
            )
        bars.sort(key=lambda b: b.ts)  # strictly-increasing ts per the ABC contract
        return bars


__all__: list[str] = [
    "DatabentoBar",
    "DatabentoClient",
    "DatabentoHistoricalDataSource",
]
