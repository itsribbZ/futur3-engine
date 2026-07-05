"""futur3.data.sources.cftc - CFTC COT concrete sources (Socrata live + in-memory replay).

Per internal design notes

- `CFTCSocrataSource`: DIY REST over `publicreporting.cftc.gov` (the recommended path per 1.3 -
  no vendor SDK, the bug-class-21 leak guard). Parses Socrata JSON rows into a normalized
  `COTReport`, mapping the flavor-specific category columns to the spec/comm blocs. Schema-drift
  is fail-closed (bug class 9): a missing expected column raises `COTSchemaDriftError` rather than
  silently mapping it to zero. `value_known_at_iso` comes from the Tue->Fri blackout helper.
- `InMemoryCOTSource`: stdlib replay of pre-fetched `COTReport`s for deterministic backtests (the
  COT analog of `ReplayDataSource`) - the engine loop queries it with zero network.

The normalized column mapping per flavor (internal design notes):
  - DISAGGREGATED (CL/GC): spec = Managed Money,   comm = Producer/Merchant/Processor/User
  - TFF (ES/NQ/MBT/MET):   spec = Leveraged Funds, comm = Dealer/Intermediary
  - LEGACY:                spec = Non-Commercial,  comm = Commercial
The columns we read avoid the CFTC schema's sic-marked typos (`noncomm_postions_spread_all`,
`swap__positions_short_all`); the schema-drift guard pins exactly the columns we DO read.

Hard seam: vendor HTTP (`requests`) lives only here, behind the `CFTCHTTPClient`
Protocol; the engine + strategy consume the `COTSource` ABC.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from typing import Final, Protocol

from futur3.data.cot_source import (
    COTSchemaDriftError,
    COTSource,
    COTSourceError,
    value_known_at_for_report,
)
from futur3.data.cot_types import COTReport, COTReportFlavor
from futur3.data.types import SourceTier, content_sha256
from futur3.data.verifier import ClockProtocol

CFTC_SOCRATA_BASE: Final[str] = "https://publicreporting.cftc.gov/resource/"
HTTP_OK: Final[int] = 200
_DEFAULT_LIMIT: Final[int] = 5000  # weekly COT for one contract over decades is < 2600 rows

# Common columns (present in every flavor).
_DATE_COL: Final[str] = "report_date_as_yyyy_mm_dd"  # the TUESDAY snapshot
_OI_COL: Final[str] = "open_interest_all"
_CODE_COL: Final[str] = "cftc_contract_market_code"
_TRADERS_COL: Final[str] = "traders_tot_all"  # optional; None when absent


@dataclass(frozen=True)
class _FlavorSchema:
    """The flavor-specific raw column names that map to the normalized spec/comm blocs."""

    spec_long: str
    spec_short: str
    comm_long: str
    comm_short: str


# Futures-only Socrata dataset ids (internal design notes). SUPPLEMENTAL (ag-only)
# is intentionally absent - out of futur3's 6-contract scope.
_DATASET_IDS: Final[dict[COTReportFlavor, str]] = {
    COTReportFlavor.LEGACY: "6dca-aqww",
    COTReportFlavor.DISAGGREGATED: "72hh-3qpy",
    COTReportFlavor.TFF: "gpe5-46if",
}

# Column names VERIFIED against the live Socrata schema (2026-05-23 discover). The live schema is
# inconsistent on the `_all` suffix: m_money/dealer carry it, but prod_merc/lev_money do NOT - the
# research doc (internal design notes) documented `_all` everywhere, which the
# schema-drift guard correctly rejected on real data. These are the live-correct names.
_FLAVOR_SCHEMAS: Final[dict[COTReportFlavor, _FlavorSchema]] = {
    COTReportFlavor.DISAGGREGATED: _FlavorSchema(
        spec_long="m_money_positions_long_all",
        spec_short="m_money_positions_short_all",
        comm_long="prod_merc_positions_long",
        comm_short="prod_merc_positions_short",
    ),
    COTReportFlavor.TFF: _FlavorSchema(
        spec_long="lev_money_positions_long",
        spec_short="lev_money_positions_short",
        comm_long="dealer_positions_long_all",
        comm_short="dealer_positions_short_all",
    ),
    COTReportFlavor.LEGACY: _FlavorSchema(
        spec_long="noncomm_positions_long_all",
        spec_short="noncomm_positions_short_all",
        comm_long="comm_positions_long_all",
        comm_short="comm_positions_short_all",
    ),
}


# ----------------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------------


class CftcSocrataError(COTSourceError):
    """Socrata transport / shape failure: non-200, malformed JSON, or a non-numeric position."""


# ----------------------------------------------------------------------------
# Vendor seam
# ----------------------------------------------------------------------------


class CFTCHTTPClient(Protocol):
    """Thin transport seam over the CFTC Socrata REST API. Returns RAW bytes so the caller can
    hash them for hash provenance before parsing."""

    def get(self, url: str, params: Mapping[str, str]) -> bytes:
        """GET `url` with `params`; return raw response bytes. Raises on transport / non-200."""
        ...


class _DefaultCFTCHTTPClient:
    """Live Socrata transport via `requests` (lazy-imported per the hard seam).

    A free Socrata app-token lifts the rate limit to 1,000 req/rolling-hour; without
    one the endpoint throttles by IP but still serves - so the token is optional.
    """

    def __init__(self, app_token: str | None = None, timeout_s: float = 30.0) -> None:
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be > 0; got {timeout_s}")
        self._app_token = app_token
        self._timeout_s = timeout_s

    def get(self, url: str, params: Mapping[str, str]) -> bytes:
        import requests

        merged = dict(params)
        if self._app_token:
            merged["$$app_token"] = self._app_token
        resp = requests.get(url, params=merged, timeout=self._timeout_s)
        if resp.status_code != HTTP_OK:
            raise CftcSocrataError(f"CFTC Socrata {url} returned HTTP {resp.status_code}")
        return resp.content


# ----------------------------------------------------------------------------
# JSON / value guards (loud on drift per the fail-loud policy)
# ----------------------------------------------------------------------------


def _loads(raw: bytes) -> object:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CftcSocrataError(f"CFTC Socrata: malformed JSON ({exc})") from exc


def _as_list(obj: object, ctx: str) -> list[object]:
    if not isinstance(obj, list):
        raise CftcSocrataError(f"CFTC Socrata {ctx}: expected JSON array, got {type(obj).__name__}")
    return obj


def _as_dict(obj: object, ctx: str) -> dict[str, object]:
    if not isinstance(obj, dict):
        raise CftcSocrataError(
            f"CFTC Socrata {ctx}: expected JSON object, got {type(obj).__name__}"
        )
    return obj


def _as_str(obj: object, ctx: str) -> str:
    if not isinstance(obj, str):
        raise CftcSocrataError(f"CFTC Socrata {ctx}: expected string, got {type(obj).__name__}")
    return obj


def _to_int(raw: object, col: str) -> int:
    """Parse a Socrata position count. Socrata returns numbers as strings ("120000"); accept
    str / int / float, reject bool (an int subclass that is never a valid count here)."""
    if isinstance(raw, bool):
        raise CftcSocrataError(f"CFTC column {col!r}: expected numeric count, got bool")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str | float):
        try:
            return int(Decimal(str(raw)))
        except (InvalidOperation, ValueError) as exc:
            raise CftcSocrataError(f"CFTC column {col!r}: not an integer: {raw!r}") from exc
    raise CftcSocrataError(f"CFTC column {col!r}: expected numeric count, got {type(raw).__name__}")


def _parse_socrata_date(raw: object, col: str) -> date:
    text = _as_str(raw, col)
    try:
        return datetime.fromisoformat(text).date() if "T" in text else date.fromisoformat(text)
    except ValueError as exc:
        raise CftcSocrataError(f"CFTC column {col!r}: date not ISO: {text!r}") from exc


# ----------------------------------------------------------------------------
# CFTCSocrataSource
# ----------------------------------------------------------------------------


class CFTCSocrataSource(COTSource):
    """COT source backed by the live CFTC Socrata open-data platform (federal, single-source)."""

    SOURCE_ID: Final[str] = "cftc_socrata"

    def __init__(self, http_client: CFTCHTTPClient, clock: ClockProtocol | None = None) -> None:
        self._http = http_client
        self._clock = clock

    @classmethod
    def from_app_token(
        cls,
        app_token: str | None = None,
        timeout_s: float = 30.0,
        clock: ClockProtocol | None = None,
    ) -> CFTCSocrataSource:
        """Build a live source. `app_token` (free Socrata signup) is optional - it only lifts the
        rate limit; the endpoint serves without one."""
        return cls(_DefaultCFTCHTTPClient(app_token, timeout_s), clock)

    @property
    def source_id(self) -> str:
        return self.SOURCE_ID

    @property
    def tier(self) -> SourceTier:
        return SourceTier.T2_MACRO  # direct federal publisher

    def _now(self) -> datetime:
        return self._clock.now_utc() if self._clock is not None else datetime.now(UTC)

    def fetch_reports(
        self,
        cftc_contract_market_code: str,
        flavor: COTReportFlavor,
        start: date,
        end: date,
    ) -> list[COTReport]:
        if flavor not in _DATASET_IDS:
            raise COTSourceError(f"CFTC Socrata source does not support flavor {flavor.value!r}")
        if end < start:
            raise ValueError(f"fetch_reports end must be >= start; got start={start} end={end}")
        schema = _FLAVOR_SCHEMAS[flavor]
        url = f"{CFTC_SOCRATA_BASE}{_DATASET_IDS[flavor]}.json"
        params = {
            "$where": (
                f"{_CODE_COL}='{cftc_contract_market_code}' "
                f"AND {_DATE_COL} >= '{start.isoformat()}' "
                f"AND {_DATE_COL} <= '{end.isoformat()}'"
            ),
            "$order": f"{_DATE_COL} ASC",
            "$limit": str(_DEFAULT_LIMIT),
        }
        raw = self._http.get(url, params)
        rows = _as_list(_loads(raw), "rows")
        now = self._now()
        reports = [self._parse_row(_as_dict(row, "row"), flavor, schema, now) for row in rows]
        return sorted(reports, key=lambda r: r.report_date)

    def _parse_row(
        self,
        row: dict[str, object],
        flavor: COTReportFlavor,
        schema: _FlavorSchema,
        now: datetime,
    ) -> COTReport:
        required = {
            _DATE_COL,
            _OI_COL,
            _CODE_COL,
            schema.spec_long,
            schema.spec_short,
            schema.comm_long,
            schema.comm_short,
        }
        missing = required - set(row)
        if missing:
            raise COTSchemaDriftError(
                f"CFTC {flavor.value} row missing columns {sorted(missing)} "
                f"(schema drift; expected {sorted(required)})"
            )
        report_date = _parse_socrata_date(row[_DATE_COL], _DATE_COL)
        traders = _to_int(row[_TRADERS_COL], _TRADERS_COL) if _TRADERS_COL in row else None
        # Per-row provenance hash over the canonical (key-sorted) row bytes - unique per row, so no
        # per-page-sha collision (known regression class).
        sha = content_sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode())
        return COTReport(
            cftc_contract_market_code=_as_str(row[_CODE_COL], _CODE_COL),
            flavor=flavor,
            report_date=report_date,
            value_known_at_iso=value_known_at_for_report(report_date),
            open_interest_all=_to_int(row[_OI_COL], _OI_COL),
            spec_long=_to_int(row[schema.spec_long], schema.spec_long),
            spec_short=_to_int(row[schema.spec_short], schema.spec_short),
            comm_long=_to_int(row[schema.comm_long], schema.comm_long),
            comm_short=_to_int(row[schema.comm_short], schema.comm_short),
            source_id=self.SOURCE_ID,
            as_of_iso=now,
            content_bytes_sha=sha,
            total_traders=traders,
        )


# ----------------------------------------------------------------------------
# InMemoryCOTSource (backtest replay)
# ----------------------------------------------------------------------------


class InMemoryCOTSource(COTSource):
    """Stdlib replay of pre-fetched `COTReport`s - the COT analog of `ReplayDataSource`.

    Holds reports in memory so a backtest queries them with zero network (deterministic). Tier is
    `T4_DERIVED`: a replay feed, even of authoritative CFTC data, is a derived source. The PIT gate
    still applies through the inherited `reports_known_at`.
    """

    def __init__(self, reports: Iterable[COTReport], source_id: str = "cot_in_memory") -> None:
        if not source_id:
            raise ValueError("InMemoryCOTSource.source_id must be non-empty")
        self._reports = list(reports)
        self._source_id = source_id

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def tier(self) -> SourceTier:
        return SourceTier.T4_DERIVED

    def fetch_reports(
        self,
        cftc_contract_market_code: str,
        flavor: COTReportFlavor,
        start: date,
        end: date,
    ) -> list[COTReport]:
        hits = [
            r
            for r in self._reports
            if r.cftc_contract_market_code == cftc_contract_market_code
            and r.flavor == flavor
            and start <= r.report_date <= end
        ]
        return sorted(hits, key=lambda r: r.report_date)


__all__: list[str] = [
    "CFTC_SOCRATA_BASE",
    "CFTCHTTPClient",
    "CFTCSocrataSource",
    "CftcSocrataError",
    "InMemoryCOTSource",
]
