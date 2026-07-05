"""futur3.data.sources.bls_macro - BlsMacroSource (LIVE event-day macro prints).

Per internal design notes

BLS Public Data API v2 is the LIVE EVENT-DAY source: it serves the freshest published print
(the 08:30 ET NFP / CPI / PPI release) with minimal lag. CRITICAL: the BLS API returns the
LATEST-REVISED value for each reference period and exposes NO per-observation publication date
or vintage. Therefore:

  * BlsMacroSource is for LIVE polling ("what did BLS just publish"). It is NOT point-in-time
    correct for backtests - later revisions would leak (bug class 5). Use FredMacroSource
    (ALFRED vintages) for backtest values. The engine routes hot-path-live -> BLS and
    backtest -> FRED (spec section 2.2); this source does not know the runtime mode.

  * value_known_at_iso (the PIT boundary) cannot come from the API. Provide a `release_lookup`
    (series + reference month -> scheduled release datetime, e.g. from the macro calendar /
    FredMacroSource.upcoming_events) for an ACCURATE boundary. Without one it falls back to a
    CONSERVATIVE end-of-next-month bound that errs toward blocking (never look-ahead) but is
    too late for live - wire a calendar for live use.

Hard seam: vendor HTTP (requests-to-BLS) lives only here, behind the `BlsHTTPClient` Protocol.
fetch_value routes its result through `enforce_pit_gate` per the MacroEventSource PIT contract.
"""

from __future__ import annotations

import json
from calendar import monthrange
from collections.abc import Callable
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Final, Protocol
from zoneinfo import ZoneInfo

from futur3.data.macro_source import MacroEventSource, MacroSourceError, enforce_pit_gate
from futur3.data.macro_types import (
    RELEASE_TIME_ET,
    MacroEvent,
    MacroPublisher,
    MacroSeries,
    MacroValue,
)
from futur3.data.types import SourceTier, _assert_tz_aware, content_sha256

ET = ZoneInfo("America/New_York")  # canonical ET
BLS_API_BASE: Final[str] = "https://api.bls.gov/publicAPI/v2/"
HTTP_OK: Final[int] = 200
_BLS_SUCCESS: Final[str] = "REQUEST_SUCCEEDED"
_MONTHLY_PERIOD_LEN: Final[int] = 3  # "M06"
_MONTHS_IN_YEAR: Final[int] = 12

# BLS series ids for the high-impact event-day prints BLS itself publishes. NFP + CPI are
# confident (CES0000000001 standard; CUUR0000SA0 per internal design notes).
# PPI WPSFD4 (final demand) is a HYPOTHESIS pending live verification (A1.26 INTEGRATION_SMOKE).
BLS_SERIES_MAP: Final[dict[MacroSeries, str]] = {
    MacroSeries.NFP: "CES0000000001",
    MacroSeries.CPI: "CUUR0000SA0",
    MacroSeries.PPI: "WPSFD4",  # HYPOTHESIS - verify exact id at live smoke
}

SERIES_PUBLISHER: Final[dict[MacroSeries, MacroPublisher]] = {
    MacroSeries.NFP: MacroPublisher.BLS,
    MacroSeries.CPI: MacroPublisher.BLS,
    MacroSeries.PPI: MacroPublisher.BLS,
}

_SERIES_BY_LENGTH: Final[tuple[MacroSeries, ...]] = tuple(
    sorted(MacroSeries, key=lambda s: len(s.value), reverse=True)
)

# series + reference month -> scheduled release moment (tz-aware). e.g. backed by the macro
# calendar or FredMacroSource.upcoming_events. Returns None if the release date is unknown.
ReleaseLookup = Callable[[MacroSeries, date], datetime | None]


# ----------------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------------


class BlsMacroError(MacroSourceError):
    """BLS-specific acquisition failure: transport non-200, non-success status, malformed JSON,
    or a response whose shape diverged from the documented contract (bug class 9 hook)."""


# ----------------------------------------------------------------------------
# Vendor seam
# ----------------------------------------------------------------------------


class BlsHTTPClient(Protocol):
    """Thin transport seam over the BLS v2 single-series GET endpoint. Returns RAW bytes so the
    caller can hash them for hash provenance before parsing."""

    def get_series(self, series_id: str) -> bytes:
        """GET `<BLS_API_BASE>timeseries/data/<series_id>`; return raw bytes (last ~3 years).

        Raises on transport failure / non-200.
        """
        ...


class _DefaultBlsHTTPClient:
    """Live BLS transport via `requests` (lazy-imported per the hard seam)."""

    def __init__(self, registration_key: str | None = None, timeout_s: float = 30.0) -> None:
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be > 0; got {timeout_s}")
        self._key = registration_key
        self._timeout_s = timeout_s

    def get_series(self, series_id: str) -> bytes:
        import requests

        params = {"registrationkey": self._key} if self._key else {}
        resp = requests.get(
            f"{BLS_API_BASE}timeseries/data/{series_id}", params=params, timeout=self._timeout_s
        )
        if resp.status_code != HTTP_OK:
            raise BlsMacroError(f"BLS series {series_id} returned HTTP {resp.status_code}")
        return resp.content


# ----------------------------------------------------------------------------
# JSON shape guards + parse helpers (self-contained per the source-decoupling convention)
# ----------------------------------------------------------------------------


def _as_obj(raw: bytes, ctx: str) -> dict[str, object]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BlsMacroError(f"BLS {ctx}: malformed JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise BlsMacroError(f"BLS {ctx}: expected JSON object, got {type(data).__name__}")
    return data


def _as_list(obj: object, ctx: str) -> list[object]:
    if not isinstance(obj, list):
        raise BlsMacroError(f"BLS {ctx}: expected JSON array, got {type(obj).__name__}")
    return obj


def _as_dict(obj: object, ctx: str) -> dict[str, object]:
    if not isinstance(obj, dict):
        raise BlsMacroError(f"BLS {ctx}: expected JSON object, got {type(obj).__name__}")
    return obj


def _as_str(obj: object, ctx: str) -> str:
    if not isinstance(obj, str):
        raise BlsMacroError(f"BLS {ctx}: expected string, got {type(obj).__name__}")
    return obj


def _series_from_event_id(event_id: str) -> MacroSeries | None:
    for series in _SERIES_BY_LENGTH:
        if event_id == series.value or event_id.startswith(series.value + "_"):
            return series
    return None


def _is_monthly_period(period: str) -> bool:
    """True for BLS monthly period codes M01..M12 (excludes M13 annual-average, Q/S/A codes)."""
    if len(period) != _MONTHLY_PERIOD_LEN or period[0] != "M" or not period[1:].isdigit():
        return False
    return 1 <= int(period[1:]) <= _MONTHS_IN_YEAR


def _is_blank_value(v: object) -> bool:
    return v is None or (isinstance(v, str) and v.strip() in ("", "-", "."))


def _parse_decimal(raw_value: str, ctx: str) -> Decimal:
    try:
        d = Decimal(raw_value)  # str input - never Decimal(float)
    except InvalidOperation as exc:
        raise BlsMacroError(f"BLS {ctx}: value not parseable as Decimal: {raw_value!r}") from exc
    if not d.is_finite():
        raise BlsMacroError(f"BLS {ctx}: value not finite: {raw_value!r}")
    return d


def _conservative_publication_bound(series: MacroSeries, ref_date: date) -> datetime:
    """Safe upper bound on a monthly series' publication: the LAST day of the month FOLLOWING the
    reference month, at the series' canonical release time (ET). Errs toward blocking (never
    look-ahead). Used only when no `release_lookup` is wired; prefer an accurate lookup."""
    if ref_date.month == _MONTHS_IN_YEAR:
        pub_year, pub_month = ref_date.year + 1, 1
    else:
        pub_year, pub_month = ref_date.year, ref_date.month + 1
    last_day = monthrange(pub_year, pub_month)[1]
    return datetime.combine(date(pub_year, pub_month, last_day), RELEASE_TIME_ET[series], tzinfo=ET)


# ----------------------------------------------------------------------------
# BlsMacroSource
# ----------------------------------------------------------------------------


class BlsMacroSource(MacroEventSource):
    """LIVE event-day BLS source (latest-revised prints; see module docstring for the backtest
    revision caveat). `upcoming_events` returns [] - BLS exposes no clean forward-calendar API;
    use FredMacroSource.upcoming_events for the calendar. `schedule_drift_check`/`healthcheck`
    inherit the ABC defaults.
    """

    SOURCE_ID: Final[str] = "bls_api_v2"

    def __init__(self, http_client: BlsHTTPClient, *, release_lookup: ReleaseLookup | None = None):
        self._http = http_client
        self._release_lookup = release_lookup

    @classmethod
    def from_registration_key(
        cls,
        registration_key: str | None = None,
        *,
        release_lookup: ReleaseLookup | None = None,
        timeout_s: float = 30.0,
    ) -> BlsMacroSource:
        """Build a live source. A free BLS API key (https://www.bls.gov/developers/) raises the
        rate limit; the keyless GET works at reduced limits."""
        return cls(
            _DefaultBlsHTTPClient(registration_key, timeout_s), release_lookup=release_lookup
        )

    @property
    def source_id(self) -> str:
        return self.SOURCE_ID

    @property
    def tier(self) -> SourceTier:
        return SourceTier.T2_MACRO  # direct government publisher

    def fetch_value(self, event_id: str, as_of_iso: datetime) -> MacroValue | None:
        _assert_tz_aware(as_of_iso, "BlsMacroSource.fetch_value as_of_iso")
        series = _series_from_event_id(event_id)
        if series is None or series not in BLS_SERIES_MAP:
            return None
        raw = self._http.get_series(BLS_SERIES_MAP[series])
        payload = _as_obj(raw, "timeseries/data")
        status = payload.get("status")
        if status != _BLS_SUCCESS:
            raise BlsMacroError(
                f"BLS request not succeeded: status={status!r} message={payload.get('message')!r}"
            )
        data = self._extract_data_points(payload)
        chosen = self._latest_monthly_point(data)
        if chosen is None:
            return None
        year_str = _as_str(chosen.get("year"), "data.year")
        if not year_str.isdigit():
            raise BlsMacroError(f"BLS data.year not numeric: {year_str!r}")
        period = _as_str(chosen.get("period"), "data.period")
        ref_date = date(int(year_str), int(period[1:]), 1)
        value = _parse_decimal(_as_str(chosen.get("value"), "data.value"), "data")
        release_dt = self._release_lookup(series, ref_date) if self._release_lookup else None
        value_known_at = (
            release_dt
            if release_dt is not None
            else _conservative_publication_bound(series, ref_date)
        )
        mv = MacroValue(
            event_id=event_id,
            series=series,
            as_of_date=ref_date,
            value_known_at_iso=value_known_at,
            source_id=self.SOURCE_ID,
            as_of_iso=as_of_iso,
            content_bytes_sha=content_sha256(raw),
            value=value,
        )
        return enforce_pit_gate(mv, as_of_iso)

    def upcoming_events(self, start: datetime, end: datetime) -> list[MacroEvent]:
        """BLS provides no clean forward-calendar API; this source is value-only. Use
        FredMacroSource.upcoming_events for the macro calendar. Returns []."""
        _assert_tz_aware(start, "BlsMacroSource.upcoming_events start")
        _assert_tz_aware(end, "BlsMacroSource.upcoming_events end")
        if end <= start:
            raise ValueError(f"upcoming_events end must be > start; got start={start} end={end}")
        return []

    @staticmethod
    def _extract_data_points(payload: dict[str, object]) -> list[object]:
        # BLS docs show Results as both an object {series:[...]} and a 1-element array; handle both.
        results_obj = payload.get("Results")
        if isinstance(results_obj, list):
            if not results_obj:
                return []
            results_obj = results_obj[0]
        results = _as_dict(results_obj, "Results")
        series_list = _as_list(results.get("series"), "Results.series")
        if not series_list:
            return []
        return _as_list(_as_dict(series_list[0], "series[0]").get("data"), "series[0].data")

    @staticmethod
    def _latest_monthly_point(data: list[object]) -> dict[str, object] | None:
        # BLS returns newest-first; take the latest monthly (M01..M12), non-blank data point.
        for dp_obj in data:
            dp = _as_dict(dp_obj, "data point")
            if not _is_monthly_period(_as_str(dp.get("period"), "data.period")):
                continue
            if _is_blank_value(dp.get("value")):
                continue
            return dp
        return None
