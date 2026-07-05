"""futur3.data.sources.fred_macro - FredMacroSource (ALFRED point-in-time macro values).

Per internal design notes

FRED + ALFRED is the BACKTEST-PRIMARY macro source: querying `series/observations` with
`realtime_start == realtime_end == <as_of_date>` returns the value AS KNOWN ON that historical
date (the vintage), which is exactly the PIT-correct revision handling `MacroValue` models. Each
returned observation carries its own `realtime_start` (the date that vintage entered the FRED
real-time window, i.e. when the value first became known) - anchored to the canonical
`RELEASE_TIME_ET`, that is the `value_known_at_iso` PIT boundary.

Date-granular vintages are conservative + correct for backtest decisions at bar boundaries.
Sub-second event-day precision (the 08:30 ET print) comes from the direct BLS source (A1.26+).

Hard seam: vendor HTTP (requests-to-FRED) lives only here, behind the `FredHTTPClient`
Protocol. `fetch_value` routes its result through `enforce_pit_gate` per the MacroEventSource
PIT contract (bug class 5).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
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

ET = ZoneInfo("America/New_York")
FRED_API_BASE: Final[str] = "https://api.stlouisfed.org/fred/"
FRED_MISSING_VALUE: Final[str] = "."
HTTP_OK: Final[int] = 200
_OBS_LOOKBACK_LIMIT: Final[int] = 12  # newest N observations of a vintage; skip trailing missings

# FRED series ids for the numeric subset FRED carries cleanly (verified vs fred.stlouisfed.org).
# Narrative/private series (FOMC_*, BEIGE_BOOK, ISM_*, CONSUMER_CONF) are intentionally absent -
# fetch_value returns None for them; they arrive via the direct Fed source (A1.26+).
# GDP_SECOND/GDP_THIRD are intentionally absent: they are later vintages of GDPC1 and surface
# naturally through fetch_value at a later as_of (the ALFRED realtime mechanism resolves them).
FRED_SERIES_MAP: Final[dict[MacroSeries, str]] = {
    MacroSeries.NFP: "PAYEMS",
    MacroSeries.CPI: "CPIAUCSL",
    MacroSeries.PPI: "PPIFIS",
    MacroSeries.GDP_ADVANCE: "GDPC1",
    MacroSeries.PCE: "PCEPI",
    MacroSeries.RETAIL_SALES: "RSAFS",
    MacroSeries.HOUSING_STARTS: "HOUST",
    MacroSeries.JOBLESS_CLAIMS: "ICSA",
    MacroSeries.JOLTS: "JTSJOL",
}

# Underlying authoritative publisher per series (FRED is the aggregator/source_id; the RELEASE
# belongs to the agency). Used to populate MacroEvent.publisher in upcoming_events.
SERIES_PUBLISHER: Final[dict[MacroSeries, MacroPublisher]] = {
    MacroSeries.NFP: MacroPublisher.BLS,
    MacroSeries.CPI: MacroPublisher.BLS,
    MacroSeries.PPI: MacroPublisher.BLS,
    MacroSeries.GDP_ADVANCE: MacroPublisher.BEA,
    MacroSeries.PCE: MacroPublisher.BEA,
    MacroSeries.RETAIL_SALES: MacroPublisher.CENSUS,
    MacroSeries.HOUSING_STARTS: MacroPublisher.CENSUS,
    MacroSeries.JOBLESS_CLAIMS: MacroPublisher.DOL,
    MacroSeries.JOLTS: MacroPublisher.BLS,
}

# Longest-value-first so e.g. "GDP_ADVANCE_..." matches GDP_ADVANCE before any shorter prefix.
_SERIES_BY_LENGTH: Final[tuple[MacroSeries, ...]] = tuple(
    sorted(MacroSeries, key=lambda s: len(s.value), reverse=True)
)


# ----------------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------------


class FredMacroError(MacroSourceError):
    """FRED-specific acquisition failure: transport non-200, malformed JSON, or a response whose
    shape diverged from the documented contract (bug class 9 schema-drift hook for FRED)."""


# ----------------------------------------------------------------------------
# Vendor seam
# ----------------------------------------------------------------------------


class FredHTTPClient(Protocol):
    """Thin transport seam over the FRED REST API. Returns RAW bytes so the caller can hash them
    for hash provenance before parsing. Implementations inject `api_key` + `file_type=json`."""

    def get(self, endpoint: str, params: Mapping[str, str]) -> bytes:
        """GET `<FRED_API_BASE><endpoint>` with `params`; return raw response bytes.

        `endpoint` is a path like "series/observations". Raises on transport failure / non-200.
        """
        ...


class _DefaultFredHTTPClient:
    """Live FRED transport via `requests` (lazy-imported per the hard seam)."""

    def __init__(self, api_key: str, timeout_s: float = 30.0) -> None:
        if not api_key:
            raise ValueError("FRED api_key must be non-empty")
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be > 0; got {timeout_s}")
        self._api_key = api_key
        self._timeout_s = timeout_s

    def get(self, endpoint: str, params: Mapping[str, str]) -> bytes:
        import requests

        merged = {**params, "api_key": self._api_key, "file_type": "json"}
        resp = requests.get(f"{FRED_API_BASE}{endpoint}", params=merged, timeout=self._timeout_s)
        if resp.status_code != HTTP_OK:
            raise FredMacroError(f"FRED {endpoint} returned HTTP {resp.status_code}")
        return resp.content


# ----------------------------------------------------------------------------
# JSON shape guards (boundary validation - loud on drift per the fail-loud policy)
# ----------------------------------------------------------------------------


def _as_obj(raw: bytes, ctx: str) -> dict[str, object]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise FredMacroError(f"FRED {ctx}: malformed JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise FredMacroError(f"FRED {ctx}: expected JSON object, got {type(data).__name__}")
    return data


def _as_list(obj: object, ctx: str) -> list[object]:
    if not isinstance(obj, list):
        raise FredMacroError(f"FRED {ctx}: expected JSON array, got {type(obj).__name__}")
    return obj


def _as_dict(obj: object, ctx: str) -> dict[str, object]:
    if not isinstance(obj, dict):
        raise FredMacroError(f"FRED {ctx}: expected JSON object, got {type(obj).__name__}")
    return obj


def _as_str(obj: object, ctx: str) -> str:
    if not isinstance(obj, str):
        raise FredMacroError(f"FRED {ctx}: expected string, got {type(obj).__name__}")
    return obj


def _series_from_event_id(event_id: str) -> MacroSeries | None:
    """Resolve the MacroSeries an event_id belongs to (event_id == "<SERIES>_<YYYY>_<MM>")."""
    for series in _SERIES_BY_LENGTH:
        if event_id == series.value or event_id.startswith(series.value + "_"):
            return series
    return None


def _parse_decimal(raw_value: str, ctx: str) -> Decimal:
    try:
        d = Decimal(raw_value)  # str input - never Decimal(float)
    except InvalidOperation as exc:
        raise FredMacroError(f"FRED {ctx}: value not parseable as Decimal: {raw_value!r}") from exc
    if not d.is_finite():
        raise FredMacroError(f"FRED {ctx}: value not finite: {raw_value!r}")
    return d


def _parse_date(raw_date: str, ctx: str) -> date:
    try:
        return date.fromisoformat(raw_date)
    except ValueError as exc:
        raise FredMacroError(f"FRED {ctx}: date not ISO YYYY-MM-DD: {raw_date!r}") from exc


# ----------------------------------------------------------------------------
# FredMacroSource
# ----------------------------------------------------------------------------


class FredMacroSource(MacroEventSource):
    """ALFRED-backed macro source. BACKTEST-PRIMARY: PIT-correct vintage values via realtime
    window queries. Drift detection (`schedule_drift_check`) is deferred to A1.25.b (inherits
    the ABC default []); live healthcheck inherits the ABC default True.
    """

    SOURCE_ID: Final[str] = "fred_alfred"

    def __init__(self, http_client: FredHTTPClient) -> None:
        self._http = http_client

    @classmethod
    def from_api_key(cls, api_key: str, timeout_s: float = 30.0) -> FredMacroSource:
        """Build a live source from a free FRED API key (https://fred.stlouisfed.org/docs/api/)."""
        return cls(_DefaultFredHTTPClient(api_key, timeout_s))

    @property
    def source_id(self) -> str:
        return self.SOURCE_ID

    @property
    def tier(self) -> SourceTier:
        return SourceTier.T3_AGGREGATOR  # aggregator: verifier prefers direct gov (T2_MACRO)

    def fetch_value(self, event_id: str, as_of_iso: datetime) -> MacroValue | None:
        _assert_tz_aware(as_of_iso, "FredMacroSource.fetch_value as_of_iso")
        series = _series_from_event_id(event_id)
        if series is None or series not in FRED_SERIES_MAP:
            return None  # unknown / narrative series this source does not carry
        as_of_date = as_of_iso.astimezone(ET).date()
        raw = self._http.get(
            "series/observations",
            {
                "series_id": FRED_SERIES_MAP[series],
                "realtime_start": as_of_date.isoformat(),
                "realtime_end": as_of_date.isoformat(),
                "sort_order": "desc",
                "limit": str(_OBS_LOOKBACK_LIMIT),
            },
        )
        observations = _as_list(_as_obj(raw, "observations").get("observations"), "observations")
        chosen: dict[str, object] | None = None
        for obs_obj in observations:
            obs = _as_dict(obs_obj, "observation")
            raw_val = obs.get("value")
            if raw_val is None or raw_val == FRED_MISSING_VALUE:
                continue  # missing data point (incl. shutdown-void); skip, do not fabricate
            chosen = obs
            break
        if chosen is None:
            return None
        value = _parse_decimal(_as_str(chosen.get("value"), "observation.value"), "observation")
        ref_date = _parse_date(_as_str(chosen.get("date"), "observation.date"), "observation")
        published = _parse_date(
            _as_str(chosen.get("realtime_start"), "observation.realtime_start"), "observation"
        )
        value_known_at = datetime.combine(published, RELEASE_TIME_ET[series], tzinfo=ET)
        mv = MacroValue(
            event_id=event_id,
            series=series,
            as_of_date=ref_date,
            value_known_at_iso=value_known_at,
            source_id=self.SOURCE_ID,
            as_of_iso=as_of_iso,
            content_bytes_sha=content_sha256(raw),
            value=value,
            vintage_as_of=as_of_date,
        )
        return enforce_pit_gate(mv, as_of_iso)

    def upcoming_events(self, start: datetime, end: datetime) -> list[MacroEvent]:
        _assert_tz_aware(start, "FredMacroSource.upcoming_events start")
        _assert_tz_aware(end, "FredMacroSource.upcoming_events end")
        if end <= start:
            raise ValueError(f"upcoming_events end must be > start; got start={start} end={end}")
        start_date = start.astimezone(ET).date()
        end_date = end.astimezone(ET).date()
        events: list[MacroEvent] = []
        for series, fred_series_id in FRED_SERIES_MAP.items():
            release_id, release_link = self._resolve_release(fred_series_id)
            raw = self._http.get(
                "release/dates",
                {
                    "release_id": str(release_id),
                    "realtime_start": start_date.isoformat(),
                    "realtime_end": end_date.isoformat(),
                    "include_release_dates_with_no_data": "true",  # forward (future) dates
                    "sort_order": "asc",
                },
            )
            page_sha = content_sha256(raw)
            release_dates = _as_list(
                _as_obj(raw, "release/dates").get("release_dates"), "release_dates"
            )
            for rd_obj in release_dates:
                rd = _as_dict(rd_obj, "release_date")
                rd_date = _parse_date(_as_str(rd.get("date"), "release_date.date"), "release_date")
                release_dt = datetime.combine(rd_date, RELEASE_TIME_ET[series], tzinfo=ET)
                if not (start <= release_dt < end):  # half-open per ABC contract
                    continue
                event_id = f"{series.value}_{rd_date.year}_{rd_date.month:02d}"
                events.append(
                    MacroEvent(
                        event_id=event_id,
                        series=series,
                        publisher=SERIES_PUBLISHER[series],
                        release_date=rd_date,
                        release_time_et=release_dt,
                        source_url=release_link
                        or f"{FRED_API_BASE}release/dates?release_id={release_id}",
                        originally_scheduled_release_time_et=release_dt,
                        # Per-event provenance, NOT per-page: one release/dates
                        # response yields many events; sha = SHA256(page_sha || event_id) keeps the
                        # page sha as a revision-detection input while uniquely fingerprinting each.
                        content_bytes_sha=content_sha256(f"{page_sha}||{event_id}".encode()),
                    )
                )
        events.sort(key=lambda e: (e.release_time_et, e.series.value))
        return events

    def _resolve_release(self, fred_series_id: str) -> tuple[int, str | None]:
        """Resolve a series' FRED release_id (+ link) dynamically - no hardcoded id guessing."""
        raw = self._http.get("series/release", {"series_id": fred_series_id})
        releases = _as_list(_as_obj(raw, "series/release").get("releases"), "releases")
        if not releases:
            raise FredMacroError(f"FRED series/release: no releases for {fred_series_id}")
        first = _as_dict(releases[0], "release")
        rid = first.get("id")
        if not isinstance(rid, int):
            raise FredMacroError(f"FRED series/release: release id not int for {fred_series_id}")
        link = first.get("link")
        return rid, link if isinstance(link, str) else None
