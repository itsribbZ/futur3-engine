"""A1.25 FredMacroSource test suite (fixture-first; no live network).

Test discipline:
- Vendor HTTP injected via FixtureFredHTTPClient (canned FRED JSON keyed by endpoint).
- ALFRED vintage path: realtime_start==realtime_end==as_of_date; per-obs realtime_start drives
  value_known_at_iso (anchored to RELEASE_TIME_ET).
- hard PIT gate (bug class 5) exercised at the boundary through the real fetch_value path.
- Missing-sentinel ('.') skipped, never fabricated.
- upcoming_events parses release/dates with dynamic series->release_id resolution; half-open.

FRED contract verified against fred.stlouisfed.org/docs/api (context7), 2026-05-22.
Live smoke gated on operator setup (free FRED API key) - see source from_api_key.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, date, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from futur3.data.macro_source import MacroEventSource, MacroSourceError
from futur3.data.macro_types import MacroPublisher, MacroSeries
from futur3.data.sources import FredMacroSource as _ExportedFredMacroSource
from futur3.data.sources.fred_macro import (
    FRED_SERIES_MAP,
    SERIES_PUBLISHER,
    FredMacroError,
    FredMacroSource,
    _DefaultFredHTTPClient,
    _series_from_event_id,
)
from futur3.data.types import SourceTier, content_sha256

ET = ZoneInfo("America/New_York")

# ============================================================================
# Fixtures / helpers
# ============================================================================


class FixtureFredHTTPClient:
    """In-memory FRED transport: returns canned bytes per endpoint; records every call."""

    def __init__(self, responses: dict[str, bytes]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, dict[str, str]]] = []

    def get(self, endpoint: str, params: Mapping[str, str]) -> bytes:
        self.calls.append((endpoint, dict(params)))
        if endpoint not in self._responses:
            raise AssertionError(f"no fixture response configured for endpoint {endpoint!r}")
        return self._responses[endpoint]

    def params_for(self, endpoint: str) -> list[dict[str, str]]:
        return [p for (ep, p) in self.calls if ep == endpoint]


def _obs_response(observations: list[dict[str, str]], realtime: str = "2026-06-10") -> bytes:
    return json.dumps(
        {
            "realtime_start": realtime,
            "realtime_end": realtime,
            "observation_start": "1776-07-04",
            "observation_end": "9999-12-31",
            "units": "lin",
            "output_type": 1,
            "file_type": "json",
            "order_by": "observation_date",
            "sort_order": "desc",
            "count": len(observations),
            "offset": 0,
            "limit": 12,
            "observations": observations,
        }
    ).encode()


def _series_release_response(
    release_id: int = 50, link: str = "https://www.bls.gov/news.release/empsit.htm"
) -> bytes:
    return json.dumps(
        {
            "realtime_start": "2026-06-10",
            "realtime_end": "2026-06-10",
            "releases": [
                {
                    "id": release_id,
                    "realtime_start": "2026-06-10",
                    "realtime_end": "2026-06-10",
                    "name": "Employment Situation",
                    "press_release": True,
                    "link": link,
                }
            ],
        }
    ).encode()


def _release_dates_response(dates: list[str], release_id: int = 50) -> bytes:
    return json.dumps(
        {
            "realtime_start": "2026-06-10",
            "realtime_end": "9999-12-31",
            "order_by": "release_date",
            "sort_order": "asc",
            "count": len(dates),
            "offset": 0,
            "limit": 10000,
            "release_dates": [{"release_id": release_id, "date": d} for d in dates],
        }
    ).encode()


# Two NFP observations, newest-first (sort_order=desc). Latest non-missing = 2026-05-01.
_NFP_OBS = [
    {
        "realtime_start": "2026-06-06",
        "realtime_end": "9999-12-31",
        "date": "2026-05-01",
        "value": "159200",
    },
    {
        "realtime_start": "2026-05-02",
        "realtime_end": "2026-06-05",
        "date": "2026-04-01",
        "value": "159025",
    },
]


def _nfp_source() -> tuple[FredMacroSource, FixtureFredHTTPClient, bytes]:
    raw = _obs_response(_NFP_OBS)
    client = FixtureFredHTTPClient({"series/observations": raw})
    return FredMacroSource(client), client, raw


# ============================================================================
# TestA1_25_Imports
# ============================================================================


class TestA1_25_Imports:
    def test_source_importable(self) -> None:
        assert FredMacroSource is not None

    def test_error_extends_macro_source_error(self) -> None:
        assert issubclass(FredMacroError, MacroSourceError)

    def test_exported_from_sources_package(self) -> None:
        assert _ExportedFredMacroSource is FredMacroSource


# ============================================================================
# TestA1_25_Maps
# ============================================================================


class TestA1_25_Maps:
    def test_series_map_keys_are_macro_series(self) -> None:
        assert all(isinstance(k, MacroSeries) for k in FRED_SERIES_MAP)

    def test_every_mapped_series_has_publisher(self) -> None:
        assert set(FRED_SERIES_MAP.keys()) == set(SERIES_PUBLISHER.keys())

    def test_narrative_series_not_mapped(self) -> None:
        for s in (MacroSeries.FOMC_STATEMENT, MacroSeries.BEIGE_BOOK, MacroSeries.ISM_MFG):
            assert s not in FRED_SERIES_MAP

    def test_known_series_ids(self) -> None:
        assert FRED_SERIES_MAP[MacroSeries.NFP] == "PAYEMS"
        assert FRED_SERIES_MAP[MacroSeries.CPI] == "CPIAUCSL"
        assert FRED_SERIES_MAP[MacroSeries.GDP_ADVANCE] == "GDPC1"


# ============================================================================
# TestA1_25_SeriesFromEventId
# ============================================================================


class TestA1_25_SeriesFromEventId:
    def test_simple_prefix(self) -> None:
        assert _series_from_event_id("NFP_2026_06") is MacroSeries.NFP

    def test_compound_series_longest_match(self) -> None:
        assert _series_from_event_id("GDP_ADVANCE_2026Q1") is MacroSeries.GDP_ADVANCE

    def test_exact_value(self) -> None:
        assert _series_from_event_id("NFP") is MacroSeries.NFP

    def test_narrative_series_resolves(self) -> None:
        # FOMC_STATEMENT IS a MacroSeries (just not FRED-mapped)
        assert _series_from_event_id("FOMC_STATEMENT_2026_06") is MacroSeries.FOMC_STATEMENT

    def test_unknown_returns_none(self) -> None:
        assert _series_from_event_id("BOGUS_2026_06") is None


# ============================================================================
# TestA1_25_Construction
# ============================================================================


class TestA1_25_Construction:
    def test_construct_with_client(self) -> None:
        source, _, _ = _nfp_source()
        assert source.source_id == "fred_alfred"

    def test_tier_is_aggregator(self) -> None:
        source, _, _ = _nfp_source()
        assert source.tier is SourceTier.T3_AGGREGATOR

    def test_from_api_key_builds_source(self) -> None:
        source = FredMacroSource.from_api_key("abcdefghijklmnopqrstuvwxyz123456")
        assert isinstance(source, FredMacroSource)

    def test_default_client_empty_key_raises(self) -> None:
        with pytest.raises(ValueError, match="api_key must be non-empty"):
            _DefaultFredHTTPClient("")

    def test_default_client_bad_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="timeout_s must be > 0"):
            _DefaultFredHTTPClient("key", timeout_s=0)

    def test_repr(self) -> None:
        source, _, _ = _nfp_source()
        assert repr(source) == "<FredMacroSource source_id='fred_alfred' tier=T3_AGGREGATOR>"


# ============================================================================
# TestA1_25_FetchValueHappy
# ============================================================================


class TestA1_25_FetchValueHappy:
    def test_returns_latest_value(self) -> None:
        source, _, raw = _nfp_source()
        mv = source.fetch_value("NFP_2026_06", datetime(2026, 6, 10, 16, 0, tzinfo=UTC))
        assert mv is not None
        assert mv.value == Decimal("159200")
        assert mv.series is MacroSeries.NFP
        assert mv.as_of_date == date(2026, 5, 1)
        assert mv.source_id == "fred_alfred"
        assert mv.content_bytes_sha == content_sha256(raw)

    def test_value_known_at_anchored_to_release_time(self) -> None:
        source, _, _ = _nfp_source()
        mv = source.fetch_value("NFP_2026_06", datetime(2026, 6, 10, 16, 0, tzinfo=UTC))
        assert mv is not None
        # realtime_start 2026-06-06 anchored to NFP release time 08:30 ET
        assert mv.value_known_at_iso == datetime(2026, 6, 6, 8, 30, tzinfo=ET)

    def test_vintage_as_of_is_et_date_of_as_of(self) -> None:
        source, _, _ = _nfp_source()
        # 16:00 UTC == 12:00 EDT -> 2026-06-10
        mv = source.fetch_value("NFP_2026_06", datetime(2026, 6, 10, 16, 0, tzinfo=UTC))
        assert mv is not None
        assert mv.vintage_as_of == date(2026, 6, 10)

    def test_realtime_params_pin_vintage(self) -> None:
        source, client, _ = _nfp_source()
        source.fetch_value("NFP_2026_06", datetime(2026, 6, 10, 16, 0, tzinfo=UTC))
        params = client.params_for("series/observations")[0]
        assert params["series_id"] == "PAYEMS"
        assert params["realtime_start"] == "2026-06-10"
        assert params["realtime_end"] == "2026-06-10"
        assert params["sort_order"] == "desc"


# ============================================================================
# TestA1_25_FetchValuePIT (bug class 5)
# ============================================================================


class TestA1_25_FetchValuePIT:
    # value_known_at = 2026-06-06 08:30 EDT == 12:30 UTC
    def test_after_publication_returned(self) -> None:
        source, _, _ = _nfp_source()
        mv = source.fetch_value("NFP_2026_06", datetime(2026, 6, 6, 12, 30, tzinfo=UTC))
        assert mv is not None  # exactly at publication boundary (<=) -> consumable

    def test_one_minute_before_publication_blocked(self) -> None:
        source, _, _ = _nfp_source()
        mv = source.fetch_value("NFP_2026_06", datetime(2026, 6, 6, 12, 29, tzinfo=UTC))
        assert mv is None

    def test_well_after_returned(self) -> None:
        source, _, _ = _nfp_source()
        mv = source.fetch_value("NFP_2026_06", datetime(2026, 6, 30, 12, 0, tzinfo=UTC))
        assert mv is not None

    def test_naive_as_of_raises(self) -> None:
        source, _, _ = _nfp_source()
        with pytest.raises(ValueError, match=r"fetch_value as_of_iso must be IANA-TZ-aware"):
            source.fetch_value("NFP_2026_06", datetime(2026, 6, 10, 16, 0))


# ============================================================================
# TestA1_25_FetchValueMissing
# ============================================================================


class TestA1_25_FetchValueMissing:
    def test_skips_missing_sentinel(self) -> None:
        obs = [
            {
                "realtime_start": "2026-06-06",
                "realtime_end": "9999-12-31",
                "date": "2026-05-01",
                "value": ".",
            },
            {
                "realtime_start": "2026-05-02",
                "realtime_end": "9999-12-31",
                "date": "2026-04-01",
                "value": "159025",
            },
        ]
        client = FixtureFredHTTPClient({"series/observations": _obs_response(obs)})
        mv = FredMacroSource(client).fetch_value(
            "NFP_2026_06", datetime(2026, 6, 10, 16, 0, tzinfo=UTC)
        )
        assert mv is not None
        assert mv.value == Decimal("159025")
        assert mv.as_of_date == date(2026, 4, 1)

    def test_all_missing_returns_none(self) -> None:
        obs = [
            {
                "realtime_start": "2026-06-06",
                "realtime_end": "9999-12-31",
                "date": "2026-05-01",
                "value": ".",
            },
            {
                "realtime_start": "2026-05-02",
                "realtime_end": "9999-12-31",
                "date": "2026-04-01",
                "value": ".",
            },
        ]
        client = FixtureFredHTTPClient({"series/observations": _obs_response(obs)})
        mv = FredMacroSource(client).fetch_value(
            "NFP_2026_06", datetime(2026, 6, 10, 16, 0, tzinfo=UTC)
        )
        assert mv is None

    def test_empty_observations_returns_none(self) -> None:
        client = FixtureFredHTTPClient({"series/observations": _obs_response([])})
        mv = FredMacroSource(client).fetch_value(
            "NFP_2026_06", datetime(2026, 6, 10, 16, 0, tzinfo=UTC)
        )
        assert mv is None


# ============================================================================
# TestA1_25_FetchValueUnmapped
# ============================================================================


class TestA1_25_FetchValueUnmapped:
    def test_narrative_series_returns_none_no_http(self) -> None:
        client = FixtureFredHTTPClient({})  # no responses -> any HTTP call would AssertionError
        mv = FredMacroSource(client).fetch_value(
            "FOMC_STATEMENT_2026_06", datetime(2026, 6, 10, 16, 0, tzinfo=UTC)
        )
        assert mv is None
        assert client.calls == []  # short-circuits before any network call

    def test_unknown_event_id_returns_none(self) -> None:
        client = FixtureFredHTTPClient({})
        mv = FredMacroSource(client).fetch_value(
            "BOGUS_2026_06", datetime(2026, 6, 10, 16, 0, tzinfo=UTC)
        )
        assert mv is None
        assert client.calls == []


# ============================================================================
# TestA1_25_FetchValueErrors
# ============================================================================


class TestA1_25_FetchValueErrors:
    def _fetch(self, raw: bytes) -> None:
        client = FixtureFredHTTPClient({"series/observations": raw})
        FredMacroSource(client).fetch_value("NFP_2026_06", datetime(2026, 6, 10, 16, 0, tzinfo=UTC))

    def test_malformed_json_raises(self) -> None:
        with pytest.raises(FredMacroError, match="malformed JSON"):
            self._fetch(b"<<<not json>>>")

    def test_missing_observations_array_raises(self) -> None:
        with pytest.raises(FredMacroError, match="expected JSON array"):
            self._fetch(json.dumps({"count": 0}).encode())

    def test_non_decimal_value_raises(self) -> None:
        obs = [
            {
                "realtime_start": "2026-06-06",
                "realtime_end": "9999-12-31",
                "date": "2026-05-01",
                "value": "not_a_number",
            }
        ]
        with pytest.raises(FredMacroError, match="not parseable as Decimal"):
            self._fetch(_obs_response(obs))

    def test_bad_date_raises(self) -> None:
        obs = [
            {
                "realtime_start": "2026-06-06",
                "realtime_end": "9999-12-31",
                "date": "2026-13-99",
                "value": "159200",
            }
        ]
        with pytest.raises(FredMacroError, match="date not ISO"):
            self._fetch(_obs_response(obs))


# ============================================================================
# TestA1_25_UpcomingEvents
# ============================================================================


class TestA1_25_UpcomingEvents:
    def _calendar_source(self, dates: list[str]) -> tuple[FredMacroSource, FixtureFredHTTPClient]:
        client = FixtureFredHTTPClient(
            {
                "series/release": _series_release_response(),
                "release/dates": _release_dates_response(dates),
            }
        )
        return FredMacroSource(client), client

    def test_produces_events_for_all_mapped_series(self) -> None:
        source, _ = self._calendar_source(["2026-07-02", "2026-08-06"])
        events = source.upcoming_events(
            datetime(2026, 7, 1, tzinfo=ET), datetime(2026, 9, 1, tzinfo=ET)
        )
        # 9 mapped series x 2 dates
        assert len(events) == len(FRED_SERIES_MAP) * 2

    def test_event_fields_correct(self) -> None:
        source, _ = self._calendar_source(["2026-07-02"])
        events = source.upcoming_events(
            datetime(2026, 7, 1, tzinfo=ET), datetime(2026, 9, 1, tzinfo=ET)
        )
        nfp = next(e for e in events if e.series is MacroSeries.NFP)
        assert nfp.publisher is MacroPublisher.BLS
        assert nfp.release_time_et == datetime(2026, 7, 2, 8, 30, tzinfo=ET)
        assert nfp.release_date == date(2026, 7, 2)
        assert nfp.event_id == "NFP_2026_07"
        assert len(nfp.content_bytes_sha) == 64

    def test_jolts_at_1000(self) -> None:
        source, _ = self._calendar_source(["2026-07-02"])
        events = source.upcoming_events(
            datetime(2026, 7, 1, tzinfo=ET), datetime(2026, 9, 1, tzinfo=ET)
        )
        jolts = next(e for e in events if e.series is MacroSeries.JOLTS)
        assert jolts.release_time_et == datetime(2026, 7, 2, 10, 0, tzinfo=ET)

    def test_sorted_by_release_time_then_series(self) -> None:
        source, _ = self._calendar_source(["2026-07-02", "2026-08-06"])
        events = source.upcoming_events(
            datetime(2026, 7, 1, tzinfo=ET), datetime(2026, 9, 1, tzinfo=ET)
        )
        keys = [(e.release_time_et, e.series.value) for e in events]
        assert keys == sorted(keys)

    def test_half_open_lower_bound(self) -> None:
        # window starts 09:00 ET on 07-02: the 08:30 series are excluded; only JOLTS (10:00) fits
        source, _ = self._calendar_source(["2026-07-02"])
        events = source.upcoming_events(
            datetime(2026, 7, 2, 9, 0, tzinfo=ET), datetime(2026, 7, 3, tzinfo=ET)
        )
        assert len(events) == 1
        assert events[0].series is MacroSeries.JOLTS

    def test_per_event_content_sha_distinct(self) -> None:
        """Regression: many events from one release/dates page must each carry a
        DISTINCT content_bytes_sha (per-event SHA256(page_sha || event_id)), never a shared one."""
        source, _ = self._calendar_source(["2026-07-02", "2026-08-06"])
        events = source.upcoming_events(
            datetime(2026, 7, 1, tzinfo=ET), datetime(2026, 9, 1, tzinfo=ET)
        )
        shas = [e.content_bytes_sha for e in events]
        assert len(shas) == len(set(shas))  # all distinct

    def test_resolves_release_id_dynamically(self) -> None:
        source, client = self._calendar_source(["2026-07-02"])
        source.upcoming_events(datetime(2026, 7, 1, tzinfo=ET), datetime(2026, 9, 1, tzinfo=ET))
        # series/release called once per mapped series to resolve release_id
        assert len(client.params_for("series/release")) == len(FRED_SERIES_MAP)
        rd_params = client.params_for("release/dates")[0]
        assert rd_params["include_release_dates_with_no_data"] == "true"


# ============================================================================
# TestA1_25_UpcomingEventsErrors
# ============================================================================


class TestA1_25_UpcomingEventsErrors:
    def test_empty_releases_raises(self) -> None:
        client = FixtureFredHTTPClient({"series/release": json.dumps({"releases": []}).encode()})
        with pytest.raises(FredMacroError, match="no releases"):
            FredMacroSource(client).upcoming_events(
                datetime(2026, 7, 1, tzinfo=ET), datetime(2026, 9, 1, tzinfo=ET)
            )

    def test_release_id_not_int_raises(self) -> None:
        bad = json.dumps({"releases": [{"id": "50", "name": "x", "link": "y"}]}).encode()
        client = FixtureFredHTTPClient({"series/release": bad})
        with pytest.raises(FredMacroError, match="release id not int"):
            FredMacroSource(client).upcoming_events(
                datetime(2026, 7, 1, tzinfo=ET), datetime(2026, 9, 1, tzinfo=ET)
            )

    def test_naive_start_raises(self) -> None:
        client = FixtureFredHTTPClient({})
        with pytest.raises(ValueError, match="start must be IANA-TZ-aware"):
            FredMacroSource(client).upcoming_events(
                datetime(2026, 7, 1), datetime(2026, 9, 1, tzinfo=ET)
            )

    def test_end_before_start_raises(self) -> None:
        client = FixtureFredHTTPClient({})
        with pytest.raises(ValueError, match="end must be > start"):
            FredMacroSource(client).upcoming_events(
                datetime(2026, 9, 1, tzinfo=ET), datetime(2026, 7, 1, tzinfo=ET)
            )


# ============================================================================
# TestA1_25_ABCCompliance
# ============================================================================


class TestA1_25_ABCCompliance:
    def test_is_macro_event_source(self) -> None:
        source, _, _ = _nfp_source()
        assert isinstance(source, MacroEventSource)

    def test_schedule_drift_check_default_empty(self) -> None:
        source, _, _ = _nfp_source()
        assert source.schedule_drift_check() == []

    def test_healthcheck_default_true(self) -> None:
        source, _, _ = _nfp_source()
        assert source.healthcheck() is True
