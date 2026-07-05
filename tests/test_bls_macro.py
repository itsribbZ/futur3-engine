"""A1.26 BlsMacroSource test suite (fixture-first; no live network).

Test discipline:
- Vendor HTTP injected via FixtureBlsHTTPClient (canned BLS JSON per series id).
- LIVE-source semantics: latest-revised print; value_known_at_iso via injected release_lookup
  (accurate) or a CONSERVATIVE end-of-next-month fallback (safe: never look-ahead).
- hard PIT gate exercised at the boundary through the real fetch_value path.
- Defensive Results parse (BLS docs show both object and 1-element-array shapes).
- Non-monthly periods (M13 annual) + blank values skipped; corrupt non-blank values fail loud.

BLS API v2 contract verified against bls.gov/developers (context7), 2026-05-22.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from futur3.data.macro_source import MacroEventSource, MacroSourceError
from futur3.data.macro_types import MacroSeries
from futur3.data.sources import BlsMacroSource as _ExportedBlsMacroSource
from futur3.data.sources.bls_macro import (
    BLS_SERIES_MAP,
    ET,
    SERIES_PUBLISHER,
    BlsMacroError,
    BlsMacroSource,
    _DefaultBlsHTTPClient,
)
from futur3.data.types import SourceTier, content_sha256

# ============================================================================
# Fixtures / helpers
# ============================================================================


class FixtureBlsHTTPClient:
    """In-memory BLS transport: returns canned bytes per series_id; records calls."""

    def __init__(self, responses: dict[str, bytes]) -> None:
        self._responses = responses
        self.calls: list[str] = []

    def get_series(self, series_id: str) -> bytes:
        self.calls.append(series_id)
        if series_id not in self._responses:
            raise AssertionError(f"no fixture response for series {series_id!r}")
        return self._responses[series_id]


def _bls_response(
    series_id: str,
    data_points: list[dict[str, object]],
    *,
    status: str = "REQUEST_SUCCEEDED",
    results_as_list: bool = False,
) -> bytes:
    inner = {"series": [{"seriesID": series_id, "data": data_points}]}
    results: object = [inner] if results_as_list else inner
    return json.dumps(
        {"status": status, "responseTime": 10, "message": [], "Results": results}
    ).encode()


_NFP_DATA = [
    {"year": "2026", "period": "M06", "periodName": "June", "value": "159200", "footnotes": [{}]},
    {"year": "2026", "period": "M05", "periodName": "May", "value": "159025", "footnotes": []},
]


def _nfp_source(
    *, results_as_list: bool = False, release_lookup: object = None
) -> tuple[BlsMacroSource, bytes]:
    raw = _bls_response("CES0000000001", _NFP_DATA, results_as_list=results_as_list)
    client = FixtureBlsHTTPClient({"CES0000000001": raw})
    src = BlsMacroSource(client, release_lookup=release_lookup)  # type: ignore[arg-type]
    return src, raw


def _lookup(mapping: dict[tuple[MacroSeries, date], datetime]):  # type: ignore[no-untyped-def]
    def fn(series: MacroSeries, ref: date) -> datetime | None:
        return mapping.get((series, ref))

    return fn


# ============================================================================
# TestA1_26_Imports
# ============================================================================


class TestA1_26_Imports:
    def test_source_importable(self) -> None:
        assert BlsMacroSource is not None

    def test_error_extends_macro_source_error(self) -> None:
        assert issubclass(BlsMacroError, MacroSourceError)

    def test_exported_from_sources_package(self) -> None:
        assert _ExportedBlsMacroSource is BlsMacroSource


# ============================================================================
# TestA1_26_Maps
# ============================================================================


class TestA1_26_Maps:
    def test_keys_are_macro_series(self) -> None:
        assert all(isinstance(k, MacroSeries) for k in BLS_SERIES_MAP)

    def test_publisher_map_matches(self) -> None:
        assert set(BLS_SERIES_MAP.keys()) == set(SERIES_PUBLISHER.keys())

    def test_known_ids(self) -> None:
        assert BLS_SERIES_MAP[MacroSeries.NFP] == "CES0000000001"
        assert BLS_SERIES_MAP[MacroSeries.CPI] == "CUUR0000SA0"

    def test_ppi_present(self) -> None:
        assert MacroSeries.PPI in BLS_SERIES_MAP


# ============================================================================
# TestA1_26_Construction
# ============================================================================


class TestA1_26_Construction:
    def test_source_id(self) -> None:
        src, _ = _nfp_source()
        assert src.source_id == "bls_api_v2"

    def test_tier_is_t2_macro(self) -> None:
        src, _ = _nfp_source()
        assert src.tier is SourceTier.T2_MACRO

    def test_from_registration_key(self) -> None:
        src = BlsMacroSource.from_registration_key("somekey")
        assert isinstance(src, BlsMacroSource)

    def test_from_registration_key_keyless(self) -> None:
        src = BlsMacroSource.from_registration_key()
        assert isinstance(src, BlsMacroSource)

    def test_default_client_bad_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="timeout_s must be > 0"):
            _DefaultBlsHTTPClient("key", timeout_s=0)

    def test_repr(self) -> None:
        src, _ = _nfp_source()
        assert repr(src) == "<BlsMacroSource source_id='bls_api_v2' tier=T2_MACRO>"


# ============================================================================
# TestA1_26_FetchValueHappy
# ============================================================================


class TestA1_26_FetchValueHappy:
    def test_returns_latest_monthly(self) -> None:
        src, raw = _nfp_source()
        mv = src.fetch_value("NFP_2026_06", datetime(2026, 9, 1, tzinfo=UTC))
        assert mv is not None
        assert mv.value == Decimal("159200")  # M06, newest monthly (not M05)
        assert mv.series is MacroSeries.NFP
        assert mv.as_of_date == date(2026, 6, 1)
        assert mv.source_id == "bls_api_v2"
        assert mv.content_bytes_sha == content_sha256(raw)


# ============================================================================
# TestA1_26_ConservativeBound (no release_lookup)
# ============================================================================


class TestA1_26_ConservativeBound:
    def test_value_known_at_is_end_of_next_month(self) -> None:
        src, _ = _nfp_source()
        mv = src.fetch_value("NFP_2026_06", datetime(2026, 9, 1, tzinfo=UTC))
        assert mv is not None
        # ref June 2026 -> conservative bound = last day of July @ NFP release time 08:30 ET
        assert mv.value_known_at_iso == datetime(2026, 7, 31, 8, 30, tzinfo=ET)

    def test_blocked_before_conservative_bound(self) -> None:
        src, _ = _nfp_source()
        # mid-July is before end-of-July bound -> PIT blocks (conservative errs toward blocking)
        mv = src.fetch_value("NFP_2026_06", datetime(2026, 7, 15, tzinfo=UTC))
        assert mv is None

    def test_returned_after_conservative_bound(self) -> None:
        src, _ = _nfp_source()
        mv = src.fetch_value("NFP_2026_06", datetime(2026, 8, 15, tzinfo=UTC))
        assert mv is not None


# ============================================================================
# TestA1_26_ReleaseLookup (accurate value_known_at)
# ============================================================================


class TestA1_26_ReleaseLookup:
    # NFP June 2026 actually released first Friday of July = 2026-07-03 08:30 ET == 12:30 UTC
    def _src(self) -> BlsMacroSource:
        lookup = _lookup(
            {(MacroSeries.NFP, date(2026, 6, 1)): datetime(2026, 7, 3, 8, 30, tzinfo=ET)}
        )
        src, _ = _nfp_source(release_lookup=lookup)
        return src

    def test_value_known_at_from_lookup(self) -> None:
        mv = self._src().fetch_value("NFP_2026_06", datetime(2026, 7, 10, tzinfo=UTC))
        assert mv is not None
        assert mv.value_known_at_iso == datetime(2026, 7, 3, 8, 30, tzinfo=ET)

    def test_boundary_at_release_returned(self) -> None:
        mv = self._src().fetch_value("NFP_2026_06", datetime(2026, 7, 3, 12, 30, tzinfo=UTC))
        assert mv is not None  # exactly at release (<=)

    def test_one_minute_before_release_blocked(self) -> None:
        mv = self._src().fetch_value("NFP_2026_06", datetime(2026, 7, 3, 12, 29, tzinfo=UTC))
        assert mv is None

    def test_naive_as_of_raises(self) -> None:
        with pytest.raises(ValueError, match=r"fetch_value as_of_iso must be IANA-TZ-aware"):
            self._src().fetch_value("NFP_2026_06", datetime(2026, 7, 10))


# ============================================================================
# TestA1_26_SkipNonMonthlyAndBlank
# ============================================================================


class TestA1_26_SkipNonMonthlyAndBlank:
    def test_skips_annual_m13(self) -> None:
        data = [
            {
                "year": "2026",
                "period": "M13",
                "periodName": "Annual",
                "value": "158000",
                "footnotes": [],
            },
            {
                "year": "2026",
                "period": "M06",
                "periodName": "June",
                "value": "159200",
                "footnotes": [],
            },
        ]
        client = FixtureBlsHTTPClient({"CES0000000001": _bls_response("CES0000000001", data)})
        mv = BlsMacroSource(client).fetch_value("NFP_2026_06", datetime(2026, 9, 1, tzinfo=UTC))
        assert mv is not None
        assert mv.value == Decimal("159200")
        assert mv.as_of_date == date(2026, 6, 1)

    def test_skips_blank_value(self) -> None:
        data = [
            {"year": "2026", "period": "M06", "periodName": "June", "value": "-", "footnotes": []},
            {
                "year": "2026",
                "period": "M05",
                "periodName": "May",
                "value": "159025",
                "footnotes": [],
            },
        ]
        client = FixtureBlsHTTPClient({"CES0000000001": _bls_response("CES0000000001", data)})
        mv = BlsMacroSource(client).fetch_value("NFP_2026_06", datetime(2026, 9, 1, tzinfo=UTC))
        assert mv is not None
        assert mv.value == Decimal("159025")

    def test_only_annual_returns_none(self) -> None:
        data = [
            {
                "year": "2026",
                "period": "M13",
                "periodName": "Annual",
                "value": "158000",
                "footnotes": [],
            }
        ]
        client = FixtureBlsHTTPClient({"CES0000000001": _bls_response("CES0000000001", data)})
        mv = BlsMacroSource(client).fetch_value("NFP_2026_06", datetime(2026, 9, 1, tzinfo=UTC))
        assert mv is None

    def test_empty_data_returns_none(self) -> None:
        client = FixtureBlsHTTPClient({"CES0000000001": _bls_response("CES0000000001", [])})
        mv = BlsMacroSource(client).fetch_value("NFP_2026_06", datetime(2026, 9, 1, tzinfo=UTC))
        assert mv is None


# ============================================================================
# TestA1_26_ResultsShape (defensive: object OR 1-element array)
# ============================================================================


class TestA1_26_ResultsShape:
    def test_results_as_object(self) -> None:
        src, _ = _nfp_source(results_as_list=False)
        mv = src.fetch_value("NFP_2026_06", datetime(2026, 9, 1, tzinfo=UTC))
        assert mv is not None and mv.value == Decimal("159200")

    def test_results_as_list(self) -> None:
        src, _ = _nfp_source(results_as_list=True)
        mv = src.fetch_value("NFP_2026_06", datetime(2026, 9, 1, tzinfo=UTC))
        assert mv is not None and mv.value == Decimal("159200")


# ============================================================================
# TestA1_26_Unmapped
# ============================================================================


class TestA1_26_Unmapped:
    def test_unmapped_series_returns_none_no_http(self) -> None:
        client = FixtureBlsHTTPClient({})
        mv = BlsMacroSource(client).fetch_value("JOLTS_2026_06", datetime(2026, 9, 1, tzinfo=UTC))
        assert mv is None
        assert client.calls == []

    def test_unknown_event_id_returns_none(self) -> None:
        client = FixtureBlsHTTPClient({})
        mv = BlsMacroSource(client).fetch_value("BOGUS_2026", datetime(2026, 9, 1, tzinfo=UTC))
        assert mv is None
        assert client.calls == []


# ============================================================================
# TestA1_26_Errors
# ============================================================================


class TestA1_26_Errors:
    def test_status_not_succeeded_raises(self) -> None:
        raw = _bls_response("CES0000000001", _NFP_DATA, status="REQUEST_NOT_PROCESSED")
        client = FixtureBlsHTTPClient({"CES0000000001": raw})
        with pytest.raises(BlsMacroError, match="not succeeded"):
            BlsMacroSource(client).fetch_value("NFP_2026_06", datetime(2026, 9, 1, tzinfo=UTC))

    def test_malformed_json_raises(self) -> None:
        client = FixtureBlsHTTPClient({"CES0000000001": b"<<<not json"})
        with pytest.raises(BlsMacroError, match="malformed JSON"):
            BlsMacroSource(client).fetch_value("NFP_2026_06", datetime(2026, 9, 1, tzinfo=UTC))

    def test_corrupt_value_fails_loud(self) -> None:
        data = [
            {"year": "2026", "period": "M06", "periodName": "June", "value": "xyz", "footnotes": []}
        ]
        client = FixtureBlsHTTPClient({"CES0000000001": _bls_response("CES0000000001", data)})
        with pytest.raises(BlsMacroError, match="not parseable as Decimal"):
            BlsMacroSource(client).fetch_value("NFP_2026_06", datetime(2026, 9, 1, tzinfo=UTC))

    def test_bad_year_raises(self) -> None:
        data = [
            {
                "year": "20XX",
                "period": "M06",
                "periodName": "June",
                "value": "159200",
                "footnotes": [],
            }
        ]
        client = FixtureBlsHTTPClient({"CES0000000001": _bls_response("CES0000000001", data)})
        with pytest.raises(BlsMacroError, match="year not numeric"):
            BlsMacroSource(client).fetch_value("NFP_2026_06", datetime(2026, 9, 1, tzinfo=UTC))


# ============================================================================
# TestA1_26_UpcomingEvents (value-only source)
# ============================================================================


class TestA1_26_UpcomingEvents:
    def test_returns_empty(self) -> None:
        src, _ = _nfp_source()
        events = src.upcoming_events(
            datetime(2026, 7, 1, tzinfo=ET), datetime(2026, 9, 1, tzinfo=ET)
        )
        assert events == []

    def test_naive_start_raises(self) -> None:
        src, _ = _nfp_source()
        with pytest.raises(ValueError, match="start must be IANA-TZ-aware"):
            src.upcoming_events(datetime(2026, 7, 1), datetime(2026, 9, 1, tzinfo=ET))

    def test_end_before_start_raises(self) -> None:
        src, _ = _nfp_source()
        with pytest.raises(ValueError, match="end must be > start"):
            src.upcoming_events(datetime(2026, 9, 1, tzinfo=ET), datetime(2026, 7, 1, tzinfo=ET))


# ============================================================================
# TestA1_26_ABCCompliance
# ============================================================================


class TestA1_26_ABCCompliance:
    def test_is_macro_event_source(self) -> None:
        src, _ = _nfp_source()
        assert isinstance(src, MacroEventSource)

    def test_schedule_drift_check_default_empty(self) -> None:
        src, _ = _nfp_source()
        assert src.schedule_drift_check() == []

    def test_healthcheck_default_true(self) -> None:
        src, _ = _nfp_source()
        assert src.healthcheck() is True
