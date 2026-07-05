"""CFTC concrete COT sources test suite (Ship 2: sources/cftc.py).

Test discipline:
- CFTCSocrataSource parses Socrata JSON -> normalized COTReport (DISAGGREGATED + TFF column maps),
  fixtures-only (ZERO network).
- Schema-drift fail-closed (bug class 9): a missing expected column raises COTSchemaDriftError.
- Per-row provenance sha (no per-page collision); as_of_iso from the injected clock.
- InMemoryCOTSource replays deterministically + inherits the hard PIT gate (reports_known_at).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from futur3.data.cot_source import COTSchemaDriftError, COTSourceError
from futur3.data.cot_types import COTReport, COTReportFlavor
from futur3.data.sources.cftc import (
    CftcSocrataError,
    CFTCSocrataSource,
    InMemoryCOTSource,
)
from futur3.data.types import SHA256_HEX_LENGTH, SourceTier

ET = ZoneInfo("America/New_York")
_FIX = Path(__file__).parent / "fixtures" / "cftc"
_DISAGG_ID = "72hh-3qpy"
_TFF_ID = "gpe5-46if"
_CLOCK_T = datetime(2026, 5, 22, 21, 0, tzinfo=UTC)
_Y0 = date(2026, 1, 1)
_Y1 = date(2026, 12, 31)


class _FixtureCFTCHTTPClient:
    """CFTCHTTPClient backed by on-disk Socrata fixture JSON, keyed by dataset id in the URL."""

    def __init__(self, dataset_to_fixture: dict[str, str]) -> None:
        self._map = dataset_to_fixture
        self.last_url: str | None = None
        self.last_params: Mapping[str, str] | None = None

    def get(self, url: str, params: Mapping[str, str]) -> bytes:
        self.last_url = url
        self.last_params = dict(params)
        for dataset_id, filename in self._map.items():
            if dataset_id in url:
                return (_FIX / filename).read_bytes()
        raise KeyError(f"no fixture mapped for url {url!r}")


class _BadJSONClient:
    def get(self, url: str, params: Mapping[str, str]) -> bytes:
        return b"{not valid json"


class _FixedClock:
    def __init__(self, fixed: datetime) -> None:
        self._fixed = fixed

    def now_utc(self) -> datetime:
        return self._fixed


def _cl_source() -> tuple[CFTCSocrataSource, _FixtureCFTCHTTPClient]:
    client = _FixtureCFTCHTTPClient({_DISAGG_ID: "disaggregated_cl.json"})
    return CFTCSocrataSource(client, clock=_FixedClock(_CLOCK_T)), client


def _cl_reports() -> list[COTReport]:
    src, _ = _cl_source()
    return src.fetch_reports("067651", COTReportFlavor.DISAGGREGATED, _Y0, _Y1)


class TestCFTCSocrataParse:
    def test_parses_disaggregated_cl_ascending(self) -> None:
        reports = _cl_reports()
        assert [r.report_date for r in reports] == [
            date(2026, 4, 28),
            date(2026, 5, 5),
            date(2026, 5, 12),
            date(2026, 5, 19),
        ]

    def test_disaggregated_normalizes_managed_money_and_producer(self) -> None:
        first = _cl_reports()[0]
        assert first.spec_long == 280_000  # m_money long
        assert first.spec_short == 160_000  # m_money short
        assert first.spec_net == 120_000
        assert first.comm_long == 600_000  # prod_merc long
        assert first.comm_short == 760_000  # prod_merc short
        assert first.comm_net == -160_000
        assert first.open_interest_all == 1_800_000
        assert first.total_traders == 350
        assert first.flavor is COTReportFlavor.DISAGGREGATED
        assert first.cftc_contract_market_code == "067651"

    def test_tff_normalizes_leveraged_funds_and_dealer(self) -> None:
        client = _FixtureCFTCHTTPClient({_TFF_ID: "tff_es.json"})
        src = CFTCSocrataSource(client, clock=_FixedClock(_CLOCK_T))
        reports = src.fetch_reports("13874A", COTReportFlavor.TFF, _Y0, _Y1)
        first = reports[0]
        assert first.spec_long == 400_000  # lev_money long
        assert first.spec_short == 500_000  # lev_money short
        assert first.spec_net == -100_000
        assert first.comm_long == 300_000  # dealer long
        assert first.comm_net == 50_000  # dealer 300k - 250k
        assert first.cftc_contract_market_code == "13874A"

    def test_value_known_at_is_following_friday_1530_et(self) -> None:
        first = _cl_reports()[0]  # Tuesday 2026-04-28
        assert first.value_known_at_iso == datetime(2026, 5, 1, 15, 30, tzinfo=ET)  # Friday

    def test_as_of_iso_from_injected_clock(self) -> None:
        assert all(r.as_of_iso == _CLOCK_T for r in _cl_reports())

    def test_per_row_sha_distinct_and_hex(self) -> None:
        shas = [r.content_bytes_sha for r in _cl_reports()]
        assert len(set(shas)) == len(shas)  # no per-page collision
        assert all(len(s) == SHA256_HEX_LENGTH for s in shas)

    def test_query_params_carry_code_and_dates(self) -> None:
        src, client = _cl_source()
        src.fetch_reports(
            "067651", COTReportFlavor.DISAGGREGATED, date(2026, 2, 1), date(2026, 6, 1)
        )
        assert client.last_url is not None
        assert _DISAGG_ID in client.last_url
        assert client.last_params is not None
        where = client.last_params["$where"]
        assert "'067651'" in where
        assert "'2026-02-01'" in where
        assert "'2026-06-01'" in where


class TestCFTCSocrataErrors:
    def test_schema_drift_raises(self) -> None:
        client = _FixtureCFTCHTTPClient({_DISAGG_ID: "disaggregated_cl_drift.json"})
        src = CFTCSocrataSource(client, clock=_FixedClock(_CLOCK_T))
        with pytest.raises(COTSchemaDriftError, match="m_money_positions_long_all"):
            src.fetch_reports("067651", COTReportFlavor.DISAGGREGATED, _Y0, _Y1)

    def test_unsupported_flavor_raises(self) -> None:
        src, _ = _cl_source()
        with pytest.raises(COTSourceError, match="supplemental"):
            src.fetch_reports("067651", COTReportFlavor.SUPPLEMENTAL, _Y0, _Y1)

    def test_end_before_start_raises(self) -> None:
        src, _ = _cl_source()
        with pytest.raises(ValueError, match="end must be >= start"):
            src.fetch_reports(
                "067651", COTReportFlavor.DISAGGREGATED, date(2026, 6, 1), date(2026, 1, 1)
            )

    def test_malformed_json_raises(self) -> None:
        src = CFTCSocrataSource(_BadJSONClient(), clock=_FixedClock(_CLOCK_T))
        with pytest.raises(CftcSocrataError, match="malformed JSON"):
            src.fetch_reports("067651", COTReportFlavor.DISAGGREGATED, _Y0, _Y1)


class TestFromAppToken:
    def test_constructs_with_token(self) -> None:
        src = CFTCSocrataSource.from_app_token("free-token")
        assert src.source_id == "cftc_socrata"
        assert src.tier is SourceTier.T2_MACRO

    def test_constructs_without_token(self) -> None:
        assert CFTCSocrataSource.from_app_token().source_id == "cftc_socrata"

    def test_bad_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="timeout_s"):
            CFTCSocrataSource.from_app_token("tok", timeout_s=0.0)


class TestInMemoryCOTSource:
    def test_replays_filtered_ascending(self) -> None:
        src = InMemoryCOTSource(_cl_reports())
        got = src.fetch_reports(
            "067651", COTReportFlavor.DISAGGREGATED, date(2026, 5, 1), date(2026, 5, 13)
        )
        assert [r.report_date for r in got] == [date(2026, 5, 5), date(2026, 5, 12)]

    def test_filters_by_contract_and_flavor(self) -> None:
        src = InMemoryCOTSource(_cl_reports())
        # CL reports queried as TFF -> none match.
        assert src.fetch_reports("067651", COTReportFlavor.TFF, _Y0, _Y1) == []
        assert src.fetch_reports("999999", COTReportFlavor.DISAGGREGATED, _Y0, _Y1) == []

    def test_reports_known_at_applies_pit_through_replay(self) -> None:
        src = InMemoryCOTSource(_cl_reports())
        # Thursday 2026-05-21: the 2026-05-19 snapshot (publishes Fri 05-22) is NOT yet known.
        as_of = datetime(2026, 5, 21, 12, 0, tzinfo=ET)
        known = src.reports_known_at(
            "067651", COTReportFlavor.DISAGGREGATED, as_of, since=date(2026, 1, 1)
        )
        assert [r.report_date for r in known] == [
            date(2026, 4, 28),
            date(2026, 5, 5),
            date(2026, 5, 12),
        ]

    def test_tier_is_t4_derived(self) -> None:
        assert InMemoryCOTSource([]).tier is SourceTier.T4_DERIVED

    def test_custom_source_id(self) -> None:
        assert InMemoryCOTSource([], source_id="cot_replay_2026").source_id == "cot_replay_2026"

    def test_empty_source_id_raises(self) -> None:
        with pytest.raises(ValueError, match="source_id"):
            InMemoryCOTSource([], source_id="")
