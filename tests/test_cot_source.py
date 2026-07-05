"""COT source ABC + PIT gate test suite (Ship 1: cot_source).

Per internal design notes:
- value_known_at_for_report: Tue/Mon snapshot -> that ISO week's Friday 15:30 ET; weekend raises.
- enforce_cot_pit_gate: the hard Tue->Fri 3-day blackout (bug class 5 / look-ahead).
- COTSource ABC abstractness + the CONCRETE reports_known_at PIT accessor (structural guard:
  a subclass implementing only the raw fetch_reports still cannot leak an unpublished snapshot).
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from futur3.data.cot_source import (
    COT_RELEASE_TIME_ET,
    COTSource,
    COTSourceError,
    enforce_cot_pit_gate,
    value_known_at_for_report,
)
from futur3.data.cot_types import COTReport, COTReportFlavor
from futur3.data.source import DataSourceError
from futur3.data.types import SourceTier, content_sha256

ET = ZoneInfo("America/New_York")
_SHA = content_sha256(b"cot-fixture")
_CL = "067651"
_GC = "088691"


def _report(
    report_date: date,
    *,
    market_code: str = _CL,
    flavor: COTReportFlavor = COTReportFlavor.DISAGGREGATED,
) -> COTReport:
    return COTReport(
        cftc_contract_market_code=market_code,
        flavor=flavor,
        report_date=report_date,
        value_known_at_iso=value_known_at_for_report(report_date),
        open_interest_all=500_000,
        spec_long=100_000,
        spec_short=60_000,
        comm_long=200_000,
        comm_short=240_000,
        source_id="fixture",
        as_of_iso=datetime(2026, 5, 22, 20, 0, tzinfo=UTC),
        content_bytes_sha=_SHA,
    )


class _FixtureCOTSource(COTSource):
    """In-memory COTSource for ABC-contract + PIT tests.

    `fetch_reports` returns rows DESCENDING by report_date on purpose, so the tests prove that
    `reports_known_at` re-sorts ascending (determinism) and applies the PIT gate.
    """

    def __init__(self, reports: list[COTReport]) -> None:
        self._reports = reports
        self.last_fetch_args: tuple[str, COTReportFlavor, date, date] | None = None

    @property
    def source_id(self) -> str:
        return "fixture_cot"

    @property
    def tier(self) -> SourceTier:
        return SourceTier.T2_MACRO

    def fetch_reports(
        self,
        cftc_contract_market_code: str,
        flavor: COTReportFlavor,
        start: date,
        end: date,
    ) -> list[COTReport]:
        self.last_fetch_args = (cftc_contract_market_code, flavor, start, end)
        hits = [
            r
            for r in self._reports
            if r.cftc_contract_market_code == cftc_contract_market_code
            and r.flavor == flavor
            and start <= r.report_date <= end
        ]
        return sorted(hits, key=lambda r: r.report_date, reverse=True)


class TestValueKnownAt:
    def test_tuesday_maps_to_same_week_friday_1530_et(self) -> None:
        got = value_known_at_for_report(date(2026, 5, 19))  # Tuesday
        assert got == datetime(2026, 5, 22, 15, 30, tzinfo=ET)  # Friday 15:30 ET
        assert got.utcoffset() == timedelta(hours=-4)  # EDT in May -> DST-aware ET

    def test_monday_snapshot_maps_to_same_week_friday(self) -> None:
        # Holiday-shift case: Tuesday was a federal holiday so the snapshot moved to Monday,
        # but the release stays that Friday.
        got = value_known_at_for_report(date(2026, 5, 18))  # Monday
        assert got == datetime(2026, 5, 22, 15, 30, tzinfo=ET)

    def test_release_time_constant_is_1530(self) -> None:
        assert (COT_RELEASE_TIME_ET.hour, COT_RELEASE_TIME_ET.minute) == (15, 30)

    @pytest.mark.parametrize("weekend", [date(2026, 5, 23), date(2026, 5, 24)])  # Sat, Sun
    def test_weekend_snapshot_raises(self, weekend: date) -> None:
        with pytest.raises(ValueError, match="weekday"):
            value_known_at_for_report(weekend)


class TestPitGate:
    def test_blocks_unpublished_snapshot(self) -> None:
        report = _report(date(2026, 5, 19))  # publishes Fri 2026-05-22 15:30 ET
        wednesday = datetime(2026, 5, 20, 14, 0, tzinfo=ET)  # before the Friday publish
        assert enforce_cot_pit_gate(report, wednesday) is None

    def test_passes_published_snapshot(self) -> None:
        report = _report(date(2026, 5, 19))
        after = datetime(2026, 5, 22, 15, 31, tzinfo=ET)
        assert enforce_cot_pit_gate(report, after) is report

    def test_boundary_is_inclusive(self) -> None:
        report = _report(date(2026, 5, 19))
        exact = datetime(2026, 5, 22, 15, 30, tzinfo=ET)  # == value_known_at_iso
        assert enforce_cot_pit_gate(report, exact) is report  # known AT the publish moment

    def test_none_passthrough(self) -> None:
        assert enforce_cot_pit_gate(None, datetime(2026, 5, 22, tzinfo=ET)) is None

    def test_naive_as_of_raises(self) -> None:
        report = _report(date(2026, 5, 19))
        with pytest.raises(ValueError, match="as_of_iso"):
            enforce_cot_pit_gate(report, datetime(2026, 5, 22, 15, 30))  # naive


class TestCOTSourceABC:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            COTSource()  # type: ignore[abstract]

    def test_partial_subclass_fails(self) -> None:
        class _Partial(COTSource):
            @property
            def source_id(self) -> str:
                return "partial"

            # missing tier + fetch_reports

        with pytest.raises(TypeError):
            _Partial()  # type: ignore[abstract]

    def test_error_rooted_at_datasource_error(self) -> None:
        assert issubclass(COTSourceError, DataSourceError)

    def test_healthcheck_default_true(self) -> None:
        assert _FixtureCOTSource([]).healthcheck() is True

    def test_repr_carries_source_id_and_tier(self) -> None:
        text = repr(_FixtureCOTSource([]))
        assert "fixture_cot" in text
        assert "T2_MACRO" in text


class TestReportsKnownAt:
    def test_applies_pit_and_sorts_ascending(self) -> None:
        # Four weekly Tuesdays ending 2026-05-19 (publish Fridays: 05-01, 05-08, 05-15, 05-22).
        tuesdays = [date(2026, 5, 19) - timedelta(weeks=i) for i in range(4)][::-1]
        src = _FixtureCOTSource([_report(t) for t in tuesdays])
        # Thursday 2026-05-21: the first 3 reports have published; the 4th (publishes 05-22) hasn't.
        as_of = datetime(2026, 5, 21, 12, 0, tzinfo=ET)
        known = src.reports_known_at(
            _CL, COTReportFlavor.DISAGGREGATED, as_of, since=date(2026, 1, 1)
        )
        assert [r.report_date for r in known] == tuesdays[:3]  # ascending; 4th gated out

    def test_passes_since_and_as_of_date_to_fetch(self) -> None:
        src = _FixtureCOTSource([_report(date(2026, 5, 19))])
        as_of = datetime(2026, 5, 25, 12, 0, tzinfo=ET)
        since = date(2026, 1, 1)
        src.reports_known_at(_CL, COTReportFlavor.DISAGGREGATED, as_of, since=since)
        assert src.last_fetch_args == (_CL, COTReportFlavor.DISAGGREGATED, since, date(2026, 5, 25))

    def test_filters_by_contract(self) -> None:
        src = _FixtureCOTSource(
            [
                _report(date(2026, 5, 19), market_code=_CL),
                _report(date(2026, 5, 19), market_code=_GC),
            ]
        )
        as_of = datetime(2026, 5, 25, tzinfo=ET)
        known = src.reports_known_at(
            _CL, COTReportFlavor.DISAGGREGATED, as_of, since=date(2026, 1, 1)
        )
        assert len(known) == 1
        assert known[0].cftc_contract_market_code == _CL

    def test_naive_as_of_raises(self) -> None:
        src = _FixtureCOTSource([])
        with pytest.raises(ValueError, match="as_of_iso"):
            src.reports_known_at(
                _CL, COTReportFlavor.DISAGGREGATED, datetime(2026, 5, 25), since=date(2026, 1, 1)
            )
