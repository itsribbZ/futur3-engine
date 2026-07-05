"""futur3.data.cot_source - COTSource ABC + COT exception hierarchy + the hard PIT gate.

Per internal design notes

Parallel to (NOT a subclass of) `futur3.data.source.DataSource` and a sibling of
`futur3.data.macro_source.MacroEventSource`: COT is a weekly positioning feed, not price bars,
so the surface is `fetch_reports` / `reports_known_at` rather than `get_bars` / `latest_settle`.
Both ABCs share the provenance discipline (SHA256, tz-aware boundaries) and the `DataSourceError`
root, so the engine catches data-acquisition failures uniformly.

COT is a SINGLE-SOURCE feed: the CFTC is the sole publisher (federally mandated under the CEA;
misreporting is criminal), so unlike multi-source bars there is no cross-source consensus - the
verifier registers a COTSource with quorum=1.

Hard PIT invariant (bug class 5 / look-ahead): the CFTC publishes a Tuesday snapshot the
following Friday at 15:30 ET. The 3-day blackout is the #1 silent bug in retail COT backtests.
Rather than trust each concrete source to remember the gate (the macro layer's per-impl
`enforce_pit_gate` convention), this ABC provides a CONCRETE `reports_known_at` built on top of
the abstract `fetch_reports` + `enforce_cot_pit_gate`, so the look-ahead guard is structural -
a subclass cannot forget it.
"""

from __future__ import annotations

import abc
from datetime import date, datetime, time, timedelta
from typing import Final
from zoneinfo import ZoneInfo

from futur3.data.cot_types import COTReport, COTReportFlavor
from futur3.data.source import DataSourceError
from futur3.data.types import SourceTier, _assert_tz_aware

# Per internal design notes:30 ET.
ET: Final[ZoneInfo] = ZoneInfo("America/New_York")
COT_RELEASE_TIME_ET: Final[time] = time(15, 30)
_FRIDAY_WEEKDAY: Final[int] = 4  # date.weekday(): Mon=0 .. Fri=4 .. Sun=6


# ----------------------------------------------------------------------------
# Exception hierarchy (rooted at DataSourceError for uniform engine catch)
# ----------------------------------------------------------------------------


class COTSourceError(DataSourceError):
    """Parent for COT-source errors.

    Extends `DataSourceError` so the engine catches COT acquisition failures with the same
    handler as bar/tick/settle/macro failures.
    """


class COTSchemaDriftError(COTSourceError):
    """Raised when the returned COT column set diverges from the pinned canonical schema.

    CFTC changed its COT schema 3 times historically (2006 Disaggregated/TFF split, 2017 crypto
    contracts, 2022 Socrata migration). Drift -> fail-closed rather than silently mis-mapping a
    renamed column to zero (bug class 9 schema-drift hook).
    """


class COTContractUnknownError(COTSourceError):
    """Raised when a source is asked for a `cftc_contract_market_code` it does not carry."""


# ----------------------------------------------------------------------------
# PIT gate + release-time helper (the bug-class-5 machinery)
# ----------------------------------------------------------------------------


def value_known_at_for_report(report_date: date) -> datetime:
    """The Friday 15:30 ET publication moment for a COT snapshot dated `report_date`.

    The CFTC snapshots positions on a Tuesday and publishes them the following Friday at 15:30 ET
    (the same ISO week). When a federal holiday falls on the Tuesday, the snapshot moves to that
    Monday but the release stays that Friday - so this maps any Mon-Fri snapshot to its own week's
    Friday (Tue -> +3 days, Mon -> +4 days).

    NOT modeled here: a Friday-holiday shift (release -> next business day). That needs the CFTC
    forward release calendar and is the concrete source's responsibility to override; this helper
    is the normal-week default.

    Raises:
        ValueError: `report_date` falls on a weekend (never a valid COT snapshot day).
    """
    weekday = report_date.weekday()
    if weekday > _FRIDAY_WEEKDAY:
        raise ValueError(
            f"COT snapshot date must be a weekday (Mon-Fri); got {report_date} (weekday {weekday})"
        )
    friday = report_date + timedelta(days=_FRIDAY_WEEKDAY - weekday)
    return datetime(
        friday.year,
        friday.month,
        friday.day,
        COT_RELEASE_TIME_ET.hour,
        COT_RELEASE_TIME_ET.minute,
        tzinfo=ET,
    )


def enforce_cot_pit_gate(report: COTReport | None, as_of_iso: datetime) -> COTReport | None:
    """The hard point-in-time filter for COT (bug class 5 / look-ahead).

    Returns `report` only if it was PUBLISHED at or before `as_of_iso`; otherwise None. This is
    the Tue->Fri 3-day blackout in code: a decision on Wednesday may not see Monday's snapshot
    (it does not publish until Friday 15:30 ET).
    """
    _assert_tz_aware(as_of_iso, "enforce_cot_pit_gate.as_of_iso")
    if report is None:
        return None
    if report.value_known_at_iso > as_of_iso:
        return None
    return report


# ----------------------------------------------------------------------------
# COTSource ABC
# ----------------------------------------------------------------------------


class COTSource(abc.ABC):
    """The seam between futur3 and the CFTC COT publisher (Socrata `publicreporting.cftc.gov`).

    Subclasses MUST implement: `source_id`, `tier`, `fetch_reports`.
    Subclasses MAY override: `healthcheck` (default True).

    PIT CONTRACT (bug class 5): consumers read history through the CONCRETE `reports_known_at`,
    which applies `enforce_cot_pit_gate` to every row - the look-ahead guard is structural, so a
    subclass that implements only the raw `fetch_reports` still cannot leak a not-yet-published
    snapshot into a backtest.
    """

    @property
    @abc.abstractmethod
    def source_id(self) -> str:
        """Stable per-source provenance string (e.g. 'cftc_socrata')."""
        ...

    @property
    @abc.abstractmethod
    def tier(self) -> SourceTier:
        """SourceTier classification. CFTC is a direct federal publisher -> `T2_MACRO`."""
        ...

    @abc.abstractmethod
    def fetch_reports(
        self,
        cftc_contract_market_code: str,
        flavor: COTReportFlavor,
        start: date,
        end: date,
    ) -> list[COTReport]:
        """Raw COT reports for `cftc_contract_market_code` with `report_date` in [start, end].

        Returns reports ascending by `report_date` (one per Tuesday snapshot in range); empty
        list if there are none (e.g. a government-shutdown gap). This is the RAW accessor - it
        applies no PIT gate; consumers use `reports_known_at` for the look-ahead-safe view.

        Raises:
            COTSchemaDriftError: returned columns diverged from the pinned schema (bug class 9).
            COTContractUnknownError: the source does not carry this market code.
            COTSourceError: any other acquisition failure.
        """
        ...

    def reports_known_at(
        self,
        cftc_contract_market_code: str,
        flavor: COTReportFlavor,
        as_of_iso: datetime,
        *,
        since: date,
    ) -> list[COTReport]:
        """PIT-correct ascending COT history known at `as_of_iso` (the look-ahead-safe view).

        Fetches reports in [`since`, `as_of_iso` date] then drops any whose `value_known_at_iso`
        is after `as_of_iso` (the Tue->Fri blackout). `since` bounds the fetch - the caller picks
        it from the strategy's lookback window.

        The result is sorted by `report_date` for total-order determinism, independent of
        the order the concrete source returned rows in.
        """
        _assert_tz_aware(as_of_iso, "COTSource.reports_known_at.as_of_iso")
        raw = self.fetch_reports(cftc_contract_market_code, flavor, since, as_of_iso.date())
        known = [r for r in raw if enforce_cot_pit_gate(r, as_of_iso) is not None]
        return sorted(known, key=lambda r: r.report_date)

    def healthcheck(self) -> bool:
        """Lightweight liveness check. Default True; network-backed sources override."""
        return True

    def __repr__(self) -> str:
        return f"<{type(self).__name__} source_id={self.source_id!r} tier={self.tier.name}>"


__all__: list[str] = [
    "COT_RELEASE_TIME_ET",
    "COTContractUnknownError",
    "COTSchemaDriftError",
    "COTSource",
    "COTSourceError",
    "enforce_cot_pit_gate",
    "value_known_at_for_report",
]
