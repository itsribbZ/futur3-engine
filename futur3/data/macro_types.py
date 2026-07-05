"""futur3.data.macro_types - frozen dataclasses for the macro-event data axis.

This module separates the macro-event data
axis from bar/tick/settle (futur3.data.types): macro events are scheduled economic releases
(NFP / CPI / FOMC / ...) whose information content is in the release reaction, not in price bars.

Core invariants (mirror futur3.data.types):
- Decimal for the published value (never float). The value is SIGNED (CPI MoM can deflate,
  GDP can contract) so there is NO positivity constraint, but it must be finite.
- IANA-TZ-aware datetimes at the boundary (naive raises).
- frozen=True for immutability + hashability.
- SHA256 provenance (content_bytes_sha) on every record for bit-reproducibility.

PIT (point-in-time) is the load-bearing invariant of this axis (bug class 5 / look-ahead):
- `MacroValue.value_known_at_iso` is the ACTUAL publication moment.
- A backtest at `as_of_iso` MUST NOT consume a value whose `value_known_at_iso > as_of_iso`.
  Centralized enforcement lives in `futur3.data.macro_source.enforce_pit_gate`.
- The 2025-10 federal shutdown PERMANENTLY voided some releases (Oct NFP household survey,
  Oct PPI). Modeled via `MacroValue.is_shutdown_void` (value is None; a number that does not
  exist must never be synthesized).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from enum import StrEnum
from typing import Final

from futur3.data.types import SHA256_HEX_LENGTH, _assert_tz_aware

# ----------------------------------------------------------------------------
# Enums
# ----------------------------------------------------------------------------


class MacroSeries(StrEnum):
    """The futur3 macro-event catalog.

    The value is the canonical series id used in `event_id` construction and provenance
    hashing. See internal design notes
    """

    NFP = "NFP"  # Employment Situation (BLS)
    CPI = "CPI"  # Consumer Price Index (BLS)
    PPI = "PPI"  # Producer Price Index (BLS)
    GDP_ADVANCE = "GDP_ADVANCE"  # GDP advance estimate (BEA)
    GDP_SECOND = "GDP_SECOND"  # GDP second estimate (BEA)
    GDP_THIRD = "GDP_THIRD"  # GDP third estimate (BEA)
    PCE = "PCE"  # Personal Consumption Expenditures (BEA)
    RETAIL_SALES = "RETAIL_SALES"  # Advance Monthly Retail Trade (Census)
    HOUSING_STARTS = "HOUSING_STARTS"  # New Residential Construction (Census)
    JOBLESS_CLAIMS = "JOBLESS_CLAIMS"  # Initial Jobless Claims (DOL)
    ISM_MFG = "ISM_MFG"  # ISM Manufacturing PMI (private)
    ISM_SERVICES = "ISM_SERVICES"  # ISM Services PMI (private)
    JOLTS = "JOLTS"  # Job Openings and Labor Turnover (BLS)
    CONSUMER_CONF = "CONSUMER_CONF"  # Consumer Confidence (Conference Board, private)
    FOMC_STATEMENT = "FOMC_STATEMENT"  # FOMC rate decision (Fed)
    FOMC_PRESS_CONF = "FOMC_PRESS_CONF"  # FOMC press conference (Fed)
    FOMC_MINUTES = "FOMC_MINUTES"  # FOMC minutes, 3 weeks post-meeting (Fed)
    BEIGE_BOOK = "BEIGE_BOOK"  # Beige Book, ~2 weeks pre-FOMC (Fed)


class MacroPublisher(StrEnum):
    """Authoritative publisher per macro series.

    Government publishers (BLS/BEA/CENSUS/DOL/FED) are T2_MACRO (federally mandated,
    criminal misreporting liability). Private publishers (ISM/CONFERENCE_BOARD) are
    substituted-with-delay per the free-first policy. FRED is the T3 aggregator (ALFRED vintages).
    """

    BLS = "BLS"
    BEA = "BEA"
    CENSUS = "CENSUS"
    DOL = "DOL"
    FED = "FED"
    ISM = "ISM"
    CONFERENCE_BOARD = "CONFERENCE_BOARD"
    FRED = "FRED"


# Per internal design notes/New_York (ET).
# This table is keyed by EVERY MacroSeries member (completeness enforced by test).
RELEASE_TIME_ET: Final[dict[MacroSeries, time]] = {
    MacroSeries.NFP: time(8, 30),
    MacroSeries.CPI: time(8, 30),
    MacroSeries.PPI: time(8, 30),
    MacroSeries.GDP_ADVANCE: time(8, 30),
    MacroSeries.GDP_SECOND: time(8, 30),
    MacroSeries.GDP_THIRD: time(8, 30),
    MacroSeries.PCE: time(8, 30),
    MacroSeries.RETAIL_SALES: time(8, 30),
    MacroSeries.HOUSING_STARTS: time(8, 30),
    MacroSeries.JOBLESS_CLAIMS: time(8, 30),
    MacroSeries.ISM_MFG: time(10, 0),
    MacroSeries.ISM_SERVICES: time(10, 0),
    MacroSeries.JOLTS: time(10, 0),
    MacroSeries.CONSUMER_CONF: time(10, 0),
    MacroSeries.FOMC_STATEMENT: time(14, 0),
    MacroSeries.FOMC_PRESS_CONF: time(14, 30),
    MacroSeries.FOMC_MINUTES: time(14, 0),
    MacroSeries.BEIGE_BOOK: time(14, 0),
}


# ----------------------------------------------------------------------------
# Records
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class MacroEvent:
    """A forward-known scheduled macro release (the CALENDAR entry).

    Published 6-12 months ahead by BLS/BEA/Census/Fed. Carries both the originally-scheduled
    moment and (post-release) the actual publication moment so the engine can detect schedule
    drift. The 2025 shutdown delayed Oct PPI from Nov-2025 to Jan-14-2026; the backtest must
    only see that value at the actual Jan-14 publication, never at the original Nov date.
    """

    event_id: str  # e.g. "NFP_2026_06"
    series: MacroSeries
    publisher: MacroPublisher
    release_date: date  # scheduled calendar date
    release_time_et: datetime  # scheduled moment, tz-aware (ET canonical)
    source_url: str
    originally_scheduled_release_time_et: datetime  # for shutdown-delay drift tracking
    content_bytes_sha: str  # SHA256 of the calendar-announcement bytes
    embargo_window_min: int = 5  # futur3 strategy blackout half-width (min)
    actual_publication_iso: datetime | None = None  # real publish moment; None until released

    def __post_init__(self) -> None:
        _assert_tz_aware(self.release_time_et, "MacroEvent.release_time_et")
        _assert_tz_aware(
            self.originally_scheduled_release_time_et,
            "MacroEvent.originally_scheduled_release_time_et",
        )
        if self.actual_publication_iso is not None:
            _assert_tz_aware(self.actual_publication_iso, "MacroEvent.actual_publication_iso")
        if not self.event_id:
            raise ValueError("MacroEvent.event_id must be non-empty")
        if not self.source_url:
            raise ValueError("MacroEvent.source_url must be non-empty")
        if self.embargo_window_min < 0:
            raise ValueError(
                f"MacroEvent.embargo_window_min must be >= 0; got {self.embargo_window_min}"
            )
        if len(self.content_bytes_sha) != SHA256_HEX_LENGTH:
            raise ValueError(
                f"MacroEvent.content_bytes_sha must be hex-SHA256 ({SHA256_HEX_LENGTH} chars); "
                f"got len={len(self.content_bytes_sha)}"
            )

    @property
    def is_published(self) -> bool:
        """True once the actual publication moment has been recorded."""
        return self.actual_publication_iso is not None


@dataclass(frozen=True)
class MacroValue:
    """A point-in-time-correct published macro value.

    `value_known_at_iso` is the PIT GATE: the engine MUST NOT consume this value at any
    backtest `as_of_iso` earlier than `value_known_at_iso` (bug class 5 / look-ahead).
    Centralized enforcement is `futur3.data.macro_source.enforce_pit_gate`.

    Revision-aware (ALFRED vintages): `vintage_as_of` records which data vintage this value
    represents. A backtest on GDP-advance-release-day must see the advance estimate, not the
    later third estimate, even though both share the same `as_of_date` reference period.

    Shutdown-void: when `is_shutdown_void` is True, `value` is None - the number does not exist
    and must never be synthesized. Read via `futur3.data.macro_source.require_value` for a
    fail-loud guard.
    """

    event_id: str
    series: MacroSeries
    as_of_date: date  # the reference PERIOD this value describes
    value_known_at_iso: datetime  # PIT GATE - actual publication moment (tz-aware)
    source_id: str
    as_of_iso: datetime  # when fetched (tz-aware)
    content_bytes_sha: str
    value: Decimal | None = None  # SIGNED headline number; None iff is_shutdown_void
    is_shutdown_void: bool = False
    vintage_as_of: date | None = None  # ALFRED vintage date (None = latest / non-vintage)

    def __post_init__(self) -> None:
        _assert_tz_aware(self.value_known_at_iso, "MacroValue.value_known_at_iso")
        _assert_tz_aware(self.as_of_iso, "MacroValue.as_of_iso")
        if not self.event_id:
            raise ValueError("MacroValue.event_id must be non-empty")
        if self.is_shutdown_void:
            if self.value is not None:
                raise ValueError(
                    "MacroValue.value must be None when is_shutdown_void=True "
                    f"(release does not exist); got {self.value}"
                )
        else:
            if self.value is None:
                raise ValueError("MacroValue.value must be set when is_shutdown_void=False")
            if not self.value.is_finite():
                raise ValueError(f"MacroValue.value must be finite; got {self.value}")
        if len(self.content_bytes_sha) != SHA256_HEX_LENGTH:
            raise ValueError(
                f"MacroValue.content_bytes_sha must be hex-SHA256 ({SHA256_HEX_LENGTH} chars); "
                f"got len={len(self.content_bytes_sha)}"
            )
