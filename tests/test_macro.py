"""A1.24 MacroEventSource ABC + macro_types test suite.

Test discipline:
- Seam-only (no live HTTP; concrete BLS/BEA/FRED sources land in A1.25+).
- ABC enforces abstractness (cannot instantiate; partial subclass fails; full concrete works).
- Dataclass validation (frozen, signed-value semantics, shutdown-void XOR, TZ-aware boundary).
- The hard PIT gate (bug class 5) is exercised end-to-end through a fixture concrete source.
- Exception hierarchy rooted at DataSourceError.

References:
- `futur3/data/macro_types.py` + `futur3/data/macro_source.py` (implementation)
- internal design notes(spec)
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from futur3.data.macro_source import (
    DriftKind,
    MacroEventSource,
    MacroSourceError,
    ScheduleDrift,
    ScheduleDriftError,
    ShutdownVoidError,
    enforce_pit_gate,
    require_value,
)
from futur3.data.macro_types import (
    RELEASE_TIME_ET,
    MacroEvent,
    MacroPublisher,
    MacroSeries,
    MacroValue,
)
from futur3.data.source import DataSourceError
from futur3.data.types import SourceTier, content_sha256

# ============================================================================
# Helpers
# ============================================================================

ET = ZoneInfo("America/New_York")
_SHA = content_sha256(b"macro-fixture")


def _valid_event() -> MacroEvent:
    return MacroEvent(
        event_id="NFP_2026_06",
        series=MacroSeries.NFP,
        publisher=MacroPublisher.BLS,
        release_date=date(2026, 6, 5),
        release_time_et=datetime(2026, 6, 5, 8, 30, tzinfo=ET),
        source_url="https://www.bls.gov/news.release/empsit.htm",
        originally_scheduled_release_time_et=datetime(2026, 6, 5, 8, 30, tzinfo=ET),
        content_bytes_sha=_SHA,
    )


def _valid_value(
    value: Decimal | None = Decimal("175000"),
    is_shutdown_void: bool = False,
    value_known_at_iso: datetime | None = None,
) -> MacroValue:
    if value_known_at_iso is None:
        value_known_at_iso = datetime(2026, 6, 5, 12, 30, tzinfo=UTC)
    return MacroValue(
        event_id="NFP_2026_06",
        series=MacroSeries.NFP,
        as_of_date=date(2026, 5, 31),
        value_known_at_iso=value_known_at_iso,
        source_id="bls_api_v2",
        as_of_iso=datetime(2026, 6, 5, 12, 30, tzinfo=UTC),
        content_bytes_sha=_SHA,
        value=value,
        is_shutdown_void=is_shutdown_void,
    )


class _FixtureMacroSource(MacroEventSource):
    """Minimal in-memory concrete source for ABC-contract + PIT tests.

    `fetch_value` routes through `enforce_pit_gate` per the PIT contract.
    """

    def __init__(self, value: MacroValue | None = None) -> None:
        self._value = value

    @property
    def source_id(self) -> str:
        return "fixture_macro"

    @property
    def tier(self) -> SourceTier:
        return SourceTier.T2_MACRO

    def upcoming_events(self, start: datetime, end: datetime) -> list[MacroEvent]:
        return []

    def fetch_value(self, event_id: str, as_of_iso: datetime) -> MacroValue | None:
        if self._value is None or self._value.event_id != event_id:
            return None
        return enforce_pit_gate(self._value, as_of_iso)


# ============================================================================
# TestA1_24_Imports
# ============================================================================


class TestA1_24_Imports:
    def test_macro_types_importable(self) -> None:
        assert MacroEvent is not None
        assert MacroValue is not None
        assert MacroSeries is not None
        assert MacroPublisher is not None
        assert RELEASE_TIME_ET is not None

    def test_macro_source_importable(self) -> None:
        assert MacroEventSource is not None
        assert enforce_pit_gate is not None
        assert require_value is not None
        assert ScheduleDrift is not None
        assert DriftKind is not None

    def test_errors_importable(self) -> None:
        assert MacroSourceError is not None
        assert ScheduleDriftError is not None
        assert ShutdownVoidError is not None


# ============================================================================
# TestA1_24_ExceptionHierarchy
# ============================================================================


class TestA1_24_ExceptionHierarchy:
    def test_macro_source_error_extends_data_source_error(self) -> None:
        assert issubclass(MacroSourceError, DataSourceError)

    def test_schedule_drift_error_extends_macro_source_error(self) -> None:
        assert issubclass(ScheduleDriftError, MacroSourceError)

    def test_shutdown_void_error_extends_macro_source_error(self) -> None:
        assert issubclass(ShutdownVoidError, MacroSourceError)

    def test_all_are_exceptions(self) -> None:
        assert issubclass(MacroSourceError, Exception)
        assert issubclass(ScheduleDriftError, Exception)
        assert issubclass(ShutdownVoidError, Exception)

    def test_siblings_distinct(self) -> None:
        assert ScheduleDriftError is not ShutdownVoidError


# ============================================================================
# TestA1_24_MacroSeriesEnum
# ============================================================================


class TestA1_24_MacroSeriesEnum:
    def test_eighteen_members(self) -> None:
        assert len(MacroSeries) == 18

    def test_is_str_enum(self) -> None:
        assert isinstance(MacroSeries.NFP, str)
        assert MacroSeries.NFP == "NFP"

    def test_value_matches_name(self) -> None:
        for member in MacroSeries:
            assert member.value == member.name

    def test_key_members_present(self) -> None:
        assert MacroSeries.NFP in MacroSeries
        assert MacroSeries.CPI in MacroSeries
        assert MacroSeries.FOMC_STATEMENT in MacroSeries
        assert MacroSeries.BEIGE_BOOK in MacroSeries


# ============================================================================
# TestA1_24_MacroPublisherEnum
# ============================================================================


class TestA1_24_MacroPublisherEnum:
    def test_eight_members(self) -> None:
        assert len(MacroPublisher) == 8

    def test_is_str_enum(self) -> None:
        assert isinstance(MacroPublisher.BLS, str)
        assert MacroPublisher.BLS == "BLS"

    def test_key_members_present(self) -> None:
        for name in ("BLS", "BEA", "CENSUS", "DOL", "FED", "ISM", "CONFERENCE_BOARD", "FRED"):
            assert name in MacroPublisher.__members__


# ============================================================================
# TestA1_24_ReleaseTimeTable
# ============================================================================


class TestA1_24_ReleaseTimeTable:
    def test_covers_every_series(self) -> None:
        """Completeness invariant: every MacroSeries has a release time, and vice versa."""
        assert set(RELEASE_TIME_ET.keys()) == set(MacroSeries)

    def test_nfp_0830(self) -> None:
        assert RELEASE_TIME_ET[MacroSeries.NFP] == time(8, 30)

    def test_fomc_statement_1400(self) -> None:
        assert RELEASE_TIME_ET[MacroSeries.FOMC_STATEMENT] == time(14, 0)

    def test_fomc_press_conf_1430(self) -> None:
        assert RELEASE_TIME_ET[MacroSeries.FOMC_PRESS_CONF] == time(14, 30)

    def test_all_values_are_time(self) -> None:
        assert all(isinstance(v, time) for v in RELEASE_TIME_ET.values())


# ============================================================================
# TestA1_24_MacroEvent
# ============================================================================


class TestA1_24_MacroEvent:
    def test_valid_construction(self) -> None:
        ev = _valid_event()
        assert ev.series is MacroSeries.NFP
        assert ev.publisher is MacroPublisher.BLS
        assert ev.embargo_window_min == 5
        assert ev.actual_publication_iso is None

    def test_is_published_false_when_unset(self) -> None:
        assert _valid_event().is_published is False

    def test_is_published_true_when_set(self) -> None:
        ev = MacroEvent(
            event_id="NFP_2026_06",
            series=MacroSeries.NFP,
            publisher=MacroPublisher.BLS,
            release_date=date(2026, 6, 5),
            release_time_et=datetime(2026, 6, 5, 8, 30, tzinfo=ET),
            source_url="https://www.bls.gov/news.release/empsit.htm",
            originally_scheduled_release_time_et=datetime(2026, 6, 5, 8, 30, tzinfo=ET),
            content_bytes_sha=_SHA,
            actual_publication_iso=datetime(2026, 6, 5, 12, 30, tzinfo=UTC),
        )
        assert ev.is_published is True

    def test_naive_release_time_raises(self) -> None:
        with pytest.raises(ValueError, match=r"MacroEvent\.release_time_et must be IANA-TZ-aware"):
            MacroEvent(
                event_id="NFP_2026_06",
                series=MacroSeries.NFP,
                publisher=MacroPublisher.BLS,
                release_date=date(2026, 6, 5),
                release_time_et=datetime(2026, 6, 5, 8, 30),  # naive
                source_url="https://example.gov",
                originally_scheduled_release_time_et=datetime(2026, 6, 5, 8, 30, tzinfo=ET),
                content_bytes_sha=_SHA,
            )

    def test_naive_originally_scheduled_raises(self) -> None:
        with pytest.raises(ValueError, match="originally_scheduled_release_time_et"):
            MacroEvent(
                event_id="NFP_2026_06",
                series=MacroSeries.NFP,
                publisher=MacroPublisher.BLS,
                release_date=date(2026, 6, 5),
                release_time_et=datetime(2026, 6, 5, 8, 30, tzinfo=ET),
                source_url="https://example.gov",
                originally_scheduled_release_time_et=datetime(2026, 6, 5, 8, 30),  # naive
                content_bytes_sha=_SHA,
            )

    def test_naive_actual_publication_raises(self) -> None:
        with pytest.raises(ValueError, match="actual_publication_iso"):
            MacroEvent(
                event_id="NFP_2026_06",
                series=MacroSeries.NFP,
                publisher=MacroPublisher.BLS,
                release_date=date(2026, 6, 5),
                release_time_et=datetime(2026, 6, 5, 8, 30, tzinfo=ET),
                source_url="https://example.gov",
                originally_scheduled_release_time_et=datetime(2026, 6, 5, 8, 30, tzinfo=ET),
                content_bytes_sha=_SHA,
                actual_publication_iso=datetime(2026, 6, 5, 12, 30),  # naive
            )

    def test_empty_event_id_raises(self) -> None:
        with pytest.raises(ValueError, match="event_id must be non-empty"):
            MacroEvent(
                event_id="",
                series=MacroSeries.NFP,
                publisher=MacroPublisher.BLS,
                release_date=date(2026, 6, 5),
                release_time_et=datetime(2026, 6, 5, 8, 30, tzinfo=ET),
                source_url="https://example.gov",
                originally_scheduled_release_time_et=datetime(2026, 6, 5, 8, 30, tzinfo=ET),
                content_bytes_sha=_SHA,
            )

    def test_empty_source_url_raises(self) -> None:
        with pytest.raises(ValueError, match="source_url must be non-empty"):
            MacroEvent(
                event_id="NFP_2026_06",
                series=MacroSeries.NFP,
                publisher=MacroPublisher.BLS,
                release_date=date(2026, 6, 5),
                release_time_et=datetime(2026, 6, 5, 8, 30, tzinfo=ET),
                source_url="",
                originally_scheduled_release_time_et=datetime(2026, 6, 5, 8, 30, tzinfo=ET),
                content_bytes_sha=_SHA,
            )

    def test_negative_embargo_raises(self) -> None:
        with pytest.raises(ValueError, match="embargo_window_min must be >= 0"):
            MacroEvent(
                event_id="NFP_2026_06",
                series=MacroSeries.NFP,
                publisher=MacroPublisher.BLS,
                release_date=date(2026, 6, 5),
                release_time_et=datetime(2026, 6, 5, 8, 30, tzinfo=ET),
                source_url="https://example.gov",
                originally_scheduled_release_time_et=datetime(2026, 6, 5, 8, 30, tzinfo=ET),
                content_bytes_sha=_SHA,
                embargo_window_min=-1,
            )

    def test_zero_embargo_allowed(self) -> None:
        ev = MacroEvent(
            event_id="NFP_2026_06",
            series=MacroSeries.NFP,
            publisher=MacroPublisher.BLS,
            release_date=date(2026, 6, 5),
            release_time_et=datetime(2026, 6, 5, 8, 30, tzinfo=ET),
            source_url="https://example.gov",
            originally_scheduled_release_time_et=datetime(2026, 6, 5, 8, 30, tzinfo=ET),
            content_bytes_sha=_SHA,
            embargo_window_min=0,
        )
        assert ev.embargo_window_min == 0

    def test_bad_sha_raises(self) -> None:
        with pytest.raises(ValueError, match="content_bytes_sha must be hex-SHA256"):
            MacroEvent(
                event_id="NFP_2026_06",
                series=MacroSeries.NFP,
                publisher=MacroPublisher.BLS,
                release_date=date(2026, 6, 5),
                release_time_et=datetime(2026, 6, 5, 8, 30, tzinfo=ET),
                source_url="https://example.gov",
                originally_scheduled_release_time_et=datetime(2026, 6, 5, 8, 30, tzinfo=ET),
                content_bytes_sha="deadbeef",  # too short
            )

    def test_frozen_immutable(self) -> None:
        ev = _valid_event()
        with pytest.raises(AttributeError):
            ev.embargo_window_min = 10  # type: ignore[misc]


# ============================================================================
# TestA1_24_MacroValue
# ============================================================================


class TestA1_24_MacroValue:
    def test_valid_construction(self) -> None:
        mv = _valid_value()
        assert mv.value == Decimal("175000")
        assert mv.is_shutdown_void is False
        assert mv.vintage_as_of is None

    def test_shutdown_void_construction(self) -> None:
        mv = _valid_value(value=None, is_shutdown_void=True)
        assert mv.value is None
        assert mv.is_shutdown_void is True

    def test_void_with_value_raises(self) -> None:
        with pytest.raises(ValueError, match="must be None when is_shutdown_void=True"):
            _valid_value(value=Decimal("1"), is_shutdown_void=True)

    def test_non_void_without_value_raises(self) -> None:
        with pytest.raises(ValueError, match="must be set when is_shutdown_void=False"):
            _valid_value(value=None, is_shutdown_void=False)

    def test_nan_value_raises(self) -> None:
        with pytest.raises(ValueError, match="value must be finite"):
            _valid_value(value=Decimal("NaN"))

    def test_infinity_value_raises(self) -> None:
        with pytest.raises(ValueError, match="value must be finite"):
            _valid_value(value=Decimal("Infinity"))

    def test_negative_value_allowed(self) -> None:
        """Macro values are SIGNED (CPI MoM can deflate; GDP can contract)."""
        mv = _valid_value(value=Decimal("-0.1"))
        assert mv.value == Decimal("-0.1")

    def test_zero_value_allowed(self) -> None:
        mv = _valid_value(value=Decimal("0"))
        assert mv.value == Decimal("0")

    def test_naive_value_known_at_raises(self) -> None:
        with pytest.raises(ValueError, match="value_known_at_iso must be IANA-TZ-aware"):
            MacroValue(
                event_id="NFP_2026_06",
                series=MacroSeries.NFP,
                as_of_date=date(2026, 5, 31),
                value_known_at_iso=datetime(2026, 6, 5, 12, 30),  # naive
                source_id="bls_api_v2",
                as_of_iso=datetime(2026, 6, 5, 12, 30, tzinfo=UTC),
                content_bytes_sha=_SHA,
                value=Decimal("175000"),
            )

    def test_naive_as_of_iso_raises(self) -> None:
        with pytest.raises(ValueError, match="as_of_iso must be IANA-TZ-aware"):
            MacroValue(
                event_id="NFP_2026_06",
                series=MacroSeries.NFP,
                as_of_date=date(2026, 5, 31),
                value_known_at_iso=datetime(2026, 6, 5, 12, 30, tzinfo=UTC),
                source_id="bls_api_v2",
                as_of_iso=datetime(2026, 6, 5, 12, 30),  # naive
                content_bytes_sha=_SHA,
                value=Decimal("175000"),
            )

    def test_empty_event_id_raises(self) -> None:
        with pytest.raises(ValueError, match="event_id must be non-empty"):
            MacroValue(
                event_id="",
                series=MacroSeries.NFP,
                as_of_date=date(2026, 5, 31),
                value_known_at_iso=datetime(2026, 6, 5, 12, 30, tzinfo=UTC),
                source_id="bls_api_v2",
                as_of_iso=datetime(2026, 6, 5, 12, 30, tzinfo=UTC),
                content_bytes_sha=_SHA,
                value=Decimal("175000"),
            )

    def test_bad_sha_raises(self) -> None:
        with pytest.raises(ValueError, match="content_bytes_sha must be hex-SHA256"):
            MacroValue(
                event_id="NFP_2026_06",
                series=MacroSeries.NFP,
                as_of_date=date(2026, 5, 31),
                value_known_at_iso=datetime(2026, 6, 5, 12, 30, tzinfo=UTC),
                source_id="bls_api_v2",
                as_of_iso=datetime(2026, 6, 5, 12, 30, tzinfo=UTC),
                content_bytes_sha="xyz",
                value=Decimal("175000"),
            )

    def test_frozen_immutable(self) -> None:
        mv = _valid_value()
        with pytest.raises(AttributeError):
            mv.value = Decimal("1")  # type: ignore[misc]

    def test_vintage_as_of_settable(self) -> None:
        mv = MacroValue(
            event_id="GDP_2026Q1_ADV",
            series=MacroSeries.GDP_ADVANCE,
            as_of_date=date(2026, 3, 31),
            value_known_at_iso=datetime(2026, 4, 29, 12, 30, tzinfo=UTC),
            source_id="fred_alfred",
            as_of_iso=datetime(2026, 4, 29, 12, 30, tzinfo=UTC),
            content_bytes_sha=_SHA,
            value=Decimal("1.6"),
            vintage_as_of=date(2026, 4, 29),
        )
        assert mv.vintage_as_of == date(2026, 4, 29)


# ============================================================================
# TestA1_24_EnforcePitGate (bug class 5 gate)
# ============================================================================


class TestA1_24_EnforcePitGate:
    def test_known_before_as_of_returned(self) -> None:
        mv = _valid_value(value_known_at_iso=datetime(2026, 6, 5, 12, 30, tzinfo=UTC))
        as_of = datetime(2026, 6, 5, 13, 0, tzinfo=UTC)  # after publication
        assert enforce_pit_gate(mv, as_of) is mv

    def test_known_after_as_of_returns_none(self) -> None:
        mv = _valid_value(value_known_at_iso=datetime(2026, 6, 5, 12, 30, tzinfo=UTC))
        as_of = datetime(2026, 6, 5, 8, 0, tzinfo=UTC)  # before publication
        assert enforce_pit_gate(mv, as_of) is None

    def test_known_exactly_at_as_of_returned(self) -> None:
        """Boundary: value known at the exact as_of instant is consumable (<=)."""
        ts = datetime(2026, 6, 5, 12, 30, tzinfo=UTC)
        mv = _valid_value(value_known_at_iso=ts)
        assert enforce_pit_gate(mv, ts) is mv

    def test_one_microsecond_before_returns_none(self) -> None:
        ts = datetime(2026, 6, 5, 12, 30, tzinfo=UTC)
        mv = _valid_value(value_known_at_iso=ts)
        assert enforce_pit_gate(mv, ts - timedelta(microseconds=1)) is None

    def test_none_input_returns_none(self) -> None:
        assert enforce_pit_gate(None, datetime(2026, 6, 5, 13, 0, tzinfo=UTC)) is None

    def test_naive_as_of_raises(self) -> None:
        mv = _valid_value()
        with pytest.raises(ValueError, match=r"enforce_pit_gate\.as_of_iso must be IANA-TZ-aware"):
            enforce_pit_gate(mv, datetime(2026, 6, 5, 13, 0))  # naive


# ============================================================================
# TestA1_24_RequireValue
# ============================================================================


class TestA1_24_RequireValue:
    def test_normal_value_returned(self) -> None:
        mv = _valid_value(value=Decimal("175000"))
        assert require_value(mv) == Decimal("175000")

    def test_void_raises_shutdown_void_error(self) -> None:
        mv = _valid_value(value=None, is_shutdown_void=True)
        with pytest.raises(ShutdownVoidError, match="shutdown-void"):
            require_value(mv)

    def test_negative_value_returned(self) -> None:
        mv = _valid_value(value=Decimal("-2.3"))
        assert require_value(mv) == Decimal("-2.3")


# ============================================================================
# TestA1_24_DriftKind
# ============================================================================


class TestA1_24_DriftKind:
    def test_three_members(self) -> None:
        assert len(DriftKind) == 3

    def test_values(self) -> None:
        assert DriftKind.MOVED == "moved"
        assert DriftKind.CANCELLED == "cancelled"
        assert DriftKind.ADDED == "added"


# ============================================================================
# TestA1_24_ScheduleDrift
# ============================================================================


class TestA1_24_ScheduleDrift:
    def test_moved_valid(self) -> None:
        d = ScheduleDrift(
            series=MacroSeries.PPI,
            event_id="PPI_2025_10",
            kind=DriftKind.MOVED,
            stored_release_time_et=datetime(2025, 11, 14, 8, 30, tzinfo=ET),
            current_release_time_et=datetime(2026, 1, 14, 8, 30, tzinfo=ET),
        )
        assert d.kind is DriftKind.MOVED

    def test_cancelled_valid(self) -> None:
        d = ScheduleDrift(
            series=MacroSeries.NFP,
            event_id="NFP_2025_10",
            kind=DriftKind.CANCELLED,
            stored_release_time_et=datetime(2025, 11, 7, 8, 30, tzinfo=ET),
            current_release_time_et=None,
        )
        assert d.current_release_time_et is None

    def test_added_valid(self) -> None:
        d = ScheduleDrift(
            series=MacroSeries.JOLTS,
            event_id="JOLTS_2026_07",
            kind=DriftKind.ADDED,
            stored_release_time_et=None,
            current_release_time_et=datetime(2026, 7, 1, 10, 0, tzinfo=ET),
        )
        assert d.stored_release_time_et is None

    def test_cancelled_with_current_raises(self) -> None:
        with pytest.raises(ValueError, match="CANCELLED drift must have current_release_time_et"):
            ScheduleDrift(
                series=MacroSeries.NFP,
                event_id="NFP_2025_10",
                kind=DriftKind.CANCELLED,
                stored_release_time_et=datetime(2025, 11, 7, 8, 30, tzinfo=ET),
                current_release_time_et=datetime(2025, 11, 7, 8, 30, tzinfo=ET),
            )

    def test_added_with_stored_raises(self) -> None:
        with pytest.raises(ValueError, match="ADDED drift must have stored_release_time_et"):
            ScheduleDrift(
                series=MacroSeries.JOLTS,
                event_id="JOLTS_2026_07",
                kind=DriftKind.ADDED,
                stored_release_time_et=datetime(2026, 7, 1, 10, 0, tzinfo=ET),
                current_release_time_et=datetime(2026, 7, 1, 10, 0, tzinfo=ET),
            )

    def test_moved_missing_stored_raises(self) -> None:
        with pytest.raises(ValueError, match="MOVED drift requires both"):
            ScheduleDrift(
                series=MacroSeries.PPI,
                event_id="PPI_2025_10",
                kind=DriftKind.MOVED,
                stored_release_time_et=None,
                current_release_time_et=datetime(2026, 1, 14, 8, 30, tzinfo=ET),
            )

    def test_moved_missing_current_raises(self) -> None:
        with pytest.raises(ValueError, match="MOVED drift requires both"):
            ScheduleDrift(
                series=MacroSeries.PPI,
                event_id="PPI_2025_10",
                kind=DriftKind.MOVED,
                stored_release_time_et=datetime(2025, 11, 14, 8, 30, tzinfo=ET),
                current_release_time_et=None,
            )

    def test_empty_event_id_raises(self) -> None:
        with pytest.raises(ValueError, match="event_id must be non-empty"):
            ScheduleDrift(
                series=MacroSeries.NFP,
                event_id="",
                kind=DriftKind.CANCELLED,
                stored_release_time_et=datetime(2025, 11, 7, 8, 30, tzinfo=ET),
                current_release_time_et=None,
            )

    def test_naive_stored_raises(self) -> None:
        with pytest.raises(ValueError, match="stored_release_time_et must be IANA-TZ-aware"):
            ScheduleDrift(
                series=MacroSeries.NFP,
                event_id="NFP_2025_10",
                kind=DriftKind.CANCELLED,
                stored_release_time_et=datetime(2025, 11, 7, 8, 30),  # naive
                current_release_time_et=None,
            )

    def test_frozen_immutable(self) -> None:
        d = ScheduleDrift(
            series=MacroSeries.NFP,
            event_id="NFP_2025_10",
            kind=DriftKind.CANCELLED,
            stored_release_time_et=datetime(2025, 11, 7, 8, 30, tzinfo=ET),
            current_release_time_et=None,
        )
        with pytest.raises(AttributeError):
            d.event_id = "x"  # type: ignore[misc]


# ============================================================================
# TestA1_24_MacroEventSourceABC
# ============================================================================


class TestA1_24_MacroEventSourceABC:
    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            MacroEventSource()  # type: ignore[abstract]

    def test_partial_subclass_still_abstract(self) -> None:
        class _Partial(MacroEventSource):
            @property
            def source_id(self) -> str:
                return "partial"

            # missing tier + upcoming_events + fetch_value

        with pytest.raises(TypeError, match="abstract"):
            _Partial()  # type: ignore[abstract]

    def test_full_concrete_works(self) -> None:
        src = _FixtureMacroSource()
        assert src.source_id == "fixture_macro"
        assert src.tier is SourceTier.T2_MACRO

    def test_default_schedule_drift_check_empty(self) -> None:
        assert _FixtureMacroSource().schedule_drift_check() == []

    def test_default_healthcheck_true(self) -> None:
        assert _FixtureMacroSource().healthcheck() is True

    def test_repr_format(self) -> None:
        assert repr(_FixtureMacroSource()) == (
            "<_FixtureMacroSource source_id='fixture_macro' tier=T2_MACRO>"
        )

    def test_fetch_value_returns_value_after_publication(self) -> None:
        mv = _valid_value(value_known_at_iso=datetime(2026, 6, 5, 12, 30, tzinfo=UTC))
        src = _FixtureMacroSource(mv)
        got = src.fetch_value("NFP_2026_06", datetime(2026, 6, 5, 13, 0, tzinfo=UTC))
        assert got is mv

    def test_fetch_value_pit_blocks_before_publication(self) -> None:
        """The concrete source routes through enforce_pit_gate (bug class 5 end-to-end)."""
        mv = _valid_value(value_known_at_iso=datetime(2026, 6, 5, 12, 30, tzinfo=UTC))
        src = _FixtureMacroSource(mv)
        got = src.fetch_value("NFP_2026_06", datetime(2026, 6, 5, 8, 0, tzinfo=UTC))
        assert got is None

    def test_fetch_value_unknown_event_returns_none(self) -> None:
        src = _FixtureMacroSource(_valid_value())
        assert src.fetch_value("CPI_2026_06", datetime(2026, 6, 5, 13, 0, tzinfo=UTC)) is None

    def test_fetch_value_boundary_at_publication_returned(self) -> None:
        ts = datetime(2026, 6, 5, 12, 30, tzinfo=UTC)
        mv = _valid_value(value_known_at_iso=ts)
        src = _FixtureMacroSource(mv)
        assert src.fetch_value("NFP_2026_06", ts) is mv
