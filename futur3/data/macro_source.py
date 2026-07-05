"""futur3.data.macro_source - MacroEventSource ABC + macro exception hierarchy + PIT gate.

Per internal design notes

Parallel to (NOT a subclass of) `futur3.data.source.DataSource`: macro events are scheduled
economic releases, not price bars, so the surface is `upcoming_events` / `fetch_value` /
`schedule_drift_check` rather than `get_bars` / `get_ticks` / `latest_settle`. Both ABCs share
the provenance discipline (SHA256 content hashes, tz-aware boundaries) and the `DataSourceError`
root, so the engine can catch data-acquisition failures uniformly.

Hard seam (CI-grep enforced): only code in `futur3.data.sources.*` imports vendor
SDKs (`fredapi`, `requests`-to-bls, etc.). Engine + strategy code consume this ABC.

Hard PIT invariant (bug class 5): `fetch_value` MUST route its result through
`enforce_pit_gate` so no not-yet-published value can leak into a backtest at an earlier as_of.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from futur3.data.macro_types import MacroEvent, MacroSeries, MacroValue
from futur3.data.source import DataSourceError
from futur3.data.types import SourceTier, _assert_tz_aware

# ----------------------------------------------------------------------------
# Exception hierarchy (rooted at DataSourceError for uniform engine catch)
# ----------------------------------------------------------------------------


class MacroSourceError(DataSourceError):
    """Parent for macro-event-source errors.

    Extends `DataSourceError` so the engine catches macro acquisition failures with the same
    handler as bar/tick/settle failures.
    """


class ScheduleDriftError(MacroSourceError):
    """Raised when a concrete source fail-closes on detected calendar drift.

    Default drift handling returns `ScheduleDrift` DATA via `schedule_drift_check`; raising
    this is the fail-closed escalation a source uses rather than silently serving a stale
    forward calendar (release moved / cancelled).
    """


class ShutdownVoidError(MacroSourceError):
    """Raised by `require_value` when a release was permanently voided.

    The 2025-10 federal shutdown class: the number does not exist (e.g. Oct-2025 NFP household
    survey was never collected). Strategies must not synthesize it - fail loud instead of
    treating a missing release as zero (bug class 5 / 8 hazard).
    """


# ----------------------------------------------------------------------------
# Schedule-drift record (DATA returned by schedule_drift_check; not an exception)
# ----------------------------------------------------------------------------


class DriftKind(StrEnum):
    """Kind of divergence between the stored forward calendar and the publisher's current one."""

    MOVED = "moved"  # release time changed (shutdown delay, weather, etc.)
    CANCELLED = "cancelled"  # release removed from publisher calendar (permanent void)
    ADDED = "added"  # a new release appeared on the publisher calendar


@dataclass(frozen=True)
class ScheduleDrift:
    """A detected divergence between futur3's stored forward calendar and the publisher's.

    Returned as DATA (not raised) by `MacroEventSource.schedule_drift_check`. Whether to
    fail-closed (raise `ScheduleDriftError`) on a given drift is the concrete source's policy.
    """

    series: MacroSeries
    event_id: str
    kind: DriftKind
    stored_release_time_et: datetime | None  # None for ADDED
    current_release_time_et: datetime | None  # None for CANCELLED

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("ScheduleDrift.event_id must be non-empty")
        if self.stored_release_time_et is not None:
            _assert_tz_aware(self.stored_release_time_et, "ScheduleDrift.stored_release_time_et")
        if self.current_release_time_et is not None:
            _assert_tz_aware(self.current_release_time_et, "ScheduleDrift.current_release_time_et")
        if self.kind is DriftKind.CANCELLED and self.current_release_time_et is not None:
            raise ValueError("CANCELLED drift must have current_release_time_et=None")
        if self.kind is DriftKind.ADDED and self.stored_release_time_et is not None:
            raise ValueError("ADDED drift must have stored_release_time_et=None")
        if self.kind is DriftKind.MOVED and (
            self.stored_release_time_et is None or self.current_release_time_et is None
        ):
            raise ValueError("MOVED drift requires both stored and current release times")


# ----------------------------------------------------------------------------
# PIT gate + value guard (the bug-class-5 + shutdown-void helpers)
# ----------------------------------------------------------------------------


def enforce_pit_gate(value: MacroValue | None, as_of_iso: datetime) -> MacroValue | None:
    """The hard point-in-time filter (bug class 5 / look-ahead).

    Returns `value` only if it was known at or before `as_of_iso`; otherwise None. Every
    `MacroEventSource.fetch_value` implementation MUST route its result through this so no
    source can leak a not-yet-published value into a backtest decision at an earlier as_of.
    """
    _assert_tz_aware(as_of_iso, "enforce_pit_gate.as_of_iso")
    if value is None:
        return None
    if value.value_known_at_iso > as_of_iso:
        return None
    return value


def require_value(mv: MacroValue) -> Decimal:
    """Read a macro value with a fail-loud guard for shutdown-void releases.

    Strategy code that cannot operate on a missing release calls this rather than treating
    None as zero (a bug class 5 / 8 hazard). Raises `ShutdownVoidError` when the release was
    voided (e.g. Oct-2025 NFP, permanently uncollected during the federal shutdown).
    """
    if mv.is_shutdown_void or mv.value is None:
        raise ShutdownVoidError(
            f"{mv.event_id} ({mv.series.value}) is shutdown-void - no value exists; "
            "strategies must not synthesize it"
        )
    return mv.value


# ----------------------------------------------------------------------------
# MacroEventSource ABC
# ----------------------------------------------------------------------------


class MacroEventSource(abc.ABC):
    """The seam between futur3 and macro-release publishers (BLS / BEA / Census / Fed / FRED).

    Subclasses MUST implement: `source_id`, `tier`, `upcoming_events`, `fetch_value`.
    Subclasses MAY override: `schedule_drift_check` (default []), `healthcheck` (default True).

    PIT CONTRACT (bug class 5): `fetch_value` MUST apply `enforce_pit_gate` before returning,
    so a not-yet-published value can never reach a backtest at an earlier as_of.
    """

    @property
    @abc.abstractmethod
    def source_id(self) -> str:
        """Stable per-source provenance string (e.g. 'bls_api_v2', 'fred_alfred')."""
        ...

    @property
    @abc.abstractmethod
    def tier(self) -> SourceTier:
        """SourceTier classification.

        Direct government publishers are `T2_MACRO`; the FRED aggregator is `T3_AGGREGATOR`
        (lower rank wins, so the verifier prefers direct over aggregator on disagreement).
        """
        ...

    @abc.abstractmethod
    def upcoming_events(self, start: datetime, end: datetime) -> list[MacroEvent]:
        """Forward-known release calendar for the half-open window [start, end) (tz-aware).

        Raises `SchemaMismatch` if the publisher calendar page schema-hash diverged (bug
        class 9). Raises `MacroSourceError` subclasses on acquisition failure.
        """
        ...

    @abc.abstractmethod
    def fetch_value(self, event_id: str, as_of_iso: datetime) -> MacroValue | None:
        """The PIT-correct published value for `event_id` as known at `as_of_iso`.

        Returns None if the release was not yet published at `as_of_iso` (PIT gate) or the
        event is unknown to this source. Implementations MUST apply `enforce_pit_gate`.

        Raises `FutureDatedSourceError` if the publisher reports a `value_known_at_iso` in the
        future relative to wall-clock now (source clock-skew / lying - bug class 5).
        """
        ...

    def schedule_drift_check(self) -> list[ScheduleDrift]:
        """Compare the stored forward calendar against the publisher's current calendar.

        Default: no drift detection (returns []). Sources backed by a live publisher calendar
        override to detect moved / cancelled / added releases (e.g. 2025-shutdown delays).
        """
        return []

    def healthcheck(self) -> bool:
        """Lightweight liveness check. Default True; network-backed sources override."""
        return True

    def __repr__(self) -> str:
        return f"<{type(self).__name__} source_id={self.source_id!r} tier={self.tier.name}>"
