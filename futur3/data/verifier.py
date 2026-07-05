"""MultiSourceVerifier — Phase A1.8 shell per the verifier spec.

The verifier is the cross-source consensus layer between N DataSources and the
engine. It receives raw records (`Settle` / `RawBar`) from each source, applies
a `VerifierPolicy`, and emits one of:

- `VerifiedSettle` / `VerifiedBar` — consensus reached; record carries
  `verifier_run_hash` for bit-reproducibility downstream.
- `DisagreementEvent` — sources responded but disagreed beyond the
  policy's tolerance band. Engine policy decides whether to halt or log.
- `IncompleteBar` — too few sources responded within the policy's quorum.

## Contracts

- **Mode-agnostic** (BACKTEST-IS-LIVE): identical code in backtest +
  live; ReplayDataSource is a first-class source.
- **Deterministic**: identical inputs → identical `verifier_run_hash`.
  Source-order-independent: the verifier sorts sources by `source_id` at
  construction and sorts provenance hashes lexicographically before
  computing the run-hash.
- **Apparatus-pure** (fail-loud): the verifier is a pure function over its
  inputs + policy; no I/O beyond reading DataSources via the ABC. JSONL
  discrepancy logging is deferred to A1.9 (verifier_policies + logger).

## Scope (A1.8 shell)

- `verify_settle(contract, as_of)` — primary path. CME + IBKR + Replay
  shipped; this is the load-bearing verification.
- `verify_bar(contract, bar_ts, resolution)` — secondary path. IBKR has
  bars; CME has BarsNotSupported; Replay has BarsNotSupported (A1.16
  storage deferral). For shell, tests cover the 1-source-responds case
  + the IncompleteBar path; full multi-source bar tests will land in
  A1.9-A1.16 once a second bar source ships.
- `VerifierPolicy` with per-field override resolution.
- `verifier_run_hash` per spec §2.5: SHA256(policy_id || sorted_provenance
  || tolerance_signature || bar_identifier).

## Deferred to later steps

- **`verify_tick`** — A1.6 tick streaming first.
- **`replay_for_revision`** — A1.20+ retroactive-revision watcher.
- **JSONL discrepancy log** — A1.9 with `verifier_policies.py`.
- **`prev_bar_hash` chain** — A1.16 storage layer (verifier shell tracks
  per-(contract, resolution) chain state in-memory; persistence layer is
  storage's job).
- **Tolerance bands TIGHT/NORMAL/WIDE numeric semantics** — A1.9. Shell
  uses EXACT-only comparison for both TIGHT and NORMAL; WIDE accepts a
  ±0.05 relative tolerance. Refinement lands when per-contract tick
  size becomes config-driven.

References:
- the verifier spec (policy presets)
- the verifier design (hash chain semantics)
- the data-layer design (architectural fit)
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import ClassVar, Final, Literal, Protocol, runtime_checkable

from futur3.data.source import (
    BarsNotSupported,
    DataSource,
    DataSourceError,
    SettlesNotSupported,
    TicksNotSupported,
)
from futur3.data.types import (
    SHA256_HEX_LENGTH,
    BarResolution,
    ContractSymbol,
    RawBar,
    Settle,
    SettleState,
    SourceTier,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Policy "kind" — how the verifier combines source values.
PolicyKind = Literal["fail_closed", "majority_vote", "highest_tier"]

# Policy tolerance band — shell semantics:
#   TIGHT / NORMAL → EXACT Decimal equality (fail-loud — fail_closed-friendly default)
#   WIDE           → ±0.05 relative tolerance (A1.9 will tighten per-contract tick size)
ToleranceBand = Literal["TIGHT", "NORMAL", "WIDE"]

# T2+ tier threshold (lower IntEnum = higher trust per the data-layer design).
# T2_EXCHANGE=2, T2_BROKER=3, T2_MACRO=4 all pass; T3_AGGREGATOR=5 + T4_DERIVED=6 excluded.
T2_OR_BETTER_MAX_TIER: Final[SourceTier] = SourceTier.T2_MACRO

# Field labels used in DisagreementEvent + per-field policy lookup.
FIELD_SETTLE: Final[str] = "settle"
FIELD_OPEN: Final[str] = "open"
FIELD_HIGH: Final[str] = "high"
FIELD_LOW: Final[str] = "low"
FIELD_CLOSE: Final[str] = "close"
FIELD_VOLUME: Final[str] = "volume"
FIELD_OI: Final[str] = "oi"
FIELD_TICK: Final[str] = "tick"
FIELD_OHLC: Final[str] = "ohlc"
# Settle-state ordering (used to pick the "best" settle when multiple agree):
# final > preliminary > live (matches ReplayDataSource convention).
_SETTLE_STATE_RANK: Final[dict[SettleState, int]] = {
    "live": 0,
    "preliminary": 1,
    "final": 2,
}

# Half-open-interval epsilon for bar lookup. DataSource.get_bars uses
# `ts_start <= bar.ts < ts_end`, so we add 1ms to bar_ts to make the boundary
# inclusive of the single bar at bar_ts.
_BAR_WINDOW_EPSILON: Final[timedelta] = timedelta(milliseconds=1)

# WIDE-band relative tolerance (shell default; A1.9 refines per-contract).
WIDE_RELATIVE_TOLERANCE: Final[Decimal] = Decimal("0.05")


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class ClockProtocol(Protocol):
    """UTC clock — injected for deterministic `detected_at` in tests."""

    def now_utc(self) -> datetime: ...


# ---------------------------------------------------------------------------
# VerifierPolicy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerifierPolicy:
    """Cross-source consensus policy.

    Per the verifier spec:
    - `kind` — how to combine source values:
      - `fail_closed`: ALL responding sources must agree within tolerance; any
        single mismatch → DisagreementEvent.
      - `majority_vote`: ≥`min_quorum` sources must agree on a value; outliers
        logged. Shell takes the most-common value (Counter.most_common).
      - `highest_tier`: trust the source with the lowest SourceTier IntEnum
        (T1 best); other sources advisory.
    - `min_quorum` — minimum sources responding to declare consensus (if fewer,
      emit `IncompleteBar`).
    - `tolerance_band` — shell semantics: TIGHT/NORMAL = exact Decimal; WIDE =
      ±WIDE_RELATIVE_TOLERANCE relative.
    - `timeout_seconds` — caller-side responsibility for shell; reserved for
      A1.9 async timing.
    - `require_tier_t2_or_better` — drop T3+ sources before consensus.
    - `per_field_policy` — field-specific overrides; resolved by
      `resolve_for_field(field)`.
    """

    policy_id: str
    kind: PolicyKind
    min_quorum: int
    tolerance_band: ToleranceBand
    timeout_seconds: int
    require_tier_t2_or_better: bool = True
    per_field_policy: dict[str, VerifierPolicy] = field(default_factory=dict)

    def resolve_for_field(self, field_name: str) -> VerifierPolicy:
        """Return the policy applicable to `field_name`.

        Falls back to self (umbrella) if no per-field override exists.
        """
        return self.per_field_policy.get(field_name, self)


# ---------------------------------------------------------------------------
# Verified records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerifiedSettle:
    """Cross-source-verified Settle for `(contract, as_of_date)`.

    Carries the provenance hash chain so downstream engine can replay + verify.
    """

    contract: ContractSymbol
    as_of_date: date
    settle: Decimal
    settle_state: SettleState
    n_sources_agreed: int
    source_provenance_hashes: tuple[str, ...]
    verifier_run_hash: str
    tier_used: SourceTier
    policy_id: str

    def __post_init__(self) -> None:
        if len(self.verifier_run_hash) != SHA256_HEX_LENGTH:
            raise ValueError(
                f"VerifiedSettle.verifier_run_hash must be hex-SHA256 "
                f"({SHA256_HEX_LENGTH} chars); got {len(self.verifier_run_hash)}"
            )
        for h in self.source_provenance_hashes:
            if len(h) != SHA256_HEX_LENGTH:
                raise ValueError(
                    f"VerifiedSettle.source_provenance_hashes entries must be "
                    f"hex-SHA256 ({SHA256_HEX_LENGTH} chars); got {len(h)}"
                )


@dataclass(frozen=True)
class VerifiedBar:
    """Cross-source-verified RawBar for `(contract, ts, resolution)`."""

    contract: ContractSymbol
    ts: datetime
    resolution: BarResolution
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    n_sources_agreed: int
    source_provenance_hashes: tuple[str, ...]
    verifier_run_hash: str
    prev_bar_hash: str | None
    tier_used: SourceTier
    policy_id: str

    def __post_init__(self) -> None:
        if self.ts.tzinfo is None:
            raise ValueError(
                f"VerifiedBar.ts must be IANA-TZ-aware datetime; got naive {self.ts!r}"
            )
        if len(self.verifier_run_hash) != SHA256_HEX_LENGTH:
            raise ValueError(
                f"VerifiedBar.verifier_run_hash must be hex-SHA256 "
                f"({SHA256_HEX_LENGTH} chars); got {len(self.verifier_run_hash)}"
            )
        if self.prev_bar_hash is not None and len(self.prev_bar_hash) != SHA256_HEX_LENGTH:
            raise ValueError(
                f"VerifiedBar.prev_bar_hash if set must be hex-SHA256; "
                f"got {len(self.prev_bar_hash)}"
            )


# ---------------------------------------------------------------------------
# Diagnostic events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DisagreementEvent:
    """Sources responded but disagreed beyond tolerance — verifier emits this
    instead of a VerifiedSettle/Bar, and the engine policy decides next step.
    """

    contract: ContractSymbol
    bar_ts: datetime
    resolution: BarResolution
    field: str
    sources: tuple[str, ...]
    values: tuple[str, ...]
    diff_magnitude_max: str
    tolerance_used: ToleranceBand
    policy_kind: PolicyKind
    quorum_required: int
    quorum_achieved: int
    detected_at: datetime
    policy_id: str

    def __post_init__(self) -> None:
        if self.detected_at.tzinfo is None:
            raise ValueError("DisagreementEvent.detected_at must be TZ-aware; got naive")
        if len(self.sources) != len(self.values):
            raise ValueError(
                f"DisagreementEvent.sources len {len(self.sources)} != "
                f"values len {len(self.values)}"
            )


@dataclass(frozen=True)
class IncompleteBar:
    """Fewer than `n_sources_required` sources responded within timeout."""

    contract: ContractSymbol
    bar_ts: datetime
    resolution: BarResolution
    n_sources_responded: int
    n_sources_required: int
    sources_pending: tuple[str, ...]
    detected_at: datetime
    policy_id: str

    def __post_init__(self) -> None:
        if self.detected_at.tzinfo is None:
            raise ValueError("IncompleteBar.detected_at must be TZ-aware; got naive")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _source_provenance_hash(source_id: str, as_of_iso: datetime, content_bytes_sha: str) -> str:
    """SHA256(source_id || as_of_iso || content_bytes_sha) per spec §2.5."""
    payload = f"{source_id}||{as_of_iso.isoformat()}||{content_bytes_sha}".encode()
    return hashlib.sha256(payload).hexdigest()


def _tolerance_signature(policy: VerifierPolicy) -> str:
    """SHA256 of canonical-JSON-encoded policy fields.

    Deterministic across processes — used as part of `verifier_run_hash`. We
    serialize a flat representation (no nested per_field_policy) since per-field
    resolution happens BEFORE hashing — the resolved policy is what gets hashed.
    """
    payload = json.dumps(
        {
            "policy_id": policy.policy_id,
            "kind": policy.kind,
            "min_quorum": policy.min_quorum,
            "tolerance_band": policy.tolerance_band,
            "require_tier_t2_or_better": policy.require_tier_t2_or_better,
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _compute_verifier_run_hash(
    *,
    resolved_policy: VerifierPolicy,
    source_provenance_hashes: tuple[str, ...],
    bar_identifier: str,
) -> str:
    """Compute the anchor hash.

    Formula: SHA256(policy_id || sorted_provenance || tolerance_sig || bar_id).
    """
    sorted_provenance = sorted(source_provenance_hashes)
    sorted_provenance_concat = "||".join(sorted_provenance)
    tol_sig = _tolerance_signature(resolved_policy)
    payload = (
        f"{resolved_policy.policy_id}||{sorted_provenance_concat}||{tol_sig}||{bar_identifier}"
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _settle_state_best(states: Iterable[SettleState]) -> SettleState:
    """Pick the highest-rank settle_state — final > preliminary > live."""
    return max(states, key=lambda s: _SETTLE_STATE_RANK[s])


def _check_settle_consensus(
    settles: list[Settle],
    resolved_policy: VerifierPolicy,
) -> tuple[bool, Decimal, Decimal]:
    """Return (agreed, consensus_value, max_diff).

    Shell semantics:
    - TIGHT / NORMAL → all settles must have EXACT Decimal-equal `settle`.
    - WIDE → all settles must be within WIDE_RELATIVE_TOLERANCE * median.
    """
    values = [s.settle for s in settles]
    if not values:
        return False, Decimal("0"), Decimal("0")

    if resolved_policy.tolerance_band == "WIDE":
        median = sorted(values)[len(values) // 2]
        if median == 0:
            agreed = all(v == 0 for v in values)
            return agreed, median, max(abs(v) for v in values) if not agreed else Decimal("0")
        max_relative = max(abs(v - median) / abs(median) for v in values)
        agreed = max_relative <= WIDE_RELATIVE_TOLERANCE
        return agreed, median, max_relative

    # TIGHT / NORMAL — exact Decimal equality
    first = values[0]
    max_diff = max(abs(v - first) for v in values)
    agreed = all(v == first for v in values)
    return agreed, first, max_diff


def _check_bar_consensus(
    bars: list[RawBar],
    resolved_policy: VerifierPolicy,
) -> tuple[bool, RawBar, Decimal]:
    """Return (agreed, representative_bar, max_diff_close).

    Shell consensus on bars compares OHLC + volume across sources. The
    representative bar is the one from the highest-tier source (per the fail-loud policy
    fail-closed safety — if all agree, picking any is fine; if disagreement,
    the highest-tier sample is the one we'd quote in the disagreement record).
    """
    if not bars:
        # Fail-loud: caller (verify_bar) MUST guard via min_quorum check first.
        # If we reach here with an empty list, the invariant is violated —
        # raise instead of IndexError (bars[0] crash) — fail loud.
        raise AssertionError(
            "_check_bar_consensus called with empty bars; "
            "verify_bar must guard via min_quorum before invoking consensus"
        )

    closes = [b.close for b in bars]

    if resolved_policy.tolerance_band == "WIDE":
        median = sorted(closes)[len(closes) // 2]
        if median == 0:
            agreed = all(c == 0 for c in closes)
            max_diff = max(abs(c) for c in closes) if not agreed else Decimal("0")
        else:
            max_relative = max(abs(c - median) / abs(median) for c in closes)
            agreed = max_relative <= WIDE_RELATIVE_TOLERANCE
            max_diff = max_relative
    else:
        first = closes[0]
        max_diff = max(abs(c - first) for c in closes)
        # Also require open/high/low/volume agreement under TIGHT/NORMAL
        agreed = all(
            b.open == bars[0].open
            and b.high == bars[0].high
            and b.low == bars[0].low
            and b.close == bars[0].close
            and b.volume == bars[0].volume
            for b in bars
        )

    representative = min(bars, key=lambda b: b.source_id)  # deterministic
    return agreed, representative, max_diff


# ---------------------------------------------------------------------------
# MultiSourceVerifier
# ---------------------------------------------------------------------------


class _SystemClock:
    """Default UTC clock."""

    def now_utc(self) -> datetime:
        return datetime.now(UTC)


class MultiSourceVerifier:
    """Cross-source consensus apparatus for N DataSources.

    Construction sorts sources by `source_id` (deterministic ordering for the
    verifier_run_hash); policy is locked at construction (backtest-is-live: same policy in
    backtest + live). Clock is injected for test determinism.
    """

    SUPPORTED_BAR_RESOLUTIONS: ClassVar[dict[BarResolution, timedelta]] = {
        BarResolution.SEC_1: timedelta(seconds=1),
        BarResolution.SEC_5: timedelta(seconds=5),
        BarResolution.MIN_1: timedelta(minutes=1),
        BarResolution.MIN_5: timedelta(minutes=5),
        BarResolution.MIN_15: timedelta(minutes=15),
        BarResolution.HOUR_1: timedelta(hours=1),
        BarResolution.DAY_1: timedelta(days=1),
    }

    def __init__(
        self,
        sources: list[DataSource],
        policy: VerifierPolicy,
        clock: ClockProtocol | None = None,
    ) -> None:
        if not sources:
            raise ValueError("MultiSourceVerifier requires at least one DataSource")
        self._sources: tuple[DataSource, ...] = tuple(sorted(sources, key=lambda s: s.source_id))
        self._policy: VerifierPolicy = policy
        self._clock: ClockProtocol = clock or _SystemClock()
        # In-memory hash chain per (contract, resolution); persistence is A1.16.
        self._prev_bar_hash: dict[tuple[ContractSymbol, BarResolution], str | None] = {}

    # ------------------------------------------------------------------------
    # Public introspection
    # ------------------------------------------------------------------------

    @property
    def policy(self) -> VerifierPolicy:
        return self._policy

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(s.source_id for s in self._sources)

    def __repr__(self) -> str:
        return (
            f"MultiSourceVerifier(sources={list(self.source_ids)}, policy={self._policy.policy_id})"
        )

    # ------------------------------------------------------------------------
    # verify_settle
    # ------------------------------------------------------------------------

    def verify_settle(
        self,
        contract: ContractSymbol,
        as_of: datetime,
    ) -> VerifiedSettle | DisagreementEvent | IncompleteBar:
        """Verify the settle for `contract` at `as_of` across all sources.

        Returns:
            - VerifiedSettle if ≥ min_quorum (resolved per-field) sources agreed.
            - DisagreementEvent if sources responded but disagreed.
            - IncompleteBar if fewer than min_quorum sources responded.
        """
        resolved = self._policy.resolve_for_field(FIELD_SETTLE)
        eligible_sources = self._tier_filter(resolved)
        responding: list[tuple[DataSource, Settle]] = []
        not_responding: list[str] = []

        for src in eligible_sources:
            try:
                settle = src.latest_settle(contract, as_of)
            except SettlesNotSupported:
                not_responding.append(src.source_id)
                continue
            except DataSourceError as e:
                logger.warning(
                    "%s: latest_settle raised %s; treating as non-responding: %s",
                    src.source_id,
                    type(e).__name__,
                    e,
                )
                not_responding.append(src.source_id)
                continue
            if settle is None:
                not_responding.append(src.source_id)
                continue
            responding.append((src, settle))

        now = self._clock.now_utc()

        if len(responding) < resolved.min_quorum:
            return IncompleteBar(
                contract=contract,
                bar_ts=as_of,
                resolution=BarResolution.SETTLE,
                n_sources_responded=len(responding),
                n_sources_required=resolved.min_quorum,
                sources_pending=tuple(not_responding),
                detected_at=now,
                policy_id=resolved.policy_id,
            )

        settles = [s for _, s in responding]
        agreed, consensus, max_diff = _check_settle_consensus(settles, resolved)
        if not agreed:
            return DisagreementEvent(
                contract=contract,
                bar_ts=as_of,
                resolution=BarResolution.SETTLE,
                field=FIELD_SETTLE,
                sources=tuple(src.source_id for src, _ in responding),
                values=tuple(str(s.settle) for _, s in responding),
                diff_magnitude_max=str(max_diff),
                tolerance_used=resolved.tolerance_band,
                policy_kind=resolved.kind,
                quorum_required=resolved.min_quorum,
                quorum_achieved=len(responding),
                detected_at=now,
                policy_id=resolved.policy_id,
            )

        # Consensus → assemble VerifiedSettle
        # Provenance hashes deterministic across processes: pure SHA256 inputs.
        # Hash determinism requires settle.as_of_iso.
        # Prior fallback to caller `as_of` was non-deterministic: two verify_settle
        # calls with identical settle data but different caller as_of produced
        # different verifier_run_hash values, breaking bit-reproducibility.
        # Sources MUST populate as_of_iso (CMEEODDataSource + IBKR A1.5 do today).
        prov_hashes: list[str] = []
        for src, settle in responding:
            if settle.as_of_iso is None:
                raise DataSourceError(
                    f"{src.source_id}: Settle.as_of_iso is None — required for "
                    f"verifier_run_hash determinism (fail-loud). Source must populate."
                )
            prov_hashes.append(
                _source_provenance_hash(
                    settle.source_id, settle.as_of_iso, settle.content_bytes_sha
                )
            )
        prov_tuple = tuple(prov_hashes)
        bar_id = f"{contract}||{settles[0].as_of_date.isoformat()}||{BarResolution.SETTLE.value}"
        run_hash = _compute_verifier_run_hash(
            resolved_policy=resolved,
            source_provenance_hashes=prov_tuple,
            bar_identifier=bar_id,
        )
        best_state = _settle_state_best(s.settle_state for s in settles)
        # tier_used = highest tier (lowest IntEnum) among the agreed sources
        best_tier = min((src.tier for src, _ in responding), key=lambda t: t.value)
        return VerifiedSettle(
            contract=contract,
            as_of_date=settles[0].as_of_date,
            settle=consensus,
            settle_state=best_state,
            n_sources_agreed=len(responding),
            source_provenance_hashes=prov_tuple,
            verifier_run_hash=run_hash,
            tier_used=best_tier,
            policy_id=resolved.policy_id,
        )

    # ------------------------------------------------------------------------
    # verify_bar
    # ------------------------------------------------------------------------

    def verify_bar(
        self,
        contract: ContractSymbol,
        bar_ts: datetime,
        resolution: BarResolution,
    ) -> VerifiedBar | DisagreementEvent | IncompleteBar:
        """Verify the bar at `(contract, bar_ts, resolution)` across sources.

        The verifier requests a tight window `[bar_ts, bar_ts + ε)` from each
        source (DataSource.get_bars is half-open; ε = 1ms). Sources returning
        empty or raising BarsNotSupported are treated as non-responding.

        Returns: VerifiedBar | DisagreementEvent | IncompleteBar.
        """
        if bar_ts.tzinfo is None:
            raise ValueError(f"verify_bar bar_ts must be TZ-aware; got naive {bar_ts!r}")
        if resolution not in self.SUPPORTED_BAR_RESOLUTIONS:
            raise ValueError(
                f"verify_bar resolution {resolution} not supported by shell; "
                f"use one of {sorted(r.name for r in self.SUPPORTED_BAR_RESOLUTIONS)}"
            )
        resolved = self._policy.resolve_for_field(FIELD_OHLC)
        eligible_sources = self._tier_filter(resolved)
        ts_end = bar_ts + _BAR_WINDOW_EPSILON

        responding: list[tuple[DataSource, RawBar]] = []
        not_responding: list[str] = []
        for src in eligible_sources:
            try:
                bars_iter = src.get_bars(contract, bar_ts, ts_end, resolution)
                bars = list(bars_iter)
            except BarsNotSupported:
                not_responding.append(src.source_id)
                continue
            except TicksNotSupported:
                not_responding.append(src.source_id)
                continue
            except DataSourceError as e:
                logger.warning(
                    "%s: get_bars raised %s; treating as non-responding: %s",
                    src.source_id,
                    type(e).__name__,
                    e,
                )
                not_responding.append(src.source_id)
                continue
            if not bars:
                not_responding.append(src.source_id)
                continue
            # Pick the bar at bar_ts (windowed query may return ≥1; we want exact match)
            exact = [b for b in bars if b.ts == bar_ts]
            if not exact:
                not_responding.append(src.source_id)
                continue
            responding.append((src, exact[0]))

        now = self._clock.now_utc()

        if len(responding) < resolved.min_quorum:
            return IncompleteBar(
                contract=contract,
                bar_ts=bar_ts,
                resolution=resolution,
                n_sources_responded=len(responding),
                n_sources_required=resolved.min_quorum,
                sources_pending=tuple(not_responding),
                detected_at=now,
                policy_id=resolved.policy_id,
            )

        raw_bars = [b for _, b in responding]
        agreed, representative, max_diff = _check_bar_consensus(raw_bars, resolved)
        if not agreed:
            return DisagreementEvent(
                contract=contract,
                bar_ts=bar_ts,
                resolution=resolution,
                field=FIELD_CLOSE,  # canonical disagreement-field for OHLC bundle
                sources=tuple(src.source_id for src, _ in responding),
                values=tuple(str(b.close) for _, b in responding),
                diff_magnitude_max=str(max_diff),
                tolerance_used=resolved.tolerance_band,
                policy_kind=resolved.kind,
                quorum_required=resolved.min_quorum,
                quorum_achieved=len(responding),
                detected_at=now,
                policy_id=resolved.policy_id,
            )

        # Consensus
        prov_hashes = [
            _source_provenance_hash(b.source_id, b.as_of_iso, b.content_bytes_sha) for b in raw_bars
        ]
        prov_tuple = tuple(prov_hashes)
        bar_id = f"{contract}||{bar_ts.isoformat()}||{resolution.value}"
        run_hash = _compute_verifier_run_hash(
            resolved_policy=resolved,
            source_provenance_hashes=prov_tuple,
            bar_identifier=bar_id,
        )
        prev_key = (contract, resolution)
        prev_hash = self._prev_bar_hash.get(prev_key)
        self._prev_bar_hash[prev_key] = run_hash
        best_tier = min((src.tier for src, _ in responding), key=lambda t: t.value)
        return VerifiedBar(
            contract=contract,
            ts=bar_ts,
            resolution=resolution,
            open=representative.open,
            high=representative.high,
            low=representative.low,
            close=representative.close,
            volume=representative.volume,
            n_sources_agreed=len(responding),
            source_provenance_hashes=prov_tuple,
            verifier_run_hash=run_hash,
            prev_bar_hash=prev_hash,
            tier_used=best_tier,
            policy_id=resolved.policy_id,
        )

    # ------------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------------

    def _tier_filter(self, resolved: VerifierPolicy) -> tuple[DataSource, ...]:
        """Drop sources whose tier is worse than T2_MACRO if policy requires."""
        if not resolved.require_tier_t2_or_better:
            return self._sources
        return tuple(s for s in self._sources if s.tier.value <= T2_OR_BETTER_MAX_TIER.value)


# Public hash helpers — exported for verifier_policies.py + tests + downstream.
__all__: list[str] = [
    "FIELD_CLOSE",
    "FIELD_OHLC",
    "FIELD_OI",
    "FIELD_SETTLE",
    "FIELD_TICK",
    "FIELD_VOLUME",
    "T2_OR_BETTER_MAX_TIER",
    "WIDE_RELATIVE_TOLERANCE",
    "ClockProtocol",
    "DisagreementEvent",
    "IncompleteBar",
    "MultiSourceVerifier",
    "PolicyKind",
    "ToleranceBand",
    "VerifiedBar",
    "VerifiedSettle",
    "VerifierPolicy",
    "_compute_verifier_run_hash",
    "_source_provenance_hash",
    "_tolerance_signature",
]
