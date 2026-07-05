"""Verifier policy presets + env-var loader — Phase A1.8 (shell) + A1.9 (expansion).

Per the verifier spec. Three presets, locked at
process startup via env-var `FUTUR3_VERIFIER_POLICY` (enforcement: same
policy in backtest + live; re-resolution mid-process is forbidden — CI test
in A1.20+ will assert no mid-process mutation).

## Phase progression invariants (immutable across all 3 presets)

- `settle.kind == "fail_closed"` always — EXACT match required across CME +
  ≥1 broker. Disagreement quarantines the contract. hard anchor invariant.
- `tick.kind == "fail_closed"` always — per-tick correctness non-negotiable
  for execution.
- `oi.kind == "highest_tier"` always — CME canonical; no cross-source
  consensus replaces it.

## POLICY_PHASE_A1_DEFAULT (A1.8)

Maximally conservative for backtest + paper-live dev:
- Umbrella: fail_closed, min_quorum=3, NORMAL, T2+ required.
- Per-field overrides: settle (quorum=2 TIGHT), tick (quorum=2 TIGHT 15s),
  ohlc (majority_vote quorum=3 NORMAL), volume (majority_vote quorum=3 WIDE),
  oi (highest_tier quorum=1 NORMAL).

## POLICY_PHASE_B1_SHADOW (A1.9)

Paper-personal live-shadow mode. Per-field policy IDENTICAL to A1 — the change
is engine-side (`RuntimeContext.degraded_mode_allowed`), not verifier-side.
This preserves backtest-is-live (same verifier code backtest + live) and decouples policy
iteration from engine iteration.

## POLICY_PHASE_C_LIVE (A1.9)

Funded-account live trading. Quorum tightens to 3-of-N (where N ≥ 5 sources
active — IBKR live + CME scraper + 1 crypto venue + Yahoo + Barchart etc.):
- Umbrella: majority_vote, min_quorum=3, NORMAL.
- settle + tick inherit A1's fail_closed exact policies (anchor invariant).
- ohlc + volume bump to policy_id `*_majority_c_v1` (same shape as A1's, new id
  reflects the phase context — useful for `verifier_run_hash` policy provenance
  + per-field log analytics).
- oi inherits A1's highest_tier policy.

## resolve_policy(env)

Reads `FUTUR3_VERIFIER_POLICY` env-var (case-insensitive). Default = PHASE_A1.
Unknown value → `UnknownPolicyError` (NO silent fallback per the fail-loud policy). Test code
injects an `env` mapping to avoid `os.environ` mutation.

References:
- the verifier spec
- the verifier design
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Final

from futur3.data.source import DataSourceError
from futur3.data.verifier import (
    FIELD_OHLC,
    FIELD_OI,
    FIELD_SETTLE,
    FIELD_TICK,
    FIELD_VOLUME,
    VerifierPolicy,
)

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class UnknownPolicyError(DataSourceError):
    """`FUTUR3_VERIFIER_POLICY` was set to a value not in the known mapping.

    Per the fail-loud policy: silent fallback to default in production
    is a bug class 19 (config drift). We raise immediately so the operator must
    fix the env-var explicitly.
    """


# ---------------------------------------------------------------------------
# Per-field overrides (assembled below into the umbrella policies)
# ---------------------------------------------------------------------------


_SETTLE_EXACT = VerifierPolicy(
    policy_id="settle_exact_v1",
    kind="fail_closed",
    min_quorum=2,
    tolerance_band="TIGHT",
    timeout_seconds=60,
    require_tier_t2_or_better=True,
)

_TICK_EXACT = VerifierPolicy(
    policy_id="tick_exact_v1",
    kind="fail_closed",
    min_quorum=2,
    tolerance_band="TIGHT",
    timeout_seconds=15,
    require_tier_t2_or_better=True,
)

_OHLC_MAJORITY_A1 = VerifierPolicy(
    policy_id="ohlc_majority_v1",
    kind="majority_vote",
    min_quorum=3,
    tolerance_band="NORMAL",
    timeout_seconds=60,
    require_tier_t2_or_better=True,
)

_VOLUME_MAJORITY_A1 = VerifierPolicy(
    policy_id="volume_majority_v1",
    kind="majority_vote",
    min_quorum=3,
    tolerance_band="WIDE",
    timeout_seconds=60,
    require_tier_t2_or_better=True,
)

_OI_TIER = VerifierPolicy(
    policy_id="oi_tier_v1",
    kind="highest_tier",
    min_quorum=1,
    tolerance_band="NORMAL",
    timeout_seconds=60,
    require_tier_t2_or_better=True,
)

# Phase-C-specific bar policies (same shape as A1's, distinct policy_id for
# per-field provenance + log analytics differentiation).
_OHLC_MAJORITY_C = VerifierPolicy(
    policy_id="ohlc_majority_c_v1",
    kind="majority_vote",
    min_quorum=3,
    tolerance_band="NORMAL",
    timeout_seconds=60,
    require_tier_t2_or_better=True,
)

_VOLUME_MAJORITY_C = VerifierPolicy(
    policy_id="volume_majority_c_v1",
    kind="majority_vote",
    min_quorum=3,
    tolerance_band="WIDE",
    timeout_seconds=60,
    require_tier_t2_or_better=True,
)


def _phase_a1_per_field() -> dict[str, VerifierPolicy]:
    """A1 per-field policy block (shared by B1 verbatim)."""
    return {
        FIELD_SETTLE: _SETTLE_EXACT,
        FIELD_TICK: _TICK_EXACT,
        FIELD_OHLC: _OHLC_MAJORITY_A1,
        FIELD_VOLUME: _VOLUME_MAJORITY_A1,
        FIELD_OI: _OI_TIER,
    }


def _phase_c_per_field() -> dict[str, VerifierPolicy]:
    """C-specific per-field — settles+ticks+oi inherit A1; ohlc+volume bump to *_c_v1."""
    return {
        FIELD_SETTLE: _SETTLE_EXACT,  # anchor invariant — same across all 3 presets
        FIELD_TICK: _TICK_EXACT,  # anchor invariant
        FIELD_OHLC: _OHLC_MAJORITY_C,  # Phase-C variant
        FIELD_VOLUME: _VOLUME_MAJORITY_C,  # Phase-C variant
        FIELD_OI: _OI_TIER,  # anchor invariant
    }


# ---------------------------------------------------------------------------
# Umbrella policy presets
# ---------------------------------------------------------------------------


POLICY_PHASE_A1_DEFAULT: Final[VerifierPolicy] = VerifierPolicy(
    policy_id="phase_a1_default_v1",
    kind="fail_closed",
    min_quorum=3,
    tolerance_band="NORMAL",
    timeout_seconds=60,
    require_tier_t2_or_better=True,
    per_field_policy=_phase_a1_per_field(),
)


POLICY_PHASE_B1_SHADOW: Final[VerifierPolicy] = VerifierPolicy(
    policy_id="phase_b1_shadow_v1",
    kind="fail_closed",
    min_quorum=3,
    tolerance_band="NORMAL",
    timeout_seconds=60,
    require_tier_t2_or_better=True,
    # Same per-field block as A1 — verifier-side is identical; degraded-mode opt-in
    # is engine-side via RuntimeContext.degraded_mode_allowed (A1.12+).
    per_field_policy=_phase_a1_per_field(),
)


POLICY_PHASE_C_LIVE: Final[VerifierPolicy] = VerifierPolicy(
    policy_id="phase_c_live_v1",
    kind="majority_vote",  # higher-N pool → majority by default
    min_quorum=3,  # 3-of-N where N ≥ 5
    tolerance_band="NORMAL",
    timeout_seconds=60,
    require_tier_t2_or_better=True,
    per_field_policy=_phase_c_per_field(),
)


# ---------------------------------------------------------------------------
# Env-var loader
# ---------------------------------------------------------------------------


VALID_POLICY_NAMES: Final[dict[str, VerifierPolicy]] = {
    "PHASE_A1": POLICY_PHASE_A1_DEFAULT,
    "PHASE_B1": POLICY_PHASE_B1_SHADOW,
    "PHASE_C": POLICY_PHASE_C_LIVE,
}

DEFAULT_POLICY_NAME: Final[str] = "PHASE_A1"
POLICY_ENV_VAR: Final[str] = "FUTUR3_VERIFIER_POLICY"


def resolve_policy(env: Mapping[str, str] | None = None) -> VerifierPolicy:
    """Resolve the active VerifierPolicy from `FUTUR3_VERIFIER_POLICY`.

    Per BACKTEST-IS-LIVE: callers should invoke this ONCE at process start
    and pass the resolved policy down. The verifier itself takes the policy
    at construction; mid-process mutation is a future CI grep failure.

    Per the fail-loud policy: unknown value raises immediately; no
    silent fallback to default in production.

    Args:
        env: Environment mapping (defaults to `os.environ`). Tests inject
            a plain dict to avoid mutating `os.environ`.

    Returns:
        VerifierPolicy: One of POLICY_PHASE_A1_DEFAULT / B1_SHADOW / C_LIVE.

    Raises:
        UnknownPolicyError: env-var set to a value not in VALID_POLICY_NAMES.
    """
    env_map: Mapping[str, str] = env if env is not None else os.environ
    raw = env_map.get(POLICY_ENV_VAR, DEFAULT_POLICY_NAME)
    selected = raw.upper()
    if selected not in VALID_POLICY_NAMES:
        raise UnknownPolicyError(
            f"{POLICY_ENV_VAR}={raw!r} not in {sorted(VALID_POLICY_NAMES)}. "
            f"Refusing silent fallback to default per the fail-loud policy."
        )
    return VALID_POLICY_NAMES[selected]


__all__: list[str] = [
    "DEFAULT_POLICY_NAME",
    "POLICY_ENV_VAR",
    "POLICY_PHASE_A1_DEFAULT",
    "POLICY_PHASE_B1_SHADOW",
    "POLICY_PHASE_C_LIVE",
    "VALID_POLICY_NAMES",
    "UnknownPolicyError",
    "resolve_policy",
]
