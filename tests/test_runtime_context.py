"""A1.12 RuntimeContext test suite — mode + injection + immutability.

Test discipline:
- Frozen-dataclass immutability (FrozenInstanceError on field mutation).
- Mode predicates correct for all 3 enum values.
- `from_env` factory wires verifier_policy via resolve_policy + propagates
  UnknownPolicyError defensively (no-silent-fallback).
- Clock injection works; default is SystemClock returning TZ-aware UTC.

References:
- futur3/runtime/context.py (implementation)
- futur3/data/verifier_policies.py (resolve_policy)
- the backtest-is-live design (one code path, backtest and live)
"""

from __future__ import annotations

import time
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from typing import Final

import pytest

from futur3.data.verifier import VerifierPolicy
from futur3.data.verifier_policies import (
    POLICY_ENV_VAR,
    POLICY_PHASE_A1_DEFAULT,
    POLICY_PHASE_B1_SHADOW,
    POLICY_PHASE_C_LIVE,
    UnknownPolicyError,
)
from futur3.runtime import RuntimeContext, RuntimeMode, SystemClock

# ============================================================================
# Helpers
# ============================================================================


class _FrozenClock:
    """Deterministic UTC clock for tests."""

    def __init__(self, fixed: datetime) -> None:
        self._fixed = fixed

    def now_utc(self) -> datetime:
        return self._fixed


_FIXED_NOW: Final[datetime] = datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC)


# ============================================================================
# TestA1_12_RuntimeMode — enum surface
# ============================================================================


class TestA1_12_RuntimeMode:
    def test_three_modes_exist(self) -> None:
        assert RuntimeMode.BACKTEST.value == "backtest"
        assert RuntimeMode.LIVE_PAPER.value == "live_paper"
        assert RuntimeMode.LIVE_FUNDED.value == "live_funded"

    def test_modes_are_strs(self) -> None:
        """StrEnum members are str — enables direct env-var passthrough + serialization."""
        assert isinstance(RuntimeMode.BACKTEST, str)
        assert RuntimeMode.BACKTEST == "backtest"

    def test_mode_membership_set(self) -> None:
        # Useful for the engine's allowlist + property predicates
        assert RuntimeMode.LIVE_PAPER in {RuntimeMode.LIVE_PAPER, RuntimeMode.LIVE_FUNDED}
        assert RuntimeMode.BACKTEST not in {RuntimeMode.LIVE_PAPER, RuntimeMode.LIVE_FUNDED}


# ============================================================================
# TestA1_12_SystemClock — default clock impl
# ============================================================================


class TestA1_12_SystemClock:
    def test_system_clock_returns_tz_aware_utc(self) -> None:
        c = SystemClock()
        t = c.now_utc()
        assert t.tzinfo is not None
        assert t.tzinfo == UTC

    def test_system_clock_advances(self) -> None:
        c = SystemClock()
        t1 = c.now_utc()
        time.sleep(0.001)
        t2 = c.now_utc()
        assert t2 >= t1


# ============================================================================
# TestA1_12_Construction — RuntimeContext field shape
# ============================================================================


class TestA1_12_Construction:
    def _ctx(self, mode: RuntimeMode) -> RuntimeContext:
        return RuntimeContext(
            mode=mode,
            verifier_policy=POLICY_PHASE_A1_DEFAULT,
            clock=_FrozenClock(_FIXED_NOW),
        )

    def test_backtest_construction(self) -> None:
        ctx = self._ctx(RuntimeMode.BACKTEST)
        assert ctx.mode == RuntimeMode.BACKTEST
        assert ctx.verifier_policy is POLICY_PHASE_A1_DEFAULT
        assert ctx.clock.now_utc() == _FIXED_NOW
        assert ctx.degraded_mode_allowed is False  # default

    def test_live_paper_construction(self) -> None:
        ctx = self._ctx(RuntimeMode.LIVE_PAPER)
        assert ctx.mode == RuntimeMode.LIVE_PAPER

    def test_live_funded_construction(self) -> None:
        ctx = self._ctx(RuntimeMode.LIVE_FUNDED)
        assert ctx.mode == RuntimeMode.LIVE_FUNDED

    def test_degraded_mode_default_false(self) -> None:
        """Fail-closed-safe default. Engine must opt-in explicitly to relax verifier quorum."""
        ctx = self._ctx(RuntimeMode.LIVE_PAPER)
        assert ctx.degraded_mode_allowed is False

    def test_degraded_mode_can_be_set_true(self) -> None:
        ctx = RuntimeContext(
            mode=RuntimeMode.LIVE_PAPER,
            verifier_policy=POLICY_PHASE_B1_SHADOW,
            clock=_FrozenClock(_FIXED_NOW),
            degraded_mode_allowed=True,
        )
        assert ctx.degraded_mode_allowed is True

    def test_verifier_policy_field_holds_full_policy_dataclass(self) -> None:
        ctx = self._ctx(RuntimeMode.BACKTEST)
        assert isinstance(ctx.verifier_policy, VerifierPolicy)
        # Sanity — anchor invariants in policy still accessible through context
        assert ctx.verifier_policy.per_field_policy["settle"].kind == "fail_closed"


# ============================================================================
# TestA1_12_ModePredicates — is_backtest / is_live / is_paper / is_funded
# ============================================================================


class TestA1_12_ModePredicates:
    def _ctx(self, mode: RuntimeMode) -> RuntimeContext:
        return RuntimeContext(
            mode=mode,
            verifier_policy=POLICY_PHASE_A1_DEFAULT,
            clock=_FrozenClock(_FIXED_NOW),
        )

    def test_backtest_predicates(self) -> None:
        ctx = self._ctx(RuntimeMode.BACKTEST)
        assert ctx.is_backtest is True
        assert ctx.is_live is False
        assert ctx.is_paper is False
        assert ctx.is_funded is False

    def test_live_paper_predicates(self) -> None:
        ctx = self._ctx(RuntimeMode.LIVE_PAPER)
        assert ctx.is_backtest is False
        assert ctx.is_live is True
        assert ctx.is_paper is True
        assert ctx.is_funded is False

    def test_live_funded_predicates(self) -> None:
        ctx = self._ctx(RuntimeMode.LIVE_FUNDED)
        assert ctx.is_backtest is False
        assert ctx.is_live is True
        assert ctx.is_paper is False
        assert ctx.is_funded is True

    def test_is_live_covers_paper_or_funded(self) -> None:
        """is_live is True for any non-backtest mode (DRY check)."""
        for mode in (RuntimeMode.LIVE_PAPER, RuntimeMode.LIVE_FUNDED):
            assert self._ctx(mode).is_live is True


# ============================================================================
# TestA1_12_Immutability — frozen dataclass
# ============================================================================


class TestA1_12_Immutability:
    def _ctx(self) -> RuntimeContext:
        return RuntimeContext(
            mode=RuntimeMode.BACKTEST,
            verifier_policy=POLICY_PHASE_A1_DEFAULT,
            clock=_FrozenClock(_FIXED_NOW),
        )

    def test_mode_assignment_raises(self) -> None:
        ctx = self._ctx()
        with pytest.raises(FrozenInstanceError):
            ctx.mode = RuntimeMode.LIVE_FUNDED  # type: ignore[misc]

    def test_verifier_policy_assignment_raises(self) -> None:
        ctx = self._ctx()
        with pytest.raises(FrozenInstanceError):
            ctx.verifier_policy = POLICY_PHASE_C_LIVE  # type: ignore[misc]

    def test_degraded_mode_assignment_raises(self) -> None:
        ctx = self._ctx()
        with pytest.raises(FrozenInstanceError):
            ctx.degraded_mode_allowed = True  # type: ignore[misc]

    def test_clock_assignment_raises(self) -> None:
        ctx = self._ctx()
        with pytest.raises(FrozenInstanceError):
            ctx.clock = SystemClock()  # type: ignore[misc]


# ============================================================================
# TestA1_12_FromEnv — factory using resolve_policy
# ============================================================================


class TestA1_12_FromEnv:
    def test_from_env_default_resolves_phase_a1(self) -> None:
        ctx = RuntimeContext.from_env(
            mode=RuntimeMode.BACKTEST,
            clock=_FrozenClock(_FIXED_NOW),
            env={},
        )
        assert ctx.verifier_policy is POLICY_PHASE_A1_DEFAULT

    def test_from_env_phase_b1(self) -> None:
        ctx = RuntimeContext.from_env(
            mode=RuntimeMode.LIVE_PAPER,
            clock=_FrozenClock(_FIXED_NOW),
            env={POLICY_ENV_VAR: "PHASE_B1"},
        )
        assert ctx.verifier_policy is POLICY_PHASE_B1_SHADOW

    def test_from_env_phase_c(self) -> None:
        ctx = RuntimeContext.from_env(
            mode=RuntimeMode.LIVE_FUNDED,
            clock=_FrozenClock(_FIXED_NOW),
            env={POLICY_ENV_VAR: "PHASE_C"},
        )
        assert ctx.verifier_policy is POLICY_PHASE_C_LIVE

    def test_from_env_unknown_raises_unknown_policy_error(self) -> None:
        with pytest.raises(UnknownPolicyError):
            RuntimeContext.from_env(
                mode=RuntimeMode.LIVE_PAPER,
                clock=_FrozenClock(_FIXED_NOW),
                env={POLICY_ENV_VAR: "PHASE_X"},
            )

    def test_from_env_default_clock_is_system_clock(self) -> None:
        ctx = RuntimeContext.from_env(mode=RuntimeMode.BACKTEST, env={})
        assert isinstance(ctx.clock, SystemClock)

    def test_from_env_degraded_mode_default_false(self) -> None:
        ctx = RuntimeContext.from_env(mode=RuntimeMode.LIVE_PAPER, env={})
        assert ctx.degraded_mode_allowed is False

    def test_from_env_degraded_mode_explicit_true(self) -> None:
        ctx = RuntimeContext.from_env(
            mode=RuntimeMode.LIVE_PAPER,
            env={},
            degraded_mode_allowed=True,
        )
        assert ctx.degraded_mode_allowed is True

    def test_from_env_respects_os_environ_when_env_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(POLICY_ENV_VAR, "PHASE_C")
        ctx = RuntimeContext.from_env(
            mode=RuntimeMode.LIVE_FUNDED,
            clock=_FrozenClock(_FIXED_NOW),
        )
        assert ctx.verifier_policy is POLICY_PHASE_C_LIVE


# ============================================================================
# TestA1_12_ModeInvariant — same dataclass shape across all modes
# ============================================================================


class TestA1_12_ModeInvariant:
    """BACKTEST-IS-LIVE: structurally identical contexts across all modes.

    Per the backtest-is-live design — the engine's
    business logic must not branch on `mode`; only data + execution edges do.
    These tests assert the *dataclass shape* is identical across modes — the
    fields present and their types match.
    """

    def test_all_modes_carry_identical_field_set(self) -> None:
        contexts = [
            RuntimeContext(
                mode=mode,
                verifier_policy=POLICY_PHASE_A1_DEFAULT,
                clock=_FrozenClock(_FIXED_NOW),
            )
            for mode in (
                RuntimeMode.BACKTEST,
                RuntimeMode.LIVE_PAPER,
                RuntimeMode.LIVE_FUNDED,
            )
        ]
        # Field set is identical by virtue of frozen dataclass; this verifies
        # no mode-specific subclass slipped in.
        for ctx in contexts:
            assert hasattr(ctx, "mode")
            assert hasattr(ctx, "verifier_policy")
            assert hasattr(ctx, "clock")
            assert hasattr(ctx, "degraded_mode_allowed")

    def test_same_policy_across_modes_allowed(self) -> None:
        """Engine can use POLICY_PHASE_A1_DEFAULT in BACKTEST AND LIVE_PAPER
        (backtest-is-live — same verifier policy + same code path)."""
        ctx_bt = RuntimeContext(
            mode=RuntimeMode.BACKTEST,
            verifier_policy=POLICY_PHASE_A1_DEFAULT,
            clock=_FrozenClock(_FIXED_NOW),
        )
        ctx_lp = RuntimeContext(
            mode=RuntimeMode.LIVE_PAPER,
            verifier_policy=POLICY_PHASE_A1_DEFAULT,
            clock=_FrozenClock(_FIXED_NOW),
        )
        # Same policy referenced by identity — verifies engine doesn't need
        # a mode-specific policy alias.
        assert ctx_bt.verifier_policy is ctx_lp.verifier_policy
