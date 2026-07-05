"""A1.9 verifier_policies test suite — 3 presets + env-var loader.

Per the verifier spec-§1.5`:
- All 3 presets share invariants: settle.kind==fail_closed (fail-loud),
  tick.kind==fail_closed, oi.kind==highest_tier.
- B1 per-field block is identical to A1 (engine-side degraded-mode opt-in
  is the only difference, not verifier-side).
- C tightens ohlc + volume to *_c_v1 ids; settles + ticks inherit A1.
- resolve_policy: PHASE_A1 default; case-insensitive env-var; unknown raises
  UnknownPolicyError (no silent fallback per the fail-loud policy).

References:
- futur3/data/verifier_policies.py (implementation)
- futur3/data/verifier.py (VerifierPolicy + DataSourceError ancestry)
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Final

import pytest

from futur3.data.source import DataSourceError
from futur3.data.verifier import VerifierPolicy
from futur3.data.verifier_policies import (
    DEFAULT_POLICY_NAME,
    POLICY_ENV_VAR,
    POLICY_PHASE_A1_DEFAULT,
    POLICY_PHASE_B1_SHADOW,
    POLICY_PHASE_C_LIVE,
    VALID_POLICY_NAMES,
    UnknownPolicyError,
    resolve_policy,
)

# All 3 presets surfaced for parametrized tests
_ALL_PRESETS: Final[list[tuple[str, VerifierPolicy]]] = [
    ("PHASE_A1", POLICY_PHASE_A1_DEFAULT),
    ("PHASE_B1", POLICY_PHASE_B1_SHADOW),
    ("PHASE_C", POLICY_PHASE_C_LIVE),
]


# ============================================================================
# TestA1_9_PresetShape — structural sanity per preset
# ============================================================================


class TestA1_9_PresetShape:
    def test_a1_umbrella_shape(self) -> None:
        p = POLICY_PHASE_A1_DEFAULT
        assert p.policy_id == "phase_a1_default_v1"
        assert p.kind == "fail_closed"
        assert p.min_quorum == 3
        assert p.tolerance_band == "NORMAL"
        assert p.require_tier_t2_or_better is True

    def test_b1_umbrella_shape(self) -> None:
        p = POLICY_PHASE_B1_SHADOW
        assert p.policy_id == "phase_b1_shadow_v1"
        assert p.kind == "fail_closed"
        assert p.min_quorum == 3
        assert p.tolerance_band == "NORMAL"
        assert p.require_tier_t2_or_better is True

    def test_c_umbrella_shape(self) -> None:
        p = POLICY_PHASE_C_LIVE
        assert p.policy_id == "phase_c_live_v1"
        assert p.kind == "majority_vote"
        assert p.min_quorum == 3
        assert p.tolerance_band == "NORMAL"
        assert p.require_tier_t2_or_better is True

    @pytest.mark.parametrize("name,preset", _ALL_PRESETS)
    def test_all_presets_carry_5_per_field_overrides(
        self, name: str, preset: VerifierPolicy
    ) -> None:
        # All 3 presets carry the same 5 keys: settle, tick, ohlc, volume, oi
        assert set(preset.per_field_policy) == {"settle", "tick", "ohlc", "volume", "oi"}, (
            f"{name} per_field keys mismatched"
        )


# ============================================================================
# TestA1_9_PhaseInvariants — anchor invariants held across all 3 presets
# ============================================================================


class TestA1_9_PhaseInvariants:
    """settle / tick / oi must NOT differ across A1, B1, C — anchor invariants."""

    @pytest.mark.parametrize("name,preset", _ALL_PRESETS)
    def test_settle_kind_is_fail_closed(self, name: str, preset: VerifierPolicy) -> None:
        assert preset.per_field_policy["settle"].kind == "fail_closed", (
            f"{name} settle.kind should be fail_closed (anchor invariant)"
        )

    @pytest.mark.parametrize("name,preset", _ALL_PRESETS)
    def test_tick_kind_is_fail_closed(self, name: str, preset: VerifierPolicy) -> None:
        assert preset.per_field_policy["tick"].kind == "fail_closed", (
            f"{name} tick.kind should be fail_closed (anchor invariant)"
        )

    @pytest.mark.parametrize("name,preset", _ALL_PRESETS)
    def test_oi_kind_is_highest_tier(self, name: str, preset: VerifierPolicy) -> None:
        assert preset.per_field_policy["oi"].kind == "highest_tier", (
            f"{name} oi.kind should be highest_tier (anchor invariant)"
        )

    @pytest.mark.parametrize("name,preset", _ALL_PRESETS)
    def test_settle_policy_id_is_settle_exact_v1(self, name: str, preset: VerifierPolicy) -> None:
        # Same policy_id across all 3 presets = same exact-settle semantics
        assert preset.per_field_policy["settle"].policy_id == "settle_exact_v1"

    @pytest.mark.parametrize("name,preset", _ALL_PRESETS)
    def test_tick_policy_id_is_tick_exact_v1(self, name: str, preset: VerifierPolicy) -> None:
        assert preset.per_field_policy["tick"].policy_id == "tick_exact_v1"

    @pytest.mark.parametrize("name,preset", _ALL_PRESETS)
    def test_settle_tolerance_band_is_tight(self, name: str, preset: VerifierPolicy) -> None:
        assert preset.per_field_policy["settle"].tolerance_band == "TIGHT"

    @pytest.mark.parametrize("name,preset", _ALL_PRESETS)
    def test_tick_tolerance_band_is_tight(self, name: str, preset: VerifierPolicy) -> None:
        assert preset.per_field_policy["tick"].tolerance_band == "TIGHT"

    @pytest.mark.parametrize("name,preset", _ALL_PRESETS)
    def test_tier_t2_or_better_required(self, name: str, preset: VerifierPolicy) -> None:
        assert preset.require_tier_t2_or_better is True
        # AND every per-field override also requires T2+
        for field_name, p in preset.per_field_policy.items():
            assert p.require_tier_t2_or_better is True, (
                f"{name}.per_field[{field_name}] should require T2+"
            )


# ============================================================================
# TestA1_9_B1MirrorsA1 — B1 per-field block must equal A1's byte-for-byte
# ============================================================================


class TestA1_9_B1MirrorsA1:
    """B1 differs from A1 only in policy_id + engine-side degraded-mode flag,
    NOT in any verifier-side per-field policy. This decouples engine + policy.
    """

    def test_b1_per_field_keys_identical_to_a1(self) -> None:
        assert set(POLICY_PHASE_B1_SHADOW.per_field_policy) == set(
            POLICY_PHASE_A1_DEFAULT.per_field_policy
        )

    @pytest.mark.parametrize("field_name", ["settle", "tick", "ohlc", "volume", "oi"])
    def test_b1_per_field_value_equal_to_a1(self, field_name: str) -> None:
        b1 = POLICY_PHASE_B1_SHADOW.per_field_policy[field_name]
        a1 = POLICY_PHASE_A1_DEFAULT.per_field_policy[field_name]
        assert b1 == a1, f"B1.{field_name} should equal A1.{field_name}"


# ============================================================================
# TestA1_9_CPolicyOverrides — C bumps ohlc + volume to *_c_v1; rest inherit A1
# ============================================================================


class TestA1_9_CPolicyOverrides:
    def test_c_ohlc_has_distinct_policy_id(self) -> None:
        c_ohlc = POLICY_PHASE_C_LIVE.per_field_policy["ohlc"]
        a1_ohlc = POLICY_PHASE_A1_DEFAULT.per_field_policy["ohlc"]
        assert c_ohlc.policy_id == "ohlc_majority_c_v1"
        assert a1_ohlc.policy_id == "ohlc_majority_v1"

    def test_c_volume_has_distinct_policy_id(self) -> None:
        c_vol = POLICY_PHASE_C_LIVE.per_field_policy["volume"]
        a1_vol = POLICY_PHASE_A1_DEFAULT.per_field_policy["volume"]
        assert c_vol.policy_id == "volume_majority_c_v1"
        assert a1_vol.policy_id == "volume_majority_v1"

    @pytest.mark.parametrize("field_name", ["settle", "tick", "oi"])
    def test_c_invariant_fields_inherit_a1(self, field_name: str) -> None:
        """settle, tick, oi: same dataclass instance as A1's (verifier-side identity)."""
        assert (
            POLICY_PHASE_C_LIVE.per_field_policy[field_name]
            == POLICY_PHASE_A1_DEFAULT.per_field_policy[field_name]
        )

    def test_c_ohlc_remains_majority_vote_quorum_3(self) -> None:
        c_ohlc = POLICY_PHASE_C_LIVE.per_field_policy["ohlc"]
        assert c_ohlc.kind == "majority_vote"
        assert c_ohlc.min_quorum == 3
        assert c_ohlc.tolerance_band == "NORMAL"

    def test_c_volume_remains_wide_band(self) -> None:
        c_vol = POLICY_PHASE_C_LIVE.per_field_policy["volume"]
        assert c_vol.tolerance_band == "WIDE"
        assert c_vol.min_quorum == 3


# ============================================================================
# TestA1_9_ResolvePolicy — env-var loader
# ============================================================================


class TestA1_9_ResolvePolicy:
    def test_default_phase_a1_when_env_unset(self) -> None:
        result = resolve_policy(env={})
        assert result is POLICY_PHASE_A1_DEFAULT
        assert DEFAULT_POLICY_NAME == "PHASE_A1"

    def test_explicit_phase_a1(self) -> None:
        result = resolve_policy(env={POLICY_ENV_VAR: "PHASE_A1"})
        assert result is POLICY_PHASE_A1_DEFAULT

    def test_explicit_phase_b1(self) -> None:
        result = resolve_policy(env={POLICY_ENV_VAR: "PHASE_B1"})
        assert result is POLICY_PHASE_B1_SHADOW

    def test_explicit_phase_c(self) -> None:
        result = resolve_policy(env={POLICY_ENV_VAR: "PHASE_C"})
        assert result is POLICY_PHASE_C_LIVE

    def test_case_insensitive_lowercase(self) -> None:
        result = resolve_policy(env={POLICY_ENV_VAR: "phase_b1"})
        assert result is POLICY_PHASE_B1_SHADOW

    def test_case_insensitive_mixed(self) -> None:
        result = resolve_policy(env={POLICY_ENV_VAR: "Phase_C"})
        assert result is POLICY_PHASE_C_LIVE

    def test_unknown_raises_unknown_policy_error(self) -> None:
        with pytest.raises(UnknownPolicyError, match="PHASE_X"):
            resolve_policy(env={POLICY_ENV_VAR: "PHASE_X"})

    def test_empty_string_raises(self) -> None:
        with pytest.raises(UnknownPolicyError):
            resolve_policy(env={POLICY_ENV_VAR: ""})

    def test_unknown_policy_error_subclasses_data_source_error(self) -> None:
        """UnknownPolicyError should be catchable as DataSourceError."""
        assert issubclass(UnknownPolicyError, DataSourceError)

    def test_uses_os_environ_when_env_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default env source is os.environ — verify via monkeypatch."""
        monkeypatch.setenv(POLICY_ENV_VAR, "PHASE_C")
        result = resolve_policy()
        assert result is POLICY_PHASE_C_LIVE

    def test_uses_default_when_env_var_missing_from_environ(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No env-var set → PHASE_A1 default."""
        monkeypatch.delenv(POLICY_ENV_VAR, raising=False)
        result = resolve_policy()
        assert result is POLICY_PHASE_A1_DEFAULT

    def test_resolve_is_deterministic(self) -> None:
        """Same env → same identity (returns the singleton constant)."""
        env = {POLICY_ENV_VAR: "PHASE_B1"}
        r1 = resolve_policy(env=env)
        r2 = resolve_policy(env=env)
        assert r1 is r2


# ============================================================================
# TestA1_9_Constants — VALID_POLICY_NAMES + DEFAULT + ENV_VAR
# ============================================================================


class TestA1_9_Constants:
    def test_valid_policy_names_keys(self) -> None:
        assert set(VALID_POLICY_NAMES) == {"PHASE_A1", "PHASE_B1", "PHASE_C"}

    def test_valid_policy_names_values_are_presets(self) -> None:
        assert VALID_POLICY_NAMES["PHASE_A1"] is POLICY_PHASE_A1_DEFAULT
        assert VALID_POLICY_NAMES["PHASE_B1"] is POLICY_PHASE_B1_SHADOW
        assert VALID_POLICY_NAMES["PHASE_C"] is POLICY_PHASE_C_LIVE

    def test_default_policy_name_constant(self) -> None:
        assert DEFAULT_POLICY_NAME == "PHASE_A1"

    def test_policy_env_var_constant(self) -> None:
        assert POLICY_ENV_VAR == "FUTUR3_VERIFIER_POLICY"


# ============================================================================
# TestA1_9_Immutability — VerifierPolicy is frozen
# ============================================================================


class TestA1_9_Immutability:
    def test_verifier_policy_frozen_dataclass(self) -> None:
        """Re-assigning fields on a frozen VerifierPolicy raises FrozenInstanceError."""
        with pytest.raises(FrozenInstanceError):
            POLICY_PHASE_A1_DEFAULT.policy_id = "tampered"  # type: ignore[misc]

    def test_resolve_returns_singleton_identity(self) -> None:
        """resolve_policy returns the module-level constant by identity, not a copy."""
        assert resolve_policy(env={POLICY_ENV_VAR: "PHASE_A1"}) is POLICY_PHASE_A1_DEFAULT
