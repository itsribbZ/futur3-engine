"""Leverage-survival bootstrap suite (futur3.execution.survival).

Covers the MATH §6 gate: the zero-leverage + all-positive sentinels (P_survive == 1.0 exactly),
monotonic non-increasing survival in leverage (more leverage -> more ruin), the single-step ruin
case real BTC exposed (a return * leverage <= -1 kills the path), seeded determinism, the
Politis-White default block length, and the loud input guards.
"""

from __future__ import annotations

import pytest

from futur3.execution.survival import (
    SURVIVAL_FLOOR,
    SurvivalError,
    survival_probability,
)

# a mixed-sign return series with real downside (worst single bar -6%)
_VOL = [0.05, -0.04, 0.03, -0.05, 0.02, -0.03, 0.04, -0.06, 0.01, -0.02, 0.03, -0.04]
_UP_ONLY = [0.01, 0.02, 0.015, 0.005, 0.03, 0.01, 0.02, 0.012]


def _p(
    returns: list[float],
    leverage: float,
    *,
    horizon: int = 60,
    n_paths: int = 500,
    block_length: int | None = 2,
    kill_switch_dd: float = 0.30,
    seed: int | None = 7,
) -> float:
    # small, fast, deterministic defaults for tests
    return survival_probability(
        returns,
        leverage,
        horizon=horizon,
        n_paths=n_paths,
        block_length=block_length,
        kill_switch_dd=kill_switch_dd,
        seed=seed,
    )


class TestSentinels:
    def test_zero_leverage_always_survives(self) -> None:
        # leverage 0 -> equity is 1*(1+0) forever -> never breaches -> P == 1.0 exactly
        assert _p(_VOL, 0.0) == 1.0

    def test_all_positive_returns_full_survival(self) -> None:
        # no down moves -> equity only grows -> survives at any leverage
        assert _p(_UP_ONLY, 5.0) == 1.0

    def test_floor_constant_matches_spec(self) -> None:
        assert SURVIVAL_FLOOR == 0.995


class TestSurvivalBehavior:
    def test_high_leverage_reduces_survival(self) -> None:
        # the leveraged-ruin failure mode: heavy leverage on a volatile series breaches the switch
        assert _p(_VOL, 5.0) < 1.0

    def test_monotonic_non_increasing_in_leverage(self) -> None:
        # same seed -> identical bootstrap paths; only leverage differs -> survival can't rise
        low, mid, high = _p(_VOL, 0.5), _p(_VOL, 2.0), _p(_VOL, 5.0)
        assert low >= mid >= high

    def test_single_step_ruin(self) -> None:
        # a -50% bar at 3x -> 1 + 3*(-0.5) = -0.5 -> instant ruin on any path drawing it
        ruin_series = [0.01, -0.50, 0.02, 0.01, 0.0, 0.02]
        p = _p(ruin_series, 3.0, block_length=1)  # iid draws -> the -0.5 gets sampled
        assert p < 1.0

    def test_deterministic_same_seed(self) -> None:
        assert _p(_VOL, 3.0) == _p(_VOL, 3.0)

    def test_seed_changes_result_distribution(self) -> None:
        # different seeds generally give (slightly) different MC estimates; both valid probabilities
        a = survival_probability(_VOL, 3.0, horizon=60, n_paths=500, block_length=2, seed=1)
        b = survival_probability(_VOL, 3.0, horizon=60, n_paths=500, block_length=2, seed=2)
        assert 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0

    def test_default_block_length_politis_white(self) -> None:
        # block_length=None -> Politis-White auto-select; must run + return a valid probability
        p = survival_probability(_VOL, 2.0, horizon=60, n_paths=300, seed=5)
        assert 0.0 <= p <= 1.0

    def test_longer_horizon_not_more_survival(self) -> None:
        # more periods = more chances to breach -> survival non-increasing in horizon (same seed)
        short = survival_probability(_VOL, 3.0, horizon=20, n_paths=400, block_length=2, seed=9)
        long = survival_probability(_VOL, 3.0, horizon=200, n_paths=400, block_length=2, seed=9)
        assert long <= short


class TestGuards:
    def test_negative_leverage_raises(self) -> None:
        with pytest.raises(SurvivalError, match="leverage must be >= 0"):
            _p(_VOL, -1.0)

    def test_horizon_below_one_raises(self) -> None:
        with pytest.raises(SurvivalError, match="horizon must be >= 1"):
            _p(_VOL, 2.0, horizon=0)

    def test_n_paths_below_one_raises(self) -> None:
        with pytest.raises(SurvivalError, match="n_paths must be >= 1"):
            _p(_VOL, 2.0, n_paths=0)

    def test_kill_switch_out_of_range_raises(self) -> None:
        with pytest.raises(SurvivalError, match=r"kill_switch_dd must be in \(0, 1\)"):
            _p(_VOL, 2.0, kill_switch_dd=1.0)

    def test_too_few_returns_raises(self) -> None:
        with pytest.raises(SurvivalError, match="need >= 2 returns"):
            _p([0.01], 2.0)
