"""WRC + stationary-bootstrap test suite (futur3.stats.reality_check + _multi_strategy).

Load-bearing tests:
- the stationary bootstrap preserves block structure (mostly +1 continuations at large block length;
  iid at block_length=1) and is deterministic per seed;
- WRC rejects when a genuinely superior strategy hides among nulls, and is well-CALIBRATED under the
  global null (low false-positive rate across many all-null universes - the property that matters);
- a benchmark that matches the edge removes the rejection (the differential is what's tested).

NB: strategy matrices are built from ONE shared RNG (`_matrix`); re-seeding per element would yield
constant series (zero variance) - a fixture trap that masquerades as a code bug.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import pytest

from futur3.stats import (
    WRC_ALPHA,
    MultiStrategyError,
    RealityCheckError,
    RealityCheckResult,
    reality_check,
)
from futur3.stats._multi_strategy import prepare_differentials, stationary_bootstrap_indices


def _matrix(means: Sequence[float], n_obs: int, sigma: float, seed: int) -> list[list[float]]:
    """N strategies (one per mean) x n_obs, all drawn from ONE rng so the series are independent."""
    rng = random.Random(seed)
    return [[rng.gauss(mu, sigma) for _ in range(n_obs)] for mu in means]


# ============================================================================
# TestStationaryBootstrap
# ============================================================================


class TestStationaryBootstrap:
    def test_indices_in_range_and_length(self) -> None:
        idx = stationary_bootstrap_indices(40, 5, random.Random(1))
        assert len(idx) == 40
        assert all(0 <= i < 40 for i in idx)

    def test_large_block_mostly_continuations(self) -> None:
        n = 200
        idx = stationary_bootstrap_indices(n, 50, random.Random(2))  # p=0.02 -> ~98% continue
        cont = sum(1 for t in range(1, n) if idx[t] == (idx[t - 1] + 1) % n)
        assert cont / (n - 1) > 0.85

    def test_block_length_one_is_iid(self) -> None:
        n = 200
        idx = stationary_bootstrap_indices(n, 1, random.Random(3))  # p=1 -> always fresh
        cont = sum(1 for t in range(1, n) if idx[t] == (idx[t - 1] + 1) % n)
        assert cont / (n - 1) < 0.10  # only coincidental continuations

    def test_determinism(self) -> None:
        a = stationary_bootstrap_indices(60, 5, random.Random(7))
        b = stationary_bootstrap_indices(60, 5, random.Random(7))
        assert a == b


# ============================================================================
# TestPrepareDifferentials
# ============================================================================


class TestPrepareDifferentials:
    def test_none_bench_is_zero(self) -> None:
        d, dbar, t, n = prepare_differentials([[1.0, 2.0, 3.0]], None)
        assert d == [[1.0, 2.0, 3.0]] and t == 3 and n == 1
        assert dbar[0] == pytest.approx(2.0)

    def test_bench_subtracted(self) -> None:
        d, _dbar, _t, _n = prepare_differentials([[1.0, 2.0, 3.0]], [0.5, 0.5, 0.5])
        assert d[0] == pytest.approx([0.5, 1.5, 2.5])

    def test_ragged_raises(self) -> None:
        with pytest.raises(MultiStrategyError, match="same number of observations"):
            prepare_differentials([[1.0, 2.0], [1.0, 2.0, 3.0]], None)

    def test_too_few_obs_raises(self) -> None:
        with pytest.raises(MultiStrategyError, match="observations"):
            prepare_differentials([[1.0]], None)

    def test_non_finite_raises(self) -> None:
        with pytest.raises(MultiStrategyError, match="finite"):
            prepare_differentials([[1.0, float("nan"), 3.0]], None)

    def test_bench_length_mismatch_raises(self) -> None:
        with pytest.raises(MultiStrategyError, match="length"):
            prepare_differentials([[1.0, 2.0, 3.0]], [0.0, 0.0])


# ============================================================================
# TestRealityCheck
# ============================================================================


class TestRealityCheck:
    def test_good_strategy_among_nulls_rejects(self) -> None:
        data = _matrix([0.5] + [0.0] * 9, n_obs=150, sigma=1.0, seed=10)  # strong, seed-robust edge
        r = reality_check(data, n_bootstrap=1000, block_length=5, seed=7)
        assert r.p_value < WRC_ALPHA
        assert r.significant is True
        assert r.max_strategy_idx == 0  # the planted edge

    def test_calibration_low_false_positive(self) -> None:
        # Under the GLOBAL null, WRC should reject only ~alpha of the time. Across 5 all-null
        # universes a correct test rejects rarely (a mis-recentered test would over-reject).
        rejections = 0
        for s in range(5):
            data = _matrix([0.0] * 6, n_obs=80, sigma=1.0, seed=100 + s)
            if reality_check(data, n_bootstrap=1000, block_length=5, seed=s).significant:
                rejections += 1
        assert rejections <= 2

    def test_benchmark_removes_edge(self) -> None:
        # A single strategy at mean ~0.25: significant vs the zero benchmark, not vs its own level.
        rng = random.Random(11)
        series = [rng.gauss(0.25, 1.0) for _ in range(200)]
        vs_zero = reality_check([series], n_bootstrap=1000, seed=7)
        vs_self = reality_check([series], [0.25] * 200, n_bootstrap=1000, seed=7)
        assert vs_zero.p_value < vs_self.p_value

    def test_determinism(self) -> None:
        data = _matrix([0.2] + [0.0] * 5, n_obs=120, sigma=1.0, seed=12)
        a = reality_check(data, n_bootstrap=1000, seed=7)
        b = reality_check(data, n_bootstrap=1000, seed=7)
        assert a.p_value == b.p_value

    def test_too_few_bootstraps_raises(self) -> None:
        with pytest.raises(RealityCheckError, match="n_bootstrap must be"):
            reality_check([[0.1, 0.2, 0.3]], n_bootstrap=500)

    def test_bad_block_length_raises(self) -> None:
        with pytest.raises(RealityCheckError, match="block_length must be"):
            reality_check([[0.1, 0.2, 0.3]], n_bootstrap=1000, block_length=0)

    def test_ragged_raises(self) -> None:
        with pytest.raises(MultiStrategyError):
            reality_check([[0.1, 0.2], [0.1, 0.2, 0.3]], n_bootstrap=1000)

    def test_result_frozen(self) -> None:
        r = reality_check(_matrix([0.1, 0.0], 50, 1.0, seed=13), n_bootstrap=1000, seed=7)
        assert isinstance(r, RealityCheckResult)
        with pytest.raises(AttributeError):
            r.p_value = 0.0  # type: ignore[misc]
