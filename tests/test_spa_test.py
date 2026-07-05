"""Hansen's SPA test suite (futur3.stats.spa_test).

Load-bearing tests:
- a genuinely superior strategy is detected (rejects); studentization makes the lower-variance of
  two equal-mean strategies the more significant one;
- SPA is at least as powerful as WRC when null strategies are present (the consistent-recentering
  payoff - SPA excludes clearly-inferior strategies that would dilute WRC's max);
- well-calibrated under the global null; deterministic per seed; clearly-inferior strategies do not
  reject.

Matrices are built from ONE shared RNG (`_matrix`) - per-element re-seeding gives constant series.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import pytest

from futur3.stats import (
    SPA_ALPHA,
    MultiStrategyError,
    SPAError,
    SPAResult,
    reality_check,
    spa_test,
)


def _matrix(means: Sequence[float], n_obs: int, sigma: float, seed: int) -> list[list[float]]:
    rng = random.Random(seed)
    return [[rng.gauss(mu, sigma) for _ in range(n_obs)] for mu in means]


# ============================================================================
# TestSPA
# ============================================================================


class TestSPA:
    def test_strong_strategy_rejects(self) -> None:
        data = _matrix([0.5] + [0.0] * 9, n_obs=200, sigma=1.0, seed=1)
        r = spa_test(data, n_bootstrap=1500, seed=7)
        assert r.p_value < SPA_ALPHA
        assert r.significant is True
        assert r.max_strategy_idx == 0

    def test_studentization_prefers_lower_variance(self) -> None:
        # Two strategies, SAME mean (0.15) but different vol. Studentizing -> the low-vol one gets
        # the larger t-stat, so SPA flags it as the best (a raw-mean test could pick either).
        rng = random.Random(2)
        low_vol = [rng.gauss(0.15, 0.5) for _ in range(300)]
        high_vol = [rng.gauss(0.15, 2.0) for _ in range(300)]
        r = spa_test([low_vol, high_vol], n_bootstrap=1500, seed=7)
        assert r.max_strategy_idx == 0  # low-vol strategy

    def test_at_least_as_powerful_as_wrc_with_nulls(self) -> None:
        # 1 modest edge among 40 nulls: WRC's max-statistic null is diluted by the nulls; SPA drops
        # them, so SPA's p-value is no larger than WRC's.
        data = _matrix([0.18] + [0.0] * 40, n_obs=250, sigma=1.0, seed=3)
        spa = spa_test(data, n_bootstrap=1500, seed=7)
        wrc = reality_check(data, n_bootstrap=1500, seed=7)
        assert spa.p_value <= wrc.p_value + 1e-9

    def test_calibration_low_false_positive(self) -> None:
        rejections = 0
        for s in range(6):
            data = _matrix([0.0] * 10, n_obs=120, sigma=1.0, seed=200 + s)
            if spa_test(data, n_bootstrap=1000, seed=s).significant:
                rejections += 1
        assert rejections <= 2

    def test_inferior_strategies_do_not_reject(self) -> None:
        data = _matrix([-0.3, -0.4, -0.2], n_obs=150, sigma=1.0, seed=4)
        r = spa_test(data, n_bootstrap=1000, seed=7)
        assert r.significant is False
        assert r.test_statistic == 0.0  # max(0, negative) -> 0

    def test_determinism(self) -> None:
        data = _matrix([0.2] + [0.0] * 5, n_obs=120, sigma=1.0, seed=5)
        a = spa_test(data, n_bootstrap=1500, seed=7)
        b = spa_test(data, n_bootstrap=1500, seed=7)
        assert a.p_value == b.p_value


# ============================================================================
# TestSPAErrors
# ============================================================================


class TestSPAErrors:
    def test_too_few_bootstraps_raises(self) -> None:
        with pytest.raises(SPAError, match="n_bootstrap must be"):
            spa_test([[0.1, 0.2, 0.3]], n_bootstrap=500)

    def test_bad_block_length_raises(self) -> None:
        with pytest.raises(SPAError, match="block_length must be"):
            spa_test([[0.1, 0.2, 0.3]], n_bootstrap=1000, block_length=0)

    def test_ragged_raises(self) -> None:
        with pytest.raises(MultiStrategyError):
            spa_test([[0.1, 0.2], [0.1, 0.2, 0.3]], n_bootstrap=1000)

    def test_result_frozen(self) -> None:
        r = spa_test(_matrix([0.1, 0.0], 50, 1.0, seed=6), n_bootstrap=1000, seed=7)
        assert isinstance(r, SPAResult)
        with pytest.raises(AttributeError):
            r.p_value = 0.0  # type: ignore[misc]
