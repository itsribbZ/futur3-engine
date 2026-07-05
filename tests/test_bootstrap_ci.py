"""BCa bootstrap test suite (futur3.stats.bootstrap_ci).

Load-bearing tests:
- ACCELERATION SIGN: right-skewed data -> accel > 0, left-skewed -> accel < 0 (standard Efron-
  Tibshirani). This is the exact detail an earlier internal note had backwards; a sign-flipped
  acceleration skews the CI the wrong way and could pass a profit-factor lower bound it shouldn't.
- MONTE-CARLO COVERAGE: a 95% BCa CI of the mean contains the true mean ~95% of the time.
- DETERMINISM: same seed -> byte-identical interval.
- G5: a profitable strategy's profit-factor BCa lower bound exceeds 1.0; a break-even one does not.
- Fail-loud: a non-finite point statistic / too many degenerate resamples -> NaN bounds + reason, not 0.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence

import pytest

from futur3.stats import PF_LOWER_THRESHOLD, BCaError, BCaResult, bca_bootstrap, profit_factor
from futur3.stats.bootstrap_ci import _quantile


def _gauss(n: int, mu: float, sigma: float, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(mu, sigma) for _ in range(n)]


def _mean(sample: Sequence[float]) -> float:  # matches Callable[[Sequence[float]], float]
    return statistics.fmean(sample)


# ============================================================================
# TestProfitFactor
# ============================================================================


class TestProfitFactor:
    def test_gains_over_losses(self) -> None:
        # gains 0.3+0.2=0.5; losses |-0.1-0.1|=0.2; PF = 2.5
        assert profit_factor([0.3, -0.1, 0.2, -0.1]) == pytest.approx(2.5)

    def test_no_losses_inf(self) -> None:
        assert profit_factor([0.1, 0.2, 0.3]) == math.inf

    def test_all_losses_zero(self) -> None:
        assert profit_factor([-0.1, -0.2]) == 0.0

    def test_no_data_nan(self) -> None:
        assert math.isnan(profit_factor([]))
        assert math.isnan(profit_factor([0.0, 0.0]))


# ============================================================================
# TestQuantile - type-7 linear interpolation
# ============================================================================


class TestQuantile:
    def test_median_odd(self) -> None:
        assert _quantile([0.0, 1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.0)

    def test_quartile(self) -> None:
        assert _quantile([0.0, 1.0, 2.0, 3.0, 4.0], 0.25) == pytest.approx(1.0)

    def test_interpolated_pair(self) -> None:
        assert _quantile([10.0, 20.0], 0.5) == pytest.approx(15.0)

    def test_endpoints(self) -> None:
        assert _quantile([5.0, 6.0, 7.0], 0.0) == pytest.approx(5.0)
        assert _quantile([5.0, 6.0, 7.0], 1.0) == pytest.approx(7.0)


# ============================================================================
# TestAccelerationSign - the corrected Efron-Tibshirani numerator
# ============================================================================


class TestAccelerationSign:
    def test_right_skew_positive_accel(self) -> None:
        rng = random.Random(40)
        right_skewed = [rng.expovariate(1.0) for _ in range(300)]  # long right tail -> a > 0
        r = bca_bootstrap(right_skewed, _mean, n_resamples=2000, seed=1)
        assert r.accel > 0.0

    def test_left_skew_negative_accel(self) -> None:
        rng = random.Random(40)
        left_skewed = [-rng.expovariate(1.0) for _ in range(300)]  # long left tail -> a < 0
        r = bca_bootstrap(left_skewed, _mean, n_resamples=2000, seed=1)
        assert r.accel < 0.0

    def test_symmetric_small_accel(self) -> None:
        r = bca_bootstrap(_gauss(400, 0.0, 1.0, seed=41), _mean, n_resamples=2000, seed=1)
        assert abs(r.accel) < 0.05  # symmetric -> ~no acceleration


# ============================================================================
# TestDeterminism
# ============================================================================


class TestDeterminism:
    def test_same_seed_identical(self) -> None:
        data = _gauss(150, 0.001, 0.01, seed=42)
        a = bca_bootstrap(data, _mean, n_resamples=2000, seed=7)
        b = bca_bootstrap(data, _mean, n_resamples=2000, seed=7)
        assert (a.lower, a.upper, a.z0, a.accel) == (b.lower, b.upper, b.z0, b.accel)

    def test_different_seed_differs(self) -> None:
        data = _gauss(150, 0.001, 0.01, seed=42)
        a = bca_bootstrap(data, _mean, n_resamples=2000, seed=7)
        b = bca_bootstrap(data, _mean, n_resamples=2000, seed=8)
        assert a.lower != b.lower or a.upper != b.upper


# ============================================================================
# TestCIProperties
# ============================================================================


class TestCIProperties:
    def test_ci_brackets_point(self) -> None:
        r = bca_bootstrap(_gauss(300, 0.002, 0.01, seed=50), _mean, n_resamples=2000, seed=3)
        assert r.lower < r.point < r.upper

    def test_higher_confidence_wider(self) -> None:
        data = _gauss(300, 0.002, 0.01, seed=51)
        narrow = bca_bootstrap(data, _mean, n_resamples=2000, confidence=0.80, seed=3)
        wide = bca_bootstrap(data, _mean, n_resamples=2000, confidence=0.99, seed=3)
        assert (wide.upper - wide.lower) > (narrow.upper - narrow.lower)


# ============================================================================
# TestMonteCarloCoverage - end-to-end calibration of the interval
# ============================================================================


class TestMonteCarloCoverage:
    def test_mean_ci_covers_true_mean_near_95pct(self) -> None:
        rng = random.Random(2025)
        mu, sigma, n, k = 0.5, 1.0, 50, 200
        covered = 0
        for i in range(k):
            sample = [rng.gauss(mu, sigma) for _ in range(n)]
            r = bca_bootstrap(sample, _mean, n_resamples=1000, seed=i)
            if r.lower <= mu <= r.upper:
                covered += 1
        # nominal 0.95; allow Monte-Carlo slack for k=200, n=50.
        assert 0.88 <= covered / k <= 1.0


# ============================================================================
# TestG5ProfitFactor
# ============================================================================


class TestG5ProfitFactor:
    def test_profitable_passes_profit_factor(self) -> None:
        profitable = _gauss(400, 0.5, 1.0, seed=60)  # strong positive drift
        r = bca_bootstrap(profitable, profit_factor, n_resamples=2000, seed=5)
        assert r.lower > PF_LOWER_THRESHOLD
        assert r.lower_exceeds(PF_LOWER_THRESHOLD) is True

    def test_breakeven_fails_g5(self) -> None:
        breakeven = _gauss(400, 0.0, 1.0, seed=61)  # zero-mean -> PF ~ 1
        r = bca_bootstrap(breakeven, profit_factor, n_resamples=2000, seed=5)
        assert r.lower_exceeds(PF_LOWER_THRESHOLD) is False


# ============================================================================
# TestDegenerate
# ============================================================================


class TestDegenerate:
    def test_constant_data_collapses_to_point(self) -> None:
        r = bca_bootstrap([5.0] * 50, _mean, n_resamples=1000, seed=1)
        assert r.accel == 0.0  # zero jackknife variance -> no acceleration
        assert r.lower == pytest.approx(5.0) and r.upper == pytest.approx(5.0)

    def test_point_not_finite_returns_nan(self) -> None:
        # profit factor with no losing trades -> inf point statistic -> undefined CI (fail-loud).
        r = bca_bootstrap([0.1, 0.2, 0.3, 0.4], profit_factor, n_resamples=1000, seed=1)
        assert math.isnan(r.lower) and math.isnan(r.upper)
        assert r.reason is not None and "not finite" in r.reason


# ============================================================================
# TestErrors
# ============================================================================


class TestErrors:
    def test_bad_confidence_raises(self) -> None:
        with pytest.raises(BCaError, match="confidence must be in"):
            bca_bootstrap([0.1, 0.2, 0.3], _mean, confidence=1.0)

    def test_too_few_resamples_raises(self) -> None:
        with pytest.raises(BCaError, match="n_resamples must be"):
            bca_bootstrap([0.1, 0.2, 0.3], _mean, n_resamples=500)

    def test_too_few_data_raises(self) -> None:
        with pytest.raises(BCaError, match="need >= 2 data points"):
            bca_bootstrap([0.1], _mean, n_resamples=1000)

    def test_non_finite_data_raises(self) -> None:
        with pytest.raises(BCaError, match="finite"):
            bca_bootstrap([0.1, float("nan"), 0.3], _mean, n_resamples=1000)

    def test_result_frozen(self) -> None:
        r = bca_bootstrap(_gauss(50, 0.01, 0.1, seed=70), _mean, n_resamples=1000, seed=1)
        assert isinstance(r, BCaResult)
        with pytest.raises(AttributeError):
            r.lower = 0.0  # type: ignore[misc]
