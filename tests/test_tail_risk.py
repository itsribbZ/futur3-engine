"""Tail-asymmetry / sizing-risk test suite (futur3.stats.tail_risk).

Hand-computed values lock: population skew (symmetric -> 0, one positive outlier -> known g1), the
outlier-collapse falsification (drop the extreme -> skew vanishes), the type-7 `_quantile` against
`statistics.quantiles(method="inclusive")`, the Groeneveld-Meeden / Bowley quantile-skew exacts, the
empirical-CVaR `tail_means` pair + its ratio (finite / inf / nan branches), and the input guards.
"""

from __future__ import annotations

import math
import statistics

import pytest

from futur3.stats.tail_risk import (
    TailRiskError,
    _quantile,
    quantile_skewness,
    skewness,
    skewness_excluding_extremes,
    tail_means,
)

_OUTLIER = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]  # right-skewed purely by the lone 100


# ============================================================================
# TestSkewness
# ============================================================================


class TestSkewness:
    def test_symmetric_is_zero(self) -> None:
        # mean 0, m3 = (-8-1+0+1+8)/5 = 0 -> skew 0
        assert skewness([-2.0, -1.0, 0.0, 1.0, 2.0]) == pytest.approx(0.0)

    def test_right_skew_hand_value(self) -> None:
        # mean 0, m2 = 20/5 = 4, m3 = 60/5 = 12, m2**1.5 = 8 -> skew 12/8 = 1.5
        assert skewness([-1.0, -1.0, -1.0, -1.0, 4.0]) == pytest.approx(1.5)

    def test_left_skew_is_negative(self) -> None:
        assert skewness([-4.0, 1.0, 1.0, 1.0, 1.0]) == pytest.approx(-1.5)

    def test_zero_variance_is_none(self) -> None:
        assert skewness([3.0, 3.0, 3.0]) is None

    def test_too_few_points_is_none(self) -> None:
        assert skewness([1.0]) is None


# ============================================================================
# TestSkewnessExcludingExtremes
# ============================================================================


class TestSkewnessExcludingExtremes:
    def test_drop_zero_is_plain_skewness(self) -> None:
        assert skewness_excluding_extremes(_OUTLIER, 0) == skewness(_OUTLIER)

    def test_full_skew_is_strongly_positive(self) -> None:
        full = skewness(_OUTLIER)
        assert full is not None and full > 1.0

    def test_dropping_the_outlier_collapses_skew(self) -> None:
        # the largest |x - mean| is 100; dropping it leaves the symmetric [1,2,3,4,5] -> skew 0
        assert skewness_excluding_extremes(_OUTLIER, 1) == pytest.approx(0.0)

    def test_drop_leaving_under_two_points_is_none(self) -> None:
        assert skewness_excluding_extremes(_OUTLIER, 5) is None

    def test_negative_drop_raises(self) -> None:
        with pytest.raises(TailRiskError, match="n_drop"):
            skewness_excluding_extremes(_OUTLIER, -1)


# ============================================================================
# TestQuantileHelper  (type-7, validated against statistics.quantiles inclusive)
# ============================================================================


class TestQuantileHelper:
    def test_matches_statistics_quantiles_inclusive_deciles(self) -> None:
        data = [float(i) for i in range(11)]  # 0..10, deciles land on integers
        expected = statistics.quantiles(data, n=10, method="inclusive")
        for k in range(1, 10):
            assert _quantile(data, k / 10.0) == pytest.approx(expected[k - 1])

    def test_interpolates_between_points(self) -> None:
        # n=4, p=0.5 -> h=1.5 -> 1 + 0.5*(2-1) = 1.5 (== the median)
        assert _quantile([0.0, 1.0, 2.0, 3.0], 0.5) == pytest.approx(1.5)

    def test_single_point(self) -> None:
        assert _quantile([7.0], 0.25) == 7.0


# ============================================================================
# TestQuantileSkewness
# ============================================================================


class TestQuantileSkewness:
    def test_symmetric_is_zero(self) -> None:
        # quartiles -1 / 0 / 1 -> (1 + -1 - 0) / (1 - -1) = 0
        assert quantile_skewness([-2.0, -1.0, 0.0, 1.0, 2.0], tail=0.25) == pytest.approx(0.0)

    def test_bowley_hand_value(self) -> None:
        # sorted [0,1,2,5,6]: Q25=1, med=2, Q75=5 -> (5 + 1 - 4) / (5 - 1) = 0.5
        assert quantile_skewness([0.0, 1.0, 2.0, 5.0, 6.0], tail=0.25) == pytest.approx(0.5)

    def test_left_body_skew_is_negative(self) -> None:
        # sorted [0,1,4,5,6]: Q25=1, med=4, Q75=5 -> (5 + 1 - 8) / (5 - 1) = -0.5
        assert quantile_skewness([0.0, 1.0, 4.0, 5.0, 6.0], tail=0.25) == pytest.approx(-0.5)

    def test_degenerate_spread_is_none(self) -> None:
        assert quantile_skewness([3.0, 3.0, 3.0, 3.0], tail=0.25) is None

    def test_too_few_points_is_none(self) -> None:
        assert quantile_skewness([1.0], tail=0.25) is None

    def test_tail_out_of_range_raises(self) -> None:
        with pytest.raises(TailRiskError, match="tail"):
            quantile_skewness([0.0, 1.0, 2.0], tail=0.0)
        with pytest.raises(TailRiskError, match="tail"):
            quantile_skewness([0.0, 1.0, 2.0], tail=0.5)
        with pytest.raises(TailRiskError, match="tail"):
            quantile_skewness([0.0, 1.0, 2.0], tail=0.6)


# ============================================================================
# TestTailMeans
# ============================================================================


class TestTailMeans:
    def test_single_observation_tails(self) -> None:
        # n=5, tail=0.2 -> k=1: worst [-5], best [3]
        tm = tail_means([-5.0, -4.0, 1.0, 2.0, 3.0], tail=0.2)
        assert tm is not None
        assert tm.n_tail == 1
        assert tm.left_mean == pytest.approx(-5.0)
        assert tm.right_mean == pytest.approx(3.0)
        assert tm.ratio == pytest.approx(0.6)  # 3 / |−5|

    def test_multi_observation_tails(self) -> None:
        # n=5, tail=0.4 -> k=2: worst mean (−5,−4)=−4.5, best mean (2,3)=2.5
        tm = tail_means([-5.0, -4.0, 1.0, 2.0, 3.0], tail=0.4)
        assert tm is not None
        assert tm.n_tail == 2
        assert tm.left_mean == pytest.approx(-4.5)
        assert tm.right_mean == pytest.approx(2.5)

    def test_symmetric_ratio_is_one(self) -> None:
        tm = tail_means([-3.0, -1.0, 1.0, 3.0], tail=0.25)
        assert tm is not None and tm.ratio == pytest.approx(1.0)

    def test_no_loss_tail_ratio_is_inf(self) -> None:
        # worst observation is 0.0 -> |left| == 0, right > 0 -> ratio +inf
        tm = tail_means([0.0, 0.0, 1.0, 2.0, 3.0], tail=0.2)
        assert tm is not None and math.isinf(tm.ratio) and tm.ratio > 0.0

    def test_all_flat_ratio_is_nan(self) -> None:
        tm = tail_means([0.0, 0.0, 0.0], tail=0.25)
        assert tm is not None and math.isnan(tm.ratio)

    def test_integer_boundary_count_is_stable(self) -> None:
        # 0.05 * 20 == 1.0 (modulo float noise) -> k must be exactly 1, not 2
        tm = tail_means([float(i) for i in range(20)], tail=0.05)
        assert tm is not None and tm.n_tail == 1

    def test_too_few_points_is_none(self) -> None:
        assert tail_means([1.0], tail=0.05) is None

    def test_tail_out_of_range_raises(self) -> None:
        with pytest.raises(TailRiskError, match="tail"):
            tail_means([0.0, 1.0, 2.0], tail=0.0)
        with pytest.raises(TailRiskError, match="tail"):
            tail_means([0.0, 1.0, 2.0], tail=0.5)
