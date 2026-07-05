"""Permutation-test suite (futur3.stats.permutation).

Load-bearing tests:
- sign_flip detects a directional edge (strong -> p tiny PASS G7; none -> p large FAIL).
- p-value uses the (#>=obs + 1)/(n_valid + 1) form -> minimum p = 1/(n_valid+1), never 0.
- mode/statistic compatibility: shuffle + mean is invariant (p ~ 1, a documented gotcha); shuffle
  + an order-dependent statistic detects autocorrelation.
- autocorrelation diagnostic: AR(1) flags `autocorr_warning` under sign_flip, not block_shuffle.
- determinism (same seed -> identical null distribution); fail-loud degenerate/error paths.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence

import pytest

from futur3.stats import (
    PERMUTATION_P_THRESHOLD,
    PermutationError,
    PermutationResult,
    mean_return,
    permutation_test,
)
from futur3.stats.bootstrap_ci import profit_factor


def _gauss(n: int, mu: float, sigma: float, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(mu, sigma) for _ in range(n)]


def _ar1(n: int, phi: float, seed: int) -> list[float]:
    rng = random.Random(seed)
    out = [0.0] * n
    for i in range(1, n):
        out[i] = phi * out[i - 1] + rng.gauss(0.0, 1.0)
    return out


def _lag1(s: Sequence[float]) -> float:
    m = statistics.fmean(s)
    var = math.fsum((x - m) ** 2 for x in s)
    if var <= 0.0:
        return 0.0
    return math.fsum((s[i] - m) * (s[i - 1] - m) for i in range(1, len(s))) / var


def _nan_if_neg_mean(s: Sequence[float]) -> float:
    m = statistics.fmean(s)
    return m if m > 0.0 else float("nan")


# ============================================================================
# TestMeanReturn
# ============================================================================


class TestMeanReturn:
    def test_mean_return(self) -> None:
        assert mean_return([1.0, 2.0, 3.0]) == pytest.approx(2.0)


# ============================================================================
# TestSignFlip - directional-edge detection
# ============================================================================


class TestSignFlip:
    def test_strong_edge_low_p(self) -> None:
        r = permutation_test(_gauss(300, 0.5, 1.0, seed=1), n_permutations=1000, seed=7)
        assert r.p_value is not None and r.p_value < PERMUTATION_P_THRESHOLD
        assert r.passes_permutation is True

    def test_no_edge_high_p(self) -> None:
        r = permutation_test(_gauss(300, 0.0, 1.0, seed=2), n_permutations=1000, seed=7)
        assert r.p_value is not None and r.p_value > PERMUTATION_P_THRESHOLD
        assert r.passes_permutation is False

    def test_negative_edge_one_sided_not_significant(self) -> None:
        # One-sided tests the UPPER tail; a negative mean sits in the lower tail -> large p.
        r = permutation_test(_gauss(300, -0.5, 1.0, seed=3), n_permutations=1000, seed=7)
        assert r.p_value is not None and r.p_value > PERMUTATION_P_THRESHOLD

    def test_two_sided_detects_negative_edge(self) -> None:
        r = permutation_test(
            _gauss(300, -0.5, 1.0, seed=3), n_permutations=1000, one_sided=False, seed=7
        )
        assert r.p_value is not None and r.p_value < PERMUTATION_P_THRESHOLD


# ============================================================================
# TestPValueForm
# ============================================================================


class TestPValueForm:
    def test_min_p_value_is_one_over_n_plus_one(self) -> None:
        # A clear edge that no permutation beats -> p hits the (0+1)/(n_valid+1) floor, never 0.
        r = permutation_test(_gauss(300, 0.6, 1.0, seed=4), n_permutations=1000, seed=7)
        assert r.n_valid == 1000
        assert r.p_value == pytest.approx(1.0 / (r.n_valid + 1))

    def test_p_value_bounded(self) -> None:
        r = permutation_test(_gauss(200, 0.0, 1.0, seed=5), n_permutations=1000, seed=7)
        assert r.p_value is not None and 0.0 < r.p_value <= 1.0


# ============================================================================
# TestModes - transform/statistic compatibility
# ============================================================================


class TestModes:
    def test_shuffle_mean_is_invariant(self) -> None:
        # The mean is permutation-invariant -> shuffling never changes it -> p ~ 1 (the gotcha).
        r = permutation_test(
            _gauss(200, 0.5, 1.0, seed=6), mean_return, n_permutations=1000, mode="shuffle", seed=7
        )
        assert r.p_value == pytest.approx(1.0)

    def test_shuffle_detects_order_dependence(self) -> None:
        # An order-dependent statistic (lag-1 ACF) on AR(1) data is significant under shuffle.
        r = permutation_test(
            _ar1(200, 0.6, seed=8), _lag1, n_permutations=1000, mode="shuffle", seed=7
        )
        assert r.p_value is not None and r.p_value < PERMUTATION_P_THRESHOLD

    def test_block_shuffle_runs(self) -> None:
        r = permutation_test(
            _gauss(200, 0.4, 1.0, seed=9),
            n_permutations=1000,
            mode="block_shuffle",
            block_length=5,
            seed=7,
        )
        assert r.p_value is not None and r.mode == "block_shuffle" and r.block_length == 5


# ============================================================================
# TestAutocorrDiagnostic
# ============================================================================


class TestAutocorrDiagnostic:
    def test_warning_on_ar1_sign_flip(self) -> None:
        r = permutation_test(_ar1(250, 0.6, seed=10), n_permutations=1000, mode="sign_flip", seed=7)
        assert r.autocorr_warning is True
        assert r.lag1_autocorr is not None and r.lag1_autocorr > 0.0

    def test_no_warning_on_iid(self) -> None:
        r = permutation_test(_gauss(250, 0.1, 1.0, seed=11), n_permutations=1000, seed=7)
        assert r.autocorr_warning is False

    def test_no_warning_under_block_shuffle(self) -> None:
        # block_shuffle is the REMEDY for autocorrelation, so it does not raise the warning.
        r = permutation_test(
            _ar1(250, 0.6, seed=10), n_permutations=1000, mode="block_shuffle", seed=7
        )
        assert r.autocorr_warning is False


# ============================================================================
# TestDeterminism
# ============================================================================


class TestDeterminism:
    def test_same_seed_identical_null(self) -> None:
        data = _gauss(150, 0.2, 1.0, seed=12)
        a = permutation_test(data, n_permutations=1000, seed=7)
        b = permutation_test(data, n_permutations=1000, seed=7)
        assert a.null_distribution == b.null_distribution and a.p_value == b.p_value

    def test_different_seed_differs(self) -> None:
        data = _gauss(150, 0.2, 1.0, seed=12)
        a = permutation_test(data, n_permutations=1000, seed=7)
        b = permutation_test(data, n_permutations=1000, seed=8)
        assert a.null_distribution != b.null_distribution


# ============================================================================
# TestResultFields
# ============================================================================


class TestResultFields:
    def test_null_distribution_length(self) -> None:
        r = permutation_test(_gauss(100, 0.2, 1.0, seed=13), n_permutations=1000, seed=7)
        assert len(r.null_distribution) == r.n_valid == 1000

    def test_result_frozen(self) -> None:
        r = permutation_test(_gauss(50, 0.2, 1.0, seed=14), n_permutations=1000, seed=7)
        assert isinstance(r, PermutationResult)
        with pytest.raises(AttributeError):
            r.p_value = 0.0  # type: ignore[misc]


# ============================================================================
# TestDegenerate + errors
# ============================================================================


class TestDegenerate:
    def test_observed_not_finite_returns_none(self) -> None:
        # profit factor with no losing trades -> inf observed -> undefined (fail-loud), no raise.
        r = permutation_test([0.1, 0.2, 0.3, 0.4], profit_factor, n_permutations=1000, seed=7)
        assert r.p_value is None
        assert r.reason is not None and "not finite" in r.reason

    def test_too_many_nan_raises(self) -> None:
        # ~half the sign-flips of a +mean series have negative mean -> NaN > 10% -> raise.
        with pytest.raises(PermutationError, match="non-finite statistic"):
            permutation_test(
                _gauss(200, 0.5, 1.0, seed=15), _nan_if_neg_mean, n_permutations=1000, seed=7
            )


class TestErrors:
    def test_bad_mode_raises(self) -> None:
        with pytest.raises(PermutationError, match="mode must be one of"):
            permutation_test([0.1, 0.2], mode="bogus", n_permutations=1000)  # type: ignore[arg-type]

    def test_too_few_permutations_raises(self) -> None:
        with pytest.raises(PermutationError, match="n_permutations must be"):
            permutation_test([0.1, 0.2, 0.3], n_permutations=500)

    def test_bad_block_length_raises(self) -> None:
        with pytest.raises(PermutationError, match="block_length must be"):
            permutation_test(
                [0.1, 0.2, 0.3], n_permutations=1000, mode="block_shuffle", block_length=0
            )

    def test_too_few_returns_raises(self) -> None:
        with pytest.raises(PermutationError, match="need >= 2 returns"):
            permutation_test([0.1], n_permutations=1000)

    def test_non_finite_returns_raises(self) -> None:
        with pytest.raises(PermutationError, match="finite"):
            permutation_test([0.1, float("inf"), 0.3], n_permutations=1000)
