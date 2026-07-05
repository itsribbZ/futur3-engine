"""Block-bootstrap test suite (futur3.cv.block_bootstrap).

Load-bearing tests:
- AUTOCORRELATION WIDENING: on AR(1) data the block-bootstrap CI of the mean is WIDER than the iid
  BCa CI. This is the entire reason §4.4 exists - the iid bootstrap underestimates the variance of a
  statistic computed on serially dependent data, so its CI is too narrow (a fake-confidence bug).
- POLITIS-WHITE: white noise -> estimated block length ~1; AR(1) rho=0.5 -> notably larger.
- NO DIVERGENCE: the "stationary" path uses the SAME `stationary_bootstrap_indices` object that
  WRC/SPA use - an identity assertion locks the two against a future copy-paste fork.
- DETERMINISM: same seed -> byte-identical result incl. the full bootstrap distribution.
- BOUNDARIES: block_len == 1 reduces to iid (length-n, full-range draws); block_len == n (fixed) is
  the trivial single-block resample (CI collapses to the point).
- Fail-loud: a non-finite point statistic -> NaN bounds + a reason, not 0.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence

import pytest

from futur3.cv import (
    BlockBootstrapError,
    BlockBootstrapResult,
    block_bootstrap,
    optimal_block_length,
)
from futur3.cv.block_bootstrap import (
    _fixed_block_indices,
    _moving_block_indices,
    stationary_bootstrap_indices,
)
from futur3.stats import bca_bootstrap, profit_factor
from futur3.stats._multi_strategy import stationary_bootstrap_indices as _stats_stationary


def _gauss(n: int, mu: float, sigma: float, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(mu, sigma) for _ in range(n)]


def _ar1(n: int, rho: float, sigma: float, seed: int) -> list[float]:
    """Zero-mean AR(1): x_t = rho*x_{t-1} + eps_t. Positive rho -> positive serial dependence."""
    rng = random.Random(seed)
    out: list[float] = []
    prev = 0.0
    for _ in range(n):
        prev = rho * prev + rng.gauss(0.0, sigma)
        out.append(prev)
    return out


def _mean(sample: Sequence[float]) -> float:  # matches Callable[[Sequence[float]], float]
    return statistics.fmean(sample)


def _width(r: BlockBootstrapResult) -> float:
    return r.upper - r.lower


# ============================================================================
# TestOptimalBlockLength - Politis-White (2004)
# ============================================================================


class TestOptimalBlockLength:
    def test_white_noise_short_block(self) -> None:
        # iid -> no serial dependence -> block length collapses toward 1.
        white = _gauss(600, 0.0, 1.0, seed=10)
        assert optimal_block_length(white) <= 4

    def test_ar1_longer_block(self) -> None:
        # rho=0.5 -> meaningful dependence -> longer block than white noise.
        ar = _ar1(600, 0.5, 1.0, seed=11)
        white = _gauss(600, 0.0, 1.0, seed=10)
        assert optimal_block_length(ar) > optimal_block_length(white)
        assert 2 <= optimal_block_length(ar) <= 40  # plausible band for n=600, rho=0.5

    def test_deterministic_pure_function(self) -> None:
        ar = _ar1(300, 0.6, 1.0, seed=12)
        assert optimal_block_length(ar) == optimal_block_length(list(ar))

    def test_constant_series_block_one(self) -> None:
        assert optimal_block_length([3.0] * 50) == 1

    def test_tiny_series_block_one(self) -> None:
        assert optimal_block_length([0.1, 0.2, 0.3]) == 1  # below the PW data floor


# ============================================================================
# TestBlockIndexGenerators - boundary structure
# ============================================================================


class TestBlockIndexGenerators:
    def test_moving_length_and_range(self) -> None:
        rng = random.Random(1)
        idx = _moving_block_indices(20, 5, rng)
        assert len(idx) == 20
        assert all(0 <= i < 20 for i in idx)

    def test_moving_block_one_is_iid_shaped(self) -> None:
        rng = random.Random(2)
        idx = _moving_block_indices(15, 1, rng)
        assert len(idx) == 15 and all(0 <= i < 15 for i in idx)

    def test_moving_full_length_is_rotation(self) -> None:
        rng = random.Random(3)
        idx = _moving_block_indices(10, 10, rng)
        assert sorted(idx) == list(range(10))  # a circular rotation visits every index once
        # consecutive mod n (contiguous wrap)
        assert all((idx[k + 1] - idx[k]) % 10 == 1 for k in range(9))

    def test_fixed_full_length_is_identity(self) -> None:
        rng = random.Random(4)
        # block_len == n: only one disjoint block (start 0) -> the original ordering.
        assert _fixed_block_indices(12, 12, rng) == list(range(12))

    def test_fixed_block_one_is_iid_shaped(self) -> None:
        rng = random.Random(5)
        idx = _fixed_block_indices(15, 1, rng)
        assert len(idx) == 15 and all(0 <= i < 15 for i in idx)


# ============================================================================
# TestModes - all three schemes run, bracket the point, report block length
# ============================================================================


class TestModes:
    @pytest.mark.parametrize("mode", ["stationary", "moving", "fixed"])
    def test_brackets_point(self, mode: str) -> None:
        data = _ar1(300, 0.4, 1.0, seed=20)
        r = block_bootstrap(data, _mean, n_resamples=2000, block_mean=8, mode=mode, seed=1)  # type: ignore[arg-type]
        assert r.lower < r.point < r.upper
        assert r.mode == mode
        assert r.block_mean == 8 and r.block_mean_estimated is False

    def test_fixed_full_block_collapses_to_point(self) -> None:
        # block_len == T: every resample is the whole series -> CI collapses to the point.
        data = _gauss(40, 0.5, 1.0, seed=21)
        r = block_bootstrap(data, _mean, n_resamples=1000, block_mean=40, mode="fixed", seed=1)
        assert r.lower == pytest.approx(r.point) and r.upper == pytest.approx(r.point)
        assert all(v == pytest.approx(r.point) for v in r.bootstrap_distribution)


# ============================================================================
# TestAutocorrelationWidening - the load-bearing §4.4 property
# ============================================================================


class TestAutocorrelationWidening:
    def test_block_ci_wider_than_iid_on_ar1(self) -> None:
        # rho=0.5 inflates var(mean) by ~(1+rho)/(1-rho)=3x; the iid bootstrap can't see this
        # and its CI is too narrow. The block bootstrap preserves the dependence -> wider, honest.
        ar = _ar1(500, 0.5, 1.0, seed=30)
        iid = bca_bootstrap(ar, _mean, n_resamples=3000, seed=1)
        block = block_bootstrap(ar, _mean, n_resamples=3000, block_mean=10, seed=1)  # stationary
        assert _width(block) > 1.15 * (iid.upper - iid.lower)

    def test_iid_data_blocks_match_iid_roughly(self) -> None:
        # On genuinely iid data the two should agree closely (no dependence to recover).
        white = _gauss(500, 0.0, 1.0, seed=31)
        iid = bca_bootstrap(white, _mean, n_resamples=3000, seed=1)
        block = block_bootstrap(white, _mean, n_resamples=3000, block_mean=5, seed=1)  # stationary
        ratio = _width(block) / (iid.upper - iid.lower)
        assert 0.7 < ratio < 1.4  # comparable; no spurious widening on iid input


# ============================================================================
# TestEstimatedBlockLength - block_mean=None routes through Politis-White
# ============================================================================


class TestEstimatedBlockLength:
    def test_none_estimates_and_flags(self) -> None:
        ar = _ar1(400, 0.5, 1.0, seed=40)
        r = block_bootstrap(ar, _mean, n_resamples=2000, block_mean=None, seed=1)
        assert r.block_mean_estimated is True
        assert r.block_mean == optimal_block_length(ar)
        assert r.block_mean >= 1


# ============================================================================
# TestDeterminism
# ============================================================================


class TestDeterminism:
    def test_same_seed_identical(self) -> None:
        data = _ar1(200, 0.4, 1.0, seed=50)
        a = block_bootstrap(data, _mean, n_resamples=2000, block_mean=6, seed=7)
        b = block_bootstrap(data, _mean, n_resamples=2000, block_mean=6, seed=7)
        assert a == b  # frozen dataclass equality covers the full bootstrap distribution too

    def test_different_seed_differs(self) -> None:
        data = _ar1(200, 0.4, 1.0, seed=50)
        a = block_bootstrap(data, _mean, n_resamples=2000, block_mean=6, seed=7)
        b = block_bootstrap(data, _mean, n_resamples=2000, block_mean=6, seed=8)
        assert a.lower != b.lower or a.upper != b.upper


# ============================================================================
# TestNoDivergence - the stationary path reuses the ONE audited primitive
# ============================================================================


class TestNoDivergence:
    def test_uses_shared_stationary_primitive(self) -> None:
        # identity, not equality: cv's "stationary" path must BE the one WRC/SPA use; a future
        # copy-paste fork into a second implementation would break this and fail loudly. The name
        # is imported from the cv SUBMODULE (the package re-exports the function `block_bootstrap`,
        # which shadows the module attribute - so the module-level name is read from the submodule).
        assert stationary_bootstrap_indices is _stats_stationary


# ============================================================================
# TestMonteCarloCoverage - block covers at least as well as iid on AR(1)
# ============================================================================


class TestMonteCarloCoverage:
    def test_block_covers_true_mean_at_least_as_well_as_iid(self) -> None:
        rng = random.Random(2026)
        true_mean, n, k = 0.0, 120, 100
        iid_cov = 0
        block_cov = 0
        for i in range(k):
            sample = []
            prev = 0.0
            for _ in range(n):
                prev = 0.5 * prev + rng.gauss(0.0, 1.0)  # AR(1) rho=0.5, zero-mean
                sample.append(prev)
            iid = bca_bootstrap(sample, _mean, n_resamples=1000, seed=i)
            block = block_bootstrap(sample, _mean, n_resamples=1000, block_mean=8, seed=i)
            iid_cov += iid.lower <= true_mean <= iid.upper
            block_cov += block.lower <= true_mean <= block.upper
        # §4.4 thesis: iid under-covers on dependent data; block is at least as good.
        assert block_cov >= iid_cov
        assert block_cov / k >= 0.80


# ============================================================================
# TestUndefined (fail-loud) + TestErrors
# ============================================================================


class TestUndefined:
    def test_point_not_finite_returns_nan(self) -> None:
        # profit factor with no losing trades -> inf point statistic -> undefined CI (fail-loud).
        r = block_bootstrap(
            [0.1, 0.2, 0.3, 0.4], profit_factor, n_resamples=1000, block_mean=2, seed=1
        )
        assert math.isnan(r.lower) and math.isnan(r.upper)
        assert r.reason is not None and "not finite" in r.reason
        assert r.bootstrap_distribution == ()


class TestErrors:
    def test_bad_confidence_raises(self) -> None:
        with pytest.raises(BlockBootstrapError, match="confidence must be in"):
            block_bootstrap([0.1, 0.2, 0.3], _mean, confidence=0.0, block_mean=2)

    def test_bad_mode_raises(self) -> None:
        with pytest.raises(BlockBootstrapError, match="mode must be one of"):
            block_bootstrap([0.1, 0.2, 0.3], _mean, block_mean=2, mode="bogus")  # type: ignore[arg-type]

    def test_too_few_resamples_raises(self) -> None:
        with pytest.raises(BlockBootstrapError, match="n_resamples must be"):
            block_bootstrap([0.1, 0.2, 0.3], _mean, n_resamples=500, block_mean=2)

    def test_too_few_data_raises(self) -> None:
        with pytest.raises(BlockBootstrapError, match="need >= 2 data points"):
            block_bootstrap([0.1], _mean, n_resamples=1000)

    def test_non_finite_data_raises(self) -> None:
        with pytest.raises(BlockBootstrapError, match="finite"):
            block_bootstrap([0.1, float("nan"), 0.3], _mean, n_resamples=1000, block_mean=2)

    def test_block_mean_too_large_raises(self) -> None:
        with pytest.raises(BlockBootstrapError, match=r"block_mean must be in"):
            block_bootstrap([0.1, 0.2, 0.3], _mean, n_resamples=1000, block_mean=99)

    def test_block_mean_zero_raises(self) -> None:
        with pytest.raises(BlockBootstrapError, match=r"block_mean must be in"):
            block_bootstrap([0.1, 0.2, 0.3], _mean, n_resamples=1000, block_mean=0)

    def test_result_frozen(self) -> None:
        r = block_bootstrap(
            _gauss(50, 0.01, 0.1, seed=70), _mean, n_resamples=1000, block_mean=4, seed=1
        )
        assert isinstance(r, BlockBootstrapResult)
        with pytest.raises(AttributeError):
            r.lower = 0.0  # type: ignore[misc]
