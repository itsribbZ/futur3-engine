"""PSR + shared Sharpe-kernel test suite (futur3.stats.probabilistic_sharpe + _sharpe_core).

Covers the higher-moment maths the validation layer depends on. The load-bearing tests:
- RAW (not excess) kurtosis: a Gaussian sample must give kurt ~ 3.0, not ~ 0 - using excess
  kurtosis in the kernel silently understates variance and INFLATES PSR/DSR (a fake-alpha bug).
- Per-period-kernel proof: PSR(threshold=0) is INVARIANT to periods_per_year (the de-annualization
  cancels); PSR(threshold!=0) correctly DOES depend on it.
- Monte-Carlo calibration: at a benchmark equal to the true Sharpe, PSR averages ~0.5 over many
  resamples - end-to-end validation of the kernel against its definition.
- Fail-loud: undefined cases (n<2, zero variance, n<10, non-positive variance term) surface None, never 0.
"""

from __future__ import annotations

import math
import random
import statistics

import pytest

from futur3.stats import PSR_THRESHOLD, PSRError, PSRResult, probabilistic_sharpe
from futur3.stats._sharpe_core import (
    SharpeMoments,
    compute_sharpe_moments,
    expected_max_sharpe,
    psr_kernel,
)
from futur3.stats.probabilistic_sharpe import min_track_record_length


def _gauss(n: int, mu: float, sigma: float, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(mu, sigma) for _ in range(n)]


# ============================================================================
# TestMoments - the raw-vs-excess kurtosis guard lives here
# ============================================================================


class TestMoments:
    def test_gaussian_raw_kurtosis_near_three(self) -> None:
        # RAW kurtosis of a Gaussian is ~3.0. If this were ~0 the code used EXCESS kurtosis,
        # which would understate the kernel variance term and inflate PSR/DSR (silent fake alpha).
        m = compute_sharpe_moments(_gauss(20000, 0.0, 1.0, seed=1))
        assert m is not None
        assert m.kurt == pytest.approx(3.0, abs=0.15)

    def test_gaussian_skew_near_zero(self) -> None:
        m = compute_sharpe_moments(_gauss(20000, 0.0, 1.0, seed=2))
        assert m is not None
        assert m.skew == pytest.approx(0.0, abs=0.1)

    def test_right_skewed_sample_positive_skew(self) -> None:
        rng = random.Random(3)
        data = [rng.expovariate(1.0) for _ in range(20000)]  # exponential -> skew ~ +2
        m = compute_sharpe_moments(data)
        assert m is not None
        assert m.skew > 1.0
        assert m.kurt > 3.0  # heavier-tailed than normal

    def test_sr_per_period_sign(self) -> None:
        up = compute_sharpe_moments(_gauss(500, 0.05, 0.1, seed=4))
        dn = compute_sharpe_moments(_gauss(500, -0.05, 0.1, seed=5))
        assert up is not None and up.sr_per_period > 0
        assert dn is not None and dn.sr_per_period < 0

    def test_too_few_returns_none(self) -> None:
        assert compute_sharpe_moments([0.01]) is None

    def test_zero_variance_none(self) -> None:
        assert compute_sharpe_moments([0.01] * 50) is None

    def test_moments_frozen(self) -> None:
        m = compute_sharpe_moments(_gauss(50, 0.01, 0.1, seed=6))
        assert isinstance(m, SharpeMoments)
        with pytest.raises(AttributeError):
            m.skew = 0.0  # type: ignore[misc]


# ============================================================================
# TestKernel - psr_kernel direct (per-period units)
# ============================================================================


class TestKernel:
    def test_at_observed_equals_half(self) -> None:
        # threshold == observed -> z = 0 -> Phi(0) = 0.5
        assert psr_kernel(0.1, 0.1, 100, 0.0, 3.0) == pytest.approx(0.5)

    def test_monotone_decreasing_in_threshold(self) -> None:
        low = psr_kernel(0.2, 0.0, 100, 0.0, 3.0)
        high = psr_kernel(0.2, 0.15, 100, 0.0, 3.0)
        assert low is not None and high is not None
        assert low > high  # a tougher benchmark lowers the probability

    def test_more_obs_sharpens(self) -> None:
        few = psr_kernel(0.2, 0.0, 30, 0.0, 3.0)
        many = psr_kernel(0.2, 0.0, 3000, 0.0, 3.0)
        assert few is not None and many is not None
        assert many > few  # same edge, more evidence -> higher confidence

    def test_negative_variance_term_none(self) -> None:
        # 1 - skew*SR + (kurt-1)/4*SR^2 = 1 - 5*1 + 0.5 = -3.5 < 0 -> undefined (fail-loud)
        assert psr_kernel(1.0, 0.0, 100, 5.0, 3.0) is None


# ============================================================================
# TestExpectedMaxSharpe - shared with DSR; SR* selection-bias term
# ============================================================================


class TestExpectedMaxSharpe:
    def test_known_value_two_trials_unit_var(self) -> None:
        # Phi^-1(0.5) = 0 so term1 vanishes; result = gamma * Phi^-1(1 - 1/(2e)).
        assert expected_max_sharpe(2, 1.0) == pytest.approx(0.5198, abs=1e-3)

    def test_monotone_increasing_in_n_trials(self) -> None:
        vals = [expected_max_sharpe(n, 1.0) for n in (2, 10, 50, 500)]
        assert vals == sorted(vals)
        assert len(set(vals)) == len(vals)  # strictly increasing

    def test_scales_with_sqrt_variance(self) -> None:
        # sigma = sqrt(var); quadrupling var doubles SR*.
        assert expected_max_sharpe(50, 4.0) == pytest.approx(2.0 * expected_max_sharpe(50, 1.0))

    def test_below_two_trials_zero(self) -> None:
        assert expected_max_sharpe(1, 1.0) == 0.0
        assert expected_max_sharpe(0, 1.0) == 0.0

    def test_nonpositive_variance_zero(self) -> None:
        assert expected_max_sharpe(50, 0.0) == 0.0
        assert expected_max_sharpe(50, -1.0) == 0.0


# ============================================================================
# TestPSR - the public function
# ============================================================================


class TestPSR:
    def test_strong_edge_high_psr(self) -> None:
        r = probabilistic_sharpe(_gauss(2000, 0.001, 0.005, seed=10), periods_per_year=252)
        assert r.psr is not None and r.psr > PSR_THRESHOLD
        assert r.passes_psr is True
        assert r.sufficient_n is True

    def test_zero_realized_edge_psr_half(self) -> None:
        # De-mean a sample so its REALIZED Sharpe is exactly 0; PSR(0) must be ~0.5 (z = 0). A
        # true-zero-mean process still produces samples with nonzero realized Sharpe that PSR
        # correctly reads as evidence over n=2000 - so we pin the realized edge, not the population.
        raw = _gauss(2000, 0.0, 0.01, seed=11)
        mean = statistics.fmean(raw)
        centered = [x - mean for x in raw]
        r = probabilistic_sharpe(centered, periods_per_year=252)
        assert r.psr is not None and r.psr == pytest.approx(0.5, abs=0.02)
        assert r.passes_psr is False

    def test_negative_edge_low_psr(self) -> None:
        r = probabilistic_sharpe(_gauss(2000, -0.001, 0.005, seed=12), periods_per_year=252)
        assert r.psr is not None and r.psr < 0.5

    def test_annualization_invariance_at_zero_threshold(self) -> None:
        # PSR(threshold=0) must be IDENTICAL regardless of periods_per_year: the per-period kernel
        # is intrinsic and the threshold de-annualizes to 0 either way. This proves the kernel is
        # NOT mistakenly fed an annualized Sharpe.
        data = _gauss(1000, 0.0005, 0.01, seed=13)
        a = probabilistic_sharpe(data, sr_threshold=0.0, periods_per_year=252)
        b = probabilistic_sharpe(data, sr_threshold=0.0, periods_per_year=78 * 252)
        assert a.psr is not None and b.psr is not None
        assert a.psr == pytest.approx(b.psr, abs=1e-12)

    def test_nonzero_threshold_depends_on_ppy(self) -> None:
        # With a non-zero benchmark the de-annualization matters, so PSR DOES move with ppy.
        data = _gauss(1000, 0.0005, 0.01, seed=14)
        a = probabilistic_sharpe(data, sr_threshold=1.0, periods_per_year=252)
        b = probabilistic_sharpe(data, sr_threshold=1.0, periods_per_year=78 * 252)
        assert a.psr is not None and b.psr is not None
        assert a.psr != pytest.approx(b.psr, abs=1e-6)

    def test_higher_threshold_lowers_psr(self) -> None:
        data = _gauss(2000, 0.001, 0.005, seed=15)
        easy = probabilistic_sharpe(data, sr_threshold=0.0, periods_per_year=252)
        hard = probabilistic_sharpe(data, sr_threshold=3.0, periods_per_year=252)
        assert easy.psr is not None and hard.psr is not None
        assert easy.psr > hard.psr

    def test_sr_annualized_value(self) -> None:
        # mean 0.02, sample std stdev([0.01,0.03]) = 0.0141..., sr_pp = 1.41421, *sqrt(4) = 2.82843.
        r = probabilistic_sharpe([0.01, 0.03], periods_per_year=4)
        assert r.sr_annualized == pytest.approx(2.82842712, abs=1e-6)
        assert r.psr is None  # n < 10 -> probability untrustworthy
        assert r.sufficient_n is False

    def test_insufficient_n_none_but_sr_reported(self) -> None:
        r = probabilistic_sharpe(_gauss(5, 0.01, 0.1, seed=16), periods_per_year=252)
        assert r.psr is None
        assert r.reason is not None and "n<10" in r.reason
        assert r.sr_annualized is not None  # descriptive value still surfaced
        assert r.skew is not None

    def test_zero_variance_none(self) -> None:
        r = probabilistic_sharpe([0.01] * 50, periods_per_year=252)
        assert r.psr is None
        assert r.reason is not None and "insufficient data" in r.reason
        assert r.sr_annualized is None


# ============================================================================
# TestPSRErrors
# ============================================================================


class TestPSRErrors:
    def test_bad_ppy_raises(self) -> None:
        with pytest.raises(PSRError, match="periods_per_year must be > 0"):
            probabilistic_sharpe([0.01, 0.02, 0.03], periods_per_year=0)

    def test_non_finite_returns_raises(self) -> None:
        with pytest.raises(PSRError, match="finite"):
            probabilistic_sharpe([0.01, float("nan"), 0.03], periods_per_year=252)

    def test_inf_returns_raises(self) -> None:
        with pytest.raises(PSRError, match="finite"):
            probabilistic_sharpe([0.01, float("inf"), 0.03], periods_per_year=252)

    def test_result_frozen(self) -> None:
        r = probabilistic_sharpe(_gauss(50, 0.01, 0.1, seed=17), periods_per_year=252)
        assert isinstance(r, PSRResult)
        with pytest.raises(AttributeError):
            r.psr = 1.0  # type: ignore[misc]


# ============================================================================
# TestPSRMonteCarloCalibration - end-to-end validation against the definition
# ============================================================================


class TestMinTRL:
    def test_none_when_sr_not_above_star(self) -> None:
        assert min_track_record_length(0.1, 0.1, 0.0, 3.0) is None  # SR == SR*
        assert min_track_record_length(0.05, 0.1, 0.0, 3.0) is None  # SR < SR*

    def test_decreases_with_higher_sr(self) -> None:
        small = min_track_record_length(0.05, 0.0, 0.0, 3.0)
        big = min_track_record_length(0.20, 0.0, 0.0, 3.0)
        assert small is not None and big is not None
        assert big < small  # a stronger edge needs fewer observations to be significant

    def test_inverts_the_psr_kernel(self) -> None:
        # at n == MinTRL, PSR(SR vs SR*) == 1 - alpha — the inversion is self-consistent.
        mtrl = min_track_record_length(0.1, 0.0, 0.0, 3.0, alpha=0.05)
        assert mtrl is not None
        psr = psr_kernel(0.1, 0.0, round(mtrl), 0.0, 3.0)
        assert psr is not None and psr == pytest.approx(0.95, abs=0.01)


class TestPSRMonteCarloCalibration:
    def test_psr_calibrated_at_true_sharpe(self) -> None:
        # If the benchmark equals the TRUE per-period Sharpe, the observed-vs-true z is ~N(0,1),
        # so PSR ~ Uniform(0,1) across resamples and its mean is ~0.5. A materially wrong kernel
        # (annualization mix-up, excess kurtosis, wrong sqrt(n) scaling) breaks this.
        rng = random.Random(2024)
        mu, sigma, n = 0.0008, 0.01, 250
        true_sr_ann = (mu / sigma) * math.sqrt(252)
        psrs: list[float] = []
        for _ in range(400):
            sample = [rng.gauss(mu, sigma) for _ in range(n)]
            res = probabilistic_sharpe(sample, sr_threshold=true_sr_ann, periods_per_year=252)
            assert res.psr is not None
            psrs.append(res.psr)
        assert statistics.fmean(psrs) == pytest.approx(0.5, abs=0.05)
