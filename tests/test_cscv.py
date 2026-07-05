"""CSCV / PBO test suite (futur3.cv.cscv) - gate G11.

Load-bearing tests:
- ROBUST family (one genuine edge + noise): the IS winner is consistently OOS-best -> PBO well below
  0.5, passes_pbo, positive IS/OOS correlation.
- OVERFIT family (each strategy spikes in one distinct segment, no real edge): the IS winner is
  whoever's spike landed in-sample, and it is OOS-poor -> PBO above 0.5, fails_g11, NEGATIVE IS/OOS
  correlation. This is the exact failure mode G11 exists to catch.
- ALL-IDENTICAL -> PBO == 0 (NOT 0.5): the winner sits exactly at the OOS median every split (w=0.5,
  logit=0) and `lambda < 0` is strict. (a predecessor implementation's docstring loosely called this 0.5; its code
  agrees with us at 0.) Documents + locks the literal Bailey-LdP definition.
- DETERMINISM: CSCV enumerates every split (no RNG) so it is a pure function - identical input gives
  an identical result (deterministic by construction).
- THRESHOLD: PBO == 0.5 FAILS G11 (borderline counts as overfit, fail-closed).
"""

from __future__ import annotations

import math
import random
import statistics

import pytest

from futur3.cv import CSCVError, CSCVResult, cscv_pbo
from futur3.cv.cscv import PBO_THRESHOLD, _per_period_sharpe
from futur3.stats import profit_factor


def _robust_family(m: int, t: int, seed: int) -> list[list[float]]:
    """Strategy 0 has a genuine, persistent edge (positive drift); the rest are pure noise."""
    rng = random.Random(seed)
    fams = [[rng.gauss(0.5, 1.0) for _ in range(t)]]
    for _ in range(m - 1):
        fams.append([rng.gauss(0.0, 1.0) for _ in range(t)])
    return fams


def _overfit_family(n_segments: int, seg_size: int, seed: int) -> list[list[float]]:
    """M == n_segments strategies; strategy i spikes ONLY in segment i, flat elsewhere. No strategy
    has a persistent edge, so the IS winner (spike in-sample) is reliably OOS-poor -> high PBO."""
    rng = random.Random(seed)
    fams: list[list[float]] = []
    for i in range(n_segments):
        series: list[float] = []
        for j in range(n_segments):
            for _ in range(seg_size):
                series.append(rng.gauss(3.0, 0.5) if j == i else rng.gauss(0.0, 0.5))
        fams.append(series)
    return fams


def _identical_family(m: int, t: int, seed: int) -> list[list[float]]:
    rng = random.Random(seed)
    base = [rng.gauss(0.1, 1.0) for _ in range(t)]
    return [list(base) for _ in range(m)]


def _make_result(pbo: float) -> CSCVResult:
    """Minimal CSCVResult to test the passes_pbo threshold without running the algorithm."""
    return CSCVResult(
        pbo=pbo,
        n_strategies=5,
        n_observations=100,
        n_segments=8,
        n_combinations=70,
        mean_logit=0.0,
        median_logit=0.0,
        is_oos_correlation=0.0,
        logits=(0.0,),
    )


# ============================================================================
# TestRobustFamily
# ============================================================================


class TestRobustFamily:
    def test_low_pbo_passes_pbo(self) -> None:
        r = cscv_pbo(_robust_family(6, 160, seed=1), n_segments=8)
        assert r.pbo < 0.5
        assert r.passes_pbo is True

    def test_positive_is_oos_correlation(self) -> None:
        r = cscv_pbo(_robust_family(6, 160, seed=2), n_segments=8)
        assert r.is_oos_correlation > 0.0  # in-sample skill carries to out-of-sample
        assert r.mean_logit > 0.0  # winner consistently above the OOS median


# ============================================================================
# TestOverfitFamily - the failure mode G11 exists to catch
# ============================================================================


class TestOverfitFamily:
    def test_high_pbo_fails_g11(self) -> None:
        r = cscv_pbo(_overfit_family(8, 6, seed=3), n_segments=8)
        assert r.pbo > 0.5
        assert r.passes_pbo is False

    def test_negative_is_oos_correlation(self) -> None:
        r = cscv_pbo(_overfit_family(8, 6, seed=4), n_segments=8)
        assert r.is_oos_correlation < 0.0  # in-sample winner is out-of-sample loser
        assert r.mean_logit < 0.0


# ============================================================================
# TestAllIdentical - PBO == 0 (the literal definition), not 0.5
# ============================================================================


class TestAllIdentical:
    def test_identical_strategies_pbo_zero(self) -> None:
        r = cscv_pbo(_identical_family(4, 160, seed=5), n_segments=8)
        assert r.pbo == 0.0
        assert all(lam == 0.0 for lam in r.logits)  # every winner sits exactly at the OOS median


# ============================================================================
# TestDeterminism (pure function, no RNG)
# ============================================================================


class TestDeterminism:
    def test_same_input_identical(self) -> None:
        fam = _robust_family(5, 160, seed=6)
        assert cscv_pbo(fam, n_segments=8) == cscv_pbo(fam, n_segments=8)

    def test_input_copy_identical(self) -> None:
        fam = _robust_family(5, 160, seed=6)
        copy = [list(s) for s in fam]
        assert cscv_pbo(fam, n_segments=8) == cscv_pbo(copy, n_segments=8)


# ============================================================================
# TestStructure - combinatorics + remainder handling
# ============================================================================


class TestStructure:
    def test_n_combinations(self) -> None:
        r = cscv_pbo(_robust_family(4, 160, seed=7), n_segments=8)
        assert r.n_combinations == math.comb(8, 4)  # 70
        assert len(r.logits) == r.n_combinations  # every split completed (default metric is finite)

    def test_drops_trailing_remainder(self) -> None:
        # T=165, S=8 -> seg_size=20, used = 160; 5 trailing observations dropped.
        r = cscv_pbo(_robust_family(3, 165, seed=8), n_segments=8)
        assert r.n_observations == 160

    def test_pbo_in_unit_interval(self) -> None:
        r = cscv_pbo(_robust_family(5, 120, seed=9), n_segments=6)
        assert 0.0 <= r.pbo <= 1.0


# ============================================================================
# TestMetricPluggable
# ============================================================================


class TestMetricPluggable:
    def test_mean_metric_robust_low_pbo(self) -> None:
        # the mean also selects the genuine-edge strategy -> still robust.
        r = cscv_pbo(_robust_family(6, 160, seed=10), n_segments=8, metric=statistics.fmean)
        assert r.pbo < 0.5

    def test_profit_factor_metric_runs(self) -> None:
        r = cscv_pbo(_robust_family(6, 160, seed=11), n_segments=8, metric=profit_factor)
        assert 0.0 <= r.pbo <= 1.0  # inf-PF folds (no losses) are skipped, the rest still rank

    def test_default_metric_is_per_period_sharpe(self) -> None:
        sample = [1.0, 2.0, 3.0]
        expected = statistics.fmean(sample) / statistics.stdev(sample)
        assert _per_period_sharpe(sample) == pytest.approx(expected)
        assert _per_period_sharpe([5.0, 5.0, 5.0]) == 0.0  # zero-variance fold -> 0.0
        assert _per_period_sharpe([1.0]) == 0.0  # too few obs -> 0.0


# ============================================================================
# TestThresholdProperty - G11 fail-closed at the boundary
# ============================================================================


class TestThresholdProperty:
    def test_threshold_constant(self) -> None:
        assert PBO_THRESHOLD == 0.5

    def test_below_threshold_passes(self) -> None:
        assert _make_result(0.49).passes_pbo is True

    def test_at_threshold_fails(self) -> None:
        assert _make_result(0.5).passes_pbo is False  # borderline counts as overfit (fail-closed)

    def test_above_threshold_fails(self) -> None:
        assert _make_result(0.6).passes_pbo is False


# ============================================================================
# TestErrors (fail-loud - contract violations raise)
# ============================================================================


class TestErrors:
    def test_odd_segments_raises(self) -> None:
        with pytest.raises(CSCVError, match="even integer"):
            cscv_pbo(_robust_family(4, 80, seed=12), n_segments=7)

    def test_too_few_segments_raises(self) -> None:
        with pytest.raises(CSCVError, match="even integer"):
            cscv_pbo(_robust_family(4, 80, seed=12), n_segments=0)

    def test_too_few_strategies_raises(self) -> None:
        with pytest.raises(CSCVError, match="need >= 2 strategies"):
            cscv_pbo([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], n_segments=2)

    def test_ragged_raises(self) -> None:
        with pytest.raises(CSCVError, match="same number"):
            cscv_pbo([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3]], n_segments=2)

    def test_non_finite_raises(self) -> None:
        with pytest.raises(CSCVError, match="finite"):
            cscv_pbo([[0.1, 0.2, float("nan"), 0.4], [0.1, 0.2, 0.3, 0.4]], n_segments=2)

    def test_insufficient_obs_per_segment_raises(self) -> None:
        # T=10, S=8 -> 1 obs/segment, below the floor of 2.
        with pytest.raises(CSCVError, match="segments needs"):
            cscv_pbo([[float(i) for i in range(10)], [float(-i) for i in range(10)]], n_segments=8)


# ============================================================================
# TestFrozen
# ============================================================================


class TestFrozen:
    def test_result_frozen(self) -> None:
        r = cscv_pbo(_robust_family(4, 80, seed=13), n_segments=8)
        assert isinstance(r, CSCVResult)
        with pytest.raises(AttributeError):
            r.pbo = 0.0  # type: ignore[misc]
