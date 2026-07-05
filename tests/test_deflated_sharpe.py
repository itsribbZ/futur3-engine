"""DSR test suite (futur3.stats.deflated_sharpe).

Load-bearing here = the SENTINEL CALIBRATIONS (a known DSR failure mode):
- a genuinely strong strategy trialled few times -> DSR > 0.95 (passes the G3 gate);
- the SAME strong returns but reported as the best of MANY high-variance trials -> DSR < 0.05
  (selection bias correctly destroys the apparent edge).
Plus the structural invariant DSR <= PSR_implied (deflation can only lower the probability) and the
honest-input contract (NaN trials dropped, n_trials < 2 -> undefined, n_obs < 10 -> undefined).
"""

from __future__ import annotations

import random

import pytest

from futur3.stats import (
    DSR_THRESHOLD,
    DSRResult,
    PSRError,
    deflated_sharpe,
    probabilistic_sharpe,
)


def _gauss(n: int, mu: float, sigma: float, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(mu, sigma) for _ in range(n)]


# ============================================================================
# TestSentinels - the fake-alpha calibration (§3.1 failure-mode 4)
# ============================================================================


class TestSentinels:
    def test_known_good_strategy_passes_dsr(self) -> None:
        # Strong, stable edge (high per-period Sharpe over many obs), trialled only twice with low
        # cross-trial variance -> little selection bias -> DSR should clear the gate.
        returns = _gauss(2000, 0.0015, 0.008, seed=20)
        result = deflated_sharpe(returns, sr_trials=[2.9, 3.0], periods_per_year=252)
        assert result.dsr is not None and result.dsr > DSR_THRESHOLD
        assert result.passes_dsr is True

    def test_overfit_winner_fails_g3(self) -> None:
        # The SAME returns, but now presented as the best of 500 wildly-varying trials. The expected
        # maximum Sharpe under that much searching is huge -> the edge is deflated to nothing.
        returns = _gauss(2000, 0.0015, 0.008, seed=20)
        rng = random.Random(21)
        many_trials = [rng.gauss(1.5, 1.5) for _ in range(500)]
        result = deflated_sharpe(returns, sr_trials=many_trials, periods_per_year=252)
        assert result.dsr is not None and result.dsr < 0.05
        assert result.passes_dsr is False
        assert result.sr_star is not None and result.sr_observed is not None
        assert result.sr_star > result.sr_observed  # searching N trials inflated the bar past it

    def test_more_trials_strictly_lower_dsr(self) -> None:
        # Holding the trial spread fixed, searching more variants raises SR* and lowers DSR.
        returns = _gauss(2000, 0.0012, 0.008, seed=22)
        few = deflated_sharpe(returns, sr_trials=[2.0, 2.4], periods_per_year=252)
        rng = random.Random(23)
        many = deflated_sharpe(
            returns, sr_trials=[rng.gauss(2.2, 0.2) for _ in range(200)], periods_per_year=252
        )
        assert few.dsr is not None and many.dsr is not None
        assert many.dsr < few.dsr


# ============================================================================
# TestDeflationInvariant
# ============================================================================


class TestDeflationInvariant:
    def test_dsr_le_psr_implied(self) -> None:
        # SR* >= 0 always, and PSR is decreasing in its threshold, so deflation can never raise the
        # probability above the unconditional PSR(0).
        returns = _gauss(1500, 0.0008, 0.01, seed=24)
        r = deflated_sharpe(returns, sr_trials=[1.0, 1.3, 0.8, 1.1], periods_per_year=252)
        assert r.dsr is not None and r.psr_implied is not None
        assert r.dsr <= r.psr_implied + 1e-12

    def test_psr_implied_matches_standalone_psr(self) -> None:
        returns = _gauss(1500, 0.0008, 0.01, seed=25)
        r = deflated_sharpe(returns, sr_trials=[1.0, 1.3], periods_per_year=252)
        standalone = probabilistic_sharpe(returns, sr_threshold=0.0, periods_per_year=252)
        assert r.psr_implied == standalone.psr

    def test_zero_trial_variance_collapses_to_psr_implied(self) -> None:
        # Identical trial Sharpes -> var 0 -> SR* = 0 -> DSR == PSR(0).
        returns = _gauss(1500, 0.001, 0.01, seed=26)
        r = deflated_sharpe(returns, sr_trials=[1.5, 1.5, 1.5], periods_per_year=252)
        assert r.sr_star == 0.0
        assert r.dsr == r.psr_implied


# ============================================================================
# TestHonestInputContract
# ============================================================================


class TestHonestInputContract:
    def test_nan_trials_dropped(self) -> None:
        returns = _gauss(1500, 0.001, 0.01, seed=27)
        r = deflated_sharpe(
            returns, sr_trials=[1.5, float("nan"), 2.0, float("inf")], periods_per_year=252
        )
        assert r.n_trials == 2  # only the two finite trial Sharpes counted
        assert r.dsr is not None

    def test_fewer_than_two_trials_none(self) -> None:
        returns = _gauss(1500, 0.001, 0.01, seed=28)
        r = deflated_sharpe(returns, sr_trials=[1.5], periods_per_year=252)
        assert r.dsr is None
        assert r.reason is not None and "n_trials<2" in r.reason
        assert r.sr_star is None
        assert r.psr_implied is not None  # the unconditional PSR is still reported

    def test_all_nan_trials_none(self) -> None:
        returns = _gauss(1500, 0.001, 0.01, seed=29)
        r = deflated_sharpe(returns, sr_trials=[float("nan"), float("nan")], periods_per_year=252)
        assert r.n_trials == 0
        assert r.dsr is None

    def test_insufficient_obs_none(self) -> None:
        # n_obs < 10 -> moments unreliable -> DSR undefined (kernel-path reason).
        r = deflated_sharpe(
            _gauss(8, 0.01, 0.1, seed=30), sr_trials=[1.0, 1.2], periods_per_year=252
        )
        assert r.dsr is None
        assert r.reason is not None and "n<10" in r.reason


# ============================================================================
# TestDSRErrors + structure
# ============================================================================


class TestDSRErrors:
    def test_bad_ppy_raises(self) -> None:
        with pytest.raises(PSRError, match="periods_per_year must be > 0"):
            deflated_sharpe([0.01, 0.02], sr_trials=[1.0, 1.2], periods_per_year=-1)

    def test_non_finite_returns_raises(self) -> None:
        with pytest.raises(PSRError, match="finite"):
            deflated_sharpe([0.01, float("nan")], sr_trials=[1.0, 1.2], periods_per_year=252)

    def test_threshold_constant(self) -> None:
        assert DSR_THRESHOLD == 0.95

    def test_result_frozen(self) -> None:
        r = deflated_sharpe(
            _gauss(50, 0.01, 0.1, seed=31), sr_trials=[1.0, 1.2], periods_per_year=252
        )
        assert isinstance(r, DSRResult)
        with pytest.raises(AttributeError):
            r.dsr = 1.0  # type: ignore[misc]
