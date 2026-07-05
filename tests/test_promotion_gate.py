"""Composite promotion-gate test suite (futur3.stats.promotion_gate).

Load-bearing tests:
- a genuinely strong strategy (high Sharpe, few low-variance trials, profitable, superior to its
  family) passes ALL FIVE gates -> promoted, failed_gates empty;
- a no-edge strategy fails -> not promoted, every gate named in failed_gates;
- failed_gates is exactly the set of failing gate properties (structural invariant), and `promoted`
  is their AND; the SPA/WRC switch is honoured; every underlying result object is kept for audit.
"""

from __future__ import annotations

import random

import pytest

from futur3.stats import (
    BCaResult,
    DSRResult,
    PermutationResult,
    PromotionDecision,
    PSRResult,
    RealityCheckResult,
    SPAResult,
    evaluate_promotion,
)


def _matrix(means: list[float], n_obs: int, sigma: float, seed: int) -> list[list[float]]:
    rng = random.Random(seed)
    return [[rng.gauss(mu, sigma) for _ in range(n_obs)] for mu in means]


def _good_strategy(n_obs: int, seed: int) -> list[float]:
    # mean 0.08, sigma 0.5 -> per-period Sharpe 0.16, annualized ~2.5 (strong, robust edge)
    rng = random.Random(seed)
    return [rng.gauss(0.08, 0.5) for _ in range(n_obs)]


# ============================================================================
# TestPromotion
# ============================================================================


class TestPromotion:
    def test_good_strategy_promoted(self) -> None:
        strat = _good_strategy(250, seed=1)
        family = [strat, *_matrix([0.0] * 5, 250, 0.5, seed=2)]
        sr_trials = [2.5, 0.1, -0.2, 0.3, 0.0, 0.1]
        d = evaluate_promotion(strat, family, sr_trials, n_resamples=1000, seed=7)
        assert d.promoted is True
        assert d.failed_gates == ()
        assert all([d.passes_dsr, d.passes_psr, d.passes_profit_factor, d.passes_permutation, d.passes_superiority])

    def test_no_edge_not_promoted(self) -> None:
        rng = random.Random(3)
        weak = [rng.gauss(0.0, 1.0) for _ in range(250)]
        family = [weak, *_matrix([0.0] * 5, 250, 1.0, seed=4)]
        d = evaluate_promotion(weak, family, [0.0] * 6, n_resamples=1000, seed=7)
        assert d.promoted is False
        assert len(d.failed_gates) > 0

    def test_failed_gates_consistency(self) -> None:
        rng = random.Random(5)
        weak = [rng.gauss(0.0, 1.0) for _ in range(200)]
        family = [weak, *_matrix([0.0] * 4, 200, 1.0, seed=6)]
        d = evaluate_promotion(weak, family, [0.0] * 5, n_resamples=1000, seed=7)
        gate_flags = {
            "deflated_sharpe": d.passes_dsr,
            "probabilistic_sharpe": d.passes_psr,
            "bca_profit_factor": d.passes_profit_factor,
            "permutation": d.passes_permutation,
            f"superiority_{d.superiority_method}": d.passes_superiority,
        }
        assert set(d.failed_gates) == {name for name, ok in gate_flags.items() if not ok}
        assert d.promoted == (len(d.failed_gates) == 0)

    def test_diagnostics_retained(self) -> None:
        strat = _good_strategy(200, seed=8)
        family = [strat, *_matrix([0.0] * 4, 200, 0.5, seed=9)]
        d = evaluate_promotion(strat, family, [2.5, 0.1, 0.0, -0.1, 0.2], n_resamples=1000, seed=7)
        assert isinstance(d.dsr, DSRResult)
        assert isinstance(d.psr, PSRResult)
        assert isinstance(d.bca_profit_factor, BCaResult)
        assert isinstance(d.permutation, PermutationResult)
        assert isinstance(d.superiority, SPAResult)

    def test_wrc_variant(self) -> None:
        strat = _good_strategy(200, seed=10)
        family = [strat, *_matrix([0.0] * 4, 200, 0.5, seed=11)]
        d = evaluate_promotion(
            strat,
            family,
            [2.5, 0.1, 0.0, -0.1, 0.2],
            n_resamples=1000,
            superiority_method="wrc",
            seed=7,
        )
        assert d.superiority_method == "wrc"
        assert isinstance(d.superiority, RealityCheckResult)

    def test_determinism(self) -> None:
        strat = _good_strategy(200, seed=12)
        family = [strat, *_matrix([0.0] * 4, 200, 0.5, seed=13)]
        trials = [2.5, 0.1, 0.0, -0.1, 0.2]
        a = evaluate_promotion(strat, family, trials, n_resamples=1000, seed=7)
        b = evaluate_promotion(strat, family, trials, n_resamples=1000, seed=7)
        assert a.promoted == b.promoted
        assert a.bca_profit_factor.lower == b.bca_profit_factor.lower
        assert a.permutation.p_value == b.permutation.p_value
        assert a.superiority.p_value == b.superiority.p_value

    def test_result_frozen(self) -> None:
        strat = _good_strategy(150, seed=14)
        family = [strat, *_matrix([0.0] * 3, 150, 0.5, seed=15)]
        d = evaluate_promotion(strat, family, [2.5, 0.1, 0.0, -0.1], n_resamples=1000, seed=7)
        assert isinstance(d, PromotionDecision)
        with pytest.raises(AttributeError):
            d.superiority_method = "wrc"  # type: ignore[misc]
