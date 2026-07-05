"""Romano-Wolf stepdown suite (futur3.stats.romano_wolf) - gate G12.

Load-bearing tests:
- FWER CONTROL (Monte-Carlo): under an all-null family (every strategy noise), P(reject any) stays
  near alpha. This is the guarantee Romano-Wolf exists to provide.
- SIGNAL DETECTION: a genuine-edge strategy among noise IS rejected (significant) and named in
  significant_indices; passes_fwer is True.
- STEPDOWN MONOTONICITY: adjusted p-values are non-decreasing down the descending-statistic order
  (the stepwise invariant).
- DETERMINISM: a fixed seed -> identical result.
"""

from __future__ import annotations

import random

import pytest

from futur3.stats.romano_wolf import (
    FWER_ALPHA,
    RomanoWolfError,
    RomanoWolfResult,
    romano_wolf_stepdown,
)


def _noise(n_strat: int, t: int, seed: int) -> list[list[float]]:
    rng = random.Random(seed)
    return [[rng.gauss(0.0, 1.0) for _ in range(t)] for _ in range(n_strat)]


def _one_edge(n_strat: int, t: int, edge: float, seed: int) -> list[list[float]]:
    """Strategy 0 has a genuine positive edge; the rest are noise."""
    rng = random.Random(seed)
    fams = [[rng.gauss(edge, 1.0) for _ in range(t)]]
    fams += [[rng.gauss(0.0, 1.0) for _ in range(t)] for _ in range(n_strat - 1)]
    return fams


# ============================================================================
# TestSignalDetection
# ============================================================================


class TestSignalDetection:
    def test_genuine_edge_rejected(self) -> None:
        fam = _one_edge(5, 250, edge=0.3, seed=1)  # t ~ 0.3*sqrt(250) ~ 4.7 -> strongly significant
        r = romano_wolf_stepdown(fam, n_bootstrap=2000, seed=7)
        assert r.reject[0] is True
        assert 0 in r.significant_indices
        assert r.passes_fwer is True

    def test_edge_strategy_has_largest_statistic(self) -> None:
        fam = _one_edge(5, 250, edge=0.3, seed=2)
        r = romano_wolf_stepdown(fam, n_bootstrap=2000, seed=7)
        assert max(range(5), key=lambda k: r.observed_stats[k]) == 0


# ============================================================================
# TestFWERControl - the Monte-Carlo guarantee
# ============================================================================


class TestFWERControl:
    def test_all_null_family_rarely_rejects(self) -> None:
        # Under all-null (every strategy noise), P(reject any) <= alpha. Count over K families.
        k_families = 40
        any_reject = 0
        for trial in range(k_families):
            fam = _noise(5, 120, seed=1000 + trial)
            r = romano_wolf_stepdown(fam, n_bootstrap=1000, seed=trial)
            any_reject += r.any_significant
        # alpha=0.05 nominal; generous upper bound for K=40 + bootstrap/MC slack.
        assert any_reject / k_families <= 0.20


# ============================================================================
# TestStepdownMonotonicity
# ============================================================================


class TestStepdownMonotonicity:
    def test_adjusted_non_decreasing_down_the_order(self) -> None:
        fam = _one_edge(6, 200, edge=0.15, seed=3)
        r = romano_wolf_stepdown(fam, n_bootstrap=2000, seed=7)
        # sort by observed statistic DESCENDING; adjusted p must be non-decreasing along that order.
        by_stat = sorted(
            zip(r.observed_stats, r.adjusted_pvalues, strict=True), key=lambda x: x[0], reverse=True
        )
        adj = [a for _, a in by_stat]
        assert all(adj[i] <= adj[i + 1] + 1e-12 for i in range(len(adj) - 1))


# ============================================================================
# TestDeterminism
# ============================================================================


class TestDeterminism:
    def test_same_seed_identical(self) -> None:
        fam = _one_edge(5, 200, edge=0.2, seed=4)
        a = romano_wolf_stepdown(fam, n_bootstrap=2000, seed=7)
        b = romano_wolf_stepdown(fam, n_bootstrap=2000, seed=7)
        assert a == b

    def test_different_seed_may_differ_but_stays_valid(self) -> None:
        fam = _one_edge(5, 200, edge=0.2, seed=4)
        a = romano_wolf_stepdown(fam, n_bootstrap=2000, seed=7)
        b = romano_wolf_stepdown(fam, n_bootstrap=2000, seed=8)
        assert len(a.adjusted_pvalues) == len(b.adjusted_pvalues) == 5  # shape stable


# ============================================================================
# TestStructure
# ============================================================================


class TestStructure:
    def test_g12_matches_any_significant(self) -> None:
        fam = _one_edge(4, 250, edge=0.3, seed=5)
        r = romano_wolf_stepdown(fam, n_bootstrap=2000, seed=7)
        assert r.passes_fwer == r.any_significant
        assert r.n_rejected == sum(r.reject)
        assert tuple(r.significant_indices) == tuple(i for i, x in enumerate(r.reject) if x)

    def test_adjusted_pvalues_in_unit_interval(self) -> None:
        fam = _noise(5, 150, seed=6)
        r = romano_wolf_stepdown(fam, n_bootstrap=1000, seed=7)
        assert all(0.0 <= p <= 1.0 for p in r.adjusted_pvalues)


# ============================================================================
# TestErrors + TestFrozen
# ============================================================================


class TestErrors:
    def test_too_few_bootstraps_raises(self) -> None:
        with pytest.raises(RomanoWolfError, match="n_bootstrap must be"):
            romano_wolf_stepdown(_noise(3, 50, seed=1), n_bootstrap=500)

    def test_bad_block_length_raises(self) -> None:
        with pytest.raises(RomanoWolfError, match="block_length must be"):
            romano_wolf_stepdown(_noise(3, 50, seed=1), n_bootstrap=1000, block_length=0)

    def test_bad_alpha_raises(self) -> None:
        with pytest.raises(RomanoWolfError, match="alpha must be in"):
            romano_wolf_stepdown(_noise(3, 50, seed=1), n_bootstrap=1000, alpha=1.5)

    def test_default_alpha_constant(self) -> None:
        assert FWER_ALPHA == 0.05


class TestFrozen:
    def test_result_frozen(self) -> None:
        r = romano_wolf_stepdown(_one_edge(4, 200, edge=0.2, seed=9), n_bootstrap=1000, seed=7)
        assert isinstance(r, RomanoWolfResult)
        with pytest.raises(AttributeError):
            r.n_rejected = 0  # type: ignore[misc]
