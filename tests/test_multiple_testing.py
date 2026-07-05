"""Multiple-testing correction suite (futur3.stats.multiple_testing) - §5.

Load-bearing tests:
- R p.adjust PARITY: adjusted p-values for [0.01,0.02,0.03,0.04,0.05] match R's p.adjust exactly -
  bonferroni [.05,.10,.15,.20,.25], holm [.05,.08,.09,.09,.09], BH all .05. The correctness anchor.
- INPUT-ORDER PRESERVATION: an unsorted p-vector returns adjusted/reject mapped back to input order.
- RELATIVE POWER: rejection counts are nested bonferroni <= holm <= bh_fdr (FWER conservative -> FDR
  liberal); mixing the two mid-stream is the §5 alpha-laundering bug, so the method is explicit.
"""

from __future__ import annotations

import pytest

from futur3.stats.multiple_testing import (
    MultipleTestError,
    MultipleTestResult,
    bh_fdr_correct,
    bonferroni_correct,
    holm_bonferroni_correct,
)

_P5 = [0.01, 0.02, 0.03, 0.04, 0.05]  # ascending; R p.adjust reference vector


# ============================================================================
# TestRAdjustParity - the correctness anchor
# ============================================================================


class TestRAdjustParity:
    def test_bonferroni_matches_r(self) -> None:
        r = bonferroni_correct(_P5, alpha=0.05)
        assert list(r.adjusted) == pytest.approx([0.05, 0.10, 0.15, 0.20, 0.25])

    def test_holm_matches_r(self) -> None:
        r = holm_bonferroni_correct(_P5, alpha=0.05)
        assert list(r.adjusted) == pytest.approx([0.05, 0.08, 0.09, 0.09, 0.09])

    def test_bh_matches_r(self) -> None:
        r = bh_fdr_correct(_P5, q=0.05)
        assert list(r.adjusted) == pytest.approx([0.05, 0.05, 0.05, 0.05, 0.05])


# ============================================================================
# TestRejection
# ============================================================================


class TestRejection:
    def test_bonferroni_rejects_only_smallest(self) -> None:
        r = bonferroni_correct(_P5, alpha=0.05)
        assert list(r.reject) == [True, False, False, False, False]
        assert r.n_reject == 1 and r.any_reject is True

    def test_bh_rejects_all_here(self) -> None:
        r = bh_fdr_correct(_P5, q=0.05)
        assert all(r.reject) and r.n_reject == 5

    def test_no_rejection_reports_false(self) -> None:
        r = bonferroni_correct([0.4, 0.5, 0.6], alpha=0.05)
        assert r.n_reject == 0 and r.any_reject is False


# ============================================================================
# TestInputOrderPreservation
# ============================================================================


class TestInputOrderPreservation:
    def test_unsorted_maps_back(self) -> None:
        # permutation of _P5: [0.04, 0.01, 0.03, 0.05, 0.02]
        r = bonferroni_correct([0.04, 0.01, 0.03, 0.05, 0.02], alpha=0.05)
        assert list(r.adjusted) == pytest.approx([0.20, 0.05, 0.15, 0.25, 0.10])
        assert list(r.reject) == [False, True, False, False, False]

    def test_holm_unsorted_maps_back(self) -> None:
        r = holm_bonferroni_correct([0.04, 0.01, 0.03, 0.05, 0.02], alpha=0.05)
        assert list(r.adjusted) == pytest.approx([0.09, 0.05, 0.09, 0.09, 0.08])


# ============================================================================
# TestRelativePower - bonferroni <= holm <= bh_fdr
# ============================================================================


class TestRelativePower:
    def test_rejection_counts_nested(self) -> None:
        p = [0.001, 0.013, 0.021, 0.030, 0.067, 0.20]
        n_bonf = bonferroni_correct(p, alpha=0.05).n_reject
        n_holm = holm_bonferroni_correct(p, alpha=0.05).n_reject
        n_bh = bh_fdr_correct(p, q=0.05).n_reject
        assert n_bonf <= n_holm <= n_bh

    def test_holm_never_more_conservative_than_bonferroni(self) -> None:
        p = [0.001, 0.013, 0.021, 0.030, 0.067, 0.20]
        bonf = bonferroni_correct(p, alpha=0.05).adjusted
        holm = holm_bonferroni_correct(p, alpha=0.05).adjusted
        assert all(h <= b + 1e-12 for h, b in zip(holm, bonf, strict=True))


# ============================================================================
# TestMonotonicity - adjusted p-values are monotone in sorted-p order
# ============================================================================


class TestMonotonicity:
    @pytest.mark.parametrize("fn", [holm_bonferroni_correct, bh_fdr_correct])
    def test_adjusted_non_decreasing_in_p_order(self, fn) -> None:  # type: ignore[no-untyped-def]
        p = [0.04, 0.01, 0.03, 0.05, 0.02]
        r = fn(p)
        by_p = [adj for _, adj in sorted(zip(r.p_values, r.adjusted, strict=True))]
        assert all(by_p[i] <= by_p[i + 1] + 1e-12 for i in range(len(by_p) - 1))


# ============================================================================
# TestEdge + TestDeterminism
# ============================================================================


class TestEdge:
    def test_single_p_value_unchanged(self) -> None:
        # N=1: every correction leaves the p-value as-is.
        assert bonferroni_correct([0.03]).adjusted[0] == pytest.approx(0.03)
        assert holm_bonferroni_correct([0.03]).adjusted[0] == pytest.approx(0.03)
        assert bh_fdr_correct([0.03]).adjusted[0] == pytest.approx(0.03)

    def test_adjusted_clamped_to_one(self) -> None:
        r = bonferroni_correct([0.5, 0.6, 0.7], alpha=0.05)  # 3 * 0.5 = 1.5 -> clamp to 1.0
        assert all(a <= 1.0 for a in r.adjusted)


class TestDeterminism:
    def test_pure_function(self) -> None:
        assert bh_fdr_correct(_P5) == bh_fdr_correct(_P5)


# ============================================================================
# TestErrors + TestFrozen
# ============================================================================


class TestErrors:
    def test_empty_raises(self) -> None:
        with pytest.raises(MultipleTestError, match="at least one"):
            bonferroni_correct([])

    def test_p_above_one_raises(self) -> None:
        with pytest.raises(MultipleTestError, match=r"\[0, 1\]"):
            bonferroni_correct([0.5, 1.5])

    def test_p_negative_raises(self) -> None:
        with pytest.raises(MultipleTestError, match=r"\[0, 1\]"):
            holm_bonferroni_correct([0.5, -0.1])

    def test_non_finite_raises(self) -> None:
        with pytest.raises(MultipleTestError, match=r"\[0, 1\]"):
            bh_fdr_correct([0.5, float("nan")])

    def test_bad_alpha_raises(self) -> None:
        with pytest.raises(MultipleTestError, match="alpha must be in"):
            bonferroni_correct([0.1, 0.2], alpha=1.0)

    def test_bad_q_raises(self) -> None:
        with pytest.raises(MultipleTestError, match="q must be in"):
            bh_fdr_correct([0.1, 0.2], q=0.0)


class TestFrozen:
    def test_result_frozen(self) -> None:
        r = bonferroni_correct(_P5)
        assert isinstance(r, MultipleTestResult)
        with pytest.raises(AttributeError):
            r.n_reject = 0  # type: ignore[misc]
