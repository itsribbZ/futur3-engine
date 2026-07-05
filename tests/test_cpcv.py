"""CPCV test suite (futur3.cv.cpcv) - combinatorial purged cross-validation.

Load-bearing tests:
- PURGE/EMBARGO DIRECTION: with one middle test block, the purged indices sit immediately BEFORE
  it (backward) and the embargoed indices immediately AFTER it (forward). Reversing the two is the
  canonical CPCV bug (AFML eq. 7.5); this test fails loudly if they flip.
- NO LEAKAGE + EXHAUSTIVE PARTITION: train and test never intersect, and every index is exactly
  one of test / train / purged / embargoed (n_test + n_train + n_purged + n_embargoed == n_samples).
- COMBINATORICS: exactly C(n_blocks, k_test) folds; blocks tile the whole series (no data dropped).
- DETERMINISM: enumerates every combination (no RNG) -> a pure function of its arguments.
"""

from __future__ import annotations

import math

import pytest

from futur3.cv import CPCVError, CPCVFold, cpcv_splits

# ============================================================================
# TestCombinatorics + block tiling
# ============================================================================


class TestCombinatorics:
    def test_fold_count_is_n_choose_k(self) -> None:
        folds = cpcv_splits(100, n_blocks=10, k_test=2)
        assert len(folds) == math.comb(10, 2)  # 45
        assert [f.fold for f in folds] == list(range(45))

    def test_each_test_block_combo_unique(self) -> None:
        folds = cpcv_splits(100, n_blocks=8, k_test=2)
        combos = [f.test_blocks for f in folds]
        assert len(set(combos)) == len(combos)  # every fold a distinct block combination
        assert all(len(c) == 2 for c in combos)

    def test_blocks_tile_the_whole_series(self) -> None:
        # k_test=1, no purge/embargo -> the test sets are exactly the blocks and cover [0, n).
        folds = cpcv_splits(105, n_blocks=10, k_test=1, purge=0, embargo=0)
        covered = sorted(i for f in folds for i in f.test_idx)
        assert covered == list(range(105))  # remainder distributed, nothing dropped


# ============================================================================
# TestPurgeEmbargoDirection - the canonical-bug catcher
# ============================================================================


class TestPurgeEmbargoDirection:
    def test_purge_backward_embargo_forward(self) -> None:
        # 10 blocks of 10; test block 5 = [50, 60); purge=3 backward, embargo=2 forward.
        folds = cpcv_splits(100, n_blocks=10, k_test=1, purge=3, embargo=2)
        f = next(fold for fold in folds if fold.test_blocks == (5,))
        train = set(f.train_idx)
        assert 46 in train  # just before the purge zone -> kept
        assert {47, 48, 49}.isdisjoint(train)  # PURGED (backward of test start 50)
        assert all(i not in train for i in range(50, 60))  # the test block itself
        assert {60, 61}.isdisjoint(train)  # EMBARGOED (forward of test end 60)
        assert 62 in train  # just after the embargo zone -> kept
        assert f.n_purged == 3
        assert f.n_embargoed == 2

    def test_first_block_has_no_purge_underflow(self) -> None:
        # test block 0 = [0, 10): there is nothing before it to purge (no negative indices).
        folds = cpcv_splits(100, n_blocks=10, k_test=1, purge=5, embargo=3)
        f = next(fold for fold in folds if fold.test_blocks == (0,))
        assert f.n_purged == 0  # nothing precedes block 0
        assert f.n_embargoed == 3  # indices 10, 11, 12 embargoed

    def test_last_block_has_no_embargo_overflow(self) -> None:
        # test block 9 = [90, 100): there is nothing after it to embargo (no out-of-range indices).
        folds = cpcv_splits(100, n_blocks=10, k_test=1, purge=4, embargo=5)
        f = next(fold for fold in folds if fold.test_blocks == (9,))
        assert f.n_purged == 4  # indices 86..89 purged
        assert f.n_embargoed == 0  # nothing follows block 9


# ============================================================================
# TestNoLeakage + exhaustive partition
# ============================================================================


class TestNoLeakage:
    def test_train_test_disjoint(self) -> None:
        for f in cpcv_splits(120, n_blocks=8, k_test=3, purge=4, embargo=2):
            assert set(f.train_idx).isdisjoint(f.test_idx)

    def test_exhaustive_partition(self) -> None:
        n = 120
        for f in cpcv_splits(n, n_blocks=8, k_test=3, purge=4, embargo=2):
            assert f.n_test + f.n_train + f.n_purged + f.n_embargoed == n

    def test_train_indices_sorted(self) -> None:
        f = cpcv_splits(100, n_blocks=10, k_test=2, purge=3, embargo=2)[0]
        assert list(f.train_idx) == sorted(f.train_idx)


# ============================================================================
# TestPurgeZero - no purge/embargo recovers plain combinatorial CV
# ============================================================================


class TestPurgeZero:
    def test_no_purge_no_embargo(self) -> None:
        for f in cpcv_splits(100, n_blocks=10, k_test=2, purge=0, embargo=0):
            assert f.n_purged == 0 and f.n_embargoed == 0
            assert f.n_train + f.n_test == 100  # every non-test index is a training index


# ============================================================================
# TestDeterminism
# ============================================================================


class TestDeterminism:
    def test_pure_function(self) -> None:
        a = cpcv_splits(100, n_blocks=10, k_test=2, purge=5, embargo=2)
        b = cpcv_splits(100, n_blocks=10, k_test=2, purge=5, embargo=2)
        assert a == b


# ============================================================================
# TestErrors
# ============================================================================


class TestErrors:
    def test_too_few_blocks_raises(self) -> None:
        with pytest.raises(CPCVError, match="n_blocks must be"):
            cpcv_splits(100, n_blocks=1)

    def test_k_test_zero_raises(self) -> None:
        with pytest.raises(CPCVError, match=r"k_test must be in"):
            cpcv_splits(100, n_blocks=10, k_test=0)

    def test_k_test_all_blocks_raises(self) -> None:
        with pytest.raises(CPCVError, match=r"k_test must be in"):
            cpcv_splits(100, n_blocks=10, k_test=10)  # no training block left

    def test_negative_purge_raises(self) -> None:
        with pytest.raises(CPCVError, match="purge must be"):
            cpcv_splits(100, n_blocks=10, k_test=2, purge=-1)

    def test_negative_embargo_raises(self) -> None:
        with pytest.raises(CPCVError, match="embargo must be"):
            cpcv_splits(100, n_blocks=10, k_test=2, embargo=-1)

    def test_fewer_samples_than_blocks_raises(self) -> None:
        with pytest.raises(CPCVError, match="must be >= n_blocks"):
            cpcv_splits(5, n_blocks=10)


# ============================================================================
# TestFrozen
# ============================================================================


class TestFrozen:
    def test_fold_frozen(self) -> None:
        f = cpcv_splits(100, n_blocks=10, k_test=2)[0]
        assert isinstance(f, CPCVFold)
        with pytest.raises(AttributeError):
            f.fold = 99  # type: ignore[misc]
