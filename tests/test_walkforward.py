"""Walk-forward fold-generation test suite (futur3.cv.walkforward).

Load-bearing tests:
- CAUSALITY (both modes): every fold has `val_start == train_end`, and every training index is
  strictly less than every validation index. This is the no-look-ahead invariant the whole module
  exists to guarantee.
- ANCHORED: training window starts at 0 and GROWS fold to fold.
- ROLLING: training window is a FIXED width that slides forward.
- DETERMINISM: a pure function of its arguments - identical args give an identical fold list.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from futur3.cv import WalkForwardError, WalkForwardFold, walk_forward_splits

# ============================================================================
# TestAnchored
# ============================================================================


class TestAnchored:
    def test_train_starts_at_zero_and_grows(self) -> None:
        folds = walk_forward_splits(100, val_window=20, min_train=20)  # anchored is the default
        assert [f.train_start for f in folds] == [0, 0, 0, 0]
        assert [f.n_train for f in folds] == [20, 40, 60, 80]  # expanding window
        assert [f.fold for f in folds] == [0, 1, 2, 3]

    def test_tiles_the_series(self) -> None:
        folds = walk_forward_splits(100, val_window=20, min_train=20)
        windows = [(f.val_start, f.val_end) for f in folds]
        assert windows == [(20, 40), (40, 60), (60, 80), (80, 100)]
        assert folds[-1].val_end == 100

    def test_partial_final_validation_window(self) -> None:
        # 95 samples, val_window 20 from start 20 -> last window [80, 95) is short, not dropped.
        folds = walk_forward_splits(95, val_window=20, min_train=20)
        assert folds[-1].val_end == 95
        assert folds[-1].n_val == 15


# ============================================================================
# TestRolling
# ============================================================================


class TestRolling:
    def test_fixed_width_train_slides(self) -> None:
        folds = walk_forward_splits(100, val_window=10, mode="rolling", train_window=30)
        assert all(f.n_train == 30 for f in folds)  # fixed width
        assert [f.train_start for f in folds][:3] == [0, 10, 20]  # slides forward by step
        assert folds[0].val_start == 30  # first validation begins after the first full window

    def test_step_controls_advance(self) -> None:
        folds = walk_forward_splits(100, val_window=10, mode="rolling", train_window=20, step=5)
        assert [f.val_start for f in folds][:3] == [20, 25, 30]


# ============================================================================
# TestCausality - the no-look-ahead invariant
# ============================================================================


class TestCausality:
    @pytest.mark.parametrize(
        ("mode", "train_window"),
        [("anchored", None), ("rolling", 25)],
    )
    def test_validation_strictly_after_training(self, mode: str, train_window: int | None) -> None:
        folds = walk_forward_splits(
            120,
            val_window=15,
            mode=mode,
            train_window=train_window,
            min_train=15,  # type: ignore[arg-type]
        )
        for f in folds:
            assert f.val_start == f.train_end  # validation begins exactly where training ends
            assert max(f.train_indices) < min(f.val_indices)  # no train index leaks forward


# ============================================================================
# TestStep - overlap / gap control
# ============================================================================


class TestStep:
    def test_default_step_non_overlapping(self) -> None:
        folds = walk_forward_splits(100, val_window=20, min_train=20)  # step defaults to val_window
        val_ranges = [(f.val_start, f.val_end) for f in folds]
        for (_, end), (nxt_start, _) in pairwise(val_ranges):
            assert end == nxt_start  # back-to-back, no overlap, no gap

    def test_small_step_overlaps(self) -> None:
        folds = walk_forward_splits(100, val_window=20, step=10, min_train=20)
        assert folds[0].val_start == 20 and folds[1].val_start == 30  # windows overlap by 10


# ============================================================================
# TestMinTrain
# ============================================================================


class TestMinTrain:
    def test_anchored_first_fold_respects_min_train(self) -> None:
        folds = walk_forward_splits(100, val_window=10, min_train=40)
        assert folds[0].n_train == 40  # first fold has exactly min_train history

    def test_rolling_below_min_train_raises(self) -> None:
        # rolling train width 10 < min_train 20 -> every fold is too small -> none formed.
        with pytest.raises(WalkForwardError, match="no fold"):
            walk_forward_splits(100, val_window=10, mode="rolling", train_window=10, min_train=20)


# ============================================================================
# TestDeterminism
# ============================================================================


class TestDeterminism:
    def test_pure_function(self) -> None:
        a = walk_forward_splits(80, val_window=10, mode="rolling", train_window=20)
        b = walk_forward_splits(80, val_window=10, mode="rolling", train_window=20)
        assert a == b


# ============================================================================
# TestFoldProperties
# ============================================================================


class TestFoldProperties:
    def test_index_ranges_and_counts(self) -> None:
        f = walk_forward_splits(50, val_window=10, min_train=20)[0]
        assert f.train_indices == range(0, 20)
        assert f.val_indices == range(20, 30)
        assert f.n_train == 20 and f.n_val == 10


# ============================================================================
# TestErrors
# ============================================================================


class TestErrors:
    def test_bad_mode_raises(self) -> None:
        with pytest.raises(WalkForwardError, match="mode must be one of"):
            walk_forward_splits(50, val_window=10, mode="bogus")  # type: ignore[arg-type]

    def test_val_window_too_small_raises(self) -> None:
        with pytest.raises(WalkForwardError, match="val_window must be"):
            walk_forward_splits(50, val_window=0)

    def test_step_too_small_raises(self) -> None:
        with pytest.raises(WalkForwardError, match="step must be"):
            walk_forward_splits(50, val_window=10, step=0)

    def test_min_train_too_small_raises(self) -> None:
        with pytest.raises(WalkForwardError, match="min_train must be"):
            walk_forward_splits(50, val_window=10, min_train=0)

    def test_n_samples_too_small_raises(self) -> None:
        with pytest.raises(WalkForwardError, match="n_samples must be"):
            walk_forward_splits(1, val_window=1)

    def test_rolling_without_train_window_raises(self) -> None:
        with pytest.raises(WalkForwardError, match="rolling mode requires train_window"):
            walk_forward_splits(50, val_window=10, mode="rolling")

    def test_anchored_with_train_window_raises(self) -> None:
        with pytest.raises(WalkForwardError, match="only used in rolling"):
            walk_forward_splits(50, val_window=10, mode="anchored", train_window=20)

    def test_windows_too_large_raises(self) -> None:
        # rolling train_window exceeds the series -> first validation start is past the end.
        with pytest.raises(WalkForwardError, match="no fold"):
            walk_forward_splits(10, val_window=5, mode="rolling", train_window=20)


# ============================================================================
# TestFrozen
# ============================================================================


class TestFrozen:
    def test_fold_frozen(self) -> None:
        f = walk_forward_splits(50, val_window=10, min_train=20)[0]
        assert isinstance(f, WalkForwardFold)
        with pytest.raises(AttributeError):
            f.train_start = 5  # type: ignore[misc]
