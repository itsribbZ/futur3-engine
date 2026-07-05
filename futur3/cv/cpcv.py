"""futur3.cv.cpcv - Combinatorial Purged Cross-Validation (§4.3; AFML Ch.7/12).

López de Prado, *Advances in Financial Machine Learning* (2018), Ch.7 (purge/embargo) + Ch.12
(the combinatorial scheme). Walk-forward gives ONE path through history; CPCV gives C(N, k) paths -
split the time axis into N contiguous blocks and, for each choice of k blocks as test, train on the
remaining N-k. Two leakage guards at the block boundaries:

  - PURGE (backward of each test block): drop training obs whose forward LABEL window would overlap
    the test interval. A label at index i spans [i, i+purge]; if that reaches into a test block the
    training label "saw" test data, so drop train indices in [test_start - purge, test_start).
  - EMBARGO (forward of each test block): drop training obs in (test_end, test_end+embargo], because
    serial correlation lets the test outcome leak into the immediately-following labels.

The DIRECTION is load-bearing: purge extends BACKWARD from a test block's start, embargo extends
FORWARD from its end. Reversing them is the canonical CPCV bug (AFML eq. 7.5) - a test asserts it.

Like walkforward, this generates the SPLITS (integer index ranges), not a model-fit loop: the hard
part is the purged combinatorial partitioning. Pure stdlib, FULLY DETERMINISTIC (enumerates every
C(N, k) combination - no RNG - deterministic by construction). Counting matches the prior art: an index is purged OR (embargoed-not-purged), never both, so
n_test + n_train + n_purged + n_embargoed == n_samples for every fold.

NOTE - the original spec sketches a per-observation label-end horizon; this first cut takes a single
fixed `purge` window (uniform label horizon), equivalent to the prior art's label-overlap purge
when every label has the same length. A per-obs label_end array can follow for a labeled ML
pipeline. Windows are INTEGER observation counts (futur3 is bar-indexed stdlib, not pandas).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Final

_DEFAULT_N_BLOCKS: Final[int] = 10
_DEFAULT_K_TEST: Final[int] = 2
_DEFAULT_PURGE: Final[int] = 5
_DEFAULT_EMBARGO: Final[int] = 2
_MIN_BLOCKS: Final[int] = 2  # need at least one train + one test block


class CPCVError(Exception):
    """Invalid CPCV configuration (n_blocks < 2, k_test outside [1, n_blocks-1], negative purge /
    embargo, or fewer observations than blocks)."""


@dataclass(frozen=True)
class CPCVFold:
    """One purged combinatorial train/test split, as sorted index tuples into the series.

    `train_idx` is post-purge-and-embargo; `test_idx` is the union of the chosen test blocks. The
    counts are mutually exclusive (purge wins over embargo), so for every fold
    `n_test + n_train + n_purged + n_embargoed == n_samples`.
    """

    fold: int  # 0-based fold number
    test_blocks: tuple[int, ...]  # which block indices form the test set
    train_idx: tuple[int, ...]  # training indices after purge + embargo (sorted)
    test_idx: tuple[int, ...]  # test indices (sorted)
    n_purged: int  # training candidates dropped by purge (backward of a test block)
    n_embargoed: int  # training candidates dropped by embargo (forward of a test block)

    @property
    def n_train(self) -> int:
        return len(self.train_idx)

    @property
    def n_test(self) -> int:
        return len(self.test_idx)


def _make_blocks(n_samples: int, n_blocks: int) -> list[tuple[int, int]]:
    """Split [0, n_samples) into n_blocks contiguous half-open [start, end) blocks of (near-)equal
    size; the first `n_samples % n_blocks` blocks get one extra observation so ALL data is used."""
    base, remainder = divmod(n_samples, n_blocks)
    blocks: list[tuple[int, int]] = []
    start = 0
    for b in range(n_blocks):
        size = base + (1 if b < remainder else 0)
        blocks.append((start, start + size))
        start += size
    return blocks


def cpcv_splits(
    n_samples: int,
    *,
    n_blocks: int = _DEFAULT_N_BLOCKS,
    k_test: int = _DEFAULT_K_TEST,
    purge: int = _DEFAULT_PURGE,
    embargo: int = _DEFAULT_EMBARGO,
) -> list[CPCVFold]:
    """Generate the C(n_blocks, k_test) purged combinatorial folds over `n_samples` observations.

    Args:
        n_samples: length of the series being split (>= n_blocks).
        n_blocks: N contiguous time blocks (>= 2). Default 10.
        k_test: blocks used as the test set per fold; in [1, n_blocks - 1]. Default 2 -> C(10,2)=45.
        purge: observations dropped from training BEFORE each test block start (the label horizon).
        embargo: observations dropped from training AFTER each test block end (serial-corr guard).

    Returns:
        Folds in deterministic combination order. Every index is exactly one of test / train /
        purged / embargoed.

    Raises:
        CPCVError: n_blocks < 2, k_test outside [1, n_blocks-1], negative purge/embargo, or
            n_samples < n_blocks.
    """
    if n_blocks < _MIN_BLOCKS:
        raise CPCVError(f"n_blocks must be >= {_MIN_BLOCKS}; got {n_blocks}")
    if not 1 <= k_test < n_blocks:
        raise CPCVError(f"k_test must be in [1, n_blocks-1] = [1, {n_blocks - 1}]; got {k_test}")
    if purge < 0:
        raise CPCVError(f"purge must be >= 0; got {purge}")
    if embargo < 0:
        raise CPCVError(f"embargo must be >= 0; got {embargo}")
    if n_samples < n_blocks:
        raise CPCVError(f"n_samples ({n_samples}) must be >= n_blocks ({n_blocks})")

    blocks = _make_blocks(n_samples, n_blocks)
    folds: list[CPCVFold] = []
    for combo in combinations(range(n_blocks), k_test):
        test_intervals = [blocks[b] for b in combo]
        test_set: set[int] = set()
        purge_set: set[int] = set()
        embargo_set: set[int] = set()
        for ts, te in test_intervals:
            test_set |= set(range(ts, te))
            purge_set |= set(range(max(0, ts - purge), ts))  # BACKWARD of the test block start
            embargo_set |= set(range(te, min(n_samples, te + embargo)))  # FORWARD of the end

        train_idx: list[int] = []
        n_purged = 0
        n_embargoed = 0
        for i in range(n_samples):
            if i in test_set:
                continue
            if i in purge_set:  # purge takes precedence over embargo (no double-count)
                n_purged += 1
            elif i in embargo_set:
                n_embargoed += 1
            else:
                train_idx.append(i)

        folds.append(
            CPCVFold(
                fold=len(folds),
                test_blocks=tuple(combo),
                train_idx=tuple(train_idx),
                test_idx=tuple(sorted(test_set)),
                n_purged=n_purged,
                n_embargoed=n_embargoed,
            )
        )
    return folds


__all__: list[str] = [
    "CPCVError",
    "CPCVFold",
    "cpcv_splits",
]
