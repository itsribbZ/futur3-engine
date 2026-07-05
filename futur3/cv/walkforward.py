"""futur3.cv.walkforward - anchored / rolling walk-forward fold generation (§4.2).

Walk-forward is the simplest time-series-aware CV: train on the past, validate on the strictly-later
future, advance, repeat. The validation window is ALWAYS after the training window, so there is no
look-ahead by construction (the bug-class-4 "look-ahead bias" the engine guards). Two variants
(Pardo 2008):

  - "anchored" (expanding): train = [0, t); the train window GROWS each fold. Best for a stable
    regime - more history is monotonically better.
  - "rolling" (sliding): train = [t - W, t); a FIXED-width window that slides forward. Best under
    regime change - the model only sees the most-recent W observations.

This module generates the SPLITS (integer index ranges into the series), not a model-fit loop: the
hard, correctness-critical part is the strictly-causal partitioning; what runs inside each fold (the
existing BacktestEngine, or a future ML model) is the caller's concern. Pure stdlib, deterministic
(a pure function of the arguments). The load-bearing invariant: `val_start == train_end` for
every fold (validation begins exactly where training ends - never earlier).

NOTE - the §4.2 spec sketches pandas Timedelta windows + a model/predict callable; futur3 is stdlib
and bar-indexed, so windows are INTEGER counts and the caller maps time -> index. More general (the
caller owns the index<->time mapping) and exactly what the bar-indexed BacktestEngine needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

WalkForwardMode = Literal["anchored", "rolling"]

_VALID_MODES: Final[tuple[WalkForwardMode, ...]] = ("anchored", "rolling")
_MIN_SAMPLES: Final[int] = 2  # need at least one train + one validation observation


class WalkForwardError(Exception):
    """Invalid walk-forward configuration (bad mode / window / step, or windows too large for the
    series so that no fold can be formed)."""


@dataclass(frozen=True)
class WalkForwardFold:
    """One strictly-causal train/validation split, as half-open index ranges into the series.

    `val_start == train_end` always - validation begins exactly where training ends, so the fold is
    look-ahead-free by construction. `[train_start, train_end)` and `[val_start, val_end)` are the
    half-open index windows; the `*_indices` properties hand back the matching `range` objects.
    """

    fold: int  # 0-based fold number
    train_start: int
    train_end: int  # exclusive
    val_start: int  # always == train_end
    val_end: int  # exclusive

    @property
    def train_indices(self) -> range:
        return range(self.train_start, self.train_end)

    @property
    def val_indices(self) -> range:
        return range(self.val_start, self.val_end)

    @property
    def n_train(self) -> int:
        return self.train_end - self.train_start

    @property
    def n_val(self) -> int:
        return self.val_end - self.val_start


def _resolve_walk_forward(
    n_samples: int,
    val_window: int,
    mode: WalkForwardMode,
    train_window: int | None,
    step: int | None,
    min_train: int,
) -> tuple[int, int]:
    """Validate the arguments; return (step, first validation start). Raises on any contract
    violation (fail-loud: no silent fallback). Factored out to keep walk_forward_splits in budget."""
    if n_samples < _MIN_SAMPLES:
        raise WalkForwardError(f"n_samples must be >= {_MIN_SAMPLES}; got {n_samples}")
    if mode not in _VALID_MODES:
        raise WalkForwardError(f"mode must be one of {_VALID_MODES}; got {mode!r}")
    if val_window < 1:
        raise WalkForwardError(f"val_window must be >= 1; got {val_window}")
    if min_train < 1:
        raise WalkForwardError(f"min_train must be >= 1; got {min_train}")
    resolved_step = val_window if step is None else step
    if resolved_step < 1:
        raise WalkForwardError(f"step must be >= 1; got {step}")
    if mode == "rolling":
        if train_window is None:
            raise WalkForwardError("rolling mode requires train_window")
        if train_window < 1:
            raise WalkForwardError(f"train_window must be >= 1; got {train_window}")
        return resolved_step, train_window
    if train_window is not None:
        raise WalkForwardError("train_window is only used in rolling mode; omit it for anchored")
    return resolved_step, min_train


def walk_forward_splits(
    n_samples: int,
    *,
    val_window: int,
    mode: WalkForwardMode = "anchored",
    train_window: int | None = None,
    step: int | None = None,
    min_train: int = 1,
) -> list[WalkForwardFold]:
    """Generate strictly-causal walk-forward folds over `n_samples` observations.

    Args:
        n_samples: length of the series being split (>= 2).
        val_window: validation window size in observations (>= 1).
        mode: "anchored" (expanding train from 0) or "rolling" (fixed-width sliding train).
        train_window: REQUIRED for "rolling" (the fixed train width, >= 1); must be omitted for
            "anchored" (where the train window is the whole past, so a fixed width is meaningless).
        step: observations to advance the validation start each fold; default = `val_window`
            (back-to-back, non-overlapping validation windows).
        min_train: minimum training observations before a fold is emitted (default 1). Anchored
            folds with too little history are skipped until the train window reaches this.

    Returns:
        Folds in chronological order. Every fold satisfies `val_start == train_end`.

    Raises:
        WalkForwardError: bad mode, val_window/step/min_train < 1, n_samples < 2, train_window
            misuse (missing for rolling / supplied for anchored), or windows too large for a fold.
    """
    step, first_val = _resolve_walk_forward(
        n_samples, val_window, mode, train_window, step, min_train
    )

    folds: list[WalkForwardFold] = []
    val_start = first_val
    while val_start < n_samples:
        val_end = min(val_start + val_window, n_samples)
        if mode == "anchored":
            train_start = 0
        else:  # rolling: train_window is not None (validated above) - assert narrows it for mypy
            assert train_window is not None
            train_start = val_start - train_window
        if val_start - train_start >= min_train:  # train window large enough to emit
            folds.append(
                WalkForwardFold(
                    fold=len(folds),
                    train_start=train_start,
                    train_end=val_start,
                    val_start=val_start,
                    val_end=val_end,
                )
            )
        val_start += step

    if not folds:
        raise WalkForwardError(
            f"no fold could be formed (n_samples={n_samples}, val_window={val_window}, "
            f"train_window={train_window}, min_train={min_train}) - windows too large"
        )
    return folds


__all__: list[str] = [
    "WalkForwardError",
    "WalkForwardFold",
    "WalkForwardMode",
    "walk_forward_splits",
]
