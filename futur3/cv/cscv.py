"""futur3.cv.cscv - Combinatorially Symmetric CV -> Probability of Backtest Overfitting.

The Probability of Backtest Overfitting gate. Bailey, Borwein, Lopez de Prado &
Zhu (2017), "The Probability of Backtest Overfitting", J. Computational Finance 20(4). The question:
you tested M strategy variants and picked the in-sample best; what is the probability it lands
below the median out-of-sample? That is the PBO. PBO >= 0.5 means selection is no better than
random (the backtest is overfit). Gate: PBO < 0.5.

Algorithm (input: M strategies x T per-period returns, sharing one time axis):
  1. Split the T-length time axis into S equal segments (S even; trailing remainder dropped).
  2. For every C(S, S/2) symmetric split into IS (chosen S/2 segments) + OOS (the other S/2):
       a. metric per strategy on IS; the IS winner n* = argmax (ties -> lowest index).
       b. midrank of n*'s OOS metric over all M -> relative rank w = rank/(M+1) in (0,1).
       c. logit lambda = log(w / (1-w)); lambda < 0 <=> n* is below the OOS median.
  3. PBO = fraction of splits with lambda < 0.
"Symmetric" = every observation is IS in exactly half the splits and OOS in the other half, which
keeps the OOS estimate look-ahead-free even though IS selection is data-driven.

Conventions: pure stdlib; FULLY DETERMINISTIC (enumerates every split - no RNG, so determinism holds by
construction). PBO is an order-statistic of the metric, so it is INVARIANT to annualizing Sharpe
(a positive scale shared by all strategies preserves the argmax and rank); the default metric is
therefore the cheap per-period Sharpe. Fail-loud: contract violations (odd S, < 2 strategies, ragged /
non-finite input, too little data for S) RAISE rather than return a misleading number. The `metric`
is pluggable (any pure `Callable[[Sequence[float]], float]`).

NOTE - all-identical strategies give PBO = 0, NOT 0.5: the winner sits exactly AT the OOS median
every split (w = 0.5, lambda = 0) and `lambda < 0` is strict, so nothing counts. (A predecessor implementation's docstring said "0.5" while its code returned 0 - a loose gloss.)
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Final

PBO_THRESHOLD: Final[float] = 0.5  # PBO (Bailey et al. 2015): must be strictly below this

_DEFAULT_N_SEGMENTS: Final[int] = 16  # S (Bailey-LdP). C(16,8)=12870; lower for short series
_DEFAULT_MIN_OBS_PER_SEGMENT: Final[int] = 2  # need >= 2 obs/segment for a sample std
_MIN_STRATEGIES: Final[int] = 2
_MIN_SEGMENTS: Final[int] = 2  # smallest even split
_MIN_OBS_FOR_STATISTIC: Final[int] = 2  # >= 2 points for a sample std / Pearson correlation
_LOGIT_EPS: Final[float] = 1e-9  # clamps the relative rank off {0, 1} so the logit stays finite


class CSCVError(Exception):
    """Invalid CSCV input (odd / too-small n_segments, < 2 strategies, ragged or non-finite returns,
    or too few observations for the requested number of segments)."""


@dataclass(frozen=True)
class CSCVResult:
    """CSCV / PBO verdict. Frozen + verbose for audit (diagnostics are features by design).
    `logits` holds every per-split logit; `is_oos_correlation` is the cross-split corr(IS, OOS) of
    the per-strategy metric (positive = skill carries over; negative = overfit selection)."""

    pbo: float  # Probability of Backtest Overfitting in [0, 1]
    n_strategies: int
    n_observations: int  # observations actually used (after dropping the trailing remainder)
    n_segments: int
    n_combinations: int  # C(n_segments, n_segments/2)
    mean_logit: float
    median_logit: float
    is_oos_correlation: float
    logits: tuple[float, ...]

    @property
    def passes_pbo(self) -> bool:
        """Gate G11: PBO strictly below 0.5 (PBO == 0.5 fails - borderline counts as overfit)."""
        return self.pbo < PBO_THRESHOLD


def _per_period_sharpe(returns: Sequence[float]) -> float:
    """Per-period Sharpe (mean / sample-std). 0.0 for a degenerate fold (< 2 obs or zero variance).

    Annualization is intentionally omitted: a positive monotone scale shared by every strategy
    changes neither the IS argmax nor the OOS rank, so PBO is invariant to it.
    """
    if len(returns) < _MIN_OBS_FOR_STATISTIC:
        return 0.0
    sd = statistics.stdev(returns)
    if sd <= 0.0 or not math.isfinite(sd):
        return 0.0
    return statistics.fmean(returns) / sd


def _metric_over(
    metric: Callable[[Sequence[float]], float],
    strat_segments: Sequence[Sequence[float]],
    combo: Sequence[int],
) -> float:
    """Apply `metric` to one strategy's returns pooled over the segments named in `combo`."""
    pooled: list[float] = []
    for j in combo:
        pooled.extend(strat_segments[j])
    return metric(pooled)


def _argmax(values: Sequence[float]) -> int:
    """Index of the maximum; ties resolved to the lowest index (deterministic)."""
    best_i = 0
    best_v = values[0]
    for i in range(1, len(values)):
        if values[i] > best_v:
            best_v = values[i]
            best_i = i
    return best_i


def _relative_rank(metrics: Sequence[float], target: int, m: int) -> float:
    """Midrank of `metrics[target]` among all M, normalized to (0, 1): rank / (M+1).

    Midrank handles ties (the tied group gets its average rank), so an all-equal fold puts the
    target exactly at the median (w = 0.5).
    """
    target_v = metrics[target]
    below = sum(1 for v in metrics if v < target_v)
    equal = sum(1 for v in metrics if v == target_v)  # includes the target itself
    rank = below + (equal + 1) / 2.0  # average rank of the tied group
    return rank / (m + 1)


def _safe_logit(w: float) -> float:
    """logit(w) = log(w / (1-w)), with w clamped into (eps, 1-eps) so the result stays finite."""
    wc = min(max(w, _LOGIT_EPS), 1.0 - _LOGIT_EPS)
    return math.log(wc / (1.0 - wc))


def _correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson correlation; 0.0 when undefined (fewer than 2 points or a zero-variance series)."""
    if len(xs) < _MIN_OBS_FOR_STATISTIC:
        return 0.0
    try:
        return statistics.correlation(xs, ys)
    except statistics.StatisticsError:
        return 0.0


def cscv_pbo(
    strategy_returns: Sequence[Sequence[float]],
    *,
    n_segments: int = _DEFAULT_N_SEGMENTS,
    metric: Callable[[Sequence[float]], float] = _per_period_sharpe,
    min_obs_per_segment: int = _DEFAULT_MIN_OBS_PER_SEGMENT,
) -> CSCVResult:
    """Probability of Backtest Overfitting via CSCV (gate G11), per Bailey-Borwein-LdP-Zhu 2017.

    Args:
        strategy_returns: M strategies, each a length-T per-period return series on one time axis
            (the same N x T family matrix the WRC/SPA gates take). M >= 2; equal length each.
        n_segments: S, an even integer >= 2 (default 16). The time axis is split into S equal
            segments; a trailing remainder of T mod S observations is dropped. Lower S for short T.
        metric: pure scalar metric over a return series (default: per-period Sharpe). PBO is
            invariant to any positive monotone transform of it.
        min_obs_per_segment: min observations per segment (default 2 - the floor for a sample std).

    Returns:
        A CSCVResult; inspect `.pbo` and `.passes_pbo`.

    Raises:
        CSCVError: odd / too-small n_segments, < 2 strategies, ragged or non-finite returns, or
            T // n_segments < min_obs_per_segment.
    """
    n_strategies = len(strategy_returns)
    if n_strategies < _MIN_STRATEGIES:
        raise CSCVError(f"need >= {_MIN_STRATEGIES} strategies; got {n_strategies}")
    if n_segments < _MIN_SEGMENTS or n_segments % 2 != 0:
        raise CSCVError(f"n_segments must be an even integer >= 2; got {n_segments}")
    if min_obs_per_segment < 1:
        raise CSCVError(f"min_obs_per_segment must be >= 1; got {min_obs_per_segment}")
    t_obs = len(strategy_returns[0])
    if any(len(s) != t_obs for s in strategy_returns):
        raise CSCVError("all strategies must have the same number of observations")
    rows = [[float(v) for v in s] for s in strategy_returns]
    if not all(math.isfinite(v) for row in rows for v in row):
        raise CSCVError("strategy returns must all be finite (no NaN / inf)")
    seg_size = t_obs // n_segments
    if seg_size < min_obs_per_segment:
        raise CSCVError(
            f"each of the {n_segments} segments needs >= {min_obs_per_segment} obs; T={t_obs} "
            f"gives {seg_size} per segment (provide more data or lower n_segments)"
        )

    used_t = seg_size * n_segments
    # per-strategy segment slices over the shared time axis (trailing remainder dropped)
    segments: list[list[list[float]]] = [
        [row[j * seg_size : (j + 1) * seg_size] for j in range(n_segments)] for row in rows
    ]
    half = n_segments // 2

    logits: list[float] = []
    is_all: list[float] = []
    oos_all: list[float] = []
    for is_combo in combinations(range(n_segments), half):
        is_set = set(is_combo)
        oos_combo = [j for j in range(n_segments) if j not in is_set]
        is_metrics = [_metric_over(metric, segments[m], is_combo) for m in range(n_strategies)]
        oos_metrics = [_metric_over(metric, segments[m], oos_combo) for m in range(n_strategies)]
        # a custom metric may be non-finite on a degenerate fold; skip it (fail-loud: never rank a NaN)
        if not all(math.isfinite(v) for v in (*is_metrics, *oos_metrics)):
            continue
        winner = _argmax(is_metrics)
        w = _relative_rank(oos_metrics, winner, n_strategies)
        logits.append(_safe_logit(w))
        is_all.extend(is_metrics)
        oos_all.extend(oos_metrics)

    if not logits:
        raise CSCVError("no split produced a finite metric (check the metric and the data)")

    pbo = sum(1 for lam in logits if lam < 0.0) / len(logits)
    return CSCVResult(
        pbo=pbo,
        n_strategies=n_strategies,
        n_observations=used_t,
        n_segments=n_segments,
        n_combinations=math.comb(n_segments, half),
        mean_logit=statistics.fmean(logits),
        median_logit=statistics.median(logits),
        is_oos_correlation=_correlation(is_all, oos_all),
        logits=tuple(logits),
    )


__all__: list[str] = [
    "PBO_THRESHOLD",
    "CSCVError",
    "CSCVResult",
    "cscv_pbo",
]
