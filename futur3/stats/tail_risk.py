"""futur3.stats.tail_risk - tail-asymmetry / sizing-risk diagnostics of a return series.

The companion to `robustness.py` (which asks "does the Sharpe survive winsorizing extremes +
cost?"). This module asks the SIZING question: which way does the tail lean, and is that lean
STRUCTURAL or a few-outlier artifact? It is decision-load-bearing -- a positive skew / right-leaning
tail is what makes a long risk premium safe to size UP; a hidden left tail (skew that goes negative
in crises) is the Lemperiere-Bouchaud trap (Sharpe ~ -skew) earlier win-rate work already burned us
on. So the numbers here gate how aggressively a sleeve may be sized, and they are unit-tested
against hand-computed values per the zero-bugs floor.

Four primitives, pure stdlib, float-domain, fail-loud (None when undefined -- never a silent NaN-number):

- `skewness` -- population skew g1 = m3/m2**1.5, via `_sharpe_core.compute_sharpe_moments` so there
  is ONE audited skew in the codebase (the same one PSR/DSR use). The HEADLINE moment, but
  3rd-moment skew is outlier-DOMINATED + has huge sampling error -- never read it alone.
- `skewness_excluding_extremes` -- the falsification: drop the `n_drop` observations with the
  largest |deviation from the mean| (the points that dominate m3) and recompute. If a big positive
  skew COLLAPSES toward 0 after dropping 1-3 days, it was an outlier artifact, not structure.
- `quantile_skewness` -- the Groeneveld-Meeden / Bowley asymmetry at tail prob `tail`:
  (Q(1-tail) + Q(tail) - 2*median) / (Q(1-tail) - Q(tail)), bounded in [-1, 1], moment-free +
  outlier-resistant. tail=0.25 is the classic Bowley quartile skew (the body); tail=0.05 reads the
  deep tails. The STABLE asymmetry read the moment-skew cannot give.
- `tail_means` -- the sizing-relevant magnitudes: mean of the worst `tail`-fraction (left, an
  empirical expected-shortfall / CVaR -- a negative number for a loss tail) vs mean of the best
  `tail`-fraction (right). `.ratio` = right / |left| > 1 <=> the up-tail dominates the down-tail
  <=> genuinely tail-safe for a long.

Deterministic; annualization-free (pure distribution shape, no periods_per_year).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from futur3.stats._sharpe_core import compute_sharpe_moments

_MIN_RETURNS: Final[int] = 2
_MAX_TAIL: Final[float] = 0.5  # tail prob must be < 0.5 (at 0.5 the quantile spread collapses to 0)


class TailRiskError(Exception):
    """Invalid tail-risk input (negative drop count, or a tail fraction outside (0, 0.5))."""


@dataclass(frozen=True)
class TailMeans:
    """Empirical expected-shortfall pair: the mean of the worst-`tail` (`left_mean`, signed -- a
    negative number for a loss tail) and best-`tail` (`right_mean`) `n_tail` observations each."""

    left_mean: float
    right_mean: float
    n_tail: int  # observations averaged in EACH tail = ceil(tail * n)

    @property
    def ratio(self) -> float:
        """right / |left| -- > 1 the up-tail dominates (tail-safe to size a long), < 1 the down-tail
        dominates (dangerous). inf when there is an up-tail but no down loss; nan when neither tail
        carries magnitude (degenerate, all-flat)."""
        denom = abs(self.left_mean)
        if denom == 0.0:
            return math.inf if self.right_mean > 0.0 else math.nan
        return self.right_mean / denom


def skewness(returns: Sequence[float]) -> float | None:
    """Population skew g1 = m3 / m2**1.5 (the SAME moment PSR/DSR use, via `_sharpe_core`). None if
    n<2 or zero variance (fail-loud). NOTE: outlier-dominated + high sampling error -- pair it with
    `skewness_excluding_extremes` (robustness) and `quantile_skewness` (the stable read)."""
    moments = compute_sharpe_moments(returns)
    return None if moments is None else moments.skew


def skewness_excluding_extremes(returns: Sequence[float], n_drop: int) -> float | None:
    """Skew after removing the `n_drop` observations with the largest |deviation from the mean| (the
    points that dominate the 3rd moment). `n_drop=0` == `skewness`. None if fewer than 2 points
    survive the drop / the survivors have zero variance. A big positive skew that collapses toward 0
    here was an outlier artifact, not structure.

    Raises:
        TailRiskError: n_drop < 0.
    """
    if n_drop < 0:
        raise TailRiskError(f"n_drop must be >= 0; got {n_drop}")
    if len(returns) - n_drop < _MIN_RETURNS:
        return None
    if n_drop == 0:
        return skewness(returns)
    mean = statistics.fmean(returns)
    # Drop the n_drop indices with the largest |x - mean| (stable: ties keep original order).
    by_extremity = sorted(range(len(returns)), key=lambda i: abs(returns[i] - mean), reverse=True)
    dropped = set(by_extremity[:n_drop])
    kept = [returns[i] for i in range(len(returns)) if i not in dropped]
    return skewness(kept)


def _quantile(ordered: Sequence[float], p: float) -> float:
    """Linear-interpolation (type-7 -- the numpy/R + `statistics.quantiles(method='inclusive')`
    default) quantile of an ALREADY-SORTED sequence; 0 <= p <= 1, len >= 1."""
    n = len(ordered)
    if n == 1:
        return ordered[0]
    h = (n - 1) * p
    lo = math.floor(h)
    hi = min(lo + 1, n - 1)
    return ordered[lo] + (h - lo) * (ordered[hi] - ordered[lo])


def quantile_skewness(returns: Sequence[float], *, tail: float = 0.05) -> float | None:
    """Groeneveld-Meeden / Bowley robust asymmetry at tail prob `tail` in (0, 0.5):
    (Q(1-tail) + Q(tail) - 2*median) / (Q(1-tail) - Q(tail)), in [-1, 1]. tail=0.25 == the classic
    Bowley quartile skew (body asymmetry); tail=0.05 reads the deep tails. Moment-free +
    outlier-resistant -- the STABLE asymmetry read. None if n<2 or the quantile spread is 0
    (degenerate).

    Raises:
        TailRiskError: tail not in (0, 0.5).
    """
    if not 0.0 < tail < _MAX_TAIL:
        raise TailRiskError(f"tail must be in (0, {_MAX_TAIL}); got {tail}")
    if len(returns) < _MIN_RETURNS:
        return None
    ordered = sorted(returns)
    q_lo = _quantile(ordered, tail)
    q_mid = _quantile(ordered, 0.5)
    q_hi = _quantile(ordered, 1.0 - tail)
    spread = q_hi - q_lo
    if spread <= 0.0:
        return None
    return (q_hi + q_lo - 2.0 * q_mid) / spread


def tail_means(returns: Sequence[float], *, tail: float = 0.05) -> TailMeans | None:
    """The worst-`tail` mean (left, signed) and best-`tail` mean (right) -- an empirical CVaR pair,
    each averaged over ceil(tail*n) observations. None if n<2. Use `.ratio` (right/|left|) as the
    sizing-safety read: > 1 the up-tail dominates.

    Raises:
        TailRiskError: tail not in (0, 0.5).
    """
    if not 0.0 < tail < _MAX_TAIL:
        raise TailRiskError(f"tail must be in (0, {_MAX_TAIL}); got {tail}")
    n = len(returns)
    if n < _MIN_RETURNS:
        return None
    ordered = sorted(returns)
    # round off float noise (e.g. 0.05*2000 -> 100.00000000000001) before ceil so the tail count
    # never jumps by one on an exact-integer boundary.
    k = max(1, math.ceil(round(tail * n, 9)))
    return TailMeans(
        left_mean=statistics.fmean(ordered[:k]),
        right_mean=statistics.fmean(ordered[-k:]),
        n_tail=k,
    )


__all__: list[str] = [
    "TailMeans",
    "TailRiskError",
    "quantile_skewness",
    "skewness",
    "skewness_excluding_extremes",
    "tail_means",
]
