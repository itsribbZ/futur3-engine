"""futur3.stats.bootstrap_ci - BCa bootstrap confidence intervals (gate G5).

Efron (1987), "Better Bootstrap Confidence Intervals", JASA 82(397); Efron & Tibshirani (1993),
*An Introduction to the Bootstrap*, ch.14. The bias-corrected-and-accelerated (BCa) interval
corrects the plain percentile bootstrap for (a) median bias (z0) and (b) skewness of the statistic's
sampling distribution (the acceleration a, from the jackknife). Real futures return distributions
are always skewed (CL's 2020 negative tail; MBT/MET heavy skew), so futur3 ships BCa PROPER, not the
plain percentile variant (a predecessor implementation set a=0).

Promotion gate (BCa per Efron & Tibshirani 1993): the profit factor's BCa lower bound must exceed 1.0. Phase A
advisory -> Phase B/C hard-gate.

Acceleration sign (load-bearing): jackknife numerator `(j_bar - j_i)**3`, j_i = leave-one-out
estimate, j_bar = their mean - the standard Efron-Tibshirani 1993 eq.14.15. An earlier internal note
had this sign-flipped; corrected (latent - the cited percentile prior art set a=0, never
exercising the acceleration). A right-skewed statistic gives a > 0, widening the upper tail.

Design choices (locked):
- Pure stdlib: `random.Random(seed)` for resampling (bit-reproducible) and NormalDist for
  Phi / Phi^-1. No numpy/scipy (neither is a declared pyproject dep; matches the rest of stats/). A
  fixed seed yields byte-identical output across runs.
- The statistic is any `Callable[[Sequence[float]], float]`; `profit_factor` is the gate's statistic. NaN
  resamples are dropped (degenerate draw); +/-inf resamples are kept (a legitimately extreme
  statistic, e.g. profit factor with no losing trades). Undefined results -> NaN bounds + a `reason`
  per the fail-loud policy, never a misleading number.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Final

PF_LOWER_THRESHOLD: Final[float] = 1.0  # BCa profit-factor lower bound must exceed this

_NORMAL: Final[NormalDist] = NormalDist()
_MIN_DATA: Final[int] = 2  # need >= 2 points to resample + jackknife
_MIN_RESAMPLES: Final[int] = 1000  # below this a tail-percentile CI has no resolution
_DEFAULT_RESAMPLES: Final[int] = 10000
_DENOM_FLOOR: Final[float] = 1e-12  # guards the BCa percentile denominator against /0 at extremes


class BCaError(Exception):
    """Invalid BCa input (bad confidence, too few resamples, non-finite or too-small data)."""


@dataclass(frozen=True)
class BCaResult:
    """BCa confidence interval. `lower`/`upper` are NaN when the interval is undefined (with a
    `reason`): a non-finite point statistic, or too many degenerate (NaN) resamples."""

    point: float  # full-sample statistic
    lower: float
    upper: float
    z0: float  # bias correction (NaN if undefined)
    accel: float  # acceleration (jackknife skewness; NaN if undefined)
    confidence: float
    n_resamples: int
    n: int  # sample size
    seed: int | None
    reason: str | None  # why the interval is undefined; else None

    def lower_exceeds(self, threshold: float) -> bool:
        """True iff the BCa lower bound is defined and strictly exceeds `threshold` (e.g. G5 uses
        `result.lower_exceeds(PF_LOWER_THRESHOLD)` on a profit-factor CI)."""
        return math.isfinite(self.lower) and self.lower > threshold


def profit_factor(returns: Sequence[float]) -> float:
    """Profit factor = sum(gains) / |sum(losses)|. inf when there are gains but no losses; NaN when
    there is nothing to divide (no losses and no gains)."""
    gains = math.fsum(r for r in returns if r > 0.0)
    losses = math.fsum(-r for r in returns if r < 0.0)  # positive magnitude
    if losses == 0.0:
        return math.inf if gains > 0.0 else math.nan
    return gains / losses


def _quantile(sorted_vals: Sequence[float], p: float) -> float:
    """Linear-interpolation quantile (type 7, the numpy/R default) of an already-sorted sequence."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    h = (n - 1) * p
    lo = math.floor(h)
    hi = min(lo + 1, n - 1)
    frac = h - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


def _undefined(
    point: float, confidence: float, n_resamples: int, n: int, seed: int | None, reason: str
) -> BCaResult:
    return BCaResult(
        point=point,
        lower=math.nan,
        upper=math.nan,
        z0=math.nan,
        accel=math.nan,
        confidence=confidence,
        n_resamples=n_resamples,
        n=n,
        seed=seed,
        reason=reason,
    )


def bca_bootstrap(
    data: Sequence[float],
    statistic: Callable[[Sequence[float]], float],
    *,
    n_resamples: int = _DEFAULT_RESAMPLES,
    confidence: float = 0.95,
    seed: int | None = None,
) -> BCaResult:
    """BCa confidence interval for `statistic` evaluated on `data`.

    Args:
        data: sample (e.g. per-trade returns). Must be finite and have >= 2 points.
        statistic: maps a sample to a scalar (e.g. `profit_factor`, `statistics.fmean`).
        n_resamples: bootstrap resamples WITH REPLACEMENT (>= 1000; default 10000).
        confidence: two-sided confidence level in (0, 1); default 0.95.
        seed: RNG seed - pass an int for bit-reproducible output.

    Raises:
        BCaError: confidence not in (0,1), n_resamples < 1000, < 2 data points, or non-finite data.
    """
    if not 0.0 < confidence < 1.0:
        raise BCaError(f"confidence must be in (0, 1); got {confidence}")
    if n_resamples < _MIN_RESAMPLES:
        raise BCaError(f"n_resamples must be >= {_MIN_RESAMPLES}; got {n_resamples}")
    x = [float(v) for v in data]
    n = len(x)
    if n < _MIN_DATA:
        raise BCaError(f"need >= {_MIN_DATA} data points; got {n}")
    if not all(math.isfinite(v) for v in x):
        raise BCaError("data must all be finite (no NaN / inf)")

    point = statistic(x)
    if not math.isfinite(point):
        return _undefined(
            point,
            confidence,
            n_resamples,
            n,
            seed,
            "point statistic is not finite (e.g. profit factor with no losses)",
        )

    # --- bootstrap distribution (with replacement); drop NaN draws, keep +/-inf ---
    rng = random.Random(seed)
    boot = [
        theta
        for _ in range(n_resamples)
        if not math.isnan(theta := statistic([x[rng.randrange(n)] for _ in range(n)]))
    ]
    n_boot = len(boot)
    if n_boot < n_resamples // 2:
        n_bad = n_resamples - n_boot
        return _undefined(
            point,
            confidence,
            n_resamples,
            n,
            seed,
            f"too many degenerate resamples ({n_bad} of {n_resamples} were NaN)",
        )

    # --- z0 bias correction (continuity-clamped to avoid +/-inf at the extremes) ---
    n_less = sum(1 for t in boot if t < point)
    n_eq = sum(1 for t in boot if t == point)
    prop = (n_less + 0.5 * n_eq) / n_boot
    prop = min(max(prop, 0.5 / n_boot), 1.0 - 0.5 / n_boot)
    z0 = _NORMAL.inv_cdf(prop)

    # --- acceleration via jackknife (Efron-Tibshirani 1993 eq.14.15; numerator (j_bar - j_i)) ---
    jack = [statistic(x[:i] + x[i + 1 :]) for i in range(n)]
    if all(math.isfinite(j) for j in jack):
        j_bar = statistics.fmean(jack)
        diffs = [j_bar - j for j in jack]
        denom = math.fsum(d * d for d in diffs)
        accel = math.fsum(d**3 for d in diffs) / (6.0 * denom**1.5) if denom > 0.0 else 0.0
    else:
        accel = 0.0  # degenerate jackknife -> fall back to bias-corrected-only (a=0)

    # --- BCa-adjusted percentiles ---
    alpha = 1.0 - confidence
    boot_sorted = sorted(boot)

    def _adjusted(z_q: float) -> float:
        denom = 1.0 - accel * (z0 + z_q)
        if abs(denom) < _DENOM_FLOOR:
            denom = math.copysign(_DENOM_FLOOR, denom) if denom != 0.0 else _DENOM_FLOOR
        return _NORMAL.cdf(z0 + (z0 + z_q) / denom)

    lower = _quantile(boot_sorted, _adjusted(_NORMAL.inv_cdf(alpha / 2.0)))
    upper = _quantile(boot_sorted, _adjusted(_NORMAL.inv_cdf(1.0 - alpha / 2.0)))
    return BCaResult(
        point=point,
        lower=lower,
        upper=upper,
        z0=z0,
        accel=accel,
        confidence=confidence,
        n_resamples=n_resamples,
        n=n,
        seed=seed,
        reason=None,
    )


__all__: list[str] = [
    "PF_LOWER_THRESHOLD",
    "BCaError",
    "BCaResult",
    "bca_bootstrap",
    "profit_factor",
]
