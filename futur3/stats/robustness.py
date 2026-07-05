"""futur3.stats.robustness - outlier-robustness + cost-sensitivity of a return series.

Two pre-registered kill-criteria for an intraday sleeve (the pre-registration):

- TRIM / robustness: a Sharpe carried by a few extreme observations is fragile (Raviv's market-
  intraday-momentum critique; the cardinal failure mode of a fat-tailed edge). `winsorized_sharpe`
  recomputes the annualized Sharpe with the most extreme `trim_frac` tails CAPPED (winsorized, n
  preserved); `tail_contribution` reports the fraction of gross |return| from the most extreme
  observations. If the Sharpe collapses under a 1-5% winsorization, the edge is outlier-driven.

- COST: an intraday edge lives or dies on transaction cost. `net_returns` subtracts a per-trade cost
  (in return units, e.g. 0.00008 = 0.8 bps) from the days the strategy actually traded.

Pure stdlib, float-domain (ratios, per `stats.performance`'s convention). Annualization via the
caller's `periods_per_year`. Deterministic.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from typing import Final

_MIN_RETURNS: Final[int] = 2
_MAX_TRIM_FRAC: Final[float] = 0.5  # winsorizing >= 50% per tail would collapse to the median


class RobustnessError(Exception):
    """Invalid robustness input (bad trim/contribution fraction, ppy, or mismatched lengths)."""


def _annualized_sharpe(returns: Sequence[float], periods_per_year: float) -> float | None:
    """Simple annualized Sharpe (mean / stdev * sqrt(ppy)); None if < 2 points or zero variance."""
    if len(returns) < _MIN_RETURNS:
        return None
    sd = statistics.stdev(returns)
    if sd == 0.0:
        return None
    return statistics.mean(returns) / sd * math.sqrt(periods_per_year)


def winsorized_sharpe(
    returns: Sequence[float], trim_frac: float, *, periods_per_year: float = 252.0
) -> float | None:
    """Annualized Sharpe with the most extreme `trim_frac` at EACH tail winsorized (capped, not
    removed -> n preserved). `trim_frac=0` is the raw Sharpe. None if < 2 points / zero variance."""
    if not 0.0 <= trim_frac < _MAX_TRIM_FRAC:
        raise RobustnessError(f"trim_frac must be in [0, {_MAX_TRIM_FRAC}); got {trim_frac}")
    if periods_per_year <= 0:
        raise RobustnessError(f"periods_per_year must be > 0; got {periods_per_year}")
    if len(returns) < _MIN_RETURNS:
        return None
    ordered = sorted(returns)
    k = int(trim_frac * len(ordered))
    if k == 0:
        return _annualized_sharpe(returns, periods_per_year)
    lo, hi = ordered[k], ordered[len(ordered) - 1 - k]
    return _annualized_sharpe([min(max(x, lo), hi) for x in returns], periods_per_year)


def tail_contribution(returns: Sequence[float], frac: float) -> float:
    """Fraction of total ABSOLUTE return from the most extreme `frac` of observations (by |return|).
    ~`frac` => no concentration (uniform); >> `frac` => outlier-driven. 0.0 if all flat."""
    if not 0.0 < frac < 1.0:
        raise RobustnessError(f"frac must be in (0, 1); got {frac}")
    total = math.fsum(abs(x) for x in returns)
    if total == 0.0:
        return 0.0
    k = max(1, int(frac * len(returns)))
    top = sorted((abs(x) for x in returns), reverse=True)[:k]
    return math.fsum(top) / total


def net_returns(
    returns: Sequence[float], traded: Sequence[bool], cost_per_trade: float
) -> tuple[float, ...]:
    """Subtract `cost_per_trade` (return units; e.g. 0.00008 = 0.8 bps) from each day the strategy
    traded; flat days are unchanged. Models the round-trip cost of the once-a-day overnight bet."""
    if len(returns) != len(traded):
        raise RobustnessError(f"returns/traded length mismatch: {len(returns)} != {len(traded)}")
    if cost_per_trade < 0.0:
        raise RobustnessError(f"cost_per_trade must be >= 0; got {cost_per_trade}")
    return tuple(r - cost_per_trade if t else r for r, t in zip(returns, traded, strict=True))


__all__ = [
    "RobustnessError",
    "net_returns",
    "tail_contribution",
    "winsorized_sharpe",
]
