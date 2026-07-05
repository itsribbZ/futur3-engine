"""futur3.stats._sharpe_core - shared kernel for the Sharpe-based validation gates.

PRIVATE module. The maths common to the Probabilistic Sharpe Ratio (PSR, gate G4) and the Deflated
Sharpe Ratio (DSR, gate G3) live here so the two public modules share ONE audited implementation of
the higher-moment correction + the expected-maximum-Sharpe term.

Convention (locked against the cited prior art):

- The PSR/DSR kernel operates on the PER-PERIOD Sharpe (mean/std at the observation frequency), NOT
  the annualized Sharpe. Mertens' (2002) standard-error formula is frequency-specific; feeding an
  annualized SR into the per-period kernel is a silent bug. Callers pass an ANNUALIZED benchmark /
  SR* and de-annualize it (/sqrt(periods_per_year)) before the kernel.
- `sr_per_period = mean / sample_std` (ddof=1).
- skewness `g3 = m3/m2**1.5` and kurtosis `g4 = m4/m2**2` use POPULATION central moments (/n). g4 is
  RAW kurtosis (3.0 for a Gaussian), NOT excess - the kernel's `(g4-1)/4` term assumes raw. Using
  excess kurtosis here understates the variance term and INFLATES DSR/PSR (a fake-alpha bug).
- `expected_max_sharpe` is the FULL Bailey-Lopez de Prado 2014 eq.5 with TRIAL VARIANCE
  (sigma * [(1-gamma)*Phi^-1(1-1/N) + gamma*Phi^-1(1-1/(N*e))]). NOTE: an earlier internal derivation note
  currently states a simplified `sqrt(2*logN)*(1-gamma+gamma/logN)` form that DROPS trial variance -
  that doc line disagrees with this (canonical) implementation + the cited prior art; flagged for
  correction, not followed here.

All probabilities use stdlib `statistics.NormalDist` for Phi (`.cdf`) and Phi^-1 (`.inv_cdf`) - no
scipy dependency (scipy is not in pyproject; `performance.py` is likewise pure-stdlib). Undefined
results surface as None (never a silent 0 / NaN-as-number) per the fail-loud policy.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Final

EULER_GAMMA: Final[float] = 0.5772156649015329  # Euler-Mascheroni constant

_MIN_RETURNS: Final[int] = 2  # need >= 2 returns to form a mean + sample std
_MIN_TRIALS_FOR_MAX: Final[int] = 2  # need >= 2 trials to estimate the trial-Sharpe variance
_NORMAL: Final[NormalDist] = NormalDist()  # standard normal; reused for Phi / Phi^-1


@dataclass(frozen=True)
class SharpeMoments:
    """Per-period Sharpe + the higher moments the PSR/DSR kernel needs.

    `kurt` is RAW kurtosis (3.0 for a Gaussian), not excess. `skew`/`kurt` use population central
    moments; `sr_per_period` uses the sample std (ddof=1) - matching the cited prior art exactly.
    """

    n: int
    sr_per_period: float
    skew: float
    kurt: float


def compute_sharpe_moments(returns: Sequence[float]) -> SharpeMoments | None:
    """Per-period Sharpe + sample skew/kurtosis, or None if undefined (n<2 or zero variance).

    Returns are taken as-is (may be negative); the caller is responsible for finiteness.
    """
    n = len(returns)
    if n < _MIN_RETURNS:
        return None
    mean = statistics.fmean(returns)
    sd = statistics.stdev(returns)  # sample std, ddof=1 - matches the cited SR_hat convention
    if sd <= 0.0 or not math.isfinite(sd):
        return None  # zero / non-finite variance -> Sharpe undefined (fail-loud: None, not 0)
    sr_per_period = mean / sd
    # Population central moments (/n) for skew/kurtosis - the Bailey-Lopez de Prado convention.
    m2 = math.fsum((x - mean) ** 2 for x in returns) / n
    if m2 <= 0.0:
        return None
    m3 = math.fsum((x - mean) ** 3 for x in returns) / n
    m4 = math.fsum((x - mean) ** 4 for x in returns) / n
    skew = m3 / m2**1.5
    kurt = m4 / m2**2  # RAW kurtosis (Gaussian -> 3.0)
    return SharpeMoments(n=n, sr_per_period=sr_per_period, skew=skew, kurt=kurt)


def psr_kernel(
    sr_per_period: float,
    sr_star_per_period: float,
    n: int,
    skew: float,
    kurt: float,
) -> float | None:
    """Shared PSR/DSR kernel: Phi((SR - SR*) * sqrt(n-1) / sqrt(1 - g3*SR + (g4-1)/4 * SR**2)).

    All Sharpe inputs are PER-PERIOD. Returns the probability the true (per-period) Sharpe exceeds
    `sr_star_per_period`, or None when the variance term is non-positive (extreme moments) - fail-loud.
    """
    denom_inner = 1.0 - skew * sr_per_period + (kurt - 1.0) / 4.0 * sr_per_period**2
    if denom_inner <= 0.0:
        return None
    z = (sr_per_period - sr_star_per_period) * math.sqrt(n - 1) / math.sqrt(denom_inner)
    return _NORMAL.cdf(z)


def expected_max_sharpe(n_trials: int, var_sr_trials: float) -> float:
    """E[max Sharpe] over `n_trials` under the no-skill null (Bailey-Lopez de Prado 2014 eq.5).

    `var_sr_trials` is the variance of the trial Sharpes; the result carries their annualization.
    Returns 0.0 when there is no selection bias to correct for (n_trials<2 or non-positive var) -
    i.e. the DSR then collapses to the unconditional PSR.
    """
    if n_trials < _MIN_TRIALS_FOR_MAX or var_sr_trials <= 0.0 or not math.isfinite(var_sr_trials):
        return 0.0
    sigma = math.sqrt(var_sr_trials)
    term1 = (1.0 - EULER_GAMMA) * _NORMAL.inv_cdf(1.0 - 1.0 / n_trials)
    term2 = EULER_GAMMA * _NORMAL.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return sigma * (term1 + term2)


def _newey_west_max_lag(n: int) -> int:
    """Newey-West (1994) automatic lag-truncation bandwidth floor(4 * (n/100)^(2/9)) (>= 1) — keeps
    the autocorrelation sum to the few lags a sample of n returns can actually estimate, so noisy
    high-order rho_k (each amplified by the (q-k) weight) do not swamp the variance-ratio."""
    return max(1, int(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def lo_autocorr_factor(
    returns: Sequence[float], q: int, *, max_lag: int | None = None
) -> float | None:
    """Lo (2002) autocorrelation-corrected annualization factor eta(q), replacing the naive sqrt(q).

    eta(q) = q / sqrt(q + 2 * sum_{k=1}^{q-1} (q - k) * rho_k), rho_k = lag-k autocorrelation of
    `returns`. For POSITIVE autocorrelation (e.g. a trend/momentum book) eta(q) < sqrt(q), so the
    Lo-corrected annualized Sharpe is LOWER than the IID one (conservative — it can only make DSR
    harder). Lags are capped at min(q-1, n-1, max_lag or the Newey-West automatic bandwidth) so the
    noisy high-order autocorrelations a short sample cannot estimate do not swamp the (q-k) sum.
    Returns None when undefined (n<2, zero variance, q<1, or an ill-conditioned
    denominator from strong negative autocorrelation) — fail-loud: never a silent fake number.
    """
    n = len(returns)
    if q < 1 or n < _MIN_RETURNS:
        return None
    mean = statistics.fmean(returns)
    var = math.fsum((x - mean) ** 2 for x in returns) / n  # population variance
    if var <= 0.0 or not math.isfinite(var):
        return None
    cap = max_lag if max_lag is not None else _newey_west_max_lag(n)
    lag_cap = min(q - 1, n - 1, cap)
    if lag_cap < 1:
        return float(q) / math.sqrt(q)  # no lags to correct for -> IID sqrt(q)
    weighted = 0.0
    for k in range(1, lag_cap + 1):
        cov_k = math.fsum((returns[i] - mean) * (returns[i - k] - mean) for i in range(k, n)) / n
        weighted += (q - k) * (cov_k / var)
    denom = q + 2.0 * weighted
    if denom <= 0.0 or not math.isfinite(denom):
        return None  # ill-conditioned (strong negative autocorrelation) -> undefined
    return float(q) / math.sqrt(denom)


__all__: list[str] = [
    "EULER_GAMMA",
    "SharpeMoments",
    "compute_sharpe_moments",
    "expected_max_sharpe",
    "lo_autocorr_factor",
    "psr_kernel",
]
