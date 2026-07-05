"""futur3.stats.probabilistic_sharpe - Probabilistic Sharpe Ratio (PSR, gate G4).

Bailey & Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier", J. of Risk 15(2). PSR is the
probability that a strategy's TRUE Sharpe exceeds a benchmark Sharpe, given the sample - correcting
the raw Sharpe for sample length + skew + (raw) kurtosis. It is the single-strategy sister of the
Deflated Sharpe Ratio (`deflated_sharpe.py`): DSR = PSR evaluated at SR* = the expected maximum
Sharpe over N trials (the selection-bias term). See `_sharpe_core` for the shared kernel + the
locked moment / annualization conventions.

Promotion gate — Probabilistic Sharpe Ratio (Bailey & Lopez de Prado 2012): PSR(0) > 0.95. Phase A advisory -> Phase C hard-gate. A raw Sharpe is
decorative; PSR is the unconditional "is this edge real?" answer for a single strategy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Final

from futur3.stats._sharpe_core import compute_sharpe_moments, lo_autocorr_factor, psr_kernel

PSR_THRESHOLD: Final[float] = 0.95  # PSR promotion threshold (Bailey & Lopez de Prado 2012)
_MIN_OBS_RELIABLE: Final[int] = 10  # below this, skew/kurtosis are too unstable to trust


class PSRError(Exception):
    """Invalid PSR input (bad periods_per_year or non-finite returns)."""


@dataclass(frozen=True)
class PSRResult:
    """Probabilistic Sharpe Ratio result. `psr` is None when undefined / untrustworthy (with a
    `reason`); descriptive fields are populated whenever computable."""

    psr: float | None
    sr_annualized: float | None
    sr_threshold: float  # the (annualized) benchmark Sharpe tested against
    skew: float | None
    kurt: float | None  # raw kurtosis (3.0 for a Gaussian)
    n: int
    sufficient_n: bool  # n >= _MIN_OBS_RELIABLE (advisory reliability flag)
    reason: str | None  # why psr is None; else None
    sr_annualized_lo: float | None = None  # P0c: Lo(2002) eta(q)-corrected Sharpe (advisory)

    @property
    def passes_psr(self) -> bool:
        """True iff PSR is defined and strictly exceeds the G4 threshold (0.95)."""
        return self.psr is not None and self.psr > PSR_THRESHOLD


def probabilistic_sharpe(
    returns: Sequence[float],
    *,
    sr_threshold: float = 0.0,
    periods_per_year: float = 252.0,
) -> PSRResult:
    """Probabilistic Sharpe Ratio of `returns` against an annualized `sr_threshold`.

    Args:
        returns: per-period strategy returns (ratios, e.g. per-bar or per-trade). Must be finite.
        sr_threshold: ANNUALIZED benchmark Sharpe to test against (default 0 = "is there an edge?").
        periods_per_year: annualization factor at the returns' sampling frequency (a parameter,
            never guessed - the caller knows the bar resolution + calendar). Must be > 0.

    Raises:
        PSRError: periods_per_year <= 0, or any non-finite return.
    """
    if periods_per_year <= 0:
        raise PSRError(f"periods_per_year must be > 0; got {periods_per_year}")
    r = [float(x) for x in returns]
    if not all(math.isfinite(x) for x in r):
        raise PSRError("returns must all be finite (no NaN / inf)")
    n = len(r)

    moments = compute_sharpe_moments(r)
    if moments is None:
        return PSRResult(
            psr=None,
            sr_annualized=None,
            sr_threshold=sr_threshold,
            skew=None,
            kurt=None,
            n=n,
            sufficient_n=False,
            reason="insufficient data (n<2 or zero variance)",
        )

    sr_ann = moments.sr_per_period * math.sqrt(periods_per_year)
    # P0c: Lo(2002) autocorrelation-corrected annualized Sharpe (advisory; does NOT alter psr/dsr).
    eta = lo_autocorr_factor(r, round(periods_per_year))
    sr_ann_lo = moments.sr_per_period * eta if eta is not None else None
    if n < _MIN_OBS_RELIABLE:
        return PSRResult(
            psr=None,
            sr_annualized=sr_ann,
            sr_threshold=sr_threshold,
            skew=moments.skew,
            kurt=moments.kurt,
            n=n,
            sufficient_n=False,
            reason=f"n<{_MIN_OBS_RELIABLE} (skew/kurtosis estimates unreliable)",
            sr_annualized_lo=sr_ann_lo,
        )

    sr_threshold_pp = sr_threshold / math.sqrt(periods_per_year)  # de-annualize into kernel units
    psr = psr_kernel(moments.sr_per_period, sr_threshold_pp, moments.n, moments.skew, moments.kurt)
    reason = None if psr is not None else "variance term non-positive (extreme skew/kurtosis)"
    return PSRResult(
        psr=psr,
        sr_annualized=sr_ann,
        sr_threshold=sr_threshold,
        skew=moments.skew,
        kurt=moments.kurt,
        n=n,
        sufficient_n=True,
        reason=reason,
        sr_annualized_lo=sr_ann_lo,
    )


def min_track_record_length(
    sr_per_period: float,
    sr_star_per_period: float,
    skew: float,
    kurt: float,
    *,
    alpha: float = 0.05,
) -> float | None:
    """LdP Minimum Track Record Length: the minimum number of observations for `sr_per_period` to be
    significant at confidence 1-alpha against `sr_star_per_period` — the algebraic inversion of
    `psr_kernel` (so it mirrors the SAME variance term PSR uses). All Sharpes are PER-PERIOD; `kurt`
    is RAW (Gaussian=3). Returns None if SR <= SR* (never significant) or the variance term is
    non-positive (extreme moments). ADVISORY pre-flight — does NOT gate.
    """
    if sr_per_period <= sr_star_per_period:
        return None
    var_term = 1.0 - skew * sr_per_period + (kurt - 1.0) / 4.0 * sr_per_period**2
    if var_term <= 0.0:
        return None
    z = NormalDist().inv_cdf(1.0 - alpha)
    return 1.0 + var_term * (z / (sr_per_period - sr_star_per_period)) ** 2


__all__: list[str] = [
    "PSR_THRESHOLD",
    "PSRError",
    "PSRResult",
    "min_track_record_length",
    "probabilistic_sharpe",
]
