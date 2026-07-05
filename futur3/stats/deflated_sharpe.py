"""futur3.stats.deflated_sharpe - Deflated Sharpe Ratio (DSR, gate G3).

Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest
Overfitting, and Non-Normality", J. of Portfolio Management 40(5). The DSR is the probability that
the observed Sharpe exceeds the EXPECTED MAXIMUM Sharpe across the N strategies/variants trialled -
i.e. PSR evaluated at SR* = E[max Sharpe over N trials]. It corrects the raw Sharpe for all three
inflation sources at once: selection bias (max-of-N), overfitting (via N + trial variance), and
non-normality (skew + raw kurtosis in the shared kernel).

Promotion gate — Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014): DSR > 0.95. Phase A advisory (small-N backtests have unstable skew /
kurtosis - a known gate-vs-N artifact) -> Phase B/C hard-gate. THE
veteran-quant promotion gate; a raw Sharpe with no DSR adjustment is rejected.

Honest-input warning: `sr_trials` must include EVERY strategy trialled in the same family (not just
the winners). A silent under-count shrinks SR* and inflates DSR - the canonical fake-alpha trap. See
`_sharpe_core.expected_max_sharpe` for the SR* formula + the derivation note.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from futur3.stats._sharpe_core import expected_max_sharpe
from futur3.stats.probabilistic_sharpe import PSRError, probabilistic_sharpe

DSR_THRESHOLD: Final[float] = 0.95  # DSR promotion threshold (Bailey & Lopez de Prado 2014)
_MIN_TRIALS: Final[int] = 2  # need >= 2 trials to estimate the trial-Sharpe variance


@dataclass(frozen=True)
class DSRResult:
    """Deflated Sharpe Ratio result. `dsr` is None when undefined (with `reason`). `psr_implied` is
    the unconditional PSR(0) reported alongside (DSR for the
    gate, PSR as the single-strategy "is it real?" answer)."""

    dsr: float | None
    psr_implied: float | None  # PSR at threshold 0 (single-strategy unconditional probability)
    sr_observed: float | None  # annualized observed Sharpe
    sr_star: float | None  # annualized expected-max Sharpe over trials (the deflation threshold)
    n_trials: int
    var_sr_trials: float | None
    skew: float | None
    kurt: float | None  # raw kurtosis (3.0 for a Gaussian)
    n: int
    reason: str | None  # why dsr is None; else None
    sr_observed_lo: float | None = None  # P0c: Lo(2002) eta(q)-corrected observed Sharpe (advisory)

    @property
    def passes_dsr(self) -> bool:
        """True iff DSR is defined and strictly exceeds the G3 threshold (0.95)."""
        return self.dsr is not None and self.dsr > DSR_THRESHOLD


def deflated_sharpe(
    returns: Sequence[float],
    sr_trials: Sequence[float],
    *,
    periods_per_year: float = 252.0,
) -> DSRResult:
    """Deflated Sharpe Ratio of `returns`, deflated by the selection bias of trialling `sr_trials`.

    Args:
        returns: per-period returns of the SELECTED (best) strategy. Must be finite.
        sr_trials: ANNUALIZED Sharpe of every strategy trialled in the family (must include the
            selected one). Non-finite entries (failed trials) are dropped - a documented contract,
            not a silent filter; n_trials = count of finite trial Sharpes.
        periods_per_year: annualization factor at the returns' sampling frequency. Must be > 0.

    Raises:
        PSRError: periods_per_year <= 0, or any non-finite return.
    """
    if periods_per_year <= 0:
        raise PSRError(f"periods_per_year must be > 0; got {periods_per_year}")
    trials = [float(s) for s in sr_trials if math.isfinite(s)]  # drop failed (NaN/inf) trials
    n_trials = len(trials)
    n_obs = len(returns)

    # PSR(0): the unconditional probability + descriptive moments (also validates `returns`).
    psr0 = probabilistic_sharpe(returns, sr_threshold=0.0, periods_per_year=periods_per_year)

    if n_trials < _MIN_TRIALS:
        return DSRResult(
            dsr=None,
            psr_implied=psr0.psr,
            sr_observed=psr0.sr_annualized,
            sr_star=None,
            n_trials=n_trials,
            var_sr_trials=None,
            skew=psr0.skew,
            kurt=psr0.kurt,
            n=n_obs,
            reason=f"n_trials<{_MIN_TRIALS} (cannot estimate trial-Sharpe variance)",
            sr_observed_lo=psr0.sr_annualized_lo,
        )

    var_sr_trials = statistics.variance(trials)  # sample variance, ddof=1
    sr_star = expected_max_sharpe(n_trials, var_sr_trials)  # annualized SR*

    psr_at_star = probabilistic_sharpe(
        returns, sr_threshold=sr_star, periods_per_year=periods_per_year
    )
    return DSRResult(
        dsr=psr_at_star.psr,
        psr_implied=psr0.psr,
        sr_observed=psr_at_star.sr_annualized,
        sr_star=sr_star,
        n_trials=n_trials,
        var_sr_trials=var_sr_trials,
        skew=psr_at_star.skew,
        kurt=psr_at_star.kurt,
        n=n_obs,
        reason=psr_at_star.reason,
        sr_observed_lo=psr_at_star.sr_annualized_lo,
    )


__all__: list[str] = ["DSR_THRESHOLD", "DSRResult", "deflated_sharpe"]
