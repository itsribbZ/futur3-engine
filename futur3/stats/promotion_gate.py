"""futur3.stats.promotion_gate - the composite Phase B -> Phase C promotion decision.

The five validation methods are COMPLEMENTARY, not redundant - each
interrogates a different failure mode, so a Phase C promotion requires ALL of them to pass at
alpha=0.05 ("all five", not majority vote - failing any one is a red flag worth investigating):

  Deflated Sharpe (DSR)   > 0.95   (selection bias + overfitting + non-normality)
  Probabilistic SR (PSR)  > 0.95   (single-strategy "is the edge real?")
  BCa profit-factor lower bound > 1.0   (distribution-aware confidence interval)
  Permutation p     < 0.05   (non-parametric "no signal" null)
  superiority (SPA default, or WRC)  reject at 0.05   (family-wise multi-strategy correction)

`evaluate_promotion` runs all five and returns a `PromotionDecision` that retains every underlying
result object (diagnostics are features by design) plus per-gate pass flags, the overall
`promoted` verdict, and the list of `failed_gates`. It DECIDES nothing on its own beyond the AND of
the five gates; the individual results carry the numbers an operator audits.

This is a promotion gate, NOT a backtest: feed OUT-OF-SAMPLE returns + the full family of trials
(honest n_trials per DSR's warning). Garbage in - a confidently-wrong promotion out.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

from futur3.stats.bootstrap_ci import (
    PF_LOWER_THRESHOLD,
    BCaResult,
    bca_bootstrap,
    profit_factor,
)
from futur3.stats.deflated_sharpe import DSRResult, deflated_sharpe
from futur3.stats.permutation import PermutationResult, permutation_test
from futur3.stats.probabilistic_sharpe import PSRResult, probabilistic_sharpe
from futur3.stats.reality_check import RealityCheckResult, reality_check
from futur3.stats.spa_test import SPAResult, spa_test

SuperiorityMethod = Literal["spa", "wrc"]

_DEFAULT_RESAMPLES: Final[int] = 10000


@dataclass(frozen=True)
class PromotionDecision:
    """Composite Phase C promotion verdict. `promoted` is the AND of the five gates; each underlying
    result is retained for audit, and `failed_gates` names the ones that did not pass."""

    dsr: DSRResult
    psr: PSRResult
    bca_profit_factor: BCaResult
    permutation: PermutationResult
    superiority: SPAResult | RealityCheckResult
    superiority_method: str

    @property
    def passes_dsr(self) -> bool:
        return self.dsr.passes_dsr

    @property
    def passes_psr(self) -> bool:
        return self.psr.passes_psr

    @property
    def passes_profit_factor(self) -> bool:
        return self.bca_profit_factor.lower_exceeds(PF_LOWER_THRESHOLD)

    @property
    def passes_permutation(self) -> bool:
        return self.permutation.passes_permutation

    @property
    def passes_superiority(self) -> bool:
        return self.superiority.significant

    @property
    def promoted(self) -> bool:
        """True iff ALL five gates pass (the Phase C bar)."""
        return all(
            (
                self.passes_dsr,
                self.passes_psr,
                self.passes_profit_factor,
                self.passes_permutation,
                self.passes_superiority,
            )
        )

    @property
    def failed_gates(self) -> tuple[str, ...]:
        """Names of the gates that did not pass (empty iff promoted)."""
        checks = (
            ("deflated_sharpe", self.passes_dsr),
            ("probabilistic_sharpe", self.passes_psr),
            ("bca_profit_factor", self.passes_profit_factor),
            ("permutation", self.passes_permutation),
            (f"superiority_{self.superiority_method}", self.passes_superiority),
        )
        return tuple(name for name, ok in checks if not ok)


def evaluate_promotion(
    strategy_returns: Sequence[float],
    family_returns: Sequence[Sequence[float]],
    sr_trials: Sequence[float],
    *,
    periods_per_year: float = 252.0,
    bench_returns: Sequence[float] | None = None,
    superiority_method: SuperiorityMethod = "spa",
    n_resamples: int = _DEFAULT_RESAMPLES,
    seed: int | None = None,
) -> PromotionDecision:
    """Run the five-gate Phase C promotion stack on the selected strategy + its family.

    Args:
        strategy_returns: the SELECTED strategy's out-of-sample per-period returns (G3/G4/G5/G7).
        family_returns: every strategy trialled in the family (N x T), incl. the selected one - the
            multi-strategy correction (SPA/WRC) corrects for having picked the winner of these.
        sr_trials: annualized Sharpes of all trials (DSR's selection-bias term; include the winner).
        periods_per_year: annualization factor for DSR/PSR (a parameter, never guessed).
        bench_returns: benchmark series for SPA/WRC, or None for the zero/cash benchmark.
        superiority_method: "spa" (default, more powerful) or "wrc" (audit reference).
        n_resamples: bootstrap/permutation/resample count shared across G5/G7/superiority.
        seed: pass an int for a bit-reproducible decision.

    Returns:
        A PromotionDecision; inspect `.promoted` and `.failed_gates`.
    """
    dsr = deflated_sharpe(strategy_returns, sr_trials, periods_per_year=periods_per_year)
    psr = probabilistic_sharpe(strategy_returns, periods_per_year=periods_per_year)
    bca_pf = bca_bootstrap(strategy_returns, profit_factor, n_resamples=n_resamples, seed=seed)
    perm = permutation_test(strategy_returns, n_permutations=n_resamples, seed=seed)

    superiority: SPAResult | RealityCheckResult
    if superiority_method == "spa":
        superiority = spa_test(family_returns, bench_returns, n_bootstrap=n_resamples, seed=seed)
    else:
        superiority = reality_check(
            family_returns, bench_returns, n_bootstrap=n_resamples, seed=seed
        )

    return PromotionDecision(
        dsr=dsr,
        psr=psr,
        bca_profit_factor=bca_pf,
        permutation=perm,
        superiority=superiority,
        superiority_method=superiority_method,
    )


__all__: list[str] = ["PromotionDecision", "SuperiorityMethod", "evaluate_promotion"]
