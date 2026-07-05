"""futur3.stats.backtest_validation - the BACKTEST -> PROMOTION-VERDICT bridge (the spine seam).

The two halves of futur3 met here: `engine.BacktestEngine` produces an equity curve per strategy
run, and `stats.promotion_gate.evaluate_promotion` decides whether returns are a real edge - but
nothing joined them. This module is that join: given a FAMILY of strategy backtests (their equity
curves) it assembles the gauntlet's inputs - the selected strategy's per-period returns, the N x T
family matrix, and the trial Sharpes for DSR's selection-bias deflation - and runs the five-gate
Phase C decision.

It stays ENGINE-FREE on purpose (imports only `performance` + `promotion_gate`, both leaf stats
modules): the caller runs the engine and passes `[r.equity_curve for r in run_results]`, so `stats`
keeps zero dependency on `engine`. The family is the set of trials you actually ran (e.g. a lookback
sweep, or several catalog strategies) - the honest n_trials DSR/SPA correct for, NOT a cherry-picked
subset (`promotion_gate` docstring: garbage in -> a confidently-wrong promotion out).

Returns convention matches `performance.compute_metrics` exactly (r_t = equity_t/equity_{t-1} - 1),
so the descriptive metrics and the validation gates see the SAME returns.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from decimal import Decimal
from typing import Final

from futur3.stats.promotion_gate import (
    PromotionDecision,
    SuperiorityMethod,
    evaluate_promotion,
)

_MIN_EQUITY_POINTS: Final[int] = 2  # need >= 2 equity points to form one return
_MIN_RETURNS_FOR_SHARPE: Final[int] = 2
_DEFAULT_RESAMPLES: Final[int] = 10000


def _annualized_sharpe(returns: Sequence[float], periods_per_year: float) -> float:
    """Raw annualized Sharpe (rf=0) of a return series; 0.0 if < 2 points or ~zero variance.

    Mirrors `performance.compute_metrics`' Sharpe but consumes the (ruin-handled) returns directly,
    so a blown-up trial yields a very negative Sharpe instead of raising on its negative equity.
    """
    if len(returns) < _MIN_RETURNS_FOR_SHARPE:
        return 0.0
    mean_r = statistics.mean(returns)
    if all(math.isclose(r, mean_r, rel_tol=1e-9, abs_tol=1e-12) for r in returns):
        return 0.0  # effectively constant -> undefined Sharpe -> 0.0 (fail-loud: never a fake huge value)
    return (mean_r / statistics.stdev(returns)) * math.sqrt(periods_per_year)


class BacktestValidationError(Exception):
    """Bridge error: an empty family, an out-of-range selection, or mismatched curve lengths."""


def equity_curve_to_returns(equity_curve: Sequence[Decimal]) -> list[float]:
    """Per-period simple returns from an equity curve - the engine -> gauntlet seam.

    r_t = equity_t / equity_{t-1} - 1. Mirrors `performance.compute_metrics`' internal convention so
    descriptive metrics and the validation gates consume identical returns.

    Args:
        equity_curve: per-bar mark-to-market equity (e.g. `RunResult.equity_curve`); strictly
            positive, >= 2 points.

    Raises:
        BacktestValidationError: fewer than 2 points, or any non-positive equity value.
    """
    if len(equity_curve) < _MIN_EQUITY_POINTS:
        raise BacktestValidationError(
            f"need >= {_MIN_EQUITY_POINTS} equity points to form a return; got {len(equity_curve)}"
        )
    equity = [float(x) for x in equity_curve]
    if equity[0] <= 0:
        raise BacktestValidationError("equity curve must START strictly positive")
    # Ruin handling: if a leveraged loss drives equity <= 0 mid-curve (a blown account), record a
    # -100% (total-loss) return at the breach, then dead-flat 0.0 for the rest. This keeps a FULL
    # length-(n-1) series (so a family stays equal-length for the gauntlet) and a ruined strategy
    # surfaces as a catastrophic return series the gates correctly reject - never a crash.
    returns: list[float] = []
    ruined = False
    for i in range(1, len(equity)):
        if ruined:
            returns.append(0.0)
            continue
        prev, cur = equity[i - 1], equity[i]
        if cur <= 0:
            returns.append(-1.0)
            ruined = True
            continue
        returns.append(cur / prev - 1.0)
    return returns


def evaluate_backtests(
    equity_curves: Sequence[Sequence[Decimal]],
    selected_index: int,
    *,
    periods_per_year: float,
    bench_curve: Sequence[Decimal] | None = None,
    superiority_method: SuperiorityMethod = "spa",
    n_resamples: int = _DEFAULT_RESAMPLES,
    seed: int | None = None,
) -> PromotionDecision:
    """Run the Phase C promotion gauntlet on a family of strategy backtests.

    Converts every equity curve to per-period returns, assembles the gauntlet inputs (the selected
    strategy's returns, the N x T family, the trial Sharpes), and runs `evaluate_promotion`.

    Args:
        equity_curves: one per trialled strategy (same bar window -> equal length). Pass
            `[r.equity_curve for r in run_results]` from the engine.
        selected_index: which curve is the SELECTED strategy under test (G3/G4/G5/G7 run on it).
        periods_per_year: annualization factor at the bars' sampling frequency (never guessed).
        bench_curve: benchmark equity curve for SPA/WRC, or None for the zero/cash benchmark.
        superiority_method: "spa" (default) or "wrc".
        n_resamples: bootstrap/permutation count shared across G5/G7/superiority.
        seed: int for a bit-reproducible verdict.

    Returns:
        The composite `PromotionDecision` (inspect `.promoted` / `.failed_gates`).

    Raises:
        BacktestValidationError: empty family, out-of-range `selected_index`, or unequal lengths.
    """
    if not equity_curves:
        raise BacktestValidationError("need >= 1 equity curve to evaluate")
    n = len(equity_curves)
    if not 0 <= selected_index < n:
        raise BacktestValidationError(f"selected_index {selected_index} out of range [0, {n})")
    lengths = {len(c) for c in equity_curves}
    if len(lengths) != 1:
        raise BacktestValidationError(
            f"all equity curves must be equal length (one comparable bar window for the family); "
            f"got lengths {sorted(lengths)}"
        )

    family_returns = [equity_curve_to_returns(c) for c in equity_curves]
    strategy_returns = family_returns[selected_index]
    # Trial Sharpes for DSR's selection-bias deflation, from the (ruin-handled) returns: a flat /
    # zero-variance trial -> 0.0; a blown-up trial -> a very negative Sharpe. Either way it counts
    # toward n_trials (the whole point of deflation is "how many did you try").
    sr_trials = [_annualized_sharpe(r, periods_per_year) for r in family_returns]
    bench_returns = equity_curve_to_returns(bench_curve) if bench_curve is not None else None

    return evaluate_promotion(
        strategy_returns,
        family_returns,
        sr_trials,
        periods_per_year=periods_per_year,
        bench_returns=bench_returns,
        superiority_method=superiority_method,
        n_resamples=n_resamples,
        seed=seed,
    )


__all__: list[str] = [
    "BacktestValidationError",
    "equity_curve_to_returns",
    "evaluate_backtests",
]
