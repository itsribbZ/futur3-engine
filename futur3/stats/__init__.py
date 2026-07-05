"""futur3.stats - backtest statistics.

Ships DESCRIPTIVE performance metrics (`performance.py`: return / Sharpe / max-DD / Calmar from an
equity curve) AND the statistical VALIDATION layer (Deflated Sharpe / PSR / BCa / permutation, the standard
gate suite) - the research-grade multi-method corroboration
framework. The two are cleanly separate: descriptive metrics are NOT a promotion gate (a raw Sharpe
is decorative until deflated); the validation gates decide whether an edge is real.

Validation modules shipped so far:
- `probabilistic_sharpe.py` - PSR (G4): single-strategy P(true Sharpe > benchmark).
- `deflated_sharpe.py` - DSR (G3): PSR deflated by the max-of-N selection bias (THE promotion gate).
- `bootstrap_ci.py` - BCa (G5): bias-corrected + accelerated bootstrap CIs (e.g. profit factor).
- `permutation.py` - permutation tests (G7): sign-flip / shuffle / block-shuffle null distributions.
- `reality_check.py` - White's Reality Check: multi-strategy (max-of-N) family-wise promotion gate.
- `spa_test.py` - Hansen's SPA: studentized, consistently-recentered superiority test (default).
- `promotion_gate.py` - composite Phase C decision: all five gates must pass (the §3.6 stack).
- `multiple_testing.py` - Bonferroni / Holm / BH-FDR corrections for N-hypothesis families (§5).
- `romano_wolf.py` - Romano-Wolf stepdown (G12): FWER control under dependence (§5 / §19).
- `backtest_validation.py` - the BACKTEST -> verdict bridge: equity curves -> the gauntlet.
The shared maths live in `_sharpe_core.py` (moments/kernel) + `_multi_strategy.py` (bootstrap).
"""

from __future__ import annotations

from futur3.stats._multi_strategy import MultiStrategyError
from futur3.stats.backtest_validation import (
    BacktestValidationError,
    equity_curve_to_returns,
    evaluate_backtests,
)
from futur3.stats.bootstrap_ci import (
    PF_LOWER_THRESHOLD,
    BCaError,
    BCaResult,
    bca_bootstrap,
    profit_factor,
)
from futur3.stats.deflated_sharpe import DSR_THRESHOLD, DSRResult, deflated_sharpe
from futur3.stats.multiple_testing import (
    MultipleTestError,
    MultipleTestResult,
    bh_fdr_correct,
    bonferroni_correct,
    holm_bonferroni_correct,
)
from futur3.stats.performance import (
    PerformanceMetrics,
    StatsError,
    TradeMetrics,
    compute_metrics,
    compute_trade_metrics,
)
from futur3.stats.permutation import (
    PERMUTATION_P_THRESHOLD,
    PermutationError,
    PermutationMode,
    PermutationResult,
    mean_return,
    permutation_test,
)
from futur3.stats.probabilistic_sharpe import (
    PSR_THRESHOLD,
    PSRError,
    PSRResult,
    probabilistic_sharpe,
)
from futur3.stats.promotion_gate import (
    PromotionDecision,
    SuperiorityMethod,
    evaluate_promotion,
)
from futur3.stats.reality_check import (
    WRC_ALPHA,
    RealityCheckError,
    RealityCheckResult,
    reality_check,
)
from futur3.stats.romano_wolf import (
    FWER_ALPHA,
    RomanoWolfError,
    RomanoWolfResult,
    romano_wolf_stepdown,
)
from futur3.stats.spa_test import SPA_ALPHA, SPAError, SPAResult, spa_test

__all__: list[str] = [
    "DSR_THRESHOLD",  # DSR gate (G3) threshold
    "PSR_THRESHOLD",  # PSR gate (G4) threshold
    "PF_LOWER_THRESHOLD",  # BCa profit-factor gate (G5) threshold
    "PERMUTATION_P_THRESHOLD",  # permutation gate (G7) threshold
    "FWER_ALPHA",
    "SPA_ALPHA",  # Hansen's SPA promotion alpha
    "WRC_ALPHA",  # White's Reality Check promotion alpha
    "BCaError",
    "BCaResult",
    "BacktestValidationError",
    "DSRResult",
    "MultiStrategyError",
    "MultipleTestError",
    "MultipleTestResult",
    "PSRError",
    "PSRResult",
    "PerformanceMetrics",
    "PermutationError",
    "PermutationMode",
    "PermutationResult",
    "PromotionDecision",
    "RealityCheckError",
    "RealityCheckResult",
    "RomanoWolfError",
    "RomanoWolfResult",
    "SPAError",
    "SPAResult",
    "StatsError",
    "SuperiorityMethod",
    "TradeMetrics",
    "bca_bootstrap",
    "bh_fdr_correct",
    "bonferroni_correct",
    "compute_metrics",
    "compute_trade_metrics",
    "deflated_sharpe",
    "equity_curve_to_returns",
    "evaluate_backtests",
    "evaluate_promotion",
    "holm_bonferroni_correct",
    "mean_return",
    "permutation_test",
    "probabilistic_sharpe",
    "profit_factor",
    "reality_check",
    "romano_wolf_stepdown",
    "spa_test",
]
