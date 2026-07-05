"""futur3.stats.reality_check - White's Reality Check (WRC, multi-strategy promotion gate).

White, H. (2000), "A Reality Check for Data Snooping", Econometrica 68(5). Tests the null "the BEST
of the N strategies does not out-perform the benchmark" - the family-wise correction for having
picked the winner out of N. The statistic is the max over strategies of sqrt(T)*mean-differential;
its null distribution comes from the stationary bootstrap (`_multi_strategy`), recentered so each
strategy's bootstrap mean has the observed mean removed (imposes the null).

Promotion gate (White 2000 Reality Check, Phase B -> Phase C): a strategy promoted to Phase C must reject
this null at alpha=0.05 vs every strategy trialled in its family. Phase B advisory -> Phase C hard.

WRC is biased CONSERVATIVE when many "null" strategies (no edge) are present - they dilute the
max-statistic. Hansen's SPA (`spa_test.py`) studentizes + recenters to fix that and is the futur3
default; WRC is retained for audit + as the simpler, well-understood reference.

p-value = (#{bootstrap max-stat >= observed} + 1) / (n_bootstrap + 1) (never exactly 0). Fail-loud: bad
inputs raise MultiStrategyError; p_value is always defined for valid input (no None).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from futur3.stats._multi_strategy import (
    bootstrap_differential_means,
    prepare_differentials,
)

WRC_ALPHA: Final[float] = 0.05  # Phase C promotion: reject the no-superiority null below this
_MIN_BOOTSTRAP: Final[int] = 1000


class RealityCheckError(Exception):
    """Invalid Reality Check input (too few bootstraps or block_length < 1)."""


@dataclass(frozen=True)
class RealityCheckResult:
    """White's Reality Check result. `p_value` is the probability of the observed (or larger) best
    strategy under the no-superiority null; reject (a real edge) when it is below alpha."""

    p_value: float
    max_strategy_idx: int  # index of the best strategy (largest mean differential)
    max_differential: float  # that strategy's mean per-period out-performance
    n_strategies: int
    n_obs: int
    block_length: int
    n_bootstrap: int
    seed: int | None

    def reject_null_at(self, alpha: float = WRC_ALPHA) -> bool:
        """True iff the no-superiority null is rejected at `alpha` (i.e. a strategy genuinely beats
        the benchmark after the max-of-N correction)."""
        return self.p_value < alpha

    @property
    def significant(self) -> bool:
        """Convenience: reject at the default WRC alpha (0.05)."""
        return self.reject_null_at(WRC_ALPHA)


def reality_check(
    strategy_returns: Sequence[Sequence[float]],
    bench_returns: Sequence[float] | None = None,
    *,
    n_bootstrap: int = 10000,
    block_length: int = 5,
    seed: int | None = None,
) -> RealityCheckResult:
    """White's Reality Check across `strategy_returns` (N x T) vs `bench_returns`.

    Args:
        strategy_returns: N strategies, each a length-T return series (same T). Must be finite.
        bench_returns: length-T benchmark series, or None for the zero/cash benchmark.
        n_bootstrap: stationary-bootstrap draws (>= 1000; default 10000).
        block_length: mean block length for the stationary bootstrap (>= 1; default 5).
        seed: pass an int for a bit-reproducible p-value.

    Raises:
        RealityCheckError: n_bootstrap < 1000 or block_length < 1.
        MultiStrategyError: ragged / too-small matrix or non-finite values.
    """
    if n_bootstrap < _MIN_BOOTSTRAP:
        raise RealityCheckError(f"n_bootstrap must be >= {_MIN_BOOTSTRAP}; got {n_bootstrap}")
    if block_length < 1:
        raise RealityCheckError(f"block_length must be >= 1; got {block_length}")

    d, dbar, n_obs, n_strategies = prepare_differentials(strategy_returns, bench_returns)
    root_t = math.sqrt(n_obs)
    observed = root_t * max(dbar)
    max_idx = max(range(n_strategies), key=lambda k: dbar[k])

    dstar = bootstrap_differential_means(
        d, n_obs, n_bootstrap=n_bootstrap, block_length=block_length, seed=seed
    )
    # Recentered max statistic per draw: max_k sqrt(T) * (dstar_k - dbar_k). Imposes the null.
    count = sum(
        1
        for row in dstar
        if max(root_t * (row[k] - dbar[k]) for k in range(n_strategies)) >= observed
    )
    p_value = (count + 1) / (n_bootstrap + 1)

    return RealityCheckResult(
        p_value=p_value,
        max_strategy_idx=max_idx,
        max_differential=dbar[max_idx],
        n_strategies=n_strategies,
        n_obs=n_obs,
        block_length=block_length,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )


__all__: list[str] = ["WRC_ALPHA", "RealityCheckError", "RealityCheckResult", "reality_check"]
