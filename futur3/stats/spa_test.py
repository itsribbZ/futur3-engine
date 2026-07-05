"""futur3.stats.spa_test - Hansen's Superior Predictive Ability test (SPA, multi-strategy gate).

Hansen, P.R. (2005), "A Test for Superior Predictive Ability", J. of Business & Economic Statistics
23(4). SPA improves on White's Reality Check (`reality_check.py`) in two ways:

1. STUDENTIZATION: each strategy's mean differential is divided by its own standard error
   (omega_k = std of sqrt(T)*dbar_k), so strategies on different scales contribute comparably.
2. CONSISTENT RECENTERING (SPA_c): strategies that are clearly INFERIOR (studentized mean below
   -sqrt(2 log log T)) are dropped from the null's max instead of inflating it. White's RC keeps all
   strategies, so a pile of useless "null" strategies dilutes its max-statistic and costs power; SPA
   stays powerful. This is why SPA is the futur3 DEFAULT (Hansen 2005 showed WRC biased under null
   strategies); WRC is retained for audit.

Test statistic: T^SPA = max(0, max_k sqrt(T)*dbar_k / omega_k). Null distribution from the shared
stationary bootstrap (`_multi_strategy`), recentered by the consistent rule. omega_k is the
bootstrap std of sqrt(T)*(dstar_k - dbar_k). Gate (Phase B -> C promotion): reject the null at
alpha=0.05. p-value = (#{bootstrap stat >= observed} + 1) / (n_bootstrap + 1).

Fail-loud: zero-variance strategies (omega_k == 0) are excluded (cannot studentize); bad inputs raise.
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

SPA_ALPHA: Final[float] = 0.05  # Phase C promotion: reject the no-superiority null below this
_MIN_BOOTSTRAP: Final[int] = 1000
_MIN_OBS_FOR_TAU: Final[int] = 3  # log(log(T)) needs T > e; clamp below this for the threshold


class SPAError(Exception):
    """Invalid SPA input (too few bootstraps or block_length < 1)."""


@dataclass(frozen=True)
class SPAResult:
    """Hansen's SPA result. `p_value` is the consistent-SPA probability of the observed (or a more
    extreme) best studentized statistic under the no-superiority null; reject below alpha."""

    p_value: float
    test_statistic: float  # T^SPA = max(0, max_k studentized mean)
    max_strategy_idx: int  # strategy with the largest studentized mean
    max_t_stat: float  # that strategy's studentized mean (pre-floor)
    n_strategies: int
    n_obs: int
    block_length: int
    n_bootstrap: int
    seed: int | None

    def reject_null_at(self, alpha: float = SPA_ALPHA) -> bool:
        """True iff the no-superiority null is rejected at `alpha` (a strategy genuinely beats the
        benchmark after the studentized, consistently-recentered max-of-N correction)."""
        return self.p_value < alpha

    @property
    def significant(self) -> bool:
        """Convenience: reject at the default SPA alpha (0.05)."""
        return self.reject_null_at(SPA_ALPHA)


def spa_test(
    strategy_returns: Sequence[Sequence[float]],
    bench_returns: Sequence[float] | None = None,
    *,
    n_bootstrap: int = 10000,
    block_length: int = 5,
    seed: int | None = None,
) -> SPAResult:
    """Hansen's (consistent) SPA test across `strategy_returns` (N x T) vs `bench_returns`.

    Args:
        strategy_returns: N strategies, each a length-T return series (same T). Must be finite.
        bench_returns: length-T benchmark series, or None for the zero/cash benchmark.
        n_bootstrap: stationary-bootstrap draws (>= 1000; default 10000).
        block_length: mean block length for the stationary bootstrap (>= 1; default 5).
        seed: pass an int for a bit-reproducible p-value.

    Raises:
        SPAError: n_bootstrap < 1000 or block_length < 1.
        MultiStrategyError: ragged / too-small matrix or non-finite values.
    """
    if n_bootstrap < _MIN_BOOTSTRAP:
        raise SPAError(f"n_bootstrap must be >= {_MIN_BOOTSTRAP}; got {n_bootstrap}")
    if block_length < 1:
        raise SPAError(f"block_length must be >= 1; got {block_length}")

    d, dbar, n_obs, n_strategies = prepare_differentials(strategy_returns, bench_returns)
    root_t = math.sqrt(n_obs)
    dstar = bootstrap_differential_means(
        d, n_obs, n_bootstrap=n_bootstrap, block_length=block_length, seed=seed
    )

    # omega_k = bootstrap std of sqrt(T)*(dstar_k - dbar_k); 0 for a zero-variance strategy.
    omega = [
        math.sqrt(math.fsum((root_t * (row[k] - dbar[k])) ** 2 for row in dstar) / n_bootstrap)
        for k in range(n_strategies)
    ]
    t_stats = [root_t * dbar[k] / omega[k] if omega[k] > 0.0 else 0.0 for k in range(n_strategies)]
    max_idx = max(range(n_strategies), key=lambda k: t_stats[k])
    observed = max(0.0, t_stats[max_idx])

    # Consistent recentering: keep strategy k's mean only if it is not clearly inferior; otherwise
    # recenter to 0 so its negative bootstrap term cannot enter the max (stops diluting the null).
    tau = math.sqrt(2.0 * math.log(math.log(max(n_obs, _MIN_OBS_FOR_TAU))))
    g = [dbar[k] if (omega[k] > 0.0 and t_stats[k] >= -tau) else 0.0 for k in range(n_strategies)]

    count = 0
    for row in dstar:
        boot_stat = 0.0  # max(0, ...) by initialising the running max at 0
        for k in range(n_strategies):
            if omega[k] > 0.0:
                val = root_t * (row[k] - g[k]) / omega[k]
                boot_stat = max(boot_stat, val)
        if boot_stat >= observed:
            count += 1
    p_value = (count + 1) / (n_bootstrap + 1)

    return SPAResult(
        p_value=p_value,
        test_statistic=observed,
        max_strategy_idx=max_idx,
        max_t_stat=t_stats[max_idx],
        n_strategies=n_strategies,
        n_obs=n_obs,
        block_length=block_length,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )


__all__: list[str] = ["SPA_ALPHA", "SPAError", "SPAResult", "spa_test"]
