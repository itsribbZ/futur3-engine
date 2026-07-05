"""futur3.stats.romano_wolf - Romano-Wolf stepdown FWER control (gate G12).

Romano, J.P. & M. Wolf (2005), "Stepwise Multiple Testing as Formalized Data Snooping", Econometrica
73(4). Where Bonferroni/Holm (`multiple_testing.py`) treat the N hypotheses as if independent and so
over-correct when strategies are CORRELATED, Romano-Wolf builds the joint null from a bootstrap and
steps down on the MAX statistic - exploiting the dependence to stay powerful while still controlling
the family-wise error rate. This is gate G12 (§16/§19; advisory Phase A/B, hard Phase C).

Algorithm (studentized, one-sided superiority - identical machinery to SPA's `spa_test.py`):
  1. Per strategy: studentized statistic t_k = sqrt(T)*dbar_k / omega_k, where dbar_k is the
     mean differential vs the benchmark and omega_k its bootstrap std (zero-variance -> excluded).
  2. Shared stationary bootstrap (`_multi_strategy`) recentered to the null: tstar_b,k =
     sqrt(T)*(dstar_b,k - dbar_k) / omega_k. The SAME resampled time indices across strategies keep
     their cross-sectional dependence (the whole point).
  3. Order strategies by DESCENDING t_k. Stepwise, over the not-yet-rejected REMAINING set R:
       p_step = (#{ b : max_{k in R} tstar_b,k >= t_(j) } + 1) / (B + 1)
     enforce the adjusted p-value is monotone non-decreasing (running max); reject while
     adjusted p <= alpha, stop at the first acceptance (you cannot reject a weaker hypothesis once
     the strongest remaining survives). As R shrinks the critical value falls -> more power than a
     single max-of-N (this is the stepdown gain).

Default decision rule: `any_significant` - at least one strategy passes
at alpha=0.05. Pure stdlib; deterministic under `seed`. Fail-loud: bad inputs raise.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from futur3.stats._multi_strategy import bootstrap_differential_means, prepare_differentials

FWER_ALPHA: Final[float] = 0.05  # FWER level: a strategy is significant when its adjusted p < this
_MIN_BOOTSTRAP: Final[int] = 1000


class RomanoWolfError(Exception):
    """Invalid Romano-Wolf input (too few bootstraps, block_length < 1, or alpha not in (0, 1))."""


@dataclass(frozen=True)
class RomanoWolfResult:
    """Romano-Wolf stepdown verdict, in the caller's input order. `reject[k]` is True when
    strategy k's no-superiority null is rejected (genuinely significant after FWER control)."""

    alpha: float
    observed_stats: tuple[float, ...]  # studentized t_k (0.0 for a zero-variance strategy)
    adjusted_pvalues: tuple[float, ...]  # stepdown-adjusted p-values
    reject: tuple[bool, ...]
    n_rejected: int
    n_strategies: int
    n_obs: int
    block_length: int
    n_bootstrap: int
    seed: int | None

    @property
    def any_significant(self) -> bool:
        """The G12 default rule: at least one strategy survives the FWER-controlled stepdown."""
        return self.n_rejected > 0

    @property
    def passes_fwer(self) -> bool:
        """Gate G12 (default `any_significant`): >= 1 strategy passes Romano-Wolf at alpha."""
        return self.any_significant

    @property
    def significant_indices(self) -> tuple[int, ...]:
        """Indices (input order) of the strategies whose null was rejected."""
        return tuple(i for i, r in enumerate(self.reject) if r)


def romano_wolf_stepdown(
    strategy_returns: Sequence[Sequence[float]],
    bench_returns: Sequence[float] | None = None,
    *,
    alpha: float = FWER_ALPHA,
    n_bootstrap: int = 10000,
    block_length: int = 5,
    seed: int | None = None,
) -> RomanoWolfResult:
    """Romano-Wolf stepdown across `strategy_returns` (N x T) vs `bench_returns` (gate G12).

    Args:
        strategy_returns: N strategies, each a length-T return series (same T). Must be finite.
        bench_returns: length-T benchmark series, or None for the zero/cash benchmark.
        alpha: family-wise error level (default 0.05).
        n_bootstrap: stationary-bootstrap draws (>= 1000; default 10000).
        block_length: mean block length for the stationary bootstrap (>= 1; default 5).
        seed: pass an int for a bit-reproducible result.

    Raises:
        RomanoWolfError: n_bootstrap < 1000, block_length < 1, or alpha not in (0, 1).
        MultiStrategyError: ragged / too-small matrix or non-finite values.
    """
    if not 0.0 < alpha < 1.0:
        raise RomanoWolfError(f"alpha must be in (0, 1); got {alpha}")
    if n_bootstrap < _MIN_BOOTSTRAP:
        raise RomanoWolfError(f"n_bootstrap must be >= {_MIN_BOOTSTRAP}; got {n_bootstrap}")
    if block_length < 1:
        raise RomanoWolfError(f"block_length must be >= 1; got {block_length}")

    d, dbar, n_obs, n_strat = prepare_differentials(strategy_returns, bench_returns)
    root_t = math.sqrt(n_obs)
    dstar = bootstrap_differential_means(
        d, n_obs, n_bootstrap=n_bootstrap, block_length=block_length, seed=seed
    )

    # studentize exactly as SPA: omega_k = bootstrap std of sqrt(T)*(dstar_k - dbar_k).
    omega = [
        math.sqrt(math.fsum((root_t * (row[k] - dbar[k])) ** 2 for row in dstar) / n_bootstrap)
        for k in range(n_strat)
    ]
    valid = [w > 0.0 for w in omega]
    t_obs = [root_t * dbar[k] / omega[k] if valid[k] else 0.0 for k in range(n_strat)]
    # recentered studentized bootstrap stats; zero-variance strategies add nothing to the max.
    tstar = [
        [root_t * (row[k] - dbar[k]) / omega[k] if valid[k] else 0.0 for k in range(n_strat)]
        for row in dstar
    ]

    order = sorted(range(n_strat), key=lambda k: t_obs[k], reverse=True)  # descending observed t
    adjusted = [1.0] * n_strat
    reject = [False] * n_strat
    remaining = set(range(n_strat))
    running = 0.0
    for idx in order:
        count = sum(
            1
            for row in tstar
            if max((row[k] for k in remaining if valid[k]), default=0.0) >= t_obs[idx]
        )
        running = max(running, (count + 1) / (n_bootstrap + 1))  # monotone non-decreasing stepdown
        adjusted[idx] = running
        if running <= alpha:
            reject[idx] = True
            remaining.discard(idx)
        else:  # strongest remaining hypothesis survives -> no weaker one can be rejected; stop
            for k in remaining:
                adjusted[k] = running
            break

    return RomanoWolfResult(
        alpha=alpha,
        observed_stats=tuple(t_obs),
        adjusted_pvalues=tuple(adjusted),
        reject=tuple(reject),
        n_rejected=sum(reject),
        n_strategies=n_strat,
        n_obs=n_obs,
        block_length=block_length,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )


__all__: list[str] = [
    "FWER_ALPHA",
    "RomanoWolfError",
    "RomanoWolfResult",
    "romano_wolf_stepdown",
]
