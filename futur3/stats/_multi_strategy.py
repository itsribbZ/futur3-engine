"""futur3.stats._multi_strategy - shared machinery for the multi-strategy gates (WRC + SPA).

PRIVATE module. White's Reality Check (`reality_check.py`) and Hansen's SPA (`spa_test.py`) both ask
"does the BEST of N strategies genuinely beat the benchmark, accounting for the fact that we picked
the best of N?" - a family-wise correction that, unlike Bonferroni, exploits the dependence between
strategies. Both need the same two pieces, which live here so there is ONE audited implementation:

1. The STATIONARY BOOTSTRAP (Politis-Romano 1994): resamples a series in geometric-length blocks
   (new block with prob 1/block_length, else advance one step with wrap-around). This preserves the
   serial dependence that an iid bootstrap would destroy - essential, because the WRC/SPA null is
   about the time-series of per-period out-performance. The SAME resampled time indices are applied
   across all N strategies each draw, preserving their cross-sectional dependence.
2. The DIFFERENTIAL means: d[k][t] = strategy_k - benchmark at t; dbar[k] = mean_t d[k][t]; and the
   bootstrap means dstar[b][k] from resampling the time axis.

Determinism: a fixed `seed` -> identical draws -> byte-reproducible bootstrap means. Pure
stdlib `random.Random`; no numpy/scipy (matches the rest of stats/).
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Final

_MIN_OBS: Final[int] = 2  # need >= 2 time observations
_MIN_STRATEGIES: Final[int] = 1


class MultiStrategyError(Exception):
    """Invalid multi-strategy input (ragged / too-small matrix, non-finite values, bad params)."""


def prepare_differentials(
    strategy_returns: Sequence[Sequence[float]],
    bench_returns: Sequence[float] | None,
) -> tuple[list[list[float]], list[float], int, int]:
    """Validate inputs and return (d, dbar, n_obs, n_strategies).

    `strategy_returns` is N strategies, each a length-T series. `bench_returns` is a length-T series
    or None (None = the zero/cash benchmark, i.e. an absolute-return null). d[k][t] = strat - bench.
    """
    n_strategies = len(strategy_returns)
    if n_strategies < _MIN_STRATEGIES:
        raise MultiStrategyError("need >= 1 strategy")
    n_obs = len(strategy_returns[0])
    if n_obs < _MIN_OBS:
        raise MultiStrategyError(f"need >= {_MIN_OBS} observations per strategy; got {n_obs}")
    if any(len(s) != n_obs for s in strategy_returns):
        raise MultiStrategyError("all strategies must have the same number of observations")
    if bench_returns is None:
        bench = [0.0] * n_obs
    else:
        bench = [float(b) for b in bench_returns]
        if len(bench) != n_obs:
            raise MultiStrategyError("bench_returns length must match the strategy length")
    d: list[list[float]] = []
    for s in strategy_returns:
        row = [float(s[t]) - bench[t] for t in range(n_obs)]
        if not all(math.isfinite(v) for v in row):
            raise MultiStrategyError("strategy / benchmark returns must all be finite")
        d.append(row)
    dbar = [math.fsum(row) / n_obs for row in d]
    return d, dbar, n_obs, n_strategies


def stationary_bootstrap_indices(n: int, block_length: int, rng: random.Random) -> list[int]:
    """One stationary-bootstrap (Politis-Romano 1994) index sequence of length `n`.

    New block with probability 1/block_length; otherwise advance one step with wrap-around.
    block_length == 1 degenerates to the iid bootstrap (no serial structure preserved).
    """
    p = 1.0 / block_length
    idx = [rng.randrange(n)]
    for _ in range(1, n):
        if rng.random() < p:
            idx.append(rng.randrange(n))
        else:
            idx.append((idx[-1] + 1) % n)
    return idx


def bootstrap_differential_means(
    d: list[list[float]],
    n_obs: int,
    *,
    n_bootstrap: int,
    block_length: int,
    seed: int | None,
) -> list[list[float]]:
    """B bootstrap draws; each returns the N per-strategy means under one shared resampling of time.

    Returns a list of `n_bootstrap` rows, each a length-N list of bootstrap differential means. The
    SAME time indices are used across strategies per draw (preserves cross-sectional dependence).
    """
    rng = random.Random(seed)
    out: list[list[float]] = []
    for _ in range(n_bootstrap):
        idx = stationary_bootstrap_indices(n_obs, block_length, rng)
        out.append([math.fsum(row[i] for i in idx) / n_obs for row in d])
    return out
