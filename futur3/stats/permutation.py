"""futur3.stats.permutation - permutation / randomization tests (gate G7).

Edgington & Onghena (2007), *Randomization Tests*; Lopez de Prado, *Advances in FML* (2018) ch.12.
Under the null "the strategy has no edge", a transform that destroys the edge but preserves the
benign structure leaves the statistic's distribution unchanged. We apply the transform
`n_permutations` times, build the null distribution, and ask how extreme the observed statistic is.

Promotion gate — permutation test: p-value < 0.05. Phase A advisory -> Phase B/C hard-gate.

p-value = (#{null >= observed} + 1) / (n_valid + 1). The +1 (Edgington) keeps the test valid: the
observed value is itself one realization under H0, so p is never exactly 0.

Modes (the transform MUST match what the statistic measures):
- "sign_flip" (default): flip each return's sign with p=0.5. Tests H0 "no DIRECTIONAL edge". Pair
  with a sign-sensitive statistic (mean / Sharpe). Preserves the magnitude/vol structure.
- "shuffle": random reorder. Tests H0 "order does not matter". Pair with an ORDER-dependent
  statistic (e.g. max-drawdown, autocorrelation) - a mean is permutation-invariant and yields p~1.
- "block_shuffle": reorder length-`block_length` blocks, preserving short-horizon serial structure.
  For autocorrelated series where plain sign_flip/shuffle would overstate significance.

A lag-1 autocorrelation diagnostic is reported; if it is significant under sign_flip/shuffle (which
assume exchangeability) `autocorr_warning` is set, recommending block_shuffle (a diagnostic, not a
block; diagnostics are features by design). Undefined results -> p_value None + reason per the fail-loud policy; > 10% NaN draws raise.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal

PermutationMode = Literal["sign_flip", "block_shuffle", "shuffle"]

PERMUTATION_P_THRESHOLD: Final[float] = 0.05  # permutation p-value must be below this
_MIN_PERMUTATIONS: Final[int] = 1000  # below this a p-value has no resolution
_MIN_RETURNS: Final[int] = 2  # need >= 2 returns to permute
_DEFAULT_PERMUTATIONS: Final[int] = 10000
_DEFAULT_BLOCK_LENGTH: Final[int] = 5
_FAIR_COIN_P: Final[float] = 0.5  # sign-flip probability per return
_MAX_NAN_FRACTION: Final[float] = 0.10  # > 10% degenerate permutations -> raise
_MIN_AUTOCORR_OBS: Final[int] = 3
_VALID_MODES: Final[frozenset[str]] = frozenset({"sign_flip", "block_shuffle", "shuffle"})


class PermutationError(Exception):
    """Invalid permutation-test input (bad mode/counts, non-finite returns, too many NaN draws)."""


@dataclass(frozen=True)
class PermutationResult:
    """Permutation-test result. `p_value` is None when undefined (with a `reason`). The full
    `null_distribution` is kept (diagnostics are features) but left out of repr for brevity."""

    p_value: float | None
    observed: float
    n_permutations: int  # requested
    n_valid: int  # permutations that produced a finite statistic
    mode: str
    one_sided: bool
    block_length: int
    lag1_autocorr: float | None
    autocorr_warning: bool
    seed: int | None
    reason: str | None
    null_distribution: tuple[float, ...] = field(repr=False, default=())

    @property
    def passes_permutation(self) -> bool:
        """True iff the p-value is defined and strictly below the G7 threshold (0.05)."""
        return self.p_value is not None and self.p_value < PERMUTATION_P_THRESHOLD


def mean_return(returns: Sequence[float]) -> float:
    """Default permutation statistic: the mean return (sign-sensitive -> tests directional edge)."""
    return statistics.fmean(returns)


def _lag1_autocorr(returns: Sequence[float]) -> float | None:
    n = len(returns)
    if n < _MIN_AUTOCORR_OBS:
        return None
    mean = statistics.fmean(returns)
    var = math.fsum((x - mean) ** 2 for x in returns)
    if var <= 0.0:
        return None
    cov = math.fsum((returns[i] - mean) * (returns[i - 1] - mean) for i in range(1, n))
    return cov / var


def _permute(returns: list[float], mode: str, block_length: int, rng: random.Random) -> list[float]:
    if mode == "sign_flip":
        return [r if rng.random() < _FAIR_COIN_P else -r for r in returns]
    if mode == "shuffle":
        out = returns[:]
        rng.shuffle(out)
        return out
    # block_shuffle: reorder contiguous blocks, preserving within-block serial structure
    n = len(returns)
    blocks = [returns[i : i + block_length] for i in range(0, n, block_length)]
    rng.shuffle(blocks)
    return [r for block in blocks for r in block]


def permutation_test(
    returns: Sequence[float],
    statistic: Callable[[Sequence[float]], float] = mean_return,
    *,
    n_permutations: int = _DEFAULT_PERMUTATIONS,
    mode: PermutationMode = "sign_flip",
    block_length: int = _DEFAULT_BLOCK_LENGTH,
    one_sided: bool = True,
    seed: int | None = None,
) -> PermutationResult:
    """Permutation test of `statistic` on `returns` under the no-edge null.

    Args:
        returns: per-period returns. Must be finite, >= 2 points.
        statistic: maps a sample to a scalar (default `mean_return`). MUST match `mode`: sign_flip
            needs a sign-sensitive statistic, shuffle an order-dependent one (see module docstring).
        n_permutations: number of permutations (>= 1000; default 10000).
        mode: "sign_flip" (default) | "shuffle" | "block_shuffle".
        block_length: block size for "block_shuffle" (>= 1; ignored otherwise).
        one_sided: if True (default) test the upper tail (positive edge); else two-sided on |stat|.
        seed: pass an int for a bit-reproducible null distribution.

    Raises:
        PermutationError: bad mode, n_permutations < 1000, block_length < 1, non-finite returns,
            < 2 returns, or > 10% of permutations producing a non-finite statistic.
    """
    if mode not in _VALID_MODES:
        raise PermutationError(f"mode must be one of {sorted(_VALID_MODES)}; got {mode!r}")
    if n_permutations < _MIN_PERMUTATIONS:
        raise PermutationError(
            f"n_permutations must be >= {_MIN_PERMUTATIONS}; got {n_permutations}"
        )
    if block_length < 1:
        raise PermutationError(f"block_length must be >= 1; got {block_length}")
    r = [float(x) for x in returns]
    n = len(r)
    if n < _MIN_RETURNS:
        raise PermutationError(f"need >= {_MIN_RETURNS} returns; got {n}")
    if not all(math.isfinite(x) for x in r):
        raise PermutationError("returns must all be finite (no NaN / inf)")

    lag1 = _lag1_autocorr(r)
    autocorr_warning = (
        lag1 is not None
        and mode in ("sign_flip", "shuffle")
        and abs(lag1) > 2.0 / math.sqrt(n)  # ~95% significance bound for lag-1 ACF
    )

    observed = statistic(r)

    def _result(
        p_value: float | None, n_valid: int, null: tuple[float, ...], reason: str | None
    ) -> PermutationResult:
        return PermutationResult(
            p_value=p_value,
            observed=observed,
            n_permutations=n_permutations,
            n_valid=n_valid,
            mode=mode,
            one_sided=one_sided,
            block_length=block_length,
            lag1_autocorr=lag1,
            autocorr_warning=autocorr_warning,
            seed=seed,
            reason=reason,
            null_distribution=null,
        )

    if not math.isfinite(observed):
        return _result(None, 0, (), "observed statistic is not finite")

    rng = random.Random(seed)
    null = [
        s
        for _ in range(n_permutations)
        if math.isfinite(s := statistic(_permute(r, mode, block_length, rng)))
    ]
    n_valid = len(null)
    n_nan = n_permutations - n_valid
    if n_nan > _MAX_NAN_FRACTION * n_permutations:
        raise PermutationError(
            f"{n_nan}/{n_permutations} permutations produced a non-finite statistic (> "
            f"{_MAX_NAN_FRACTION:.0%}); statistic is too unstable for this data"
        )

    if one_sided:
        n_ge = sum(1 for s in null if s >= observed)
    else:
        abs_obs = abs(observed)
        n_ge = sum(1 for s in null if abs(s) >= abs_obs)
    p_value = (n_ge + 1) / (n_valid + 1)
    return _result(p_value, n_valid, tuple(null), None)


__all__: list[str] = [
    "PERMUTATION_P_THRESHOLD",
    "PermutationError",
    "PermutationMode",
    "PermutationResult",
    "mean_return",
    "permutation_test",
]
