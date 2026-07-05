"""futur3.stats.multiple_testing - family-wise / false-discovery corrections (§5).

Testing N hypotheses at level alpha makes the family-wise error rate (FWER, the chance of >= 1 false
positive) ~ 1 - (1-alpha)^N -> 0.64 at alpha=0.05, N=20. futur3 runs N strategies x M params x K
windows = hundreds-to-thousands of tests/quarter (§5.2), so without correction the "discoveries"
are mostly chance - the dominant alpha-illusion source. Three corrections:

  - bonferroni_correct (Dunn 1961): reject p_i < alpha/N. FWER control, conservative. PROMOTION.
  - holm_bonferroni_correct (Holm 1979): step-down; same FWER control, uniformly more powerful.
  - bh_fdr_correct (Benjamini-Hochberg 1995): controls the false-DISCOVERY rate (expected fraction
    of rejections that are false), not FWER. Less conservative. EXPLORATION use.

Policy (§5.3): exploration reports BH-FDR (q); promotion reports Bonferroni (alpha). The method is
chosen by WHICH function you call - there is no implicit default (mixing FDR and FWER mid-stream is
the canonical alpha-laundering bug, §5 fm 4). Romano-Wolf stepdown - the FWER-with-dependence
gate G12 - lives in `romano_wolf.py`.

Adjusted p-values follow R's `p.adjust` ("bonferroni" / "holm" / "BH"); `reject` and `adjusted` come
back in the caller's INPUT order. Pure stdlib, deterministic (no RNG). Fail-loud: out-of-
range inputs raise rather than silently clamp.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

_DEFAULT_ALPHA: Final[float] = 0.05


class MultipleTestError(Exception):
    """Invalid multiple-testing input (empty / non-finite / out-of-[0,1] p-values, or a level not in
    (0, 1))."""


@dataclass(frozen=True)
class MultipleTestResult:
    """Outcome of a multiple-comparison correction, in the caller's input order.

    `level` is alpha for the FWER methods (bonferroni / holm) or q for the FDR method (bh_fdr).
    `adjusted` are the R `p.adjust`-style adjusted p-values; `reject[i]` iff `adjusted[i] <= level`.
    """

    method: str
    level: float
    p_values: tuple[float, ...]
    adjusted: tuple[float, ...]
    reject: tuple[bool, ...]
    n_reject: int

    @property
    def any_reject(self) -> bool:
        """True iff at least one hypothesis is rejected (the §5.3 `any_significant` rule)."""
        return self.n_reject > 0


def _validate(p_values: Sequence[float], level: float, level_name: str) -> list[float]:
    if not 0.0 < level < 1.0:
        raise MultipleTestError(f"{level_name} must be in (0, 1); got {level}")
    p = [float(v) for v in p_values]
    if not p:
        raise MultipleTestError("need at least one p-value")
    if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in p):
        raise MultipleTestError("p-values must be finite and in [0, 1]")
    return p


def _result(
    method: str, level: float, p: list[float], adjusted: list[float], reject: list[bool]
) -> MultipleTestResult:
    return MultipleTestResult(
        method=method,
        level=level,
        p_values=tuple(p),
        adjusted=tuple(adjusted),
        reject=tuple(reject),
        n_reject=sum(reject),
    )


def bonferroni_correct(
    p_values: Sequence[float], *, alpha: float = _DEFAULT_ALPHA
) -> MultipleTestResult:
    """Bonferroni FWER correction (Dunn 1961): adjusted_i = min(1, N * p_i); reject if <= alpha."""
    p = _validate(p_values, alpha, "alpha")
    n = len(p)
    adjusted = [min(1.0, pi * n) for pi in p]
    reject = [a <= alpha for a in adjusted]
    return _result("bonferroni", alpha, p, adjusted, reject)


def holm_bonferroni_correct(
    p_values: Sequence[float], *, alpha: float = _DEFAULT_ALPHA
) -> MultipleTestResult:
    """Holm step-down FWER correction (Holm 1979). Sorts ascending and tests p_(r) against the
    multiplier (N - r); the adjusted p-value is the running max so it is monotone non-decreasing."""
    p = _validate(p_values, alpha, "alpha")
    n = len(p)
    order = sorted(range(n), key=lambda i: p[i])  # original indices, ascending by p
    adjusted = [0.0] * n
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (n - rank) * p[idx]))  # multiplier N-rank; cumulative max
        adjusted[idx] = running
    reject = [a <= alpha for a in adjusted]
    return _result("holm", alpha, p, adjusted, reject)


def bh_fdr_correct(p_values: Sequence[float], *, q: float = _DEFAULT_ALPHA) -> MultipleTestResult:
    """Benjamini-Hochberg FDR (1995). Sorts ascending; the step-up adjusted p-value is the running
    min (from the largest rank down) of min(1, p_(r) * N / r), giving FDR <= q on rejection."""
    p = _validate(p_values, q, "q")
    n = len(p)
    order = sorted(range(n), key=lambda i: p[i])  # original indices, ascending by p
    adjusted = [0.0] * n
    running = 1.0
    for rank in range(n - 1, -1, -1):  # walk from the largest p down, taking the running min
        idx = order[rank]
        running = min(running, 1.0, p[idx] * n / (rank + 1))  # 1-based i = rank + 1
        adjusted[idx] = running
    reject = [a <= q for a in adjusted]
    return _result("bh_fdr", q, p, adjusted, reject)


__all__: list[str] = [
    "MultipleTestError",
    "MultipleTestResult",
    "bh_fdr_correct",
    "bonferroni_correct",
    "holm_bonferroni_correct",
]
