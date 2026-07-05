"""futur3.cv.block_bootstrap - block bootstrap confidence intervals for autocorrelated series.

The iid BCa bootstrap (`stats/bootstrap_ci.py`) resamples
single observations, which destroys serial dependence; on autocorrelated returns its CI is biased
TOO NARROW (Politis-Romano 1994). Block resampling draws CONTIGUOUS runs, preserving the local
dependence so the interval is honest. Three block schemes:

  - "stationary" (Politis-Romano 1994; futur3 default): random geometric-length blocks. It reuses
    the ONE audited `stationary_bootstrap_indices` already shared by WRC/SPA - IMPORTED from
    `stats/_multi_strategy`, never re-copied, so the two cannot diverge.
  - "moving" (Kuensch 1989): overlapping fixed-length blocks, circular so all n start positions are
    valid (no end-of-series under-weighting).
  - "fixed": non-overlapping disjoint blocks resampled whole.

Block length: pass `block_mean` explicitly, or leave it None to estimate it from the data's own
autocorrelation via Politis & White (2004) automatic selection (`optimal_block_length`).

Conventions (match the rest of stats/): pure stdlib (`random.Random` -> bit-reproducibility);
PERCENTILE interval, not BCa - block resamples are not iid so the jackknife acceleration does
not apply; NaN resamples dropped, +/-inf kept (a legitimately extreme statistic); an undefined point
statistic -> NaN bounds + a `reason` (fail-loud: never a misleading number). `bootstrap_distribution`
retains every resample draw (diagnostics are features by design).

NOTE - the original spec sketched numpy/pandas signatures and a `null_distribution` field; futur3 source
is stdlib-only (matching all of stats/) and the field is named `bootstrap_distribution` because a CI
uses the bootstrap distribution of the ESTIMATE, not a null. Spec prose is aspirational there.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, Literal

from futur3.stats._multi_strategy import stationary_bootstrap_indices
from futur3.stats.bootstrap_ci import _quantile

BlockMode = Literal["stationary", "moving", "fixed"]

_VALID_MODES: Final[tuple[BlockMode, ...]] = ("stationary", "moving", "fixed")
_MIN_DATA: Final[int] = 2  # need >= 2 points to form a block resample + a CI
_MIN_RESAMPLES: Final[int] = 1000  # below this a tail-percentile CI has no resolution
_DEFAULT_RESAMPLES: Final[int] = 10000
_MIN_PW_DATA: Final[int] = 8  # below this the Politis-White autocorrelation estimate is unstable
_FLAT_TOP_HALF: Final[float] = 0.5  # flat-top lag-window half-width (unit weight within +/- this)


class BlockBootstrapError(Exception):
    """Invalid block-bootstrap input (bad confidence / mode / block length, too few resamples or
    data, non-finite data)."""


@dataclass(frozen=True)
class BlockBootstrapResult:
    """Block-bootstrap percentile confidence interval. `lower`/`upper` are NaN when the interval is
    undefined (with a `reason`): a non-finite point statistic or too many degenerate resamples.
    `block_mean_estimated` is True when the length was chosen by Politis-White, not given."""

    point: float  # full-sample statistic
    lower: float
    upper: float
    confidence: float
    block_mean: int  # block length actually used
    block_mean_estimated: bool  # True iff chosen by Politis-White (block_mean=None on input)
    mode: str
    n_resamples: int
    n: int  # sample size
    seed: int | None
    bootstrap_distribution: tuple[float, ...]  # every (non-NaN) resample draw; () when undefined
    reason: str | None  # why the interval is undefined; else None

    def lower_exceeds(self, threshold: float) -> bool:
        """True iff the lower bound is defined and strictly exceeds `threshold`."""
        return math.isfinite(self.lower) and self.lower > threshold


# ---------------------------------------------------------------------------
# Block index generators - each returns a length-n list of resampled indices.
# ---------------------------------------------------------------------------


def _moving_block_indices(n: int, block_len: int, rng: random.Random) -> list[int]:
    """Circular moving-block (Kuensch 1989): overlapping length-`block_len` blocks from any of the n
    circular start positions, concatenated then trimmed to n. block_len == 1 degenerates to iid."""
    idx: list[int] = []
    while len(idx) < n:
        start = rng.randrange(n)
        idx.extend((start + j) % n for j in range(block_len))
    return idx[:n]


def _fixed_block_indices(n: int, block_len: int, rng: random.Random) -> list[int]:
    """Non-overlapping fixed blocks: the disjoint blocks [0,L), [L,2L), ... are resampled whole (the
    last block is short when n is not a multiple of L). block_len == n returns the original sample;
    block_len == 1 degenerates to iid."""
    starts = list(range(0, n, block_len))
    idx: list[int] = []
    while len(idx) < n:
        s = starts[rng.randrange(len(starts))]
        idx.extend(range(s, min(s + block_len, n)))
    return idx[:n]


def _block_indices(mode: BlockMode, n: int, block_len: int, rng: random.Random) -> list[int]:
    """Dispatch to the block scheme. `mode` is validated by the caller (no silent fallback)."""
    if mode == "stationary":
        return stationary_bootstrap_indices(n, block_len, rng)
    if mode == "moving":
        return _moving_block_indices(n, block_len, rng)
    return _fixed_block_indices(n, block_len, rng)  # "fixed"


# ---------------------------------------------------------------------------
# Politis & White (2004) automatic block-length selection.
# ---------------------------------------------------------------------------


def _autocovariances(centered: Sequence[float], max_lag: int, n: int) -> list[float]:
    """Biased sample autocovariances gamma(k), k=0..max_lag (divisor n). `centered` = x - xbar."""
    return [
        math.fsum(centered[t] * centered[t + k] for t in range(n - k)) / n
        for k in range(max_lag + 1)
    ]


def optimal_block_length(data: Sequence[float]) -> int:
    """Politis & White (2004) automatic block length for the STATIONARY bootstrap.

    Follows the canonical Patton reference implementation: pick the bandwidth `m_hat` as the largest
    lag with significant autocorrelation (the first run of K_N insignificant lags then marks
    the cutoff), set M = 2*m_hat, then b_SB = ((2*Ghat^2)/D_SB)^(1/3) * n^(1/3) using the flat-top
    lag window. White noise -> ~1; positively autocorrelated data -> larger. Result is clamped to
    [1, ceil(min(3*sqrt(n), n/3))]. Pure function: identical input -> identical L.
    """
    x = [float(v) for v in data]
    n = len(x)
    if n < _MIN_PW_DATA:
        return 1
    mean = math.fsum(x) / n
    centered = [v - mean for v in x]

    k_n = max(5, math.ceil(math.sqrt(math.log10(n))))  # K_N: insignificant-run window length
    m_max = math.ceil(math.sqrt(n)) + k_n
    gamma = _autocovariances(centered, m_max, n)
    g0 = gamma[0]
    if g0 <= 0.0:  # constant series -> no dependence to preserve
        return 1
    rho = [g / g0 for g in gamma]
    threshold = 2.0 * math.sqrt(math.log10(n) / n)  # rho-significance band (~1.96/2 sigma)

    # m_hat = lag just before the first run of K_N consecutive insignificant autocorrelations.
    m_hat: int | None = None
    for kk in range(1, m_max - k_n + 2):
        if all(abs(rho[kk + j]) < threshold for j in range(k_n)):
            m_hat = kk - 1
            break
    if m_hat is None:  # no insignificant run found -> largest significant lag (else 1)
        sig = [k for k in range(1, m_max + 1) if abs(rho[k]) >= threshold]
        m_hat = max(sig) if sig else 1
    m_hat = max(m_hat, 1)
    big_m = min(2 * m_hat, m_max)

    # Flat-top lag window (Politis-Romano 1995): 1 for |s|<=1/2, taper to 0 by |s|=1.
    def _flat_top(s: float) -> float:
        a = abs(s)
        if a <= _FLAT_TOP_HALF:
            return 1.0
        if a <= 1.0:
            return 2.0 * (1.0 - a)
        return 0.0

    g_hat = 0.0
    d_inner = 0.0
    for k in range(-big_m, big_m + 1):
        lam = _flat_top(k / big_m)
        gk = gamma[abs(k)]
        g_hat += lam * abs(k) * gk
        d_inner += lam * gk
    d_sb = 2.0 * d_inner**2
    if d_sb <= 0.0 or g_hat <= 0.0:
        return 1
    b = ((2.0 * g_hat**2) / d_sb) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    b_max = math.ceil(min(3.0 * math.sqrt(n), n / 3.0))
    return max(1, min(int(b + 0.5), b_max))


def block_bootstrap(
    data: Sequence[float],
    statistic: Callable[[Sequence[float]], float],
    *,
    n_resamples: int = _DEFAULT_RESAMPLES,
    block_mean: int | None = None,
    mode: BlockMode = "stationary",
    confidence: float = 0.95,
    seed: int | None = None,
) -> BlockBootstrapResult:
    """Block-bootstrap percentile CI for `statistic` on autocorrelated `data`.

    Args:
        data: sample (e.g. per-period returns). Must be finite with >= 2 points.
        statistic: maps a sample to a scalar (e.g. `statistics.fmean` or `profit_factor`).
        n_resamples: block resamples (>= 1000; default 10000).
        block_mean: block length (>= 1, <= n). None -> Politis-White `optimal_block_length(data)`.
        mode: "stationary" (default), "moving", or "fixed".
        confidence: two-sided confidence level in (0, 1); default 0.95.
        seed: RNG seed - pass an int for bit-reproducible output.

    Raises:
        BlockBootstrapError: bad confidence / mode / block length, n_resamples < 1000, < 2 points,
            or non-finite data.
    """
    if not 0.0 < confidence < 1.0:
        raise BlockBootstrapError(f"confidence must be in (0, 1); got {confidence}")
    if mode not in _VALID_MODES:
        raise BlockBootstrapError(f"mode must be one of {_VALID_MODES}; got {mode!r}")
    if n_resamples < _MIN_RESAMPLES:
        raise BlockBootstrapError(f"n_resamples must be >= {_MIN_RESAMPLES}; got {n_resamples}")
    x = [float(v) for v in data]
    n = len(x)
    if n < _MIN_DATA:
        raise BlockBootstrapError(f"need >= {_MIN_DATA} data points; got {n}")
    if not all(math.isfinite(v) for v in x):
        raise BlockBootstrapError("data must all be finite (no NaN / inf)")

    if block_mean is None:  # explicit branch so mypy narrows block_len to int (not int | None)
        block_len = optimal_block_length(x)
        estimated = True
    else:
        block_len = block_mean
        estimated = False
    if block_len < 1 or block_len > n:
        raise BlockBootstrapError(f"block_mean must be in [1, {n}]; got {block_len}")

    point = statistic(x)
    if not math.isfinite(point):
        return BlockBootstrapResult(
            point=point,
            lower=math.nan,
            upper=math.nan,
            confidence=confidence,
            block_mean=block_len,
            block_mean_estimated=estimated,
            mode=mode,
            n_resamples=n_resamples,
            n=n,
            seed=seed,
            bootstrap_distribution=(),
            reason="point statistic is not finite (e.g. profit factor with no losses)",
        )

    # --- block bootstrap distribution; drop NaN draws, keep +/-inf (matches bca_bootstrap) ---
    rng = random.Random(seed)
    boot: list[float] = []
    for _ in range(n_resamples):
        idx = _block_indices(mode, n, block_len, rng)
        theta = statistic([x[i] for i in idx])
        if not math.isnan(theta):
            boot.append(theta)
    n_boot = len(boot)
    if n_boot < n_resamples // 2:
        n_bad = n_resamples - n_boot
        return BlockBootstrapResult(
            point=point,
            lower=math.nan,
            upper=math.nan,
            confidence=confidence,
            block_mean=block_len,
            block_mean_estimated=estimated,
            mode=mode,
            n_resamples=n_resamples,
            n=n,
            seed=seed,
            bootstrap_distribution=tuple(boot),
            reason=f"too many degenerate resamples ({n_bad} of {n_resamples} were NaN)",
        )

    alpha = 1.0 - confidence
    boot_sorted = sorted(boot)
    lower = _quantile(boot_sorted, alpha / 2.0)
    upper = _quantile(boot_sorted, 1.0 - alpha / 2.0)
    return BlockBootstrapResult(
        point=point,
        lower=lower,
        upper=upper,
        confidence=confidence,
        block_mean=block_len,
        block_mean_estimated=estimated,
        mode=mode,
        n_resamples=n_resamples,
        n=n,
        seed=seed,
        bootstrap_distribution=tuple(boot),
        reason=None,
    )


__all__: list[str] = [
    "BlockBootstrapError",
    "BlockBootstrapResult",
    "BlockMode",
    "block_bootstrap",
    "optimal_block_length",
]
