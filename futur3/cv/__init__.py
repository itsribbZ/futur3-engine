"""futur3.cv - time-series cross-validation + dependency-aware resampling (§4 backtesting).

Where `stats/` provides the inferential machinery (DSR/PSR/BCa/permutation/WRC/SPA), `cv/`
provides the data-partitioning + serial-dependence-handling machinery the inference runs inside:

  - block_bootstrap (§4.4) - block resampling for autocorrelated series (iid biases the CI)
  - cscv / G11 (§4.5, §18) - CSCV probability-of-backtest-overfitting gate (Bailey-LdP 2017)
  - walkforward (§4.2) - anchored / rolling strictly-causal fold generation
  - cpcv (§4.3) - combinatorial purged cross-validation (purge backward + embargo forward)

Pure stdlib throughout (matches stats/); deterministic under a seed.
"""

from __future__ import annotations

from futur3.cv.block_bootstrap import (
    BlockBootstrapError,
    BlockBootstrapResult,
    BlockMode,
    block_bootstrap,
    optimal_block_length,
)
from futur3.cv.cpcv import (
    CPCVError,
    CPCVFold,
    cpcv_splits,
)
from futur3.cv.cscv import (
    PBO_THRESHOLD,
    CSCVError,
    CSCVResult,
    cscv_pbo,
)
from futur3.cv.walkforward import (
    WalkForwardError,
    WalkForwardFold,
    WalkForwardMode,
    walk_forward_splits,
)

__all__: list[str] = [
    "PBO_THRESHOLD",
    "BlockBootstrapError",
    "BlockBootstrapResult",
    "BlockMode",
    "CPCVError",
    "CPCVFold",
    "CSCVError",
    "CSCVResult",
    "WalkForwardError",
    "WalkForwardFold",
    "WalkForwardMode",
    "block_bootstrap",
    "cpcv_splits",
    "cscv_pbo",
    "optimal_block_length",
    "walk_forward_splits",
]
