"""futur3.data.intraday_continuous - roll-aware view over a PRE-SPLICED continuous intraday series.

Databento `.c.0` continuous splices the front
month across rolls but is NOT roll-adjusted: at each roll the underlying contract (instrument_id)
switches and the price jumps by the calendar spread. Measured on the 3yr hourly cache: ES ~+1.1%
per roll (contango), CL ~-0.6% (backwardation), ~13-17x a typical hourly move. Counted as a return
that jump is a phantom -- the exact roll-contamination that `continuous.py` (NUL) fixes for the
DAILY per-contract pipeline.

The NUL gold standard (`continuous.ContinuousSeries`) books the TRUE roll spread from both contracts
priced on the same roll date, so it needs OVERLAPPING per-contract data (parent symbology). A `.c.0`
continuous stream has one contract at a time (no overlap), so it CANNOT feed the NUL builder.

This module is the $0 EXPLORATION tier: keep the raw spliced prices, FLAG each roll boundary (where
the underlying contract changes), and EXCLUDE the single contaminated cross-contract return there.
Conservative -- it drops a real return too (~12-36 per 3yr, negligible vs ~17.6k bars) and does NOT
book the real roll cost, so it cannot flatter a strategy either. A survivor MUST be re-confirmed on
NUL gold-standard parent data before trading; this tier only guarantees we never MANUFACTURE a false
edge from the roll jumps. Pure stdlib + Decimal, deterministic. See `continuous.py` for NUL.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from futur3.data.types import RawBar


class IntradayContinuousError(Exception):
    """Pre-spliced continuous-series contract violation (empty, length mismatch, ts disorder)."""


def _roll_flags(bars: Sequence[RawBar]) -> tuple[bool, ...]:
    """True at index i iff `bars[i]` starts a new underlying contract (a roll boundary)."""
    if not bars:
        return ()
    flags = [False]
    flags.extend(bars[i].contract != bars[i - 1].contract for i in range(1, len(bars)))
    return tuple(flags)


@dataclass(frozen=True)
class RollExcludedSeries:
    """A pre-spliced continuous intraday series with roll boundaries flagged + the contaminated
    cross-contract return excluded -- the $0 EXPLORATION-tier roll handling (see the module
    docstring; this is NOT the NUL gold standard, which needs overlapping per-contract data).

    `roll_flags[i]` is True iff `bars[i]` is the first bar of a new underlying contract.
    """

    root: str
    bars: tuple[RawBar, ...]
    roll_flags: tuple[bool, ...]

    def __post_init__(self) -> None:
        if not self.bars:
            raise IntradayContinuousError("RollExcludedSeries has no bars")
        if len(self.bars) != len(self.roll_flags):
            raise IntradayContinuousError(
                f"bars/roll_flags length mismatch: {len(self.bars)} != {len(self.roll_flags)}"
            )
        if self.roll_flags[0]:
            raise IntradayContinuousError("roll_flags[0] must be False (first bar starts no roll)")
        for prev, cur in zip(self.bars, self.bars[1:], strict=False):
            if cur.ts <= prev.ts:
                raise IntradayContinuousError(
                    f"bars must be strictly increasing in ts; got {prev.ts} >= {cur.ts}"
                )

    @property
    def n_rolls(self) -> int:
        """Number of roll boundaries in the series."""
        return sum(self.roll_flags)

    def roll_clean_returns(self) -> tuple[float | None, ...]:
        """Per-bar simple close-to-close returns with the cross-contract return EXCLUDED (`None`)
        at each roll boundary. Length == len(bars); index 0 is `None` (no prior bar). A `None`
        marks a return that MUST NOT enter any PnL / Sharpe / signal computation."""
        out: list[float | None] = [None]
        for i in range(1, len(self.bars)):
            prev_close = float(self.bars[i - 1].close)
            if self.roll_flags[i] or prev_close == 0.0:
                out.append(None)  # roll boundary (phantom jump) or undefined base -> exclude
                continue
            out.append(float(self.bars[i].close) / prev_close - 1.0)
        return tuple(out)


def build_roll_excluded_series(root: str, bars: Sequence[RawBar]) -> RollExcludedSeries:
    """Build a `RollExcludedSeries` from a pre-spliced continuous `bars` stream for `root`.

    `bars` must be in strictly-increasing ts order, each carrying its underlying contract identity
    (which CHANGES at a roll -- the loader stamps it from the Databento instrument_id). Rolls are
    detected as contract-identity changes; `roll_clean_returns()` then excludes the contaminated
    boundary return.
    """
    ordered = tuple(bars)
    return RollExcludedSeries(root=root, bars=ordered, roll_flags=_roll_flags(ordered))


__all__ = [
    "IntradayContinuousError",
    "RollExcludedSeries",
    "build_roll_excluded_series",
]
