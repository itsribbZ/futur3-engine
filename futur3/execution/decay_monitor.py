"""futur3.execution.decay_monitor - forward-paper decay monitor.

Watches the accumulating PaperLedger and answers the pre-live question: is the forward-paper edge
still ALIVE, or has it decayed? Over the recorded per-session net $ PnL series it computes the
full-sample Sharpe, a TRAILING-WINDOW (recent) Sharpe, the max drawdown of the cumulative-P&L curve,
and a HEALTH verdict. The pre-live health check is read from LIVE paper results -- NOT from a
backtest number -- and this monitor reads it continuously as paper sessions accumulate.

HEALTH logic: while fewer than `min_sessions` are recorded the verdict is WARMING_UP (never a false
alarm on a thin sample). Once enough sessions exist, the edge is HEALTHY iff cumulative P&L > 0 AND
the trailing Sharpe > 0; otherwise DECAYED (re-pin or shelve). Pure stdlib, deterministic;
operates on the PnL series (PaperLedger.pnl_series()), decoupled from the ledger's storage.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Final

_DEFAULT_PPY: Final[float] = 252.0  # one paper session per trading day
_DEFAULT_ROLLING_WINDOW: Final[int] = 63  # ~one quarter of trading days
_DEFAULT_MIN_SESSIONS: Final[int] = 21  # ~one month before the verdict is meaningful
_MIN_FOR_SHARPE: Final[int] = 2  # need >= 2 points for a sample stdev


class DecayVerdict(StrEnum):
    """The monitor's health verdict for the forward-paper sleeve."""

    WARMING_UP = "warming_up"  # too few sessions to judge yet
    HEALTHY = "healthy"  # cumulative > 0 AND trailing Sharpe > 0
    DECAYED = "decayed"  # cumulative <= 0 OR trailing Sharpe <= 0


@dataclass(frozen=True)
class DecayStatus:
    """The decay monitor's read of the forward-paper PnL series."""

    n_sessions: int
    cumulative_pnl: Decimal
    full_sharpe: float | None
    rolling_sharpe: float | None  # trailing `rolling_window` sessions
    rolling_window: int
    max_drawdown: Decimal  # worst peak-to-trough of the cumulative curve ($, >= 0)
    verdict: DecayVerdict
    reason: str

    @property
    def healthy(self) -> bool:
        return self.verdict is DecayVerdict.HEALTHY


def _sharpe(pnl: Sequence[float], periods_per_year: float) -> float | None:
    """Annualized Sharpe of a $ PnL series (mean/stdev * sqrt(ppy)); None if undefined."""
    n = len(pnl)
    if n < _MIN_FOR_SHARPE:
        return None
    mean = sum(pnl) / n
    var = sum((x - mean) ** 2 for x in pnl) / (n - 1)
    if var <= 0:
        return None
    return (mean / math.sqrt(var)) * math.sqrt(periods_per_year)


def _max_drawdown(pnl: Sequence[Decimal]) -> Decimal:
    """Worst peak-to-trough drop of the cumulative-PnL curve ($, >= 0)."""
    peak = cum = Decimal(0)
    mdd = Decimal(0)
    for x in pnl:
        cum += x
        peak = max(peak, cum)
        mdd = max(mdd, peak - cum)
    return mdd


def assess_decay(
    pnl: Sequence[Decimal],
    *,
    periods_per_year: float = _DEFAULT_PPY,
    rolling_window: int = _DEFAULT_ROLLING_WINDOW,
    min_sessions: int = _DEFAULT_MIN_SESSIONS,
) -> DecayStatus:
    """Assess whether the forward-paper edge (per-session net $ PnL series) is still alive."""
    n = len(pnl)
    cumulative = sum(pnl, Decimal(0))
    full_sr = _sharpe([float(x) for x in pnl], periods_per_year)
    recent = pnl[-rolling_window:]
    rolling_sr = _sharpe([float(x) for x in recent], periods_per_year)
    mdd = _max_drawdown(pnl)

    roll = "n/a" if rolling_sr is None else f"{rolling_sr:+.2f}"
    if n < min_sessions:
        verdict = DecayVerdict.WARMING_UP
        reason = f"warming up: {n} < {min_sessions} sessions -- not enough to judge"
    elif cumulative <= 0 or (rolling_sr is not None and rolling_sr <= 0):
        # decay = cumulative trends negative OR the recent Sharpe has rolled non-positive
        verdict = DecayVerdict.DECAYED
        reason = f"DECAYED: cumulative ${cumulative:,.0f}, trailing-{rolling_window} Sharpe {roll}"
    else:
        verdict = DecayVerdict.HEALTHY
        reason = f"cumulative ${cumulative:,.0f} > 0, trailing-{rolling_window} Sharpe {roll}"

    return DecayStatus(
        n_sessions=n,
        cumulative_pnl=cumulative,
        full_sharpe=full_sr,
        rolling_sharpe=rolling_sr,
        rolling_window=rolling_window,
        max_drawdown=mdd,
        verdict=verdict,
        reason=reason,
    )


__all__ = [
    "DecayStatus",
    "DecayVerdict",
    "assess_decay",
]
