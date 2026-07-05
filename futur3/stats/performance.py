"""futur3.stats.performance - DESCRIPTIVE backtest performance metrics.

Computes the standard summary metrics from a BacktestEngine equity curve: total + annualized
return, annualized volatility, (raw) annualized Sharpe, max drawdown, Calmar.

SCOPE - DESCRIPTIVE, NOT VALIDATION. A single raw Sharpe is
"decorative - only multi-method corroboration counts." These metrics REPORT performance; they do
NOT decide whether an edge is real. The statistical VALIDATION layer (Deflated Sharpe / PSR / BCa
bootstrap / permutation) is the separate,
research-grade body of work that adjusts these raw numbers for selection bias + overfitting + non-
normality. This module is the descriptive foundation that layer builds on - never a promotion gate.

Conventions (documented because they matter):
- Per-period simple returns r_t = equity_t / equity_{t-1} - 1 over the equity curve.
- `periods_per_year` is the annualization factor at the curve's sampling frequency (a PARAMETER,
  never guessed: ~252 daily; ~252*78 RTH-5min equities; futures sessions differ - the caller, who
  knows the bar resolution + market calendar, supplies it).
- Sharpe (annualized, risk-free default 0): (mean(r) - rf/ppy) / stdev(r) * sqrt(ppy).
- Max drawdown is a POSITIVE magnitude in [0, 1] (0.15 = a 15% peak-to-trough decline).
- Stats are float-domain (ratios, not money) - the Decimal-domain rule governs prices/quantities,
  not derived statistics. Undefined statistics surface as None (never a silent 0) per the fail-loud policy.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Final

_MIN_EQUITY_POINTS: Final[int] = 2  # need >= 2 equity points to form a return
_MIN_RETURNS_FOR_STDEV: Final[int] = 2  # sample stdev needs >= 2 returns
_MIN_TRADES_FOR_SKEW: Final[int] = 3  # population skew needs >= 3 points to be meaningful


class StatsError(Exception):
    """Performance-metric computation error (degenerate / invalid equity curve)."""


@dataclass(frozen=True)
class PerformanceMetrics:
    """Descriptive performance summary of an equity curve. Ratio fields are None when undefined
    (too few points / zero variance / zero drawdown) rather than a misleading 0."""

    n_periods: int  # number of return periods (= len(equity_curve) - 1)
    total_return: float
    annualized_return: float
    annualized_volatility: float | None
    sharpe: float | None  # annualized, raw (NOT deflated - see module docstring)
    max_drawdown: float  # positive magnitude in [0, 1]
    calmar: float | None


@dataclass(frozen=True)
class TradeMetrics:
    """Per-trade (closed round-trip) summary - the foundation for win-rate / payoff analysis.
    Money fields are Decimal (the money unit); ratios + the skew are float. Undefined fields are
    None (fail-loud: never a misleading 0).

    Conventions (they matter):
    - A "trade" is one CLOSED economic bet (flat-to-flat) per `MockBroker.trade_pnls`: rolls are
      folded into the bet they continue, partial scale-outs accumulate into it, and a position still
      OPEN at run end is excluded (no realized outcome).
    - win_rate = n_wins / n_trades (a scratch trade, pnl == 0, sits in the denominator only).
    - avg_win  = mean PnL of winning trades (> 0); avg_loss = mean MAGNITUDE of losing trades (< 0).
    - payoff_ratio = avg_win / avg_loss (reward:risk b); breakeven win_rate = 1 / (1 + b).
    - expectancy = mean PnL across ALL trades = win_rate*avg_win - loss_rate*avg_loss, where
      loss_rate = n_losses / n_trades. The exact per-trade edge; positive iff profit_factor > 1.
    - profit_factor = sum(wins) / sum(|losses|); None when there are no losses (undefined/infinite),
      0.0 when there are trades but no wins. Mirrors the `bootstrap_ci.profit_factor` intent.
    - pnl_skew = population skewness of per-trade PnL (negative = fat left tail -- the structural
      cost a high win rate usually hides); None if < 3 trades or zero variance.
    """

    n_trades: int
    n_wins: int
    n_losses: int
    n_scratch: int
    win_rate: float | None
    avg_win: Decimal | None
    avg_loss: Decimal | None  # positive magnitude
    payoff_ratio: float | None
    expectancy: Decimal | None
    profit_factor: float | None
    pnl_skew: float | None


def _max_drawdown(equity: Sequence[float]) -> float:
    peak = equity[0]
    mdd = 0.0
    for x in equity:
        peak = max(peak, x)
        dd = (peak - x) / peak
        mdd = max(mdd, dd)
    return mdd


def compute_metrics(
    equity_curve: Sequence[Decimal],
    *,
    periods_per_year: float,
    risk_free_rate: float = 0.0,
) -> PerformanceMetrics:
    """Descriptive metrics for `equity_curve` (e.g. `RunResult.equity_curve`).

    Args:
        equity_curve: per-bar mark-to-market equity (strictly positive, >= 2 points).
        periods_per_year: annualization factor at the curve's sampling frequency.
        risk_free_rate: annualized risk-free rate (default 0).

    Raises:
        StatsError: fewer than 2 points, or any non-positive equity value.
    """
    if periods_per_year <= 0:
        raise StatsError(f"periods_per_year must be > 0; got {periods_per_year}")
    n = len(equity_curve)
    if n < _MIN_EQUITY_POINTS:
        raise StatsError(f"need >= 2 equity points to compute returns; got {n}")
    equity = [float(x) for x in equity_curve]
    if any(x <= 0 for x in equity):
        raise StatsError("equity curve must be strictly positive to compute returns")

    returns = [equity[i] / equity[i - 1] - 1.0 for i in range(1, n)]
    n_returns = len(returns)
    total_return = equity[-1] / equity[0] - 1.0
    annualized_return = (equity[-1] / equity[0]) ** (periods_per_year / n_returns) - 1.0
    max_dd = _max_drawdown(equity)

    volatility: float | None = None
    sharpe: float | None = None
    if n_returns >= _MIN_RETURNS_FOR_STDEV:
        mean_r = statistics.mean(returns)
        # Effectively-constant returns (incl. exact-flat) have only float-noise variance: report
        # zero volatility + UNDEFINED Sharpe. Without this guard, float rounding on identical
        # returns yields a tiny nonzero stdev (~1e-15) and a meaningless ASTRONOMICAL Sharpe
        # (~1e16) - a dangerous metric. math.isclose is the principled float-equality test.
        if all(math.isclose(r, mean_r, rel_tol=1e-9, abs_tol=1e-12) for r in returns):
            volatility = 0.0
        else:
            sd = statistics.stdev(returns)  # sample stdev (ddof=1)
            volatility = sd * math.sqrt(periods_per_year)
            mean_excess = mean_r - risk_free_rate / periods_per_year
            sharpe = (mean_excess / sd) * math.sqrt(periods_per_year)

    calmar = (annualized_return / max_dd) if max_dd > 0 else None

    return PerformanceMetrics(
        n_periods=n_returns,
        total_return=total_return,
        annualized_return=annualized_return,
        annualized_volatility=volatility,
        sharpe=sharpe,
        max_drawdown=max_dd,
        calmar=calmar,
    )


def _pnl_skew(pnls: Sequence[float]) -> float | None:
    """Population skewness (g1 = m3 / m2**1.5) of the per-trade PnL distribution, or None if there
    are < 3 trades or zero variance. Negative skew = a fat left tail (rare large losses) - the
    structural risk a high win rate is usually hiding. Population-moment convention matches
    `stats._sharpe_core.compute_sharpe_moments`."""
    n = len(pnls)
    if n < _MIN_TRADES_FOR_SKEW:
        return None
    mean = math.fsum(pnls) / n
    m2 = math.fsum((x - mean) ** 2 for x in pnls) / n
    if m2 <= 0.0:
        return None  # zero variance -> skew undefined (fail-loud: None, not 0)
    m3 = math.fsum((x - mean) ** 3 for x in pnls) / n
    return m3 / math.pow(m2, 1.5)  # math.pow -> float (m2 > 0; avoids float**float = Any)


def compute_trade_metrics(trade_pnls: Sequence[Decimal]) -> TradeMetrics:
    """Descriptive per-trade metrics from CLOSED-trade PnLs (e.g. `RunResult.trade_pnls`).

    Unlike `compute_metrics`, this does NOT raise on an empty sequence: zero trades is a legitimate
    (if useless) backtest outcome, reported as n_trades=0 with None ratios so a caller/gate can read
    "never traded" without catching an exception. Win/loss classification is exact (Decimal domain).
    """
    n = len(trade_pnls)
    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p < 0]
    n_wins = len(wins)
    n_losses = len(losses)

    win_rate = (n_wins / n) if n > 0 else None
    avg_win = (sum(wins, Decimal("0")) / n_wins) if n_wins > 0 else None
    avg_loss = (-sum(losses, Decimal("0")) / n_losses) if n_losses > 0 else None  # positive mag
    payoff_ratio = (
        float(avg_win / avg_loss) if (avg_win is not None and avg_loss is not None) else None
    )
    expectancy = (sum(trade_pnls, Decimal("0")) / n) if n > 0 else None

    gross_win = sum(wins, Decimal("0"))
    gross_loss = -sum(losses, Decimal("0"))  # positive magnitude
    profit_factor = float(gross_win / gross_loss) if gross_loss > 0 else None

    return TradeMetrics(
        n_trades=n,
        n_wins=n_wins,
        n_losses=n_losses,
        n_scratch=n - n_wins - n_losses,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        payoff_ratio=payoff_ratio,
        expectancy=expectancy,
        profit_factor=profit_factor,
        pnl_skew=_pnl_skew([float(p) for p in trade_pnls]),
    )


__all__: list[str] = [
    "PerformanceMetrics",
    "StatsError",
    "TradeMetrics",
    "compute_metrics",
    "compute_trade_metrics",
]
