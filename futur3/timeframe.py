"""futur3.timeframe -- bar-resolution -> annualization factor (`periods_per_year`).

The stats layer annualizes a Sharpe / volatility by `periods_per_year`: the number of equity-curve
return observations at the curve's sampling frequency that occur in one year. `stats.performance`
documents the convention but keeps `periods_per_year` a CALLER-supplied parameter ("never guessed:
~252 daily; ~252*78 RTH-5min equities"). This module is the single, tested place that DERIVES that
factor from a `BarResolution` + a trading calendar, so the daily->intraday rebuild has ONE
source of truth instead of the ~40 hardcoded `252`s scattered across research code.

Convention (reproduces the `stats.performance` docstring exactly):

    periods_per_year = sessions_per_year * hours_per_session * (bars per hour at the resolution)

DAY_1 / SETTLE are one bar per session (= sessions_per_year, independent of session length). Worked
check: MIN_5 at RTH (6.5h, 252d) -> 252 * 6.5 * 12 = 252 * 78 = 19656.

Session presets (`hours_per_session`, `sessions_per_year`):
- RTH_HOURS 6.5: equity-index cash session 09:30-16:00 ET (RTH-anchored ES/NQ strategies).
- GLOBEX_EQUITY_HOURS 23: ES/NQ electronic Globex session (1h daily maintenance break).
- GLOBEX_ENERGY_HOURS 22: CL electronic session.
- CRYPTO_HOURS 24 + CRYPTO_DAYS_PER_YEAR 365: 24/7 venues.

The CALLER picks the basis matching the strategy's actual return series: an RTH last-30-min trade
annualizes on RTH; a 24h-marked curve on the full session; a once-per-day trade series at DAY_1
(252) regardless of the bar resolution it was built from.
"""

from __future__ import annotations

from typing import Final

from futur3.data.types import BarResolution

RTH_HOURS: Final[float] = 6.5  # equity-index cash session (09:30-16:00 ET = 6.5h)
GLOBEX_EQUITY_HOURS: Final[float] = 23.0  # ES/NQ electronic session (1h daily break)
GLOBEX_ENERGY_HOURS: Final[float] = 22.0  # CL electronic session
CRYPTO_HOURS: Final[float] = 24.0

TRADING_DAYS_PER_YEAR: Final[int] = 252
CRYPTO_DAYS_PER_YEAR: Final[int] = 365

# Bars per hour for each sub-hourly resolution; DAY_1 / SETTLE handled separately (one per session).
_BARS_PER_HOUR: Final[dict[BarResolution, float]] = {
    BarResolution.SEC_1: 3600.0,
    BarResolution.SEC_5: 720.0,
    BarResolution.MIN_1: 60.0,
    BarResolution.MIN_5: 12.0,
    BarResolution.MIN_15: 4.0,
    BarResolution.HOUR_1: 1.0,
}


def resolve_ppy(
    resolution: BarResolution,
    *,
    hours_per_session: float = RTH_HOURS,
    sessions_per_year: float = TRADING_DAYS_PER_YEAR,
) -> float:
    """Periods-per-year annualization factor for an equity curve sampled at `resolution`.

    Args:
        resolution: bar granularity of the equity-curve / return series being annualized.
        hours_per_session: trading hours in one session (default `RTH_HOURS` 6.5; pass
            `GLOBEX_EQUITY_HOURS` / `GLOBEX_ENERGY_HOURS` / `CRYPTO_HOURS` for electronic / 24-7).
            Ignored for DAY_1 / SETTLE (one bar per session).
        sessions_per_year: trading sessions per year (default 252; `CRYPTO_DAYS_PER_YEAR` 365).

    Returns:
        `periods_per_year` (> 0) for `stats.performance.compute_metrics` and the gauntlet.

    Raises:
        ValueError: `hours_per_session` or `sessions_per_year` <= 0, or an unmapped resolution.
    """
    if hours_per_session <= 0:
        raise ValueError(f"hours_per_session must be > 0; got {hours_per_session}")
    if sessions_per_year <= 0:
        raise ValueError(f"sessions_per_year must be > 0; got {sessions_per_year}")
    if resolution in (BarResolution.DAY_1, BarResolution.SETTLE):
        return float(sessions_per_year)
    bars_per_hour = _BARS_PER_HOUR.get(resolution)
    if bars_per_hour is None:
        raise ValueError(f"no periods_per_year mapping for resolution {resolution!r}")
    return float(sessions_per_year) * float(hours_per_session) * bars_per_hour


__all__ = [
    "CRYPTO_DAYS_PER_YEAR",
    "CRYPTO_HOURS",
    "GLOBEX_ENERGY_HOURS",
    "GLOBEX_EQUITY_HOURS",
    "RTH_HOURS",
    "TRADING_DAYS_PER_YEAR",
    "resolve_ppy",
]
