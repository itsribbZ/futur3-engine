"""futur3.contracts — Contract & market metadata.

Per the build plan. The home for CME market metadata used by the
roll-handling layer:
- `CMETradingCalendar` (this ship) — computed CME full-closure trading-day calendar; the
  calendar `roll_executor.py` needs for deriving roll_target/roll_deadline as T-N *trading*
  days from LTD/FND.
"""

from __future__ import annotations

from futur3.contracts.trading_calendar import (
    CMETradingCalendar,
    TradingCalendarError,
)

__all__: list[str] = [
    "CMETradingCalendar",
    "TradingCalendarError",
]
