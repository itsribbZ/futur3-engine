"""futur3.engine - the backtest/live orchestrator.

`BacktestEngine` runs a `Strategy` over verified bars through the position-sizing hard-gate + slippage against
a MockBroker. The signal->size->gate->order decision core is BACKTEST-IS-LIVE: a
future LiveEngine reuses it, differing only in bar source + fill arrival.
"""

from __future__ import annotations

from futur3.engine.backtest import BacktestEngine, RunResult

__all__: list[str] = [
    "BacktestEngine",
    "RunResult",
]
