"""futur3.execution.adapters — Concrete BrokerAdapter implementations.

Per internal design notes-1.4`:
- `IBKRBrokerAdapter` — paper-personal + IBKR live (Phase A1.13 shell; live wiring A1.13.b).
- `TopstepXBrokerAdapter` — funded execution (Phase A1.14 STUB; full integration Phase B).
- `MockBroker` — backtest fixture (deferred — built when first consumed by engine).
"""

from __future__ import annotations

from futur3.execution.adapters.ibkr_broker import (
    IBKRBrokerAdapter,
    IBKRBrokerNotImplemented,
)
from futur3.execution.adapters.mock_broker import MockBroker
from futur3.execution.adapters.topstepx_broker import (
    TopstepXBrokerAdapter,
    TopstepXBrokerNotImplemented,
)

__all__: list[str] = [
    "IBKRBrokerAdapter",
    "IBKRBrokerNotImplemented",
    "MockBroker",
    "TopstepXBrokerAdapter",
    "TopstepXBrokerNotImplemented",
]
