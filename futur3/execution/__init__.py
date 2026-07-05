"""futur3.execution — Broker + execution layer (Phase A1.13+).

Per internal design notes:
- `BrokerAdapter` ABC (this module) + 3 concrete impls (MockBroker for backtest,
  IBKRBrokerAdapter for paper-personal, TopstepXBrokerAdapter for funded).
- `Order` / `OrderEvent` / `Position` / `AccountMetrics` frozen dataclasses.
- `OrderType` / `Side` / `Duration` enums.
- Exception hierarchy: `BrokerError` -> {`OrderTypeUnsupportedError`,
  `BrokerNotConnectedError`, `IBKRBrokerNotImplemented`, ...}.
"""

from __future__ import annotations

from futur3.execution.broker import (
    AccountMetrics,
    BrokerAdapter,
    BrokerError,
    BrokerNotConnectedError,
    Duration,
    Order,
    OrderEvent,
    OrderType,
    OrderTypeUnsupportedError,
    Position,
    Side,
)
from futur3.execution.decay_monitor import (
    DecayStatus,
    DecayVerdict,
    assess_decay,
)
from futur3.execution.margin_source import (
    MarginSource,
    MarginSourceError,
    StaticMarginSource,
)
from futur3.execution.paper_ledger import (
    PaperLedger,
    PaperLedgerError,
    PaperSession,
)
from futur3.execution.risk_manager import (
    CONTRACT_INITIAL_MARGIN,
    CONTRACT_MULTIPLIER,
    BindingConstraint,
    RiskManager,
    RiskManagerError,
    RiskParams,
    SizingDecision,
    multiplier_for,
)
from futur3.execution.roll_executor import (
    ROOT_CYCLE,
    OpenPosition,
    RollCalendar,
    RollCalendarBuilder,
    RollCalendarEntry,
    RollDecision,
    RollDivergenceCheck,
    RollDivergenceResult,
    RollExecutor,
    RollExecutorError,
    RollMethod,
    StaticRollCalendar,
    StaticRollDivergenceCheck,
    next_contract,
    parse_contract,
)
from futur3.execution.slippage import (
    CONTRACT_TICK_SIZE,
    SlippageBand,
    SlippageError,
    SlippageModel,
    TickHaircutSlippageModel,
    UnknownSlippageBandError,
    resolve_slippage_band,
    tick_size_for,
)

__all__: list[str] = [
    "CONTRACT_INITIAL_MARGIN",
    "CONTRACT_MULTIPLIER",
    "CONTRACT_TICK_SIZE",
    "ROOT_CYCLE",
    "AccountMetrics",
    "BindingConstraint",
    "BrokerAdapter",
    "BrokerError",
    "BrokerNotConnectedError",
    "DecayStatus",
    "DecayVerdict",
    "Duration",
    "MarginSource",
    "MarginSourceError",
    "OpenPosition",
    "Order",
    "OrderEvent",
    "OrderType",
    "OrderTypeUnsupportedError",
    "PaperLedger",
    "PaperLedgerError",
    "PaperSession",
    "Position",
    "RiskManager",
    "RiskManagerError",
    "RiskParams",
    "RollCalendar",
    "RollCalendarBuilder",
    "RollCalendarEntry",
    "RollDecision",
    "RollDivergenceCheck",
    "RollDivergenceResult",
    "RollExecutor",
    "RollExecutorError",
    "RollMethod",
    "Side",
    "SizingDecision",
    "SlippageBand",
    "SlippageError",
    "SlippageModel",
    "StaticMarginSource",
    "StaticRollCalendar",
    "StaticRollDivergenceCheck",
    "TickHaircutSlippageModel",
    "UnknownSlippageBandError",
    "assess_decay",
    "multiplier_for",
    "next_contract",
    "parse_contract",
    "resolve_slippage_band",
    "tick_size_for",
]
