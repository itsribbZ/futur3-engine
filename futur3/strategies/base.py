"""futur3.strategies.base - Strategy ABC + Signal contract (strategy-layer foundation).

The per-strategy seam: each strategy consumes a cross-source-verified bar history and emits a
`Signal` for the latest bar. The composite/decision layer aggregates
many strategies' signals into one trade decision; this module ships the single-strategy contract
those aggregators (and the backtest engine) consume.

Signal field semantics per the decision layer:
- `direction` is the discrete side in {-1 short, 0 flat, +1 long} (convention: `signal_i ∈ [-1,+1]`);
- `confidence ∈ [0, 1]` is the strategy's self-confidence (from its primary model probability).

DESIGN DECISIONS (UNSPECIFIED in docs as a Python contract - fixed + documented here):
1. `full_kelly_fraction` carries the NON-NEGATIVE Kelly edge magnitude f* = mu/sigma^2
   (f* = mu/sigma^2, Kelly 1956); the SIGN lives in `direction`. This is forced by the
   sizing contract: `RiskManager.size_position` treats full_kelly_fraction <= 0 as `no_edge` and
   returns 0 contracts (`risk_manager.py:170`), so a SHORT must emit direction=-1 with a POSITIVE
   magnitude - never a negative f*. The engine reads `direction` to choose BUY/SELL and
   `full_kelly_fraction` to size.
2. A strategy returns `Signal | None`: None = "no actionable signal" (insufficient history / no
   edge). A `Signal` with direction==0 is an explicit-flat view and MUST carry zero magnitude.
3. `generate_signal` consumes a chronological `VerifiedBar` history (post-verifier, not RawBar,
   per the decision-layer design) + the `RuntimeContext` so any wall-time read goes through
   `ctx.clock` (BACKTEST-IS-LIVE - never `datetime.now()`).
"""

from __future__ import annotations

import abc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from futur3.data.types import ContractSymbol
from futur3.data.verifier import VerifiedBar
from futur3.runtime import RuntimeContext


class StrategyError(Exception):
    """Strategy-layer error (malformed Signal, misconfigured strategy params)."""


@dataclass(frozen=True)
class Signal:
    """One strategy's directional signal for a (contract, ts). Immutable + hashable."""

    contract: ContractSymbol
    ts: datetime
    strategy_id: str
    direction: int  # -1 short / 0 flat / +1 long
    full_kelly_fraction: Decimal  # >= 0; Kelly edge magnitude f* for sizing (sign is in direction)
    confidence: Decimal  # [0, 1]

    def __post_init__(self) -> None:
        if self.direction not in (-1, 0, 1):
            raise StrategyError(f"Signal.direction must be -1/0/+1; got {self.direction}")
        if self.full_kelly_fraction < 0:
            raise StrategyError(
                f"Signal.full_kelly_fraction must be >= 0 (sign lives in direction); "
                f"got {self.full_kelly_fraction}"
            )
        if not (Decimal(0) <= self.confidence <= Decimal(1)):
            raise StrategyError(f"Signal.confidence must be in [0, 1]; got {self.confidence}")
        if self.direction == 0 and self.full_kelly_fraction != 0:
            raise StrategyError(
                "Signal: direction==0 (flat) requires full_kelly_fraction==0; "
                f"got {self.full_kelly_fraction}"
            )


class Strategy(abc.ABC):
    """Abstract single-strategy signal generator. Concrete strategies implement
    `generate_signal`; the composite layer + backtest engine consume them through this interface."""

    @property
    @abc.abstractmethod
    def strategy_id(self) -> str:
        """Stable strategy id (e.g. "example_momentum") - composite-layer key + audit tag."""

    @abc.abstractmethod
    def generate_signal(self, history: Sequence[VerifiedBar], ctx: RuntimeContext) -> Signal | None:
        """Emit a Signal for the LATEST bar in `history`, or None if no actionable signal.

        Args:
            history: VerifiedBars in ascending-ts order; the last element is the current bar.
                     Strategies needing N lookback bars return None until enough history exists.
            ctx: RuntimeContext - clock source (backtest-is-live: read wall-time via ctx.clock.now_utc(),
                 never datetime.now()) + mode.

        Returns:
            Signal for `history[-1]`, or None.
        """


class CrossSectionalStrategy(abc.ABC):
    """Abstract cross-sectional (portfolio) strategy: ranks a SET of contracts jointly and emits a
    Signal per contract.

    Unlike `Strategy` (one contract's history -> one Signal), a cross-sectional strategy needs
    every contract's history at once - e.g. a ranking strategy longs the top-ranked contracts and shorts the
    bottom-ranked, a view the single-contract ABC cannot express. This is a SIBLING ABC, not a
    subclass: the two have genuinely different shapes.

    NOT yet consumed by the single-stream `BacktestEngine` (which runs one `Strategy` over one bar
    stream). A multi-contract engine - the composite / decision layer (Phase C) - wires it; until
    then it stands alone with its own tests, the same build-ahead-of-wiring discipline the PBO / FWER
    gates follow (advisory until `promotion_gate` integration).
    """

    @property
    @abc.abstractmethod
    def strategy_id(self) -> str:
        """Stable strategy id - composite-layer key + audit tag."""

    @abc.abstractmethod
    def generate_signals(
        self,
        histories: Mapping[ContractSymbol, Sequence[VerifiedBar]],
        ctx: RuntimeContext,
    ) -> dict[ContractSymbol, Signal]:
        """Emit a Signal for each ranked contract (the long + short legs), keyed by contract.

        Args:
            histories: per-contract VerifiedBar histories in ascending-ts order. Contracts with
                insufficient history are dropped; if fewer than 2 rankable contracts remain the
                cross-section is undefined and {} is returned.
            ctx: RuntimeContext - clock source (backtest-is-live) + mode.

        Returns:
            {contract: Signal} for the emitted legs; {} when no actionable cross-section exists.
        """


__all__: list[str] = [
    "CrossSectionalStrategy",
    "Signal",
    "Strategy",
    "StrategyError",
]
