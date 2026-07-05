"""PropAwareMockBroker — prop-account risk-limit overlay on MockBroker.

Adds a pre-trade hard-gate enforcing the three survival limits a trailing-
drawdown prop firm (Topstep et al.) imposes, WITHOUT touching the underlying
MockBroker wire. It exists so forward runs can be measured at a deployable,
risk-guarded size profile rather than at a scale-free account with no guard.

Three gates (all on RISK-INCREASING orders only):

- **Position cap** — the *resulting* position may not exceed ``max_contracts``
  (deployable size, e.g. 1 MNQ-equivalent).
- **Daily Loss Limit (DLL)** — halt once the session's realized loss reaches
  ``daily_loss_limit``.
- **Trailing Max Loss Limit (MLL)** — halt once equity falls
  ``trailing_max_loss_limit`` below its trailing high-water mark.

## Compatibility

The base ``MockBroker`` is UNCHANGED; this subclass is used ONLY by new
prop-scale forward runs, so existing regression artifacts that instantiate the
plain ``MockBroker`` stay byte-identical.

## Design notes

- **Risk-reducing orders are ALWAYS permitted** — even when halted. You must be
  able to flatten/trim a position regardless of limit state; only risk-
  INCREASING (open/add) orders are gated. Risk-reducing = the order strictly
  shrinks ``|position|``.
- **Halts LATCH** — once DLL or trailing-MLL trips, the broker stays halted for
  the rest of its life (Topstep terminates the account on MLL breach; DLL locks
  for the day). ``reset()`` clears it. Each forward session spawns a fresh
  broker, so the DLL naturally scopes to one session ("today"); the account-
  level trailing peak is carried across sessions via ``initial_equity_peak``.
- **Trailing peak** updates whenever equity is observed — on every
  ``is_trade_allowed`` (increasing) check and every ``get_account_metrics()``
  poll (the engine marks-to-market each bar, keeping the trail faithful).
- All limits are validated ``> 0`` at construction (fail-loud; no silent
  defaults).
"""

from __future__ import annotations

from decimal import Decimal

from futur3.data.types import ContractSymbol
from futur3.execution.adapters.mock_broker import (
    _DEFAULT_ACCOUNT_ID,
    _DEFAULT_STARTING_EQUITY,
    MockBroker,
)
from futur3.execution.broker import AccountMetrics, OrderType, Side


class PropAwareMockBroker(MockBroker):
    """MockBroker + DLL / trailing-MLL / position-cap pre-trade gates.

    Behaviour is identical to MockBroker for any order that (a) is risk-reducing
    or (b) sits within all three limits. The proven replay path never uses this
    class; canonical regression artifacts are external to this repo and unaffected.
    """

    BROKER_ID: str = "prop-mock"

    def __init__(
        self,
        *,
        daily_loss_limit: Decimal,
        trailing_max_loss_limit: Decimal,
        max_contracts: int,
        initial_equity_peak: Decimal | None = None,
        starting_equity: Decimal = _DEFAULT_STARTING_EQUITY,
        account_id: str = _DEFAULT_ACCOUNT_ID,
        block_all_trades: bool = False,
        restrict_supported_to: frozenset[OrderType] | None = None,
        clock_now_fn: object | None = None,
    ) -> None:
        super().__init__(
            starting_equity=starting_equity,
            account_id=account_id,
            block_all_trades=block_all_trades,
            restrict_supported_to=restrict_supported_to,
            clock_now_fn=clock_now_fn,
        )
        if daily_loss_limit <= 0:
            raise ValueError(f"daily_loss_limit must be > 0; got {daily_loss_limit}")
        if trailing_max_loss_limit <= 0:
            raise ValueError(
                f"trailing_max_loss_limit must be > 0; got {trailing_max_loss_limit}"
            )
        if max_contracts < 1:
            raise ValueError(f"max_contracts must be >= 1; got {max_contracts}")
        if initial_equity_peak is not None and initial_equity_peak < starting_equity:
            raise ValueError(
                f"initial_equity_peak ({initial_equity_peak}) must be >= "
                f"starting_equity ({starting_equity})"
            )
        self._daily_loss_limit = daily_loss_limit
        self._trailing_max_loss_limit = trailing_max_loss_limit
        self._max_contracts = max_contracts
        self._initial_equity_peak: Decimal = (
            initial_equity_peak if initial_equity_peak is not None else starting_equity
        )
        self._equity_peak: Decimal = self._initial_equity_peak
        self._halted: str | None = None

    # ----- Introspection (read-only) ---------------------------------------

    @property
    def equity_peak(self) -> Decimal:
        return self._equity_peak

    @property
    def halt_reason(self) -> str | None:
        return self._halted

    @property
    def max_contracts(self) -> int:
        return self._max_contracts

    @property
    def daily_loss_limit(self) -> Decimal:
        return self._daily_loss_limit

    @property
    def trailing_max_loss_limit(self) -> Decimal:
        return self._trailing_max_loss_limit

    # ----- Internals -------------------------------------------------------

    def _current_equity(self) -> Decimal:
        """Synchronous mirror of get_account_metrics' equity formula."""
        unrealized = sum(
            (p.unrealized_pnl for p in self._positions.values()),
            Decimal("0"),
        )
        return self._starting_equity + self._realized_pnl_today + unrealized

    def _observe(self, equity: Decimal) -> None:
        self._equity_peak = max(self._equity_peak, equity)

    # ----- Gated interface -------------------------------------------------

    def is_trade_allowed(
        self,
        contract: ContractSymbol,
        side: Side,
        quantity: int,
    ) -> tuple[bool, str | None]:
        base_ok, reason = super().is_trade_allowed(contract, side, quantity)
        if not base_ok:
            return base_ok, reason

        existing = self._positions.get(contract)
        current_qty = existing.quantity if existing is not None else 0
        delta = quantity if side == Side.BUY else -quantity
        resulting_qty = current_qty + delta

        # Risk-REDUCING orders (strictly shrink |position|) are ALWAYS allowed —
        # you must be able to flatten/trim even when the account is halted.
        if abs(resulting_qty) < abs(current_qty):
            return True, None

        # Risk-increasing (open / add) from here — apply the prop gates.
        block = self._increasing_block_reason(abs(resulting_qty))
        if block is not None:
            return False, f"PropGuard: {block}"
        return True, None

    def _increasing_block_reason(self, resulting_abs_qty: int) -> str | None:
        """Reason a risk-increasing order is blocked, or None if permitted.

        Side effects: refreshes the trailing equity peak and LATCHES
        ``_halted`` on a DLL or trailing-MLL breach.
        """
        if self._halted is not None:
            return f"account halted ({self._halted})"
        if resulting_abs_qty > self._max_contracts:
            return (
                f"resulting position {resulting_abs_qty} exceeds "
                f"max_contracts {self._max_contracts}"
            )
        equity = self._current_equity()
        self._observe(equity)
        if self._realized_pnl_today <= -self._daily_loss_limit:
            self._halted = (
                f"DLL breached: realized {self._realized_pnl_today} "
                f"<= -{self._daily_loss_limit}"
            )
            return self._halted
        drawdown = self._equity_peak - equity
        if drawdown >= self._trailing_max_loss_limit:
            self._halted = (
                f"trailing MLL breached: drawdown {drawdown} "
                f">= {self._trailing_max_loss_limit}"
            )
            return self._halted
        return None

    async def get_account_metrics(self) -> AccountMetrics:
        metrics = await super().get_account_metrics()
        self._observe(metrics.equity)
        return metrics

    def reset(self) -> None:
        super().reset()
        self._equity_peak = self._initial_equity_peak
        self._halted = None
