"""futur3.engine.portfolio_backtest — the multi-contract cross-sectional backtest engine.

The single-stream `BacktestEngine` runs ONE `Strategy` over ONE bar stream. A
`CrossSectionalStrategy` ranks a SET of contracts jointly
and emits a Signal per leg — which the single engine cannot express. This engine runs a
`CrossSectionalStrategy` over time-aligned per-contract bar streams against ONE `MockBroker` account
and produces a SINGLE blended `RunResult` (one daily equity curve + one flat-to-flat trade ledger),
so the EXISTING sealed gauntlet scores it unchanged — the objective's metric layer is
contract-agnostic (it reads `RunResult.equity_curve` + `trade_pnls`).

DATE-AXIS loop (the load-bearing correctness property — look-ahead-free + daily-marked):

    for date d in sorted(union of all contracts' bar dates):
        for each contract with a bar on d (deterministic date, then contract order):
            broker.advance(bar)          # fill orders from PRIOR dates at d's open
            history[contract].append(bar); last_close[contract] = bar.close
        execute any rolls dated d (all roots; paired trade -> continuous equity)
        mtm_equity = realized + sum unrealized over positions at d's closes   # ONE daily point
        signals = strategy.generate_signals(history, ctx)   # decide on d's closes
        for contract, signal in signals: size on the BLENDED equity -> order  # fills at d+1's open

An order decided on date d fills at d+1's open — never at a price already observed (bug class 4/5).
Equity is marked ONCE PER DATE (daily), so `equity_curve` len == #dates (NOT #bars, unlike the
single engine) and `periods_per_year=252` + the return distribution stay correct.

Reuses the single engine's `_mark_to_market` / `_net_position` (ONE source of truth for the MTM +
netting math) + `RunResult`; `BacktestEngine` is untouched (byte-identical). Sizing: each leg is
sized by `RiskManager` against the SHARED blended equity (the one account funds every leg) + the
per-contract Kelly/margin/leverage caps. Optional `risk_parity` re-weights each leg's Kelly by its
inverse trailing vol so a basket isn't vol-dominated; off by default. All-Decimal, det.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from decimal import Decimal
from itertools import groupby

from futur3.data.continuous import RollEvent
from futur3.data.types import ContractSymbol
from futur3.data.verifier import VerifiedBar
from futur3.engine.backtest import RunResult, _mark_to_market, _net_position
from futur3.execution.adapters.mock_broker import MockBroker
from futur3.execution.broker import Order, OrderType, Position, Side
from futur3.execution.risk_manager import RiskManager, SizingDecision, multiplier_for
from futur3.execution.slippage import SlippageModel
from futur3.execution.vol_parity import inverse_vol_weights
from futur3.runtime import RuntimeContext
from futur3.strategies.base import CrossSectionalStrategy, Signal


def _trailing_returns(bars: Sequence[VerifiedBar], lookback: int) -> list[Decimal]:
    """Simple returns of the last `lookback`+1 closes of `bars` (fewer if the stream is short)."""
    closes = [b.close for b in bars[-(lookback + 1) :]]
    return [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))]


class PortfolioBacktestEngine:
    """Runs a `CrossSectionalStrategy` over time-aligned per-contract bar streams against one
    `MockBroker`, ranking + trading all contracts jointly each date and emitting a single blended
    `RunResult` the existing sealed gauntlet scores unchanged. The date-axis loop is look-ahead-free
    and marks equity daily (see module docstring). `target_position` (default False) trades the
    delta to each leg's signed target net — a FLAT (direction 0) signal then closes that leg; legacy
    mode places the full sized clip per leg."""

    def __init__(
        self,
        *,
        ctx: RuntimeContext,
        broker: MockBroker,
        strategy: CrossSectionalStrategy,
        risk: RiskManager,
        slippage: SlippageModel | None = None,
        target_position: bool = False,
        roll_events: Sequence[RollEvent] | None = None,
        risk_parity: bool = False,
        vol_lookback: int = 60,
    ) -> None:
        self._ctx = ctx
        self._broker = broker
        self._strategy = strategy
        self._risk = risk
        self._slippage = slippage
        self._target_position = target_position
        self._risk_parity = risk_parity
        self._vol_lookback = vol_lookback
        # All roots' roll events grouped by date (multiple roots can roll on the same date).
        self._rolls_by_date: dict[date, list[RollEvent]] = {}
        for event in roll_events or ():
            self._rolls_by_date.setdefault(event.roll_date, []).append(event)

    async def run(self, histories: Mapping[ContractSymbol, Sequence[VerifiedBar]]) -> RunResult:
        """Run the cross-sectional strategy over `histories` (per-contract VerifiedBar streams)."""
        # Two axes: the mapping KEY is the ranking stream id (a root's continuous series); each
        # bar's `.contract` is the tradeable expiry (switches across rolls). History/ranking keys
        # by stream id; execution (orders/positions/MTM/rolls) keys by the expiry.
        indexed = sorted(
            ((b.ts.date(), str(key), key, b) for key, bars in histories.items() for b in bars),
            key=lambda t: (t[0], t[1]),
        )
        history: dict[ContractSymbol, list[VerifiedBar]] = {key: [] for key in histories}
        last_close: dict[ContractSymbol, Decimal] = {}
        equity_curve: list[Decimal] = []
        signals_emitted = 0
        orders = 0
        fills = 0
        rolls_executed = 0
        total_roll_cost = Decimal(0)

        for d, group in groupby(indexed, key=lambda t: t[0]):
            for _, _, key, bar in group:
                fills += len(self._broker.advance(bar, self._slippage))
                last_close[bar.contract] = bar.close
                history[key].append(bar)
            rolled_n, roll_cost = await self._execute_rolls(d)
            rolls_executed += rolled_n
            total_roll_cost += roll_cost

            metrics = await self._broker.get_account_metrics()
            positions = await self._broker.get_positions()
            mtm_equity = _mark_to_market(metrics.equity, positions, last_close)
            equity_curve.append(mtm_equity)
            if mtm_equity <= 0:  # ruined: keep marking the (ruined) curve, place no new orders
                continue

            signals = self._strategy.generate_signals(history, self._ctx)
            if not signals:
                continue
            signals_emitted += len(signals)
            weights = self._leg_weights(signals, history)
            orders += await self._place_signals(signals, positions, last_close, mtm_equity, weights)

        final_metrics = await self._broker.get_account_metrics()
        final_positions = await self._broker.get_positions()
        final_equity = equity_curve[-1] if equity_curve else final_metrics.equity
        return RunResult(
            bars_processed=len(indexed),
            signals_generated=signals_emitted,
            orders_placed=orders,
            fills=fills,
            final_equity=final_equity,
            realized_pnl=final_metrics.realized_pnl_today,
            final_positions=tuple(final_positions),
            equity_curve=tuple(equity_curve),
            rolls_executed=rolls_executed,
            total_roll_cost=total_roll_cost,
            trade_pnls=self._broker.trade_pnls,
        )

    async def _execute_rolls(self, d: date) -> tuple[int, Decimal]:
        """Execute every roll dated `d` (all roots): roll each held front position into its back
        contract at the roll's settle prices (paired trade -> continuous equity). `roll_position`
        returns 0 when flat in the front, so only held positions actually roll."""
        events = self._rolls_by_date.get(d)
        if not events:
            return 0, Decimal(0)
        rolled_total = 0
        cost_total = Decimal(0)
        for event in events:
            rolled = self._broker.roll_position(
                event.old_contract, event.new_contract, event.front_settle, event.back_settle
            )
            if rolled == 0:
                continue
            rolled_total += 1
            cost_total += (
                (event.back_settle - event.front_settle)
                * rolled
                * multiplier_for(event.new_contract)
            )
        return rolled_total, cost_total

    def _leg_weights(
        self,
        signals: Mapping[ContractSymbol, Signal],
        history: Mapping[ContractSymbol, Sequence[VerifiedBar]],
    ) -> Mapping[ContractSymbol, Decimal] | None:
        """Inverse-vol (risk-parity) weight per signalled STREAM, or None when risk_parity is off
        (legacy equal-Kelly). Vol = stdev of each stream's trailing `vol_lookback` returns."""
        if not self._risk_parity:
            return None
        return inverse_vol_weights(
            {key: _trailing_returns(history[key], self._vol_lookback) for key in signals}
        )

    async def _place_signals(
        self,
        signals: Mapping[ContractSymbol, Signal],
        positions: Sequence[Position],
        last_close: Mapping[ContractSymbol, Decimal],
        mtm_equity: Decimal,
        weights: Mapping[ContractSymbol, Decimal] | None,
    ) -> int:
        """Size + place each leg's order on the shared blended equity (orders fill next date). The
        position snapshot is pre-this-date's-orders (orders fill at the next advance), so every leg
        nets against the same current state — deterministic in sorted-contract order. The
        `weights` (risk_parity) re-scales each leg's Kelly by inverse-vol; None => equal-Kelly."""
        orders = 0
        # Execute by Signal.contract (the tradeable expiry), not the ranking key.
        for key, signal in sorted(signals.items(), key=lambda kv: str(kv[1].contract)):
            contract = signal.contract
            price = last_close.get(contract)
            if price is None:
                continue  # no close seen yet for this leg
            if signal.direction == 0:
                if self._target_position:
                    orders += await self._flatten(contract, positions)
                continue
            kelly = signal.full_kelly_fraction
            if weights is not None:
                kelly *= weights.get(key, Decimal(1))
            decision = self._risk.size_position(contract, kelly, mtm_equity, price)
            if decision.contracts <= 0:
                continue  # capped to zero (no_edge / margin / leverage) -> no trade for this leg
            resolved = self._resolve_order(decision, signal, positions, contract)
            if resolved is None:
                continue
            side, quantity = resolved
            allowed, _reason = self._broker.is_trade_allowed(contract, side, quantity)
            if not allowed:
                continue
            await self._broker.place_order(
                Order(
                    contract=contract,
                    side=side,
                    quantity=quantity,
                    order_type=OrderType.MKT,
                    client_order_id=f"{signal.strategy_id}@{signal.ts.isoformat()}#{contract}",
                )
            )
            orders += 1
        return orders

    def _resolve_order(
        self,
        decision: SizingDecision,
        signal: Signal,
        positions: Sequence[Position],
        contract: ContractSymbol,
    ) -> tuple[Side, int] | None:
        """(side, quantity) for this leg's order, or None for no trade. Target-position mode trades
        the delta to the signed net target (caps govern total per-leg leverage); legacy places the
        full sized clip."""
        if self._target_position:
            target = decision.contracts if signal.direction > 0 else -decision.contracts
            delta = target - _net_position(positions, contract)
            if delta == 0:
                return None
            return (Side.BUY if delta > 0 else Side.SELL, abs(delta))
        side = Side.BUY if signal.direction > 0 else Side.SELL
        return (side, decision.contracts)

    async def _flatten(self, contract: ContractSymbol, positions: Sequence[Position]) -> int:
        """Target-position flat-exit: close any open position in `contract` (fills next advance).
        Returns 1 if a closing order was placed, else 0."""
        net = _net_position(positions, contract)
        if net == 0:
            return 0
        side = Side.SELL if net > 0 else Side.BUY
        allowed, _reason = self._broker.is_trade_allowed(contract, side, abs(net))
        if not allowed:
            return 0
        await self._broker.place_order(
            Order(
                contract=contract,
                side=side,
                quantity=abs(net),
                order_type=OrderType.MKT,
                client_order_id=f"flat@{contract}",
            )
        )
        return 1


__all__: list[str] = [
    "PortfolioBacktestEngine",
]
