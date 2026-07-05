"""futur3.engine.backtest - BacktestEngine: the runnable backtest orchestrator.

Closes the loop the data/verifier/broker triumvirate made wireable: per bar it fills prior orders,
asks the strategy for a signal, sizes it through the position-sizing hard-gate, checks the broker trade gate,
and places a market order. The signal -> size -> gate -> order DECISION core is mode-agnostic (
BACKTEST-IS-LIVE) - a future LiveEngine reuses it verbatim, differing only in where bars come from
(a stream) and how fills arrive (the broker's event stream rather than `advance`).

LOOK-AHEAD-FREE loop ordering (the load-bearing correctness property):
    for bar in bars:
        broker.advance(bar)          # fill orders from PRIOR bars at THIS bar's open
        history.append(bar)
        signal = strategy.generate_signal(history, ctx)   # decide on THIS bar (its close)
        ... size + gate ... -> place_order               # fills at the NEXT bar's advance
An order decided on bar t's close therefore fills at bar t+1's open - never at a price the
strategy already observed (bug class 4/5). An order placed on the final bar never fills (no
next bar) - realistic; RunResult reflects state as of the last bar.

NOTE: this takes a concrete `MockBroker` (not the `BrokerAdapter` ABC) because backtest fills
are driven by `MockBroker.advance`, which is intentionally NOT on the ABC (live brokers fill from
the real market). Every OTHER broker call here (`get_account_metrics`, `is_trade_allowed`,
`place_order`, `get_positions`) is an ABC method, so the decision core stays adapter-agnostic.

MARK-TO-MARKET: MockBroker reports realized PnL only (its unrealized_pnl is always 0). The engine
marks open positions to market itself each bar — unrealized = sum over positions of
(last_close[contract] - avg_entry) * signed_qty * multiplier — to produce a per-bar `equity_curve`
(for the Phase 8 Sharpe/drawdown stats) and to size on the CURRENT account value (MTM equity), not
realized-only. Multi-contract safe: each held contract is marked at its own last-seen close.

SURVIVAL GATE (opt-in): with `survival_gate=True` the loop feeds each contract's trailing
close-to-close price returns into `size_position(returns=...)`, activating the MATH §6 leverage-
survival bootstrap as a 4th tighten-only cap that de-levers a position whose leverage would risk
account ruin (the failure mode real BTC at 3x exposed). DEFAULT OFF — `survival_gate=False` calls
`size_position` exactly as before (returns=None), so the gate is dormant and backtests are
byte-identical. The returns fed are UNIT-EXPOSURE price returns (the bootstrap scales them by the
proposed leverage); feeding the already-leveraged equity-curve returns would double-count leverage.

TARGET-POSITION MODE (opt-in): with `target_position=True` the loop trades the DELTA to a
signed TARGET net position each bar instead of placing the full sized clip. The gate-capped size IS
the target TOTAL exposure, so the survival/leverage caps govern total position leverage and a
persistent-signal strategy can't ruin the account by stacking a fresh clip every bar - the dominant
ruin driver real BTC daily exposed (the per-bar survival cap alone can't tame a position built from
124 individually-"survivable" clips). DEFAULT OFF preserves the legacy clip-placing behavior, so
existing backtests stay byte-identical; the two opt-in flags compose: target-position + survival
gate = one bounded position whose total leverage is gate-capped to survivable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Final

from futur3.data.continuous import RollEvent
from futur3.data.types import ContractSymbol
from futur3.data.verifier import VerifiedBar
from futur3.execution.adapters.mock_broker import MockBroker
from futur3.execution.broker import Order, OrderType, Position, Side
from futur3.execution.risk_manager import RiskManager, SizingDecision, multiplier_for
from futur3.execution.slippage import SlippageModel
from futur3.runtime import RuntimeContext
from futur3.strategies.base import Signal, Strategy

_MIN_CLOSES_FOR_SURVIVAL: Final[int] = 3  # >=3 closes -> >=2 returns (bootstrap minimum)


@dataclass(frozen=True)
class RunResult:
    """Deterministic outcome of a backtest run (same bars + strategy -> same result)."""

    bars_processed: int
    signals_generated: int  # non-None signals the strategy emitted
    orders_placed: int
    fills: int  # orders that auto-filled via advance (an order on the last bar never fills)
    final_equity: Decimal  # mark-to-market (realized + unrealized) as of the last bar
    realized_pnl: Decimal
    final_positions: tuple[Position, ...]
    equity_curve: tuple[Decimal, ...]  # MTM equity after each processed bar (len == bars_processed)
    survival_capped_bars: int = 0  # bars the survival cap strictly de-levered (0 = gate off)
    rolls_executed: int = 0  # contract rolls performed (0 = no roll_events or nothing to roll)
    total_roll_cost: Decimal = Decimal(0)  # G9 diagnostic: sum (back-front)*qty*mult over rolls
    trade_pnls: tuple[Decimal, ...] = ()  # PnL of each CLOSED bet (flat-to-flat) -> win rate


def _mark_to_market(
    realized_equity: Decimal,
    positions: Sequence[Position],
    last_close: dict[ContractSymbol, Decimal],
) -> Decimal:
    """MTM account equity = realized equity + unrealized PnL of open positions at last-seen close.

    `realized_equity` is MockBroker's equity (starting + realized; its own unrealized is always 0),
    so this adds the real unrealized: sum of (price - avg_entry) * signed_qty * multiplier.
    """
    unrealized = Decimal(0)
    for p in positions:
        if p.quantity == 0:
            continue
        price = last_close.get(p.contract)
        if price is None:  # never marked (shouldn't happen - a fill implies a seen bar)
            continue
        unrealized += (price - p.avg_entry_price) * p.quantity * multiplier_for(p.contract)
    return realized_equity + unrealized


def _net_position(positions: Sequence[Position], contract: ContractSymbol) -> int:
    """Net signed quantity in `contract` (0 if flat); the base for the target-position delta."""
    return sum((p.quantity for p in positions if p.contract == contract), 0)


class BacktestEngine:
    """Runs a `Strategy` over a fixed `VerifiedBar` sequence against a MockBroker, with hard-gated sizing
    (Kelly/margin/leverage + a survival cap), opt-in target-position netting, and slippage. The
    signal->size->gate->order core is the mode-agnostic engine logic."""

    def __init__(
        self,
        *,
        ctx: RuntimeContext,
        broker: MockBroker,
        strategy: Strategy,
        risk: RiskManager,
        slippage: SlippageModel | None = None,
        survival_gate: bool = False,
        survival_seed: int = 0,
        target_position: bool = False,
        roll_events: Sequence[RollEvent] | None = None,
        stop_loss_pct: Decimal | None = None,
    ) -> None:
        """Wire the backtest. `survival_gate` (default False) activates the leverage-survival
        cap (MATH §6) in the loop by feeding each contract's trailing price returns to the sizer;
        OFF reproduces the original 3-cap behavior exactly. `survival_seed` is the base seed for the
        per-bar survival bootstrap (seed = survival_seed + bar_index) for determinism.
        `target_position` (default False) trades the DELTA to a signed target net position each bar
        instead of placing the full sized clip; ON makes the gate govern TOTAL leverage and prevents
        clip-stacking ruin. Both flags default OFF -> byte-identical to the legacy engine.

        `stop_loss_pct` (default None=off): flatten an open position whose adverse move from its
        average entry exceeds this fraction -- caps mean-reversion's negative-skew tail.
        """
        self._ctx = ctx
        self._broker = broker
        self._strategy = strategy
        self._risk = risk
        self._slippage = slippage
        self._survival_gate = survival_gate
        self._survival_seed = survival_seed
        self._target_position = target_position
        if stop_loss_pct is not None and stop_loss_pct <= 0:
            raise ValueError(f"stop_loss_pct must be > 0 when set; got {stop_loss_pct}")
        self._stop_loss_pct = stop_loss_pct
        # W1.5: NUL roll events keyed by roll date; on that bar the held position rolls into the
        # new contract at the roll's settle prices. None/empty -> no rolling (byte-identical).
        self._roll_by_date: dict[date, RollEvent] = (
            {e.roll_date: e for e in roll_events} if roll_events else {}
        )

    # The per-bar pipeline (advance->MTM->ruin->stop->signal->size->gate->order) is one cohesive,
    # look-ahead-free sequence; the docstring treats its ordering as the load-bearing correctness
    # property, so run() is kept whole and PLR0915 is suppressed rather than split into pieces.
    async def run(self, bars: Sequence[VerifiedBar]) -> RunResult:  # noqa: PLR0915
        history: list[VerifiedBar] = []
        last_close: dict[ContractSymbol, Decimal] = {}
        closes_by_contract: dict[ContractSymbol, list[Decimal]] = {}
        equity_curve: list[Decimal] = []
        signals = 0
        orders = 0
        fills = 0
        survival_capped = 0
        rolls_executed = 0
        total_roll_cost = Decimal(0)

        for i, bar in enumerate(bars):
            # 1. Fill orders from prior bars against THIS bar (MKT at open) - look-ahead-free.
            fills += len(self._broker.advance(bar, self._slippage))
            last_close[bar.contract] = bar.close

            # W1.5: roll the held position into the new contract at the roll's settle prices
            # (paired trade) so equity stays continuous - no phantom jump.
            rolled_n, roll_cost = await self._execute_roll(bar)
            rolls_executed += rolled_n
            total_roll_cost += roll_cost

            if self._survival_gate:  # track per-contract closes for the survival gate
                closes_by_contract.setdefault(bar.contract, []).append(bar.close)

            # 2. Mark to market at this bar's close (post-advance positions): this MTM equity is
            #    both the bar's equity-curve point AND the account value the sizing gate uses.
            metrics = await self._broker.get_account_metrics()
            positions = await self._broker.get_positions()
            mtm_equity = _mark_to_market(metrics.equity, positions, last_close)
            equity_curve.append(mtm_equity)
            history.append(bar)

            # A ruined account (a leveraged loss wiped MTM equity to <= 0) can't size a position
            # - size_position requires equity > 0. Stop trading; keep marking the (ruined) curve.
            # The engine does not model forced liquidation, so an open position keeps marking but
            # NO new orders are placed once ruined (real BTC at 3x leverage exposed this path).
            if mtm_equity <= 0:
                continue

            # Stop-loss overlay: cut a position whose adverse move from entry exceeds the stop
            # (caps the negative-skew tail). Off (None) -> byte-identical; fills next bar.
            if self._stop_loss_pct is not None and await self._apply_stop_loss(bar, positions):
                orders += 1
                continue  # exited on the stop; ignore this bar's signal

            # 3. Decide on this bar.
            signal = self._strategy.generate_signal(history, self._ctx)
            if signal is None:
                continue
            signals += 1
            if signal.direction == 0:
                # Under target-position netting a FLAT signal flattens the open position -- the
                # mean-reversion exit-at-mean. Legacy mode holds (byte-identical).
                if self._target_position:
                    orders += await self._flatten_on_flat(signal, positions, bar)
                continue

            # 4. Size through the sizing hard-gate (Kelly/margin/leverage [+ survival]) on MTM equity.
            #    When on, the survival gate uses the contract's trailing returns to de-lever a
            #    ruinous position (see `_survival_inputs`).
            returns, seed = self._survival_inputs(closes_by_contract, signal.contract, i)
            decision = self._risk.size_position(
                signal.contract,
                signal.full_kelly_fraction,
                mtm_equity,
                bar.close,
                returns=returns,
                seed=seed,
            )
            if decision.binding_constraint == "survival":
                survival_capped += 1  # the survival cap strictly de-levered this bar (audit counter)
            if decision.contracts <= 0:
                continue  # capped to zero (no_edge / leverage / margin / survival) -> no trade

            # 5. Resolve the order (target-position delta vs legacy full clip); None -> no trade.
            resolved = self._resolve_order(decision, signal, positions)
            if resolved is None:
                continue
            side, quantity = resolved

            # 6. Broker trade gate, then place a market order (fills at the next bar's advance).
            allowed, _reason = self._broker.is_trade_allowed(signal.contract, side, quantity)
            if not allowed:
                continue
            await self._broker.place_order(
                Order(
                    contract=signal.contract,
                    side=side,
                    quantity=quantity,
                    order_type=OrderType.MKT,
                    client_order_id=f"{signal.strategy_id}@{bar.ts.isoformat()}",
                )
            )
            orders += 1

        final_metrics = await self._broker.get_account_metrics()
        final_positions = await self._broker.get_positions()
        final_equity = equity_curve[-1] if equity_curve else final_metrics.equity
        return RunResult(
            bars_processed=len(bars),
            signals_generated=signals,
            orders_placed=orders,
            fills=fills,
            final_equity=final_equity,
            realized_pnl=final_metrics.realized_pnl_today,
            final_positions=tuple(final_positions),
            equity_curve=tuple(equity_curve),
            survival_capped_bars=survival_capped,
            rolls_executed=rolls_executed,
            total_roll_cost=total_roll_cost,
            trade_pnls=self._broker.trade_pnls,
        )

    async def _execute_roll(self, bar: VerifiedBar) -> tuple[int, Decimal]:
        """W1.5: roll the held position into the new contract at the roll's settle prices when
        `bar` falls on a roll date (paired trade -> continuous equity, no phantom jump). Returns
        (rolls_executed_delta, roll_cost_delta); (0, Decimal(0)) when this bar is not a roll date,
        no position is held in the expiring contract, or the roll moved nothing."""
        if not self._roll_by_date:
            return 0, Decimal(0)
        event = self._roll_by_date.get(bar.ts.date())
        if event is None:
            return 0, Decimal(0)
        pre_roll = await self._broker.get_positions()
        if _net_position(pre_roll, event.old_contract) == 0:
            return 0, Decimal(0)
        rolled = self._broker.roll_position(
            event.old_contract,
            event.new_contract,
            event.front_settle,
            event.back_settle,
        )
        if rolled == 0:
            return 0, Decimal(0)
        cost = (
            (event.back_settle - event.front_settle) * rolled * multiplier_for(event.new_contract)
        )
        return 1, cost

    def _survival_inputs(
        self,
        closes_by_contract: dict[ContractSymbol, list[Decimal]],
        contract: ContractSymbol,
        bar_index: int,
    ) -> tuple[list[float] | None, int | None]:
        """Trailing UNIT-EXPOSURE price returns + a per-bar seed for the survival gate.

        Returns (None, None) when the gate is off or fewer than `_MIN_CLOSES_FOR_SURVIVAL` closes
        exist (too little history for the bootstrap). These are the contract's own price returns -
        the bootstrap scales them by the proposed leverage, so feeding the leveraged equity-curve
        returns would double-count leverage. The per-bar seed keeps the run deterministic.
        """
        if not self._survival_gate:
            return None, None
        closes = closes_by_contract.get(contract)
        if closes is None or len(closes) < _MIN_CLOSES_FOR_SURVIVAL:
            return None, None
        returns = [float(closes[j] / closes[j - 1] - 1) for j in range(1, len(closes))]
        return returns, self._survival_seed + bar_index

    def _resolve_order(
        self, decision: SizingDecision, signal: Signal, positions: Sequence[Position]
    ) -> tuple[Side, int] | None:
        """(side, quantity) for this bar's order, or None for no trade.

        TARGET-POSITION mode trades the DELTA to a signed net target: the gate-capped size IS the
        target total exposure, so the caps govern TOTAL leverage and the account can't blow up by
        stacking a fresh clip every bar. Legacy default places the full sized clip.
        """
        if self._target_position:
            target = decision.contracts if signal.direction > 0 else -decision.contracts
            delta = target - _net_position(positions, signal.contract)
            if delta == 0:
                return None
            return (Side.BUY if delta > 0 else Side.SELL, abs(delta))
        side = Side.BUY if signal.direction > 0 else Side.SELL
        return (side, decision.contracts)

    async def _flatten_on_flat(
        self, signal: Signal, positions: Sequence[Position], bar: VerifiedBar
    ) -> int:
        """Flat-exit (target-position only): close any open position in `signal.contract`,
        filling at the next advance. Returns 1 if a closing order was placed, else 0."""
        net = _net_position(positions, signal.contract)
        if net == 0:
            return 0
        close_side = Side.SELL if net > 0 else Side.BUY
        allowed, _reason = self._broker.is_trade_allowed(signal.contract, close_side, abs(net))
        if not allowed:
            return 0
        await self._broker.place_order(
            Order(
                contract=signal.contract,
                side=close_side,
                quantity=abs(net),
                order_type=OrderType.MKT,
                client_order_id=f"{signal.strategy_id}@{bar.ts.isoformat()}",
            )
        )
        return 1

    async def _apply_stop_loss(self, bar: VerifiedBar, positions: Sequence[Position]) -> bool:
        """Stop-loss: if the open position in `bar.contract` is adverse beyond stop_loss_pct of
        its avg entry, place a flattening order (fills next advance). Returns True iff it fired."""
        pos = next((p for p in positions if p.contract == bar.contract and p.quantity != 0), None)
        if pos is None or pos.avg_entry_price <= 0 or self._stop_loss_pct is None:
            return False
        if pos.quantity > 0:
            adverse = (pos.avg_entry_price - bar.close) / pos.avg_entry_price
        else:
            adverse = (bar.close - pos.avg_entry_price) / pos.avg_entry_price
        if adverse < self._stop_loss_pct:
            return False
        close_side = Side.SELL if pos.quantity > 0 else Side.BUY
        allowed, _reason = self._broker.is_trade_allowed(
            bar.contract, close_side, abs(pos.quantity)
        )
        if not allowed:
            return False
        await self._broker.place_order(
            Order(
                contract=bar.contract,
                side=close_side,
                quantity=abs(pos.quantity),
                order_type=OrderType.MKT,
                client_order_id=f"stop@{bar.ts.isoformat()}",
            )
        )
        return True


__all__: list[str] = [
    "BacktestEngine",
    "RunResult",
]
