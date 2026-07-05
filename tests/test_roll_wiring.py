"""Engine roll-as-paired-trade wiring test suite.

Per the build plan. Two layers:
- `MockBroker.roll_position`: closes the front leg (realizing PnL) + re-opens the same signed
  qty in back at back_fill (avg_entry = back price, no gap carried). Long / short / flat.
- `BacktestEngine` with opt-in `roll_events`: on a roll date a held position is rolled into the
  new contract at the roll's settle prices BEFORE the bar's mark-to-market. The KILLER test holds
  a position across a CLF26->CLG26 roll (raw +$0.70 gap) and asserts the equity step across the
  roll equals the REAL front move (Dec11->Dec12 front), NOT the roll gap — the roll-gap contamination
  fixed. roll_events=None stays byte-identical (covered by the existing 1698-test suite).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, date, datetime
from decimal import Decimal

from futur3.data import SourceTier
from futur3.data.continuous import RollEvent
from futur3.data.types import BarResolution, ContractSymbol, content_sha256
from futur3.data.verifier import VerifiedBar
from futur3.data.verifier_policies import POLICY_PHASE_A1_DEFAULT
from futur3.engine import BacktestEngine, RunResult
from futur3.execution import RiskManager, multiplier_for
from futur3.execution.adapters.mock_broker import MockBroker
from futur3.execution.broker import Order, OrderType, Side
from futur3.runtime import RuntimeContext, RuntimeMode, SystemClock
from futur3.strategies.base import Signal, Strategy

_F = ContractSymbol("CLF26")
_G = ContractSymbol("CLG26")
_ROLL = date(2025, 12, 12)  # CLF26 roll_target (W1.2-verified)


# ============================================================================
# TestW1_5_RollPosition (broker mechanic)
# ============================================================================


def _mkt(side: Side, qty: int, contract: ContractSymbol = _F) -> Order:
    return Order(contract=contract, side=side, quantity=qty, order_type=OrderType.MKT)


def _long(qty: int, price: str) -> MockBroker:
    b = MockBroker()
    oid = asyncio.run(b.place_order(_mkt(Side.BUY, qty)))
    b.fill_order(oid, Decimal(price))
    return b


class TestW1_5_RollPosition:
    def test_long_transfers_and_realizes(self) -> None:
        b = _long(3, "75.00")
        rolled = b.roll_position(_F, _G, Decimal("75.40"), Decimal("76.10"))
        assert rolled == 3
        pos = {str(p.contract): p for p in asyncio.run(b.get_positions())}
        assert pos["CLF26"].quantity == 0  # front closed
        assert pos["CLG26"].quantity == 3  # exposure transferred
        assert pos["CLG26"].avg_entry_price == Decimal("76.10")  # at back price (no gap carried)
        mult = multiplier_for(_F)
        m = asyncio.run(b.get_account_metrics())
        assert m.realized_pnl_today == (Decimal("75.40") - Decimal("75.00")) * 3 * mult  # front leg

    def test_short_transfers(self) -> None:
        b = MockBroker()
        oid = asyncio.run(b.place_order(_mkt(Side.SELL, 2)))
        b.fill_order(oid, Decimal("75.00"))
        rolled = b.roll_position(_F, _G, Decimal("75.40"), Decimal("76.10"))
        assert rolled == -2
        pos = {str(p.contract): p for p in asyncio.run(b.get_positions())}
        assert pos["CLF26"].quantity == 0
        assert pos["CLG26"].quantity == -2
        assert pos["CLG26"].avg_entry_price == Decimal("76.10")

    def test_flat_is_noop(self) -> None:
        assert MockBroker().roll_position(_F, _G, Decimal("75.40"), Decimal("76.10")) == 0


# ============================================================================
# TestW1_5_EngineRollWiring (the no-phantom-jump proof)
# ============================================================================


class _BuyOnceHold(Strategy):
    """Goes long once (first bar), then flat-signals — establishes a position and holds it."""

    @property
    def strategy_id(self) -> str:
        return "test_buy_once_hold"

    def generate_signal(self, history: Sequence[VerifiedBar], ctx: RuntimeContext) -> Signal | None:
        bar = history[-1]
        first = len(history) == 1
        return Signal(
            contract=bar.contract,
            ts=bar.ts,
            strategy_id=self.strategy_id,
            direction=1 if first else 0,
            full_kelly_fraction=Decimal("20") if first else Decimal("0"),
            confidence=Decimal("0.9"),
        )


def _vbar(contract: ContractSymbol, day: date, o: str, h: str, low: str, c: str) -> VerifiedBar:
    ts = datetime(day.year, day.month, day.day, tzinfo=UTC)
    return VerifiedBar(
        contract=contract,
        ts=ts,
        resolution=BarResolution.DAY_1,
        open=Decimal(o),
        high=Decimal(h),
        low=Decimal(low),
        close=Decimal(c),
        volume=100,
        n_sources_agreed=2,
        source_provenance_hashes=(content_sha256(b"s1"),),
        verifier_run_hash=content_sha256(f"{contract}|{ts}|{c}".encode()),
        prev_bar_hash=None,
        tier_used=SourceTier.T2_EXCHANGE,
        policy_id="PHASE_A1",
    )


# CLF26 trades Dec 8-11 (front), then CLG26 Dec 12-16 (back) — the NUL-spliced stream. The raw
# front->back level gap is ~ +$0.70 (front Dec11 close 75.30 -> back Dec12 close 76.10).
_ROLL_BARS: list[VerifiedBar] = [
    _vbar(_F, date(2025, 12, 8), "75.00", "75.10", "74.90", "75.00"),  # 0 signal bar
    _vbar(_F, date(2025, 12, 9), "75.05", "75.20", "75.00", "75.10"),  # 1 fills BUY @ open 75.05
    _vbar(_F, date(2025, 12, 10), "75.10", "75.30", "75.05", "75.20"),  # 2
    _vbar(_F, date(2025, 12, 11), "75.20", "75.40", "75.15", "75.30"),  # 3 last front bar
    _vbar(_G, date(2025, 12, 12), "76.00", "76.20", "75.95", "76.10"),  # 4 ROLL bar (back_settle)
    _vbar(_G, date(2025, 12, 15), "76.10", "76.40", "76.05", "76.20"),  # 5
    _vbar(_G, date(2025, 12, 16), "76.20", "76.50", "76.15", "76.30"),  # 6
]
# front_settle = CLF26's close on the roll date (Dec12), known from the raw contract (W1.4), not
# present in the spliced stream; back_settle = the Dec12 bar's close.
_ROLL_EVENT = RollEvent(
    roll_date=_ROLL,
    old_contract=_F,
    new_contract=_G,
    front_settle=Decimal("75.40"),
    back_settle=Decimal("76.10"),
    roll_gap=Decimal("0.70"),
)


def _ctx() -> RuntimeContext:
    return RuntimeContext(
        mode=RuntimeMode.BACKTEST, verifier_policy=POLICY_PHASE_A1_DEFAULT, clock=SystemClock()
    )


class TestW1_5_EngineRollWiring:
    def _run(self) -> RunResult:
        engine = BacktestEngine(
            ctx=_ctx(),
            broker=MockBroker(),
            strategy=_BuyOnceHold(),
            risk=RiskManager(),
            roll_events=[_ROLL_EVENT],
        )
        return asyncio.run(engine.run(_ROLL_BARS))

    def test_roll_executed_and_position_transferred(self) -> None:
        res = self._run()
        assert res.rolls_executed == 1
        final = {str(p.contract): p for p in res.final_positions}
        assert "CLF26" not in final or final["CLF26"].quantity == 0  # front closed
        assert final["CLG26"].quantity > 0  # exposure now in the back contract

    def test_no_phantom_jump_across_roll(self) -> None:
        res = self._run()
        qty = {str(p.contract): p for p in res.final_positions}["CLG26"].quantity
        mult = multiplier_for(_G)
        step = res.equity_curve[4] - res.equity_curve[3]  # equity step across the roll
        # The held position's REAL move Dec11->Dec12 is the FRONT move 75.30 -> 75.40 = +0.10.
        assert step == Decimal("0.10") * qty * mult
        # NOT the raw front-vs-back gap (76.10 - 75.30 = 0.80) — that was the phantom roll gap.
        assert step != Decimal("0.80") * qty * mult

    def test_total_roll_cost_diagnostic(self) -> None:
        res = self._run()
        qty = {str(p.contract): p for p in res.final_positions}["CLG26"].quantity
        mult = multiplier_for(_G)
        assert res.total_roll_cost == Decimal("0.70") * qty * mult
