"""Per-trade metric + ledger test suite.

Three layers:
- `compute_trade_metrics` (futur3.stats.performance): hand-calculated trade-PnL sequences with known
  win rate / payoff / expectancy / profit factor / skew, plus the degenerate cases (empty, all-wins,
  all-losses, scratch, < 3 trades, zero variance) that surface None (fail-loud), never a misleading 0.
- `MockBroker.trade_pnls`: the per-trade ledger books ONE flat-to-flat bet per round-trip -- partial
  scale-outs accumulate into one trade, a flip books the closed bet then starts the next, an open
  position at the end is excluded, and `reset()` clears it.
- The roll-fold KILLER: a position held across a contract roll books exactly ONE trade (the roll's
  front-leg close is NOT a separate trade -- its PnL is carried into the back contract), so rolls
  cannot inflate or skew the win rate on the CL/GC race markets.

References: mock_broker.py, stats/performance.py, test_roll_wiring.py
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from futur3.data.types import ContractSymbol
from futur3.execution import multiplier_for
from futur3.execution.adapters.mock_broker import MockBroker
from futur3.execution.broker import Order, OrderType, Side
from futur3.stats import TradeMetrics, compute_trade_metrics

_CL_F = ContractSymbol("CLF26")
_CL_G = ContractSymbol("CLG26")


def _pnls(values: list[str]) -> list[Decimal]:
    return [Decimal(v) for v in values]


def _fill(
    b: MockBroker, side: Side, qty: int, price: str, contract: ContractSymbol = _CL_F
) -> None:
    """Place + fully fill a MKT order (the broker's discretionary-trade path)."""
    oid = asyncio.run(
        b.place_order(Order(contract=contract, side=side, quantity=qty, order_type=OrderType.MKT))
    )
    b.fill_order(oid, Decimal(price))


# ============================================================================
# TestKnownValues - the metric math
# ============================================================================


class TestKnownValues:
    def test_win_rate_payoff_expectancy(self) -> None:
        # 3 wins @ +100, 2 losses @ -50 -> WR 0.6, avg_win 100, avg_loss 50, payoff 2.0,
        # expectancy (300-100)/5 = 40, profit_factor 300/100 = 3.0
        m = compute_trade_metrics(_pnls(["100", "100", "100", "-50", "-50"]))
        assert m.n_trades == 5
        assert m.n_wins == 3
        assert m.n_losses == 2
        assert m.n_scratch == 0
        assert m.win_rate == pytest.approx(0.6)
        assert m.avg_win == Decimal("100")
        assert m.avg_loss == Decimal("50")
        assert m.payoff_ratio == pytest.approx(2.0)
        assert m.expectancy == Decimal("40")
        assert m.profit_factor == pytest.approx(3.0)

    def test_returns_trade_metrics_type(self) -> None:
        assert isinstance(compute_trade_metrics(_pnls(["1", "-1"])), TradeMetrics)

    def test_breakeven_identity(self) -> None:
        # expectancy MUST equal win_rate*avg_win - loss_rate*avg_loss (loss_rate = n_losses/n)
        m = compute_trade_metrics(_pnls(["10", "20", "-5", "-15", "30"]))
        assert m.win_rate is not None and m.avg_win is not None and m.avg_loss is not None
        loss_rate = m.n_losses / m.n_trades
        identity = m.win_rate * float(m.avg_win) - loss_rate * float(m.avg_loss)
        assert m.expectancy is not None
        assert float(m.expectancy) == pytest.approx(identity)

    def test_scratch_sits_in_denominator(self) -> None:
        # a 0-PnL trade is neither win nor loss but counts in n_trades (the win_rate denominator)
        m = compute_trade_metrics(_pnls(["100", "0", "-50"]))
        assert (m.n_trades, m.n_wins, m.n_losses, m.n_scratch) == (3, 1, 1, 1)
        assert m.win_rate == pytest.approx(1 / 3)


# ============================================================================
# TestDegenerate - None, never a misleading 0 (fail-loud)
# ============================================================================


class TestDegenerate:
    def test_empty_is_zero_trades_not_error(self) -> None:
        m = compute_trade_metrics([])
        assert m.n_trades == 0
        assert m.win_rate is None
        assert m.avg_win is None
        assert m.avg_loss is None
        assert m.payoff_ratio is None
        assert m.expectancy is None
        assert m.profit_factor is None
        assert m.pnl_skew is None

    def test_all_wins_payoff_and_pf_undefined(self) -> None:
        m = compute_trade_metrics(_pnls(["100", "200", "300"]))
        assert m.win_rate == pytest.approx(1.0)
        assert m.avg_loss is None
        assert m.payoff_ratio is None  # no losses -> reward:risk undefined
        assert m.profit_factor is None  # no losses -> infinite -> None (fail-loud)

    def test_all_losses_pf_is_zero(self) -> None:
        m = compute_trade_metrics(_pnls(["-100", "-200", "-300"]))
        assert m.win_rate == pytest.approx(0.0)
        assert m.avg_win is None
        assert m.payoff_ratio is None
        assert m.profit_factor == 0.0  # zero gross win over real losses -> a meaningful 0


# ============================================================================
# TestSkew - the high-win-rate tail-risk signal
# ============================================================================


class TestSkew:
    def test_negative_skew_detected(self) -> None:
        # many small wins + one big loss -> left-skewed (the steamroller-tail signature)
        m = compute_trade_metrics(_pnls(["1", "1", "1", "1", "1", "-50"]))
        assert m.pnl_skew is not None and m.pnl_skew < 0

    def test_positive_skew_detected(self) -> None:
        m = compute_trade_metrics(_pnls(["-1", "-1", "-1", "-1", "-1", "50"]))
        assert m.pnl_skew is not None and m.pnl_skew > 0

    def test_too_few_for_skew(self) -> None:
        assert compute_trade_metrics(_pnls(["10", "-5"])).pnl_skew is None

    def test_zero_variance_skew_none(self) -> None:
        assert compute_trade_metrics(_pnls(["10", "10", "10"])).pnl_skew is None


# ============================================================================
# TestBrokerTradeLedger - flat-to-flat bookkeeping on the MockBroker
# ============================================================================


class TestBrokerTradeLedger:
    def test_single_round_trip_books_one_trade(self) -> None:
        b = MockBroker()
        _fill(b, Side.BUY, 1, "100")
        _fill(b, Side.SELL, 1, "110")
        m = asyncio.run(b.get_account_metrics())
        assert len(b.trade_pnls) == 1
        assert b.trade_pnls[0] == m.realized_pnl_today  # the one closed bet == realized PnL
        assert b.trade_pnls[0] > 0  # bought 100, sold 110 -> win

    def test_open_position_not_booked(self) -> None:
        b = MockBroker()
        _fill(b, Side.BUY, 1, "100")
        assert b.trade_pnls == ()  # still open -> no realized outcome (excluded from WR)

    def test_partial_scale_out_is_one_trade(self) -> None:
        b = MockBroker()
        _fill(b, Side.BUY, 4, "100")
        _fill(b, Side.SELL, 2, "110")
        assert b.trade_pnls == ()  # partial close: the bet continues, nothing booked yet
        _fill(b, Side.SELL, 2, "120")
        m = asyncio.run(b.get_account_metrics())
        assert len(b.trade_pnls) == 1  # both scale-outs fold into ONE flat-to-flat trade
        assert b.trade_pnls[0] == m.realized_pnl_today

    def test_flip_books_closed_bet_then_next(self) -> None:
        b = MockBroker()
        _fill(b, Side.BUY, 2, "100")
        _fill(b, Side.SELL, 5, "110")  # closes the long (booked) AND opens short 3 @ 110
        assert len(b.trade_pnls) == 1
        assert b.trade_pnls[0] > 0  # long 100 -> 110
        _fill(b, Side.BUY, 3, "90")  # closes the short
        assert len(b.trade_pnls) == 2
        assert b.trade_pnls[1] > 0  # short 110 -> 90 -> win

    def test_two_round_trips_preserve_sign_and_order(self) -> None:
        b = MockBroker()
        _fill(b, Side.BUY, 1, "100")
        _fill(b, Side.SELL, 1, "110")  # +win
        _fill(b, Side.BUY, 1, "200")
        _fill(b, Side.SELL, 1, "190")  # -loss
        assert len(b.trade_pnls) == 2
        assert b.trade_pnls[0] > 0
        assert b.trade_pnls[1] < 0

    def test_reset_clears_ledger(self) -> None:
        b = MockBroker()
        _fill(b, Side.BUY, 1, "100")
        _fill(b, Side.SELL, 1, "110")
        assert len(b.trade_pnls) == 1
        b.reset()
        assert b.trade_pnls == ()


# ============================================================================
# TestRollFold - the KILLER: rolls must NOT pollute the win rate
# ============================================================================


class TestRollFold:
    def test_roll_alone_books_no_trade(self) -> None:
        b = MockBroker()
        _fill(b, Side.BUY, 3, "75.00")
        b.roll_position(_CL_F, _CL_G, Decimal("75.40"), Decimal("76.10"))
        assert (
            b.trade_pnls == ()
        )  # a roll is not a trade; the bet is still open in the back contract

    def test_roll_folds_into_one_trade(self) -> None:
        b = MockBroker()
        _fill(b, Side.BUY, 3, "75.00")
        b.roll_position(_CL_F, _CL_G, Decimal("75.40"), Decimal("76.10"))
        _fill(b, Side.SELL, 3, "77.00", contract=_CL_G)  # finally close the (rolled) bet
        mult = multiplier_for(_CL_G)
        # ONE trade = front seg (75.40-75.00) + back seg (77.00-76.10), folded across the roll
        expected = (
            ((Decimal("75.40") - Decimal("75.00")) + (Decimal("77.00") - Decimal("76.10")))
            * 3
            * mult
        )
        assert b.trade_pnls == (expected,)

    def test_roll_ledger_reconciles_with_realized(self) -> None:
        b = MockBroker()
        _fill(b, Side.BUY, 3, "75.00")
        b.roll_position(_CL_F, _CL_G, Decimal("75.40"), Decimal("76.10"))
        _fill(b, Side.SELL, 3, "77.00", contract=_CL_G)
        m = asyncio.run(b.get_account_metrics())
        assert sum(b.trade_pnls, Decimal("0")) == m.realized_pnl_today  # ledger == realized PnL
