"""Decay-monitor test suite (futur3.execution.decay_monitor) - the forward-paper health gate.

Locks: WARMING_UP under min_sessions, HEALTHY on a positive series, DECAYED via negative cumulative
AND via a non-positive trailing Sharpe (cumulative still > 0), the max-drawdown computation, and the
degenerate constant-positive series (undefined Sharpe but positive cumulative -> HEALTHY).
"""

from __future__ import annotations

from decimal import Decimal

from futur3.execution.decay_monitor import DecayVerdict, assess_decay


class TestVerdict:
    def test_warming_up_under_min_sessions(self) -> None:
        st = assess_decay([Decimal("10")] * 5, min_sessions=21)
        assert st.verdict is DecayVerdict.WARMING_UP
        assert not st.healthy
        assert st.n_sessions == 5

    def test_healthy_on_positive_series(self) -> None:
        pnl = [Decimal(x) for x in ("10", "8", "12", "9", "11", "7", "13", "10")]
        st = assess_decay(pnl, min_sessions=3, rolling_window=5)
        assert st.verdict is DecayVerdict.HEALTHY
        assert st.healthy
        assert st.full_sharpe is not None and st.full_sharpe > 0
        assert st.rolling_sharpe is not None and st.rolling_sharpe > 0
        assert st.cumulative_pnl == Decimal("80")

    def test_decayed_on_negative_cumulative(self) -> None:
        pnl = [Decimal(x) for x in ("5", "-8", "-7", "4", "-9", "-6")]
        st = assess_decay(pnl, min_sessions=3, rolling_window=5)
        assert st.verdict is DecayVerdict.DECAYED
        assert not st.healthy
        assert st.cumulative_pnl < 0

    def test_decayed_on_negative_trailing_sharpe(self) -> None:
        # cumulative stays POSITIVE (+75) but the recent window has rolled negative
        pnl = [Decimal("20")] * 5 + [Decimal(x) for x in ("-5", "-6", "-4", "-7", "-3")]
        st = assess_decay(pnl, min_sessions=3, rolling_window=5)
        assert st.cumulative_pnl == Decimal("75")  # still net positive
        assert st.rolling_sharpe is not None and st.rolling_sharpe < 0
        assert st.verdict is DecayVerdict.DECAYED  # the recent roll-over trips it

    def test_constant_positive_is_healthy_not_decayed(self) -> None:
        # zero-variance positive series -> undefined Sharpe, but positive cumulative is NOT decay
        st = assess_decay([Decimal("10")] * 30, min_sessions=3, rolling_window=5)
        assert st.full_sharpe is None  # var == 0
        assert st.rolling_sharpe is None
        assert st.verdict is DecayVerdict.HEALTHY
        assert st.healthy


class TestMaxDrawdown:
    def test_peak_to_trough(self) -> None:
        # cumulative curve: 100, 70, 20, 40 -> worst peak(100)->trough(20) = 80
        pnl = [Decimal("100"), Decimal("-30"), Decimal("-50"), Decimal("20")]
        st = assess_decay(pnl, min_sessions=1, rolling_window=4)
        assert st.max_drawdown == Decimal("80")

    def test_monotonic_up_has_zero_drawdown(self) -> None:
        st = assess_decay([Decimal("5")] * 10, min_sessions=1)
        assert st.max_drawdown == Decimal("0")
