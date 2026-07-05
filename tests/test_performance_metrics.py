"""Descriptive performance-metric test suite (futur3.stats.performance).

Synthetic equity curves (type-compatible with BacktestEngine RunResult.equity_curve) with known
total return / max drawdown / Sharpe sign, plus the degenerate cases (flat, zero-variance, too
few points, non-positive equity) that must surface None / StatsError rather than a misleading 0.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from futur3.stats import PerformanceMetrics, StatsError, compute_metrics


def _curve(values: list[str]) -> list[Decimal]:
    return [Decimal(v) for v in values]


# ============================================================================
# TestKnownValues
# ============================================================================


class TestKnownValues:
    def test_total_return(self) -> None:
        m = compute_metrics(_curve(["100", "110"]), periods_per_year=1)
        assert m.total_return == pytest.approx(0.10)

    def test_max_drawdown(self) -> None:
        # peak 120 -> trough 90 = 25% drawdown
        m = compute_metrics(_curve(["100", "120", "90", "110"]), periods_per_year=252)
        assert m.max_drawdown == pytest.approx(0.25)

    def test_annualized_return_formula(self) -> None:
        # (110/100)^(ppy/num_returns) - 1 ; ppy=2, num_returns=1 -> 1.1^2 - 1 = 0.21
        m = compute_metrics(_curve(["100", "110"]), periods_per_year=2)
        assert m.annualized_return == pytest.approx(0.21)

    def test_n_periods_is_returns_count(self) -> None:
        m = compute_metrics(_curve(["100", "101", "102", "103"]), periods_per_year=252)
        assert m.n_periods == 3  # 4 equity points -> 3 returns


# ============================================================================
# TestSharpeAndDirection
# ============================================================================


class TestSharpeAndDirection:
    def test_uptrend_positive_sharpe(self) -> None:
        m = compute_metrics(_curve(["100", "101", "103", "104", "107"]), periods_per_year=252)
        assert m.sharpe is not None and m.sharpe > 0
        assert m.total_return > 0
        assert m.annualized_volatility is not None and m.annualized_volatility > 0

    def test_downtrend_negative_sharpe(self) -> None:
        m = compute_metrics(_curve(["107", "104", "103", "101", "100"]), periods_per_year=252)
        assert m.sharpe is not None and m.sharpe < 0
        assert m.total_return < 0

    def test_drawdown_then_recovery_calmar(self) -> None:
        m = compute_metrics(_curve(["100", "120", "90", "150"]), periods_per_year=252)
        assert m.max_drawdown == pytest.approx(0.25)
        assert m.calmar is not None  # annualized_return / 0.25


# ============================================================================
# TestDegenerate
# ============================================================================


class TestDegenerate:
    def test_flat_curve(self) -> None:
        m = compute_metrics(_curve(["100", "100", "100", "100"]), periods_per_year=252)
        assert m.total_return == pytest.approx(0.0)
        assert m.max_drawdown == pytest.approx(0.0)
        assert m.annualized_volatility == pytest.approx(0.0)
        assert m.sharpe is None  # zero variance -> undefined, NOT a silent 0
        assert m.calmar is None  # zero drawdown -> undefined

    def test_zero_variance_constant_returns(self) -> None:
        # geometric +10% each step -> identical returns -> stdev 0 -> Sharpe undefined
        m = compute_metrics(_curve(["100", "110", "121", "133.1"]), periods_per_year=252)
        assert m.sharpe is None
        assert m.annualized_volatility == pytest.approx(0.0)
        assert m.total_return > 0  # still rose

    def test_two_points_volatility_none(self) -> None:
        m = compute_metrics(_curve(["100", "110"]), periods_per_year=252)
        assert m.total_return == pytest.approx(0.10)
        assert m.annualized_volatility is None  # need >= 2 returns for stdev
        assert m.sharpe is None

    def test_monotonic_up_zero_drawdown(self) -> None:
        m = compute_metrics(_curve(["100", "110", "120", "130"]), periods_per_year=252)
        assert m.max_drawdown == pytest.approx(0.0)
        assert m.calmar is None


# ============================================================================
# TestErrors
# ============================================================================


class TestErrors:
    def test_too_few_points(self) -> None:
        with pytest.raises(StatsError, match="need >= 2 equity points"):
            compute_metrics(_curve(["100"]), periods_per_year=252)

    def test_non_positive_equity(self) -> None:
        with pytest.raises(StatsError, match="strictly positive"):
            compute_metrics(_curve(["100", "0", "50"]), periods_per_year=252)

    def test_bad_periods_per_year(self) -> None:
        with pytest.raises(StatsError, match="periods_per_year must be > 0"):
            compute_metrics(_curve(["100", "110"]), periods_per_year=0)

    def test_result_is_frozen(self) -> None:
        m = compute_metrics(_curve(["100", "110"]), periods_per_year=1)
        assert isinstance(m, PerformanceMetrics)
        with pytest.raises(AttributeError):
            m.total_return = 0.0  # type: ignore[misc]
