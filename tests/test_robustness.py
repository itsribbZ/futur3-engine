"""Robustness / cost-sensitivity test suite (futur3.stats.robustness).

Hand-computed values lock: winsorized Sharpe (trim=0 == raw; capping the tails), tail-contribution
concentration (one outlier -> ~1.0; uniform -> ~frac), per-trade cost subtraction on traded days,
and the input guards.
"""

from __future__ import annotations

import math
import statistics

import pytest

from futur3.stats.robustness import (
    RobustnessError,
    net_returns,
    tail_contribution,
    winsorized_sharpe,
)

_DATA = [0.01, 0.02, -0.01, 0.03, 0.20]  # 4 normal + 1 positive outlier


# ============================================================================
# TestWinsorizedSharpe
# ============================================================================


class TestWinsorizedSharpe:
    def test_trim_zero_is_raw_sharpe(self) -> None:
        expected = statistics.mean(_DATA) / statistics.stdev(_DATA) * math.sqrt(252)
        assert winsorized_sharpe(_DATA, 0.0) == pytest.approx(expected)

    def test_winsorize_caps_both_tails(self) -> None:
        # trim 0.2 of n=5 -> k=1: cap to [ordered[1], ordered[3]] = [0.01, 0.03].
        # capped = [0.01, 0.02, 0.01, 0.03, 0.03] -> mean 0.02, stdev 0.01 -> SR_pp 2.0.
        assert winsorized_sharpe(_DATA, 0.2) == pytest.approx(2.0 * math.sqrt(252))

    def test_zero_variance_is_none(self) -> None:
        assert winsorized_sharpe([0.01, 0.01, 0.01], 0.0) is None

    def test_too_few_points_is_none(self) -> None:
        assert winsorized_sharpe([0.01], 0.0) is None

    def test_trim_frac_out_of_range_raises(self) -> None:
        with pytest.raises(RobustnessError, match="trim_frac"):
            winsorized_sharpe(_DATA, 0.5)
        with pytest.raises(RobustnessError, match="trim_frac"):
            winsorized_sharpe(_DATA, -0.1)

    def test_bad_ppy_raises(self) -> None:
        with pytest.raises(RobustnessError, match="periods_per_year"):
            winsorized_sharpe(_DATA, 0.0, periods_per_year=0.0)


# ============================================================================
# TestTailContribution
# ============================================================================


class TestTailContribution:
    def test_single_outlier_dominates(self) -> None:
        # one big observation carries ~all of the gross |return|
        assert tail_contribution([0.0, 0.0, 0.0, 0.0, 1.0], 0.2) == pytest.approx(1.0)

    def test_uniform_matches_frac(self) -> None:
        # 5 equal observations, top 20% (=1 of 5) = 0.1 / 0.5 = 0.2
        assert tail_contribution([0.1, 0.1, 0.1, 0.1, 0.1], 0.2) == pytest.approx(0.2)

    def test_all_flat_is_zero(self) -> None:
        assert tail_contribution([0.0, 0.0, 0.0], 0.2) == 0.0

    def test_frac_out_of_range_raises(self) -> None:
        with pytest.raises(RobustnessError, match="frac"):
            tail_contribution(_DATA, 0.0)
        with pytest.raises(RobustnessError, match="frac"):
            tail_contribution(_DATA, 1.0)


# ============================================================================
# TestNetReturns
# ============================================================================


class TestNetReturns:
    def test_cost_hits_only_traded_days(self) -> None:
        out = net_returns([0.01, 0.02, 0.0], [True, False, True], 0.001)
        assert out == pytest.approx((0.009, 0.02, -0.001))

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(RobustnessError, match="length mismatch"):
            net_returns([0.01, 0.02], [True], 0.001)

    def test_negative_cost_raises(self) -> None:
        with pytest.raises(RobustnessError, match="cost_per_trade"):
            net_returns([0.01], [True], -0.001)
