"""Annualization-factor test suite (futur3.timeframe.resolve_ppy).

Locks the `periods_per_year` convention against `stats.performance`'s documented anchors (252 daily;
252*78 RTH-5min), covers every BarResolution, the session-preset overrides (Globex ETH / crypto
24-7), DAY_1 / SETTLE session-length invariance, and the non-positive-input guards.
"""

from __future__ import annotations

import pytest

from futur3.data.types import BarResolution
from futur3.timeframe import (
    CRYPTO_DAYS_PER_YEAR,
    CRYPTO_HOURS,
    GLOBEX_EQUITY_HOURS,
    RTH_HOURS,
    TRADING_DAYS_PER_YEAR,
    resolve_ppy,
)

# ============================================================================
# TestDocumentedConvention - must match stats.performance anchors exactly
# ============================================================================


class TestDocumentedConvention:
    def test_daily_is_252(self) -> None:
        assert resolve_ppy(BarResolution.DAY_1) == pytest.approx(252.0)

    def test_settle_is_252(self) -> None:
        assert resolve_ppy(BarResolution.SETTLE) == pytest.approx(252.0)

    def test_rth_5min_matches_252x78(self) -> None:
        # stats.performance docstring anchor: "~252*78 RTH-5min equities"
        assert resolve_ppy(BarResolution.MIN_5) == pytest.approx(252 * 78)

    def test_rth_hourly_is_1638(self) -> None:
        assert resolve_ppy(BarResolution.HOUR_1) == pytest.approx(1638.0)  # 252 * 6.5

    def test_rth_1min_is_98280(self) -> None:
        assert resolve_ppy(BarResolution.MIN_1) == pytest.approx(98_280.0)  # 252 * 390

    def test_rth_15min(self) -> None:
        assert resolve_ppy(BarResolution.MIN_15) == pytest.approx(252 * 26)  # 6.5h * 4 = 26/day

    def test_rth_1sec(self) -> None:
        assert resolve_ppy(BarResolution.SEC_1) == pytest.approx(252 * 6.5 * 3600)

    def test_rth_5sec(self) -> None:
        assert resolve_ppy(BarResolution.SEC_5) == pytest.approx(252 * 6.5 * 720)


# ============================================================================
# TestSessionPresets - caller picks the basis matching the return series
# ============================================================================


class TestSessionPresets:
    def test_globex_equity_hourly(self) -> None:
        assert resolve_ppy(
            BarResolution.HOUR_1, hours_per_session=GLOBEX_EQUITY_HOURS
        ) == pytest.approx(252 * 23)

    def test_crypto_minutes_per_year(self) -> None:
        # 24/7: 365 * 24 * 60 = 525600 minutes per year
        assert resolve_ppy(
            BarResolution.MIN_1,
            hours_per_session=CRYPTO_HOURS,
            sessions_per_year=CRYPTO_DAYS_PER_YEAR,
        ) == pytest.approx(525_600.0)

    def test_finer_resolution_gives_more_periods(self) -> None:
        assert (
            resolve_ppy(BarResolution.HOUR_1)
            < resolve_ppy(BarResolution.MIN_1)
            < resolve_ppy(BarResolution.SEC_1)
        )

    def test_presets_have_expected_values(self) -> None:
        assert RTH_HOURS == 6.5
        assert TRADING_DAYS_PER_YEAR == 252
        assert GLOBEX_EQUITY_HOURS == 23.0
        assert CRYPTO_HOURS == 24.0
        assert CRYPTO_DAYS_PER_YEAR == 365


# ============================================================================
# TestSessionLengthInvariance - DAY_1 / SETTLE ignore hours_per_session
# ============================================================================


class TestSessionLengthInvariance:
    @pytest.mark.parametrize("resolution", [BarResolution.DAY_1, BarResolution.SETTLE])
    def test_daily_ignores_session_hours(self, resolution: BarResolution) -> None:
        assert resolve_ppy(resolution, hours_per_session=23.0) == pytest.approx(
            resolve_ppy(resolution, hours_per_session=6.5)
        )

    def test_daily_scales_with_sessions_per_year(self) -> None:
        assert resolve_ppy(
            BarResolution.DAY_1, sessions_per_year=CRYPTO_DAYS_PER_YEAR
        ) == pytest.approx(365.0)


# ============================================================================
# TestGuards - non-positive inputs surface ValueError (never a silent 0)
# ============================================================================


class TestGuards:
    def test_zero_hours_raises(self) -> None:
        with pytest.raises(ValueError, match="hours_per_session"):
            resolve_ppy(BarResolution.HOUR_1, hours_per_session=0.0)

    def test_negative_hours_raises(self) -> None:
        with pytest.raises(ValueError, match="hours_per_session"):
            resolve_ppy(BarResolution.MIN_1, hours_per_session=-1.0)

    def test_zero_sessions_raises(self) -> None:
        with pytest.raises(ValueError, match="sessions_per_year"):
            resolve_ppy(BarResolution.HOUR_1, sessions_per_year=0)

    def test_negative_sessions_raises(self) -> None:
        with pytest.raises(ValueError, match="sessions_per_year"):
            resolve_ppy(BarResolution.HOUR_1, sessions_per_year=-252)


# ============================================================================
# TestExhaustiveness - every BarResolution resolves under the default basis
# ============================================================================


class TestExhaustiveness:
    def test_all_resolutions_resolve_positive(self) -> None:
        for resolution in BarResolution:
            assert resolve_ppy(resolution) > 0
