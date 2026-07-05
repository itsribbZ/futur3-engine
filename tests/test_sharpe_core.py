"""P0c Lo(2002) autocorrelation-correction test suite (_sharpe_core.lo_autocorr_factor).

The naive annualization sqrt(q) assumes IID returns. For a positively-autocorrelated book (trend /
momentum) the true annualized Sharpe is LOWER (Lo 2002) — eta(q) < sqrt(q) — so the naive number is
optimistic. P0c reports the eta(q)-corrected Sharpe ALONGSIDE the naive one (advisory; the psr/dsr
probabilities themselves are unchanged). These tests pin the factor's behavior + the conservative
direction that matters for a momentum CTA.
"""

from __future__ import annotations

import math
import random

import pytest

from futur3.stats._sharpe_core import lo_autocorr_factor
from futur3.stats.deflated_sharpe import deflated_sharpe
from futur3.stats.probabilistic_sharpe import probabilistic_sharpe

_Q = 252
_SQRT_Q = math.sqrt(_Q)


def _ar1(rho: float, n: int = 500, *, drift: float = 0.0, seed: int = 7) -> list[float]:
    """Deterministic AR(1) series r_t = rho*r_{t-1} + N(0, 0.01) + drift (lag-1 autocorr ~= rho)."""
    rng = random.Random(seed)
    out: list[float] = []
    prev = 0.0
    for _ in range(n):
        prev = rho * prev + rng.gauss(0.0, 0.01)
        out.append(prev + drift)
    return out


class TestLoAutocorrFactor:
    def test_no_lags_equals_sqrt_q(self) -> None:
        # max_lag=0 -> no autocorrelation correction -> reduces to the naive IID sqrt(q).
        assert lo_autocorr_factor(_ar1(0.0), _Q, max_lag=0) == pytest.approx(_SQRT_Q)

    def test_positive_autocorr_below_sqrt_q(self) -> None:
        eta = lo_autocorr_factor(_ar1(0.7), _Q)
        assert eta is not None
        assert eta < _SQRT_Q  # positive autocorrelation -> conservative (lower than naive sqrt(q))

    def test_negative_autocorr_not_below_sqrt_q(self) -> None:
        # mean-reverting: the correction must NOT be optimistic -> eta > sqrt(q), or None if the
        # variance-ratio denominator is ill-conditioned (strong negative autocorrelation).
        eta = lo_autocorr_factor(_ar1(-0.5), _Q)
        assert eta is None or eta > _SQRT_Q

    def test_constant_series_is_none(self) -> None:
        assert lo_autocorr_factor([0.01] * 40, _Q) is None  # zero variance

    def test_too_short_is_none(self) -> None:
        assert lo_autocorr_factor([0.01], _Q) is None

    def test_bad_q_is_none(self) -> None:
        assert lo_autocorr_factor(_ar1(0.0), 0) is None


class TestLoIntegration:
    def test_psr_reports_conservative_lo_sharpe(self) -> None:
        r = _ar1(0.7, drift=0.002)  # positively autocorrelated, positive mean
        res = probabilistic_sharpe(r, periods_per_year=float(_Q))
        assert res.sr_annualized is not None
        assert res.sr_annualized_lo is not None
        assert res.sr_annualized_lo < res.sr_annualized  # eta(q) < sqrt(q) for positive autocorr
        assert res.psr is not None  # psr still computed — additive field did not change behavior

    def test_dsr_reports_lo_sharpe(self) -> None:
        r = _ar1(0.7, drift=0.002)
        res = deflated_sharpe(r, [0.5, 0.8, 1.0], periods_per_year=float(_Q))
        assert res.sr_observed_lo is not None
        assert res.dsr is not None  # dsr math unchanged by the additive Lo field
