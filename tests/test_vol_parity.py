"""Tests for inverse-vol (risk-parity) leg weighting.

The weights re-distribute one shared bet so each leg contributes ~equal risk: weight_i ~ 1/vol_i,
normalized to mean 1. Degenerate legs (< 2 returns / zero vol) get the neutral mean inverse-vol so a
flat leg can never demand an infinite position.
"""

from __future__ import annotations

from decimal import Decimal

from futur3.execution.vol_parity import inverse_vol_weights


def _rets(*vals: str) -> list[Decimal]:
    return [Decimal(v) for v in vals]


class TestInverseVolWeights:
    def test_empty_input(self) -> None:
        assert inverse_vol_weights({}) == {}

    def test_equal_vol_gives_equal_unit_weights(self) -> None:
        # identical return vol -> exactly equal weights, each ~1 (modulo Decimal ULP from sqrt/norm)
        rets = _rets("0.01", "-0.01", "0.01", "-0.01")
        w = inverse_vol_weights({"A": rets, "B": list(rets)})
        assert w["A"] == w["B"]
        assert abs(w["A"] - Decimal(1)) < Decimal("1e-20")

    def test_higher_vol_gets_lower_weight_ratio_matches_vol_ratio(self) -> None:
        # B has 3x A's vol -> A's weight is 3x B's; mean weight stays 1
        w = inverse_vol_weights(
            {
                "A": _rets("0.01", "-0.01", "0.01", "-0.01"),
                "B": _rets("0.03", "-0.03", "0.03", "-0.03"),
            }
        )
        assert w["A"] > w["B"]
        assert abs(w["A"] / w["B"] - Decimal(3)) < Decimal("0.0001")  # weight ratio == vol ratio
        assert abs((w["A"] + w["B"]) / 2 - Decimal(1)) < Decimal("1e-12")  # mean weight 1

    def test_flat_leg_is_neutral_not_infinite(self) -> None:
        # a zero-vol leg can't contribute equal risk -> capped to the neutral mean, never infinite
        w = inverse_vol_weights(
            {
                "LO": _rets("0.01", "-0.01", "0.01", "-0.01"),
                "HI": _rets("0.04", "-0.04", "0.04", "-0.04"),
                "FLAT": _rets("0", "0", "0", "0"),
            }
        )
        assert w["LO"] > w["HI"]  # lower vol -> higher weight
        assert all(v > 0 for v in w.values())  # every weight finite + positive (FLAT not infinite)
        assert abs(sum(w.values(), start=Decimal(0)) - Decimal(3)) < Decimal("1e-12")  # mean 1

    def test_all_degenerate_falls_back_to_unit_weights(self) -> None:
        w = inverse_vol_weights({"X": _rets("0", "0"), "Y": _rets("5")})  # zero vol + too-short
        assert w == {"X": Decimal(1), "Y": Decimal(1)}
