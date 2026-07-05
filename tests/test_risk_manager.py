"""A1.21 risk_manager test suite (sizing hard-gate).

Test discipline:
- Tighten-only composite: contracts = min(quarter-Kelly, margin-budget, leverage-cap).
- Each cap formula verified with exact Decimal arithmetic + the binding-constraint label.
- Kelly / margin / leverage binding cases all exercised; no_edge short-circuit.
- Per-contract margin + multiplier registries match the broker execution design +
  markets_microstructure point values.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from futur3.data.types import ContractSymbol
from futur3.execution import RiskManager as _PkgRiskManager
from futur3.execution.risk_manager import (
    CONTRACT_INITIAL_MARGIN,
    CONTRACT_MULTIPLIER,
    RiskManager,
    RiskManagerError,
    RiskParams,
    SizingDecision,
)

# ============================================================================
# TestA1_21_Imports
# ============================================================================


class TestA1_21_Imports:
    def test_importable(self) -> None:
        assert RiskManager is not None
        assert RiskParams is not None
        assert SizingDecision is not None

    def test_exported_from_execution_package(self) -> None:
        assert _PkgRiskManager is RiskManager


# ============================================================================
# TestA1_21_Registries
# ============================================================================


class TestA1_21_Registries:
    def test_margin_configured_contracts(self) -> None:
        assert set(CONTRACT_INITIAL_MARGIN) == {
            "ES",
            "MES",
            "NQ",
            "MNQ",
            "CL",
            "MCL",
            "GC",
            "MGC",
            "MBT",
            "MET",
            "YM",  # broaden wave 1
            "RTY",
            "6E",
            "6A",
            "ZN",
            "ZB",
            "6J",  # broaden wave 2
            "6B",
            "6C",
            "6S",
            "HG",
            "SI",
            "ZT",
            "ZF",
            "UB",
            "NKD",
        }

    def test_multiplier_ten_contracts(self) -> None:
        assert set(CONTRACT_MULTIPLIER) == set(CONTRACT_INITIAL_MARGIN)

    def test_known_margins(self) -> None:
        assert CONTRACT_INITIAL_MARGIN["ES"] == Decimal("15400")
        assert CONTRACT_INITIAL_MARGIN["MBT"] == Decimal("2800")  # corrected 1/50-BTC

    def test_known_multipliers(self) -> None:
        assert CONTRACT_MULTIPLIER["ES"] == Decimal("50")
        assert CONTRACT_MULTIPLIER["CL"] == Decimal("1000")
        assert CONTRACT_MULTIPLIER["MBT"] == Decimal("0.1")


# ============================================================================
# TestA1_21_RiskParams
# ============================================================================


class TestA1_21_RiskParams:
    def test_defaults(self) -> None:
        p = RiskParams()
        assert p.kelly_multiplier == Decimal("0.25")  # Quarter Kelly
        assert p.margin_budget_pct == Decimal("0.50")
        assert p.leverage_cap == Decimal("3")

    def test_bad_kelly_multiplier_raises(self) -> None:
        with pytest.raises(ValueError, match="kelly_multiplier must be in"):
            RiskParams(kelly_multiplier=Decimal("0"))

    def test_kelly_multiplier_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="kelly_multiplier must be in"):
            RiskParams(kelly_multiplier=Decimal("1.5"))

    def test_bad_margin_pct_raises(self) -> None:
        with pytest.raises(ValueError, match="margin_budget_pct must be in"):
            RiskParams(margin_budget_pct=Decimal("0"))

    def test_bad_leverage_raises(self) -> None:
        with pytest.raises(ValueError, match="leverage_cap must be > 0"):
            RiskParams(leverage_cap=Decimal("0"))

    def test_frozen(self) -> None:
        p = RiskParams()
        with pytest.raises(AttributeError):
            p.leverage_cap = Decimal("5")  # type: ignore[misc]


# ============================================================================
# TestA1_21_SizingKellyBinds
# ============================================================================


class TestA1_21_SizingKellyBinds:
    def test_es_kelly_binds(self) -> None:
        # ES @ 5000: notional 250k, margin 15.4k. equity 1M, full_kelly 2.0.
        # kelly_capital = 2.0 * 0.25 * 1M = 500k -> 500k/250k = 2 contracts.
        # margin = 1M*0.5/15400 = 32; leverage = 1M*3/250k = 12. min = 2 (kelly).
        d = RiskManager().size_position(
            ContractSymbol("ESM26"), Decimal("2.0"), Decimal("1000000"), Decimal("5000")
        )
        assert d.contracts == 2
        assert d.binding_constraint == "kelly"
        assert d.kelly_contracts == 2
        assert d.margin_contracts == 32
        assert d.leverage_contracts == 12
        assert d.notional_per_contract == Decimal("250000")
        assert d.margin_per_contract == Decimal("15400")

    def test_mes_kelly_binds(self) -> None:
        d = RiskManager().size_position(
            ContractSymbol("MESM26"), Decimal("4.0"), Decimal("100000"), Decimal("5000")
        )
        # notional 25k; kelly = 4*0.25*100k/25k = 4; margin = 100k*0.5/1540 = 32; leverage = 12.
        assert d.contracts == 4
        assert d.binding_constraint == "kelly"


# ============================================================================
# TestA1_21_SizingLeverageBinds
# ============================================================================


class TestA1_21_SizingLeverageBinds:
    def test_es_leverage_binds(self) -> None:
        # full_kelly huge -> kelly cap large; leverage 12 is tightest.
        d = RiskManager().size_position(
            ContractSymbol("ESM26"), Decimal("100"), Decimal("1000000"), Decimal("5000")
        )
        assert d.kelly_contracts == 100
        assert d.margin_contracts == 32
        assert d.leverage_contracts == 12
        assert d.contracts == 12
        assert d.binding_constraint == "leverage"


# ============================================================================
# TestA1_21_SizingMarginBinds
# ============================================================================


class TestA1_21_SizingMarginBinds:
    def test_mbt_margin_binds(self) -> None:
        # MBT @ 95000: notional 9500, margin 2800. equity 100k, full_kelly 50.
        # kelly = 50*0.25*100k/9500 = 131; margin = 100k*0.5/2800 = 17; leverage = 100k*3/9500 = 31.
        d = RiskManager().size_position(
            ContractSymbol("MBTM26"), Decimal("50"), Decimal("100000"), Decimal("95000")
        )
        assert d.kelly_contracts == 131
        assert d.margin_contracts == 17
        assert d.leverage_contracts == 31
        assert d.contracts == 17
        assert d.binding_constraint == "margin"

    def test_margin_binds_with_high_leverage_param(self) -> None:
        # Raise leverage_cap so leverage doesn't bind -> margin (32) binds for ES.
        rm = RiskManager(RiskParams(leverage_cap=Decimal("20")))
        d = rm.size_position(
            ContractSymbol("ESM26"), Decimal("100"), Decimal("1000000"), Decimal("5000")
        )
        assert d.leverage_contracts == 80
        assert d.margin_contracts == 32
        assert d.contracts == 32
        assert d.binding_constraint == "margin"


# ============================================================================
# TestA1_21_NoEdge
# ============================================================================


class TestA1_21_NoEdge:
    def test_zero_kelly_no_position(self) -> None:
        d = RiskManager().size_position(
            ContractSymbol("ESM26"), Decimal("0"), Decimal("1000000"), Decimal("5000")
        )
        assert d.contracts == 0
        assert d.binding_constraint == "no_edge"
        assert d.kelly_contracts == 0
        # caps still computed for audit transparency
        assert d.margin_contracts == 32
        assert d.leverage_contracts == 12

    def test_negative_kelly_no_position(self) -> None:
        d = RiskManager().size_position(
            ContractSymbol("ESM26"), Decimal("-1.5"), Decimal("1000000"), Decimal("5000")
        )
        assert d.contracts == 0
        assert d.binding_constraint == "no_edge"


# ============================================================================
# TestA1_21_Validation
# ============================================================================


class TestA1_21_Validation:
    def test_zero_equity_raises(self) -> None:
        with pytest.raises(ValueError, match="account_equity must be > 0"):
            RiskManager().size_position(
                ContractSymbol("ESM26"), Decimal("1"), Decimal("0"), Decimal("5000")
            )

    def test_negative_equity_raises(self) -> None:
        with pytest.raises(ValueError, match="account_equity must be > 0"):
            RiskManager().size_position(
                ContractSymbol("ESM26"), Decimal("1"), Decimal("-100"), Decimal("5000")
            )

    def test_zero_price_raises(self) -> None:
        with pytest.raises(ValueError, match="price must be > 0"):
            RiskManager().size_position(
                ContractSymbol("ESM26"), Decimal("1"), Decimal("1000000"), Decimal("0")
            )

    def test_unknown_contract_raises(self) -> None:
        with pytest.raises(RiskManagerError, match="no initial margin configured"):
            RiskManager().size_position(
                ContractSymbol("XYM26"), Decimal("1"), Decimal("1000000"), Decimal("5000")
            )

    def test_short_symbol_raises(self) -> None:
        with pytest.raises(RiskManagerError, match="too short to parse root"):
            RiskManager().size_position(
                ContractSymbol("ES"), Decimal("1"), Decimal("1000000"), Decimal("5000")
            )


# ============================================================================
# TestA1_21_DecimalExactness
# ============================================================================


class TestA1_21_DecimalExactness:
    def test_cl_notional_exact(self) -> None:
        # CL @ 75.50: notional = 75.50 * 1000 = 75500 exact (no float drift)
        d = RiskManager().size_position(
            ContractSymbol("CLN26"), Decimal("10"), Decimal("1000000"), Decimal("75.50")
        )
        assert d.notional_per_contract == Decimal("75500.00")

    def test_floor_truncates_not_rounds(self) -> None:
        # leverage = 1M*3/250k = 12.0 exactly; ensure exact int 12 (not 11/13)
        d = RiskManager().size_position(
            ContractSymbol("ESM26"), Decimal("100"), Decimal("1000000"), Decimal("5000")
        )
        assert d.leverage_contracts == 12


# ============================================================================
# TestA1_21_SizingDecision
# ============================================================================


class TestA1_21_SizingDecision:
    def test_negative_contracts_raises(self) -> None:
        with pytest.raises(ValueError, match="contracts must be >= 0"):
            SizingDecision(
                contracts=-1,
                binding_constraint="kelly",
                kelly_contracts=0,
                margin_contracts=0,
                leverage_contracts=0,
                notional_per_contract=Decimal("1"),
                margin_per_contract=Decimal("1"),
            )

    def test_frozen(self) -> None:
        d = RiskManager().size_position(
            ContractSymbol("ESM26"), Decimal("2.0"), Decimal("1000000"), Decimal("5000")
        )
        with pytest.raises(AttributeError):
            d.contracts = 99  # type: ignore[misc]


# ============================================================================
# TestA1_21_SurvivalGate  (leverage-survival bootstrap as the opt-in 4th cap)
# ============================================================================

# big down bars (-15% to -20%): at 3x leverage these breach the 30% kill switch -> gate must tighten
_RUINOUS = [0.05, -0.20, 0.03, -0.04, 0.06, -0.18, 0.02, -0.05, 0.04, -0.03, 0.05, -0.15]
_BENIGN = [0.01, 0.008, 0.012, 0.009, 0.011, 0.007, 0.013, 0.01, 0.009, 0.012, 0.008, 0.01]


def _surv_rm() -> RiskManager:
    # small bootstrap for fast, deterministic tests (defaults are spec B=10000)
    return RiskManager(RiskParams(survival_paths=200, survival_horizon=60))


class TestA1_21_SurvivalGate:
    def test_returns_none_skips_gate_backward_compatible(self) -> None:
        # no returns -> survival cap not evaluated -> identical to the 3-cap behavior
        d = RiskManager().size_position(
            ContractSymbol("MESU26"), Decimal("20"), Decimal("50000"), Decimal("5000")
        )
        assert d.survival_contracts is None
        assert d.contracts == d.leverage_contracts == 6  # leverage binds at 3x

    def test_ruinous_returns_tighten_below_leverage_cap(self) -> None:
        d = _surv_rm().size_position(
            ContractSymbol("MESU26"),
            Decimal("20"),
            Decimal("50000"),
            Decimal("5000"),
            returns=_RUINOUS,
            seed=7,
        )
        assert d.leverage_contracts == 6
        assert d.survival_contracts is not None
        assert 0 <= d.survival_contracts < d.leverage_contracts  # the gate de-levered the position
        assert d.contracts == d.survival_contracts
        assert d.binding_constraint == "survival"

    def test_benign_returns_do_not_tighten(self) -> None:
        d = _surv_rm().size_position(
            ContractSymbol("MESU26"),
            Decimal("20"),
            Decimal("50000"),
            Decimal("5000"),
            returns=_BENIGN,
            seed=7,
        )
        # low-vol returns survive 3x -> survival cap == base -> leverage stays the binding cap
        assert d.survival_contracts == 6
        assert d.contracts == 6
        assert d.binding_constraint == "leverage"

    def test_deterministic_with_seed(self) -> None:
        a = _surv_rm().size_position(
            ContractSymbol("MESU26"),
            Decimal("20"),
            Decimal("50000"),
            Decimal("5000"),
            returns=_RUINOUS,
            seed=11,
        )
        b = _surv_rm().size_position(
            ContractSymbol("MESU26"),
            Decimal("20"),
            Decimal("50000"),
            Decimal("5000"),
            returns=_RUINOUS,
            seed=11,
        )
        assert a.survival_contracts == b.survival_contracts

    def test_no_edge_reports_survival_none(self) -> None:
        d = _surv_rm().size_position(
            ContractSymbol("MESU26"),
            Decimal("-1"),
            Decimal("50000"),
            Decimal("5000"),
            returns=_RUINOUS,
            seed=7,
        )
        assert d.binding_constraint == "no_edge"
        assert d.contracts == 0
        assert d.survival_contracts is None

    def test_bad_survival_params_raise(self) -> None:
        with pytest.raises(ValueError, match="survival_floor must be in"):
            RiskParams(survival_floor=Decimal("1.5"))
        with pytest.raises(ValueError, match="kill_switch_dd must be in"):
            RiskParams(kill_switch_dd=Decimal("0"))
