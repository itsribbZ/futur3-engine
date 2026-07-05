"""A1.22a MarginSource seam test suite.

Per internal design notes: the injectable margin seam that makes live CME SPAN margins a drop-in for the static Phase
A1 estimates in RiskManager WITH NO FORMULA CHANGE. Covers:
- StaticMarginSource: lookup hit, unknown-root MarginSourceError ("no initial margin configured"),
  known_roots; runtime_checkable MarginSource protocol conformance.
- RiskManager wiring: default uses CONTRACT_INITIAL_MARGIN unchanged; an injected MarginSource
  flows into the sizing math (margin_per_contract + margin_contracts reflect it); the historical
  RiskManagerError error contract is preserved through the seam; positional (params, source) ctor.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from futur3.data.types import ContractSymbol
from futur3.execution import (
    CONTRACT_INITIAL_MARGIN,
    MarginSource,
    MarginSourceError,
    RiskManager,
    RiskManagerError,
    RiskParams,
    StaticMarginSource,
)
from futur3.execution.margin_source import StaticMarginSource as _ModStaticMarginSource

# ============================================================================
# TestA1_22a_StaticMarginSource
# ============================================================================


class TestA1_22a_StaticMarginSource:
    def test_initial_margin_hit(self) -> None:
        src = StaticMarginSource({"ES": Decimal("15400")})
        assert src.initial_margin("ES") == Decimal("15400")

    def test_unknown_root_raises(self) -> None:
        src = StaticMarginSource({"ES": Decimal("15400")})
        with pytest.raises(MarginSourceError, match="no initial margin configured"):
            src.initial_margin("NQ")

    def test_unknown_root_lists_known(self) -> None:
        src = StaticMarginSource({"ES": Decimal("15400"), "NQ": Decimal("22800")})
        with pytest.raises(MarginSourceError, match=r"known roots: \['ES', 'NQ'\]"):
            src.initial_margin("CL")

    def test_known_roots(self) -> None:
        src = StaticMarginSource({"ES": Decimal("15400"), "MES": Decimal("1540")})
        assert src.known_roots() == frozenset({"ES", "MES"})

    def test_copies_input_mapping(self) -> None:
        # mutating the caller's dict after construction must not change the source
        d = {"ES": Decimal("15400")}
        src = StaticMarginSource(d)
        d["ES"] = Decimal("1")
        assert src.initial_margin("ES") == Decimal("15400")

    def test_satisfies_protocol(self) -> None:
        assert isinstance(StaticMarginSource({}), MarginSource)

    def test_package_export_matches_module(self) -> None:
        assert StaticMarginSource is _ModStaticMarginSource


# ============================================================================
# TestA1_22a_RiskManagerWiring
# ============================================================================


class TestA1_22a_RiskManagerWiring:
    def test_default_source_uses_static_registry(self) -> None:
        rm = RiskManager()
        d = rm.size_position(
            ContractSymbol("ESM26"), Decimal("2.0"), Decimal("50000"), Decimal("5260")
        )
        assert d.margin_per_contract == CONTRACT_INITIAL_MARGIN["ES"]

    def test_default_margin_source_property(self) -> None:
        assert isinstance(RiskManager().margin_source, MarginSource)

    def test_injected_source_flows_into_sizing(self) -> None:
        rm = RiskManager(margin_source=StaticMarginSource({"ES": Decimal("1000")}))
        d = rm.size_position(
            ContractSymbol("ESM26"), Decimal("2.0"), Decimal("50000"), Decimal("5260")
        )
        assert d.margin_per_contract == Decimal("1000")
        # margin_contracts = floor(equity * 0.50 / margin) = floor(50000 * 0.5 / 1000) = 25
        assert d.margin_contracts == 25

    def test_injected_margin_is_unique_binder(self) -> None:
        # MES micro (notional 26,300) with high kelly + high leverage cap so kelly (23) and
        # leverage (190) both exceed the injected-margin cap (10) -> margin is the unique binder.
        rm = RiskManager(
            RiskParams(leverage_cap=Decimal("100")),
            StaticMarginSource({"MES": Decimal("2500")}),
        )
        d = rm.size_position(
            ContractSymbol("MESU26"), Decimal("50"), Decimal("50000"), Decimal("5260")
        )
        assert d.margin_per_contract == Decimal("2500")
        assert d.margin_contracts == 10  # floor(50000 * 0.5 / 2500)
        assert d.kelly_contracts == 23 and d.leverage_contracts == 190
        assert d.contracts == 10
        assert d.binding_constraint == "margin"

    def test_injected_unknown_root_preserves_error_contract(self) -> None:
        rm = RiskManager(margin_source=StaticMarginSource({}))  # knows nothing
        with pytest.raises(RiskManagerError, match="no initial margin configured"):
            rm.size_position(
                ContractSymbol("ESM26"), Decimal("2.0"), Decimal("50000"), Decimal("5260")
            )

    def test_positional_params_and_source(self) -> None:
        # ctor stays backward-compatible: params positional, margin_source 2nd positional
        rm = RiskManager(
            RiskParams(leverage_cap=Decimal("20")),
            StaticMarginSource({"ES": Decimal("9999")}),
        )
        d = rm.size_position(
            ContractSymbol("ESM26"), Decimal("2.0"), Decimal("50000"), Decimal("5260")
        )
        assert d.margin_per_contract == Decimal("9999")
        assert rm.params.leverage_cap == Decimal("20")
