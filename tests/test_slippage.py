"""A1.19 SlippageModel + TickHaircutSlippageModel test suite.

Test discipline:
- ABC enforces abstractness; TickHaircut is the Phase A1 default (3 bands).
- Per-side tick haircut applied ADVERSELY (BUY up, SELL down) by ticks * tick_size.
- Exact Decimal arithmetic (never float) verified per contract + band.
- Tick sizes match the T1/T2-cited table in internal design notes (all 10 contracts).
- Env loader (FUTUR3_SLIPPAGE_BAND) defaults CONSERVATIVE, case-insensitive, unknown raises (fail-loud).
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from futur3.data.types import ContractSymbol
from futur3.execution import TickHaircutSlippageModel as _PkgTickHaircut
from futur3.execution.broker import Side
from futur3.execution.slippage import (
    CONTRACT_TICK_SIZE,
    SlippageError,
    SlippageModel,
    TickHaircutSlippageModel,
    UnknownSlippageBandError,
    resolve_slippage_band,
    tick_size_for,
)

_TS = datetime(2026, 6, 5, 14, 0, tzinfo=UTC)


# ============================================================================
# TestA1_19_Imports
# ============================================================================


class TestA1_19_Imports:
    def test_model_importable(self) -> None:
        assert SlippageModel is not None
        assert TickHaircutSlippageModel is not None

    def test_errors_importable(self) -> None:
        assert issubclass(UnknownSlippageBandError, SlippageError)

    def test_exported_from_execution_package(self) -> None:
        assert _PkgTickHaircut is TickHaircutSlippageModel


# ============================================================================
# TestA1_19_ContractTickSize
# ============================================================================


class TestA1_19_ContractTickSize:
    def test_configured_contracts_present(self) -> None:
        assert set(CONTRACT_TICK_SIZE) == {
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

    def test_known_tick_sizes(self) -> None:
        assert CONTRACT_TICK_SIZE["ES"] == Decimal("0.25")
        assert CONTRACT_TICK_SIZE["CL"] == Decimal("0.01")
        assert CONTRACT_TICK_SIZE["GC"] == Decimal("0.10")
        assert CONTRACT_TICK_SIZE["MBT"] == Decimal("5")
        assert CONTRACT_TICK_SIZE["MET"] == Decimal("0.5")

    def test_tick_size_for_parses_root(self) -> None:
        assert tick_size_for(ContractSymbol("ESM26")) == Decimal("0.25")
        assert tick_size_for(ContractSymbol("MESM26")) == Decimal("0.25")
        assert tick_size_for(ContractSymbol("MBTM26")) == Decimal("5")

    def test_unknown_root_raises(self) -> None:
        with pytest.raises(SlippageError, match="no tick size configured"):
            tick_size_for(ContractSymbol("XYM26"))

    def test_short_symbol_raises(self) -> None:
        with pytest.raises(SlippageError, match="too short to parse root"):
            tick_size_for(ContractSymbol("ES"))


# ============================================================================
# TestA1_19_BandTicks
# ============================================================================


class TestA1_19_BandTicks:
    def test_conservative_two_per_side(self) -> None:
        assert TickHaircutSlippageModel("CONSERVATIVE").ticks_per_side == 2

    def test_realistic_one_per_side(self) -> None:
        assert TickHaircutSlippageModel("REALISTIC").ticks_per_side == 1

    def test_pessimistic_four_per_side(self) -> None:
        assert TickHaircutSlippageModel("PESSIMISTIC").ticks_per_side == 4


# ============================================================================
# TestA1_19_Construction
# ============================================================================


class TestA1_19_Construction:
    def test_default_band_conservative(self) -> None:
        assert TickHaircutSlippageModel().band == "CONSERVATIVE"

    def test_explicit_band(self) -> None:
        assert TickHaircutSlippageModel("PESSIMISTIC").band == "PESSIMISTIC"

    def test_bad_band_raises(self) -> None:
        with pytest.raises(UnknownSlippageBandError, match="Unknown slippage band"):
            TickHaircutSlippageModel("AGGRESSIVE")  # type: ignore[arg-type]

    def test_repr(self) -> None:
        assert repr(TickHaircutSlippageModel("REALISTIC")) == (
            "<TickHaircutSlippageModel band='REALISTIC' ticks_per_side=1>"
        )


# ============================================================================
# TestA1_19_ApplySlippageDirection
# ============================================================================


class TestA1_19_ApplySlippageDirection:
    def test_buy_fills_higher(self) -> None:
        m = TickHaircutSlippageModel("CONSERVATIVE")  # 2 ticks x 0.25 = 0.50
        out = m.apply_slippage(Decimal("5000.00"), Side.BUY, 1, ContractSymbol("ESM26"), _TS)
        assert out == Decimal("5000.50")

    def test_sell_fills_lower(self) -> None:
        m = TickHaircutSlippageModel("CONSERVATIVE")
        out = m.apply_slippage(Decimal("5000.00"), Side.SELL, 1, ContractSymbol("ESM26"), _TS)
        assert out == Decimal("4999.50")

    def test_realistic_one_tick(self) -> None:
        m = TickHaircutSlippageModel("REALISTIC")
        out = m.apply_slippage(Decimal("5000.00"), Side.BUY, 1, ContractSymbol("ESM26"), _TS)
        assert out == Decimal("5000.25")

    def test_pessimistic_four_ticks(self) -> None:
        m = TickHaircutSlippageModel("PESSIMISTIC")
        out = m.apply_slippage(Decimal("5000.00"), Side.BUY, 1, ContractSymbol("ESM26"), _TS)
        assert out == Decimal("5001.00")


# ============================================================================
# TestA1_19_ApplySlippagePerContract (exact Decimal, CONSERVATIVE band)
# ============================================================================


class TestA1_19_ApplySlippagePerContract:
    def _buy(self, price: str, symbol: str) -> Decimal:
        m = TickHaircutSlippageModel("CONSERVATIVE")
        return m.apply_slippage(Decimal(price), Side.BUY, 1, ContractSymbol(symbol), _TS)

    def test_nq(self) -> None:
        assert self._buy("18000.00", "NQM26") == Decimal("18000.50")  # 2 x 0.25

    def test_cl(self) -> None:
        assert self._buy("75.50", "CLN26") == Decimal("75.52")  # 2 x 0.01

    def test_gc(self) -> None:
        assert self._buy("2900.0", "GCQ26") == Decimal("2900.20")  # 2 x 0.10

    def test_mbt(self) -> None:
        assert self._buy("95000", "MBTM26") == Decimal("95010")  # 2 x 5

    def test_met(self) -> None:
        assert self._buy("3400.0", "METM26") == Decimal("3401.0")  # 2 x 0.5

    def test_cl_decimal_precision_exact(self) -> None:
        # 0.01 tick must not drift via float; result is exact Decimal
        out = self._buy("75.51", "MCLN26")
        assert out == Decimal("75.53")
        assert str(out) == "75.53"


# ============================================================================
# TestA1_19_ApplySlippageValidation
# ============================================================================


class TestA1_19_ApplySlippageValidation:
    def setup_method(self) -> None:
        self.m = TickHaircutSlippageModel()

    def test_zero_price_raises(self) -> None:
        with pytest.raises(ValueError, match="intended_price must be > 0"):
            self.m.apply_slippage(Decimal("0"), Side.BUY, 1, ContractSymbol("ESM26"), _TS)

    def test_negative_price_raises(self) -> None:
        with pytest.raises(ValueError, match="intended_price must be > 0"):
            self.m.apply_slippage(Decimal("-1"), Side.BUY, 1, ContractSymbol("ESM26"), _TS)

    def test_zero_quantity_raises(self) -> None:
        with pytest.raises(ValueError, match="quantity must be > 0"):
            self.m.apply_slippage(Decimal("5000"), Side.BUY, 0, ContractSymbol("ESM26"), _TS)

    def test_naive_ts_raises(self) -> None:
        with pytest.raises(ValueError, match="apply_slippage ts must be IANA-TZ-aware"):
            self.m.apply_slippage(
                Decimal("5000"), Side.BUY, 1, ContractSymbol("ESM26"), datetime(2026, 6, 5, 14, 0)
            )

    def test_unknown_contract_raises(self) -> None:
        with pytest.raises(SlippageError, match="no tick size configured"):
            self.m.apply_slippage(Decimal("100"), Side.BUY, 1, ContractSymbol("XYM26"), _TS)


# ============================================================================
# TestA1_19_QuantityIndependence
# ============================================================================


class TestA1_19_QuantityIndependence:
    def test_haircut_independent_of_quantity(self) -> None:
        m = TickHaircutSlippageModel("CONSERVATIVE")
        one = m.apply_slippage(Decimal("5000.00"), Side.BUY, 1, ContractSymbol("ESM26"), _TS)
        hundred = m.apply_slippage(Decimal("5000.00"), Side.BUY, 100, ContractSymbol("ESM26"), _TS)
        assert one == hundred == Decimal("5000.50")


# ============================================================================
# TestA1_19_EnvLoader
# ============================================================================


class TestA1_19_EnvLoader:
    def test_default_when_unset(self) -> None:
        assert resolve_slippage_band({}) == "CONSERVATIVE"

    def test_explicit(self) -> None:
        assert resolve_slippage_band({"FUTUR3_SLIPPAGE_BAND": "PESSIMISTIC"}) == "PESSIMISTIC"

    def test_case_insensitive(self) -> None:
        assert resolve_slippage_band({"FUTUR3_SLIPPAGE_BAND": "realistic"}) == "REALISTIC"

    def test_whitespace_trimmed(self) -> None:
        assert resolve_slippage_band({"FUTUR3_SLIPPAGE_BAND": "  CONSERVATIVE  "}) == "CONSERVATIVE"

    def test_unknown_raises(self) -> None:
        with pytest.raises(UnknownSlippageBandError, match="Unknown FUTUR3_SLIPPAGE_BAND"):
            resolve_slippage_band({"FUTUR3_SLIPPAGE_BAND": "WILD"})

    def test_from_env_constructs_model(self) -> None:
        m = TickHaircutSlippageModel.from_env({"FUTUR3_SLIPPAGE_BAND": "REALISTIC"})
        assert m.band == "REALISTIC"


# ============================================================================
# TestA1_19_ABCCompliance
# ============================================================================


class TestA1_19_ABCCompliance:
    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            SlippageModel()  # type: ignore[abstract]

    def test_partial_subclass_still_abstract(self) -> None:
        class _Partial(SlippageModel):
            pass  # does not implement apply_slippage

        with pytest.raises(TypeError, match="abstract"):
            _Partial()  # type: ignore[abstract]

    def test_tick_haircut_is_slippage_model(self) -> None:
        assert isinstance(TickHaircutSlippageModel(), SlippageModel)
