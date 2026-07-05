"""futur3.execution.slippage - SlippageModel ABC + TickHaircutSlippageModel (Phase A1).

Per internal design notes.

The slippage model adjusts an intended fill price to a realistic executed price for BACKTEST
cost realism. Per BACKTEST-IS-LIVE it is injected via RuntimeContext:
- Backtest mode: apply before the MockBroker fill (model.apply_slippage(...) -> fill_order(...)).
- Paper/live mode: NOT applied - real broker fills already embed execution slippage.

Phase A1 ships the ABC + `TickHaircutSlippageModel` (a fixed per-side tick haircut, 3 bands).
The volume-dependent `AlmgrenChrissSlippageModel` (Phase B) and empirically-calibrated
`MarketRealitySlippageModel` (Phase C) extend this ABC when those phases arrive.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal
from typing import Final, Literal

from futur3.data.types import ContractSymbol, _assert_tz_aware
from futur3.execution.broker import Side

SlippageBand = Literal["CONSERVATIVE", "REALISTIC", "PESSIMISTIC"]

SLIPPAGE_BAND_ENV_VAR: Final[str] = "FUTUR3_SLIPPAGE_BAND"
DEFAULT_BAND: Final[SlippageBand] = "CONSERVATIVE"

# Per-side tick haircut by band. Round-trip = 2x per-side: CONSERVATIVE 4 RT / REALISTIC 2 RT /
# PESSIMISTIC 8 RT (internal design notes).
_BAND_TICKS_PER_SIDE: Final[dict[SlippageBand, int]] = {
    "CONSERVATIVE": 2,
    "REALISTIC": 1,
    "PESSIMISTIC": 4,
}

# Price increment (tick size) per contract root. T1/T2-cited in internal microstructure notes
# (ES/MES/NQ/MNQ section, CL/MCL, GC/MGC, MBT outright $5/BTC, MET outright 0.5 index pt).
_SYMBOL_SUFFIX_LEN: Final[int] = 3  # <month_code><year_2digit>, e.g. "M26"
CONTRACT_TICK_SIZE: Final[dict[str, Decimal]] = {
    "ES": Decimal("0.25"),
    "MES": Decimal("0.25"),
    "NQ": Decimal("0.25"),
    "MNQ": Decimal("0.25"),
    "CL": Decimal("0.01"),
    "MCL": Decimal("0.01"),
    "GC": Decimal("0.10"),
    "MGC": Decimal("0.10"),
    "MBT": Decimal("5"),  # $5/BTC index move = $0.50/contract (0.1 BTC)
    "MET": Decimal("0.5"),  # 0.5 index pt = $0.05/contract (0.1 ETH)
    # Broaden wave 1 (tick value = tick x multiplier, CME-verified):
    "YM": Decimal("1"),  # 1 index pt x $5 = $5
    "RTY": Decimal("0.10"),  # 0.1 pt x $50 = $5
    "6E": Decimal("0.00005"),  # x $125k = $6.25
    "6A": Decimal("0.00005"),  # x $100k = $5
    "ZN": Decimal("0.015625"),  # 1/2 of 1/32 x $1000 = $15.625
    "ZB": Decimal("0.03125"),  # 1/32 x $1000 = $31.25
    # Broaden wave 2 (CME-verified; tick x mult = tick value):
    "6J": Decimal("0.0000005"),  # x 12.5M JPY = $6.25 (raw ~0.0063-0.007 USD/yen; scale-checked)
    "6B": Decimal("0.0001"),  # x 62.5k GBP = $6.25
    "6C": Decimal("0.00005"),  # x 100k CAD = $5.00
    "6S": Decimal("0.0001"),  # x 125k CHF = $12.50
    "HG": Decimal("0.0005"),  # copper, x 25,000 lb = $12.50
    "SI": Decimal("0.005"),  # silver, x 5,000 oz = $25.00
    "ZT": Decimal("0.00390625"),  # 1/8 of 1/32 x $2000 = $7.8125
    "ZF": Decimal("0.0078125"),  # 1/4 of 1/32 x $1000 = $7.8125
    "UB": Decimal("0.03125"),  # 1/32 x $1000 = $31.25
    "NKD": Decimal("5"),  # Nikkei 225, 5 idx pt x $5 = $25.00
}


# ----------------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------------


class SlippageError(Exception):
    """Base for slippage-model errors (unknown contract, unknown band)."""


class UnknownSlippageBandError(SlippageError):
    """Configured slippage band is not one of CONSERVATIVE / REALISTIC / PESSIMISTIC.

    Raised loudly (never silent-fallback per the fail-loud policy) by the band resolver / model constructor.
    """


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _contract_root(contract: ContractSymbol) -> str:
    """Strip the <month_code><year_2digit> suffix, e.g. ContractSymbol("ESM26") -> "ES"."""
    s = str(contract)
    if len(s) <= _SYMBOL_SUFFIX_LEN:
        raise SlippageError(f"contract symbol too short to parse root: {s!r}")
    return s[:-_SYMBOL_SUFFIX_LEN]


def tick_size_for(contract: ContractSymbol) -> Decimal:
    """Price increment for a contract. Raises SlippageError if the root is not configured."""
    root = _contract_root(contract)
    try:
        return CONTRACT_TICK_SIZE[root]
    except KeyError as exc:
        raise SlippageError(
            f"no tick size configured for contract root {root!r} (from {contract!r}); "
            f"known roots: {sorted(CONTRACT_TICK_SIZE)}"
        ) from exc


def resolve_slippage_band(env: Mapping[str, str]) -> SlippageBand:
    """Read FUTUR3_SLIPPAGE_BAND from `env` (default CONSERVATIVE; case-insensitive).

    Unknown value raises UnknownSlippageBandError - no silent fallback (fail-loud).
    """
    raw = env.get(SLIPPAGE_BAND_ENV_VAR, DEFAULT_BAND)
    normalized = raw.strip().upper()
    if normalized not in _BAND_TICKS_PER_SIDE:
        raise UnknownSlippageBandError(
            f"Unknown {SLIPPAGE_BAND_ENV_VAR}={raw!r}; expected one of "
            f"{sorted(_BAND_TICKS_PER_SIDE)}"
        )
    return normalized


# ----------------------------------------------------------------------------
# SlippageModel ABC + TickHaircutSlippageModel
# ----------------------------------------------------------------------------


class SlippageModel(abc.ABC):
    """Maps an intended fill price to a realistic executed price for backtest cost realism."""

    @abc.abstractmethod
    def apply_slippage(
        self,
        intended_price: Decimal,
        side: Side,
        quantity: int,
        contract: ContractSymbol,
        ts: datetime,
    ) -> Decimal:
        """Return the executed fill price after slippage.

        Adverse by convention: a BUY fills no better than (>=) intended; a SELL fills no
        better than (<=) intended. `ts` must be IANA-TZ-aware.
        """
        ...


class TickHaircutSlippageModel(SlippageModel):
    """Fixed per-side tick haircut, selectable across 3 bands (Phase A1 default).

    CONSERVATIVE = 2 ticks/side, REALISTIC = 1, PESSIMISTIC = 4. The haircut is constant per
    fill (quantity-independent); quantity-proportional impact is the Phase B AlmgrenChriss model.
    """

    def __init__(self, band: SlippageBand = DEFAULT_BAND) -> None:
        if band not in _BAND_TICKS_PER_SIDE:
            raise UnknownSlippageBandError(
                f"Unknown slippage band {band!r}; expected one of {sorted(_BAND_TICKS_PER_SIDE)}"
            )
        self._band: SlippageBand = band

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> TickHaircutSlippageModel:
        """Construct from FUTUR3_SLIPPAGE_BAND (default CONSERVATIVE)."""
        return cls(resolve_slippage_band(env))

    @property
    def band(self) -> SlippageBand:
        return self._band

    @property
    def ticks_per_side(self) -> int:
        return _BAND_TICKS_PER_SIDE[self._band]

    def apply_slippage(
        self,
        intended_price: Decimal,
        side: Side,
        quantity: int,
        contract: ContractSymbol,
        ts: datetime,
    ) -> Decimal:
        if intended_price <= 0:
            raise ValueError(f"intended_price must be > 0; got {intended_price}")
        if quantity <= 0:
            raise ValueError(f"quantity must be > 0; got {quantity}")
        _assert_tz_aware(ts, "TickHaircutSlippageModel.apply_slippage ts")
        adjustment = tick_size_for(contract) * _BAND_TICKS_PER_SIDE[self._band]
        return intended_price + adjustment if side is Side.BUY else intended_price - adjustment

    def __repr__(self) -> str:
        return (
            f"<TickHaircutSlippageModel band={self._band!r} ticks_per_side={self.ticks_per_side}>"
        )
