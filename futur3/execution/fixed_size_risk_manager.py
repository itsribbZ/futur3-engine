"""FixedSizeRiskManager — trade a FIXED deployable contract count.

A prop or small self-funded account trades a FIXED small size (e.g. 1 MNQ), NOT a
size that scales with equity and price the way `RiskManager.size_position` does.
On the 3yr NQ minute cache, the equity/leverage-scaled sizer yields 1-3 MNQ
depending on the NQ price level (≈2 lots at 2023's ~12k prices, ≈1 at ~21k) — so
measuring prop survival at "the sized count" conflates the edge with a wandering
position size. This subclass pins the size to `contracts`, so the prop-survival
runner trades a clean, deployable, constant 1 MNQ.

Semantics:
- A positive-edge signal sizes to exactly `contracts`, CAPPED so it never exceeds
  what the base margin / leverage / survival caps can afford (you cannot trade size
  you cannot fund). For 1 MNQ on a $50k account those caps are ≥3, so the pin holds.
- A no-edge signal (`full_kelly_fraction <= 0`) still sizes to 0 (preserved).
- The Kelly/margin/leverage/survival diagnostics from the base sizer are PRESERVED
  on the returned SizingDecision (audit trail); only `contracts` is overridden.

This is additive — it subclasses RiskManager and overrides one method. The proven
forward-paper runners use the plain RiskManager and are unaffected.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from decimal import Decimal

from futur3.data.types import ContractSymbol
from futur3.execution.margin_source import MarginSource
from futur3.execution.risk_manager import (
    RiskManager,
    RiskParams,
    SizingDecision,
)


class FixedSizeRiskManager(RiskManager):
    """RiskManager that pins position size to a fixed deployable contract count."""

    def __init__(
        self,
        *,
        contracts: int,
        params: RiskParams | None = None,
        margin_source: MarginSource | None = None,
    ) -> None:
        super().__init__(params, margin_source)
        if contracts < 1:
            raise ValueError(f"contracts must be >= 1; got {contracts}")
        self._fixed = contracts

    @property
    def fixed_contracts(self) -> int:
        return self._fixed

    def size_position(
        self,
        contract: ContractSymbol,
        full_kelly_fraction: Decimal,
        account_equity: Decimal,
        price: Decimal,
        *,
        returns: Sequence[float] | None = None,
        seed: int | None = None,
    ) -> SizingDecision:
        base = super().size_position(
            contract,
            full_kelly_fraction,
            account_equity,
            price,
            returns=returns,
            seed=seed,
        )
        if full_kelly_fraction <= 0:
            return base  # no edge -> 0 contracts, unchanged
        # Never exceed what the real caps can fund/survive.
        affordable = [base.margin_contracts, base.leverage_contracts]
        if base.survival_contracts is not None:
            affordable.append(base.survival_contracts)
        return replace(base, contracts=min(self._fixed, min(affordable)))
