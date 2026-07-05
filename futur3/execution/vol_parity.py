"""futur3.execution.vol_parity - inverse-volatility (risk-parity) leg weighting.

A cross-sectional basket sizes every leg by ONE shared Kelly edge, but legs differ in volatility, so
equal-Kelly = UNEQUAL risk (a high-vol / high-notional market dominates the blended P&L - internal
research flagged this as why a basket is vol-dominated). Inverse-vol weighting spreads the bet
so each leg contributes ~equal risk: weight_i ~ 1/stdev(returns_i), normalized so the MEAN weight is
1 - it RE-WEIGHTS, it does NOT change the overall bet scale (the RiskManager still caps each leg).

A leg with too little history or zero vol gets the neutral mean inverse-vol (no penalty); all-equal
vols -> all weights 1 (== no reweight). Pure stdlib + Decimal, deterministic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Final

_MIN_VOL_OBS: Final[int] = 2  # need >= 2 returns for a sample stdev


def _stdev(returns: Sequence[Decimal]) -> Decimal | None:
    """Sample stdev of `returns`, or None when undefined (< 2 returns) or degenerate (zero)."""
    n = len(returns)
    if n < _MIN_VOL_OBS:
        return None
    mean = sum(returns, start=Decimal(0)) / n
    var = sum(((r - mean) ** 2 for r in returns), start=Decimal(0)) / (n - 1)
    return var.sqrt() if var > 0 else None


def inverse_vol_weights[K](returns_by_key: Mapping[K, Sequence[Decimal]]) -> dict[K, Decimal]:
    """Normalized inverse-vol weights (mean == 1) for risk-parity leg sizing.

    weight_i ~ 1/stdev(returns_i), rescaled so the mean weight is 1 (preserves bet scale; only
    redistributes risk). Degenerate legs (< 2 returns or zero vol) get the neutral mean inverse-vol;
    if NOTHING is measurable, every weight is 1 (no reweight). Empty input -> empty."""
    if not returns_by_key:
        return {}
    inv: dict[K, Decimal | None] = {}
    for key, rets in returns_by_key.items():
        sd = _stdev(rets)
        inv[key] = (Decimal(1) / sd) if sd is not None else None
    finite = [x for x in inv.values() if x is not None]
    if not finite:  # nothing measurable -> neutral, all weights 1
        return {key: Decimal(1) for key in returns_by_key}
    neutral = sum(finite, start=Decimal(0)) / len(finite)
    raw = {key: (x if x is not None else neutral) for key, x in inv.items()}
    mean_raw = sum(raw.values(), start=Decimal(0)) / len(raw)
    return {key: raw[key] / mean_raw for key in raw}


__all__: list[str] = ["inverse_vol_weights"]
