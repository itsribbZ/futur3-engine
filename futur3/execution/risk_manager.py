"""futur3.execution.risk_manager - position-sizing hard-gate (Phase A1).

Per the sizing design notes

Tighten-only composite sizing (principle A6): the number of contracts is the MINIMUM of
independent caps, so adding a cap can only ever reduce size, never increase it:

    contracts = min(
        quarter_kelly_contracts,   # MATH section 8: (k * f*) * equity / notional, k = 0.25
        margin_budget_contracts,   # MATH section 9: equity * 0.50 / margin_per_contract
        leverage_cap_contracts,    # MATH section 10: equity * 3 / contract_notional
    )

Phase A1 defaults (LOCKED internal design notes): Quarter Kelly (0.25) +
50% margin budget + 3x leverage cap. (MATH section 10 quotes a general 5x; the tighter 3x is the
Phase A1 lock.) The leverage-survival bootstrap is a further cap that
defaults to NOT BINDING until historical returns + the Phase 8 stats layer are wired (section 11:
"all caps default to infinity until their subsystem is wired"); it is intentionally absent here.

Per-contract initial margins come from an injectable `MarginSource` (A1.22a). The Phase A1 default
is `StaticMarginSource` over the static 2026 estimates in internal design notes(marked "verify Phase A1"); a live SPAN-backed source drops into the same slot with no change
to the sizing math (MATH section 9 is SPAN-ready by construction).

Composes with the engine loop: the engine reads `broker.get_account_metrics().equity`, then calls
`size_position(...)`. Mode-agnostic (backtest-is-live); no broker import here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from typing import Final, Literal

from futur3.data.types import ContractSymbol
from futur3.execution.margin_source import MarginSource, MarginSourceError, StaticMarginSource
from futur3.execution.survival import survival_probability

_SYMBOL_SUFFIX_LEN: Final[int] = 3  # <month_code><year_2digit>, e.g. "M26"
_MIN_SURVIVAL_RETURNS: Final[int] = 2  # the survival bootstrap needs >= 2 returns

# Per-contract initial margin (USD), static 2026 estimates per internal design notes
# section 4.2 ("verify Phase A1"). Wrapped by the default StaticMarginSource; inject a live
# SPAN-backed MarginSource to override (A1.22a). GC reflects the Jan-2026 %-of-notional regime.
# MBT corrected to 1/50-of-BTC ($2,800, not the prior $25K estimate).
CONTRACT_INITIAL_MARGIN: Final[dict[str, Decimal]] = {
    "ES": Decimal("15400"),
    "MES": Decimal("1540"),
    "NQ": Decimal("22800"),
    "MNQ": Decimal("2280"),
    "CL": Decimal("7200"),
    "MCL": Decimal("720"),
    "GC": Decimal("26100"),
    "MGC": Decimal("2610"),
    "MBT": Decimal("2800"),
    "MET": Decimal("1800"),
    # Universe broaden wave 1 (CME-verified; margins APPROXIMATE -> pull live before deployment):
    "YM": Decimal("1700"),
    "RTY": Decimal("3000"),
    "6E": Decimal("2600"),
    "6A": Decimal("1800"),
    "ZN": Decimal("2200"),
    "ZB": Decimal("4000"),
    # Universe broaden wave 2 (CME roots; margins APPROXIMATE May-2026 -> pull live SPAN before deploy):
    "6J": Decimal("2860"),
    "6B": Decimal("2090"),
    "6C": Decimal("990"),
    "6S": Decimal("4400"),
    "HG": Decimal("13200"),
    "SI": Decimal("46230"),
    "ZT": Decimal("1320"),
    "ZF": Decimal("1375"),
    "UB": Decimal("5665"),
    "NKD": Decimal("21874"),
}

# Contract point multiplier (USD per 1.0 price move) = tick_value / tick_size, T1/T2-cited from
# internal design notes. Used for contract_notional = price * multiplier.
CONTRACT_MULTIPLIER: Final[dict[str, Decimal]] = {
    "ES": Decimal("50"),
    "MES": Decimal("5"),
    "NQ": Decimal("20"),
    "MNQ": Decimal("2"),
    "CL": Decimal("1000"),
    "MCL": Decimal("100"),
    "GC": Decimal("100"),
    "MGC": Decimal("10"),
    "MBT": Decimal("0.1"),
    "MET": Decimal("0.1"),
    # Broaden wave 1: equity (YM $5/pt, RTY $50/pt), FX (6E $125k, 6A $100k), rates (ZN/ZB $1k/pt).
    "YM": Decimal("5"),
    "RTY": Decimal("50"),
    "6E": Decimal("125000"),
    "6A": Decimal("100000"),
    "ZN": Decimal("1000"),
    "ZB": Decimal("1000"),
    # Broaden wave 2: FX (6J 12.5M JPY, 6B 62.5k GBP, 6C 100k CAD, 6S 125k CHF), metals
    # (HG 25k lb, SI 5k oz), rates (ZT $2k/pt, ZF/UB $1k/pt), equity (NKD $5/idx-pt).
    "6J": Decimal("12500000"),
    "6B": Decimal("62500"),
    "6C": Decimal("100000"),
    "6S": Decimal("125000"),
    "HG": Decimal("25000"),
    "SI": Decimal("5000"),
    "ZT": Decimal("2000"),
    "ZF": Decimal("1000"),
    "UB": Decimal("1000"),
    "NKD": Decimal("5"),
}

BindingConstraint = Literal["kelly", "margin", "leverage", "survival", "no_edge"]


class RiskManagerError(Exception):
    """Risk-manager error (e.g. contract root not configured in margin/multiplier registries)."""


@dataclass(frozen=True)
class RiskParams:
    """Phase A1 sizing parameters (tighten-only). All defaults locked by design."""

    kelly_multiplier: Decimal = Decimal("0.25")  # Quarter Kelly (MATH section 8, k=0.25)
    margin_budget_pct: Decimal = Decimal("0.50")  # <= 50% of equity into margin (MATH section 9)
    leverage_cap: Decimal = Decimal("3")  # 3x notional (Phase A1 lock; general default 5x)
    # Leverage-survival bootstrap - the 4th cap, ACTIVE only when historical
    # returns are passed to size_position. Defaults match the spec; callers running the sim per-bar
    # should lower survival_paths for speed.
    survival_floor: Decimal = Decimal("0.995")  # keep a size only if P_survive >= this
    kill_switch_dd: Decimal = Decimal("0.30")  # kill-switch max drawdown in the survival sim
    survival_horizon: int = 252  # survival-path length (1 trading year)
    survival_paths: int = 10000  # survival-sim bootstrap paths B

    def __post_init__(self) -> None:
        if not 0 < self.kelly_multiplier <= 1:
            raise ValueError(f"kelly_multiplier must be in (0, 1]; got {self.kelly_multiplier}")
        if not 0 < self.margin_budget_pct <= 1:
            raise ValueError(f"margin_budget_pct must be in (0, 1]; got {self.margin_budget_pct}")
        if self.leverage_cap <= 0:
            raise ValueError(f"leverage_cap must be > 0; got {self.leverage_cap}")
        if not 0 < self.survival_floor < 1:
            raise ValueError(f"survival_floor must be in (0, 1); got {self.survival_floor}")
        if not 0 < self.kill_switch_dd < 1:
            raise ValueError(f"kill_switch_dd must be in (0, 1); got {self.kill_switch_dd}")
        if self.survival_horizon < 1:
            raise ValueError(f"survival_horizon must be >= 1; got {self.survival_horizon}")
        if self.survival_paths < 1:
            raise ValueError(f"survival_paths must be >= 1; got {self.survival_paths}")


@dataclass(frozen=True)
class SizingDecision:
    """Operator-readable sizing result: the chosen size, which cap bound it, and every
    candidate so the decision is auditable ("with these inputs, by rule X, sized N contracts")."""

    contracts: int
    binding_constraint: BindingConstraint
    kelly_contracts: int
    margin_contracts: int
    leverage_contracts: int
    notional_per_contract: Decimal
    margin_per_contract: Decimal
    survival_contracts: int | None = None  # survival cap; None = gate not run (no returns)

    def __post_init__(self) -> None:
        if self.contracts < 0:
            raise ValueError(f"contracts must be >= 0; got {self.contracts}")


def _contract_root(contract: ContractSymbol) -> str:
    s = str(contract)
    if len(s) <= _SYMBOL_SUFFIX_LEN:
        raise RiskManagerError(f"contract symbol too short to parse root: {s!r}")
    return s[:-_SYMBOL_SUFFIX_LEN]


def _lookup(
    registry: dict[str, Decimal], root: str, contract: ContractSymbol, what: str
) -> Decimal:
    try:
        return registry[root]
    except KeyError as exc:
        raise RiskManagerError(
            f"no {what} configured for contract root {root!r} (from {contract!r}); "
            f"known roots: {sorted(registry)}"
        ) from exc


def _floor_to_int(value: Decimal) -> int:
    """Floor a non-negative Decimal to int (ROUND_DOWN == floor for non-negatives)."""
    return int(value.to_integral_value(rounding=ROUND_DOWN))


class RiskManager:
    """Position sizing: tighten-only min of Kelly / margin / leverage caps."""

    def __init__(
        self,
        params: RiskParams | None = None,
        margin_source: MarginSource | None = None,
    ) -> None:
        self._params = params if params is not None else RiskParams()
        # A1.22a SPAN-ready seam: default is the static §4.2 estimates; inject a live SPAN-backed
        # MarginSource to override with NO change to the sizing math below.
        self._margin_source: MarginSource = (
            margin_source
            if margin_source is not None
            else StaticMarginSource(CONTRACT_INITIAL_MARGIN)
        )

    @property
    def params(self) -> RiskParams:
        return self._params

    @property
    def margin_source(self) -> MarginSource:
        return self._margin_source

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
        """Return the gate-capped contract count for `contract` given the strategy's full Kelly
        fraction (f* = mu / sigma^2), current account equity, and current price.

        Tighten-only: min(quarter-Kelly, margin-budget, leverage-cap[, survival]). A non-positive
        `full_kelly_fraction` (no edge) yields zero contracts.

        `returns` (optional historical per-period returns) ACTIVATES the leverage-survival
        bootstrap (MATH section 6): the size is further capped so P(survive the kill switch) >=
        survival_floor. When None (default) the survival cap is not evaluated (backward-compatible).
        `seed` makes the survival bootstrap bit-reproducible.
        """
        if account_equity <= 0:
            raise ValueError(f"account_equity must be > 0; got {account_equity}")
        if price <= 0:
            raise ValueError(f"price must be > 0; got {price}")
        root = _contract_root(contract)
        try:
            margin = self._margin_source.initial_margin(root)
        except MarginSourceError as exc:
            # Preserve the historical error contract (RiskManagerError + "no initial margin
            # configured") regardless of which MarginSource is wired.
            raise RiskManagerError(str(exc)) from exc
        multiplier = _lookup(CONTRACT_MULTIPLIER, root, contract, "multiplier")
        notional = price * multiplier

        margin_contracts = _floor_to_int(account_equity * self._params.margin_budget_pct / margin)
        leverage_contracts = _floor_to_int(account_equity * self._params.leverage_cap / notional)

        if full_kelly_fraction <= 0:
            return SizingDecision(
                contracts=0,
                binding_constraint="no_edge",
                kelly_contracts=0,
                margin_contracts=margin_contracts,
                leverage_contracts=leverage_contracts,
                notional_per_contract=notional,
                margin_per_contract=margin,
                survival_contracts=None,
            )

        kelly_capital = full_kelly_fraction * self._params.kelly_multiplier * account_equity
        kelly_contracts = _floor_to_int(kelly_capital / notional)

        base = min(kelly_contracts, margin_contracts, leverage_contracts)
        survival_contracts = (
            self._survival_cap(returns, base, notional, account_equity, seed)
            if returns is not None and base > 0
            else None
        )
        contracts = base if survival_contracts is None else min(base, survival_contracts)
        return SizingDecision(
            contracts=contracts,
            binding_constraint=_binding(
                kelly_contracts, margin_contracts, leverage_contracts, survival_contracts
            ),
            kelly_contracts=kelly_contracts,
            margin_contracts=margin_contracts,
            leverage_contracts=leverage_contracts,
            notional_per_contract=notional,
            margin_per_contract=margin,
            survival_contracts=survival_contracts,
        )

    def _survival_cap(
        self,
        returns: Sequence[float],
        base: int,
        notional: Decimal,
        account_equity: Decimal,
        seed: int | None,
    ) -> int:
        """Largest contract count <= `base` whose implied leverage passes the survival bootstrap.

        Implied leverage of n contracts = n * notional / account_equity. P_survive is monotonic
        non-increasing in n, so the surviving region is [0, cap]; binary-search the boundary.
        Too little history (< 2 returns) -> no cap (the bootstrap needs serial structure).
        """
        if len(returns) < _MIN_SURVIVAL_RETURNS:
            return base
        p = self._params

        def survives(n: int) -> bool:
            if n <= 0:
                return True  # no position always survives
            leverage = float(n * notional / account_equity)
            return survival_probability(
                returns,
                leverage,
                horizon=p.survival_horizon,
                n_paths=p.survival_paths,
                kill_switch_dd=float(p.kill_switch_dd),
                seed=seed,
            ) >= float(p.survival_floor)

        if survives(base):
            return base
        lo, hi = 0, base  # invariant: survives(lo) and not survives(hi)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if survives(mid):
                lo = mid
            else:
                hi = mid
        return lo


def _binding(kelly: int, margin: int, leverage: int, survival: int | None) -> BindingConstraint:
    caps: list[tuple[BindingConstraint, int]] = [
        ("kelly", kelly),
        ("margin", margin),
        ("leverage", leverage),
    ]
    if survival is not None:
        caps.append(("survival", survival))
    smallest = min(v for _, v in caps)
    for name, v in caps:  # ties resolve to the earlier cap (survival only when strictly tighter)
        if v == smallest:
            return name
    return "kelly"  # unreachable (caps is non-empty)


def multiplier_for(contract: ContractSymbol) -> Decimal:
    """USD point multiplier for `contract`'s root (e.g. ES -> 50, MES -> 5).

    Public accessor over CONTRACT_MULTIPLIER + the root parse; used by the engine for
    mark-to-market position valuation. Raises RiskManagerError if the root is not configured.
    """
    root = _contract_root(contract)
    return _lookup(CONTRACT_MULTIPLIER, root, contract, "multiplier")
