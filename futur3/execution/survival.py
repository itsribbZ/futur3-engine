"""futur3.execution.survival - leverage-survival bootstrap (kill-switch survival gate).

The hard gate that prevents account blow-up under leverage (the failure mode real BTC data exposed:
a leveraged bet rode a -21% trend past zero equity). A synthesis of (
Vince 1992 Optimal-f + Kelly 1956 + Politis-Romano 1994 stationary bootstrap):

  1. Empirical return distribution = the per-bar (or per-trade) returns from history.
  2. Block-bootstrap B paths, each `horizon` periods, block length L = Politis-White auto-select.
  3. Equity path at the leverage:  E_t = E_(t-1) * (1 + r_t * leverage),  E_0 = 1.
  4. A path SURVIVES iff min_t E_t > 1 - kill_switch_dd  (never breaches the kill-switch drawdown;
     a single r_t * leverage <= -1 ruins the path - exactly the leveraged-ruin case).
  5. P_survive = (#paths_survived) / B.   Gate: promote a size only if P_survive >= 0.995.

FLOAT domain (a derived statistic, not a price/quantity - the Decimal-domain rule governs the
latter, matching `stats/`). Deterministic under a fixed `seed`. Reuses the Politis-White
block-length selector; the path generator mirrors `stationary_bootstrap_indices`' recurrence but
is parameterized for horizon != N (the stats helper fixes output length == data length, which the
survival horizon does not) - the locked stats function is reused where it fits and left untouched.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Final

from futur3.cv.block_bootstrap import optimal_block_length

SURVIVAL_FLOOR: Final[float] = 0.995  # < 0.5% of bootstrap paths may breach the kill switch
_DEFAULT_HORIZON: Final[int] = 252  # one trading year (MATH §6)
_DEFAULT_PATHS: Final[int] = 10000  # B (MATH §6)
_DEFAULT_KILL_DD: Final[float] = 0.30  # kill-switch max drawdown (configurable; Topstep tighter)
_MIN_RETURNS: Final[int] = 2  # need >= 2 returns for any serial structure
_RUIN_EQUITY: Final[float] = 0.0


class SurvivalError(Exception):
    """Survival-bootstrap input error (bad leverage / horizon / kill-switch / too few returns)."""


def _stationary_path(
    output_len: int, data_len: int, new_block_p: float, rng: random.Random
) -> list[int]:
    """A Politis-Romano stationary-bootstrap index path of length `output_len` into `[0, data_len)`.

    Same recurrence as `stats.stationary_bootstrap_indices` (new block w.p. `new_block_p`, else
    advance one step with wrap-around) but with output length decoupled from the data length, which
    the survival horizon requires.
    """
    idx = [rng.randrange(data_len)]
    for _ in range(1, output_len):
        if rng.random() < new_block_p:
            idx.append(rng.randrange(data_len))
        else:
            idx.append((idx[-1] + 1) % data_len)
    return idx


def survival_probability(
    returns: Sequence[float],
    leverage: float,
    *,
    horizon: int = _DEFAULT_HORIZON,
    n_paths: int = _DEFAULT_PATHS,
    kill_switch_dd: float = _DEFAULT_KILL_DD,
    block_length: int | None = None,
    seed: int | None = None,
) -> float:
    """P(account survives `horizon` periods at `leverage`) via block-bootstrap (survival gate).

    Args:
        returns: historical per-period simple returns (the empirical distribution to resample).
        leverage: the proposed position's leverage multiplier (>= 0; notional / equity).
        horizon: path length in periods (default 252 = 1y).
        n_paths: bootstrap paths B (default 10000).
        kill_switch_dd: kill-switch max drawdown in (0, 1); a path dies if equity falls to or below
            `1 - kill_switch_dd` (or below 0 = ruin).
        block_length: stationary-bootstrap mean block length; None -> Politis-White auto-select.
        seed: int for a bit-reproducible probability.

    Returns:
        P_survive in [0, 1] = fraction of paths whose equity never breached the kill switch.

    Raises:
        SurvivalError: leverage < 0, horizon < 1, n_paths < 1, kill_switch_dd not in (0, 1), or
            fewer than 2 returns.
    """
    if leverage < 0:
        raise SurvivalError(f"leverage must be >= 0; got {leverage}")
    if horizon < 1:
        raise SurvivalError(f"horizon must be >= 1; got {horizon}")
    if n_paths < 1:
        raise SurvivalError(f"n_paths must be >= 1; got {n_paths}")
    if not 0.0 < kill_switch_dd < 1.0:
        raise SurvivalError(f"kill_switch_dd must be in (0, 1); got {kill_switch_dd}")
    rs = [float(r) for r in returns]
    if len(rs) < _MIN_RETURNS:
        raise SurvivalError(f"need >= {_MIN_RETURNS} returns; got {len(rs)}")

    bl = block_length if block_length is not None else optimal_block_length(rs)
    bl = max(1, min(bl, len(rs)))
    new_block_p = 1.0 / bl
    kill_floor = 1.0 - kill_switch_dd
    rng = random.Random(seed)
    n = len(rs)

    survived = 0
    for _ in range(n_paths):
        equity = 1.0
        alive = True
        for i in _stationary_path(horizon, n, new_block_p, rng):
            equity *= 1.0 + rs[i] * leverage
            if equity <= kill_floor or equity <= _RUIN_EQUITY:
                alive = False
                break
        if alive:
            survived += 1
    return survived / n_paths


__all__: list[str] = [
    "SURVIVAL_FLOOR",
    "SurvivalError",
    "survival_probability",
]
