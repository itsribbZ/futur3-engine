"""RuntimeContext — BACKTEST-IS-LIVE injection seam.

Phase A1.12 per the data-layer design + the backtest-is-live design.

The RuntimeContext is the single mode-aware object passed through the engine.
Same dataclass API in BACKTEST + LIVE_PAPER + LIVE_FUNDED; only `mode` and
the wired DataSource / BrokerAdapter instances differ.

## Why a single context (not per-call mode args)

Lesson from a predecessor engine: scattered mode checks
("`if backtest: do_x else: do_y`") proliferate into an unmaintainable
patchwork. RuntimeContext centralizes mode + dependency injection so
business logic is mode-agnostic. Mode-aware code lives ONLY in:

- DataSource selection (ReplayDataSource in BACKTEST, IBKR/CME in LIVE).
- BrokerAdapter selection (MockBroker in BACKTEST, IBKRBrokerAdapter in
  LIVE_PAPER, TopstepXBrokerAdapter in LIVE_FUNDED).
- SlippageModel application (active in BACKTEST; absent in LIVE — broker fills
  already include slippage).

Strategies, verifier, risk_manager, decision layer — all consume RuntimeContext
without branching on mode (they call the abstractions; the context's
injected instances do the right thing).

## Scope (A1.12 shell)

- `RuntimeMode` enum (BACKTEST, LIVE_PAPER, LIVE_FUNDED).
- `RuntimeContext` frozen dataclass carrying mode + verifier_policy + clock
  + degraded_mode_allowed (engine-side flag).
- Mode-helper predicates (`is_backtest`, `is_live`, `is_paper`, `is_funded`).
- `from_env()` factory resolving `verifier_policy` via
  `verifier_policies.resolve_policy()`.
- `SystemClock` default ClockProtocol impl.

## Deferred to later steps

- **DataSource + BrokerAdapter wiring** — A1.13 BrokerAdapter ABC will land
  the broker injection slot; A1.10-A1.11 will land the multi-source wiring.
  For now, the RuntimeContext carries the policy + clock; engine code that
  needs sources passes them as separate args until A1.13 collapses them.
- **Calendar injection** — A1+ when calendar code lands.
- **degraded_mode auto-default by mode** — engine policy lives in A1.13+;
  RuntimeContext stays apparatus-pure (defaults to False; callers set
  explicitly per phase).

References:
- the backtest-is-live design (one code path, backtest and live)
- the data-layer design (3hr estimate)
- the verifier spec (B1 degraded-mode opt-in)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from futur3.data.verifier import ClockProtocol, VerifierPolicy
from futur3.data.verifier_policies import resolve_policy

# ---------------------------------------------------------------------------
# RuntimeMode
# ---------------------------------------------------------------------------


class RuntimeMode(StrEnum):
    """Engine execution mode.

    Per BACKTEST-IS-LIVE: the verifier + strategies + decision layer are
    mode-agnostic. Mode affects only DataSource + BrokerAdapter + SlippageModel
    wiring.

    Values are lower-case canonical strings (StrEnum); compatible with
    env-var passthrough + serialization without `.value` boilerplate.
    """

    BACKTEST = "backtest"
    LIVE_PAPER = "live_paper"
    LIVE_FUNDED = "live_funded"


# ---------------------------------------------------------------------------
# SystemClock — default ClockProtocol impl
# ---------------------------------------------------------------------------


class SystemClock:
    """Default ClockProtocol implementation — `datetime.now(UTC)` each call.

    Tests inject a frozen-time clock instead (see `tests/conftest.py:FakeClock`
    + `tests/test_ibkr_historical.py:FixtureClock`).
    """

    def now_utc(self) -> datetime:
        return datetime.now(UTC)


# ---------------------------------------------------------------------------
# RuntimeContext
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuntimeContext:
    """Mode-aware engine context carrying policy + clock + degraded-mode flag.

    Construction is mode-explicit (no defaults — caller MUST pick a mode).
    Frozen immutability + structural shape both contribute to BACKTEST-IS-LIVE:
    backtest + live runs receive STRUCTURALLY IDENTICAL contexts, differing only
    in the `mode` field + the injected sources/broker downstream.

    Args:
        mode: RuntimeMode selecting BACKTEST / LIVE_PAPER / LIVE_FUNDED.
        verifier_policy: Resolved VerifierPolicy preset (POLICY_PHASE_A1_DEFAULT /
            B1_SHADOW / C_LIVE). Typically passed from `resolve_policy(env)`.
        clock: ClockProtocol — UTC clock. Production = SystemClock; tests inject
            FakeClock for determinism.
        degraded_mode_allowed: Engine-side flag — if True, engine MAY proceed
            with N-1 verifier sources after a 24h-max manual opt-in (B1 paper
            mode). False in BACKTEST + LIVE_FUNDED by default (the verifier
            policy itself is unchanged across phases across modes; this flag gates
            the ENGINE's tolerance of IncompleteBar events).
    """

    mode: RuntimeMode
    verifier_policy: VerifierPolicy
    clock: ClockProtocol
    degraded_mode_allowed: bool = False

    # ------------------------------------------------------------------------
    # Mode helper predicates
    # ------------------------------------------------------------------------

    @property
    def is_backtest(self) -> bool:
        return self.mode == RuntimeMode.BACKTEST

    @property
    def is_live(self) -> bool:
        """True for LIVE_PAPER or LIVE_FUNDED — anything non-backtest."""
        return self.mode in (RuntimeMode.LIVE_PAPER, RuntimeMode.LIVE_FUNDED)

    @property
    def is_paper(self) -> bool:
        return self.mode == RuntimeMode.LIVE_PAPER

    @property
    def is_funded(self) -> bool:
        return self.mode == RuntimeMode.LIVE_FUNDED

    # ------------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------------

    @classmethod
    def from_env(
        cls,
        *,
        mode: RuntimeMode,
        clock: ClockProtocol | None = None,
        env: Mapping[str, str] | None = None,
        degraded_mode_allowed: bool = False,
    ) -> RuntimeContext:
        """Resolve verifier_policy from `FUTUR3_VERIFIER_POLICY` env-var.

        Per the fail-loud policy: unknown env-var value raises
        `UnknownPolicyError` (propagated from resolve_policy). No silent
        fallback to default.

        Args:
            mode: Mandatory RuntimeMode selection.
            clock: ClockProtocol; defaults to SystemClock if None.
            env: Environment mapping; defaults to `os.environ` via resolve_policy.
            degraded_mode_allowed: Engine-side opt-in flag (default False).

        Returns:
            RuntimeContext with verifier_policy resolved from env-var.

        Raises:
            UnknownPolicyError: `FUTUR3_VERIFIER_POLICY` set to unknown value.
        """
        policy = resolve_policy(env=env)
        return cls(
            mode=mode,
            verifier_policy=policy,
            clock=clock if clock is not None else SystemClock(),
            degraded_mode_allowed=degraded_mode_allowed,
        )
