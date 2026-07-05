"""futur3.runtime — BACKTEST-IS-LIVE injection seam.

This package owns `RuntimeContext` — the single object that carries
mode-aware configuration through the engine. Same `RuntimeContext` API in
BACKTEST + LIVE_PAPER + LIVE_FUNDED; only the `mode` field differs.

Per the backtest-is-live design:
- ONE engine, RuntimeContext injection.
- Mode-agnostic business logic; mode-aware only at the data + execution edges.
- Eliminates the scattered mode-patcher anti-pattern.
"""

from __future__ import annotations

from futur3.runtime.context import (
    RuntimeContext,
    RuntimeMode,
    SystemClock,
)

__all__: list[str] = [
    "RuntimeContext",
    "RuntimeMode",
    "SystemClock",
]
