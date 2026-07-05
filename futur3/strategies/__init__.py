"""Strategy plugin interface.

Concrete strategy/alpha implementations are intentionally NOT included in this
public engine distribution. The engine consumes any object implementing the
``Strategy`` / ``CrossSectionalStrategy`` interface defined in :mod:`base`.
"""

from __future__ import annotations

from futur3.strategies.base import (
    CrossSectionalStrategy,
    Signal,
    Strategy,
    StrategyError,
)

__all__ = ["CrossSectionalStrategy", "Signal", "Strategy", "StrategyError"]
