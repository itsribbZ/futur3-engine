"""futur3.tools — Operator utility scripts (Phase A1.15+).

- `verify_facts` — walks HYPOTHESIS markers in research + canonical docs and
  enumerates them so the operator can verify externally before promotion to RESOLVED.
"""

from __future__ import annotations

from futur3.tools.verify_facts import (
    HypothesisMarker,
    WalkReport,
    walk_hypothesis,
)

__all__: list[str] = [
    "HypothesisMarker",
    "WalkReport",
    "walk_hypothesis",
]
