"""futur3.execution.margin_source - injectable per-contract initial margin provider (A1.22a).

The seam that lets live CME SPAN margins replace the static Phase A1 estimates in `risk_manager`
WITH NO FORMULA CHANGE (internal design notes_h_prop_firms_span.md
section 2.8). `RiskManager` consumes a `MarginSource`, so swapping the static map for a future
`SpanFileMarginSource` needs zero change to the sizing math - exactly the "drops into the
margin slot" integration the storage/risk close notes describe.

Phase A1 ships `StaticMarginSource` over broker-documented margin estimates.
The SPAN file parser is intentionally NOT included here (and NOT stubbed - A1.19 rule): parsing
CME's proprietary `.spn` (XML) / `.pa2` (packed binary) / chunked daily risk-parameter files
correctly requires a real sample file to validate against (the full
risk-array model is a later phase). A parser tested only against self-invented fixtures would give
false confidence and risk a silent margin error - the exact silent-error catastrophe class (one wrong
margin -> wrong position size -> blown account). It lands as another `MarginSource` impl once a
real `.spn` sample is in hand (free from ftp://ftp.cmegroup.com/pub/span/data/cme).
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Protocol, runtime_checkable


class MarginSourceError(Exception):
    """Margin lookup failed (contract root not present in this source)."""


@runtime_checkable
class MarginSource(Protocol):
    """Per-contract-root initial margin provider. Mode-agnostic: the same interface serves the
    static Phase A1 estimates and a future live SPAN ingest, so business logic is source-agnostic.
    """

    def initial_margin(self, root: str) -> Decimal:
        """USD initial (overnight) margin for ONE contract of `root` (e.g. "ES", "MES").

        Raises:
            MarginSourceError: `root` is not known to this source.
        """
        ...

    def known_roots(self) -> frozenset[str]:
        """The set of contract roots this source can price."""
        ...


class StaticMarginSource:
    """Fixed per-root margin map (Phase A1 default = internal design notes
    estimates). A live `SpanFileMarginSource` (deferred) is a drop-in replacement that satisfies
    the same `MarginSource` protocol."""

    def __init__(self, margins: Mapping[str, Decimal]) -> None:
        self._margins: dict[str, Decimal] = dict(margins)

    def initial_margin(self, root: str) -> Decimal:
        try:
            return self._margins[root]
        except KeyError as exc:
            # Message intentionally contains "no initial margin configured" so RiskManager can
            # re-surface it as RiskManagerError with the unchanged error contract.
            raise MarginSourceError(
                f"no initial margin configured for contract root {root!r}; "
                f"known roots: {sorted(self._margins)}"
            ) from exc

    def known_roots(self) -> frozenset[str]:
        return frozenset(self._margins)
