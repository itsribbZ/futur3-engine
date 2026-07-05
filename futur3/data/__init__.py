"""futur3.data — DataSource ABC + MultiSourceVerifier + storage layer.

Hard seam: no code outside `futur3/data/sources/` imports vendor SDKs directly.
All vendor calls wrapped behind DataSource subclasses; verifier sits between sources and engine.
"""

from futur3.data.cot_types import (
    COT_CONTRACT_SPECS,
    COTContractSpec,
    COTReport,
    COTReportFlavor,
)
from futur3.data.macro_types import (
    RELEASE_TIME_ET,
    MacroEvent,
    MacroPublisher,
    MacroSeries,
    MacroValue,
)
from futur3.data.types import (
    BarResolution,
    ContractSymbol,
    RawBar,
    RawTick,
    Settle,
    SettleState,
    Side,
    SourceTier,
)

__all__ = [
    "COT_CONTRACT_SPECS",
    "RELEASE_TIME_ET",
    "BarResolution",
    "COTContractSpec",
    "COTReport",
    "COTReportFlavor",
    "ContractSymbol",
    "MacroEvent",
    "MacroPublisher",
    "MacroSeries",
    "MacroValue",
    "RawBar",
    "RawTick",
    "Settle",
    "SettleState",
    "Side",
    "SourceTier",
]
