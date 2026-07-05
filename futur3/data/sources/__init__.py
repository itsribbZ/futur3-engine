"""futur3.data.sources — concrete DataSource implementations.

**Hard seam** (per the data-layer design): only code in this package
may import vendor SDKs (`curl_cffi`, `ib_async`, `ccxt`, `bs4`, `lxml`, etc.).

Code outside `futur3.data.sources.*` must access external data through the
`DataSource` ABC (in `futur3.data.source`) — never via direct vendor SDK calls.

This seam:
- eliminates bug class 2 (monkey-patching of vendor SDKs in tests) by construction
- enables wholesale source swap (e.g., reroute IBKR → Tradovate without touching the engine)
- isolates vendor breakage to a single layer
- enables `ReplayDataSource` (A1.7) to substitute fixture-backed data for ANY source
"""

from __future__ import annotations

__all__: list[str] = [
    "BackoffQueue",
    "BackoffQueueConfig",
    "BinanceUSCryptoDataSource",
    "BitstampCryptoDataSource",
    "BlsMacroSource",
    "CMEEODDataSource",
    "CcxtCryptoDataSource",
    "CoinbaseCryptoDataSource",
    "FredMacroSource",
    "GeminiCryptoDataSource",
    "IBKRHistoricalDataSource",
    "KrakenCryptoDataSource",
    "ReplayDataSource",
]


from futur3.data.sources.backoff_queue import BackoffQueue, BackoffQueueConfig
from futur3.data.sources.bls_macro import BlsMacroSource
from futur3.data.sources.cme_eod import CMEEODDataSource
from futur3.data.sources.crypto import (
    BinanceUSCryptoDataSource,
    BitstampCryptoDataSource,
    CcxtCryptoDataSource,
    CoinbaseCryptoDataSource,
    GeminiCryptoDataSource,
    KrakenCryptoDataSource,
)
from futur3.data.sources.fred_macro import FredMacroSource
from futur3.data.sources.ibkr_historical import IBKRHistoricalDataSource
from futur3.data.sources.replay import ReplayDataSource
