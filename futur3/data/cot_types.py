"""futur3.data.cot_types - frozen dataclasses for the CFTC Commitment of Traders (COT) data axis.

This module separates the COT data axis
from bar/tick/settle (`futur3.data.types`) and from macro events (`futur3.data.macro_types`):
the CFTC publishes a weekly trader-positioning snapshot whose information content is in the
positioning data, not in price. The snapshot is normalized here to the two
trader blocs the strategy layer consumes - SPECULATOR vs COMMERCIAL/HEDGER.

Core invariants (mirror `futur3.data.types` + `macro_types`):
- int contract counts (a position is a whole number of contracts); long/short are >= 0, the
  derived net is signed.
- IANA-TZ-aware datetimes at the boundary (naive raises).
- frozen=True for immutability + hashability.
- SHA256 provenance (`content_bytes_sha`) on every record for bit-reproducibility.

PIT (point-in-time) is the load-bearing invariant of this axis (bug class 5 / look-ahead) and is the
#1 silent bug in retail COT backtests (internal design notes):
- A report's `report_date` is the TUESDAY snapshot; the CFTC does not PUBLISH it until the
  following FRIDAY 15:30 ET. `COTReport.value_known_at_iso` is that Friday publication moment.
- A backtest at `as_of_iso` MUST NOT consume a report whose `value_known_at_iso > as_of_iso`.
  The 3-day Tue->Fri blackout is mandatory; centralized enforcement lives in
  `futur3.data.cot_source.enforce_cot_pit_gate` (+ the `COTSource.reports_known_at` accessor).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import Final

from futur3.data.types import SHA256_HEX_LENGTH, _assert_tz_aware

# ----------------------------------------------------------------------------
# Enums
# ----------------------------------------------------------------------------


class COTReportFlavor(StrEnum):
    """CFTC COT report flavor. Per internal design notes

    futur3's 6-contract universe uses TFF (financial: ES/NQ/MBT/MET) + DISAGGREGATED (physical:
    CL/GC). LEGACY is the pre-2006/2010 historical baseline (Commercial vs Non-Commercial);
    SUPPLEMENTAL covers only 13 agricultural contracts and is out of scope.
    """

    LEGACY = "legacy"
    DISAGGREGATED = "disaggregated"
    TFF = "tff"
    SUPPLEMENTAL = "supplemental"


# ----------------------------------------------------------------------------
# Contract-code mapping (HYPOTHESIS - pin via live Socrata discover before live use)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class COTContractSpec:
    """Maps a futur3 contract root to its CFTC COT identity (flavor + market code).

    The `cftc_contract_market_code` is a 6-char per-market-per-exchange identifier. The codes in
    `COT_CONTRACT_SPECS` are HYPOTHESIS per internal design notes (web
    research): they must be pinned against the live Socrata endpoint - a
    `WHERE market_and_exchange_names LIKE '%E-MINI S&P 500%'` discovery - and recorded canonically
    before any LIVE use. Backtest correctness does not depend on the literal value (it is a join
    key); the discovery step exists so live trading never queries a wrong market.
    """

    root: str
    flavor: COTReportFlavor
    cftc_contract_market_code: str
    crypto_thin: bool = False  # MBT/MET: <50 reportable traders is common -> thin-signal gate

    def __post_init__(self) -> None:
        if not self.root:
            raise ValueError("COTContractSpec.root must be non-empty")
        if not self.cftc_contract_market_code:
            raise ValueError(
                f"COTContractSpec.cftc_contract_market_code must be non-empty (root {self.root})"
            )


# Contract-code mapping. CL (067651), ES (13874A), GC (088691), MBT (133741), MET (146021) are
# LIVE-VERIFIED against the Socrata catalog (CL/ES/MBT/MET 2026-05-23 discover; GC confirmed
# 2026-05-23 via DISAGGREGATED fetch - 435 reports 2018-01..2026-04). The
# crypto codes use the PARENT CME contract (BITCOIN / ETHER CASH SETTLED), the institutional
# smart-money positioning gauge; the Micro Bitcoin/Ether listings (133742 /
# 146022) are more retail. An earlier draft MBT code "1330E1" was bogus (absent from the live
# data) - corrected here. NQ (209742) remains HYPOTHESIS pending the same live confirm.
COT_CONTRACT_SPECS: Final[dict[str, COTContractSpec]] = {
    "ES": COTContractSpec("ES", COTReportFlavor.TFF, "13874A"),
    "NQ": COTContractSpec("NQ", COTReportFlavor.TFF, "209742"),
    "CL": COTContractSpec("CL", COTReportFlavor.DISAGGREGATED, "067651"),
    "GC": COTContractSpec("GC", COTReportFlavor.DISAGGREGATED, "088691"),
    "MBT": COTContractSpec("MBT", COTReportFlavor.TFF, "133741", crypto_thin=True),
    "MET": COTContractSpec("MET", COTReportFlavor.TFF, "146021", crypto_thin=True),
}


# ----------------------------------------------------------------------------
# Records
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class COTReport:
    """A point-in-time-correct weekly CFTC COT positioning snapshot for one contract.

    Positioning is NORMALIZED to the two trader blocs the strategy layer consumes - `spec_*`
    (the SPECULATOR proxy) and `comm_*` (the COMMERCIAL/HEDGER proxy). Which CFTC category each
    maps to depends on `flavor` (internal design notes):

      - DISAGGREGATED (CL/GC): spec = Managed Money,    comm = Producer/Merchant/Processor/User
      - TFF (ES/NQ/MBT/MET):   spec = Leveraged Funds,  comm = Dealer/Intermediary
      - LEGACY:                spec = Non-Commercial,   comm = Commercial

    The flavor-specific raw-column extraction (including the CFTC schema's sic-marked typos like
    `noncomm_postions_spread_all`) lives in the concrete source (`futur3.data.sources.cftc`); this
    record is the normalized, provenance-stamped result.

    `value_known_at_iso` is the hard PIT GATE: the Friday 15:30 ET publication moment of the
    `report_date` (Tuesday) snapshot. A backtest must never consume this report at an as_of
    earlier than `value_known_at_iso` (the mandatory Tue->Fri 3-day blackout, bug class 5).
    """

    cftc_contract_market_code: str  # 6-char CFTC market code (e.g. "067651" for CL)
    flavor: COTReportFlavor
    report_date: date  # the TUESDAY snapshot (CFTC `report_date_as_yyyy_mm_dd`)
    value_known_at_iso: datetime  # PIT GATE - Friday 15:30 ET publication moment (tz-aware)
    open_interest_all: int  # total open interest as of the Tuesday snapshot
    spec_long: int  # speculator long  (Managed Money / Leveraged Funds / Non-Comm)
    spec_short: int  # speculator short
    comm_long: int  # commercial/hedger long  (Producer-Merchant / Dealer / Comm)
    comm_short: int  # commercial/hedger short
    source_id: str
    as_of_iso: datetime  # when fetched (tz-aware)
    content_bytes_sha: str
    total_traders: int | None = None  # reportable trader count (crypto thin-signal gate); None=NA

    def __post_init__(self) -> None:
        _assert_tz_aware(self.value_known_at_iso, "COTReport.value_known_at_iso")
        _assert_tz_aware(self.as_of_iso, "COTReport.as_of_iso")
        if not self.cftc_contract_market_code:
            raise ValueError("COTReport.cftc_contract_market_code must be non-empty")
        for name, count in (
            ("open_interest_all", self.open_interest_all),
            ("spec_long", self.spec_long),
            ("spec_short", self.spec_short),
            ("comm_long", self.comm_long),
            ("comm_short", self.comm_short),
        ):
            if count < 0:
                raise ValueError(f"COTReport.{name} must be >= 0 (contract count); got {count}")
        if self.total_traders is not None and self.total_traders < 0:
            raise ValueError(
                f"COTReport.total_traders must be >= 0 if set; got {self.total_traders}"
            )
        if len(self.content_bytes_sha) != SHA256_HEX_LENGTH:
            raise ValueError(
                f"COTReport.content_bytes_sha must be hex-SHA256 ({SHA256_HEX_LENGTH} chars); "
                f"got len={len(self.content_bytes_sha)}"
            )

    @property
    def spec_net(self) -> int:
        """Speculator net position (long - short). Signed."""
        return self.spec_long - self.spec_short

    @property
    def comm_net(self) -> int:
        """Commercial/hedger net position (long - short). Signed."""
        return self.comm_long - self.comm_short


__all__: list[str] = [
    "COT_CONTRACT_SPECS",
    "COTContractSpec",
    "COTReport",
    "COTReportFlavor",
]
