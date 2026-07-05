"""CMEEODDataSource — scraper of CME public settlement pages.

PRIMARY ANCHOR (T2_EXCHANGE) for daily settlement values across the 10-contract
futur3 universe per the verifier design. Free-first $0/mo.
Authoritative for settlement-price cross-source verification (verifier
Phase A1 = N≥3 sources; settle is the fail-closed anchor field per internal notes).

Architecture:
- URL pattern: https://www.cmegroup.com/markets/{group}/{family}/{product}.settlements.html
- Schema: Month · Open · High · Low · Last · Change · Settle · Est. Volume · Prior Day OI
- Cloudflare WAF defense via `curl_cffi` browser-fingerprint TLS impersonation
- Schema-signature validation on every fetch (bug class 9 — silent vendor schema drift)
- Settle state detection (preliminary vs final) from page status indicator
- Persistent Parquet archive at `data/cme_eod_archive/contract={root}/year={YYYY}/data.parquet`
- Append-only with idempotent dedupe on (contract, as_of_date, settle_state, content_bytes_sha)
  — preserves preliminary→final transitions AND retroactive revisions for verifier diff detection

Critical invariants:
- `oi_prior` is PRIOR-DAY OI per CME convention (internal notes). Field name LOCKED to
  prevent bug class 4 (look-ahead) mis-binding to nonexistent same-day OI.
- All datetimes IANA-TZ-aware at the source boundary (Settle.__post_init__ enforces).
- Decimal coercion preserves CME's display precision (no float).
- Comma thousands separators stripped from volume/OI before int conversion.
- "+" prefix stripped from change before Decimal conversion.
- WAF detection BEFORE HTML parse (raw-byte marker scan) — short-circuits on block.

Test discipline:
- Default `pytest` run hits ZERO live endpoints; HTTPClient + ClockProtocol are injected
  protocols so fixture-only tests cover the entire parse + archive surface.
- Live-network smoke gated behind `@pytest.mark.integration` (A1.3.10 procedure doc).

References:
- the verifier design-1.6 (CME public settlements spec)
- the data-layer design (DataSource ABC contract)
- the data-layer design (schema-signature + provenance hash chain)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Iterable
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, ClassVar, Final, Protocol, runtime_checkable

import polars as pl
from bs4 import BeautifulSoup
from bs4.element import Tag

from futur3.data.source import (
    BarsNotSupported,
    CMEScrapeError,
    ContractNotConfigured,
    DataSource,
    MalformedSettlementPage,
    SchemaMismatch,
    WAFBlockedError,
)
from futur3.data.types import (
    BarResolution,
    ContractSymbol,
    RawBar,
    Settle,
    SettleState,
    SourceTier,
    content_sha256,
)

# Module-level magic-value constants (per AAA discipline; named usage > inline literals)
HTTP_OK: Final[int] = 200
MIN_CONTRACT_SYMBOL_LENGTH: Final[int] = 4  # <ROOT><MONTH_CODE><YY> minimum (e.g., "ESM26")
MONTH_DISPLAY_PARTS: Final[int] = 2  # "<MMM> <YY>" on CME page
YEAR_DIGIT_COUNT: Final[int] = 2  # 2-digit year-suffix (internal notes)

logger = logging.getLogger(__name__)


# ============================================================================
# Injectable protocols (testability via dependency injection)
# ============================================================================


@runtime_checkable
class HTTPClient(Protocol):
    """HTTP fetch interface — injected so tests substitute fixture loaders."""

    def fetch(self, url: str) -> tuple[bytes, str]:
        """Fetch URL → (response_bytes, content_sha256_hex). Raises CMEScrapeError on failure."""
        ...

    def healthcheck(self, url: str) -> bool:
        """Lightweight HEAD probe; returns True if reachable."""
        ...


@runtime_checkable
class ClockProtocol(Protocol):
    """Clock interface — injected for deterministic tests."""

    def now_utc(self) -> datetime:
        """Return current UTC datetime (TZ-aware)."""
        ...


# ============================================================================
# Default implementations (production)
# ============================================================================


class _DefaultCMEHTTPClient:
    """Default HTTP client using `curl_cffi` browser-fingerprint TLS.

    Defense against Cloudflare WAF per internal notes (plain
    `requests` returns HTTP 403 on cmegroup.com). `curl_cffi` impersonates a
    modern Chrome TLS fingerprint via the libcurl-impersonate library.

    Tests should inject a fixture-based HTTPClient instead — this default is
    only constructed when CMEEODDataSource() is called with no http_client arg.
    """

    DEFAULT_TIMEOUT: ClassVar[float] = 30.0
    DEFAULT_IMPERSONATE: ClassVar[str] = "chrome120"

    # `_session` typed as Any: curl_cffi exposes a closed Literal for `impersonate`
    # and a generic Session that mypy can't resolve without paying the import cost
    # at module top. Keeping the type loose here is acceptable for an internal default.
    _session: Any
    _timeout: float

    def __init__(
        self,
        impersonate: str = DEFAULT_IMPERSONATE,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        # Lazy import — only triggered when this default client is instantiated.
        # Fixture-based tests never hit this path (they pass http_client=).
        from curl_cffi import requests as curl_requests

        # curl_cffi types impersonate as a closed Literal; we accept dynamic str so
        # config (env-var, future browser-version refresh) can override without forcing
        # callers to construct a Literal. Runtime validation lives in curl_cffi.
        self._session = curl_requests.Session(impersonate=impersonate)  # type: ignore[arg-type]
        self._timeout = timeout

    def fetch(self, url: str) -> tuple[bytes, str]:
        try:
            resp = self._session.get(url, timeout=self._timeout)
        except Exception as e:
            raise CMEScrapeError(f"CME fetch failed for {url}: {type(e).__name__}: {e}") from e
        if resp.status_code != HTTP_OK:
            raise CMEScrapeError(f"CME fetch returned HTTP {resp.status_code} for {url}")
        body: bytes = resp.content
        return body, content_sha256(body)

    def healthcheck(self, url: str) -> bool:
        try:
            resp = self._session.head(url, timeout=5.0)
            return resp.status_code in (200, 301, 302)
        except Exception as e:
            logger.debug("CME healthcheck %s failed: %s", url, e)
            return False


class _SystemClock:
    """Default clock returning UTC datetime (TZ-aware)."""

    def now_utc(self) -> datetime:
        return datetime.now(UTC)


# ============================================================================
# Internal control flow
# ============================================================================


class _SkipRow(Exception):
    """Internal: row should be skipped (e.g., no-settle no-trade row).

    Not exposed outside `_parse_settlement_page` → no external dependency on this.
    """


# ============================================================================
# Main class
# ============================================================================


class CMEEODDataSource(DataSource):
    """CME public settlement pages scraper (10-contract universe).

    See module docstring for architecture overview, invariants, and references.
    """

    SOURCE_ID: ClassVar[str] = "cme_public_settlements"
    CME_URL_BASE: ClassVar[str] = "https://www.cmegroup.com"

    # 10-contract URL registry per internal notes
    URLS: ClassVar[dict[str, str]] = {
        # Confirmed URLs (T2 web-search verified)
        "ES": "/markets/equities/sp/e-mini-sandp500.settlements.html",
        "NQ": "/markets/equities/nasdaq/e-mini-nasdaq-100.settlements.html",
        "CL": "/markets/energy/crude-oil/light-sweet-crude.settlements.html",
        "GC": "/markets/metals/precious/gold.settlements.html",
        "MBT": "/markets/cryptocurrencies/bitcoin/micro-bitcoin.settlements.html",
        "MET": "/markets/cryptocurrencies/ether/micro-ether.settlements.html",
        # HYPOTHESIS URL patterns — verify on first live fetch per internal notes
        "MES": "/markets/equities/sp/micro-e-mini-sandp-500.settlements.html",
        "MNQ": "/markets/equities/nasdaq/micro-e-mini-nasdaq-100.settlements.html",
        "MCL": "/markets/energy/crude-oil/micro-crude-oil.settlements.html",
        "MGC": "/markets/metals/precious/micro-gold.settlements.html",
    }

    HYPOTHESIS_URL_ROOTS: ClassVar[frozenset[str]] = frozenset({"MES", "MNQ", "MCL", "MGC"})

    # Tick sizes per internal notes — reserved for verifier tolerance bands (not used for
    # quantize() here; CMEEODDataSource preserves source decimal precision verbatim).
    TICK_SIZES: ClassVar[dict[str, Decimal]] = {
        "ES": Decimal("0.25"),
        "MES": Decimal("0.25"),
        "NQ": Decimal("0.25"),
        "MNQ": Decimal("0.25"),
        "CL": Decimal("0.01"),
        "MCL": Decimal("0.01"),
        "GC": Decimal("0.10"),
        "MGC": Decimal("0.10"),
        "MBT": Decimal("5"),
        "MET": Decimal("0.50"),
    }

    # Canonical column headers per internal notes (LOCKED — bug class 9 anchor)
    EXPECTED_COLUMN_HEADERS: ClassVar[tuple[str, ...]] = (
        "Month",
        "Open",
        "High",
        "Low",
        "Last",
        "Change",
        "Settle",
        "Est. Volume",
        "Prior Day OI",
    )

    # CME month code mapping (standard futures month letter codes)
    MONTH_NAME_TO_CODE: ClassVar[dict[str, str]] = {
        "JAN": "F",
        "FEB": "G",
        "MAR": "H",
        "APR": "J",
        "MAY": "K",
        "JUN": "M",
        "JUL": "N",
        "AUG": "Q",
        "SEP": "U",
        "OCT": "V",
        "NOV": "X",
        "DEC": "Z",
    }
    VALID_MONTH_CODES: ClassVar[frozenset[str]] = frozenset(MONTH_NAME_TO_CODE.values())

    # Cloudflare WAF detection markers (raw-byte scan pre-parse)
    WAF_MARKERS: ClassVar[tuple[bytes, ...]] = (
        b"cf-please-wait",
        b"Checking if the site connection is secure",
        b"Just a moment...",
        b"_cf_chl_opt",
        b"cf_chl_",
    )

    # Placeholder values for missing-data cells (internal notes known issue 5)
    PLACEHOLDER_VALUES: ClassVar[frozenset[str]] = frozenset({"---", "--", "", "-", "n/a", "N/A"})

    # Default archive root (relative to CWD; .gitignore excludes data/)
    DEFAULT_ARCHIVE_ROOT: ClassVar[Path] = Path("data/cme_eod_archive")

    def __init__(
        self,
        archive_root: Path | None = None,
        http_client: HTTPClient | None = None,
        clock: ClockProtocol | None = None,
    ) -> None:
        self._archive_root: Path = archive_root or self.DEFAULT_ARCHIVE_ROOT
        self._http: HTTPClient = http_client or _DefaultCMEHTTPClient()
        self._clock: ClockProtocol = clock or _SystemClock()
        self._expected_schema_signature: str = self._compute_schema_signature(
            self.EXPECTED_COLUMN_HEADERS
        )

    # ------------------------------------------------------------------------
    # DataSource ABC contract
    # ------------------------------------------------------------------------

    @property
    def source_id(self) -> str:
        return self.SOURCE_ID

    @property
    def tier(self) -> SourceTier:
        return SourceTier.T2_EXCHANGE

    def get_bars(
        self,
        contract: ContractSymbol,
        ts_start: datetime,
        ts_end: datetime,
        resolution: BarResolution,
    ) -> Iterable[RawBar]:
        """Not supported — CME public settlement pages publish 1 settle/day, not bars.

        Use IBKRHistoricalDataSource (A1.4) for intraday + daily bars.
        """
        raise BarsNotSupported(
            f"{self.source_id}: settlement pages publish daily settle only "
            f"(use IBKRHistoricalDataSource for bars)"
        )

    def latest_settle(
        self,
        contract: ContractSymbol,
        as_of: datetime,
    ) -> Settle | None:
        """Fetch + parse + archive ALL settles on the contract's page, return the
        Settle matching `contract` (or None if its month is not on the page).

        Per DataSource ABC: `as_of` may be naive OR TZ-aware; `as_of.date()` is used
        for the Settle's `as_of_date` (business-date the settle applies to). Caller
        is responsible for querying within the page's publication window.

        Raises:
            ContractNotConfigured: contract root not in URLS registry, or invalid format
            WAFBlockedError: Cloudflare challenge page returned instead of HTML
            MalformedSettlementPage: HTML missing <table>/<thead>/<tbody>, empty rows,
                or cell-count drift
            SchemaMismatch: column headers diverged from canonical (bug class 9 alarm)
            CMEScrapeError: network or HTTP-status failure
        """
        root, _, _ = self._parse_contract_symbol(contract)
        settles = self.fetch_all_for_root(root, as_of=as_of)
        for s in settles:
            if s.contract == contract:
                return s
        return None

    def healthcheck(self) -> bool:
        """Lightweight liveness check via HEAD on CME root URL."""
        try:
            return self._http.healthcheck(self.CME_URL_BASE)
        except Exception as e:
            logger.debug("CME healthcheck error: %s", e)
            return False

    # ------------------------------------------------------------------------
    # Public API beyond ABC
    # ------------------------------------------------------------------------

    def fetch_all_for_root(
        self,
        contract_root: str,
        as_of: datetime,
    ) -> list[Settle]:
        """Fetch + parse + archive ALL settles on `contract_root`'s settlement page.

        Returns the full list of parsed Settle records (one per contract-month on the
        page). Same data is written to the local Parquet archive (idempotent dedupe).

        Use this when querying multiple months for the same root — one HTTP fetch
        covers all months on the page. Saves rate-limit budget vs. N calls to
        `latest_settle`.
        """
        if contract_root not in self.URLS:
            raise ContractNotConfigured(
                f"{self.source_id}: contract root {contract_root!r} not in URL registry. "
                f"Configured roots: {sorted(self.URLS)}"
            )
        if contract_root in self.HYPOTHESIS_URL_ROOTS:
            logger.info(
                "Fetching HYPOTHESIS URL for %s — first live verify needed (internal notes)",
                contract_root,
            )

        url = self.CME_URL_BASE + self.URLS[contract_root]
        fetched_at = self._clock.now_utc()
        as_of_date = as_of.date()

        content_bytes, content_sha = self._http.fetch(url)
        if self._is_waf_block(content_bytes):
            raise WAFBlockedError(
                f"{self.source_id}: Cloudflare WAF challenge returned for {url}. "
                f"Verify curl_cffi browser-impersonation is enabled (plain "
                f"`requests` gets HTTP 403 on cmegroup.com)."
            )

        settles = self._parse_settlement_page(
            content_bytes,
            contract_root=contract_root,
            content_sha=content_sha,
            as_of_iso=fetched_at,
            as_of_date=as_of_date,
        )
        self._write_archive(contract_root, settles)
        return settles

    # ------------------------------------------------------------------------
    # Private: detection + provenance helpers
    # ------------------------------------------------------------------------

    def _is_waf_block(self, content_bytes: bytes) -> bool:
        """Detect Cloudflare challenge page in raw response bytes (pre-parse)."""
        return any(marker in content_bytes for marker in self.WAF_MARKERS)

    def _compute_schema_signature(self, headers: tuple[str, ...]) -> str:
        """SHA256 fingerprint of canonical header tuple.

        Stable across runs — used for bug class 9 (schema drift) detection.
        JSON serialization with `sort_keys=True` ensures byte-stable input
        regardless of Python's dict iteration order.
        """
        payload = json.dumps(list(headers), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    # ------------------------------------------------------------------------
    # Private: contract symbol parsing
    # ------------------------------------------------------------------------

    def _parse_contract_symbol(self, contract: ContractSymbol) -> tuple[str, str, int]:
        """ESM26 → ('ES', 'M', 26). MBTM26 → ('MBT', 'M', 26).

        Algorithm: last 2 chars = year_2dig; char at -3 = month code; rest = root.
        Validates each layer; raises ContractNotConfigured with detailed diagnostic
        on any inconsistency.
        """
        s = str(contract)
        if len(s) < MIN_CONTRACT_SYMBOL_LENGTH:
            raise ContractNotConfigured(
                f"{self.source_id}: contract {s!r} too short; expected <ROOT><MONTH_CODE><YY>"
            )
        year_str = s[-2:]
        if not year_str.isdigit():
            raise ContractNotConfigured(
                f"{self.source_id}: contract {s!r} year-suffix {year_str!r} not 2 digits"
            )
        year_2dig = int(year_str)
        month_code = s[-3]
        if month_code not in self.VALID_MONTH_CODES:
            raise ContractNotConfigured(
                f"{self.source_id}: contract {s!r} invalid month code {month_code!r}; "
                f"valid CME codes: {sorted(self.VALID_MONTH_CODES)}"
            )
        root = s[:-3]
        if root not in self.URLS:
            raise ContractNotConfigured(
                f"{self.source_id}: contract {s!r} root {root!r} not in URL registry. "
                f"Configured: {sorted(self.URLS)}"
            )
        return root, month_code, year_2dig

    def _parse_month_display(self, display: str) -> tuple[str, int]:
        """'JUN 26' → ('M', 26). 'MAR 27' → ('H', 27).

        Raises MalformedSettlementPage on un-parseable inputs.
        """
        parts = display.strip().upper().split()
        if len(parts) != MONTH_DISPLAY_PARTS:
            raise MalformedSettlementPage(
                f"{self.source_id}: cannot parse month display {display!r}; expected '<MMM> <YY>'"
            )
        month_name, year_str = parts
        if month_name not in self.MONTH_NAME_TO_CODE:
            raise MalformedSettlementPage(
                f"{self.source_id}: unknown month name {month_name!r} in {display!r}; "
                f"valid: {sorted(self.MONTH_NAME_TO_CODE)}"
            )
        if not year_str.isdigit() or len(year_str) != YEAR_DIGIT_COUNT:
            raise MalformedSettlementPage(
                f"{self.source_id}: cannot parse year {year_str!r} in {display!r}; "
                f"expected 2 digits"
            )
        return self.MONTH_NAME_TO_CODE[month_name], int(year_str)

    # ------------------------------------------------------------------------
    # Private: HTML parsing
    # ------------------------------------------------------------------------

    def _parse_settlement_page(
        self,
        content_bytes: bytes,
        contract_root: str,
        content_sha: str,
        as_of_iso: datetime,
        as_of_date: date,
    ) -> list[Settle]:
        """Parse HTML → list[Settle], validating structure + schema upfront."""
        soup = BeautifulSoup(content_bytes, "lxml")
        table = self._find_settlement_table(soup, contract_root)

        # Extract + validate headers (bug class 9 detection)
        thead = table.find("thead")
        if not isinstance(thead, Tag):
            raise MalformedSettlementPage(
                f"{self.source_id}: no <thead> in settlement table for {contract_root}"
            )
        header_row = thead.find("tr")
        if not isinstance(header_row, Tag):
            raise MalformedSettlementPage(
                f"{self.source_id}: no header <tr> in settlement table for {contract_root}"
            )
        headers = tuple(th.get_text(strip=True) for th in header_row.find_all("th"))
        actual_signature = self._compute_schema_signature(headers)
        if actual_signature != self._expected_schema_signature:
            raise SchemaMismatch(
                f"{self.source_id}: column headers {headers!r} != expected "
                f"{self.EXPECTED_COLUMN_HEADERS!r} for {contract_root}. "
                f"actual_signature={actual_signature[:16]}... "
                f"expected_signature={self._expected_schema_signature[:16]}..."
            )

        # Extract rows
        tbody = table.find("tbody")
        if not isinstance(tbody, Tag):
            raise MalformedSettlementPage(
                f"{self.source_id}: no <tbody> in settlement table for {contract_root}"
            )
        rows = tbody.find_all("tr")
        if not rows:
            raise MalformedSettlementPage(
                f"{self.source_id}: empty <tbody> — no settlement data for {contract_root}"
            )

        settle_state = self._detect_settle_state(soup)

        results: list[Settle] = []
        for row_idx, tr in enumerate(rows):
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(cells) != len(self.EXPECTED_COLUMN_HEADERS):
                raise MalformedSettlementPage(
                    f"{self.source_id}: row {row_idx} has {len(cells)} cells; expected "
                    f"{len(self.EXPECTED_COLUMN_HEADERS)} for {contract_root}"
                )
            try:
                settle = self._build_settle(
                    cells=cells,
                    contract_root=contract_root,
                    content_sha=content_sha,
                    as_of_iso=as_of_iso,
                    as_of_date=as_of_date,
                    settle_state=settle_state,
                )
            except _SkipRow:
                continue
            results.append(settle)
        return results

    def _find_settlement_table(self, soup: BeautifulSoup, contract_root: str) -> Tag:
        """Locate the settlement table via fallback selectors (id → class → last-resort)."""
        for tag in (
            soup.find("table", id="settlementsTable"),
            soup.find("table", class_="cmeTable"),
            soup.find("table", class_="main-table"),
            soup.find("table"),
        ):
            if isinstance(tag, Tag):
                return tag
        raise MalformedSettlementPage(
            f"{self.source_id}: no <table> element found for {contract_root}"
        )

    def _detect_settle_state(self, soup: BeautifulSoup) -> SettleState:
        """Find 'Preliminary' or 'Final' marker on the page.

        Primary: `<span class="status-value">` text. Fallback: full-page text scan.
        Conservative default: 'preliminary' (lower trust until proven final).
        """
        status_span = soup.find("span", class_="status-value")
        if isinstance(status_span, Tag):
            text = status_span.get_text(strip=True).lower()
            if "final" in text:
                return "final"
            if "preliminary" in text:
                return "preliminary"
        # Fallback: full-page text scan
        page_text = soup.get_text(separator=" ", strip=True)
        has_final = bool(re.search(r"\bFinal\b", page_text))
        has_preliminary = bool(re.search(r"\bPreliminary\b", page_text))
        if has_final and not has_preliminary:
            return "final"
        return "preliminary"

    def _build_settle(
        self,
        cells: list[str],
        contract_root: str,
        content_sha: str,
        as_of_iso: datetime,
        as_of_date: date,
        settle_state: SettleState,
    ) -> Settle:
        """Build a Settle from parsed cell strings.

        Raises `_SkipRow` for rows where the settle itself is missing (no-data row).

        For rows where settle is present but OHLC/volume/OI are placeholders
        (no-trade row per internal notes), fills missing OHLC with settle value
        and volume/OI with 0 — this preserves the regulatory-published settle while
        flagging zero trading activity to downstream consumers.
        """
        (
            month_disp,
            open_str,
            high_str,
            low_str,
            last_str,
            change_str,
            settle_str,
            volume_str,
            oi_prior_str,
        ) = cells

        month_code, year_2dig = self._parse_month_display(month_disp)
        contract = ContractSymbol(f"{contract_root}{month_code}{year_2dig:02d}")

        settle_value = self._parse_decimal(settle_str)
        if settle_value is None:
            # No settle on this row → entire row is meaningless; skip
            raise _SkipRow()

        open_value = self._parse_decimal(open_str)
        high_value = self._parse_decimal(high_str)
        low_value = self._parse_decimal(low_str)
        last_value = self._parse_decimal(last_str)
        change_value = self._parse_decimal(change_str)
        volume_value = self._parse_int(volume_str)
        oi_prior_value = self._parse_int(oi_prior_str)

        # No-trade row handling (internal notes): settle published, OHLC missing
        if open_value is None:
            open_value = settle_value
        if high_value is None:
            high_value = settle_value
        if low_value is None:
            low_value = settle_value
        if last_value is None:
            last_value = settle_value
        if change_value is None:
            change_value = Decimal("0")
        if volume_value is None:
            volume_value = 0
        if oi_prior_value is None:
            oi_prior_value = 0

        # Per-row content_bytes_sha for cross-contract
        # uniqueness. Previously every Settle from one page fetch received the
        # SAME page-level content_sha. ESM26 + ESU26 from one page → identical
        # (source_id, as_of_iso, content_bytes_sha) → source_provenance_hash
        # collision. Fix: hash (page_sha || contract) so each row is uniquely
        # fingerprinted while preserving page_sha as input (revision detection +
        # WAF-defense still work; archive dedupe still idempotent per-row).
        per_row_content_sha = content_sha256(f"{content_sha}||{contract}".encode())
        return Settle(
            contract=contract,
            as_of_date=as_of_date,
            settle=settle_value,
            settle_state=settle_state,
            open=open_value,
            high=high_value,
            low=low_value,
            last=last_value,
            change=change_value,
            volume_est=volume_value,
            oi_prior=oi_prior_value,
            source_id=self.SOURCE_ID,
            as_of_iso=as_of_iso,
            content_bytes_sha=per_row_content_sha,
            cme_month_code=month_code,
        )

    def _parse_decimal(self, raw: str) -> Decimal | None:
        """Decimal coercion with placeholder + comma + sign handling.

        Returns None for known placeholder values ('---', '', '-', 'n/a', etc.).
        Raises MalformedSettlementPage on un-coercible non-placeholder strings.
        """
        s = raw.strip()
        if s in self.PLACEHOLDER_VALUES:
            return None
        s = s.replace(",", "")
        if s.startswith("+"):
            s = s[1:]
        try:
            return Decimal(s)
        except (ValueError, ArithmeticError) as e:
            raise MalformedSettlementPage(
                f"{self.source_id}: cannot parse {raw!r} as Decimal: {e}"
            ) from e

    def _parse_int(self, raw: str) -> int | None:
        """Int coercion with placeholder + comma handling."""
        s = raw.strip()
        if s in self.PLACEHOLDER_VALUES:
            return None
        s = s.replace(",", "")
        try:
            return int(s)
        except ValueError as e:
            raise MalformedSettlementPage(
                f"{self.source_id}: cannot parse {raw!r} as int: {e}"
            ) from e

    # ------------------------------------------------------------------------
    # Private: Parquet archive
    # ------------------------------------------------------------------------

    def _write_archive(self, contract_root: str, settles: list[Settle]) -> None:
        """Append-only Parquet write with idempotent dedupe (internal notes + §2.6).

        Path: `data/cme_eod_archive/contract={root}/year={YYYY}/data.parquet`
        Dedupe key: `(contract, as_of_date, settle_state, content_bytes_sha)`

        Semantics:
        - Identical key (same fetch repeated) → no-op (idempotent)
        - Same (contract, date, state) BUT different content_bytes_sha → BOTH kept
          (preserves retroactive revision history for verifier `RetroactiveRevision`
          detection per internal notes)
        - Different settle_state (preliminary vs final) for same (contract, date) →
          BOTH kept (internal notes: preliminary→final transition is required for
          BACKTEST-IS-LIVE backtest-vs-replay equivalence)
        """
        if not settles:
            return

        by_year: dict[int, list[Settle]] = {}
        for s in settles:
            by_year.setdefault(s.as_of_date.year, []).append(s)

        for year, year_settles in by_year.items():
            archive_dir = self._archive_root / f"contract={contract_root}" / f"year={year}"
            archive_dir.mkdir(parents=True, exist_ok=True)
            archive_path = archive_dir / "data.parquet"

            new_df = self._settles_to_dataframe(year_settles)

            if archive_path.exists():
                existing_df = pl.read_parquet(archive_path)
                combined = pl.concat([existing_df, new_df], how="vertical_relaxed")
                deduped = combined.unique(
                    subset=["contract", "as_of_date", "settle_state", "content_bytes_sha"],
                    keep="last",
                    maintain_order=True,
                )
                deduped.write_parquet(archive_path)
            else:
                new_df.write_parquet(archive_path)

    def _settles_to_dataframe(self, settles: list[Settle]) -> pl.DataFrame:
        """Convert list[Settle] → polars DataFrame for Parquet write.

        Decimal → string (lossless precision). Datetime → microsecond UTC. Dates → naive date.
        """
        rows: list[dict[str, object]] = []
        for s in settles:
            rows.append(
                {
                    "contract": str(s.contract),
                    "as_of_date": s.as_of_date,
                    "settle": str(s.settle),
                    "settle_state": s.settle_state,
                    "open": str(s.open),
                    "high": str(s.high),
                    "low": str(s.low),
                    "last": str(s.last),
                    "change": str(s.change),
                    "volume_est": s.volume_est,
                    "oi_prior": s.oi_prior,
                    "source_id": s.source_id,
                    "as_of_iso": s.as_of_iso,
                    "content_bytes_sha": s.content_bytes_sha,
                    "cme_month_code": s.cme_month_code,
                }
            )
        return pl.DataFrame(rows)
