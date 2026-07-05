"""A1.3 KEYSTONE test suite — CMEEODDataSource fixture-based coverage.

Test discipline:
- ALL tests fixture-only (zero live network)
- Boundary-invariant tests (TZ-aware, oi_prior binding, Decimal-not-float)
- Schema-drift detection tests (bug class 9)
- Error-path coverage (WAF, malformed, schema drift, contract-not-configured)
- Idempotency tests on archive write
- Revision-preservation tests (preliminary→final, content_bytes_sha drift)

Target: 35+ new tests; combined with 25 existing smoke tests → 60+ total green.

References:
- futur3/data/sources/cme_eod.py (implementation)
- tests/fixtures/cme_eod/*.html (fixture HTML files)
- tests/conftest.py (FixtureHTTPClient + FakeClock)
- the verifier design (spec source of truth)
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest

from futur3.data.source import (
    BarsNotSupported,
    CMEScrapeError,
    ContractNotConfigured,
    DataSource,
    DataSourceError,
    MalformedSettlementPage,
    SchemaMismatch,
    TicksNotSupported,
    WAFBlockedError,
)
from futur3.data.sources.cme_eod import (
    ClockProtocol,
    CMEEODDataSource,
    HTTPClient,
    _SkipRow,
)
from futur3.data.types import (
    BarResolution,
    ContractSymbol,
    SourceTier,
)
from tests.conftest import FakeClock, FixtureHTTPClient

# ============================================================================
# TestA1_3_Imports — module imports, class constants, ABC compliance
# ============================================================================


class TestA1_3_Imports:
    """Module imports + class-level constants are correct per internal notes."""

    def test_source_imports_clean(self) -> None:
        assert CMEEODDataSource is not None
        assert issubclass(CMEEODDataSource, DataSource)

    def test_protocols_are_runtime_checkable(self) -> None:
        # Verify the protocol interfaces are properly defined
        assert hasattr(HTTPClient, "fetch")
        assert hasattr(HTTPClient, "healthcheck")
        assert hasattr(ClockProtocol, "now_utc")

    def test_10_urls_configured(self) -> None:
        """internal notes: 6 confirmed + 4 HYPOTHESIS = 10 contract URLs."""
        assert len(CMEEODDataSource.URLS) == 10
        confirmed = {"ES", "NQ", "CL", "GC", "MBT", "MET"}
        hypothesis = {"MES", "MNQ", "MCL", "MGC"}
        assert set(CMEEODDataSource.URLS) == confirmed | hypothesis
        assert frozenset(hypothesis) == CMEEODDataSource.HYPOTHESIS_URL_ROOTS

    def test_tick_sizes_present_for_all_10_contracts(self) -> None:
        """internal notes: tick size required for every URL'd contract."""
        assert set(CMEEODDataSource.TICK_SIZES) == set(CMEEODDataSource.URLS)
        # Spot-check known values
        assert CMEEODDataSource.TICK_SIZES["ES"] == Decimal("0.25")
        assert CMEEODDataSource.TICK_SIZES["CL"] == Decimal("0.01")
        assert CMEEODDataSource.TICK_SIZES["GC"] == Decimal("0.10")
        assert CMEEODDataSource.TICK_SIZES["MBT"] == Decimal("5")
        assert CMEEODDataSource.TICK_SIZES["MET"] == Decimal("0.50")

    def test_canonical_column_headers(self) -> None:
        """internal notes LOCKED schema — bug class 9 anchor."""
        assert CMEEODDataSource.EXPECTED_COLUMN_HEADERS == (
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
        # Must have 9 columns
        assert len(CMEEODDataSource.EXPECTED_COLUMN_HEADERS) == 9
        # "Prior Day OI" is the load-bearing field name (bug class 4 prevention)
        assert "Prior Day OI" in CMEEODDataSource.EXPECTED_COLUMN_HEADERS
        # NEVER "Today's OI" or similar
        assert not any("Today" in h for h in CMEEODDataSource.EXPECTED_COLUMN_HEADERS)

    def test_month_name_to_code_mapping_complete(self) -> None:
        """All 12 calendar months mapped to standard CME letter codes."""
        mapping = CMEEODDataSource.MONTH_NAME_TO_CODE
        assert len(mapping) == 12
        # Spot-check known values
        assert mapping["JAN"] == "F"
        assert mapping["MAR"] == "H"
        assert mapping["JUN"] == "M"
        assert mapping["SEP"] == "U"
        assert mapping["DEC"] == "Z"
        # All codes are uppercase single-letter
        assert all(len(code) == 1 and code.isupper() for code in mapping.values())


# ============================================================================
# TestA1_3_ContractParsing — ContractSymbol and month-display parsing
# ============================================================================


class TestA1_3_ContractParsing:
    """ContractSymbol + month-display parsing edge cases."""

    @pytest.fixture
    def source(self, cme_archive_tmp: Path) -> CMEEODDataSource:
        # Doesn't need HTTP client for these tests
        return CMEEODDataSource(
            archive_root=cme_archive_tmp,
            http_client=FixtureHTTPClient({}, Path(".")),  # never called
            clock=FakeClock(datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC)),
        )

    @pytest.mark.parametrize(
        "contract_str,expected_root,expected_month,expected_year",
        [
            ("ESM26", "ES", "M", 26),
            ("ESU26", "ES", "U", 26),
            ("ESZ26", "ES", "Z", 26),
            ("ESH27", "ES", "H", 27),
            ("NQM26", "NQ", "M", 26),
            ("CLN26", "CL", "N", 26),
            ("CLV26", "CL", "V", 26),
            ("GCQ26", "GC", "Q", 26),
            ("MBTM26", "MBT", "M", 26),
            ("METM26", "MET", "M", 26),
            ("MESM26", "MES", "M", 26),
            ("MNQM26", "MNQ", "M", 26),
            ("MCLN26", "MCL", "N", 26),
            ("MGCQ26", "MGC", "Q", 26),
        ],
    )
    def test_parse_valid_contract_symbol(
        self,
        source: CMEEODDataSource,
        contract_str: str,
        expected_root: str,
        expected_month: str,
        expected_year: int,
    ) -> None:
        root, month, year = source._parse_contract_symbol(ContractSymbol(contract_str))
        assert root == expected_root
        assert month == expected_month
        assert year == expected_year

    def test_parse_contract_too_short(self, source: CMEEODDataSource) -> None:
        with pytest.raises(ContractNotConfigured, match="too short"):
            source._parse_contract_symbol(ContractSymbol("ESM"))

    def test_parse_contract_non_digit_year(self, source: CMEEODDataSource) -> None:
        with pytest.raises(ContractNotConfigured, match="not 2 digits"):
            source._parse_contract_symbol(ContractSymbol("ESMXX"))

    def test_parse_contract_invalid_month_code(self, source: CMEEODDataSource) -> None:
        # 'T' is NOT a valid CME month code
        with pytest.raises(ContractNotConfigured, match="invalid month code"):
            source._parse_contract_symbol(ContractSymbol("EST26"))

    def test_parse_contract_unknown_root(self, source: CMEEODDataSource) -> None:
        # 'ZB' (US Treasury bonds) not in our 10-contract universe by design
        with pytest.raises(ContractNotConfigured, match="not in URL registry"):
            source._parse_contract_symbol(ContractSymbol("ZBM26"))

    @pytest.mark.parametrize(
        "display,expected_code,expected_year",
        [
            ("JUN 26", "M", 26),
            ("SEP 26", "U", 26),
            ("DEC 26", "Z", 26),
            ("MAR 27", "H", 27),
            ("JUL 26", "N", 26),
            ("AUG 26", "Q", 26),
            ("FEB 27", "G", 27),
            ("APR 27", "J", 27),
            ("jun 26", "M", 26),  # lowercase tolerated
            ("  DEC 26  ", "Z", 26),  # whitespace stripped
        ],
    )
    def test_parse_valid_month_display(
        self,
        source: CMEEODDataSource,
        display: str,
        expected_code: str,
        expected_year: int,
    ) -> None:
        code, year = source._parse_month_display(display)
        assert code == expected_code
        assert year == expected_year

    def test_parse_invalid_month_name(self, source: CMEEODDataSource) -> None:
        with pytest.raises(MalformedSettlementPage, match="unknown month name"):
            source._parse_month_display("FOO 26")

    def test_parse_malformed_month_display(self, source: CMEEODDataSource) -> None:
        with pytest.raises(MalformedSettlementPage, match="cannot parse month display"):
            source._parse_month_display("JUN")  # missing year

    def test_parse_invalid_year_format(self, source: CMEEODDataSource) -> None:
        with pytest.raises(MalformedSettlementPage, match="cannot parse year"):
            source._parse_month_display("JUN 2026")  # 4 digits not 2


# ============================================================================
# TestA1_3_LowLevelParsers — decimal, int, WAF, settle-state
# ============================================================================


class TestA1_3_LowLevelParsers:
    """Decimal + int + WAF + settle-state low-level parsing routines."""

    @pytest.fixture
    def source(self, cme_archive_tmp: Path) -> CMEEODDataSource:
        return CMEEODDataSource(
            archive_root=cme_archive_tmp,
            http_client=FixtureHTTPClient({}, Path(".")),
            clock=FakeClock(datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC)),
        )

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("5247.25", Decimal("5247.25")),
            ("0.25", Decimal("0.25")),
            ("78.45", Decimal("78.45")),
            ("2345.60", Decimal("2345.60")),
            ("67895.00", Decimal("67895.00")),
            ("+12.75", Decimal("12.75")),
            ("-12.75", Decimal("-12.75")),
            ("1,234.56", Decimal("1234.56")),
            ("  5247.25  ", Decimal("5247.25")),
        ],
    )
    def test_parse_decimal_valid(
        self, source: CMEEODDataSource, raw: str, expected: Decimal
    ) -> None:
        assert source._parse_decimal(raw) == expected

    @pytest.mark.parametrize("raw", ["---", "--", "", "-", "n/a", "N/A", "  ---  "])
    def test_parse_decimal_placeholders_return_none(
        self, source: CMEEODDataSource, raw: str
    ) -> None:
        assert source._parse_decimal(raw) is None

    def test_parse_decimal_invalid_raises_malformed(self, source: CMEEODDataSource) -> None:
        with pytest.raises(MalformedSettlementPage, match="cannot parse"):
            source._parse_decimal("not a number")

    def test_parse_decimal_preserves_precision(self, source: CMEEODDataSource) -> None:
        """Decimal coercion must NOT round through float."""
        assert source._parse_decimal("0.1") == Decimal("0.1")
        # If we accidentally went through float: float("0.1") = 0.1000000000...4
        # Decimal("0.1") = exactly 0.1
        result = source._parse_decimal("0.1")
        assert result is not None
        assert str(result) == "0.1"

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("1,234,567", 1_234_567),
            ("234567", 234_567),
            ("0", 0),
            ("1234", 1234),
        ],
    )
    def test_parse_int_valid(self, source: CMEEODDataSource, raw: str, expected: int) -> None:
        assert source._parse_int(raw) == expected

    @pytest.mark.parametrize("raw", ["---", "", "n/a"])
    def test_parse_int_placeholders_return_none(self, source: CMEEODDataSource, raw: str) -> None:
        assert source._parse_int(raw) is None

    def test_waf_detection_on_cloudflare_fixture(
        self, source: CMEEODDataSource, cme_fixtures_dir: Path
    ) -> None:
        content = (cme_fixtures_dir / "waf_block_cloudflare.html").read_bytes()
        assert source._is_waf_block(content) is True

    def test_waf_detection_negative_on_normal_html(
        self, source: CMEEODDataSource, cme_fixtures_dir: Path
    ) -> None:
        content = (cme_fixtures_dir / "es_jun26_preliminary.html").read_bytes()
        assert source._is_waf_block(content) is False

    def test_schema_signature_stable(self, source: CMEEODDataSource) -> None:
        """Same input must produce identical hash byte-for-byte (bit-repro invariant)."""
        sig_a = source._compute_schema_signature(CMEEODDataSource.EXPECTED_COLUMN_HEADERS)
        sig_b = source._compute_schema_signature(CMEEODDataSource.EXPECTED_COLUMN_HEADERS)
        assert sig_a == sig_b
        # SHA256 is 64 hex chars
        assert len(sig_a) == 64

    def test_schema_signature_changes_on_drift(self, source: CMEEODDataSource) -> None:
        canonical = source._compute_schema_signature(CMEEODDataSource.EXPECTED_COLUMN_HEADERS)
        drifted = source._compute_schema_signature(
            (
                "Month",
                "Open",
                "High",
                "Low",
                "Last",
                "Change",
                "Settlement Price",
                "Est. Volume",
                "Prior Day OI",
            ),
        )
        assert canonical != drifted


# ============================================================================
# TestA1_3_ABCCompliance — DataSource ABC contract
# ============================================================================


class TestA1_3_ABCCompliance:
    """DataSource ABC contract: source_id, tier, get_bars, ticks default."""

    @pytest.fixture
    def source(self, cme_archive_tmp: Path) -> CMEEODDataSource:
        return CMEEODDataSource(
            archive_root=cme_archive_tmp,
            http_client=FixtureHTTPClient({}, Path(".")),
            clock=FakeClock(datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC)),
        )

    def test_source_id_is_stable(self, source: CMEEODDataSource) -> None:
        assert source.source_id == "cme_public_settlements"
        assert source.source_id == source.SOURCE_ID

    def test_tier_is_t2_exchange(self, source: CMEEODDataSource) -> None:
        assert source.tier == SourceTier.T2_EXCHANGE

    def test_get_bars_raises_bars_not_supported(self, source: CMEEODDataSource) -> None:
        as_of = datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC)
        with pytest.raises(BarsNotSupported, match="settlement pages publish daily"):
            list(
                source.get_bars(
                    ContractSymbol("ESM26"),
                    as_of,
                    as_of,
                    BarResolution.DAY_1,
                )
            )

    def test_get_ticks_defaults_to_unsupported(self, source: CMEEODDataSource) -> None:
        as_of = datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC)
        with pytest.raises(TicksNotSupported):
            list(source.get_ticks(ContractSymbol("ESM26"), as_of, as_of))

    def test_repr_includes_source_id_and_tier(self, source: CMEEODDataSource) -> None:
        rep = repr(source)
        assert "cme_public_settlements" in rep
        assert "T2_EXCHANGE" in rep


# ============================================================================
# TestA1_3_FullPageParse — per-contract happy-path parsing
# ============================================================================


class TestA1_3_FullPageParse:
    """End-to-end fetch+parse for each contract fixture."""

    @pytest.fixture
    def s7_as_of(self) -> datetime:
        return datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC)

    def _make_source(
        self,
        cme_archive_tmp: Path,
        all_contracts_http_client: FixtureHTTPClient,
        fake_clock_s7: FakeClock,
    ) -> CMEEODDataSource:
        return CMEEODDataSource(
            archive_root=cme_archive_tmp,
            http_client=all_contracts_http_client,
            clock=fake_clock_s7,
        )

    def test_es_fetch_returns_4_settles(
        self,
        cme_archive_tmp: Path,
        all_contracts_http_client: FixtureHTTPClient,
        fake_clock_s7: FakeClock,
        s7_as_of: datetime,
    ) -> None:
        source = self._make_source(cme_archive_tmp, all_contracts_http_client, fake_clock_s7)
        settles = source.fetch_all_for_root("ES", as_of=s7_as_of)
        assert len(settles) == 4  # Jun, Sep, Dec 26 + Mar 27

    def test_es_jun26_settle_values(
        self,
        cme_archive_tmp: Path,
        all_contracts_http_client: FixtureHTTPClient,
        fake_clock_s7: FakeClock,
        s7_as_of: datetime,
    ) -> None:
        source = self._make_source(cme_archive_tmp, all_contracts_http_client, fake_clock_s7)
        settle = source.latest_settle(ContractSymbol("ESM26"), s7_as_of)
        assert settle is not None
        assert settle.contract == "ESM26"
        assert settle.settle == Decimal("5247.25")
        assert settle.open == Decimal("5234.25")
        assert settle.high == Decimal("5251.50")
        assert settle.low == Decimal("5228.75")
        assert settle.last == Decimal("5247.00")
        assert settle.change == Decimal("12.75")
        assert settle.volume_est == 1_234_567
        assert settle.oi_prior == 2_345_678
        assert settle.settle_state == "preliminary"
        assert settle.cme_month_code == "M"
        assert settle.source_id == "cme_public_settlements"
        # Boundary invariant: TZ-aware as_of_iso
        assert settle.as_of_iso is not None
        assert settle.as_of_iso.tzinfo is not None
        # Provenance hash present
        assert len(settle.content_bytes_sha) == 64

    def test_es_sep26_settle(
        self,
        cme_archive_tmp: Path,
        all_contracts_http_client: FixtureHTTPClient,
        fake_clock_s7: FakeClock,
        s7_as_of: datetime,
    ) -> None:
        source = self._make_source(cme_archive_tmp, all_contracts_http_client, fake_clock_s7)
        settle = source.latest_settle(ContractSymbol("ESU26"), s7_as_of)
        assert settle is not None
        assert settle.settle == Decimal("5274.50")
        assert settle.cme_month_code == "U"

    def test_nq_returns_correct_decimal_precision(
        self,
        cme_archive_tmp: Path,
        all_contracts_http_client: FixtureHTTPClient,
        fake_clock_s7: FakeClock,
        s7_as_of: datetime,
    ) -> None:
        source = self._make_source(cme_archive_tmp, all_contracts_http_client, fake_clock_s7)
        settle = source.latest_settle(ContractSymbol("NQM26"), s7_as_of)
        assert settle is not None
        assert settle.settle == Decimal("18498.75")
        assert settle.contract == "NQM26"

    def test_cl_jul26_settles(
        self,
        cme_archive_tmp: Path,
        all_contracts_http_client: FixtureHTTPClient,
        fake_clock_s7: FakeClock,
        s7_as_of: datetime,
    ) -> None:
        source = self._make_source(cme_archive_tmp, all_contracts_http_client, fake_clock_s7)
        settle = source.latest_settle(ContractSymbol("CLN26"), s7_as_of)
        assert settle is not None
        assert settle.settle == Decimal("78.89")
        assert settle.cme_month_code == "N"
        # CL has 6 contract rows
        settles = source.fetch_all_for_root("CL", as_of=s7_as_of)
        assert len(settles) == 6

    def test_gc_returns_metals_precision(
        self,
        cme_archive_tmp: Path,
        all_contracts_http_client: FixtureHTTPClient,
        fake_clock_s7: FakeClock,
        s7_as_of: datetime,
    ) -> None:
        source = self._make_source(cme_archive_tmp, all_contracts_http_client, fake_clock_s7)
        settle = source.latest_settle(ContractSymbol("GCQ26"), s7_as_of)
        assert settle is not None
        assert settle.settle == Decimal("2369.20")

    def test_mbt_crypto_contract(
        self,
        cme_archive_tmp: Path,
        all_contracts_http_client: FixtureHTTPClient,
        fake_clock_s7: FakeClock,
        s7_as_of: datetime,
    ) -> None:
        source = self._make_source(cme_archive_tmp, all_contracts_http_client, fake_clock_s7)
        settle = source.latest_settle(ContractSymbol("MBTM26"), s7_as_of)
        assert settle is not None
        assert settle.settle == Decimal("67895.00")
        assert settle.contract == "MBTM26"

    def test_met_crypto_contract(
        self,
        cme_archive_tmp: Path,
        all_contracts_http_client: FixtureHTTPClient,
        fake_clock_s7: FakeClock,
        s7_as_of: datetime,
    ) -> None:
        source = self._make_source(cme_archive_tmp, all_contracts_http_client, fake_clock_s7)
        settle = source.latest_settle(ContractSymbol("METM26"), s7_as_of)
        assert settle is not None
        assert settle.settle == Decimal("3266.00")

    def test_micro_contracts_route_correctly(
        self,
        cme_archive_tmp: Path,
        all_contracts_http_client: FixtureHTTPClient,
        fake_clock_s7: FakeClock,
        s7_as_of: datetime,
    ) -> None:
        """HYPOTHESIS-URL Micro contracts must route + parse without error."""
        source = self._make_source(cme_archive_tmp, all_contracts_http_client, fake_clock_s7)
        # All four Micro roots
        for micro_contract in ["MESM26", "MNQM26", "MCLN26", "MGCQ26"]:
            settle = source.latest_settle(ContractSymbol(micro_contract), s7_as_of)
            assert settle is not None, f"Micro {micro_contract} failed to route"

    def test_latest_settle_returns_none_for_unlisted_month(
        self,
        cme_archive_tmp: Path,
        all_contracts_http_client: FixtureHTTPClient,
        fake_clock_s7: FakeClock,
        s7_as_of: datetime,
    ) -> None:
        source = self._make_source(cme_archive_tmp, all_contracts_http_client, fake_clock_s7)
        # ESM30 is not in the fixture (which has Jun/Sep/Dec 26 + Mar 27)
        settle = source.latest_settle(ContractSymbol("ESM30"), s7_as_of)
        assert settle is None

    def test_as_of_date_passed_through_correctly(
        self,
        cme_archive_tmp: Path,
        all_contracts_http_client: FixtureHTTPClient,
        fake_clock_s7: FakeClock,
        s7_as_of: datetime,
    ) -> None:
        source = self._make_source(cme_archive_tmp, all_contracts_http_client, fake_clock_s7)
        settle = source.latest_settle(ContractSymbol("ESM26"), s7_as_of)
        assert settle is not None
        assert settle.as_of_date == date(2026, 5, 21)


# ============================================================================
# TestA1_3_ErrorPaths — schema drift, WAF, malformed pages
# ============================================================================


class TestA1_3_ErrorPaths:
    """All scraper error paths must raise the correct typed exception."""

    @pytest.fixture
    def s7_as_of(self) -> datetime:
        return datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC)

    def _source_with(
        self,
        fixture_filename: str,
        archive: Path,
        fixtures_dir: Path,
        clock: FakeClock,
    ) -> CMEEODDataSource:
        return CMEEODDataSource(
            archive_root=archive,
            http_client=FixtureHTTPClient(
                {"e-mini-sandp500.settlements.html": fixture_filename},
                fixtures_dir,
            ),
            clock=clock,
        )

    def test_schema_drift_raises_schema_mismatch(
        self,
        cme_archive_tmp: Path,
        cme_fixtures_dir: Path,
        fake_clock_s7: FakeClock,
        s7_as_of: datetime,
    ) -> None:
        source = self._source_with(
            "es_schema_drift.html", cme_archive_tmp, cme_fixtures_dir, fake_clock_s7
        )
        with pytest.raises(SchemaMismatch, match="column headers"):
            source.latest_settle(ContractSymbol("ESM26"), s7_as_of)

    def test_waf_block_raises_waf_blocked_error(
        self,
        cme_archive_tmp: Path,
        cme_fixtures_dir: Path,
        fake_clock_s7: FakeClock,
        s7_as_of: datetime,
    ) -> None:
        source = self._source_with(
            "waf_block_cloudflare.html", cme_archive_tmp, cme_fixtures_dir, fake_clock_s7
        )
        with pytest.raises(WAFBlockedError, match="Cloudflare WAF"):
            source.latest_settle(ContractSymbol("ESM26"), s7_as_of)

    def test_empty_table_raises_malformed(
        self,
        cme_archive_tmp: Path,
        cme_fixtures_dir: Path,
        fake_clock_s7: FakeClock,
        s7_as_of: datetime,
    ) -> None:
        source = self._source_with(
            "empty_table.html", cme_archive_tmp, cme_fixtures_dir, fake_clock_s7
        )
        with pytest.raises(MalformedSettlementPage, match="empty"):
            source.latest_settle(ContractSymbol("ESM26"), s7_as_of)

    def test_no_table_raises_malformed(
        self,
        cme_archive_tmp: Path,
        cme_fixtures_dir: Path,
        fake_clock_s7: FakeClock,
        s7_as_of: datetime,
    ) -> None:
        source = self._source_with(
            "malformed_no_tbody.html", cme_archive_tmp, cme_fixtures_dir, fake_clock_s7
        )
        with pytest.raises(MalformedSettlementPage, match="no <table>"):
            source.latest_settle(ContractSymbol("ESM26"), s7_as_of)

    def test_unconfigured_root_raises(
        self,
        cme_archive_tmp: Path,
        cme_fixtures_dir: Path,
        fake_clock_s7: FakeClock,
        s7_as_of: datetime,
    ) -> None:
        source = self._source_with(
            "es_jun26_preliminary.html", cme_archive_tmp, cme_fixtures_dir, fake_clock_s7
        )
        with pytest.raises(ContractNotConfigured, match=r"root.*not in URL registry"):
            source.fetch_all_for_root("ZB", as_of=s7_as_of)

    def test_hierarchy_cme_subclasses_data_source_error(self) -> None:
        """All CME-specific exceptions catchable via parent CMEScrapeError + DataSourceError."""
        assert issubclass(WAFBlockedError, CMEScrapeError)
        assert issubclass(MalformedSettlementPage, CMEScrapeError)
        assert issubclass(CMEScrapeError, DataSourceError)
        assert issubclass(SchemaMismatch, DataSourceError)
        assert issubclass(ContractNotConfigured, DataSourceError)


# ============================================================================
# TestA1_3_PlaceholderHandling — no-trade rows captured per internal notes
# ============================================================================


class TestA1_3_PlaceholderHandling:
    """No-trade rows: settle present, OHLC/volume/OI placeholders."""

    def test_placeholders_row_preserves_settle_with_ohlc_stub(
        self,
        cme_archive_tmp: Path,
        cme_fixtures_dir: Path,
        fake_clock_s7: FakeClock,
    ) -> None:
        as_of = datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC)
        source = CMEEODDataSource(
            archive_root=cme_archive_tmp,
            http_client=FixtureHTTPClient(
                {"e-mini-sandp500.settlements.html": "es_with_placeholders.html"},
                cme_fixtures_dir,
            ),
            clock=fake_clock_s7,
        )
        settles = source.fetch_all_for_root("ES", as_of=as_of)
        # 3 rows: JUN 26 (real trade) + JUN 28 + JUN 29 (no-trade with settle only)
        assert len(settles) == 3

        # JUN 28 row: settle = 5350.00, OHLC stubbed to settle, vol/OI = 0
        jun28 = next(s for s in settles if s.contract == "ESM28")
        assert jun28.settle == Decimal("5350.00")
        assert jun28.open == Decimal("5350.00")
        assert jun28.high == Decimal("5350.00")
        assert jun28.low == Decimal("5350.00")
        assert jun28.last == Decimal("5350.00")
        assert jun28.change == Decimal("0")
        assert jun28.volume_est == 0
        assert jun28.oi_prior == 0

    def test_all_placeholder_row_skipped(
        self,
        cme_archive_tmp: Path,
        cme_fixtures_dir: Path,
        fake_clock_s7: FakeClock,
    ) -> None:
        """A row where EVEN SETTLE is placeholder should be skipped entirely.

        Currently no fixture has this case (every row has settle); test the
        _SkipRow control flow directly.
        """
        source = CMEEODDataSource(
            archive_root=cme_archive_tmp,
            http_client=FixtureHTTPClient({}, Path(".")),
            clock=fake_clock_s7,
        )
        with pytest.raises(_SkipRow):
            source._build_settle(
                cells=["JUN 26", "---", "---", "---", "---", "---", "---", "---", "---"],
                contract_root="ES",
                content_sha="0" * 64,
                as_of_iso=fake_clock_s7.now_utc(),
                as_of_date=date(2026, 5, 21),
                settle_state="preliminary",
            )


# ============================================================================
# TestA1_3_PreliminaryFinal — settle_state transition
# ============================================================================


class TestA1_3_PreliminaryFinal:
    """Preliminary vs final settle_state must propagate from page indicator."""

    def test_preliminary_state_detected(
        self,
        cme_archive_tmp: Path,
        cme_fixtures_dir: Path,
        fake_clock_s7: FakeClock,
    ) -> None:
        as_of = datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC)
        source = CMEEODDataSource(
            archive_root=cme_archive_tmp,
            http_client=FixtureHTTPClient(
                {"e-mini-sandp500.settlements.html": "es_jun26_preliminary.html"},
                cme_fixtures_dir,
            ),
            clock=fake_clock_s7,
        )
        settle = source.latest_settle(ContractSymbol("ESM26"), as_of)
        assert settle is not None
        assert settle.settle_state == "preliminary"

    def test_final_state_detected(
        self,
        cme_archive_tmp: Path,
        cme_fixtures_dir: Path,
        fake_clock_s7: FakeClock,
    ) -> None:
        as_of = datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC)
        source = CMEEODDataSource(
            archive_root=cme_archive_tmp,
            http_client=FixtureHTTPClient(
                {"e-mini-sandp500.settlements.html": "es_jun26_final.html"},
                cme_fixtures_dir,
            ),
            clock=fake_clock_s7,
        )
        settle = source.latest_settle(ContractSymbol("ESM26"), as_of)
        assert settle is not None
        assert settle.settle_state == "final"

    def test_preliminary_vs_final_settle_values_differ(
        self,
        cme_archive_tmp: Path,
        cme_fixtures_dir: Path,
        fake_clock_s7: FakeClock,
    ) -> None:
        """Same contract, different fixtures: Dec26 settle is revised in final."""
        as_of = datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC)
        prelim_source = CMEEODDataSource(
            archive_root=cme_archive_tmp,
            http_client=FixtureHTTPClient(
                {"e-mini-sandp500.settlements.html": "es_jun26_preliminary.html"},
                cme_fixtures_dir,
            ),
            clock=fake_clock_s7,
        )
        final_source = CMEEODDataSource(
            archive_root=cme_archive_tmp,
            http_client=FixtureHTTPClient(
                {"e-mini-sandp500.settlements.html": "es_jun26_final.html"},
                cme_fixtures_dir,
            ),
            clock=fake_clock_s7,
        )
        prelim_dec = prelim_source.latest_settle(ContractSymbol("ESZ26"), as_of)
        final_dec = final_source.latest_settle(ContractSymbol("ESZ26"), as_of)
        assert prelim_dec is not None
        assert final_dec is not None
        # The Dec26 settle moved 5299.25 → 5299.50 between preliminary and final
        assert prelim_dec.settle == Decimal("5299.25")
        assert final_dec.settle == Decimal("5299.50")
        assert prelim_dec.settle_state == "preliminary"
        assert final_dec.settle_state == "final"


# ============================================================================
# TestA1_3_Archive — Parquet write, idempotency, revision preservation
# ============================================================================


class TestA1_3_Archive:
    """Persistent archive correctness: writes, idempotency, revisions."""

    def _source(
        self,
        archive: Path,
        fixture_filename: str,
        fixtures_dir: Path,
        clock: FakeClock,
    ) -> CMEEODDataSource:
        return CMEEODDataSource(
            archive_root=archive,
            http_client=FixtureHTTPClient(
                {"e-mini-sandp500.settlements.html": fixture_filename},
                fixtures_dir,
            ),
            clock=clock,
        )

    def test_archive_written_on_first_fetch(
        self,
        cme_archive_tmp: Path,
        cme_fixtures_dir: Path,
        fake_clock_s7: FakeClock,
    ) -> None:
        as_of = datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC)
        source = self._source(
            cme_archive_tmp, "es_jun26_preliminary.html", cme_fixtures_dir, fake_clock_s7
        )
        source.fetch_all_for_root("ES", as_of=as_of)

        archive_path = cme_archive_tmp / "contract=ES" / "year=2026" / "data.parquet"
        assert archive_path.exists()
        df = pl.read_parquet(archive_path)
        assert len(df) == 4  # 4 ES months

    def test_archive_idempotent_on_same_content(
        self,
        cme_archive_tmp: Path,
        cme_fixtures_dir: Path,
        fake_clock_s7: FakeClock,
    ) -> None:
        """Fetching identical content twice MUST NOT double-write rows."""
        as_of = datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC)
        source = self._source(
            cme_archive_tmp, "es_jun26_preliminary.html", cme_fixtures_dir, fake_clock_s7
        )
        source.fetch_all_for_root("ES", as_of=as_of)
        source.fetch_all_for_root("ES", as_of=as_of)

        archive_path = cme_archive_tmp / "contract=ES" / "year=2026" / "data.parquet"
        df = pl.read_parquet(archive_path)
        assert len(df) == 4  # Still 4, NOT 8

    def test_archive_preserves_preliminary_and_final(
        self,
        cme_archive_tmp: Path,
        cme_fixtures_dir: Path,
        fake_clock_s7: FakeClock,
    ) -> None:
        """preliminary AND final states for same date both archived (revision history)."""
        as_of = datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC)
        prelim = self._source(
            cme_archive_tmp, "es_jun26_preliminary.html", cme_fixtures_dir, fake_clock_s7
        )
        final = self._source(
            cme_archive_tmp, "es_jun26_final.html", cme_fixtures_dir, fake_clock_s7
        )
        prelim.fetch_all_for_root("ES", as_of=as_of)
        final.fetch_all_for_root("ES", as_of=as_of)

        archive_path = cme_archive_tmp / "contract=ES" / "year=2026" / "data.parquet"
        df = pl.read_parquet(archive_path)
        # 4 months × 2 states = 8 rows
        assert len(df) == 8
        states = set(df["settle_state"].to_list())
        assert states == {"preliminary", "final"}

    def test_archive_path_uses_year_from_as_of_date(
        self,
        cme_archive_tmp: Path,
        cme_fixtures_dir: Path,
        fake_clock_s7: FakeClock,
    ) -> None:
        as_of = datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC)
        source = self._source(
            cme_archive_tmp, "es_jun26_preliminary.html", cme_fixtures_dir, fake_clock_s7
        )
        source.fetch_all_for_root("ES", as_of=as_of)

        # Path scheme: contract={root}/year={YYYY}/data.parquet
        contract_dir = cme_archive_tmp / "contract=ES"
        year_dir = contract_dir / "year=2026"
        assert contract_dir.exists()
        assert year_dir.exists()
        assert (year_dir / "data.parquet").exists()

    def test_archive_decimal_precision_round_trip(
        self,
        cme_archive_tmp: Path,
        cme_fixtures_dir: Path,
        fake_clock_s7: FakeClock,
    ) -> None:
        """Decimal values must round-trip Parquet write → read without precision loss."""
        as_of = datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC)
        source = self._source(
            cme_archive_tmp, "es_jun26_preliminary.html", cme_fixtures_dir, fake_clock_s7
        )
        source.fetch_all_for_root("ES", as_of=as_of)

        archive_path = cme_archive_tmp / "contract=ES" / "year=2026" / "data.parquet"
        df = pl.read_parquet(archive_path)
        # Find ESM26 row
        esm26_row = df.filter(pl.col("contract") == "ESM26").row(0, named=True)
        # Decimal stored as string for losslessness
        assert esm26_row["settle"] == "5247.25"
        assert esm26_row["open"] == "5234.25"
        # Re-parsing as Decimal preserves
        assert Decimal(esm26_row["settle"]) == Decimal("5247.25")

    def test_archive_no_write_for_empty_input(
        self,
        cme_archive_tmp: Path,
        cme_fixtures_dir: Path,
        fake_clock_s7: FakeClock,
    ) -> None:
        """Direct call to _write_archive with empty list is no-op."""
        source = self._source(
            cme_archive_tmp, "es_jun26_preliminary.html", cme_fixtures_dir, fake_clock_s7
        )
        source._write_archive("ES", [])
        # No directories should have been created
        assert not (cme_archive_tmp / "contract=ES").exists()


# ============================================================================
# TestA1_3_Healthcheck — mocked liveness check
# ============================================================================


class TestA1_3_Healthcheck:
    def test_healthcheck_returns_true_via_fixture_client(
        self,
        cme_archive_tmp: Path,
        cme_fixtures_dir: Path,
        fake_clock_s7: FakeClock,
    ) -> None:
        source = CMEEODDataSource(
            archive_root=cme_archive_tmp,
            http_client=FixtureHTTPClient({}, cme_fixtures_dir),
            clock=fake_clock_s7,
        )
        # FixtureHTTPClient.healthcheck always returns True
        assert source.healthcheck() is True


# ============================================================================
# TestPerRowContentSha — corrective: per-row content_bytes_sha
# ============================================================================


class TestPerRowContentSha:
    """Corrective batch — fix from an internal audit.

    Prior bug: all Settles from one page fetch shared the same `content_bytes_sha`
    (the page-level SHA256). ESM26 + ESU26 + ESZ26 + ESH27 all had identical
    (source_id, as_of_iso, content_bytes_sha) tuples → identical
    source_provenance_hash. Cross-contract uniqueness invariant violated.

    Fix: per-row `content_bytes_sha = SHA256(page_sha || contract)`. Page-level
    hash is preserved as input (revision detection + WAF-defense still work).
    """

    @pytest.fixture
    def s7_as_of(self) -> datetime:
        return datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC)

    def _make_source(
        self,
        cme_archive_tmp: Path,
        all_contracts_http_client: FixtureHTTPClient,
        fake_clock_s7: FakeClock,
    ) -> CMEEODDataSource:
        return CMEEODDataSource(
            archive_root=cme_archive_tmp,
            http_client=all_contracts_http_client,
            clock=fake_clock_s7,
        )

    def test_all_es_contracts_from_one_page_have_distinct_content_sha(
        self,
        cme_archive_tmp: Path,
        all_contracts_http_client: FixtureHTTPClient,
        fake_clock_s7: FakeClock,
        s7_as_of: datetime,
    ) -> None:
        """Each contract from same page fetch gets a unique content_bytes_sha."""
        source = self._make_source(cme_archive_tmp, all_contracts_http_client, fake_clock_s7)
        settles = source.fetch_all_for_root("ES", as_of=s7_as_of)
        assert len(settles) == 4  # Jun, Sep, Dec 26 + Mar 27

        shas = [s.content_bytes_sha for s in settles]
        # All 4 must be distinct (M3 invariant)
        assert len(set(shas)) == 4, (
            f"Expected 4 distinct content_bytes_sha for 4 contracts; "
            f"got {len(set(shas))} unique values: {shas}"
        )
        # All must be hex-SHA256 (64 chars)
        for sha in shas:
            assert len(sha) == 64
            assert all(c in "0123456789abcdef" for c in sha)

    def test_per_row_hash_deterministic_across_two_fetches(
        self,
        cme_archive_tmp: Path,
        all_contracts_http_client: FixtureHTTPClient,
        fake_clock_s7: FakeClock,
        s7_as_of: datetime,
    ) -> None:
        """Same page bytes + same contract → same per-row content_bytes_sha."""
        src1 = self._make_source(cme_archive_tmp, all_contracts_http_client, fake_clock_s7)
        settle1 = src1.latest_settle(ContractSymbol("ESM26"), s7_as_of)

        # New source instance, same fixture → must produce identical hash
        src2 = self._make_source(cme_archive_tmp, all_contracts_http_client, fake_clock_s7)
        settle2 = src2.latest_settle(ContractSymbol("ESM26"), s7_as_of)

        assert settle1 is not None and settle2 is not None
        assert settle1.content_bytes_sha == settle2.content_bytes_sha

    def test_different_contracts_same_page_have_different_hashes(
        self,
        cme_archive_tmp: Path,
        all_contracts_http_client: FixtureHTTPClient,
        fake_clock_s7: FakeClock,
        s7_as_of: datetime,
    ) -> None:
        """ESM26 vs ESU26 from the same page fetch must differ."""
        source = self._make_source(cme_archive_tmp, all_contracts_http_client, fake_clock_s7)
        s_m = source.latest_settle(ContractSymbol("ESM26"), s7_as_of)
        s_u = source.latest_settle(ContractSymbol("ESU26"), s7_as_of)
        assert s_m is not None and s_u is not None
        # Pre-fix: these were IDENTICAL (page-level SHA). Post-fix: DIFFERENT.
        assert s_m.content_bytes_sha != s_u.content_bytes_sha
