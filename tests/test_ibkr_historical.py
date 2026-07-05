"""A1.4 IBKRHistoricalDataSource test suite — fixture-based coverage.

Test discipline:
- ALL tests fixture-only (zero IB Gateway connection)
- FixtureIBClient supports happy-path + error injection (connect-fail, req-fail)
- Boundary invariants: TZ-aware, Decimal-not-float, oi=None for IBKR bars
- Exchange routing across all 10 contracts verified
- BackoffQueue integration verified via real BackoffQueue instance
- ABC compliance (get_ticks raises, latest_settle raises, get_bars yields)

References:
- futur3/data/sources/ibkr_historical.py (implementation)
- futur3/data/sources/backoff_queue.py (rate-limit dep)
- internal design notes (spec source)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from futur3.data.source import (
    BarsNotSupported,
    ContractNotConfigured,
    DataSource,
    DataSourceError,
    IBKRConnectionError,
    IBKRError,
    IBKRReqError,
    TicksNotSupported,
)
from futur3.data.sources.backoff_queue import BackoffQueue
from futur3.data.sources.ibkr_historical import (
    DEFAULT_PORT_LIVE,
    DEFAULT_PORT_PAPER,
    ClockProtocol,
    IBClient,
    IBKRHistoricalDataSource,
)
from futur3.data.types import (
    BarResolution,
    ContractSymbol,
    RawBar,
    SourceTier,
)

# ============================================================================
# Test fixtures: FixtureIBClient + FakeClock
# ============================================================================


@dataclass
class FixtureIBContract:
    """Fake IBKR Future contract (replaces ib_async.Future in tests)."""

    symbol: str
    last_trade_date: str
    exchange: str


@dataclass
class FixtureIBBar:
    """Fake ib_async.BarData (only fields IBKRHistoricalDataSource reads)."""

    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class FixtureIBClient:
    """In-memory IBClient for tests — supports happy-path + error injection.

    Configure via constructor:
        bars_by_contract: dict[(symbol, last_trade_date, exchange) → list[FixtureIBBar]]
        connect_should_fail: True → connect() raises IBKRConnectionError
        req_should_fail: True → req_historical_data() raises generic Exception
    """

    bars_by_contract: dict[tuple[str, str, str], list[FixtureIBBar]] = field(default_factory=dict)
    connect_should_fail: bool = False
    req_should_fail: bool = False
    connect_call_count: int = 0
    req_call_count: int = 0
    last_req_args: dict[str, Any] = field(default_factory=dict)
    _connected: bool = False

    def connect(self, host: str, port: int, client_id: int) -> None:
        self.connect_call_count += 1
        if self.connect_should_fail:
            raise IBKRConnectionError(f"FixtureIBClient injected connect failure for {host}:{port}")
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def build_future_contract(
        self,
        symbol: str,
        last_trade_date: str,
        exchange: str,
    ) -> FixtureIBContract:
        return FixtureIBContract(
            symbol=symbol,
            last_trade_date=last_trade_date,
            exchange=exchange,
        )

    def req_historical_data(
        self,
        contract: Any,
        end_datetime: str,
        duration: str,
        bar_size: str,
        what_to_show: str,
        use_rth: int,
    ) -> list[FixtureIBBar]:
        self.req_call_count += 1
        self.last_req_args = {
            "contract": contract,
            "end_datetime": end_datetime,
            "duration": duration,
            "bar_size": bar_size,
            "what_to_show": what_to_show,
            "use_rth": use_rth,
        }
        if self.req_should_fail:
            raise RuntimeError("FixtureIBClient injected req failure")
        key = (contract.symbol, contract.last_trade_date, contract.exchange)
        return self.bars_by_contract.get(key, [])


@dataclass
class FixtureClock:
    """Deterministic UTC clock for IBKR tests."""

    fixed_time: datetime

    def now_utc(self) -> datetime:
        return self.fixed_time


# ============================================================================
# Shared fixtures
# ============================================================================


@pytest.fixture
def fixed_clock() -> FixtureClock:
    """Fixture reference time."""
    return FixtureClock(fixed_time=datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC))


@pytest.fixture
def empty_ib_client() -> FixtureIBClient:
    """Connects fine; returns empty bar list for any contract."""
    return FixtureIBClient()


@pytest.fixture
def es_bars_ib_client() -> FixtureIBClient:
    """Pre-populated with ESM26 daily bars for happy-path tests."""
    bars = [
        FixtureIBBar(
            date=datetime(2026, 5, 19, 22, 0, 0, tzinfo=UTC),
            open=5230.25,
            high=5248.75,
            low=5225.50,
            close=5240.00,
            volume=1_200_000,
        ),
        FixtureIBBar(
            date=datetime(2026, 5, 20, 22, 0, 0, tzinfo=UTC),
            open=5240.50,
            high=5258.25,
            low=5237.75,
            close=5252.00,
            volume=1_150_000,
        ),
        FixtureIBBar(
            date=datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
            open=5252.25,
            high=5263.50,
            low=5248.75,
            close=5260.00,
            volume=1_280_000,
        ),
    ]
    return FixtureIBClient(bars_by_contract={("ES", "202606", "CME"): bars})


def make_source(
    ib_client: FixtureIBClient,
    clock: FixtureClock,
    *,
    use_rth: bool = False,
) -> IBKRHistoricalDataSource:
    """Helper: build IBKRHistoricalDataSource wired to fixture deps."""
    return IBKRHistoricalDataSource(
        ib_client=ib_client,
        backoff_queue=BackoffQueue(),
        clock=clock,
        use_rth=use_rth,
    )


# ============================================================================
# TestIBKR_Imports — class constants + module imports
# ============================================================================


class TestIBKR_Imports:
    def test_imports_clean(self) -> None:
        assert IBKRHistoricalDataSource is not None
        assert issubclass(IBKRHistoricalDataSource, DataSource)

    def test_protocols_runtime_checkable(self) -> None:
        assert hasattr(IBClient, "connect")
        assert hasattr(IBClient, "req_historical_data")
        assert hasattr(ClockProtocol, "now_utc")

    def test_universe_contracts_have_exchange_mapping(self) -> None:
        """10-contract pull universe including ZB (CBOT)."""
        exchanges = IBKRHistoricalDataSource.EXCHANGE_BY_ROOT
        assert len(exchanges) == 11
        assert set(exchanges) == {
            "ES",
            "MES",
            "NQ",
            "MNQ",
            "CL",
            "MCL",
            "GC",
            "MGC",
            "MBT",
            "MET",
            "ZB",  # CBOT Treasury bond
        }

    def test_exchange_routing_correct(self) -> None:
        """Per internal notes: equity index + crypto on CME; energy on NYMEX; metals on COMEX;
        Treasuries on CBOT."""
        exchanges = IBKRHistoricalDataSource.EXCHANGE_BY_ROOT
        for root in ("ES", "MES", "NQ", "MNQ", "MBT", "MET"):
            assert exchanges[root] == "CME", f"{root} should be CME"
        for root in ("CL", "MCL"):
            assert exchanges[root] == "NYMEX", f"{root} should be NYMEX"
        for root in ("GC", "MGC"):
            assert exchanges[root] == "COMEX", f"{root} should be COMEX"
        assert exchanges["ZB"] == "CBOT", "ZB should be CBOT"

    def test_bar_size_map_covers_essential_resolutions(self) -> None:
        """BarResolution → IBKR barSizeSetting per internal notes"""
        m = IBKRHistoricalDataSource.BAR_SIZE_MAP
        assert m[BarResolution.SEC_1] == "1 secs"
        assert m[BarResolution.SEC_5] == "5 secs"
        assert m[BarResolution.MIN_1] == "1 min"
        assert m[BarResolution.MIN_5] == "5 mins"
        assert m[BarResolution.MIN_15] == "15 mins"
        assert m[BarResolution.HOUR_1] == "1 hour"
        assert m[BarResolution.DAY_1] == "1 day"
        # SETTLE intentionally absent — raises BarsNotSupported

    def test_month_code_complete(self) -> None:
        m = IBKRHistoricalDataSource.MONTH_CODE_TO_NUMBER
        assert len(m) == 12
        assert m["F"] == 1
        assert m["M"] == 6
        assert m["U"] == 9
        assert m["Z"] == 12

    def test_default_ports_match_ibkr_convention(self) -> None:
        """IB Gateway: 4001 live, 4002 paper per internal notes"""
        assert DEFAULT_PORT_LIVE == 4001
        assert DEFAULT_PORT_PAPER == 4002


# ============================================================================
# TestIBKR_ContractParsing — ContractSymbol → root/month/year + IB Contract build
# ============================================================================


class TestIBKR_ContractParsing:
    @pytest.fixture
    def source(
        self,
        empty_ib_client: FixtureIBClient,
        fixed_clock: FixtureClock,
    ) -> IBKRHistoricalDataSource:
        return make_source(empty_ib_client, fixed_clock)

    @pytest.mark.parametrize(
        "symbol,expected_root,expected_code,expected_year",
        [
            ("ESM26", "ES", "M", 26),
            ("NQU26", "NQ", "U", 26),
            ("CLN26", "CL", "N", 26),
            ("GCQ26", "GC", "Q", 26),
            ("MBTM26", "MBT", "M", 26),
            ("METM26", "MET", "M", 26),
            ("MESH27", "MES", "H", 27),
            ("MGCZ26", "MGC", "Z", 26),
        ],
    )
    def test_parse_valid_contract(
        self,
        source: IBKRHistoricalDataSource,
        symbol: str,
        expected_root: str,
        expected_code: str,
        expected_year: int,
    ) -> None:
        root, code, year = source._parse_contract_symbol(ContractSymbol(symbol))
        assert root == expected_root
        assert code == expected_code
        assert year == expected_year

    def test_unknown_root_raises(self, source: IBKRHistoricalDataSource) -> None:
        with pytest.raises(ContractNotConfigured, match=r"not in EXCHANGE_BY_ROOT"):
            source._parse_contract_symbol(ContractSymbol("ZNM26"))  # 10y T-note, not configured

    def test_invalid_month_code_raises(self, source: IBKRHistoricalDataSource) -> None:
        with pytest.raises(ContractNotConfigured, match="invalid month code"):
            source._parse_contract_symbol(ContractSymbol("EST26"))

    def test_non_digit_year_raises(self, source: IBKRHistoricalDataSource) -> None:
        with pytest.raises(ContractNotConfigured, match="not 2 digits"):
            source._parse_contract_symbol(ContractSymbol("ESMXX"))

    def test_too_short_raises(self, source: IBKRHistoricalDataSource) -> None:
        with pytest.raises(ContractNotConfigured, match="too short"):
            source._parse_contract_symbol(ContractSymbol("ES"))

    @pytest.mark.parametrize(
        "symbol,expected_last_trade,expected_exchange",
        [
            ("ESM26", "202606", "CME"),
            ("NQU26", "202609", "CME"),
            ("CLN26", "202607", "NYMEX"),
            ("GCQ26", "202608", "COMEX"),
            ("MBTH27", "202703", "CME"),
            ("METZ26", "202612", "CME"),
        ],
    )
    def test_build_ib_contract_routes_to_correct_exchange(
        self,
        source: IBKRHistoricalDataSource,
        symbol: str,
        expected_last_trade: str,
        expected_exchange: str,
    ) -> None:
        ib_contract = source._build_ib_contract(ContractSymbol(symbol))
        assert ib_contract.symbol == symbol[:-3]  # root
        assert ib_contract.last_trade_date == expected_last_trade
        assert ib_contract.exchange == expected_exchange

    def test_century_disambiguation_27_means_2027(self, source: IBKRHistoricalDataSource) -> None:
        """ContractSymbol convention: 00-49 → 2000+; 50-99 → 1900+."""
        ib_contract = source._build_ib_contract(ContractSymbol("ESH27"))
        assert ib_contract.last_trade_date == "202703"


# ============================================================================
# TestIBKR_BarSizeMapping — BarResolution → barSizeSetting
# ============================================================================


class TestIBKR_BarSizeMapping:
    @pytest.fixture
    def source(
        self,
        empty_ib_client: FixtureIBClient,
        fixed_clock: FixtureClock,
    ) -> IBKRHistoricalDataSource:
        return make_source(empty_ib_client, fixed_clock)

    def test_supported_resolutions_map_correctly(self, source: IBKRHistoricalDataSource) -> None:
        assert source._map_bar_size(BarResolution.MIN_1) == "1 min"
        assert source._map_bar_size(BarResolution.DAY_1) == "1 day"

    def test_settle_resolution_raises_bars_not_supported(
        self, source: IBKRHistoricalDataSource
    ) -> None:
        with pytest.raises(BarsNotSupported, match="SETTLE"):
            source._map_bar_size(BarResolution.SETTLE)


# ============================================================================
# TestIBKR_DurationStr — ts range → IBKR durationStr
# ============================================================================


class TestIBKR_DurationStr:
    @pytest.fixture
    def source(
        self,
        empty_ib_client: FixtureIBClient,
        fixed_clock: FixtureClock,
    ) -> IBKRHistoricalDataSource:
        return make_source(empty_ib_client, fixed_clock)

    def test_seconds_range_uses_s_unit(self, source: IBKRHistoricalDataSource) -> None:
        start = datetime(2026, 5, 21, 21, 0, 0, tzinfo=UTC)
        end = datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC)
        # 3600 seconds
        assert source._compute_duration_str(start, end) == "3600 S"

    def test_day_range_uses_d_unit(self, source: IBKRHistoricalDataSource) -> None:
        start = datetime(2026, 5, 19, 0, 0, 0, tzinfo=UTC)
        end = datetime(2026, 5, 21, 0, 0, 0, tzinfo=UTC)
        assert source._compute_duration_str(start, end) == "2 D"

    def test_fractional_day_rounds_up(self, source: IBKRHistoricalDataSource) -> None:
        # 1.5 days
        start = datetime(2026, 5, 19, 0, 0, 0, tzinfo=UTC)
        end = datetime(2026, 5, 20, 12, 0, 0, tzinfo=UTC)
        assert source._compute_duration_str(start, end) == "2 D"


# ============================================================================
# TestIBKR_EndDateTime — format ts_end for IBKR
# ============================================================================


class TestIBKR_EndDateTime:
    @pytest.fixture
    def source(
        self,
        empty_ib_client: FixtureIBClient,
        fixed_clock: FixtureClock,
    ) -> IBKRHistoricalDataSource:
        return make_source(empty_ib_client, fixed_clock)

    def test_utc_datetime_formats_correctly(self, source: IBKRHistoricalDataSource) -> None:
        ts = datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC)
        assert source._format_end_datetime(ts) == "20260521 22:00:00 UTC"

    def test_naive_datetime_raises(self, source: IBKRHistoricalDataSource) -> None:
        with pytest.raises(ValueError, match="TZ-aware"):
            source._format_end_datetime(datetime(2026, 5, 21, 22, 0, 0))

    def test_non_utc_tz_coerces_to_utc(self, source: IBKRHistoricalDataSource) -> None:
        # 18:00 CT = 23:00 UTC (CDT = UTC-5)
        ts = datetime(2026, 5, 21, 18, 0, 0, tzinfo=ZoneInfo("America/Chicago"))
        result = source._format_end_datetime(ts)
        assert "UTC" in result
        assert "23:00:00" in result


# ============================================================================
# TestIBKR_ABCCompliance — DataSource ABC contract
# ============================================================================


class TestIBKR_ABCCompliance:
    @pytest.fixture
    def source(
        self,
        empty_ib_client: FixtureIBClient,
        fixed_clock: FixtureClock,
    ) -> IBKRHistoricalDataSource:
        return make_source(empty_ib_client, fixed_clock)

    def test_source_id_stable(self, source: IBKRHistoricalDataSource) -> None:
        assert source.source_id == "ibkr_tws_historical"
        assert source.source_id == source.SOURCE_ID

    def test_tier_is_t2_broker(self, source: IBKRHistoricalDataSource) -> None:
        assert source.tier == SourceTier.T2_BROKER

    def test_get_ticks_raises_unsupported(self, source: IBKRHistoricalDataSource) -> None:
        as_of = datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC)
        with pytest.raises(TicksNotSupported, match=r"deferred to A1\.6"):
            list(source.get_ticks(ContractSymbol("ESM26"), as_of, as_of))

    def test_latest_settle_returns_none_when_no_bars(
        self, source: IBKRHistoricalDataSource
    ) -> None:
        """A1.5 — empty IB response (pre-listing / expired w/o includeExpired) → None."""
        as_of = datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC)
        assert source.latest_settle(ContractSymbol("ESM26"), as_of) is None

    def test_repr_includes_source_id_and_tier(self, source: IBKRHistoricalDataSource) -> None:
        rep = repr(source)
        assert "ibkr_tws_historical" in rep
        assert "T2_BROKER" in rep


# ============================================================================
# TestIBKR_ConnectionLifecycle — lazy connect, idempotent, error injection
# ============================================================================


class TestIBKR_ConnectionLifecycle:
    def test_lazy_connect_on_first_get_bars(
        self, es_bars_ib_client: FixtureIBClient, fixed_clock: FixtureClock
    ) -> None:
        source = make_source(es_bars_ib_client, fixed_clock)
        assert es_bars_ib_client.connect_call_count == 0
        # First get_bars should trigger connect
        list(
            source.get_bars(
                ContractSymbol("ESM26"),
                datetime(2026, 5, 18, 0, 0, 0, tzinfo=UTC),
                datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
                BarResolution.DAY_1,
            )
        )
        assert es_bars_ib_client.connect_call_count == 1

    def test_subsequent_calls_dont_reconnect(
        self, es_bars_ib_client: FixtureIBClient, fixed_clock: FixtureClock
    ) -> None:
        source = make_source(es_bars_ib_client, fixed_clock)
        for _ in range(3):
            list(
                source.get_bars(
                    ContractSymbol("ESM26"),
                    datetime(2026, 5, 18, 0, 0, 0, tzinfo=UTC),
                    datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
                    BarResolution.DAY_1,
                )
            )
        assert es_bars_ib_client.connect_call_count == 1  # connected ONCE
        assert es_bars_ib_client.req_call_count == 3

    def test_connect_failure_raises_ibkr_connection_error(self, fixed_clock: FixtureClock) -> None:
        client = FixtureIBClient(connect_should_fail=True)
        source = make_source(client, fixed_clock)
        with pytest.raises(IBKRConnectionError, match="injected connect failure"):
            list(
                source.get_bars(
                    ContractSymbol("ESM26"),
                    datetime(2026, 5, 18, 0, 0, 0, tzinfo=UTC),
                    datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
                    BarResolution.DAY_1,
                )
            )

    def test_disconnect_explicit(
        self, es_bars_ib_client: FixtureIBClient, fixed_clock: FixtureClock
    ) -> None:
        source = make_source(es_bars_ib_client, fixed_clock)
        list(
            source.get_bars(
                ContractSymbol("ESM26"),
                datetime(2026, 5, 18, 0, 0, 0, tzinfo=UTC),
                datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
                BarResolution.DAY_1,
            )
        )
        assert es_bars_ib_client.is_connected()
        source.disconnect()
        assert not es_bars_ib_client.is_connected()

    def test_healthcheck_reflects_connection_state(
        self, es_bars_ib_client: FixtureIBClient, fixed_clock: FixtureClock
    ) -> None:
        source = make_source(es_bars_ib_client, fixed_clock)
        assert source.healthcheck() is False  # not connected yet
        es_bars_ib_client.connect("127.0.0.1", 4002, 1)
        assert source.healthcheck() is True


# ============================================================================
# TestIBKR_GetBarsHappyPath — full pipeline fixture → RawBar
# ============================================================================


class TestIBKR_GetBarsHappyPath:
    @pytest.fixture
    def source(
        self,
        es_bars_ib_client: FixtureIBClient,
        fixed_clock: FixtureClock,
    ) -> IBKRHistoricalDataSource:
        return make_source(es_bars_ib_client, fixed_clock)

    def test_returns_3_bars_for_es(self, source: IBKRHistoricalDataSource) -> None:
        bars = list(
            source.get_bars(
                ContractSymbol("ESM26"),
                datetime(2026, 5, 18, 0, 0, 0, tzinfo=UTC),
                datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
                BarResolution.DAY_1,
            )
        )
        assert len(bars) == 3
        assert all(isinstance(b, RawBar) for b in bars)

    def test_decimal_precision_preserved(self, source: IBKRHistoricalDataSource) -> None:
        bars = list(
            source.get_bars(
                ContractSymbol("ESM26"),
                datetime(2026, 5, 18, 0, 0, 0, tzinfo=UTC),
                datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
                BarResolution.DAY_1,
            )
        )
        first = bars[0]
        # Float→Decimal via str() preserves exact representation
        assert first.open == Decimal("5230.25")
        assert first.high == Decimal("5248.75")
        assert first.close == Decimal("5240.00")
        assert isinstance(first.open, Decimal)

    def test_volume_is_int(self, source: IBKRHistoricalDataSource) -> None:
        bars = list(
            source.get_bars(
                ContractSymbol("ESM26"),
                datetime(2026, 5, 18, 0, 0, 0, tzinfo=UTC),
                datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
                BarResolution.DAY_1,
            )
        )
        assert bars[0].volume == 1_200_000
        assert isinstance(bars[0].volume, int)

    def test_oi_is_none_for_ibkr(self, source: IBKRHistoricalDataSource) -> None:
        """IBKR daily bars don't include open-interest field."""
        bars = list(
            source.get_bars(
                ContractSymbol("ESM26"),
                datetime(2026, 5, 18, 0, 0, 0, tzinfo=UTC),
                datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
                BarResolution.DAY_1,
            )
        )
        assert bars[0].oi is None

    def test_tz_aware_ts(self, source: IBKRHistoricalDataSource) -> None:
        bars = list(
            source.get_bars(
                ContractSymbol("ESM26"),
                datetime(2026, 5, 18, 0, 0, 0, tzinfo=UTC),
                datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
                BarResolution.DAY_1,
            )
        )
        for b in bars:
            assert b.ts.tzinfo is not None
            assert b.ts.tzinfo.utcoffset(b.ts) == timedelta(0)  # UTC

    def test_source_id_propagates(self, source: IBKRHistoricalDataSource) -> None:
        bars = list(
            source.get_bars(
                ContractSymbol("ESM26"),
                datetime(2026, 5, 18, 0, 0, 0, tzinfo=UTC),
                datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
                BarResolution.DAY_1,
            )
        )
        assert all(b.source_id == "ibkr_tws_historical" for b in bars)

    def test_content_bytes_sha_is_64_hex(self, source: IBKRHistoricalDataSource) -> None:
        bars = list(
            source.get_bars(
                ContractSymbol("ESM26"),
                datetime(2026, 5, 18, 0, 0, 0, tzinfo=UTC),
                datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
                BarResolution.DAY_1,
            )
        )
        for b in bars:
            assert len(b.content_bytes_sha) == 64
            assert all(c in "0123456789abcdef" for c in b.content_bytes_sha)

    def test_content_sha_unique_per_bar(self, source: IBKRHistoricalDataSource) -> None:
        bars = list(
            source.get_bars(
                ContractSymbol("ESM26"),
                datetime(2026, 5, 18, 0, 0, 0, tzinfo=UTC),
                datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
                BarResolution.DAY_1,
            )
        )
        shas = {b.content_bytes_sha for b in bars}
        assert len(shas) == len(bars)  # all distinct

    def test_request_uses_correct_parameters(
        self, source: IBKRHistoricalDataSource, es_bars_ib_client: FixtureIBClient
    ) -> None:
        list(
            source.get_bars(
                ContractSymbol("ESM26"),
                datetime(2026, 5, 18, 0, 0, 0, tzinfo=UTC),
                datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
                BarResolution.DAY_1,
            )
        )
        last = es_bars_ib_client.last_req_args
        assert last["bar_size"] == "1 day"
        assert last["what_to_show"] == "TRADES"
        assert last["use_rth"] == 0  # default false → int(False) = 0
        assert "20260522" in last["end_datetime"]
        # Duration: 4 days delta → "4 D"
        assert last["duration"] == "4 D"

    def test_use_rth_true_sends_1(
        self, es_bars_ib_client: FixtureIBClient, fixed_clock: FixtureClock
    ) -> None:
        source = make_source(es_bars_ib_client, fixed_clock, use_rth=True)
        list(
            source.get_bars(
                ContractSymbol("ESM26"),
                datetime(2026, 5, 18, 0, 0, 0, tzinfo=UTC),
                datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
                BarResolution.DAY_1,
            )
        )
        assert es_bars_ib_client.last_req_args["use_rth"] == 1


# ============================================================================
# TestIBKR_GetBarsErrorPaths — error handling
# ============================================================================


class TestIBKR_GetBarsErrorPaths:
    @pytest.fixture
    def empty_source(
        self,
        empty_ib_client: FixtureIBClient,
        fixed_clock: FixtureClock,
    ) -> IBKRHistoricalDataSource:
        return make_source(empty_ib_client, fixed_clock)

    def test_ts_end_before_ts_start_raises_value_error(
        self, empty_source: IBKRHistoricalDataSource
    ) -> None:
        start = datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC)
        end = datetime(2026, 5, 21, 21, 0, 0, tzinfo=UTC)
        with pytest.raises(ValueError, match="must be after"):
            list(empty_source.get_bars(ContractSymbol("ESM26"), start, end, BarResolution.MIN_1))

    def test_ts_end_equals_ts_start_raises(self, empty_source: IBKRHistoricalDataSource) -> None:
        ts = datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC)
        with pytest.raises(ValueError, match="must be after"):
            list(empty_source.get_bars(ContractSymbol("ESM26"), ts, ts, BarResolution.MIN_1))

    def test_unknown_contract_root_raises(self, empty_source: IBKRHistoricalDataSource) -> None:
        with pytest.raises(ContractNotConfigured):
            list(
                empty_source.get_bars(
                    ContractSymbol("ZNM26"),  # 10y T-note not in our universe
                    datetime(2026, 5, 18, 0, 0, 0, tzinfo=UTC),
                    datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
                    BarResolution.DAY_1,
                )
            )

    def test_settle_resolution_raises_bars_not_supported(
        self, empty_source: IBKRHistoricalDataSource
    ) -> None:
        with pytest.raises(BarsNotSupported, match="SETTLE"):
            list(
                empty_source.get_bars(
                    ContractSymbol("ESM26"),
                    datetime(2026, 5, 18, 0, 0, 0, tzinfo=UTC),
                    datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
                    BarResolution.SETTLE,
                )
            )

    def test_req_failure_wraps_to_ibkr_req_error(self, fixed_clock: FixtureClock) -> None:
        client = FixtureIBClient(req_should_fail=True)
        source = make_source(client, fixed_clock)
        with pytest.raises(IBKRReqError, match="reqHistoricalData failed"):
            list(
                source.get_bars(
                    ContractSymbol("ESM26"),
                    datetime(2026, 5, 18, 0, 0, 0, tzinfo=UTC),
                    datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
                    BarResolution.DAY_1,
                )
            )

    def test_exception_hierarchy(self) -> None:
        assert issubclass(IBKRConnectionError, IBKRError)
        assert issubclass(IBKRReqError, IBKRError)
        assert issubclass(IBKRError, DataSourceError)


# ============================================================================
# TestIBKR_TimestampCoercion — IB BarData.date → TZ-aware UTC
# ============================================================================


class TestIBKR_TimestampCoercion:
    @pytest.fixture
    def source(
        self,
        empty_ib_client: FixtureIBClient,
        fixed_clock: FixtureClock,
    ) -> IBKRHistoricalDataSource:
        return make_source(empty_ib_client, fixed_clock)

    def test_m1_naive_datetime_raises(self, source: IBKRHistoricalDataSource) -> None:
        """Naive datetime → IBKRReqError, NOT silent UTC.

        Prior behavior silently coerced naive (account-TZ-leaked) → UTC,
        which shifted bars 5-6h if IB Gateway was on CT (default). Fail-loud: refuse
        silent TZ assumption; raise so caller surfaces the bug.
        """
        naive = datetime(2026, 5, 21, 22, 0, 0)  # no tzinfo
        with pytest.raises(IBKRReqError, match="naive datetime"):
            source._coerce_ts_to_utc(naive)

    def test_m1_naive_iso_string_raises(self, source: IBKRHistoricalDataSource) -> None:
        """Naive ISO string → IBKRReqError."""
        naive_iso = "2026-05-21T22:00:00"  # no TZ suffix
        with pytest.raises(IBKRReqError, match="naive ISO datetime"):
            source._coerce_ts_to_utc(naive_iso)

    def test_aware_datetime_preserved(self, source: IBKRHistoricalDataSource) -> None:
        aware = datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC)
        result = source._coerce_ts_to_utc(aware)
        assert result == aware

    def test_epoch_int_converted_to_utc(self, source: IBKRHistoricalDataSource) -> None:
        # 2026-05-21T22:00:00Z = epoch 1779336000
        epoch = 1779379200  # adjust as needed
        result = source._coerce_ts_to_utc(epoch)
        assert result.tzinfo == UTC

    def test_iso_string_converted(self, source: IBKRHistoricalDataSource) -> None:
        iso = "2026-05-21T22:00:00+00:00"
        result = source._coerce_ts_to_utc(iso)
        assert result.tzinfo is not None
        assert result.year == 2026

    def test_daily_date_anchored_to_noon_utc(self, source: IBKRHistoricalDataSource) -> None:
        """IB returns DAILY bars as a plain `date` (formatDate=2 gives an epoch int only for
        intraday). The fix anchors at noon UTC so the session date is stable in BOTH UTC and CME tz
        (engine keys on ts.date()). Regression for a live "cannot coerce date" crash."""
        result = source._coerce_ts_to_utc(date(2026, 2, 4))
        assert result == datetime(2026, 2, 4, 12, 0, tzinfo=UTC)
        assert result.date() == date(2026, 2, 4)  # stable in UTC
        # ... and in CME tz (noon UTC never crosses a midnight boundary in US zones)
        assert result.astimezone(ZoneInfo("America/Chicago")).date() == date(2026, 2, 4)

    def test_bad_type_raises(self, source: IBKRHistoricalDataSource) -> None:
        with pytest.raises(IBKRReqError, match="cannot coerce"):
            source._coerce_ts_to_utc([1, 2, 3])  # invalid type


# ============================================================================
# TestIBKR_BackoffIntegration — real BackoffQueue records IBKR contract keys
# ============================================================================


class TestIBKR_BackoffIntegration:
    def test_request_increments_global_queue_count(
        self, es_bars_ib_client: FixtureIBClient, fixed_clock: FixtureClock
    ) -> None:
        queue = BackoffQueue()
        source = IBKRHistoricalDataSource(
            ib_client=es_bars_ib_client,
            backoff_queue=queue,
            clock=fixed_clock,
        )
        assert queue.global_count() == 0
        list(
            source.get_bars(
                ContractSymbol("ESM26"),
                datetime(2026, 5, 18, 0, 0, 0, tzinfo=UTC),
                datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
                BarResolution.DAY_1,
            )
        )
        assert queue.global_count() == 1

    def test_per_contract_key_format(
        self, es_bars_ib_client: FixtureIBClient, fixed_clock: FixtureClock
    ) -> None:
        queue = BackoffQueue()
        source = IBKRHistoricalDataSource(
            ib_client=es_bars_ib_client,
            backoff_queue=queue,
            clock=fixed_clock,
        )
        list(
            source.get_bars(
                ContractSymbol("ESM26"),
                datetime(2026, 5, 18, 0, 0, 0, tzinfo=UTC),
                datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
                BarResolution.DAY_1,
            )
        )
        # Key format: <contract>@<exchange>@<what_to_show>
        assert queue.per_contract_count("ESM26@CME@TRADES") == 1


# ============================================================================
# TestA1_5_LatestSettle — daily-bar → Settle conversion + Friday-lag handling
# ============================================================================


class TestA1_5_LatestSettle:
    """A1.5 — IBKRHistoricalDataSource.latest_settle.

    Per internal notes: IBKR daily-bar close IS the official settlement once
    published; preliminary/final state is derived heuristically from elapsed
    time since session close (18h Mon-Thu, 42h Fri lag thresholds).
    """

    # ----- Happy path: ESM26 settle, fixed clock post-Thu-close = preliminary -----

    def test_happy_path_returns_settle(
        self,
        es_bars_ib_client: FixtureIBClient,
        fixed_clock: FixtureClock,
    ) -> None:
        """ES daily bar at 2026-05-21 22:00 UTC (Thu 17:00 CT close) +
        as_of also 2026-05-21 22:00 UTC = 0h elapsed → 'preliminary'."""
        source = make_source(es_bars_ib_client, fixed_clock)
        as_of = datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC)
        settle = source.latest_settle(ContractSymbol("ESM26"), as_of)
        assert settle is not None
        assert settle.contract == ContractSymbol("ESM26")
        # Latest bar's close (5260.00) becomes both settle + last
        assert settle.settle == Decimal("5260.00")
        assert settle.last == Decimal("5260.00")
        # Prior close 5252.00 → change = 5260 - 5252 = 8
        assert settle.change == Decimal("8.00")
        assert settle.open == Decimal("5252.25")
        assert settle.high == Decimal("5263.50")
        assert settle.low == Decimal("5248.75")
        assert settle.volume_est == 1_280_000

    def test_settle_state_preliminary_within_weekday_lag(
        self,
        es_bars_ib_client: FixtureIBClient,
    ) -> None:
        """Thu 22:00 UTC close, queried Thu 22:00 UTC → 0h elapsed < 18h → preliminary."""
        clock = FixtureClock(fixed_time=datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC))
        source = make_source(es_bars_ib_client, clock)
        as_of = datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC)
        settle = source.latest_settle(ContractSymbol("ESM26"), as_of)
        assert settle is not None
        assert settle.settle_state == "preliminary"

    def test_settle_state_final_after_weekday_lag(
        self,
        es_bars_ib_client: FixtureIBClient,
    ) -> None:
        """Thu 22:00 UTC close, queried Fri 17:00 UTC → 19h elapsed > 18h → final."""
        clock = FixtureClock(fixed_time=datetime(2026, 5, 22, 17, 0, 0, tzinfo=UTC))
        source = make_source(es_bars_ib_client, clock)
        as_of = datetime(2026, 5, 22, 17, 0, 0, tzinfo=UTC)
        settle = source.latest_settle(ContractSymbol("ESM26"), as_of)
        assert settle is not None
        assert settle.settle_state == "final"

    def test_settle_state_preliminary_within_friday_extended_lag(self) -> None:
        """Fri 21:00 UTC close (Fri 16:00 CT), queried Sat 13:00 UTC = 16h
        elapsed < 42h → preliminary (Friday extended-lag window)."""
        bars = [
            FixtureIBBar(
                date=datetime(2026, 5, 22, 21, 0, 0, tzinfo=UTC),  # Fri 16:00 CT
                open=5260.0,
                high=5275.0,
                low=5258.0,
                close=5270.0,
                volume=1_100_000,
            ),
        ]
        ib_client = FixtureIBClient(bars_by_contract={("ES", "202606", "CME"): bars})
        clock = FixtureClock(fixed_time=datetime(2026, 5, 23, 13, 0, 0, tzinfo=UTC))
        source = make_source(ib_client, clock)
        as_of = datetime(2026, 5, 23, 13, 0, 0, tzinfo=UTC)
        settle = source.latest_settle(ContractSymbol("ESM26"), as_of)
        assert settle is not None
        assert settle.settle_state == "preliminary"

    def test_settle_state_final_after_friday_extended_lag(self) -> None:
        """Fri 21:00 UTC close + Sun 18:00 UTC query = 45h elapsed > 42h → final."""
        bars = [
            FixtureIBBar(
                date=datetime(2026, 5, 22, 21, 0, 0, tzinfo=UTC),  # Fri 16:00 CT
                open=5260.0,
                high=5275.0,
                low=5258.0,
                close=5270.0,
                volume=1_100_000,
            ),
        ]
        ib_client = FixtureIBClient(bars_by_contract={("ES", "202606", "CME"): bars})
        clock = FixtureClock(fixed_time=datetime(2026, 5, 24, 18, 0, 0, tzinfo=UTC))
        source = make_source(ib_client, clock)
        as_of = datetime(2026, 5, 24, 18, 0, 0, tzinfo=UTC)
        settle = source.latest_settle(ContractSymbol("ESM26"), as_of)
        assert settle is not None
        assert settle.settle_state == "final"

    def test_settle_state_friday_lag_distinguishes_from_weekday(self) -> None:
        """Fri 21:00 UTC + Sat 18:00 UTC = 21h elapsed. Weekday cutoff (18h) would
        say 'final', Friday cutoff (42h) says 'preliminary'. Ensures Friday path fires."""
        bars = [
            FixtureIBBar(
                date=datetime(2026, 5, 22, 21, 0, 0, tzinfo=UTC),  # Fri 16:00 CT
                open=5260.0,
                high=5275.0,
                low=5258.0,
                close=5270.0,
                volume=1_100_000,
            ),
        ]
        ib_client = FixtureIBClient(bars_by_contract={("ES", "202606", "CME"): bars})
        clock = FixtureClock(fixed_time=datetime(2026, 5, 23, 18, 0, 0, tzinfo=UTC))
        source = make_source(ib_client, clock)
        as_of = datetime(2026, 5, 23, 18, 0, 0, tzinfo=UTC)
        settle = source.latest_settle(ContractSymbol("ESM26"), as_of)
        assert settle is not None
        assert settle.settle_state == "preliminary"

    # ----- Edge cases: no bars / single-bar / naive datetime -----

    def test_returns_none_when_ib_returns_empty(
        self,
        empty_ib_client: FixtureIBClient,
        fixed_clock: FixtureClock,
    ) -> None:
        """Empty IB response (pre-listing, expired without includeExpired) → None."""
        source = make_source(empty_ib_client, fixed_clock)
        as_of = datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC)
        assert source.latest_settle(ContractSymbol("ESM26"), as_of) is None

    def test_single_bar_change_defaults_to_zero(
        self,
        fixed_clock: FixtureClock,
    ) -> None:
        """First session of contract life: only 1 bar available → change = Decimal(0)."""
        bars = [
            FixtureIBBar(
                date=datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
                open=5230.0,
                high=5248.0,
                low=5225.0,
                close=5240.0,
                volume=1_000_000,
            ),
        ]
        ib_client = FixtureIBClient(bars_by_contract={("ES", "202606", "CME"): bars})
        source = make_source(ib_client, fixed_clock)
        settle = source.latest_settle(
            ContractSymbol("ESM26"),
            datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        )
        assert settle is not None
        assert settle.change == Decimal("0")
        assert settle.settle == Decimal("5240.00")

    def test_naive_as_of_raises_value_error(
        self,
        es_bars_ib_client: FixtureIBClient,
        fixed_clock: FixtureClock,
    ) -> None:
        """Naive datetime → ValueError (TZ-aware always; fail-loud)."""
        source = make_source(es_bars_ib_client, fixed_clock)
        with pytest.raises(ValueError, match="TZ-aware"):
            source.latest_settle(
                ContractSymbol("ESM26"),
                datetime(2026, 5, 21, 22, 0, 0),  # naive
            )

    def test_non_utc_tz_coerces_to_utc(
        self,
        es_bars_ib_client: FixtureIBClient,
        fixed_clock: FixtureClock,
    ) -> None:
        """as_of in CT TZ → coerced to UTC for IBKR endDateTime."""
        source = make_source(es_bars_ib_client, fixed_clock)
        # 17:00 CT = 22:00 UTC (CDT)
        as_of_ct = datetime(2026, 5, 21, 17, 0, 0, tzinfo=ZoneInfo("America/Chicago"))
        settle = source.latest_settle(ContractSymbol("ESM26"), as_of_ct)
        assert settle is not None
        # endDateTime passed to IB should be UTC-stringified
        end_dt = es_bars_ib_client.last_req_args["end_datetime"]
        assert "22:00:00" in end_dt
        assert "UTC" in end_dt

    # ----- Decimal precision + TZ + provenance invariants -----

    def test_decimal_precision_preserved(
        self,
        fixed_clock: FixtureClock,
    ) -> None:
        """Float close like 5240.12345 → Decimal('5240.12345') exact via str() coercion."""
        bars = [
            FixtureIBBar(
                date=datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
                open=5230.5,
                high=5248.7,
                low=5225.3,
                close=5240.12345,
                volume=999_999,
            ),
        ]
        ib_client = FixtureIBClient(bars_by_contract={("ES", "202606", "CME"): bars})
        source = make_source(ib_client, fixed_clock)
        settle = source.latest_settle(
            ContractSymbol("ESM26"),
            datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        )
        assert settle is not None
        assert settle.settle == Decimal("5240.12345")
        # No IEEE-754 leak: Decimal(str(float)) preserves the printed-string exactly
        assert str(settle.settle) == "5240.12345"

    def test_as_of_iso_uses_clock_now(
        self,
        es_bars_ib_client: FixtureIBClient,
    ) -> None:
        """Settle.as_of_iso reflects clock.now_utc() at fetch time, not as_of param."""
        clock_now = datetime(2026, 5, 22, 9, 30, 0, tzinfo=UTC)
        clock = FixtureClock(fixed_time=clock_now)
        source = make_source(es_bars_ib_client, clock)
        settle = source.latest_settle(
            ContractSymbol("ESM26"),
            datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        )
        assert settle is not None
        assert settle.as_of_iso == clock_now

    def test_as_of_date_uses_session_close_in_ct(
        self,
        es_bars_ib_client: FixtureIBClient,
        fixed_clock: FixtureClock,
    ) -> None:
        """Bar at 2026-05-21 22:00 UTC = 2026-05-21 17:00 CT → as_of_date = 2026-05-21."""
        source = make_source(es_bars_ib_client, fixed_clock)
        settle = source.latest_settle(
            ContractSymbol("ESM26"),
            datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        )
        assert settle is not None
        assert settle.as_of_date == date(2026, 5, 21)

    def test_content_bytes_sha_is_64_hex(
        self,
        es_bars_ib_client: FixtureIBClient,
        fixed_clock: FixtureClock,
    ) -> None:
        """content_bytes_sha must be hex-SHA256 (64 chars)."""
        source = make_source(es_bars_ib_client, fixed_clock)
        settle = source.latest_settle(
            ContractSymbol("ESM26"),
            datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        )
        assert settle is not None
        assert len(settle.content_bytes_sha) == 64
        assert all(c in "0123456789abcdef" for c in settle.content_bytes_sha)

    def test_content_bytes_sha_is_deterministic(
        self,
        fixed_clock: FixtureClock,
    ) -> None:
        """Same bar payload → same content_bytes_sha (bit-reproducibility)."""
        bars = [
            FixtureIBBar(
                date=datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
                open=5230.0,
                high=5248.0,
                low=5225.0,
                close=5240.0,
                volume=1_000_000,
            ),
        ]
        ib_client_a = FixtureIBClient(bars_by_contract={("ES", "202606", "CME"): bars})
        ib_client_b = FixtureIBClient(bars_by_contract={("ES", "202606", "CME"): bars})
        source_a = make_source(ib_client_a, fixed_clock)
        source_b = make_source(ib_client_b, fixed_clock)
        as_of = datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC)
        settle_a = source_a.latest_settle(ContractSymbol("ESM26"), as_of)
        settle_b = source_b.latest_settle(ContractSymbol("ESM26"), as_of)
        assert settle_a is not None and settle_b is not None
        assert settle_a.content_bytes_sha == settle_b.content_bytes_sha

    def test_cme_month_code_extracted_from_contract(
        self,
        fixed_clock: FixtureClock,
    ) -> None:
        """ESM26 → cme_month_code='M' (June)."""
        bars = [
            FixtureIBBar(
                date=datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
                open=5230.0,
                high=5248.0,
                low=5225.0,
                close=5240.0,
                volume=1_000_000,
            ),
        ]
        ib_client = FixtureIBClient(bars_by_contract={("ES", "202606", "CME"): bars})
        source = make_source(ib_client, fixed_clock)
        settle = source.latest_settle(
            ContractSymbol("ESM26"),
            datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        )
        assert settle is not None
        assert settle.cme_month_code == "M"

    def test_source_id_matches_class_constant(
        self,
        es_bars_ib_client: FixtureIBClient,
        fixed_clock: FixtureClock,
    ) -> None:
        """Settle.source_id = SOURCE_ID for verifier-side provenance routing."""
        source = make_source(es_bars_ib_client, fixed_clock)
        settle = source.latest_settle(
            ContractSymbol("ESM26"),
            datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        )
        assert settle is not None
        assert settle.source_id == IBKRHistoricalDataSource.SOURCE_ID
        assert settle.source_id == "ibkr_tws_historical"

    def test_oi_prior_is_zero_for_ibkr(
        self,
        es_bars_ib_client: FixtureIBClient,
        fixed_clock: FixtureClock,
    ) -> None:
        """IBKR daily bars don't publish OI — structurally zero (A1.9 verifier policy)."""
        source = make_source(es_bars_ib_client, fixed_clock)
        settle = source.latest_settle(
            ContractSymbol("ESM26"),
            datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        )
        assert settle is not None
        assert settle.oi_prior == 0

    # ----- IBKR API call shape (BackoffQueue + endDateTime + duration) -----

    def test_request_uses_2d_duration_for_prior_diff(
        self,
        es_bars_ib_client: FixtureIBClient,
        fixed_clock: FixtureClock,
    ) -> None:
        """Need 2 days for change = latest - prior diff."""
        source = make_source(es_bars_ib_client, fixed_clock)
        source.latest_settle(
            ContractSymbol("ESM26"),
            datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        )
        assert es_bars_ib_client.last_req_args["duration"] == "2 D"
        assert es_bars_ib_client.last_req_args["bar_size"] == "1 day"
        assert es_bars_ib_client.last_req_args["what_to_show"] == "TRADES"

    def test_use_rth_propagates_from_source_config(
        self,
        es_bars_ib_client: FixtureIBClient,
        fixed_clock: FixtureClock,
    ) -> None:
        """use_rth=True → IB call gets use_rth=1; default → 0."""
        source_rth = make_source(es_bars_ib_client, fixed_clock, use_rth=True)
        source_rth.latest_settle(
            ContractSymbol("ESM26"),
            datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        )
        assert es_bars_ib_client.last_req_args["use_rth"] == 1

        # Reset and verify default
        es_bars_ib_client.last_req_args = {}
        source_eth = make_source(es_bars_ib_client, fixed_clock)
        source_eth.latest_settle(
            ContractSymbol("ESM26"),
            datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        )
        assert es_bars_ib_client.last_req_args["use_rth"] == 0

    def test_backoff_queue_acquired_on_settle_call(
        self,
        es_bars_ib_client: FixtureIBClient,
        fixed_clock: FixtureClock,
    ) -> None:
        """latest_settle must go through BackoffQueue (rate-limit apparatus)."""
        queue = BackoffQueue()
        source = IBKRHistoricalDataSource(
            ib_client=es_bars_ib_client,
            backoff_queue=queue,
            clock=fixed_clock,
        )
        assert queue.global_count() == 0
        source.latest_settle(
            ContractSymbol("ESM26"),
            datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        )
        assert queue.global_count() == 1
        assert queue.per_contract_count("ESM26@CME@TRADES") == 1

    # ----- Error paths -----

    def test_connect_failure_raises_connection_error(
        self,
        fixed_clock: FixtureClock,
    ) -> None:
        ib_client = FixtureIBClient(connect_should_fail=True)
        source = make_source(ib_client, fixed_clock)
        with pytest.raises(IBKRConnectionError):
            source.latest_settle(
                ContractSymbol("ESM26"),
                datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
            )

    def test_req_failure_wraps_as_ibkr_req_error(
        self,
        fixed_clock: FixtureClock,
    ) -> None:
        """RuntimeError from IB → IBKRReqError preserves chain (raise from)."""
        ib_client = FixtureIBClient(req_should_fail=True)
        source = make_source(ib_client, fixed_clock)
        with pytest.raises(IBKRReqError, match="latest_settle reqHistoricalData failed"):
            source.latest_settle(
                ContractSymbol("ESM26"),
                datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
            )

    def test_unknown_contract_root_raises_not_configured(
        self,
        empty_ib_client: FixtureIBClient,
        fixed_clock: FixtureClock,
    ) -> None:
        """ZNM26 (10y T-note, not in the configured universe) → ContractNotConfigured."""
        source = make_source(empty_ib_client, fixed_clock)
        with pytest.raises(ContractNotConfigured, match="not in EXCHANGE_BY_ROOT"):
            source.latest_settle(
                ContractSymbol("ZNM26"),
                datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
            )

    def test_ibkr_error_hierarchy_intact(
        self,
        fixed_clock: FixtureClock,
    ) -> None:
        """IBKRConnectionError + IBKRReqError both inherit IBKRError + DataSourceError."""
        ib_client = FixtureIBClient(req_should_fail=True)
        source = make_source(ib_client, fixed_clock)
        with pytest.raises(IBKRError):
            source.latest_settle(
                ContractSymbol("ESM26"),
                datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
            )
        ib_client.req_should_fail = True
        with pytest.raises(DataSourceError):
            source.latest_settle(
                ContractSymbol("ESM26"),
                datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
            )

    # ----- Sort defensiveness: unordered IB response handled -----

    def test_unsorted_bars_sorted_ascending_internally(
        self,
        fixed_clock: FixtureClock,
    ) -> None:
        """If IB returns bars out-of-order, latest_settle picks the truly-latest."""
        bars = [
            FixtureIBBar(  # day 2 (actual latest)
                date=datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
                open=5252.0,
                high=5263.0,
                low=5248.0,
                close=5260.0,
                volume=1_280_000,
            ),
            FixtureIBBar(  # day 1 (sent first, but earlier)
                date=datetime(2026, 5, 20, 22, 0, 0, tzinfo=UTC),
                open=5240.0,
                high=5258.0,
                low=5237.0,
                close=5252.0,
                volume=1_150_000,
            ),
        ]
        ib_client = FixtureIBClient(bars_by_contract={("ES", "202606", "CME"): bars})
        source = make_source(ib_client, fixed_clock)
        settle = source.latest_settle(
            ContractSymbol("ESM26"),
            datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        )
        assert settle is not None
        # Latest is 5260, prior is 5252 → change = 8 (regardless of input order)
        assert settle.settle == Decimal("5260.00")
        assert settle.change == Decimal("8.00")
