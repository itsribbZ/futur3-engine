"""A1.10a CcxtCryptoDataSource + CoinbaseCryptoDataSource test suite.

Test discipline:
- ALL tests fixture-only (zero network).
- FixtureCcxtExchange injects canned OHLCV + error scenarios.
- Boundary invariants: TZ-aware ts, Decimal-not-float prices, content_bytes_sha
  64-hex, half-open [ts_start, ts_end) window filter.
- Hard seam: vendor ccxt never touched in tests; CcxtExchange Protocol enforced.

References:
- futur3/data/sources/crypto.py (implementation)
- internal crypto-data notes (Coinbase spec)
- the data-layer design (Phase A1 step)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import ClassVar

import pytest

from futur3.data.source import (
    ContractNotConfigured,
    DataSource,
    DataSourceError,
    SettlesNotSupported,
    TicksNotSupported,
)
from futur3.data.sources.crypto import (
    BinanceUSCryptoDataSource,
    BitstampCryptoDataSource,
    CcxtCryptoDataSource,
    CcxtExchange,
    CoinbaseCryptoDataSource,
    GeminiCryptoDataSource,
    KrakenCryptoDataSource,
)
from futur3.data.types import (
    BarResolution,
    ContractSymbol,
    SourceTier,
)

# ============================================================================
# FixtureCcxtExchange — hard-seam stub
# ============================================================================


@dataclass
class FixtureCcxtExchange:
    """In-memory CcxtExchange satisfying the structural Protocol.

    Stores canned OHLCV per (symbol, timeframe) key; supports error injection.
    Records calls for assertion.
    """

    id: str = "coinbase"
    ohlcv_by_key: dict[tuple[str, str], list[list[float]]] = field(default_factory=dict)
    fetch_should_fail: bool = False
    fetch_exception_type: type[Exception] = RuntimeError
    fetch_call_count: int = 0
    last_fetch_args: dict[str, object] = field(default_factory=dict)

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        since: int | None = None,
        limit: int | None = None,
    ) -> list[list[float]]:
        self.fetch_call_count += 1
        self.last_fetch_args = {
            "symbol": symbol,
            "timeframe": timeframe,
            "since": since,
            "limit": limit,
        }
        if self.fetch_should_fail:
            raise self.fetch_exception_type(
                f"FixtureCcxtExchange: injected fetch failure for {symbol}/{timeframe}"
            )
        return list(self.ohlcv_by_key.get((symbol, timeframe), []))


# ============================================================================
# Helper builders
# ============================================================================


def _ms(dt: datetime) -> int:
    """TZ-aware datetime → ccxt epoch ms."""
    return int(dt.astimezone(UTC).timestamp() * 1000)


def _coinbase(exchange: FixtureCcxtExchange) -> CoinbaseCryptoDataSource:
    return CoinbaseCryptoDataSource(exchange=exchange)


# Standard 3-bar fixture: 2026-05-19/20/21 daily BTC closes, 1d resolution
def _three_daily_btc_bars() -> list[list[float]]:
    d1 = _ms(datetime(2026, 5, 19, 0, 0, 0, tzinfo=UTC))
    d2 = _ms(datetime(2026, 5, 20, 0, 0, 0, tzinfo=UTC))
    d3 = _ms(datetime(2026, 5, 21, 0, 0, 0, tzinfo=UTC))
    return [
        [d1, 67000.0, 67250.0, 66800.0, 67100.0, 1234.5],
        [d2, 67100.0, 67400.0, 66900.0, 67200.0, 1180.2],
        [d3, 67200.0, 67500.0, 67000.0, 67350.0, 1320.0],
    ]


# ============================================================================
# TestA1_10a_Protocol — CcxtExchange shape sanity
# ============================================================================


class TestA1_10a_Protocol:
    def test_protocol_runtime_checkable(self) -> None:
        assert hasattr(CcxtExchange, "fetch_ohlcv")
        # runtime_checkable lets isinstance work for structural matching
        fx = FixtureCcxtExchange()
        assert isinstance(fx, CcxtExchange)

    def test_fixture_satisfies_protocol(self) -> None:
        fx = FixtureCcxtExchange(id="coinbase")
        assert fx.id == "coinbase"
        result = fx.fetch_ohlcv("BTC/USD", timeframe="1d")
        assert result == []


# ============================================================================
# TestA1_10a_BaseConfigRequired — base class enforces VENUE_NAME/SOURCE_ID/SYMBOL_MAP
# ============================================================================


class TestA1_10a_BaseConfigRequired:
    def test_base_class_no_venue_name_raises(self) -> None:
        """Bare CcxtCryptoDataSource has empty class-level constants → raise."""
        with pytest.raises(ContractNotConfigured, match="VENUE_NAME"):
            CcxtCryptoDataSource(exchange=FixtureCcxtExchange())

    def test_subclass_missing_source_id_raises(self) -> None:
        class _BadSubclass(CcxtCryptoDataSource):
            VENUE_NAME: ClassVar[str] = "coinbase"
            # SOURCE_ID + SYMBOL_MAP intentionally not set

        with pytest.raises(ContractNotConfigured, match="SOURCE_ID"):
            _BadSubclass(exchange=FixtureCcxtExchange())

    def test_subclass_missing_symbol_map_raises(self) -> None:
        class _BadSubclass(CcxtCryptoDataSource):
            VENUE_NAME: ClassVar[str] = "coinbase"
            SOURCE_ID: ClassVar[str] = "coinbase_advanced"
            # SYMBOL_MAP intentionally not set

        with pytest.raises(ContractNotConfigured, match="SYMBOL_MAP"):
            _BadSubclass(exchange=FixtureCcxtExchange())


# ============================================================================
# TestA1_10a_CoinbaseShape — concrete subclass class-level config
# ============================================================================


class TestA1_10a_CoinbaseShape:
    def test_is_datasource(self) -> None:
        assert issubclass(CoinbaseCryptoDataSource, CcxtCryptoDataSource)
        assert issubclass(CoinbaseCryptoDataSource, DataSource)

    def test_venue_name(self) -> None:
        assert CoinbaseCryptoDataSource.VENUE_NAME == "coinbase"

    def test_source_id(self) -> None:
        src = _coinbase(FixtureCcxtExchange())
        assert src.source_id == "coinbase_advanced"
        assert src.source_id == CoinbaseCryptoDataSource.SOURCE_ID

    def test_tier_t2_exchange(self) -> None:
        src = _coinbase(FixtureCcxtExchange())
        assert src.tier == SourceTier.T2_EXCHANGE

    def test_symbol_map_btc_eth(self) -> None:
        m = CoinbaseCryptoDataSource.SYMBOL_MAP
        assert m["BTCUSD"] == "BTC/USD"
        assert m["ETHUSD"] == "ETH/USD"

    def test_repr_includes_source_id_and_tier(self) -> None:
        src = _coinbase(FixtureCcxtExchange())
        rep = repr(src)
        assert "coinbase_advanced" in rep
        assert "T2_EXCHANGE" in rep


# ============================================================================
# TestA1_10a_GetBars_Happy — round-trip OHLCV → RawBar
# ============================================================================


class TestA1_10a_GetBars_Happy:
    def _src(self) -> tuple[CoinbaseCryptoDataSource, FixtureCcxtExchange]:
        fx = FixtureCcxtExchange(
            id="coinbase",
            ohlcv_by_key={("BTC/USD", "1d"): _three_daily_btc_bars()},
        )
        return _coinbase(fx), fx

    def test_returns_three_bars(self) -> None:
        src, _ = self._src()
        bars = list(
            src.get_bars(
                ContractSymbol("BTCUSD"),
                datetime(2026, 5, 19, 0, 0, 0, tzinfo=UTC),
                datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
                BarResolution.DAY_1,
            )
        )
        assert len(bars) == 3

    def test_bar_prices_are_decimal_via_str(self) -> None:
        src, _ = self._src()
        bars = list(
            src.get_bars(
                ContractSymbol("BTCUSD"),
                datetime(2026, 5, 19, 0, 0, 0, tzinfo=UTC),
                datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
                BarResolution.DAY_1,
            )
        )
        # Decimal exact equality on first bar's close
        assert bars[0].close == Decimal("67100.0")
        assert isinstance(bars[0].close, Decimal)
        assert isinstance(bars[0].open, Decimal)
        assert isinstance(bars[0].high, Decimal)
        assert isinstance(bars[0].low, Decimal)

    def test_bar_ts_tz_aware_utc(self) -> None:
        src, _ = self._src()
        bars = list(
            src.get_bars(
                ContractSymbol("BTCUSD"),
                datetime(2026, 5, 19, 0, 0, 0, tzinfo=UTC),
                datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
                BarResolution.DAY_1,
            )
        )
        for b in bars:
            assert b.ts.tzinfo is not None
            assert b.ts.tzinfo == UTC

    def test_bar_contract_preserved(self) -> None:
        src, _ = self._src()
        bars = list(
            src.get_bars(
                ContractSymbol("BTCUSD"),
                datetime(2026, 5, 19, 0, 0, 0, tzinfo=UTC),
                datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
                BarResolution.DAY_1,
            )
        )
        for b in bars:
            assert b.contract == ContractSymbol("BTCUSD")

    def test_oi_is_none_for_spot(self) -> None:
        src, _ = self._src()
        bars = list(
            src.get_bars(
                ContractSymbol("BTCUSD"),
                datetime(2026, 5, 19, 0, 0, 0, tzinfo=UTC),
                datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
                BarResolution.DAY_1,
            )
        )
        for b in bars:
            assert b.oi is None

    def test_content_bytes_sha_64_hex(self) -> None:
        src, _ = self._src()
        bars = list(
            src.get_bars(
                ContractSymbol("BTCUSD"),
                datetime(2026, 5, 19, 0, 0, 0, tzinfo=UTC),
                datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
                BarResolution.DAY_1,
            )
        )
        for b in bars:
            assert len(b.content_bytes_sha) == 64
            assert all(c in "0123456789abcdef" for c in b.content_bytes_sha)

    def test_content_bytes_sha_unique_per_bar(self) -> None:
        src, _ = self._src()
        bars = list(
            src.get_bars(
                ContractSymbol("BTCUSD"),
                datetime(2026, 5, 19, 0, 0, 0, tzinfo=UTC),
                datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
                BarResolution.DAY_1,
            )
        )
        shas = {b.content_bytes_sha for b in bars}
        assert len(shas) == len(bars)

    def test_source_id_propagated_to_raw_bar(self) -> None:
        src, _ = self._src()
        bars = list(
            src.get_bars(
                ContractSymbol("BTCUSD"),
                datetime(2026, 5, 19, 0, 0, 0, tzinfo=UTC),
                datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
                BarResolution.DAY_1,
            )
        )
        for b in bars:
            assert b.source_id == "coinbase_advanced"

    def test_volume_int_from_float(self) -> None:
        src, _ = self._src()
        bars = list(
            src.get_bars(
                ContractSymbol("BTCUSD"),
                datetime(2026, 5, 19, 0, 0, 0, tzinfo=UTC),
                datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
                BarResolution.DAY_1,
            )
        )
        # 1234.5 → 1234 (Decimal int truncation)
        assert bars[0].volume == 1234


# ============================================================================
# TestA1_10a_GetBars_Window — half-open [ts_start, ts_end) filter
# ============================================================================


class TestA1_10a_GetBars_Window:
    def test_excludes_bar_at_ts_end_boundary(self) -> None:
        """Bar at ts_end exactly is EXCLUDED (half-open right boundary)."""
        fx = FixtureCcxtExchange(
            id="coinbase",
            ohlcv_by_key={("BTC/USD", "1d"): _three_daily_btc_bars()},
        )
        src = _coinbase(fx)
        # ts_end exactly at May 21 → excludes that day's bar
        bars = list(
            src.get_bars(
                ContractSymbol("BTCUSD"),
                datetime(2026, 5, 19, 0, 0, 0, tzinfo=UTC),
                datetime(2026, 5, 21, 0, 0, 0, tzinfo=UTC),
                BarResolution.DAY_1,
            )
        )
        # Only May 19 + May 20 should be included
        assert len(bars) == 2
        assert bars[0].ts == datetime(2026, 5, 19, 0, 0, 0, tzinfo=UTC)
        assert bars[1].ts == datetime(2026, 5, 20, 0, 0, 0, tzinfo=UTC)

    def test_includes_bar_at_ts_start_boundary(self) -> None:
        """Bar at ts_start exactly IS included (half-open left boundary)."""
        fx = FixtureCcxtExchange(
            id="coinbase",
            ohlcv_by_key={("BTC/USD", "1d"): _three_daily_btc_bars()},
        )
        src = _coinbase(fx)
        bars = list(
            src.get_bars(
                ContractSymbol("BTCUSD"),
                datetime(2026, 5, 19, 0, 0, 0, tzinfo=UTC),
                datetime(2026, 5, 20, 0, 0, 0, tzinfo=UTC),
                BarResolution.DAY_1,
            )
        )
        assert len(bars) == 1
        assert bars[0].ts == datetime(2026, 5, 19, 0, 0, 0, tzinfo=UTC)


# ============================================================================
# TestA1_10a_GetBars_FetchCall — ccxt call shape
# ============================================================================


class TestA1_10a_GetBars_FetchCall:
    def test_fetch_called_with_btcusd_to_btc_slash_usd(self) -> None:
        fx = FixtureCcxtExchange(
            id="coinbase",
            ohlcv_by_key={("BTC/USD", "1d"): []},
        )
        src = _coinbase(fx)
        list(
            src.get_bars(
                ContractSymbol("BTCUSD"),
                datetime(2026, 5, 19, 0, 0, 0, tzinfo=UTC),
                datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
                BarResolution.DAY_1,
            )
        )
        assert fx.last_fetch_args["symbol"] == "BTC/USD"
        assert fx.last_fetch_args["timeframe"] == "1d"

    def test_fetch_since_is_epoch_ms_of_ts_start(self) -> None:
        fx = FixtureCcxtExchange(
            id="coinbase",
            ohlcv_by_key={("BTC/USD", "1m"): []},
        )
        src = _coinbase(fx)
        ts_start = datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC)
        list(
            src.get_bars(
                ContractSymbol("BTCUSD"),
                ts_start,
                datetime(2026, 5, 21, 22, 30, 0, tzinfo=UTC),
                BarResolution.MIN_1,
            )
        )
        expected_ms = int(ts_start.timestamp() * 1000)
        assert fx.last_fetch_args["since"] == expected_ms

    @pytest.mark.parametrize(
        "resolution,expected_tf",
        [
            (BarResolution.MIN_1, "1m"),
            (BarResolution.MIN_5, "5m"),
            (BarResolution.MIN_15, "15m"),
            (BarResolution.HOUR_1, "1h"),
            (BarResolution.DAY_1, "1d"),
        ],
    )
    def test_resolution_to_timeframe_mapping(
        self,
        resolution: BarResolution,
        expected_tf: str,
    ) -> None:
        fx = FixtureCcxtExchange(
            id="coinbase",
            ohlcv_by_key={("BTC/USD", expected_tf): []},
        )
        src = _coinbase(fx)
        list(
            src.get_bars(
                ContractSymbol("BTCUSD"),
                datetime(2026, 5, 19, 0, 0, 0, tzinfo=UTC),
                datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
                resolution,
            )
        )
        assert fx.last_fetch_args["timeframe"] == expected_tf


# ============================================================================
# TestA1_10a_GetBars_Errors — defensive validation + ccxt error wrap
# ============================================================================


class TestA1_10a_GetBars_Errors:
    def test_naive_ts_start_raises(self) -> None:
        src = _coinbase(FixtureCcxtExchange())
        with pytest.raises(ValueError, match="ts_start must be TZ-aware"):
            list(
                src.get_bars(
                    ContractSymbol("BTCUSD"),
                    datetime(2026, 5, 19, 0, 0, 0),
                    datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
                    BarResolution.DAY_1,
                )
            )

    def test_naive_ts_end_raises(self) -> None:
        src = _coinbase(FixtureCcxtExchange())
        with pytest.raises(ValueError, match="ts_end must be TZ-aware"):
            list(
                src.get_bars(
                    ContractSymbol("BTCUSD"),
                    datetime(2026, 5, 19, 0, 0, 0, tzinfo=UTC),
                    datetime(2026, 5, 22, 0, 0, 0),
                    BarResolution.DAY_1,
                )
            )

    def test_ts_end_before_ts_start_raises(self) -> None:
        src = _coinbase(FixtureCcxtExchange())
        with pytest.raises(ValueError, match="must be after"):
            list(
                src.get_bars(
                    ContractSymbol("BTCUSD"),
                    datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
                    datetime(2026, 5, 19, 0, 0, 0, tzinfo=UTC),
                    BarResolution.DAY_1,
                )
            )

    def test_unknown_contract_raises_not_configured(self) -> None:
        src = _coinbase(FixtureCcxtExchange())
        with pytest.raises(ContractNotConfigured, match="not in SYMBOL_MAP"):
            list(
                src.get_bars(
                    ContractSymbol("DOGEUSD"),
                    datetime(2026, 5, 19, 0, 0, 0, tzinfo=UTC),
                    datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
                    BarResolution.DAY_1,
                )
            )

    def test_unsupported_resolution_raises(self) -> None:
        src = _coinbase(FixtureCcxtExchange())
        # SEC_1 not in TIMEFRAME_MAP
        with pytest.raises(ValueError, match="SEC_1 not supported"):
            list(
                src.get_bars(
                    ContractSymbol("BTCUSD"),
                    datetime(2026, 5, 19, 0, 0, 0, tzinfo=UTC),
                    datetime(2026, 5, 19, 0, 1, 0, tzinfo=UTC),
                    BarResolution.SEC_1,
                )
            )

    def test_ccxt_fetch_error_wraps_to_data_source_error(self) -> None:
        fx = FixtureCcxtExchange(fetch_should_fail=True)
        src = _coinbase(fx)
        with pytest.raises(DataSourceError, match="fetch_ohlcv failed"):
            list(
                src.get_bars(
                    ContractSymbol("BTCUSD"),
                    datetime(2026, 5, 19, 0, 0, 0, tzinfo=UTC),
                    datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
                    BarResolution.DAY_1,
                )
            )

    def test_malformed_ohlcv_row_raises_data_source_error(self) -> None:
        """Row with < 6 fields → DataSourceError."""
        d1 = _ms(datetime(2026, 5, 19, 0, 0, 0, tzinfo=UTC))
        fx = FixtureCcxtExchange(
            id="coinbase",
            ohlcv_by_key={
                # 5 fields instead of 6 — malformed
                ("BTC/USD", "1d"): [[d1, 67000.0, 67250.0, 66800.0, 67100.0]],
            },
        )
        src = _coinbase(fx)
        with pytest.raises(DataSourceError, match="malformed OHLCV row"):
            list(
                src.get_bars(
                    ContractSymbol("BTCUSD"),
                    datetime(2026, 5, 19, 0, 0, 0, tzinfo=UTC),
                    datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
                    BarResolution.DAY_1,
                )
            )


# ============================================================================
# TestA1_10a_ABCCompliance — get_ticks + latest_settle defaults
# ============================================================================


class TestA1_10a_ABCCompliance:
    def test_get_ticks_raises_default(self) -> None:
        src = _coinbase(FixtureCcxtExchange())
        as_of = datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC)
        with pytest.raises(TicksNotSupported):
            list(src.get_ticks(ContractSymbol("BTCUSD"), as_of, as_of))

    def test_latest_settle_raises_default(self) -> None:
        """Crypto spot has no settle — ABC default raise."""
        src = _coinbase(FixtureCcxtExchange())
        with pytest.raises(SettlesNotSupported):
            src.latest_settle(
                ContractSymbol("BTCUSD"),
                datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
            )

    def test_healthcheck_true(self) -> None:
        src = _coinbase(FixtureCcxtExchange())
        assert src.healthcheck() is True


# ============================================================================
# TestA1_10a_Precision — Decimal-via-str precision preserved
# ============================================================================


class TestA1_10a_Precision:
    def test_exact_decimal_round_trip(self) -> None:
        """Float 67000.12345 → Decimal('67000.12345') exact (no IEEE-754 leak)."""
        fx = FixtureCcxtExchange(
            id="coinbase",
            ohlcv_by_key={
                ("BTC/USD", "1d"): [
                    [
                        _ms(datetime(2026, 5, 19, 0, 0, 0, tzinfo=UTC)),
                        67000.5,
                        67250.7,
                        66800.3,
                        67100.12345,
                        1000.0,
                    ],
                ],
            },
        )
        src = _coinbase(fx)
        bars = list(
            src.get_bars(
                ContractSymbol("BTCUSD"),
                datetime(2026, 5, 19, 0, 0, 0, tzinfo=UTC),
                datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
                BarResolution.DAY_1,
            )
        )
        assert bars[0].close == Decimal("67100.12345")
        # No float→Decimal IEEE-754 leak — exact str-repr preservation
        assert str(bars[0].close) == "67100.12345"

    def test_content_bytes_sha_deterministic_same_input(self) -> None:
        """Same payload → same sha (bit-repro)."""
        rows = _three_daily_btc_bars()
        fx_a = FixtureCcxtExchange(ohlcv_by_key={("BTC/USD", "1d"): rows})
        fx_b = FixtureCcxtExchange(ohlcv_by_key={("BTC/USD", "1d"): rows})
        src_a = _coinbase(fx_a)
        src_b = _coinbase(fx_b)
        bars_a = list(
            src_a.get_bars(
                ContractSymbol("BTCUSD"),
                datetime(2026, 5, 19, 0, 0, 0, tzinfo=UTC),
                datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
                BarResolution.DAY_1,
            )
        )
        bars_b = list(
            src_b.get_bars(
                ContractSymbol("BTCUSD"),
                datetime(2026, 5, 19, 0, 0, 0, tzinfo=UTC),
                datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
                BarResolution.DAY_1,
            )
        )
        assert [b.content_bytes_sha for b in bars_a] == [b.content_bytes_sha for b in bars_b]


# ============================================================================
# TestA1_10b_Kraken — BRR constituent 2-of-4
# ============================================================================


class TestA1_10b_Kraken:
    def test_class_constants(self) -> None:
        assert KrakenCryptoDataSource.VENUE_NAME == "kraken"
        assert KrakenCryptoDataSource.SOURCE_ID == "kraken_spot"
        assert KrakenCryptoDataSource.SYMBOL_MAP["BTCUSD"] == "BTC/USD"
        assert KrakenCryptoDataSource.SYMBOL_MAP["ETHUSD"] == "ETH/USD"

    def test_construction_with_fixture(self) -> None:
        src = KrakenCryptoDataSource(exchange=FixtureCcxtExchange(id="kraken"))
        assert src.source_id == "kraken_spot"
        assert src.tier == SourceTier.T2_EXCHANGE

    def test_get_bars_round_trip(self) -> None:
        fx = FixtureCcxtExchange(
            id="kraken",
            ohlcv_by_key={("BTC/USD", "1d"): _three_daily_btc_bars()},
        )
        src = KrakenCryptoDataSource(exchange=fx)
        bars = list(
            src.get_bars(
                ContractSymbol("BTCUSD"),
                datetime(2026, 5, 19, 0, 0, 0, tzinfo=UTC),
                datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
                BarResolution.DAY_1,
            )
        )
        assert len(bars) == 3
        assert all(b.source_id == "kraken_spot" for b in bars)


# ============================================================================
# TestA1_10c_Bitstamp — BRR constituent 3-of-4
# ============================================================================


class TestA1_10c_Bitstamp:
    def test_class_constants(self) -> None:
        assert BitstampCryptoDataSource.VENUE_NAME == "bitstamp"
        assert BitstampCryptoDataSource.SOURCE_ID == "bitstamp_spot"
        assert BitstampCryptoDataSource.SYMBOL_MAP["BTCUSD"] == "BTC/USD"

    def test_construction_with_fixture(self) -> None:
        src = BitstampCryptoDataSource(exchange=FixtureCcxtExchange(id="bitstamp"))
        assert src.source_id == "bitstamp_spot"
        assert src.tier == SourceTier.T2_EXCHANGE

    def test_get_bars_round_trip(self) -> None:
        fx = FixtureCcxtExchange(
            id="bitstamp",
            ohlcv_by_key={("BTC/USD", "1d"): _three_daily_btc_bars()},
        )
        src = BitstampCryptoDataSource(exchange=fx)
        bars = list(
            src.get_bars(
                ContractSymbol("BTCUSD"),
                datetime(2026, 5, 19, 0, 0, 0, tzinfo=UTC),
                datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
                BarResolution.DAY_1,
            )
        )
        assert len(bars) == 3
        assert all(b.source_id == "bitstamp_spot" for b in bars)


# ============================================================================
# TestA1_10d_Gemini — BRR constituent 4-of-4
# ============================================================================


class TestA1_10d_Gemini:
    def test_class_constants(self) -> None:
        assert GeminiCryptoDataSource.VENUE_NAME == "gemini"
        assert GeminiCryptoDataSource.SOURCE_ID == "gemini_spot"
        assert GeminiCryptoDataSource.SYMBOL_MAP["BTCUSD"] == "BTC/USD"

    def test_construction_with_fixture(self) -> None:
        src = GeminiCryptoDataSource(exchange=FixtureCcxtExchange(id="gemini"))
        assert src.source_id == "gemini_spot"
        assert src.tier == SourceTier.T2_EXCHANGE

    def test_get_bars_round_trip(self) -> None:
        fx = FixtureCcxtExchange(
            id="gemini",
            ohlcv_by_key={("BTC/USD", "1d"): _three_daily_btc_bars()},
        )
        src = GeminiCryptoDataSource(exchange=fx)
        bars = list(
            src.get_bars(
                ContractSymbol("BTCUSD"),
                datetime(2026, 5, 19, 0, 0, 0, tzinfo=UTC),
                datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
                BarResolution.DAY_1,
            )
        )
        assert len(bars) == 3
        assert all(b.source_id == "gemini_spot" for b in bars)


# ============================================================================
# TestA1_10e_BinanceUS — oracle (NOT BRR constituent)
# ============================================================================


class TestA1_10e_BinanceUS:
    def test_class_constants(self) -> None:
        assert BinanceUSCryptoDataSource.VENUE_NAME == "binanceus"
        assert BinanceUSCryptoDataSource.SOURCE_ID == "binanceus_oracle"
        assert BinanceUSCryptoDataSource.SYMBOL_MAP["BTCUSD"] == "BTC/USD"

    def test_construction_with_fixture(self) -> None:
        src = BinanceUSCryptoDataSource(exchange=FixtureCcxtExchange(id="binanceus"))
        assert src.source_id == "binanceus_oracle"
        assert src.tier == SourceTier.T2_EXCHANGE

    def test_get_bars_round_trip(self) -> None:
        fx = FixtureCcxtExchange(
            id="binanceus",
            ohlcv_by_key={("BTC/USD", "1d"): _three_daily_btc_bars()},
        )
        src = BinanceUSCryptoDataSource(exchange=fx)
        bars = list(
            src.get_bars(
                ContractSymbol("BTCUSD"),
                datetime(2026, 5, 19, 0, 0, 0, tzinfo=UTC),
                datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
                BarResolution.DAY_1,
            )
        )
        assert len(bars) == 3
        assert all(b.source_id == "binanceus_oracle" for b in bars)


# ============================================================================
# TestA1_10_CrossVenue — 5-venue interop sanity
# ============================================================================


class TestA1_10_CrossVenue:
    """Verifier-side cross-venue interop sanity — all 5 venues emit RawBar
    with distinct source_id but identical structural shape."""

    def test_all_five_sources_have_distinct_source_ids(self) -> None:
        ids = {
            CoinbaseCryptoDataSource.SOURCE_ID,
            KrakenCryptoDataSource.SOURCE_ID,
            BitstampCryptoDataSource.SOURCE_ID,
            GeminiCryptoDataSource.SOURCE_ID,
            BinanceUSCryptoDataSource.SOURCE_ID,
        }
        assert len(ids) == 5  # all distinct

    def test_all_five_sources_have_distinct_venue_names(self) -> None:
        names = {
            CoinbaseCryptoDataSource.VENUE_NAME,
            KrakenCryptoDataSource.VENUE_NAME,
            BitstampCryptoDataSource.VENUE_NAME,
            GeminiCryptoDataSource.VENUE_NAME,
            BinanceUSCryptoDataSource.VENUE_NAME,
        }
        assert len(names) == 5

    def test_all_five_sources_share_canonical_btc_usd_mapping(self) -> None:
        """ccxt unifies all venues to canonical 'BTC/USD' — verifier consensus
        works because all 5 sources query the same canonical symbol."""
        for cls in (
            CoinbaseCryptoDataSource,
            KrakenCryptoDataSource,
            BitstampCryptoDataSource,
            GeminiCryptoDataSource,
            BinanceUSCryptoDataSource,
        ):
            assert cls.SYMBOL_MAP["BTCUSD"] == "BTC/USD"

    def test_all_five_sources_emit_distinct_provenance_hashes(self) -> None:
        """Same bar data but different source_id → distinct content_bytes_sha
        (provenance includes source venue). Critical for verifier to detect
        which source each VerifiedSettle/Bar agreement came from."""
        rows = _three_daily_btc_bars()
        sources = [
            CoinbaseCryptoDataSource(
                exchange=FixtureCcxtExchange(
                    id="coinbase",
                    ohlcv_by_key={("BTC/USD", "1d"): rows},
                )
            ),
            KrakenCryptoDataSource(
                exchange=FixtureCcxtExchange(
                    id="kraken",
                    ohlcv_by_key={("BTC/USD", "1d"): rows},
                )
            ),
            BitstampCryptoDataSource(
                exchange=FixtureCcxtExchange(
                    id="bitstamp",
                    ohlcv_by_key={("BTC/USD", "1d"): rows},
                )
            ),
            GeminiCryptoDataSource(
                exchange=FixtureCcxtExchange(
                    id="gemini",
                    ohlcv_by_key={("BTC/USD", "1d"): rows},
                )
            ),
            BinanceUSCryptoDataSource(
                exchange=FixtureCcxtExchange(
                    id="binanceus",
                    ohlcv_by_key={("BTC/USD", "1d"): rows},
                )
            ),
        ]
        first_bar_shas: list[str] = []
        for src in sources:
            bars = list(
                src.get_bars(
                    ContractSymbol("BTCUSD"),
                    datetime(2026, 5, 19, 0, 0, 0, tzinfo=UTC),
                    datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
                    BarResolution.DAY_1,
                )
            )
            first_bar_shas.append(bars[0].content_bytes_sha)
        # All 5 distinct (provenance hash includes venue name)
        assert len(set(first_bar_shas)) == 5
