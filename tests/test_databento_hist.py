"""DatabentoHistoricalDataSource test suite (Ship 6).

Per the DataSource ABC contract:
- maps plain DatabentoBar records -> RawBar (Decimal OHLC preserved, tz-aware, DAY_1, per-bar sha);
- half-open [ts_start, ts_end) range filter + strictly-increasing ts;
- BarsNotSupported (non-daily), ContractNotConfigured (unknown root), FutureDatedSourceError
  (bug class 5), naive-datetime guard;
- continuous-symbol resolution (CLZ26 -> CL.c.0). ZERO network / no databento SDK / no API key.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from futur3.data.source import (
    BarsNotSupported,
    ContractNotConfigured,
    FutureDatedSourceError,
)
from futur3.data.sources.databento_hist import (
    DatabentoBar,
    DatabentoHistoricalDataSource,
)
from futur3.data.types import SHA256_HEX_LENGTH, BarResolution, ContractSymbol, SourceTier

_NOW = datetime(2026, 5, 23, 12, 0, tzinfo=UTC)


class _FixedClock:
    def __init__(self, fixed: datetime) -> None:
        self._fixed = fixed

    def now_utc(self) -> datetime:
        return self._fixed


class _FixtureClient:
    """DatabentoClient backed by an in-memory bar list; captures the last call args."""

    def __init__(self, bars: list[DatabentoBar]) -> None:
        self._bars = bars
        self.last_call: tuple[str, str, str, datetime, datetime] | None = None

    def fetch_ohlcv_1d(
        self, dataset: str, symbol: str, stype_in: str, start: datetime, end: datetime
    ) -> list[DatabentoBar]:
        self.last_call = (dataset, symbol, stype_in, start, end)
        return list(self._bars)


def _bar(day: int, close: str) -> DatabentoBar:
    c = Decimal(close)
    return DatabentoBar(
        ts=datetime(2024, 1, day, tzinfo=UTC),
        open=c,
        high=c + Decimal("0.50"),
        low=c - Decimal("0.50"),
        close=c,
        volume=1000 + day,
    )


def _source(bars: list[DatabentoBar]) -> tuple[DatabentoHistoricalDataSource, _FixtureClient]:
    client = _FixtureClient(bars)
    return DatabentoHistoricalDataSource(client, clock=_FixedClock(_NOW)), client


_CL = ContractSymbol("CLZ26")
_START = datetime(2024, 1, 2, tzinfo=UTC)
_END = datetime(2024, 1, 5, tzinfo=UTC)


class TestGetBars:
    def test_maps_records_to_rawbars(self) -> None:
        src, _ = _source([_bar(2, "70.50"), _bar(3, "71.00"), _bar(4, "70.25")])
        bars = list(src.get_bars(_CL, _START, _END, BarResolution.DAY_1))
        assert len(bars) == 3
        first = bars[0]
        assert first.contract == _CL
        assert first.open == Decimal("70.50")  # Decimal preserved exactly
        assert first.high == Decimal("71.00")
        assert first.low == Decimal("70.00")
        assert first.close == Decimal("70.50")
        assert first.volume == 1002
        assert first.oi is None
        assert first.resolution is BarResolution.DAY_1
        assert first.source_id == "databento_glbx_mdp3"
        assert first.as_of_iso == _NOW

    def test_half_open_range_filter(self) -> None:
        # bar on the end boundary (01-05) is excluded; start boundary (01-02) included.
        src, _ = _source([_bar(2, "70"), _bar(3, "71"), _bar(5, "72")])
        bars = list(src.get_bars(_CL, _START, _END, BarResolution.DAY_1))
        days = [b.ts.day for b in bars]
        assert days == [2, 3]  # 01-05 == ts_end excluded (half-open)

    def test_sorted_chronologically(self) -> None:
        src, _ = _source([_bar(4, "70"), _bar(2, "71"), _bar(3, "72")])  # out of order
        bars = list(src.get_bars(_CL, _START, _END, BarResolution.DAY_1))
        assert [b.ts.day for b in bars] == [2, 3, 4]

    def test_per_bar_sha_distinct_and_hex(self) -> None:
        src, _ = _source([_bar(2, "70"), _bar(3, "71"), _bar(4, "72")])
        shas = [b.content_bytes_sha for b in src.get_bars(_CL, _START, _END, BarResolution.DAY_1)]
        assert len(set(shas)) == 3
        assert all(len(s) == SHA256_HEX_LENGTH for s in shas)

    def test_resolves_continuous_symbol(self) -> None:
        src, client = _source([_bar(2, "70")])
        list(src.get_bars(_CL, _START, _END, BarResolution.DAY_1))
        assert client.last_call is not None
        assert client.last_call[1] == "CL.c.0"  # CLZ26 -> CL continuous front-month
        assert client.last_call[2] == "continuous"


class TestGuards:
    def test_non_daily_resolution_raises(self) -> None:
        src, _ = _source([])
        with pytest.raises(BarsNotSupported, match="DAY_1"):
            list(src.get_bars(_CL, _START, _END, BarResolution.MIN_1))

    def test_unconfigured_root_raises(self) -> None:
        src, _ = _source([_bar(2, "70")])
        with pytest.raises(ContractNotConfigured, match="ZQZ26"):
            list(src.get_bars(ContractSymbol("ZQZ26"), _START, _END, BarResolution.DAY_1))

    def test_future_dated_bar_raises(self) -> None:
        src, _ = _source(
            [
                DatabentoBar(
                    ts=datetime(2099, 1, 1, tzinfo=UTC),
                    open=Decimal("1"),
                    high=Decimal("1"),
                    low=Decimal("1"),
                    close=Decimal("1"),
                    volume=1,
                )
            ]
        )
        with pytest.raises(FutureDatedSourceError, match="2099"):
            list(src.get_bars(_CL, _START, datetime(2100, 1, 1, tzinfo=UTC), BarResolution.DAY_1))

    def test_naive_start_raises(self) -> None:
        src, _ = _source([])
        with pytest.raises(ValueError, match="ts_start"):
            list(src.get_bars(_CL, datetime(2024, 1, 2), _END, BarResolution.DAY_1))

    def test_naive_end_raises(self) -> None:
        src, _ = _source([])
        with pytest.raises(ValueError, match="ts_end"):
            list(src.get_bars(_CL, _START, datetime(2024, 1, 5), BarResolution.DAY_1))


class TestConstruction:
    def test_root_only_symbol_resolves(self) -> None:
        src, client = _source([_bar(2, "70")])
        list(src.get_bars(ContractSymbol("GC"), _START, _END, BarResolution.DAY_1))
        assert client.last_call is not None
        assert client.last_call[1] == "GC.c.0"

    def test_tier_and_source_id(self) -> None:
        src, _ = _source([])
        assert src.tier is SourceTier.T1_EXCHANGE_HISTORICAL
        assert src.source_id == "databento_glbx_mdp3"

    def test_from_api_key_constructs(self) -> None:
        src = DatabentoHistoricalDataSource.from_api_key("db-test-key")
        assert src.source_id == "databento_glbx_mdp3"

    def test_repr(self) -> None:
        assert "databento_glbx_mdp3" in repr(_source([])[0])
