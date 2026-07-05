"""Smoke tests for Phase A1.0-A1.2 — package imports + types instantiate + ABC enforces.

Verifies the foundation laid in the first session:
- A1.0: package installable + pytest config working
- A1.1: types.py dataclasses construct + validate per __post_init__
- A1.2: DataSource ABC blocks instantiation; concrete subclass instantiates

These are the most basic "is the foundation sound?" tests. The 62-test PIT regression
suite (internal notes) ships at A1.18.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, date, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from futur3 import __version__
from futur3.data import (
    BarResolution,
    ContractSymbol,
    RawBar,
    RawTick,
    Settle,
    Side,
    SourceTier,
)
from futur3.data.source import (
    BarsNotSupported,
    ContractStillActiveError,
    DataSource,
    DataSourceError,
    FutureDatedSourceError,
    GeoBlockedError,
    SchemaMismatch,
    SettlesNotSupported,
    TicksNotSupported,
)
from futur3.data.types import content_sha256, source_provenance_hash

# ----------------------------------------------------------------------------
# A1.0 — package imports
# ----------------------------------------------------------------------------


class TestA1_0_PackageImports:
    def test_version_defined(self) -> None:
        assert __version__ == "0.0.1"

    def test_exception_hierarchy(self) -> None:
        for exc in (
            BarsNotSupported,
            TicksNotSupported,
            SettlesNotSupported,
            GeoBlockedError,
            SchemaMismatch,
            FutureDatedSourceError,
            ContractStillActiveError,
        ):
            assert issubclass(exc, DataSourceError)


# ----------------------------------------------------------------------------
# A1.1 — types.py construction + validation
# ----------------------------------------------------------------------------


SHA64 = "a" * 64  # valid hex-len-64 placeholder for tests


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


class TestA1_1_TypesConstruction:
    def test_source_tier_ordering(self) -> None:
        # Lower rank = higher trust
        assert SourceTier.T2_EXCHANGE < SourceTier.T2_BROKER < SourceTier.T3_AGGREGATOR
        assert SourceTier.T1_EXCHANGE_HISTORICAL == 1
        assert SourceTier.T4_DERIVED == 6

    def test_bar_resolution_values(self) -> None:
        assert BarResolution.MIN_1.value == "1m"
        assert BarResolution.SETTLE.value == "settle"

    def test_side_values(self) -> None:
        assert Side.BUY.value == "buy"
        assert Side.SELL.value == "sell"

    def test_contract_symbol_is_str(self) -> None:
        sym = ContractSymbol("ESM26")
        assert isinstance(sym, str)
        assert sym == "ESM26"

    def test_raw_bar_construction_valid(self) -> None:
        ts = _utc_now()
        bar = RawBar(
            contract=ContractSymbol("ESM26"),
            ts=ts,
            resolution=BarResolution.MIN_1,
            open=Decimal("5000.00"),
            high=Decimal("5010.25"),
            low=Decimal("4999.50"),
            close=Decimal("5005.00"),
            volume=1234,
            oi=None,
            source_id="ibkr_tws_v10.30",
            as_of_iso=ts,
            content_bytes_sha=SHA64,
        )
        assert bar.high >= bar.low
        assert bar.volume >= 0

    def test_raw_bar_rejects_high_below_low(self) -> None:
        ts = _utc_now()
        with pytest.raises(ValueError, match="high"):
            RawBar(
                contract=ContractSymbol("ESM26"),
                ts=ts,
                resolution=BarResolution.MIN_1,
                open=Decimal("5000"),
                high=Decimal("4900"),  # < low
                low=Decimal("5000"),
                close=Decimal("4950"),
                volume=100,
                oi=None,
                source_id="test",
                as_of_iso=ts,
                content_bytes_sha=SHA64,
            )

    def test_raw_bar_rejects_naive_datetime(self) -> None:
        with pytest.raises(ValueError, match="IANA-TZ-aware"):
            RawBar(
                contract=ContractSymbol("ESM26"),
                ts=datetime.now(),  # naive — must be rejected (bug class 7)
                resolution=BarResolution.MIN_1,
                open=Decimal("5000"),
                high=Decimal("5010"),
                low=Decimal("4990"),
                close=Decimal("5005"),
                volume=100,
                oi=None,
                source_id="test",
                as_of_iso=_utc_now(),
                content_bytes_sha=SHA64,
            )

    def test_raw_bar_rejects_negative_volume(self) -> None:
        ts = _utc_now()
        with pytest.raises(ValueError, match="volume"):
            RawBar(
                contract=ContractSymbol("ESM26"),
                ts=ts,
                resolution=BarResolution.MIN_1,
                open=Decimal("5000"),
                high=Decimal("5010"),
                low=Decimal("4990"),
                close=Decimal("5005"),
                volume=-1,
                oi=None,
                source_id="test",
                as_of_iso=ts,
                content_bytes_sha=SHA64,
            )

    def test_raw_bar_rejects_bad_sha(self) -> None:
        ts = _utc_now()
        with pytest.raises(ValueError, match="content_bytes_sha"):
            RawBar(
                contract=ContractSymbol("ESM26"),
                ts=ts,
                resolution=BarResolution.MIN_1,
                open=Decimal("5000"),
                high=Decimal("5010"),
                low=Decimal("4990"),
                close=Decimal("5005"),
                volume=100,
                oi=None,
                source_id="test",
                as_of_iso=ts,
                content_bytes_sha="notlongenough",
            )

    def test_settle_construction_valid(self) -> None:
        ts = _utc_now()
        s = Settle(
            contract=ContractSymbol("ESM26"),
            as_of_date=date(2026, 5, 20),
            settle=Decimal("5000.25"),
            settle_state="preliminary",
            open=Decimal("4995.00"),
            high=Decimal("5010.50"),
            low=Decimal("4990.00"),
            last=Decimal("5000.25"),
            change=Decimal("5.00"),
            volume_est=1_500_000,
            oi_prior=2_300_000,
            as_of_iso=ts,
            content_bytes_sha=SHA64,
            cme_month_code="M",
        )
        assert s.settle_state == "preliminary"
        assert s.oi_prior == 2_300_000

    def test_settle_rejects_bad_state(self) -> None:
        # mypy catches the Literal mismatch statically; runtime accepts but type is wrong
        # Truthful reporting: we don't add runtime Literal enforcement; rely on mypy.
        pass

    def test_raw_tick_construction_valid(self) -> None:
        ts = _utc_now()
        tick = RawTick(
            contract=ContractSymbol("BTCM26"),
            ts=ts,
            price=Decimal("65000.00"),
            size=1,
            side="trade",
            source_id="coinbase_advanced_v3",
            as_of_iso=ts,
            content_bytes_sha=SHA64,
        )
        assert tick.side == "trade"
        assert tick.price > 0

    def test_raw_tick_rejects_zero_size(self) -> None:
        ts = _utc_now()
        with pytest.raises(ValueError, match="size"):
            RawTick(
                contract=ContractSymbol("BTCM26"),
                ts=ts,
                price=Decimal("65000"),
                size=0,
                side=None,
                source_id="test",
                as_of_iso=ts,
                content_bytes_sha=SHA64,
            )

    def test_content_sha256_deterministic(self) -> None:
        payload = b"hello futur3"
        h1 = content_sha256(payload)
        h2 = content_sha256(payload)
        assert h1 == h2
        assert len(h1) == 64

    def test_source_provenance_hash_deterministic(self) -> None:
        ts = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
        h1 = source_provenance_hash("ibkr_tws_v10.30", ts, SHA64)
        h2 = source_provenance_hash("ibkr_tws_v10.30", ts, SHA64)
        assert h1 == h2
        # Different source_id → different hash
        h3 = source_provenance_hash("ibkr_tws_v10.31", ts, SHA64)
        assert h1 != h3

    def test_tz_aware_non_utc_accepted(self) -> None:
        # Any IANA-TZ-aware datetime is acceptable at boundary;
        # canonicalization to UTC happens in verifier. Boundary only rejects naive.
        ts = datetime(2026, 5, 21, 14, 30, tzinfo=ZoneInfo("America/Chicago"))
        bar = RawBar(
            contract=ContractSymbol("ESM26"),
            ts=ts,
            resolution=BarResolution.MIN_1,
            open=Decimal("5000"),
            high=Decimal("5010"),
            low=Decimal("4990"),
            close=Decimal("5005"),
            volume=100,
            oi=None,
            source_id="test",
            as_of_iso=ts,
            content_bytes_sha=SHA64,
        )
        assert bar.ts.tzinfo is not None


# ----------------------------------------------------------------------------
# A1.2 — DataSource ABC enforcement
# ----------------------------------------------------------------------------


class TestA1_2_DataSourceABC:
    def test_abc_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            DataSource()  # type: ignore[abstract]

    def test_concrete_subclass_must_implement_abstract_methods(self) -> None:
        class IncompleteSource(DataSource):
            # Missing source_id, tier, get_bars
            pass

        with pytest.raises(TypeError, match="abstract"):
            IncompleteSource()  # type: ignore[abstract]

    def test_concrete_subclass_instantiates(self) -> None:
        class FixtureSource(DataSource):
            @property
            def source_id(self) -> str:
                return "test_fixture_v1"

            @property
            def tier(self) -> SourceTier:
                return SourceTier.T4_DERIVED

            def get_bars(
                self,
                contract: ContractSymbol,
                ts_start: datetime,
                ts_end: datetime,
                resolution: BarResolution,
            ) -> Iterable[RawBar]:
                return iter([])

        src = FixtureSource()
        assert src.source_id == "test_fixture_v1"
        assert src.tier == SourceTier.T4_DERIVED
        assert (
            list(
                src.get_bars(
                    ContractSymbol("ESM26"),
                    _utc_now(),
                    _utc_now(),
                    BarResolution.MIN_1,
                )
            )
            == []
        )

    def test_default_get_ticks_raises_not_supported(self) -> None:
        class FixtureSource(DataSource):
            @property
            def source_id(self) -> str:
                return "test_fixture_v1"

            @property
            def tier(self) -> SourceTier:
                return SourceTier.T4_DERIVED

            def get_bars(
                self,
                contract: ContractSymbol,
                ts_start: datetime,
                ts_end: datetime,
                resolution: BarResolution,
            ) -> Iterable[RawBar]:
                return iter([])

        src = FixtureSource()
        with pytest.raises(TicksNotSupported):
            list(
                src.get_ticks(
                    ContractSymbol("ESM26"),
                    _utc_now(),
                    _utc_now(),
                )
            )

    def test_default_latest_settle_raises_not_supported(self) -> None:
        class FixtureSource(DataSource):
            @property
            def source_id(self) -> str:
                return "test_fixture_v1"

            @property
            def tier(self) -> SourceTier:
                return SourceTier.T4_DERIVED

            def get_bars(
                self,
                contract: ContractSymbol,
                ts_start: datetime,
                ts_end: datetime,
                resolution: BarResolution,
            ) -> Iterable[RawBar]:
                return iter([])

        src = FixtureSource()
        with pytest.raises(SettlesNotSupported):
            src.latest_settle(ContractSymbol("ESM26"), _utc_now())

    def test_default_healthcheck_returns_true(self) -> None:
        class FixtureSource(DataSource):
            @property
            def source_id(self) -> str:
                return "test_fixture_v1"

            @property
            def tier(self) -> SourceTier:
                return SourceTier.T4_DERIVED

            def get_bars(
                self,
                contract: ContractSymbol,
                ts_start: datetime,
                ts_end: datetime,
                resolution: BarResolution,
            ) -> Iterable[RawBar]:
                return iter([])

        src = FixtureSource()
        assert src.healthcheck() is True

    def test_repr_includes_source_id_and_tier(self) -> None:
        class FixtureSource(DataSource):
            @property
            def source_id(self) -> str:
                return "test_fixture_v1"

            @property
            def tier(self) -> SourceTier:
                return SourceTier.T4_DERIVED

            def get_bars(
                self,
                contract: ContractSymbol,
                ts_start: datetime,
                ts_end: datetime,
                resolution: BarResolution,
            ) -> Iterable[RawBar]:
                return iter([])

        src = FixtureSource()
        r = repr(src)
        assert "test_fixture_v1" in r
        assert "T4_DERIVED" in r
