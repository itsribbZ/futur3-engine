"""A1.11 BRRReconstructor test suite — 4-of-7 reconstruction coverage.

Test discipline:
- ALL tests fixture-only (zero live network, zero ccxt instantiation).
- Boundary-invariant tests (DST window math, Decimal precision, hash determinism).
- Graceful-degradation tests (venue dropout per partition).
- Error-path coverage (unsupported contract, all partitions degraded).

References:
- `futur3/data/brr.py` (implementation)
- internal crypto-data notes (spec source of truth)
- internal crypto-data notes (algorithm)
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import ClassVar

import pytest

from futur3.data.brr import (
    BRRError,
    BRRReconstruction,
    BRRReconstructor,
    BRRWindowEmpty,
    InsufficientVenues,
)
from futur3.data.source import (
    BarsNotSupported,
    DataSource,
    DataSourceError,
    SettlesNotSupported,
    TicksNotSupported,
)
from futur3.data.types import (
    BarResolution,
    ContractSymbol,
    RawBar,
    RawTick,
    Settle,
    SourceTier,
)

# ============================================================================
# Mock DataSource — emits canned RawBars in a window
# ============================================================================


@dataclass
class MockBarSource(DataSource):
    """Returns the configured `bars_to_return` filtered by the requested window.

    Used to inject deterministic OHLCV streams into BRRReconstructor without
    spinning up ccxt or hitting the network.
    """

    SOURCE_ID: ClassVar[str] = "mock_bar"

    fixed_source_id: str = "mock_bar"
    fixed_tier: SourceTier = SourceTier.T2_EXCHANGE
    bars_to_return: list[RawBar] = field(default_factory=list)
    raise_bars_not_supported: bool = False
    raise_data_source_error: bool = False

    @property
    def source_id(self) -> str:
        return self.fixed_source_id

    @property
    def tier(self) -> SourceTier:
        return self.fixed_tier

    def get_bars(
        self,
        contract: ContractSymbol,
        ts_start: datetime,
        ts_end: datetime,
        resolution: BarResolution,
    ) -> Iterable[RawBar]:
        if self.raise_bars_not_supported:
            raise BarsNotSupported(f"{self.source_id}: configured raise")
        if self.raise_data_source_error:
            raise DataSourceError(f"{self.source_id}: configured raise")
        return [
            b
            for b in self.bars_to_return
            if ts_start <= b.ts < ts_end and b.resolution == resolution and b.contract == contract
        ]

    def get_ticks(
        self,
        contract: ContractSymbol,
        ts_start: datetime,
        ts_end: datetime,
    ) -> Iterable[RawTick]:
        raise TicksNotSupported(f"{self.source_id}: mock")

    def latest_settle(
        self,
        contract: ContractSymbol,
        as_of: datetime,
    ) -> Settle | None:
        raise SettlesNotSupported(f"{self.source_id}: mock")


# ============================================================================
# Helpers
# ============================================================================


def _sha(seed: str) -> str:
    """Deterministic 64-char hex sha for fixture bars."""
    return (seed.encode().hex() * 16)[:64]


def _make_bar(
    *,
    ts: datetime,
    close: str,
    source_id: str,
    contract: str = "BTCUSD",
    resolution: BarResolution = BarResolution.MIN_5,
    volume: int = 100,
) -> RawBar:
    return RawBar(
        contract=ContractSymbol(contract),
        ts=ts,
        resolution=resolution,
        open=Decimal(close),
        high=Decimal(close),
        low=Decimal(close),
        close=Decimal(close),
        volume=volume,
        oi=None,
        source_id=source_id,
        as_of_iso=datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        content_bytes_sha=_sha(f"{source_id}_{ts.isoformat()}_{close}"),
    )


def _make_full_window_bars(
    *,
    source_id: str,
    ts_start_utc: datetime,
    base_price: Decimal,
    contract: str = "BTCUSD",
) -> list[RawBar]:
    """12 x 5-min bars spanning a 1hr window starting at `ts_start_utc`.

    Each bar's close is `base_price + i * 0.01` so partitions have distinct values.
    """
    bars = []
    for i in range(12):
        ts = ts_start_utc + timedelta(minutes=i * 5)
        close = base_price + Decimal(i) * Decimal("0.01")
        bars.append(
            _make_bar(
                ts=ts,
                close=str(close),
                source_id=source_id,
                contract=contract,
            )
        )
    return bars


def _4_venue_setup_summer(
    base_price: Decimal = Decimal("60000.00"),
    contract: str = "BTCUSD",
) -> list[MockBarSource]:
    """Four mock venues with 12 x 5-min bars across the BST 14:00-15:00 UTC window.

    fix_date = 2026-05-21 (BST in effect; London 15:00 = UTC 14:00).
    """
    ts_start_utc = datetime(2026, 5, 21, 14, 0, 0, tzinfo=UTC)
    venues = [
        MockBarSource(
            fixed_source_id=f"venue_{name}",
            bars_to_return=_make_full_window_bars(
                source_id=f"venue_{name}",
                ts_start_utc=ts_start_utc,
                base_price=base_price + offset,
                contract=contract,
            ),
        )
        for name, offset in [
            ("coinbase", Decimal("0.00")),
            ("kraken", Decimal("0.10")),
            ("bitstamp", Decimal("0.20")),
            ("gemini", Decimal("0.30")),
        ]
    ]
    return venues


# ============================================================================
# TestA1_11_Imports
# ============================================================================


class TestA1_11_Imports:
    def test_brr_reconstructor_importable(self) -> None:
        assert BRRReconstructor is not None

    def test_brr_reconstruction_dataclass_importable(self) -> None:
        assert BRRReconstruction is not None

    def test_error_hierarchy(self) -> None:
        # BRRError inherits from DataSourceError so existing catch-blocks work
        assert issubclass(BRRError, DataSourceError)
        assert issubclass(BRRWindowEmpty, BRRError)
        assert issubclass(InsufficientVenues, BRRError)


# ============================================================================
# TestA1_11_Construction
# ============================================================================


class TestA1_11_Construction:
    def test_construct_with_4_sources_ok(self) -> None:
        venues = _4_venue_setup_summer()
        r = BRRReconstructor(venues)
        assert len(r.venue_ids) == 4

    def test_construct_with_2_sources_minimum_ok(self) -> None:
        venues = _4_venue_setup_summer()[:2]
        r = BRRReconstructor(venues)
        assert len(r.venue_ids) == 2

    def test_construct_with_1_source_raises_insufficient_venues(self) -> None:
        venues = _4_venue_setup_summer()[:1]
        with pytest.raises(InsufficientVenues, match="at least 2 sources"):
            BRRReconstructor(venues)

    def test_construct_with_0_sources_raises(self) -> None:
        with pytest.raises(InsufficientVenues):
            BRRReconstructor([])

    def test_venue_ids_sorted_for_determinism(self) -> None:
        # Pass in non-alphabetical order; expect sorted result
        venues = _4_venue_setup_summer()
        scrambled = [venues[3], venues[1], venues[0], venues[2]]
        r = BRRReconstructor(scrambled)
        assert r.venue_ids == (
            "venue_bitstamp",
            "venue_coinbase",
            "venue_gemini",
            "venue_kraken",
        )

    def test_repr_lists_venue_ids(self) -> None:
        venues = _4_venue_setup_summer()
        r = BRRReconstructor(venues)
        rep = repr(r)
        assert "venue_coinbase" in rep
        assert "venue_kraken" in rep


# ============================================================================
# TestA1_11_LondonWindow — DST-aware window math
# ============================================================================


class TestA1_11_LondonWindow:
    def test_summer_bst_window_is_14_15_utc(self) -> None:
        """BST: London 15:00 = UTC 14:00; 16:00 = 15:00."""
        # 2026-05-21 is well into BST (BST runs late-March → late-October).
        start, end = BRRReconstructor._london_window_utc(date(2026, 5, 21))
        assert start == datetime(2026, 5, 21, 14, 0, 0, tzinfo=UTC)
        assert end == datetime(2026, 5, 21, 15, 0, 0, tzinfo=UTC)

    def test_winter_gmt_window_is_15_16_utc(self) -> None:
        """GMT: London 15:00 = UTC 15:00; 16:00 = 16:00."""
        # 2026-01-15 is in GMT (BST starts late March).
        start, end = BRRReconstructor._london_window_utc(date(2026, 1, 15))
        assert start == datetime(2026, 1, 15, 15, 0, 0, tzinfo=UTC)
        assert end == datetime(2026, 1, 15, 16, 0, 0, tzinfo=UTC)

    def test_window_is_exactly_one_hour(self) -> None:
        start, end = BRRReconstructor._london_window_utc(date(2026, 5, 21))
        assert end - start == timedelta(hours=1)

    def test_dst_spring_forward_day_window_still_one_hour(self) -> None:
        """2026 BST starts 29 March (last Sun). 15:00-16:00 still 1hr."""
        start, end = BRRReconstructor._london_window_utc(date(2026, 3, 29))
        assert end - start == timedelta(hours=1)

    def test_dst_fall_back_day_window_still_one_hour(self) -> None:
        """2026 BST ends 25 October (last Sun). 15:00-16:00 still 1hr."""
        start, end = BRRReconstructor._london_window_utc(date(2026, 10, 25))
        assert end - start == timedelta(hours=1)


# ============================================================================
# TestA1_11_HappyPath
# ============================================================================


class TestA1_11_HappyPath:
    def test_4_venues_all_responding_full_window(self) -> None:
        """4 venues x 12 partitions all complete -> all partitions valid."""
        venues = _4_venue_setup_summer()
        r = BRRReconstructor(venues)
        result = r.reconstruct(ContractSymbol("BTCUSD"), date(2026, 5, 21))

        assert isinstance(result, BRRReconstruction)
        assert result.contract == "BTCUSD"
        assert result.fix_date == date(2026, 5, 21)
        assert result.num_partitions_valid == 12
        assert all(v is not None for v in result.partition_values)
        assert len(result.venue_ids) == 4
        assert len(result.content_bytes_sha) == 64

    def test_reconstructed_value_within_venue_price_band(self) -> None:
        """BRR ≈ arithmetic mean across venues + partitions; must land in input band."""
        venues = _4_venue_setup_summer(base_price=Decimal("60000.00"))
        r = BRRReconstructor(venues)
        result = r.reconstruct(ContractSymbol("BTCUSD"), date(2026, 5, 21))
        # Venues span 60000.00 - 60000.30 base + 0-0.11 partition spread →
        # max ≈ 60000.41, min ≈ 60000.00. BRR must lie in that band.
        assert result.reconstructed_value > Decimal("60000.00")
        assert result.reconstructed_value < Decimal("60000.50")

    def test_ethusd_supported_contract(self) -> None:
        venues = _4_venue_setup_summer(
            base_price=Decimal("3000.00"),
            contract="ETHUSD",
        )
        r = BRRReconstructor(venues)
        result = r.reconstruct(ContractSymbol("ETHUSD"), date(2026, 5, 21))
        assert result.contract == "ETHUSD"
        assert result.num_partitions_valid == 12

    def test_partition_values_are_per_partition_means(self) -> None:
        """Exact-math check: partition 0 mean equals known input."""
        # 4 venues at 60000.00 / 60000.10 / 60000.20 / 60000.30 in partition 0.
        # Mean = (60000.00 + 60000.10 + 60000.20 + 60000.30) / 4 = 60000.15
        venues = _4_venue_setup_summer(base_price=Decimal("60000.00"))
        r = BRRReconstructor(venues)
        result = r.reconstruct(ContractSymbol("BTCUSD"), date(2026, 5, 21))
        assert result.partition_values[0] == Decimal("60000.15")

    def test_final_brr_is_mean_of_partition_means(self) -> None:
        """Each partition mean is base + i*0.01 + 0.15 (the venue-spread offset).
        Final BRR = mean of (60000.15 + 60000.16 + ... + 60000.26) = 60000.205.
        """
        venues = _4_venue_setup_summer(base_price=Decimal("60000.00"))
        r = BRRReconstructor(venues)
        result = r.reconstruct(ContractSymbol("BTCUSD"), date(2026, 5, 21))
        # Manual compute: sum from i=0..11 of (60000.15 + i*0.01) / 12.
        # = 60000.15 + (0+1+...+11)/12 * 0.01 = 60000.15 + 66/12 * 0.01
        # = 60000.15 + 0.055 = 60000.205
        assert result.reconstructed_value == Decimal("60000.205")


# ============================================================================
# TestA1_11_GracefulDegradation
# ============================================================================


class TestA1_11_GracefulDegradation:
    def test_one_venue_missing_one_partition_other_three_carry(self) -> None:
        """One venue missing partition i: other 3 venues still aggregate fine."""
        venues = _4_venue_setup_summer()
        # Drop the 5th bar (index 4 = partition 4) from venue_coinbase.
        cb = venues[0]
        cb.bars_to_return = [b for j, b in enumerate(cb.bars_to_return) if j != 4]

        r = BRRReconstructor(venues)
        result = r.reconstruct(ContractSymbol("BTCUSD"), date(2026, 5, 21))
        # All 12 partitions should still be valid (3 venues per partition >= 2).
        assert result.num_partitions_valid == 12

    def test_all_venues_missing_one_partition_marks_partition_none(self) -> None:
        """If ALL venues lack partition i, partition_values[i] is None."""
        venues = _4_venue_setup_summer()
        for v in venues:
            # Drop bar at partition index 6 from each venue
            v.bars_to_return = [b for j, b in enumerate(v.bars_to_return) if j != 6]

        r = BRRReconstructor(venues)
        result = r.reconstruct(ContractSymbol("BTCUSD"), date(2026, 5, 21))
        assert result.num_partitions_valid == 11
        assert result.partition_values[6] is None
        # Other partitions still valid
        for i, v in enumerate(result.partition_values):
            if i != 6:
                assert v is not None

    def test_two_venues_missing_one_partition_marks_partition_none(self) -> None:
        """Below-quorum (< MIN_VENUES_PER_PARTITION=2) → partition None."""
        venues = _4_venue_setup_summer()
        # Drop bar 3 from venues 0, 1, 2 → only venue 3 has it (N=1 < 2)
        for v in venues[:3]:
            v.bars_to_return = [b for j, b in enumerate(v.bars_to_return) if j != 3]
        r = BRRReconstructor(venues)
        result = r.reconstruct(ContractSymbol("BTCUSD"), date(2026, 5, 21))
        assert result.partition_values[3] is None
        assert result.num_partitions_valid == 11

    def test_two_venue_minimum_setup_succeeds(self) -> None:
        """N=2 venues both with full windows → all 12 partitions valid."""
        venues = _4_venue_setup_summer()[:2]
        r = BRRReconstructor(venues)
        result = r.reconstruct(ContractSymbol("BTCUSD"), date(2026, 5, 21))
        assert result.num_partitions_valid == 12


# ============================================================================
# TestA1_11_Determinism — hash chain
# ============================================================================


class TestA1_11_Determinism:
    def test_same_inputs_same_content_sha(self) -> None:
        venues_a = _4_venue_setup_summer()
        venues_b = _4_venue_setup_summer()  # rebuilt identical fixtures
        r_a = BRRReconstructor(venues_a)
        r_b = BRRReconstructor(venues_b)
        result_a = r_a.reconstruct(ContractSymbol("BTCUSD"), date(2026, 5, 21))
        result_b = r_b.reconstruct(ContractSymbol("BTCUSD"), date(2026, 5, 21))
        assert result_a.content_bytes_sha == result_b.content_bytes_sha
        assert result_a.reconstructed_value == result_b.reconstructed_value

    def test_source_order_independence(self) -> None:
        """Scramble source order at construction → same content_sha."""
        venues_a = _4_venue_setup_summer()
        venues_b = list(reversed(_4_venue_setup_summer()))
        r_a = BRRReconstructor(venues_a)
        r_b = BRRReconstructor(venues_b)
        result_a = r_a.reconstruct(ContractSymbol("BTCUSD"), date(2026, 5, 21))
        result_b = r_b.reconstruct(ContractSymbol("BTCUSD"), date(2026, 5, 21))
        assert result_a.content_bytes_sha == result_b.content_bytes_sha

    def test_different_dates_different_content_sha(self) -> None:
        venues = _4_venue_setup_summer()
        r = BRRReconstructor(venues)
        result_may21 = r.reconstruct(ContractSymbol("BTCUSD"), date(2026, 5, 21))
        # New venues with bars for a different date so windows don't overlap
        ts_other = datetime(2026, 5, 22, 14, 0, 0, tzinfo=UTC)
        other_venues = [
            MockBarSource(
                fixed_source_id=v.fixed_source_id,
                bars_to_return=_make_full_window_bars(
                    source_id=v.fixed_source_id,
                    ts_start_utc=ts_other,
                    base_price=Decimal("60000.00"),
                ),
            )
            for v in venues
        ]
        r2 = BRRReconstructor(other_venues)
        result_may22 = r2.reconstruct(ContractSymbol("BTCUSD"), date(2026, 5, 22))
        assert result_may21.content_bytes_sha != result_may22.content_bytes_sha

    def test_content_sha_is_hex_sha256(self) -> None:
        venues = _4_venue_setup_summer()
        r = BRRReconstructor(venues)
        result = r.reconstruct(ContractSymbol("BTCUSD"), date(2026, 5, 21))
        assert len(result.content_bytes_sha) == 64
        assert all(c in "0123456789abcdef" for c in result.content_bytes_sha)


# ============================================================================
# TestA1_11_Errors
# ============================================================================


class TestA1_11_Errors:
    def test_unsupported_contract_raises(self) -> None:
        venues = _4_venue_setup_summer()
        r = BRRReconstructor(venues)
        with pytest.raises(DataSourceError, match="not supported"):
            r.reconstruct(ContractSymbol("ESM26"), date(2026, 5, 21))

    def test_empty_window_raises_brr_window_empty(self) -> None:
        """No bars at all → all 12 partitions degraded → BRRWindowEmpty."""
        # Empty bar sets across all venues
        venues = [
            MockBarSource(fixed_source_id="venue_a", bars_to_return=[]),
            MockBarSource(fixed_source_id="venue_b", bars_to_return=[]),
        ]
        r = BRRReconstructor(venues)
        with pytest.raises(BRRWindowEmpty, match="0 of 12 partitions"):
            r.reconstruct(ContractSymbol("BTCUSD"), date(2026, 5, 21))

    def test_data_source_error_propagates(self) -> None:
        """A venue raising DataSourceError mid-reconstruct propagates (no swallow)."""
        venues = _4_venue_setup_summer()
        venues[0].raise_data_source_error = True
        r = BRRReconstructor(venues)
        with pytest.raises(DataSourceError, match="configured raise"):
            r.reconstruct(ContractSymbol("BTCUSD"), date(2026, 5, 21))


# ============================================================================
# TestA1_11_DataclassValidation
# ============================================================================


class TestA1_11_DataclassValidation:
    def _valid_kwargs(self) -> dict[str, object]:
        return {
            "contract": ContractSymbol("BTCUSD"),
            "fix_date": date(2026, 5, 21),
            "reconstructed_value": Decimal("60000.20"),
            "partition_values": tuple(Decimal("60000.00") for _ in range(12)),
            "num_partitions_valid": 12,
            "venue_ids": ("venue_a", "venue_b"),
            "content_bytes_sha": "a" * 64,
        }

    def test_valid_construction(self) -> None:
        BRRReconstruction(**self._valid_kwargs())  # type: ignore[arg-type]

    def test_wrong_partition_count_raises(self) -> None:
        kw = self._valid_kwargs()
        kw["partition_values"] = tuple(Decimal("60000.00") for _ in range(10))
        with pytest.raises(ValueError, match="must have 12 entries"):
            BRRReconstruction(**kw)  # type: ignore[arg-type]

    def test_num_partitions_valid_below_1_raises(self) -> None:
        kw = self._valid_kwargs()
        kw["num_partitions_valid"] = 0
        with pytest.raises(ValueError, match="num_partitions_valid must be"):
            BRRReconstruction(**kw)  # type: ignore[arg-type]

    def test_num_partitions_valid_above_12_raises(self) -> None:
        kw = self._valid_kwargs()
        kw["num_partitions_valid"] = 13
        with pytest.raises(ValueError, match="num_partitions_valid must be"):
            BRRReconstruction(**kw)  # type: ignore[arg-type]

    def test_reconstructed_value_zero_raises(self) -> None:
        kw = self._valid_kwargs()
        kw["reconstructed_value"] = Decimal("0")
        with pytest.raises(ValueError, match="must be > 0"):
            BRRReconstruction(**kw)  # type: ignore[arg-type]

    def test_reconstructed_value_negative_raises(self) -> None:
        kw = self._valid_kwargs()
        kw["reconstructed_value"] = Decimal("-1")
        with pytest.raises(ValueError, match="must be > 0"):
            BRRReconstruction(**kw)  # type: ignore[arg-type]

    def test_bad_sha_length_raises(self) -> None:
        kw = self._valid_kwargs()
        kw["content_bytes_sha"] = "abc"
        with pytest.raises(ValueError, match="hex-SHA256"):
            BRRReconstruction(**kw)  # type: ignore[arg-type]

    def test_too_few_venue_ids_raises(self) -> None:
        kw = self._valid_kwargs()
        kw["venue_ids"] = ("venue_a",)
        with pytest.raises(ValueError, match="at least 2"):
            BRRReconstruction(**kw)  # type: ignore[arg-type]

    def test_frozen_dataclass_immutable(self) -> None:
        rec = BRRReconstruction(**self._valid_kwargs())  # type: ignore[arg-type]
        with pytest.raises(AttributeError):
            rec.reconstructed_value = Decimal("99999.99")  # type: ignore[misc]
