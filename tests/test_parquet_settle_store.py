"""A1.16c ParquetSettleStore test suite (real Parquet round-trips in tmp_path).

Per the `SettleStore` ABC contract:
- Round-trip exactness: Decimal lossless via string (incl. negative `change`), tz-aware
  `as_of_iso` (and the nullable `as_of_iso=None` case), empty `content_bytes_sha`, cme_month_code.
- Idempotent dedupe on (contract, as_of_date, settle_state, content_bytes_sha); preliminary->final
  transitions preserved; retroactive revisions (distinct sha) preserved; counts accurate.
- Inclusive date window (daily cadence); year partition fan-out; multi-contract filter.
- Deterministic ascending (as_of_date, settle_state-rank live<preliminary<final, sha) order;
  end<start raises ValueError; empty/missing -> []; short contract raises StorageError.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from futur3.data.types import ContractSymbol, Settle, SettleState, content_sha256
from futur3.storage import ParquetSettleStore as _PkgParquetSettleStore
from futur3.storage.abcs import SettleStore, StorageError
from futur3.storage.parquet_settle_store import ParquetSettleStore

# Module constant (a name reference, not an inline call) so it can be a default arg without
# tripping ruff B008, while still letting tests pass `as_of_iso=None` explicitly (nullable case).
_DEFAULT_AS_OF_ISO = datetime(2026, 5, 21, 23, 0, tzinfo=UTC)


def _settle(
    *,
    contract: str = "ESM26",
    as_of_date: date | None = None,
    settle: str = "5260.50",
    state: SettleState = "final",
    change: str = "12.25",
    sha: str | None = None,
    as_of_iso: datetime | None = _DEFAULT_AS_OF_ISO,
    oi_prior: int = 2_100_000,
    cme_month_code: str = "M",
) -> Settle:
    as_of_date = as_of_date or date(2026, 5, 21)
    return Settle(
        contract=ContractSymbol(contract),
        as_of_date=as_of_date,
        settle=Decimal(settle),
        settle_state=state,
        open=Decimal("5248.25"),
        high=Decimal("5263.00"),
        low=Decimal("5247.75"),
        last=Decimal("5260.25"),
        change=Decimal(change),
        volume_est=1_350_000,
        oi_prior=oi_prior,
        source_id="cme_public_settlements",
        as_of_iso=as_of_iso,
        content_bytes_sha=sha
        if sha is not None
        else content_sha256(f"{contract}|{as_of_date}|{settle}|{state}".encode()),
        cme_month_code=cme_month_code,
    )


# ============================================================================
# TestA1_16c_Imports / Construction
# ============================================================================


class TestA1_16c_Imports:
    def test_importable(self) -> None:
        assert ParquetSettleStore is not None

    def test_exported_from_storage_package(self) -> None:
        assert _PkgParquetSettleStore is ParquetSettleStore

    def test_is_settle_store(self, tmp_path: Path) -> None:
        assert isinstance(ParquetSettleStore(tmp_path), SettleStore)


class TestA1_16c_Construction:
    def test_backend_id(self, tmp_path: Path) -> None:
        assert ParquetSettleStore(tmp_path).backend_id == "parquet_local"

    def test_healthcheck_creates_dir(self, tmp_path: Path) -> None:
        base = tmp_path / "store"
        assert ParquetSettleStore(base).healthcheck() is True
        assert base.is_dir()


# ============================================================================
# TestA1_16c_RoundTrip
# ============================================================================


class TestA1_16c_RoundTrip:
    def _read_one(self, store: ParquetSettleStore, contract: str = "ESM26") -> Settle:
        out = list(
            store.read_settles(ContractSymbol(contract), date(2026, 1, 1), date(2026, 12, 31))
        )
        assert len(out) == 1
        return out[0]

    def test_write_then_read_exact(self, tmp_path: Path) -> None:
        store = ParquetSettleStore(tmp_path)
        s = _settle(settle="5260.50", change="12.25", oi_prior=2_100_000)
        result = store.write_settles([s])
        assert result.rows_written == 1
        assert result.dedupe_count == 0

        got = self._read_one(store)
        assert got.contract == ContractSymbol("ESM26")
        assert got.as_of_date == date(2026, 5, 21)
        assert got.settle == Decimal("5260.50")
        assert got.settle_state == "final"
        assert got.open == Decimal("5248.25")
        assert got.high == Decimal("5263.00")
        assert got.low == Decimal("5247.75")
        assert got.last == Decimal("5260.25")
        assert got.change == Decimal("12.25")
        assert got.volume_est == 1_350_000
        assert got.oi_prior == 2_100_000
        assert got.source_id == "cme_public_settlements"
        assert got.cme_month_code == "M"
        assert got.content_bytes_sha == s.content_bytes_sha

    def test_negative_change_round_trips(self, tmp_path: Path) -> None:
        store = ParquetSettleStore(tmp_path)
        store.write_settles([_settle(change="-37.75")])
        assert self._read_one(store).change == Decimal("-37.75")

    def test_high_precision_decimal_exact(self, tmp_path: Path) -> None:
        store = ParquetSettleStore(tmp_path)
        store.write_settles([_settle(settle="5260.123456789")])
        assert self._read_one(store).settle == Decimal("5260.123456789")

    def test_as_of_iso_tz_aware_round_trips(self, tmp_path: Path) -> None:
        store = ParquetSettleStore(tmp_path)
        s = _settle(as_of_iso=datetime(2026, 5, 21, 23, 0, tzinfo=UTC))
        store.write_settles([s])
        got = self._read_one(store)
        assert got.as_of_iso is not None
        assert got.as_of_iso.tzinfo is not None  # never naive (bug class 7)
        assert got.as_of_iso == s.as_of_iso

    def test_as_of_iso_none_round_trips(self, tmp_path: Path) -> None:
        store = ParquetSettleStore(tmp_path)
        store.write_settles([_settle(as_of_iso=None)])
        assert self._read_one(store).as_of_iso is None

    def test_all_none_as_of_iso_batch_does_not_break_schema(self, tmp_path: Path) -> None:
        # The bars `oi` lesson: an all-None nullable column must keep its typed dtype.
        store = ParquetSettleStore(tmp_path)
        store.write_settles(
            [
                _settle(as_of_date=date(2026, 5, 20), as_of_iso=None, sha=content_sha256(b"n1")),
                _settle(as_of_date=date(2026, 5, 21), as_of_iso=None, sha=content_sha256(b"n2")),
            ]
        )
        # a later write WITH a real timestamp must concat cleanly onto the all-null column
        store.write_settles([_settle(as_of_date=date(2026, 5, 22), sha=content_sha256(b"t1"))])
        out = list(
            store.read_settles(ContractSymbol("ESM26"), date(2026, 5, 20), date(2026, 5, 22))
        )
        assert len(out) == 3
        assert [s.as_of_iso is None for s in out] == [True, True, False]

    def test_empty_content_sha_round_trips(self, tmp_path: Path) -> None:
        store = ParquetSettleStore(tmp_path)
        store.write_settles([_settle(sha="")])  # dataclass allows empty sha
        assert self._read_one(store).content_bytes_sha == ""


# ============================================================================
# TestA1_16c_Dedupe / Revision / PreliminaryFinal
# ============================================================================


class TestA1_16c_Dedupe:
    def _read_all(self, store: ParquetSettleStore) -> list[Settle]:
        return list(
            store.read_settles(ContractSymbol("ESM26"), date(2026, 1, 1), date(2026, 12, 31))
        )

    def test_rewrite_same_is_idempotent(self, tmp_path: Path) -> None:
        store = ParquetSettleStore(tmp_path)
        s = _settle()
        first = store.write_settles([s])
        assert first.rows_written == 1 and first.dedupe_count == 0
        second = store.write_settles([s])
        assert second.rows_written == 0
        assert second.dedupe_count == 1
        assert len(self._read_all(store)) == 1

    def test_internal_batch_dedupe(self, tmp_path: Path) -> None:
        store = ParquetSettleStore(tmp_path)
        s = _settle()
        result = store.write_settles([s, s, s])
        assert result.rows_written == 1
        assert result.dedupe_count == 2

    def test_revision_distinct_sha_both_kept(self, tmp_path: Path) -> None:
        store = ParquetSettleStore(tmp_path)
        d = date(2026, 5, 21)
        v1 = _settle(as_of_date=d, settle="5260.50", state="final", sha=content_sha256(b"v1"))
        v2 = _settle(as_of_date=d, settle="5260.75", state="final", sha=content_sha256(b"v2"))
        store.write_settles([v1])
        store.write_settles([v2])
        out = self._read_all(store)
        assert len(out) == 2  # same (contract, date, state) but distinct sha -> both kept
        assert {s.settle for s in out} == {Decimal("5260.50"), Decimal("5260.75")}

    def test_preliminary_then_final_both_kept(self, tmp_path: Path) -> None:
        store = ParquetSettleStore(tmp_path)
        d = date(2026, 5, 21)
        prelim = _settle(
            as_of_date=d, settle="5260.50", state="preliminary", sha=content_sha256(b"p")
        )
        final = _settle(as_of_date=d, settle="5260.75", state="final", sha=content_sha256(b"f"))
        store.write_settles([prelim, final])
        out = self._read_all(store)
        assert len(out) == 2
        assert [s.settle_state for s in out] == ["preliminary", "final"]  # rank order


# ============================================================================
# TestA1_16c_Window
# ============================================================================


class TestA1_16c_Window:
    def _three_days(self, store: ParquetSettleStore) -> None:
        store.write_settles(
            [
                _settle(as_of_date=date(2026, 5, 20), settle="1", sha=content_sha256(b"a")),
                _settle(as_of_date=date(2026, 5, 21), settle="2", sha=content_sha256(b"b")),
                _settle(as_of_date=date(2026, 5, 22), settle="3", sha=content_sha256(b"c")),
            ]
        )

    def test_inclusive_date_window(self, tmp_path: Path) -> None:
        store = ParquetSettleStore(tmp_path)
        self._three_days(store)
        out = list(
            store.read_settles(ContractSymbol("ESM26"), date(2026, 5, 20), date(2026, 5, 22))
        )
        assert [s.settle for s in out] == [Decimal("1"), Decimal("2"), Decimal("3")]

    def test_window_excludes_outside(self, tmp_path: Path) -> None:
        store = ParquetSettleStore(tmp_path)
        self._three_days(store)
        out = list(
            store.read_settles(ContractSymbol("ESM26"), date(2026, 5, 21), date(2026, 5, 21))
        )
        assert [s.settle for s in out] == [Decimal("2")]  # single-day inclusive window

    def test_ascending_order_written_out_of_order(self, tmp_path: Path) -> None:
        store = ParquetSettleStore(tmp_path)
        store.write_settles(
            [
                _settle(as_of_date=date(2026, 5, 22), settle="3", sha=content_sha256(b"c")),
                _settle(as_of_date=date(2026, 5, 20), settle="1", sha=content_sha256(b"a")),
                _settle(as_of_date=date(2026, 5, 21), settle="2", sha=content_sha256(b"b")),
            ]
        )
        out = list(
            store.read_settles(ContractSymbol("ESM26"), date(2026, 5, 20), date(2026, 5, 22))
        )
        assert [s.settle for s in out] == [Decimal("1"), Decimal("2"), Decimal("3")]


# ============================================================================
# TestA1_16c_Partitioning / MultiContract
# ============================================================================


class TestA1_16c_Partitioning:
    def test_spans_years(self, tmp_path: Path) -> None:
        store = ParquetSettleStore(tmp_path)
        store.write_settles(
            [
                _settle(as_of_date=date(2025, 12, 31), settle="1", sha=content_sha256(b"a")),
                _settle(as_of_date=date(2026, 6, 30), settle="2", sha=content_sha256(b"b")),
                _settle(as_of_date=date(2027, 1, 5), settle="3", sha=content_sha256(b"c")),
            ]
        )
        base = tmp_path / "futures" / "settles" / "ES"
        assert (base / "year=2025" / "data.parquet").exists()
        assert (base / "year=2026" / "data.parquet").exists()
        assert (base / "year=2027" / "data.parquet").exists()
        out = list(
            store.read_settles(ContractSymbol("ESM26"), date(2025, 1, 1), date(2027, 12, 31))
        )
        assert [s.settle for s in out] == [Decimal("1"), Decimal("2"), Decimal("3")]

    def test_multi_contract_same_root_filtered(self, tmp_path: Path) -> None:
        store = ParquetSettleStore(tmp_path)
        d = date(2026, 5, 21)
        store.write_settles(
            [
                _settle(contract="ESM26", as_of_date=d, settle="5260", sha=content_sha256(b"m")),
                _settle(contract="ESU26", as_of_date=d, settle="5280", sha=content_sha256(b"u")),
            ]
        )
        out = list(store.read_settles(ContractSymbol("ESM26"), d, d))
        assert len(out) == 1
        assert out[0].contract == ContractSymbol("ESM26")
        assert out[0].settle == Decimal("5260")


# ============================================================================
# TestA1_16c_EdgeCases
# ============================================================================


class TestA1_16c_EdgeCases:
    def test_empty_write_zero_result(self, tmp_path: Path) -> None:
        result = ParquetSettleStore(tmp_path).write_settles([])
        assert result.rows_written == 0
        assert result.dedupe_count == 0

    def test_read_missing_partition_empty(self, tmp_path: Path) -> None:
        out = list(
            ParquetSettleStore(tmp_path).read_settles(
                ContractSymbol("NQM26"), date(2026, 1, 1), date(2026, 12, 31)
            )
        )
        assert out == []

    def test_end_before_start_raises(self, tmp_path: Path) -> None:
        store = ParquetSettleStore(tmp_path)
        with pytest.raises(ValueError, match=r"as_of_date_end .* < as_of_date_start"):
            list(store.read_settles(ContractSymbol("ESM26"), date(2026, 5, 21), date(2026, 5, 20)))

    def test_short_contract_root_raises(self, tmp_path: Path) -> None:
        store = ParquetSettleStore(tmp_path)
        with pytest.raises(StorageError, match="too short to parse root"):
            store.write_settles([_settle(contract="ES")])


# ============================================================================
# TestA1_16c_Determinism
# ============================================================================


class TestA1_16c_Determinism:
    def test_reread_identical(self, tmp_path: Path) -> None:
        store = ParquetSettleStore(tmp_path)
        store.write_settles(
            [
                _settle(as_of_date=date(2026, 5, 21), settle="2", sha=content_sha256(b"b")),
                _settle(as_of_date=date(2026, 5, 20), settle="1", sha=content_sha256(b"a")),
            ]
        )
        first = [
            (s.as_of_date, s.settle)
            for s in store.read_settles(
                ContractSymbol("ESM26"), date(2026, 5, 20), date(2026, 5, 21)
            )
        ]
        second = [
            (s.as_of_date, s.settle)
            for s in store.read_settles(
                ContractSymbol("ESM26"), date(2026, 5, 20), date(2026, 5, 21)
            )
        ]
        assert first == second
        assert first == [
            (date(2026, 5, 20), Decimal("1")),
            (date(2026, 5, 21), Decimal("2")),
        ]

    def test_settle_state_rank_order_on_same_day(self, tmp_path: Path) -> None:
        # live < preliminary < final, regardless of write order, on the same as_of_date
        store = ParquetSettleStore(tmp_path)
        d = date(2026, 5, 21)
        store.write_settles(
            [
                _settle(as_of_date=d, state="final", settle="3", sha=content_sha256(b"f")),
                _settle(as_of_date=d, state="live", settle="1", sha=content_sha256(b"l")),
                _settle(as_of_date=d, state="preliminary", settle="2", sha=content_sha256(b"p")),
            ]
        )
        out = list(store.read_settles(ContractSymbol("ESM26"), d, d))
        assert [s.settle_state for s in out] == ["live", "preliminary", "final"]
