"""PaperLedger test suite (futur3.execution.paper_ledger) - the forward-paper P&L accumulator.

Locks: append+read round-trip (Decimal/date/None survive JSONL), record() cumulative
accumulation, the chronological guard (a same-day / out-of-order double-run raises, never
double-counts), the empty ledger, a flat (non-qualifying) session, and the derived pnl_series /
cumulative_pnl accessors.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from futur3.execution.paper_ledger import PaperLedger, PaperLedgerError, PaperSession


def _ledger(tmp_path: Path) -> PaperLedger:
    return PaperLedger(tmp_path / "paper" / "nq_overnight.jsonl")


class TestRoundTrip:
    def test_append_read_preserves_fields(self, tmp_path: Path) -> None:
        led = _ledger(tmp_path)
        s = PaperSession(
            session_date=date(2026, 1, 5),
            qualified=True,
            contracts=3,
            entry_price=Decimal("21000.25"),
            exit_price=Decimal("21010.50"),
            session_pnl=Decimal("615.00"),
            cost=Decimal("1.20"),
            cumulative_pnl=Decimal("615.00"),
        )
        led.append(s)
        back = led.read()
        assert back == [s]  # frozen dataclass equality: every field (incl. Decimals) survived

    def test_flat_session_none_prices(self, tmp_path: Path) -> None:
        led = _ledger(tmp_path)
        s = PaperSession(
            session_date=date(2026, 1, 6),
            qualified=False,
            contracts=0,
            entry_price=None,
            exit_price=None,
            session_pnl=Decimal("0"),
            cost=Decimal("0"),
            cumulative_pnl=Decimal("0"),
        )
        led.append(s)
        assert led.read() == [s]

    def test_empty_ledger_reads_empty(self, tmp_path: Path) -> None:
        led = _ledger(tmp_path)
        assert led.read() == []
        assert led.cumulative_pnl() == Decimal(0)
        assert led.pnl_series() == []


class TestRecord:
    def test_record_accumulates_cumulative(self, tmp_path: Path) -> None:
        led = _ledger(tmp_path)
        led.record(
            session_date=date(2026, 1, 5),
            qualified=True,
            contracts=1,
            entry_price=Decimal("21000"),
            exit_price=Decimal("21010"),
            session_pnl=Decimal("200"),
            cost=Decimal("1"),
        )
        led.record(
            session_date=date(2026, 1, 6),
            qualified=False,
            contracts=0,
            entry_price=None,
            exit_price=None,
            session_pnl=Decimal("0"),
            cost=Decimal("0"),
        )
        s3 = led.record(
            session_date=date(2026, 1, 7),
            qualified=True,
            contracts=1,
            entry_price=Decimal("21010"),
            exit_price=Decimal("20990"),
            session_pnl=Decimal("-400"),
            cost=Decimal("1"),
        )
        assert s3.cumulative_pnl == Decimal("-200")  # 200 + 0 - 400
        assert led.cumulative_pnl() == Decimal("-200")
        assert led.pnl_series() == [Decimal("200"), Decimal("0"), Decimal("-400")]

    def test_record_rejects_non_chronological(self, tmp_path: Path) -> None:
        led = _ledger(tmp_path)
        led.record(
            session_date=date(2026, 1, 7),
            qualified=True,
            contracts=1,
            entry_price=Decimal("21000"),
            exit_price=Decimal("21010"),
            session_pnl=Decimal("200"),
            cost=Decimal("1"),
        )
        with pytest.raises(PaperLedgerError, match="non-chronological"):
            led.record(  # same date again -> double-run guard
                session_date=date(2026, 1, 7),
                qualified=False,
                contracts=0,
                entry_price=None,
                exit_price=None,
                session_pnl=Decimal("0"),
                cost=Decimal("0"),
            )
