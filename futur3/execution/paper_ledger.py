"""futur3.execution.paper_ledger - append-only forward-paper P&L ledger.

The forward-paper harness (the $0 true-OOS closer + deployment vehicle) trades the overnight sleeve
forward in time and accumulates realized paper P&L across sessions. This ledger is the PERSISTENT
accumulator: one JSONL row per trading session, append-only. A months-long paper run survives
process restarts -- the MockBroker is re-derived from replay each run, but the realized record
persists here, so the accumulating OOS P&L (and the decay monitor watching it) is durable.

Pure stdlib, deterministic. Decimals serialize as STRINGS (the price/PnL Decimal rule -- never
float); dates as ISO. `record()` enforces chronological appends (a same-date double-run raises,
never silently double-counts). The Sharpe / drawdown the pre-live gate needs are DERIVED from
`pnl_series()` by the decay monitor -- the ledger stores only realized facts.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Final

_SCHEMA: Final[str] = "futur3.paper_ledger.v1"


class PaperLedgerError(Exception):
    """Paper-ledger misuse (malformed row, non-chronological append)."""


@dataclass(frozen=True)
class PaperSession:
    """One trading session's forward-paper outcome (realized facts only)."""

    session_date: date
    qualified: bool  # did the conditioner fire (a traded night)?
    contracts: int  # net contracts held in the window (0 when flat / not qualified)
    entry_price: Decimal | None  # window entry fill (None when flat)
    exit_price: Decimal | None  # window exit fill (None when flat)
    session_pnl: Decimal  # realized $ PnL for the session, NET of cost
    cost: Decimal  # $ cost charged this session
    cumulative_pnl: Decimal  # running sum of session_pnl through this session

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": _SCHEMA,
            "session_date": self.session_date.isoformat(),
            "qualified": self.qualified,
            "contracts": self.contracts,
            "entry_price": None if self.entry_price is None else str(self.entry_price),
            "exit_price": None if self.exit_price is None else str(self.exit_price),
            "session_pnl": str(self.session_pnl),
            "cost": str(self.cost),
            "cumulative_pnl": str(self.cumulative_pnl),
        }

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> PaperSession:
        ep, xp = d["entry_price"], d["exit_price"]
        return cls(
            session_date=date.fromisoformat(str(d["session_date"])),
            qualified=bool(d["qualified"]),
            contracts=int(d["contracts"]),
            entry_price=None if ep is None else Decimal(str(ep)),
            exit_price=None if xp is None else Decimal(str(xp)),
            session_pnl=Decimal(str(d["session_pnl"])),
            cost=Decimal(str(d["cost"])),
            cumulative_pnl=Decimal(str(d["cumulative_pnl"])),
        )


class PaperLedger:
    """Append-only JSONL ledger of `PaperSession` rows (one file = one paper run)."""

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def read(self) -> list[PaperSession]:
        """All sessions in append order ([] if the ledger does not exist yet)."""
        if not self._path.exists():
            return []
        out: list[PaperSession] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(PaperSession.from_json(json.loads(line)))
        return out

    def append(self, session: PaperSession) -> None:
        """Append one already-formed session row (no cumulative computation)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(session.to_json()) + "\n")

    def record(
        self,
        *,
        session_date: date,
        qualified: bool,
        contracts: int,
        entry_price: Decimal | None,
        exit_price: Decimal | None,
        session_pnl: Decimal,
        cost: Decimal,
    ) -> PaperSession:
        """Compute `cumulative_pnl` from the last row and append. Raises on a non-chronological date
        (a same-day double-run never silently double-counts)."""
        prior = self.read()
        if prior and session_date <= prior[-1].session_date:
            raise PaperLedgerError(
                f"non-chronological append: {session_date} <= last {prior[-1].session_date}"
            )
        cumulative = (prior[-1].cumulative_pnl if prior else Decimal(0)) + session_pnl
        session = PaperSession(
            session_date=session_date,
            qualified=qualified,
            contracts=contracts,
            entry_price=entry_price,
            exit_price=exit_price,
            session_pnl=session_pnl,
            cost=cost,
            cumulative_pnl=cumulative,
        )
        self.append(session)
        return session

    def pnl_series(self) -> list[Decimal]:
        """Per-session net $ PnL series (decay monitor derives Sharpe / drawdown from this)."""
        return [s.session_pnl for s in self.read()]

    def cumulative_pnl(self) -> Decimal:
        """Total realized paper P&L to date (0 on an empty ledger)."""
        rows = self.read()
        return rows[-1].cumulative_pnl if rows else Decimal(0)


__all__ = [
    "PaperLedger",
    "PaperLedgerError",
    "PaperSession",
]
