# futur3 — futures trading engine

**An event-driven futures-trading engine: the execution, data-integrity, and validation apparatus — built strategy-agnostic.** This repository is the engine and its test suite. The alpha/strategy implementations are intentionally not included; the engine runs any object that implements the `Strategy` interface.

Built solo as a systems-engineering project, apparatus-first — correctness and reproducibility before any profitability claim.

- **69 test modules** across the data, storage, execution, statistics, and engine layers
- Deterministic, bit-reproducible runs · no silent fallbacks (every degraded path raises loudly)

## What it does

- **Multi-source data layer** — ingests bars/settlements from IBKR, CME, and five crypto venues, then cross-verifies them through a fail-closed `MultiSourceVerifier` (majority-vote / highest-tier) with point-in-time correctness.
- **Backtest-is-live engine** — one code path; a `RuntimeContext` injects `BACKTEST` / `LIVE_PAPER` / `LIVE_FUNDED` mode, so the code that backtests is the code that trades.
- **Execution layer** — a `BrokerAdapter` abstraction (IBKR, TopstepX, and a deterministic `MockBroker`), a tick-haircut slippage model, and calendar-spread roll logic.
- **Risk management** — position sizing hard-gated to the tightest of quarter-Kelly, a margin cap, and a leverage cap.
- **Statistical hygiene** — walk-forward + combinatorial-purged cross-validation, Deflated / Probabilistic Sharpe, and a Reality Check — built in to fight backtest overfitting.
- **Storage** — Parquet (hive-partitioned) + DuckDB; idempotent with revision handling and a byte-reproducible canonical sort.

## Architecture

```
data/        multi-source ingestion + fail-closed verification (point-in-time)
storage/     Parquet / DuckDB persistence (deterministic, idempotent)
execution/   broker adapters (IBKR / TopstepX / Mock), slippage, risk manager, roll
runtime/     RuntimeContext — backtest / paper / live injection seam
engine/      event-driven backtest loop (consumes the Strategy interface)
cv/          combinatorial-purged + walk-forward cross-validation
stats/       deflated / probabilistic Sharpe, Reality Check
strategies/  the Strategy / Signal interface only (concrete alphas not included)
```

## Run the tests

```bash
python -m venv .venv
source .venv/Scripts/activate        # Windows; use .venv/bin/activate on macOS/Linux
pip install -e ".[dev]"
pytest -q
```

## Design constraints

- **Backtest-is-live** — one engine, mode injected at the seam; business logic is mode-agnostic.
- **Apparatus-first** — zero-bug discipline before any profitability claim.
- **Deterministic** — same inputs produce the same output hash, byte-for-byte.
- **No silent fallbacks** — every degraded path raises loudly rather than guessing.

> The strategy/alpha layer is intentionally omitted. This repository demonstrates the engineering — data integrity, execution, risk, and statistical validation — not a trading edge.

## License

Proprietary — Copyright (c) 2026 Jacob Ribbe. All Rights Reserved. See [LICENSE](LICENSE).
