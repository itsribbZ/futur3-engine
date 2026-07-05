# IBKR Live-Network Integration Smoke Procedure

Manual one-time procedure to validate A1.4 `IBKRHistoricalDataSource` against a live
IB Gateway connection. Run by the operator BEFORE relying on the scraper for
production data ingestion (A1.5+ paper trading).

## Why this is manual

Design constraints (fixture-only default test suite):
- Default `pytest` MUST hit zero live endpoints (already enforced by `@pytest.mark.integration`).
- Live IB Gateway smoke involves: real credentials, real rate limits, real WAF-equivalent
  guards, daily auth re-cycle at 23:45 CT.
- Procedure runs ONE-TIME per fresh install + RE-RUN at each new IBKR Gateway version.

## Prerequisites

1. **IB Gateway installed** (free; https://www.interactivebrokers.com/en/trading/ibgateway-stable.php).
   Latest stable (TWS 10.30+ minimum).
2. **IBKR paper account** signed up (instant; free).
3. **`ib_async >= 2.0.1`** installed (already in `pyproject.toml`).
4. **Paper account market data subscription enabled** in Client Portal
   (Settings → Market Data Subscriptions → "US Securities Snapshot & Futures Value Bundle"
   — $10/mo paper share via live credit OR free 15-min-delayed for backtest-only mode).
5. **Re-classify account as non-professional** in Client Portal (defaults to pro = 10× cost).
6. **Market Data API Acknowledgement** signed in Client Portal.

## Procedure A — IB Gateway launch + paper login

1. Start IB Gateway in **paper mode** (port 4002).
2. Log in with paper account credentials.
3. Confirm Gateway shows "Connected" status.
4. **Critical setting**: API → Configure → Settings → check "Enable ActiveX and Socket Clients" + un-check
   "Read-Only API" (the default Read-Only blocks placing orders, but historical-data fetch may also gate on it).
5. Confirm "Socket port" reads 4002.

If Gateway can't be launched headlessly, use **IBC** (IBController, https://github.com/IbcAlpha/IBC)
to automate the daily reset (23:45 CT). For initial smoke testing, manual Gateway start is fine.

## Procedure B — Bare ib_async connection probe

Verify the SDK can connect before testing futur3's wrapper:

```bash
cd <repo-root>
.venv/Scripts/python.exe -c "
from ib_async import IB
ib = IB()
ib.connect('127.0.0.1', 4002, clientId=1)
print(f'isConnected: {ib.isConnected()}')
print(f'serverVersion: {ib.client.serverVersion()}')
ib.disconnect()
"
```

**Expected**: `isConnected: True` + a serverVersion integer.

**If fails with timeout**: Gateway not running on port 4002, or "Enable ActiveX and Socket Clients" is OFF.

**If fails with API error**: re-check Market Data API Acknowledgement + non-professional classification.

## Procedure C — IBKRHistoricalDataSource happy-path smoke (ES daily bars)

Use futur3's wrapper to fetch ES front-month 1y of daily bars:

```bash
.venv/Scripts/python.exe -c "
from datetime import datetime, UTC, timedelta
from futur3.data.sources import IBKRHistoricalDataSource
from futur3.data.types import ContractSymbol, BarResolution

src = IBKRHistoricalDataSource()  # default paper port 4002

# Pick front-month ES — adjust to current quarterly
# (June=M, September=U, December=Z, March=H)
contract = ContractSymbol('ESM26')  # ES June 2026

ts_end = datetime.now(UTC)
ts_start = ts_end - timedelta(days=365)

bars = list(src.get_bars(contract, ts_start, ts_end, BarResolution.DAY_1))
print(f'Fetched {len(bars)} daily bars for {contract}')
if bars:
    first, last = bars[0], bars[-1]
    print(f'  first: ts={first.ts} settle-ish={first.close}')
    print(f'  last:  ts={last.ts} settle-ish={last.close}')
    print(f'  source_id: {first.source_id}')
    print(f'  content_bytes_sha sample: {first.content_bytes_sha[:16]}...')
src.disconnect()
"
```

**Expected**: ~250 trading days of bars returned, all RawBar instances, ts in UTC.

**If fails with no permissions error**: paper account doesn't have the market data subscription enabled.

**If fails with timeout**: rate-limit was hit; wait 60s and retry.

## Procedure D — Rate-limit reality check (intentional violation)

Verify BackoffQueue actually throttles vs. IBKR's silent disconnect on over-rate:

```bash
.venv/Scripts/python.exe -c "
from datetime import datetime, UTC, timedelta
import time
from futur3.data.sources import IBKRHistoricalDataSource
from futur3.data.types import ContractSymbol, BarResolution

src = IBKRHistoricalDataSource()
contract = ContractSymbol('ESM26')

# Try to fire 10 requests in rapid succession (violates 5/2s per-contract)
start = time.monotonic()
for i in range(10):
    bars = list(src.get_bars(
        contract,
        datetime.now(UTC) - timedelta(days=2),
        datetime.now(UTC),
        BarResolution.DAY_1,
    ))
    print(f'iter {i+1}: {len(bars)} bars, elapsed={time.monotonic()-start:.2f}s')

src.disconnect()
"
```

**Expected**: 10 requests complete successfully but BackoffQueue inserts ~2s spacing per 5
requests, so total elapsed ≥ 4s. No IBKR throttle / disconnect should occur.

**Pass criteria**: all 10 requests succeed; Gateway connection stays alive.

**Fail criteria**: Gateway disconnects (rate-limit violation) OR IBKR returns error code 162 (pacing).

## Procedure E — Daily 23:45 CT disconnect simulation

Schedule a long-running test that holds the connection across the IBKR daily reset:

```bash
# Run starting before 23:30 CT
.venv/Scripts/python.exe -c "
from datetime import datetime, UTC, timedelta
import time
from futur3.data.sources import IBKRHistoricalDataSource
from futur3.data.types import ContractSymbol, BarResolution

src = IBKRHistoricalDataSource()
contract = ContractSymbol('ESM26')

# Fetch once per minute for 60 minutes (crosses 23:45 CT reset)
for i in range(60):
    try:
        bars = list(src.get_bars(
            contract,
            datetime.now(UTC) - timedelta(days=1),
            datetime.now(UTC),
            BarResolution.DAY_1,
        ))
        print(f'iter {i+1}: {len(bars)} bars, connected={src.healthcheck()}')
    except Exception as e:
        print(f'iter {i+1}: EXCEPTION {type(e).__name__}: {e}')
    time.sleep(60)
src.disconnect()
"
```

**Expected behavior** at 23:45 CT:
1. Gateway disconnects (we'll see `IBKRConnectionError` or `healthcheck()=False`)
2. A1.4 SKELETON does NOT auto-reconnect — caller must handle.
3. A1.7 will add auto-reconnect logic.

**Pass criteria**: error is surfaced cleanly as `IBKRConnectionError`. Behavior matches docs.

**Fail criteria**: silent disconnect or hang.

## Procedure F — All 10 contracts URL routing verification

Confirm all 10 contracts in our universe route to the correct exchange and fetch bars:

```bash
.venv/Scripts/python.exe -c "
from datetime import datetime, UTC, timedelta
import time
from futur3.data.sources import IBKRHistoricalDataSource
from futur3.data.types import ContractSymbol, BarResolution

src = IBKRHistoricalDataSource()

contracts = [
    'ESM26', 'NQM26', 'MESM26', 'MNQM26',
    'CLN26', 'MCLN26', 'GCQ26', 'MGCQ26',
    'MBTM26', 'METM26',
]
for symbol in contracts:
    try:
        bars = list(src.get_bars(
            ContractSymbol(symbol),
            datetime.now(UTC) - timedelta(days=7),
            datetime.now(UTC),
            BarResolution.DAY_1,
        ))
        print(f'{symbol}: {len(bars)} bars ✓')
    except Exception as e:
        print(f'{symbol}: FAIL {type(e).__name__}: {e}')
    time.sleep(2)  # rate-limit pacing
src.disconnect()
"
```

**Expected**: All 10 contracts return ~5 daily bars (7 days minus weekends).

**Common failure mode**: "No market data permissions for ..." → subscribe to bundle.

## Logging this procedure

After running, log results in your run log under the date you ran integration smoke:
- Procedures A-F pass/fail
- Any rate-limit findings
- Any contract permission gaps
- Time at which daily 23:45 CT disconnect occurred (procedure E)

The smoke procedure is repeatable — re-run quarterly to catch IB Gateway version changes
and any IBKR API surface evolution.

## When NOT to run

- Inside the default `pytest` cycle (already gated by `@pytest.mark.integration` which is disabled)
- Without a paper market data subscription (will silently return empty bar lists)
- During the 23:45 CT daily reset window unless specifically testing procedure E
- From an unrecognized IP (would muddle the operator-attribution audit trail)
- Before completing the A1.4 baseline triple-green (which this doc accompanies)
