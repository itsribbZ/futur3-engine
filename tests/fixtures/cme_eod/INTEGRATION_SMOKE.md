# CME Live-Network Integration Smoke Procedure

Manual one-time procedure to validate A1.3 `CMEEODDataSource` against live CME public settlement pages. Run by the operator BEFORE relying on the scraper for production data ingestion.

## Why this is manual

Design constraints:
- Default `pytest` MUST hit zero live endpoints (fast + deterministic + offline-safe).
- Live network smoke is operationally significant (rate limits, WAF challenges, schema reality-check) and must be gated behind explicit operator action.
- Procedure runs ONE-TIME per discovery cycle (URL verify) + ONE-TIME per schema-baseline capture.

## Prerequisites

1. `curl_cffi >= 0.15` installed (already in pyproject runtime deps).
2. Network access to `https://www.cmegroup.com` from your IP.
3. Run within publication window: prefer **18:30 CT** (post-preliminary) or **09:30 CT next morning** (post-final).
4. **Respect the rate limit**: 1 req/2s per page ceiling. The procedure includes built-in pacing.

## Procedure A — verify HYPOTHESIS URLs (one-time per discovery cycle)

For each HYPOTHESIS-flagged contract root (MES, MNQ, MCL, MGC), confirm the URL pattern actually resolves to a settlement page on cmegroup.com.

```bash
cd <repo-root>
.venv/Scripts/python.exe -c "
from datetime import datetime, UTC
from futur3.data.sources import CMEEODDataSource

src = CMEEODDataSource()
now = datetime.now(UTC)

# Probe HYPOTHESIS URLs one at a time, 2s spacing
import time
for root in ['MES', 'MNQ', 'MCL', 'MGC']:
    try:
        settles = src.fetch_all_for_root(root, as_of=now)
        print(f'OK {root}: {len(settles)} settles. Front-month: {settles[0].contract if settles else None}')
    except Exception as e:
        print(f'FAIL {root}: {type(e).__name__}: {e}')
    time.sleep(2)
"
```

**Expected results**:
- ✅ All 4 HYPOTHESIS roots return >0 settles → URL patterns confirmed, promote from HYPOTHESIS → confirmed in `HYPOTHESIS_URL_ROOTS` class set (remove from frozenset).
- ❌ Any returns `WAFBlockedError` → `curl_cffi` impersonation working but CME challenged the request. Retry from different network or different `impersonate=` value (e.g., chrome131).
- ❌ Any returns `CMEScrapeError` with HTTP 404 → URL pattern incorrect. Manual browser-verify the URL, update `URLS` dict in `cme_eod.py`.

After successful HYPOTHESIS verification, record the confirmation in your run log (promote HYPOTHESIS → CONFIRMED).

## Procedure B — capture baseline schema hashes (one-time per discovery)

After confirming all 10 URLs, capture the canonical schema signature per contract by running a real fetch + validating against expected.

```bash
.venv/Scripts/python.exe -c "
from datetime import datetime, UTC
import time
from futur3.data.sources import CMEEODDataSource

src = CMEEODDataSource()
now = datetime.now(UTC)
print(f'Expected schema sig: {src._expected_schema_signature[:16]}...')

for root in ['ES', 'NQ', 'CL', 'GC', 'MBT', 'MET']:
    try:
        settles = src.fetch_all_for_root(root, as_of=now)
        if settles:
            print(f'OK {root}: {len(settles)} settles, state={settles[0].settle_state}')
            print(f'   sample: contract={settles[0].contract} settle={settles[0].settle} oi_prior={settles[0].oi_prior}')
    except Exception as e:
        print(f'FAIL {root}: {type(e).__name__}: {e}')
    time.sleep(2)
"
```

**Expected**: every confirmed root parses without raising `SchemaMismatch`. Schema sig stable across all 6 confirmed contracts.

If `SchemaMismatch` raised on a confirmed contract → CME has changed their HTML schema since the parser was written. This is a real-world bug class 9 detection working as designed. Investigate the page manually, update `EXPECTED_COLUMN_HEADERS`, re-run.

## Procedure C — `oi_prior` reality-check (CRITICAL)

This is the bug class 4 prevention test. Manually inspect ONE real settlement and verify the "Prior Day OI" field on the page corresponds to YESTERDAY's open interest, NOT today's.

```bash
# Fetch ES front-month
.venv/Scripts/python.exe -c "
from datetime import datetime, UTC
from futur3.data.sources import CMEEODDataSource
src = CMEEODDataSource()
settles = src.fetch_all_for_root('ES', as_of=datetime.now(UTC))
front = settles[0]
print(f'Contract: {front.contract}')
print(f'as_of_date: {front.as_of_date}')
print(f'oi_prior (CME convention = YESTERDAY): {front.oi_prior:,}')
print(f'settle: {front.settle}')
"

# Then, in a browser, open the CME settlement page and visually confirm:
# 1. The page's "Prior Day OI" column == the printed oi_prior value
# 2. The CME page explicitly labels it as PRIOR-DAY (not "Today's OI")
```

**Pass**: oi_prior value matches CME page's "Prior Day OI" column AND the column header explicitly says "Prior Day OI" (not "OI" or "Open Interest").

**Fail**: any mismatch → STOP. Re-read parser code. There is no acceptable alternative. Bug class 4 must be air-tight.

## Procedure D — preliminary → final transition validation

Run at two distinct times:
1. **First fetch at 18:30 CT** (post-preliminary publish): capture `settle_state == "preliminary"`.
2. **Second fetch at 09:30 CT next morning** (post-final publish): capture `settle_state == "final"`.

```bash
# Run after 18:00 CT
.venv/Scripts/python.exe -c "
from datetime import datetime, UTC
from futur3.data.sources import CMEEODDataSource
src = CMEEODDataSource()
settles = src.fetch_all_for_root('ES', as_of=datetime.now(UTC))
print(f'18:30 fetch: state={settles[0].settle_state}, settle={settles[0].settle}')
"

# Run after 09:30 CT next morning
# (same command — settle_state should now be 'final')
```

Then verify the archive has BOTH states:

```bash
.venv/Scripts/python.exe -c "
import polars as pl
df = pl.read_parquet('data/cme_eod_archive/contract=ES/year=2026/data.parquet')
print(df.group_by('settle_state').agg(pl.len()))
print(df.filter(pl.col('contract') == df['contract'][0]).select(['settle_state', 'settle']))
"
```

**Expected**: archive has rows for BOTH `preliminary` and `final` states, with settle values from both publications.

## Procedure E — WAF defense validation

Verify `curl_cffi` browser-fingerprint actually defeats Cloudflare WAF:

```bash
# Test 1: curl_cffi via our scraper — should succeed
.venv/Scripts/python.exe -c "
from datetime import datetime, UTC
from futur3.data.sources import CMEEODDataSource
try:
    settles = CMEEODDataSource().fetch_all_for_root('ES', as_of=datetime.now(UTC))
    print(f'curl_cffi PASS: {len(settles)} settles fetched')
except Exception as e:
    print(f'curl_cffi FAIL: {type(e).__name__}: {e}')
"

# Test 2: plain requests — should get 403 (validates WAF is active + curl_cffi is required)
.venv/Scripts/python.exe -c "
import requests
url = 'https://www.cmegroup.com/markets/equities/sp/e-mini-sandp500.settlements.html'
r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
print(f'requests status: {r.status_code} (expect 403; if 200, WAF is dormant on this IP)')
"
```

If `requests` gets 200 — that's fine, it means CME isn't actively challenging this network egress at this moment. But `curl_cffi` must still succeed. The defense is correct regardless of whether WAF is currently active.

If `curl_cffi` gets blocked but `requests` works — anomaly, investigate. (Unlikely; `curl_cffi` impersonation matches real Chrome.)

## Procedure F — rate-limit reality check

Run 5 fetches with built-in 2s pacing — confirm no throttling.

```bash
.venv/Scripts/python.exe -c "
from datetime import datetime, UTC
import time
from futur3.data.sources import CMEEODDataSource
src = CMEEODDataSource()
for i in range(5):
    settles = src.fetch_all_for_root('ES', as_of=datetime.now(UTC))
    print(f'Iter {i+1}: {len(settles)} settles')
    time.sleep(2)
"
```

If any iteration raises `WAFBlockedError` or `CMEScrapeError` with status 429 → CME's per-IP rate-limit is tighter than 2s/req. Increase to 3-5s/req in `_DefaultCMEHTTPClient`.

## Logging this procedure

After running, log results in your run log under the date you ran integration smoke. Include:
- Procedures A-F pass/fail per root
- Any HYPOTHESIS URLs promoted/rejected
- Any rate-limit findings (default 2s/req OK or needs adjustment)
- Any schema-drift findings (CME column rename or addition)

The smoke procedure is repeatable — re-run quarterly to catch CME page evolution drift.

## When NOT to run

- Inside the default `pytest` cycle (already gated by `@pytest.mark.integration`)
- During CME trading hours (you'll be sampling preliminary settles in the middle of price discovery)
- From an unrecognized IP (would muddle the operator-attribution audit trail)
- During the BACKTEST-IS-LIVE Phase A1.16+ replay-harness work (replay must use archived fixtures, NOT live fetches)
