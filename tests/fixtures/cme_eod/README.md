# CME EOD Settlement Fixture Set

Hand-crafted synthetic HTML fixtures representing the canonical CME public settlement page structure. Used by `tests/test_cme_eod.py` for fixture-first parser dev (A1.3 keystone).

## Constraints

- Synthetic-but-realistic prices (round-number patterns) — NEVER real market values; protects against accidental look-ahead in tests.
- Schema mirrors CME canonical column order: `Month · Open · High · Low · Last · Change · Settle · Est. Volume · Prior Day OI`.
- "Prior Day OI" column is PRIOR day per CME convention — bug class 4 prevention via field naming.
- "---" placeholders for missing data per CME convention.
- All Decimal-coercible values; volumes/OI with comma thousands separators.

## Canonical happy-path fixtures (6 confirmed-URL contracts)

| File | Contract | State | Notes |
|---|---|---|---|
| `es_jun26_preliminary.html` | ES (E-mini S&P 500) | preliminary | 4 months: Jun/Sep/Dec/Mar — quarterly cycle |
| `es_jun26_final.html` | ES | final | Same date, Dec26 settle revised 5299.25→5299.50 (drift test) |
| `nq_jun26_preliminary.html` | NQ (E-mini Nasdaq-100) | preliminary | 4 quarterly months; 0.25 tick |
| `cl_jul26_preliminary.html` | CL (Light Sweet Crude) | preliminary | 6 monthly serial contracts; $0.01 tick |
| `gc_aug26_preliminary.html` | GC (Gold) | preliminary | 6 contracts (Feb/Apr/Jun/Aug/Oct/Dec); $0.10 tick |
| `mbt_jun26_preliminary.html` | MBT (Micro Bitcoin) | preliminary | post-2026-05-29 24/7 era; $5 tick |
| `met_jun26_preliminary.html` | MET (Micro Ether) | preliminary | $0.50 tick |

## HYPOTHESIS-URL contracts (URL pattern only, awaiting live verify)

| File | Contract | Notes |
|---|---|---|
| `mes_jun26_preliminary.html` | MES (Micro E-mini S&P 500) | Same structure as ES; HYPOTHESIS URL |
| `mnq_jun26_preliminary.html` | MNQ (Micro E-mini Nasdaq-100) | Same structure as NQ; HYPOTHESIS URL |
| `mcl_jul26_preliminary.html` | MCL (Micro WTI Crude) | Same as CL; HYPOTHESIS URL |
| `mgc_aug26_preliminary.html` | MGC (Micro Gold) | Same as GC; HYPOTHESIS URL |

## Edge-case fixtures

| File | Tests |
|---|---|
| `es_schema_drift.html` | "Settle" column renamed to "Settlement Price" → MUST raise SchemaMismatch |
| `es_with_placeholders.html` | Far-dated month rows with `---` cells (volume/OI nil) → preserved as None, not 0 |
| `waf_block_cloudflare.html` | Cloudflare 403 challenge response → MUST raise WAFBlockedError |
| `empty_table.html` | Table present but `<tbody>` empty → MUST raise MalformedSettlementPage |
| `malformed_no_tbody.html` | No `<table>` at all → MUST raise MalformedSettlementPage |

## Provenance

Fixtures created 2026-05-21 for the A1.3 keystone implementation. Synthetic prices designed to NEVER match real CME quotes — test outputs are reproducible from these inputs only.
