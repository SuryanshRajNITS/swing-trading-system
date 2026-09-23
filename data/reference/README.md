# Universe reference data

These CSV files are UTF-8 source-controlled reference data. They contain no credentials.

- `securities.csv` assigns stable internal `security_id` values. ISIN is blank unless a cited official record validates it.
- `symbol_history.csv` maps each security to its exchange symbol and company name over an inclusive `effective_from` and exclusive `effective_to` interval.
- `universe_membership.csv` stores date-bounded membership in `NIFTY50_PIT_TRADABLE`, `NIFTY50_PIT_INDEX`, and the separate `NIFTY50_CURRENT` snapshot. Membership dates are inclusive at `effective_from` and exclusive at `effective_to`.
- `universe_events.csv` records member additions/removals, identity changes, and the review with no replacements. It preserves `announcement_date` separately from `effective_date`; a blank announcement date means the available source did not provide a prior announcement for that specific event.
- `broker_instrument_history.csv` stores broker token mappings with the same date-bound convention. Tokens are broker identifiers, not security identity.
- `universe_coverage.csv` declares the exact date spans that can be queried. Queries outside a universe's coverage fail; they never fall back to another universe.
- `sources.csv` records source URLs and descriptions for the rows above.

## Universe distinctions

`NIFTY50_PIT_INDEX` preserves index-methodology corporate-action placeholders such as `DUMMYITC`. `NIFTY50_PIT_TRADABLE` excludes dummy placeholders. A listed corporate-action security may temporarily be a 51st tradable constituent until NSE Indices removes it; those rows have `membership_type=CORPORATE_ACTION` and retain their official effective dates.

`NIFTY50_CURRENT` is an explicitly stored one-day snapshot dated 2026-09-23. It was materialized from the reviewed effective-dated event ledger for the current-universe interface. It is not used by point-in-time queries and is not automatically refreshed. `current_members()` fails outside its declared coverage until the snapshot is refreshed.

The initial roster is the December 2022 NIFTY 50 component table in the SEBI January 2023 Bulletin. The initial reference-data state is applied from 2023-01-01. The primary research window is 2023-01-01 through 2026-09-22; the 2026-09-30 BSE change is included for boundary verification beyond that cutoff.

## Kite snapshot import and audit

To add mappings from an existing locally saved NSE instrument snapshot, without making a Kite API request:

```powershell
python -m src.data.universe.snapshot --path data/raw/instruments/nse_instruments_YYYY-MM-DD.csv
```

The importer only maps symbols it can resolve and records a one-day interval for that snapshot date. It does not claim historical token stability.

To report uncovered broker mappings for a date range:

```powershell
python -m src.data.universe.audit --start 2023-01-01 --end 2026-09-22
```
