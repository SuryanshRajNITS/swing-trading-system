# Hourly Kite market data

The PIT hourly pipeline requests Kite's `60minute` historical interval and stores
normalized OHLCV under:

```text
data/processed/market_data/<universe>/60minute/<security_id>/<symbol>_<token>.csv
```

Coverage, broker mapping provenance, market-data-access provenance, requested
and observed ranges, row count, missing full-bar diagnostics, acquisition
timestamp, and SHA-256 are recorded in
`data/processed/hourly_market_data_manifest.csv`. Read-only 60-minute probe
results are separately written to
`data/processed/hourly_access_probe_manifest.csv`. Raw/processed data are
ignored by Git.

## Timestamp and session convention

Kite candle timestamps identify the **start** of each candle. The official Kite
historical API reference describes the candle format; Kite staff confirm that
timestamps mark the beginning of the interval and are aligned from the exchange
session open ([API reference](https://kite.trade/docs/connect/v3/historical/),
[timestamp clarification](https://kite.trade/forum/discussion/15973/timestamp-in-candle-data-is-open-time-or-candle-close-time)).

For the NSE cash session, regular 60-minute starts are 09:15, 10:15, 11:15,
12:15, 13:15, and 14:15 Asia/Kolkata. A 15:15 timestamp is accepted and retained
as Kite's possible final partial interval from 15:15 to the 15:30 close. It is
optional in missing-full-bar diagnostics. The pipeline never creates a
15:30-16:15 candle. Timestamps retain their instant and are serialized with the
Asia/Kolkata UTC offset; a naive provider timestamp is interpreted as
Asia/Kolkata, consistent with the existing historical-data normalization.

Weekends, off-grid times, and dates absent from the local validated daily
dataset's exchange-session calendar are rejected on ingestion. Missing hourly
candles are reported and never filled.

## Identity and acquisition safety

Instrument tokens are first resolved through `UniverseProvider` by stable
`security_id` and explicit mapping effective dates. Where that documentary
mapping is absent, the pipeline may resolve the current NSE equity contract
from the saved Kite instrument snapshot and use its token only after a
successful read-only `60minute` historical probe on a local validated session
when the same stable security is known to exist. That result authorizes market
data access; it does not create or imply historical broker-mapping provenance.

The current repository's Kite mapping snapshot covers 2026-09-23 only. The
pipeline keeps its provenance separate from access proof, records both in the
hourly manifest, and fails closed for ambiguous corporate-action identity
boundaries. Only specifically source-reviewed continuity renames (currently
ZOMATO/ETERNAL and LTIM/LTM) may cross a symbol boundary through the current
token; an unclassified identity event fails closed. HDFC Ltd has no current
token in the saved snapshot and remains an explicit no-Kite-access gap unless
separately supported by an evidenced historical mapping.

Run `python -m src.data.hourly probe` first. The downloader consumes only the
matching, snapshot-current probe report and successful rows. It uses Kite's
existing 400-calendar-day maximum for `60minute` chunks, the shared request
throttle, bounded retries, and deterministic conflict-checked merges. It
refreshes missing full-bar sessions and the most recent observed session when
data already exists. Failed/unavailable ranges are recorded as
`NO_KITE_HISTORICAL_ACCESS`; they are not removed from the PIT universe.
