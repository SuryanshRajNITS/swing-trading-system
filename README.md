# Indian Equity Swing Trading System

## Purpose

This project is being developed as a research-first, backtest-first, and paper-trading-first system for deterministic swing-trading research in Indian equities. Strategy ideas are hypotheses to define and evaluate; the system is not an AI discretionary trader.

## Current status

Milestone 1 provides the project foundation: settings loading and validation, basic logging, and a minimal SQLite initialization/health check. The deterministic trading strategy and backtesting engine have not been implemented.

The repository includes manual Kite Connect authentication, read-only session verification, NSE instrument snapshots, historical candle access, and a date-aware point-in-time NIFTY 50 universe. A resumable daily PIT dataset builder and official NSE daily CSV importer are available. Historical dataset generation requires an authenticated Kite session; strategy/backtesting, 60-minute and 15-minute bulk data, and broker orders are not implemented. Live trading is disabled by default (`trading_mode: paper` and `live_trading_enabled: false`).

## Architecture

Configuration, logging, and SQLite remain small foundation modules. The data layer separates broker-token mappings from market-data files: files are keyed by stable `security_id`, and each manifest row records its actual data source and symbol used. The daily dataset is the source for a deterministic weekly resampler; weekly bars are exposed only after the final observed exchange session closes. Broker integrations remain separate from strategy logic.

## Requirements

- Python 3.12
- Dependencies listed in `requirements.txt`

## Setup

Create and activate a virtual environment from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` only when local integration credentials are eventually needed. The current foundation does not read or require credentials. Never commit `.env` or put real credentials in project files.

## Run the foundation

From the repository root:

```powershell
python -m src.main
```

This loads `config/settings.yaml`, initializes the configured local SQLite file, checks logging, and reports the configured mode. The database and application log are created under `data/database/` and `logs/` and are ignored by Git.

## Kite Connect authentication foundation

Authentication uses the official Kite Connect Python client and a manual browser login. Set `KITE_API_KEY` and `KITE_API_SECRET` in a local `.env` file, then run:

```powershell
python -m src.broker.auth
```

The application displays the Kite login URL, opens it in the browser, and listens for the registered callback at `http://127.0.0.1:8000/callback`. Complete Zerodha login and 2FA manually. The request token is exchanged through the Kite SDK and is never printed or logged. The resulting access token is encrypted with Windows DPAPI for the current Windows user and saved under the ignored `data/.secrets/` directory. It is reused only for the current `Asia/Kolkata` calendar day. This module does not automate credentials or 2FA, fetch market data, or place orders. Live trading remains disabled.

To verify today's saved session with a read-only profile request, run:

```powershell
python -m src.broker.session
```

The verifier uses only the API key and saved access token. It does not display profile data or place orders.

Historical-data date bounds are configured in `config/settings.yaml`:

```yaml
data:
  history_start_date: "2023-01-01"
  history_end_date: "yesterday"
```

The settings loader validates this range and resolves `yesterday` at load time using `Asia/Kolkata`.

To download and validate the current NSE instrument master into an ignored, date-stamped CSV snapshot, run:

```powershell
python -m src.broker.instruments
```

This is a read-only Kite request. The snapshot contains exchange instrument metadata; it does not establish historical NIFTY 50 membership.

### Historical candles

After creating today's Kite session and downloading the NSE instrument snapshot, request historical candles for explicitly selected NSE equity symbols:

```powershell
python -m src.data.historical --symbols INFY,TCS --intervals day,60minute,15minute
```

The date range comes from `data.history_start_date` and `data.history_end_date` in `config/settings.yaml`. Supported intervals are `day`, `60minute`, and `15minute`; omit `--intervals` to request all three. Files are normalized CSVs under ignored `data/raw/historical/<interval>/`. The command reads instrument tokens from the latest NSE snapshot, requires the symbols to be NSE equity instruments, uses the saved access token, and makes historical-data requests only. It does not place orders.

Choose the symbols intentionally. The latest NSE instrument snapshot is only a token lookup; it does not establish historical NIFTY 50 membership or backdate broker mappings.

### Point-in-time daily dataset

The daily dataset builder takes the union of tradable security identities that belonged to the requested PIT universe. It defaults to warmup beginning 2022-07-01, writes under `data/processed/market_data/<universe>/day/<security_id>/`, and records market-data provenance in `data/processed/market_data_manifest.csv`. The manifest does not require a Kite token: Kite, the local INFY/TCS files, and explicit official NSE imports have separately identified sources. No forward-filled or synthetic candles are written.

First run a read-only historical coverage probe:

```powershell
python -m src.data.kite_probe --universe NIFTY50_PIT_TRADABLE --start 2023-01-01 --end 2026-09-22 --warmup-start 2022-07-01
```

The probe uses current snapshot tokens only to request read-only historical candles. It records data-response evidence separately and never modifies `broker_instrument_history.csv`. A matching probe manifest is reused by the builder when its date range and current instrument snapshot are unchanged. It fails closed for the pre-demerger Tata Motors identity; the current TMPV token is never used to retrieve the older TATAMOTORS interval.

To build the daily evaluation dataset, including the explicitly selected official NSE secondary source for HDFC Ltd and pre-demerger TATAMOTORS, run:

```powershell
python -m src.data.dataset build --universe NIFTY50_PIT_TRADABLE --start 2023-01-01 --end 2026-09-22 --warmup-start 2022-07-01 --intervals day --include-nse-secondary
```

On Windows PowerShell, use the same command after activating `.venv` as above. The builder is resumable: valid complete per-security files are reused. Kite calls are throttled to at most three per second and retried with bounded exponential backoff. Daily request chunks are capped at 2,000 calendar days. The requested period fits one chunk per security; count roughly 60 Kite probes plus up to 60 Kite downloads, as well as one NSE request each for HDFC and pre-demerger TATAMOTORS. Existing valid files reduce the download count.

HDFC and old TATAMOTORS daily rows come from NSE's official Security-wise Archives CSV export, only when `--include-nse-secondary` is specified. Their source and page reference are recorded in the manifest. The two Tata symbol eras are stored as separate files under the same stable security ID and are not concatenated. NSE's official archive interface limits each export request to five years; requests spanning more than five years must be split. No secondary data source is used silently.

Audit broker mapping and market-data coverage separately:

```powershell
python -m src.data.dataset audit --universe NIFTY50_PIT_TRADABLE --start 2023-01-01 --end 2026-09-22 --interval day
```

Market-data coverage checks validated rows, whether data brackets each PIT membership range, and interior dates against the union of dates in validated local series. It reports absent data as gaps and does not fill them. The calendar is inferred from observed NSE equity datasets rather than independently downloaded from an official calendar feed, so an exchange session missing from every local file cannot be detected by this audit.

Weekly candles are derived locally, never downloaded natively. `src.data.weekly.resample_daily_to_weekly()` aggregates observed sessions only and accepts an exchange session calendar; `last_completed_weekly_candle(as_of_timestamp, ...)` withholds the current week until its final session close. No weekly strategy is implemented.

## Run tests

```powershell
python -m pytest
```

The tests cover settings and safety defaults, SQLite initialization/health, mocked Kite authentication/session and historical-data behavior, instrument validation, daily market-data provenance/validation, historical probes, official NSE CSV parsing, and weekly close-time look-ahead protection. No strategy tests exist yet because no strategy logic has been implemented.
