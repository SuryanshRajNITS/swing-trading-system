# Indian Equity Swing Trading System

## Purpose

This project is being developed as a research-first, backtest-first, and paper-trading-first system for deterministic swing-trading research in Indian equities. Strategy ideas are hypotheses to define and evaluate; the system is not an AI discretionary trader.

## Current status

Milestone 1 provides the project foundation: settings loading and validation, basic logging, and a minimal SQLite initialization/health check. Trading strategy and market-data features have not been implemented.

The manual Kite Connect authentication foundation, read-only session verification, NSE instrument snapshot, and historical candle downloader are implemented. Historical candles are not downloaded until you run the command below. Point-in-time NIFTY 50 membership is not yet connected, so the downloader requires you to explicitly choose symbols. Live trading is disabled by default (`trading_mode: paper` and `live_trading_enabled: false`). No broker orders are placed.

## Architecture

The initial foundation keeps configuration, logging, database setup, the application entry point, and tests in small modules. Future work can add normalized market data, deterministic strategy and risk components, backtesting, paper execution, and Telegram reporting in stages. Broker integrations remain separate from strategy logic.

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

Choose the symbols intentionally. The latest NSE instrument snapshot is only a token lookup; it is not point-in-time NIFTY 50 membership. Historical universe membership and reproducible historical dataset orchestration remain to be implemented.

## Run tests

```powershell
python -m pytest
```

The tests cover settings and safety defaults, SQLite initialization/health, logging, mocked Kite authentication/session and historical-data behavior, and instrument-master validation and persistence. No strategy tests exist yet because no strategy logic has been implemented.
