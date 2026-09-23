# Indian Equity Swing Trading System

## Purpose

This project is being developed as a research-first, backtest-first, and paper-trading-first system for deterministic swing-trading research in Indian equities. Strategy ideas are hypotheses to define and evaluate; the system is not an AI discretionary trader.

## Current status

Milestone 1 provides the project foundation: settings loading and validation, basic logging, and a minimal SQLite initialization/health check. Trading strategy and market-data features have not been implemented.

Zerodha/Kite Connect and Telegram are not connected. Live trading is disabled by default (`trading_mode: paper` and `live_trading_enabled: false`). No broker orders are placed by this foundation.

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

## Run tests

```powershell
python -m pytest
```

The tests cover settings and safety defaults, SQLite initialization/health, and basic logging setup. No strategy tests exist yet because no strategy logic has been implemented.
