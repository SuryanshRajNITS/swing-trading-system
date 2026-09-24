# Indian Equity Swing Trading System

## Purpose

This project is being developed as a research-first, backtest-first, and paper-trading-first system for deterministic swing-trading research in Indian equities. Strategy ideas are hypotheses to define and evaluate; the system is not an AI discretionary trader.

## Current status

The project foundation, read-only Kite Connect integration, point-in-time daily NIFTY 50 dataset, deterministic backtest engine, and first daily-only research hypothesis (`sr_baseline_v1`) are available. The baseline is development research, not a validated trading strategy. Hourly/15-minute data and logic, paper/live execution, and broker orders are not implemented. Live trading is disabled by default (`trading_mode: paper` and `live_trading_enabled: false`).

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

### Backtest foundation

The backtest engine consumes validated, manifested daily files and the point-in-time universe. Decisions use a shared `StrategyContext`; `EntrySignal`, `ExitSignal`, and `RiskPolicy` are decision contracts intended to remain independent of broker fills. Historical entry/exit fills are simulated only in `src.backtest`; no Kite execution or order API is used. Signals are timestamped at the 15:30 IST daily close and execute at 09:15 IST on the next available data session. Positions are monitored each session against daily OHLC. If both stop and target fall inside the same daily range, stop is applied first; a gap through a stop fills at the open. Exit timestamps for daily-bar stops/targets use 15:30 as the bar label and explicitly do not claim an inferred intraday fill time.

The initial risk policy sizes to `equity * max_risk_per_trade / (entry - stop)`, then caps the whole-share quantity to available cash after estimated entry costs. It does not use leverage. Each date checks that every PIT-eligible security has a daily candle; missing data stops the run instead of silently dropping that security. The engine exposes only daily rows up to the current close and only the most recently completed weekly candle. Current-week weekly data is not visible until the final session closes.

Choose exactly one research phase when running the smoke strategy. Other phases are not run automatically:

```powershell
python -m src.backtest.run --phase development --strategy smoke
```

Available phases are `development` (2023-01-01–2024-12-31), `validation` (2025), `final_oos` (2026-01-01–2026-09-22), and `stress` (2020–2022). The engine accepts explicit date bounds with `--start` and `--end`; phases are not parameter optimization instructions. The smoke rule is only an engine fixture: it emits a signal when a daily close exceeds the prior close, places a synthetic stop at that candle's low, and uses a configurable 2R target. It is not the real strategy and says nothing about profitability.

Backtest run JSON (metadata, config hash, events, equity path, cash, reserved capital, realized and unrealized P&L, open positions, closed-trade ledger, and metrics) is written under ignored `data/processed/backtests/`. The run ID is deterministic for strategy parameters, data, and configuration; `created_at` is an audit timestamp and changes across reruns. The dataset version hashes the selected manifest and candle files.

The default transaction-cost scenario is `CURRENT_ZERODHA_EQUITY_DELIVERY_2026`, verified against the official [Zerodha charges schedule](https://zerodha.com/charges), its [DP charge details](https://support.zerodha.com/category/account-opening/resident-individual/ri-charges/articles/what-do-dp-charges-mean), and Zerodha's [STT explanation](https://support.zerodha.com/category/account-opening/resident-individual/ri-charges/articles/how-is-the-securities-transaction-tax-stt-calculated). It configures ₹0 retail delivery brokerage; STT 0.1% on both buy and sell; current NSE transaction charges 0.00307%; SEBI ₹10/crore; NSE IPFT ₹0.01/crore; GST 18% on brokerage, transaction, SEBI, and IPFT charges; buy stamp duty 0.015%; and ₹15.34 DP charge per scrip per sell date. The DP default assumes a male primary holder and includes its 18% GST. Brokerage and DP values remain configurable.

The official schedule changes over time: for example, Zerodha's [notice](https://zerodha.com/marketintel/bulletin/391488/revision-in-transactions-charges-from-1st-october-2024) records NSE delivery transaction charges changing to 0.00297% from October 1, 2024. The current rates are only a current-live-cost comparison and are **not** backdated to reproduce exact 2023–2026 charges. The model rounds STT to the nearest rupee using the published Zerodha convention. Other proportional charges are retained at their computed precision; contract-note-level rounding can cause small differences.

Slippage is separate from statutory/broker charges and defaults to zero because it is strategy/execution dependent, not an official fixed rate. Configure `slippage_bps` before interpreting live-cost results. Select `--cost-scenario raw_zero` only for an explicit zero-cost/raw comparator; that mode disables both transaction and operating costs.

Fixed operating costs are separate from trading cash and per-trade costs. `config/settings.yaml` defaults to Kite Connect ₹500/month, hosting ₹0/month, and other costs ₹0/month; the latter two are configurable and not estimates. Each calendar month touched by the inclusive evaluation dates is charged in full, including partial first/last months. By default these costs affect only `economic_net_profit` / `economic_net_return`; they do not change fills, sizing cash, completed-trade P&L, or R-multiple. The run metrics report trading return, transaction costs, Kite, hosting, other operating costs, charged months, operating total, and economic return separately.

Metrics include return, trade count, win rate, average win/loss, profit factor, expectancy, average/median R, maximum drawdown and drawdown percentage, average exposure, and average holding period. Annualized metrics are intentionally omitted. Open positions at the end are marked to the final available close and reported separately; they are not force-liquidated or counted as completed trades.

## Run tests

```powershell
python -m pytest
```

The tests cover settings and safety defaults, SQLite initialization/health, mocked Kite authentication/session and historical-data behavior, instrument validation, daily market-data provenance/validation, historical probes, official NSE CSV parsing, weekly close-time look-ahead protection, and the synthetic portfolio/event-driven backtest foundation. The smoke strategy is not the production strategy.

### First strategy research baseline

`sr_baseline_v1` is the first deterministic strategy hypothesis. It implements two daily-only setup types: a repeated support test and a resistance breakout followed by a retest. Both require a confirmed swing-derived price zone; swing pivots become visible only after their configured right-hand bars have completed. Nearby same-side pivots are clustered by the configured percentage tolerance. Support signals require an earlier, separated touch followed by a reaction, then a fresh revisit whose close is at/above zone center with a minimum close-location score. Breakout signals require a close above a resistance zone plus threshold; optional volume confirmation is calculated against the previous N sessions. A later retest must return near the frozen breakout zone, remain above the invalidation boundary, and close above zone center with a minimum close-location score. Breakouts time out after the configured number of sessions. Same-day simultaneous setup signals for one security are both rejected as `MULTIPLE_SETUP_CONFLICT`.

Support stops are below the support-zone lower bound by a configurable price-percentage buffer. Breakout-retest stops are below the lower of the retest low and reclaimed zone lower bound, also with a configurable buffer. Signals execute at the next session open. The 1.5R/2R/2.5R/3R target is calculated from the executable next-session fill and structural stop. The existing engine resolves daily-bar stop/target ambiguity stop-first. A separately switchable weekly filter uses only calendar weeks before the signal's current week; it requires the latest such weekly close above its configurable trailing weekly close average and, by default, above the preceding weekly close. This deliberately avoids current-week Friday-close leakage.

All strategy parameters live under `backtest.strategy` in `config/settings.yaml`. The research CLI is locked to 2023-01-01 through 2024-12-31; it parses only the dated prefix through 2024-12-31 and does not load/hash later daily candles. Later validation and final-OOS performance must not be used to adjust this baseline. The default cost scenario is the current Zerodha delivery schedule; slippage can be run separately at 0, 5, or 10 bps. Results are research observations and do not establish profitability.

Examples (run separately; all outputs are saved under ignored `data/processed/backtests/sr_baseline_v1/`):

```powershell
# Support-test only, 2R, daily only
python -m src.backtest.strategy_run --setup support_test --target-r 2 --weekly-filter off --slippage-bps 0

# Breakout-retest only, 2R, daily only
python -m src.backtest.strategy_run --setup breakout_retest --target-r 2 --weekly-filter off --slippage-bps 0

# Both setups, 2R, daily only
python -m src.backtest.strategy_run --setup both --target-r 2 --weekly-filter off --slippage-bps 0

# Combined setup with weekly filter
python -m src.backtest.strategy_run --setup both --target-r 2 --weekly-filter on --slippage-bps 0

# Target variants: repeat with --target-r 1.5, 2, 2.5, and 3
python -m src.backtest.strategy_run --setup both --target-r 1.5 --weekly-filter off --slippage-bps 0

# Slippage sensitivity: repeat with --slippage-bps 0, 5, and 10
python -m src.backtest.strategy_run --setup both --target-r 2 --weekly-filter off --slippage-bps 5

# Explicit raw-fee-zero comparator (slippage remains the selected value)
python -m src.backtest.strategy_run --setup both --target-r 2 --weekly-filter off --slippage-bps 0 --cost-scenario raw_zero
```

Each JSON result contains run metadata, complete closed-trade ledger, diagnostics and rejection reasons, setup-level summaries, monthly returns, equity curve, R and holding-period distributions, exit counts, largest winners/losers, transaction costs, and fixed-cost economic returns. The output reports open positions separately at the period end.

Generate candle-level failure diagnostics from the saved baseline run (this reads only the 2023-2024 dated dataset prefix and does not rerun the strategy):

```powershell
python -m src.backtest.sr_diagnostics
```

The reproducible CSV/JSON diagnostics and their definitions are written under `data/processed/backtests/sr_baseline_v1/diagnostics/` (ignored local research outputs).
