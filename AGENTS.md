# AGENTS.md

# Indian Equity Swing Trading System

## 0. Purpose

This repository is being built as a research-first, backtest-first, paper-trading-first and eventually live-trading swing system for Indian equities using Zerodha Kite Connect.

The project converts trading concepts learned from the user's supplied trading-course/video material into:

1. Explicit rules
2. Deterministic algorithms
3. Historical backtests
4. Failure analysis
5. Walk-forward/out-of-sample validation
6. Paper trading
7. Telegram reporting
8. Eventually, controlled live execution

The system must NOT behave like an AI discretionary trader.

The LLM/Codex is an engineering and research assistant.

The strategy itself must be deterministic and reproducible.

---

# 1. Core philosophy

## 1.1 Strategy decisions must be deterministic

Never implement:

- "chart looks strong"
- "this stock feels bullish"
- "strong momentum" without a measurable definition
- "best-looking setup"
- "strongest sector" without an objective ranking
- "AI thinks this stock will rise"

Every such concept must be converted into measurable rules.

For example:

Instead of:

    "Strong breakout"

define measurable variables such as:

    breakout_close > resistance
    breakout_distance >= X * ATR
    volume_ratio >= Y
    consolidation_length >= N candles

The exact parameters must be configurable and tested.

---

## 1.2 Do not assume that a source's claimed edge is true

The trading videos supplied by the user contain trading methodologies, observations and claimed success rates.

Treat them as hypotheses.

Do NOT encode statements such as:

- "this setup works 70% of the time"
- "this pattern is highly reliable"
- "1:3 always works"

as facts.

Instead:

1. Convert the idea into a rule.
2. Backtest it.
3. Measure it.
4. Test it out-of-sample.
5. Analyze failure cases.

---

# 2. V1 scope

V1 is deliberately small.

## Universe

Initial universe:

- NIFTY 50 equities
- Liquid stocks only
- Valid instrument/token mapping
- Exclude unsuitable/illiquid/circuit-prone names where appropriate

Do not begin with hundreds of stocks.

After the baseline works, the universe can be expanded.

## Direction

V1:

- Long-only
- Cash equity
- Swing trades

Do NOT implement options in V1.

Do NOT implement short-selling strategies in V1.

## Trading style

Swing trading only.

V1 is NOT an intraday system.

Intraday scanners discussed in the source material should not be implemented in V1.

---

# 3. Overall architecture

The intended architecture is:

    MARKET DATA
          |
          v
    UNIVERSE
          |
          v
    MARKET REGIME
          |
          v
    SECTOR STRENGTH
          |
          v
    STOCK SCREENER
          |
          v
    SETUP DETECTION
          |
          v
    MULTI-TIMEFRAME CONFIRMATION
          |
          v
    RISK ENGINE
          |
          +------------+
          |            |
          v            v
       PAPER        LIVE
          |            |
          +-----+------+
                |
                v
          TRADE LOGGER
                |
                v
        FAILURE ANALYSIS
                |
                v
            RESEARCH

Backtesting must use the same strategy/risk logic as paper/live execution.

---

# 4. Technology architecture

Initial stack:

- Python 3.12
- Zerodha Kite Connect
- pandas
- numpy
- SQLite
- SQLAlchemy where useful
- python-dotenv
- Pydantic where useful
- APScheduler
- pytest
- requests where required
- Telegram Bot API

Do NOT introduce unnecessarily complex infrastructure.

Do NOT introduce:

- React
- Vite
- FastAPI
- Docker
- Kubernetes
- Kafka
- Redis
- PostgreSQL

for V1 unless a concrete requirement appears later.

The earlier Zerodha tutorial demonstrated React/Vite/FastAPI as a possible UI architecture.

We deliberately are NOT using that UI architecture in V1.

Telegram is the V1 interface.

---

# 5. Repository structure

Target architecture:

    swing-trading-system/
    |
    +-- .venv/
    +-- .env
    +-- .env.example
    +-- .gitignore
    +-- AGENTS.md
    +-- README.md
    +-- requirements.txt
    |
    +-- config/
    |   +-- settings.yaml
    |
    +-- data/
    |   +-- raw/
    |   +-- processed/
    |   +-- database/
    |
    +-- logs/
    |
    +-- src/
    |   +-- __init__.py
    |   |
    |   +-- main.py
    |   |
    |   +-- broker/
    |   |   +-- __init__.py
    |   |   +-- kite_client.py
    |   |
    |   +-- data/
    |   |   +-- __init__.py
    |   |   +-- downloader.py
    |   |   +-- market_data.py
    |   |   +-- instruments.py
    |   |
    |   +-- strategy/
    |   |   +-- __init__.py
    |   |   +-- universe.py
    |   |   +-- market_structure.py
    |   |   +-- sector_strength.py
    |   |   +-- screener.py
    |   |   +-- setups.py
    |   |   +-- signals.py
    |   |
    |   +-- risk/
    |   |   +-- __init__.py
    |   |   +-- position_sizing.py
    |   |   +-- risk_manager.py
    |   |
    |   +-- execution/
    |   |   +-- __init__.py
    |   |   +-- paper_trader.py
    |   |   +-- live_trader.py
    |   |
    |   +-- notifications/
    |   |   +-- __init__.py
    |   |   +-- telegram.py
    |   |
    |   +-- reports/
    |   |   +-- __init__.py
    |   |   +-- sod.py
    |   |   +-- eod.py
    |   |
    |   +-- backtest/
    |       +-- __init__.py
    |       +-- engine.py
    |       +-- metrics.py
    |       +-- analysis.py
    |
    +-- tests/
        +-- __init__.py

Do not create every module immediately.

Build incrementally.

---

# 6. Security

Never put credentials into:

- Python files
- YAML files
- README
- AGENTS.md
- Git commits
- Telegram messages
- logs

Use:

    .env

Example:

    KITE_API_KEY=
    KITE_API_SECRET=
    KITE_ACCESS_TOKEN=
    TELEGRAM_BOT_TOKEN=
    TELEGRAM_CHAT_ID=

`.env` must be in `.gitignore`.

`.env.example` contains placeholders only.

Never ask the user to paste secrets into ChatGPT.

---

# 7. Execution safety

Live trading must be OFF by default.

Use explicit modes:

    backtest
    paper
    alert_only
    live

Configuration should look conceptually like:

    execution:
      mode: paper

Never infer live trading merely because API credentials exist.

Never place a live order during development/testing unless the user explicitly enables live mode.

Before live execution, implement:

- duplicate-order protection
- position reconciliation
- order-status reconciliation
- restart recovery
- daily risk limit
- kill switch
- broker-error handling
- data-error handling
- audit logging

---

# 8. Trading knowledge learned from source material

The following concepts came from the user's supplied trading videos/transcripts.

They are SOURCE-DERIVED CONCEPTS.

They are NOT automatically proven profitable rules.

---

# 9. Liquidity and universe selection

The source material emphasizes beginning with liquid stocks.

Initial universe:

- NIFTY 50

Reasons discussed in the source:

- better liquidity
- lower risk of problematic execution
- fewer circuit-related issues
- easier systematic testing

Implementation must still apply objective liquidity/data-quality checks.

Potential future filters:

- average traded value
- average volume
- spread
- F&O availability where relevant
- minimum price

Do not arbitrarily add filters without testing.

---

# 10. Multi-timeframe analysis

The source material strongly emphasizes:

    WEEKLY
       |
       v
    DAILY
       |
       v
     1H
       |
       v
     15M

## Weekly

Purpose:

- overall trend
- major market structure
- higher highs
- higher lows
- major support/resistance
- broad directional context

## Daily

Purpose:

- setup detection
- breakout/consolidation
- support/resistance
- momentum
- volume
- primary swing signal

## 1H

Purpose:

- structural stop region
- local swing structure
- breakout/retest confirmation
- tighter risk definition

## 15M

Purpose:

- entry refinement
- trigger confirmation
- local structural stop

Important:

Do not require every timeframe to independently produce a buy signal.

The hierarchy should answer:

1. Is higher-timeframe structure acceptable?
2. Is there a valid daily setup?
3. Can risk be defined?
4. Is there a precise entry trigger?

---

# 11. Market structure

The supplied material covers:

- swing highs
- swing lows
- Higher High (HH)
- Higher Low (HL)
- Lower High (LH)
- Lower Low (LL)
- uptrend
- downtrend
- consolidation

A bullish structure generally corresponds to:

    HH
    HL
    HH
    HL

A bearish structure generally corresponds to:

    LH
    LL
    LH
    LL

These must be converted into deterministic swing-detection rules.

Do NOT rely on visual chart interpretation.

Potential implementation:

- configurable pivot lookback
- configurable minimum swing distance
- optional ATR normalization

Parameters must be configurable.

---

# 12. Support and resistance

Source-derived concepts:

- support/resistance is better represented as a zone than a single exact price
- fresh levels are preferred
- repeated tests may weaken a level
- avoid entering in the middle of a major range
- breakout through resistance can be a setup
- retest of breakout level can be a setup

Implementation must define:

- how swing levels are identified
- zone width
- what constitutes a test
- how many tests count as repeated
- how old a level can be
- how breakout distance is measured
- how retest is detected

Do not use subjective chart interpretation.

---

# 13. Avoid the middle of the range

One repeated source concept:

Do not buy when price is simply sitting between major support and resistance.

A trade should have a defined edge.

Possible measurable approach:

    distance_to_resistance
    distance_to_support
    ATR

Potentially reject trades where:

    distance_to_resistance is too small for intended reward

or where price is located in the middle of a defined range.

Exact formula must be tested.

---

# 14. Breakout setups

Major source-derived setup:

    resistance
       |
       |       breakout
       |          /
       |        /
       |_______/________
          consolidation

Breakout candidates can involve:

- repeated resistance tests
- consolidation
- range-high breakout
- strong close
- volume expansion
- higher-timeframe bullish structure

But:

    close > resistance

alone is not sufficient to claim a high-quality breakout.

Research progressively:

### Baseline

Price breakout only.

### Experiment 1

Breakout + volume.

### Experiment 2

Breakout + market structure.

### Experiment 3

Breakout + consolidation.

### Experiment 4

Breakout + retest.

Do not combine every filter immediately.

---

# 15. Range breakout + volume screener

One source screener is:

    Range High Breakout + Volume

Candidate logic should eventually involve:

- defined consolidation/range
- defined range high
- breakout above range high
- volume relative to baseline

Possible variables:

    range_high
    range_low
    range_width
    consolidation_length
    breakout_distance
    volume_ratio

Do not hard-code thresholds until tested.

---

# 16. 52-week-high breakout

Source material also discusses:

    52-week high breakout

Candidate:

    current price > rolling 52-week high

But test:

- intraday penetration versus closing breakout
- volume confirmation
- distance above high
- market regime
- sector strength

Do not assume a 52-week high breakout automatically produces a trade.

---

# 17. Higher-high / strong-structure screener

Candidate stocks may be selected where:

- swing structure is bullish
- HH/HL sequence is present
- price maintains structural support

Again, exact definitions must be deterministic.

---

# 18. Two-day strength / breakout concept

The source discusses a screener involving:

- two days of strong/high closes or range behavior
- relatively low/controlled volume in one version
- waiting for retracement

This is a hypothesis to test.

Do not implement the wording literally.

Define measurable characteristics such as:

- close location within daily range
- body/range ratio
- consecutive higher closes
- ATR-relative range
- volume behavior
- retracement depth

Then backtest.

---

# 19. Momentum → pause → continuation

One of the important price-action concepts:

    LARGE IMPULSE
          |
          v
       PAUSE
          |
          v
    CONTINUATION

Source terminology includes:

- large/strong candle
- smaller candles
- pause/consolidation
- continuation/retracement

Convert this into measurable variables.

Possible variables:

- impulse body / ATR
- impulse range / ATR
- number of pause candles
- maximum retracement
- volume contraction
- continuation breakout
- stop location

Do NOT assume the success percentages mentioned in the source are valid.

---

# 20. Rounding / cup-type structures

Source material discusses:

- rounding bottom
- cup-like structures
- cup and handle

These are difficult to define objectively.

They should initially be treated as research candidates.

Possible objective research:

- left-side decline
- rounded low
- recovery
- symmetry constraints
- handle/consolidation
- breakout above prior resistance

Do not implement a highly subjective pattern detector until there is a clear mathematical definition.

---

# 21. Other price-action patterns

Source material includes concepts such as:

- W pattern
- M pattern
- ascending triangle
- pole and flag
- pennant
- gap-up/gap-down context
- retest
- 50% retracement

These should NOT all be part of V1.

Keep them in the research backlog.

Implement only when:

1. pattern definition is deterministic
2. sufficient historical sample exists
3. independent backtest is possible

---

# 22. Screener versus strategy

This distinction is important.

A:

    SCREENER

answers:

    "Where should I look?"

A:

    STRATEGY

answers:

    "Should I trade it?"

A stock can pass the screener and still be rejected because:

- higher timeframe structure is poor
- resistance is too close
- stop is too wide
- risk cannot be sized
- sector is weak
- market regime is poor
- trigger did not occur

---

# 23. Intraday scanners from source material

The source material also discusses:

- gap-up reversal
- open-high
- open-low
- intraday sprout
- top losers
- 5m/15m intraday scanners

These are OUTSIDE V1.

Do not implement them into the swing strategy.

Keep them as future research only.

---

# 24. Sector strength

Another major concept learned from the source:

> strongest stock inside the strongest sector

The source discusses:

- NSE sectoral indices
- market-cap-weighted sector indices
- equal-weight sector baskets
- niche/subsector/TJI-type classifications
- comparing 1D / 1W / 1M / 3M / 6M performance
- looking at broad sector participation rather than one heavyweight dominating an index

Important:

"strongest sector" must become a numerical ranking.

---

# 25. Equal-weight sector baskets

Market-cap-weighted sector indices can be dominated by a few large constituents.

The source discusses building equal-weight baskets.

Conceptually:

For sector S:

    sector_return =
        average(constituent_returns)

Potential metrics:

- 1D return
- 1W return
- 1M return
- 3M return
- 6M return
- relative return versus NIFTY 50
- constituent breadth

Do not assume equal-weight is better.

Test whether it improves trading results.

---

# 26. Strongest stock inside strongest sector

Once sector strength is quantified, stock strength can be measured relative to:

- NIFTY
- sector
- peers

Potential variables:

    stock_return_1M
    stock_return_3M
    stock_return_6M
    relative_strength_vs_sector
    relative_strength_vs_nifty

But:

    strongest stock ≠ automatic buy

It must still pass the setup and risk engine.

---

# 27. Supertrend

The source material uses Supertrend on daily charts for stock inspection.

It also discusses Supertrend as a trailing mechanism.

For V1:

- Supertrend may be tested as a trailing exit.
- It should not automatically become the primary entry signal.
- Parameters must be configurable.
- Different parameters must be tested without overfitting.

---

# 28. Renko

The source discusses Renko/Renko-like charts as a way of visualizing price movement without the normal time emphasis.

For V1:

- Renko is NOT a core signal generator.
- It may be used for research/visualization later.
- Do not introduce it into live trading without separate validation.

---

# 29. Long-only rationale from source

The source material favors long-side equity trading and discusses practical limitations around short-side equity trading and F&O.

For V1:

    LONG ONLY

This is an engineering scope decision.

Do not treat the source's market-direction observations as guaranteed future performance.

---

# 30. Entry design

Every trade must have:

- symbol
- setup
- trigger
- entry price
- stop price
- risk/share
- quantity
- capital used
- target/exit method
- timestamp

Possible entry models to test:

1. breakout close
2. next-session confirmation
3. breakout retest
4. 15m confirmation after daily setup

Do not select the "best" model by intuition.

Compare them statistically.

---

# 31. Structural stop loss

Source principle:

> stop loss should be based on market structure, not an arbitrary percentage.

Possible stop anchors:

- daily swing low
- 1H swing low
- 15m swing low
- below breakout/retest failure level

The stop must represent a point where the original trade thesis is invalidated.

If the structural stop is too wide for the account risk limit:

    REJECT TRADE

Do not arbitrarily move the stop closer just to make position sizing possible.

---

# 32. Risk management

Source material discusses approximately:

- 1–2% risk per trade

Our initial V1 engineering baseline:

    DEFAULT RISK = 1% OF ACCOUNT EQUITY

For ₹10,000:

    maximum planned risk ≈ ₹100

This is a configurable default, not a guarantee or recommendation of profitability.

---

# 33. Position sizing

Core formula:

    risk_per_share = entry_price - stop_price

    quantity_by_risk =
        floor(max_rupee_risk / risk_per_share)

Then also check:

- available capital
- maximum position value
- broker constraints
- liquidity
- quantity validity
- portfolio exposure
- sector exposure

The risk engine must return:

    APPROVED

or:

    REJECTED + explicit reason

Never silently reject a trade.

---

# 34. Reward/risk

The source repeatedly discusses:

    1:3 risk/reward

Do NOT hard-code 1:3 as a proven optimum.

Test:

- 1.5R
- 2R
- 2.5R
- 3R
- trailing
- hybrid target + trailing

Compare:

- expectancy
- drawdown
- average R
- profit factor
- trade count
- robustness

---

# 35. Trailing exits

Source material emphasizes trailing because strong trends can continue for long periods.

Potential methods:

- Supertrend
- moving average
- swing-low trail
- ATR-based trail

Each must be independently tested.

V1 should avoid overly complicated partial exits.

---

# 36. Re-entry / pyramiding

Source material discusses:

- split entry
- re-entry
- pyramiding

Do NOT implement automatic pyramiding in V1.

Reason:

- small ₹10K account
- increased complexity
- increased correlation/exposure
- more difficult backtesting
- more difficult execution

Keep as future research.

---

# 37. Sector concentration

Source material warns about taking multiple stocks from the same sector because failures can correlate.

Implement configurable limits such as:

- maximum positions per sector
- maximum total risk per sector
- optional correlation-based exposure limit

Do not arbitrarily eliminate diversification opportunities.

Test the effect.

---

# 38. Market regime

Before stock selection, calculate a market regime.

Potential objective inputs:

- NIFTY trend
- NIFTY moving-average relationship
- market breadth
- volatility
- recent drawdown
- percentage of constituents above trend measures

Possible states:

    BULLISH
    NEUTRAL
    BEARISH

The labels are implementation conveniences.

The underlying calculations must be explicit.

Initially regime should be:

- context
- filter
- research feature

It must not be an LLM judgment.

---

# 39. Future global market context

Future V2/V3 may include:

- US markets
- Asian markets
- European markets
- USD/INR
- crude oil
- India VIX
- RBI decisions
- Fed decisions
- major government actions
- geopolitical events
- major corporate news
- sector news
- macroeconomic events

This is NOT V1.

When implemented:

1. Store source.
2. Store timestamp.
3. Normalize data.
4. Define objective rules.
5. Backtest effect.
6. Keep news/context separate from core signal generation.

AI-generated news sentiment must NOT override deterministic strategy logic.

---

# 40. Backtesting methodology

The supplied material emphasizes:

    formulate
        ↓
    implement
        ↓
    backtest
        ↓
    inspect results
        ↓
    analyze failures
        ↓
    controlled optimization
        ↓
    test on different stocks/instruments

This is the project's research loop.

---

# 41. Required backtest metrics

## Returns

- total return
- CAGR
- average trade
- profit factor
- expectancy

## Risk

- maximum drawdown
- drawdown duration
- worst trade
- worst consecutive losses
- volatility

## Trade quality

- number of trades
- win rate
- average winner
- average loser
- average R
- median R
- maximum favorable excursion
- maximum adverse excursion

## Robustness

Break results down by:

- stock
- sector
- year
- market regime
- volatility regime
- setup type

Also include:

- transaction costs
- slippage assumptions
- gap risk
- data quality
- corporate actions
- survivorship bias considerations

---

# 42. False-breakout analysis

A key lesson from the supplied backtesting material:

Do not simply add filters whenever a backtest loses.

Instead investigate:

    WHY DID THIS TRADE FAIL?

Possible categories:

- false breakout
- breakout without follow-through
- immediate reversal
- weak market regime
- weak sector
- volume failure
- resistance too close
- entry too late
- excessive stop
- gap against position
- retest failure
- setup misclassification
- data issue
- execution/slippage issue

Then form a hypothesis.

Example:

    Hypothesis:
    Breakouts with low relative volume fail more often.

Then test:

    baseline
    versus
    baseline + volume filter

Do not stack ten filters after seeing historical failures.

---

# 43. Walk-forward validation

Avoid optimizing everything on the same historical period.

Use:

    TRAIN
      ↓
    VALIDATION
      ↓
    TEST

And/or:

    WALK-FORWARD

Example concept:

    train → validate → move forward → train → validate → move forward

The final test period should remain as independent as practical.

---

# 44. Cross-stock validation

A strategy must not depend on one stock.

Test across:

- multiple NIFTY 50 stocks
- multiple sectors
- different market regimes
- bull periods
- bearish periods
- sideways periods

If a rule only works on one stock, it should not be considered validated.

---

# 45. Trade record

Every trade should record enough information for audit and research.

Minimum fields:

    trade_id
    strategy_version
    parameter_version
    timestamp
    symbol
    sector
    setup_type

    weekly_structure
    daily_structure
    hourly_structure
    fifteen_minute_trigger

    support_level
    resistance_level

    breakout_strength
    volume_ratio

    entry_price
    stop_price
    initial_risk_per_share
    quantity
    capital_used
    rupee_risk

    target_R
    exit_method

    exit_price
    exit_timestamp

    gross_pnl
    costs
    net_pnl
    R_multiple

    market_regime
    sector_strength
    stock_strength

    signal_status
    execution_status

    failure_reason
    notes

---

# 46. Failure analysis

Failure analysis is a first-class feature.

Do not report only:

    "Strategy lost money."

Instead classify failures.

Possible categories:

- FALSE_BREAKOUT
- NO_FOLLOW_THROUGH
- IMMEDIATE_REVERSAL
- WEAK_MARKET
- WEAK_SECTOR
- RESISTANCE_TOO_CLOSE
- LOW_VOLUME
- LATE_ENTRY
- GAP_AGAINST_POSITION
- RETEST_FAILURE
- STRUCTURE_INVALIDATED
- DATA_ERROR
- EXECUTION_ERROR
- SLIPPAGE_ERROR

Categories may expand based on evidence.

---

# 47. AI's role

AI/Codex is allowed to:

- write code
- debug code
- refactor
- analyze backtests
- analyze failure patterns
- propose hypotheses
- document decisions
- generate test cases
- investigate data issues
- explain metrics

AI is NOT allowed to:

- invent a trade
- override deterministic strategy
- bypass risk limits
- decide to buy because a chart "looks good"
- change parameters silently
- claim profitability without evidence
- fabricate market data
- fabricate backtest results
- fabricate broker responses
- enable live trading by itself

If AI proposes a new strategy rule:

    hypothesis
       ↓
    document
       ↓
    implement
       ↓
    backtest
       ↓
    validate
       ↓
    compare against baseline
       ↓
    decide whether to retain

---

# 48. Telegram interface

Telegram is the V1 interface.

The user wants two planned daily messages.

## SOD

Start-of-Day message should contain:

- date
- market regime
- sector strength
- candidate stocks
- setup
- trigger
- entry
- stop
- quantity
- rupee risk
- target/trailing method
- reason/rule evidence
- relevant overnight context
- WATCH / TRADE-IF-TRIGGERED status

Avoid excessive internal calculations.

## EOD

End-of-Day message should contain:

- trades executed
- entry
- exit
- P&L
- R multiple
- cumulative P&L
- ignored signals
- rejected trades
- rejection reasons
- failed setups
- failure categories
- system health
- API/data errors

---

# 49. Scheduling

Approximate workflow:

    PREMARKET
        ↓
    SOD generation
        ↓
    MARKET HOURS
        ↓
    signal monitoring
        ↓
    POST MARKET
        ↓
    position reconciliation
        ↓
    EOD report

Use:

- APScheduler
- Windows Task Scheduler if useful

Do not hard-code assumptions about holidays.

Use a proper trading calendar/data source when scheduling becomes production-critical.

---

# 50. Zerodha architecture

Broker-specific logic belongs under:

    src/broker/

Do not put Kite API calls directly inside strategy modules.

Broker abstraction should eventually support:

- authentication
- profile
- instruments
- historical candles
- quotes
- orders
- positions
- holdings
- order status
- reconciliation

Strategy should consume normalized internal data.

---

# 51. Historical data

Historical data should support:

- daily
- weekly
- 1H
- 15m

where available and appropriate.

Normalize data into consistent OHLCV format.

Account for:

- timezone
- missing candles
- holidays
- corporate actions
- instrument-token changes
- symbol changes
- duplicate data
- malformed records

Timezone:

    Asia/Kolkata

---

# 52. Database

SQLite is the initial database.

Potential logical entities:

- instruments
- candles
- sectors
- sector_members
- market_regime
- setups
- signals
- orders
- positions
- trades
- backtest_runs
- strategy_versions
- parameter_versions

Do not create a huge schema before the corresponding feature exists.

---

# 53. Strategy versioning

Every strategy change should have a version.

Example:

    strategy_version = "0.1.0"

If a trading rule changes:

- increment version
- document change
- backtest
- compare against previous version

Never silently change historical strategy definitions.

---

# 54. Configuration

Keep strategy parameters outside the code where practical.

Examples:

    account:
      starting_capital: 10000

    risk:
      risk_per_trade_pct: 1.0

    execution:
      mode: paper

    market:
      timezone: Asia/Kolkata

    strategy:
      universe: NIFTY50

Use sensible defaults.

Avoid magic numbers.

---

# 55. Testing

Every important strategy component must have unit tests.

Examples:

- swing-high detection
- swing-low detection
- HH/HL detection
- support/resistance
- breakout detection
- volume ratio
- position sizing
- risk rejection
- target calculation
- trailing stop
- sector ranking
- signal state transitions

Strategy changes should include regression tests where appropriate.

Never change tests merely to make a failing strategy pass.

---

# 56. Coding style

Prefer:

- Python 3.12
- type hints
- dataclasses/Pydantic models where appropriate
- pure functions for strategy calculations
- small modules
- small functions
- explicit names
- structured logging
- deterministic behavior
- clear errors

Avoid:

- giant classes
- giant files
- hidden global state
- hard-coded credentials
- magic numbers
- broker logic inside strategy
- Telegram logic inside strategy
- catch-all exception suppression
- unnecessary abstraction
- unnecessary infrastructure

---

# 57. Git discipline

Use small meaningful commits.

Examples:

    chore: initialize project scaffold

    feat: add kite historical data client

    feat: add market structure detector

    feat: add breakout screener

    feat: add risk engine

    feat: add backtest engine

    test: validate breakout detection

Never commit:

- .env
- API keys
- Telegram tokens
- access tokens
- secrets
- private credentials

---

# 58. Development milestones

## Milestone 0 — Environment

Already completed:

- Python 3.12
- virtual environment
- Git
- GitHub
- VS Code
- Codex

## Milestone 1 — Project foundation

Build:

- project scaffold
- .gitignore
- .env.example
- README
- settings
- logging
- SQLite foundation
- basic tests

Do NOT connect to Zerodha yet.

## Milestone 2 — Zerodha integration

Build:

- authentication
- profile
- instrument master
- historical candles
- normalized market data

No live orders.

## Milestone 3 — Historical dataset

Build:

- NIFTY 50 universe
- daily data
- weekly data
- 1H data
- 15m data
- data validation
- reproducible preprocessing

## Milestone 4 — Market structure

Build:

- swing highs
- swing lows
- HH/HL/LH/LL
- support/resistance
- trend
- consolidation

## Milestone 5 — Screener

Build:

- range breakout
- 52-week high
- higher-high structure
- two-day strength/retracement hypothesis
- liquidity filters

## Milestone 6 — Setup engine

Build and test:

- breakout
- breakout/retest
- momentum-pause-continuation

Research later:

- rounding/cup
- triangles
- flags
- W/M
- other patterns

## Milestone 7 — Multi-timeframe engine

Build:

    Weekly
      ↓
    Daily
      ↓
    1H
      ↓
    15m

## Milestone 8 — Risk engine

Build:

- 1% default account risk
- structural stop
- position sizing
- sector concentration
- portfolio risk

## Milestone 9 — Backtesting

Build:

- engine
- trade ledger
- realistic costs
- slippage
- metrics
- failure analysis
- walk-forward testing

## Milestone 10 — Paper trading

Build:

- live data
- signal generation
- simulated execution
- position tracking
- Telegram alerts

## Milestone 11 — SOD/EOD reporting

Build:

- SOD
- EOD
- system health

## Milestone 12 — Controlled live execution

Only after explicit user approval and adequate validation.

---

# 59. V1 baseline strategy

The initial baseline should be intentionally simple.

    UNIVERSE
        NIFTY 50

    DIRECTION
        LONG ONLY

    HIGHER TIMEFRAME
        Weekly + Daily

    CANDIDATE
        Daily range/high breakout

    CONFIRMATION
        Price structure + volume

    ENTRY
        Explicit breakout trigger

    STOP
        Structural swing-based stop
        refined using 1H/15m

    RISK
        1% account equity

    EXIT
        Compare fixed R targets
        versus trailing exits

    PORTFOLIO
        Avoid excessive sector concentration

    EXECUTION
        Paper / alert initially

    INTERFACE
        Telegram SOD + EOD

    VALIDATION
        Backtest + walk-forward + failure analysis

This baseline is NOT claimed to be profitable.

It is the initial research hypothesis.

---

# 60. What to do when requirements are ambiguous

If ambiguity affects:

- entry
- stop
- exit
- position size
- market regime
- universe
- execution

do NOT silently invent a trading rule.

Instead:

1. identify ambiguity
2. explain possible interpretations
3. propose objectively testable alternatives
4. ask the user if a strategic decision is required

For ordinary engineering decisions that do not affect strategy behavior:

- choose the simplest maintainable approach
- document it

---

# 61. What Codex must NEVER do

NEVER:

- place a live order merely to test connectivity
- commit secrets
- expose credentials
- enable live mode by default
- claim profitability without evidence
- fabricate historical data
- fabricate backtest results
- fabricate broker responses
- infer missing market data
- silently alter strategy rules
- bypass risk management
- optimize only for maximum historical P&L
- turn V1 into a complex platform
- treat LLM confidence as trading probability
- create discretionary trading logic hidden inside code

---

# 62. Definition of done for V1

V1 is technically complete only when:

1. Historical data can be downloaded and validated.
2. NIFTY 50 universe is reproducible.
3. Strategy rules are deterministic.
4. Strategy is versioned.
5. Backtest produces a reproducible trade ledger.
6. Costs and reasonable slippage are included.
7. Risk sizing works correctly.
8. Paper trading uses the same signal engine.
9. Telegram SOD/EOD reports work.
10. Failures are categorized.
11. Walk-forward/out-of-sample validation exists.
12. Live execution remains disabled until explicitly enabled.
13. System can explain why every candidate was accepted or rejected.

---

# 63. Development rule for Codex

Work incrementally.

At the beginning of each task:

1. Read this AGENTS.md.
2. Inspect the existing repository.
3. Identify the current milestone.
4. Do not implement future milestones unnecessarily.
5. Preserve existing behavior.
6. Add tests with important logic.
7. Run relevant tests.
8. Report:
   - files changed
   - what changed
   - tests run
   - test results
   - assumptions made
   - anything requiring user/strategy approval

Do not implement a new trading rule merely because it appears plausible.

The goal is not to create the largest trading system.

The goal is to create a system whose behavior can be understood, reproduced, tested, challenged, and eventually trusted.