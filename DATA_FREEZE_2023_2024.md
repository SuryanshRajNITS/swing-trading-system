# Research Data Freeze: 2023-2024

## Frozen scope

- Research period: 2023-01-01 through 2024-12-31.
- Daily/weekly warmup begins 2022-07-01 where manifested data exists.
- Hourly warmup requested from 2022-10-01; actual first bars depend on security/listing/provider availability.
- PIT universe: `NIFTY50_PIT_TRADABLE`, resolved by effective-dated membership and symbol history; eligibility is date-specific and excludes non-tradable index placeholders.
- 2025 and 2026 remain untouched validation/OOS periods.
- Data freeze status: `PASS`.
- Future strategy results must reference freeze version `NIFTY50_PIT_2023_2024_V1` and the manifest hash set in `data/processed/manifests/data_freeze_2023_2024.json`.

## Dataset status

- PIT securities expected across the period: 55 unique identities.
- Daily usable: 55 identities with manifested daily candles overlapping their PIT membership.
- Weekly: derived locally from daily candles using Monday-Sunday calendar-week grouping; OHLCV uses observed sessions only, no synthetic rows. A week becomes available only after its final session close.
- NIFTY 50 benchmark: 495 daily price-return sessions, 2023-01-02 through 2024-12-31; no pre-2023 benchmark warmup is present.
- Hourly: 52 usable 60-minute identities after exclusions; Kite-native candle start timestamps preserved.
- Hourly unavailable: HDFC (`NO_CURRENT_TOKEN`); LTIM (`FAIL_CLOSED_IDENTITY_BOUNDARY`).
- Hourly excluded: TATAMOTORS / NSEEQ-0055 (`HOURLY_IDENTITY_PRICE_MISMATCH`). Its daily/weekly data remains available where PIT membership and daily coverage permit; any 1H-dependent opportunity must be `HOURLY_DATA_UNAVAILABLE`. No TMPV historical hourly substitution is permitted.
- Quarantined provider-invalid hourly candles: 3. No interpolation, synthetic bars, or silent substitution is used.

## Special sessions

Hourly session validation uses explicit exchange overrides. Within 2023-2024 these include 2023-11-12 and 2024-11-01 Muhurat sessions; 2024-01-20 special regular session; and 2024-03-02 and 2024-05-18 special split sessions. Exchange trading windows remain separate from observed Kite bar-start grids. No timestamps are snapped and no bars are synthesized.

## Cross-timeframe price-integrity findings

Daily OHLC was compared with each same-day hourly aggregate (first open, maximum high, minimum low, last close). Strict OHLC equality was measured at ₹0.05, but failures are treated as provider OHLC variance and are not corrected.
At the strict ₹0.05 tolerance, 52 of the 52 usable identities have at least one mismatched session (23149 security-session pairs).
The 52 usable hourly series are classified as `PRICE_SCALE_VALIDATED_WITH_PROVIDER_OHLC_VARIANCE` under the persistent-scale rule (median hourly-last-close / daily-close differs from 1.0 by more than 1%). Persistent scale divergence IDs: none.

| Field | Median absolute difference (₹) | Median absolute difference (%) | P95 absolute difference (%) | Maximum (%) | >0.25% | >0.50% | >1.00% | >2.00% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| open | 0.000000 | 0.000000 | 0.000000 | 14.484733 | 55 | 18 | 3 | 1 |
| high | 0.000000 | 0.000000 | 0.011504 | 14.462889 | 48 | 16 | 7 | 3 |
| low | 0.000000 | 0.000000 | 0.019857 | 2.861576 | 48 | 15 | 7 | 3 |
| close | 1.150000 | 0.103355 | 0.382335 | 3.269631 | 3658 | 502 | 53 | 6 |

All per-security distributions and every field/session discrepancy over 1% are in `data/processed/manifests/cross_timeframe_discrepancy_stats_2023_2024.csv` and `data/processed/manifests/cross_timeframe_discrepancies_over_1pct_2023_2024.csv`. The >1% report contains 70 field-level rows across 68 security-date pairs.

## Provenance and limitations

- PIT membership, symbol intervals, identity and source records come from the versioned `data/reference/` ledgers; security tokens are never treated as security identity.
- Daily equity candles use the manifested source per security (Kite historical data, with official NSE archives where manifested); the exact dated data prefix and source-manifest slice hashes are in the freeze JSON.
- Weekly candles are deterministic derivations from the frozen daily prefix using `src/data/weekly.py`; there is no separate persisted weekly market-data dataset.
- Hourly candles use Zerodha Kite historical data. The freeze hash includes only 52 hourly identities allowed for research, the explicit exclusion file, quarantine file, and data-source manifest slice.
- NIFTY benchmark is the user-provided official NIFTY 50 Historical PR export combined at `data/raw/reference/nifty50_daily.csv`. It is a price-return series, not a total-return series.
- OHLC discrepancies can exceed ₹0.05, including isolated larger deviations. Prices have not been adjusted to force aggregation equality; consult the >1% list before interpreting affected dates.
- HDFC and LTIM have no usable hourly series for the frozen window. TATAMOTORS hourly data is excluded due a roughly 0.689 median close ratio versus its daily series.
- This freeze is a research-data integrity checkpoint, not a claim that strategy results are profitable or that every provider candle is error-free.
- Full test suite at freeze creation: 187 passed, 0 failed, 0 skipped (Python 3.12.10, pytest 8.4.2).

Freeze manifest: `data/processed/manifests/data_freeze_2023_2024.json`.
