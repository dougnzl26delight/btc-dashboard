# Monthly OOS Revalidation - 2026-05-10

## [1] Max-history single backtest

- Window: 2020-01-21 -> 2026-05-10 (2301 days)
- Final equity: $4,109,074
- Annualized:   +80.29%
- Sharpe:       +1.40
- Max DD:       +40.15%
- Trades:       163

## [2] Walk-forward 5 OOS folds

| Fold | Window | Days | Annlzd | Sharpe | DD | BAH |
|---|---|---|---|---|---|---|
| 1 | 2020-01-21 -> 2021-04-24 | 459 | +301.3% | +2.80 | 25% | +462.1% |
| 2 | 2021-04-25 -> 2022-07-28 | 459 | +76.7% | +1.41 | 33% | -18.7% |
| 3 | 2022-07-29 -> 2023-10-31 | 459 | +25.3% | +0.72 | 40% | +7.3% |
| 4 | 2023-11-01 -> 2025-02-02 | 459 | +98.7% | +1.54 | 38% | +145.9% |
| 5 | 2025-02-03 -> 2026-05-10 | 461 | -7.3% | -0.12 | 33% | -49.7% |

- Mean OOS Sharpe:    +1.27  (baseline: +1.27)
- Latest fold Sharpe: -0.12
- Mean OOS annualized: +98.92%
- Folds positive:     4/5

- **ALERT [RECENT_NEGATIVE]**: Latest fold Sharpe -0.12 < 0

## [3] Parameter neighborhood check (3x3 grid)

| ATR | Cap | Annlzd | Sharpe | DD |
|---|---|---|---|---|
| 3.0 | 0.10 | +50.9% | +1.16 | 40% |
| 3.0 | 0.15 | +54.6% | +1.18 | 40% |
| 3.0 | 0.20 | +54.6% | +1.17 | 41% |
| 4.0 | 0.10 | +74.9% | +1.38 | 37% |
| 4.0 | 0.15 | +80.3% | +1.40 | 40% | (production)
| 4.0 | 0.20 | +47.7% | +1.08 | 41% |
| 5.0 | 0.10 | +55.2% | +1.18 | 38% |
| 5.0 | 0.15 | +58.1% | +1.26 | 39% |
| 5.0 | 0.20 | +58.7% | +1.21 | 39% |

- Production params remain at local optimum.

## Reminder

This is REPORTING ONLY. Do NOT change production parameters
based on one monthly drift signal. Per Charter:
- Wait for 2 consecutive months of drift before considering change
- All parameter changes require fresh backtest + 60-day paper test