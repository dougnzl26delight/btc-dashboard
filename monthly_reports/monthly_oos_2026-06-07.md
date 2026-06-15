# Monthly OOS Revalidation - 2026-06-07

## [1] Max-history single backtest

- Window: 2020-02-19 -> 2026-06-07 (2300 days)
- Final equity: $4,096,650
- Annualized:   +80.25%
- Sharpe:       +1.40
- Max DD:       +40.15%
- Trades:       159

## [2] Walk-forward 5 OOS folds

| Fold | Window | Days | Annlzd | Sharpe | DD | BAH |
|---|---|---|---|---|---|---|
| 1 | 2020-02-19 -> 2021-05-23 | 459 | +240.9% | +2.28 | 32% | +250.6% |
| 2 | 2021-05-24 -> 2022-08-26 | 459 | +95.1% | +1.68 | 33% | -20.8% |
| 3 | 2022-08-27 -> 2023-11-29 | 459 | +70.4% | +1.24 | 40% | +59.4% |
| 4 | 2023-11-30 -> 2025-03-03 | 459 | +39.4% | +0.93 | 39% | +46.0% |
| 5 | 2025-03-04 -> 2026-06-07 | 460 | -7.3% | -0.12 | 33% | -53.8% |

- Mean OOS Sharpe:    +1.20  (baseline: +1.27)
- Latest fold Sharpe: -0.12
- Mean OOS annualized: +87.69%
- Folds positive:     4/5

- **ALERT [RECENT_NEGATIVE]**: Latest fold Sharpe -0.12 < 0

## [3] Parameter neighborhood check (3x3 grid)

| ATR | Cap | Annlzd | Sharpe | DD |
|---|---|---|---|---|
| 3.0 | 0.10 | +49.1% | +1.15 | 40% |
| 3.0 | 0.15 | +52.7% | +1.16 | 40% |
| 3.0 | 0.20 | +52.8% | +1.15 | 41% |
| 4.0 | 0.10 | +74.9% | +1.39 | 37% |
| 4.0 | 0.15 | +80.3% | +1.40 | 40% | (production)
| 4.0 | 0.20 | +47.6% | +1.09 | 41% |
| 5.0 | 0.10 | +56.5% | +1.21 | 38% |
| 5.0 | 0.15 | +59.4% | +1.29 | 39% |
| 5.0 | 0.20 | +60.1% | +1.24 | 39% |

- Production params remain at local optimum.

## Reminder

This is REPORTING ONLY. Do NOT change production parameters
based on one monthly drift signal. Per Charter:
- Wait for 2 consecutive months of drift before considering change
- All parameter changes require fresh backtest + 60-day paper test