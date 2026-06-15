# Backlog — quant improvements deferred from initial build

These are real gaps in the current code that practitioners would flag.
Not implemented yet because (a) the existing strategies haven't earned
their place in live trading, and (b) building infrastructure for
strategies that will be retired is wasted effort.

## Validation infrastructure

- [x] **Combinatorial Purged K-Fold CV (CPCV)** — López de Prado AFML Ch. 7.4.
  Built in `core/cpcv.py`. Smoke test: 6 groups, k=2 → 15 combinations,
  TSMOM mean OOS Sharpe 0.67 across 10000 OOS obs (vs walk-forward's
  1630 OOS obs).
- [ ] **Reproducibility hashing** — snapshot input data hash + library
  versions with each evidence-ledger claim. Without this, results drift
  silently when Binance updates historical data.
- [ ] **Bonferroni / FDR correction** — when running CV across many parameter
  combos, deflated Sharpe's `num_trials` is a guess. Track actual trial count.

## Strategy machinery

- [x] **Triple-barrier exit labels** — López de Prado AFML Ch. 3.
  Built in `core/exits.py`. Smoke test: TSMOM(30,90) → 62% win rate,
  +1.22% avg, 4-day avg hold across 21 trades.
- [x] **Fractional Kelly sizing** — Carver (2015) standard 0.25×.
  Built in `core/sizing.py`. Smoke test: TSMOM full Kelly = 2.29x,
  fractional clipped to 0.20.
- [x] **Multi-asset cross-sectional momentum** — built in
  `research/cross_sectional.py`. Best result: xs_momentum_30 with
  alpha_t = 2.35 (only signal that crossed alpha threshold).
- [x] **Meta-labeling** — built in `core/meta_labeling.py`. Smoke test on
  TSMOM events: 147 events, base rate 47%, mean CV accuracy 50.8% =
  +3.9 percentage points lift. Top predictive feature: price_vs_sma200
  (0.17 importance).

## Risk + regime

- [x] **Tail risk overlay** — Daniel/Moskowitz (2016) "Momentum Crashes".
  Built in `core/tail_overlay.py`. Vol-scales a strategy by its own EWMA
  realized vol; defaults to 1.0 in calm regimes, shrinks during crashes.
- [x] **Drawdown-based position scaling** — Carver (2015) Systematic
  Trading Ch. 9. Built in `core/drawdown_scale.py`. Continuous scalar
  from 1.0 (≤10% DD) ramping to 0.0 (≥30% DD).
- [ ] **HMM regime detection** — Hamilton (1989). Two- or three-state Markov
  switching on BTC vol or returns; cleaner than 200-day SMA.
- [x] **GARCH(1,1) volatility** — Engle (1982), Bollerslev (1986). Built in
  `core/garch_vol.py`. BTC fit: alpha=0.078, beta=0.899, persistence=0.977
  (highly persistent). Forecast 37% vs realized 33%.
- [x] **Hierarchical Risk Parity** — López de Prado (2016). Built in
  `core/hrp.py`. BTC/ETH/SOL HRP weights: 0.49/0.27/0.24, portfolio variance
  reduced 15% vs equal-weight.
- [x] **Ledoit-Wolf shrinkage covariance** — Ledoit/Wolf (2003). Built in
  `core/shrinkage.py`. Standard upgrade to sample covariance for portfolio
  optimization stability.

## Microstructure / execution

- [ ] **Almgren-Chriss impact model** — only matters at $1M+ position size.
  Defer until book size warrants.
- [ ] **Limit-order queue position model** — for resting limit orders.
- [ ] **TWAP / VWAP execution** — slice large orders across time.
- [ ] **Slippage model from real fill data** — current 5bps assumption is a
  guess. Once we have live fills, fit an empirical slippage distribution.

## Signal sources (per CLAUDE.md alpha ranking)

- [ ] ETF flow signal (Farside scrape)
- [ ] Exchange in/outflows (CryptoQuant or Glassnode)
- [ ] Stablecoin supply expansion (CoinGecko)
- [ ] Whale wallet tracking (Etherscan, curated address list)
- [ ] Cross-exchange basis (Binance vs Kraken vs Coinbase same pair)

## Operations

- [ ] Daily P&L attribution (port from stocks `daily_attribution.py`)
- [ ] Champion-challenger A/B harness (port from stocks `champion_challenger.py`)
- [ ] Survivorship-bias check on historical pair lists
- [ ] Wakelock setup as `Crypto_*` Windows scheduled task

## Going-live gate

A strategy can flip `VALIDATED = False` -> `True` only after passing,
in this order:

1. `evaluate_strict()`: walk-forward mean OOS Sharpe > 0.5, min fold > 0
2. Factor decomposition: |alpha t-stat| > 2.0
3. Deflated Sharpe on OOS-concatenated returns: DSR > 0.95 AND |t-stat| > 3.0
4. 60+ closed paper trades on the live broker (per CLAUDE.md)
5. Manual review of evidence ledger entries

Only THEN is `BINANCE_TESTNET=false` flipped, real keys provisioned,
and small allocations sized via fractional Kelly.
