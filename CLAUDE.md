# Crypto Trading System

## Project context

Separate from `C:\Users\dougn\Documents\Trading` (which is the stocks/options
system on Alpaca paper). This project is a clean-slate crypto trading rig.

**User**: solo retail trader, NZ timezone (NZST/NZDT, UTC+12/+13).
Currently running a 10-strategy quantitative stock system on Alpaca paper.
Same JS-style discipline expected here: multi-paper-grounded, regime-gated,
honest evidence collection, cap-controlled risk.

## Key differences from the stocks system (don't copy-paste assumptions)

| Dimension                | Stocks system (existing)         | Crypto system (this one)                      |
|--------------------------|----------------------------------|-----------------------------------------------|
| Market hours             | US 09:30-16:00 ET, weekdays only | 24/7 — no off-window                          |
| Broker                   | Alpaca paper                     | Binance NZ (decided 2026-05-09; FSP1003864)   |
| Universe                 | ~3000 US equities                | BTC/ETH/SOL + maybe top 20 alts; not 3000     |
| Daily volatility         | ~1-2% on liquid names            | ~3-8% on BTC, 5-15% on alts                   |
| Stop tolerance           | MAX_LOSS = 8%                    | MAX_LOSS = 15-20% (vol-adjusted)              |
| Signal sources           | SEC EDGAR, congress, 8-K, Form 4 | On-chain wallet moves, funding rates, basis,  |
|                          |                                  | exchange flows, ETF flows (BTC/ETH spot ETFs) |
| Tail hedge               | SPY puts                         | BTC puts (Deribit), short perps, or stablecoin|
| Cron schedule windows    | Aligned to US session            | Continuous; needs idle-window detection       |
| Calendar effects         | FOMC, CPI, NFP, earnings         | Halving, ETF approval/inflow days, CPI still  |
| Position cap             | $5k per name on $100k book       | Wider per-name (5-10%) due to fewer names     |

## Constraints / guardrails

- **Paper first.** Always start on a paper / sandbox account. No live
  capital until the system has 60+ closed trades on paper.
- **Honest deflated Sharpe.** Same discipline as the stocks system —
  every strategy gets evaluated against Bailey/LdP DSR + Harvey/Liu/Zhu
  t > 3.0 hurdle before sizing up.
- **Wakelock.** This machine sleeps if not protected. Re-use the wakelock
  approach from `Documents\Trading\wakelock.py` (Windows
  SetThreadExecutionState) but install a SEPARATE scheduled task with a
  unique name (e.g. `Crypto_wakelock`) so the two don't fight.
- **No code copy-paste from the stocks system without justification.**
  Each pattern must earn its place here on its own merits — different
  market behaviour means many idioms don't transfer.

## Rough alpha sources to investigate (NOT to build day 1)

In rough order of edge-likelihood for retail crypto:

1. **ETF flow signals** (BlackRock IBIT, Fidelity FBTC, etc. daily inflows)
   — published by Farside Investors, lag ~1 day, simple regime signal.
2. **Funding rate divergence** — when perp funding diverges from basis,
   short the rich side. Free data on Binance/Bybit.
3. **Exchange inflows / outflows** — large BTC moving onto exchange =
   sell pressure. CryptoQuant free tier or Glassnode trial.
4. **Stablecoin supply expansion / contraction** — USDT/USDC market cap
   changes correlate with crypto flows. CoinGecko free.
5. **Whale wallet tracking** — top 100 BTC wallets, ETH whales. Etherscan
   + manual address curation. Slow to build but free.
6. **Basis trade** — perp vs spot funding arbitrage. Requires 2 venues
   (one for spot, one for perp), but mechanically simple.
7. **Long-term momentum** — TSMOM works in crypto too (Asness 2012).
   12-month return positive → long, negative → cash/short.
8. **Mean reversion at multi-σ extremes** — same OU-style stat arb that
   works in stocks works in BTC pairs (e.g. ETH/BTC, BNB/BTC).

Skip until evidence:
- "AI sentiment" off Twitter/Reddit (too noisy for retail).
- DeFi yield farming (separate workflow, not really trading).
- NFT signals (illiquid, manipulation-prone).

## Project structure (proposed; not enforced)

```
CryptoTrading/
├── CLAUDE.md             ← this file
├── README.md             ← short repo overview
├── .env.example          ← API key placeholders
├── .env                  ← (gitignored) actual keys
├── .gitignore
├── requirements.txt      ← deps
├── core/
│   ├── broker.py         ← Coinbase/Kraken adapter (broker abstraction)
│   ├── data.py           ← price + on-chain data router
│   └── risk.py           ← shared risk caps + regime gates
├── signals/
│   ├── etf_flow.py       ← BlackRock/Fidelity ETF inflows
│   ├── funding_basis.py  ← funding rate divergence
│   ├── tsmom.py          ← long-term momentum
│   └── ...
├── strategies/
│   ├── btc_trend.py      ← BTC TSMOM
│   ├── eth_btc_pair.py   ← stat arb
│   └── ...
├── ops/
│   ├── wakelock.py       ← reuse pattern from stocks system
│   ├── watchdog.py       ← health check
│   └── alerts.py         ← Telegram/email
└── tests/
```

## What to do FIRST (when actually starting work)

1. ~~**Pick a broker.**~~ Done 2026-05-09 — Binance NZ. Coinbase live is
   broken for NZ (no NZD on/off-ramp). Kraken's spot test env is qualified-
   clients-only. Binance NZ is FSP-registered, has a full public testnet,
   and the largest universe.
2. **Set up `.env`** with API keys. Copy `.env.example` to `.env`, fill in
   `BINANCE_API_KEY`/`BINANCE_API_SECRET` (testnet keys are sufficient for
   now — get them at https://testnet.binance.vision).
3. ~~**Build `core/broker.py`**~~ Done 2026-05-09 — `paper`/`live` modes,
   testnet support, generic quote currency in `PaperState`.
4. **Run paper for 30 days BEFORE adding strategies.** Just BTC buy-and-hold
   on Binance, prove the wiring works end-to-end.
5. **Build `core/data.py`** — first signal source (ETF flow, simplest to
   start).
6. **Then** add the first signal-driven strategy (TSMOM or funding basis),
   and only then iterate.

## Notes on tooling

- Use `python` (3.11+) — same as stocks system.
- Use `ccxt` for unified broker access (Coinbase/Kraken/Binance) — it's
  the standard.
- Use `requests` for ETF flow scraping (Farside has no API but the HTML
  is parseable).
- Cache aggressively, same TTL pattern as stocks system's `signal_cache.py`.
- Email alerts via existing SMTP setup (re-use from stocks `alerts.py`).

## Don'ts

- Don't run more than one cron at a time on this machine. The stocks
  system already has 30+ scheduled tasks. Crypto crons should be
  prefixed `Crypto_` to avoid name collision.
- Don't share data caches between projects.
- Don't auto-trade until paper-verified for 30+ days.
- Don't build all 8 strategies before 1 is profitable.

## Status

**Day 0+** as of 2026-05-09 — broker adapter shipped (`core/broker.py`),
tested live against Binance mainnet ticks (paper-bought USDT 1k of BTC,
state persisted). Next: 30-day paper buy-and-hold to prove wiring before
adding any signal-driven strategy.

**Broker:** Binance NZ. Quote currency: USDT. Live testnet routing via
`BINANCE_TESTNET=true` env var. Paper mode is local fill simulator against
live mainnet ticks (no keys required).
