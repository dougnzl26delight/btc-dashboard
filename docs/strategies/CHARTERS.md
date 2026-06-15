# Strategy Charters

One page per sleeve. Forces articulation of edge in 2 sentences. If a sleeve
can't answer "What's the edge?" plainly, it gets cut from the rig.

**Last updated:** 2026-05-28
**Maintained by:** Operator (solo). Review on every parameter change.

---

## 1. BAH BTC (Buy-and-Hold Bitcoin sleeve)

**Edge hypothesis (2 sentences):** Bitcoin's 4-year halving cycle has produced
positive 4-year returns in every cycle since 2012, driven by predictable supply
shocks. Holding through cycle drawdowns historically outperforms market-timing
attempts after costs.

**Entry rules:**
- Buy on first cycle if no position
- Top-up to target notional monthly OR when drift exceeds ±5pp from target
- Target: 10% of bankroll (reduced from 20% in 2026-05-28)

**Exit rules:**
- No exit during normal operation (passive sleeve)
- **Cycle-aware overlay:** Halve position if Mayer Multiple < 0.7 AND lower-low for 30 days
- Sleeve circuit breaker: scale 50% at -5% sleeve DD, 25% at -7.5%, paused at -10%

**Hypothesized post-cost Sharpe:** 0.5-0.9 (cycle-dependent, much higher in bull years)
**Expected max DD:** 25-40% in bear cycles. Tolerable BECAUSE of cycle thesis.
**Kill criteria:**
- 6 consecutive months of -10% returns (regime change)
- 4-year cycle pattern breaks (no new ATH within 18 months post-halving)

**Risks:**
- Regulatory shock (e.g., US ETF redemption suspension)
- 2035 halving cycle decay (returns shrinking each cycle)
- Long-term concentration risk if BTC dominance keeps falling

---

## 2. XSMOM (Cross-sectional momentum)

**Edge hypothesis:** Within a basket of crypto majors, recent winners
outperform recent losers over 14-day periods because retail/institutional
attention follows performance. Equally long top-N / short bottom-N captures
this dispersion.

**Entry rules:**
- Universe: 8 major pairs (BTC, ETH, SOL, BNB, AVAX, LINK, DOT, ATOM)
- Rank by 14-day return
- Long top 2 (20% weight each), short bottom 2 (-10% each) → 30% gross exposure
- Rebalance every 14 days
- 30% of paper account allocation, scaled by drawdown CB + Sharpe gate

**Exit rules:**
- Rebalance closes/opens positions on 14-day cycle
- No intraday stops (rebalance is the exit)

**Hypothesized post-cost Sharpe:** 0.3-0.5 (low — costs eat momentum signals fast)
**Expected max DD:** 15-25%
**Kill criteria:**
- 90-day live Sharpe < 0 → pause and reassess
- Correlation to BTC > 0.95 → strategy no longer additive

**Risks:**
- Pair-selection lookahead (Binance delistings/listings)
- Funding cost on short legs during persistent uptrends
- Regime-dependent: works in trending markets, fails in chop

---

## 3. Pro_trend (Trend follower with pyramiding, v5)

**Edge hypothesis:** Crypto exhibits strong serial correlation when above
its 200-day SMA — most large moves continue. Donchian-20 breakouts confirmed
by positive TSMOM_30 + MACD_hist filter the false breakouts.

**Entry rules:**
- LONG: price > SMA200 AND high ≥ Donchian-20 high AND TSMOM_30 > 0 AND MACD_hist > 0
- SHORT (re-enabled 2026-05-28): price < SMA200 AND low ≤ Donchian-20 low AND TSMOM_30 < -0.10 AND MACD_hist < 0
- ATR-based position sizing, max 30% gross per pair
- Pyramid: add unit on ATR breakout in trend direction

**Exit rules:**
- Trail stop at 3-ATR from extreme
- Hard exit on opposite Donchian breach
- Sleeve circuit breaker (5/7.5/10% tiers)

**Hypothesized post-cost Sharpe:** 0.8-1.4 (highest of all sleeves; survived cost model)
**Expected max DD:** 20-30%
**Kill criteria:**
- 60-day live Sharpe < 0.5
- Hit rate < 35% (typical trend system runs 40-50%)
- 4 consecutive losing trades on same pair → pause that pair

**Risks:**
- Whipsaw losses in chop (the SMA200 filter mitigates but doesn't eliminate)
- Short squeezes in bear-relief bounces (the v5 short filter mitigates)
- Pyramid sizing can over-concentrate single pair

---

## 4. Basis arb (Funding rate capture)

**Edge hypothesis:** When perp futures funding rate is persistently positive,
shorting perp + buying spot captures the funding income while remaining
market-neutral. Funding > round-trip costs = positive carry.

**Entry rules:**
- 8h-funding > +5bp (annualized ~5.5%) sustained
- Open: long spot + short perp at same notional
- Max 5 concurrent positions, $30k each (was $30k paper baseline; live sized smaller)

**Exit rules:**
- Funding flips negative for 2 consecutive 8h periods → close
- 30-day hold cap
- Sleeve CB and global VaR cap apply

**Hypothesized post-cost Sharpe:** 1.0-2.0 IF executed cleanly (pure arb, low vol)
**Expected max DD:** < 5% (market-neutral by construction)
**Kill criteria:**
- Funding regime change (sustained negative): pause sleeve
- Spot-perp basis blowout (>1%): close all positions

**Risks:**
- Liquidation risk on perp leg if margin insufficient
- Spot-perp price divergence during exchange stress
- Funding payments are NOT guaranteed (exchange policy changes)

---

## 5. Oversold bounce (Tactical long mean-reversion)

**Edge hypothesis:** When 3+ crypto majors simultaneously hit RSI < 25, the
universe is in regime-wide capitulation. Mean-reversion bounces of 20-50%
occur within 1-3 weeks in 65-70% of cases historically.

**Entry rules:**
- Cross-section: ≥3 pairs at RSI(14) < 25 in same scan
- Enter top 5 most-oversold names, equal-weight
- Total basket = 15% of bankroll (≈3% each), spot long
- Stop-loss: 2% below recent 20-day low at entry

**Exit rules:**
- RSI > 50 (recovered)
- +20% from entry (target)
- Stop-loss hit
- 30-day time cap

**Hypothesized post-cost Sharpe:** 0.5-1.0 (high R:R per trade, infrequent)
**Expected max DD:** 10-15% (single basket can lose -15% if all positions hit stops)
**Kill criteria:**
- 5 consecutive losing baskets → strategy decay
- Hit rate falls below 40% over 12 trades
- Sleeve CB

**Risks:**
- "Catching falling knives" in structural breakdowns
- Bear-market reliefs DON'T mark the bottom — sized to be wrong sometimes
- Universe correlation = all 5 positions can lose together

---

## 6. Overbought fade (Tactical short mean-reversion)

**Edge hypothesis:** In confirmed bear regime (BTC < SMA200 AND 14d return
< -10%), relief rallies that push 3+ majors to RSI > 70 are typically failed
bounces. Shorting these captures the reversal back down to the cycle bottom.

**Entry rules:**
- Hard regime gate: BTC must be in BEAR (below SMA200 + 14d ret < -10%)
- Cross-section: ≥3 pairs at RSI(14) > 70
- Short top 3 most-overbought on perp, equal-weight
- Total basket = 10% of bankroll (≈3.3% each)
- Stop: 2% above recent 10-day high

**Exit rules:**
- RSI < 50
- +15% gain (price dropped 15%)
- Stop hit
- 14-day time cap

**Hypothesized post-cost Sharpe:** 0.4-0.8 (lower than longs due to squeeze risk)
**Expected max DD:** 15-20% (squeezes can be vicious)
**Kill criteria:**
- 4 consecutive stopped-out trades
- Regime flips to BULL → permanently disable until new bear

**Risks:**
- Short squeezes in oversold rebounds — primary failure mode
- Funding cost during prolonged hold
- Borrowable supply on perp (Binance can run out on some alts)

---

## 7. Spot orchestrator (multi-strategy ensemble)

**Edge hypothesis:** A weighted blend of 5 weak-edge signals (TSMOM, short-term
momentum, vol breakout, funding contrarian, diverse_mom_ethbtc) has lower
variance than any single signal while preserving most of the directional edge.

**Entry rules:**
- Long-only (gated)
- Portfolio vol target 20% annualized
- Risk parity (inverse-vol per pair)
- Concordance dampener: scale down when too many signals agree (regime-change risk)
- Daily VaR limit: 1% of equity (hard gate)

**Exit rules:**
- Signal flip to negative or zero → close
- Bear regime → no new entries (existing positions ride)
- Sleeve CB

**Hypothesized post-cost Sharpe:** 0.2-0.5 (low — ensemble of weak signals)
**Expected max DD:** 15-25%
**Kill criteria:**
- 60-day live Sharpe < 0
- Concordance permanently > 0.95 (signals collapsed to one)

**Risks:**
- Most underlying signals showed < 0 post-cost Sharpe — heavy lift to net positive
- May be culled per cost-model re-evaluation
- Daily VaR check could starve it of capital in volatile regimes

---

## Review schedule

- **Weekly (every Sunday):** scan scorecard, flag any sleeve below threshold
- **Monthly (1st of month):** full walk-forward retraining of parameters
- **Quarterly:** complete charter review, prune sleeves that haven't earned capital

## Decision rules

A sleeve gets PROMOTED to full allocation when:
- 90-day live Sharpe > 1.0 after costs
- Max DD < 10%
- Hit rate matches or exceeds backtest by 10pp

A sleeve gets DEMOTED to half size when:
- 60-day live Sharpe falls below half of backtest
- Cumulative cost drag > 50% of gross return

A sleeve gets KILLED when:
- 90-day live Sharpe < 0
- Strategy charter cannot be defended (edge can't be articulated)
- Operator can't explain a single losing trade

---

**Last principle:** if you can't write your edge in 2 sentences for a sleeve,
you don't have one. Cut it.
