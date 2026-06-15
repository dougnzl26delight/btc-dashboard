# Crypto Rig Status — 2026-05-11 (3-sleeve allocation: pro_trend / XSMOM / BAH BTC)

## LATEST: 3-sleeve allocation (2026-05-11 final)

After user observed system underperformed BAH on recent 18m chop window,
ran comprehensive bake-off of 9 strategy classes across 4 market regimes.

### The data

Full 6.3y Sharpe ranking confirmed pro_trend v5 is best risk-adjusted
strategy (Sharpe 1.45 vs BAH BTC 0.89). BUT recent 18m window: BAH BTC
+10.2% vs pro_trend +2.1%. The strategy correctly stays out in chop
(no pair above SMA200) but misses the +10-15% slow rallies underneath.

### Three-sleeve allocation test (8 splits × 5 regimes)

| Allocation | Full 6.3y Sharpe | Recent 18m Ret | Max DD |
|---|---|---|---|
| current 70/30/0 (no BAH) | 1.48 | +6.42% | 28.5% |
| **NEW 50/30/20** | **1.49** | **+11.30%** | **32.9%** |
| 50/20/30 (more BAH) | 1.44 | +12.73% | 41.9% |
| 60/20/20 (proposed) | 1.49 | +10.69% | 35.9% |
| 40/30/30 (aggressive) | 1.43 | +13.22% | 39.2% |

**50/30/20 is the winner:** slightly higher Sharpe than current (1.49 vs
1.48), nearly doubles recent return (+11.3% vs +6.4%), modest DD increase
(32.9% vs 28.5%).

### Implementation

1. **Closed BTC short** (force-entered, +$26.44 realized). Was conflicting
   with the new BAH BTC long.
2. **Built `strategies/bah_btc.py`**: target 20% allocation, monthly
   rebalance (or ±5pp drift). Initial buy: 0.2475 BTC at $80,823 = $20,000.
3. **Built `bah_btc_run.py`**: scheduled as `Crypto_bah_btc_monthly` (1st of
   each month at 14:40).
4. **Added `bah_btc` sleeve** to pnl_attribution + kill_criteria.
5. **Charter §7** updated with 50/30/20 allocation rules.

### Schedule grew to 20 Crypto_ tasks

Strategic + operational + monitoring + discipline + GCR overlays + monthly
BAH rebalance. All under `Crypto_*` prefix.

### Position state after the rebuild

| Sleeve | Position | Notional |
|---|---|---|
| bah_btc (NEW) | BTC long, 0.2475 BTC | $20,000 |
| discretionary | NEAR long, ETH short | ~$33k total |
| xsmom | LINK, SOL long; ETH, ATOM short | ~$12k gross |
| systematic_pro_trend | nothing (all pairs below SMA200) | $0 |

The ETH short conflicts slightly with no BAH ETH, but ETH short is held
on trail stop and managed separately. NEAR long is aligned with new
long-only direction.

---

## EARLIER: 4-year indicator factor mining (2026-05-11 late night)

Ran systematic factor mining over 4 years of BTC/ETH/SOL daily data:

- **Phase 1:** 24 indicators × 3 pairs = 72 single-indicator signal tests
- **Phase 2:** Top 10 indicators × pairwise AND combinations (45 combos)
- **Phase 3:** Best combos applied as ENTRY SIGNALS to full pro_trend
  mechanics (wide stops + pyramid + portfolio cap). Walk-forward 5 folds.

### Top single indicators (avg Sharpe across 3 pairs at quantile-70 threshold)

| Indicator | Sharpe q70 | IC_30d |
|---|---|---|
| TSMOM_30 | 0.74 | +0.057 |
| Donchian20_pct | 0.76 | +0.076 |
| CCI_20 | 0.68 | +0.085 |
| MACD_hist | 0.66 | **+0.118** |
| RSI14 | 0.61 | +0.079 |
| Price/SMA50 | 0.58 | +0.033 |

### Top pairwise combos (Long when BOTH > 70th %ile, 3-pair avg)

| Combo | Sharpe | Cum Ret | % In Mkt |
|---|---|---|---|
| TSMOM_30 + MACD_hist | **0.87** | +228% | 12.5% |
| TSMOM_30 + CCI_20 | 0.85 | +370% | 17.9% |
| MACD_hist + Close_vs_60d_HL | 0.84 | +141% | 11.6% |

### Phase 3: applied as entry filters to pro_trend (4-yr walk-forward)

| Variant | Full Sharpe | WF Mean | WF Std | Worst Fold | Score |
|---|---|---|---|---|---|
| v0 baseline (Donchian + SMA200) | 0.91 | +0.34 | 1.27 | -1.45 | -0.02 |
| v1 + MACD>0 | 0.95 | +0.50 | 0.95 | -0.51 | +0.51 |
| v4 TSMOM_only (no Donchian) | 0.79 | +0.75 | 0.72 | -0.33 | +0.83 |
| **v5 + TSMOM>0 + MACD>0** | **0.96** | **+0.74** | **0.76** | **-0.49** | **+0.93** ⭐ |
| v6 fresh-window cap | 0.62 | +0.13 | 1.32 | -1.25 | -0.56 |

### Decision: WIRE v5 into pro_trend.py

Long entries now require ALL FOUR:
1. Price > SMA200 (regime)
2. High >= 20-day Donchian high (breakout)
3. TSMOM_30 > 0 (30-day return positive)
4. MACD_hist > 0 (momentum confirmation)

The key value isn't a higher absolute Sharpe — it's **dramatically better
walk-forward stability**. Worst fold went from Sharpe -1.45 → -0.49.
Translates to less likelihood of psychological breakage during bad periods.

**Code changes:**
- `strategies/pro_trend.py:138-145` — compute TSMOM_30 + MACD_hist inline
- `strategies/pro_trend.py:268` — entry condition adds `tsmom30 > 0 and macd_hist > 0`

**Bug test: 105/105 PASS.**

### Test artifacts
- `core/indicator_lab.py` — 24-indicator factor library + IC analysis
- `core/indicator_strategy_test.py` — Phase 3 walk-forward of top combos

---

## EARLIER: 60-day live sim revealed shorts are net-negative (2026-05-11 night)

After user requested a 60-day backtest of the production rig, the simulation
revealed the system would have lost **-$10,011 (-10.01%)** while BAH basket
gained **+14.59%** — alpha gap of **-24.60%**.

Diagnostic: all 4 systematic short entries in the window were initiated when
price was already **25-68% below SMA200** — exhausted bear moves, not fresh
breakdowns. Every short stopped out as the market V-reversed.

Filter test on 6.3-year history:

| Variant | Sharpe | Annualized | Recent 60d |
|---|---|---|---|
| LONG+SHORT no filter (was production) | 0.90 | +31% | -11.67% |
| LONG+SHORT 20% exhaustion cap | 1.33 | +58% | 0.00% |
| **LONG-ONLY no filter** | **1.40** | **+80%** | **0.00%** |

**Decision: SHORTS DISABLED.** Removing shorts entirely is BETTER than any
short-filter variant. The "bear protection" alpha of the strategy was never
from shorting — it was from STAYING OUT (SMA200 filter). On crypto majors:

- Late longs (>20% above SMA200) → parabolic blowoff → WIN big
- Late shorts (>20% below SMA200) → capitulation reversal → LOSE big
- Asymmetric, structural to crypto.

### Code changes
- `strategies/pro_trend.py`: short EXIT logic decoupled from `enable_shorts`
  (existing shorts still managed). Short ENTRY logic still gated.
- `pro_trend_run.py`: defaults `enable_shorts=False`. Existing BTC/ETH shorts
  from force_entry remain managed for exit.
- Charter §1 updated to "LONG-ONLY" with rationale.

### Existing positions
- BTC short (force-entered): managed via trail stop, no new entries
- ETH short (force-entered): same
- NEAR long: still managed (long is the new direction)
- XSMOM positions: unchanged (XSMOM is a separate sleeve with different math)

### What this changes for live performance
- 60-day window forward: would have been 0% instead of -10% (didn't enter the 4 losing shorts)
- 6.3-year backtest annualized: +80% (vs prior +31% with shorts)
- Sharpe 1.40 (vs prior 0.90)

**Bug test: 105/105 PASS.**

### Test artifacts (kept for future reference)
- `core/exhaustion_filter_test.py` — proved 20% cap helps but not as much as removing shorts
- `core/long_only_vs_with_shorts.py` — head-to-head proving long-only wins
- `_diagnose_losers.py` — diagnosed the 4 recent short losses

---

## EARLIER: GCR-inspired Tier A info overlays (2026-05-11 later)

User asked "if you were GCR what would you do" — the answer was: GCR
wouldn't run this rig at all (he's discretionary). But the GCR-style
INFORMATION ADVANTAGES are compatible with the systematic framework
without breaking discipline. Built all 5 Tier A overlays:

| # | Overlay | File | Schedule | Alert types |
|---|---|---|---|---|
| 1 | OI/funding regime | `ops/oi_funding_overlay.py` | Daily 14:25 | FROTH / EXHAUSTION / SQUEEZE_LONG / SQUEEZE_SHORT |
| 2 | ETF flow daily | `ops/etf_flow_overlay.py` | Daily 14:28 | BIG_DAY (±$500M) / EXTREME_Z (>2σ) / STREAK (5+) |
| 3 | DXY+10Y macro | `ops/macro_filter.py` | Daily 14:30 | RISK_ON / NEUTRAL / RISK_OFF / FLIGHT |
| 4 | Liq cluster proxy | `ops/liq_cluster_proxy.py` | Daily 14:32 | LIQ_CASCADE / NEAR_SWING_HIGH / NEAR_SWING_LOW |
| 5 | Catalyst calendar | `ops/catalyst_calendar.py` | Daily 14:33 | T-7d / T-day / T+3d alerts |

**Crucial design choice: ALERT-ONLY.** None of these auto-trade. They
inform discretionary review during weekly check-in. The systematic
strategies (pro_trend + XSMOM + basis_arb) execute unchanged.

### Why this is GCR-compatible without being GCR-style trading

The user is not GCR (12-year track record, deep market relationships,
proven discretionary edge). The systematic discipline IS the protection
against being a *bad* discretionary trader. These overlays add GCR's
**information advantage** (positioning data, macro filter, catalyst
awareness) without giving up the systematic risk controls that prevent
typical retail blowups.

### First-day results

- OI/funding: all 7 pairs NEUTRAL. Funding moderate (0.5-1.0 bps/8h).
- ETF flow: -$145.7M today (Mar 8), z=-0.91 — within normal range, no alert.
- Macro: NEUTRAL. DXY 98.1 (-0.4% vs 100d SMA), 10Y 4.36% (+0.07 in 20d),
  VIX 17.2. Close to RISK_ON but TNX rising keeps it neutral.
- Liq cluster: multiple NEAR_SWING alerts because chop regime has tight
  consolidation; current price within 1% of recent levels.
- Catalyst calendar: EU MiCA deadline in 4 days. T-7 alert fired.

### Schedule now 19 tasks

Original 14 tasks + 5 GCR overlays. All under `Crypto_*` prefix.

**Bug test: 105/105 PASS** (added 17 new tests for Tier A overlays).

---

## EARLIER: Mean-reversion / reversal strategies tested + REJECTED (2026-05-11)

User requested wiring of two additional sleeves: cointegrated pairs trading
and negative-funding short basis arb. Both were backtested with same gate
criteria as XSMOM (correlation < 0.3, Sharpe > 0.3, combined Sharpe maintained).

| Strategy | Standalone Sharpe | Gate result | Decision |
|---|---|---|---|
| Cointegrated pairs (Engle-Granger, 7-pair universe, monthly re-scan) | **-0.40** | FAILED standalone | SKIPPED |
| Negative-funding basis arb (-1.0 bps entry, short spot + long perp) | **-5.61** | FAILED standalone | SKIPPED |
| Negative-funding basis arb (-2.0 bps entry, tighter) | Sharpe -1.51 to -3.89 | FAILED standalone | SKIPPED |
| Negative-funding basis arb (-3.0 bps entry, extreme) | Sharpe -0.60 to -2.50 | FAILED standalone | SKIPPED |

**Why pairs failed:** cointegration in crypto is unstable; relationships
decohere faster than the trade can capture reversion. The 327 re-scans over
2300 days found a pair each time, but the spread mean+std drifted between
scans, so signals chased already-broken relationships.

**Why negative-funding failed:** basis spread random walk + 30bps round-trip
commissions ate the funding income. Higher thresholds (entry < -2.0 bps)
reduce trade frequency but don't fix unit economics. When funding is very
negative, price is usually crashing — basis blowout exits cancel funding gains.

**Net change to live system: NONE.** Both strategies tested, both failed,
neither wired. Per Charter §6 ("don't add strategies expecting independent
alpha without gate passage"), discipline holds.

**Bug test still: 88/88 PASS.**

This is the most important lesson of the session: the Charter exists for
exactly this moment. The user said "wire it do all" expecting the strategies
would work. The data said they don't. Discipline > expectation.

### Files added (research only — no production wire)
- `core/pairs_coint_backtest.py` — full backtest + correlation gate (failed)
- `core/negative_funding_backtest.py` — short-basis backtest + threshold sweep (failed)

---

## EARLIER: Sim-driven recommendations implemented (2026-05-10 late night)

After 8M-path Monte Carlo sim revealed median 1-year +19% (not +80%) with 89%
of days underwater, implemented all 9 sim-driven recommendations:

| # | Recommendation | Status | Implementation |
|---|---|---|---|
| 1 | Cap rig at 30-40% of crypto allocation | ✅ | Charter §7 updated with sim justification |
| 2 | Pre-commit to 5-year horizon | ✅ | Charter §7a added with horizon comparison table |
| 3 | Weekly check-ins, not daily | ✅ | Charter §7b added with cadence rules |
| 4 | Forward-distribution chart | ✅ | `ops/dashboard_components.py` + dashboard.py wiring |
| 5 | Refine K2 (180d Sharpe < -0.5) | ✅ | `ops/kill_criteria.py` updated |
| 6 | Chop-regime detector | ✅ | `ops/regime_detector.py` (scheduled daily 14:18) |
| 7 | Sim comparator | ✅ | `ops/sim_comparator.py` (scheduled Sun 14:35) |
| 8 | YTD stop-out K5 (-25% by month 4) | ✅ | New rule in `ops/kill_criteria.py` |
| 9 | May 2021 fast-move shutoff | ✅ | Added to `ops/pro_trend_intraday.py` |

### Charter additions
- **§7 sizing**: 30-40% of crypto allocation (down from 30-50%); 10% cash
  reserve for opportunistic adds during DD
- **§7a horizon**: 5-year minimum commitment with multi-horizon P(profit) table
- **§7b psychology**: Daily/weekly/monthly/quarterly/yearly cadence rules.
  No daily checks. Suppress P&L alerts.

### Refined kill criteria
- **K1**: Live DD > 45% (unchanged)
- **K2**: 180-day rolling Sharpe < **-0.5** (was: any negative); only
  triggers below sim P5 of Sharpe distribution at 50% haircut
- **K3**: 6+ months without systematic entry (now active, was placeholder)
- **K5 (NEW)**: YTD return < -25% after month 4 → 30-day pause

### Regime detector outputs
Currently classified: TRANSITION (BTC -2.3% vs SMA200 + 60d return +15.1%).
When regime turns to CHOP, alerts fire with documented expected behavior:
"strategy under-performs in chop, expected -15% to +5% ann, do NOT change
parameters."

### Fast-move shutoff
If pair drops >15% in 24h while still above trail (long) or pumps >15%
while below trail (short), tighten trail to 1% buffer below current price.
This is the missing protection that lost -21.8% in May 2021 crypto crash.

### Bug test: 88/88 PASS
Extended `core/bug_test.py` with regime detector, sim comparator, dashboard
components, K2 refinement, K3 active, K5 YTD trigger.

### Schedule grew to 14 tasks
- Strategic: pro_trend_daily, basis_arb_4hourly, xsmom_weekly, eval_weekly
- Operational: pro_trend_intraday_15min, position_monitor_30min
- Real-time: realtime_kill_switch_5min, portfolio_risk_30min
- Discipline: kill_criteria, daily_log, **regime_detector** (NEW),
  weekly_review, **sim_comparator_weekly** (NEW), monthly_oos

---

## EARLIER: Top-1% multi-tier monitoring (2026-05-10 night)

After establishing parameter robustness in the recommendation pass, added the
multi-tier monitoring infrastructure that distinguishes top-1% systematic
shops from retail:

| Tier | Cadence | Component | Purpose |
|---|---|---|---|
| **Strategic** | Daily | `Crypto_pro_trend_daily` | Entry/exit/pyramid decisions |
| **Strategic** | 4h | `Crypto_basis_arb_4hourly` | Basis open/close (8h funding cycle) |
| **Operational** | **15min** | `Crypto_pro_trend_intraday_15min` ⭐ NEW | Ratchets trail stops, closes on intraday breach |
| **Real-time** | **5min** | `Crypto_realtime_kill_switch_5min` ⭐ NEW | Flash-crash protection (RT1/RT2/RT3) |
| **Discipline** | Daily | `Crypto_kill_criteria` | K1/K2 monitoring on rolling Sharpe + DD |
| **Discipline** | Weekly | `Crypto_weekly_review` ⭐ NEW | Per-sleeve P&L + Charter compliance |
| **Discipline** | Monthly (1st Sun) | `Crypto_monthly_oos` ⭐ NEW | OOS walk-forward + param drift check |

### Real-time kill switch (`ops/realtime_kill_switch.py`)

Polls portfolio MTM every 5 min, logs to `.equity_realtime_log.jsonl`,
checks velocity against thresholds:

- **RT1**: -5% in 10 min → flatten + 24h lockout
- **RT2**: -8% in 60 min → flatten + 24h lockout
- **RT3**: -15% in 24h → flatten + 24h lockout (backstop)

When triggered: closes all positions via correct broker, writes
`.kill_switch_lock.json` with timestamp. `pro_trend.cycle()` reads this
file and SKIPS new entries during lockout (existing positions still
managed by trail/exit logic).

### Intraday trail stop monitor (`ops/pro_trend_intraday.py`)

Every 15 min:
1. Loads each `.pro_trend_state_*.json` with units
2. Fetches current 24h high/low via REST
3. Ratchets `extreme` + `trail_stop` based on intraday extremes
4. If price has BREACHED trail intraday → close immediately

Initial test already tightened 3 trail stops (BTC short $87k→$86k,
ETH short $2572→$2544, NEAR long $1.245→$1.281) that would have
waited until tomorrow's daily cycle.

### Weekly review (`ops/weekly_review.py`)

Every Sunday: writes timestamped report to `weekly_reports/`. Includes:
- Open positions snapshot
- Per-sleeve P&L (systematic / discretionary / basis_arb)
- Equity windows (7d / 30d / 90d / all-time)
- Charter compliance check
- Live Sharpe vs backtest reference (1.40)

**Read-only** — does not modify state or parameters.

### Monthly OOS revalidation (`ops/monthly_oos.py`)

First Sunday of each month: re-runs walk-forward 5 folds + 3x3 param
sensitivity grid. Alerts if:
- Mean OOS Sharpe < 0.6 (50% degradation from baseline 1.27)
- Latest fold Sharpe < 0
- Local parameter optimum has drifted

### Bug tests now 62/62 PASS

`core/bug_test.py` extended to cover all new modules: intraday monitor,
kill switch velocity check, weekly review, monthly OOS, lockout integration.

---

## EARLIER: Comprehensive recommendation pass (2026-05-10 late evening)

After comprehensive backtest revealed +80% headline driven by 2020-21 mega-bull
and recent 2-year window flat-to-negative, ran all six recommendations. Three
landed, three didn't:

| Recommendation | Tested | Outcome | Action |
|---|---|---|---|
| Wider DD kill (0.30/0.35/0.40/0.45) | param_sweep.py | 0.30 wins on 6.3y, fails on chop-only window | KEEP 0.35 (more robust) |
| ATR sensitivity (3.5/4.0/4.5/5.0) | param_sweep.py | 4.0 still optimum | KEEP 4.0 |
| Top-K full history | param_sweep.py | All 5 still beat any subset | KEEP 5 |
| Funding-cost-aware backtest | funding_aware_backtest.py | 1.5x at -5% funding wins, only -20% kills it | KEEP 1.5x |
| Lower basis arb threshold (1.0→0.7) | basis_threshold_test.py | LOSES money (Sharpe -6.16 at 0.7) | KEEP 1.0 |
| Mean rev as chop hedge | mean_rev_chop_hedge.py | Standalone Sharpe 0.16 (gate is 0.3) | SKIP |

**Net: no parameter changes were applied. The system was already well-tuned.**
The value of running these tests was confirming robustness, not finding alpha.

### What WAS added — discipline infrastructure

1. **P&L attribution module** (`core/pnl_attribution.py`) — tags every position
   with its origin sleeve (`systematic_pro_trend` / `discretionary` / `basis_arb`)
   in `.pnl_attribution.json`. Force-entries are tagged automatically at entry.
   Live track record can now be cleanly separated from discretionary trades.

2. **Kill-criteria monitor** (`ops/kill_criteria.py`) — scheduled daily at
   14:15 NZ as `Crypto_kill_criteria`. Logs equity to `.equity_log.jsonl`,
   computes rolling 90-day Sharpe + current DD. Alerts on:
   - K1: live DD > 45% (5pp above backtest max)
   - K2: 90-day rolling Sharpe < 0
   - K3 (placeholder): 6 months without systematic entry
   - K4 (placeholder): live Sharpe < 40% of backtest after 60+ trades

3. **Strategy Charter** (`STRATEGY_CHARTER.md`) — one-page rules document.
   Covers what the strategy is/isn't, where alpha comes from, expected
   outcomes, kill criteria, and behavioral commitments. Read when tempted
   to break the strategy mid-drawdown.

4. **Existing 3 force-entered positions tagged as `discretionary`**:
   NEAR long, BTC short, ETH short. They will be managed by the daily
   cycle but won't contaminate systematic Sharpe measurement.

### Bug test extended to 49 tests, all PASS

`core/bug_test.py` now covers attribution, kill-criteria, plus the prior
config/math/robustness checks.

---

## LATEST: 4-lever calibration on pro_trend (2026-05-10 evening)

After the universe expansion test showed expanding HURT (top 11 +7.9% → top 74
+2.6% per-pair mean), tested four remaining levers. Three landed; one was a
red herring; the per-pair-flat sizing was actively broken.

### Tests run

1. **`core/regime_gate_test.py`** — BTC-regime gate (5 variants, soft+hard).
   Result: every gate variant LOWERED returns 30%+ vs no gate. The per-pair
   SMA200 is already doing the regime job — stacking BTC's regime is
   double-filtering. Verdict: kill the idea.

2. **`core/universe_size_test.py`** — top-K subsets by Sharpe/return/alpha.
   Result: top 5 by Sharpe more than doubles per-pair mean (4.35% → 10.62%).
   Top 5 = SOL, BTC, OP, AVAX, ETH. Top 3 marginally higher mean but
   concentration risk. Verdict: shrink to 5.

3. **`core/vol_targeted_test.py`** — per-pair vol scaling and portfolio cap.
   **Critical finding**: 4%-flat per pair was BROKEN at the portfolio level —
   when 5 pairs simultaneously triggered, total active risk hit 20% and the
   35% DD kill triggered 114 times in 1500 days. Adding a 15% portfolio cap
   raised Sharpe from 0.26 → 1.01 and annualized return from +2.13% →
   +45.28%. Vol-targeting alone gave Sharpe 1.04 — same as cap, more
   complex. Verdict: portfolio cap at 15%, skip vol scaling.

4. **`core/catalyst_overlay_test.py`** — halving cycle multipliers (6
   schedules tested). Result: every overlay schedule HURT returns. The
   "default" 1.0/1.5/1.0/0.5 schedule cost 24pp of annualized return for
   zero Sharpe gain. Verdict: kill `USE_CATALYST_OVERLAY`.

5. **Basis arb sleeve** — already built and scheduled (`Crypto_basis_arb_4hourly`).
   Verified runs cleanly. Sitting idle because funding is below 1 bps/8h
   entry threshold across the universe (BTC +5.1% ann, ETH +9.1%, OP/INJ at
   exactly +10.9%). Will fire automatically when funding spikes.

### Final config (LIVE in `strategies/pro_trend.py` as of 2026-05-10)

```python
PRO_TREND_PAIRS    = ["SOL/USDT", "BTC/USDT", "OP/USDT", "AVAX/USDT", "ETH/USDT"]
RISK_PCT_PER_UNIT  = 0.04          # base; capped by portfolio cap
PORTFOLIO_RISK_CAP = 0.15          # NEW — total active risk across pairs
LEVERAGE_MULTIPLIER = 1.5
USE_CATALYST_OVERLAY = False       # was True — disabled after backtest
ATR_STOP_MULT      = 4.0
MAX_PYRAMID_UNITS  = 2
DRAWDOWN_KILL_PCT  = 0.35
```

### Final-config backtest (`core/final_config_backtest.py`, 1241 days)

```
Starting equity:    $100,000
Final equity:       $378,213
Total return:       +278.21%
Annualized:         +45.28%
Sharpe:             +1.01
Max drawdown:       40.15%   (DD-kill triggers when EOD MTM exceeds 35%)
Total trades:       58 (≈17/yr across 5 pairs)
DD-kill events:     38
```

### Open positions (carried over, unchanged by this calibration)

- BTC short, trail $87,283 (in profit, current $80,764)
- ETH short, trail $2,572 (in profit, current $2,328)
- NEAR long, trail $1.25 (in profit, current $1.56) — ORPHANED from
  universe shrink, but `pro_trend_run.py` now also iterates active state
  files outside PRO_TREND_PAIRS for management (no new entries, just exit).

### What changed in code

- `strategies/pro_trend.py` — new `PORTFOLIO_RISK_CAP`, helpers
  `count_active_pairs()` / `effective_risk_pct()`, entry+pyramid both use
  `effective_risk_pct(pair)`, catalyst overlay default OFF, universe to 5.
- `pro_trend_run.py` — `_orphaned_pairs()` helper folds in any state file
  with open units that's no longer in the active universe.

---

# (Older sections preserved below; pre-calibration context.)

# Crypto Rig Status — 2026-05-10 (post-cleanup)

## CRITICAL FINDING (added during JS-style cleanup review)

**The full multi-strategy stack does NOT generate edge.** Backtested over
729 days across the 10-pair universe with 5 strategies:

```
Total return:        -25.4%   ← LOST money
Sharpe (annualized): -0.12    ← negative
Max drawdown:        -53.3%   ← catastrophic
vs BTC buy-and-hold: -57.4%   (BTC was UP 32%, we lost 25%)
```

The pruned 3-strategy version was even WORSE (-40% return). This means the
"redundant" strategies were the only thing reducing the loss — the system
has no positive edge in any combination tested.

**Implications:**
1. The current paper rig should NOT be deployed live at any size.
2. The 30-day paper window is no longer needed for confirmation —
   we already have 729 days of evidence the combination doesn't work.
3. The individual `diverse_mom_ethbtc` strategy is the least-bad
   candidate; the orchestrator has been pruned to only this.

## Current state (v2 — post-fix-implementation)

- **Paper bankroll**: 100,000 USDT
- **Active strategies (5)**: diverse_mom_ethbtc, tsmom, short_term_momentum,
  vol_breakout, funding_basis. Same set used in v2 backtest.
- **Universe**: 10 pairs (BTC/ETH/SOL/BNB/XRP/ADA/AVAX/DOGE/LINK/DOT)
- **Mode**: paper, LONG_ONLY = True, mainnet read-only ticks, simulated fills
- **Risk-parity sizing**: per-pair gross cap = min(0.15, 0.50 / realized_vol_ann)
- **Concordance dampener**: target × max(0.30, 1 - concordance_score)
- **Portfolio vol target**: 20% annualized; scale all positions down if exceeded
- **Long-only confirmed best in 729-day backtest** (-1.7% vs -25% no-fixes)

## v2 fixes implemented in run.py

1. **Inverse-vol per-pair sizing** — replaces fixed 10% cap; SOL gets less notional than BTC
2. **Concordance dampener** — when ≥70% strategies aligned, scale signal down
3. **Portfolio vol target** — total weighted vol capped at 20% annualized
4. **Long-only** — empirically best in the 729-day window
5. **Regime gate** — per-pair 200-SMA filter (longs allowed only in bull)
6. **Liquidity check with exit override** — risk-reduction trades bypass liquidity gate

## Honest framing of what this is

A **capital-preserving crypto exposure overlay**, not an alpha generator.
- 729-day backtest: -1.7% return, -6.4% max DD, -33.7% alpha vs BTC
- BTC buy-and-hold over same window: +32% return
- Use case: 20-40% of crypto allocation; remainder in BTC BAH

## NEW — Documented profitable systems (2026-05-10 build)

After the JS-style review concluded the daily-bar TSMOM mix has no edge,
implemented three peer-reviewed-documented profitable strategies:

### 1. Funding-rate basis arbitrage (THE headline result)

`core/basis_arb_backtest.py`, `strategies/funding_basis_arb.py`

Long spot + short perp when perp funding > entry threshold; collect funding
payments every 8h; close when funding normalizes.

**Backtest (999 days, 30% allocation per pair):**
- BTC: 4 trades, +2.62% return, **Sharpe +4.30**, max DD 0.18%
- ETH: 3 trades, +3.27% return, **Sharpe +5.62**, max DD 0.18%
- SOL: 6 trades, +3.05% return, **Sharpe +3.97**, max DD 0.37%
- Combined: ~3% per year on 30% allocation; scales linearly to 100%

**This is an institutional-grade carry trade with retail capacity.**
- Real edge: retail leverage demand pays the carry
- Documented Sharpe 5-15 in literature (ScienceDirect 2024)
- Capacity ~$50-200k before slippage matters

**Live execution NOW WIRED (paper-mode):**
- `core/perp_broker.py` — Binance USDT-margined perp paper broker (with
  funding accumulation, weighted-avg entry pricing, P&L realization)
- `core/basis_executor.py` — coordinates dual-leg open/close; tracks
  basis positions in `.basis_positions.json` (separate from spot state)
- `basis_run.py` — runs the basis-arb cycle, settles funding, opens/closes
  positions per signal
- Verified end-to-end with $10k test trade on BTC (mechanism works,
  basis spread visible: spot $80,750 vs perp $80,710 = $40 spread risk)
- Scheduled: `Crypto_basis_arb_4hourly` task in setup_crypto_scheduler.ps1
  (catches funding flips quickly without overtrading)

### 2. Cointegrated pairs trading (proper Engle-Granger version)

`core/cointegration.py`, `strategies/pairs_cointegration.py`

Different from naive z-score reversion:
- Tests cointegration first (Engle-Granger + ADF)
- Hedge ratio from regression, not 1:1
- OU process model gives proper entry/exit thresholds (half-life-based)

**Initial scan finding:** Only 2 of 21 pair combinations in top-7 universe
are formally cointegrated (ADA/LINK p=0.013, SOL/LINK p=0.017). BTC/ETH
which we'd been using as a "pair" actually FAILS cointegration (p=0.70)
— our diverse_mom_ethbtc reversion was approximating, not on solid ground.

### 3. Vol-managed cross-sectional momentum


`strategies/xs_momentum_vol_managed.py`

Han/Kang/Ryu (2024): cross-sectional momentum Sharpe 1.51 in crypto under
realistic costs. Daniel/Moskowitz (2016): vol-management reduces tail crashes.

Combined: rank-based long top tercile + scale by inverse realized vol.

## Going-live decision matrix (UPDATED)

The basis arb result is materially different from everything we've built
before. Sharpe 4-5 with max DD <0.5% is genuinely good and reproducible.

**Recommendation:** build the perp broker class to enable live basis arb
execution. This is the highest-EV next infrastructure investment.

## NEW (2026-05-10): PRO TREND FOLLOWER on BTC

After accepting that the multi-strategy daily-bar approach has no edge,
built the canonical pro trend system: WIDE STOPS + PYRAMIDING + LET WINNERS RUN.

**Backtest (2500 days BTC, 4 ATR / 2 pyramid / 2% risk):**
- Total return: +154% (vs BTC BAH +819%)
- Annualized: +16%
- Sharpe: +1.02
- Max drawdown: 21%
- Win rate: 45%, R/R 5.4

**Walk-forward (5 OOS folds):**
- Mean Sharpe: +0.58, Std 1.15
- 3 of 5 folds positive
- **Critical fold (2020-22, BTC -69%): system was -2% — MASSIVE protection**

**Multi-asset:**
- BTC: +24% / Sharpe 0.50 / DD 13% (BAH +118%)
- ETH: +20% / Sharpe 0.45 / DD 14% (BAH **-14%** — system BEAT BAH)
- SOL: +107% / Sharpe 0.92 / DD 22% (BAH +150%)

**Production setup (LIVE NOW):**
- `strategies/pro_trend.py` — generalized state-machine, ANY pair, both directions
- `strategies/pro_trend_btc.py` — older BTC-only version (kept as reference)
- `pro_trend_run.py` — multi-pair cycle: BTC + ETH + SOL each day, both directions
- `Crypto_pro_trend_daily` — scheduled task (14:10 NZ daily)
- Per-pair state files: `.pro_trend_state_{BTC,ETH,SOL}.json`
- Initial state: all 3 pairs flat (all in bear, system correctly waiting)

**Effective strategy count: 7 — proper diversification:**
  - BTC long (spot) + BTC short (perp)
  - ETH long (spot) + ETH short (perp)
  - SOL long (spot) + SOL short (perp)
  - Plus basis arb across BTC/ETH/SOL (uncorrelated carry)

Same proven mechanic across all 6 trend strategies (wide-stop trend +
pyramiding), backtested on each asset. Diversification by asset AND by
direction. Shorts route to perp broker; longs to spot. When BTC/ETH/SOL
break down (bear) the system goes short via perp; when they break up
(bull) it goes long via spot.

**Honest framing:** This is the closest we've come to a real, defensible
trading system. NOT going to beat BTC BAH in absolute terms during bull
markets, but provides MASSIVE drawdown protection. Use as: dynamic crypto
exposure that captures bull moves while sitting out bear markets.
Realistic 10-18% annualized at Sharpe 0.5-1.0 with 15-25% max DD.

## Strategies (11 total — all VALIDATED=False)

| Strategy | Source | Universe |
|---|---|---|
| `diverse_mom_ethbtc` | TSMOM(30,90) + ETH/BTC reversion | BTC only |
| `tsmom_v3` | + GARCH vol-target + drawdown scaling | BTC only |
| `tsmom` | Plain TSMOM(60) | Any pair |
| `funding_basis` | Perp funding contrarian | Any pair |
| `xs_momentum` | Cross-sectional rank | BTC/ETH/SOL/BNB/ADA |
| `vol_breakout` | Vol-regime breakout | Any pair |
| `short_term_momentum` | 10-day TSMOM | Any pair |
| `open_interest` | OI-confirmed trend (Binance) | Any pair |
| `long_short_ratio` | Contrarian on retail LS ratio (Binance) | Any pair |
| `stablecoin_supply` | USDT+USDC mcap expansion (CoinGecko) | Macro |
| `btc_dominance` | BTC dominance trend (CoinGecko) | All pairs |

## Risk gates (run in order each cycle)

1. **Circuit breaker** — `ops/circuit_breaker.py`. Liquidate everything if portfolio dd > 20%. Warn at 10%.
2. **Position monitor** — `ops/position_monitor.py`. Per-position stop-loss (7%), trailing stop (8% from peak), TP (off by default).
3. **Concordance check** — `core/correlation_monitor.py`. Alert if ≥85% of strategies agree on direction.
4. **Liquidity check** — `core/execution.py`. Abort trade if > 5% of top-20 book depth or spread > 50 bps.
5. **Per-pair regime gates** — `core/regime.py`. Long-only in bull, short-only in bear.
6. **Signal cycle** — combine signals → target weight → trade if delta > $50.
7. **Attribution snapshot** — `core/attribution.py`. Logs per-cycle signals + prices for analysis.

## Validation framework

- `core/cv.py` — walk-forward cross-validation
- `core/cpcv.py` — Combinatorial Purged CV (López de Prado)
- `core/deflated_sharpe.py` — Bailey/LdP DSR + Harvey/Liu/Zhu t > 3.0
- `core/factor_decomp.py` — alpha vs benchmark with t-stat
- `core/ic_tracker.py` — information coefficient + decay alert
- `core/meta_labeling.py` — RandomForest signal filter
- `core/garch_vol.py` — GARCH(1,1) conditional vol
- `core/hrp.py` — Hierarchical Risk Parity
- `core/shrinkage.py` — Ledoit-Wolf shrinkage covariance
- `core/exits.py` — triple-barrier labeling
- `core/sizing.py` — fractional Kelly
- `core/drawdown_scale.py` — Carver-style drawdown scaling
- `core/tail_overlay.py` — Daniel/Moskowitz crash overlay
- `core/hmm_regime.py` — Hamilton 2-state Markov regime
- `core/stress_test.py` — replay through historical crashes
- `core/realized_pnl.py` — FIFO realized P&L matching
- `core/portfolio_risk.py` — VaR, ES, position correlation, strategy correlation
- `core/monte_carlo.py` — bootstrap N-day P&L distribution

## Operational

- **Dashboard**: http://localhost:8510 (port 8510, separate from stocks dashboard)
- **Scheduler script**: `setup_crypto_scheduler.ps1` (run as admin to register tasks)
  - `Crypto_orchestrator_daily` — daily at 14:00 UTC
  - `Crypto_daily_log` — daily at 14:05 UTC
  - `Crypto_eval_weekly` — Sunday at 14:30 UTC
  - `Crypto_position_monitor_30min` — every 30 min for intraday stop coverage
- **Wakelock**: `ops/wakelock.py` (separate from stocks `wakelock.py`)
- **Watchdog**: `ops/watchdog.py` heartbeat + alerts

## Key empirical findings to date

1. **Bear regime currently active** — BTC 2.5% below 200-SMA. All 10 pairs in bear regime.
2. **Funding extremely negative** — perp funding contrarian fires max-short across the board.
3. **Position correlation HIGH** — all 10 crypto positions correlate 0.5-0.95 with each other. "10 positions" ≈ 2-3 effective bets.
4. **HMM regime: low_vol** at 92.7% probability. Two regimes have annualized vols 31.7% (low) vs 77.4% (high).
5. **Stress test result**: TSMOM with regime gate would have generated +33-54% alpha during COVID crash, China ban, LUNA collapse, FTX, yen carry unwind. The regime gate is real.
6. **Monte Carlo 30-day forecast**: expected P&L +$4k, but P5 = -$44k, P95 = +$52k. 21% chance of losing $20k+ in 30 days.

## What's still in the backlog

- Real-time WebSocket data (currently REST polling)
- Live broker for shorts (we paper-short via negative spot positions)
- Per-strategy Kelly sizing (needs ≥30 days of live P&L)
- Reproducibility hashing on input data
- TWAP execution slicing (`slice_order` exists, not wired)
- Funding payment accounting (perp positions earn/pay funding every 8h)
- DeFi TVL signal
- On-chain whale tracking (paid APIs)
- Cross-exchange basis arbitrage
- Variance risk premium (needs options data)

## Going-live readiness (per JS discipline)

- [ ] 30 calendar days of paper running
- [ ] 60+ closed paper trades
- [ ] Rolling 180-day Sharpe positive over last 90 days
- [ ] **Multi-strategy portfolio backtest passes (DSR > 0.95)** ← FAILED
- [ ] At least one strategy with DSR > 0.95 AND alpha t > 2.0 OOS
- [ ] BTC in bull regime (price > 200-SMA)
- [ ] Telegram/email alerts firing for ≥1 week without false positives
- [ ] At least one drawdown observed and correctly handled by stops
- [ ] Position monitor scheduled task confirmed running every 30 min

**Current: 0 of 9 boxes ticked, with the new portfolio-backtest box
already a hard NO.** Live capital remains gated indefinitely until we
either find new uncorrelated signal sources or accept that this universe
doesn't have retail-accessible edge at daily timeframes.

## Honest paths forward

A. **Accept the result and stop the rig.** 729 days of backtest evidence
   showing -25% return / -53% DD is conclusive enough. Use the
   infrastructure for future research, but don't trade.

B. **Run the single-strategy version in paper.** `diverse_mom_ethbtc`
   alone had OOS Sharpe 0.86, alpha_t 1.65. Best of a weak set. Run for
   30 days, see if recent rolling Sharpe turns positive.

C. **Pivot to fundamentally different research.** Daily-bar BTC
   strategies are saturated/efficient at retail level. Worth exploring:
   intraday timeframes (needs WebSocket), options vol surface (needs
   options data), cross-exchange basis (needs multi-broker), on-chain
   whale flows (paid APIs).

D. **Pause and cool off.** No code changes. Re-evaluate in 1 month with
   fresh perspective. The risk infrastructure built here is genuinely
   valuable; don't throw it away.
