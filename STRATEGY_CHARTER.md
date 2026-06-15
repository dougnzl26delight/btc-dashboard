# Strategy Charter — pro_trend crypto rig

**Drafted:** 2026-05-10  
**Read this when:** considering changes mid-strategy, drawdown panic, or any
"this isn't working" thought.

---

## 1. What this strategy IS

**LONG-ONLY wide-stop trend follower with momentum-confirmed entries**
on a 5-pair crypto universe (SOL, BTC, OP, AVAX, ETH), 1.5× leverage via perp longs.

**Entry requires ALL of:**
1. Price > SMA200 (regime filter)
2. Today's high >= 20-day Donchian high (fresh breakout)
3. **TSMOM_30 > 0** (30-day return positive)
4. **MACD_hist > 0** (12/26/9 MACD histogram positive)

Conditions 3-4 added 2026-05-11 after 4-year indicator factor mining
(indicator_lab.py + indicator_strategy_test.py) showed they dramatically
improve walk-forward stability (WF Sharpe std 0.76 vs 1.27 baseline; worst
fold -0.49 vs -1.45). The added confirmation prevents entries on weak
breakouts and chop-regime false starts.

**As of 2026-05-11: SHORTS DISABLED.** 60-day live-style backtest revealed
that on crypto majors, the Donchian-20-low + SMA200 short signal fires at
CAPITULATION BOTTOMS, not continuation breakdowns. All 4 systematic shorts
in the most recent 60 days lost money (-$10k aggregate). Full-history test:
removing shorts raises Sharpe from 0.90 → 1.40 and annualized from +31% → +80%.
Existing short positions (force-entered) are still managed for exits but no
new shorts will be entered.

- Entry: Donchian-20 breakout AND price above SMA200 (long), or breakdown +
  below SMA200 (short)
- Stops: 4× ATR trailing
- Pyramid: add 2nd unit at +2× ATR continuation
- Risk: 4% per pair OR `15%/n_active` whichever lower (portfolio cap)
- Exit: trail stop hit, SMA200 break, or portfolio drawdown 30% (kill switch)
- Data: daily bars, evaluated at 14:10 NZ via `Crypto_pro_trend_daily`

## 2. What this strategy is NOT

- Not a profit maximizer in absolute terms (BAH wins in pure bulls)
- Not a continuous alpha generator
- Not a chop-regime strategy (under-participates in sideways markets)
- Not a high-frequency strategy (~17 trades/year/pair)
- Not diversified outside crypto — all 5 pairs are crypto-correlated
- Not validated for live: 0 closed systematic trades to date (2026-05-10)

## 3. Where the alpha comes from

**Bear-market protection.** SMA200 + trail stops keep capital out during
crashes; backtest shows +60% alpha vs BAH during LUNA, +31% during FTX,
+60% during full 2022 bear. The big wins compound this protected capital
into bull-cycle entries.

**NOT from**:
- Out-trading retail in bull rallies (BAH usually wins)
- Picking winners (universe is fixed at 5)
- Timing tops (we ride trends until trail stop hits)

## 4. Expected outcomes (updated 2026-05-11 after rebuild)

| Metric | Realistic range | Backtest reference (6.3y, long-only) |
|---|---|---|
| Annualized return | +18% to +35% | +80% (mega-bull biased) |
| Sharpe | 0.6 to 0.9 | 1.40 (full) / 0.96 (4y v5) |
| Max drawdown | 35-45% | 40% |
| Months underwater (typical) | 6-18 | up to 18 |
| Months underwater (max) | 24+ | 17 |
| Trades per year | ~15-20 (across 5 pairs) | 17 (long-only) |

**Forecast slightly UP after the 2026-05-11 rebuild** (shorts disabled,
v5 momentum-confirmed entries):
- The shorts that were costing money are now blocked
- The v5 filter (TSMOM + MACD) cut worst walk-forward fold from
  Sharpe -1.45 → -0.49 (3× better tail protection)
- 60-day live-style sim on the most recent data: -10.01% (old config)
  → +0.25% (new config). The recent regime that hurt is now neutral.

**Most likely 3-year outcome on $100k:** $160-220k, with at least one
12-month underwater stretch en route.

## 5. Kill criteria (when to PAUSE the rig)

- **K1 — Drawdown:** Live max DD exceeds **45%** (5pp above backtest max)
- **K2 — Sustained negative Sharpe:** Rolling 90-day Sharpe < 0 for 60
  consecutive days
- **K3 — No entries:** 6 months pass with zero new systematic entries
  (suggests SMA200 filter is structurally broken or universe is wrong)
- **K4 — Live vs backtest divergence:** After 60+ closed trades, live
  realized Sharpe < 40% of backtest Sharpe (i.e., < 0.56)

When ANY criterion triggers: **stop adding capital, do not close existing
positions, review with a 2-week cooling period before any change.**

## 6. Behavioral commitments (the hard part)

### I will NOT:
- **Force-enter against the system.** Every `force_entry.py` use breaks the
  evidence-based discipline. The strategy's "wait" signal IS the alpha.
- **Tinker with parameters during a drawdown.** A bad month is not a
  parameter problem; it's the cost of the strategy.
- **Add new strategies expecting independent alpha.** Three prior strategies
  were tested and failed to add value. Stop adding.
- **Expand the universe back beyond 5 pairs.** This was tested and disproven.
- **Iterate weekly.** Each retest inflates trial count and raises the deflated-
  Sharpe hurdle. Implement once, then **wait 60 days minimum** for evidence.
- **Size up live before paper passes 60 closed trades AND 90-day live
  Sharpe is positive.**
- **Confuse activity with edge.** Sitting in cash IS the strategy in chop
  regimes. The backtest spent 30%+ of days flat; that's normal.

### I will:
- **Run the daily cycle.** Let the scheduler do its job.
- **Track per-sleeve P&L** (systematic vs discretionary) so the live
  evidence is uncontaminated by force-entries.
- **Read this charter** when I think the strategy is broken.
- **Pause if K1 or K4 fires.** No exceptions.

## 7. Capital allocation rules

### Within-rig 3-sleeve split (2026-05-11 rebuild)

| Sleeve | Target % | Role |
|---|---|---|
| **pro_trend v5** | **50%** | Trend follower; bear protection via SMA200 filter |
| **XSMOM** | **30%** | Cross-sectional momentum; uncorrelated to pro_trend |
| **BAH BTC** | **20%** | Passive directional exposure; catches chop-regime upside |

Justification: three_sleeve_test.py 2026-05-11 showed this allocation:
- Improves 6.3y Sharpe from 1.48 (70/30/0) to 1.49 (50/30/20)
- Doubles recent 18-month return (+6.4% → +11.3%)
- Modestly increases max DD (28.5% → 32.9%)

Rebalance: BAH BTC sleeve rebalances monthly (1st of each month) or on
±5pp drift from target. Pro_trend and XSMOM operate via their own
dynamics; allocations re-emerge from each cycle's risk budgeting.

### Total-portfolio rules

- **This rig: 30-40% of total crypto allocation**, NOT 100%
  - Justification: 8M-path Monte Carlo (max_simulation.py 2026-05-10) shows
    P(>50% DD) = 5.4%. Don't put more capital here than you can lose half of.
- **Remainder: BTC BAH** (the bull-cycle return engine — strategy under-
  participates in pure bulls, BAH carries that)
- **Outside crypto:** at least 50% of total portfolio in stocks/bonds rig
  (genuine diversification — all 5 crypto pairs correlate 0.6-0.95)
- **Cash reserve:** keep ~10% of liquid capital uncommitted to opportunistically
  add during DD episodes (buy when the rig is deep underwater).

## 7a. Time horizon commitment

**This is a 5-YEAR strategy. Anything shorter is path-dependent gambling.**

Monte Carlo evidence (8M paths, 50% Sharpe haircut applied):

| Horizon | P(profit) | P50 outcome on $100k |
|---|---|---|
| 30 days | 41.7% | $98,564 |
| 90 days | 54.4% | $102,222 |
| 1 year | 66.0% | $118,892 |
| 2 years | 72.8% | $143,742 |
| **5 years** | **83.8%** | **$255,602** |

**Don't judge the strategy on horizons under 1 year.** P(profit at 30d) is 42% —
worse than a coin flip. Time-diversification is the alpha.

## 7b. Check-in cadence (psychology rules)

**89% of days are spent underwater** (below previous peak). Daily checking =
daily disappointment. Therefore:

- **Daily**: do not check. Scheduler runs autonomously.
- **Weekly (Sunday)**: read `weekly_reports/weekly_<date>.md` only.
- **Monthly (1st Sunday)**: read `monthly_reports/monthly_oos_<date>.md`.
- **Quarterly**: review live-vs-sim percentile tracker. Inside [P5, P95] = OK.
- **Yearly**: full re-evaluation, including kill criteria assessment.

Suppress daily Telegram/email alerts on equity/P&L. Only K1/K2/K4/K5
critical alerts reach the user. The kill_criteria daemon still runs but
quietly.

## 8. Discretionary force-entries (if any)

Currently active (force-entered against system 2026-05-10):
- NEAR/USDT long (entry $1.556, trail $1.25)
- BTC/USDT short (entry ~$95k, trail $87,283) 
- ETH/USDT short (entry ~$2,800, trail $2,572)

These are **tagged as `discretionary` sleeve** and tracked separately from
systematic P&L. They will be managed by the daily cycle's exit/pyramid
logic but they are NOT systematic evidence.

**No new force-entries.** Period.

## 9. Configuration (live as of 2026-05-10)

```python
PRO_TREND_PAIRS    = ["SOL/USDT", "BTC/USDT", "OP/USDT", "AVAX/USDT", "ETH/USDT"]
SMA_FILTER         = 200
DONCHIAN_WINDOW    = 20
ATR_PERIOD         = 14
ATR_STOP_MULT      = 4.0       # validated 2026-05-10 (param_sweep)
PYRAMID_ATR_STEP   = 2.0
MAX_PYRAMID_UNITS  = 2
RISK_PCT_PER_UNIT  = 0.04
PORTFOLIO_RISK_CAP = 0.15      # critical: prevents simultaneous-entry blowout
DRAWDOWN_KILL_PCT  = 0.35      # 0.30 wins on long history, but 0.35 is more
                                # robust to chop regimes (param_sweep 2026-05-10)
LEVERAGE_MULTIPLIER = 1.5      # validated incl. funding cost
USE_CATALYST_OVERLAY = False   # tested, disabled — overlay HURT 24pp/yr
ROUND_TRIP_BPS     = 30
```

Basis arb sleeve:
```python
ENTRY_FUNDING_BPS_8H = 1.0     # validated optimum (lowering to 0.7 lost money)
EXIT_FUNDING_BPS_8H  = 0.3
```

## 10. Schedule (top-1% multi-tier monitoring)

**Strategic tier** — daily, decisions on entries/exits/pyramids:
- `Crypto_pro_trend_daily` — 14:10 NZ
- `Crypto_basis_arb_4hourly` — every 4 hours (funding cycle is 8h)
- `Crypto_eval_weekly` — Sunday 14:30 NZ

**Operational tier** — intraday risk management:
- `Crypto_pro_trend_intraday_15min` — every 15 min, ratchets trail stops + closes
  on intraday breach (key for flash moves between daily cycles)
- `Crypto_position_monitor_30min` — legacy position monitor (older orchestrator)

**Real-time risk tier** — flash-crash protection:
- `Crypto_realtime_kill_switch_5min` — every 5 min, monitors MTM velocity:
  - RT1: -5% in 10min → flatten + 24h lockout
  - RT2: -8% in 60min → flatten + 24h lockout
  - RT3: -15% in 24h → flatten + 24h lockout
  - When locked: pro_trend skips new entries; existing positions still managed

**Discipline tier** — equity logging + kill criteria:
- `Crypto_kill_criteria` — daily 14:15 NZ (logs equity, K1/K2/K3/K4 alerts)
- `Crypto_daily_log` — daily 14:05 NZ
- `Crypto_weekly_review` — Sunday 14:30 NZ (per-sleeve P&L + charter compliance)
- `Crypto_monthly_oos` — first Sunday only (walk-forward + param drift check)

---

**The strategy will look broken. That is expected. Do not break it in response.**
