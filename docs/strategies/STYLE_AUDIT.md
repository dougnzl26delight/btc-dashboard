# Style Audit — Schwager's *Market Wizards* discipline

> "Combining the methodologies of even the world's best traders results in much worse results."
> "Every successful trader Schwager interviewed found a style of trading that fits their personality very specifically."
> — Daniel Scrivner, synthesis of *Market Wizards* series

This document audits each sleeve against three psychology-fit criteria. Sleeves
that fail two of three are flagged for removal — **not because they don't backtest
well, but because the operator will sabotage them under pressure.**

## The 3 Schwager questions per sleeve

1. **Can I sleep at night with this position size?**
2. **When this sleeve has a 4-loss streak, will I leave it alone or will I want to override / disable?**
3. **Does the trade frequency match my screen time (30-60 min/day per `claudeMd`)?**

Score 1 if yes, 0 if no. Sum across 3.

## Audit — fill in honestly, kill failures

| Sleeve | Q1: Sleep with size? | Q2: Leave alone on 4-loss? | Q3: Freq matches 30-60 min/day? | Score | Verdict |
|---|---|---|---|---|---|
| **BAH BTC** | 1 (small, long-cycle) | 1 (passive, designed to ride through) | 1 (monthly rebal) | **3/3** | KEEP — pure style match |
| **XSMOM** | 1 (14d hold, modest size) | likely 1 (14d cadence absorbs streaks) | 1 (rebalance every 14d) | **3/3** | KEEP — cross-section is psychologically clean |
| **basis_arb (post-W14)** | 1 (delta-neutral) | likely 1 (carry trade — losses are slow bleed) | 1 (4h cadence, ~1/week new) | **3/3** | KEEP — but only on Drift/Hyperliquid long-term |
| **oversold_bounce** | 1 ($3k per pair) | TEST — when streak hits, do you stay? | 1 (entries 1-2× per month) | **2/3 → TEST** | Keep on probation; verify Q2 in live |
| **overbought_fade** | 1 ($3.3k per pair) | TEST — short squeezes are emotionally hard | 1 (entries 1-2× per month) | **2/3 → TEST** | Same; biggest psychology risk of all sleeves |
| **pro_trend** | 1 (1% risk per unit) | TEST — pyramiding into trends in chop is brutal | 1 (daily check) | **2/3 → TEST** | Audit after first 4-loss streak |
| **grid_trader** | 1 (small grid steps) | likely 1 (mechanical, no emotion) | 1 (5-min cadence — automated) | **3/3** | KEEP — true automation; minimal override risk |
| **intraday_momentum (long)** | 1 (small per-trade) | likely 1 (15-min — too fast for emotional override) | TEST — does 15-min firing distract you? | **2/3 → TEST** | Disable if it interrupts focus during work |
| **intraday_momentum_short** | 1 (small per-trade) | TEST — shorts on retail psychology is the hardest | 1 (15-min cadence) | **2/3 → TEST** | Most likely candidate for cull |
| **consolidation_breakout** | 1 (15% per trade) | likely 1 (multi-day setups; less emotional) | 1 (daily scan) | **3/3** | KEEP — matches the Livermore patience profile |

## Sleeves to actively cull after Day 30 walk-forward

Per Schwager, run each sleeve LIVE PAPER through one 4-loss streak. After that:

- **If you opened the dashboard 5+ times during the streak** = Q1 failed; cut the sleeve
- **If you wanted to "tweak the parameters" mid-streak** = Q2 failed; cut the sleeve
- **If the sleeve fires during work hours and breaks your focus** = Q3 failed; either schedule shift or cut

## The Schwager prime directive

You do not need every sleeve. You need the sleeves that you will OBEY WITHOUT INTERFERENCE.

A rig running 3 obedient sleeves with Sharpe 0.5 each will compound. A rig
running 8 sleeves where you override 2 of them quarterly will not.

After walk-forward day 30, this doc gets updated. After day 90, sleeves that
failed are REMOVED from the orchestrator.

## Sleeve reduction roadmap

**Target by Day 90 of walk-forward:** 4-6 sleeves maximum.

Default cull candidates (most likely to fail Schwager audit in live trading):

1. `overbought_fade` — shorting in retail psychology is the hardest single skill
2. `intraday_momentum_short` — same problem, smaller
3. `pro_trend` shorts component — same problem
4. `grid_trader` — if it causes panic during BTC -10% day, cut it

Keep candidates (passed Schwager audit during build review):

1. `BAH BTC` — cycle thesis, psychologically clean
2. `XSMOM` — cross-sectional with 14d cadence
3. `basis_arb` — delta-neutral carry (post-Drift migration)
4. `oversold_bounce` — but only if Q2 verified in live
5. `consolidation_breakout` — Livermore pattern, matches operator profile
