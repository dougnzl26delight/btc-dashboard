# Cash-Yield Deployment Plan

**Generated:** 2026-05-28
**Status:** Manual setup required — cannot be automated by the rig (CeFi APIs differ; opening accounts needs human KYC)

## Current state

- Spot account cash: **~$99,837 idle**
- Perp account cash: **~$98,176 idle** (collateralizing $8.6k of XSMOM positions)
- **Total idle dry powder: ~$190k** earning 0% in the paper sim. In live deployment this is real opportunity cost.

## Why this matters

A top-1% trader treats idle cash as a position with a measurable expected return.
At current rates:
- USDC/USDT in CeFi yield products: **5-8% APY**
- 1-year T-bill equivalents (US): **~4.5%**
- NZ OCR-linked term deposits: **~5.25%**

On $190k that's **$8,500 - $15,200/year** in pure carry, no market risk. Not deploying it is leaving money on the table.

## Recommended split

| Bucket | Amount | Allocation | Vehicle | Why |
|---|---|---|---|---|
| **Deployable yield** | $100k | 50% | Binance Flexible Earn (USDT) or KuCoin Lending | 6-8% APY, no lockup, withdraw anytime |
| **Short-dated income** | $50k | 25% | NZ term deposits (3-6 month) OR US T-bill ETF (BIL/SHV via Hatch) | 4.5-5.25% guaranteed, FDIC/Crown-backed |
| **Hot dry powder** | $30k | 15% | Stays in stablecoin on exchange | Available within minutes to deploy on flash crashes / forced liquidations |
| **Trade margin reserve** | $10k | 10% | Stays in spot account cash | Covers XSMOM rebalances + tactical trades without dipping into yield-earning buckets |

## Operational rules

1. **Never let "hot dry powder" go above $30k** — if cash builds up from sleeve exits, sweep excess to the yield bucket the next morning.

2. **Never deploy yield-bucket capital intra-day** — minimum 24h notice. This forces discipline. Tactical bets must use the hot bucket.

3. **Audit monthly** — track the yield earned vs the opportunity cost of any deployed capital. If a tactical trade had 6-month payback but only 2% return, hot-bucket capital was wasted vs the 6% yield alternative.

4. **Rebalance on rate moves > 100 bps** — if CeFi yield drops below 4%, shift weight to T-bills. If above 10% (rare), shift more to CeFi. Don't churn for <100bp changes.

## Specific vehicle recommendations

### Crypto-native yield (50% of bucket)

**Binance Flexible Earn — USDT**
- Currently paying ~6.5% APY (varies daily)
- No lockup, withdraw to spot anytime
- Capital efficient — same exchange as your trading
- Counterparty risk: Binance (acceptable given size)

**Alternative:** Kraken USDC, Bybit Earn, Aave (DeFi — slightly higher yield but smart-contract risk)

**Avoid:** Anchor-style yield products promising 15%+ — these are unsustainable

### Short-dated income (25% of bucket)

**For NZ residents:**
- Sharesies or Hatch → BIL ETF (US 1-3 month T-bills) — currently ~5% gross, ~3.4% after NZ tax
- ANZ/ASB/Westpac 6-month term deposit — currently ~5.25% gross
- Kiwi Bonds (Treasury) — guaranteed but slightly lower yield

### Hot dry powder (15% of bucket)

Stays in USDT on the exchange. The rig can deploy it instantly via tactical signals (mean reversion bounce strategy = task #6).

## Implementation steps

1. **Today:** Document the split (this file). No actual movement until you decide.
2. **This week:** Open Binance Flexible Earn position with $50k USDT. Test withdrawal-back flow with $1k first.
3. **This week:** Open Sharesies/Hatch account if not already. Deposit $50k NZD-equivalent.
4. **Next week:** Buy BIL or NZ term deposit with the $50k. Set calendar reminder for roll-over.
5. **Ongoing:** Monthly review — earned yield vs opportunity cost.

## What NOT to do

- ❌ Don't lock all $190k in fixed-term products (no liquidity for tactical trades)
- ❌ Don't chase 10%+ DeFi yields without understanding smart-contract risk
- ❌ Don't put yield-bucket capital on a different exchange than your trading account — friction = lost trades
- ❌ Don't deploy yield-bucket on the same day you exit a trade (24h cooling period rule)

## Expected impact on rig P&L

Conservative scenario (50% deployed at 6%, 25% at 5%, 25% idle):
- Annual yield on $190k = $100k × 6% + $50k × 5% + $40k × 0% = **$6,000 + $2,500 = $8,500/year**
- That's **+4.5% on the idle capital** with zero market-risk added

This is roughly equivalent to one extra alpha generator running 24/7 with Sharpe ∞ (no volatility). It's free.

---

**Last updated:** 2026-05-28
**Next review:** First Sunday of each month — verify yields haven't drifted, rebalance if needed
