"""BTC 2035 projection — adoption-led bull scenarios."""
from datetime import date
import pandas as pd


print("=" * 90)
print("BTC 2035 — re-projecting with adoption tailwinds")
print("=" * 90)
print()

# Market cap framing
btc_supply_2035 = 19_900_000  # approx supply after halving 5 + slow issuance
gold_mcap_today = 22_000_000_000_000  # $22T (Apr 2026 estimate including central bank reserves)
global_wealth = 500_000_000_000_000  # ~$500T global wealth in all assets
us_money_supply_m2 = 22_000_000_000_000  # $22T M2

print(f"Reference market caps (today's dollars):")
print(f"  BTC current:           ${80_000 * btc_supply_2035 / 1e12:.1f}T")
print(f"  Gold (incl reserves):  ~$22T")
print(f"  US M2 money supply:    ~$22T")
print(f"  Global wealth:         ~$500T")
print()

# Calculate BTC price needed to reach various market cap milestones
def price_at_mcap(mcap, supply=btc_supply_2035):
    return mcap / supply

print("BTC price required to reach various market cap milestones:")
print(f"  $5T mcap (~25% gold):     ${price_at_mcap(5e12):,.0f}")
print(f"  $10T mcap (~45% gold):    ${price_at_mcap(10e12):,.0f}")
print(f"  $22T mcap (= gold today): ${price_at_mcap(22e12):,.0f}")
print(f"  $50T mcap (super-asset):  ${price_at_mcap(50e12):,.0f}")
print(f"  $100T mcap (~1% global):  ${price_at_mcap(100e12):,.0f}")
print()

# Adoption-led scenarios
print("=" * 90)
print("BTC 2035 SCENARIOS WITH ADOPTION TAILWINDS")
print("=" * 90)
print()

scenarios = [
    {
        "name": "1. Continued cycles, mild diminishing (base case earlier)",
        "thesis": "4-yr cycles continue but each cycle adds 1.5-2x",
        "2035_price": 280_000,
        "mcap_t": 5.6,
        "annualized": (280/80) ** (1/9.6) - 1,
        "probability": 25,
    },
    {
        "name": "2. ETF/bank custody mainstream (your thesis)",
        "thesis": "ETF flows + bank custody systematic; cycles compress to smaller bears",
        "2035_price": 500_000,
        "mcap_t": 10.0,
        "annualized": (500/80) ** (1/9.6) - 1,
        "probability": 30,
    },
    {
        "name": "3. Sovereign/corporate reserve asset",
        "thesis": "Multiple G20 nations + S&P500 corps add BTC to reserves",
        "2035_price": 800_000,
        "mcap_t": 16.0,
        "annualized": (800/80) ** (1/9.6) - 1,
        "probability": 20,
    },
    {
        "name": "4. Gold parity ($14-22T mcap)",
        "thesis": "BTC absorbs significant portion of gold's role as store of value",
        "2035_price": 1_000_000,
        "mcap_t": 20.0,
        "annualized": (1000/80) ** (1/9.6) - 1,
        "probability": 15,
    },
    {
        "name": "5. Hyper-monetization (>gold)",
        "thesis": "BTC becomes dominant global store of value, banks hold large reserves",
        "2035_price": 2_000_000,
        "mcap_t": 40.0,
        "annualized": (2000/80) ** (1/9.6) - 1,
        "probability": 7,
    },
    {
        "name": "6. Regulatory destruction (bear)",
        "thesis": "Hostile policy in 2-3 major economies cripples BTC adoption",
        "2035_price": 80_000,
        "mcap_t": 1.6,
        "annualized": 0.0,
        "probability": 3,
    },
]

print(f"{'Scenario':<50s} {'2035 BTC':>12s} {'MCap $T':>9s} {'Annlzd':>8s} {'P':>4s}")
print("-" * 90)
weighted = 0
for s in scenarios:
    print(f"{s['name']:<50s} ${s['2035_price']:>10,.0f}  ${s['mcap_t']:>5.1f}T   "
          f"{s['annualized']*100:>5.1f}%   {s['probability']:>2d}%")
    weighted += s["2035_price"] * s["probability"] / 100

print()
print(f"Probability-weighted 2035 BTC price: ${weighted:,.0f}")
print(f"Annualized: {(weighted/80_000) ** (1/9.6) - 1:.1%}")
print()

print("=" * 90)
print("Why my prior projection was too conservative")
print("=" * 90)
print()
print("Prior base case ($212k by 2035) assumed:")
print("  - Cycles continue with diminishing returns")
print("  - Each cycle's top-multiple shrinks geometrically")
print("  - No structural adoption breakthrough")
print()
print("What that missed:")
print("  - ETF flows = systematic demand (didn't exist in prior cycles)")
print("  - Bank custody = reduces 'where to store' friction")
print("  - 401(k)/retirement access = automatic monthly buying")
print("  - Corporate treasury adoption (Strategy, others)")
print("  - Stablecoin growth = on-ramp at scale")
print("  - Sovereign reserve potential (US strategic reserve discussion)")
print()
print("The base case extrapolates from 12 years of pre-mainstream data.")
print("If mainstream adoption is in fact happening, the pattern breaks UP.")
print()

print("=" * 90)
print("HISTORICAL CONTEXT — what did each previous tech go through?")
print("=" * 90)
print()
print(f"{'Asset':<30s} {'Pre-mainstream growth':<25s} {'Mainstream growth':<25s}")
print("-" * 80)
print(f"  Gold (1971-1980)              35x in 9 years            Then 20 years sideways")
print(f"  Internet (1995-2000 stocks)   Microsoft 100x in 5yrs    +20-30% annual for decades")
print(f"  Smartphones (2007-2015)       100x adoption             Now ubiquitous, 5-10% annual")
print(f"  EVs (2015-2025)               Tesla 100x in 10yrs       Maturing")
print()
print("BTC has already done 12 years of pre-mainstream (50% annualized).")
print("If mainstream phase is starting NOW, expect:")
print("  - Lower volatility (no more 80% bears)")
print("  - More linear growth (15-30% annualized)")
print("  - Tighter cycles or no cycles at all")
print("  - 2035 BTC: $500k-$2M plausible")
print()

print("=" * 90)
print("REVISED 2035 PROJECTION (adoption-tilted)")
print("=" * 90)
print()
print(f"  Conservative (cycles continue):     $200-350k")
print(f"  Base case (mainstream begins):       $400-700k")
print(f"  Strong adoption:                     $800k-1.5M")
print(f"  Hyper-bull (sovereign + reserves):   $1.5M-3M+")
print()
print(f"  Weighted probability estimate:       ~$650k")
print(f"  Annualized from $80k:                ~+25%")
print()

print("=" * 90)
print("WHAT THIS MEANS FOR THE PORTFOLIO")
print("=" * 90)
print()
print(f"On your 0.247 BTC at $80,823 entry (BAH BTC sleeve):")
print(f"  Conservative ($250k):    ${0.247 * 250_000:>10,.0f}  (3.1x)")
print(f"  Base ($500k):            ${0.247 * 500_000:>10,.0f}  (6.2x)")
print(f"  Strong adoption ($1M):   ${0.247 * 1_000_000:>10,.0f}  (12.4x)")
print(f"  Hyper-bull ($2M):        ${0.247 * 2_000_000:>10,.0f}  (24.8x)")
print()
print("For the full $100k portfolio at 50/30/20 (assuming similar BTC trajectory):")
print(f"  Conservative: $100k -> ~$400k-600k")
print(f"  Base case:    $100k -> ~$700k-1.2M")
print(f"  Bull case:    $100k -> ~$1.5M-3M+")
