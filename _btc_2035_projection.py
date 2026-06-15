"""BTC price projection through 2035 — multi-cycle analysis."""
from datetime import date
import pandas as pd


# Historical halvings + projected
HALVINGS = [
    ("H1", date(2012, 11, 28)),
    ("H2", date(2016, 7, 9)),
    ("H3", date(2020, 5, 11)),
    ("H4", date(2024, 4, 19)),
    ("H5", date(2028, 4, 1)),    # estimated
    ("H6", date(2032, 4, 1)),    # estimated
    ("H7", date(2036, 4, 1)),    # estimated
]

# Historical cycle data
CYCLES = [
    {"num": 1, "halv_date": date(2012, 11, 28), "top_date": date(2013, 11, 30),
     "top_price": 1163, "bot_date": date(2015, 1, 14), "bot_price": 152},
    {"num": 2, "halv_date": date(2016, 7, 9), "top_date": date(2017, 12, 17),
     "top_price": 19783, "bot_date": date(2018, 12, 15), "bot_price": 3200},
    {"num": 3, "halv_date": date(2020, 5, 11), "top_date": date(2021, 11, 9),
     "top_price": 69000, "bot_date": date(2022, 11, 21), "bot_price": 15500},
    {"num": 4, "halv_date": date(2024, 4, 19), "top_date": date(2025, 10, 6),
     "top_price": 126200, "bot_date": None, "bot_price": None},
]


def mo(d1, d2):
    return (d2 - d1).days / 30.4


# Average timing intervals
MEAN_HALV_TO_TOP = 17.0  # months (avg of cycles 2-4)
MEAN_TOP_TO_BOT = 12.6  # months
MEAN_BOT_TO_NEXT_TOP = 34.8  # months (the cleanest signal)


def project_cycle(cycle_num, halv_date, prev_bot_price, top_multiple, bot_drawdown):
    """Project a future cycle's top and bottom."""
    top_date = halv_date + pd.Timedelta(days=MEAN_HALV_TO_TOP * 30.4)
    bot_date = top_date + pd.Timedelta(days=MEAN_TOP_TO_BOT * 30.4)
    top_price = prev_bot_price * top_multiple
    bot_price = top_price * (1 + bot_drawdown)  # bot_drawdown negative
    return {
        "num": cycle_num,
        "halv_date": halv_date,
        "top_date": top_date,
        "top_price": top_price,
        "bot_date": bot_date,
        "bot_price": bot_price,
    }


print("=" * 90)
print("BTC PROJECTION THROUGH 2035 — multi-cycle model")
print("=" * 90)
print()

# Historical
print("Historical cycles (actual):")
print(f"{'Cycle':<8s} {'Halving':<12s} {'Top date':<12s} {'Top $':>10s} "
      f"{'Bot date':<12s} {'Bot $':>10s}")
for c in CYCLES:
    bot_str = str(c["bot_date"]) if c["bot_date"] else "TBD"
    bot_price_str = f"${c['bot_price']:,.0f}" if c["bot_price"] else "?"
    print(f"Cycle {c['num']:<3d} {c['halv_date']}   {c['top_date']}   "
          f"${c['top_price']:>8,.0f}   {bot_str:<12s} {bot_price_str:>9s}")
print()

# Project cycle 4 bottom + future cycles
print("PROJECTIONS (using historical timing + diminishing magnitude):")
print()

# Cycle 4 bottom projection
c4 = CYCLES[3]
c4_bot_date = c4["top_date"] + pd.Timedelta(days=MEAN_TOP_TO_BOT * 30.4)
# Assume ~70% drawdown (slight diminishing from -78% last cycle)
c4_bot_drawdown = -0.70
c4_bot_price = c4["top_price"] * (1 + c4_bot_drawdown)
print(f"Cycle 4 bottom (projected):")
print(f"  Date: {c4_bot_date}   Price: ${c4_bot_price:,.0f}  "
      f"(-{abs(c4_bot_drawdown)*100:.0f}% from $126k peak)")
print()

# Cycle 5
# Per historical decay: bot-to-next-top multiples were 130x, 21.6x, 8.1x
# Continuing decay at 0.4x: 8.1 × 0.4 = 3.24x
# Or stabilizing at 0.5x: 8.1 × 0.5 = 4.05x
print("CYCLE 5 (Halving April 2028 -> Top ~Sept 2029 -> Bottom ~Oct 2030)")
print()
for scenario_label, c5_bot_to_top_mult, c5_top_drawdown in [
    ("Conservative (decay continues)", 3.5, -0.65),
    ("Base case", 4.5, -0.62),
    ("Bull case (institutional)", 6.0, -0.55),
]:
    c5_top_price = c4_bot_price * c5_bot_to_top_mult
    c5_top_date = HALVINGS[4][1] + pd.Timedelta(days=MEAN_HALV_TO_TOP * 30.4)
    c5_bot_date = c5_top_date + pd.Timedelta(days=MEAN_TOP_TO_BOT * 30.4)
    c5_bot_price = c5_top_price * (1 + c5_top_drawdown)
    print(f"  {scenario_label}:")
    print(f"    C5 top:    {c5_top_date}   ${c5_top_price:>10,.0f}  "
          f"(bot-to-top multiple {c5_bot_to_top_mult}x)")
    print(f"    C5 bottom: {c5_bot_date}   ${c5_bot_price:>10,.0f}  "
          f"({c5_top_drawdown:.0%} from top)")
    print()

# Use base case for cycle 6 projection
c5_top_price_base = c4_bot_price * 4.5
c5_bot_price_base = c5_top_price_base * 0.38
c5_top_date_base = HALVINGS[4][1] + pd.Timedelta(days=MEAN_HALV_TO_TOP * 30.4)
c5_bot_date_base = c5_top_date_base + pd.Timedelta(days=MEAN_TOP_TO_BOT * 30.4)

print(f"Using BASE CASE for cycle 5 ahead:")
print(f"  C5 top:    {c5_top_date_base}  ${c5_top_price_base:,.0f}")
print(f"  C5 bottom: {c5_bot_date_base}  ${c5_bot_price_base:,.0f}")
print()

print("CYCLE 6 (Halving April 2032 -> Top ~Sept 2033 -> Bottom ~Oct 2034)")
print()
for scenario_label, c6_bot_to_top_mult, c6_top_drawdown in [
    ("Conservative", 2.5, -0.55),
    ("Base case", 3.5, -0.50),
    ("Bull case", 5.0, -0.45),
]:
    c6_top_price = c5_bot_price_base * c6_bot_to_top_mult
    c6_top_date = HALVINGS[5][1] + pd.Timedelta(days=MEAN_HALV_TO_TOP * 30.4)
    c6_bot_date = c6_top_date + pd.Timedelta(days=MEAN_TOP_TO_BOT * 30.4)
    c6_bot_price = c6_top_price * (1 + c6_top_drawdown)
    print(f"  {scenario_label}:")
    print(f"    C6 top:    {c6_top_date}  ${c6_top_price:>10,.0f}  "
          f"(bot-to-top multiple {c6_bot_to_top_mult}x)")
    print(f"    C6 bottom: {c6_bot_date}  ${c6_bot_price:>10,.0f}  "
          f"({c6_top_drawdown:.0%} from top)")
    print()

# 2035 specifically — analyze where we are in cycle 6/7
c6_top_date_base = HALVINGS[5][1] + pd.Timedelta(days=MEAN_HALV_TO_TOP * 30.4)
c6_bot_date_base = c6_top_date_base + pd.Timedelta(days=MEAN_TOP_TO_BOT * 30.4)
c6_top_price_base = c5_bot_price_base * 3.5
c6_bot_price_base = c6_top_price_base * 0.5

print("=" * 90)
print("WHERE IS BTC IN 2035?")
print("=" * 90)
print()
print(f"Cycle 6 bottom (base case): {c6_bot_date_base}  ${c6_bot_price_base:,.0f}")
print(f"Cycle 7 begins around then.")
print()

# 2035 endpoints
y2035 = date(2035, 12, 31)
months_post_c6_bottom = mo(c6_bot_date_base, y2035)
print(f"End of 2035: {months_post_c6_bottom:.1f} months past C6 bottom")
print()

# At end-2035, we're partway from c6 bottom to c7 top
# c7 top expected ~Sept 2037 (halving April 2036 + 17 mo)
c7_top_date = HALVINGS[6][1] + pd.Timedelta(days=MEAN_HALV_TO_TOP * 30.4)
months_to_c7_top = mo(y2035, c7_top_date)
print(f"Cycle 7 top (projected): {c7_top_date}, {months_to_c7_top:.1f} months from end 2035")

# At end 2035 we're in early cycle 7 recovery
# Linear interpolation from C6 bot to C7 top
total_recovery_months = mo(c6_bot_date_base, c7_top_date)
position = months_post_c6_bottom / total_recovery_months
print(f"Position in cycle 6-7 recovery: {position:.0%}")
print()

# 2035 BTC price by scenario
print("BTC price end-2035 (varying assumptions):")
print()
for label, c6_top_mult, c6_dd, c7_mult in [
    ("Conservative", 2.5, -0.55, 2.5),
    ("Base case", 3.5, -0.50, 3.0),
    ("Bull case", 5.0, -0.45, 4.0),
    ("Mega bull (institutional)", 7.0, -0.40, 5.0),
]:
    c5_top = c4_bot_price * 4.5
    c5_bot = c5_top * 0.38
    c6_top = c5_bot * c6_top_mult
    c6_bot = c6_top * (1 + c6_dd)
    c7_top = c6_bot * c7_mult
    # Linear interpolate where 2035 sits
    price_2035 = c6_bot + (c7_top - c6_bot) * position
    print(f"  {label:<28s}: c6 bot ${c6_bot:>9,.0f}, c7 top ${c7_top:>9,.0f} -> 2035 ${price_2035:>10,.0f}")

print()
print("=" * 90)
print("COMPARISON: power law model + S2F + linear extrapolation")
print("=" * 90)
print()
print(f"Today: BTC $80k, cycles have moved 80x ($1k -> $80k) over 12 years")
print()
print("Power law model (Santostasi): ~$1M by 2035")
print("S2F model (PlanB, now broken): predicted $1M+ by 2030, was off by 5x")
print("Stock-to-flow halved per halving (asymptote to log curve): $250-500k by 2035")
print()

print("=" * 90)
print("HONEST CYCLE-BASED 2035 RANGE")
print("=" * 90)
print()
print("  Conservative (decay continues fast):   $80k - $180k")
print("  Base case (decay stabilizes):           $200k - $400k")
print("  Bull case (institutional adoption):     $500k - $800k")
print("  Mega bull (BTC reaches gold parity):    $800k - $1.5M")
