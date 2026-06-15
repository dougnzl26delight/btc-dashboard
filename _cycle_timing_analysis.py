"""Halving-cycle timing analysis — are the time intervals consistent?"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from core import data


# Known cycle structure
HALVINGS = [
    ("Halving 1", date(2012, 11, 28)),
    ("Halving 2", date(2016, 7, 9)),
    ("Halving 3", date(2020, 5, 11)),
    ("Halving 4", date(2024, 4, 19)),
]

# Cycle tops (post-halving peaks)
TOPS = [
    ("Cycle 1 top", date(2013, 11, 30), 1163),   # historical
    ("Cycle 2 top", date(2017, 12, 17), 19783),  # historical
    ("Cycle 3 top", date(2021, 11, 9), 69000),
    ("Cycle 4 top", date(2025, 10, 6), 126200),  # confirmed
]

# Cycle bottoms (post-top capitulation lows)
BOTTOMS = [
    ("Cycle 1 bottom", date(2015, 1, 14), 152),
    ("Cycle 2 bottom", date(2018, 12, 15), 3200),
    ("Cycle 3 bottom", date(2022, 11, 21), 15500),
    ("Cycle 4 bottom", None, None),  # TBD
]


def months_between(d1, d2):
    return (d2 - d1).days / 30.4


print("=" * 90)
print("HALVING-CYCLE TIMING ANALYSIS")
print("=" * 90)
print()

print(f"{'Cycle':<10s} {'Halving':<12s} {'Top':<12s} {'Bottom':<12s} "
      f"{'Halv->Top':>10s} {'Top->Bot':>10s} {'Halv->Bot':>10s}")
print("-" * 90)

for i in range(4):
    halv_name, halv_date = HALVINGS[i]
    top_name, top_date, top_price = TOPS[i]
    bot_name, bot_date, bot_price = BOTTOMS[i]

    halv_to_top = months_between(halv_date, top_date) if top_date else None
    top_to_bot = months_between(top_date, bot_date) if bot_date else None
    halv_to_bot = months_between(halv_date, bot_date) if bot_date else None

    print(f"Cycle {i+1:<3d}  {halv_date}   {top_date}   "
          f"{str(bot_date) if bot_date else '?':<12s} "
          f"{halv_to_top:>9.1f}mo "
          f"{top_to_bot if top_to_bot else '?':>9}{'mo' if top_to_bot else ''} "
          f"{halv_to_bot if halv_to_bot else '?':>9}{'mo' if halv_to_bot else ''}")

print()
print("=" * 90)
print("PRICE MAGNITUDE PER CYCLE")
print("=" * 90)
print(f"{'Cycle':<10s} {'Halving$':<12s} {'Top$':<12s} {'Top/Halv':>10s} "
      f"{'Bottom$':<12s} {'Bot/Top':>10s} {'Cycle gain':>12s}")
print("-" * 90)

halving_prices = [12.50, 657, 8821, 64200]  # approximate BTC price at each halving
for i in range(4):
    halv_name, halv_date = HALVINGS[i]
    halv_price = halving_prices[i]
    top_name, top_date, top_price = TOPS[i]
    bot_name, bot_date, bot_price = BOTTOMS[i]

    top_mult = top_price / halv_price if top_price and halv_price else None
    bot_mult = bot_price / top_price if bot_price and top_price else None
    cycle_gain = (bot_price / halving_prices[i-1] - 1) if i > 0 and bot_price else None

    bot_str = f"${bot_price:,.0f}" if bot_price else "?"
    bot_mult_str = f"{bot_mult:.2f}" if bot_mult else "?"
    gain_str = f"{cycle_gain:+.0%}" if cycle_gain is not None else "?"
    print(f"Cycle {i+1:<3d}  ${halv_price:>10,.0f}  ${top_price:>10,.0f}  "
          f"{top_mult:>9.1f}x  {bot_str:<11s} {bot_mult_str:>9s}   {gain_str:>10s}")

print()
print("=" * 90)
print("CYCLE 4 PROJECTION using each prior cycle's exact timing")
print("=" * 90)
print()

cycle4_halv = date(2024, 4, 19)
cycle4_top = date(2025, 10, 6)
today = date.today()

print(f"Cycle 4 halving:   {cycle4_halv}")
print(f"Cycle 4 top:       {cycle4_top}")
print(f"Today:             {today}")
print(f"Months halving->top:    {months_between(cycle4_halv, cycle4_top):.1f}")
print(f"Months top->today:      {months_between(cycle4_top, today):.1f}")
print()

print(f"{'Reference':<22s} {'Top->Bottom':>11s} {'Halv->Bottom':>13s} {'Projected Bottom Date':<22s}")
print("-" * 80)

for i in range(3):
    halv = HALVINGS[i][1]
    top = TOPS[i][1]
    bot = BOTTOMS[i][1]
    top_to_bot_days = (bot - top).days
    halv_to_bot_days = (bot - halv).days

    proj_from_top = cycle4_top + pd.Timedelta(days=top_to_bot_days)
    proj_from_halv = cycle4_halv + pd.Timedelta(days=halv_to_bot_days)

    print(f"  Per Cycle {i+1:<2d}        "
          f"{months_between(top, bot):>9.1f}mo   "
          f"{months_between(halv, bot):>11.1f}mo   "
          f"  {proj_from_top} (top+) / {proj_from_halv} (halv+)")

print()
# Mean timing
mean_top_to_bot = sum((BOTTOMS[i][1] - TOPS[i][1]).days for i in range(3)) / 3
mean_halv_to_bot = sum((BOTTOMS[i][1] - HALVINGS[i][1]).days for i in range(3)) / 3
proj_from_top_mean = cycle4_top + pd.Timedelta(days=mean_top_to_bot)
proj_from_halv_mean = cycle4_halv + pd.Timedelta(days=mean_halv_to_bot)
print(f"  AVERAGE          {mean_top_to_bot/30.4:>9.1f}mo   {mean_halv_to_bot/30.4:>11.1f}mo   "
      f"  {proj_from_top_mean} (top+) / {proj_from_halv_mean} (halv+)")
print()

print("=" * 90)
print("PERCENT THROUGH THE BEAR")
print("=" * 90)
print()
today_dt = date.today()
top_to_today_days = (today_dt - cycle4_top).days

for i in range(3):
    top = TOPS[i][1]
    bot = BOTTOMS[i][1]
    cycle_dur = (bot - top).days
    pct = top_to_today_days / cycle_dur * 100
    days_remaining = cycle_dur - top_to_today_days
    print(f"  If cycle 4 mirrors cycle {i+1}: we are {pct:.0f}% through bear, "
          f"~{days_remaining/30.4:.1f} months until bottom")

avg_cycle_days = mean_top_to_bot
avg_pct = top_to_today_days / avg_cycle_days * 100
avg_remaining = avg_cycle_days - top_to_today_days
print(f"  Average pattern:           we are {avg_pct:.0f}% through bear, "
      f"~{avg_remaining/30.4:.1f} months until bottom")

print()
print("=" * 90)
print("BTC ATH RETRACEMENT TARGETS based on prior cycle drawdowns")
print("=" * 90)

cycle4_peak = 126200
df = data.ohlcv_extended("BTC/USDT", days_back=300)
current_btc = float(df["close"].iloc[-1])
current_dd = (current_btc / cycle4_peak - 1)
print(f"  Cycle 4 peak:    ${cycle4_peak:,.0f}")
print(f"  Current BTC:     ${current_btc:,.0f}  ({current_dd:+.1%} from peak)")
print()
print(f"{'Reference':<20s} {'Drawdown':>10s} {'Bottom target':>16s} {'From here':>11s}")
print("-" * 80)
for i in range(3):
    top_p = TOPS[i][2]
    bot_p = BOTTOMS[i][2]
    dd = bot_p / top_p - 1
    target = cycle4_peak * (1 + dd)
    from_here = target / current_btc - 1
    print(f"  Cycle {i+1} pattern    {dd:>9.1%}  "
          f"${target:>13,.0f}  {from_here:>+10.1%}")

avg_dd = sum((BOTTOMS[i][1] - TOPS[i][1]).days for i in range(3))  # wrong; recompute drawdowns
drawdowns = [BOTTOMS[i][2] / TOPS[i][2] - 1 for i in range(3)]
avg_dd_pct = sum(drawdowns) / 3
target_avg = cycle4_peak * (1 + avg_dd_pct)
from_here_avg = target_avg / current_btc - 1
print(f"  AVERAGE          {avg_dd_pct:>9.1%}  ${target_avg:>13,.0f}  {from_here_avg:>+10.1%}")
