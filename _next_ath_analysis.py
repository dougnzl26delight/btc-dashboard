"""Next ATH timing + magnitude — analyzing bottom-to-next-top across cycles."""
from datetime import date
import pandas as pd


HALVINGS = [
    ("Halving 1", date(2012, 11, 28)),
    ("Halving 2", date(2016, 7, 9)),
    ("Halving 3", date(2020, 5, 11)),
    ("Halving 4", date(2024, 4, 19)),
    ("Halving 5", date(2028, 4, 1)),  # estimated
]

CYCLES = [
    # (cycle, halving_date, halving_price, top_date, top_price, bottom_date, bottom_price)
    (1, date(2012, 11, 28), 12.50, date(2013, 11, 30), 1163, date(2015, 1, 14), 152),
    (2, date(2016, 7, 9), 657, date(2017, 12, 17), 19783, date(2018, 12, 15), 3200),
    (3, date(2020, 5, 11), 8821, date(2021, 11, 9), 69000, date(2022, 11, 21), 15500),
    (4, date(2024, 4, 19), 64200, date(2025, 10, 6), 126200, None, None),  # bottom TBD
]


def mo(d1, d2):
    return (d2 - d1).days / 30.4


print("=" * 90)
print("BOTTOM -> NEXT-TOP TIMING (the all-important cycle interval)")
print("=" * 90)
print()
print(f"{'Cycle pair':<22s} {'Bot date':<12s} {'Next top':<12s} {'Months':>10s} "
      f"{'Bot $':>10s} {'Top $':>10s} {'Multiple':>10s}")
print("-" * 95)

bot_to_next_top_months = []
bot_to_next_top_multiples = []
for i in range(3):
    cy_n = CYCLES[i]
    cy_next = CYCLES[i + 1]
    if cy_n[5] is None or cy_next[3] is None:
        continue
    bot_date = cy_n[5]
    bot_price = cy_n[6]
    next_top_date = cy_next[3]
    next_top_price = cy_next[4]
    months = mo(bot_date, next_top_date)
    multiple = next_top_price / bot_price
    bot_to_next_top_months.append(months)
    bot_to_next_top_multiples.append(multiple)
    print(f"Cycle {cy_n[0]} bot -> {cy_next[0]} top  "
          f"{bot_date}   {next_top_date}   "
          f"{months:>8.1f}mo  "
          f"${bot_price:>8,.0f}  ${next_top_price:>8,.0f}   "
          f"{multiple:>8.1f}x")

print()
print(f"Mean months bottom-to-next-top:    {sum(bot_to_next_top_months)/len(bot_to_next_top_months):.1f}")
print(f"Std dev (range):                    {max(bot_to_next_top_months)-min(bot_to_next_top_months):.1f}")
print()

print("=" * 90)
print("TOP -> TOP TIMING AND MAGNITUDE")
print("=" * 90)
print()
print(f"{'Cycle pair':<22s} {'Top dates':<28s} {'Months':>10s} "
      f"{'Top->Top mult':>15s} {'Decay rate':>12s}")
print("-" * 95)

top_to_top_months = []
top_to_top_multiples = []
for i in range(3):
    cy_n = CYCLES[i]
    cy_next = CYCLES[i + 1]
    months = mo(cy_n[3], cy_next[3])
    mult = cy_next[4] / cy_n[4]
    top_to_top_months.append(months)
    top_to_top_multiples.append(mult)
    decay = ""
    if i > 0:
        decay = f"{mult / top_to_top_multiples[i - 1]:.2f}x"
    print(f"Cycle {cy_n[0]} -> {cy_next[0]} top   "
          f"{cy_n[3]} -> {cy_next[3]}  "
          f"{months:>8.1f}mo   {mult:>13.2f}x  {decay:>10s}")

print()
print("=" * 90)
print("CYCLE 5 PROJECTION (assuming cycle 4 bottom ~Oct 24, 2026)")
print("=" * 90)

# Average of prior 3 cycles
mean_bot_to_top = sum(bot_to_next_top_months) / len(bot_to_next_top_months)
projected_c4_bottom = date(2026, 10, 24)
projected_c5_top_avg = projected_c4_bottom + pd.Timedelta(days=mean_bot_to_top * 30.4)
print(f"  Cycle 4 bottom (projected):  {projected_c4_bottom}")
print(f"  Mean bottom->top months:     {mean_bot_to_top:.1f}")
print(f"  Projected cycle 5 ATH date:  {projected_c5_top_avg}")
print()
print(f"  Halving 5 (est):              {HALVINGS[4][1]}")
print(f"  Halving 5 -> projected top:   {mo(HALVINGS[4][1], projected_c5_top_avg):.1f} months")
print()

print("=" * 90)
print("CYCLE 5 ATH PRICE PROJECTION")
print("=" * 90)
print()
print("Approach 1: Top-to-top multiples (diminishing pattern)")
print(f"  C2/C1: 17.0x   C3/C2: 3.49x   C4/C3: 1.83x")
print(f"  Decay ratio: C3-vs-C2 = {3.49/17.0:.2f}, C4-vs-C3 = {1.83/3.49:.2f}")
print(f"  If decay continues at 0.5x:  C5/C4 = 1.83 * 0.5 = 0.92x  -> C5 top would be LOWER (unlikely)")
print(f"  If decay stabilizes at 0.5:  C5/C4 = 0.92x -> $116k (BELOW current cycle top — unrealistic)")
print()

# More reasonable: assume diminishing trend stabilizes
print("Approach 2: Conservative diminishing returns (likely floor)")
c4_top = 126200
scenarios = {
    "1.4x C4 (extreme diminishing)": c4_top * 1.4,
    "1.5x C4 (mild diminishing)":    c4_top * 1.5,
    "1.83x C4 (same as C4/C3)":      c4_top * 1.83,
    "2.5x C4 (institutional rally)": c4_top * 2.5,
    "3.5x C4 (decay reverses)":      c4_top * 3.5,
}
for label, price in scenarios.items():
    print(f"  {label:<35s} -> ${price:>10,.0f}")
print()

print("Approach 3: Bottom-to-next-top multiples (per historical pattern)")
print(f"  C1 bot $152 -> C2 top $19,783 = 130x")
print(f"  C2 bot $3,200 -> C3 top $69,000 = 21.6x")
print(f"  C3 bot $15,500 -> C4 top $126,200 = 8.1x")
print(f"  Decay: 130/21.6 = 6.0x reduction, 21.6/8.1 = 2.7x reduction")
print()

# Project from various cycle 4 bottoms
print(f"  If cycle 4 bottom = X, cycle 5 ATH = X * (8.1 / [decay])")
print(f"{'C4 bottom':<15s} {'C5/C4 multiple':>16s} {'Cycle 5 ATH':>15s}")
for c4_bot in [25000, 35000, 50000, 60000]:
    # Continuing decay
    for next_mult_label, next_mult in [
        ("3x (severe decay)", 3.0),
        ("4x (moderate decay)", 4.0),
        ("5x (mild decay)", 5.0),
    ]:
        c5_ath = c4_bot * next_mult
        print(f"  ${c4_bot:>10,.0f}    {next_mult_label:<25s}   ${c5_ath:>12,.0f}")
    print()

print("=" * 90)
print("MOST LIKELY CYCLE 5 SCENARIO (weighted average)")
print("=" * 90)
print()
print("Combining the most-likely cycle 4 bottom ($35-45k range)")
print("with most-likely C5/C4-bottom multiplier (4-5x):")
print()
print(f"{'Scenario':<35s} {'Bottom':>10s} {'Multiplier':>12s} {'ATH':>15s}")
print(f"  Conservative (-65% bear, 3x rebound) ${44000:>10,.0f}        3.0x      ${44000*3:>12,.0f}")
print(f"  Base case (-72% bear, 4x rebound)    ${35000:>10,.0f}        4.0x      ${35000*4:>12,.0f}")
print(f"  Aggressive (-72% bear, 5x rebound)   ${35000:>10,.0f}        5.0x      ${35000*5:>12,.0f}")
print(f"  Supercycle (-50% bear, 6x rebound)   ${63000:>10,.0f}        6.0x      ${63000*6:>12,.0f}")
print()

print("=" * 90)
print("TIMING SUMMARY")
print("=" * 90)
print()
print(f"  Cycle 4 bottom (projected):  Oct 2026 (~5 months away)")
print(f"  Halving 5:                    April 2028 (~24 months away)")
print(f"  Cycle 5 ATH (projected):     {projected_c5_top_avg}")
print(f"                               (~{mo(date.today(), projected_c5_top_avg):.1f} months from today)")
print()
print(f"  Bottom-to-next-ATH timing has been EXACTLY consistent across 3 cycles:")
for i, m in enumerate(bot_to_next_top_months):
    print(f"    Cycle {i+1}->{i+2}: {m:.1f} months")
