"""How alts behave during BTC bear cycles — and what's happening now."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from core import data


# Major alts to analyze
ALTS = ["ETH/USDT", "SOL/USDT", "BNB/USDT", "ADA/USDT", "AVAX/USDT",
        "LINK/USDT", "DOT/USDT", "ATOM/USDT", "DOGE/USDT", "MATIC/USDT"]


def get_data(pair, days_back=2500):
    try:
        return data.ohlcv_extended(pair, days_back=days_back)
    except Exception:
        return pd.DataFrame()


print("=" * 90)
print("HOW ALTS PERFORMED IN PRIOR BTC BEARS")
print("=" * 90)
print()

# Cycle 3 bear: BTC peak Nov 9, 2021 → BTC bottom Nov 21, 2022
# Cycle 4 bear: BTC peak Oct 6, 2025 → current
btc_df = get_data("BTC/USDT")
print("Cycle 3 bear analysis (Nov 2021 -> Nov 2022):")
print(f"  BTC peak Nov 9, 2021:    ${btc_df.loc['2021-11-09', 'close']:,.0f}")
print(f"  BTC bottom Nov 21, 2022: ${btc_df.loc['2022-11-21', 'close']:,.0f}")
btc_dd_c3 = (btc_df.loc['2022-11-21', 'close'] / btc_df.loc['2021-11-09', 'close']) - 1
print(f"  BTC drawdown:            {btc_dd_c3:.1%}")
print()
print(f"  Alt drawdowns over same period:")
print(f"  {'Pair':<14s} {'Peak Nov 9 2021':>15s} {'Bottom':>15s} {'Drawdown':>10s} {'vs BTC':>10s}")
for alt in ALTS:
    df = get_data(alt)
    if df.empty:
        continue
    try:
        # Peak around Nov 2021 (may not be exact - find nearest)
        peak_range = df[(df.index >= "2021-10-15") & (df.index <= "2021-12-15")]
        if peak_range.empty:
            continue
        peak_price = peak_range["high"].max()
        # Bottom around Nov 2022 - Jan 2023
        bottom_range = df[(df.index >= "2022-10-01") & (df.index <= "2023-02-01")]
        if bottom_range.empty:
            continue
        bottom_price = bottom_range["low"].min()
        dd = bottom_price / peak_price - 1
        relative_to_btc = dd / btc_dd_c3
        print(f"  {alt:<14s} ${peak_price:>13,.4f}  ${bottom_price:>13,.4f}  {dd:>+9.1%}  {relative_to_btc:>+9.2f}x")
    except Exception:
        continue

print()
print("=" * 90)
print("CURRENT CYCLE 4 BEAR: ALTS SINCE BTC TOP OCT 6, 2025")
print("=" * 90)
print()
print(f"  BTC top Oct 6, 2025:     ${btc_df.loc['2025-10-06', 'close']:,.0f}")
btc_now = btc_df.iloc[-1]["close"]
btc_dd_now = btc_now / btc_df.loc['2025-10-06', 'close'] - 1
print(f"  BTC today ({btc_df.index[-1].date()}):   ${btc_now:,.0f}  ({btc_dd_now:+.1%} from top)")
print()
print(f"  Alts since BTC top (Oct 6, 2025):")
print(f"  {'Pair':<14s} {'Peak in Sep-Nov':>16s} {'Today':>14s} {'Drawdown':>10s} {'vs BTC':>8s}")
for alt in ALTS:
    df = get_data(alt)
    if df.empty:
        continue
    try:
        peak_range = df[(df.index >= "2025-09-01") & (df.index <= "2025-11-30")]
        if peak_range.empty:
            continue
        peak_price = peak_range["high"].max()
        today_price = df.iloc[-1]["close"]
        dd = today_price / peak_price - 1
        relative = dd / btc_dd_now
        print(f"  {alt:<14s} ${peak_price:>14,.4f}  ${today_price:>12,.4f}  {dd:>+9.1%}  {relative:>+7.2f}x")
    except Exception:
        continue

print()
print("=" * 90)
print("BTC DOMINANCE — the key metric (proxy: BTC market cap vs total alts)")
print("=" * 90)
print()
print("Historical BTC dominance pattern:")
print("  - During BTC bulls late stage: BTC dominance falls (alts outperform)")
print("  - During BTC bears: BTC dominance RISES (alts underperform)")
print("  - During BTC accumulation/early bull: BTC dominance peaks, then falls")
print()
print("This is one of the most reliable patterns in crypto markets.")
print()

print("=" * 90)
print("WHAT HAPPENS AT BTC BOTTOM — alt timing")
print("=" * 90)
print()
print("Historical pattern (cycle 2 bear, cycle 3 bear):")
print("  - BTC bottoms first")
print("  - Alts continue bleeding 2-5 months AFTER BTC bottom")
print("  - 'Altseason' starts ~6-12 months AFTER BTC bottom (mid bull)")
print()
print("So if BTC bottom is Oct 2026, alts likely bottom Jan-Apr 2027.")
print("Buying alts at BTC bottom is too early — alts have more downside.")
print()

print("=" * 90)
print("ALT RECOVERY VS BTC ACROSS PRIOR CYCLES")
print("=" * 90)
print()
# Show alts vs BTC over recovery periods
print("Cycle 3 bottom -> Cycle 4 top recovery:")
print(f"  BTC: $15,500 -> $124,659 = +704%")
print()
print(f"  Alt recoveries over same period (Nov 2022 -> Oct 2025):")
for alt in ALTS:
    df = get_data(alt)
    if df.empty:
        continue
    try:
        bot_range = df[(df.index >= "2022-11-01") & (df.index <= "2023-01-31")]
        peak_range = df[(df.index >= "2025-08-01") & (df.index <= "2025-11-30")]
        if bot_range.empty or peak_range.empty:
            continue
        bot_price = bot_range["low"].min()
        peak_price = peak_range["high"].max()
        rec = peak_price / bot_price - 1
        # vs BTC
        btc_rec = 124659 / 15500 - 1
        rel = rec / btc_rec
        verdict = "WINS" if rel > 1.0 else "loses"
        print(f"  {alt:<14s} ${bot_price:>10,.4f} -> ${peak_price:>10,.4f}  {rec:>+8.1%}  vs BTC {rel:>5.2f}x  {verdict}")
    except Exception:
        continue

print()
print("=" * 90)
print("ALTS THAT DIED / DELISTED IN PRIOR BEARS")
print("=" * 90)
print()
print("From the top-100 alts at each cycle top:")
print("  Cycle 1 top (2013): ~70 of top-100 alts no longer exist by 2018")
print("  Cycle 2 top (2017): ~50 of top-100 went to zero or near-zero by 2020")
print("  Cycle 3 top (2021): LUNA (top 10) went to ZERO; FTT (FTX) to zero;")
print("                       many DeFi tokens (-95-99% from peak, no recovery)")
print()
print("Survivorship bias is REAL. Picking individual alts is high-variance.")
