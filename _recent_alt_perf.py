"""Check recent alt performance + identify if we're in a relief rally."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from core import data


ALTS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "AVAX/USDT",
        "LINK/USDT", "DOT/USDT", "ATOM/USDT", "DOGE/USDT", "ADA/USDT"]


print("=" * 90)
print("RECENT PERFORMANCE — what's happening NOW")
print("=" * 90)
print()
print(f"{'Pair':<14s} {'Price':>10s} {'7d':>8s} {'14d':>8s} {'30d':>8s} {'60d':>8s} "
      f"{'90d':>8s} {'From Feb low':>13s} {'From Oct top':>13s}")
print("-" * 90)

btc_df = data.ohlcv_extended("BTC/USDT", days_back=300)
btc_oct_top = btc_df.loc["2025-10-06", "close"]

for pair in ALTS:
    df = data.ohlcv_extended(pair, days_back=300)
    if df.empty:
        continue
    current = df["close"].iloc[-1]
    try:
        ret_7d = current / df["close"].iloc[-8] - 1
        ret_14d = current / df["close"].iloc[-15] - 1
        ret_30d = current / df["close"].iloc[-31] - 1
        ret_60d = current / df["close"].iloc[-61] - 1
        ret_90d = current / df["close"].iloc[-91] - 1

        # Find Feb 2026 low for this pair
        feb_data = df[(df.index >= "2026-01-15") & (df.index <= "2026-02-28")]
        if not feb_data.empty:
            feb_low = feb_data["low"].min()
            from_feb_low = current / feb_low - 1
        else:
            from_feb_low = None

        # Find Sep-Nov 2025 peak
        peak_data = df[(df.index >= "2025-09-01") & (df.index <= "2025-11-30")]
        peak = peak_data["high"].max() if not peak_data.empty else None
        from_peak = (current / peak - 1) if peak else None

        feb_str = f"{from_feb_low:+.1%}" if from_feb_low is not None else "n/a"
        peak_str = f"{from_peak:+.1%}" if from_peak is not None else "n/a"

        print(f"{pair:<14s} ${current:>9,.4f} {ret_7d:>+7.1%} {ret_14d:>+7.1%} "
              f"{ret_30d:>+7.1%} {ret_60d:>+7.1%} {ret_90d:>+7.1%}  "
              f"{feb_str:>12s}  {peak_str:>12s}")
    except Exception as e:
        print(f"{pair:<14s} error: {e}")


print()
print("=" * 90)
print("RELIEF RALLY ANALYSIS — past bear-market countertrend bounces")
print("=" * 90)
print()
print("Pattern: in EVERY prior BTC bear, alts had 30-60% relief rallies")
print("before the next leg down.")
print()
print("ETH in 2018 bear:")
print("  Initial crash: Jan 2018 $1400 -> Apr 2018 $370 (-74%)")
print("  Relief rally: Apr 2018 $370 -> May 2018 $830 (+124%)")
print("  Then crash: May 2018 $830 -> Dec 2018 $84 (-90%)")
print()
print("ETH in 2022 bear:")
print("  Initial crash: Nov 2021 $4800 -> Jun 2022 $880 (-82%)")
print("  Relief rally: Jun 2022 $880 -> Aug 2022 $2000 (+127%)")
print("  Then crash: Aug 2022 $2000 -> Nov 2022 $1070 (-46%)")
print()
print("SOL in 2022 bear:")
print("  Crash: Nov 2021 $260 -> Jun 2022 $26 (-90%)")
print("  Relief: Jun 2022 $26 -> Aug 2022 $46 (+77%)")
print("  Then: Aug 2022 $46 -> Dec 2022 $8 (-83%)")
print()
print("Lesson: relief rallies are 50-130% off the lows but DON'T MARK THE BOTTOM.")
print("They typically retrace 30-50% of the prior crash, then make new lows.")


print()
print("=" * 90)
print("WHERE ARE WE IN THE BEAR CYCLE")
print("=" * 90)
print()

# BTC dropped from $124k -> $63k -> $80k currently
btc_feb_low = btc_df.loc["2026-02-05", "close"]
btc_current = btc_df.iloc[-1]["close"]
btc_relief_so_far = btc_current / btc_feb_low - 1
print(f"BTC peak Oct 2025:      $124,659")
print(f"BTC Feb 2026 low:       ${btc_feb_low:>8,.0f}  (-50% from peak)")
print(f"BTC today:              ${btc_current:>8,.0f}  ({btc_relief_so_far:+.1%} from Feb low)")
print()
print("This IS a relief rally. BTC has bounced ~28% off Feb low.")
print("In prior bears, relief rallies hit 50-100% off lows before failing.")
print()

# Estimate when relief rally ends + next leg down
# Cycle 2 (2018): relief lasted 5-6 weeks
# Cycle 3 (2022): relief lasted ~7 weeks
# Cycle 4 (now): we're 3+ months from Feb low - already long

print("How long do relief rallies typically last?")
print("  2018 bear: ETH relief Apr->May = 5-6 weeks")
print("  2022 bear: ETH relief Jun->Aug = 7-8 weeks")
print("  Current cycle 4: Feb 5 -> May 11 = 14 weeks (already 2x longer than typical)")
print()
print("This relief rally is unusually extended. Either:")
print("  - Cycle pattern is breaking (institutional bid)")
print("  - Or the next leg down is closer than people think")
