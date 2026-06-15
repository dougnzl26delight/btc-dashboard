"""Diagnose what conditions the 4 losing shorts had at entry."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from core import data


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)


SHORT_ENTRIES = [
    ("OP/USDT",   "2026-03-22", 0.1146),
    ("BTC/USDT",  "2026-03-27", 66407.28),
    ("AVAX/USDT", "2026-03-27", 8.79),
    ("SOL/USDT",  "2026-03-29", 81.44),
]

print("Diagnosing what each losing short looked like AT ENTRY:")
print("=" * 100)
print(f"{'Pair':<12s} {'Entry Date':<12s} {'Entry':>10s} {'SMA200':>10s} {'%-SMA200':>10s} "
      f"{'SMA50':>10s} {'SMA50/200':>10s} {'RSI14':>7s} {'90d-low':>11s} {'%-from-low':>11s}")
print("-" * 100)

for pair, entry_date, entry_price in SHORT_ENTRIES:
    df = data.ohlcv_extended(pair, days_back=400)
    entry_ts = pd.Timestamp(entry_date, tz="UTC")
    # Get data UP TO entry date (not including)
    pre = df[df.index < entry_ts]
    if len(pre) < 200:
        print(f"{pair}: insufficient pre-entry data")
        continue
    last = float(pre["close"].iloc[-1])
    sma200 = float(pre["close"].rolling(200).mean().iloc[-1])
    sma50 = float(pre["close"].rolling(50).mean().iloc[-1])
    rsi14 = float(rsi(pre["close"], 14).iloc[-1])
    low_90d = float(pre["low"].tail(90).min())
    pct_below_sma200 = (last / sma200 - 1)
    sma_ratio = sma50 / sma200 - 1
    pct_from_low = (last - low_90d) / low_90d

    print(f"{pair:<12s} {entry_date:<12s} ${entry_price:>8.4f}  ${sma200:>8.4f}  "
          f"{pct_below_sma200:>+8.1%}  ${sma50:>8.4f}  {sma_ratio:>+8.1%}  "
          f"{rsi14:>6.1f}   ${low_90d:>9.4f}  {pct_from_low:>+9.1%}")

print()
print("INTERPRETATION:")
print("  - %-SMA200: how far below SMA200 at entry (negative = below)")
print("  - SMA50/200: SMA50 relative to SMA200 (negative = downtrend confirmed)")
print("  - RSI14: 14-day RSI (< 30 = oversold)")
print("  - %-from-low: how far above the 90-day low (small % = exhausted move)")
