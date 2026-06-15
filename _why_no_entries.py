"""Diagnose why pro_trend hasn't taken systematic entries."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import data
from strategies import pro_trend


print("PRO_TREND ENTRY GATE ANALYSIS")
print("=" * 80)
print(f"Entry rules:")
print(f"  LONG:  price > SMA200 AND today's high >= 20-day Donchian high")
print(f"  SHORT: price < SMA200 AND today's low <= 20-day Donchian low")
print(f"  ATR period: {pro_trend.ATR_PERIOD}d, ATR stop multiplier: {pro_trend.ATR_STOP_MULT}")
print()

print(f"{'Pair':<12s} {'Price':>10s} {'SMA200':>10s} {'Regime':<6s} "
      f"{'Donch-Hi':>10s} {'Donch-Lo':>10s} {'Dist to LONG':>13s} {'Dist to SHORT':>14s}")
print("-" * 100)

for pair in pro_trend.PRO_TREND_PAIRS:
    df = data.ohlcv_extended(pair, days_back=250)
    if df.empty or len(df) < 220:
        print(f"{pair:<12s} insufficient data")
        continue
    last = float(df["close"].iloc[-1])
    sma200 = float(df["close"].rolling(200).mean().iloc[-1])
    donch_hi = float(df["high"].rolling(20).max().shift(1).iloc[-1])
    donch_lo = float(df["low"].rolling(20).min().shift(1).iloc[-1])

    in_bull = last > sma200
    regime = "BULL" if in_bull else "BEAR"

    # To trigger LONG: need price > SMA200 AND high >= donch_hi
    # We're below SMA200 — so first need price to RISE +X% to SMA200
    pct_to_sma = (sma200 - last) / last if not in_bull else 0
    pct_to_donch_hi = (donch_hi - last) / last
    dist_long = max(pct_to_sma, pct_to_donch_hi)

    # To trigger SHORT: need price < SMA200 (TRUE) AND low <= donch_lo
    # We're already below SMA200, so just need price to FALL to donch_lo
    pct_to_donch_lo = (last - donch_lo) / last  # positive if above donch_lo
    dist_short = pct_to_donch_lo if not in_bull else float('inf')

    print(f"{pair:<12s} ${last:>8.4f}  ${sma200:>8.4f}  {regime:<5s} "
          f" ${donch_hi:>8.4f}  ${donch_lo:>8.4f}  "
          f"{dist_long:>+12.1%}  {dist_short:>+13.1%}")

print()
print("INTERPRETATION:")
print("  - 'Dist to LONG' = % move needed UP for long trigger (needs price > SMA200 + Donch break)")
print("  - 'Dist to SHORT' = % move needed DOWN for short trigger (price has to drop to 20d low)")
print("  - LARGE positive = far from triggering; negative would mean already triggered")
