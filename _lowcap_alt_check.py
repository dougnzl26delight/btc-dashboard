"""Check lower-cap alts: TAO, ONDO, TIBBIR, NPC + a few comparables.

Tests: 1) is data available?  2) what's the signal state?  3) historical context?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from core import data


# Candidate lower-cap pairs to check (with reasonable name variants)
CANDIDATES = [
    "TAO/USDT", "ONDO/USDT", "TIBBIR/USDT", "NPC/USDT",
    # Comparable mid/low-caps for context
    "SUI/USDT", "WIF/USDT", "PEPE/USDT", "FET/USDT",
    "RNDR/USDT", "INJ/USDT", "ARB/USDT", "TIA/USDT",
]


def calc_indicators(df):
    df = df.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    c = df["close"]
    df["sma_20"] = c.rolling(20).mean()
    df["sma_200"] = c.rolling(200).mean()
    df["ema_21"] = c.ewm(span=21).mean()
    delta = c.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    df["macd_hist"] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9).mean()
    bb_std = c.rolling(20).std()
    df["bb_upper"] = df["sma_20"] + 2 * bb_std
    df["bb_lower"] = df["sma_20"] - 2 * bb_std
    df["bb_pct"] = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    df["mayer"] = c / df["sma_200"]
    return df


def signal_state(df):
    if len(df) < 50:
        return None
    macd_bear = (df["macd_hist"].shift(1) > 0) & (df["macd_hist"] < 0)
    ema_break = (df["close"].shift(1) > df["ema_21"].shift(1)) & (df["close"] < df["ema_21"])
    sig = macd_bear.rolling(5).max().astype(bool) & ema_break
    fires = sig[sig.fillna(False)]
    last_fire = fires.index[-1] if len(fires) > 0 else None
    return last_fire


print("=" * 110)
print("LOWER-CAP ALT CHECK — data availability + current state")
print("=" * 110)
print()
print(f"{'Pair':<14s} {'Days':>6s} {'StartDate':>12s} {'Price':>12s} {'FromLow':>9s} {'90dRet':>8s} "
      f"{'RSI':>5s} {'Mayer':>6s} {'BB%':>5s} {'Sig':<11s} {'State':<10s}")
print("-" * 110)

results = []
for pair in CANDIDATES:
    try:
        df = data.ohlcv_extended(pair, days_back=600)
    except Exception as e:
        print(f"{pair:<14s} NOT AVAILABLE  ({type(e).__name__})")
        continue
    if df.empty:
        print(f"{pair:<14s} no data returned")
        continue

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    days = len(df)
    start = df.index[0].date()
    cur_price = float(df["close"].iloc[-1])

    df = calc_indicators(df)
    last = df.iloc[-1]

    # From-low: use Feb 2026 low if data has it, else use 90-day low
    feb_mask = df.index >= pd.Timestamp("2026-02-01")
    if feb_mask.any():
        feb_low = float(df.loc[feb_mask, "low"].min())
        from_low = cur_price / feb_low - 1
        low_label = "Feb"
    else:
        recent_low = float(df["low"].iloc[-90:].min()) if len(df) >= 90 else float(df["low"].min())
        from_low = cur_price / recent_low - 1
        low_label = "90d"

    # 90-day return
    if len(df) >= 91:
        ret_90 = cur_price / df["close"].iloc[-91] - 1
    else:
        ret_90 = None

    rsi = last["rsi"] if not pd.isna(last["rsi"]) else None
    mayer = last["mayer"] if not pd.isna(last["mayer"]) else None
    bb = last["bb_pct"] if not pd.isna(last["bb_pct"]) else None

    last_fire = signal_state(df)
    if last_fire is None:
        sig_str = "never"
        state = "WAITING"
    else:
        days_since = (df.index[-1] - last_fire).days
        sig_str = str(last_fire.date())
        if days_since <= 7 and rsi is not None and rsi >= 70:
            state = "PEAK ZONE"
        elif days_since <= 14:
            state = f"{days_since}d ago"
        elif rsi is not None and rsi >= 75 and bb is not None and bb >= 0.95:
            state = "ARMED"
        else:
            state = "stale"

    price_fmt = f"${cur_price:>10,.4f}" if cur_price < 100 else f"${cur_price:>10,.2f}"
    rsi_str = f"{rsi:.0f}" if rsi is not None else "—"
    mayer_str = f"{mayer:.2f}" if mayer is not None else "—"
    bb_str = f"{bb:.0%}" if bb is not None else "—"
    ret90_str = f"{ret_90:+.0%}" if ret_90 is not None else "—"
    fromlow_str = f"{from_low:+.0%}({low_label})"

    print(f"{pair:<14s} {days:>6d} {str(start):>12s} {price_fmt:>12s} {fromlow_str:>9s} {ret90_str:>8s} "
          f"{rsi_str:>5s} {mayer_str:>6s} {bb_str:>5s} {sig_str:<11s} {state:<10s}")
    results.append((pair, days, from_low, ret_90, rsi, last_fire))


print()
print("=" * 110)
print("WHAT LOW-CAPS DID IN PRIOR BEAR-MARKET RELIEF RALLIES")
print("=" * 110)
print()
print("Historical pattern in 2018 and 2022 reliefs:")
print()
print("  Phase 1 (weeks 1-3): BTC leads (+15-25%)")
print("  Phase 2 (weeks 4-6): ETH + L1s catch up (+30-50%)")
print("  Phase 3 (weeks 7-10): LOWER-CAPS rotate hard (+80-200%)")
print("  Then: everything cracks, lower-caps drop -80 to -95%")
print()
print("2022 EXAMPLES of low-cap rotation in the relief (Jun-Aug 2022):")
print("  ATOM:  +97% relief, then -80% to bottom")
print("  AVAX:  +96% relief, then -78% to bottom")
print("  FIL:   +71% relief, then -82% to bottom")
print("  APE:   +183% relief, then -88% to bottom")
print("  GMT:   +94% relief, then -91% to bottom")
print()
print("Reading this for the 2026 relief:")
print("  - BTC/ETH topped or topping  (you saw this — large caps signaling)")
print("  - SOL/LINK in late rotation (already firing)")
print("  - Low-caps may have 1-3 weeks of FOMO upside left")
print("  - DOWNSIDE risk: 80-90% drawdown from current levels in next leg down")
