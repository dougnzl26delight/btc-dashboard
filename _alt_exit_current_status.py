"""Show current indicator readings + best-signal status for each alt position."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import numpy as np
from core import data


PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "LINK/USDT",
         "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "BNB/USDT", "DOT/USDT", "ATOM/USDT"]


def calc_indicators(df):
    df = df.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_200"] = df["close"].rolling(200).mean()
    df["ema_21"] = df["close"].ewm(span=21).mean()
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd_hist"] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_upper"] = df["sma_20"] + 2 * bb_std
    df["bb_lower"] = df["sma_20"] - 2 * bb_std
    df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    df["mayer"] = df["close"] / df["sma_200"]
    return df


def best_signal_fired_when(df):
    """Best signal: MACD bear cross + close<EMA21 within 5d. Return last fire date."""
    macd_bear = (df["macd_hist"].shift(1) > 0) & (df["macd_hist"] < 0)
    ema_break = (df["close"].shift(1) > df["ema_21"].shift(1)) & (df["close"] < df["ema_21"])
    sig = macd_bear.rolling(5).max().astype(bool) & ema_break
    fires = sig[sig.fillna(False)]
    if fires.empty:
        return None
    return fires.index[-1]


print("=" * 100)
print("CURRENT ALT POSITIONS — indicator readings + best-signal status")
print("=" * 100)
print(f"Best signal (from backtest): MACD bear cross + close<EMA21 within 5 days")
print(f"Historical performance: 66.6% capture, +126% avoided drawdown, avg -0.8d from peak")
print()

print(f"{'Pair':<10s} {'Price':>10s} {'RSI':>5s} {'Mayer':>6s} {'BB%':>6s} "
      f"{'MACDh':>8s} {'EMA21':>9s} {'From low':>9s} {'LastFire':<12s} {'Status':<10s}")
print("-" * 100)

for pair in PAIRS:
    try:
        df = data.ohlcv_extended(pair, days_back=400)
    except Exception:
        continue
    if df.empty:
        continue
    df = calc_indicators(df)
    last = df.iloc[-1]
    feb_low = float(df.loc[df.index >= pd.Timestamp("2026-02-01"), "low"].min())
    from_low = last["close"] / feb_low - 1
    above_ema = "above" if last["close"] > last["ema_21"] else "BELOW"
    fire_date = best_signal_fired_when(df)
    if fire_date is None:
        last_fire = "never"
        days_since = None
    else:
        last_fire = str(fire_date.date())
        days_since = (df.index[-1] - fire_date).days
    if days_since is None:
        status = "ARMED"
    elif days_since < 7:
        status = "FIRING NOW"
    elif days_since < 30:
        status = f"{days_since}d ago"
    else:
        status = "stale"

    price_fmt = f"${last['close']:>9,.4f}" if last['close'] < 100 else f"${last['close']:>9,.2f}"
    print(f"{pair:<10s} {price_fmt:>10s} {last['rsi']:>5.0f} {last['mayer']:>6.2f} "
          f"{last['bb_pct']:>5.0%} {last['macd_hist']:>+8.4f} {above_ema:>9s} "
          f"{from_low:>+8.1%} {last_fire:<12s} {status:<10s}")

print()
print("=" * 100)
print("INTERPRETATION")
print("=" * 100)
print()
print("RSI: <30 oversold, >70 overbought. Current relief tops in 2018/2022 had RSI 75-85.")
print("Mayer: price/SMA200. <0.7 deep bear, 1.0 = at trend, >1.2 stretched, >1.5 frothy.")
print("  In 2018 ETH relief, Mayer peaked at 1.4. In 2022 BTC relief, Mayer peaked at 1.1.")
print("BB%: where in 20-day Bollinger Band. >100% = above upper band (overbought).")
print("MACDh: histogram value. POSITIVE = uptrend, flipping NEGATIVE = first sell trigger.")
print()
print("ACTION RULES (per the backtest):")
print("  STATUS = 'FIRING NOW' (last 7 days): sell 1/3 of position immediately")
print("  STATUS = '~ago' (already fired): sell tranche on next bounce of >3%")
print("  STATUS = 'ARMED' (not yet fired): hold, but tighten stop to EMA21 break")
print()
print("Backtest caveat: n=5 historical samples. 2026 ETF-era dynamics may differ.")
