"""Backtest rule-based exit signals against historical bear-market relief rallies.

Tests candidate indicators against the 2018 and 2022 bear reliefs (n=5 samples)
to find which signal would have captured the top in REAL TIME.

For each signal, measures:
  - When it fired relative to the actual peak (negative = early, positive = late)
  - Capture ratio (% of full relief retained at exit)
  - Avoided drawdown (% of next-leg-down avoided)
  - Whether it would have caught the current 2026 relief
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import numpy as np
from core import data


# Historical bear-market relief rallies (verified from OHLCV data)
# (label, pair, rally_start, rally_peak, post_rally_low_date)
RELIEFS = [
    ("2018_BTC", "BTC/USDT", "2018-04-06", "2018-05-05", "2018-06-29"),
    ("2018_ETH", "ETH/USDT", "2018-04-08", "2018-05-05", "2018-06-29"),
    ("2022_BTC", "BTC/USDT", "2022-06-18", "2022-08-15", "2022-11-21"),
    ("2022_ETH", "ETH/USDT", "2022-06-18", "2022-08-13", "2022-11-09"),
    ("2022_SOL", "SOL/USDT", "2022-06-30", "2022-09-09", "2022-12-29"),
]

# Current 2026 relief context (open-ended — peak unknown)
CURRENT_RELIEFS = [
    ("2026_BTC", "BTC/USDT", "2026-02-05"),
    ("2026_ETH", "ETH/USDT", "2026-02-05"),
    ("2026_SOL", "SOL/USDT", "2026-02-05"),
    ("2026_LINK", "LINK/USDT", "2026-02-05"),
    ("2026_DOGE", "DOGE/USDT", "2026-02-05"),
    ("2026_ADA", "ADA/USDT", "2026-02-05"),
    ("2026_AVAX", "AVAX/USDT", "2026-02-05"),
]


def calc_indicators(df):
    """Compute all candidate indicators on daily OHLCV."""
    df = df.copy()

    # Moving averages
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df["sma_200"] = df["close"].rolling(200).mean()
    df["ema_21"] = df["close"].ewm(span=21).mean()

    # RSI(14) daily
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # Stochastic (14, 3)
    lo14 = df["low"].rolling(14).min()
    hi14 = df["high"].rolling(14).max()
    df["stoch_k"] = (100 * (df["close"] - lo14) / (hi14 - lo14)).rolling(3).mean()

    # CCI(20)
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma_tp = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df["cci"] = (tp - sma_tp) / (0.015 * mad)

    # MACD
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_sig"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_sig"]

    # Bollinger Bands
    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # Mayer Multiple
    df["mayer"] = df["close"] / df["sma_200"]

    # ATR(14)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # Donchian channel
    df["donch_high_20"] = df["high"].rolling(20).max()
    df["donch_low_10"] = df["low"].rolling(10).min()
    df["donch_low_20"] = df["low"].rolling(20).min()

    # Weekly RSI
    weekly = df.resample("W").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    delta_w = weekly["close"].diff()
    gain_w = delta_w.where(delta_w > 0, 0).rolling(14).mean()
    loss_w = (-delta_w.where(delta_w < 0, 0)).rolling(14).mean()
    rs_w = gain_w / loss_w.replace(0, np.nan)
    weekly["rsi_w"] = 100 - (100 / (1 + rs_w))
    df["rsi_w"] = weekly["rsi_w"].reindex(df.index, method="ffill")

    # Chandelier exit (2 ATR below 20-day high)
    df["chand_long"] = df["donch_high_20"] - 2 * df["atr"]

    # 3 ATR from high
    df["chand3_long"] = df["donch_high_20"] - 3 * df["atr"]

    # % from 20-day high
    df["pct_from_20h"] = df["close"] / df["donch_high_20"] - 1

    return df


SIGNAL_DEFS = [
    # Reversal signals (most natural exit triggers)
    ("close<SMA20",
     lambda df: (df["close"].shift(1) > df["sma_20"].shift(1)) & (df["close"] < df["sma_20"])),
    ("close<EMA21",
     lambda df: (df["close"].shift(1) > df["ema_21"].shift(1)) & (df["close"] < df["ema_21"])),
    ("Donchian-10 break",
     lambda df: df["close"] < df["donch_low_10"].shift(1)),
    ("Chandelier-2ATR",
     lambda df: df["close"] < df["chand_long"]),
    ("Chandelier-3ATR",
     lambda df: df["close"] < df["chand3_long"]),
    # Overbought triggers
    ("RSI_d>75",
     lambda df: df["rsi"] > 75),
    ("RSI_d>80",
     lambda df: df["rsi"] > 80),
    ("RSI_w>65",
     lambda df: df["rsi_w"] > 65),
    ("RSI_w>70",
     lambda df: df["rsi_w"] > 70),
    ("BB_pct>1.0",
     lambda df: df["bb_pct"] > 1.0),
    # Reversal of overbought
    ("RSI_d cross<70",
     lambda df: (df["rsi"].shift(1) > 70) & (df["rsi"] < 70)),
    ("Stoch cross<80",
     lambda df: (df["stoch_k"].shift(1) > 80) & (df["stoch_k"] < 80)),
    ("MACD bear cross",
     lambda df: (df["macd_hist"].shift(1) > 0) & (df["macd_hist"] < 0)),
    # Position/cycle
    ("Mayer>1.3",
     lambda df: df["mayer"] > 1.3),
    ("Mayer>1.5",
     lambda df: df["mayer"] > 1.5),
    # Combos
    ("RSI_d>70 & close<SMA20",
     lambda df: (df["rsi"] > 70) & (df["close"] < df["sma_20"])),
    ("BB_pct>1 & RSI_d>70",
     lambda df: (df["bb_pct"] > 1.0) & (df["rsi"] > 70)),
    ("RSI_d>70 then MACD bear",
     lambda df: ((df["rsi"].rolling(7).max() > 70) & (df["macd_hist"].shift(1) > 0) & (df["macd_hist"] < 0))),
    ("RSI_w>60 & close<SMA20",
     lambda df: (df["rsi_w"] > 60) & (df["close"].shift(1) > df["sma_20"].shift(1)) & (df["close"] < df["sma_20"])),
    ("Mayer>1.2 & close<SMA20",
     lambda df: (df["mayer"] > 1.2) & (df["close"].shift(1) > df["sma_20"].shift(1)) & (df["close"] < df["sma_20"])),
    # Refined production-grade combos (require multiple confirmations)
    ("MACD bear+close<EMA21 w/in 5d",
     lambda df: (((df["macd_hist"].shift(1) > 0) & (df["macd_hist"] < 0)).rolling(5).max().astype(bool) &
                ((df["close"].shift(1) > df["ema_21"].shift(1)) & (df["close"] < df["ema_21"])))),
    ("MACD bear & RSI was >70 w/in 10d",
     lambda df: ((df["macd_hist"].shift(1) > 0) & (df["macd_hist"] < 0) &
                 (df["rsi"].rolling(10).max() > 70))),
    ("RSI was >70 then close<EMA21",
     lambda df: ((df["rsi"].rolling(10).max() > 70) &
                 (df["close"].shift(1) > df["ema_21"].shift(1)) & (df["close"] < df["ema_21"]))),
    ("Trail: close < 10d-high * (1-0.08)",
     lambda df: df["close"] < df["high"].rolling(10).max() * 0.92),
]


def evaluate_signal(df, rstart, rpeak, plow_date, condition):
    """For one relief + one signal, find first fire and measure quality."""
    start_dt = pd.Timestamp(rstart)
    peak_dt = pd.Timestamp(rpeak)
    plow_dt = pd.Timestamp(plow_date)

    low_price = df.loc[start_dt:peak_dt, "low"].min()
    peak_price = df.loc[start_dt:peak_dt, "high"].max()
    post_low = df.loc[peak_dt:plow_dt, "low"].min()
    full_relief = peak_price / low_price - 1
    if full_relief <= 0:
        return None

    # Find first fire date within the rally + immediate post-peak (rally + 14 days)
    window_end = plow_dt
    window_cond = condition.loc[start_dt:window_end].fillna(False)
    fires = window_cond[window_cond]
    if fires.empty:
        return None
    fire_dt = fires.index[0]
    fire_price = float(df.loc[fire_dt, "close"])

    days_from_peak = (fire_dt - peak_dt).days
    captured_pct = fire_price / low_price - 1
    capture_ratio = captured_pct / full_relief
    avoided_loss = fire_price / post_low - 1

    return {
        "fire_dt": fire_dt,
        "fire_price": fire_price,
        "days_from_peak": days_from_peak,
        "captured_pct": captured_pct,
        "capture_ratio": capture_ratio,
        "avoided_loss": avoided_loss,
        "low_price": low_price,
        "peak_price": peak_price,
        "post_low": post_low,
        "full_relief": full_relief,
    }


def main():
    print("=" * 105)
    print("ALT EXIT SIGNAL BACKTEST — historical bear-market reliefs (n=5)")
    print("=" * 105)

    all_rows = []

    for label, pair, rstart, rpeak, plow in RELIEFS:
        try:
            df = data.ohlcv_extended(pair, days_back=4500)
        except Exception as e:
            print(f"   skip {pair}: {e}")
            continue
        if df.empty:
            print(f"   skip {label}: no data")
            continue
        # Normalize timezone — strip tz if present so all comparisons are tz-naive
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        if pd.Timestamp(rstart) < df.index[0]:
            print(f"   skip {label}: data starts {df.index[0].date()}, need {rstart}")
            continue
        df = calc_indicators(df)

        start_dt = pd.Timestamp(rstart)
        peak_dt = pd.Timestamp(rpeak)
        plow_dt = pd.Timestamp(plow)
        try:
            actual_low = float(df.loc[start_dt:peak_dt, "low"].min())
            actual_peak = float(df.loc[start_dt:peak_dt, "high"].max())
            actual_post_low = float(df.loc[peak_dt:plow_dt, "low"].min())
        except Exception:
            continue

        print()
        print(f"--- {label} ({pair}) ---")
        print(f"   Window: {rstart} -> {rpeak} -> {plow}")
        print(f"   Low ${actual_low:,.2f}  Peak ${actual_peak:,.2f}  PostLow ${actual_post_low:,.2f}")
        print(f"   Relief +{actual_peak/actual_low - 1:.0%}   Drop from peak {actual_post_low/actual_peak - 1:.0%}")

        for sig_name, cond_fn in SIGNAL_DEFS:
            try:
                cond = cond_fn(df)
                res = evaluate_signal(df, rstart, rpeak, plow, cond)
                if res is None:
                    continue
                res["relief"] = label
                res["signal"] = sig_name
                all_rows.append(res)
            except Exception as e:
                print(f"   sig '{sig_name}' err: {e}")

    if not all_rows:
        print("No data.")
        return

    res_df = pd.DataFrame(all_rows)

    print()
    print("=" * 105)
    print("AGGREGATE — signal quality across all 5 reliefs")
    print("=" * 105)
    print()
    summary = res_df.groupby("signal").agg(
        n=("fire_dt", "count"),
        avg_days=("days_from_peak", "mean"),
        avg_capture=("capture_ratio", "mean"),
        avg_avoided=("avoided_loss", "mean"),
        worst_days=("days_from_peak", "max"),
        best_capture=("capture_ratio", "max"),
        worst_capture=("capture_ratio", "min"),
    )

    # Quality score: capture ratio * avoided downside
    summary["score"] = summary["avg_capture"] * (1 + summary["avg_avoided"])
    summary = summary.sort_values("score", ascending=False)

    print(f"{'Signal':<28s} {'N':>3s} {'AvgDays':>9s} {'Capture':>9s} {'Avoided':>9s} {'WorstCap':>10s} {'Score':>8s}")
    print("-" * 105)
    for sig, r in summary.iterrows():
        print(f"{sig:<28s} {int(r['n']):>3d} {r['avg_days']:>+8.1f}d {r['avg_capture']:>+8.1%} {r['avoided_loss_placeholder'] if False else r['avg_avoided']:>+8.1%} {r['worst_capture']:>+9.1%} {r['score']:>+7.2f}")

    print()
    print("Notes:")
    print("  - AvgDays: fire date vs peak. 0 = perfect, -7 = a week early, +7 = a week late")
    print("  - Capture: % of relief retained at exit (100% = sold AT peak; 50% = sold midway)")
    print("  - Avoided: how much further the price fell after the fire date (negative = none)")
    print("  - Score: capture x (1+avoided). Higher = better all-round")

    # Per-relief detail for top 6 signals
    top_sigs = summary.head(6).index.tolist()
    print()
    print("=" * 105)
    print(f"TOP SIGNAL DETAIL — per-relief")
    print("=" * 105)
    for sig in top_sigs:
        rows = res_df[res_df["signal"] == sig]
        print()
        print(f"  [{sig}]")
        for _, r in rows.iterrows():
            print(f"    {r['relief']:<10s} fired {str(r['fire_dt'].date()):<12s} @${r['fire_price']:>10,.2f}  "
                  f"days_from_peak {r['days_from_peak']:>+4d}  "
                  f"capture {r['capture_ratio']:>+6.0%}  avoided {r['avoided_loss']:>+6.0%}")

    # Apply best signal to current 2026 reliefs
    print()
    print("=" * 105)
    print("CURRENT 2026 RELIEF — applying top signals to live data")
    print("=" * 105)

    best_signal_name = summary.head(1).index[0]
    best_signal_fn = next(fn for n, fn in SIGNAL_DEFS if n == best_signal_name)

    print(f"\nApplying best signal '{best_signal_name}' to alts (rally start 2026-02-05):")
    print()
    print(f"{'Pair':<12s} {'Current':>10s} {'Low->Now':>10s} {'SignalStatus':>14s} {'FireDate':<14s}")
    print("-" * 80)

    for label, pair, rstart in CURRENT_RELIEFS:
        try:
            df = data.ohlcv_extended(pair, days_back=600)
        except Exception:
            continue
        if df.empty:
            continue
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = calc_indicators(df)
        start_dt = pd.Timestamp(rstart)
        if df[df.index >= start_dt].empty:
            continue
        try:
            low_p = float(df.loc[start_dt:, "low"].min())
            cur_p = float(df["close"].iloc[-1])
        except Exception:
            continue

        cond = best_signal_fn(df).fillna(False)
        recent_cond = cond.loc[start_dt:]
        fired_dates = recent_cond[recent_cond]
        if len(fired_dates) > 0:
            fire_date_str = str(fired_dates.index[-1].date())
            status = "FIRED"
        else:
            fire_date_str = "-"
            status = "ARMED"

        print(f"{pair:<12s} ${cur_p:>9,.4f} {cur_p/low_p - 1:>+9.1%} {status:>14s} {fire_date_str:<14s}")

    # Also show top 3 signals across all pairs
    print()
    print("Top 3 signals status on each current 2026 relief (most recent fire date):")
    top3 = summary.head(3).index.tolist()
    print()
    header = f"{'Pair':<12s} " + " ".join(f"{s[:18]:>20s}" for s in top3)
    print(header)
    print("-" * len(header))
    for label, pair, rstart in CURRENT_RELIEFS:
        try:
            df = data.ohlcv_extended(pair, days_back=600)
        except Exception:
            continue
        if df.empty:
            continue
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = calc_indicators(df)
        start_dt = pd.Timestamp(rstart)
        if df[df.index >= start_dt].empty:
            continue
        row = [f"{pair:<12s}"]
        for sig in top3:
            cond_fn = next(fn for n, fn in SIGNAL_DEFS if n == sig)
            cond = cond_fn(df).fillna(False)
            recent_cond = cond.loc[start_dt:]
            fired = recent_cond[recent_cond]
            if len(fired) > 0:
                # Most recent fire in last 14 days?
                last_fire = fired.index[-1]
                days_ago = (df.index[-1] - last_fire).days
                if days_ago < 14:
                    row.append(f"{'FIRED ' + str(last_fire.date()):>20s}")
                else:
                    row.append(f"{'past (' + str(days_ago) + 'd ago)':>20s}")
            else:
                row.append(f"{'armed':>20s}")
        print(" ".join(row))


if __name__ == "__main__":
    main()
