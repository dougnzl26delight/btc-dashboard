"""8-year BTC cycle analysis — 25+ indicators tested against known tops/bottoms.

Identifies which single indicators AND combinations gave reliable signals
at each known cycle extreme over the past 8 years, then applies the best
combination to today's data.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import data


HALVINGS = [
    date(2016, 7, 9),
    date(2020, 5, 11),
    date(2024, 4, 19),
]


# Known BTC cycle tops and bottoms (from 2018 onwards) — CORRECTED 2026-05-11
# Cycle 4 true top is Oct 2025 (not Jan 2025). Recent low Feb 2026 may or may not
# be THE cycle 4 bottom (history suggests bottom around month 28-38 post-halving,
# which would be Aug 2026 - Jun 2027).
KNOWN_EXTREMES = [
    # (date_str, type, label, approx_price)
    ("2018-12-15", "bottom", "2018 cycle bottom (post-2017 bear)", 3236),
    ("2019-06-26", "top",    "2019 mini-bull top",                 13800),
    ("2020-03-13", "bottom", "COVID crash bottom",                 4950),
    ("2021-04-14", "top",    "2021 first peak",                    64800),
    ("2021-07-20", "bottom", "Mid-2021 capitulation",              29278),
    ("2021-11-09", "top",    "2021 TRUE cycle top",                69000),
    ("2022-11-21", "bottom", "FTX collapse bottom (cycle 2 low)",  15500),
    ("2024-03-13", "top",    "Post-ETF interim peak",              73750),
    ("2024-08-05", "bottom", "Aug 2024 correction low",            49300),
    ("2025-10-06", "top",    "Cycle 4 TRUE top",                  124659),  # CORRECTED
    ("2026-02-05", "bottom", "Feb 2026 low (interim?)",            62910),  # NEW
]


# ============================================================================
# Indicator computations
# ============================================================================

def rsi(s, period=14):
    delta = s.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def add_all_indicators(df):
    df = df.copy()
    c = df["close"]
    h, l = df["high"], df["low"]
    v = df["volume"]

    # ----- Trend / Mean Indicators -----
    df["sma50"] = c.rolling(50).mean()
    df["sma200"] = c.rolling(200).mean()
    df["sma100"] = c.rolling(100).mean()
    df["sma111"] = c.rolling(111).mean()
    df["sma350"] = c.rolling(350).mean()
    df["sma_weekly_200"] = c.rolling(200 * 7).mean()  # ~200-week
    df["sma_1y"] = c.rolling(365).mean()

    # Mayer Multiple
    df["mayer"] = c / df["sma200"]
    df["mayer_50w"] = c / df["sma_weekly_200"]

    # Pi Cycle: 111 SMA vs 350 SMA × 2
    df["pi_top_flag"] = (df["sma111"] >= df["sma350"] * 2).astype(int)
    df["pi_top_distance"] = df["sma111"] / (df["sma350"] * 2) - 1

    # 200-week SMA distance (multiple)
    df["wk200_multiple"] = c / df["sma_weekly_200"]

    # Distance from prior all-time-high (rolling)
    df["ath_running"] = c.cummax()
    df["dist_from_ath"] = c / df["ath_running"] - 1

    # ----- Momentum -----
    df["rsi_14d"] = rsi(c, 14)
    df["rsi_21d"] = rsi(c, 21)
    # Weekly RSI approximation (using weekly resample)
    weekly = c.resample("W").last()
    weekly_rsi = rsi(weekly, 14)
    df["rsi_weekly"] = weekly_rsi.reindex(df.index, method="ffill")
    # Monthly RSI
    monthly = c.resample("ME").last()
    monthly_rsi = rsi(monthly, 14)
    df["rsi_monthly"] = monthly_rsi.reindex(df.index, method="ffill")

    # MACD
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9).mean()
    df["macd_hist"] = (macd - macd_signal) / c
    df["macd_norm"] = macd / c

    # ----- Volatility / Range -----
    df["bb_mid"] = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std
    df["bb_pct"] = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)

    # Realized vol
    ret = c.pct_change()
    df["rvol_20"] = ret.rolling(20).std() * np.sqrt(365)
    df["rvol_60"] = ret.rolling(60).std() * np.sqrt(365)
    df["rvol_90"] = ret.rolling(90).std() * np.sqrt(365)

    # ATR / price
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr14"] / c

    # ----- Oscillators -----
    # Stochastic %K
    low_n = l.rolling(14).min()
    high_n = h.rolling(14).max()
    df["stoch_k"] = 100 * (c - low_n) / (high_n - low_n).replace(0, np.nan)

    # Williams %R
    df["williams_r"] = -100 * (high_n - c) / (high_n - low_n).replace(0, np.nan)

    # CCI
    typ = (h + l + c) / 3
    sma_typ = typ.rolling(20).mean()
    mad = typ.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=False)
    df["cci"] = (typ - sma_typ) / (0.015 * mad.replace(0, np.nan))

    # ----- Returns -----
    df["ret_30d"] = c.pct_change(30)
    df["ret_60d"] = c.pct_change(60)
    df["ret_90d"] = c.pct_change(90)
    df["ret_180d"] = c.pct_change(180)
    df["ret_365d"] = c.pct_change(365)

    # ----- Volume -----
    df["vol_z_30d"] = (v - v.rolling(30).mean()) / v.rolling(30).std().replace(0, np.nan)
    # OBV slope
    obv = (np.sign(c.diff()) * v).cumsum()
    df["obv_slope_30d"] = obv.pct_change(30)

    # ----- MVRV/NUPL approximations (using price-only data) -----
    # MVRV proxy: current price / 365d avg price
    df["mvrv_proxy"] = c / df["sma_1y"]
    # NUPL proxy: cumulative unrealized profit ratio (rough)
    # = (current - rolling mean cost basis) / current
    df["nupl_proxy"] = (c - df["sma_1y"]) / c

    # ----- Halving cycle -----
    def _months_since_halving(d):
        date_only = d.date() if hasattr(d, "date") else d
        past = [h for h in HALVINGS if h <= date_only]
        return (date_only - max(past)).days / 30.4 if past else 999
    df["months_post_halving"] = df.index.map(_months_since_halving)

    # ----- ADX -----
    plus_dm = h.diff().where((h.diff() > l.diff().abs()) & (h.diff() > 0), 0)
    minus_dm = l.diff().abs().where((l.diff().abs() > h.diff()) & (l.diff() < 0), 0)
    atr = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx_14"] = dx.rolling(14).mean()

    return df


# ============================================================================
# Analysis
# ============================================================================

def analyze_extremes(df, extremes):
    """Record indicator values at each known cycle extreme."""
    df = add_all_indicators(df)
    rows = []
    for date_str, kind, label, approx_price in extremes:
        ts = pd.Timestamp(date_str, tz="UTC")
        available = df.index[df.index <= ts]
        if len(available) == 0:
            continue
        actual_ts = available[-1]
        row = df.loc[actual_ts]
        rec = {
            "date": date_str, "type": kind, "label": label,
            "actual_date": str(actual_ts.date()),
            "actual_price": float(row["close"]),
            "expected_price": approx_price,
        }
        # Record key indicator values
        for ind in ["mayer", "mayer_50w", "wk200_multiple", "pi_top_flag",
                    "pi_top_distance", "rsi_14d", "rsi_weekly", "rsi_monthly",
                    "macd_hist", "bb_pct", "rvol_20", "rvol_60", "rvol_90",
                    "stoch_k", "williams_r", "cci", "ret_30d", "ret_60d",
                    "ret_90d", "ret_180d", "ret_365d", "vol_z_30d",
                    "obv_slope_30d", "mvrv_proxy", "nupl_proxy",
                    "months_post_halving", "adx_14", "dist_from_ath",
                    "atr_pct"]:
            if ind in df.columns:
                val = row[ind]
                rec[ind] = float(val) if pd.notna(val) else None
        rows.append(rec)
    return pd.DataFrame(rows)


def find_top_signals(extremes_df, ind_cols, threshold_pct=80):
    """For each indicator, find threshold that distinguishes tops from bottoms."""
    tops = extremes_df[extremes_df["type"] == "top"]
    bottoms = extremes_df[extremes_df["type"] == "bottom"]
    if len(tops) < 2 or len(bottoms) < 2:
        return pd.DataFrame()
    rows = []
    for ind in ind_cols:
        if ind not in extremes_df.columns:
            continue
        top_vals = tops[ind].dropna()
        bot_vals = bottoms[ind].dropna()
        if len(top_vals) < 2 or len(bot_vals) < 2:
            continue
        # Spearman-like: do tops have systematically higher values than bottoms?
        top_mean = top_vals.mean()
        bot_mean = bot_vals.mean()
        top_min = top_vals.min()
        top_max = top_vals.max()
        bot_min = bot_vals.min()
        bot_max = bot_vals.max()
        # Distinguishability: gap between top_min and bot_max
        if top_mean > bot_mean:
            direction = "high_at_top"
            gap = top_min - bot_max
            cleanly_separates = top_min > bot_max
        else:
            direction = "low_at_top"
            gap = bot_min - top_max
            cleanly_separates = bot_min > top_max
        rows.append({
            "indicator": ind, "direction": direction,
            "top_mean": top_mean, "bot_mean": bot_mean,
            "top_range": f"[{top_min:.2f}, {top_max:.2f}]",
            "bot_range": f"[{bot_min:.2f}, {bot_max:.2f}]",
            "gap": gap,
            "cleanly_separates": cleanly_separates,
        })
    return pd.DataFrame(rows).sort_values("gap", ascending=False)


# ============================================================================

if __name__ == "__main__":
    print("=" * 100)
    print("8-YEAR BTC CYCLE ANALYSIS — indicators at known tops/bottoms")
    print("=" * 100)
    print()

    df = data.ohlcv_extended("BTC/USDT", days_back=3000)
    print(f"Data: {df.index[0].date()} to {df.index[-1].date()} ({len(df)} days)")
    print()

    extremes_df = analyze_extremes(df, KNOWN_EXTREMES)

    # Print indicator table at each event
    print("INDICATOR VALUES AT EACH KNOWN CYCLE EXTREME")
    print("-" * 100)
    print(f"{'Date':<12s} {'Type':<7s} {'Price':>9s} {'Mayer':>6s} {'Wk200x':>7s} "
          f"{'PiTop':>6s} {'RSIw':>5s} {'RSIm':>5s} {'BB%':>5s} "
          f"{'Ret90':>7s} {'Ret180':>7s} {'Ret365':>7s} {'MVRV':>5s} {'MosHv':>6s}")
    for _, r in extremes_df.iterrows():
        print(f"{r['date']:<12s} {r['type']:<7s} ${r['actual_price']:>7,.0f}  "
              f"{r['mayer']:>5.2f}  {r['wk200_multiple']:>6.2f}  "
              f"{r['pi_top_flag']:>5.0f}  {r['rsi_weekly']:>4.0f}  "
              f"{r['rsi_monthly']:>4.0f}  {r['bb_pct']:>4.2f}  "
              f"{r['ret_90d']:>+6.1%} {r['ret_180d']:>+6.1%} "
              f"{r['ret_365d']:>+6.1%}  {r['mvrv_proxy']:>4.2f}  "
              f"{r['months_post_halving']:>5.1f}")
    print()

    # Find indicators with clean separation
    print("=" * 100)
    print("WHICH INDICATORS CLEANLY DISTINGUISH TOPS FROM BOTTOMS?")
    print("=" * 100)
    ind_cols = ["mayer", "mayer_50w", "wk200_multiple", "rsi_14d", "rsi_weekly",
                "rsi_monthly", "macd_hist", "bb_pct", "rvol_60", "stoch_k",
                "williams_r", "cci", "ret_90d", "ret_180d", "ret_365d",
                "mvrv_proxy", "nupl_proxy", "atr_pct", "dist_from_ath"]
    signal_df = find_top_signals(extremes_df, ind_cols)
    print(f"{'Indicator':<22s} {'Direction':<14s} {'Top range':<22s} "
          f"{'Bot range':<22s} {'Gap':>7s} {'Clean':>6s}")
    for _, r in signal_df.iterrows():
        marker = "YES" if r["cleanly_separates"] else "no"
        print(f"{r['indicator']:<22s} {r['direction']:<14s} {r['top_range']:<22s} "
              f"{r['bot_range']:<22s} {r['gap']:>+6.2f}   {marker:>4s}")
    print()

    # ----- Find SAFE thresholds using only cleanly separating indicators -----
    print("=" * 100)
    print("PROPOSED THRESHOLDS (using cleanly-separating indicators)")
    print("=" * 100)
    clean = signal_df[signal_df["cleanly_separates"]]
    if not clean.empty:
        for _, r in clean.iterrows():
            ind = r["indicator"]
            tops = extremes_df[extremes_df["type"] == "top"][ind].dropna()
            bots = extremes_df[extremes_df["type"] == "bottom"][ind].dropna()
            if r["direction"] == "high_at_top":
                top_threshold = float(min(tops.min(), bots.max() + (tops.min() - bots.max()) / 2))
                bot_threshold = float(max(bots.max(), bots.max() + 0.01))
                # Use midpoint for clean threshold
                midpoint = (tops.min() + bots.max()) / 2
                print(f"  {ind:<22s}: TOP if > {midpoint:.2f}, BOTTOM if < {midpoint:.2f}")
            else:
                midpoint = (bots.min() + tops.max()) / 2
                print(f"  {ind:<22s}: TOP if < {midpoint:.2f}, BOTTOM if > {midpoint:.2f}")
    else:
        print("  No indicators give 100% clean separation across all 5+ events.")
        print("  This is expected — markets evolve. Look at majority-rule instead.")
    print()

    # ----- Current values + composite vote -----
    print("=" * 100)
    print("CURRENT INDICATOR VALUES")
    print("=" * 100)
    df_now = add_all_indicators(df).iloc[-1]
    print(f"Date: {df.index[-1].date()}, Price: ${df_now['close']:,.0f}")
    print()

    # For each indicator, classify current value as "top-like", "bottom-like", or "neutral"
    print(f"{'Indicator':<22s} {'Current':>10s} {'Top mean':>10s} {'Bot mean':>10s} "
          f"{'Classification':<18s}")
    composite_top_votes = 0
    composite_bottom_votes = 0
    composite_neutral = 0
    composite_total = 0
    for _, r in signal_df.iterrows():
        ind = r["indicator"]
        if ind not in df_now.index:
            continue
        cur_val = float(df_now[ind])
        top_mean = r["top_mean"]
        bot_mean = r["bot_mean"]
        # Where is current vs the two means?
        if r["direction"] == "high_at_top":
            # higher = more top-like
            if cur_val > top_mean * 0.95:
                cls = "near-top"
                composite_top_votes += 1
            elif cur_val < bot_mean * 1.05:
                cls = "near-bottom"
                composite_bottom_votes += 1
            else:
                cls = "neutral"
                composite_neutral += 1
        else:
            if cur_val < top_mean * 1.05:
                cls = "near-top"
                composite_top_votes += 1
            elif cur_val > bot_mean * 0.95:
                cls = "near-bottom"
                composite_bottom_votes += 1
            else:
                cls = "neutral"
                composite_neutral += 1
        composite_total += 1
        print(f"  {ind:<22s} {cur_val:>+9.2f}  {top_mean:>+9.2f}  {bot_mean:>+9.2f}   {cls:<18s}")
    print()
    print(f"COMPOSITE VOTE: {composite_top_votes} top-like / {composite_bottom_votes} bottom-like / "
          f"{composite_neutral} neutral (out of {composite_total} indicators)")
    print()

    # ----- Halving context -----
    months_post_halving_now = df_now["months_post_halving"]
    print(f"Months since Apr 2024 halving: {months_post_halving_now:.1f}")
    print(f"Historical cycle top window (14-18mo post-halving): ", end="")
    if 14 <= months_post_halving_now <= 18:
        print("INSIDE — typical top window")
    elif months_post_halving_now < 14:
        print(f"NOT YET — top window starts in {14 - months_post_halving_now:.1f} months")
    elif months_post_halving_now > 18:
        print(f"PASSED — top window ended {months_post_halving_now - 18:.1f} months ago")
    print(f"Historical cycle bottom window (28-38mo post-halving): ", end="")
    if 28 <= months_post_halving_now <= 38:
        print("INSIDE — typical bottom window")
    elif months_post_halving_now < 28:
        print(f"NOT YET — bottom window in {28 - months_post_halving_now:.1f} months")
    elif months_post_halving_now > 38:
        print(f"PASSED — bottom window ended {months_post_halving_now - 38:.1f} months ago")
    print()

    # ----- Final synthesis -----
    print("=" * 100)
    print("SYNTHESIS: where are we in the cycle?")
    print("=" * 100)
    top_ratio = composite_top_votes / composite_total if composite_total > 0 else 0
    bot_ratio = composite_bottom_votes / composite_total if composite_total > 0 else 0
    print(f"  Top-like indicators:     {top_ratio:.0%}")
    print(f"  Bottom-like indicators:  {bot_ratio:.0%}")
    print(f"  Neutral indicators:      {composite_neutral/composite_total:.0%}")
    print()
    if top_ratio >= 0.60:
        regime_call = "TOP-LIKE (consider de-risking)"
    elif bot_ratio >= 0.60:
        regime_call = "BOTTOM-LIKE (consider accumulating)"
    elif top_ratio > bot_ratio + 0.10:
        regime_call = "Tilting toward overextended"
    elif bot_ratio > top_ratio + 0.10:
        regime_call = "Tilting toward oversold"
    else:
        regime_call = "MID-CYCLE / TRANSITION — no extreme signal"
    print(f"  COMPOSITE CALL: {regime_call}")
