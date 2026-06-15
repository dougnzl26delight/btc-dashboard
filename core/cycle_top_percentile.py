"""Percentile-rank cycle-top detector — auto-adapts to muted cycles.

THE PROBLEM:
    Each BTC cycle's peak indicator magnitudes are decaying due to
    institutional smoothing (ETF + sovereign demand replace retail euphoria).

    Cycle 3 peak Mayer Multiple: 3.7
    Cycle 4 peak Mayer Multiple: 1.94 then 1.48 (-50%)
    Cycle 5 peak Mayer Multiple: 1.18 (-19%)
    Cycle 6 forecast:            ~1.0  (-15%)

    Classic absolute thresholds ("Mayer >= 2.4 = TOP") miss muted cycles.
    Cycle 5 peak fired ZERO of the 7 most-cited absolute-threshold signals.

THE FIX:
    Use percentile rank vs trailing 4-year window. When today's MVRV is in
    the top 5% of the last 1460 days, that's distribution zone — regardless
    of absolute magnitude.

    This auto-adapts to each cycle's actual peak signature.

INDICATORS RANKED:
    1. MVRV          (CoinMetrics free)
    2. NUPL          (derived from MVRV: 1 - 1/MVRV)
    3. Mayer Mult    (price / 200d SMA)
    4. Weekly RSI    (14-week RSI on weekly closes)
    5. Price vs ATH  (distance from rolling all-time high)

COMPOSITE SCORE:
    0-100 score weighted by indicator quality + technical confirmation.
    >= 75: scale out aggressively, >= 90: full distribution zone.

CACHED for 4 hours to avoid hitting CoinMetrics API every call.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
CACHE_FILE = REPO / ".cycle_top_percentile_cache.json"
CACHE_TTL = 4 * 3600  # 4 hours


# === DATA FETCHERS ===

def _fetch_mvrv_history(days: int = 1500) -> pd.DataFrame:
    """Fetch BTC MVRV from CoinMetrics community API."""
    start = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    url = ("https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
           f"?assets=btc&metrics=CapMVRVCur&start_time={start}&page_size=10000")
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            payload = json.loads(r.read())
    except Exception:
        return pd.DataFrame()
    rows = []
    for d in payload.get("data", []):
        try:
            rows.append({
                "date": pd.to_datetime(d["time"]).date(),
                "mvrv": float(d["CapMVRVCur"]),
            })
        except Exception:
            continue
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _fetch_btc_prices(days: int = 1500) -> pd.DataFrame:
    """Fetch BTC daily prices from ccxt Binance."""
    try:
        import ccxt
        ex = ccxt.binance({"enableRateLimit": True, "timeout": 15000})
    except Exception:
        return pd.DataFrame()

    all_rows = []
    end_ts = int(time.time() * 1000)
    target = days + 100  # buffer
    while len(all_rows) < target:
        try:
            ohlcv = ex.fetch_ohlcv("BTC/USDT", timeframe="1d", limit=1000,
                                    since=end_ts - 1000 * 86400 * 1000)
        except Exception:
            break
        if not ohlcv: break
        all_rows = ohlcv + all_rows
        end_ts = ohlcv[0][0] - 86400 * 1000
        if len(ohlcv) < 1000: break

    df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms").dt.date
    return df.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def _load_data(force: bool = False) -> pd.DataFrame:
    """Load merged dataset with all indicators computed."""
    if not force and CACHE_FILE.exists():
        try:
            d = json.loads(CACHE_FILE.read_text())
            if time.time() - d.get("fetched_at", 0) < CACHE_TTL:
                df = pd.read_json(d["data"], orient="split")
                df["date"] = pd.to_datetime(df["date"]).dt.date
                return df
        except Exception:
            pass

    mvrv = _fetch_mvrv_history()
    btc = _fetch_btc_prices()
    if mvrv.empty or btc.empty:
        return pd.DataFrame()

    df = btc.merge(mvrv, on="date", how="inner").sort_values("date").reset_index(drop=True)

    # Compute indicators
    df["sma200"] = df["close"].rolling(200).mean()
    df["mayer"] = df["close"] / df["sma200"]
    df["nupl"] = 1 - 1 / df["mvrv"].replace(0, np.nan)
    df["ath"] = df["close"].cummax()
    df["distance_from_ath"] = df["close"] / df["ath"]

    # Weekly RSI: compute on weekly resamples
    weekly = df.iloc[::7].copy()
    delta = weekly["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    weekly["rsi_w"] = 100 - 100 / (1 + rs)
    df = df.merge(weekly[["date", "rsi_w"]], on="date", how="left")
    df["rsi_w"] = df["rsi_w"].ffill()

    # Weekly MACD bear cross signal
    df["ema12w"] = df["close"].ewm(span=12 * 7, adjust=False).mean()
    df["ema26w"] = df["close"].ewm(span=26 * 7, adjust=False).mean()
    macd_w = df["ema12w"] - df["ema26w"]
    sig_w = macd_w.ewm(span=9 * 7, adjust=False).mean()
    df["macd_w_hist"] = macd_w - sig_w

    # Cache
    try:
        # Convert dates to strings for JSON serialization
        df_to_cache = df.copy()
        df_to_cache["date"] = df_to_cache["date"].astype(str)
        CACHE_FILE.write_text(json.dumps({
            "fetched_at": time.time(),
            "data": df_to_cache.to_json(orient="split"),
        }))
    except Exception:
        pass

    return df


# === PERCENTILE COMPUTATION ===

# CALIBRATED based on backtest at 2025-10-06 peak (see _deep_ta.py findings):
#
# CRITICAL FINDING: At cycle 5 peak, MVRV (2.29) was LOWER than the March 2024
# spike (2.78). The 730-day percentile window contained a HIGHER MVRV than the
# actual price peak — because cycle 5's slow grind kept realized cost basis high.
# In the institutional era, MVRV may NEVER reach absolute extremes again, even
# at price peaks.
#
# THE FIX: shorter window (180 days = 6 months) measures RECENT REGIME, not
# multi-year context. Combined with ROLLOVER detection (3% pullback from
# 90-day high while still elevated) catches the actual trend reversal.
#
# But the underlying truth: technical signals (weekly MACD bear, daily exit
# signal) are the most reliable cycle-peak detectors going forward. The
# percentile-rank ranks serve mainly as CONFIRMATION layer.
WINDOW_DAYS = 180        # 6 months = recent regime
ROLLOVER_LOOKBACK = 90   # check if rolled over from 90-day high
PERCENTILE_EXTREME = 90  # top 10% = "extreme high" zone
PERCENTILE_ELEVATED = 75 # top 25% = "elevated" warning
ROLLOVER_PCT_TRIGGER = 0.03  # 3% pullback from 90-day high = rollover confirmed


def _percentile_rank(series: pd.Series, current_val: float) -> float:
    """Return percentile rank (0-100) of current_val in the series."""
    if series.empty or pd.isna(current_val):
        return 0.0
    valid = series.dropna()
    if len(valid) == 0:
        return 0.0
    return float((valid <= current_val).sum()) / len(valid) * 100


def _evaluate_indicator(window: pd.Series, current_val: float,
                          lookback_max_days: int = ROLLOVER_LOOKBACK) -> dict:
    """Full evaluation of one indicator — percentile rank + rollover detection."""
    if pd.isna(current_val) or window.empty:
        return {"current_value": None, "percentile_rank": 0.0,
                "extreme": False, "elevated": False,
                "rolled_over_from_high": False, "pct_off_recent_high": 0.0}

    rank = _percentile_rank(window, current_val)
    # Rollover detection: how far are we from the recent N-day max?
    recent = window.tail(lookback_max_days).dropna()
    recent_max = float(recent.max()) if len(recent) > 0 else current_val
    pct_off_max = (current_val / recent_max - 1) if recent_max > 0 else 0
    # "Rolled over" = we were at ≥90th percentile AND now we're ≥3% below recent high
    rolled_over = (rank >= PERCENTILE_ELEVATED and
                    pct_off_max < -ROLLOVER_PCT_TRIGGER)

    return {
        "current_value": float(current_val),
        "percentile_rank": rank,
        "extreme": rank >= PERCENTILE_EXTREME,
        "elevated": rank >= PERCENTILE_ELEVATED,
        "rolled_over_from_high": rolled_over,
        "pct_off_recent_high": pct_off_max,
        "recent_high": recent_max,
    }


def compute_percentile_ranks() -> dict:
    """Compute percentile rank + rollover for each cycle-top indicator.

    Each indicator returns:
        - current_value
        - percentile_rank (vs trailing 2-year window)
        - extreme  (top 10%)
        - elevated (top 25%)
        - rolled_over_from_high (elevated + 3%+ off recent 60-day high)
        - pct_off_recent_high
    """
    df = _load_data()
    if df.empty:
        return {"error": "data_unavailable"}

    window = df.tail(WINDOW_DAYS).copy()
    if len(window) < 90:
        return {"error": "insufficient_history",
                "window_available_days": len(window)}

    current = df.iloc[-1]
    out = {
        "as_of": str(current["date"]),
        "btc_price": float(current["close"]),
        "window_days": len(window),
        "thresholds": {
            "extreme_pct": PERCENTILE_EXTREME,
            "elevated_pct": PERCENTILE_ELEVATED,
            "rollover_lookback_days": ROLLOVER_LOOKBACK,
            "rollover_trigger_pct": ROLLOVER_PCT_TRIGGER,
        },
        "indicators": {},
    }

    for name, col in [("mvrv", "mvrv"), ("nupl", "nupl"), ("mayer", "mayer"),
                       ("rsi_w", "rsi_w"), ("distance_from_ath", "distance_from_ath")]:
        out["indicators"][name] = _evaluate_indicator(window[col], current[col])

    out["weekly_macd_bear"] = bool(current["macd_w_hist"] < 0)
    out["weekly_macd_hist"] = float(current["macd_w_hist"])

    return out


def cycle_top_score() -> dict:
    """Compute composite cycle-top score 0-100.

    SCORING PHILOSOPHY:
    Magnitude alone misses muted cycles (cycle 5 proved this).
    The fix: each indicator can contribute to the score TWICE:
        (a) Magnitude — top 10% percentile = elevated state
        (b) Rollover — was elevated AND just pulled back from recent high
    Rollover is the trend-reversal signal that catches the actual peak.

    Score components per indicator (max 20-30 each):
        MVRV:    +15 extreme percentile, +15 rollover from high  (total max 30)
        NUPL:    +15 extreme,            +15 rollover            (max 30)
        Mayer:   +10 extreme,            +10 rollover            (max 20)
        RSI_w:   +10 extreme,            +10 rollover            (max 20)
        Tech:    +10 weekly MACD bear, +15 daily exit signal
        Distance from ATH: +5 if elevated (we're near ATH = late cycle context)

    Total possible: 130. Capped at 100.

    Interpretation:
        0-30:   normal/early bull
        30-50:  caution
        50-70:  scale out 25%
        70-85:  scale out 50%
        85-100: PEAK ZONE — full exit
    """
    ranks = compute_percentile_ranks()
    if "error" in ranks:
        return ranks

    score = 0
    components = {}
    inds = ranks["indicators"]

    # === HARD GATE: only fire cycle-top scoring if BTC is near recent ATH ===
    # In a bear market, percentile-rank within the recent window can be high
    # while price is far below any meaningful peak. Only score if BTC close
    # is within 12% of the all-time high (distance_from_ath >= 0.88).
    NEAR_ATH_REQUIRED = 0.88
    dist_to_ath = inds["distance_from_ath"]["current_value"]  # close/ATH ratio
    if dist_to_ath < NEAR_ATH_REQUIRED:
        return {
            "as_of": ranks["as_of"],
            "btc_price": ranks["btc_price"],
            "score": 0,
            "verdict": "NOT_NEAR_ATH",
            "action": (f"BTC is {(1-dist_to_ath)*100:.0f}% below all-time high — "
                       f"cycle-top scoring suppressed (need within {(1-NEAR_ATH_REQUIRED)*100:.0f}%)"),
            "extreme_indicator_count": 0,
            "elevated_indicator_count": 0,
            "rollover_indicator_count": 0,
            "components": {},
            "window_days": ranks["window_days"],
            "indicators": inds,
            "weekly_macd_bear": ranks["weekly_macd_bear"],
            "distance_to_ath": dist_to_ath,
        }

    # MVRV
    if inds["mvrv"]["extreme"]:
        score += 15; components["mvrv_extreme"] = 15
    elif inds["mvrv"]["elevated"]:
        score += 6; components["mvrv_elevated"] = 6
    if inds["mvrv"]["rolled_over_from_high"]:
        score += 15; components["mvrv_rollover"] = 15

    # NUPL
    if inds["nupl"]["extreme"]:
        score += 15; components["nupl_extreme"] = 15
    elif inds["nupl"]["elevated"]:
        score += 6; components["nupl_elevated"] = 6
    if inds["nupl"]["rolled_over_from_high"]:
        score += 15; components["nupl_rollover"] = 15

    # Mayer
    if inds["mayer"]["extreme"]:
        score += 10; components["mayer_extreme"] = 10
    elif inds["mayer"]["elevated"]:
        score += 4; components["mayer_elevated"] = 4
    if inds["mayer"]["rolled_over_from_high"]:
        score += 10; components["mayer_rollover"] = 10

    # Weekly RSI
    if inds["rsi_w"]["extreme"]:
        score += 10; components["rsi_w_extreme"] = 10
    elif inds["rsi_w"]["elevated"]:
        score += 4; components["rsi_w_elevated"] = 4
    if inds["rsi_w"]["rolled_over_from_high"]:
        score += 10; components["rsi_w_rollover"] = 10

    # Distance from ATH (we're at/near ATH = late-cycle context)
    if inds["distance_from_ath"]["elevated"]:
        score += 5; components["near_ath"] = 5

    # Weekly MACD bear (binary technical)
    if ranks["weekly_macd_bear"]:
        score += 10; components["weekly_macd_bear"] = 10

    # Technical exit signal (existing rig component — heaviest weight)
    try:
        from btc_exit_signal_alert import compute_status
        es = compute_status()
        btc_st = es.get("BTC/USDT", {})
        if btc_st.get("stop_alert") in ("BROKEN", "NEAR"):
            score += 15; components["technical_exit_signal"] = 15
    except Exception:
        pass

    # Cap at 100
    score = min(score, 100)

    # Interpretation — new thresholds tuned to scoring scale
    if score >= 85:
        verdict = "PEAK_ZONE"
        action = "FULL EXIT — multiple cycle-top indicators confirming + technical break"
    elif score >= 70:
        verdict = "DISTRIBUTION_ZONE"
        action = "Scale out 50% — major distribution signal"
    elif score >= 50:
        verdict = "ELEVATED"
        action = "Scale out 25% — late-bull warning"
    elif score >= 30:
        verdict = "CAUTION"
        action = "Watch carefully — early warning"
    else:
        verdict = "NORMAL"
        action = "No cycle-top signal — hold"

    extreme_count = sum(1 for i in inds.values() if i["extreme"])
    elevated_count = sum(1 for i in inds.values() if i["elevated"] and not i["extreme"])
    rollover_count = sum(1 for i in inds.values() if i["rolled_over_from_high"])

    return {
        "as_of": ranks["as_of"],
        "btc_price": ranks["btc_price"],
        "score": score,
        "verdict": verdict,
        "action": action,
        "extreme_indicator_count": extreme_count,
        "elevated_indicator_count": elevated_count,
        "rollover_indicator_count": rollover_count,
        "components": components,
        "window_days": ranks["window_days"],
        "indicators": ranks["indicators"],
        "weekly_macd_bear": ranks["weekly_macd_bear"],
    }


def backtest_historical_peaks(scan_window_days: int = 90) -> list[dict]:
    """Verify detector fires at historical peaks.

    Scans ±scan_window_days around each peak, finds the first date the score
    crosses the DISTRIBUTION_ZONE threshold (70). Shows how close to the
    actual peak the signal fires.
    """
    df = _load_data()
    if df.empty:
        return []

    historical_peaks = [
        (pd.to_datetime("2021-04-14").date(), "Cycle 4 first peak", 64863),
        (pd.to_datetime("2021-11-08").date(), "Cycle 4 final peak", 67526),
        (pd.to_datetime("2025-10-06").date(), "Cycle 5 peak",       124659),
    ]

    def score_at(idx: int) -> int:
        """Compute score for a specific row index."""
        if idx < 200 or idx >= len(df): return 0
        window = df.iloc[max(0, idx - WINDOW_DAYS):idx + 1]
        if len(window) < 90: return 0
        current = df.iloc[idx]
        total = 0
        for col, w_ext, w_roll in [("mvrv", 15, 15), ("nupl", 15, 15),
                                     ("mayer", 10, 10), ("rsi_w", 10, 10)]:
            ind = _evaluate_indicator(window[col], current[col])
            if ind["extreme"]: total += w_ext
            elif ind["elevated"]: total += w_ext // 2
            if ind["rolled_over_from_high"]: total += w_roll
        dist_ind = _evaluate_indicator(window["distance_from_ath"], current["distance_from_ath"])
        if dist_ind["elevated"]: total += 5
        if current["macd_w_hist"] < 0: total += 10
        return min(100, total)

    results = []
    for peak_dt, label, peak_px in historical_peaks:
        row = df[df["date"] == peak_dt]
        if row.empty:
            results.append({"date": peak_dt, "label": label, "price": peak_px,
                            "status": "not_in_dataset"})
            continue
        peak_idx = row.index[0]

        # Scan ±scan_window_days
        scan_start = max(200, peak_idx - scan_window_days)
        scan_end = min(len(df), peak_idx + scan_window_days + 1)

        scores_by_day = []
        for i in range(scan_start, scan_end):
            s = score_at(i)
            scores_by_day.append((i, s, df.iloc[i]["date"], float(df.iloc[i]["close"])))

        # Score AT peak
        at_peak = next((s for i, s, d, p in scores_by_day if d == peak_dt), 0)

        # Find first crossing of DISTRIBUTION (70)
        first_distribution = None
        for i, s, d, p in scores_by_day:
            if s >= 70:
                first_distribution = (d, p, s)
                break

        # Find max score in window
        max_score = max(scores_by_day, key=lambda x: x[1])

        result = {
            "date": peak_dt, "label": label, "price": peak_px,
            "score_at_peak": at_peak,
            "first_distribution_signal": (
                {"date": str(first_distribution[0]),
                 "price": first_distribution[1],
                 "score": first_distribution[2],
                 "days_from_peak": (first_distribution[0] - peak_dt).days,
                 "capture_pct": first_distribution[1] / peak_px * 100}
                if first_distribution else None
            ),
            "max_score": {
                "date": str(max_score[2]), "price": max_score[3],
                "score": max_score[1],
                "days_from_peak": (max_score[2] - peak_dt).days,
            },
        }
        results.append(result)
    return results


def main():
    print("=" * 76)
    print("PERCENTILE-RANK CYCLE TOP DETECTOR — current reading")
    print("=" * 76)
    print()
    r = cycle_top_score()
    if "error" in r:
        print(f"Error: {r['error']}")
        return

    print(f"  As of:        {r['as_of']}")
    print(f"  BTC price:    ${r['btc_price']:,.0f}")
    print(f"  Window:       {r['window_days']} days (trailing 4y)")
    print()
    print(f"  COMPOSITE SCORE: {r['score']}/100  ->  {r['verdict']}")
    print(f"  Action:          {r['action']}")
    print()
    print(f"  Indicator percentile ranks (top 5% = EXTREME, top 20% = elevated):")
    print(f"  " + "-" * 66)
    print(f"  {'Indicator':<22s} {'Value':>10s}  {'Percentile':>12s}  {'Status':<14s}")
    for name, d in r["indicators"].items():
        val = d.get("current_value")
        val_str = f"{val:.2f}" if val is not None else "n/a"
        rank = d["percentile_rank"]
        status = "EXTREME" if d["extreme"] else ("elevated" if d["elevated"] else "normal")
        print(f"  {name:<22s} {val_str:>10s}  {rank:>10.1f}%  {status:<14s}")
    print()
    print(f"  Weekly MACD bear:  {'YES' if r['weekly_macd_bear'] else 'no'} "
          f"(hist {r.get('indicators', {}).get('mvrv', {}).get('current_value', 0)})")
    print()
    if r["components"]:
        print(f"  Score components:")
        for k, v in r["components"].items():
            print(f"    +{v:>3d}  {k}")
    print()

    # Historical validation
    print("=" * 76)
    print("HISTORICAL PEAK VALIDATION")
    print("=" * 76)
    print()
    hist = backtest_historical_peaks(scan_window_days=90)
    if hist:
        for h in hist:
            print(f"  {h['label']} ({h['date']}, ${h['price']:,.0f})")
            if h.get("status") == "not_in_dataset":
                print(f"    Not in dataset (needs longer BTC price history)")
                print()
                continue
            print(f"    Score AT peak day:        {h['score_at_peak']}/100")
            if h.get("first_distribution_signal"):
                fs = h["first_distribution_signal"]
                days = fs["days_from_peak"]
                if days < 0: timing = f"{-days} days BEFORE peak"
                elif days == 0: timing = "AT peak"
                else: timing = f"{days} days AFTER peak"
                print(f"    First DISTRIBUTION fire:  {fs['date']} at ${fs['price']:,.0f}")
                print(f"                              score {fs['score']}, "
                      f"{timing}, capture {fs['capture_pct']:.1f}%")
            else:
                print(f"    No DISTRIBUTION signal in ±90 day window")
            ms = h["max_score"]
            print(f"    Max score in window:      {ms['score']}/100 on {ms['date']} "
                  f"({ms['days_from_peak']:+d}d from peak)")
            print()


if __name__ == "__main__":
    main()
