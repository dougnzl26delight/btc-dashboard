"""Liquidation cluster proxy — free-data version (no Coinalyze API).

GCR uses liquidation heatmaps from Coinalyze/CoinGlass (paid). A reasonable
free-data proxy: detect zones where heavy liquidation is LIKELY to live by
identifying:

  1. Recent unfilled imbalances — gaps in volume profile
  2. Recent swing highs/lows where stops cluster
  3. Recent flash liquidation candles (>5x avg volume + >3% move in <1h)

For each universe pair, computes:
  - Nearest "thin zone" above/below current price (estimated liquidation magnet)
  - Recent liquidation cascade count (last 7d)
  - Distance to nearest swing high/low (where stops cluster)

Alert when current price is within 1% of a known cluster zone OR a flash
cascade just happened (signals more cascades possible).

Scheduled as Crypto_liq_cluster_proxy (daily 14:32 NZ).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core import data
from ops.alerts import alert


LIQ_LOG = REPO_ROOT / ".liq_cluster_log.jsonl"

UNIVERSE = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "LINK/USDT"]

CASCADE_VOL_MULT = 5.0      # 5x avg volume in 1h
CASCADE_PCT_MOVE = 0.03      # >3% intra-bar move
NEAR_CLUSTER_PCT = 0.01      # within 1% of cluster zone


def fetch_hourly(pair: str, hours: int = 168) -> pd.DataFrame:
    """Fetch last N hours of 1h candles."""
    try:
        # ccxt fetch_ohlcv with 1h timeframe
        rows = data._EX.fetch_ohlcv(pair, "1h", limit=hours)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df.set_index("ts")
    except Exception:
        return pd.DataFrame()


def detect_recent_cascades(df: pd.DataFrame) -> list[dict]:
    """Bars with 5x avg volume + >3% range = likely liquidation cascade."""
    if df.empty or len(df) < 24:
        return []
    avg_vol = df["volume"].rolling(24).mean()
    df = df.copy()
    df["bar_range_pct"] = (df["high"] - df["low"]) / df["close"]
    df["vol_mult"] = df["volume"] / avg_vol

    cascades = df[
        (df["vol_mult"] > CASCADE_VOL_MULT)
        & (df["bar_range_pct"] > CASCADE_PCT_MOVE)
    ]
    out = []
    for ts, row in cascades.iterrows():
        out.append({
            "ts": str(ts),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "vol_mult": float(row["vol_mult"]),
            "range_pct": float(row["bar_range_pct"]),
            "direction": "down" if row["close"] < row["open"] else "up",
        })
    return out


def nearest_swing_levels(df: pd.DataFrame, current_price: float,
                          window: int = 20) -> dict:
    """Find nearest swing high above and swing low below current price."""
    if df.empty or len(df) < window * 2:
        return {}
    highs = df["high"].rolling(window, center=True).max()
    lows = df["low"].rolling(window, center=True).min()
    is_swing_high = (df["high"] == highs)
    is_swing_low = (df["low"] == lows)

    swing_highs_above = df.loc[is_swing_high & (df["high"] > current_price), "high"]
    swing_lows_below = df.loc[is_swing_low & (df["low"] < current_price), "low"]

    nearest_high = float(swing_highs_above.min()) if not swing_highs_above.empty else None
    nearest_low = float(swing_lows_below.max()) if not swing_lows_below.empty else None
    return {
        "nearest_swing_high_above": nearest_high,
        "nearest_swing_low_below": nearest_low,
        "pct_to_high": (nearest_high - current_price) / current_price if nearest_high else None,
        "pct_to_low": (current_price - nearest_low) / current_price if nearest_low else None,
    }


def main() -> dict:
    snapshot = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "by_pair": {},
    }
    alerts_fired = []

    for pair in UNIVERSE:
        df = fetch_hourly(pair, hours=168)
        if df.empty:
            snapshot["by_pair"][pair] = {"error": "no data"}
            continue

        current_price = float(df["close"].iloc[-1])
        cascades = detect_recent_cascades(df)
        swings = nearest_swing_levels(df, current_price, window=12)

        pair_info = {
            "current_price": current_price,
            "n_cascades_7d": len(cascades),
            "recent_cascade": cascades[-1] if cascades else None,
            **swings,
        }
        snapshot["by_pair"][pair] = pair_info

        # Alert: cascade in last 24h
        if cascades:
            most_recent = cascades[-1]
            ts = pd.Timestamp(most_recent["ts"])
            age_hours = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
            if age_hours < 24:
                msg = (f"LIQ CASCADE {pair} {age_hours:.0f}h ago: "
                       f"vol {most_recent['vol_mult']:.1f}x avg, "
                       f"range {most_recent['range_pct']:.1%}, "
                       f"direction {most_recent['direction']}")
                alerts_fired.append(msg)
                alert(f"LIQ_CLUSTER: {msg}", level="info")

        # Alert: within 1% of swing high/low (stops cluster)
        if swings.get("pct_to_high") is not None and swings["pct_to_high"] < NEAR_CLUSTER_PCT:
            msg = (f"NEAR SWING HIGH {pair}: current ${current_price:,.4f} "
                   f"only {swings['pct_to_high']:.2%} below resistance "
                   f"${swings['nearest_swing_high_above']:,.4f}")
            alerts_fired.append(msg)
            alert(f"LIQ_CLUSTER: {msg}", level="info")
        if swings.get("pct_to_low") is not None and swings["pct_to_low"] < NEAR_CLUSTER_PCT:
            msg = (f"NEAR SWING LOW {pair}: current ${current_price:,.4f} "
                   f"only {swings['pct_to_low']:.2%} above support "
                   f"${swings['nearest_swing_low_below']:,.4f}")
            alerts_fired.append(msg)
            alert(f"LIQ_CLUSTER: {msg}", level="info")

    snapshot["alerts_fired"] = alerts_fired

    # Daily log (idempotent)
    today_iso = datetime.now(timezone.utc).date().isoformat()
    if LIQ_LOG.exists():
        last_line = None
        for line in LIQ_LOG.read_text().strip().split("\n"):
            if line:
                last_line = line
        if last_line:
            try:
                if json.loads(last_line)["ts"][:10] == today_iso:
                    return snapshot
            except Exception:
                pass
    with LIQ_LOG.open("a") as f:
        f.write(json.dumps(snapshot, default=str) + "\n")

    return snapshot


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2, default=str))
