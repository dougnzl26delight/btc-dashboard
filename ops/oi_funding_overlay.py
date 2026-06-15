"""OI / funding regime overlay — GCR-style positioning awareness.

Reads perp open interest + funding rate for the 5 universe pairs + a few
extra majors. Classifies each into one of:

  FROTH       : high OI (>P80 of 60d) + high funding (>1.5 bps/8h)
                  -> retail bullish positioning at peaks; top warning
  EXHAUSTION  : high OI (>P80) + negative funding (<-0.5 bps/8h)
                  -> retail bearish positioning at troughs; bottom hint
  SQUEEZE_LONG: high OI + crashing price + extreme positive funding
                  -> short squeeze potential
  SQUEEZE_SHORT: high OI + spiking price + extreme negative funding
                  -> long squeeze potential
  NEUTRAL     : everything else

Alert-only. Does NOT auto-trade. Provides regime context for discretionary
review during the weekly check-in.

Scheduled as Crypto_oi_funding_overlay (daily 14:25 NZ).
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


OI_FUNDING_LOG = REPO_ROOT / ".oi_funding_log.jsonl"

UNIVERSE_PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "AVAX/USDT",
    "LINK/USDT", "DOGE/USDT",
]

# Thresholds
OI_PCT_THRESHOLD = 0.80
FROTH_FUNDING_BPS_8H = 1.5
NEGATIVE_FUNDING_BPS_8H = -0.5
EXTREME_FUNDING_BPS_8H = 3.0


def fetch_oi_history(pair: str, days_back: int = 60) -> pd.Series:
    """Fetch OI history; returns Series of notional OI in USDT."""
    perp_pair = f"{pair}:USDT" if ":" not in pair else pair
    try:
        # ccxt fetch_open_interest_history — daily bars
        rows = data._EX.fetch_open_interest_history(
            perp_pair, timeframe="1d", limit=days_back
        )
        if not rows:
            return pd.Series(dtype=float)
        df = pd.DataFrame([{
            "ts": pd.Timestamp(r["timestamp"], unit="ms", tz="UTC"),
            "oi_value": float(r.get("openInterestValue", 0) or r.get("openInterestAmount", 0)),
        } for r in rows])
        return df.set_index("ts")["oi_value"]
    except Exception as e:
        return pd.Series(dtype=float)


def latest_funding_bps_8h(pair: str) -> float:
    perp_pair = f"{pair}:USDT" if ":" not in pair else pair
    try:
        f = data.funding_history(perp_pair, limit=1)
        if f.empty:
            return 0.0
        return float(f["funding_rate"].iloc[-1]) * 10_000.0
    except Exception:
        return 0.0


def recent_price_action(pair: str, days: int = 7) -> dict:
    try:
        df = data.ohlcv_extended(pair, days_back=days + 5)
        if df.empty:
            return {}
        recent = df.tail(days)
        return {
            "ret_7d": float(recent["close"].iloc[-1] / recent["close"].iloc[0] - 1),
            "max_drawdown_7d": float(
                (1 - recent["close"] / recent["close"].cummax()).max()
            ),
            "current_price": float(recent["close"].iloc[-1]),
        }
    except Exception:
        return {}


def classify(oi_pct: float, funding_bps: float, ret_7d: float, dd_7d: float) -> str:
    """Map OI/funding/price into regime label."""
    high_oi = oi_pct > OI_PCT_THRESHOLD

    if high_oi and funding_bps > EXTREME_FUNDING_BPS_8H and dd_7d > 0.10:
        return "SQUEEZE_LONG"  # high funding + crashing = shorts about to get crushed back
    if high_oi and funding_bps < -1.0 and ret_7d > 0.10:
        return "SQUEEZE_SHORT"  # negative funding + spiking = longs about to get blown out
    if high_oi and funding_bps > FROTH_FUNDING_BPS_8H:
        return "FROTH"
    if high_oi and funding_bps < NEGATIVE_FUNDING_BPS_8H:
        return "EXHAUSTION"
    return "NEUTRAL"


def main() -> dict:
    snapshot = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "by_pair": {},
    }
    alerts_fired = []
    summary_count = {"FROTH": 0, "EXHAUSTION": 0, "SQUEEZE_LONG": 0,
                     "SQUEEZE_SHORT": 0, "NEUTRAL": 0}

    for pair in UNIVERSE_PAIRS:
        oi_series = fetch_oi_history(pair, days_back=60)
        funding_bps = latest_funding_bps_8h(pair)
        price_action = recent_price_action(pair)

        if oi_series.empty:
            snapshot["by_pair"][pair] = {"error": "no OI data"}
            continue

        current_oi = float(oi_series.iloc[-1])
        oi_pct = float((oi_series < current_oi).mean())  # current OI's percentile
        ret_7d = price_action.get("ret_7d", 0)
        dd_7d = price_action.get("max_drawdown_7d", 0)
        regime = classify(oi_pct, funding_bps, ret_7d, dd_7d)
        summary_count[regime] += 1

        snapshot["by_pair"][pair] = {
            "current_oi_usdt": current_oi,
            "oi_pct_of_60d": oi_pct,
            "funding_bps_8h": funding_bps,
            "funding_ann_pct": funding_bps * 3 * 365 / 100,
            "ret_7d": ret_7d,
            "max_dd_7d": dd_7d,
            "regime": regime,
        }

        if regime != "NEUTRAL":
            msg = f"{regime} {pair}: OI P{oi_pct*100:.0f}, funding {funding_bps:+.2f}bps/8h, 7d ret {ret_7d:+.1%}"
            alerts_fired.append(msg)
            alert(f"OI_FUNDING [{regime}]: {msg}", level="info")

    snapshot["summary"] = summary_count
    snapshot["alerts_fired"] = alerts_fired

    # Append to log (one entry per day)
    today = datetime.now(timezone.utc).date().isoformat()
    if OI_FUNDING_LOG.exists():
        last_line = None
        for line in OI_FUNDING_LOG.read_text().strip().split("\n"):
            if line:
                last_line = line
        if last_line:
            try:
                last = json.loads(last_line)
                if last["ts"][:10] == today:
                    return snapshot
            except Exception:
                pass
    with OI_FUNDING_LOG.open("a") as f:
        f.write(json.dumps(snapshot, default=str) + "\n")

    return snapshot


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2, default=str))
