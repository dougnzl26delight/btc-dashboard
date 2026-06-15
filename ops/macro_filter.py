"""Macro filter — DXY, 10Y yield, VIX cross-asset regime.

GCR consideration: crypto rallies rarely sustain during USD strength +
rising real yields + risk-off. This module fetches:
  - DXY (US dollar index, ticker ^DXY via yfinance)
  - 10Y treasury yield (^TNX)
  - VIX (^VIX)

Classifies macro regime:
  RISK_ON      : DXY weak, 10Y stable/falling, VIX low (< 18)
  NEUTRAL      : mixed signals
  RISK_OFF     : DXY breaking out (above 200d) + 10Y rising fast OR VIX > 25
  FLIGHT       : DXY spiking + VIX > 30 = capitulation regime

Alerts when macro flips, especially into RISK_OFF/FLIGHT during pro_trend
long positions.

Scheduled as Crypto_macro_filter (daily 14:30 NZ — after equity close).
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

from ops.alerts import alert

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False


MACRO_LOG = REPO_ROOT / ".macro_filter_log.jsonl"


def fetch_macro_series(period: str = "6mo") -> dict:
    """Returns DXY, 10Y, VIX as Series."""
    if not HAS_YF:
        return {}
    out = {}
    for ticker, label in [("DX-Y.NYB", "DXY"), ("^TNX", "TNX"), ("^VIX", "VIX")]:
        try:
            data = yf.Ticker(ticker).history(period=period)
            if not data.empty:
                out[label] = data["Close"]
        except Exception as e:
            print(f"  fetch {label} failed: {e}")
            continue
    return out


def classify_macro(dxy: pd.Series, tnx: pd.Series, vix: pd.Series) -> dict:
    """Classify current macro regime."""
    if dxy.empty or tnx.empty or vix.empty:
        return {"regime": "unknown", "reason": "missing data"}

    # Latest values
    dxy_now = float(dxy.iloc[-1])
    tnx_now = float(tnx.iloc[-1])
    vix_now = float(vix.iloc[-1])

    # DXY trend: above/below 100d SMA
    dxy_sma = float(dxy.rolling(100).mean().iloc[-1]) if len(dxy) >= 100 else dxy_now
    dxy_breakout_up = dxy_now > dxy_sma * 1.02
    dxy_breakdown = dxy_now < dxy_sma * 0.98

    # 10Y velocity: change over last 20 days
    tnx_20d_change = float(tnx_now - tnx.iloc[-20]) if len(tnx) >= 21 else 0
    tnx_rising_fast = tnx_20d_change > 0.5  # +0.50 pp in 20 days = aggressive

    # VIX regime
    vix_low = vix_now < 18
    vix_high = vix_now > 25
    vix_extreme = vix_now > 30

    # Classification
    if dxy_breakout_up and vix_extreme:
        regime = "FLIGHT"
        reason = (f"DXY +{(dxy_now/dxy_sma-1)*100:.1f}% above 100d SMA, "
                  f"VIX {vix_now:.1f} > 30")
    elif dxy_breakout_up and (tnx_rising_fast or vix_high):
        regime = "RISK_OFF"
        reason = (f"DXY breakout (+{(dxy_now/dxy_sma-1)*100:.1f}% vs 100d), "
                  f"10Y {'rising fast' if tnx_rising_fast else 'stable'}, "
                  f"VIX {vix_now:.1f}")
    elif dxy_breakdown and vix_low and not tnx_rising_fast:
        regime = "RISK_ON"
        reason = (f"DXY weak (-{(1-dxy_now/dxy_sma)*100:.1f}% vs 100d), "
                  f"VIX {vix_now:.1f} < 18, 10Y stable")
    else:
        regime = "NEUTRAL"
        reason = (f"Mixed: DXY {dxy_now:.1f} (vs SMA {dxy_sma:.1f}), "
                  f"10Y {tnx_now:.2f} (Δ20d {tnx_20d_change:+.2f}), "
                  f"VIX {vix_now:.1f}")

    return {
        "regime": regime,
        "reason": reason,
        "dxy": dxy_now,
        "dxy_sma100": dxy_sma,
        "dxy_pct_vs_sma": dxy_now / dxy_sma - 1,
        "tnx": tnx_now,
        "tnx_20d_change": tnx_20d_change,
        "vix": vix_now,
    }


def main() -> dict:
    if not HAS_YF:
        return {"status": "yfinance not installed",
                "msg": "pip install yfinance to enable macro filter"}

    series = fetch_macro_series(period="6mo")
    if len(series) < 3:
        return {"status": "incomplete data", "got": list(series.keys())}

    result = classify_macro(series["DXY"], series["TNX"], series["VIX"])
    snapshot = {
        "ts": datetime.now(timezone.utc).isoformat(),
        **result,
    }

    # Alert on regime entries (RISK_OFF or FLIGHT)
    if result["regime"] in ("RISK_OFF", "FLIGHT"):
        # Check previous regime
        prior_regime = None
        if MACRO_LOG.exists():
            for line in MACRO_LOG.read_text().strip().split("\n")[::-1]:
                if line:
                    try:
                        prior_regime = json.loads(line)["regime"]
                        break
                    except Exception:
                        continue
        if prior_regime != result["regime"]:
            alert(
                f"MACRO -> {result['regime']}: {result['reason']}. "
                f"Crypto longs at elevated risk; consider reduced pro_trend allocation.",
                level="warning",
            )

    # Append daily log
    today_iso = datetime.now(timezone.utc).date().isoformat()
    if MACRO_LOG.exists():
        last_line = None
        for line in MACRO_LOG.read_text().strip().split("\n"):
            if line:
                last_line = line
        if last_line:
            try:
                if json.loads(last_line)["ts"][:10] == today_iso:
                    return snapshot
            except Exception:
                pass
    with MACRO_LOG.open("a") as f:
        f.write(json.dumps(snapshot, default=str) + "\n")

    return snapshot


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2, default=str))
