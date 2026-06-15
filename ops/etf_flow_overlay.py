"""Daily ETF flow overlay — GCR-style catalyst awareness.

Wraps research/etf_flows.py — fetches recent BTC ETF flows (Farside),
computes z-score vs 30-day distribution. Alerts on:
  - Big day (>$500M net inflow or outflow)
  - >2σ above/below recent mean (extreme regime)
  - Streak of 5+ consecutive same-direction days

Alert-only; no auto-trading. GCR uses ETF flow as a regime indicator
and pre-positions on extremes.

Scheduled as Crypto_etf_flow_overlay (daily 14:28 NZ).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from research.etf_flows import fetch_etf_flows, etf_flow_signal
from ops.alerts import alert


ETF_LOG = REPO_ROOT / ".etf_flow_log.jsonl"

BIG_DAY_THRESHOLD_M = 500.0   # +/- $500M = "big day"
EXTREME_Z_THRESHOLD = 2.0      # 2σ event
STREAK_THRESHOLD = 5            # 5 consecutive same-direction days


def main() -> dict:
    try:
        flows = fetch_etf_flows()
    except Exception as e:
        return {"status": "error", "error": f"fetch_etf_flows failed: {e}"}

    if flows.empty or len(flows) < 7:
        return {"status": "insufficient_data", "n_days": len(flows)}

    recent_30 = flows.tail(30)
    mean_30 = float(recent_30.mean())
    std_30 = float(recent_30.std())
    today_flow = float(flows.iloc[-1])
    today_z = (today_flow - mean_30) / std_30 if std_30 > 0 else 0
    today_date = str(flows.index[-1].date())

    # Recent streak
    signs = np.sign(flows.tail(10).values)
    streak = 1
    for i in range(len(signs) - 2, -1, -1):
        if signs[i] == signs[-1] and signs[i] != 0:
            streak += 1
        else:
            break

    # Signal calc (from research module)
    sig = etf_flow_signal(flows, ema_window=7)
    signal_value = float(sig.iloc[-1]) if not sig.empty else 0

    snapshot = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "today_date": today_date,
        "today_flow_m": today_flow,
        "30d_mean_m": mean_30,
        "30d_std_m": std_30,
        "today_z_score": today_z,
        "streak_days": streak,
        "streak_direction": "inflow" if signs[-1] > 0 else "outflow",
        "signal_value": signal_value,
        "alerts_fired": [],
    }

    # Big day alert
    if abs(today_flow) > BIG_DAY_THRESHOLD_M:
        msg = (f"ETF FLOW BIG DAY: ${today_flow:+,.0f}M on {today_date} "
               f"(z={today_z:+.2f})")
        snapshot["alerts_fired"].append(msg)
        alert(msg, level="info")

    # Extreme z-score alert
    if abs(today_z) > EXTREME_Z_THRESHOLD:
        direction = "INFLOW" if today_z > 0 else "OUTFLOW"
        msg = (f"ETF FLOW EXTREME {direction}: z={today_z:+.2f} > "
               f"{EXTREME_Z_THRESHOLD} (today ${today_flow:+,.0f}M, "
               f"30d mean ${mean_30:+,.0f}M)")
        snapshot["alerts_fired"].append(msg)
        alert(msg, level="warning")

    # Streak alert
    if streak >= STREAK_THRESHOLD:
        msg = (f"ETF FLOW STREAK: {streak} consecutive "
               f"{snapshot['streak_direction']} days")
        snapshot["alerts_fired"].append(msg)
        alert(msg, level="info")

    # Append daily log
    today_iso = datetime.now(timezone.utc).date().isoformat()
    if ETF_LOG.exists():
        last_line = None
        for line in ETF_LOG.read_text().strip().split("\n"):
            if line:
                last_line = line
        if last_line:
            try:
                if json.loads(last_line)["ts"][:10] == today_iso:
                    return snapshot
            except Exception:
                pass
    with ETF_LOG.open("a") as f:
        f.write(json.dumps(snapshot, default=str) + "\n")

    return snapshot


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2, default=str))
