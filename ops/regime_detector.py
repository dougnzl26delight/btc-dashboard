"""Crypto regime detector — bull / bear / chop classifier.

Sim evidence: pro_trend works in clear bull (2023 +274%) and clear bear
(2022 -9% vs BAH -64% = MASSIVE protection), but UNDER-performs in chop
(2024 +2%, 2025 -9%). The user's psychological challenge is recognizing
"strategy is in its weakness zone, this is documented behavior" rather
than "strategy is broken, must change something."

This detector classifies the current crypto regime daily. When regime =
chop, alerts: "Expected weakness; do not change parameters."

Classification logic (BTC-anchored, since BTC drives all 5 universe pairs):
  - 30-day realized vol (annualized)
  - 200-day price vs SMA200 (% above/below)
  - 60-day return vs flat
  - ADX-style trend strength

Regimes:
  BULL:  price > SMA200, 60d return > +15%, vol moderate-high
  BEAR:  price < SMA200, 60d return < -15%
  CHOP:  abs(price/SMA200 - 1) < 5%, 60d return in [-10%, +10%], low vol
  TRANSITION: anything that doesn't match the above

Outputs:
  - Current regime label
  - Days in current regime
  - Last 4 regime transitions
  - Expected strategy behavior in current regime (from sim evidence)
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


REGIME_LOG = REPO_ROOT / ".regime_log.jsonl"

# Expected strategy behavior per regime (from max_simulation.py + stress replays)
REGIME_EXPECTATIONS = {
    "BULL": {
        "expected_strategy_behavior": "Strong positive returns, sometimes lagging BAH",
        "expected_annualized": "+50-150%",
        "expected_max_dd": "20-30%",
        "do_not_panic_about": "DDs up to 30% — this is normal in bull regimes",
    },
    "BEAR": {
        "expected_strategy_behavior": "Sit in cash or short positions, mostly flat to mildly negative",
        "expected_annualized": "-5% to +5%",
        "expected_max_dd": "5-15%",
        "do_not_panic_about": "Lack of new entries — strategy is correctly waiting",
    },
    "CHOP": {
        "expected_strategy_behavior": "UNDER-PERFORMS — small whipsaw losses; few real trades",
        "expected_annualized": "-15% to +5%",
        "expected_max_dd": "20-30%",
        "do_not_panic_about": "Long flat-to-down stretches are documented behavior",
    },
    "TRANSITION": {
        "expected_strategy_behavior": "Mixed — depends on which way it resolves",
        "expected_annualized": "highly variable",
        "expected_max_dd": "variable",
        "do_not_panic_about": "Mixed signals; system manages on existing rules",
    },
}


def classify_btc_regime() -> dict:
    """Classify current BTC regime using daily bars."""
    df = data.ohlcv_extended("BTC/USDT", days_back=300)
    if df.empty or len(df) < 250:
        return {"error": "insufficient BTC data"}
    df = df.copy()
    df["sma200"] = df["close"].rolling(200).mean()
    df["sma50"] = df["close"].rolling(50).mean()
    df["ret60"] = df["close"].pct_change(60)
    df["ret30"] = df["close"].pct_change(30)
    df["realized_vol_30d"] = (
        df["close"].pct_change().rolling(30).std() * np.sqrt(365)
    )
    df = df.dropna()

    last = df.iloc[-1]
    price = float(last["close"])
    sma200 = float(last["sma200"])
    sma50 = float(last["sma50"])
    ret60 = float(last["ret60"])
    ret30 = float(last["ret30"])
    vol = float(last["realized_vol_30d"])

    pct_vs_sma200 = price / sma200 - 1
    sma_separation = abs(sma50 / sma200 - 1)

    # Classification rules
    if price > sma200 and ret60 > 0.15 and vol > 0.30:
        regime = "BULL"
    elif price < sma200 and ret60 < -0.15:
        regime = "BEAR"
    elif (abs(pct_vs_sma200) < 0.05
            and abs(ret60) < 0.10
            and vol < 0.45):
        regime = "CHOP"
    else:
        regime = "TRANSITION"

    return {
        "regime": regime,
        "price": price,
        "sma200": sma200,
        "sma50": sma50,
        "pct_vs_sma200": pct_vs_sma200,
        "sma_separation": sma_separation,
        "ret60d": ret60,
        "ret30d": ret30,
        "realized_vol_30d_ann": vol,
        "expectations": REGIME_EXPECTATIONS[regime],
    }


def append_regime_log(snapshot: dict) -> None:
    """Append; idempotent — only one entry per UTC date."""
    today = datetime.now(timezone.utc).date().isoformat()
    if REGIME_LOG.exists():
        last_line = None
        for line in REGIME_LOG.read_text().strip().split("\n"):
            if line:
                last_line = line
        if last_line:
            try:
                last = json.loads(last_line)
                if last["ts"][:10] == today:
                    return
            except Exception:
                pass
    snapshot["ts"] = datetime.now(timezone.utc).isoformat()
    with REGIME_LOG.open("a") as f:
        f.write(json.dumps(snapshot, default=str) + "\n")


def load_regime_log() -> list[dict]:
    if not REGIME_LOG.exists():
        return []
    rows = []
    for line in REGIME_LOG.read_text().strip().split("\n"):
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def days_in_current_regime(log: list[dict]) -> int:
    if not log:
        return 0
    current = log[-1]["regime"]
    n = 1
    for entry in reversed(log[:-1]):
        if entry["regime"] != current:
            break
        n += 1
    return n


def recent_transitions(log: list[dict], n_recent: int = 4) -> list[dict]:
    """Last N regime changes, in chronological order."""
    if len(log) < 2:
        return []
    transitions = []
    for i in range(1, len(log)):
        if log[i]["regime"] != log[i - 1]["regime"]:
            transitions.append({
                "date": log[i]["ts"][:10],
                "from": log[i - 1]["regime"],
                "to": log[i]["regime"],
            })
    return transitions[-n_recent:]


def main() -> dict:
    snap = classify_btc_regime()
    if "error" in snap:
        return snap

    append_regime_log(snap)

    log = load_regime_log()
    days_in_regime = days_in_current_regime(log)
    transitions = recent_transitions(log)

    # Alert on regime entry into CHOP — the "do not interfere" reminder
    if snap["regime"] == "CHOP" and days_in_regime == 1 and len(log) > 1:
        prior_regime = log[-2]["regime"] if len(log) > 1 else "unknown"
        alert(
            f"REGIME -> CHOP (from {prior_regime}). Strategy under-performs in chop. "
            f"Expected: {snap['expectations']['expected_annualized']} ann return. "
            f"DO NOT change parameters during chop regimes — this is documented "
            f"strategy behavior, not a malfunction. Read STRATEGY_CHARTER.md.",
            level="warning",
        )

    return {
        "current": snap,
        "days_in_regime": days_in_regime,
        "recent_transitions": transitions,
    }


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2, default=str))
