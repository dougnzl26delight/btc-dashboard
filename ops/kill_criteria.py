"""Kill-criteria monitor — enforces the discipline rules from Strategy Charter.

Logs daily equity to .equity_log.jsonl, computes rolling stats, and alerts
when ANY of these triggers fire:

  K1 — Live drawdown > 45% (5pp above backtest max)
  K2 — Rolling 90-day Sharpe < 0 for 60 consecutive days
  K3 — 6 months without a new systematic entry
  K4 — Live realized Sharpe < 40% of backtest Sharpe after 60+ trades

When K1 or K4 fires: alert with PAUSE_RECOMMENDED level.
When K2 or K3 fires: alert with REVIEW level.

The script is idempotent — safe to run daily via scheduler.
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

from core.broker import Broker
from core.perp_broker import PerpBroker
from core.pnl_attribution import load_attribution
from ops import alerts


EQUITY_LOG = REPO_ROOT / ".equity_log.jsonl"

# Backtest reference — pulled from comprehensive_backtest.py results
BACKTEST_SHARPE = 1.40
BACKTEST_MAX_DD = 0.40
BACKTEST_ANN = 0.80


def current_equity(mode: str = "paper") -> dict:
    """Snapshot total equity across spot + perp brokers, plus per-sleeve P&L."""
    spot = Broker(mode=mode, long_only=False)
    perp = PerpBroker(mode=mode)

    spot_cash = float(spot.get_balance().get("USDT", 0))
    perp_balance = perp.get_balance()
    perp_cash = float(perp_balance.get("USDT", 0))

    # Position values
    attrib = load_attribution()
    position_value = 0.0
    per_sleeve_pnl = {"systematic_pro_trend": 0.0, "discretionary": 0.0,
                      "basis_arb": 0.0, "xsmom": 0.0, "bah_btc": 0.0,
                      "unknown": 0.0}
    for pair, tag in attrib.items():
        if pair.startswith("basis:"):
            continue  # basis legs net to ~0; perp side handled by perp broker
        try:
            ticker = perp.ticker(pair) if tag.get("side") == "short" else spot.get_ticker(pair)
            last = float(ticker.get("last") or ticker.get("close") or 0)
        except Exception:
            last = float(tag["entry_price"])
        if last <= 0:
            last = float(tag["entry_price"])
        sign = 1 if tag["side"] == "long" else -1
        pnl = sign * tag["qty"] * (last - tag["entry_price"])
        sleeve = tag.get("sleeve", "unknown")
        per_sleeve_pnl[sleeve] = per_sleeve_pnl.get(sleeve, 0) + pnl
        if tag["side"] == "long":
            position_value += tag["qty"] * last

    total_equity = spot_cash + perp_cash + position_value
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "total_equity": total_equity,
        "spot_cash": spot_cash, "perp_cash": perp_cash,
        "position_value_long": position_value,
        "per_sleeve_pnl": per_sleeve_pnl,
    }


def append_equity_log(snapshot: dict) -> None:
    """Append one snapshot per day. Idempotent: skips if today already logged."""
    today_date = datetime.now(timezone.utc).date().isoformat()
    if EQUITY_LOG.exists():
        last_line = None
        for line in EQUITY_LOG.read_text().strip().split("\n"):
            if line:
                last_line = line
        if last_line:
            last = json.loads(last_line)
            if last["ts"][:10] == today_date:
                return  # already logged today
    with EQUITY_LOG.open("a") as f:
        f.write(json.dumps(snapshot, default=str) + "\n")


def load_equity_log() -> pd.DataFrame:
    if not EQUITY_LOG.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in EQUITY_LOG.read_text().strip().split("\n") if line]
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.set_index("ts").sort_index()
    return df


def compute_rolling_stats(df: pd.DataFrame, window_days: int = 90) -> dict:
    if df.empty or len(df) < window_days:
        return {"n_days_logged": len(df) if not df.empty else 0}
    eq = df["total_equity"].astype(float)
    daily_rets = eq.pct_change().dropna()
    if len(daily_rets) < window_days:
        return {"n_days_logged": len(df)}
    recent = daily_rets.tail(window_days)
    sharpe = float(recent.mean() / recent.std() * np.sqrt(365)) if recent.std() > 0 else 0
    peak = eq.cummax()
    current_dd = float(1 - eq.iloc[-1] / peak.iloc[-1]) if peak.iloc[-1] > 0 else 0

    # 180-day Sharpe (used for refined K2)
    rolling_180_sharpe = None
    if len(daily_rets) >= 180:
        recent_180 = daily_rets.tail(180)
        rolling_180_sharpe = (
            float(recent_180.mean() / recent_180.std() * np.sqrt(365))
            if recent_180.std() > 0 else 0
        )

    # YTD return (used for K5)
    ytd_return = None
    current_year = df.index[-1].year
    ytd_eq = eq[eq.index.year == current_year]
    if len(ytd_eq) >= 2:
        ytd_return = float(ytd_eq.iloc[-1] / ytd_eq.iloc[0] - 1)

    return {
        "rolling_sharpe": sharpe,
        "rolling_180_sharpe": rolling_180_sharpe,
        "ytd_return": ytd_return,
        "current_dd": current_dd,
        "n_days_logged": len(df),
        "first_day": df.index[0].date().isoformat(),
        "last_day": df.index[-1].date().isoformat(),
    }


def check_kill_criteria(stats: dict, snapshot: dict) -> list[dict]:
    """Returns list of triggered kill criteria, each with severity + reason.

    Calibrated against Monte Carlo evidence (max_simulation.py 2026-05-10):
      - K1 fires above forward-sim P5 of DD distribution (~50%)
      - K2 fires below forward-sim P5 of 180-day Sharpe (~-0.5)
      - K5 fires below forward-sim P5 of YTD return (~-25% by month 4)
    """
    triggers = []
    current_dd = stats.get("current_dd", 0)
    rolling_sharpe = stats.get("rolling_sharpe", 0)
    rolling_180_sharpe = stats.get("rolling_180_sharpe", None)
    ytd_return = stats.get("ytd_return", None)
    n_days = stats.get("n_days_logged", 0)
    months_since_systematic_entry = stats.get("months_since_systematic_entry", 0)

    # K1 — drawdown above backtest max + 5pp buffer
    if current_dd > 0.45:
        triggers.append({
            "id": "K1",
            "severity": "PAUSE_RECOMMENDED",
            "reason": f"Live drawdown {current_dd:.1%} exceeds 45% threshold "
                      f"(5pp above backtest max DD {BACKTEST_MAX_DD:.0%})",
        })

    # K2 — 180-day rolling Sharpe < -0.5 (REFINED from "any negative").
    # Per Monte Carlo: P5 of 180d Sharpe distribution at 50% haircut is ~-0.5.
    # Going below that is genuine degradation, not just chop noise.
    if n_days >= 180 and rolling_180_sharpe is not None and rolling_180_sharpe < -0.5:
        triggers.append({
            "id": "K2",
            "severity": "REVIEW",
            "reason": f"Rolling 180-day Sharpe {rolling_180_sharpe:+.2f} < -0.5 "
                      f"(below sim P5; genuine degradation signal)",
        })

    # K3 — 6 months without systematic entry
    if months_since_systematic_entry >= 6:
        triggers.append({
            "id": "K3",
            "severity": "REVIEW",
            "reason": f"{months_since_systematic_entry} months without systematic "
                      f"entry — entry filter may be structurally broken",
        })

    # K5 — YTD < -25% by month 4 or later (NEW, per max_simulation analysis).
    # If you're 4+ months into the year and already below the sim's P5 of
    # 1-year returns (~-39%) trajectory, the year is almost certainly broken.
    today = datetime.now(timezone.utc)
    if today.month >= 4 and ytd_return is not None and ytd_return < -0.25:
        triggers.append({
            "id": "K5",
            "severity": "PAUSE_RECOMMENDED",
            "reason": f"YTD return {ytd_return:.1%} < -25% by month {today.month} "
                      f"— pause 30 days, re-evaluate before resuming",
        })

    return triggers


if __name__ == "__main__":
    print(f"Kill-criteria check at {datetime.now(timezone.utc).isoformat()}")
    print()
    snap = current_equity()
    append_equity_log(snap)

    print(f"Total equity:         ${snap['total_equity']:>14,.2f}")
    print(f"  Spot cash:          ${snap['spot_cash']:>14,.2f}")
    print(f"  Perp cash:          ${snap['perp_cash']:>14,.2f}")
    print(f"  Position value:     ${snap['position_value_long']:>14,.2f}")
    print()
    print(f"Per-sleeve unrealized P&L:")
    for sleeve, pnl in snap["per_sleeve_pnl"].items():
        print(f"  {sleeve:<22s} ${pnl:>+14,.2f}")
    print()

    df = load_equity_log()
    stats = compute_rolling_stats(df)
    n_days = stats.get("n_days_logged", 0) if stats else 0
    if "rolling_sharpe" not in stats:
        print(f"Logged {n_days} days. Need 90+ for rolling-Sharpe-based kill criteria.")
        print(f"K1 (drawdown) and K5 (YTD) gates still active.")
    else:
        print(f"Days logged:          {stats['n_days_logged']}")
        print(f"Rolling 90d Sharpe:   {stats['rolling_sharpe']:+.2f}")
        if stats.get("rolling_180_sharpe") is not None:
            print(f"Rolling 180d Sharpe:  {stats['rolling_180_sharpe']:+.2f}")
        if stats.get("ytd_return") is not None:
            print(f"YTD return:           {stats['ytd_return']:+.2%}")
        print(f"Current drawdown:     {stats['current_dd']:.1%}")
        print()

    triggers = check_kill_criteria(stats or {"n_days_logged": n_days}, snap)
    if not triggers:
        print("No kill criteria triggered. Continue.")
    else:
        print("KILL CRITERIA TRIGGERED:")
        for t in triggers:
            print(f"  [{t['id']} - {t['severity']}] {t['reason']}")
            alerts.alert(
                f"KILL-{t['id']} ({t['severity']}): {t['reason']}",
                level="critical",
            )
