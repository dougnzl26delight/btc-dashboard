"""Sim comparator — locate live performance in forward Monte Carlo distribution.

The single best psychological tool: knowing whether your DD is normal or broken.

Each month, this script:
  1. Loads cached Monte Carlo distribution (from max_simulation.py output)
  2. Computes live N-day return for matching horizons
  3. Locates each live observation as a percentile of the sim distribution
  4. Reports: "live 30-day return is at P15 of sim — within normal range"
     or "live 90-day return is at P3 of sim — significantly worse than expected"

Alert fires when live percentile falls below P5 for 3+ consecutive periods —
that's genuine evidence of strategy degradation, not just unlucky chop.

Cached forward distribution lives in monte_carlo_results/. Refreshed by the
monthly_oos.py task (first Sunday) which now also re-runs Monte Carlo.

Scheduled as Crypto_sim_comparator_weekly.
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


EQUITY_LOG = REPO_ROOT / ".equity_log.jsonl"
SIM_DIR = REPO_ROOT / "monte_carlo_results"
COMPARATOR_LOG = REPO_ROOT / ".sim_comparator_log.jsonl"

# Production strategy in the sim ("70/30 combined" most matches live config
# now that XSMOM is wired). Block bootstrap, 50% haircut as realistic.
TARGET_STRATEGY = "70/30 combined"
TARGET_HAIRCUT = 0.5
TARGET_METHOD = "block"

# Horizons we'll compare at, in days
COMPARE_HORIZONS = [30, 90, 365, 730]


def load_equity_log() -> pd.DataFrame:
    if not EQUITY_LOG.exists():
        return pd.DataFrame()
    rows = []
    for line in EQUITY_LOG.read_text().strip().split("\n"):
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.set_index("ts").sort_index()
    return df


def load_latest_sim() -> dict | None:
    """Find newest max_sim_*.json in monte_carlo_results/ and return its results."""
    if not SIM_DIR.exists():
        return None
    files = sorted(SIM_DIR.glob("max_sim_*.json"))
    if not files:
        return None
    latest = files[-1]
    return {"file": latest.name, "results": json.loads(latest.read_text())}


def find_sim_record(sim_results: list[dict], horizon: int) -> dict | None:
    """Locate the sim record matching our target strategy + haircut + method + horizon."""
    for r in sim_results:
        if (r.get("strategy") == TARGET_STRATEGY
                and r.get("horizon_days") == horizon
                and r.get("haircut") == TARGET_HAIRCUT
                and r.get("method") == TARGET_METHOD):
            return r
    return None


def live_n_day_return(eq_df: pd.DataFrame, n: int) -> float | None:
    """Compute live N-day return; None if not enough data."""
    if eq_df.empty or len(eq_df) < 2:
        return None
    eq = eq_df["total_equity"].astype(float)
    today_idx = eq.index[-1]
    cutoff = today_idx - pd.Timedelta(days=n)
    past = eq[eq.index <= cutoff]
    if past.empty:
        return None
    return float(eq.iloc[-1] / past.iloc[-1] - 1)


def percentile_in_sim(live_value: float, sim_record: dict) -> float:
    """Map live observation to a percentile in the sim distribution.

    The sim record stores P5/P25/P50/P75/P95 of total_returns. We do
    piecewise linear interpolation between known percentiles.
    """
    pcts = [(5, sim_record["tot_p5"]),
            (25, sim_record["tot_p25"]),
            (50, sim_record["tot_p50"]),
            (75, sim_record["tot_p75"]),
            (95, sim_record["tot_p95"])]

    # Below P5
    if live_value <= pcts[0][1]:
        # Extrapolate linearly using P5-P25 slope
        slope = (pcts[1][0] - pcts[0][0]) / max(pcts[1][1] - pcts[0][1], 1e-9)
        rank = pcts[0][0] + slope * (live_value - pcts[0][1])
        return max(0.1, rank)

    # Above P95
    if live_value >= pcts[-1][1]:
        slope = (pcts[-1][0] - pcts[-2][0]) / max(pcts[-1][1] - pcts[-2][1], 1e-9)
        rank = pcts[-1][0] + slope * (live_value - pcts[-1][1])
        return min(99.9, rank)

    # Interpolate between known points
    for i in range(len(pcts) - 1):
        p_lo, v_lo = pcts[i]
        p_hi, v_hi = pcts[i + 1]
        if v_lo <= live_value <= v_hi:
            slope = (p_hi - p_lo) / max(v_hi - v_lo, 1e-9)
            return p_lo + slope * (live_value - v_lo)
    return 50.0


def append_log(comparison: dict) -> None:
    today = datetime.now(timezone.utc).date().isoformat()
    if COMPARATOR_LOG.exists():
        last_line = None
        for line in COMPARATOR_LOG.read_text().strip().split("\n"):
            if line:
                last_line = line
        if last_line:
            try:
                last = json.loads(last_line)
                if last["ts"][:10] == today:
                    return
            except Exception:
                pass
    comparison["ts"] = datetime.now(timezone.utc).isoformat()
    with COMPARATOR_LOG.open("a") as f:
        f.write(json.dumps(comparison, default=str) + "\n")


def consecutive_below_p5(log: list[dict], horizon: int) -> int:
    """Count consecutive entries where live percentile at this horizon was < 5."""
    n = 0
    for entry in reversed(log):
        h_data = entry.get("by_horizon", {}).get(str(horizon))
        if h_data is None:
            break
        if h_data["live_percentile"] < 5:
            n += 1
        else:
            break
    return n


def main() -> dict:
    eq_df = load_equity_log()
    sim_data = load_latest_sim()

    if eq_df.empty:
        return {"status": "no_equity_log", "msg": "Need equity log entries first."}
    if sim_data is None:
        return {"status": "no_sim_data",
                "msg": "Run core/max_simulation.py first to generate sim baseline."}

    n_days_logged = (eq_df.index[-1] - eq_df.index[0]).days
    by_horizon = {}
    alerts_fired = []

    for horizon in COMPARE_HORIZONS:
        if n_days_logged < horizon:
            continue
        live_ret = live_n_day_return(eq_df, horizon)
        if live_ret is None:
            continue

        sim_rec = find_sim_record(sim_data["results"], horizon)
        if sim_rec is None:
            continue

        pct = percentile_in_sim(live_ret, sim_rec)
        verdict = ("STRONG (above P75)" if pct > 75
                   else "OK (P50-P75)" if pct > 50
                   else "OK (P25-P50)" if pct > 25
                   else "BELOW (P5-P25)" if pct > 5
                   else "FAR BELOW SIM (below P5)")

        by_horizon[str(horizon)] = {
            "live_return": live_ret,
            "live_percentile": pct,
            "verdict": verdict,
            "sim_p5": sim_rec["tot_p5"],
            "sim_p50": sim_rec["tot_p50"],
            "sim_p95": sim_rec["tot_p95"],
        }

    if not by_horizon:
        return {"status": "insufficient_history",
                "msg": f"Need at least {min(COMPARE_HORIZONS)} days of equity log; have {n_days_logged}"}

    comparison = {
        "n_days_logged": n_days_logged,
        "sim_file": sim_data["file"],
        "by_horizon": by_horizon,
    }
    append_log(comparison)

    # Alert if live has been below P5 for 3+ consecutive checks at any horizon
    log = []
    if COMPARATOR_LOG.exists():
        for line in COMPARATOR_LOG.read_text().strip().split("\n"):
            if line:
                try:
                    log.append(json.loads(line))
                except Exception:
                    continue

    for horizon in COMPARE_HORIZONS:
        consec = consecutive_below_p5(log, horizon)
        if consec >= 3:
            msg = (f"Live {horizon}-day return below sim P5 for "
                   f"{consec} consecutive checks. Genuine degradation signal.")
            alert(msg, level="critical")
            alerts_fired.append(msg)

    comparison["alerts_fired"] = alerts_fired
    return comparison


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2, default=str))
