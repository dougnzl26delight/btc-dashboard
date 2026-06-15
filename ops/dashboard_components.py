"""Dashboard widgets — forward-distribution envelope + sim-percentile rank.

Importable into the existing dashboard.py to add the "this is normal" chart.
Reads:
  - .equity_log.jsonl  (live equity points)
  - monte_carlo_results/max_sim_*.json  (sim distribution)

Provides:
  - forward_envelope_chart() — Plotly figure overlaying live equity on
    P5/P25/P50/P75/P95 forward simulation envelope
  - current_sim_percentile_rank() — dict: live percentile at each horizon
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
EQUITY_LOG = REPO_ROOT / ".equity_log.jsonl"
SIM_DIR = REPO_ROOT / "monte_carlo_results"

# Same target as sim_comparator.py
TARGET_STRATEGY = "70/30 combined"
TARGET_HAIRCUT = 0.5
TARGET_METHOD = "block"


def _load_equity() -> pd.DataFrame:
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
    return df.set_index("ts").sort_index()


def _latest_sim_results() -> list[dict]:
    if not SIM_DIR.exists():
        return []
    files = sorted(SIM_DIR.glob("max_sim_*.json"))
    if not files:
        return []
    return json.loads(files[-1].read_text())


def _sim_record(results: list[dict], horizon: int) -> dict | None:
    for r in results:
        if (r.get("strategy") == TARGET_STRATEGY
                and r.get("horizon_days") == horizon
                and r.get("haircut") == TARGET_HAIRCUT
                and r.get("method") == TARGET_METHOD):
            return r
    return None


def forward_envelope_data(start_equity: float = 100_000.0,
                           horizons: list[int] | None = None) -> dict:
    """Return data for the envelope chart: timestamps + bands + live equity.

    The envelope is constructed by stitching sim percentiles at each horizon.
    For days between horizons, we linearly interpolate the percentile bands.
    """
    horizons = horizons or [30, 90, 365, 730, 1825]
    sim_results = _latest_sim_results()
    if not sim_results:
        return {"error": "no sim data"}

    # Build envelope: for each horizon, get P5/P25/P50/P75/P95 of sim
    # Convert total_return to cumulative equity values from start_equity.
    points = [(0, start_equity, start_equity, start_equity, start_equity, start_equity)]
    for h in horizons:
        rec = _sim_record(sim_results, h)
        if not rec:
            continue
        points.append((
            h,
            start_equity * (1 + rec["tot_p5"]),
            start_equity * (1 + rec["tot_p25"]),
            start_equity * (1 + rec["tot_p50"]),
            start_equity * (1 + rec["tot_p75"]),
            start_equity * (1 + rec["tot_p95"]),
        ))

    eq_df = _load_equity()
    if not eq_df.empty:
        first_day = eq_df.index[0]
        live_days = (eq_df.index - first_day).days
        live_equity = eq_df["total_equity"].astype(float).values.tolist()
    else:
        first_day = pd.Timestamp.now(tz="UTC")
        live_days = []
        live_equity = []

    return {
        "envelope_days": [p[0] for p in points],
        "envelope_p5":   [p[1] for p in points],
        "envelope_p25":  [p[2] for p in points],
        "envelope_p50":  [p[3] for p in points],
        "envelope_p75":  [p[4] for p in points],
        "envelope_p95":  [p[5] for p in points],
        "live_days": list(live_days),
        "live_equity": live_equity,
        "first_day": str(first_day.date()),
    }


def forward_envelope_chart(start_equity: float = 100_000.0):
    """Plotly figure: live equity overlaid on sim envelope. Returns Figure or None."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    data = forward_envelope_data(start_equity)
    if "error" in data:
        return None

    fig = go.Figure()

    # P95 -> P5 fill (light)
    fig.add_trace(go.Scatter(
        x=data["envelope_days"], y=data["envelope_p95"],
        line=dict(width=0), name="P95",
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=data["envelope_days"], y=data["envelope_p5"],
        line=dict(width=0), name="P5-P95 range",
        fill="tonexty", fillcolor="rgba(100,150,200,0.15)",
    ))

    # P75 -> P25 fill (medium)
    fig.add_trace(go.Scatter(
        x=data["envelope_days"], y=data["envelope_p75"],
        line=dict(width=0), name="P75",
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=data["envelope_days"], y=data["envelope_p25"],
        line=dict(width=0), name="P25-P75 range",
        fill="tonexty", fillcolor="rgba(100,150,200,0.30)",
    ))

    # P50 line (median)
    fig.add_trace(go.Scatter(
        x=data["envelope_days"], y=data["envelope_p50"],
        mode="lines", line=dict(color="rgb(70,120,180)", dash="dot", width=2),
        name="Sim P50 (median)",
    ))

    # Live equity
    if data["live_days"]:
        fig.add_trace(go.Scatter(
            x=data["live_days"], y=data["live_equity"],
            mode="lines+markers",
            line=dict(color="rgb(220,80,30)", width=3),
            name="LIVE",
        ))

    fig.update_layout(
        title=("Forward Monte Carlo envelope vs live equity. "
               "Inside the band = normal. Below P5 = degradation signal."),
        xaxis_title=f"Days since {data['first_day']}",
        yaxis_title="Equity ($)",
        height=400,
        hovermode="x unified",
    )
    return fig


def current_sim_percentile_rank(start_equity: float = 100_000.0) -> dict:
    """Returns live percentile at each horizon, suitable for a metrics card."""
    data = forward_envelope_data(start_equity)
    if "error" in data:
        return {"error": data["error"]}
    eq_df = _load_equity()
    if eq_df.empty:
        return {"status": "no_equity_log"}

    n_days = (eq_df.index[-1] - eq_df.index[0]).days
    live_eq_now = float(eq_df["total_equity"].iloc[-1])
    live_return_now = live_eq_now / start_equity - 1

    sim_results = _latest_sim_results()
    out = {}
    for horizon in [30, 90, 365, 730]:
        if n_days < horizon:
            continue
        rec = _sim_record(sim_results, horizon)
        if not rec:
            continue
        pcts = [(5, rec["tot_p5"]), (25, rec["tot_p25"]), (50, rec["tot_p50"]),
                (75, rec["tot_p75"]), (95, rec["tot_p95"])]
        # locate live_return_now in this distribution
        if live_return_now <= pcts[0][1]:
            rank = 5 - (pcts[0][1] - live_return_now) * 100
            rank = max(0.1, rank)
        elif live_return_now >= pcts[-1][1]:
            rank = 95 + (live_return_now - pcts[-1][1]) * 100
            rank = min(99.9, rank)
        else:
            rank = 50.0
            for i in range(len(pcts) - 1):
                if pcts[i][1] <= live_return_now <= pcts[i + 1][1]:
                    p_lo, v_lo = pcts[i]
                    p_hi, v_hi = pcts[i + 1]
                    rank = p_lo + (p_hi - p_lo) * (live_return_now - v_lo) / max(v_hi - v_lo, 1e-9)
                    break
        out[horizon] = {
            "live_return": live_return_now,
            "percentile": rank,
            "sim_p5": rec["tot_p5"],
            "sim_p50": rec["tot_p50"],
            "sim_p95": rec["tot_p95"],
        }
    return out


if __name__ == "__main__":
    print("Forward envelope data:")
    print(json.dumps(forward_envelope_data(), indent=2, default=str)[:1500])
    print()
    print("Current percentile rank:")
    print(json.dumps(current_sim_percentile_rank(), indent=2, default=str))
