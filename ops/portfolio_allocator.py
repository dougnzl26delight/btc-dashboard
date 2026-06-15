"""Portfolio allocator — HRP-driven sleeve weights with monthly rebalance cadence.

Lopez de Prado HRP (Hierarchical Risk Parity, AFML Ch 16) replaces hardcoded
sleeve allocations once the rig has accumulated enough live return data.

Pre-Day-30:
    Sleeve allocations are HARDCODED in sub-account funding
    (BAH 10%, XSMOM 30%, oversold_bounce 15%, etc.)

Day 30+:
    This module computes HRP weights from the live sleeve return series and
    persists them to .hrp_weights.json. Sleeve runners can multiply their
    baseline allocation by hrp_weight(sleeve) / baseline_weight to lean into
    sleeves with stronger / more diversifying live edges.

Cadence: monthly recompute (1st of each month) — HRP is stable to small
return changes; weekly recompute is wasted compute. Daily snapshot for
dashboard rendering is cheap and live-updating.

Read by: dashboard.py (display), sleeve runners (optional weight scaler).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

WEIGHTS_FILE = REPO_ROOT / ".hrp_weights.json"

# Minimum days of returns per sleeve before HRP can compute. Below this we
# return hardcoded baselines and don't write the HRP file.
MIN_DAYS_PER_SLEEVE = 14

# Hardcoded baseline weights — what the rig uses pre-Day-30. These match the
# sub-account funding plan (200k bankroll split into spot 100k + perp 100k).
BASELINE_WEIGHTS = {
    "bah_btc":            0.10,
    "oversold_bounce":    0.15,
    "intraday_momentum":  0.10,
    "grid_trader":        0.10,
    "consolidation_breakout": 0.10,
    "xsmom":              0.30,
    "pro_trend":          0.30,
    "overbought_fade":    0.10,
    "basis_arb":          0.20,
}


def compute_and_persist() -> dict:
    """Compute HRP weights from live sleeve returns + persist.

    Returns dict with weights + meta. Persists to .hrp_weights.json so the
    dashboard and sleeve runners can read without re-running the computation.
    """
    try:
        from core.hrp_allocation import compute_sleeve_hrp
    except Exception as e:
        return {"error": f"hrp_allocation import failed: {e}"}

    result = compute_sleeve_hrp(days=60)
    if result.get("error"):
        # Insufficient data — write a "waiting" marker and fall back to baseline
        meta = {
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "status": "waiting_for_data",
            "min_days_required": MIN_DAYS_PER_SLEEVE,
            "weights": BASELINE_WEIGHTS,
            "weight_source": "hardcoded_baseline",
        }
        WEIGHTS_FILE.write_text(json.dumps(meta, indent=2, default=str))
        return meta

    weights = result["weights"]
    meta = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "n_sleeves": result["n_sleeves"],
        "n_observations": result["n_observations"],
        "weights": weights,
        "weight_source": "hrp",
        "verdict": result.get("verdict"),
    }
    WEIGHTS_FILE.write_text(json.dumps(meta, indent=2, default=str))
    return meta


def read_weights() -> dict:
    """Return current weights dict — HRP if available, baseline otherwise.

    Safe to call from sleeve runners; never raises.
    """
    if WEIGHTS_FILE.exists():
        try:
            d = json.loads(WEIGHTS_FILE.read_text())
            return d.get("weights", BASELINE_WEIGHTS)
        except Exception:
            return BASELINE_WEIGHTS
    return BASELINE_WEIGHTS


def hrp_scale_for(sleeve: str) -> float:
    """Return HRP weight / baseline weight as a multiplier in [0.0, 3.0].

    Sleeve runners can multiply their baseline allocation by this to lean
    into sleeves HRP prefers. Returns 1.0 (no effect) when HRP is still
    in warm-up or sleeve isn't in the weights map.
    """
    weights = read_weights()
    hrp_w = weights.get(sleeve)
    baseline_w = BASELINE_WEIGHTS.get(sleeve)
    if hrp_w is None or baseline_w is None or baseline_w <= 0:
        return 1.0
    return max(0.0, min(3.0, hrp_w / baseline_w))


def status() -> dict:
    """Dashboard-friendly snapshot."""
    if WEIGHTS_FILE.exists():
        try:
            d = json.loads(WEIGHTS_FILE.read_text())
            d["file_age_hours"] = (
                datetime.now(timezone.utc)
                - datetime.fromisoformat(d.get("computed_at", "2020-01-01T00:00:00+00:00"))
            ).total_seconds() / 3600
            return d
        except Exception:
            pass
    return {
        "status": "no_file_yet",
        "weights": BASELINE_WEIGHTS,
        "weight_source": "hardcoded_baseline",
    }


def main():
    """CLI: compute + display."""
    print("=" * 70)
    print("PORTFOLIO ALLOCATOR — HRP sleeve weights (Lopez de Prado AFML 16)")
    print("=" * 70)
    r = compute_and_persist()
    if r.get("status") == "waiting_for_data":
        print()
        print(f"  Status: WAITING — need >= {MIN_DAYS_PER_SLEEVE} days per sleeve")
        print(f"  Currently using BASELINE weights (hardcoded):")
        for s, w in sorted(BASELINE_WEIGHTS.items(), key=lambda x: -x[1]):
            print(f"    {s:<24s} {w*100:>5.1f}%")
        return

    print()
    print(f"  HRP weights computed from {r['n_observations']} obs across {r['n_sleeves']} sleeves")
    print(f"  Verdict: {r.get('verdict')}")
    print()
    print(f"  {'Sleeve':<24s} {'HRP %':>8s}  {'Baseline %':>11s}  {'Scale':>7s}")
    for sleeve, hw in sorted(r["weights"].items(), key=lambda x: -x[1]):
        bw = BASELINE_WEIGHTS.get(sleeve, 0.0)
        scale = hw / bw if bw > 0 else float("nan")
        scale_str = f"{scale:.2f}x" if bw > 0 else "n/a"
        print(f"  {sleeve:<24s} {hw*100:>7.2f}%  {bw*100:>10.2f}%  {scale_str:>7s}")


if __name__ == "__main__":
    main()
