"""Regime transition email alert — fires when macro regime shifts
between RISK_ON, LATE_CYCLE, RECESSIONARY_BEAR.

Runs via Crypto_regime_check Windows scheduled task (daily).
Watches for transitions; sends critical-level email on shift.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = REPO_ROOT / ".btc_regime_state.json"


def _load_state() -> dict:
    if not STATE_FILE.exists():
        return {"last_regime": None, "last_check": None,
                "last_buckets": None, "last_vetoes": []}
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {"last_regime": None, "last_check": None,
                "last_buckets": None, "last_vetoes": []}


def _save_state(s: dict) -> None:
    try: STATE_FILE.write_text(json.dumps(s, indent=2))
    except Exception: pass


def check_regime(send_email: bool = True) -> dict:
    """Check if regime changed since last run. Alert on transition or
    new veto activation."""
    from core.btc_unified_decision import unified_decision

    r = unified_decision(current_equity_pct=70, total_stake_nzd=130_000)
    current_regime = r["regime"]
    current_vetoes = sorted(r["vetoes_active"])

    state = _load_state()
    last_regime = state.get("last_regime")
    last_vetoes = sorted(state.get("last_vetoes") or [])

    state["last_check"] = datetime.now(timezone.utc).isoformat()
    state["last_regime"] = current_regime
    state["last_buckets"] = r["regime_buckets"]
    state["last_vetoes"] = current_vetoes
    _save_state(state)

    regime_changed = last_regime is not None and last_regime != current_regime
    new_vetoes = sorted(set(current_vetoes) - set(last_vetoes))

    if not (regime_changed or new_vetoes):
        return {"regime": current_regime, "changed": False,
                "alert_sent": False,
                "message": f"No change. Still in {current_regime}, "
                            f"vetoes: {current_vetoes or 'none'}"}

    # Build alert
    if regime_changed:
        subject = (f"!! REGIME SHIFT: {last_regime or 'first run'} -> "
                    f"{current_regime} !!")
    else:
        subject = f"!! NEW VETO ACTIVE: {','.join(new_vetoes)} ({current_regime}) !!"

    t = r["target_allocation_pct"]
    n = r["target_allocation_nzd"]
    sb = r["staging_basket_pct"]
    sn = r["staging_basket_nzd"]
    sc = r["scorecards"]
    b = r["regime_buckets"]

    body = f"""MACRO REGIME UPDATE

Regime shift:    {last_regime or 'first observation'} -> {current_regime}
New vetoes:      {new_vetoes or 'none'}
Active vetoes:   {current_vetoes or 'none'}

================================================================
REGIME BUCKETS
================================================================
  Growth deterioration:   {b['growth']}/4
  Plumbing stress:        {b['plumbing']}/4
  Credit cycle:           {b['credit']}/3
  Yield curve un-invert:  {r['regime_curve_uninvert']}
  Liquidity z-score:      {r['liquidity']['z']:+.2f}

================================================================
SCORECARDS (regime-modulated thresholds)
================================================================
  Top Confirmation:    {sc['top']['n_met']}/{sc['top']['n_total']}    -> {sc['top']['action']}
  Early Rotation:      {sc['early']['n_firing']}/{sc['early']['n_total']}    -> {sc['early']['action']}
  BTC Bottom:          {sc['bottom']['n_met']}/{sc['bottom']['n_total']}

================================================================
TARGET ALLOCATION (NZ${n['btc'] + n['equity'] + n['staging']:,} stake)
================================================================
  Equity:    {t['equity']:.1f}%  -> NZ${n['equity']:,}
  BTC:       {t['btc']:.1f}%  -> NZ${n['btc']:,}
  Staging:   {t['staging']:.1f}%  -> NZ${n['staging']:,}
    BIL:        {sb.get('BIL',0)}%  -> NZ${sn.get('BIL',0):,}
    VTIP:       {sb.get('VTIP',0)}%  -> NZ${sn.get('VTIP',0):,}
    GLDM:       {sb.get('GLDM',0)}%  -> NZ${sn.get('GLDM',0):,}
  Rotation now: NZ${r['rotation_nzd']:,}

================================================================
ACTION
================================================================
  {sc['top']['action']} on equity side. {sc['early']['action']} on early signal.

  Rationale: {r['staging_basket_rationale']}

================================================================
NEXT STEPS
================================================================
Dashboard: http://localhost:8511 (Overview tab -> Unified Decision)
Re-check:  python -m core.btc_regime_alert
"""

    if send_email:
        try:
            from ops.alerts import alert
            alert(body, level="critical", subject=subject)
            return {"regime": current_regime, "changed": True,
                    "alert_sent": True, "subject": subject}
        except Exception as e:
            return {"regime": current_regime, "changed": True,
                    "alert_sent": False, "error": str(e)}
    return {"regime": current_regime, "changed": True,
            "alert_sent": False, "preview": body}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--no-email", action="store_true")
    args = p.parse_args()
    r = check_regime(send_email=not args.no_email)
    print(json.dumps(r, indent=2, default=str))


if __name__ == "__main__":
    main()
