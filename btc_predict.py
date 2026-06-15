"""BTC PREDICTION MACHINE — root-level CLI launcher.

Usage:
    python btc_predict.py              # run prediction (uses 4h cache)
    python btc_predict.py --force      # force refresh all signals
    python btc_predict.py --json       # output JSON instead of text
    python btc_predict.py --brief      # short summary only
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.btc_prediction import state_of_btc, main as full_report


def brief_summary(s: dict) -> str:
    """One-paragraph summary."""
    px = s["btc_price"]
    regime = s["regime"]
    short = s["horizons"]["short_term"]
    medium = s["horizons"]["medium_term"]
    long_ = s["horizons"]["long_term"]
    tgt_30 = s["price_targets"].get("short_term", {})
    tgt_90 = s["price_targets"].get("medium_term", {})

    return (
        f"BTC ${px:,.0f}  |  REGIME: {regime}\n"
        f"Short (1-30d):  {short['interpretation']} (score {short['direction_score']:+.2f}, {short['confidence']} confidence)\n"
        f"Medium (1-6m):  {medium['interpretation']} (score {medium['direction_score']:+.2f}, {medium['confidence']} confidence)\n"
        f"Long (6m-2y):   {long_['interpretation']} (score {long_['direction_score']:+.2f}, {long_['confidence']} confidence)\n"
        f"\n30-day range:   ${tgt_30.get('p25', 0):,.0f} - ${tgt_30.get('p75', 0):,.0f}  median ${tgt_30.get('median', 0):,.0f}\n"
        f"90-day range:   ${tgt_90.get('p25', 0):,.0f} - ${tgt_90.get('p75', 0):,.0f}  median ${tgt_90.get('median', 0):,.0f}"
    )


def cli():
    args = sys.argv[1:]
    force = "--force" in args
    as_json = "--json" in args
    brief = "--brief" in args

    if as_json:
        s = state_of_btc(force=force)
        # Trim signals to keep JSON readable
        s_short = {k: v for k, v in s.items() if k != "signals"}
        s_short["signals_summary"] = {
            cat: {"category_score":
                  sum(d.get("score", 0) for d in cs.values()
                      if isinstance(d, dict) and d.get("score") is not None) /
                  max(1, sum(1 for d in cs.values()
                              if isinstance(d, dict) and d.get("score") is not None))}
            for cat, cs in s["signals"].items()
            if isinstance(cs, dict) and not cs.get("error")
        }
        print(json.dumps(s_short, indent=2, default=str))
        return

    if brief:
        s = state_of_btc(force=force)
        print()
        print(brief_summary(s))
        return

    # Full text report
    full_report()


if __name__ == "__main__":
    cli()
