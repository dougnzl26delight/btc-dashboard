"""Theme-breadth scoring for the bottom-confirmation gate.

WHY: the 10 hard criteria are NOT independent — 3 are miner signals
(puell / hashrate_ribbon / hashrate_drawdown), 2 are cost-basis, 2 are demand-
flow. Counting them as 10 equal votes overstates confirmation: a "6/10" can be
just 3 independent themes echoing. For a single, hard-to-reverse rotation, what
de-risks the call is BREADTH ACROSS ORTHOGONAL THEMES, not a raw signal count.

This regroups the criteria into 6 orthogonal themes and scores by how many
themes confirm, with a MOMENTUM / price-turn theme MANDATORY for a full-deploy
call (don't deploy full size into a still-falling tape just because on-chain is
cheap). Each theme is tagged ETF-ROBUST (supply / miner / time — barely moved by
ETF demand) vs ETF-DISTORTED (demand / flow — heavily moved), so a muted /
ETF-mutated cycle is read through the lenses that still work.

Pure-additive: callers keep their existing n_met/n_total; this adds a `themes`
block + a breadth-based deploy recommendation alongside. Nothing is removed.
"""
from __future__ import annotations

# Injected pseudo-criteria the 10-criterion gate lacks natively, supplied by the
# caller: "price_turn" (a real weekly price reclaim, momentum theme) and
# "deriv_reset" (funding/OI reset, derivatives theme).
THEMES = [
    {"key": "cost_basis", "label": "Cost-basis cheapness", "etf": "robust",
     "members": ["realized_cap_drawdown", "mvrv_z", "price_drawdown"]},
    {"key": "miner", "label": "Miner capitulation", "etf": "robust",
     "members": ["puell_multiple", "hashrate_ribbon", "hashrate_drawdown"]},
    {"key": "time_cycle", "label": "Cycle-clock window", "etf": "robust",
     "members": ["halving_day"]},
    {"key": "momentum", "label": "Price-turn / momentum", "etf": "distorted",
     "members": ["sth_mvrv_reclaim", "price_turn"]},
    {"key": "demand_flows", "label": "Spot demand / flows", "etf": "distorted",
     "members": ["coinbase_premium", "cb_premium_streak"]},
    {"key": "derivatives", "label": "Derivatives reset", "etf": "distorted",
     "members": ["deriv_reset"]},
]
MOMENTUM_THEME = "momentum"   # mandatory for a FULL-deploy call


def theme_breadth(criteria: list, extra_met: dict | None = None) -> dict:
    """criteria: list of {id, met, ...} from the scorecard.
    extra_met: injected pseudo-criteria id -> bool (price_turn, deriv_reset).
    Returns themes + breadth + a momentum-mandatory deploy recommendation."""
    met_ids = {c["id"] for c in criteria if c.get("met")}
    avail_ids = {c["id"] for c in criteria}
    if extra_met:
        # only KNOWN (non-None) extras count; a None extra is a data gap, not a fail
        known = {k: v for k, v in extra_met.items() if v is not None}
        met_ids |= {k for k, v in known.items() if v}
        avail_ids |= set(known.keys())

    themes = []
    for t in THEMES:
        present = [m for m in t["members"] if m in avail_ids]
        hit = [m for m in t["members"] if m in met_ids]
        # a theme with NO present members is "unknown" (data gap), not "failed"
        status = "met" if hit else ("unknown" if not present else "not_met")
        themes.append({
            "key": t["key"], "label": t["label"], "etf": t["etf"],
            "met": bool(hit), "status": status,
            "n_members_met": len(hit), "members_met": hit,
        })

    themes_met = sum(1 for t in themes if t["met"])
    themes_total = len(THEMES)
    momentum_met = any(t["met"] for t in themes if t["key"] == MOMENTUM_THEME)
    time_met = any(t["met"] for t in themes if t["key"] == "time_cycle")
    robust_met = sum(1 for t in themes if t["met"] and t["etf"] == "robust")

    # Deploy by BREADTH gated on TIMING. Cheapness themes (cost-basis/miner/flows)
    # say "good price"; timing themes (momentum=price turned, time_cycle=in window)
    # say "the bottom is actually near". Cheap-but-not-timed = small tranche only;
    # a FULL deploy REQUIRES price to have turned (no knife-catching).
    if themes_met >= 4 and momentum_met:
        action, level = "FULL DEPLOY", "DEPLOY"
    elif themes_met >= 3 and (momentum_met or time_met):
        action, level = "SCALE IN (50-75%)", "SCALE_IN"
    elif themes_met >= 3:
        action, level = "EARLY-PLUS — broad but NO timing signal; small tranche only", "EARLY"
    elif themes_met >= 2:
        action, level = "EARLY — first small tranche only", "EARLY"
    else:
        action, level = "WAIT — breadth insufficient", "WAIT"

    return {
        "themes": themes,
        "themes_met": themes_met,
        "themes_total": themes_total,
        "momentum_met": momentum_met,
        "robust_themes_met": robust_met,
        "deploy_action": action,
        "deploy_level": level,
        "summary": (f"{themes_met}/{themes_total} themes ({robust_met} ETF-robust)"
                    + (" · price turned" if momentum_met else " · NO price-turn yet")),
    }


def rolling_percentile_rank(series, lookback: int = 730):
    """Percentile (0..1) of the latest value within the last `lookback` points.
    Converts absolute thresholds into muted-cycle-robust reads: e.g. Puell 0.62
    can still flag if 0.62 is its lowest-decile value of the last 2 years, even
    though the old hard '<0.50' never trips in a shallow ETF-era cycle."""
    try:
        import numpy as np
        s = [x for x in list(series)[-lookback:] if x is not None]
        arr = np.asarray(s, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) < 30:
            return None
        return float((arr <= arr[-1]).sum() / len(arr))
    except Exception:
        return None


if __name__ == "__main__":
    demo = [
        {"id": "realized_cap_drawdown", "met": True},
        {"id": "mvrv_z", "met": False},
        {"id": "price_drawdown", "met": True},
        {"id": "puell_multiple", "met": False},
        {"id": "hashrate_ribbon", "met": False},
        {"id": "hashrate_drawdown", "met": False},
        {"id": "halving_day", "met": False},
        {"id": "sth_mvrv_reclaim", "met": False},
        {"id": "coinbase_premium", "met": True},
        {"id": "cb_premium_streak", "met": False},
    ]
    r = theme_breadth(demo, extra_met={"price_turn": False, "deriv_reset": None})
    print(r["summary"], "->", r["deploy_action"])
    for t in r["themes"]:
        print(f"  [{'X' if t['met'] else ' '}] {t['label']:<26} {t['etf']:<9} {t['status']}")
