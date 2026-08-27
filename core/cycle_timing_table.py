"""Cross-cycle bottom TIMING + DEPTH table.

Shows halving->peak->bottom day counts and max drawdown for each cycle, so the
"the bottom window is still ~weeks ahead, and this cycle is half as deep"
picture stays live. Current-cycle row + the projection update each refresh.

Output is PURE PYTHON ({'rows':[...], 'asof':...}) so the pickled panel is
immune to numpy/pandas version drift on the cloud (same rule as
cycle_peak_table). Rendered as a markdown table in btc_prediction_dashboard.py.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from core import data

# Fixed historical anchors (verified from our own price history).
HALV = {"C3": "2016-07-09", "C4": "2020-05-11", "C5": "2024-04-20"}
PEAK = {"C3": ("2017-12-17", 19783), "C4": ("2021-11-10", 69000),
        "C5": ("2025-10-06", 126198)}
BOT = {"C3": ("2018-12-15", 3212), "C4": ("2022-11-21", 15781)}  # C5 not in yet


def _d(a: str, b: str) -> int:
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


def _fmtk(v) -> str:
    try:
        v = float(v)
    except Exception:
        return "—"
    return f"${v / 1000:.1f}k" if v >= 1000 else f"${v:,.0f}"


def cycle_timing_table() -> dict:
    today = datetime.now(timezone.utc).date().isoformat()
    rows = []
    for c in ("C3", "C4"):
        h = HALV[c]
        pd_, pp = PEAK[c]
        bd, bp = BOT[c]
        rows.append({
            "cycle": c.replace("C", "Cycle "),
            "peak": _fmtk(pp), "h2p": _d(h, pd_), "p2b": _d(pd_, bd),
            "h2b": _d(h, bd), "bottom": _fmtk(bp),
            "drawdown": f"{round((bp / pp - 1) * 100)}%",
        })

    # Current cycle (live).
    h = HALV["C5"]
    pd_, pp = PEAK["C5"]
    cur = lo_p = None
    try:
        df = data.ohlcv_extended("BTC/USDT", days_back=400, timeframe="1d")
        idx = [str(x)[:10] for x in df.index]
        cl = [float(x) for x in df["close"].tolist()]
        cur = cl[-1]
        since = [(idx[i], cl[i]) for i in range(len(idx)) if idx[i] >= pd_]
        if since:
            _, lo_p = min(since, key=lambda t: t[1])
    except Exception:
        pass
    cur_dd = round((cur / pp - 1) * 100) if cur else None
    max_dd = round((lo_p / pp - 1) * 100) if lo_p else None
    rows.append({
        "cycle": "**Cycle 5 (now)**", "peak": _fmtk(pp), "h2p": _d(h, pd_),
        "p2b": f"now +{_d(pd_, today)}d", "h2b": f"now +{_d(h, today)}d",
        "bottom": _fmtk(cur) if cur else "—",
        "drawdown": (f"{max_dd}% low / {cur_dd}% now" if cur else "—"),
    })

    # Projection (average of the two prior cycles).
    ap2b = round((rows[0]["p2b"] + rows[1]["p2b"]) / 2)
    ah2b = round((rows[0]["h2b"] + rows[1]["h2b"]) / 2)
    proj_date = (date.fromisoformat(pd_) + timedelta(days=ap2b)).isoformat()
    rows.append({
        "cycle": "Cycle 5 *projected*", "peak": "—", "h2p": "—",
        "p2b": f"~+{ap2b}d", "h2b": f"~+{ah2b}d",
        "bottom": "~$68k (analog)", "drawdown": "~-45% (analog)",
    })

    return {"rows": rows, "asof": today,
            "projected_bottom_date": proj_date,
            "days_to_projected": _d(today, proj_date)}


if __name__ == "__main__":
    r = cycle_timing_table()
    for row in r["rows"]:
        print(row)
    print("projected bottom:", r["projected_bottom_date"],
          "| days away:", r["days_to_projected"])
