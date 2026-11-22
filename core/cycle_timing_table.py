"""Cross-cycle TIMING table: halving -> top -> bottom -> next halving, every cycle.

Schema 2 (2026-11-22, Dave: "add the timing table to the dashboard"). Everything
is on DAILY CLOSES (the old table mixed intraday tops with closes); it adds the
2013-15 cycle and a 'bottom -> next halving' column, shows cycle 5 twice ('lowest
close so far' and 'if the low is now'), and lists the bottom WINDOW each timing
rule implies with where today sits. Rendered on the Today and Research tabs by
btc_prediction_dashboard._render_cycle_timing.

Historical anchors were computed from daily closes on 2026-11-22 (Bitstamp before
Yahoo's 2014-09-17 start, Yahoo after); cycle 5 is recomputed live each refresh.
Output is PURE PYTHON so the pickled panel survives numpy/pandas drift on Cloud.
No "~" anywhere in the strings: a pair of them renders as strikethrough.
"""
from __future__ import annotations

import json
import urllib.request
from datetime import date, datetime, timedelta, timezone

# (label, halving, top date, top close, bottom date, bottom close, next halving)
HIST = [
    ("2013–15", "2012-11-28", "2013-12-04", 1132.01, "2015-01-14", 178.10, "2016-07-09"),
    ("2017–18", "2016-07-09", "2017-12-16", 19497.40, "2018-12-15", 3236.76, "2020-05-11"),
    ("2021–22", "2020-05-11", "2021-11-08", 67566.83, "2022-11-21", 15787.28, "2024-04-20"),
]
C5_HALVING = "2024-04-20"
NEXT_HALVING_BLOCK = 1_050_000
FALLBACK_HALVING_GAP_D = 1440       # 2020->2024 gap; only if the block-pace estimate fails


def _d(a, b) -> int:
    return (date.fromisoformat(str(b)[:10]) - date.fromisoformat(str(a)[:10])).days


def _fd(d) -> str:
    d = date.fromisoformat(str(d)[:10])
    return f"{d.day} {d:%b %Y}"


def _cell(d, px: float) -> str:
    return f"{_fd(d)} · ${px:,.0f}"


def _pct(x: float) -> str:
    return f"{round(x):+d}%".replace("-", "−").replace("+", "")


def _span(a: date, b: date) -> str:
    if a.year == b.year:
        return f"{a.day} {a:%b} – {b.day} {b:%b %Y}"
    return f"{a.day} {a:%b %Y} – {b.day} {b:%b %Y}"


def _status(a: date, b: date, today: date) -> str:
    if today < a:
        return f"starts in {(a - today).days} days"
    if today <= b:
        return f"open — {(b - today).days} days left"
    return f"ended {(today - b).days} days ago"


def _next_halving_estimate() -> tuple[date, str]:
    """Block 1,050,000 at the pace of the last 20,160 blocks (~140 days). Never raises."""
    def get(u):
        req = urllib.request.Request(u, headers={"User-Agent": "btcdelight-precompute"})
        return urllib.request.urlopen(req, timeout=15).read()
    try:
        tip = int(get("https://mempool.space/api/blocks/tip/height"))

        def ts(h):
            hsh = get(f"https://mempool.space/api/block-height/{h}").decode().strip()
            return int(json.loads(get(f"https://mempool.space/api/block/{hsh}"))["timestamp"])
        n = 20160
        t_tip, t_old = ts(tip), ts(tip - n)
        per = (t_tip - t_old) / n
        eta = datetime.fromtimestamp(t_tip + (NEXT_HALVING_BLOCK - tip) * per, timezone.utc).date()
        return eta, f"block {tip:,} at the last 20,160 blocks' pace of {per / 60:.2f} min per block"
    except Exception:
        eta = date.fromisoformat(C5_HALVING) + timedelta(days=FALLBACK_HALVING_GAP_D)
        return eta, (f"live block data unavailable, so assumed {FALLBACK_HALVING_GAP_D} days "
                     f"after the last halving")


def cycle_timing_table() -> dict:
    today = datetime.now(timezone.utc).date()
    rows, h2p, p2b, h2b, b2nh, dd = [], [], [], [], [], []
    for lab, hv, td, tp, bd, bp, nh in HIST:
        r = {"cycle": lab, "kind": "hist", "peak": _cell(td, tp), "bottom": _cell(bd, bp),
             "h2p": _d(hv, td), "p2b": _d(td, bd), "h2b": _d(hv, bd),
             "drawdown": _pct((bp / tp - 1) * 100), "b2nh": _d(bd, nh)}
        rows.append(r)
        h2p.append(r["h2p"]); p2b.append(r["p2b"]); h2b.append(r["h2b"]); b2nh.append(r["b2nh"])
        dd.append(round((bp / tp - 1) * 100))
    nxt, nxt_how = _next_halving_estimate()

    # Cycle 5, live from daily closes (Yahoo + exchange gap-fill, today's candle dropped).
    from core.bottom_status import _daily_closes
    closes, _filled = _daily_closes()
    h0 = date.fromisoformat(C5_HALVING)
    top_end = (h0 + timedelta(days=700)).isoformat()    # every cycle topped within ~550d
    win = [(d, c) for d, c in closes if C5_HALVING <= d <= top_end]
    if not win:
        raise RuntimeError("no cycle-5 daily closes")
    td, tp = max(win, key=lambda x: x[1])
    bd, bp = min(((d, c) for d, c in closes if d >= td), key=lambda x: x[1])
    last_d, last_c = closes[-1]
    try:
        from core import data
        spot = float(data.btc_spot()) or last_c
    except Exception:
        spot = last_c
    t0, b0 = date.fromisoformat(td), date.fromisoformat(bd)
    c5_h2p, p_low = _d(C5_HALVING, td), _d(td, bd)
    rows.append({"cycle": "2025–26 · lowest close so far", "kind": "low",
                 "peak": _cell(td, tp), "bottom": _cell(bd, bp),
                 "h2p": c5_h2p, "p2b": p_low, "h2b": _d(C5_HALVING, bd),
                 "drawdown": _pct((bp / tp - 1) * 100), "b2nh": f"≈{(nxt - b0).days} (est)"})
    if (today - b0).days > 1:
        rows.append({"cycle": "2025–26 · if the low is now", "kind": "now",
                     "peak": _cell(td, tp), "bottom": f"now · ${spot:,.0f}",
                     "h2p": c5_h2p, "p2b": (today - t0).days, "h2b": (today - h0).days,
                     "drawdown": _pct((spot / tp - 1) * 100), "b2nh": f"≈{(nxt - today).days} (est)"})

    # Bottom windows implied by each earlier-cycle timing rule.
    w1 = (t0 + timedelta(days=min(p2b)), t0 + timedelta(days=max(p2b)))
    w2 = (h0 + timedelta(days=min(h2b)), h0 + timedelta(days=max(h2b)))
    w2b = (h0 + timedelta(days=min(h2b[1:])), h0 + timedelta(days=max(h2b[1:])))
    w3 = (nxt - timedelta(days=max(b2nh)), nxt - timedelta(days=min(b2nh)))
    windows = [
        {"rule": f"Top + {min(p2b)}–{max(p2b)} days", "window": _span(*w1),
         "status": _status(*w1, today)},
        {"rule": f"Halving + {min(h2b)}–{max(h2b)} days",
         "window": f"{_span(*w2)} (last two cycles: {_span(*w2b)})", "status": _status(*w2, today)},
        {"rule": f"Bottom {min(b2nh)}–{max(b2nh)} days before the next halving (≈{_fd(nxt)})",
         "window": _span(*w3), "status": _status(*w3, today)},
    ]

    notes = []
    lo, hi = min(h2p[1:]), max(h2p[1:])
    if lo <= c5_h2p <= hi:
        notes.append(f"**The top was on schedule:** {c5_h2p} days after the halving, between the "
                     f"last two cycles ({lo} and {hi}).")
    else:
        notes.append(f"**The top came {c5_h2p} days after the halving** "
                     f"(earlier cycles: {', '.join(map(str, h2p))}).")
    if p_low < min(p2b):
        notes.append(f"**The lowest close so far ({_fd(bd)}) came early:** {p_low} days after the "
                     f"top, {min(p2b) - p_low}–{max(p2b) - p_low} days sooner than any earlier cycle.")
    elif p_low <= max(p2b):
        notes.append(f"**The lowest close so far ({_fd(bd)}) came {p_low} days after the top**, "
                     f"inside the earlier range ({min(p2b)}–{max(p2b)}).")
    else:
        notes.append(f"**The lowest close so far ({_fd(bd)}) came {p_low} days after the top**, "
                     f"later than any earlier cycle (longest: {max(p2b)}).")
    d_top, d_h = (today - t0).days, (today - h0).days
    msg = (f"**Today is {d_top} days after the top** (longest bear so far: {max(p2b)}) "
           f"**and {d_h} after the halving** (latest bottom so far: {max(h2b)}).")
    if d_top > max(p2b) and d_h > max(h2b):
        msg += (" Both are past every earlier cycle — if the low isn't in soon, this becomes the "
                "longest bear market on record.")
    notes.append(msg)
    notes.append(f"**Depth:** the lowest close is {_pct((bp / tp - 1) * 100)} from the top, versus "
                 f"{_pct(max(dd))} to {_pct(min(dd))} in earlier cycles.")

    proj = t0 + timedelta(days=round(sum(p2b[1:]) / 2))   # legacy keys for the old renderer
    return {"schema": 2, "rows": rows, "windows": windows, "notes": notes,
            "next_halving_est": nxt.isoformat(), "next_halving_method": nxt_how,
            "asof": today.isoformat(), "latest_close": [last_d, round(last_c, 2)],
            "projected_bottom_date": proj.isoformat(), "days_to_projected": (proj - today).days}


if __name__ == "__main__":
    r = cycle_timing_table()
    for row in r["rows"]:
        print(row)
    for w in r["windows"]:
        print(w)
    for n in r["notes"]:
        print("-", n)
    print("next halving:", r["next_halving_est"], "|", r["next_halving_method"])
