"""Past cycle bottoms overlay — the recurring 'bounce -> lower low -> bull'
pattern at BTC major lows, with the current cycle overlaid.

Each episode is indexed to its first major low (=100) and plotted by days since
that low, so a dip BELOW 100 = a lower low made AFTER the initial bounce. The
current cycle (gold) shows where we sit versus history.

Returns a Plotly figure DICT (dashboard convention; see core/btc_swift_dials.py).
Cached as panel 'past_bottoms' by precompute_dashboard.py; rendered with
st.plotly_chart in btc_prediction_dashboard.py.
"""
from __future__ import annotations

from datetime import date

from core import data

BG = "#0e1117"
TEXT = "#d4d4d4"
MUTED = "#888"

# label, anchor-window start, anchor-window end, forward days, colour, is_current
EPISODES = [
    ("2018 bottom",    "2018-10-01", "2018-11-14", 260, "#4ea1ff", False),
    ("2021 mid-cycle", "2021-05-10", "2021-05-25", 130, "#a78bfa", False),
    ("2022 bottom",    "2022-06-10", "2022-06-30", 300, "#f472b6", False),
    ("2026 (now)",     "2026-06-10", "2026-07-10", 400, "#f0b90b", True),
]


def _empty(msg: str) -> dict:
    return {"data": [], "layout": {
        "title": {"text": "Past cycle bottoms", "font": {"color": TEXT}},
        "annotations": [{"text": msg, "showarrow": False,
                         "font": {"color": MUTED, "size": 13}, "x": 0.5, "y": 0.5}],
        "paper_bgcolor": BG, "plot_bgcolor": BG,
        "xaxis": {"visible": False}, "yaxis": {"visible": False}}}


def past_bottoms_chart() -> dict:
    try:
        df = data.ohlcv_extended("BTC/USDT", days_back=3500, timeframe="1d")
        idx = [str(x)[:10] for x in df.index]
        close = [float(c) for c in df["close"].tolist()]
    except Exception as e:
        return _empty(f"data unavailable: {e}")
    if len(idx) < 200:
        return _empty("insufficient history")

    traces, annos = [], []
    for label, a, b, fwd, color, cur in EPISODES:
        win = [(i, idx[i], close[i]) for i in range(len(idx)) if a <= idx[i] <= b]
        if not win:
            continue
        ai, ad, ap = min(win, key=lambda t: t[2])   # anchor = the first major low
        if ap <= 0:
            continue
        ad_d = date.fromisoformat(ad)
        xs, ys = [], []
        low_x, low_y = 0, 100.0
        for j in range(ai, min(ai + fwd, len(idx))):
            dnum = (date.fromisoformat(idx[j]) - ad_d).days
            v = close[j] / ap * 100.0
            xs.append(dnum)
            ys.append(round(v, 1))
            if v < low_y:
                low_y, low_x = v, dnum
        traces.append({
            "type": "scatter", "mode": "lines", "name": label,
            "x": xs, "y": ys,
            "line": {"color": color, "width": 4 if cur else 2},
            "opacity": 1.0 if cur else 0.85,
            "hovertemplate": f"<b>{label}</b><br>day %{{x}}<br>%{{y:.0f}} "
                             f"(first low = 100)<extra></extra>",
        })
        # annotate a genuine lower low (dipped below the first low) on historical lines
        if low_y < 99 and not cur:
            annos.append({"x": low_x, "y": low_y,
                          "text": f"{label}: lower low {low_y - 100:+.0f}%",
                          "showarrow": True, "arrowhead": 2, "arrowcolor": color,
                          "font": {"color": color, "size": 10}, "ay": 28, "ax": 0})
    if not traces:
        return _empty("no episode data")

    # where the current line currently sits (last point)
    cur_tr = next((t for t in traces if t["name"].startswith("2026")), None)
    if cur_tr and cur_tr["y"]:
        annos.append({"x": cur_tr["x"][-1], "y": cur_tr["y"][-1],
                      "text": f"now {cur_tr['y'][-1] - 100:+.0f}% (no lower low yet)",
                      "showarrow": True, "arrowhead": 2, "arrowcolor": "#f0b90b",
                      "font": {"color": "#f0b90b", "size": 10}, "ay": -25, "ax": 0})

    return {"data": traces, "layout": {
        "title": {"text": "<b>Past cycle bottoms: bounce &#8594; lower low &#8594; bull</b><br>"
                          "<span style='font-size:11px;color:#888'>each indexed to its first "
                          "major low (=100); a dip below 100 = a lower low after the bounce</span>",
                  "font": {"color": TEXT, "size": 14}},
        "paper_bgcolor": BG, "plot_bgcolor": BG,
        "margin": {"l": 55, "r": 20, "t": 72, "b": 46}, "height": 450,
        "font": {"color": TEXT, "family": "Inter, sans-serif"},
        "xaxis": {"title": "Days since first major low", "gridcolor": "#222",
                  "color": TEXT, "tickfont": {"size": 10}, "range": [-10, 310]},
        "yaxis": {"title": "Price (first major low = 100)", "gridcolor": "#222",
                  "color": TEXT, "tickfont": {"size": 10}},
        "legend": {"x": 0.02, "y": 0.02, "font": {"size": 11, "color": TEXT},
                   "bgcolor": "rgba(14,17,23,0.7)"},
        "shapes": [{"type": "line", "x0": 0, "x1": 1, "xref": "paper",
                    "y0": 100, "y1": 100, "yref": "y",
                    "line": {"color": "#666", "width": 1, "dash": "dash"}}],
        "annotations": annos, "hovermode": "closest",
    }}


if __name__ == "__main__":
    f = past_bottoms_chart()
    print("traces:", len(f["data"]))
    for t in f["data"]:
        print(f"  {t['name']:<15} pts={len(t['x'])} start={t['y'][0] if t['y'] else '?'} "
              f"min={min(t['y']) if t['y'] else '?'} last={t['y'][-1] if t['y'] else '?'}")
    print("annotations:", [a["text"] for a in f["layout"]["annotations"]])
