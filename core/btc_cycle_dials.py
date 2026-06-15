"""Cycle Dials — the Swift cycle indicators as at-a-glance gauges.

The Charts tab had 8 time-series charts that each need interpretation
(where's the danger zone, where are we now, is that bull or bear?).
A gauge answers all three instantly: colored zones + a needle.

Each dial: GREEN = bottom/buy zone, YELLOW = neutral, RED = top/sell zone.
A headline counts how many dials sit in buy vs sell zones — the single
fastest read of "where are we in the cycle".

Values are pulled from already-computed caches (swift_indicators,
swift_watch, native scorecards) — no new data fetching.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BG = "#0e1117"
TEXT = "#d4d4d4"
MUTED = "#888"
GREEN = "rgba(34, 197, 94, 0.55)"
GREEN_LITE = "rgba(34, 197, 94, 0.30)"
YELLOW = "rgba(240, 185, 11, 0.40)"
ORANGE = "rgba(249, 115, 22, 0.45)"
RED = "rgba(239, 68, 68, 0.55)"


def _gauge(value, vmin, vmax, zones, title, sub, suffix="", number_fmt=".2f"):
    """Build one Plotly gauge.

    zones: list of (start, end, color) translucent bands across the axis.
    """
    if value is None:
        return {
            "data": [], "layout": {
                "title": {"text": f"<b>{title}</b><br><span style='font-size:10px;color:#888'>"
                          f"data unavailable</span>", "font": {"color": TEXT, "size": 12}},
                "paper_bgcolor": BG, "height": 200,
                "xaxis": {"visible": False}, "yaxis": {"visible": False},
            }
        }
    v = max(vmin, min(vmax, float(value)))
    return {
        "data": [{
            "type": "indicator", "mode": "gauge+number",
            "value": round(float(value), 3),
            "number": {"suffix": suffix, "font": {"size": 26, "color": "#fff"},
                       "valueformat": number_fmt},
            "title": {"text": f"<b>{title}</b><br><span style='font-size:10px;color:#888'>{sub}</span>",
                      "font": {"size": 12, "color": TEXT}},
            "gauge": {
                "axis": {"range": [vmin, vmax], "tickwidth": 1, "tickcolor": MUTED,
                         "tickfont": {"size": 8, "color": MUTED}},
                "bar": {"color": "rgba(255,255,255,0.9)", "thickness": 0.22},
                "bgcolor": BG, "borderwidth": 0,
                "steps": [{"range": [z[0], z[1]], "color": z[2]} for z in zones],
                "threshold": {"line": {"color": "#fff", "width": 3},
                              "thickness": 0.9, "value": v},
            },
        }],
        "layout": {
            "paper_bgcolor": BG, "plot_bgcolor": BG,
            "margin": {"l": 14, "r": 14, "t": 40, "b": 6},
            "height": 200, "font": {"color": TEXT, "family": "Inter, sans-serif"},
        },
    }


def _classify(value, buy_below, sell_above):
    """BUY (cheap), NEUTRAL, or SELL (expensive) by thresholds."""
    if value is None:
        return "?"
    v = float(value)
    if v <= buy_below:
        return "BUY"
    if v >= sell_above:
        return "SELL"
    return "NEUTRAL"


def all_cycle_dials() -> dict:
    """Build the full dial board + an at-a-glance summary."""
    from core.dashboard_cache import get_cached

    si = get_cached("swift_indicators") or {}
    sw = get_cached("swift_watch") or {}
    nb = get_cached("btc_native_bottom_scorecard") or {}

    gr = (si.get("golden_ratio_multiplier") or {}).get("multiplier")
    tym = (si.get("two_year_ma_multiplier") or {}).get("multiplier")
    nupl = (si.get("lth_nupl") or {}).get("nupl")
    mvrv = (si.get("lth_nupl") or {}).get("mvrv")
    logdev = (si.get("log_regression") or {}).get("deviation_pct")
    risk = (sw.get("risk_index") or {}).get("risk_index")

    # Mayer from native bottom scorecard status text ("Mayer Multiple 0.80 ...")
    mayer = None
    for c in nb.get("criteria", []):
        if "Mayer Multiple" in (c.get("label") or ""):
            v = c.get("value")
            if isinstance(v, (int, float)):
                mayer = v
            else:
                import re
                m = re.search(r"Mayer Multiple\s+([0-9.]+)", c.get("status", "") or "")
                if m:
                    mayer = float(m.group(1))
            break

    dials = {
        "mayer": {
            "fig": _gauge(mayer, 0, 3,
                          [(0, 1, GREEN), (1, 1.5, GREEN_LITE), (1.5, 2.4, YELLOW), (2.4, 3, RED)],
                          "Mayer Multiple", "price / 200d MA · <0.6 deep value, >2.4 top", suffix="×"),
            "zone": _classify(mayer, 1.0, 2.4),
        },
        "mvrv": {
            "fig": _gauge(mvrv, 0, 6,
                          [(0, 1, GREEN), (1, 2, GREEN_LITE), (2, 3.7, YELLOW), (3.7, 5, ORANGE), (5, 6, RED)],
                          "MVRV Ratio", "market / realized value · <1 capitulation, >5 euphoria", suffix="×"),
            "zone": _classify(mvrv, 1.0, 3.7),
        },
        "golden_ratio": {
            "fig": _gauge(gr, 0, 6,
                          [(0, 1.6, GREEN), (1.6, 3, YELLOW), (3, 5, ORANGE), (5, 6, RED)],
                          "Golden Ratio Mult", "price / 350d MA · <1.6 accumulate, >5 cycle top", suffix="×"),
            "zone": _classify(gr, 1.6, 5.0),
        },
        "two_year_ma": {
            "fig": _gauge(tym, 0, 6,
                          [(0, 1, GREEN), (1, 3, YELLOW), (3, 5, ORANGE), (5, 6, RED)],
                          "2-Year MA Mult", "price / 2y MA · <1 bottom band, >5 top band", suffix="×"),
            "zone": _classify(tym, 1.0, 5.0),
        },
        "nupl": {
            "fig": _gauge(nupl, -0.5, 1.0,
                          [(-0.5, 0, GREEN), (0, 0.25, GREEN_LITE), (0.25, 0.5, YELLOW),
                           (0.5, 0.75, ORANGE), (0.75, 1.0, RED)],
                          "NUPL", "net unrealized P/L · <0 capitulation, >0.75 euphoria", number_fmt=".2f"),
            "zone": _classify(nupl, 0.0, 0.75),
        },
        "log_regression": {
            "fig": _gauge(logdev, -80, 80,
                          [(-80, -40, GREEN), (-40, -10, GREEN_LITE), (-10, 20, YELLOW),
                           (20, 50, ORANGE), (50, 80, RED)],
                          "Log Regression", "% from fair-value model · deep- to over-valued", suffix="%", number_fmt=".0f"),
            "zone": _classify(logdev, -10, 50),
        },
        "risk_index": {
            "fig": _gauge(risk, 0, 1,
                          [(0, 0.25, GREEN), (0.25, 0.4, GREEN_LITE), (0.4, 0.6, YELLOW),
                           (0.6, 0.8, ORANGE), (0.8, 1.0, RED)],
                          "Swift Risk Index", "composite 0=max buy → 1=max sell", number_fmt=".2f"),
            "zone": _classify(risk, 0.4, 0.8),
        },
    }

    zones = [d["zone"] for d in dials.values() if d["zone"] != "?"]
    n_buy = zones.count("BUY")
    n_sell = zones.count("SELL")
    n_neutral = zones.count("NEUTRAL")
    n_total = len(zones)

    if n_buy >= n_sell and n_buy >= 3:
        headline = "ACCUMULATION ZONE"
        head_color = "#22c55e"
    elif n_sell >= 3:
        headline = "DISTRIBUTION ZONE"
        head_color = "#ef4444"
    else:
        headline = "NEUTRAL / MID-CYCLE"
        head_color = "#f0b90b"

    return {
        "dials": dials,
        "summary": {
            "headline": headline,
            "head_color": head_color,
            "n_buy": n_buy, "n_sell": n_sell, "n_neutral": n_neutral, "n_total": n_total,
        },
    }


def main():
    r = all_cycle_dials()
    s = r["summary"]
    print(f"HEADLINE: {s['headline']}  ({s['n_buy']} buy / {s['n_neutral']} neutral / {s['n_sell']} sell)")
    for k, d in r["dials"].items():
        print(f"  {k:16s} zone={d['zone']}")


if __name__ == "__main__":
    main()
