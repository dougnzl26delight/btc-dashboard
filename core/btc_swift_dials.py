"""Swift Dials — high-impact visual indicators for the BTC dashboard.

Per Swift's gap analysis: 5 indicators that should be on first screen of
Overview but weren't. All visuals — meant to be glanced at, not read.

  1. Halving Clock          — donut chart of cycle progression
  2. BTC Dominance gauge    — 0-100% with regime zones
  3. S2F Deflection gauge   — price / S2F model fair value
  4. Open Interest gauge    — froth vs capitulation
  5. Cycle 4 vs 5 overlay   — normalized price comparison

All return Plotly figure dicts. Compose with st.plotly_chart(fig, use_container_width=True).
"""
from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 3-color palette (matches dashboard)
GREEN  = "#22c55e"
YELLOW = "#f0b90b"
RED    = "#ef4444"
BG     = "#0e1117"
TEXT   = "#d4d4d4"
MUTED  = "#888"


def _empty_fig(title: str, msg: str = "data unavailable") -> dict:
    """Fallback figure when upstream data fails."""
    return {
        "data": [], "layout": {
            "title": {"text": title, "font": {"color": TEXT}},
            "annotations": [{"text": msg, "showarrow": False,
                             "font": {"color": MUTED, "size": 13},
                             "x": 0.5, "y": 0.5}],
            "paper_bgcolor": BG, "plot_bgcolor": BG,
            "xaxis": {"visible": False}, "yaxis": {"visible": False},
        }
    }


# =================================================================
# 1) HALVING CLOCK — donut chart of cycle progression
# =================================================================
def halving_clock_chart() -> dict:
    """Circular donut showing where we are in the halving cycle.

    Green = first 18mo (accumulation -> peak window)
    Yellow = peak window (16-19 months post-halving)
    Red = bear/distribution (post-peak)
    Blue marker = current position
    """
    try:
        from core.halving_clock import current_halving_position, HISTORICAL
    except Exception:
        return _empty_fig("Halving Clock")

    pos = current_halving_position()
    days_post = pos.get("days_post_halving", 0)
    cycle_n = pos.get("current_cycle", 5)
    pct_thru = pos.get("pct_through_cycle", 0) or 0
    proj_peak = pos.get("projected_peak_date")
    proj_bot = pos.get("projected_bottom_date")

    # Cycle is ~1460 days (4 years between halvings)
    CYCLE_LEN = 1460
    # Phase boundaries (days post-halving)
    ACCUM_END  = 365   # first year — accumulation
    PEAK_START = 480   # peak window opens (16 mo)
    PEAK_END   = 570   # peak window closes (19 mo)
    BEAR_END   = CYCLE_LEN

    # 4 segments
    values = [ACCUM_END, PEAK_START - ACCUM_END, PEAK_END - PEAK_START, BEAR_END - PEAK_END]
    labels = ["Accumulation (0-12mo)", "Run-up (12-16mo)", "PEAK WINDOW (16-19mo)", "Bear/distribution (19+ mo)"]
    colors = [GREEN, GREEN, YELLOW, RED]

    # Current position needle approximated by an extra trace
    needle_angle = (days_post / CYCLE_LEN) * 360

    fig = {
        "data": [
            {
                "type": "pie", "values": values, "labels": labels,
                "marker": {"colors": colors, "line": {"color": BG, "width": 2}},
                "hole": 0.62, "rotation": 0, "direction": "clockwise",
                "sort": False, "textinfo": "none",
                "hovertemplate": "%{label}<br>%{value} days<extra></extra>",
                "showlegend": False,
            }
        ],
        "layout": {
            "paper_bgcolor": BG, "plot_bgcolor": BG,
            "margin": {"l": 10, "r": 10, "t": 30, "b": 10},
            "height": 280,
            "font": {"color": TEXT, "family": "Inter, sans-serif"},
            "annotations": [
                {
                    "text": (f"<b style='font-size:24px;color:#fff'>Day {days_post}</b><br>"
                             f"<span style='font-size:11px;color:#888'>of cycle {cycle_n} "
                             f"({pct_thru:.0f}% through)</span><br>"
                             f"<span style='font-size:10px;color:#888'>"
                             f"Peak proj: {proj_peak}</span>"),
                    "showarrow": False, "x": 0.5, "y": 0.5, "align": "center",
                },
            ],
            "showlegend": False,
        }
    }
    return fig


# =================================================================
# 2) BTC DOMINANCE GAUGE — 0-100% with regime zones
# =================================================================
def btc_dominance_gauge() -> dict:
    """Gauge showing BTC dominance % with regime zones.

    < 45%   ALT SEASON       (red — risk-on, alts beat BTC)
    45-55%  BALANCED         (yellow — neutral)
    55-65%  BTC DOMINANT     (green — BTC outperforming)
    > 65%   BTC HEGEMONY     (yellow — extreme; usually bear-rally or capitulation)
    """
    try:
        from core.btc_dominance import fetch_dominance, regime_classification
    except Exception:
        return _empty_fig("BTC Dominance")

    try:
        d = fetch_dominance()
        dom = d.get("btc_dominance_pct", 0)
        eth = d.get("eth_dominance_pct", 0)
        reg = regime_classification(dom)
        regime_name = reg.get("regime", "?")
    except Exception:
        return _empty_fig("BTC Dominance", "fetch failed")

    fig = {
        "data": [{
            "type": "indicator", "mode": "gauge+number",
            "value": round(dom, 1),
            "number": {"suffix": "%", "font": {"size": 38, "color": "#fff"}},
            "title": {"text": f"<b>BTC Dominance</b><br><span style='font-size:11px;color:#888'>"
                              f"Regime: {regime_name} · ETH {eth:.1f}%</span>",
                      "font": {"size": 13, "color": TEXT}},
            "gauge": {
                "axis": {"range": [30, 80], "tickwidth": 1, "tickcolor": MUTED,
                         "tickfont": {"size": 9, "color": MUTED}},
                "bar": {"color": "rgba(255,255,255,0.85)", "thickness": 0.25},
                "bgcolor": BG, "borderwidth": 0,
                "steps": [
                    {"range": [30, 45], "color": "rgba(239, 68, 68, 0.45)"},   # alt season
                    {"range": [45, 55], "color": "rgba(240, 185, 11, 0.45)"},  # balanced
                    {"range": [55, 65], "color": "rgba(34, 197, 94, 0.45)"},   # BTC dominant
                    {"range": [65, 80], "color": "rgba(240, 185, 11, 0.45)"},  # hegemony
                ],
                "threshold": {"line": {"color": "#fff", "width": 3},
                              "thickness": 0.9, "value": dom},
            },
        }],
        "layout": {
            "paper_bgcolor": BG, "plot_bgcolor": BG,
            "margin": {"l": 18, "r": 18, "t": 36, "b": 10},
            "height": 260,
            "font": {"color": TEXT, "family": "Inter, sans-serif"},
        },
    }
    return fig


# =================================================================
# 3) STOCK-TO-FLOW DEFLECTION GAUGE
# =================================================================
def s2f_deflection_gauge() -> dict:
    """Gauge showing price / S2F model fair value (deflection).

    < 0.3   DEEP UNDERVALUED  (red zone — historical bottoms)
    0.3-0.7 UNDERVALUED        (yellow — accumulation)
    0.7-1.3 FAIR VALUE          (green — neutral)
    1.3-2.0 OVERVALUED          (yellow — distribution)
    > 2.0   EXTREMELY OVERVALUED (red — historical tops)

    Note: model is DEGRADED post-cycle 4. Use as one input, not gospel.
    """
    try:
        from core.btc_more_signals import stock_to_flow_deflection
    except Exception:
        return _empty_fig("Stock-to-Flow")

    try:
        s2f = stock_to_flow_deflection()
        ratio = s2f.get("s2f_ratio", 1.0) or 1.0
        s2f_fair = s2f.get("s2f_fair_value", 0)
        defl_pct = s2f.get("deflection_pct", 0)
    except Exception:
        return _empty_fig("Stock-to-Flow", "fetch failed")

    # Color semantic: low ratio = price BELOW fair value = BUY zone = GREEN.
    # High ratio = price ABOVE fair value = SELL zone = RED.
    # Was previously red at both ends (technically "extreme" but UX inversion —
    # user sees red and panics, even when action is BUY at the low end).
    fig = {
        "data": [{
            "type": "indicator", "mode": "gauge+number",
            "value": round(ratio, 2),
            "number": {"suffix": "×", "font": {"size": 38, "color": "#fff"}},
            "title": {"text": f"<b>S2F Deflection</b><br><span style='font-size:11px;color:#888'>"
                              f"Fair: ${s2f_fair:,.0f} · "
                              f"{'over' if defl_pct>0 else 'under'} by {abs(defl_pct):.0f}% "
                              f"(low=BUY, high=SELL; model degraded post-cycle 4)</span>",
                      "font": {"size": 13, "color": TEXT}},
            "gauge": {
                "axis": {"range": [0, 3.0], "tickvals": [0, 0.3, 0.7, 1.3, 2.0, 3.0],
                         "tickwidth": 1, "tickcolor": MUTED,
                         "tickfont": {"size": 9, "color": MUTED}},
                "bar": {"color": "rgba(255,255,255,0.85)", "thickness": 0.25},
                "bgcolor": BG, "borderwidth": 0,
                "steps": [
                    {"range": [0,   0.3], "color": "rgba(34, 197, 94, 0.60)"},   # DEEP BUY
                    {"range": [0.3, 0.7], "color": "rgba(34, 197, 94, 0.35)"},   # BUY
                    {"range": [0.7, 1.3], "color": "rgba(240, 185, 11, 0.40)"},  # FAIR / neutral
                    {"range": [1.3, 2.0], "color": "rgba(239, 68, 68, 0.35)"},   # SELL
                    {"range": [2.0, 3.0], "color": "rgba(239, 68, 68, 0.60)"},   # EXTREME SELL
                ],
                "threshold": {"line": {"color": "#fff", "width": 3},
                              "thickness": 0.9, "value": min(ratio, 3.0)},
            },
        }],
        "layout": {
            "paper_bgcolor": BG, "plot_bgcolor": BG,
            "margin": {"l": 18, "r": 18, "t": 36, "b": 10},
            "height": 260,
            "font": {"color": TEXT, "family": "Inter, sans-serif"},
        },
    }
    return fig


# =================================================================
# 4) OPEN INTEREST GAUGE
# =================================================================
def open_interest_gauge() -> dict:
    """Gauge showing aggregated BTC perpetual OI z-score.

    Sign + magnitude tell the story:
      score > +0.6  : OI froth — speculative overheat, top risk
      score in [-0.6, 0.6] : normal range
      score < -0.6  : OI capitulation — bottom signal

    score is normalized z-score across 30 days. open_interest_signal returns [-1, 1].
    """
    try:
        from signals.open_interest import open_interest_signal
    except Exception:
        return _empty_fig("Open Interest")

    try:
        score = open_interest_signal("BTC/USDT")
        # Map [-1, 1] to a meaningful gauge value: -100 to +100
        gauge_val = round(score * 100, 1)
    except Exception:
        return _empty_fig("Open Interest", "fetch failed")

    # Underlying signal is OI * sign-of-price-return (trend-confirmation),
    # NOT pure froth/capitulation. Re-label honestly to avoid misleading reads.
    if score > 0.6:
        zone = "STRONG UPTREND CONFIRM (OI + price up)"
    elif score > 0.3:
        zone = "WEAK UPTREND"
    elif score > -0.3:
        zone = "NEUTRAL / RANGING"
    elif score > -0.6:
        zone = "WEAK DOWNTREND"
    else:
        zone = "STRONG DOWNTREND CONFIRM (OI + price down)"

    fig = {
        "data": [{
            "type": "indicator", "mode": "gauge+number",
            "value": gauge_val,
            "number": {"suffix": "", "font": {"size": 38, "color": "#fff"}},
            "title": {"text": f"<b>OI Trend Confirmation</b><br>"
                              f"<span style='font-size:11px;color:#888'>{zone}</span>",
                      "font": {"size": 13, "color": TEXT}},
            "gauge": {
                "axis": {"range": [-100, 100], "tickwidth": 1, "tickcolor": MUTED,
                         "tickfont": {"size": 9, "color": MUTED},
                         "tickvals": [-100, -60, -30, 0, 30, 60, 100]},
                "bar": {"color": "rgba(255,255,255,0.85)", "thickness": 0.25},
                "bgcolor": BG, "borderwidth": 0,
                "steps": [
                    {"range": [-100, -60], "color": "rgba(34, 197, 94, 0.55)"},   # capitulation = buy
                    {"range": [-60, -30],  "color": "rgba(34, 197, 94, 0.30)"},
                    {"range": [-30,  30],  "color": "rgba(240, 185, 11, 0.30)"},
                    {"range": [ 30,  60],  "color": "rgba(239, 68, 68, 0.30)"},
                    {"range": [ 60, 100],  "color": "rgba(239, 68, 68, 0.55)"},   # froth = sell
                ],
                "threshold": {"line": {"color": "#fff", "width": 3},
                              "thickness": 0.9, "value": gauge_val},
            },
        }],
        "layout": {
            "paper_bgcolor": BG, "plot_bgcolor": BG,
            "margin": {"l": 18, "r": 18, "t": 36, "b": 10},
            "height": 260,
            "font": {"color": TEXT, "family": "Inter, sans-serif"},
        },
    }
    return fig


# =================================================================
# 5) CYCLE 4 vs CYCLE 5 OVERLAY — normalized to halving = day 0
# =================================================================
def cycle_overlay_chart() -> dict:
    """Line chart: cycle 4 (2020-22) price vs cycle 5 (2024-26) price,
    both normalized: x = days post-halving, y = price / halving-day-price.

    Shows where current cycle stands vs the previous one.
    """
    try:
        import pandas as pd
        from datetime import datetime
        from core.halving_clock import HISTORICAL
        from core import data
    except Exception:
        return _empty_fig("Cycle overlay")

    try:
        # Pull price history for both cycles. ohlcv_extended returns DatetimeIndex.
        df_all = data.ohlcv_extended("BTC/USDT", days_back=365 * 6)
        if "date" not in df_all.columns:
            df_all = df_all.copy()
            df_all["date"] = df_all.index
        df_all["date"] = pd.to_datetime(df_all["date"])
        # Strip timezone for comparison with tz-naive halving dates
        if df_all["date"].dt.tz is not None:
            df_all["date"] = df_all["date"].dt.tz_localize(None)

        traces = []
        for cycle_n, color, name in [(3, "#666", "Cycle 3 (2016-18)"),
                                      (4, "#888", "Cycle 4 (2020-22)"),
                                      (5, "#4a90e2", "Cycle 5 (2024-26)")]:
            halv_date = HISTORICAL[cycle_n]["halving"]
            df_cyc = df_all[df_all["date"] >= pd.Timestamp(halv_date)].copy()
            if df_cyc.empty: continue
            df_cyc["days_post"] = (df_cyc["date"] - pd.Timestamp(halv_date)).dt.days
            df_cyc = df_cyc[df_cyc["days_post"] <= 1100]  # cap at 3yr post-halving
            if df_cyc.empty: continue
            base_price = df_cyc.iloc[0]["close"]
            if base_price <= 0: continue
            df_cyc["norm"] = df_cyc["close"] / base_price
            traces.append({
                "type": "scatter", "mode": "lines",
                "x": df_cyc["days_post"].tolist(),
                "y": df_cyc["norm"].round(2).tolist(),
                "name": name, "line": {"color": color, "width": 2},
                "hovertemplate": f"<b>{name}</b><br>Day %{{x}} post-halving<br>"
                                  f"%{{y}}× halving-day price<extra></extra>",
            })

        # Mark cycle 4 peak with a vertical line
        c4_peak_days = HISTORICAL[4]["days_to_peak"]
        peak_shape = {
            "type": "line", "x0": c4_peak_days, "x1": c4_peak_days,
            "y0": 0, "y1": 1, "yref": "paper",
            "line": {"color": RED, "width": 1, "dash": "dot"},
        }
        peak_anno = {
            "x": c4_peak_days, "y": 1, "yref": "paper",
            "text": f"Cycle 4 peak<br>day {c4_peak_days}",
            "showarrow": False, "font": {"color": RED, "size": 9},
            "yshift": 6, "bgcolor": BG,
        }
    except Exception as e:
        return _empty_fig("Cycle overlay", f"compute failed: {e}")

    fig = {
        "data": traces,
        "layout": {
            "title": {"text": "<b>Cycle 4 vs Cycle 5 — normalized to halving day</b>",
                      "font": {"color": TEXT, "size": 13}},
            "paper_bgcolor": BG, "plot_bgcolor": BG,
            "margin": {"l": 50, "r": 20, "t": 50, "b": 40},
            "height": 320,
            "font": {"color": TEXT, "family": "Inter, sans-serif"},
            "xaxis": {"title": "Days post-halving", "gridcolor": "#222",
                      "color": TEXT, "tickfont": {"size": 10}},
            "yaxis": {"title": "Price × halving-day", "gridcolor": "#222",
                      "color": TEXT, "tickfont": {"size": 10}},
            "legend": {"x": 0.02, "y": 0.98, "font": {"size": 11, "color": TEXT},
                       "bgcolor": "rgba(14,17,23,0.7)"},
            "shapes": [peak_shape], "annotations": [peak_anno],
            "hovermode": "x unified",
        },
    }
    return fig


# =================================================================
# Convenience: compute everything in one call (for precompute cache)
# =================================================================
def all_swift_dials() -> dict:
    """Compute all 5 dials and return as dict. For precompute_dashboard caching."""
    return {
        "halving_clock":    halving_clock_chart(),
        "btc_dominance":    btc_dominance_gauge(),
        "s2f_deflection":   s2f_deflection_gauge(),
        "open_interest":    open_interest_gauge(),
        "cycle_overlay":    cycle_overlay_chart(),
    }


if __name__ == "__main__":
    out = all_swift_dials()
    for k, v in out.items():
        n_traces = len(v.get("data", []))
        print(f"  {k}: {n_traces} traces, height={v.get('layout', {}).get('height', '?')}")
