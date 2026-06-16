"""Phillip Swift / LookIntoBitcoin chart suite.

Generates interactive Plotly charts that show full historical context
for each indicator — not just point values. This is what Swift's
LookIntoBitcoin.com is famous for.

Charts produced:
  1. Rainbow Chart           — BTC price with 9 log-regression color bands
  2. Pi Cycle Top history    — 111d MA / (350d MA × 2) ratio with cross line
  3. Pi Cycle Bottom history — 150d MA / (471d MA × 0.745) ratio
  4. Golden Ratio Mult bands — price / 350d MA with Fibonacci bands
  5. 2y MA Multiplier bands  — price / 2y MA with 5× top band
  6. MVRV bands              — MVRV history with capitulation/euphoria zones
  7. Puell Multiple bands    — historical with bottom/top zones
  8. HODL Waves heatmap      — realized cap velocity over time (proxy)

Each returns a plotly Figure dict ready for st.plotly_chart().
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _btc_history(period: str = "max") -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
        df = yf.Ticker("BTC-USD").history(period=period)
        if df is None or df.empty: return None
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception:
        return None


def _cm(metric: str, days: int = 3650) -> Optional[pd.Series]:
    try:
        from core.btc_pro_signals import _cm as _coinmetrics
        df = _coinmetrics(metric, days=days)
        if df is None or df.empty: return None
        return df.iloc[:, 0]
    except Exception:
        return None


def _miner_revenue(days: int = 2200) -> Optional[pd.Series]:
    """Daily miner revenue (USD). CoinMetrics RevUSD is now paywalled on the
    free tier, so source it from blockchain.com's free charts API instead."""
    try:
        from core.btc_pro_signals import _blockchain_info
        yrs = max(2, days // 365 + 1)
        df = _blockchain_info("miners-revenue", timespan=f"{yrs}years")
        if df is None or df.empty: return None
        return df.iloc[:, 0]
    except Exception:
        return None


def _realized_cap(days: int = 1500) -> Optional[pd.Series]:
    """Realized cap. CoinMetrics CapRealUSD is now paywalled, so derive it the
    same way the cost-basis module does: market cap / MVRV (both still free)."""
    try:
        mc = _cm("CapMrktCurUSD", days=days)
        mv = _cm("CapMVRVCur", days=days)
        if mc is None or mv is None or mc.empty or mv.empty: return None
        j = mc.to_frame("mc").join(mv.to_frame("mv"), how="inner").dropna()
        if j.empty: return None
        return (j["mc"] / j["mv"]).rename("rcap")
    except Exception:
        return None


def _safe_fig_dict(fig) -> dict:
    """Plotly figure -> pure-Python dict with NO numpy arrays.

    fig.to_dict() keeps numpy ndarrays in the trace data; pickling those makes
    the cached panel fail to load/render whenever the numpy (or plotly) version
    on the render host (Streamlit Cloud) differs from the build host (the GitHub
    Action) — numpy 1.x<->2.x pickle buffers are incompatible. fig.to_json()
    uses plotly's own encoder to emit plain JSON lists, so json.loads() yields a
    version-proof dict that pickles and renders identically everywhere.
    """
    import json
    return json.loads(fig.to_json())


def _base_layout(title: str, ylog: bool = False) -> dict:
    return {
        "title": {"text": title, "font": {"size": 13, "color": "#ccc"}},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "#0e1117",
        "font": {"color": "#ccc", "family": "Inter, sans-serif", "size": 10},
        "xaxis": {"gridcolor": "#222", "linecolor": "#444", "tickfont": {"size": 9}},
        "yaxis": {
            "gridcolor": "#222", "linecolor": "#444", "tickfont": {"size": 9},
            "type": "log" if ylog else "linear",
        },
        "margin": {"l": 50, "r": 20, "t": 40, "b": 30},
        "height": 280,
        "showlegend": False,
        "hovermode": "x unified",
    }


# ============================================================
# 1. RAINBOW CHART — Swift's iconic visual
# ============================================================

def rainbow_chart() -> Optional[dict]:
    """BTC log-scale price with 9 color bands from log regression model."""
    df = _btc_history("max")
    if df is None or len(df) < 100: return None
    import plotly.graph_objects as go

    df = df[["Close"]].copy()
    # Days since genesis
    GENESIS = pd.Timestamp("2009-01-03")
    df["days"] = (df.index - GENESIS).days

    # Log regression: log10(price) = a * log10(days) + b
    # Use the well-known LookIntoBitcoin approximation (slow drift)
    a, b = 5.84, -17.01

    df["log_days"] = np.log10(df["days"].clip(lower=1))
    df["model"] = 10 ** (a * df["log_days"] + b)

    # 9 rainbow bands as multipliers of model
    bands = [
        (-50, "Fire Sale BUY",     "rgba(38,166,154,0.5)"),
        (-25, "BUY!",                "rgba(102,187,106,0.4)"),
        (0,    "Accumulate",          "rgba(174,213,129,0.3)"),
        (50,   "Cheap",                "rgba(212,225,87,0.3)"),
        (100,  "Fair Value",           "rgba(255,235,59,0.3)"),
        (200,  "Resistance",           "rgba(255,167,38,0.3)"),
        (300,  "FOMO",                  "rgba(255,112,67,0.4)"),
        (500,  "MAX BUBBLE",            "rgba(239,68,68,0.5)"),
    ]

    fig = go.Figure()
    # Build bands
    prev_mult = -100
    for pct, label, color in bands:
        mult = 1 + pct / 100
        fig.add_trace(go.Scatter(
            x=df.index, y=df["model"] * mult, mode="lines",
            line={"color": "rgba(0,0,0,0)", "width": 0},
            fill="tonexty" if prev_mult > -100 else None,
            fillcolor=color, showlegend=False, hoverinfo="skip",
        ))
        prev_mult = mult

    # Price on top
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        line={"color": "white", "width": 2},
        name="BTC", hovertemplate="$%{y:,.0f}<extra></extra>",
    ))

    # Current value annotation
    last_price = float(df["Close"].iloc[-1])
    last_model = float(df["model"].iloc[-1])
    dev = (last_price / last_model - 1) * 100
    fig.add_annotation(
        xref="paper", yref="paper", x=0.99, y=0.02,
        text=f"BTC ${last_price:,.0f}  ({dev:+.0f}% from model)",
        showarrow=False, bgcolor="rgba(0,0,0,0.6)", bordercolor="#444",
        font={"color": "white", "size": 11},
    )

    # Band labels at the right edge — the iconic part of a rainbow chart.
    for _pct, _lbl, _col in bands:
        fig.add_annotation(
            x=df.index[-1], y=last_model * (1 + _pct / 100), text=_lbl,
            xanchor="left", yanchor="middle", showarrow=False,
            font={"size": 8, "color": "rgba(255,255,255,0.6)"},
        )

    layout = _base_layout("BTC Rainbow Chart — log-regression bands (Swift / LookIntoBitcoin)", ylog=True)
    layout["height"] = 360
    layout["margin"] = {"l": 50, "r": 92, "t": 40, "b": 30}
    fig.update_layout(**layout)
    return _safe_fig_dict(fig)


# ============================================================
# 2. PI CYCLE TOP — ratio history
# ============================================================

def pi_cycle_top_chart() -> Optional[dict]:
    """111d MA / (350d MA × 2) ratio. Cross = 1.0 = cycle top within 3 days."""
    df = _btc_history("8y")
    if df is None or len(df) < 350: return None
    import plotly.graph_objects as go

    closes = df["Close"]
    ma111 = closes.rolling(111).mean()
    ma350x2 = closes.rolling(350).mean() * 2
    ratio = ma111 / ma350x2

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ratio.index, y=ratio.values, line={"color": "#22c55e", "width": 2},
        name="Pi Top Ratio",
        hovertemplate="%{y:.2f}<extra></extra>",
    ))
    # Cross line at 1.0
    fig.add_hline(y=1.0, line_dash="dash", line_color="#ef4444",
                   annotation_text="Cross = CYCLE TOP",
                   annotation_position="top right",
                   annotation_font_color="#ef4444")
    # Approach band at 0.95
    fig.add_hline(y=0.95, line_dash="dot", line_color="#f0b90b",
                   annotation_text="Approach zone",
                   annotation_position="bottom right",
                   annotation_font_color="#f0b90b")

    # Current value
    cur = float(ratio.iloc[-1]) if not pd.isna(ratio.iloc[-1]) else 0
    fig.add_annotation(
        xref="paper", yref="paper", x=0.99, y=0.02,
        text=f"Current: {cur:.3f}",
        showarrow=False, bgcolor="rgba(0,0,0,0.6)", bordercolor="#444",
        font={"color": "white", "size": 11},
    )

    fig.update_layout(**_base_layout("Pi Cycle Top — 111d MA / (350d MA × 2)"))
    return _safe_fig_dict(fig)


# ============================================================
# 3. PI CYCLE BOTTOM — ratio history
# ============================================================

def pi_cycle_bottom_chart() -> Optional[dict]:
    """150d MA / (471d MA × 0.745). Below 1.0 = bottom signal."""
    df = _btc_history("8y")
    if df is None or len(df) < 471: return None
    import plotly.graph_objects as go

    closes = df["Close"]
    ma150 = closes.rolling(150).mean()
    threshold = closes.rolling(471).mean() * 0.745
    ratio = ma150 / threshold

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ratio.index, y=ratio.values, line={"color": "#22c55e", "width": 2},
        hovertemplate="%{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=1.0, line_dash="dash", line_color="#22c55e",
                   annotation_text="Cross = CYCLE BOTTOM",
                   annotation_position="top right",
                   annotation_font_color="#22c55e")

    cur = float(ratio.iloc[-1]) if not pd.isna(ratio.iloc[-1]) else 0
    fig.add_annotation(
        xref="paper", yref="paper", x=0.99, y=0.95,
        text=f"Current: {cur:.3f}",
        showarrow=False, bgcolor="rgba(0,0,0,0.6)", bordercolor="#444",
        font={"color": "white", "size": 11},
    )

    fig.update_layout(**_base_layout("Pi Cycle Bottom — 150d MA / (471d MA × 0.745)"))
    return _safe_fig_dict(fig)


# ============================================================
# 4. GOLDEN RATIO MULTIPLIER — Fibonacci bands
# ============================================================

def golden_ratio_chart() -> Optional[dict]:
    """Price / 350d MA history with horizontal bands at 1.6/2/3/5/8/13/21."""
    df = _btc_history("8y")
    if df is None or len(df) < 350: return None
    import plotly.graph_objects as go

    closes = df["Close"]
    ma350 = closes.rolling(350).mean()
    mult = closes / ma350

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mult.index, y=mult.values, line={"color": "white", "width": 2},
        hovertemplate="%{y:.2f}× 350d MA<extra></extra>",
    ))

    # Fibonacci bands
    bands = [
        (1.0, "350d MA", "#22c55e"),
        (1.6, "Acceleration", "#9ccc65"),
        (2.0, "Resistance", "#ffeb3b"),
        (3.0, "Bull Market", "#ffa726"),
        (5.0, "Late Bull", "#ff7043"),
        (8.0, "Top Cap", "#ef4444"),
        (13.0, "Major Top", "#b91c1c"),
        (21.0, "MAX BUBBLE", "#7b1fa2"),
    ]
    for level, label, color in bands:
        fig.add_hline(y=level, line_dash="dot", line_color=color,
                       annotation_text=f"{level}× {label}",
                       annotation_position="top right",
                       annotation_font_color=color,
                       annotation_font_size=9)

    cur = float(mult.iloc[-1]) if not pd.isna(mult.iloc[-1]) else 0
    fig.add_annotation(
        xref="paper", yref="paper", x=0.99, y=0.02,
        text=f"Current: {cur:.2f}× 350d MA",
        showarrow=False, bgcolor="rgba(0,0,0,0.7)", bordercolor="#444",
        font={"color": "white", "size": 11},
    )

    layout = _base_layout("Golden Ratio Multiplier — price / 350d MA with Fibonacci bands", ylog=True)
    layout["height"] = 320
    fig.update_layout(**layout)
    return _safe_fig_dict(fig)


# ============================================================
# 5. 2-YEAR MA MULTIPLIER bands
# ============================================================

def two_year_ma_chart() -> Optional[dict]:
    """Price / 2y MA history with horizontal band at 5× (top zone)."""
    df = _btc_history("8y")
    if df is None or len(df) < 730: return None
    import plotly.graph_objects as go

    closes = df["Close"]
    ma2y = closes.rolling(730).mean()
    mult = closes / ma2y

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mult.index, y=mult.values, line={"color": "white", "width": 2},
        hovertemplate="%{y:.2f}× 2y MA<extra></extra>",
    ))

    bands = [
        (0.6, "Deep Capitulation", "#22c55e"),
        (1.0, "2y MA", "#9ccc65"),
        (2.0, "Mid-Cycle", "#ffeb3b"),
        (3.5, "Late Cycle", "#ff7043"),
        (5.0, "TOP ZONE", "#ef4444"),
    ]
    for level, label, color in bands:
        fig.add_hline(y=level, line_dash="dot", line_color=color,
                       annotation_text=f"{level}× {label}",
                       annotation_position="top right",
                       annotation_font_color=color,
                       annotation_font_size=9)

    cur = float(mult.iloc[-1]) if not pd.isna(mult.iloc[-1]) else 0
    fig.add_annotation(
        xref="paper", yref="paper", x=0.99, y=0.02,
        text=f"Current: {cur:.2f}× 2y MA",
        showarrow=False, bgcolor="rgba(0,0,0,0.7)", bordercolor="#444",
        font={"color": "white", "size": 11},
    )

    fig.update_layout(**_base_layout("2-Year MA Multiplier"))
    return _safe_fig_dict(fig)


# ============================================================
# 6. MVRV with capitulation/euphoria zones
# ============================================================

def mvrv_bands_chart() -> Optional[dict]:
    """MVRV ratio history with horizontal zones."""
    s = _cm("CapMVRVCur", days=2200)
    if s is None or len(s) < 100: return None
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s.index, y=s.values, line={"color": "#22c55e", "width": 2},
        hovertemplate="MVRV %{y:.2f}<extra></extra>",
    ))

    bands = [
        (1.0, "HODLer underwater (CAPITULATION)", "#16a34a"),
        (1.6, "Recovery", "#9ccc65"),
        (2.4, "Bull market", "#ffa726"),
        (3.7, "EUPHORIA (top zone)", "#b91c1c"),
    ]
    for level, label, color in bands:
        fig.add_hline(y=level, line_dash="dot", line_color=color,
                       annotation_text=label,
                       annotation_position="top right",
                       annotation_font_color=color,
                       annotation_font_size=9)

    cur = float(s.iloc[-1]) if not pd.isna(s.iloc[-1]) else 0
    fig.add_annotation(
        xref="paper", yref="paper", x=0.99, y=0.02,
        text=f"Current: {cur:.2f}",
        showarrow=False, bgcolor="rgba(0,0,0,0.7)", bordercolor="#444",
        font={"color": "white", "size": 11},
    )

    fig.update_layout(**_base_layout("MVRV ratio with cycle bands"))
    return _safe_fig_dict(fig)


# ============================================================
# 7. Puell Multiple bands
# ============================================================

def puell_bands_chart() -> Optional[dict]:
    """Puell history = daily miner revenue / 365d MA."""
    rev = _miner_revenue(days=2200)
    if rev is None or len(rev) < 365: return None
    import plotly.graph_objects as go

    ma365 = rev.rolling(365, min_periods=30).mean()
    puell = rev / ma365

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=puell.index, y=puell.values, line={"color": "#22c55e", "width": 1.5},
        hovertemplate="Puell %{y:.2f}<extra></extra>",
    ))

    bands = [
        (0.5, "BOTTOM ZONE (miner cap)", "#16a34a"),
        (1.0, "Equilibrium", "#9ccc65"),
        (2.5, "TOP ZONE (miner extreme)", "#b91c1c"),
    ]
    for level, label, color in bands:
        fig.add_hline(y=level, line_dash="dot", line_color=color,
                       annotation_text=label,
                       annotation_position="top right",
                       annotation_font_color=color,
                       annotation_font_size=9)

    cur = float(puell.iloc[-1]) if not pd.isna(puell.iloc[-1]) else 0
    fig.add_annotation(
        xref="paper", yref="paper", x=0.99, y=0.02,
        text=f"Current: {cur:.2f}",
        showarrow=False, bgcolor="rgba(0,0,0,0.7)", bordercolor="#444",
        font={"color": "white", "size": 11},
    )

    layout = _base_layout("Puell Multiple with bands", ylog=True)
    fig.update_layout(**layout)
    return _safe_fig_dict(fig)


# ============================================================
# 8. HODL Waves proxy heatmap (realized cap velocity over time)
# ============================================================

def hodl_waves_heatmap() -> Optional[dict]:
    """Realized cap velocity over time — low = LTH dominant = bullish.

    Free-tier proxy. Real HODL Waves require paid CoinMetrics SplyAct1yr.
    """
    rc = _realized_cap(days=1500)
    if rc is None or len(rc) < 60: return None
    import plotly.graph_objects as go

    # 30d velocity (% change in realized cap over 30d, annualized)
    velocity = rc.pct_change(30) * 100 * 12
    velocity = velocity.dropna()
    if len(velocity) < 30: return None

    fig = go.Figure()
    # Color by velocity level
    fig.add_trace(go.Scatter(
        x=velocity.index, y=velocity.values,
        mode="lines",
        line={"color": "#22c55e", "width": 1.5},
        fill="tozeroy", fillcolor="rgba(38,166,154,0.2)",
        hovertemplate="Velocity %{y:.0f}%<extra></extra>",
    ))

    # Zones (annualized)
    bands = [
        (0,    "LTH DOMINANT (bullish)", "#16a34a"),
        (30,   "Mixed", "#ffeb3b"),
        (100,  "Active distribution", "#ef4444"),
    ]
    for level, label, color in bands:
        fig.add_hline(y=level, line_dash="dot", line_color=color,
                       annotation_text=label,
                       annotation_position="top right",
                       annotation_font_color=color,
                       annotation_font_size=9)

    cur = float(velocity.iloc[-1]) if not pd.isna(velocity.iloc[-1]) else 0
    fig.add_annotation(
        xref="paper", yref="paper", x=0.99, y=0.95,
        text=f"Current 30d annualized: {cur:.0f}%",
        showarrow=False, bgcolor="rgba(0,0,0,0.7)", bordercolor="#444",
        font={"color": "white", "size": 11},
    )

    fig.update_layout(**_base_layout("HODL proxy — Realized Cap 30d velocity (annualized)"))
    return _safe_fig_dict(fig)


# ============================================================
# Aggregator
# ============================================================

def all_swift_charts() -> dict:
    """Return all 8 chart figure dicts. Heavy compute — cache aggressively."""
    out = {}
    chart_funcs = [
        ("rainbow",         rainbow_chart),
        ("pi_cycle_top",    pi_cycle_top_chart),
        ("pi_cycle_bottom", pi_cycle_bottom_chart),
        ("golden_ratio",    golden_ratio_chart),
        ("two_year_ma",     two_year_ma_chart),
        ("mvrv_bands",      mvrv_bands_chart),
        ("puell_bands",     puell_bands_chart),
        ("hodl_waves",      hodl_waves_heatmap),
    ]
    for name, fn in chart_funcs:
        try:
            out[name] = fn()
        except Exception as e:
            out[name] = None
    return out


def main():
    r = all_swift_charts()
    for name, fig in r.items():
        status = "OK" if fig else "FAILED"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
