"""BTC Prediction Dashboard — pro-tier UX (v2 redesign).

Sticky sidebar with persistent verdict + countdown + scorecard.
Hero verdict banner.
Five tabs: Overview / Cycle Math / On-Chain / Technical / Detail.

Same analytics modules as before; UX is rebuilt for clarity + density.

Run:  streamlit run btc_prediction_dashboard.py --server.port 8511
URL:  http://localhost:8511
"""

from __future__ import annotations

import sys
import html
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))


# === Page config ===
st.set_page_config(
    page_title="Dave's BTC Dashboard",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)
# === Auto-refresh ===
# Repaint the whole dashboard every few minutes so it stays current on an
# always-open screen WITHOUT a manual browser refresh: each rerun re-fetches the
# live price (60s cache) and repaints all panels, and the periodic activity also
# keeps the Streamlit Cloud app from going to sleep. Guarded so a missing
# component (e.g. a local run before `pip install streamlit-autorefresh`)
# degrades gracefully to no-autorefresh.
_AUTOREFRESH_MS = 60_000  # 60s — re-pulls the live-data overlay (publish_live_cache.py) for a near-live feel
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=_AUTOREFRESH_MS, key="dashboard_autorefresh")
except Exception:
    pass  # component unavailable — fall back to manual browser refresh


# ─────────────────────────────────────────────────────────────────
# PRIVACY: the dashboard URL is shared with friends. When False, all
# personal figures (NZ$ amounts, personal allocation %, your equity
# exposure) are hidden — only signals/market data show. Email alerts
# to the operator's own inbox still include the NZ$ amounts.
# Flip to True only if the dashboard is private again.
# ─────────────────────────────────────────────────────────────────
SHOW_PERSONAL = False


def _swift_fig(d):
    """Rebuild a cached figure dict into a Figure, tolerating plotly-version drift.

    A cached figure is JSON built by one plotly version; the render host may run a
    different plotly. skip_invalid=True drops any property the render plotly doesn't
    recognise (instead of raising), so a cross-version cached figure still renders
    rather than going blank. Used by the Swift cycle charts (built off-host)."""
    import plotly.graph_objects as go
    return go.Figure(d, skip_invalid=True)

def _money(amount, fallback: str = "—") -> str:
    """Render an NZ$ amount, or a redaction placeholder when sanitised."""
    if not SHOW_PERSONAL:
        return fallback
    try:
        return f"NZ${float(amount):,.0f}"
    except Exception:
        return fallback

def _money_paren(amount) -> str:
    """ ' (NZ$X)' when private, '' when sanitised — for inline parentheticals."""
    if not SHOW_PERSONAL:
        return ""
    try:
        return f" (NZ${float(amount):,.0f})"
    except Exception:
        return ""


# === Theme ===
C = {
    # 3-color palette (Swift rationalization): GREEN / YELLOW / RED
    # Used consistently across all verdict cards, gauges, and scorecard cells.
    "green":     "#22c55e",   # primary green — deploy / buy / bullish
    "yellow":    "#f0b90b",   # primary yellow — hold / watch / neutral
    "red":       "#ef4444",   # primary red — exit / trim / bearish
    # Legacy keys — now ALIGNED to the canonical green/red so there is exactly
    # ONE bullish green and ONE bearish red across the whole product.
    "bull":      "#22c55e",   # == green (was teal #22c55e)
    "deep_bull": "#16a34a",   # single darker green (was #16a34a)
    "bear":      "#ef4444",   # == red (was #ef4444)
    "deep_bear": "#b91c1c",   # single darker red (was #b91c1c)
    "neutral":   "#f0b90b",
    "extreme":   "#7b1fa2",
    "accent":    "#4a90e2",
    "muted":     "#888",
    "card_bg":   "#1a1d24",
    "card_br":   "#2a2d34",
    "lth":       "#9c27b0",
    "sth":       "#f0b90b",
    "text":      "#d4d4d4",
}

# Palette shorthands — prefer these over raw hex so colour stays single-source.
GREEN, AMBER, RED = C["green"], C["yellow"], C["red"]
DEEP_GREEN, DEEP_RED, MUTED = C["deep_bull"], C["deep_bear"], C["muted"]

CHART_LAYOUT = dict(
    plot_bgcolor="#0e1117",
    paper_bgcolor="#0e1117",
    font=dict(family="Inter, system-ui, sans-serif", size=11, color=C["text"]),
    margin=dict(l=24, r=24, t=40, b=40),
    hoverlabel=dict(font=dict(family="Inter", size=11)),
)


# === Custom CSS ===
st.markdown(
    """
<style>
.block-container { padding-top: 0.4rem; padding-bottom: 2rem; }
/* Collapse Streamlit's built-in top toolbar strip so the masthead sits
   flush at the very top. Transparent + slim (not display:none) keeps the
   sidebar-toggle control functional. */
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 1.25rem !important;
}
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    font-size: 15px;
    font-weight: 500;
    padding: 8px 16px;
    background: #1a1d24;
    border-radius: 6px 6px 0 0;
}
.stTabs [aria-selected="true"] { background: #2a2d34; }
section[data-testid="stSidebar"] { background: #0a0d12; }
.metric-block {
    background: #1a1d24;
    padding: 12px 14px;
    border-radius: 6px;
    border-left: 3px solid #4a90e2;
    margin-bottom: 8px;
}
.metric-label {
    font-size: 10px;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 2px;
}
.metric-value {
    font-size: 22px;
    font-weight: 700;
    color: #fff;
    line-height: 1.1;
}
.metric-sub {
    font-size: 12px;
    color: #aaa;
    margin-top: 2px;
}
.verdict-hero {
    padding: 20px 26px;
    border-radius: 10px;
    border-left: 8px solid var(--verdict-color);
    background: linear-gradient(90deg, var(--verdict-bg) 0%, #1a1d24 100%);
}
.verdict-title {
    font-size: 11px;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 4px;
}
.verdict-text {
    font-size: 44px;
    font-weight: 800;
    line-height: 1;
    color: var(--verdict-color);
}
.verdict-sub {
    font-size: 15px;
    color: #ccc;
    margin-top: 8px;
}
.verdict-detail {
    font-size: 13px;
    color: #888;
    margin-top: 10px;
}
.section-header {
    font-size: 16px;
    font-weight: 600;
    color: #d4d4d4;
    margin: 20px 0 8px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #2a2d34;
}
.compact-table { font-size: 12px; }
h2, h3 { color: #d4d4d4; font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)


# === Mobile-responsive CSS + iOS PWA tags ===
# Designed phone-first per Phillip Swift's review. Desktop view unchanged
# (all rules gated behind @media (max-width: 768px)).
st.markdown(
    """
<!-- iOS PWA: when user does Share->Add to Home Screen, app behaves native-ish -->
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="BTC">
<meta name="mobile-web-app-capable" content="yes">
<meta name="theme-color" content="#0e1117">
<meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover, maximum-scale=5.0">

<style>
/* ═══════════════════════════════════════════════════════════════
   GLOBAL — prevent horizontal page-scroll on ANY screen size.
   This MUST come before media queries. Wide content (tables, code,
   Plotly) should scroll inside its OWN container, never push the body.
   ═══════════════════════════════════════════════════════════════ */
html, body {
    overflow-x: hidden !important;
    max-width: 100vw !important;
}
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main,
.main > div,
section.main {
    overflow-x: hidden !important;
    max-width: 100vw !important;
}
/* Plotly charts default to a min-width that overflows phone — clamp them */
.js-plotly-plot, .plot-container, .plotly {
    max-width: 100% !important;
    width: 100% !important;
}

/* CRITICAL MOBILE FIX: prevent Plotly + iframes from capturing scroll touch.
   touch-action: pan-x pan-y means "single-finger swipes scroll the PAGE,
   not the chart." Pinch-zoom + double-tap-zoom on the chart are blocked.
   This stops the "graph zooms when I'm just trying to scroll past it" bug.
   Applied at ALL screen sizes — even on desktop with touch screens. */
.js-plotly-plot,
.js-plotly-plot .main-svg,
.js-plotly-plot .plot-container,
.plotly,
.plot-container,
.stPlotlyChart,
iframe {
    touch-action: pan-x pan-y !important;
}
/* Belt-and-braces: also disable Plotly's drag selection on touch */
.js-plotly-plot .draglayer,
.js-plotly-plot .nsewdrag,
.js-plotly-plot .drag {
    touch-action: pan-x pan-y !important;
}
/* Iframe embeds (LookIntoBitcoin etc) must be width-100 */
iframe { max-width: 100% !important; box-sizing: border-box !important; }
/* Long code blocks / unbreakable text should wrap or scroll-within */
pre, code { white-space: pre-wrap !important; word-break: break-word !important; }
/* DataFrames + tables: scroll INSIDE the cell, not on the page */
[data-testid="stDataFrame"],
[data-testid="stTable"],
.stTable {
    max-width: 100% !important;
    overflow-x: auto !important;
}
/* Markdown text wraps long URLs / hashes */
.stMarkdown { word-wrap: break-word !important; overflow-wrap: anywhere !important; }

/* ═══════════════════════════════════════════════════════════════
   PHONE (≤768px) — stack columns, compress text, scrollable tabs
   ═══════════════════════════════════════════════════════════════ */
@media (max-width: 768px) {
    /* CARDS WITH INLINE STYLES — force them to be phone-sized
       The dashboard has many st.markdown cards with inline styles like
       padding:14px, font-size:28px, min-height:140px. Override them all
       so nothing sticks past the viewport edge. */
    [data-testid="stMarkdownContainer"] div[style*="padding:14px"],
    [data-testid="stMarkdownContainer"] div[style*="padding: 14px"] {
        padding: 10px 12px !important;
        min-height: auto !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
        overflow-wrap: anywhere !important;
        word-break: break-word !important;
    }
    /* Big numbers in cards — shrink from 28-30px → 22px so they fit */
    [data-testid="stMarkdownContainer"] div[style*="font-size:28px"],
    [data-testid="stMarkdownContainer"] div[style*="font-size: 28px"],
    [data-testid="stMarkdownContainer"] div[style*="font-size:30px"],
    [data-testid="stMarkdownContainer"] div[style*="font-size: 30px"] {
        font-size: 22px !important;
    }
    /* Mid-size text */
    [data-testid="stMarkdownContainer"] div[style*="font-size:22px"],
    [data-testid="stMarkdownContainer"] div[style*="font-size: 22px"],
    [data-testid="stMarkdownContainer"] div[style*="font-size:24px"],
    [data-testid="stMarkdownContainer"] div[style*="font-size: 24px"] {
        font-size: 18px !important;
    }
    [data-testid="stMarkdownContainer"] div[style*="font-size:20px"],
    [data-testid="stMarkdownContainer"] div[style*="font-size: 20px"] {
        font-size: 16px !important;
    }
    /* CRITICAL: Sticky-header negative-margin pull-out — on desktop it
       extends -20px past the .block-container padding (24px) for edge-to-edge
       look. On phone (where container padding is 0.6rem ≈ 10px), -20px pulls
       BEYOND the viewport edge, clipping the BTC Deploy block on the right.
       Constrain the pull to match phone padding so nothing escapes. */
    [data-testid="stMarkdownContainer"] div[style*="margin:-20px -20px"],
    [data-testid="stMarkdownContainer"] div[style*="margin: -20px -20px"],
    [data-testid="stMarkdownContainer"] div[style*="margin:0 -20px"],
    [data-testid="stMarkdownContainer"] div[style*="margin: 0 -20px"] {
        margin: 0 -10px 14px -10px !important;
        max-width: 100vw !important;
        overflow-x: hidden !important;
    }
    /* Masthead on phone: keep badge+wordmark on one line, hide the date stamp */
    [data-testid="stMarkdownContainer"] div[style*="letter-spacing:2.6px"] {
        letter-spacing: 1.6px !important;
        font-size: 8.5px !important;
    }
    [data-testid="stMarkdownContainer"] div[style*="white-space:nowrap"][style*="text-align:right"] {
        display: none !important;
    }
    /* Sticky-header BTC Deploy block — relax min-width on phone */
    [data-testid="stMarkdownContainer"] div[style*="min-width:140px"],
    [data-testid="stMarkdownContainer"] div[style*="min-width: 140px"] {
        min-width: 100% !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }
    /* Sticky-header child blocks — each takes full row, not side-by-side */
    [data-testid="stMarkdownContainer"] div[style*="display:flex"][style*="flex-wrap:wrap"] > div,
    [data-testid="stMarkdownContainer"] div[style*="display: flex"][style*="flex-wrap: wrap"] > div {
        flex: 0 0 100% !important;
        max-width: 100% !important;
        border-left: none !important;
        border-top: 1px solid #2a2d36 !important;
        box-sizing: border-box !important;
    }
    /* Verdict label (18px on desktop) → 16px on phone — keep readable */
    [data-testid="stMarkdownContainer"] span[style*="font-size:18px"],
    [data-testid="stMarkdownContainer"] span[style*="font-size: 18px"] {
        font-size: 16px !important;
    }
    /* All inline-style div tag content should fit phone width */
    [data-testid="stMarkdownContainer"] div {
        max-width: 100% !important;
        overflow-wrap: anywhere !important;
    }
    [data-testid="stMarkdownContainer"] span {
        word-break: break-word !important;
    }

    /* Tighter outer padding so we use every pixel */
    .block-container {
        padding: 0.4rem 0.6rem 4rem 0.6rem !important;
        max-width: 100vw !important;
        overflow-x: hidden !important;
        box-sizing: border-box !important;
    }

    /* Force EVERY first-level streamlit container to clamp to viewport */
    [data-testid="stVerticalBlock"],
    [data-testid="stVerticalBlockBorderWrapper"],
    [data-testid="element-container"] {
        max-width: 100% !important;
        overflow-x: hidden !important;
    }

    /* CRITICAL: stack st.columns() vertically on phone.
       Default Streamlit puts 3 cols side-by-side which crushes everything. */
    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        gap: 8px !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="column"],
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }

    /* Verdict hero — compact */
    .verdict-hero {
        padding: 14px 16px !important;
        border-left-width: 6px !important;
    }
    .verdict-text { font-size: 28px !important; }
    .verdict-sub { font-size: 13px !important; }
    .verdict-detail { font-size: 11px !important; }
    .verdict-title { font-size: 10px !important; letter-spacing: 1px !important; }

    /* Metric cards */
    .metric-block {
        padding: 8px 10px !important;
        margin-bottom: 6px !important;
    }
    .metric-value { font-size: 18px !important; }
    .metric-label { font-size: 9px !important; }
    .metric-sub { font-size: 10px !important; }

    /* Tabs container — contain scroll inside ITSELF so it doesn't push the page */
    .stTabs {
        max-width: 100% !important;
        overflow-x: hidden !important;
    }
    /* Tabs — horizontal scroll WITHIN the bar, no page-scroll, big thumb-tap targets */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px !important;
        overflow-x: auto !important;
        overflow-y: hidden !important;
        -webkit-overflow-scrolling: touch !important;
        flex-wrap: nowrap !important;
        scrollbar-width: none !important;
        padding-bottom: 4px !important;
        max-width: 100% !important;
    }
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 12px !important;
        font-size: 13px !important;
        min-width: max-content !important;
        flex-shrink: 0 !important;
        min-height: 44px !important;  /* Apple HIG min tap target */
    }

    /* Plotly charts — cap heights so they don't dominate the screen */
    .stPlotlyChart, .js-plotly-plot { max-height: 65vh !important; }
    .js-plotly-plot .plotly { max-height: 65vh !important; }

    /* Iframe embeds — same */
    iframe { max-height: 75vh !important; width: 100% !important; }

    /* Headings smaller on phone */
    h1 { font-size: 22px !important; line-height: 1.2 !important; }
    h2 { font-size: 17px !important; line-height: 1.25 !important; }
    h3 { font-size: 14px !important; line-height: 1.3 !important; }

    /* Section header tighter */
    .section-header {
        font-size: 13px !important;
        margin: 12px 0 6px 0 !important;
        padding-bottom: 4px !important;
    }

    /* Buttons — bigger tap targets */
    .stButton button, .stDownloadButton button {
        min-height: 44px !important;
        padding: 10px 16px !important;
        font-size: 14px !important;
        width: 100% !important;
    }

    /* Streamlit metric (st.metric) — smaller on phone */
    [data-testid="stMetric"] {
        padding: 6px 4px !important;
    }
    [data-testid="stMetricValue"] { font-size: 18px !important; }
    [data-testid="stMetricLabel"] { font-size: 10px !important; }
    [data-testid="stMetricDelta"] { font-size: 11px !important; }

    /* Sidebar — auto overlay (not push) on phone */
    section[data-testid="stSidebar"] {
        width: 85vw !important;
        max-width: 320px !important;
        min-width: 0 !important;
    }

    /* Data frames — horizontal scroll instead of crush */
    [data-testid="stDataFrame"] {
        font-size: 11px !important;
    }
    [data-testid="stDataFrame"] > div {
        overflow-x: auto !important;
    }

    /* Markdown body text */
    .stMarkdown p, .stMarkdown li { font-size: 13px !important; line-height: 1.4 !important; }
    .stMarkdown code { font-size: 11px !important; }

    /* Expander headers — bigger tap targets */
    .streamlit-expanderHeader, [data-testid="stExpander"] summary {
        min-height: 44px !important;
        padding: 10px 12px !important;
        font-size: 13px !important;
    }

    /* Captions */
    .stCaption, [data-testid="stCaptionContainer"] {
        font-size: 11px !important;
    }
}

/* ═══════════════════════════════════════════════════════════════
   TABLET (769-1024px) — light density reduction
   ═══════════════════════════════════════════════════════════════ */
@media (min-width: 769px) and (max-width: 1024px) {
    .block-container { padding: 0.8rem 1rem !important; }
    .verdict-text { font-size: 36px !important; }
    .verdict-sub { font-size: 14px !important; }
}

/* ═══════════════════════════════════════════════════════════════
   iOS-specific (Safari mobile)
   ═══════════════════════════════════════════════════════════════ */
@supports (-webkit-touch-callout: none) {
    /* Prevent zoom-on-focus by ensuring inputs are >= 16px font */
    input, select, textarea {
        font-size: 16px !important;
    }
    /* Smooth momentum scroll everywhere */
    body, .main, .block-container {
        -webkit-overflow-scrolling: touch;
    }
    /* Respect notch/safe-areas */
    .block-container {
        padding-left: max(0.6rem, env(safe-area-inset-left)) !important;
        padding-right: max(0.6rem, env(safe-area-inset-right)) !important;
        padding-bottom: max(4rem, env(safe-area-inset-bottom)) !important;
    }
}

/* ═══════════════════════════════════════════════════════════════
   DARK MODE — ensure system bar matches our dark theme
   ═══════════════════════════════════════════════════════════════ */
@media (prefers-color-scheme: dark) {
    html { background: #0e1117; }
}
</style>
""",
    unsafe_allow_html=True,
)


# === Cached state loaders (4h TTL, matches disk cache) ===
# All heavy panels wrap @st.cache_data with disk_cached so values survive
# streamlit restarts. On a cold start (disk + streamlit both empty), each
# function returns its compute. On a streamlit restart (disk warm), values
# load INSTANTLY from disk. On compute failure, last known good is reused.
from core.dashboard_cache import disk_cached  # noqa: E402


@st.cache_data(ttl=60)
@disk_cached("state_of_btc", ttl=86400)
def get_state(force_refresh: int = 0):
    from core.btc_prediction import state_of_btc
    return state_of_btc()


@st.cache_data(ttl=60)
@disk_cached("bottom_signals", ttl=86400)
def get_bottom_signals(force_refresh: int = 0):
    from core.btc_bottom_signals import all_bottom_signals
    return all_bottom_signals()


# Live BTC price — 60s TTL so the displayed price stays current
# (state cache stays 4h for heavy signals; price comes from this faster fetch)
@st.cache_data(ttl=60, show_spinner=False)
def get_live_btc_ticker() -> dict:
    """Full BTC ticker (last / percentage / low / high), 60s TTL shared page-wide.
    Tries several venues so it works from ANY region: Binance is geo-blocked on
    US cloud hosts (Streamlit Cloud), so US-accessible venues (Coinbase, Kraken)
    are tried first; Binance is the fallback (and works fine from NZ)."""
    import ccxt
    for ex_id, sym in (("kraken", "BTC/USD"), ("coinbase", "BTC/USD"),
                       ("binance", "BTC/USDT"), ("bitstamp", "BTC/USD")):
        try:
            t = getattr(ccxt, ex_id)({"timeout": 7000, "enableRateLimit": True}).fetch_ticker(sym)
            last = float(t.get("last") or 0)
            if last > 0:
                return {"last": last, "percentage": float(t.get("percentage") or 0),
                        "low": float(t.get("low") or 0), "high": float(t.get("high") or 0)}
        except Exception:
            continue
    return {"last": 0.0, "percentage": 0.0, "low": 0.0, "high": 0.0}


def get_live_btc_price() -> float:
    """Fast spot price — delegates to the shared cached ticker."""
    return get_live_btc_ticker().get("last", 0.0) or 0.0


# Cost-basis analytics — these hit CoinMetrics API.
# Cache 4h to match the underlying signal cadence.
# Macro Rotation Tracker (equities → BTC) — cached 30 min
# (changes more often than other signals as it depends on live SPY)
@st.cache_data(ttl=60)
@disk_cached("rotation", ttl=86400)
def cached_rotation():
    from core.btc_macro_rotation import rotation_phase
    return rotation_phase()


# Top Confirmation Scorecard — cached 30 min
# (hits FRED 7 times, ~28s cold; cache makes warm renders instant)
@st.cache_data(ttl=60)
@disk_cached("top_scorecard", ttl=86400)
def cached_top_scorecard():
    from core.btc_top_scorecard import (
        top_confirmation_scorecard, phased_exit_recommendation, historical_backtest
    )
    return {
        "scorecard": top_confirmation_scorecard(),
        "recommendation": phased_exit_recommendation(current_equity_pct=70),
        "backtest": historical_backtest(),
    }


# Early Rotation Signal — Druckenmiller/PTJ/Zulauf leading indicators.
# Pre-empts the standard top scorecard by 3-9 months. Routes equity to
# CASH (not BTC) when BTC isn't bottomed yet.
@st.cache_data(ttl=60)
@disk_cached("early_rotation", ttl=86400)
def cached_early_rotation():
    from core.btc_early_rotation import early_rotation_signal
    return early_rotation_signal(current_equity_pct=70, current_btc_pct=30, total_stake_nzd=130_000)


# Unified Decision Engine — top-tier macro layer + regime state machine +
# all scorecards + liquidity overlay + staging basket. Single source of truth.
@st.cache_data(ttl=60)
@disk_cached("unified_decision", ttl=86400)
def cached_unified_decision():
    from core.btc_unified_decision import unified_decision
    return unified_decision(current_equity_pct=70, current_btc_pct=30, total_stake_nzd=130_000)


# Top 1% Predictor Engine — calibrated framework
# (IC table + theme composites + BTC state + Kelly+vol-targeted sizing
#  + failure detection). 10-module quantitative architecture.
@st.cache_data(ttl=60)
@disk_cached("predictor_engine", ttl=86400)
def cached_predictor_engine():
    from core.predictor_engine import predictor_engine_state
    return predictor_engine_state(
        current_equity_pct=70, current_btc_pct=30, total_stake_nzd=130_000
    )


@st.cache_data(ttl=60)
@disk_cached("date_predictions", ttl=86400)
def cached_date_predictions():
    """Combined date predictions (indicator extrapolation + cycle 4 analog +
    macro calendar + convergence). Cached 4h."""
    from core.btc_date_predictions import (
        indicator_extrapolation, cycle_4_analog,
        macro_calendar, bottom_date_convergence,
    )
    return {
        "extrapolation":   indicator_extrapolation(),
        "cycle_4_analog":  cycle_4_analog(),
        "macro_calendar":  macro_calendar(180),
        "convergence":     bottom_date_convergence(),
    }


@st.cache_data(ttl=60)
@disk_cached("realized_price", ttl=86400)
def cached_realized_price():
    from core.btc_cost_basis import realized_price
    return realized_price()


@st.cache_data(ttl=60)
@disk_cached("sth_cost_basis", ttl=86400)
def cached_sth_cost_basis():
    from core.btc_cost_basis import sth_cost_basis
    return sth_cost_basis()


@st.cache_data(ttl=60)
@disk_cached("realized_cap_drawdown", ttl=86400)
def cached_realized_cap_drawdown_depth():
    from core.btc_cost_basis import realized_cap_drawdown_depth
    return realized_cap_drawdown_depth()


@st.cache_data(ttl=14400)
def cached_bottom_probability_distribution(price_bucket: int):
    """price_bucket is current price rounded to nearest $1000 — cache key."""
    from core.btc_cost_basis import bottom_probability_distribution
    return bottom_probability_distribution()


# Olson TA layer — fetches OHLCV, computes MACD/RSI/HA. Cache 1h
# (timeframes are 1w/3w so daily change is small).
@st.cache_data(ttl=60)
@disk_cached("olson", ttl=86400)
def cached_olson():
    from core.btc_jesse_olson import olson_combined_verdict
    return olson_combined_verdict()


# 90d candle chart data — cached 30 min (intraday price moves matter
# but the chart is 90 days so cadence is daily-ish).
@st.cache_data(ttl=60)
@disk_cached("ohlcv_90d", ttl=86400)
def cached_ohlcv_90d():
    from core import data
    return data.ohlcv_extended("BTC/USDT", days_back=90)


# Scorecard + trigger — cheap dict lookups, but state-dependent.
# Cache 5 min so they update faster than the heavy state cache.
@st.cache_data(ttl=300)
def cached_scorecard_and_trigger(_state):
    """Both wrapped together because they both read state. _state arg
    makes Streamlit invalidate when state changes."""
    from core.btc_bottom_scorecard import bottom_confirmation_scorecard
    from core.btc_etf_aware_trigger import etf_aware_bottom_trigger
    sc = bottom_confirmation_scorecard(_state, compute_breadth=False)  # hot path: no yfinance
    trigger = etf_aware_bottom_trigger(_state)
    return sc, trigger


# === Load state ===
try:
    state = get_state()
    bottom_sigs = get_bottom_signals()
except Exception as e:
    st.error(f"Failed to load prediction state: {e}")
    st.stop()


btc_price = state.get("btc_price", 0) or 73000
# Override with live price (60s cache) so display is current
# even when state cache is stale (4h TTL on heavy signals)
_live_price = get_live_btc_price()
if _live_price > 0:
    btc_price = _live_price

# Canonical regime — prefer the freshest source (unified_decision, 30min TTL)
# over state's 4h cache to keep all panels in sync.
def _canonical_regime() -> str:
    try:
        _r = cached_unified_decision().get("regime")
        if _r: return _r
    except Exception: pass
    return state.get("regime", "?")
regime = _canonical_regime()
horizons = state.get("horizons", {})
ensemble = state.get("ensemble", {})

# Cycle 5 ATH
CYCLE5_PEAK_PRICE = 124659
CYCLE5_PEAK_DATE = datetime(2025, 10, 6).date()
pct_from_ath = (btc_price / CYCLE5_PEAK_PRICE - 1) * 100


# === Compute derived analytics ===
from core.btc_bottom_scorecard import bottom_confirmation_scorecard
from core.halving_clock import (
    current_halving_position, pattern_projected_targets,
    cycle_phase_from_halving_day, MEAN_DAYS_TO_PEAK, MEAN_DAYS_TO_BOTTOM,
    PEAK_STD_DEV, BOTTOM_STD_DEV, HISTORICAL,
)
from core.btc_cost_basis import (
    realized_price as _rp_fn, sth_cost_basis as _sth_fn,
    realized_cap_drawdown_depth as _rcd_fn,
    bottom_probability_distribution as _pdb_fn,
)

# Use cached wrappers so each refresh hits Streamlit cache, not the network
sc, trigger = cached_scorecard_and_trigger(state)
pos = current_halving_position()
phase_info = cycle_phase_from_halving_day(pos["days_post_halving"])
ppt = pattern_projected_targets(btc_price)
rp = cached_realized_price()
sth = cached_sth_cost_basis()
rcd = cached_realized_cap_drawdown_depth()
pdb = cached_bottom_probability_distribution(int(btc_price / 1000))

verdict_label = trigger["verdict_label"]
verdict_color = trigger["color"]
verdict_sub = trigger["rationale"][:90] + ("..." if len(trigger["rationale"]) > 90 else "")


def _find_signal(name: str, *cats: str) -> Optional[dict]:
    """Find signal by name in given categories (default all)."""
    search = cats or ("onchain", "fundamentals", "flows", "technical",
                       "macro", "sentiment", "derivatives", "liquidations",
                       "cycle", "cycle_outlook", "options_adv", "regime_models")
    for cat in search:
        d = state.get("signals", {}).get(cat, {})
        if name in d and isinstance(d[name], dict):
            return d[name]
    return None


def metric_card(label: str, value: str, sub: str = "", accent: str = None) -> str:
    """Generate a compact metric card HTML."""
    accent = accent or C["accent"]
    return (
        f"<div style='background:{C['card_bg']}; padding:12px 14px; border-radius:6px; "
        f"border-left:3px solid {accent}; margin-bottom:8px;'>"
        f"<div style='font-size:10px; color:#888; text-transform:uppercase; letter-spacing:0.5px;'>{label}</div>"
        f"<div style='font-size:22px; font-weight:700; color:#fff; line-height:1.1;'>{value}</div>"
        f"<div style='font-size:12px; color:#aaa; margin-top:2px;'>{sub}</div>"
        f"</div>"
    )


def _seg_bar(n, total, color, height: int = 9) -> str:
    """Segmented progress bar HTML: `n` of `total` cells filled with `color`,
    the rest muted grey. A glanceable 'X of N' visual for scorecards."""
    try:
        total = max(1, int(total))
        n = max(0, min(int(n), total))
    except Exception:
        return ""
    cells = "".join(
        f"<div style='flex:1; height:{height}px; border-radius:2px; "
        f"background:{color if i < n else '#2a2d36'};'></div>"
        for i in range(total)
    )
    return f"<div style='display:flex; gap:2px; margin-top:8px;'>{cells}</div>"


def _crit_tiles(criteria, met_color: str = "#22c55e", min_width: int = 200) -> str:
    """Met/unmet checklist as a colored tile grid (green = met, grey = not)."""
    cells = []
    for c in (criteria or []):
        met = bool(c.get("met"))
        bg = met_color if met else "#2a2d36"
        ico = "🔥" if met else "○"
        lbl = c.get("label") or "?"
        sub = (c.get("status") or "")[:64]
        cells.append(
            f"<div style='flex:1 1 calc(50% - 8px); min-width:{min_width}px; padding:7px 10px; "
            f"border-radius:6px; background:{bg}22; border-left:3px solid {bg};'>"
            f"<div style='font-size:12px; color:#eee;'>{ico} {lbl}</div>"
            f"<div style='font-size:10px; color:#999; line-height:1.3;'>{sub}</div></div>")
    return f"<div style='display:flex; flex-wrap:wrap; gap:8px;'>{''.join(cells)}</div>"


def _diverging_fig(labels, values, height=None):
    """Horizontal diverging bar (green ≥0 right / red <0 left), sorted ascending by value."""
    pairs = sorted(zip(list(values), list(labels)), key=lambda p: p[0])
    vals = [p[0] for p in pairs]
    labs = [p[1] for p in pairs]
    cols = ["#22c55e" if v >= 0 else "#ef4444" for v in vals]
    vmax = max((abs(v) for v in vals), default=1) or 1
    fig = go.Figure(go.Bar(
        x=vals, y=labs, orientation="h",
        marker=dict(color=cols, line=dict(color="#0e1117", width=1)),
        text=[f"{v:+.2f}" for v in vals], textposition="outside",
        textfont=dict(size=11, color="#cccccc"), hoverinfo="skip"))
    fig.add_vline(x=0, line=dict(color="#888888", width=1))
    fig.update_layout(**CHART_LAYOUT, height=height or max(220, 40 * len(vals) + 50),
                      xaxis=dict(range=[-vmax * 1.35, vmax * 1.35], gridcolor="#2a2d34", zeroline=False),
                      yaxis=dict(automargin=True), showlegend=False)
    return fig


# === ANYA UX HELPERS ===
def _age_badge(panel_key: str) -> str:
    """Inline HTML badge showing age of cached panel — green/yellow/red by freshness."""
    try:
        from core.dashboard_cache import cache_age_seconds
        age = cache_age_seconds(panel_key)
    except Exception:
        return ""
    if age is None:
        return "<span style='font-size:10px;color:#888;'>● live</span>"
    mins = max(0, int(age / 60))    # clamp negative (clock skew)
    if mins < 30:
        c = "#22c55e"
    elif mins < 240:
        c = "#f0b90b"
    else:
        c = "#ef4444"
    if mins < 60:
        txt = f"{mins}m ago"
    else:
        hrs = mins // 60
        txt = f"{hrs}h ago"
    return f"<span style='font-size:10px;color:{c};font-weight:600;'>● updated {txt}</span>"


# Centralized verdict-level → color (avoid drift between renders)
def _top_color(level: str) -> str:
    """Top scorecard verdict color — whitelist, not fallthrough.
    Unknown level returns neutral, NOT red (was a real bug)."""
    if level in ("HOLD", "DORMANT", "LOW_RISK", "QUIET"):
        return "#22c55e"
    if level in ("WATCH", "BUILDING"):
        return "#f0b90b"
    if level in ("TRIM_25", "SCALE_OUT_25", "SCALE_OUT_50", "EXIT_75",
                  "EXIT_100", "CONFIRMED_TOP"):
        return "#ef4444"
    return "#888"  # unknown -> neutral (safer than red)


def _bottom_color(level: str) -> str:
    """Bottom scorecard verdict color — whitelist, not fallthrough."""
    if level in ("STRONG_BUY", "DEEP_VALUE", "EXTREME", "ACCUMULATE",
                  "BUY_ZONE", "DCA"):
        return "#22c55e"
    if level in ("WATCH", "WARMING"):
        return "#f0b90b"
    if level in ("HOLD", "DORMANT", "QUIET"):
        return "#888"
    return "#888"  # unknown -> neutral


def _dormant_status(n_met: int, n_total: int, kind: str) -> str:
    """Per-scorecard 'is this dormant or active?' message under the count.

    kind: 'top' or 'bottom'. Different language for each.
    Per Anya: 'Without this, the user thinks the dashboard is broken when
    numbers look low. Make the dormant-state intentional.'
    """
    pct = (n_met / n_total) if n_total else 0
    if kind == "top":
        if n_met == 0:
            return ("✓ <b style='color:#22c55e;'>DORMANT</b> — no cycle-top risk firing. "
                    "This is correct in early-cycle / accumulation phases.")
        if pct < 0.25:
            return ("✓ <b style='color:#22c55e;'>LOW RISK</b> — top signals quiet. "
                    "Will activate when 4+ criteria fire (~25% of board).")
        if pct < 0.5:
            return ("⚠️ <b style='color:#f0b90b;'>BUILDING</b> — top signals are firing. "
                    "Stay alert; consider trimming as criteria accumulate.")
        if pct < 0.75:
            return ("🔻 <b style='color:#ef4444;'>HIGH RISK</b> — multiple top signals firing. "
                    "Phased exit strongly recommended.")
        return ("🚨 <b style='color:#ef4444;'>CONFIRMED TOP</b> — overwhelming cycle-top "
                "evidence. Exit core holding NOW.")
    else:  # bottom
        if n_met == 0:
            return ("✓ <b style='color:#888;'>DORMANT</b> — no bottom-buy signals firing. "
                    "Will activate in bear market when 4+ criteria fire.")
        if pct < 0.25:
            return ("✓ <b style='color:#888;'>QUIET</b> — too early. Wait for 4+ criteria.")
        if pct < 0.5:
            return ("⚠️ <b style='color:#f0b90b;'>WARMING</b> — value building. "
                    "Begin DCA accumulation at 6+ criteria.")
        if pct < 0.75:
            return ("🔼 <b style='color:#22c55e;'>BUY ZONE</b> — multiple bottom signals firing. "
                    "Deploy planned crypto allocation.")
        return ("🚨 <b style='color:#22c55e;'>DEEP VALUE</b> — generational bottom evidence. "
                "Deploy aggressively, including reserve capital.")


# ╔══════════════════════════════════════════════════════════════╗
# ║                        SIDEBAR                                ║
# ╚══════════════════════════════════════════════════════════════╝
with st.sidebar:
    # VERDICT BADGE (always visible — ETF-aware trigger)
    st.markdown(
        f"<div style='background:{verdict_color}; padding:18px 16px; border-radius:8px; "
        f"text-align:center; margin-bottom:14px;'>"
        f"<div style='font-size:10px; color:#fff; opacity:0.85; text-transform:uppercase; letter-spacing:1px;'>"
        f"Trigger {trigger['trigger_id']}</div>"
        f"<div style='font-size:24px; font-weight:800; color:#fff; line-height:1.1;'>{verdict_label}</div>"
        f"<div style='font-size:11px; color:#fff; opacity:0.85; margin-top:6px; line-height:1.3;'>"
        f"Scorecard {sc['n_met']}/{sc['n_total']} + ETF {trigger['etf_status'].replace('_', ' ').lower()}"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    # KEY METRICS
    st.markdown(
        metric_card("BTC", f"${btc_price:,.0f}", f"{pct_from_ath:+.1f}% from ATH",
                     C["accent"]),
        unsafe_allow_html=True,
    )

    days_to_bot = pos["days_to_pattern_bottom"]
    st.markdown(
        metric_card(
            "Pattern bottom (T-)",
            f"{days_to_bot}d" if days_to_bot > 0 else f"{abs(days_to_bot)}d ago",
            pos["projected_bottom_date"].strftime("%b %d, %Y"),
            C["lth"],
        ),
        unsafe_allow_html=True,
    )

    sc_color = (C["deep_bull"] if sc["n_met"] >= 6 else
                 C["bull"] if sc["n_met"] >= 4 else
                 C["neutral"] if sc["n_met"] >= 2 else C["bear"])
    st.markdown(
        metric_card("Scorecard", f"{sc['n_met']}/{sc['n_total']}",
                     sc["verdict_level"].replace("_", " "), sc_color),
        unsafe_allow_html=True,
    )

    if rcd and not rcd.get("error"):
        rcd_color = (C["deep_bull"] if rcd["current_drawdown_pct"] < -20 else
                      C["bull"] if rcd["current_drawdown_pct"] < -15 else
                      C["neutral"] if rcd["current_drawdown_pct"] < -10 else C["bear"])
        st.markdown(
            metric_card("Realized Cap drawdown",
                         f"{rcd['current_drawdown_pct']:+.1f}%",
                         f"need -15% for bottom zone", rcd_color),
            unsafe_allow_html=True,
        )

    st.divider()

    if st.button("Force refresh", width='stretch'):
        st.cache_data.clear()
        for f in [".btc_prediction_cache.json", ".btc_bottom_signals_cache.json"]:
            (REPO_ROOT / f).unlink(missing_ok=True)
        st.rerun()

    st.caption(f"UTC: {datetime.now(timezone.utc).strftime('%H:%M:%S')}")
    st.caption("Cache: 4h • Manual refresh (Ctrl+R / F5)")
    st.markdown(
        "<div style='margin-top:12px; font-size:11px; color:#888;'>"
        "<a href='http://localhost:8510' style='color:#4a90e2;'>Main rig (8510)</a> · "
        "<a href='http://localhost:8511' style='color:#4a90e2;'>This (8511)</a><br>"
        "Weekly: <code>python -m core.btc_weekly_report</code>"
        "</div>",
        unsafe_allow_html=True,
    )


# ╔══════════════════════════════════════════════════════════════╗
# ║  MASTHEAD — Dave's BTC Dashboard (renders FIRST, above all)    ║
# ╚══════════════════════════════════════════════════════════════╝
try:
    from datetime import datetime as _dt_mh
    from zoneinfo import ZoneInfo as _ZI_mh
    # NZ time — this renders on the Streamlit Cloud server (UTC), so a naive
    # now() would show yesterday's date all NZ morning.
    _mh_date = _dt_mh.now(_ZI_mh("Pacific/Auckland")).strftime("%a %d %b %Y")
except Exception:
    _mh_date = ""
st.markdown(
    f"<div style='display:flex; align-items:center; justify-content:space-between; "
    f"padding:6px 2px 12px 2px; margin:-6px 0 12px 0; "
    f"border-bottom:1px solid #232936;'>"
    # Left: orange ₿ badge + two-line wordmark
    f"<div style='display:flex; align-items:center; gap:13px;'>"
    f"<div style='width:40px; height:40px; min-width:40px; border-radius:11px; "
    f"background:linear-gradient(135deg, #f7931a 0%, #c2410c 100%); "
    f"box-shadow:0 2px 10px rgba(247,147,26,0.35); "
    f"display:flex; align-items:center; justify-content:center; "
    f"font-size:22px; font-weight:800; color:#fff;'>&#8383;</div>"
    f"<div>"
    f"<div style='font-size:21px; font-weight:800; color:#fff; "
    f"letter-spacing:0.2px; line-height:1.05; font-family:Inter, system-ui, sans-serif;'>"
    f"Dave&rsquo;s <span style='background:linear-gradient(90deg, #f7931a, #fbbf24); "
    f"-webkit-background-clip:text; background-clip:text; color:transparent;'>"
    f"BTC Dashboard</span></div>"
    f"<div style='font-size:9.5px; color:#6b7280; text-transform:uppercase; "
    f"letter-spacing:2.6px; margin-top:4px; font-weight:600;'>"
    f"Cycle Intelligence &middot; Rotation System</div>"
    f"</div></div>"
    # Right: quiet date stamp (hidden on phone via mobile CSS)
    f"<div style='font-size:10px; color:#4b5563; letter-spacing:1px; "
    f"text-align:right; white-space:nowrap; padding-left:10px;'>{_mh_date}</div>"
    f"</div>",
    unsafe_allow_html=True,
)


# ╔══════════════════════════════════════════════════════════════╗
# ║   STICKY HEADER — BTC price + 24h + master verdict + changes  ║
# ║   (Swift insistence: single source of truth for price)        ║
# ╚══════════════════════════════════════════════════════════════╝
try:
    _live_px = get_live_btc_price()
    if _live_px <= 0: _live_px = btc_price  # fallback
    _ud_hdr = cached_unified_decision()
    _pe_hdr = cached_predictor_engine()
    _hdr_regime = _ud_hdr.get("regime", "?")
    _hdr_top_z = _pe_hdr.get("decision_composites", {}).get("top", 0)
    _hdr_bottom_z = _pe_hdr.get("decision_composites", {}).get("bottom", 0)
    _hdr_top_action = _ud_hdr.get("scorecards", {}).get("top", {}).get("action", "?")
    _hdr_pct_from_ath = (_live_px / CYCLE5_PEAK_PRICE - 1) * 100
    # 3-color palette — single source of truth (was a duplicate dict)
    PAL = C
    # Regime to color
    _reg_color = PAL["green"] if _hdr_regime == "RISK_ON" else (
        PAL["yellow"] if _hdr_regime == "LATE_CYCLE" else PAL["red"])
    # Get 24h change
    _24h_ok = False
    try:
        _t = get_live_btc_ticker()
        _24h_pct = float(_t.get("percentage", 0))
        _24h_low = float(_t.get("low", 0))
        _24h_high = float(_t.get("high", 0))
        _24h_ok = _t.get("last", 0) > 0
    except Exception:
        _24h_pct = 0; _24h_low = 0; _24h_high = 0
    # If fetch failed, render dash instead of lying with +0.00%
    if not _24h_ok:
        _24h_pct_str = "—"
        _24h_range_str = "— · —"
        _24h_color = PAL["muted"]
    else:
        _24h_pct_str = f"{_24h_pct:+.2f}%"
        _24h_range_str = f"${_24h_low:,.0f}-${_24h_high:,.0f}"
        _24h_color = PAL["green"] if _24h_pct > 0 else PAL["red"]
    # Anya redesign: VERDICT first (colored BG), then context. Glance from
    # across the room and you SEE the color of the action.
    st.markdown(
        f"<div style='display:flex; flex-wrap:wrap; gap:0; align-items:stretch; "
        f"margin:0 -20px 14px -20px; border-bottom:2px solid #2a2d36; "
        f"font-size:13px;'>"
        # BTC DEPLOY block (etf_aware_bottom_trigger) — colored BG, biggest, leftmost
        # Answers: "Should I deploy crypto reserve into BTC RIGHT NOW?"
        f"<div style='background:{verdict_color}; padding:13px 22px; "
        f"display:flex; flex-direction:column; justify-content:center; min-width:155px;'>"
        f"<span style='font-size:11px; color:#000; opacity:0.78; text-transform:uppercase; "
        f"letter-spacing:1.5px; font-weight:700;'>BTC Deploy</span>"
        f"<span style='font-size:25px; font-weight:800; color:#000; line-height:1.05; "
        f"margin-top:3px;'>{verdict_label}</span>"
        f"</div>"
        # EQUITY ACTION block (top_confirmation_scorecard action) — independent decision
        # Answers: "Should I be trimming/exiting stocks RIGHT NOW?"
        f"<div style='background:#0e1117; padding:13px 22px; "
        f"display:flex; flex-direction:column; justify-content:center; border-left:1px solid #2a2d36;'>"
        f"<span style='font-size:11px; color:#888; text-transform:uppercase; letter-spacing:1px;'>Equity Action</span>"
        f"<span style='font-size:20px; font-weight:700; color:#fff; line-height:1.05; margin-top:3px;'>"
        f"{_hdr_top_action}</span>"
        f"</div>"
        # Price block
        f"<div style='background:#0e1117; padding:13px 22px; "
        f"display:flex; flex-direction:column; justify-content:center; border-left:1px solid #2a2d36;'>"
        f"<span style='font-size:11px; color:#888; text-transform:uppercase; letter-spacing:1px;'>BTC</span>"
        f"<div style='display:flex; gap:6px; align-items:baseline; margin-top:3px;'>"
        f"<span style='font-size:22px; font-weight:800; color:white;'>${_live_px:,.0f}</span>"
        f"<span style='font-size:13px; color:{_24h_color}; font-weight:600;'>{_24h_pct_str}</span>"
        f"</div></div>"
        # Context cluster (regime + ath + top/bot z) — desktop only via wrap
        f"<div style='background:#0e1117; padding:13px 22px; "
        f"display:flex; gap:20px; align-items:center; flex:1; border-left:1px solid #2a2d36;"
        f"flex-wrap:wrap;'>"
        f"<div style='color:#888; font-size:13px;'>Regime <b style='color:{_reg_color};'>"
        f"{_hdr_regime.replace('_',' ')}</b></div>"
        f"<div style='color:#888; font-size:13px;'>From ATH <b style='color:#ccc;'>"
        f"{_hdr_pct_from_ath:+.1f}%</b></div>"
        f"<div style='color:#888; font-size:13px;'>Top z <b style='color:#ccc;'>"
        f"{_hdr_top_z:+.2f}</b></div>"
        f"<div style='color:#888; font-size:13px;'>Bottom z <b style='color:#ccc;'>"
        f"{_hdr_bottom_z:+.2f}</b></div>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
except Exception as _e:
    # Surface the failure instead of silently hiding the header.
    st.caption(f"⚠️ Header — temporarily unavailable")

# ╔══════════════════════════════════════════════════════════════╗
# ║                          TABS                                 ║
# ╚══════════════════════════════════════════════════════════════╝
# ╔══════════════════════════════════════════════════════════════════════╗
# ║ REGION MAP (auto-generated 2026-07-04b — post fresh-eyes review)      ║
# ║ Line numbers are exact (map block itself = 34 lines, already counted).║
# ║ Dependency audit: regions are self-contained (preamble + own locals), ║
# ║ so any region can be extracted to a module without breaking others.   ║
# ║ Add NEW panels to the correct home tab — not wherever is convenient.  ║
# ║ Today renders slots first: _today_hero, _today_guard, then appends.   ║
# ╠══════════════════════════════════════════════════════════════════════╣
# ║  line  1225  tab_signals    -> Signals                  (   4 lines)  ║
# ║  line  1229  tab_today      -> Today                    (  18 lines)  ║
# ║  line  1247  tab_gurus      -> Research                 (   6 lines)  ║
# ║  line  1253  _today_guard   -> Today (slot 2)           (  37 lines)  ║
# ║  line  1290  tab_research   -> Research                 ( 424 lines)  ║
# ║  line  1714  _today_hero    -> Today (slot 1)           (  49 lines)  ║
# ║  line  1763  tab_research   -> Research                 (  77 lines)  ║
# ║  line  1840  tab_today      -> Today                    (  42 lines)  ║
# ║  line  1882  tab_playbook   -> Playbook                 ( 213 lines)  ║
# ║  line  2095  tab_research   -> Research                 ( 219 lines)  ║
# ║  line  2314  tab_signals    -> Signals                  ( 967 lines)  ║
# ║  line  3281  tab_research   -> Research                 (  79 lines)  ║
# ║  line  3360  tab_signals    -> Signals                  (1167 lines)  ║
# ║  line  4527  _unified_top   -> Signals (renders FIRST)  ( 275 lines)  ║
# ║  line  4802  tab_signals    -> Signals                  ( 253 lines)  ║
# ║  line  5055  tab_playbook   -> Playbook                 ( 589 lines)  ║
# ║  line  5644  tab_simple     -> Today                    ( 256 lines)  ║
# ║  line  5900  tab_cycle      -> Research                 ( 473 lines)  ║
# ║  line  6373  tab_onchain    -> Research                 ( 273 lines)  ║
# ║  line  6646  tab_technical  -> Research                 ( 160 lines)  ║
# ║  line  6806  tab_detail     -> Research                 ( 111 lines)  ║
# ║  line  6917  tab_charts     -> Signals                  ( 109 lines)  ║
# ║  line  7026  tab_scorecards -> Research                 ( 126 lines)  ║
# ║  line  7152  tab_macro      -> Signals                  ( 291 lines)  ║
# ║  line  7443  tab_exit       -> Playbook                 ( 176 lines)  ║
# ╚══════════════════════════════════════════════════════════════════════╝
# 2026-07-04 guru restructure: 8 tabs -> 4.
#   Today    = verdict + cycle position + Simpleton summary
#   Signals  = the live engine (Unified Decision Engine renders FIRST)
#   Playbook = ALL execution: rotation trigger, 4-trigger matrix, checklist, tax
#   Research = backtests, validation, Swift suite, guru feeds, detail
# No content deleted - every panel re-homed. Legacy tab vars are aliased so
# existing `with tab_x:` blocks render into their new homes.
(tab_today, tab_signals, tab_playbook, tab_research) = st.tabs([
    "🎯 Today",
    "📡 Signals",
    "🚪 Playbook",
    "🔬 Research",
])
# Reserve the TOP of Signals for the Unified Decision Engine: computed late in
# the script, rendered first via this container slot.
with tab_signals:
    _unified_top = st.container()
# Today renders in slot order: hero first, then the campaign guardrail -
# regardless of where their code executes in the file.
with tab_today:
    _today_hero = st.container()
    _today_guard = st.container()

# Legacy tab-variable map (old `with tab_x:` blocks -> new homes)
tab_simple     = tab_today
tab_overview   = tab_today      # hero banner; rest of its body re-homed inline below
tab_charts     = tab_signals
tab_macro      = tab_signals
tab_exit       = tab_playbook
tab_scorecards = tab_research
tab_detail     = tab_research
tab_gurus      = tab_research
tab_cycle      = tab_research   # cycle-math deep dive
tab_onchain    = tab_research   # on-chain detail
tab_technical  = tab_research   # Olson technical detail

# ── 👁 GURU PANEL tab — legends scored off the live signals ───────────────────
with tab_gurus:
    st.markdown("### 👁 Guru Panel — what would the legends think?")
    st.caption("Legendary crypto & macro top/bottom callers, scored off your LIVE signals. "
               "Framework-based sanity check — not investment advice.")


with _today_guard:   # <- 2026-07-04 review fix: daily guardrail belongs on Today
    # ── 🧭 Campaign thesis health — pre-registered kill-criteria ──────────────
    try:
        @st.cache_data(ttl=1800, show_spinner=False)
        def _thesis_check():
            from core.campaign_kill_criteria import campaign_thesis_check
            return campaign_thesis_check()
        _tc = _thesis_check() or {}
        _tcol = _tc.get("color", "#22c55e")

        def _scol(stt):
            return ("#ef4444" if stt == "TRIPPED" else
                    "#f0b90b" if stt == "WARNING" else "#22c55e")
        _rows = "".join(
            f"<div style='display:flex; gap:10px; align-items:baseline; padding:5px 0; "
            f"border-top:1px solid #20242c;'>"
            f"<span style='font-size:10px; font-weight:700; min-width:74px; "
            f"color:{_scol(c['status'])};'>{c['status']}</span>"
            f"<div><div style='font-size:12px; color:#ddd; font-weight:600;'>{c['name']}</div>"
            f"<div style='font-size:11px; color:#888;'>{c.get('current', '')}</div>"
            f"<div style='font-size:10px; color:#777; margin-top:1px;'>{c['detail']}</div></div></div>"
            for c in _tc.get("criteria", []))
        st.markdown(
            f"<div style='padding:16px 20px; border-radius:10px; background:#13161c; "
            f"border-left:6px solid {_tcol}; margin-bottom:14px;'>"
            f"<div style='display:flex; justify-content:space-between; align-items:baseline;'>"
            f"<span style='font-size:11px; color:#888; text-transform:uppercase; letter-spacing:1.5px; "
            f"font-weight:700;'>🧭 Campaign thesis health · what would prove me wrong</span>"
            f"<span style='font-size:20px; font-weight:800; color:{_tcol};'>{_tc.get('verdict', '?')}</span>"
            f"</div>{_rows}"
            f"<div style='font-size:10px; color:#777; margin-top:8px;'>Pre-registered falsification "
            f"criteria — decided in advance so a green dashboard can't lull you. Not advice.</div>"
            f"</div>", unsafe_allow_html=True)
    except Exception:
        pass


with tab_research:   # back to Guru Panel content
    # ── 🧭 GURU-GRADE DECISION UPGRADES (2026-06-13) ───────────────────────────
    # Theme-breadth gate · banded/time deploy · regime tag · BTC-vs-equity · ETF
    # flow quality. Surfaces the surgical upgrades that harden the framework for
    # an ETF-mutated cycle without changing the architecture. Not advice.
    try:
        from core.dashboard_cache import get_cached as _gc
        # READ pre-warmed disk caches only — never compute on render. The background
        # precompute task fills regime_tag / btc_equity_relval / etf_flow_quality /
        # bottom_confirmation (each hits yfinance/SPX/QQQ/DXY and was the load lag).
        _rg = _gc("regime_tag") or {}
        _rv = _gc("btc_equity_relval") or {}
        _eq = _gc("etf_flow_quality") or {}
        _bd = _gc("bottom_confirmation") or {}
        # 2026-07-07 signals audit: rotation_check's writer was removed ~Jun 12
        # (orphaned panel, 24 days stale) — rotation_trigger is the live panel
        # carrying the identical deploy_plan, refreshed every precompute cycle.
        _plan = (_gc("rotation_trigger") or {}).get("deploy_plan") or {}
        if not any([_rg, _rv, _eq, _bd]):
            st.caption("🧭 Guru-grade upgrades — warming up (first precompute cycle)…")

        st.markdown(
            f"<div style='padding:14px 18px; border-radius:10px; background:#13161c; "
            f"border-left:6px solid #4a90e2; margin-bottom:12px;'>"
            f"<div style='font-size:11px; color:#888; text-transform:uppercase; letter-spacing:1.2px; "
            f"font-weight:700;'>🧭 Regime &amp; playbook</div>"
            f"<div style='font-size:20px; font-weight:800; color:#cfe3ff;'>{html.escape(str(_rg.get('regime','?')))}</div>"
            f"<div style='font-size:12px; color:#aaa; margin-top:4px;'>{html.escape(str(_rg.get('detail','')))}</div>"
            f"<div style='font-size:11px; color:#888; margin-top:4px;'>Closest historical analog: "
            f"<b style='color:#ccc;'>{html.escape(str(_rg.get('analog','?')))}</b> "
            f"({html.escape(str(_rg.get('analog_confidence','?')))} confidence)</div></div>",
            unsafe_allow_html=True)

        _themes = _bd.get("themes", []) or []
        _chips = "".join(
            f"<span style='display:inline-block; margin:2px 4px 2px 0; padding:2px 8px; border-radius:10px; "
            f"font-size:10px; background:{'#15351f' if t.get('met') else '#241c16'}; "
            f"color:{'#22c55e' if t.get('met') else '#888'}; "
            f"border:1px solid {'#22c55e55' if t.get('met') else '#333'};'>"
            f"{'✓' if t.get('met') else '·'} {html.escape(str(t.get('label','')))} "
            f"<span style='opacity:.6'>({html.escape(str(t.get('etf','')))})</span></span>"
            for t in _themes)
        _dl = _bd.get("deploy_level", "?")
        _dlcol = {"DEPLOY": "#22c55e", "SCALE_IN": "#f0b90b", "EARLY": "#f0b90b", "WAIT": "#888"}.get(_dl, "#888")
        st.markdown(
            f"<div style='padding:14px 18px; border-radius:10px; background:#13161c; "
            f"border-left:6px solid {_dlcol}; margin-bottom:12px;'>"
            f"<div style='display:flex; justify-content:space-between; align-items:baseline; flex-wrap:wrap;'>"
            f"<span style='font-size:11px; color:#888; text-transform:uppercase; letter-spacing:1.2px; "
            f"font-weight:700;'>🎯 Deploy gate · theme breadth (not raw count)</span>"
            f"<span style='font-size:15px; font-weight:800; color:{_dlcol};'>{html.escape(str(_bd.get('deploy_action','?')))}</span></div>"
            f"<div style='margin-top:8px;'>{_chips}</div>"
            f"<div style='font-size:11px; color:#999; margin-top:6px;'>{html.escape(str(_bd.get('breadth_summary','')))} "
            f"&middot; full deploy needs <b>4/6 themes + a price-turn</b>.</div>"
            # 2026-07-08 gate audit: show met vs FIRM vs mechanisms so a fragile
            # threshold-nick reading (e.g. 5/10 where 2 are hair-over-the-line and
            # the premium/hashrate pairs double-count) is transparent.
            f"<div style='font-size:11px; color:#c9a227; margin-top:6px;'>"
            f"Raw <b>{_bd.get('n_met','?')}/{_bd.get('n_total','?')}</b> met, but only "
            f"<b>{_bd.get('n_firm','?')} firm</b> ({_bd.get('n_marginal',0)} marginal, within "
            f"threshold buffer) &middot; <b>{_bd.get('n_mechanisms_firm','?')}/"
            f"{_bd.get('n_mechanisms_met','?')}</b> firm INDEPENDENT mechanisms — the "
            f"capital gate keys off firm mechanisms, so it won't flicker on daily BTC noise."
            f"</div></div>",
            unsafe_allow_html=True)

        if _plan:
            _frac = _plan.get("fraction_pct", 0)
            _fcol = "#22c55e" if _frac >= 100 else "#f0b90b" if _frac > 0 else "#888"
            st.markdown(
                f"<div style='padding:12px 18px; border-radius:10px; background:#13161c; "
                f"border-left:6px solid {_fcol}; margin-bottom:12px;'>"
                f"<div style='font-size:11px; color:#888; text-transform:uppercase; letter-spacing:1.2px; "
                f"font-weight:700;'>💧 Banded scale-in plan (BTC leg)</div>"
                f"<div style='font-size:22px; font-weight:800; color:{_fcol};'>Deploy {_frac}%</div>"
                f"<div style='font-size:12px; color:#aaa; margin-top:2px;'>{html.escape(str(_plan.get('reason','')))}</div>"
                f"<div style='font-size:10px; color:#777; margin-top:4px;'>"
                f"{_plan.get('weeks_in_band',0)}w in the $45-58k band &middot; "
                f"time-deploy {'ARMED' if _plan.get('time_deploy') else 'off'} — "
                f"the grind-proofing tranche: deploy on time if the band holds past the cycle window "
                f"even without a capitulation signal.</div></div>",
                unsafe_allow_html=True)

        _rvs = _rv.get("score") or 0
        _rvcol = "#22c55e" if _rvs >= 66 else "#f0b90b" if _rvs >= 33 else "#ef4444"
        _eql = str(_eq.get("label", "?"))
        _eqcol = ("#22c55e" if "REAL" in _eql else
                  "#ef4444" if ("CARRY" in _eql or "DISTRIB" in _eql) else "#888")
        st.markdown(
            f"<div style='display:flex; gap:10px; margin-bottom:4px; flex-wrap:wrap;'>"
            f"<div style='flex:1; min-width:200px; padding:12px 16px; border-radius:10px; "
            f"background:#13161c; border-left:5px solid {_rvcol};'>"
            f"<div style='font-size:10px; color:#888; text-transform:uppercase;'>BTC vs equities (rotation premise)</div>"
            f"<div style='font-size:15px; font-weight:800; color:{_rvcol};'>{html.escape(str(_rv.get('tier','?')))}</div>"
            f"<div style='font-size:10px; color:#999; margin-top:2px;'>{html.escape(str(_rv.get('detail','')))}</div></div>"
            f"<div style='flex:1; min-width:200px; padding:12px 16px; border-radius:10px; "
            f"background:#13161c; border-left:5px solid {_eqcol};'>"
            f"<div style='font-size:10px; color:#888; text-transform:uppercase;'>ETF flow quality</div>"
            f"<div style='font-size:15px; font-weight:800; color:{_eqcol};'>{html.escape(_eql)}</div>"
            f"<div style='font-size:10px; color:#999; margin-top:2px;'>{html.escape(str(_eq.get('detail','')))}</div></div>"
            f"</div>",
            unsafe_allow_html=True)
        st.caption("Guru-grade upgrades (2026-06-13): theme-breadth gate · banded/time deploy · regime tag · "
                   "BTC-vs-equity · ETF flow quality. Honest bottom = ~$44-60k, mid-2026 → Q1-2027; the "
                   "$52-57k / Oct-2026 point is the base case, not a promise. Not advice.")
        st.markdown("<hr style='border-color:#2a2d36; margin:14px 0;'>", unsafe_allow_html=True)
    except Exception:
        st.caption("Guru-grade upgrades — temporarily unavailable.")

    try:
        from core.guru_panel import guru_panel as _guru_panel
        _gp = _guru_panel()
        _vc = "#22c55e" if _gp.get("verdict") == "ON TRACK" else "#f0b90b"
        _ct = _gp.get("counts", {})
        st.markdown(
            f"<div style='padding:16px 20px; border-radius:10px; background:#13161c; "
            f"border-left:6px solid {_vc}; margin-bottom:14px;'>"
            f"<div style='font-size:11px; color:#888; text-transform:uppercase; letter-spacing:1.5px;'>"
            f"Panel verdict</div>"
            f"<div style='font-size:30px; font-weight:800; color:{_vc};'>{_gp.get('verdict')}</div>"
            f"<div style='font-size:13px; color:#ccc; margin-top:6px; line-height:1.6;'>"
            f"{_gp.get('summary')}</div>"
            f"<div style='font-size:12px; color:#888; margin-top:8px;'>"
            f"<b style='color:#22c55e;'>{_ct.get('on_track', 0)}</b> on track &middot; "
            f"<b style='color:#f0b90b;'>{_ct.get('wait', 0) + _ct.get('caution', 0)}</b> wait/caution &middot; "
            f"<b style='color:#ef4444;'>{_ct.get('dissent', 0)}</b> dissent</div>"
            f"</div>", unsafe_allow_html=True)

        def _render_guru_group(title, cat):
            st.markdown(f"#### {title}")
            for gu in [x for x in _gp.get("gurus", []) if x["category"] == cat]:
                _badge = "" if gu.get("live") else (
                    "<span style='font-size:9px; color:#666; border:1px solid #333; "
                    "border-radius:3px; padding:0 4px; margin-left:6px;'>stance</span>")
                st.markdown(
                    f"<div style='padding:12px 16px; border-radius:8px; background:#1a1d24; "
                    f"border-left:5px solid {gu['color']}; margin-bottom:8px;'>"
                    f"<div style='display:flex; justify-content:space-between; align-items:baseline;'>"
                    f"<span style='font-size:15px; font-weight:700; color:#fff;'>{gu['name']}{_badge}</span>"
                    f"<span style='font-size:12px; font-weight:700; color:{gu['color']};'>"
                    f"{gu['verdict_label']}</span></div>"
                    f"<div style='font-size:11px; color:#888; margin-top:2px;'>{gu['framework']}</div>"
                    f"<div style='font-size:12px; color:#ddd; margin-top:5px;'>📊 {gu['reading']}</div>"
                    f"<div style='font-size:12px; color:#aaa; margin-top:3px; line-height:1.5;'>"
                    f"{gu['detail']}</div></div>", unsafe_allow_html=True)

        _gc1, _gc2 = st.columns(2)
        with _gc1:
            _render_guru_group("🪙 Crypto cycle panel", "crypto")
        with _gc2:
            _render_guru_group("📉 Equity / macro panel", "macro")
        st.caption(_gp.get("as_of_note", ""))
    except Exception:
        st.info("Guru panel temporarily unavailable.")

    # ── 🎲 Prediction-market overlay (live real-money crowd odds) ──────────────
    st.markdown("<hr style='border-color:#2a2d36; margin:18px 0;'>", unsafe_allow_html=True)
    st.markdown("#### 🎲 What real money is betting — Polymarket crowd odds")
    st.caption("Live prediction-market prices. Real money is often sharper than any single "
               "indicator — divergence from our model is the signal. Not investment advice.")
    try:
        @st.cache_data(ttl=1800, show_spinner=False)
        def _pm_odds():
            from core.prediction_markets import prediction_market_odds
            return prediction_market_odds(max_items=20)
        _pm = _pm_odds() or {}
        if _pm.get("error") or not _pm.get("by_category"):
            st.caption("Prediction-market odds temporarily unavailable.")
        else:
            _CATEMOJI = {"btc": "₿ Bitcoin price", "crypto": "🪙 Crypto (other)",
                         "rates": "🏦 Fed / rates", "recession": "📉 Recession / macro",
                         "equities": "📊 Equities"}
            for _cat in ["btc", "rates", "recession", "equities", "crypto"]:
                _rows = (_pm.get("by_category") or {}).get(_cat)
                if not _rows:
                    continue
                st.markdown(f"**{_CATEMOJI.get(_cat, _cat)}**")
                for _m in _rows:
                    _yp = _m.get("yes_prob")
                    _pct = f"{_yp*100:.0f}%" if _yp is not None else "—"
                    _col = ("#22c55e" if (_yp is not None and _yp >= 0.6) else
                            "#ef4444" if (_yp is not None and _yp <= 0.4) else "#f0b90b")
                    st.markdown(
                        f"<div style='display:flex; gap:12px; align-items:baseline; "
                        f"padding:3px 0; border-bottom:1px solid #20242c;'>"
                        f"<span style='font-size:15px; font-weight:800; color:{_col}; "
                        f"min-width:48px;'>{_pct}</span>"
                        f"<span style='font-size:12px; color:#ccc;'>{html.escape(_m.get('question', '')[:92])}</span>"
                        f"<span style='font-size:10px; color:#666; margin-left:auto; "
                        f"white-space:nowrap;'>${_m.get('volume', 0):,.0f} · "
                        f"{_m.get('end_date', '')}</span></div>", unsafe_allow_html=True)
                st.write("")
            st.caption(f"{_pm.get('n_total', '?')} relevant live markets · {_pm.get('source', '')} "
                       "· % = crowd's implied 'Yes' probability")
    except Exception:
        st.caption("Prediction-market odds temporarily unavailable.")

    # ── 🩸 The Bear Case — weekly red-team / devil's advocate ──────────────────
    st.markdown("<hr style='border-color:#2a2d36; margin:18px 0;'>", unsafe_allow_html=True)
    st.markdown("#### 🩸 The Bear Case — your in-house devil's advocate")
    st.caption("The strongest arguments that you're WRONG, compiled off live signals and emailed "
               "weekly. If none of them change your mind, that's earned conviction. Not advice.")
    try:
        @st.cache_data(ttl=1800, show_spinner=False)
        def _red_team():
            from core.red_team import red_team_report
            return red_team_report()
        _rtm = _red_team() or {}
        _sevcol = {"HIGH": "#ef4444", "MEDIUM": "#f0b90b", "LOW": "#888"}
        for _a in _rtm.get("arguments", []):
            _sc = _sevcol.get(_a.get("severity"), "#888")
            st.markdown(
                f"<div style='padding:11px 15px; border-radius:8px; background:#1a1d24; "
                f"border-left:4px solid {_sc}; margin-bottom:7px;'>"
                f"<div style='display:flex; justify-content:space-between; align-items:baseline;'>"
                f"<span style='font-size:13px; font-weight:700; color:#eee;'>{_a.get('title', '')}</span>"
                f"<span style='font-size:9px; font-weight:700; color:{_sc};'>{_a.get('severity', '')}</span>"
                f"</div>"
                + (f"<div style='font-size:11px; color:#999; margin-top:3px;'>📍 {_a.get('evidence', '')}"
                   f"</div>" if _a.get('evidence') else "")
                + f"<div style='font-size:11px; color:#bbb; margin-top:3px; line-height:1.5;'>"
                  f"{_a.get('why', '')}</div></div>", unsafe_allow_html=True)
        st.caption("Emailed every Sunday. The opposition rests.")
    except Exception:
        st.caption("Red-team temporarily unavailable.")

    # ── 🩺 Data Health — freshness + denominator-drift self-check ──────────────
    st.markdown("<hr style='border-color:#2a2d36; margin:18px 0;'>", unsafe_allow_html=True)
    try:
        @st.cache_data(ttl=300, show_spinner=False)
        def _data_health_cached():
            from core.data_health import data_health
            return data_health()
        _dh = _data_health_cached() or {}
        _dhc = _dh.get("color", "#888")
        st.markdown(
            f"<div style='padding:11px 16px; border-radius:8px; background:#13161c; "
            f"border-left:5px solid {_dhc};'>"
            f"<span style='font-size:11px; color:#888; text-transform:uppercase; "
            f"letter-spacing:1.2px; font-weight:700;'>🩺 Data Health</span> "
            f"<span style='font-size:15px; font-weight:800; color:{_dhc}; margin-left:8px;'>"
            f"{_dh.get('verdict', '?')}</span>"
            f"<span style='font-size:11px; color:#aaa; margin-left:10px;'>"
            f"{_dh.get('n_tracked', 0)} indicators · {_dh.get('n_stale', 0)} stale · "
            f"{_dh.get('n_aging', 0)} aging · {len(_dh.get('dead_feeds', []))} dead · "
            f"{len(_dh.get('drift', []))} drift</span></div>", unsafe_allow_html=True)
        with st.expander("🩺 Per-indicator freshness (every cache: age vs budget)"):
            if _dh.get("drift"):
                st.error("DENOMINATOR DRIFT — a scorecard total changed and labels need updating: "
                         + ", ".join(f"{d['key']} shows {d['actual']} (expected {d['expected']})"
                                     for d in _dh["drift"]))
            if _dh.get("dead_feeds"):
                st.caption("Dead feeds (paywalled/unavailable, known): "
                           + ", ".join(d['label'].strip() for d in _dh['dead_feeds']))
            for _it in sorted(_dh.get("items", []), key=lambda x: (x['age_h'] or 0), reverse=True):
                _ic = {"FRESH": "#22c55e", "AGING": "#f0b90b", "STALE": "#ef4444",
                       "MISSING": "#ef4444"}.get(_it["status"], "#888")
                _a = f"{_it['age_h']:.1f}h" if _it["age_h"] is not None else "—"
                st.markdown(
                    f"<div style='display:flex; gap:10px; font-size:11px; padding:1px 0;'>"
                    f"<span style='color:{_ic}; font-weight:700; min-width:60px;'>{_it['status']}</span>"
                    f"<span style='color:#ccc; min-width:220px;'>{_it['key']}</span>"
                    f"<span style='color:#888;'>{_a} / {_it['budget_h']}h budget</span></div>",
                    unsafe_allow_html=True)
        st.caption("Auto-checks every cache's age + scorecard totals each load. STALE or DRIFT "
                   "gets emailed. DEGRADED = the 2 known paywalled feeds (NVT, funding) — expected.")
    except Exception:
        st.caption("Data-health check temporarily unavailable.")

    # ── 📊 Olson signal scorecard — auto-graded track record (decide on data) ──
    st.markdown("<hr style='border-color:#2a2d36; margin:18px 0;'>", unsafe_allow_html=True)
    st.markdown("#### 📊 Jesse Olson — signal scorecard (auto-graded)")
    st.caption("Logs his directional calls and grades them by 30-day forward return → real hit-rate "
               "WITH payoff (R) + expectancy. Builds over months; small samples mean little. "
               "Use this to judge his paid tier on data, not marketing. Not advice.")
    try:
        @st.cache_data(ttl=1800, show_spinner=False)
        def _olson_scorecard_cached():
            from core.olson_scorecard import olson_scorecard
            return olson_scorecard()
        _osc = _olson_scorecard_cached() or {}
        _hit = _osc.get("hit_rate_pct") or 0
        _pay = _osc.get("payoff_R")
        _exp = _osc.get("expectancy_pct")
        _vc = ("#22c55e" if (_exp is not None and _exp > 0.5) else
               "#ef4444" if (_exp is not None and _exp < -0.5) else "#f0b90b")
        st.markdown(
            f"<div style='padding:14px 18px; border-radius:10px; background:#13161c; "
            f"border-left:6px solid {_vc};'>"
            f"<div style='font-size:11px; color:#888; text-transform:uppercase; "
            f"letter-spacing:1.2px; font-weight:700;'>Verdict</div>"
            f"<div style='font-size:18px; font-weight:800; color:{_vc};'>{_osc.get('verdict', '?')}</div>"
            f"<div style='font-size:12px; color:#ccc; margin-top:8px;'>"
            f"Hit-rate <b>{_hit:.0f}%</b> ({_osc.get('n_scored', 0)} graded) &middot; "
            f"Payoff <b>{_pay if _pay is not None else '—'}R</b> &middot; "
            f"Expectancy <b>{_exp if _exp is not None else '—'}%</b>/call &middot; "
            f"{_osc.get('n_pending', 0)} pending &middot; {_osc.get('n_logged', 0)} logged</div>"
            f"<div style='font-size:10px; color:#777; margin-top:6px;'>"
            f"Curated calls hand-verified; new calls auto-graded by {_osc.get('horizon_days', 30)}d "
            f"forward return (direction only — coarser). Payoff R = avg win ÷ avg loss.</div></div>",
            unsafe_allow_html=True)
        with st.expander("📋 The call log (newest first)"):
            for _c in _osc.get("calls", [])[:25]:
                _oc = {"RIGHT": "#22c55e", "WRONG": "#ef4444", "PENDING": "#888",
                       "FLAT": "#888"}.get(_c.get("outcome"), "#888")
                _fwd = f"{_c['fwd_return']:+.0f}%" if _c.get("fwd_return") is not None else ""
                st.markdown(
                    f"<div style='display:flex; gap:10px; font-size:11px; padding:2px 0; "
                    f"border-bottom:1px solid #20242c;'>"
                    f"<span style='color:{_oc}; font-weight:700; min-width:60px;'>{_c.get('outcome', '?')}</span>"
                    f"<span style='color:#888; min-width:74px;'>{_c.get('date', '')}</span>"
                    f"<span style='color:#bbb; min-width:44px;'>{_c.get('asset', '')}</span>"
                    f"<span style='color:#999; min-width:34px;'>{_fwd}</span>"
                    f"<span style='color:#aaa;'>{html.escape(str(_c.get('thesis', ''))[:70])}</span></div>",
                    unsafe_allow_html=True)
        st.caption("Updated daily. To judge his PAID tier: let it run ~3–6 months, then look at "
                   "payoff R + expectancy — not the headline hit-rate.")
    except Exception:
        st.caption("Olson scorecard temporarily unavailable.")

    # ── 🧮 Benjamin Cowen — risk-metric analyst (live + auto-graded) ──────────
    st.markdown("<hr style='border-color:#2a2d36; margin:18px 0;'>", unsafe_allow_html=True)
    st.markdown("#### 🧮 Benjamin Cowen — risk-metric analyst (live + auto-graded)")
    st.caption("Quant & cautious by nature: accumulate when his risk metric is low, distribute when high; "
               "'diminishing returns' each cycle. His method is already encoded in the Cowen log-regression "
               "dial in the Guru Panel above. Tweets auto-fetch every ~2h; new calls auto-graded by 30-day "
               "forward return. Not advice.")
    try:
        # Cycle stance — framework-level + his published Q2-2026 view (durable; the
        # live specifics arrive in the tweet feed below as they're posted).
        st.markdown(
            "<div style='padding:12px 16px; border-radius:10px; background:#13161c; "
            "border-left:6px solid #4a90e2;'>"
            "<div style='font-size:11px; color:#888; text-transform:uppercase; letter-spacing:1.2px; "
            "font-weight:700;'>His cycle stance (Q2-2026 published view)</div>"
            "<div style='font-size:13px; color:#ccc; margin-top:6px; line-height:1.5;'>"
            "Bottom <b>not in</b> — he puts ~75% odds the cycle low is still ahead, most likely "
            "<b>~Oct 2026</b> (about a year after the Oct-2025 top). Posture: <b>capital preservation</b>; "
            "rallies are tactical, not a new bull. Keeps <b>sub-$40k</b> on the table before the low.</div>"
            "<div style='font-size:11px; color:#9ab; margin-top:8px; line-height:1.5;'>"
            "↪ <b>Lines up with the rotation plan:</b> his ~Oct-2026 bottom ≈ your deploy window, and "
            "'preserve capital until the low' ≈ armed-but-not-triggered. His sub-$40k tail is the live "
            "reason to scale in across $57k→$50k→$45k rather than dump the whole rotation at first touch.</div>"
            "<div style='font-size:10px; color:#666; margin-top:6px;'>"
            "Framework + his published view, sourced from his Q2-2026 risk memo &amp; X posts — not a "
            "confirmed live quote. The feed below is the live record.</div></div>",
            unsafe_allow_html=True)

        @st.cache_data(ttl=1800, show_spinner=False)
        def _cowen_scorecard_cached():
            from core.guru_scorecard import cowen_scorecard
            return cowen_scorecard()
        _csc = _cowen_scorecard_cached() or {}
        _chit = _csc.get("hit_rate_pct") or 0
        _cpay = _csc.get("payoff_R")
        _cexp = _csc.get("expectancy_pct")
        _cvc = ("#22c55e" if (_cexp is not None and _cexp > 0.5) else
                "#ef4444" if (_cexp is not None and _cexp < -0.5) else "#f0b90b")
        st.markdown(
            f"<div style='padding:14px 18px; border-radius:10px; background:#13161c; "
            f"border-left:6px solid {_cvc}; margin-top:10px;'>"
            f"<div style='font-size:11px; color:#888; text-transform:uppercase; "
            f"letter-spacing:1.2px; font-weight:700;'>Signal scorecard verdict</div>"
            f"<div style='font-size:18px; font-weight:800; color:{_cvc};'>{_csc.get('verdict', '?')}</div>"
            f"<div style='font-size:12px; color:#ccc; margin-top:8px;'>"
            f"Hit-rate <b>{_chit:.0f}%</b> ({_csc.get('n_scored', 0)} graded) &middot; "
            f"Payoff <b>{_cpay if _cpay is not None else '—'}R</b> &middot; "
            f"Expectancy <b>{_cexp if _cexp is not None else '—'}%</b>/call &middot; "
            f"{_csc.get('n_pending', 0)} pending &middot; {_csc.get('n_logged', 0)} logged</div>"
            f"<div style='font-size:10px; color:#777; margin-top:6px;'>"
            f"Seeded from his curated record; new calls auto-graded by {_csc.get('horizon_days', 30)}d "
            f"forward return. Builds over months — small samples mean little.</div></div>",
            unsafe_allow_html=True)
        with st.expander("📋 Cowen call log (newest first)"):
            for _c in _csc.get("calls", [])[:25]:
                _oc = {"RIGHT": "#22c55e", "WRONG": "#ef4444", "PENDING": "#888",
                       "FLAT": "#888"}.get(_c.get("outcome"), "#888")
                _fwd = f"{_c['fwd_return']:+.0f}%" if _c.get("fwd_return") is not None else ""
                st.markdown(
                    f"<div style='display:flex; gap:10px; font-size:11px; padding:2px 0; "
                    f"border-bottom:1px solid #20242c;'>"
                    f"<span style='color:{_oc}; font-weight:700; min-width:60px;'>{_c.get('outcome', '?')}</span>"
                    f"<span style='color:#888; min-width:74px;'>{_c.get('date', '')}</span>"
                    f"<span style='color:#bbb; min-width:44px;'>{_c.get('asset', '')}</span>"
                    f"<span style='color:#999; min-width:34px;'>{_fwd}</span>"
                    f"<span style='color:#aaa;'>{html.escape(str(_c.get('thesis', ''))[:70])}</span></div>",
                    unsafe_allow_html=True)

        # ─── LIVE COWEN TWEETS (from nitter monitor; populates within ~2h) ───
        try:
            import json as _cjson
            _cowen_cache = REPO_ROOT / ".guru_benjamincowen_tweets_cache.json"
            if _cowen_cache.exists():
                _cwd = _cjson.loads(_cowen_cache.read_text())
                _cw_tweets = _cwd.get("tweets", [])
                _cw_updated = _cwd.get("updated", "")
                _cw_show = ([t for t in _cw_tweets if t.get("relevance") == "HIGH"][:5] +
                            [t for t in _cw_tweets if t.get("relevance") == "MEDIUM"][:3])
                if _cw_show:
                    st.markdown(
                        f"<div class='section-header' style='font-size:14px; margin-top:12px; "
                        f"color:#ccc;'>🎙️ Latest @benjamincowen tweets "
                        f"<span style='font-size:10px; color:#888;'>(updated {_cw_updated[:16]})</span></div>",
                        unsafe_allow_html=True)
                    for _t in _cw_show[:6]:
                        _rel = _t.get("relevance", "?")
                        _rel_color = ("#ef4444" if _rel == "HIGH" else "#f0b90b")
                        _txt = html.escape((_t.get("text") or _t.get("title", ""))[:280])
                        _link = html.escape(_t.get("link", ""))
                        _pub = html.escape((_t.get("pub", "") or "")[:16])
                        st.markdown(
                            f"<div style='padding:8px 12px; margin-bottom:6px; background:#13161c; "
                            f"border-radius:6px; border-left:3px solid {_rel_color}; font-size:12px;'>"
                            f"<div style='display:flex; justify-content:space-between; align-items:baseline;'>"
                            f"<span style='font-size:9px; color:{_rel_color}; font-weight:700; "
                            f"text-transform:uppercase;'>{_rel}</span>"
                            f"<span style='font-size:9px; color:#888;'>{_pub}</span></div>"
                            f"<div style='color:#ccc; margin-top:3px; line-height:1.4;'>{_txt}</div>"
                            f"<a href='{_link}' target='_blank' style='font-size:10px; color:#4a90e2;'>"
                            f"open on X →</a></div>",
                            unsafe_allow_html=True)
                else:
                    st.caption("📡 Live tweets: cache present but no relevant posts yet.")
            else:
                st.caption("📡 Live tweets: waiting for first fetch (monitor runs every ~2h; "
                           "nitter availability varies).")
        except Exception as _cte:
            st.caption("Cowen tweets temporarily unavailable.")
    except Exception:
        st.caption("Cowen panel temporarily unavailable.")


# ─────────────────────────────────────────────────────────────────
# OVERVIEW TAB — price chart, scorecard, RCap thermometer, action
# ─────────────────────────────────────────────────────────────────
with _today_hero:   # hero banner (verdict + cycle) - renders FIRST on Today
    # ╔══════════════════════════════════════════════════════════════╗
    # ║                        HERO BANNER                            ║
    # ╚══════════════════════════════════════════════════════════════╝
    _hero = st.columns([3, 2])
    with _hero[0]:
        rcd_str = f"{rcd['current_drawdown_pct']:+.1f}%" if rcd and not rcd.get("error") else "?"
        st.markdown(
            f"<div style='padding:20px 26px; border-radius:10px; "
            f"border-left:8px solid {verdict_color}; "
            f"background:linear-gradient(90deg, {verdict_color}22 0%, #1a1d24 100%);'>"
            f"<div style='font-size:11px; color:#888; text-transform:uppercase; letter-spacing:1.5px;'>"
            f"Current verdict</div>"
            f"<div style='font-size:44px; font-weight:800; color:{verdict_color}; line-height:1;'>"
            f"{verdict_label}</div>"
            f"<div style='font-size:15px; color:#ccc; margin-top:8px;'>{verdict_sub}</div>"
            f"<div style='font-size:13px; color:#888; margin-top:10px;'>"
            f"Bottom (base case) ~<b style='color:#ccc;'>{days_to_bot}d</b> "
            f"({pos['projected_bottom_date'].strftime('%b %Y')}); honest window "
            f"<b style='color:#ccc;'>~mid-2026→Q1-2027</b>, band <b style='color:#ccc;'>~$44-60k</b>. "
            f"Scorecard <b style='color:#ccc;'>{sc['n_met']}/{sc['n_total']}</b>. "
            f"Realized Cap drawdown <b style='color:#ccc;'>{rcd_str}</b> "
            f"(need -15% min). "
            f"BTC <b style='color:#ccc;'>${btc_price:,.0f}</b> ({pct_from_ath:+.0f}% from ATH)."
            f"</div></div>",
            unsafe_allow_html=True,
        )

    with _hero[1]:
        cycle_length = (pos["next_halving"] - pos["current_halving"]).days
        st.markdown(
            f"<div style='padding:20px 26px; background:#1a1d24; border-radius:10px; height:100%;'>"
            f"<div style='font-size:11px; color:#888; text-transform:uppercase; letter-spacing:1.5px;'>"
            f"Cycle position</div>"
            f"<div style='font-size:24px; font-weight:700; color:#fff; line-height:1.2; margin-top:4px;'>"
            f"{regime.replace('_', ' ').title()}</div>"
            f"<div style='font-size:13px; color:#aaa; margin-top:6px;'>"
            f"{phase_info['description']}</div>"
            f"<div style='margin-top:14px; background:#2a2d34; height:8px; border-radius:4px; overflow:hidden;'>"
            f"<div style='background:{verdict_color}; height:100%; width:{pos['pct_through_cycle']:.0f}%;'></div>"
            f"</div>"
            f"<div style='font-size:12px; color:#888; margin-top:8px;'>"
            f"Day {pos['days_post_halving']} of {cycle_length} ({pos['pct_through_cycle']:.0f}% through cycle 5)"
            f"</div></div>",
            unsafe_allow_html=True,
        )



with tab_research:   # <- 2026-07-04 restructure
    # ═══════════════════════════════════════════════════════════════════
    # ── 🎙️ JESSE OLSON — latest X feed + ideas digest (on-demand refresh) ──────
    try:
        _ohdr = st.columns([3, 1])
        with _ohdr[0]:
            st.markdown("#### 🎙️ Jesse Olson — latest feed & ideas")
        with _ohdr[1]:
            _olson_refresh = st.button("🔄 Refresh feed", key="olson_refresh_btn")
        if _olson_refresh:
            with st.spinner("Pulling @JesseOlson live from nitter…"):
                try:
                    from _scheduler.jesse_olson_monitor import refresh_cache_only
                    _rr = refresh_cache_only()
                except Exception as _re:
                    _rr = {"ok": False, "error": str(_re)[:80]}
            if _rr.get("ok"):
                st.success(f"Refreshed — {_rr.get('n')} tweets pulled live.")
            else:
                st.warning(f"Couldn't refresh ({_rr.get('error', '?')}); showing last cached feed.")

        # 🤖 AI prose read (read-only cache — generated by precompute in the
        # background, NEVER here; the public page can't spend tokens).
        try:
            from core.olson_ai_summary import olson_ai_summary_cached
            _ai = olson_ai_summary_cached()
            if _ai.get("summary"):
                st.markdown(
                    f"**🤖 AI read of his latest posts** "
                    f"<span style='font-size:10px; color:#888;'>"
                    f"({html.escape(str(_ai.get('model','')).split('-2025')[0] or 'Haiku')}, "
                    f"updated {html.escape(str(_ai.get('generated',''))[:16])})</span>",
                    unsafe_allow_html=True)
                st.markdown(_ai["summary"])   # safe markdown (unsafe_allow_html=False = XSS-safe)
            elif not _ai.get("enabled"):
                st.caption("🤖 AI summary: add ANTHROPIC_API_KEY to .env to enable a written read "
                           "(the levels digest below is always free).")
        except Exception:
            pass

        from core.olson_feed_summary import olson_feed_summary
        _ofs = olson_feed_summary(max_tweets=5)
        _chips = "".join(
            f"<span style='display:inline-block; margin:2px 4px 2px 0; padding:2px 9px; "
            f"border-radius:10px; font-size:11px; background:#1c2430; color:#9fb6c9;'>"
            f"{html.escape(str(a))}: {html.escape(', '.join(v.get('levels', [])[:3]))}</span>"
            for a, v in _ofs.get("by_asset", {}).items() if v.get("levels"))
        st.markdown(
            f"<div style='padding:10px 14px; border-radius:10px; background:#13161c; "
            f"border-left:5px solid #4a90e2; margin-bottom:8px;'>"
            f"<div style='font-size:13px; color:#ddd; font-weight:600;'>{html.escape(str(_ofs.get('read','')))}</div>"
            f"<div style='margin-top:6px;'>{_chips}</div>"
            f"<div style='font-size:10px; color:#777; margin-top:6px;'>"
            f"From his cached X feed · {_ofs.get('n',0)} tweets · updated {html.escape(str(_ofs.get('updated','?')))} "
            f"(auto every 2h; the button forces a live pull). Levels = prices he's naming; "
            f"read his posts below for his actual view — not advice.</div></div>",
            unsafe_allow_html=True)
        for _t in _ofs.get("tweets", []):
            _rc = ("#ef4444" if _t.get("relevance") == "HIGH" else
                   "#f0b90b" if _t.get("relevance") == "MEDIUM" else "#888")
            _lv = (" · levels: " + ", ".join(_t["levels"])) if _t.get("levels") else ""
            st.markdown(
                f"<div style='padding:7px 12px; margin-bottom:5px; background:#13161c; border-radius:6px; "
                f"border-left:3px solid {_rc}; font-size:12px;'>"
                f"<div style='display:flex; justify-content:space-between; align-items:baseline;'>"
                f"<span style='font-size:9px; color:{_rc}; font-weight:700; text-transform:uppercase;'>"
                f"{html.escape(str(_t.get('relevance','?')))}{html.escape(_lv)}</span>"
                f"<span style='font-size:9px; color:#888;'>{html.escape(str(_t.get('pub','')))}</span></div>"
                f"<div style='color:#ccc; margin-top:3px; line-height:1.4;'>{html.escape(str(_t.get('text','')))}</div>"
                f"<a href='{html.escape(str(_t.get('link','')))}' target='_blank' "
                f"style='font-size:10px; color:#4a90e2;'>open on X →</a></div>",
                unsafe_allow_html=True)
        st.markdown("<hr style='border-color:#2a2d36; margin:12px 0;'>", unsafe_allow_html=True)
    except Exception:
        st.caption("Jesse Olson feed — temporarily unavailable.")


with tab_today:   # <- 2026-07-04 review fix: "what changed" belongs on Today
    # 🆕 DAILY CHANGE LOG — what's different since yesterday (Swift insistence)
    # ═══════════════════════════════════════════════════════════════════
    try:
        from core.btc_change_log import get_diff
        _cl = get_diff()
        if _cl and not _cl.get("first_observation"):
            n_changes = _cl.get("n_changes", 0)
            if n_changes > 0:
                price_diff = _cl.get("price_diff")
                # Build the change pills
                pills_html = ""
                if price_diff:
                    px_color = "#22c55e" if price_diff["arrow"] == "↑" else "#ef4444"
                    pills_html += (
                        f"<span style='display:inline-block; padding:4px 10px; margin:2px 4px 2px 0; "
                        f"border-radius:14px; background:#13161c; border:1px solid {px_color}; "
                        f"font-size:11px; color:{px_color};'>{price_diff['text']}</span>"
                    )
                for d in _cl.get("diffs", [])[:12]:
                    arrow = d.get("arrow", "≠")
                    color = "#22c55e" if arrow == "↑" else ("#ef4444" if arrow == "↓" else "#f0b90b")
                    pills_html += (
                        f"<span style='display:inline-block; padding:4px 10px; margin:2px 4px 2px 0; "
                        f"border-radius:14px; background:#13161c; border:1px solid {color}; "
                        f"font-size:11px; color:{color};'>{d['text']}</span>"
                    )
                st.markdown(
                    f"<div style='padding:10px 14px; margin-bottom:14px; border-radius:8px; "
                    f"background:#0e1117; border-left:4px solid #f0b90b;'>"
                    f"<div style='font-size:10px; color:#888; text-transform:uppercase; "
                    f"letter-spacing:1.5px; margin-bottom:6px;'>"
                    f"Today's Changes ({n_changes})</div>"
                    f"<div>{pills_html}</div></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.caption("Today's changes: no movement vs yesterday's snapshot.")
    except Exception as _e:
        pass


with tab_playbook:   # <- 2026-07-04 restructure
    # ═══════════════════════════════════════════════════════════════════
    # 🔄 ROTATION TRIGGER — single-shot equity→BTC rotation (top of page)
    # When ANY of 3 paths fires (2-of-2 each), email arrives with NZ$ amounts.
    # Until then: status is ARMED/WARMING, this panel is mostly informational.
    # ═══════════════════════════════════════════════════════════════════
    try:
        from core.dashboard_cache import get_cached as _gc_rt
        _rt = _gc_rt("rotation_trigger")
        if not _rt:
            from core.rotation_trigger import evaluate_rotation_trigger
            _rt = evaluate_rotation_trigger()

        _rt_status = _rt.get("overall", "?")
        _rt_color = _rt.get("color", "#888")
        _rt_fired = _rt.get("fired", False)
        _rt_paths = _rt.get("paths", [])
        _rt_action = _rt.get("action", {}) or {}

        _status_emoji = {"FIRED": "🔥", "WARMING": "🟡", "ARMED": "🟢"}.get(_rt_status, "❓")
        _status_subtitle = {
            "FIRED":   "EXECUTE TODAY — sell all equity, same-day buy BTC",
            "WARMING": "1-of-4 equity-stress signals firing. Conditions building.",
            "ARMED":   "All 4 equity signals armed and watching. No action needed.",
        }.get(_rt_status, "")

        # Build 1+2+3 readouts: cycle era + confidence + effective signals
        _rt_era = _rt.get("cycle_era", "?")
        _rt_scale = _rt.get("cycle_scale", 1.0) or 1.0
        _rt_conf_pct = _rt.get("confidence_pct", 0) or 0
        _rt_conf_tier = _rt.get("confidence_tier", "?")
        _rt_eff_sig = _rt.get("effective_signals", "?")

        # Hero status bar with validation footers
        st.markdown(
            f"<div style='padding:16px 20px; border-radius:10px; "
            f"background:#13161c; border:3px solid {_rt_color}; "
            f"margin-bottom:14px;'>"
            f"<div style='display:flex; justify-content:space-between; align-items:baseline;'>"
            f"<span style='font-size:11px; color:#888; text-transform:uppercase; "
            f"letter-spacing:1.5px; font-weight:700;'>"
            f"🔄 ROTATION TRIGGER · Equity → BTC</span>"
            f"<span>{_age_badge('rotation_trigger')}</span></div>"
            f"<div style='font-size:36px; font-weight:800; color:{_rt_color}; "
            f"line-height:1.1; margin-top:4px;'>{_status_emoji} {_rt_status}</div>"
            f"<div style='font-size:13px; color:#ccc; margin-top:6px;'>"
            f"{_status_subtitle}</div>"
            # Validation summary footer
            f"<div style='display:flex; flex-wrap:wrap; gap:14px; "
            f"margin-top:10px; padding-top:10px; border-top:1px solid #2a2d36;'>"
            f"<div style='font-size:11px; color:#888;'>"
            f"<b style='color:#fff;'>Confidence:</b> "
            f"<span style='color:#4a90e2;'>{_rt_conf_pct:.0f}%</span> ({_rt_conf_tier})</div>"
            f"<div style='font-size:11px; color:#888;'>"
            f"<b style='color:#fff;'>Effective signals:</b> "
            f"<span style='color:#4a90e2;'>{_rt_eff_sig}</span> clusters</div>"
            f"<div style='font-size:11px; color:#888;'>"
            f"<b style='color:#fff;'>Cycle era:</b> "
            f"<span style='color:#4a90e2;'>{_rt_era}</span> · "
            f"thresholds × {_rt_scale:.2f}</div>"
            f"</div></div>", unsafe_allow_html=True,
        )

        # Historical context — what this status has meant at past NASDAQ tops.
        # Prominent box when FIRED/WARMING; collapsed expander when ARMED.
        try:
            from core.rotation_trigger import leadtime_context as _ltc
            _lc = _ltc(_rt_status)
            _ep_rows = "".join(
                f"<div style='display:flex; justify-content:space-between; gap:10px; "
                f"padding:4px 0; border-top:1px solid #20242c; font-size:11px;'>"
                f"<span style='color:#bbb;'>{_e['year']} · {_e['kind']}</span>"
                f"<span style='color:#888;'>out <b style=\"color:#ddd;\">{_e['exec_off']:+d}%</b> · "
                f"dodged <b style=\"color:#22c55e;\">{_e['dodged']:+d}%</b> · "
                f"<b style=\"color:#4a90e2;\">{_e['lead_days']}d</b> before low</span></div>"
                for _e in _lc["episodes"])
            _ctx_inner = (
                f"<div style='font-size:12.5px; color:#ccc; line-height:1.55;'>{_lc['lead']}</div>"
                f"<div style='margin-top:8px;'>{_ep_rows}</div>"
                f"<div style='font-size:9.5px; color:#777; margin-top:7px;'>{_lc['source']}</div>")
            if _rt_status in ("FIRED", "WARMING"):
                _ctx_b = "#ef4444" if _rt_status == "FIRED" else "#f0b90b"
                st.markdown(
                    f"<div style='padding:13px 18px; border-radius:10px; background:#13161c; "
                    f"border-left:5px solid {_ctx_b}; margin-bottom:14px;'>"
                    f"<div style='font-size:11px; color:#fff; text-transform:uppercase; "
                    f"letter-spacing:1.2px; font-weight:700; margin-bottom:6px;'>"
                    f"📜 {_lc['headline']}</div>{_ctx_inner}</div>",
                    unsafe_allow_html=True)
            else:
                with st.expander(f"📜 {_lc['headline']}"):
                    st.markdown(_ctx_inner, unsafe_allow_html=True)
        except Exception:
            pass

        # If FIRED — action card. Amounts shown only when SHOW_PERSONAL (private);
        # otherwise signal-only ("amounts in your email").
        if _rt_fired:
            if SHOW_PERSONAL:
                _legs = (
                    f"<b>Leg 1 — SELL EQUITY:</b> "
                    f"<span style='color:#f0b90b;'>{_money(_rt_action.get('sell_equity_nzd', 0))}</span><br>"
                    f"<b>Leg 2 — BUY BTC:</b> "
                    f"<span style='color:#22c55e;'>{_money(_rt_action.get('buy_btc_nzd', 0))}</span> "
                    f"(~{_rt_action.get('btc_amount_estimate', 0):.4f} BTC at "
                    f"${_rt.get('btc_price', 0):,.0f})<br>"
                    f"<b>Reserve:</b> {_money(_rt_action.get('cash_reserve_nzd', 0))} cash"
                )
            else:
                _legs = (
                    f"<b>Leg 1 — SELL all equity</b> → cash<br>"
                    f"<b>Leg 2 — BUY BTC</b> with the proceeds, same trading window "
                    f"(BTC ${_rt.get('btc_price', 0):,.0f})<br>"
                    f"<span style='color:#888;'>Exact amounts are in the operator's email alert.</span>"
                )
            st.markdown(
                f"<div style='padding:16px 20px; border-radius:10px; "
                f"background:linear-gradient(90deg, rgba(239,68,68,0.20) 0%, #13161c 100%); "
                f"border:3px solid #ef4444; margin-bottom:14px;'>"
                f"<div style='font-size:11px; color:#ef4444; text-transform:uppercase; "
                f"letter-spacing:1.5px; font-weight:700;'>"
                f"🚨 EXECUTE ROTATION TODAY</div>"
                f"<div style='font-size:14px; color:#fff; margin-top:8px; line-height:1.5;'>"
                f"{_legs}"
                f"</div></div>",
                unsafe_allow_html=True,
            )

        # Single-trigger 4-condition scoreboard (2-of-4 to fire)
        _p1_data = _rt_paths[0] if _rt_paths else {}
        _p1_conds = _p1_data.get("conditions", [])
        _p1_score = _p1_data.get("score", "0/4")
        _p1_fired = _p1_data.get("fired", False)
        _p1_border = "#ef4444" if _p1_fired else (
            "#f0b90b" if _p1_score != "0/4" else "#22c55e")

        # 4 condition cards in a 2x2 grid (stacks on phone)
        _cc1, _cc2 = st.columns(2)
        for _idx, _cond in enumerate(_p1_conds):
            col = _cc1 if _idx % 2 == 0 else _cc2
            with col:
                _mark = "✓ FIRING" if _cond.get("met") else "○ dormant"
                _mc = "#22c55e" if _cond.get("met") else "#888"
                st.markdown(
                    f"<div style='padding:10px 12px; border-radius:6px; "
                    f"background:#13161c; border-left:3px solid {_mc}; "
                    f"margin-bottom:8px;'>"
                    f"<div style='display:flex; justify-content:space-between;'>"
                    f"<span style='font-size:9px; color:#888; text-transform:uppercase;'>"
                    f"Signal {_idx+1}</span>"
                    f"<span style='font-size:9px; color:{_mc}; font-weight:700;'>{_mark}</span>"
                    f"</div>"
                    f"<div style='font-size:12px; color:#ccc; margin-top:4px; line-height:1.3;'>"
                    f"{_cond.get('label', '')}</div>"
                    f"<div style='font-size:10px; color:#888; margin-top:2px;'>"
                    f"{_cond.get('current', '')}</div>"
                    f"</div>", unsafe_allow_html=True,
                )

        st.caption(
            f"**EQUITY-PRIORITY** logic: trigger fires when **2 of 4 equity signals** "
            f"confirm. Currently **{_p1_score}** firing. Designed to fire EARLY on equity "
            "weakness rather than wait for BTC bottom — your stated priority is "
            "avoiding equity drawdown."
        )

        # ── 🛡️ Tail hedge (the crash safety net — Burry/Hussman scenario) ──────
        try:
            @st.cache_data(ttl=1800, show_spinner=False)
            def _tail_hedge_status():
                from core.tail_hedge import compute_hedge_recommendation
                return compute_hedge_recommendation()
            _th = _tail_hedge_status() or {}
            _urg = str(_th.get("urgency", "optional")).upper()
            _thc = {"OPTIONAL": "#22c55e", "RECOMMENDED": "#f0b90b", "ADVISED": "#f0b90b",
                    "URGENT": "#ef4444"}.get(_urg, "#22c55e")
            _rfc = _th.get("risk_factor_count", 0)
            _budg = (_th.get("max_premium_pct_of_bankroll") or 0) * 100
            _reasons = _th.get("reasoning") or []
            _struct = _th.get("suggested_structure")
            _rlist = "".join(
                f"<div style='font-size:11px; color:#bbb; margin-top:2px;'>· {r}</div>"
                for r in _reasons[:4])
            st.markdown(
                f"<div style='padding:14px 18px; border-radius:10px; background:#13161c; "
                f"border-left:6px solid {_thc}; margin-bottom:6px;'>"
                f"<div style='display:flex; justify-content:space-between; align-items:baseline;'>"
                f"<span style='font-size:11px; color:#888; text-transform:uppercase; "
                f"letter-spacing:1.5px; font-weight:700;'>🛡️ Tail hedge · crash protection</span>"
                f"<span style='font-size:18px; font-weight:800; color:{_thc};'>{_urg}</span></div>"
                f"<div style='font-size:12px; color:#ccc; margin-top:6px;'>"
                f"Risk factors firing: <b style='color:{_thc};'>{_rfc}/6</b> &middot; "
                f"max premium budget <b>{_budg:.2f}%</b> of bankroll &middot; "
                f"action: <b>{'BUY PROTECTION NOW' if _th.get('should_hedge') else 'no hedge needed yet'}</b></div>"
                f"{_rlist}"
                + (f"<div style='font-size:11px; color:#888; margin-top:4px;'>Suggested structure: "
                   f"{_struct}</div>" if _struct else "")
                + "<div style='font-size:10px; color:#777; margin-top:6px;'>This is your protection "
                  "against the scenario where stocks AND Bitcoin crash together. It arms itself as "
                  "risk factors (VIX spike, credit stress, etc.) accumulate.</div>"
                "</div>", unsafe_allow_html=True)
        except Exception:
            st.caption("🛡️ Tail hedge status temporarily unavailable.")

        # (BTC TOP scale-out ladder lives at the bottom of the Playbook tab —
        #  years away, kept out of the daily flow. Email alerts fire regardless.)

        st.markdown("<hr style='border-color:#2a2d36; margin:18px 0;'>",
                    unsafe_allow_html=True)
    except Exception as _rte:
        st.caption(f"Rotation trigger — temporarily unavailable")


with tab_research:   # <- 2026-07-04 restructure
    # ═══════════════════════════════════════════════════════════════════
    # 📊 STATISTICAL VALIDATION — does this dashboard actually have edge?
    # Backtest + correlation + sensitivity + confidence + cycle-6 adjustment.
    # Surfaces the quant lens so user can SEE the math, not just trust it.
    # ═══════════════════════════════════════════════════════════════════
    try:
        from core.dashboard_cache import get_cached as _gc_v
        _val = _gc_v("rotation_validation")
        if not _val:
            from core.rotation_validation import all_validation
            _val = all_validation()

        st.markdown(
            f"<div class='section-header' style='font-size:16px; font-weight:700; "
            f"color:#4a90e2; margin-top:8px; display:flex; justify-content:space-between;'>"
            f"<span>📊 Statistical Validation — does this trigger actually have edge?</span>"
            f"<span>{_age_badge('rotation_validation')}</span></div>",
            unsafe_allow_html=True,
        )
        # 2026-07-07 claim-validity audit (H2): honest framing so this panel
        # isn't read as statistical proof. The sensitivity test is real; the
        # "confidence %" is a HEURISTIC index (includes a fixed neutral cycle-6
        # term), the backtest is n=3 cycles, and cycle 5's "bottom" is an
        # ESTIMATE ($58k) not a realized low — so the backtest partly grades
        # against a guess. Treat as a structured sanity check, not an edge proof.
        st.caption("Sanity check, not a statistical edge proof (n=3 cycles). "
                   "2026-07-08 rebuild: the score is now an honest EVIDENCE tally "
                   "(independent mechanisms firing + trigger-path progress) — the "
                   "old hardcoded cycle-6 fudge is removed. The backtest averages "
                   "over 2 REALIZED bottoms only (cycle 5 shown but excluded — its "
                   "bottom is an estimate). Directional, not a probability.")

        # Row 1: 4 headline metric cards
        _v1, _v2, _v3, _v4 = st.columns(4)
        _conf = _val.get("confidence", {}) or {}
        _corr = _val.get("correlation", {}) or {}
        _cyc6 = _val.get("cycle6", {}) or {}
        _sens = _val.get("sensitivity", {}) or {}

        # Card 1: Confidence score
        with _v1:
            _conf_pct = _conf.get("confidence_pct", 0) or 0
            _conf_tier = _conf.get("tier", "?")
            _conf_color = ("#22c55e" if _conf_pct >= 75 else
                            "#f0b90b" if _conf_pct >= 50 else
                            "#888" if _conf_pct >= 25 else "#ef4444")
            st.markdown(
                f"<div style='padding:12px 14px; border-radius:8px; background:#13161c; "
                f"border-left:4px solid {_conf_color}; min-height:120px;'>"
                f"<div style='font-size:10px; color:#888; text-transform:uppercase;'>"
                f"Confidence score</div>"
                f"<div style='font-size:28px; font-weight:800; color:{_conf_color}; "
                f"line-height:1.1;'>{_conf_pct:.0f}%</div>"
                f"<div style='font-size:11px; color:#ccc; font-weight:600;'>{_conf_tier}</div>"
                f"<div style='font-size:10px; color:#aaa; margin-top:4px;'>"
                f"% of independent clusters firing</div>"
                f"</div>", unsafe_allow_html=True,
            )

        # Card 2: Cycle-6 era detector
        with _v2:
            _era = _cyc6.get("era", "?")
            _dd = _cyc6.get("current_dd_pct", 0) or 0
            _hist_dd = _cyc6.get("avg_historical_dd", 0) or 0
            _scale = _cyc6.get("suggested_scale", 1.0) or 1.0
            _era_color = ("#ef4444" if _era == "ETF_MUTED" else
                           "#f0b90b" if _era == "MILD_MUTED" else "#22c55e")
            st.markdown(
                f"<div style='padding:12px 14px; border-radius:8px; background:#13161c; "
                f"border-left:4px solid {_era_color}; min-height:120px;'>"
                f"<div style='font-size:10px; color:#888; text-transform:uppercase;'>"
                f"Cycle-6 detector</div>"
                f"<div style='font-size:18px; font-weight:800; color:{_era_color}; "
                f"line-height:1.1;'>{_era}</div>"
                f"<div style='font-size:11px; color:#aaa; margin-top:4px;'>"
                f"DD: {_dd:.1f}% vs hist {_hist_dd:.0f}%</div>"
                f"<div style='font-size:11px; color:#aaa;'>"
                f"Thresholds × {_scale:.2f}</div>"
                f"</div>", unsafe_allow_html=True,
            )

        # Card 3: Effective signals (correlation deduplication)
        with _v3:
            _raw = _corr.get("raw_firing", "0/15")
            _eff = _corr.get("clusters_firing", "0/6")
            _eff_pct = _corr.get("effective_pct", 0) or 0
            _eff_color = "#22c55e" if _eff_pct >= 50 else "#f0b90b" if _eff_pct >= 25 else "#888"
            st.markdown(
                f"<div style='padding:12px 14px; border-radius:8px; background:#13161c; "
                f"border-left:4px solid {_eff_color}; min-height:120px;'>"
                f"<div style='font-size:10px; color:#888; text-transform:uppercase;'>"
                f"Effective signals</div>"
                f"<div style='font-size:28px; font-weight:800; color:{_eff_color}; "
                f"line-height:1.1;'>{_eff}</div>"
                f"<div style='font-size:10px; color:#aaa; margin-top:4px;'>"
                f"clusters firing (dedup)</div>"
                f"<div style='font-size:10px; color:#888;'>"
                f"raw: {_raw}</div>"
                f"</div>", unsafe_allow_html=True,
            )

        # Card 4: Threshold sensitivity
        with _v4:
            _sens_interp = _sens.get("interpretation", "?")[:80]
            _is_robust = "ROBUST" in _sens_interp.upper()
            _sens_color = "#22c55e" if _is_robust else "#f0b90b"
            _sens_short = ("ROBUST" if _is_robust else
                            "BORDERLINE" if "BORDERLINE" in _sens_interp.upper() else
                            "CONFIRMED" if "CONFIRMED" in _sens_interp.upper() else "MIXED")
            st.markdown(
                f"<div style='padding:12px 14px; border-radius:8px; background:#13161c; "
                f"border-left:4px solid {_sens_color}; min-height:120px;'>"
                f"<div style='font-size:10px; color:#888; text-transform:uppercase;'>"
                f"Threshold sensitivity</div>"
                f"<div style='font-size:20px; font-weight:800; color:{_sens_color}; "
                f"line-height:1.1; margin-top:4px;'>{_sens_short}</div>"
                f"<div style='font-size:10px; color:#aaa; margin-top:6px; line-height:1.3;'>"
                f"trigger tested ±10% on all thresholds</div>"
                f"</div>", unsafe_allow_html=True,
            )

        # Cycle-6 explanatory message
        if _cyc6.get("message"):
            _bg_color = ("rgba(239,68,68,0.15)" if _cyc6.get("era") == "ETF_MUTED"
                          else "rgba(240,185,11,0.15)" if _cyc6.get("era") == "MILD_MUTED"
                          else "rgba(34,197,94,0.15)")
            st.markdown(
                f"<div style='padding:10px 14px; margin-top:10px; border-radius:6px; "
                f"background:{_bg_color}; border-left:3px solid {_era_color};'>"
                f"<div style='font-size:11px; color:#fff; line-height:1.4;'>"
                f"<b>Cycle-6 modifier ({_era}):</b> {_cyc6.get('message')}</div>"
                f"</div>", unsafe_allow_html=True,
            )

        # Historical backtest expander
        with st.expander("📜 Historical backtest — when would the trigger have fired?", expanded=False):
            _bt = _val.get("backtest", {}) or {}
            _bt_results = _bt.get("results", []) or []
            if _bt_results:
                import pandas as _pd
                rows = []
                for r in _bt_results:
                    if r.get("would_have_fired"):
                        _est = r.get("bottom_is_estimate")
                        rows.append({
                            "Cycle":            f"Cycle {r.get('cycle', '?')}" + (" (est.)" if _est else ""),
                            "Fire date":        r.get("fire_date", "?"),
                            "Fire price":       f"${r.get('fire_price', 0):,.0f}",
                            "Actual bottom":    (r.get("actual_bottom", "?") + " ~est" if _est
                                                 else r.get("actual_bottom", "?")),
                            "Bottom price":     f"${r.get('actual_btm_price', 0):,.0f}" + ("*" if _est else ""),
                            "Days vs bottom":   ("excl." if _est else str(r.get("days_vs_bottom", 0))),
                            "% from low":       ("excl." if _est else f"{r.get('pct_from_bottom', 0):+.1f}%"),
                        })
                    else:
                        rows.append({
                            "Cycle":            f"Cycle {r.get('cycle', '?')}",
                            "Fire date":        "did not fire",
                            "Fire price":       "—",
                            "Actual bottom":    "—",
                            "Bottom price":     "—",
                            "Days vs bottom":   "—",
                            "% from low":       "—",
                        })
                st.dataframe(_pd.DataFrame(rows), width='stretch', hide_index=True)
                if _bt.get("avg_days_vs_bottom") is not None:
                    st.caption(
                        f"**Average**: trigger would have fired "
                        f"**{abs(int(_bt.get('avg_days_vs_bottom', 0)))} days "
                        f"{'before' if _bt.get('avg_days_vs_bottom', 0) < 0 else 'after'}** "
                        f"the actual bottom, at "
                        f"**{_bt.get('avg_pct_from_bottom', 0):+.1f}%** from the absolute low "
                        f"(over {_bt.get('n_fired_realized', _bt.get('n_realized_cycles','?'))} "
                        f"REALIZED bottoms; cycle 5 excluded — *estimated bottom). "
                        f"Price-only proxies; the real 15-signal scorecard fires more precisely."
                    )

        # Correlation cluster expander
        with st.expander("🔬 Signal correlation — independent vs redundant", expanded=False):
            _cl = _corr.get("cluster_breakdown", {}) or {}
            st.caption(_corr.get("interpretation", ""))
            for name, info in _cl.items():
                if info.get("n_total", 0) == 0: continue
                _firing = info.get("n_firing", 0)
                _total = info.get("n_total", 0)
                _active = info.get("active", False)
                _color = "#22c55e" if _active else "#888"
                _badge = "🔥 firing" if _active else "○ dormant"
                st.markdown(
                    f"<div style='padding:6px 10px; margin-bottom:4px; "
                    f"background:#13161c; border-radius:4px; border-left:3px solid {_color};'>"
                    f"<div style='display:flex; justify-content:space-between;'>"
                    f"<span style='font-size:11px; color:#ccc; font-weight:600;'>{name}</span>"
                    f"<span style='font-size:11px; color:{_color};'>"
                    f"{_firing}/{_total} · {_badge}</span></div>"
                    f"<div style='font-size:10px; color:#888; margin-top:2px;'>"
                    f"{' · '.join(info.get('labels', []))[:200]}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # Sensitivity scenarios expander
        with st.expander("⚖️ Sensitivity bands — strict vs loose thresholds", expanded=False):
            _scenarios = _sens.get("scenarios", {}) or {}
            for label in ["strict", "baseline", "loose"]:
                s = _scenarios.get(label, {})
                if not s: continue
                _st = s.get("status", "?")
                _bs = s.get("best_score", "?")
                _color = ("#ef4444" if s.get("fired") else
                            "#f0b90b" if _st == "WARMING" else "#22c55e")
                st.markdown(
                    f"<div style='padding:8px 12px; margin-bottom:4px; "
                    f"background:#13161c; border-radius:4px; border-left:3px solid {_color};'>"
                    f"<div style='display:flex; justify-content:space-between;'>"
                    f"<span style='font-size:11px; color:#ccc; font-weight:600;'>"
                    f"{label.upper()} (+/-10%)</span>"
                    f"<span style='font-size:11px; color:{_color};'>"
                    f"{_st} · score {_bs}/2</span></div>"
                    f"<div style='font-size:10px; color:#888; margin-top:2px;'>"
                    f"BTC overwhelming: {s.get('thresholds', {}).get('btc_bottom_overwhelming')}/16 · "
                    f"BTC moderate: {s.get('thresholds', {}).get('btc_bottom_moderate')}/16 · "
                    f"QQQ gap: ${s.get('thresholds', {}).get('qqq_gap_level')}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<hr style='border-color:#2a2d36; margin:18px 0;'>",
                    unsafe_allow_html=True)
    except Exception as _ve:
        st.caption(f"Validation panel — temporarily unavailable")


with tab_signals:   # <- 2026-07-04 restructure
    # ═══════════════════════════════════════════════════════════════════
    # 🔺 BTC BOTTOM WATCH — when to BUY BTC (primary focus, post-peak phase)
    # ALL signals in this block are BTC-specific. Equity tracking is below
    # in its own EQUITY TOP WATCH section to keep the two decisions distinct.
    # ═══════════════════════════════════════════════════════════════════
    try:
        from core.dashboard_cache import get_cached as _gc
        st.markdown(
            "<div class='section-header' style='font-size:18px; font-weight:700; "
            "color:#22c55e; margin-top:8px;'>"
            "🔺 BTC BOTTOM WATCH — <span style='font-size:13px; color:#888;'>"
            "when to BUY BTC (cycle signals)</span></div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Two scorecards by design — this **16-signal watch-list** is the wide radar "
            "(*how close are we?*); the tighter **10-signal hard-confirmation gate** is what "
            "actually fires the deploy triggers (deploy at 7/10). Different jobs, different totals."
        )

        # Row 1: bottom scorecard verdict + bottom date + ETF trigger + halving day
        _bw_c1, _bw_c2, _bw_c3, _bw_c4 = st.columns(4)

        # Card 1: Native bottom scorecard (THE key panel)
        with _bw_c1:
            _nb = _gc("btc_native_bottom_scorecard")
            if _nb:
                _lv = _nb.get("verdict_level", "HOLD")
                _n = _nb.get("n_met") or 0
                _tot = _nb.get("n_total") or 16
                _bw_col = _bottom_color(_lv)
                st.markdown(
                    f"<div style='padding:14px; border-radius:8px; background:#13161c; "
                    f"border:2px solid {_bw_col}; min-height:140px;'>"
                    f"<div style='font-size:10px; color:#888; text-transform:uppercase;'>🔺 BTC BOTTOM <span style='color:#22c55e;'>(buy BTC?)</span></div>"
                    f"<div style='font-size:30px; font-weight:800; color:{_bw_col}; line-height:1;'>"
                    f"{_n}<span style='font-size:14px; color:#888;'>/{_tot}</span></div>"
                    f"<div style='font-size:11px; color:#ccc; font-weight:600; "
                    f"margin-top:4px;'>{_lv.replace('_',' ')}</div>"
                    f"<div style='font-size:10px; color:#aaa; margin-top:6px; line-height:1.4;'>"
                    f"{_dormant_status(_n, _tot, 'bottom')}</div>"
                    f"</div>", unsafe_allow_html=True,
                )

        # Card 2: ETF-aware bottom trigger (the master verdict)
        with _bw_c2:
            try:
                _etf_v = verdict_label
                _etf_c = verdict_color
                _etf_sub = verdict_sub
                st.markdown(
                    f"<div style='padding:14px; border-radius:8px; background:#13161c; "
                    f"border:2px solid {_etf_c}; min-height:140px;'>"
                    f"<div style='font-size:10px; color:#888; text-transform:uppercase;'>ETF-Aware Trigger</div>"
                    f"<div style='font-size:22px; font-weight:800; color:{_etf_c}; "
                    f"line-height:1.1; margin-top:4px;'>{_etf_v}</div>"
                    f"<div style='font-size:10px; color:#aaa; margin-top:8px; line-height:1.3;'>{_etf_sub}</div>"
                    f"</div>", unsafe_allow_html=True,
                )
            except Exception: pass

        # Card 3: Bottom date convergence (WHEN) -- bottom_date_convergence()
        # returns: ev_date, n_methods, spread_days, summary, estimates
        with _bw_c3:
            try:
                _dp = _gc("date_predictions") or {}
                _conv = _dp.get("convergence", {}) or {}
                _date = _conv.get("ev_date", "?")
                _n_methods = _conv.get("n_methods", 0) or 0
                _spread = _conv.get("spread_days", 0) or 0
                # Days until ev_date
                _days_out = "?"
                try:
                    if _date and _date != "?":
                        _ev_d = datetime.fromisoformat(_date).date()
                        _days_out = (_ev_d - datetime.now(timezone.utc).date()).days
                except Exception:
                    pass
                # Confidence proxy: tighter spread + more methods = higher confidence
                if _n_methods >= 4 and _spread < 120:
                    _conf = "HIGH"
                elif _n_methods >= 3 and _spread < 200:
                    _conf = "MEDIUM"
                else:
                    _conf = "LOW"
                st.markdown(
                    f"<div style='padding:14px; border-radius:8px; background:#13161c; "
                    f"border:2px solid #f0b90b; min-height:140px;'>"
                    f"<div style='font-size:10px; color:#888; text-transform:uppercase;'>"
                    f"Bottom Date Convergence</div>"
                    f"<div style='font-size:18px; font-weight:800; color:#f0b90b; "
                    f"line-height:1; margin-top:4px;'>{_date}</div>"
                    f"<div style='font-size:11px; color:#aaa; margin-top:6px;'>"
                    f"~<b>{_days_out}</b> days away</div>"
                    f"<div style='font-size:11px; color:#aaa;'>"
                    f"{_n_methods} methods · ±{_spread}d spread · <b>{_conf}</b></div>"
                    f"</div>", unsafe_allow_html=True,
                )
            except Exception as _ee:
                st.caption(f"convergence — temporarily unavailable")

        # Card 4: Halving Day + Cycle phase
        with _bw_c4:
            try:
                _hp = pos  # already loaded earlier
                _days_post = _hp.get("days_post_halving", 0)
                _proj_bot = _hp.get("projected_bottom_date", "?")
                _phase = phase_info.get("phase", "?") if isinstance(phase_info, dict) else "?"
                st.markdown(
                    f"<div style='padding:14px; border-radius:8px; background:#13161c; "
                    f"border:2px solid #4a90e2; min-height:140px;'>"
                    f"<div style='font-size:10px; color:#888; text-transform:uppercase;'>"
                    f"Halving Cycle (cycle 5)</div>"
                    f"<div style='font-size:24px; font-weight:800; color:#4a90e2; "
                    f"line-height:1; margin-top:4px;'>Day {_days_post}</div>"
                    f"<div style='font-size:11px; color:#aaa; margin-top:6px;'>"
                    f"Phase: <b>{_phase}</b></div>"
                    f"<div style='font-size:11px; color:#aaa;'>"
                    f"Pattern bottom: <b>{_proj_bot}</b></div>"
                    f"</div>", unsafe_allow_html=True,
                )
            except Exception:
                st.caption("halving data not ready")

        # Row 2: Realized Cap drawdown thermometer + Bottom signals composite
        _bw_r2c1, _bw_r2c2 = st.columns([1, 1])
        with _bw_r2c1:
            try:
                _rcd_val = _gc("realized_cap_drawdown") or rcd
                if _rcd_val:
                    # Correct key per realized_cap_drawdown_depth():
                    #   current_drawdown_pct (NOT drawdown_pct), note (NOT verdict)
                    _dd = _rcd_val.get("current_drawdown_pct", 0) or 0
                    _verdict = (_rcd_val.get("note") or
                                _rcd_val.get("verdict") or
                                "no drawdown yet")
                    # Color: green when drawdown deep (buying zone)
                    if _dd <= -25: _c = "#22c55e"
                    elif _dd <= -15: _c = "#f0b90b"
                    elif _dd <= -5:  _c = "#aaa"
                    else: _c = "#888"
                    # Thermometer bar (0 = no DD, -50 = deep value)
                    _pct_fill = min(abs(_dd) / 50 * 100, 100)
                    st.markdown(
                        f"<div style='padding:14px; border-radius:8px; background:#13161c; "
                        f"border-left:4px solid {_c};'>"
                        f"<div style='display:flex; justify-content:space-between;'>"
                        f"<span style='font-size:10px; color:#888; text-transform:uppercase;'>"
                        f"Realized Cap Drawdown — how deep is the pain?</span>"
                        f"<span>{_age_badge('realized_cap_drawdown')}</span></div>"
                        f"<div style='font-size:28px; font-weight:800; color:{_c}; line-height:1.1;'>"
                        f"{_dd:.1f}%</div>"
                        f"<div style='background:#222; border-radius:4px; height:8px; "
                        f"margin-top:8px; overflow:hidden;'>"
                        f"<div style='background:{_c}; width:{_pct_fill}%; height:100%;'></div></div>"
                        f"<div style='font-size:10px; color:#888; margin-top:4px;'>"
                        f"0% (ATH)&nbsp;&nbsp;|&nbsp;&nbsp;-15% (typical bear)&nbsp;&nbsp;|"
                        f"&nbsp;&nbsp;-25%+ (capitulation)</div>"
                        f"<div style='font-size:12px; color:#ccc; margin-top:6px;'>{_verdict}</div>"
                        f"</div>", unsafe_allow_html=True,
                    )
            except Exception as _ee:
                st.caption(f"RCap drawdown — temporarily unavailable")

        with _bw_r2c2:
            try:
                _bs = _gc("bottom_signals") or bottom_sigs
                if _bs:
                    # bottom_signals returns {signal_name: {value, score, note, ...}, ...}
                    # No top-level composite. Compute mean of per-signal scores.
                    _scores = [v.get("score", 0) for v in _bs.values()
                                if isinstance(v, dict) and isinstance(v.get("score"), (int, float))]
                    _comp = (sum(_scores) / len(_scores)) if _scores else 0
                    # Best interpretation: pick the note from the highest-scoring signal
                    _best = max(((v.get("score", 0), v.get("note", ""))
                                  for v in _bs.values()
                                  if isinstance(v, dict)),
                                default=(0, ""))
                    _interp = (_best[1] or "")[:140]
                    if _comp >= 0.6:    _bc = "#22c55e"
                    elif _comp >= 0.3:  _bc = "#f0b90b"
                    else:               _bc = "#888"
                    # Bar (0 to 1)
                    _bar_fill = min(_comp * 100, 100)
                    st.markdown(
                        f"<div style='padding:14px; border-radius:8px; background:#13161c; "
                        f"border-left:4px solid {_bc};'>"
                        f"<div style='display:flex; justify-content:space-between;'>"
                        f"<span style='font-size:10px; color:#888; text-transform:uppercase;'>"
                        f"Bottom Signals Composite — how close are we?</span>"
                        f"<span>{_age_badge('bottom_signals')}</span></div>"
                        f"<div style='font-size:28px; font-weight:800; color:{_bc}; line-height:1.1;'>"
                        f"{_comp:.2f}<span style='font-size:14px; color:#888;'>/1.00</span></div>"
                        f"<div style='background:#222; border-radius:4px; height:8px; "
                        f"margin-top:8px; overflow:hidden;'>"
                        f"<div style='background:{_bc}; width:{_bar_fill}%; height:100%;'></div></div>"
                        f"<div style='font-size:10px; color:#888; margin-top:4px;'>"
                        f"0 (no signal)&nbsp;&nbsp;|&nbsp;&nbsp;0.3 (warming)&nbsp;&nbsp;|"
                        f"&nbsp;&nbsp;0.6+ (firing)</div>"
                        f"<div style='font-size:12px; color:#ccc; margin-top:6px;'>{_interp}</div>"
                        f"</div>", unsafe_allow_html=True,
                    )
            except Exception as _ee:
                st.caption(f"bottom signals — temporarily unavailable")

        # ─── OLSON BTC TARGET BAND ($52k-$57k bearish W pattern) ───
        # When BTC enters this band, Olson's call has played out -> consider deploy
        try:
            _btc_now = float(get_live_btc_ticker().get("last", 0) or 0)
            _olson_lo, _olson_hi = 52_000, 57_000
            _pct_to_band_top = (_olson_hi / _btc_now - 1) * 100 if _btc_now else 0
            _pct_to_band_bot = (_olson_lo / _btc_now - 1) * 100 if _btc_now else 0

            if _olson_lo <= _btc_now <= _olson_hi:
                _ob_status = "IN_BAND"
                _ob_color = "#22c55e"
                _ob_msg = (f"🎯 BTC ${_btc_now:,.0f} is INSIDE Olson's bearish W target "
                           f"band (${_olson_lo:,.0f}-${_olson_hi:,.0f}). "
                           f"His top call is hitting — consider scaling INTO BTC.")
            elif _btc_now > _olson_hi:
                _ob_status = "ABOVE"
                _ob_color = "#f0b90b"
                _ob_msg = (f"BTC ${_btc_now:,.0f} above Olson's target. "
                           f"{_pct_to_band_top:+.1f}% to enter band top (${_olson_hi:,.0f}). "
                           f"If his W pattern plays out, expect drop to $52-57k.")
            else:
                _ob_status = "BELOW"
                _ob_color = "#ef4444"
                _ob_msg = (f"BTC ${_btc_now:,.0f} BELOW Olson's $52k floor. "
                           f"His target exceeded — deeper drawdown than predicted.")

            st.markdown(
                f"<div style='padding:12px 14px; margin:10px 0; border-radius:8px; "
                f"background:#13161c; border-left:4px solid {_ob_color};'>"
                f"<div style='font-size:10px; color:#888; text-transform:uppercase; "
                f"letter-spacing:1px;'>"
                f"🎯 Olson BTC Target — bearish W pattern $52k-$57k</div>"
                f"<div style='font-size:18px; font-weight:700; color:{_ob_color}; "
                f"margin-top:4px;'>{_ob_status}</div>"
                f"<div style='font-size:12px; color:#ccc; margin-top:4px; line-height:1.4;'>"
                f"{_ob_msg}</div></div>",
                unsafe_allow_html=True,
            )
        except Exception as _obe:
            st.caption(f"Olson BTC target band: {_obe}")

        # ─── LATEST SWIFT TWEETS (parallel to Olson tweets in equity section) ───
        try:
            import json as _json2
            _sw_cache = REPO_ROOT / ".guru_positivecrypto_tweets_cache.json"
            if _sw_cache.exists():
                _sw_data = _json2.loads(_sw_cache.read_text())
                _sw_tweets = _sw_data.get("tweets", [])
                _sw_updated = _sw_data.get("updated", "")
                _sw_high = [t for t in _sw_tweets if t.get("relevance") == "HIGH"][:4]
                _sw_med = [t for t in _sw_tweets if t.get("relevance") == "MEDIUM"][:3]
                _sw_show = _sw_high + _sw_med
                if _sw_show:
                    st.markdown(
                        f"<div class='section-header' style='font-size:14px; "
                        f"margin-top:12px; color:#ccc;'>"
                        f"🎙️ Latest @PositiveCrypto (Phillip Swift) "
                        f"<span style='font-size:10px; color:#888;'>"
                        f"(updated {_sw_updated[:16]})</span></div>",
                        unsafe_allow_html=True,
                    )
                    for _t in _sw_show[:5]:
                        _r = _t.get("relevance", "?")
                        _rc = "#ef4444" if _r == "HIGH" else "#f0b90b"
                        _tx = html.escape((_t.get("text") or _t.get("title", ""))[:280])
                        _lk = _t.get("link", "")
                        _pb = (_t.get("pub", "") or "")[:16]
                        st.markdown(
                            f"<div style='padding:8px 12px; margin-bottom:6px; "
                            f"background:#13161c; border-radius:6px; "
                            f"border-left:3px solid {_rc}; font-size:12px;'>"
                            f"<div style='display:flex; justify-content:space-between; "
                            f"align-items:baseline;'>"
                            f"<span style='font-size:9px; color:{_rc}; "
                            f"font-weight:700;'>{_r}</span>"
                            f"<span style='font-size:9px; color:#888;'>{_pb}</span></div>"
                            f"<div style='color:#ccc; margin-top:3px; line-height:1.4;'>"
                            f"{_tx}</div>"
                            f"<a href='{_lk}' target='_blank' "
                            f"style='font-size:10px; color:#4a90e2;'>open on X →</a>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
        except Exception as _swe:
            st.caption(f"Swift tweets: {_swe}")

        # Detail expander — full bottom criteria list, only if user wants it
        with st.expander("📋 See all 15 bottom criteria + status of each", expanded=False):
            _nb_detail = _gc("btc_native_bottom_scorecard")
            if _nb_detail and _nb_detail.get("criteria"):
                import pandas as _pd
                _rows = [{"✓": "🔥" if c.get("met") else "○",
                          "Criterion": c.get("label", "?"),
                          "Status": (c.get("status", "?") or "")[:90]}
                         for c in _nb_detail.get("criteria", [])]
                st.dataframe(_pd.DataFrame(_rows), width='stretch', hide_index=True)
            else:
                st.caption("bottom criteria detail not available")

        st.markdown("<hr style='border-color:#2a2d36; margin:18px 0;'>",
                    unsafe_allow_html=True)
    except Exception as _e:
        st.caption(f"BTC Bottom Watch — temporarily unavailable")

    # ═══════════════════════════════════════════════════════════════════
    # 🟦 GLASSNODE-GRADE PROXIES — free-data approximations of paid-tier signals
    # Three signals James Check called the most predictive, adapted from
    # free CoinMetrics community data.
    # ═══════════════════════════════════════════════════════════════════
    try:
        from core.dashboard_cache import get_cached as _gc_gn
        _gn = _gc_gn("glassnode_proxies")
        if not _gn:
            from core.glassnode_proxies import all_glassnode_proxies
            _gn = all_glassnode_proxies()

        _gn_lth = _gn.get("lth_npc", {}) or {}
        _gn_asopr = _gn.get("asopr", {}) or {}
        _gn_cohort = _gn.get("cohort_pl", {}) or {}
        _gn_firing = _gn.get("n_firing", 0)

        # Hero status
        _gn_color = ("#22c55e" if _gn_firing >= 2 else
                       "#f0b90b" if _gn_firing >= 1 else "#888")
        _gn_msg = {
            3: "🚨 ALL 3 Glassnode proxies firing — generational bottom signal",
            2: "🔥 2 of 3 Glassnode proxies firing — strong bottom evidence",
            1: "⚠️ 1 of 3 firing — building",
            0: "○ All dormant — no on-chain bottom confirmation yet",
        }.get(_gn_firing, "?")
        st.markdown(
            f"<div style='padding:14px 18px; border-radius:10px; "
            f"background:#13161c; border:2px solid {_gn_color}; "
            f"margin-bottom:12px;'>"
            f"<div style='display:flex; justify-content:space-between; align-items:baseline;'>"
            f"<span style='font-size:11px; color:#888; text-transform:uppercase; "
            f"letter-spacing:1.5px; font-weight:700;'>"
            f"🟦 Glassnode-grade proxies · Check + Hagerty framework</span>"
            f"<span>{_age_badge('glassnode_proxies')}</span></div>"
            f"<div style='font-size:20px; font-weight:700; color:{_gn_color}; "
            f"margin-top:6px;'>{_gn_msg}</div>"
            f"</div>", unsafe_allow_html=True,
        )

        # 3 cards
        _gnc1, _gnc2, _gnc3 = st.columns(3)

        # Card 1: LTH NPC (with Check's velocity ask)
        with _gnc1:
            _lth_color = _gn_lth.get("color", "#888")
            _lth_phase = _gn_lth.get("phase", "?")
            _lth_npc = _gn_lth.get("npc_30d_pct", 0) or 0
            _lth_vel = _gn_lth.get("velocity_label", "?")
            _lth_vel_pct = _gn_lth.get("velocity_pct", 0) or 0
            _lth_interp = (_gn_lth.get("interpretation") or "")[:120]
            _lth_fire = _gn_lth.get("bottom_signal", False)
            _lth_badge = "🔥 FIRING" if _lth_fire else "○ dormant"
            _vel_color = ("#22c55e" if _lth_vel == "ACCELERATING" else
                            "#f0b90b" if _lth_vel == "STEADY" else "#ef4444")
            st.markdown(
                f"<div style='padding:12px 14px; border-radius:8px; background:#13161c; "
                f"border-left:4px solid {_lth_color}; min-height:180px;'>"
                f"<div style='display:flex; justify-content:space-between;'>"
                f"<span style='font-size:10px; color:#888; text-transform:uppercase;'>"
                f"LTH Net Position Change</span>"
                f"<span style='font-size:10px; color:{_lth_color};'>{_lth_badge}</span>"
                f"</div>"
                f"<div style='font-size:20px; font-weight:800; color:{_lth_color}; "
                f"margin-top:4px;'>{_lth_phase}</div>"
                f"<div style='font-size:11px; color:#aaa; margin-top:4px;'>"
                f"{_lth_npc:+.2f}% 30d ratio change</div>"
                f"<div style='font-size:11px; margin-top:4px;'>"
                f"<span style='color:#888;'>Velocity:</span> "
                f"<b style='color:{_vel_color};'>{_lth_vel}</b> "
                f"<span style='color:#888;'>({_lth_vel_pct:+.2f}%/wk)</span></div>"
                f"<div style='font-size:11px; color:#ccc; margin-top:6px; line-height:1.4;'>"
                f"{_lth_interp}</div>"
                f"</div>", unsafe_allow_html=True,
            )

        # Card 2: aSOPR
        with _gnc2:
            _as_color = _gn_asopr.get("color", "#888")
            _as_zone = _gn_asopr.get("zone", "?")
            _as_val = _gn_asopr.get("asopr_proxy", 1.0) or 1.0
            _as_interp = (_gn_asopr.get("interpretation") or "")[:140]
            _as_fire = _gn_asopr.get("bottom_signal", False)
            _as_badge = "🔥 FIRING" if _as_fire else "○ dormant"
            st.markdown(
                f"<div style='padding:12px 14px; border-radius:8px; background:#13161c; "
                f"border-left:4px solid {_as_color}; min-height:160px;'>"
                f"<div style='display:flex; justify-content:space-between;'>"
                f"<span style='font-size:10px; color:#888; text-transform:uppercase;'>"
                f"aSOPR proxy</span>"
                f"<span style='font-size:10px; color:{_as_color};'>{_as_badge}</span>"
                f"</div>"
                f"<div style='font-size:24px; font-weight:800; color:{_as_color}; "
                f"margin-top:4px;'>{_as_val:.3f}</div>"
                f"<div style='font-size:11px; color:#ccc; font-weight:600; margin-top:2px;'>"
                f"{_as_zone}</div>"
                f"<div style='font-size:11px; color:#ccc; margin-top:6px; line-height:1.4;'>"
                f"{_as_interp}</div>"
                f"</div>", unsafe_allow_html=True,
            )

        # Card 3: Cohort P/L
        with _gnc3:
            _co_color = _gn_cohort.get("color", "#888")
            _co_phase = _gn_cohort.get("phase", "?")
            _co_sth = _gn_cohort.get("sth_pl_pct", 0) or 0
            _co_lth = _gn_cohort.get("lth_pl_pct", 0) or 0
            _co_interp = (_gn_cohort.get("interpretation") or
                            _gn_cohort.get("error", "?"))[:140]
            _co_fire = _gn_cohort.get("bottom_signal", False)
            _co_badge = "🔥 FIRING" if _co_fire else "○ dormant"
            if _gn_cohort.get("error"):
                _co_phase = "PENDING DATA"
                _co_color = "#888"
                _co_interp = "Cost basis data refreshing — usually populates within 30 min."
            st.markdown(
                f"<div style='padding:12px 14px; border-radius:8px; background:#13161c; "
                f"border-left:4px solid {_co_color}; min-height:160px;'>"
                f"<div style='display:flex; justify-content:space-between;'>"
                f"<span style='font-size:10px; color:#888; text-transform:uppercase;'>"
                f"Cohort P/L (STH vs LTH)</span>"
                f"<span style='font-size:10px; color:{_co_color};'>{_co_badge}</span>"
                f"</div>"
                f"<div style='font-size:18px; font-weight:800; color:{_co_color}; "
                f"margin-top:4px;'>{_co_phase}</div>"
                f"<div style='font-size:11px; color:#aaa; margin-top:4px;'>"
                f"STH: {_co_sth:+.1f}% · LTH: {_co_lth:+.1f}%</div>"
                f"<div style='font-size:11px; color:#ccc; margin-top:6px; line-height:1.4;'>"
                f"{_co_interp}</div>"
                f"</div>", unsafe_allow_html=True,
            )

        st.caption(
            "⚠️ **Critical**: these proxies are NOT in the 16-signal bottom scorecard."
            "If both LTH NPC + aSOPR are firing, your TRUE bottom-signal count is "
            "**5/16 (not 3/16)**— Path 3 trigger may be 1 signal away from FIRE."
        )

        # === aSOPR historical chart (Hagerty ask) ===
        with st.expander("📈 aSOPR proxy — 60-day history (Hagerty chart view)", expanded=False):
            try:
                _hist = _gn_asopr.get("history_60d", []) or []
                if _hist:
                    import pandas as _pd
                    _df = _pd.DataFrame(_hist)
                    _df["d"] = _pd.to_datetime(_df["d"])
                    _df = _df.set_index("d")
                    st.line_chart(_df["v"], height=240, width='stretch')
                    st.caption(
                        f"Capitulation zone: < 0.98 (green = bottom signal). "
                        f"Current 7d avg: **{_gn_asopr.get('asopr_proxy', 0):.3f}**. "
                        f"Historical bottoms (Dec 2018, Nov 2022) showed sustained "
                        f"<0.98 for 2-4 weeks before bottom."
                    )
                else:
                    st.caption("aSOPR history pending — usually populates within 30 min of first run.")
            except Exception as _ahe:
                st.caption(f"chart — temporarily unavailable")

        # === GURU TRACK RECORDS (Anya's ask) ===
        with st.expander("🎙️ Guru track records — how reliable is each source?", expanded=False):
            st.caption("✅ Outcomes graded OBJECTIVELY from forward price action "
                       "(guru_grader) — not hand-scored. hit rate = right/(right+wrong) "
                       "over decisive calls; marginal & still-open calls excluded. "
                       "Remaining caveat: SURVIVORSHIP — the SET of calls is "
                       "author-selected (n≈3–4/guru), so read tiers as directional, "
                       "not definitive.")
            try:
                _gi = _gc_gn("guru_intelligence")
                if not _gi:
                    from core.guru_intelligence import all_guru_intelligence
                    _gi = all_guru_intelligence()
                _tracks = _gi.get("track_records", {}) or {}
                for handle, info in _tracks.items():
                    _hr = info.get("hit_rate_pct")
                    _tier = info.get("tier", "?")
                    _n_scored = info.get("n_calls_scored", 0)
                    # objective grading: a % on <3 decisive calls is noise — show n.
                    _hr_disp = (f"n={_n_scored} decisive · too few to rate"
                                if _tier == "INSUFFICIENT" or _hr is None
                                else f"{_hr:.0f}% graded ({_n_scored} decisive) · {_tier}")
                    _hr_color = ("#888" if (_tier == "INSUFFICIENT" or _hr is None)
                                 else "#22c55e" if _hr >= 75
                                 else "#f0b90b" if _hr >= 50 else "#ef4444")
                    st.markdown(
                        f"<div style='padding:10px 12px; margin-bottom:8px; "
                        f"background:#13161c; border-radius:6px; "
                        f"border-left:4px solid {_hr_color};'>"
                        f"<div style='display:flex; justify-content:space-between;'>"
                        f"<span style='font-size:13px; color:#fff; font-weight:600;'>"
                        f"{info.get('name', handle)} <span style='color:#888; font-size:11px;'>"
                        f"@{handle}</span></span>"
                        f"<span style='font-size:14px; color:{_hr_color}; font-weight:700;'>"
                        f"{_hr_disp}</span></div>"
                        f"<div style='font-size:11px; color:#aaa; margin-top:4px;'>"
                        f"{info.get('specialty', '')}</div>"
                        f"<div style='font-size:11px; color:#888; margin-top:4px;'>"
                        f"{info.get('n_right', 0)}/{info.get('n_calls_scored', 0)} decisive right · "
                        f"{info.get('n_marginal', 0)} marginal · "
                        f"{info.get('n_pending', len(info.get('pending_calls', [])))} open</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                # Recent HIGH calls aggregator
                _recent = _gi.get("recent_calls", []) or []
                if _recent:
                    st.markdown("**Recent HIGH-relevance calls (last 48h):**")
                    for c in _recent[:5]:
                        st.markdown(
                            f"<div style='font-size:11px; color:#ccc; margin-bottom:4px; "
                            f"padding-left:8px; border-left:2px solid #4a90e2;'>"
                            f"<b>{c.get('name', '?')}</b> ({c.get('pub', '')}): "
                            f"{c.get('text', '')[:200]}</div>",
                            unsafe_allow_html=True,
                        )
                # 📺 Latest guru YouTube uploads — free, IP-unblocked feed that
                # replaces the dead Nitter/Twitter scrape (works on the cloud too).
                import html as _h
                _gy = _gc_gn("guru_youtube") or {}
                _gylist = _gy.get("gurus", []) or []
                if _gylist:
                    st.markdown("**📺 Latest guru videos (YouTube — live):**")
                    for _g in _gylist:
                        for _v in (_g.get("videos", []) or [])[:2]:
                            st.markdown(
                                f"<div style='font-size:11px; color:#ccc; margin-bottom:4px; "
                                f"padding-left:8px; border-left:2px solid #ef4444;'>"
                                f"<b>{_h.escape(str(_g.get('name','?')))}</b> "
                                f"<span style='color:#888;'>({_h.escape(str(_v.get('date','')))})</span> "
                                f"<a href='{_h.escape(str(_v.get('url','')))}' target='_blank' "
                                f"style='color:#7fb3ff; text-decoration:none;'>"
                                f"{_h.escape(str(_v.get('title',''))[:90])}</a></div>",
                                unsafe_allow_html=True,
                            )
            except Exception as _gie:
                st.caption(f"guru track records: {_gie}")

        st.markdown("<hr style='border-color:#2a2d36; margin:18px 0;'>",
                    unsafe_allow_html=True)
    except Exception as _gne:
        st.caption(f"Glassnode proxies — temporarily unavailable")

    # ═══════════════════════════════════════════════════════════════════
    # 🔻 EQUITY TOP WATCH — when to SELL STOCKS (separate from BTC)
    # NZ context: 70% of stake is in equities; this section tells you when
    # to rotate out. Distinct decision from BTC bottom hunt above.
    # Sources:
    #   - QQQ Olson technical layer (589 trapdoor, 200wMA, MACD, RSI)
    #   - top_confirmation_scorecard (10 hard macro criteria)
    # ═══════════════════════════════════════════════════════════════════
    try:
        from core.dashboard_cache import get_cached as _gc_eq
        st.markdown(
            "<div class='section-header' style='font-size:18px; font-weight:700; "
            "color:#ef4444; margin-top:8px;'>"
            "🔻 EQUITY TOP WATCH — <span style='font-size:13px; color:#888;'>"
            "when to SELL STOCKS (Olson technicals + macro signals)</span></div>",
            unsafe_allow_html=True,
        )

        # ─── QQQ OLSON LAYER (prominent — first thing in this section) ───
        try:
            _qq = _gc_eq("equity_olson")
            if not _qq:
                from core.equity_olson import qqq_olson_verdict
                _qq = qqq_olson_verdict()

            _qq_tier = _qq.get("tier", "?")
            _qq_emoji = _qq.get("tier_emoji", "")
            _qq_color = _qq.get("color", "#888")
            _qq_action = _qq.get("action", "")
            _qq_close = _qq.get("last_close", 0) or 0
            _qq_gap = _qq.get("gap_level", 589)
            _qq_pct_gap = _qq.get("pct_to_gap", 0) or 0
            _qq_pct_dma = _qq.get("pct_to_dma200", 0) or 0
            _qq_pct_wma = _qq.get("pct_to_wma200", 0) or 0
            _qq_wma = _qq.get("wma200", 0) or 0
            _qq_dma = _qq.get("dma200", 0) or 0
            _qq_sigs = _qq.get("signals", {}) or {}
            _qq_macd = _qq_sigs.get("macd_3w", {}) or {}
            _qq_ha = _qq_sigs.get("heikin_ashi", {}) or {}
            _qq_rsi = _qq_sigs.get("rsi_divergence", {}) or {}

            # Hero verdict bar
            st.markdown(
                f"<div style='padding:16px 20px; border-radius:10px; "
                f"background:#13161c; border:3px solid {_qq_color}; "
                f"margin-bottom:14px;'>"
                f"<div style='display:flex; justify-content:space-between; align-items:baseline;'>"
                f"<span style='font-size:11px; color:#888; text-transform:uppercase; "
                f"letter-spacing:1.5px; font-weight:700;'>"
                f"🎯 Jesse Olson QQQ Layer · 589 Trapdoor Watch</span>"
                f"<span>{_age_badge('equity_olson')}</span></div>"
                f"<div style='font-size:32px; font-weight:800; color:{_qq_color}; "
                f"line-height:1.1; margin-top:4px;'>{_qq_emoji} {_qq_tier}</div>"
                f"<div style='font-size:13px; color:#ccc; margin-top:8px; line-height:1.4;'>"
                f"{_qq_action}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # 3 metric cards — QQQ current, 589 gap distance, 200wMA distance
            _q1, _q2, _q3 = st.columns(3)
            with _q1:
                # Color the price card by RSI state if overbought
                _px_border = "#ef4444" if _qq_rsi.get("rsi_overbought") else "#4a90e2"
                _px_caption = (f"RSI {_qq_rsi.get('rsi', 0):.0f} "
                               f"({_qq_rsi.get('trend', '?')})")
                st.markdown(
                    f"<div style='padding:12px 14px; border-radius:8px; background:#13161c; "
                    f"border-left:4px solid {_px_border};'>"
                    f"<div style='font-size:10px; color:#888; text-transform:uppercase;'>"
                    f"QQQ current</div>"
                    f"<div style='font-size:24px; font-weight:800; color:#fff; "
                    f"line-height:1.1;'>${_qq_close:,.0f}</div>"
                    f"<div style='font-size:11px; color:#aaa; margin-top:4px;'>{_px_caption}</div>"
                    f"</div>", unsafe_allow_html=True,
                )
            with _q2:
                # Distance to 589 trapdoor — color by zone
                _gap_color = ("#22c55e" if _qq_pct_gap < -10 else
                              "#f0b90b" if _qq_pct_gap < -5 else "#ef4444")
                _gap_label = ("SAFE distance" if _qq_pct_gap < -10 else
                              "WATCH zone" if _qq_pct_gap < -5 else
                              "DANGER — at trapdoor" if _qq_pct_gap < 0 else
                              "BROKEN — below gap")
                st.markdown(
                    f"<div style='padding:12px 14px; border-radius:8px; background:#13161c; "
                    f"border-left:4px solid {_gap_color};'>"
                    f"<div style='font-size:10px; color:#888; text-transform:uppercase;'>"
                    f"To Olson 589 gap</div>"
                    f"<div style='font-size:24px; font-weight:800; color:{_gap_color}; "
                    f"line-height:1.1;'>{_qq_pct_gap:+.1f}%</div>"
                    f"<div style='font-size:11px; color:#aaa; margin-top:4px;'>"
                    f"target ${_qq_gap:,.0f} · {_gap_label}</div>"
                    f"</div>", unsafe_allow_html=True,
                )
            with _q3:
                # Distance to 200wMA — the generational support
                _wma_color = "#888" if _qq_pct_wma < -25 else "#f0b90b"
                st.markdown(
                    f"<div style='padding:12px 14px; border-radius:8px; background:#13161c; "
                    f"border-left:4px solid {_wma_color};'>"
                    f"<div style='font-size:10px; color:#888; text-transform:uppercase;'>"
                    f"To 200-week SMA</div>"
                    f"<div style='font-size:24px; font-weight:800; color:{_wma_color}; "
                    f"line-height:1.1;'>{_qq_pct_wma:+.1f}%</div>"
                    f"<div style='font-size:11px; color:#aaa; margin-top:4px;'>"
                    f"target ${_qq_wma:,.0f} · generational support</div>"
                    f"</div>", unsafe_allow_html=True,
                )

            # Tech signals row — MACD + HA + 200dMA
            _t1, _t2, _t3 = st.columns(3)
            with _t1:
                _macd_color = "#22c55e" if _qq_macd.get("bullish") else "#ef4444"
                _macd_trend = _qq_macd.get("trend", "?")
                _macd_hist = _qq_macd.get("histogram", 0)
                st.markdown(
                    f"<div style='padding:10px 12px; border-radius:6px; background:#13161c; "
                    f"border-left:3px solid {_macd_color}; margin-top:8px;'>"
                    f"<div style='font-size:10px; color:#888;'>3-WEEK MACD</div>"
                    f"<div style='font-size:14px; color:{_macd_color}; font-weight:700;'>"
                    f"{_macd_trend}</div>"
                    f"<div style='font-size:10px; color:#aaa;'>"
                    f"histogram {_macd_hist:+.2f}</div></div>",
                    unsafe_allow_html=True,
                )
            with _t2:
                _ha_color = "#22c55e" if _qq_ha.get("bullish") else "#ef4444"
                _ha_recent = " ".join(_qq_ha.get("recent_colors", [])[-8:])
                st.markdown(
                    f"<div style='padding:10px 12px; border-radius:6px; background:#13161c; "
                    f"border-left:3px solid {_ha_color}; margin-top:8px;'>"
                    f"<div style='font-size:10px; color:#888;'>WEEKLY HEIKIN ASHI</div>"
                    f"<div style='font-size:14px; color:{_ha_color}; font-weight:700;'>"
                    f"{_qq_ha.get('current_color', '?')} × {_qq_ha.get('streak_weeks', 0)}w</div>"
                    f"<div style='font-size:10px; color:#aaa;'>"
                    f"last 8: {_ha_recent}</div></div>",
                    unsafe_allow_html=True,
                )
            with _t3:
                _dma_color = "#22c55e" if _qq_pct_dma < 0 else "#ef4444"
                st.markdown(
                    f"<div style='padding:10px 12px; border-radius:6px; background:#13161c; "
                    f"border-left:3px solid {_dma_color}; margin-top:8px;'>"
                    f"<div style='font-size:10px; color:#888;'>200-DAY SMA</div>"
                    f"<div style='font-size:14px; color:{_dma_color}; font-weight:700;'>"
                    f"${_qq_dma:,.0f} ({_qq_pct_dma:+.1f}%)</div>"
                    f"<div style='font-size:10px; color:#aaa;'>"
                    f"near-term support</div></div>",
                    unsafe_allow_html=True,
                )

            # ─── SEMIS LEADING TELL + OLSON GAP ROADMAP ───
            _s1, _s2 = st.columns(2)
            with _s1:
                try:
                    _sem = _gc_eq("equity_semis")
                    if not _sem:
                        from core.equity_semis import semis_tell
                        _sem = semis_tell()
                    if _sem and not _sem.get("error"):
                        _sem_color = _sem.get("color", "#888")
                        _sem_tier = _sem.get("tier", "?")
                        _sem_off = _sem.get("pct_off_high", 0) or 0
                        _sem_v50 = _sem.get("pct_vs_50", 0) or 0
                        st.markdown(
                            f"<div style='padding:10px 12px; border-radius:6px; "
                            f"background:#13161c; border-left:3px solid {_sem_color}; "
                            f"margin-top:8px;'>"
                            f"<div style='font-size:10px; color:#888;'>"
                            f"🔬 SEMIS TELL (SOXX) — leads QQQ</div>"
                            f"<div style='font-size:14px; color:{_sem_color}; font-weight:700;'>"
                            f"{_sem_tier}</div>"
                            f"<div style='font-size:10px; color:#aaa;'>"
                            f"{_sem_off:+.1f}% off high · {_sem_v50:+.1f}% vs 50dMA · "
                            f"earlier than QQQ</div></div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.caption("semis tell: data pending")
                except Exception as _se:
                    st.caption(f"semis: {_se}")
            with _s2:
                _ng = _qq.get("next_gap")
                _ngp = _qq.get("next_gap_pct")
                _gf = _qq.get("gaps_filled", 0)
                _gt = _qq.get("gaps_total", 0)
                if _ng:
                    st.markdown(
                        f"<div style='padding:10px 12px; border-radius:6px; "
                        f"background:#13161c; border-left:3px solid #f0b90b; "
                        f"margin-top:8px;'>"
                        f"<div style='font-size:10px; color:#888;'>"
                        f"🎯 OLSON NEXT GAP (downside roadmap)</div>"
                        f"<div style='font-size:14px; color:#f0b90b; font-weight:700;'>"
                        f"${_ng:,.0f} ({_ngp:+.1f}%)</div>"
                        f"<div style='font-size:10px; color:#aaa;'>"
                        f"{_gf}/{_gt} gaps filled · price fills these in sequence down</div></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<div style='padding:10px 12px; border-radius:6px; "
                        f"background:#13161c; border-left:3px solid #22c55e; "
                        f"margin-top:8px;'>"
                        f"<div style='font-size:10px; color:#888;'>"
                        f"🎯 OLSON GAP ROADMAP</div>"
                        f"<div style='font-size:13px; color:#22c55e; font-weight:700;'>"
                        f"above all mapped gaps</div>"
                        f"<div style='font-size:10px; color:#aaa;'>"
                        f"no downside gap immediately below</div></div>",
                        unsafe_allow_html=True,
                    )

            # Equity drawdown risk — market % only (personal NZ$ shown when private)
            if SHOW_PERSONAL:
                _stake_nzd = 130_000
                _equity_nzd = int(_stake_nzd * 0.70)
                _dd_if_wma = int(_equity_nzd * (_qq_pct_wma / 100))
                _dd_if_gap = int(_equity_nzd * (_qq_pct_gap / 100))
                _exposure_line = (
                    f"To 589 gap: <b style='color:#f0b90b;'>{_dd_if_gap:+,} NZD</b> "
                    f"on your <b>{_money(_equity_nzd)}</b> equity&nbsp;|&nbsp;"
                    f"To 200wMA: <b style='color:#ef4444;'>{_dd_if_wma:+,} NZD</b> "
                    f"({_qq_pct_wma:+.1f}%)"
                )
            else:
                _exposure_line = (
                    f"To 589 gap: <b style='color:#f0b90b;'>{_qq_pct_gap:+.1f}%</b>&nbsp;|&nbsp;"
                    f"To 200wMA: <b style='color:#ef4444;'>{_qq_pct_wma:+.1f}%</b> "
                    f"(potential equity drawdown if QQQ retraces, ~1:1)"
                )
            st.markdown(
                f"<div style='padding:10px 12px; margin-top:8px; background:#13161c; "
                f"border-radius:6px; border-left:3px solid #888;'>"
                f"<div style='font-size:11px; color:#888;'>"
                f"📊 Equity drawdown risk if QQQ retraces:</div>"
                f"<div style='font-size:12px; color:#ccc; margin-top:4px;'>"
                f"{_exposure_line}</div></div>",
                unsafe_allow_html=True,
            )

            # ─── LATEST OLSON TWEETS (from nitter monitor) ───
            try:
                import json as _json
                _olson_cache = REPO_ROOT / ".jesse_olson_tweets_cache.json"
                if _olson_cache.exists():
                    _olson_data = _json.loads(_olson_cache.read_text())
                    _olson_tweets = _olson_data.get("tweets", [])
                    _olson_updated = _olson_data.get("updated", "")
                    # Show top 5 HIGH-relevance, fallback to MEDIUM
                    _high = [t for t in _olson_tweets if t.get("relevance") == "HIGH"][:5]
                    _med = [t for t in _olson_tweets if t.get("relevance") == "MEDIUM"][:3]
                    _show = _high + _med
                    if _show:
                        st.markdown(
                            f"<div class='section-header' style='font-size:14px; "
                            f"margin-top:12px; color:#ccc;'>"
                            f"🎙️ Latest @JesseOlson tweets "
                            f"<span style='font-size:10px; color:#888;'>"
                            f"(updated {_olson_updated[:16]})</span></div>",
                            unsafe_allow_html=True,
                        )
                        for _t in _show[:6]:
                            _rel = _t.get("relevance", "?")
                            _rel_color = ("#ef4444" if _rel == "HIGH" else "#f0b90b")
                            _txt = html.escape((_t.get("text") or _t.get("title", ""))[:280])
                            _link = _t.get("link", "")
                            _pub = (_t.get("pub", "") or "")[:16]
                            st.markdown(
                                f"<div style='padding:8px 12px; margin-bottom:6px; "
                                f"background:#13161c; border-radius:6px; "
                                f"border-left:3px solid {_rel_color}; font-size:12px;'>"
                                f"<div style='display:flex; justify-content:space-between; "
                                f"align-items:baseline;'>"
                                f"<span style='font-size:9px; color:{_rel_color}; "
                                f"font-weight:700; text-transform:uppercase;'>{_rel}</span>"
                                f"<span style='font-size:9px; color:#888;'>{_pub}</span>"
                                f"</div>"
                                f"<div style='color:#ccc; margin-top:3px; line-height:1.4;'>"
                                f"{_txt}</div>"
                                f"<a href='{_link}' target='_blank' "
                                f"style='font-size:10px; color:#4a90e2;'>open on X →</a>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
            except Exception as _ote:
                st.caption(f"Olson tweets: {_ote}")

            st.markdown(
                "<hr style='border-color:#2a2d36; margin:14px 0;'>",
                unsafe_allow_html=True,
            )
        except Exception as _qe:
            st.caption(f"QQQ Olson layer — temporarily unavailable")

        _ts = _gc_eq("top_scorecard") or {}
        _scorecard = _ts.get("scorecard") if isinstance(_ts, dict) else None
        _recommendation = _ts.get("recommendation") if isinstance(_ts, dict) else None

        if not _scorecard:
            try:
                _scorecard = cached_top_scorecard().get("scorecard", {})
                _recommendation = cached_top_scorecard().get("recommendation")
            except Exception: pass

        # Equity top scorecard summary card row
        _eq_c1, _eq_c2, _eq_c3 = st.columns([1.2, 1, 1])

        # Card 1: Scorecard verdict + n_met (uses actual keys from top_confirmation_scorecard)
        with _eq_c1:
            try:
                # Real keys: verdict_level (HOLD/WATCH/TRIM_25/...), n_met, n_total, verdict (text)
                _eq_level = (_scorecard or {}).get("verdict_level") or "HOLD"
                _eq_n_hard = (_scorecard or {}).get("n_met") or 0
                _eq_n_hard_total = (_scorecard or {}).get("n_total") or 10
                _eq_color = _top_color(_eq_level if isinstance(_eq_level, str) else "HOLD")
                # Map verdict_level to readable display
                _eq_label_map = {
                    "HOLD": "HOLD STOCKS", "WATCH": "WATCH (mild stress)",
                    "TRIM_25": "TRIM 25%", "TRIM": "TRIM",
                    "SCALE_OUT_25": "SCALE OUT 25%", "SCALE_OUT_50": "SCALE OUT 50%",
                    "EXIT_75": "EXIT 75%", "EXIT_100": "EXIT 100%",
                    "DEFENSIVE": "DEFENSIVE",
                }
                _eq_display = _eq_label_map.get(_eq_level, _eq_level)
                st.markdown(
                    f"<div style='padding:14px; border-radius:8px; background:#13161c; "
                    f"border:2px solid {_eq_color}; min-height:140px;'>"
                    f"<div style='display:flex; justify-content:space-between; align-items:baseline;'>"
                    f"<span style='font-size:10px; color:#888; text-transform:uppercase;'>"
                    f"🔻 Equity TOP <span style='color:#ef4444;'>(sell stocks?)</span></span>"
                    f"<span>{_age_badge('top_scorecard')}</span></div>"
                    f"<div style='font-size:28px; font-weight:800; color:{_eq_color}; "
                    f"line-height:1; margin-top:4px;'>{_eq_n_hard}<span style='font-size:14px; "
                    f"color:#888;'>/{_eq_n_hard_total}</span></div>"
                    f"<div style='font-size:11px; color:#ccc; font-weight:600; margin-top:4px;'>"
                    f"{_eq_display}</div>"
                    f"<div style='font-size:11px; color:#aaa; margin-top:6px; line-height:1.4;'>"
                    f"{_dormant_status(_eq_n_hard, _eq_n_hard_total, 'top')}</div>"
                    f"</div>", unsafe_allow_html=True,
                )
            except Exception as _ee:
                st.caption(f"equity scorecard — temporarily unavailable")

        # Card 2: Equity drawdown-risk context (personal allocation hidden when sanitised)
        with _eq_c2:
            try:
                if SHOW_PERSONAL:
                    _c2_title = "Your Equity Exposure"
                    _c2_big = "70%"
                    _c2_sub = f"{_money(91_000)} of {_money(130_000)} stake in stocks"
                    _c2_foot = "BTC: 30% · Cash: 0%"
                else:
                    # Market-only: QQQ's distance to its 200wMA = max equity drawdown risk
                    _q_wma = 0.0
                    try:
                        _q_wma = (_gc_eq("equity_olson") or {}).get("pct_to_wma200", 0) or 0
                    except Exception:
                        pass
                    _c2_title = "Equity Drawdown Risk"
                    _c2_big = f"{_q_wma:+.0f}%"
                    _c2_sub = "QQQ distance to its 200-week SMA"
                    _c2_foot = "= worst-case equity drawdown if it fully retests"
                st.markdown(
                    f"<div style='padding:14px; border-radius:8px; background:#13161c; "
                    f"border:2px solid #4a90e2; min-height:140px;'>"
                    f"<div style='font-size:10px; color:#888; text-transform:uppercase;'>"
                    f"{_c2_title}</div>"
                    f"<div style='font-size:28px; font-weight:800; color:#4a90e2; "
                    f"line-height:1; margin-top:4px;'>{_c2_big}</div>"
                    f"<div style='font-size:11px; color:#aaa; margin-top:6px;'>{_c2_sub}</div>"
                    f"<div style='font-size:10px; color:#888; margin-top:4px;'>{_c2_foot}</div>"
                    f"</div>", unsafe_allow_html=True,
                )
            except Exception: pass

        # Card 3: Recommendation summary (uses actual keys from phased_exit_recommendation)
        with _eq_c3:
            try:
                _rec_pct = 0
                _rec_action = "HOLD position"
                _sell_nzd = 0
                if _recommendation and isinstance(_recommendation, dict):
                    # Real keys: equity_to_sell_pct_of_stake, rationale, sell_nzd
                    _rec_pct = _recommendation.get("equity_to_sell_pct_of_stake", 0) or 0
                    _rec_action = (_recommendation.get("rationale") or "HOLD position")[:80]
                    _sell_nzd = _recommendation.get("sell_nzd", 0) or 0
                _rec_color = "#ef4444" if _rec_pct > 0 else "#22c55e"
                _sell_suffix = (f" ({_money(_sell_nzd)})"
                                if (SHOW_PERSONAL and _sell_nzd > 0) else "")
                st.markdown(
                    f"<div style='padding:14px; border-radius:8px; background:#13161c; "
                    f"border:2px solid {_rec_color}; min-height:140px;'>"
                    f"<div style='font-size:10px; color:#888; text-transform:uppercase;'>"
                    f"Phased Exit Plan</div>"
                    f"<div style='font-size:28px; font-weight:800; color:{_rec_color}; "
                    f"line-height:1; margin-top:4px;'>{_rec_pct:.0f}%</div>"
                    f"<div style='font-size:11px; color:#aaa; margin-top:6px;'>"
                    f"of equities to sell now{_sell_suffix}</div>"
                    f"<div style='font-size:11px; color:#ccc; margin-top:4px; line-height:1.3;'>"
                    f"{_rec_action}</div>"
                    f"</div>", unsafe_allow_html=True,
                )
            except Exception: pass

        # Detail expander — full criteria list of equity scorecard
        with st.expander("📋 See all equity-TOP criteria + status of each", expanded=False):
            try:
                _eq_crit = (_scorecard or {}).get("criteria") or \
                           (_scorecard or {}).get("hard_criteria") or []
                if _eq_crit:
                    import pandas as _pd
                    _eq_rows = [{"✓": "🔥" if c.get("met") else "○",
                                  "Macro Criterion": c.get("label") or c.get("name", "?"),
                                  "Status": (c.get("status") or c.get("note", "") or "")[:90]}
                                 for c in _eq_crit]
                    st.dataframe(_pd.DataFrame(_eq_rows), width='stretch', hide_index=True)
                else:
                    st.caption("equity criteria detail not in cache yet")
            except Exception as _ee:
                st.caption(f"equity detail: {_ee}")

        st.markdown("<hr style='border-color:#2a2d36; margin:18px 0;'>",
                    unsafe_allow_html=True)
    except Exception as _e:
        st.caption(f"Equity Top Watch — temporarily unavailable")


with tab_research:   # <- 2026-07-04 restructure
    # ═══════════════════════════════════════════════════════════════════
    # 🆕 SWIFT DIALS — high-impact glance-able indicators (BTC cycle context)
    # Built per Swift's gap analysis: Halving Clock, BTC Dominance, S2F,
    # Open Interest, Cycle 4 vs 5 overlay. All Plotly figures pre-computed.
    # ═══════════════════════════════════════════════════════════════════
    try:
        from core.dashboard_cache import get_cached
        _dials = get_cached("swift_dials")
        if not _dials:
            from core.btc_swift_dials import all_swift_dials
            _dials = all_swift_dials()

        st.markdown(
            f"<div class='section-header' style='display:flex; justify-content:space-between; "
            f"align-items:baseline;'>"
            f"<span>📊 Swift Dials — cycle position + dominance + derivatives</span>"
            f"<span>{_age_badge('swift_dials')}</span></div>",
            unsafe_allow_html=True,
        )

        # Row 1: 4 gauges (halving clock, dominance, S2F, OI)
        _g1, _g2, _g3, _g4 = st.columns(4)
        with _g1:
            if _dials.get("halving_clock"):
                st.plotly_chart(_dials["halving_clock"], width='stretch',
                                config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False, "displaylogo": False})
        with _g2:
            if _dials.get("btc_dominance"):
                st.plotly_chart(_dials["btc_dominance"], width='stretch',
                                config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False, "displaylogo": False})
        with _g3:
            if _dials.get("s2f_deflection"):
                st.plotly_chart(_dials["s2f_deflection"], width='stretch',
                                config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False, "displaylogo": False})
        with _g4:
            if _dials.get("open_interest"):
                st.plotly_chart(_dials["open_interest"], width='stretch',
                                config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False, "displaylogo": False})

        # Row 2: Cycle 4 vs 5 overlay (full width)
        if _dials.get("cycle_overlay"):
            st.plotly_chart(_dials["cycle_overlay"], width='stretch',
                            config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False, "displaylogo": False})

        # Quick caption explaining the dials
        st.caption(
            "**Halving Clock**: where we are in the 4-year cycle. **BTC Dominance**: "
            "BTC vs alts split — < 45% favors alts, 55-65% favors BTC. **S2F Deflection**: "
            "price vs Plan B's model (degraded post-cycle 4, use as context only). "
            "**Open Interest z-score**: derivatives positioning extreme — froth at top, "
            "capitulation at bottom. **Cycle overlay**: where cycle 5 stands vs cycle 4 path."
        )
    except Exception as _e:
        st.caption(f"Swift Dials — temporarily unavailable")


    # ═══════════════════════════════════════════════════════════════════
    # 🆕 LIQUIDATION HEATMAP — Coinglass embed (free)
    # Shows stop-loss clusters above & below price — where liquidations
    # will trigger. Predicts directional sweeps.
    # ═══════════════════════════════════════════════════════════════════
    with st.expander("🔥 Liquidation heatmap (Coinglass) — where stops cluster", expanded=False):
        # Coinglass blocks iframe embedding (X-Frame-Options), so an embed renders
        # blank — link out instead (same pattern as Mempool / Bitcoin Mag Pro).
        st.markdown(
            "<div style='padding:12px 16px; background:#13161c; border-radius:8px; "
            "border-left:3px solid #f0b90b;'>"
            "<div style='font-size:13px; color:#ccc;'>Coinglass blocks embedding, so this "
            "opens on their site:</div>"
            "<a href='https://www.coinglass.com/LiquidationData' target='_blank' "
            "style='display:inline-block; margin-top:8px; font-size:14px; font-weight:700; "
            "color:#4a90e2; text-decoration:none;'>🔥 Open the live liquidation heatmap on Coinglass →</a>"
            "<div style='font-size:11px; color:#888; margin-top:8px;'>"
            "Bright zones = stop clusters; price often sweeps these before reversing.</div></div>",
            unsafe_allow_html=True,
        )


with tab_signals:   # <- 2026-07-04 review fix: Cockpit leads the section
    # ═══════════════════════════════════════════════════════════════════
    # 🎯 COCKPIT — single-glance answer view
    # Five gauges + theme strip + big verdict card. Everything below is
    # supporting detail (collapsed by default).
    # ═══════════════════════════════════════════════════════════════════
    try:
        _pe = cached_predictor_engine()
        _ud = cached_unified_decision()
        _dec = _pe.get("decision_composites", {})
        _themes = _pe.get("theme_composites", {})
        _btc = _pe.get("btc_state", {})
        _regime = _ud.get("regime", "UNKNOWN")
        _liq_z = _ud.get("liquidity", {}).get("z", 0.0)
        _vetoes = _ud.get("vetoes_active", [])
        _curr_alloc = _ud.get("current_allocation_pct", {})
        _target_alloc = _ud.get("target_allocation_pct", {})
        _delta = _ud.get("delta_pct", {})

        # ─── Helper: build a gauge ───
        def _build_gauge(value, title, vmin, vmax, zones, fmt=".2f", suffix=""):
            """zones: list of (start, end, color)"""
            import plotly.graph_objects as go
            steps = [{"range": [s, e], "color": c} for s, e, c in zones]
            v = value if value is not None else 0
            v = max(vmin, min(vmax, v))
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=v,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": f"<b>{title}</b>", "font": {"size": 13, "color": "#ccc"}},
                number={"font": {"size": 26, "color": "white"},
                        "valueformat": fmt, "suffix": suffix},
                gauge={
                    "axis": {"range": [vmin, vmax],
                              "tickfont": {"size": 9, "color": "#888"},
                              "tickwidth": 1, "tickcolor": "#444"},
                    "bar": {"color": "rgba(255,255,255,0.85)", "thickness": 0.25},
                    "bgcolor": "#0e1117",
                    "borderwidth": 1,
                    "bordercolor": "#2a2d36",
                    "steps": steps,
                    "threshold": {
                        "line": {"color": "white", "width": 3},
                        "thickness": 0.85,
                        "value": v,
                    },
                },
            ))
            fig.update_layout(
                margin=dict(l=20, r=20, t=50, b=15),
                height=200,
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "white", "family": "Inter, sans-serif"},
            )
            return fig

        # ─── Big headline verdict card ───
        # === DUAL VERDICT: equity rotation + BTC deploy (two separate decisions) ===
        _top_sc = _ud.get("scorecards", {}).get("top", {})
        _early_sc = _ud.get("scorecards", {}).get("early", {})
        _bottom_sc = _ud.get("scorecards", {}).get("bottom", {})

        _top_z = _dec.get("top", 0)
        _bottom_z = _dec.get("bottom", 0)
        _btc_prob = _btc.get("bottom_probability", 0.5)
        _btc_state_label = _btc.get("state", "UNCONFIRMED")

        # --- EQUITY VERDICT ---
        _eq_label = "HOLD"; _eq_color = C.get("bull", "#22c55e")
        _eq_detail = "No trigger crossed — stay allocated"
        _eq_next = "TRIM if top scorecard ≥ 3/10 OR early rotation ≥ 2/7"
        if _top_sc.get("action") in ("BEAR_CONFIRMED", "FULL_ROTATION"):
            _eq_label = _top_sc["action"].replace("_", " "); _eq_color = C.get("deep_bear", "#b91c1c")
            _eq_detail = f"Top {_top_sc['n_met']}/{_top_sc['n_total']} — major equity exit"
            _eq_next = "Execute trim now"
        elif _top_sc.get("action") == "DEFENSIVE":
            _eq_label = "DEFENSIVE"; _eq_color = C.get("bear", "#ef4444")
            _eq_detail = f"Top {_top_sc['n_met']}/{_top_sc['n_total']} — reduce to 50%"
            _eq_next = "BEAR if top ≥ 7/10"
        elif _top_sc.get("action") == "TRIM":
            _eq_label = "TRIM"; _eq_color = C.get("neutral", "#f0b90b")
            _eq_detail = f"Top {_top_sc['n_met']}/{_top_sc['n_total']} — reduce equity 25%"
            _eq_next = "DEFENSIVE if top ≥ 5/10"
        elif _early_sc.get("action") == "ROTATE_TO_CASH":
            _eq_label = "ROTATE TO CASH"; _eq_color = C.get("bear", "#ef4444")
            _eq_detail = f"Early rotation {_early_sc['n_firing']}/{_early_sc['n_total']} firing"
            _eq_next = "Confirm with top scorecard trigger"
        elif _top_z > 1.5:
            _eq_label = "WATCH"; _eq_color = C.get("neutral", "#f0b90b")
            _eq_detail = f"Calibrated z={_top_z:+.2f} extreme — no scorecard trigger yet"
            _eq_next = "Tighten stops; act when scorecard fires"
        elif _early_sc.get("action") == "WATCH":
            _eq_label = "WATCH"; _eq_color = C.get("neutral", "#f0b90b")
            _eq_detail = f"Early rotation {_early_sc['n_firing']}/{_early_sc['n_total']} — leading"
            _eq_next = "REDUCE if early ≥ 3/7"

        # --- BTC VERDICT ---
        # Display the SAME 16-signal native scorecard as the Scorecards tab (single
        # source of truth) so the bottom count is consistent everywhere on the
        # dashboard. The verdict LABEL below is still driven by the composite
        # z-score (_bottom_z) — unchanged; only the displayed count is unified.
        from core.dashboard_cache import get_cached as _gcv
        _nbs = _gcv("btc_native_bottom_scorecard") or {}
        _bn = _nbs.get("n_met", _bottom_sc.get("n_met", 0))
        _bnt = _nbs.get("n_total", 16)
        _btc_label = "WAIT"; _btc_color = C.get("muted", "#888")
        _btc_detail = f"Bottom {_bn}/{_bnt} — not at bottom"
        _btc_next = "WATCH as the 16-signal bottom checklist builds OR ETF flows turn positive"
        if _btc_state_label == "DEEP_GENERATIONAL":
            _btc_label = "DEPLOY 50%"; _btc_color = C.get("deep_bull", "#16a34a")
            _btc_detail = "Deep capitulation confirmed — historic precedent"
            _btc_next = "Half upfront, DCA rest over 30d"
        elif _btc_state_label == "SHALLOW_ETF_DRIVEN":
            _btc_label = "DCA"; _btc_color = C.get("bull", "#22c55e")
            _btc_detail = "ETF-driven bottom forming — 20% initial + 80% DCA"
            _btc_next = "DEPLOY_50 if MVRV-Z drops < -1.5"
        elif _bottom_z > 1.0:
            _btc_label = "ACCUMULATE"; _btc_color = C.get("bull", "#22c55e")
            _btc_detail = f"Bottom composite z={_bottom_z:+.2f} — forming"
            _btc_next = "DCA as the bottom checklist confirms (see Scorecards tab)"
        elif _bottom_z > 0:
            _btc_label = "WATCH"; _btc_color = C.get("neutral", "#f0b90b")
            _btc_detail = f"Composite z={_bottom_z:+.2f} — early signal"
            _btc_next = "ACCUMULATE if composite z > 1.0"

        _regime_emoji = {"RISK_ON":"🟢","LATE_CYCLE":"🟡","RECESSIONARY_BEAR":"🔴"}.get(_regime,"⚪")
        # LIVE BTC price — 60s cache via Binance ticker (not the 4h state cache)
        _btc_price_live = get_live_btc_price()
        if _btc_price_live > 0:
            _btc_price = _btc_price_live
            _btc_price_source = "live"
        else:
            # Fallback to cached state if live unavailable
            _state = get_state()
            _btc_price = _state.get("btc_price") or _state.get("price_usd") or 0
            _btc_price_source = "cached"
        # BTC cycle position
        try:
            _hpos = current_halving_position()
            _cycle_day = _hpos.get("days_post_halving", 0)
            _cycle_pct = _hpos.get("pct_through_cycle", 0)
        except Exception:
            _cycle_day = 0; _cycle_pct = 0

        # Allocation block — personal (current/Δ + NZ$ stake) only when private;
        # otherwise show just the model's TARGET allocation (a signal, not a position).
        if SHOW_PERSONAL:
            _alloc_html = (
                "<div style='padding:14px 18px; border-radius:8px; margin-bottom:14px; "
                "background:#13161c; border:1px solid #2a2d36;'>"
                "<div style='font-size:10px; color:#888; letter-spacing:1.5px; "
                "text-transform:uppercase; margin-bottom:8px;'>Your Allocation (NZ$130k stake)</div>"
                "<div style='display:flex; gap:24px; flex-wrap:wrap; font-size:13px;'>"
                f"<div><span style='color:#888;'>Current:</span> "
                f"<b style='color:{C.get('bear')};'>{_curr_alloc.get('equity',0):.0f}%</b> eq / "
                f"<b style='color:{C.get('accent')};'>{_curr_alloc.get('btc',0):.0f}%</b> BTC / "
                f"<b style='color:{C.get('bull')};'>{_curr_alloc.get('staging',0):.0f}%</b> cash</div>"
                f"<div><span style='color:#888;'>Target:</span> "
                f"<b style='color:{C.get('bear')};'>{_target_alloc.get('equity',0):.0f}%</b> / "
                f"<b style='color:{C.get('accent')};'>{_target_alloc.get('btc',0):.0f}%</b> / "
                f"<b style='color:{C.get('bull')};'>{_target_alloc.get('staging',0):.0f}%</b></div>"
                "</div></div>"
            )
        else:
            _alloc_html = (
                "<div style='padding:14px 18px; border-radius:8px; margin-bottom:14px; "
                "background:#13161c; border:1px solid #2a2d36;'>"
                "<div style='font-size:10px; color:#888; letter-spacing:1.5px; "
                "text-transform:uppercase; margin-bottom:8px;'>Model Target Allocation (signal)</div>"
                "<div style='font-size:13px;'>"
                f"<b style='color:{C.get('bear')};'>{_target_alloc.get('equity',0):.0f}%</b> equity / "
                f"<b style='color:{C.get('accent')};'>{_target_alloc.get('btc',0):.0f}%</b> BTC / "
                f"<b style='color:{C.get('bull')};'>{_target_alloc.get('staging',0):.0f}%</b> cash</div>"
                "</div>"
            )

        # === Dual verdict cards ===
        st.markdown(
            f"""
<div style='display:flex; gap:14px; margin-bottom:14px; flex-wrap:wrap;'>
  <div style='flex:1; min-width:300px; padding:18px 22px; border-radius:10px;
              background:linear-gradient(135deg, {_eq_color}26 0%, #13161c 70%);
              border-left:6px solid {_eq_color};'>
    <div style='font-size:10px; color:#888; letter-spacing:2px; text-transform:uppercase;'>Equity Rotation</div>
    <div style='font-size:30px; font-weight:800; color:{_eq_color}; line-height:1.05; margin:4px 0 6px 0;'>{_eq_label}</div>
    <div style='font-size:12px; color:#bbb; min-height:18px;'>{_eq_detail}</div>
    <div style='font-size:11px; color:#888; margin-top:8px; font-style:italic;'>↑ Next: {_eq_next}</div>
  </div>
  <div style='flex:1; min-width:300px; padding:18px 22px; border-radius:10px;
              background:linear-gradient(135deg, {_btc_color}26 0%, #13161c 70%);
              border-left:6px solid {_btc_color};'>
    <div style='font-size:10px; color:#888; letter-spacing:2px; text-transform:uppercase;'>BTC Deploy</div>
    <div style='font-size:30px; font-weight:800; color:{_btc_color}; line-height:1.05; margin:4px 0 6px 0;'>{_btc_label}</div>
    <div style='font-size:12px; color:#bbb; min-height:18px;'>{_btc_detail}</div>
    <div style='font-size:11px; color:#888; margin-top:8px; font-style:italic;'>↑ Next: {_btc_next}</div>
  </div>
</div>
<div style='padding:12px 18px; border-radius:8px; margin-bottom:14px; background:#13161c;
            border:1px solid #2a2d36; display:flex; gap:24px; flex-wrap:wrap; font-size:12px;'>
  <div><span style='color:#888;'>Macro regime:</span> <b style='color:#ccc;'>{_regime_emoji} {_regime}</b></div>
  <div><span style='color:#888;'>BTC phase:</span> <b style='color:#ccc;'>{_btc_state_label.replace('_',' ')}</b></div>
  <div><span style='color:#888;'>BTC spot:</span> <b style='color:#ccc;'>${_btc_price:,.0f}</b> <span style='color:#888; font-size:10px;'>({_btc_price_source})</span></div>
  <div><span style='color:#888;'>Cycle 5:</span> <b style='color:#ccc;'>day {_cycle_day} ({_cycle_pct:.0f}%)</b></div>
  <div><span style='color:#888;'>Liquidity z:</span> <b style='color:#ccc;'>{_liq_z:+.2f}</b></div>
  <div><span style='color:#888;'>Vetoes:</span> <b style='color:{C.get("bear") if _vetoes else "#ccc"};'>{len(_vetoes)} active</b></div>
</div>
{_alloc_html}""",
            unsafe_allow_html=True,
        )

        # ─── 5 Gauges row ───
        # Use Plotly indicator gauges for at-glance reading
        try:
            import plotly.graph_objects as go
            _gcols = st.columns(5)

            # Plotly only accepts 6-char hex OR rgba() — NOT 8-char hex with alpha.
            # Pre-defined translucent zone colors:
            GREEN_LITE = "rgba(34, 197, 94, 0.25)"
            GREEN_MID  = "rgba(34, 197, 94, 0.50)"
            YELLOW_LITE = "rgba(240, 185, 11, 0.30)"
            RED_LITE   = "rgba(239, 68, 68, 0.30)"
            RED_MID    = "rgba(239, 68, 68, 0.55)"

            # Gauge 1: Top composite (equity exit pressure)
            with _gcols[0]:
                fig = _build_gauge(
                    value=_top_z,
                    title="Top Score (exit equity)",
                    vmin=-3, vmax=3,
                    zones=[(-3, 0, GREEN_LITE), (0, 1.5, YELLOW_LITE), (1.5, 3, RED_MID)],
                    fmt=".2f",
                )
                st.plotly_chart(fig, width='stretch', config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False, "displaylogo": False})

            # Gauge 2: Bottom composite (BTC deploy signal)
            with _gcols[1]:
                fig = _build_gauge(
                    value=_bottom_z,
                    title="BTC Bottom (deploy)",
                    vmin=-3, vmax=3,
                    zones=[(-3, 0, RED_LITE), (0, 1.5, YELLOW_LITE), (1.5, 3, GREEN_MID)],
                    fmt=".2f",
                )
                st.plotly_chart(fig, width='stretch', config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False, "displaylogo": False})

            # Gauge 3: BTC bottom probability (sigmoid)
            with _gcols[2]:
                fig = _build_gauge(
                    value=_btc_prob * 100,
                    title="Bottom Probability",
                    vmin=0, vmax=100,
                    zones=[(0, 30, RED_LITE), (30, 70, YELLOW_LITE), (70, 100, GREEN_MID)],
                    fmt=".0f", suffix="%",
                )
                st.plotly_chart(fig, width='stretch', config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False, "displaylogo": False})

            # Gauge 4: Liquidity z-score
            with _gcols[3]:
                fig = _build_gauge(
                    value=_liq_z,
                    title="Liquidity Z",
                    vmin=-2, vmax=2,
                    zones=[(-2, -0.5, RED_LITE), (-0.5, 0.5, YELLOW_LITE), (0.5, 2, GREEN_MID)],
                    fmt=".2f",
                )
                st.plotly_chart(fig, width='stretch', config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False, "displaylogo": False})

            # Gauge 5: Cycle position (days since halving / cycle length).
            # Use pct_through_cycle from halving_clock (real next-halving date diff),
            # NOT a hard-coded 1460 — keeps in sync with the hero card.
            with _gcols[4]:
                try:
                    pos = current_halving_position()
                    cycle_pct = pos.get("pct_through_cycle", 0) or 0
                    cycle_pct = max(0, min(100, cycle_pct))
                except Exception:
                    cycle_pct = 50
                fig = _build_gauge(
                    value=cycle_pct,
                    title="Cycle 5 Position",
                    vmin=0, vmax=100,
                    zones=[(0, 25, GREEN_LITE), (25, 75, YELLOW_LITE), (75, 100, RED_MID)],
                    fmt=".0f", suffix="%",
                )
                st.plotly_chart(fig, width='stretch', config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False, "displaylogo": False})

        except ImportError:
            st.warning("Install plotly for gauges: `pip install plotly`")

        # ─── BTC-NATIVE TOP SCORECARD (the gap from cycle 5 backtest) ───
        try:
            from core.dashboard_cache import get_cached as _gc
            _btc_top = _gc("btc_native_top_scorecard")
            if _btc_top:
                _btc_n_met = _btc_top.get("n_met", 0)
                _btc_n_total = _btc_top.get("n_total", 16)  # canonical: native top has 16 criteria
                _btc_level = _btc_top.get("verdict_level", "HOLD")
                _btc_top_color = {
                    "EXIT_75":      C.get("deep_bear", "#b91c1c"),
                    "SCALE_OUT_50": C.get("bear", "#ef4444"),
                    "TRIM_25":      C.get("neutral", "#f0b90b"),
                    "WATCH":        C.get("accent", "#4a90e2"),
                    "HOLD":         C.get("bull", "#22c55e"),
                }.get(_btc_level, C.get("muted", "#888"))
                st.markdown(
                    f"<div style='padding:14px 18px; border-radius:8px; margin-bottom:14px; "
                    f"background:#13161c; border-left:4px solid {_btc_top_color};'>"
                    f"<div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;'>"
                    f"<div><span style='font-size:10px; color:#888; text-transform:uppercase; letter-spacing:1.5px;'>"
                    f"🔻 BTC TOP — sell BTC? (cycle signals)</span><br>"
                    f"<span style='font-size:22px; font-weight:700; color:{_btc_top_color};'>"
                    f"{_btc_level.replace('_',' ')}</span> "
                    f"<span style='font-size:14px; color:#888;'>({_btc_n_met}/{_btc_n_total} signals)</span></div>"
                    f"<div style='font-size:12px; color:#aaa; text-align:right; max-width:60%;'>"
                    f"{_btc_top.get('verdict', '')[:140]}</div>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
                with st.expander(f"📋 All 10 BTC-native top criteria — {_btc_n_met} firing"):
                    rows = []
                    for c in _btc_top.get("criteria", []):
                        rows.append({
                            "✓": "🔥" if c.get("met") else "○",
                            "Criterion": c.get("label", "?"),
                            "Status": c.get("status", "?")[:80],
                        })
                    if rows:
                        st.dataframe(pd.DataFrame(rows),
                                      width='stretch', hide_index=True)
                    st.caption(
                        "These 10 are BTC-cycle native (Pi Cycle, MVRV-Z, Puell, NUPL, "
                        "STH-MVRV, aSOPR, Hash Ribbon, RSI div, MACD bear, Realized Cap @ ATH). "
                        "Built to catch BTC tops that the equity-side Top Scorecard misses "
                        "(like cycle 5 Oct'25 where SPY kept rising 60d after BTC peaked)."
                    )
        except Exception as _e:
            pass

        # ─── SWIFT CHART SUITE — full historical context (Rainbow, Pi Cycle history, multiplier bands) ───
        with st.expander("📊 Phillip Swift chart suite — full historical context (Rainbow, Pi Cycle, GR Multiplier, MVRV)"):
            try:
                from core.dashboard_cache import get_cached as _gc
                _sc = _gc("swift_charts")
                if _sc:
                    import plotly.graph_objects as go
                    _CFG2 = {"displayModeBar": False, "scrollZoom": False, "doubleClick": False, "displaylogo": False}
                    def _d(k):
                        f = _sc.get(k)
                        if f:
                            # unique key per chart: this suite shares figures with the
                            # "Cycle charts (live)" block, so without distinct keys Streamlit
                            # raises StreamlitDuplicateElementId and BOTH copies go blank.
                            st.plotly_chart(_swift_fig(f), width='stretch', config=_CFG2, key=f"swsuite_{k}")
                    # Rainbow chart full width (most iconic)
                    _d("rainbow")
                    # 2 columns for the multiplier history charts
                    _r1 = st.columns(2)
                    with _r1[0]: _d("pi_cycle_top")
                    with _r1[1]: _d("pi_cycle_bottom")
                    _r2 = st.columns(2)
                    with _r2[0]: _d("golden_ratio")
                    with _r2[1]: _d("two_year_ma")
                    _r3 = st.columns(2)
                    with _r3[0]: _d("mvrv_bands")
                    with _r3[1]: _d("puell_bands")
                    _d("hodl_waves")
                    st.caption(
                        "8 LookIntoBitcoin-style charts with full historical context. Rainbow chart = "
                        "log regression with 8 color bands. Pi Cycle Top/Bottom show ratio history vs cross "
                        "threshold. Golden Ratio Multiplier = price/350d MA with Fibonacci bands. "
                        "2y MA Multiplier = price/2y MA with 5× top band. MVRV with capitulation/euphoria "
                        "zones. Puell Multiple with bottom/top zones. HODL proxy via realized cap velocity."
                    )
            except Exception as _e:
                st.error(f"Swift charts — temporarily unavailable")

        # ─── PHILLIP SWIFT WATCH — Risk Index + extended indicators + content monitoring ───
        with st.expander("👤 Phillip Swift Watch — Bitcoin Risk Index + extended Swift indicators + live monitoring"):
            try:
                from core.dashboard_cache import get_cached as _gc
                _sw = _gc("swift_watch")
                if _sw:
                    # === Bitcoin Risk Index — his signature meta-indicator ===
                    ri = _sw.get("risk_index", {})
                    if not ri.get("error"):
                        idx = ri.get("risk_index", 0.5)
                        zone = ri.get("zone", "?")
                        emoji = ri.get("emoji", "")
                        action = ri.get("action", "?")
                        # Color: green at 0, red at 1
                        bar_color = (C.get("deep_bull") if idx < 0.2 else
                                       C.get("bull") if idx < 0.4 else
                                       C.get("neutral") if idx < 0.6 else
                                       C.get("bear") if idx < 0.8 else C.get("deep_bear"))
                        st.markdown(
                            f"<div style='padding:18px 22px; border-radius:10px; "
                            f"background:linear-gradient(90deg, {bar_color}22 0%, #13161c 100%); "
                            f"border-left:6px solid {bar_color}; margin-bottom:14px;'>"
                            f"<div style='font-size:11px; color:#888; text-transform:uppercase; letter-spacing:2px;'>"
                            f"Bitcoin Risk Index (Phillip Swift's composite)</div>"
                            f"<div style='display:flex; align-items:center; gap:22px; flex-wrap:wrap;'>"
                            f"<div style='font-size:48px; font-weight:800; color:{bar_color}; line-height:1;'>"
                            f"{idx:.2f}</div>"
                            f"<div><div style='font-size:22px; font-weight:700; color:{bar_color};'>{emoji} {zone}</div>"
                            f"<div style='font-size:13px; color:#aaa;'>{action}</div>"
                            f"<div style='font-size:11px; color:#f0b90b; margin-top:3px;'>"
                            f"Long-term <b>value</b> gauge — not a 'buy now' timing call. "
                            f"Timing is governed by the bottom scorecard &amp; rotation plan.</div>"
                            f"</div></div>"
                            f"<div style='height:14px; background:#222; border-radius:7px; margin-top:14px; position:relative;'>"
                            f"<div style='width:{idx*100:.0f}%; height:100%; "
                            f"background:linear-gradient(90deg, {C.get('deep_bull')} 0%, "
                            f"{C.get('bull')} 25%, {C.get('neutral')} 50%, "
                            f"{C.get('bear')} 75%, {C.get('deep_bear')} 100%); border-radius:7px;'></div>"
                            f"<div style='position:absolute; top:-4px; left:{idx*100:.0f}%; "
                            f"width:6px; height:22px; background:white; transform:translateX(-3px);'></div>"
                            f"</div>"
                            f"<div style='display:flex; justify-content:space-between; font-size:10px; color:#888; margin-top:4px;'>"
                            f"<span>0.0 MAX BUY</span><span>0.5 NEUTRAL</span><span>1.0 MAX SELL</span></div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        # Component breakdown
                        comps = ri.get("components", {})
                        st.caption(
                            f"Components: MVRV {comps.get('mvrv_score',0):.2f} (40%) · "
                            f"Mayer {comps.get('mayer_score',0):.2f} (15%) · "
                            f"NUPL {comps.get('nupl_score',0):.2f} (15%) · "
                            f"Pi {comps.get('pi_score',0):.2f} (10%) · "
                            f"GR {comps.get('gr_score',0):.2f} (10%) · "
                            f"RR {comps.get('rr_score',0):.2f} (10%)"
                        )

                    # === Extended indicators ===
                    st.markdown(
                        "<div style='font-size:11px; color:#888; text-transform:uppercase; "
                        "letter-spacing:1.5px; margin:12px 0 6px 0;'>Extended Swift Indicators</div>",
                        unsafe_allow_html=True,
                    )
                    _ec = st.columns(3)
                    # Thermocap
                    tm = _sw.get("thermocap", {})
                    with _ec[0]:
                        if not tm.get("error"):
                            st.metric("Thermocap Multiple",
                                       f"{tm.get('multiplier', 0):.1f}×",
                                       f"{tm.get('emoji', '')} {tm.get('zone', '?')}")
                        else: st.metric("Thermocap Multiple", "—", "data unavailable")
                    # Profitable Days
                    pd_info = _sw.get("profitable_days", {})
                    with _ec[1]:
                        if not pd_info.get("error"):
                            st.metric("Profitable Days",
                                       f"{pd_info.get('profitable_pct', 0):.1f}%",
                                       f"{pd_info.get('emoji', '')} {pd_info.get('zone', '?')}")
                        else: st.metric("Profitable Days", "—", "data unavailable")
                    # 200wMA
                    wma = _sw.get("two_hundred_wma", {})
                    with _ec[2]:
                        if not wma.get("error"):
                            st.metric("vs 200-week MA",
                                       f"{wma.get('pct_vs_ma', 0):+.1f}%",
                                       f"{wma.get('emoji', '')} {wma.get('zone', '?')}")
                        else: st.metric("vs 200-week MA", "—", "data unavailable")

                    # === Content monitoring ===
                    content = _sw.get("content", {})
                    if content:
                        st.markdown("---")
                        st.markdown(
                            "<div style='font-size:11px; color:#888; text-transform:uppercase; "
                            "letter-spacing:1.5px; margin:6px 0;'>Live Content to Monitor</div>",
                            unsafe_allow_html=True,
                        )
                        _cc = st.columns(2)
                        with _cc[0]:
                            tw = content.get("twitter", {})
                            st.markdown(
                                f"**🐦 Twitter:** [{tw.get('handle', '?')}]({tw.get('url', '#')})<br>"
                                f"<small style='color:#888;'>{tw.get('description', '')}</small>",
                                unsafe_allow_html=True,
                            )
                            lib = content.get("lookintobitcoin", {})
                            st.markdown(
                                f"**📊 Master dashboard:** [{lib.get('name', '?')}]({lib.get('url', '#')})<br>"
                                f"<small style='color:#888;'>{lib.get('description', '')}</small>",
                                unsafe_allow_html=True,
                            )
                            bmp = content.get("bitcoin_magazine_pro", {})
                            st.markdown(
                                f"**📝 Bitcoin Magazine Pro:** [{bmp.get('name', '?')}]({bmp.get('url', '#')})<br>"
                                f"<small style='color:#888;'>{bmp.get('description', '')}</small>",
                                unsafe_allow_html=True,
                            )
                            nl = content.get("newsletter", {})
                            st.markdown(
                                f"**📧 Newsletter:** [{nl.get('name', '?')}]({nl.get('url', '#')})<br>"
                                f"<small style='color:#888;'>{nl.get('description', '')}</small>",
                                unsafe_allow_html=True,
                            )
                        with _cc[1]:
                            st.markdown("**📺 YouTube channels with frequent Swift appearances:**")
                            for name, url, descr in content.get("youtube_channels", []):
                                st.markdown(
                                    f"- [{name}]({url})  <small style='color:#888;'>— {descr}</small>",
                                    unsafe_allow_html=True,
                                )

                    # === Embedded Twitter timeline ===
                    st.markdown("---")
                    st.markdown(
                        "<div style='font-size:11px; color:#888; text-transform:uppercase; "
                        "letter-spacing:1.5px; margin:6px 0;'>@PositiveCrypto Latest Tweets (embedded)</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        """<a class="twitter-timeline" data-theme="dark"
                          data-height="500"
                          href="https://twitter.com/PositiveCrypto?ref_src=twsrc%5Etfw">
                          Tweets by @PositiveCrypto</a>
                        <script async src="https://platform.twitter.com/widgets.js"
                          charset="utf-8"></script>""",
                        unsafe_allow_html=True,
                    )

            except Exception as _e:
                st.warning(f"Swift Watch — temporarily unavailable")

        # ─── FREE-TIER PROXIES for paid metrics + EXTERNAL CHART EMBEDS ───
        with st.expander("🔧 Free-tier proxies for paid Glassnode metrics (HODL Waves, Reserve Risk, CVDD, NRP&L)"):
            try:
                from core.dashboard_cache import get_cached as _gc
                _pr = _gc("free_proxies")
                if _pr:
                    proxy_rows = []
                    for key, info in _pr.items():
                        if key == "asof" or not isinstance(info, dict): continue
                        if info.get("error"):
                            proxy_rows.append({
                                "Metric": key.replace("_", " "),
                                "Status": "data unavailable",
                                "Confidence": info.get("confidence", "—"),
                                "Reading": "—",
                            })
                            continue
                        proxy_rows.append({
                            "Metric": key.replace("_", " "),
                            "Status": "OK",
                            "Confidence": info.get("confidence", "—"),
                            "Reading": info.get("interpretation", "?")[:75],
                        })
                    if proxy_rows:
                        st.dataframe(pd.DataFrame(proxy_rows),
                                      width='stretch', hide_index=True)
                    st.caption(
                        "HIGH confidence = same math as paid version. "
                        "MEDIUM = good directional proxy. "
                        "LOW = rough approximation. "
                        "Data falls back gracefully when CoinMetrics free tier is unavailable."
                    )
            except Exception as _e:
                st.warning(f"Proxies — temporarily unavailable")

        with st.expander("🌐 OFFICIAL paid-tier charts (embedded from public sites — these ARE the real ones)"):
            st.caption(
                "These iframes embed the OFFICIAL paid-tier charts from LookIntoBitcoin.com "
                "(Phillip Swift's site), Woobull.com (Willy Woo), Mempool.space, and Coinglass. "
                "All publicly accessible — same charts the gurus look at, no paid subscription needed."
            )
            try:
                from core.btc_external_embeds import EMBEDS, render_embed_html
                # Selection: most important ones for daily use
                _featured = [
                    "lookintobitcoin_pi_top",
                    "lookintobitcoin_pi_bottom",
                    "lookintobitcoin_rainbow",
                    "lookintobitcoin_mvrv_z",
                    "lookintobitcoin_reserve_risk",
                    "lookintobitcoin_hodl_waves",
                    "lookintobitcoin_nupl",
                    "lookintobitcoin_puell",
                    "lookintobitcoin_realized_price",
                    "lookintobitcoin_golden_ratio",
                    "lookintobitcoin_2y_mult",
                    "lookintobitcoin_thermo_cap",
                    "coinglass_funding",
                    "coinglass_liquidations",
                    "mempool_hash_rate",
                ]
                _tabs = st.tabs(["Swift signature", "On-chain", "Derivatives", "Mining"])
                with _tabs[0]:  # Swift
                    for key in ["lookintobitcoin_pi_top", "lookintobitcoin_pi_bottom",
                                  "lookintobitcoin_rainbow", "lookintobitcoin_golden_ratio",
                                  "lookintobitcoin_2y_mult"]:
                        if key in EMBEDS:
                            label, url, descr, _, h = EMBEDS[key]
                            st.markdown(f"**{label}** — {descr}")
                            st.markdown(render_embed_html(url, height=h), unsafe_allow_html=True)
                with _tabs[1]:  # On-chain
                    for key in ["lookintobitcoin_mvrv_z", "lookintobitcoin_reserve_risk",
                                  "lookintobitcoin_hodl_waves", "lookintobitcoin_nupl",
                                  "lookintobitcoin_realized_price", "lookintobitcoin_puell"]:
                        if key in EMBEDS:
                            label, url, descr, _, h = EMBEDS[key]
                            st.markdown(f"**{label}** — {descr}")
                            st.markdown(render_embed_html(url, height=h), unsafe_allow_html=True)
                with _tabs[2]:  # Derivatives
                    for key in ["coinglass_funding", "coinglass_liquidations"]:
                        if key in EMBEDS:
                            label, url, descr, _, h = EMBEDS[key]
                            st.markdown(f"**{label}** — {descr}")
                            st.markdown(render_embed_html(url, height=h), unsafe_allow_html=True)
                with _tabs[3]:  # Mining
                    for key in ["mempool_hash_rate", "mempool_difficulty", "mempool_block_rewards",
                                  "lookintobitcoin_thermo_cap"]:
                        if key in EMBEDS:
                            label, url, descr, _, h = EMBEDS[key]
                            st.markdown(f"**{label}** — {descr}")
                            st.markdown(render_embed_html(url, height=h), unsafe_allow_html=True)
            except Exception as _e:
                st.warning(f"Embeds — temporarily unavailable")

        # ─── SWIFT INDICATOR SUITE (LookIntoBitcoin signature) ───
        try:
            from core.dashboard_cache import get_cached as _gc
            _swift = _gc("swift_indicators")
            if _swift:
                grm = _swift.get("golden_ratio_multiplier", {})
                tym = _swift.get("two_year_ma_multiplier", {})
                lr = _swift.get("log_regression", {})
                cm = _swift.get("cap_models", {})
                nupl = _swift.get("lth_nupl", {})
                hodl = _swift.get("hodl_waves", {})

                st.markdown(
                    "<div style='font-size:10px; color:#888; text-transform:uppercase; "
                    "letter-spacing:2px; margin:14px 0 6px 0;'>"
                    "Phillip Swift / LookIntoBitcoin Indicators</div>",
                    unsafe_allow_html=True,
                )

                # 6 indicator cards
                _scols = st.columns(3)
                cards = []

                # Golden Ratio Multiplier
                if not grm.get("error"):
                    grm_emoji = grm.get("current_emoji", "")
                    grm_mult = grm.get("multiplier", 0)
                    grm_zone = grm.get("current_zone", "?")
                    color = C.get("bull") if grm_mult < 1.5 else (C.get("accent") if grm_mult < 5 else C.get("bear"))
                    cards.append((
                        "Golden Ratio Mult",
                        f"{grm_mult:.2f}× 350d MA",
                        f"{grm_emoji} {grm_zone}", color
                    ))

                # 2y MA Multiplier
                if not tym.get("error"):
                    tym_mult = tym.get("multiplier", 0)
                    tym_zone = tym.get("current_zone", "?")
                    tym_emoji = tym.get("current_emoji", "")
                    color = (C.get("bull") if tym_mult < 1 else
                             C.get("accent") if tym_mult < 3.5 else C.get("bear"))
                    cards.append((
                        "2y MA Multiplier",
                        f"{tym_mult:.2f}× 2y MA",
                        f"{tym_emoji} {tym_zone}", color
                    ))

                # Log Regression
                if not lr.get("error"):
                    dev = lr.get("deviation_pct", 0)
                    zone = lr.get("zone", "?")
                    emoji = lr.get("emoji", "")
                    color = (C.get("bull") if dev < -25 else
                             C.get("accent") if dev < 100 else C.get("bear"))
                    cards.append((
                        "Log Regression",
                        f"{dev:+.0f}% from model",
                        f"{emoji} {zone}", color
                    ))

                # LTH-NUPL
                if not nupl.get("error"):
                    n = nupl.get("lth_nupl", 0)
                    zone = nupl.get("zone", "?")
                    emoji = nupl.get("emoji", "")
                    color = (C.get("bear") if n > 0.7 else
                             C.get("accent") if n > 0.25 else C.get("bull"))
                    cards.append((
                        "LTH-NUPL",
                        f"{n:.2f}",
                        f"{emoji} {zone}", color
                    ))

                # Cap Models
                if not cm.get("error"):
                    bc = cm.get("bottom_cap", 0)
                    tc = cm.get("top_cap", 0)
                    cards.append((
                        "Cap Models",
                        f"${bc:,.0f} — ${tc:,.0f}",
                        "Bottom Cap → Top Cap", C.get("muted")
                    ))
                else:
                    cards.append(("Cap Models", "unavailable", "data missing", C.get("muted")))

                # HODL Waves proxy
                if hodl and not hodl.get("error"):
                    lth_pct = round(hodl.get("lth_supply_pct", hodl.get("lth_pct_est", 0)) or 0)
                    label = hodl.get("label") or ("HODLer-dominated" if lth_pct >= 70 else "distributing")
                    color = C.get("bull") if lth_pct > 60 else C.get("muted")
                    cards.append((
                        "HODL Waves (est)",
                        f"~{lth_pct}% LTH supply",
                        label, color
                    ))
                else:
                    cards.append(("HODL Waves (est)", "unavailable", "needs CoinMetrics", C.get("muted")))

                # Render cards in 3 columns
                for i, (title, value, subline, color) in enumerate(cards):
                    col = _scols[i % 3]
                    with col:
                        st.markdown(
                            f"<div style='padding:12px 14px; border-radius:8px; "
                            f"margin-bottom:10px; background:#13161c; border-left:3px solid {color};'>"
                            f"<div style='font-size:10px; color:#888; text-transform:uppercase;'>{title}</div>"
                            f"<div style='font-size:18px; font-weight:700; color:{color}; margin:4px 0 2px 0;'>{value}</div>"
                            f"<div style='font-size:11px; color:#aaa;'>{subline}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
        except Exception:
            pass

        # ─── BTC-NATIVE BOTTOM SCORECARD (guru-tier) ───
        try:
            from core.dashboard_cache import get_cached as _gc
            _btc_bot = _gc("btc_native_bottom_scorecard")
            if _btc_bot:
                _bn = _btc_bot.get("n_met", 0)
                _bnt = _btc_bot.get("n_total", 16)  # native bottom scorecard total (live count)
                _bl = _btc_bot.get("verdict_level", "HOLD")
                _bc = {
                    "EXTREME":     C.get("deep_bull", "#16a34a"),
                    "DEEP_VALUE":  C.get("deep_bull", "#16a34a"),
                    "STRONG_BUY":  C.get("bull", "#22c55e"),
                    "ACCUMULATE":  C.get("bull", "#22c55e"),
                    "WATCH":       C.get("accent", "#4a90e2"),
                    "HOLD":        C.get("muted", "#888"),
                }.get(_bl, C.get("muted", "#888"))
                st.markdown(
                    f"<div style='padding:14px 18px; border-radius:8px; margin-bottom:14px; "
                    f"background:#13161c; border-left:4px solid {_bc};'>"
                    f"<div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;'>"
                    f"<div><span style='font-size:10px; color:#888; text-transform:uppercase; letter-spacing:1.5px;'>"
                    f"🔺 BTC BOTTOM — buy BTC? (guru-tier cycle signals)</span><br>"
                    f"<span style='font-size:22px; font-weight:700; color:{_bc};'>"
                    f"{_bl.replace('_',' ')}</span> "
                    f"<span style='font-size:14px; color:#888;'>({_bn}/{_bnt} signals)</span></div>"
                    f"<div style='font-size:12px; color:#aaa; text-align:right; max-width:60%;'>"
                    f"{_btc_bot.get('verdict', '')[:140]}</div>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
                with st.expander(f"📋 All 12 guru-tier bottom criteria — {_bn} firing"):
                    rows = []
                    for c in _btc_bot.get("criteria", []):
                        rows.append({
                            "✓": "🔥" if c.get("met") else "○",
                            "Criterion": c.get("label", "?"),
                            "Status": c.get("status", "?")[:90],
                        })
                    if rows:
                        st.dataframe(pd.DataFrame(rows),
                                      width='stretch', hide_index=True)
                    st.caption(
                        "12 guru-tier bottom signals from the top crypto bottom-callers: "
                        "Pi Cycle Bottom (Swift), Hash Ribbon Golden Cross (Edwards), "
                        "NVT Signal (Woo), Mayer Multiple (Mayer), Price < LTH cost basis (Checkmate), "
                        "2y MA (Loukas), Funding rate extreme, Cycle day analog, AHR999, "
                        "Realized Cap DD, MVRV-Z capitulation, Coinbase Premium. "
                        "Calibrated on 2015/2018/2020/2022 cycle bottoms."
                    )
        except Exception:
            pass

        # ─── PATTERN TARGET ZONES (where is price vs supply/support) ───
        try:
            from core.dashboard_cache import get_cached as _gc
            _zones = _gc("pattern_zones")
            if _zones and "zones" in _zones:
                # Use LIVE BTC price (60s) for display, not the precomputed cache
                # price (can be hours old). The zone classification was computed
                # against the cached price, so distance % is from that snapshot.
                _curr_price = btc_price  # synced with sticky header
                _curr_zone = _zones.get("current_zone", "?")
                st.markdown(
                    f"<div style='font-size:10px; color:#888; text-transform:uppercase; "
                    f"letter-spacing:2px; margin:8px 0 6px 0;'>"
                    f"Pattern Target Zones &mdash; BTC ${_curr_price:,.0f} ({_curr_zone})</div>",
                    unsafe_allow_html=True,
                )
                # Compact zone strip
                _zone_cells = []
                for z in _zones["zones"]:
                    inside = z["status"] == "INSIDE"
                    _dp = z.get("distance_pct", 0)
                    if inside:
                        border = "3px solid #ffffff"; bg = "rgba(255,255,255,0.12)"; color = "#fff"
                    else:
                        _base = "239,68,68" if _dp > 0 else "34,197,94"
                        _alpha = max(0.08, min(0.42, 0.42 - abs(_dp) / 100))
                        border = "1px solid #2a2d36"; bg = f"rgba({_base},{_alpha:.2f})"; color = "#ddd"
                    _zone_cells.append(
                        f"<div style='flex:1; padding:8px 6px; border-radius:5px; "
                        f"background:{bg}; border:{border}; text-align:center; min-width:80px;'>"
                        f"<div style='font-size:9px; color:#999;'>{z['label'][:20]}</div>"
                        f"<div style='font-size:11px; color:{color}; font-weight:600;'>"
                        f"${z['low']/1000:.0f}-${z['high']/1000:.0f}k</div>"
                        f"<div style='font-size:9px; color:#aaa;'>{z['distance_pct']:+.1f}%</div>"
                        f"</div>"
                    )
                st.markdown(
                    "<div style='display:flex; gap:4px; flex-wrap:wrap; margin-bottom:14px;'>" +
                    "".join(_zone_cells) + "</div>",
                    unsafe_allow_html=True,
                )
        except Exception:
            pass

        # ─── ETF flow regime indicator ───
        try:
            from core.dashboard_cache import get_cached as _gc
            _etf_reg = _gc("etf_regime")
            if _etf_reg and _etf_reg.get("regime") != "DATA_UNAVAILABLE":
                _etf_regime_name = _etf_reg.get("regime", "?")
                _etf_color = {
                    "STRONG_INFLOW":      C.get("deep_bull", "#16a34a"),
                    "ACCUMULATION":       C.get("bull", "#22c55e"),
                    "NEUTRAL":            C.get("muted", "#888"),
                    "DISTRIBUTION":       C.get("neutral", "#f0b90b"),
                    "HEAVY_OUTFLOW":      C.get("bear", "#ef4444"),
                    "CAPITULATION_FLOW":  C.get("deep_bear", "#b91c1c"),
                }.get(_etf_regime_name, C.get("muted"))
                _top_warn = _etf_reg.get("top_warning")
                _bot_warn = _etf_reg.get("bottom_warning")
                warn_text = ""
                if _top_warn:
                    warn_text = " ⚠ HEAVY OUTFLOWS NEAR PEAK — top distribution signal"
                elif _bot_warn:
                    warn_text = " ⚠ CAPITULATION FLOWS + DEEP DRAWDOWN — bottom forming"
                _f5 = _etf_reg.get('flows_5d_M', 0) or 0
                _f30 = _etf_reg.get('flows_30d_M', 0) or 0
                _f60 = _etf_reg.get('flows_60d_M', 0) or 0
                _fmax = max(1.0, abs(_f5), abs(_f30), abs(_f60))

                def _fbar(_lbl, _v):
                    _w = abs(_v) / _fmax * 100
                    _col = "#22c55e" if _v >= 0 else "#ef4444"
                    return (f"<div style='display:flex; align-items:center; gap:6px; margin:2px 0;'>"
                            f"<span style='font-size:9px; color:#888; width:26px;'>{_lbl}</span>"
                            f"<div style='flex:1; height:8px; background:#1a1d24; border-radius:2px;'>"
                            f"<div style='width:{_w:.0f}%; height:100%; background:{_col}; "
                            f"border-radius:2px;'></div></div>"
                            f"<span style='font-size:10px; color:#ccc; width:62px; text-align:right;'>"
                            f"${_v:+,.0f}M</span></div>")
                st.markdown(
                    f"<div style='padding:10px 14px; border-radius:6px; margin-bottom:14px; "
                    f"background:#13161c; border-left:3px solid {_etf_color}; "
                    f"display:flex; justify-content:space-between; gap:14px; flex-wrap:wrap; "
                    f"align-items:center;'>"
                    f"<div><span style='font-size:10px; color:#888;'>ETF Flow Regime:</span> "
                    f"<b style='color:{_etf_color}; font-size:14px;'>{_etf_regime_name.replace('_',' ')}</b>"
                    f"<span style='color:#aaa; font-size:11px;'>{warn_text}</span></div>"
                    f"<div style='min-width:220px;'>"
                    + _fbar("5d", _f5) + _fbar("30d", _f30) + _fbar("60d", _f60)
                    + "</div></div>",
                    unsafe_allow_html=True,
                )
        except Exception:
            pass

        # ─── 6-theme heat strip ───
        st.markdown(
            "<div style='font-size:10px; color:#888; text-transform:uppercase; "
            "letter-spacing:2px; margin:8px 0 6px 0;'>"
            "Theme Composites &mdash; positive = bullish equity, negative = bearish</div>",
            unsafe_allow_html=True,
        )
        _theme_order = ["LIQUIDITY", "CREDIT", "GROWTH", "VALUATION", "SENTIMENT", "BTC_ONCHAIN"]
        _th_labs, _th_vals = [], []
        for t in _theme_order:
            info = _themes.get(t, {}) or {}
            _z = info.get("z", 0)
            _th_vals.append(float(_z) if _z is not None else 0.0)
            _th_labs.append(t.replace("_", " ").title())
        st.plotly_chart(_diverging_fig(_th_labs, _th_vals, height=260),
                        width='stretch',
                        config={"displayModeBar": False, "displaylogo": False})

        # ─── Vetoes banner if any ───
        if _vetoes:
            st.error(f"⛔ Active vetoes: {', '.join(_vetoes)} — these override the standard scorecard math")

        # ─── Quick reference: what each gauge means ───
        with st.expander("ℹ️ Quick reference — what each gauge & score means"):
            st.markdown("""
**Top Score** (`Top composite z`): equity exit pressure derived from VALUATION + SENTIMENT themes.
- `< 0`: no exit signal — bull continues
- `0 to +1.5`: forming — tighten stops, build cash buffer
- `> +1.5`: extreme — historically associated with major tops (1999, 2007, 2021)

**BTC Bottom** (`Bottom composite z`): BTC bottoming signal derived from BTC_ONCHAIN + LIQUIDITY themes.
- `< 0`: no bottom signal — don't add BTC
- `0 to +1.5`: approaching — start scaling in carefully
- `> +1.5`: deep capitulation — deploy aggressively

**Bottom Probability**: sigmoid of bottom composite z. >70% = strong deploy signal.

**Liquidity Z**: net liquidity (WALCL - WTREGEN - RRP) z-score over 2y window.
- `< -0.5`: tight — risk-off conditions
- `> +0.5`: supportive — pro-cyclical for risk assets

**Cycle 5 Position**: days since halving 4 (Apr 2024) as % of 4-year cycle. ~25% = early bull, ~75% = late cycle.

**Theme strip**: each composite is sign-corrected so positive = bullish equity. VALUATION negative means "expensive = bearish forward."

**Verdict card**:
- HOLD: no signals firing
- WATCH: 2+ early indicators firing, no action yet
- TRIM: top scorecard 3/10+, reduce equity 25%
- DEFENSIVE: 5/10+, reduce equity 50%
- BEAR_CONFIRMED: 7/10+, reduce equity 80%
- FULL_ROTATION: 9/10+, exit equity almost entirely
- ROTATE_TO_CASH: early rotation 5/7+ firing
""")

        st.divider()
    except Exception as _e:
        st.error(f"Cockpit error: {_e}")
        import traceback; st.code(traceback.format_exc())

    # ═══════════════════════════════════════════════════════════════════
    # TOP 1% PREDICTOR ENGINE — calibrated framework (10 quant modules)
    # ═══════════════════════════════════════════════════════════════════
    try:
        pe = cached_predictor_engine()
        st.markdown(
            "<div class='section-header'>🧠 Top 1% Predictor Engine — calibrated framework</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "10-module quantitative architecture: rolling z-score standardization, "
            "Information Coefficient table (Spearman vs forward returns), 6 theme "
            "composites (LIQUIDITY/CREDIT/GROWTH/VALUATION/SENTIMENT/BTC_ONCHAIN), "
            "BTC state classifier (DEEP/SHALLOW/UNCONFIRMED), Kelly + vol-targeted "
            "position sizing, walk-forward backtests, Monte Carlo stress, failure "
            "detection. This is the research-grade engine — the rule-based engine "
            "below remains for auditable decisions."
        )

        # Header row — IC table + signal coverage + failure status
        ic = pe.get("ic_table", {})
        n_snap = pe.get("raw_signal_snapshot_n", 0)
        failures = pe.get("failure_checks", {})

        _pe_top = st.columns([1, 1, 1, 1])
        with _pe_top[0]:
            st.metric("IC Table signals", ic.get("n_signals", 0))
        with _pe_top[1]:
            st.metric("Signal snapshot", n_snap)
        with _pe_top[2]:
            st.metric("Regime", pe.get("regime", "?"))
        with _pe_top[3]:
            n_fail = failures.get("n_failures", 0)
            st.metric("Failures", n_fail,
                       delta="halt" if failures.get("halt_required") else "ok")

        # Decision composites — the 3 most important numbers
        dec = pe.get("decision_composites", {})
        themes = pe.get("theme_composites", {})

        st.markdown(
            f"<div style='padding:14px 18px; border-radius:8px; "
            f"background:#13161c; border:1px solid #2a2d36; margin:10px 0;'>"
            f"<span style='color:#888; font-size:11px; text-transform:uppercase;'>"
            f"Calibrated Decision Composites (z-scores)</span><br>"
            f"<b>Top composite:</b> {dec.get('top', 0):+.2f} "
            f"<span style='color:#888;'>(high = exit equity)</span>  &nbsp;|&nbsp; "
            f"<b>Early composite:</b> {dec.get('early', 0):+.2f} "
            f"<span style='color:#888;'>(high = rotate cash)</span>  &nbsp;|&nbsp; "
            f"<b>Bottom composite:</b> {dec.get('bottom', 0):+.2f} "
            f"<span style='color:#888;'>(high = deploy BTC)</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.caption("↑ The 6 theme composites are shown as the diverging bar above.")

        # BTC State + Calibrated allocation
        st.markdown("<br>", unsafe_allow_html=True)
        _alloc_cols = st.columns([1, 1])
        with _alloc_cols[0]:
            btc_s = pe.get("btc_state", {})
            plan = pe.get("btc_entry_plan", {})
            state_color = {
                "DEEP_GENERATIONAL":       C.get("bull", "#22c55e"),
                "SHALLOW_ETF_DRIVEN":      C.get("accent", "#f0b90b"),
                "BTC_BOTTOM_UNCONFIRMED":  C.get("muted", "#888"),
            }.get(btc_s.get("state"), C.get("muted", "#888"))
            st.markdown(
                f"<div style='padding:14px 18px; border-radius:8px; "
                f"background:#13161c; border-left:4px solid {state_color};'>"
                f"<div style='font-size:10px; color:#888; text-transform:uppercase;'>"
                f"BTC Bottom State</div>"
                f"<div style='font-size:20px; font-weight:700; color:{state_color}; margin-top:6px;'>"
                f"{btc_s.get('state', '?').replace('_', ' ')}</div>"
                f"<div style='font-size:13px; color:#ccc; margin-top:6px;'>"
                f"Composite z: <b>{btc_s.get('composite_z', 0):+.2f}</b> | "
                f"Probability: <b>{btc_s.get('bottom_probability', 0):.1%}</b> | "
                f"Components: {btc_s.get('n_components_used', 0)}/8"
                f"</div>"
                f"<div style='font-size:12px; color:#aaa; margin-top:8px;'>"
                f"Entry plan: <b>{plan.get('initial_tranche_pct', 0)}%</b> initial + "
                f"<b>{plan.get('dca_remaining_pct', 0)}%</b> DCA over "
                f"<b>{plan.get('dca_days', 0)}</b>d (max alloc {plan.get('max_alloc_pct', 0)}%)"
                f"</div></div>",
                unsafe_allow_html=True,
            )
        with _alloc_cols[1]:
            cal = pe.get("calibrated_allocation", {}).get("weights_pct", {})
            cal_nzd = pe.get("calibrated_allocation", {}).get("nzd", {})
            rule = pe.get("rule_based_allocation", {})
            st.markdown(
                f"<div style='padding:14px 18px; border-radius:8px; "
                f"background:#13161c; border-left:4px solid {C.get('accent', '#f0b90b')};'>"
                f"<div style='font-size:10px; color:#888; text-transform:uppercase;'>"
                f"Calibrated Allocation (Kelly + vol-target)</div>"
                f"<div style='font-size:13px; color:#ccc; margin-top:8px;'>"
                f"Equity: <b>{cal.get('equity', 0):.1f}%</b> "
                f"{_money_paren(cal_nzd.get('equity', 0))} "
                f"<span style='color:#888;'>vs rule {rule.get('equity', 0):.1f}%</span><br>"
                f"BTC: <b>{cal.get('btc', 0):.1f}%</b> "
                f"{_money_paren(cal_nzd.get('btc', 0))} "
                f"<span style='color:#888;'>vs rule {rule.get('btc', 0):.1f}%</span><br>"
                f"Staging: <b>{cal.get('staging', 0):.1f}%</b> "
                f"{_money_paren(cal_nzd.get('staging', 0))} "
                f"<span style='color:#888;'>vs rule {rule.get('staging', 0):.1f}%</span>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

        # IC table + top signals expandable
        with st.expander(f"📊 IC Table — top signals by predictive power "
                          f"({ic.get('n_signals', 0)} validated)"):
            from core.research.ic_table import load_ic_table
            tbl = load_ic_table()
            if tbl:
                rows = []
                for name, row in list(tbl.items())[:20]:
                    rows.append({
                        "Signal": name,
                        "IC (SPY)": row.get("best_ic_spy"),
                        "Horizon (days)": row.get("best_h_spy"),
                        "Robust": row.get("robust_spy"),
                        "OOS Gate": "✓" if row.get("oos_pass_spy") else "✗",
                        "IC (BTC)": row.get("best_ic_btc"),
                    })
                df_disp = pd.DataFrame(rows)
                # Round numeric columns
                for col in ("IC (SPY)", "IC (BTC)"):
                    if col in df_disp.columns:
                        df_disp[col] = pd.to_numeric(df_disp[col], errors="coerce").round(3)
                st.dataframe(df_disp, width='stretch', hide_index=True)
                st.caption(
                    "IC > 0.05 = decent | IC > 0.10 = rare/valuable | IC > 0.15 = gold. "
                    "OOS Gate passes if |IC| > 0.03 at BOTH 6m AND 12m. "
                    "Robust = neighboring horizons agree on sign."
                )
            else:
                st.info("IC table not yet computed. Will refresh on next precompute cycle.")

        # Failure check status
        with st.expander("🚨 Engine failure detection — kill criteria status"):
            checks = failures.get("checks_run", {})
            active = failures.get("active_failures", [])
            for check_name, fired in checks.items():
                emoji = "🔥" if fired else "✓"
                color = C.get("bear", "#ef4444") if fired else C.get("bull", "#22c55e")
                st.markdown(
                    f"<div style='padding:6px 10px; margin:3px 0; border-left:3px solid {color};'>"
                    f"<b>{emoji} {check_name}</b></div>",
                    unsafe_allow_html=True,
                )
            if active:
                st.markdown("**Active failures:**")
                for f in active:
                    st.error(
                        f"[{f['severity']}] {f['name']}: action = {f.get('action')}"
                    )
            else:
                st.success("All failure checks passing.")

        # The 10 modules manifest
        with st.expander("🏗️ The 10 modules — research-grade architecture"):
            st.markdown("""
**Phase 1-3: Research Pipeline**
- `core/research/standardize.py` — rolling z-scores (10y window), percentile ranks, velocity
- `core/research/ic_table.py` — Spearman IC across horizons, optimal-horizon search, OOS gating
- `core/research/walk_forward.py` — expanding-window walk-forward, threshold stability gates

**Phase 4-5: Composites + Regime**
- `core/composites.py` — 6 theme composites (LIQUIDITY, CREDIT, GROWTH, VALUATION, SENTIMENT, BTC_ONCHAIN), IC-weighted
- `core/regime_hmm.py` — 3-state Gaussian HMM validator for the rule-based regime classifier

**Phase 6-7: Sizing + BTC State**
- `core/position_size.py` — Kelly fraction (capped at 25%) + vol-targeted (12% target) + drawdown brake + turnover constraint
- `core/btc_state.py` — DEEP_GENERATIONAL / SHALLOW_ETF_DRIVEN / BTC_BOTTOM_UNCONFIRMED classifier with entry plans

**Phase 8-9: Backtest + Stress**
- `core/backtest/replay.py` — historical scenario replay (2000-03, 2007-09, 2018, 2020, 2022, 2024-25)
- `core/backtest/stress.py` — Monte Carlo bootstrap of regime returns + synthetic stagflation/policy-error/etf-mania scenarios

**Phase 10: Failure Detection**
- `core/failure_detection.py` — 6 kill criteria: underperform 12m, drawdown >15%, HMM disagreement, IC decay, regime thrashing, missed BTC upcycle

**Coordinator:**
- `core/predictor_engine.py` — orchestrator that pulls all 10 modules into one dashboard call

**Falsification gates** built into the engine:
- IC < 0.03 at 6m AND 12m → demote signal
- Walk-forward CoV > 0.20 → threshold overfit, drop
- Monte Carlo P5 Sharpe < 0 → engine fails stress
- HMM vs rule-based < 80% agreement → regime mis-calibrated
- Realized drawdown < -15% → halt rebalancing + alert
""")

    except Exception as e:
        st.error(f"⚠ Predictor Engine error: {e}")
        import traceback
        st.code(traceback.format_exc())

    st.divider()

    # ═══════════════════════════════════════════════════════════════════
    # 📚 LEGACY / DETAIL — superseded panels kept for reference
    # Everything below this point is detail-only. The Cockpit + Swift
    # panels above are the primary decision view (Swift's recommendation).
    # ═══════════════════════════════════════════════════════════════════
    st.markdown(
        "<div style='padding:14px 18px; margin:14px 0; border-radius:8px; "
        "background:#13161c; border-left:4px solid #888;'>"
        "<div style='font-size:11px; color:#888; text-transform:uppercase; "
        "letter-spacing:1.5px; margin-bottom:4px;'>Legacy / Detail Panels</div>"
        "<div style='font-size:13px; color:#aaa;'>The <b>Unified Decision Engine and "
        "Cockpit above</b> give the verdict. Panels below add the supporting scorecards, "
        "history and confirmation detail behind it.</div>"
        "</div>",
        unsafe_allow_html=True,
    )


with _unified_top:   # <- 2026-07-04 restructure
    # ═══════════════════════════════════════════════════════════════════
    # UNIFIED DECISION ENGINE — single source of truth at the top
    # Combines: macro layer (8 leading) + regime state machine
    # + 3 scorecards + liquidity overlay + staging basket
    # ═══════════════════════════════════════════════════════════════════
    try:
        ud = cached_unified_decision()
        regime = ud["regime"]
        regime_color = {
            "RISK_ON":           C.get("bull", "#22c55e"),
            "LATE_CYCLE":        C.get("accent", "#f0b90b"),
            "RECESSIONARY_BEAR": C.get("deep_bear", "#ef4444"),
        }.get(regime, C.get("muted", "#888"))
        regime_emoji = {
            "RISK_ON":           "🟢",
            "LATE_CYCLE":        "🟡",
            "RECESSIONARY_BEAR": "🔴",
        }.get(regime, "⚪")

        st.markdown(
            f"<div class='section-header'>🎯 Unified Decision — {regime_emoji} Regime: "
            f"<span style='color:{regime_color};'>{regime}</span></div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Single source of truth: 8 incremental macro signals (OECD CLI, LEI, MOVE, "
            "RRP, credit impulse, SLOOS, Sahm, claims) + regime state machine "
            "(RISK_ON / LATE_CYCLE / RECESSIONARY_BEAR) + all 3 scorecards "
            "(Top, Early Rotation, BTC Bottom) + liquidity overlay + 3-asset staging "
            "basket (BIL/VTIP/GLDM). Druckenmiller/PTJ/Zulauf/Howell/Hayes framework."
        )

        # Regime banner with bucket scores
        b = ud["regime_buckets"]
        liq_z = ud["liquidity"]["z"]
        vetoes = ud["vetoes_active"]

        liq_color = C.get("bull", "#22c55e") if liq_z > 0 else C.get("bear", "#ef4444")
        _ud_cols = st.columns([2, 1, 1])
        with _ud_cols[0]:
            st.markdown(
                f"<div style='padding:18px 22px; border-radius:8px; "
                f"border-left:6px solid {regime_color}; "
                f"background:linear-gradient(90deg, {regime_color}22 0%, #1a1d24 100%);'>"
                f"<div style='font-size:11px; color:#888; text-transform:uppercase; letter-spacing:1.5px;'>"
                f"Macro Regime</div>"
                f"<div style='font-size:30px; font-weight:800; color:{regime_color}; line-height:1.1; margin:6px 0;'>"
                f"{regime_emoji} {regime}</div>"
                f"<div style='font-size:13px; color:#ccc;'>"
                f"Growth: <b>{b['growth']}/4</b> | Plumbing: <b>{b['plumbing']}/4</b> | "
                f"Credit: <b>{b['credit']}/3</b> | "
                f"Liquidity z: <b style='color:{liq_color};'>"
                f"{liq_z:+.2f}</b>"
                f"</div></div>",
                unsafe_allow_html=True,
            )
        with _ud_cols[1]:
            t = ud["target_allocation_pct"]
            n = ud["target_allocation_nzd"]
            st.markdown(
                f"<div style='padding:14px 18px; border-radius:8px; "
                f"background:#1a1d24; border:1px solid #333; height:100%;'>"
                f"<div style='font-size:10px; color:#888; text-transform:uppercase;'>Target allocation</div>"
                f"<div style='font-size:14px; color:#ccc; margin-top:6px;'>"
                f"<b style='color:{C.get('bear', '#ef4444')};'>Equity</b>: "
                f"{t['equity']:.0f}%{_money_paren(n['equity'])}<br>"
                f"<b style='color:{C.get('accent', '#f0b90b')};'>BTC</b>: "
                f"{t['btc']:.0f}%{_money_paren(n['btc'])}<br>"
                f"<b style='color:{C.get('bull', '#22c55e')};'>Staging</b>: "
                f"{t['staging']:.0f}%{_money_paren(n['staging'])}"
                f"</div></div>",
                unsafe_allow_html=True,
            )
        with _ud_cols[2]:
            sb = ud["staging_basket_pct"]
            sn = ud["staging_basket_nzd"]
            st.markdown(
                f"<div style='padding:14px 18px; border-radius:8px; "
                f"background:#1a1d24; border:1px solid #333; height:100%;'>"
                f"<div style='font-size:10px; color:#888; text-transform:uppercase;'>Staging basket</div>"
                f"<div style='font-size:14px; color:#ccc; margin-top:6px;'>"
                f"<b>BIL</b>: {sb.get('BIL',0)}%{_money_paren(sn.get('BIL',0))}<br>"
                f"<b>VTIP</b>: {sb.get('VTIP',0)}%{_money_paren(sn.get('VTIP',0))}<br>"
                f"<b>GLDM</b>: {sb.get('GLDM',0)}%{_money_paren(sn.get('GLDM',0))}"
                f"</div></div>",
                unsafe_allow_html=True,
            )

        # Vetoes banner if any active
        if vetoes:
            st.markdown(
                f"<div style='padding:12px 18px; border-radius:6px; margin-top:8px; "
                f"background:#3a1a1a; border-left:4px solid {C.get('deep_bear', '#ef4444')};'>"
                f"<b>⛔ Active vetoes:</b> {', '.join(vetoes)}<br>"
                f"<span style='color:#aaa; font-size:12px;'>"
                f"These override the standard scorecard math.</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Scorecard summary row (use _ud_sc to avoid clobbering global `sc`)
        _ud_sc = ud["scorecards"]
        thr = ud["regime_thresholds"]
        st.markdown(
            f"<div style='padding:10px 14px; border-radius:6px; margin-top:8px; "
            f"background:#13161c; border:1px solid #2a2d36;'>"
            f"<span style='color:#888; font-size:11px;'>SCORECARDS (regime-adjusted thresholds):</span><br>"
            f"<b>Top:</b> {_ud_sc['top']['n_met']}/{_ud_sc['top']['n_total']} "
            f"<span style='color:{C.get('accent')};'>→ {_ud_sc['top']['action']}</span> "
            f"(trim≥{thr['top']['trim']}, def≥{thr['top']['defensive']}, "
            f"bear≥{thr['top']['bear_confirmed']}, full≥{thr['top']['full_rotation']})  &nbsp;|&nbsp; "
            f"<b>Early:</b> {_ud_sc['early']['n_firing']}/{_ud_sc['early']['n_total']} "
            f"<span style='color:{C.get('accent')};'>→ {_ud_sc['early']['action']}</span>"
            f"{' ⚡accel' if _ud_sc['early']['accelerating'] else ''}  &nbsp;|&nbsp; "
            f"<b>BTC Bottom:</b> {_ud_sc['bottom']['n_met']}/{_ud_sc['bottom']['n_total']} "
            f"(partial≥{thr['btc_partial']}, full≥{thr['btc_full']}, max alloc {thr['max_btc_pct']:.0f}%)"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Rotation amount banner if non-zero
        if ud["rotation_nzd"] > 0:
            st.markdown(
                f"<div style='padding:14px 18px; border-radius:8px; margin-top:8px; "
                f"background:#1a1d24; border-left:4px solid {C.get('bear')};'>"
                f"<b>💼 Rotation now:</b> Move <b style='color:{C.get('bear')};'>{_money(ud['rotation_nzd'])}</b> "
                f"from equity → staging basket. Equity {ud['current_equity_pct']:.0f}% → "
                f"{t['equity']:.0f}%. Staging basket allocation:<br>"
                f"<span style='color:#aaa; font-size:13px;'>{ud['staging_basket_rationale']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Expandables
        with st.expander("📊 All 8 incremental macro signals"):
            macro_rows = []
            labels = {
                "oecd_cli_6m":       "1. OECD CLI 6m change",
                "cb_lei_yoy":        "2. CB LEI YoY (CLI proxy)",
                "move_elevated":     "3. MOVE Index (bond vol)",
                "dollar_liq_stress": "4. Dollar plumbing (RRP + SOFR-IORB)",
                "credit_impulse":    "5. US credit impulse (BUSLOANS)",
                "sloos_tightening":  "6. SLOOS net tightening (banks)",
                "sahm_rule":         "7. Sahm Rule",
                "claims_cross":      "8. Initial claims 4w vs 12m MA",
            }
            for key, sig in ud.get("macro_signals_summary", {}).items():
                if key == "asof": continue
                macro_rows.append({
                    "#": labels.get(key, key),
                    "Status": "🔥 FIRING" if sig.get("firing") else "✓ ok",
                    "Detail": sig.get("status", "")[:100],
                })
            if macro_rows:
                st.dataframe(pd.DataFrame(macro_rows),
                              width='stretch', hide_index=True)

        with st.expander("📐 Regime thresholds + sizing logic"):
            st.markdown(f"""
**Current regime: {regime}** ({regime_emoji})

**Why this regime**:
- Growth bucket {b['growth']}/4: OECD CLI, LEI, Sahm Rule, claims
- Plumbing bucket {b['plumbing']}/4: MOVE, RRP, SOFR-IORB, HY spread
- Credit bucket {b['credit']}/3: credit impulse, SLOOS, yield curve uninvert
- Yield curve un-invert: {ud['regime_curve_uninvert']}

**Regime classification rules**:
- `RECESSIONARY_BEAR`: Sahm fires OR curve un-inverts OR growth ≥3 OR (growth ≥2 AND plumbing ≥2)
- `LATE_CYCLE`: growth ≥1 OR plumbing ≥2 OR credit ≥1
- `RISK_ON`: clean bill of health

**Thresholds for {regime}** (looser → tighter as regime worsens):
- Top Scorecard: TRIM≥{thr['top']['trim']}, DEFENSIVE≥{thr['top']['defensive']}, BEAR≥{thr['top']['bear_confirmed']}, FULL≥{thr['top']['full_rotation']}
- Early Rotation: WATCH≥{thr['early']['watch']}, REDUCE≥{thr['early']['reduce']}, ROTATE_CASH≥{thr['early']['rotate_to_cash']}
- BTC deploy: partial≥{thr['btc_partial']}/8, full≥{thr['btc_full']}/8, max alloc {thr['max_btc_pct']:.0f}%
- Baseline equity: {thr['baseline_equity_pct']:.0f}%

**BTC continuous sizing formula** (NOT hard buckets):
```
base = ramp(bottom_n, partial, full)  # 0..1
regime_mult = {{RISK_ON: 0.7, LATE_CYCLE: 1.0, RECESSIONARY_BEAR: 1.3}}
liq_mult = clip(1 + 0.2 * liq_z, 0.5, 1.5)
target_btc = base * regime_mult * liq_mult * max_alloc
```
This means BTC sizing scales smoothly with bottom score, regime, and liquidity — Hayes "buy fear" multiplier in recessionary bear gives 1.3x boost.

**Vetoes** (override scorecard math):
- `force_cash_move_spike`: MOVE > 150 → forces cash
- `no_btc_during_collapse`: BEAR + liq_z<-1 + credit impulse<-3 → bans BTC
- `no_equity_add_recession_start`: curve uninverted + claims rising → caps equity at 10%
""")

        with st.expander("🧠 Why this is the pro framework"):
            st.markdown("""
**The 3-tier rotation framework (Druckenmiller / PTJ / Zulauf / Howell / Hayes):**

1. **Equity → Cash (preserve)** when leading macro/credit signals fire.
   Druckenmiller: *"I'm 60% cash before recession starts. I cannot afford to be wrong on this."*

2. **Cash → BTC (deploy at low)** when BTC bottom scorecard confirms AND
   correlation has decoupled AND liquidity z-score is supportive.
   Hayes: *"Buy fear. The fiscal dominance trade requires patience but
   pays asymmetrically."*

3. **Staging basket** during the cash phase isn't 100% T-bills:
   - **BIL** (T-bills): liquid, 4-5% yield, near-zero risk — base layer
   - **VTIP** (short TIPS): captures real-yield compression as Fed pivots
   - **GLDM** (gold): crisis hedge when fiscal dominance + low real yields

**Why the regime layer matters** (decision asymmetry):
- In RISK_ON: act late (high thresholds), preserve equity upside
- In LATE_CYCLE: act earlier (lower thresholds), tighten stops
- In RECESSIONARY_BEAR: act aggressively (lowest thresholds + 1.3x BTC mult on bottom)

**The 2022 lesson encoded**: SPY -25%, BTC -77%.
Without regime awareness, naive "rotate equity → BTC at top" lost MORE
than staying in equity. This engine routes through CASH (T-bills/TIPS/gold)
until BTC bottom actually fires.
""")
    except Exception as e:
        st.error(f"⚠ Unified Decision error: {e}")
        import traceback
        st.code(traceback.format_exc())

    st.divider()

    # === 90d BTC price chart with cost-basis lines ===
    st.markdown("<div class='section-header'>BTC Price — last 90 days with cost-basis levels</div>",
                 unsafe_allow_html=True)
    try:
        px_df = cached_ohlcv_90d()
        if not px_df.empty:
            px_df = px_df.sort_index() if not px_df.index.is_monotonic_increasing else px_df
            _pf = go.Figure()
            _pf.add_trace(go.Candlestick(
                x=px_df.index, open=px_df["open"], high=px_df["high"],
                low=px_df["low"], close=px_df["close"],
                increasing_line_color=C["bull"], decreasing_line_color=C["bear"],
                increasing_fillcolor=C["bull"], decreasing_fillcolor=C["bear"],
                name="BTC",
            ))
            # Cost-basis lines
            if rp and not rp.get("error"):
                _pf.add_hline(y=rp["value"], line=dict(color=C["lth"], width=2),
                              annotation_text=f"LTH cost basis ${rp['value']:,.0f}",
                              annotation_position="left",
                              annotation_font_color=C["lth"])
            if sth and not sth.get("error"):
                _pf.add_hline(y=sth["value"], line=dict(color=C["sth"], width=2),
                              annotation_text=f"STH cost basis ${sth['value']:,.0f}",
                              annotation_position="left",
                              annotation_font_color=C["sth"])
            if pdb and not pdb.get("error"):
                _pf.add_hline(y=pdb["expected_value_price"],
                              line=dict(color=C["bull"], width=2, dash="dash"),
                              annotation_text=f"EV bottom ${pdb['expected_value_price']:,.0f}",
                              annotation_position="right",
                              annotation_font_color=C["bull"])
            _pf.add_hline(y=btc_price, line=dict(color="#fff", width=1, dash="dot"),
                          annotation_text=f"NOW ${btc_price:,.0f}",
                          annotation_position="right",
                          annotation_font_color="#fff")
            _pf.update_layout(
                **CHART_LAYOUT, height=380,
                xaxis=dict(rangeslider=dict(visible=False), gridcolor="#2a2d34"),
                yaxis=dict(gridcolor="#2a2d34", title="Price (USD)"),
                showlegend=False,
            )
            st.plotly_chart(_pf, width='stretch')
    except Exception as e:
        st.caption(f"Price chart unavailable: {e}")


with tab_signals:   # <- 2026-07-04 restructure
    # === EQUITY TOP CONFIRMATION SCORECARD (when to exit equities) ===
    # Primary rotation-side scorecard; renders in Signals below the divider.
    # Its headline verdict also rolls up to the Unified Decision Engine on top.
    try:
        _top_bundle = cached_top_scorecard()
        top_sc = _top_bundle["scorecard"]
        exit_rec = _top_bundle["recommendation"]
        # historical_backtest used later via the same bundle
        from core.btc_top_scorecard import historical_backtest  # noqa: F401

        top_color = {
            "FULL_ROTATION":  C["deep_bear"],
            "BEAR_CONFIRMED": C["bear"],
            "DEFENSIVE":      C["neutral"],
            "TRIM":           C["accent"],
            "HOLD":           C["muted"],
        }.get(top_sc["verdict_level"], C["muted"])

        st.markdown(
            "<div class='section-header'>🔻 Equity TOP — sell stocks? (macro/AAII/NAAIM/breadth)</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "10 hard criteria for equity top — Druckenmiller/Marks/Grantham framework. "
            "Phased exits: 3/10 → TRIM | 5/10 → DEFENSIVE | 7/10 → BEAR CONFIRMED | 9/10 → FULL ROTATION. "
            "Backtested at 2000/2008/2020/2022 tops."
        )

        # Big verdict banner
        _top_cols = st.columns([3, 2])
        with _top_cols[0]:
            st.markdown(
                f"<div style='padding:18px 22px; border-radius:8px; "
                f"border-left:6px solid {top_color}; "
                f"background:linear-gradient(90deg, {top_color}22 0%, #1a1d24 100%);'>"
                f"<div style='font-size:11px; color:#888; text-transform:uppercase; letter-spacing:1.5px;'>"
                f"Top Scorecard ({top_sc['n_met']}/{top_sc['n_total']} criteria firing)</div>"
                f"<div style='font-size:30px; font-weight:800; color:{top_color}; line-height:1.1; margin:6px 0;'>"
                f"{top_sc['verdict_level'].replace('_', ' ')}</div>"
                f"<div style='font-size:14px; color:#ccc;'>{top_sc['verdict']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with _top_cols[1]:
            st.markdown(metric_card(
                "Reduce equity to",
                f"{top_sc['reduce_to_pct']}%",
                "of original allocation",
                top_color,
            ), unsafe_allow_html=True)
            if SHOW_PERSONAL:  # personal NZD holdings — hidden on public view
                st.markdown(metric_card(
                    "Sell from equities",
                    f"${exit_rec['sell_nzd']:,.0f}",
                    f"NZD (of ${exit_rec['current_equity_nzd']:,.0f} held)",
                    C["bear"],
                ), unsafe_allow_html=True)

        st.info(f"💡 **Recommendation**: {exit_rec['rationale']}")

        # Criteria table
        with st.expander(f"📋 All 10 criteria — {top_sc['n_met']} firing"):
            sc_rows = []
            for c in top_sc["criteria"]:
                mark = "🔴 FIRING" if c["met"] else "⚪ not yet"
                sc_rows.append({
                    "Status": mark,
                    "Criterion": c["label"],
                    "Reading": c["status"],
                    "Why it matters": c["rationale"][:80],
                })
            st.dataframe(pd.DataFrame(sc_rows), width='stretch', hide_index=True)

        # Historical backtest
        with st.expander("📊 Historical backtest — did this work at past tops?"):
            bt = _top_bundle["backtest"]
            for period in bt["periods"]:
                n_fired_at_peak = period["n_met_at_peak"]
                n_fired_after = period["n_met_30d_after"]
                st.markdown(
                    f"**{period['label']}** ({period['peak_date']}) — "
                    f"{n_fired_at_peak}/10 at peak, {n_fired_after}/10 30d after"
                )
                st.markdown(f"  *{period['outcome']}*")
            st.caption("✓ At ALL 4 historical tops, scorecard reached BEAR_CONFIRMED (7+/10) within 30 days of peak.")

        # Phased exit explainer
        with st.expander("📐 How the phased exit works"):
            st.markdown(
                "| Criteria firing | Phase | Equity → reduce to | Druckenmiller wisdom |\n"
                "|---|---|---|---|\n"
                "| 0-2/10 | HOLD | 100% (no change) | No warning signals |\n"
                "| **3/10** | **TRIM** | 75% (sell 25%) | Early warning — take some off |\n"
                "| **5/10** | **DEFENSIVE** | 50% (cut in half) | Bear forming — get defensive |\n"
                "| **7/10** | **BEAR CONFIRMED** | 20% (cut to 1/5) | Bear confirmed — preserve capital |\n"
                "| **9/10** | **FULL ROTATION** | 5% (defensives only) | Time to be cash + BTC + defensives |\n"
            )

    except Exception as e:
        st.error(f"⚠ Top Scorecard error: {e}")
        import traceback
        st.code(traceback.format_exc())

    st.divider()

    # === EARLY ROTATION SIGNAL (Druckenmiller/PTJ/Zulauf — pre-empts standard top) ===
    try:
        er = cached_early_rotation()
        action_color = {
            "ROTATE_TO_BTC":   C["deep_bear"],   # urgent
            "ROTATE_TO_CASH":  C["bear"],         # serious
            "REDUCE_TO_CASH":  C["neutral"],      # warning
            "WATCH":           C["accent"],       # heads up
            "HOLD":            C["muted"],        # all clear
        }.get(er["action"], C["muted"])
        urgency_color = {
            "IMMEDIATE": C["deep_bear"],
            "HIGH":      C["bear"],
            "MEDIUM":    C["neutral"],
            "LOW":       C["muted"],
        }.get(er["urgency"], C["muted"])

        st.markdown(
            "<div class='section-header'>⚡ Early Rotation Signal — before equities dive</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Druckenmiller / Paul Tudor Jones / Felix Zulauf leading indicators. "
            "Pre-empts the standard Top Scorecard by 3-9 months. CRUCIAL: when leading "
            "signals fire but BTC isn't bottomed yet, this routes equity to CASH (not BTC) "
            "to avoid the 2022 mistake (SPY -25%, BTC -77%)."
        )

        _er_cols = st.columns([3, 2])
        with _er_cols[0]:
            st.markdown(
                f"<div style='padding:18px 22px; border-radius:8px; "
                f"border-left:6px solid {action_color}; "
                f"background:linear-gradient(90deg, {action_color}22 0%, #1a1d24 100%);'>"
                f"<div style='font-size:11px; color:#888; text-transform:uppercase; letter-spacing:1.5px;'>"
                f"Early Rotation Action ({er['n_firing']}/{er['n_total']} leading indicators firing)</div>"
                f"<div style='font-size:30px; font-weight:800; color:{action_color}; line-height:1.1; margin:6px 0;'>"
                f"{er['action'].replace('_', ' ')}</div>"
                f"<div style='font-size:14px; color:#ccc; line-height:1.5;'>"
                f"<b>Destination:</b> {er['destination']}  &nbsp;|&nbsp;  "
                f"<b style='color:{urgency_color};'>Urgency: {er['urgency']}</b>"
                f"</div></div>",
                unsafe_allow_html=True,
            )
        with _er_cols[1]:
            st.markdown(
                f"<div style='padding:14px 18px; border-radius:8px; "
                f"background:#1a1d24; border:1px solid #333;'>"
                f"<div style='font-size:10px; color:#888; text-transform:uppercase;'>"
                f"Target equity allocation</div>"
                f"<div style='font-size:24px; font-weight:700; color:{action_color};'>"
                f"{er['target_equity_pct']}%</div>"
                f"<div style='font-size:11px; color:#aaa;'>"
                f"{(_money(er['target_equity_nzd']) + ' (was ' + _money(er['current_equity_nzd']) + ')') if SHOW_PERSONAL else 'equity reduced per signal'}"
                f"</div>"
                f"<div style='font-size:10px; color:#888; text-transform:uppercase; margin-top:8px;'>"
                f"Move now</div>"
                f"<div style='font-size:18px; font-weight:600; color:{C['bear']};'>"
                f"{_money(er['rotation_nzd'])}"
                f"</div></div>",
                unsafe_allow_html=True,
            )

        # Recommendation box with the "why"
        st.markdown(
            f"<div style='padding:14px 18px; border-radius:8px; "
            f"background:#1a1d24; border-left:3px solid {C['accent']}; "
            f"margin-top:12px;'>"
            f"<b>💡 Pro logic:</b> {er['rationale']}<br>"
            f"<span style='color:#888; font-size:12px;'>BTC bottom proximity: "
            f"{er['btc_bottom_n_met']}/{er.get('btc_bottom_n_total') or 10} scorecard firing ({er['btc_bottom_zone']}). "
            f"{'BTC ready — rotate equity → BTC directly.' if er['btc_bottom_zone'] == 'AT_BOTTOM' else 'BTC not ready — route equity → CASH first.'}"
            f"</span></div>",
            unsafe_allow_html=True,
        )

        # All 7 leading indicators
        with st.expander(
            f"📋 All 7 leading indicators — {er['n_firing']} firing"
        ):
            ind_rows = []
            indicator_labels = {
                "small_caps_leading_down":   "1. Russell 2000 / SPY ratio (small caps top first)",
                "defensive_sector_rotation": "2. Defensive sectors outperforming (XLP/XLU/XLV vs XLK/XLY)",
                "yield_curve_resteepening":  "3. Yield curve re-steepening from inversion",
                "hy_spread_widening":        "4. HY credit spreads widening fast",
                "spy_below_200d":            "5. SPY < 200d MA by >1.5% (PTJ rule)",
                "vix_backwardation":         "6. VIX9D/VIX3M backwardation persistent",
                "btc_spy_correlation_high":  "7. BTC-SPY 30d correlation > 0.7",
            }
            for key, ind in er["indicators"].items():
                mark = "🔥 FIRING" if ind.get("firing") else "✓ ok"
                ind_rows.append({
                    "#": indicator_labels.get(key, key),
                    "Status": mark,
                    "Detail": ind.get("status", "")[:90],
                })
            st.dataframe(pd.DataFrame(ind_rows), width='stretch', hide_index=True)

        # Acceleration + BTC proximity context
        with st.expander("📊 Why route to CASH vs BTC? The 2022 lesson"):
            st.markdown("""
**Druckenmiller / Paul Tudor Jones / Felix Zulauf rotation framework:**

1. **Standard rotation (naive)**: Wait for equities to top → rotate to BTC.
   - Fatal flaw: BTC has 1.5x beta to SPY in liquidity crunches.
   - 2022: SPY -25%, BTC **-77%**. Rotating equity to BTC at the top
     would have lost you MORE than staying in equity.
   - 2020 Mar: SPY -34% in 1 month, BTC -50% same time.

2. **Pro rotation (this signal)**:
   - **Phase 1**: Leading indicators crack → rotate equity → **CASH** (BIL/SGOV).
     - Capital preservation first. Cash earns 4-5% yield while you wait.
   - **Phase 2**: BTC bottom scorecard fires (6/8+) → rotate cash → **BTC**.
     - Maximum compounding when BTC has actually bottomed.

3. **The decision tree this signal implements**:
   - 5+ leading firing AND BTC at bottom (≥6/8) AND BTC decoupled → **ROTATE_TO_BTC**
   - 5+ leading firing AND BTC not bottomed → **ROTATE_TO_CASH**
   - 3-4 leading firing AND accelerating → **ROTATE_TO_CASH** (urgent)
   - 3-4 leading firing not accelerating → **REDUCE_TO_CASH** (1/3 out)
   - 2 leading firing → **WATCH** (tighten stops, no action)
   - 0-1 leading firing → **HOLD** (stay allocated)

**Why these 7 indicators specifically:**
- Small caps lead large caps by 3-6 months at tops (1999, 2007, 2021).
- Defensive rotation = pros already exiting before headlines catch up.
- Yield curve un-inverting = 100% recession signal since 1970s.
- HY spread widening = credit market sees stress 1-3 months early.
- 200d break = PTJ's hard line. "If 200d breaks I'm out. Period."
- VIX backwardation persistent = regime shift accepted.
- BTC-SPY correlation high = BTC is NOT a hedge, it's leveraged SPY.

**What pros do with cash during the wait**:
- BIL (1-3 mo T-bills): 4-5% yield, near-zero risk
- SGOV (0-3 mo T-bills): same idea, slightly different duration
- Money market funds: same yield, instant liquidity
- DO NOT hold in equity broker cash account (often <1% yield)
""")
    except Exception as e:
        st.error(f"⚠ Early Rotation Signal error: {e}")
        import traceback
        st.code(traceback.format_exc())

    st.divider()


with tab_playbook:   # <- 2026-07-04 review fix: the deploy trigger IS execution
    # === ETF-AWARE BOTTOM TRIGGER (combines scorecard + ETF flows) ===
    st.markdown("<div class='section-header'>🔺 ETF-Aware Bottom Trigger — when to deploy BTC</div>",
                 unsafe_allow_html=True)
    st.caption(
        "Combines hard scorecard count with ETF flow direction. ETF-era cycles "
        "may never hit 6/8 scorecard — this surfaces shallow bottoms (Trigger 1A) "
        "where institutions absorb supply."
    )
    _tg_cols = st.columns([4, 2])
    with _tg_cols[0]:
        st.markdown(
            f"<div style='padding:18px 22px; border-radius:8px; "
            f"border-left:6px solid {trigger['color']}; "
            f"background:linear-gradient(90deg, {trigger['color']}22 0%, #1a1d24 100%);'>"
            f"<div style='font-size:10px; color:#888; text-transform:uppercase; letter-spacing:1.5px;'>"
            f"Active trigger</div>"
            f"<div style='font-size:13px; color:#aaa; margin-top:2px;'>"
            f"{trigger['trigger_name']}</div>"
            f"<div style='font-size:32px; font-weight:800; color:{trigger['color']}; line-height:1; margin-top:6px;'>"
            f"{trigger['verdict_label']}</div>"
            f"<div style='font-size:13px; color:#ccc; margin-top:10px;'>"
            f"{trigger['rationale']}</div>"
            f"<div style='font-size:12px; color:#888; margin-top:8px;'>"
            f"<b>Next levels:</b> {trigger['next_levels']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with _tg_cols[1]:
        # Mini cards for ETF status + entry zone
        etf_color = {
            "STRONG_POSITIVE": C["bull"], "POSITIVE": C["bull"],
            "FLAT": C["muted"], "NEGATIVE": C["bear"], "STRONG_NEGATIVE": C["deep_bear"],
            "UNKNOWN": C["muted"],
        }.get(trigger["etf_status"], C["muted"])
        st.markdown(metric_card(
            "Deploy %",
            f"{trigger['deploy_pct']}%",
            ("of stake" if trigger['deploy_pct'] else "no deploy yet"),
            trigger["color"],
        ), unsafe_allow_html=True)
        st.markdown(metric_card(
            "ETF status",
            trigger["etf_status"].replace("_", " "),
            f"5d: ${trigger['etf_5d_M']:+,.0f}M | 30d: ${trigger['etf_30d_M']:+,.0f}M",
            etf_color,
        ), unsafe_allow_html=True)
        st.markdown(metric_card(
            "Entry zone",
            trigger["entry_zone"],
            "expected if trigger fires",
            C["lth"],
        ), unsafe_allow_html=True)

    # 4-trigger matrix for clarity
    with st.expander("How the 4 triggers work"):
        st.markdown(
            "| Trigger | Conditions | Action | Entry zone |\n"
            "|---|---|---|---|\n"
            "| **2** | Scorecard ≥7/10 (any ETF) | **DEPLOY 100%** — traditional bottom confirmed | $50-60k |\n"
            "| **1B** | Scorecard ≥5/10 + ETF outflows | **SCALE IN 75%** — real bottom forming | $55-65k |\n"
            "| **1A** | Scorecard ≥5/10 + ETF inflows/flat | **SCALE IN 50%** — shallow ETF-era bottom | $60-70k |\n"
            "| **1C** | Scorecard ≥3/10 + Clemente/Alden strong | **SCALE IN 60%** — institutional early bottom | $60-70k |\n"
            "| **EARLY** | Scorecard 3-4/10 | Watch — prepare capital | n/a |\n"
            "| **WAIT** | Scorecard 0-2/10 | Cash is a position | n/a |\n"
            "| **4** | Strong ETF outflows + low scorecard | WAIT LONGER — bear deepening | $45-55k possible |\n"
        )
        st.caption(
            "Key insight: ETF-era cycles may never reach scorecard 7/10 because "
            "institutional accumulation absorbs the supply that would normally "
            "trigger retail capitulation. Triggers 1A/1B account for this."
        )

    st.divider()

    # === MACRO ROTATION TRACKER (equities → BTC) ===
    st.markdown("<div class='section-header'>Macro Rotation — equities → BTC timing (top-tier)</div>",
                 unsafe_allow_html=True)
    st.caption(
        "Top-tier rotation indicator using Druckenmiller/Howell/Alden/PTJ framework: "
        "HY credit spreads + VIX term structure + Kelly-criterion sizing + DCA pace + "
        "historical backtest validation. Verified at 2018 and 2022 BTC bottoms."
    )
    try:
        rot = cached_rotation()
        if rot and not rot.get("error"):
            # Color by phase
            phase_colors = {
                "PRE_ROTATION":  C["muted"],
                "WATCH":         C["neutral"],
                "ACTIVE":        C["bull"],
                "ACTIVE_DEEP":   C["bull"],
                "AGGRESSIVE":    C["deep_bull"],
                "COMPLETE":      C["deep_bull"],
            }
            phase_color = phase_colors.get(rot["phase_id"], C["accent"])
            spy = rot["spy"]; btc = rot["btc"]
            corr = rot.get("correlation") or {}
            liq = rot.get("liquidity") or {}

            # Big verdict banner
            _rot_cols = st.columns([3, 2])
            with _rot_cols[0]:
                st.markdown(
                    f"<div style='padding:18px 22px; border-radius:8px; "
                    f"border-left:6px solid {phase_color}; "
                    f"background:linear-gradient(90deg, {phase_color}22 0%, #1a1d24 100%);'>"
                    f"<div style='font-size:10px; color:#888; text-transform:uppercase; letter-spacing:1.5px;'>"
                    f"Rotation phase: {rot['phase_id'].replace('_', ' ')}</div>"
                    f"<div style='font-size:30px; font-weight:800; color:{phase_color}; line-height:1.1; margin:6px 0;'>"
                    f"{rot['action']}</div>"
                    f"<div style='font-size:14px; color:#ccc;'>{rot['rationale']}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if rot.get("notes"):
                    for n in rot["notes"]:
                        st.caption(f"• {n}")
            with _rot_cols[1]:
                st.markdown(metric_card(
                    "Recommended rotation %",
                    f"{rot['deploy_pct']}%",
                    "of equity stake → BTC",
                    phase_color,
                ), unsafe_allow_html=True)
                liq_phase = liq.get("phase", "?").replace("_", " ")
                liq_color = (C["bull"] if "EXPANS" in liq.get("phase", "")
                             else C["bear"] if "CONTRACT" in liq.get("phase", "")
                             else C["muted"])
                st.markdown(metric_card(
                    "Liquidity phase (Howell)",
                    liq_phase,
                    f"{liq.get('chg_30d_pct', 0):+.1f}% 30d",
                    liq_color,
                ), unsafe_allow_html=True)

            # Side-by-side asset comparison
            st.markdown("<div class='section-header'>Asset comparison</div>",
                        unsafe_allow_html=True)
            _ab_cols = st.columns(3)
            with _ab_cols[0]:
                spy_color = (C["bear"] if spy["drawdown_pct"] < -15
                             else C["neutral"] if spy["drawdown_pct"] < -5
                             else C["muted"])
                st.markdown(metric_card(
                    "SPY (S&P 500)",
                    f"${spy['current_price']:,.0f}",
                    f"{spy['drawdown_pct']:+.1f}% from peak | 30d: {spy['chg_30d_pct']:+.1f}%",
                    spy_color,
                ), unsafe_allow_html=True)
            with _ab_cols[1]:
                btc_color = (C["deep_bull"] if btc["drawdown_pct"] < -50
                             else C["bull"] if btc["drawdown_pct"] < -35
                             else C["neutral"])
                st.markdown(metric_card(
                    "BTC",
                    f"${btc['current_price']:,.0f}",
                    f"{btc['drawdown_pct']:+.1f}% from cycle 5 peak | 30d: {btc['chg_30d_pct']:+.1f}%",
                    btc_color,
                ), unsafe_allow_html=True)
            with _ab_cols[2]:
                cv = corr.get("corr_30d")
                if cv is not None:
                    corr_label = ("HIGH (moving together)" if cv > 0.6
                                  else "MODERATE" if cv > 0.3
                                  else "LOW (divergence)")
                    corr_color = C["bear"] if cv > 0.6 else C["bull"] if cv < 0.2 else C["neutral"]
                else:
                    corr_label = "n/a"
                    corr_color = C["muted"]
                st.markdown(metric_card(
                    "BTC-SPY 30d correlation",
                    f"{cv:.2f}" if cv is not None else "—",
                    corr_label,
                    corr_color,
                ), unsafe_allow_html=True)

            # === THIS WEEK'S EXECUTION PLAN (operator-only — hidden on the public view) ===
            if SHOW_PERSONAL:
              try:
                from core.btc_rotation_planner import weekly_rotation_plan, _load_log
                plan = weekly_rotation_plan(rot)
                log = _load_log()
                st.markdown("##### This week's execution plan")
                _ex_cols = st.columns([3, 2])
                with _ex_cols[0]:
                    st.markdown(
                        f"<div style='padding:16px 20px; background:{C['accent']}22; "
                        f"border-left:6px solid {C['accent']}; border-radius:6px;'>"
                        f"<div style='font-size:11px; color:#888; text-transform:uppercase;'>"
                        f"This week's action</div>"
                        f"<div style='font-size:18px; font-weight:600; color:#fff; margin:6px 0;'>"
                        f"{plan['this_week_action']}</div>"
                        f"<div style='font-size:12px; color:#aaa;'>"
                        f"Phase: {plan['rotation_phase']} | "
                        f"Tranche: ${plan['tranche_amount_nzd']:,.0f} NZD | "
                        f"Pace: {plan['pace_status']}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with _ex_cols[1]:
                    st.markdown(metric_card(
                        "Deployed so far",
                        f"${plan['deployed_so_far_nzd']:,.0f}",
                        f"{plan['deployed_pct']:.0f}% of plan ({plan['tranches_done']}/{plan['tranches_total']} tranches)",
                        C["bull"],
                    ), unsafe_allow_html=True)

                with st.expander("📋 Step-by-step execution checklist"):
                    for step in plan.get("execution_checklist", []):
                        st.markdown(f"- {step}")
                    st.markdown("---")
                    st.markdown("**Recommended sells (high-beta first):**")
                    for s in plan.get("sell_recommendations", []):
                        st.markdown(f"- **{s['category']}** ({s['examples']}) — {s['rationale']}")
                    st.markdown("**Recommended buys (BTC ETFs):**")
                    for b in plan.get("buy_recommendations", []):
                        st.markdown(f"- **{b['ticker']}** ({b['name']}, expense {b['expense_ratio']})")
                    st.info(
                        "💡 **Phase change alerts**: Crypto_rotation_check task runs daily 9am NZT. "
                        "You'll get an email the day the phase changes (WATCH → ACTIVE → etc)."
                    )

                # Recent trades log
                trades = log.get("trades", [])
                if trades:
                    with st.expander(f"📒 Rotation log — {len(trades)} trades executed"):
                        log_rows = []
                        for t in trades[-20:]:
                            log_rows.append({
                                "Date":     t.get("date", "?"),
                                "Sold":     f"${t.get('sell_nzd', 0):,.0f} {t.get('sell_ticker', '?')}",
                                "Bought":   f"${t.get('buy_nzd', 0):,.0f} {t.get('buy_ticker', '?')}",
                                "Notes":    t.get("notes", "")[:40],
                            })
                        st.dataframe(pd.DataFrame(log_rows), width='stretch',
                                      hide_index=True)
              except Exception as e:
                st.caption(f"Execution plan unavailable: {e}")

            # === NEW: Top-tier additions (HY spreads, VIX, Kelly, DCA) ===
            st.markdown("<div class='section-header'>Top-tier signals (Druckenmiller / PTJ / Howell)</div>",
                        unsafe_allow_html=True)
            _tier_cols = st.columns(4)
            with _tier_cols[0]:
                hy = rot.get("hy_credit_spreads") or {}
                if hy and not hy.get("error"):
                    hy_phase = hy.get("phase", "?")
                    hy_score = hy.get("score", 0)
                    hy_color = (C["bull"] if hy_score > 0.2
                                else C["bear"] if hy_score < -0.2
                                else C["muted"])
                    st.markdown(metric_card(
                        "HY credit spreads",
                        hy_phase.replace("_", " ")[:24],
                        f"score {hy_score:+.2f} — Druckenmiller's #1 signal",
                        hy_color,
                    ), unsafe_allow_html=True)
            with _tier_cols[1]:
                vix = rot.get("vix_term_structure") or {}
                if vix and not vix.get("error"):
                    vix_phase = vix.get("phase", "?")
                    vix_v = vix.get("vix", 0)
                    vix_ratio = vix.get("term_ratio", 1.0)
                    vix_score = vix.get("score", 0)
                    vix_color = (C["bull"] if vix_score > 0.3
                                 else C["bear"] if vix_score < -0.3
                                 else C["muted"])
                    st.markdown(metric_card(
                        "VIX term structure",
                        f"VIX {vix_v:.1f}, ratio {vix_ratio:.2f}",
                        vix_phase[:28],
                        vix_color,
                    ), unsafe_allow_html=True)
            with _tier_cols[2]:
                kelly = rot.get("kelly_details", {})
                base = kelly.get("base_pct", 0)
                kelly_pct = kelly.get("kelly_adjusted_pct", 0)
                vol_m = kelly.get("vol_multiplier", 1)
                dd_m = kelly.get("drawdown_multiplier", 1)
                sig_m = kelly.get("signal_multiplier", 1)
                st.markdown(metric_card(
                    "Kelly-sized deploy",
                    f"{kelly_pct}%",
                    f"Base {base}% × vol {vol_m:.2f} × dd {dd_m:.2f} × sig {sig_m:.2f}",
                    C["accent"],
                ), unsafe_allow_html=True)
            with _tier_cols[3]:
                dca = rot.get("dca", {})
                weeks = dca.get("weeks", "?")
                tranches = dca.get("tranches", "?")
                pct_per = dca.get("pct_per_tranche", "?")
                st.markdown(metric_card(
                    "DCA pace (Saylor-style)",
                    f"{weeks}wk / {tranches} tranches",
                    f"{pct_per}% per tranche — {dca.get('frequency', '?')}",
                    C["lth"],
                ), unsafe_allow_html=True)
            # Full DCA recommendation
            if dca:
                st.info(f"**Execution plan**: {dca.get('recommendation', '')}")

            # Confirming signals count
            cs = rot.get("confirming_signals", 0)
            if cs >= 3:
                st.success(f"🟢 {cs} confirming signals firing — high-conviction setup")
            elif cs >= 1:
                st.info(f"🟡 {cs} confirming signal(s) — proceed with caution")
            else:
                st.warning(f"⚠️ Only {cs} confirming signals — wait for more confirmation")

            # === Druckenmiller layer: valuation, yield curve, DXY ===
            st.markdown("<div class='section-header'>Druckenmiller layer — valuation + macro confirmation</div>",
                        unsafe_allow_html=True)
            _druck_cols = st.columns(3)
            with _druck_cols[0]:
                val = rot.get("earnings_valuation") or {}
                if val and not val.get("error"):
                    erp = val.get("equity_risk_premium_pp", 0)
                    pe = val.get("trailing_pe", 0)
                    val_color = (C["bull"] if erp < 0 else C["neutral"] if erp < 2 else C["bear"])
                    st.markdown(metric_card(
                        "SPY valuation (ERP)",
                        f"{erp:+.1f}pp",
                        f"P/E {pe:.0f} | E-yield {val.get('earnings_yield_pct', 0):.1f}% vs 10y {val.get('treasury_10y_pct', 0):.1f}%",
                        val_color,
                    ), unsafe_allow_html=True)
            with _druck_cols[1]:
                yc = rot.get("yield_curve") or {}
                if yc and not yc.get("error"):
                    yc_phase = yc.get("phase", "?")
                    yc_color = (C["bull"] if "RE_STEEPENING" in yc_phase
                                else C["bear"] if "INVERTED" in yc_phase
                                else C["muted"])
                    st.markdown(metric_card(
                        "Yield curve (2y10y)",
                        f"{yc.get('value', 0):+.2f}pp",
                        yc_phase[:30],
                        yc_color,
                    ), unsafe_allow_html=True)
            with _druck_cols[2]:
                dxy = rot.get("currency_dynamics") or {}
                if dxy and not dxy.get("error"):
                    dxy_phase = dxy.get("phase", "?")
                    dxy_color = (C["bull"] if "WEAK" in dxy_phase
                                 else C["bear"] if "STRONG" in dxy_phase
                                 else C["muted"])
                    st.markdown(metric_card(
                        "DXY (USD strength)",
                        f"{dxy.get('value', 0):.1f}",
                        f"{dxy.get('chg_30d_pct', 0):+.1f}% 30d — {dxy_phase[:20]}",
                        dxy_color,
                    ), unsafe_allow_html=True)

            # === RISK MANAGEMENT ===
            risk = rot.get("risk_management", {})
            if risk and SHOW_PERSONAL:  # operator-only — NZD sizing, hidden on public view
                st.markdown("<div class='section-header'>Risk management (PTJ rules)</div>",
                            unsafe_allow_html=True)
                _risk_cols = st.columns(4)
                with _risk_cols[0]:
                    st.markdown(metric_card(
                        "Deploy NOW",
                        f"${risk['deploy_nzd']:,.0f}",
                        f"NZD ({risk['actual_deploy_pct']}% of $130k stake)",
                        C["accent"],
                    ), unsafe_allow_html=True)
                with _risk_cols[1]:
                    st.markdown(metric_card(
                        "Reserve for BTC",
                        f"${risk['remaining_btc_capacity_nzd']:,.0f}",
                        "NZD for additional tranches (up to 50% cap)",
                        C["lth"],
                    ), unsafe_allow_html=True)
                with _risk_cols[2]:
                    st.markdown(metric_card(
                        "Cash buffer",
                        f"${risk['cash_buffer_nzd']:,.0f}",
                        f"NZD untouched ({risk['min_cash_buffer_pct']}% rule)",
                        C["muted"],
                    ), unsafe_allow_html=True)
                with _risk_cols[3]:
                    st.markdown(metric_card(
                        "Stop loss",
                        f"-{risk['stop_loss_pct']:.0f}%",
                        "from entry (VIX-adjusted)",
                        C["bear"],
                    ), unsafe_allow_html=True)
                st.caption(f"💡 {risk['note']}")
                st.caption(f"⚠️ Stops: {risk['suggested_stop_25pct']}. {risk['suggested_stop_35pct']}.")

            # === WHAT TO SELL ===
            with st.expander("🔄 What to sell from equities first (PTJ framework)"):
                sells = rot.get("what_to_sell") or {}
                if sells:
                    st.markdown("**Sell FIRST (high-beta / overvalued):**")
                    for s in sells.get("sell_first", []):
                        st.markdown(f"- **{s['category']}** — {s['rationale']}\n  *Examples: {s['examples']}*")
                    st.markdown("**Sell LAST (defensive ballast):**")
                    for s in sells.get("sell_last", []):
                        st.markdown(f"- **{s['category']}** — {s['rationale']}\n  *Examples: {s['examples']}*")
                    st.info(f"💡 {sells.get('tax_loss_harvest_first', '')}")

            # === NZ TAX CONSIDERATIONS ===
            tax = rot.get("nz_tax_considerations") or {}
            if tax:
                tax_color = "warning" if tax.get("over_fif_threshold") else "info"
                with st.expander("🇳🇿 NZ tax considerations (FIF regime)"):
                    if tax.get("over_fif_threshold"):
                        st.warning(f"⚠️ {tax.get('note', '')}")
                    else:
                        st.info(tax.get("note", ""))
                    st.markdown("**Considerations:**")
                    for desc, status in tax.get("considerations", []):
                        st.markdown(f"- **{desc}** — {status}")
                    st.markdown("**Action items:**")
                    for ai in tax.get("action_items", []):
                        st.markdown(f"- {ai}")

            # === HISTORICAL BACKTEST ===
            with st.expander("📊 Historical backtest — did this indicator work at past BTC bottoms?"):
                try:
                    from core.btc_macro_rotation import historical_backtest
                    bt = historical_backtest()
                    st.markdown("**Verification at known historical bottoms:**")
                    for period in bt.get("periods", []):
                        st.markdown(f"**{period['label']}** "
                                     f"— BTC peak ${period['btc_peak']:,.0f} → bottom ${period['btc_bottom']:,.0f}")
                        bt_rows = []
                        for t in period.get("tests", []):
                            bt_rows.append({
                                "Test point": t["label"],
                                "Date": t["date"],
                                "SPY price": f"${t['spy_price']:.0f}",
                                "BTC price": f"${t['btc_price']:,.0f}",
                                "SPY DD": f"{t['spy_dd']:+.1f}%",
                                "BTC DD": f"{t['btc_dd']:+.1f}%",
                                "Indicator phase": t["phase_signaled"],
                            })
                        if bt_rows:
                            st.dataframe(pd.DataFrame(bt_rows), width='stretch',
                                          hide_index=True)
                    st.caption(
                        "Each period shows what the indicator WOULD have said at key "
                        "dates leading up to and at the historical bottom. AGGRESSIVE or "
                        "ACTIVE at the bottom = indicator correctly identified the rotation."
                    )
                    if bt.get("summary_lines"):
                        for line in bt["summary_lines"]:
                            st.markdown(f"- ✓ {line}")
                except Exception as e:
                    st.caption(f"Backtest unavailable: {e}")

            # Phase guide
            with st.expander("How the 5 rotation phases work"):
                st.markdown(
                    "| Phase | When | Action | Deploy % |\n"
                    "|---|---|---|---|\n"
                    "| **PRE_ROTATION** | Both SPY + BTC elevated | Hold equities, wait | 0% |\n"
                    "| **WATCH** | SPY near peak + BTC discounted (-40% or worse) | Begin rotation | 20% |\n"
                    "| **ACTIVE** | SPY pulling back + BTC in bottom zone | Accelerate rotation | 40-60% |\n"
                    "| **AGGRESSIVE** | SPY correcting + BTC capitulation + liquidity turn | High-conviction | 75-90% |\n"
                    "| **COMPLETE** | BTC recovering, SPY still soft | Finish rotation | 100% |\n"
                )
                st.caption(
                    "Based on the Howell/Alden framework: global liquidity drives "
                    "everything. BTC bottoms first (highest beta sensor), equities follow "
                    "within 1-4 weeks. Exception: crypto-contagion years (e.g., 2022 FTX), "
                    "when BTC LAGS equities at the bottom."
                )
        else:
            st.caption(f"Rotation tracker: {rot.get('error', 'unavailable')}")
    except Exception as e:
        st.caption(f"Rotation tracker unavailable: {e}")

    # === Macro Drivers (ETF flows, Net Liquidity, Stablecoins, Hash price) ===
    st.markdown("<div class='section-header'>Macro Drivers — paid-tier-equivalent free signals</div>",
                 unsafe_allow_html=True)
    st.caption("18-signal premium-free layer: ETF flows + stablecoins + Net Liquidity + Deribit Greeks + 14 more")
    _md_cols = st.columns(4)
    _macro_signals = [
        ("etf_flows",         "ETF flows (5d)",      "$M institutional flow",   "M"),
        ("stablecoin_supply", "Stablecoin supply",   "$B fresh liquidity",      "B"),
        ("net_liquidity",     "Net Liquidity",       "Bloomberg $20K/yr metric", "T"),
        ("hash_price",        "Hash price",          "$/TH/day miner econ",     ""),
    ]
    for col, (sig_name, label, sub, unit) in zip(_md_cols, _macro_signals):
        sig = _find_signal(sig_name)
        with col:
            if sig is None or sig.get("error"):
                st.markdown(metric_card(label, "—", "unavailable", C["muted"]),
                              unsafe_allow_html=True)
                continue
            score = sig.get("score", 0)
            val = sig.get("value")
            acc = (C["deep_bull"] if score > 0.5 else C["bull"] if score > 0.2
                   else C["neutral"] if score > -0.2 else C["bear"] if score > -0.5
                   else C["deep_bear"])
            if sig_name == "etf_flows":
                disp_val = f"${sig.get('last_5d_M', 0):+,.0f}M"
                disp_sub = f"5d total ({sig.get('last_30d_M', 0):+,.0f}M 30d)"
            elif sig_name == "stablecoin_supply":
                disp_val = f"${val/1e9:.0f}B"
                disp_sub = f"30d {sig.get('chg_30d_pct', 0):+.1f}%"
            elif sig_name == "net_liquidity":
                disp_val = f"${val:.2f}T"
                disp_sub = f"30d {sig.get('chg_30d_pct', 0):+.1f}%"
            elif sig_name == "hash_price":
                disp_val = f"${val:.3f}"
                disp_sub = "miner stress" if score > 0.3 else "neutral"
            else:
                disp_val = str(val)[:14]
                disp_sub = sub
            st.markdown(metric_card(label, disp_val, disp_sub, acc),
                          unsafe_allow_html=True)

    # === Scorecard + Realized Cap thermometer ===
    _ov = st.columns([3, 2])

    with _ov[0]:
        st.markdown("<div class='section-header'>Bottom Confirmation Scorecard</div>",
                     unsafe_allow_html=True)
        st.caption(
            "Hard criteria — historically required 6+ for actual bottom. "
            "Soft signals (Reserve Risk, halving forward) fire on projection; "
            "these confirm reality."
        )
        _sc_crit = sc.get("criteria", [])
        _sc_met = sum(1 for c in _sc_crit if c.get("met"))
        _sc_tot = len(_sc_crit)
        _sc_col = (C["deep_bull"] if _sc_met >= 6 else C["bull"] if _sc_met >= 4
                   else C["neutral"] if _sc_met >= 2 else C["muted"])
        st.markdown(
            f"<div style='font-size:22px; font-weight:800; color:{_sc_col};'>{_sc_met}"
            f"<span style='font-size:14px; color:#888;'> / {_sc_tot} hard criteria met</span></div>"
            f"{_seg_bar(_sc_met, _sc_tot, _sc_col)}",
            unsafe_allow_html=True)
        st.markdown(_crit_tiles(_sc_crit, _sc_col), unsafe_allow_html=True)

    with _ov[1]:
        st.markdown("<div class='section-header'>Realized Cap drawdown</div>",
                     unsafe_allow_html=True)
        st.caption("THE bottom indicator. Need -15% min for historical band.")
        if rcd and not rcd.get("error"):
            current_dd = rcd["current_drawdown_pct"]
            _tf = go.Figure()
            _tf.add_shape(type="rect", x0=-30, x1=0, y0=0.4, y1=0.6,
                          fillcolor="#2a2d34", line=dict(width=0))
            bar_color = (C["deep_bull"] if current_dd < -25 else
                         C["bull"] if current_dd < -15 else
                         C["neutral"] if current_dd < -10 else C["bear"])
            _tf.add_shape(type="rect", x0=current_dd, x1=0, y0=0.4, y1=0.6,
                          fillcolor=bar_color, opacity=0.7, line=dict(width=0))
            for x, lbl, col in [
                (-10, "Bear", C["muted"]),
                (-15, "Bottom entry", C["bull"]),
                (-20, "Bottom mid", C["deep_bull"]),
                (-25, "Bottom deep", C["deep_bull"]),
            ]:
                _tf.add_vline(x=x, line_color=col, line_width=1, line_dash="dot",
                              annotation_text=lbl, annotation_position="top",
                              annotation_font_size=9, annotation_font_color=col)
            _tf.add_vline(x=current_dd, line_color="#fff", line_width=3,
                          annotation_text=f"NOW {current_dd:+.1f}%",
                          annotation_position="bottom", annotation_font_size=12)
            # Override margin to be tighter for the thermometer (no full **CHART_LAYOUT
            # spread because we want a custom margin)
            _tf.update_layout(
                plot_bgcolor=CHART_LAYOUT["plot_bgcolor"],
                paper_bgcolor=CHART_LAYOUT["paper_bgcolor"],
                font=CHART_LAYOUT["font"],
                height=170,
                xaxis=dict(title="Drawdown (%) — ◀ deeper = better entry", range=[-32, 2],
                           gridcolor="#2a2d34"),
                yaxis=dict(visible=False),
                margin=dict(l=20, r=20, t=30, b=40),
            )
            st.plotly_chart(_tf, width='stretch')

            # Quick cost-basis summary
            cb_html = ""
            if rp and not rp.get("error"):
                cb_html += f"<div style='font-size:13px; color:#aaa;'>LTH cost basis: <b style='color:{C['lth']};'>${rp['value']:,.0f}</b></div>"
            if sth and not sth.get("error"):
                cb_html += f"<div style='font-size:13px; color:#aaa;'>STH cost basis: <b style='color:{C['sth']};'>${sth['value']:,.0f}</b> ({sth['price_vs_sth_pct']:+.1f}%)</div>"
            if pdb and not pdb.get("error"):
                cb_html += f"<div style='font-size:13px; color:#aaa;'>EV bottom: <b style='color:{C['bull']};'>${pdb['expected_value_price']:,.0f}</b> ({pdb['expected_value_chg_pct']:+.1f}%)</div>"
            if cb_html:
                st.markdown(cb_html, unsafe_allow_html=True)
with tab_simple:
    # ═══════════════════════════════════════════════════════════════════
    # 🔰 SIMPLETON SUMMARY — the whole dashboard in plain English
    # ═══════════════════════════════════════════════════════════════════
    st.markdown("## 🔰 Simpleton Summary")
    st.caption("The whole dashboard boiled down to plain English — no jargon, no market-speak. "
               "Everything here updates by itself.")

    def _sg(_k):
        try:
            from core.dashboard_cache import get_cached as _gcx
            return _gcx(_k) or {}
        except Exception:
            return {}

    _cd_s = _sg("cycle_dials").get("summary", {})
    _conv_s = _sg("date_predictions").get("convergence", {})
    _tsc_s = _sg("top_scorecard").get("scorecard", {})
    _rt_s = _sg("rotation_trigger")
    _olq_s = _sg("equity_olson")
    _sem_s = _sg("equity_semis")
    _p = btc_price or 0

    # --- Is Bitcoin cheap? (cycle gauges) ---
    _hl = _cd_s.get("headline", "")
    _nb = _cd_s.get("n_buy", 0)
    _nt = _cd_s.get("n_total", 7) or 7
    _is_cheap = "ACCUMULATION" in _hl
    _is_exp = "DISTRIBUTION" in _hl
    if _is_cheap:
        _btc_status, _btc_sub, _btc_col = ("Cheap on long-term value",
            f"{_nb} of {_nt} cycle gauges say 'good value' — but the bottom (the best buy) may still be lower", "#22c55e")
    elif _is_exp:
        _btc_status, _btc_sub, _btc_col = ("No — looks expensive",
            "cycle gauges are flashing caution", "#ef4444")
    else:
        _btc_status, _btc_sub, _btc_col = ("Around fair value",
            f"{_nb} of {_nt} cycle gauges say cheap", "#f0b90b")

    # --- Bottom timing ---
    _evd = None
    _ev = _conv_s.get("ev_date", "2026-10-23")
    try:
        _evd = datetime.strptime(_ev, "%Y-%m-%d").date()
        _days_to = (_evd - datetime.now(timezone.utc).date()).days
        _ev_my = _evd.strftime("%B %Y")
    except Exception:
        _days_to, _ev_my = None, "late 2026"

    # --- Stocks (shares) state ---
    _tn = _tsc_s.get("n_met", 0)
    _ttot = _tsc_s.get("n_total", 10) or 10
    _olt = _olq_s.get("tier", "")
    _semt = _sem_s.get("tier", "")
    if _tn <= 2 and _olt in ("SAFE", ""):
        _stk_status, _stk_col = "Calm, near highs", "#22c55e"
        _stk_sub = f"only {_tn} of {_ttot} 'toppy' warning signs are on"
    elif _tn >= 5 or _olt in ("DANGER", "EXIT", "CAUTION"):
        _stk_status, _stk_col = "Looking stretched", "#ef4444"
        _stk_sub = f"{_tn} of {_ttot} 'toppy' warning signs are on"
    else:
        _stk_status, _stk_col = "Mostly calm", "#f0b90b"
        _stk_sub = (f"{_tn} of {_ttot} warning signs on; chip stocks {_semt.lower()}"
                    if _semt else f"{_tn} of {_ttot} warning signs on")

    # --- The plan (rotation trigger) ---
    _fire = _rt_s.get("firing_paths", []) or []
    if not _fire:
        _plan_status, _plan_sub, _plan_col = "WAITING", "Nothing to do yet — sit tight", "#22c55e"
    elif len(_fire) >= 2:
        _plan_status, _plan_sub, _plan_col = ("TIME TO ACT",
            "Signal to move shares → Bitcoin has fired", "#ef4444")
    else:
        _plan_status, _plan_sub, _plan_col = "GETTING READY", "Early warning — start preparing", "#f0b90b"

    # === HERO sentence ===
    _days_txt = f" — about <b>{_days_to} days</b> away" if _days_to and _days_to > 0 else ""
    _hero = (
        f"Bitcoin is <b>${_p:,.0f}</b> right now and looks "
        f"<b style='color:{_btc_col};'>{_btc_status.lower()}</b>. The big opportunity — the cycle "
        f"<b>bottom</b> (the best time to buy heavily) — is expected around <b>$52,000–$57,000</b>, "
        f"most likely <b>{_ev_my}</b>{_days_txt}. Meanwhile shares (US stocks) look "
        f"<b style='color:{_stk_col};'>{_stk_status.lower()}</b>. The plan is simple: shift money from "
        f"shares into Bitcoin when shares start to wobble — and right now the system says "
        f"<b style='color:{_plan_col};'>{_plan_status}</b>."
    )
    st.markdown(
        f"<div style='padding:18px 22px; border-radius:10px; background:#13161c; "
        f"border-left:6px solid #f0b90b; font-size:16px; line-height:1.65; color:#e8e8e8;'>"
        f"{_hero}</div>", unsafe_allow_html=True)
    st.write("")

    # === 📜 What this usually means (historical context for the plan) ===
    try:
        from core.rotation_trigger import leadtime_context as _ltc2
        _lc2 = _ltc2(_plan_status)
        _eg = ("For example: in the slow 2022 decline this kind of signal got you out near the "
               "top and side-stepped a further ~31% fall over the next 10 months. In the fast "
               "2020 crash it gave far less room — which is what the separate safety-net (the "
               "tail hedge) is there for.")
        _body2 = (f"<div style='font-size:13px; color:#ddd; line-height:1.6;'>{_lc2['plain']}</div>"
                  f"<div style='font-size:12px; color:#aaa; line-height:1.55; margin-top:7px;'>{_eg}</div>")
        if _plan_status in ("TIME TO ACT", "GETTING READY"):
            st.markdown(
                f"<div style='padding:14px 18px; border-radius:10px; background:#13161c; "
                f"border-left:5px solid {_plan_col}; margin:2px 0 10px;'>"
                f"<div style='font-size:11px; color:#fff; text-transform:uppercase; "
                f"letter-spacing:1.2px; font-weight:700; margin-bottom:6px;'>"
                f"📜 What this usually means</div>{_body2}</div>", unsafe_allow_html=True)
        else:
            with st.expander("📜 What happens when the plan fires?"):
                st.markdown(_body2, unsafe_allow_html=True)
    except Exception:
        pass

    # === 📅 Today's update — the daily 6am plain-English brief ===
    st.markdown("#### 📅 Today's update — what changed in the last 24 hours")
    _brief = None
    try:
        from pathlib import Path as _BP
        import json as _bjson
        _bpath = _BP(__file__).resolve().parent / ".simpleton_daily_brief.json"
        if _bpath.exists():
            _brief = _bjson.loads(_bpath.read_text(encoding="utf-8"))
    except Exception:
        _brief = None
    if _brief and _brief.get("lines"):
        import re as _bre

        def _b2h(_s):
            return _bre.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", _s or "")
        _blines = "".join(
            f"<div style='font-size:13px; color:#ddd; margin-top:6px; line-height:1.5;'>{_b2h(_ln)}</div>"
            for _ln in _brief.get("lines", []))
        st.markdown(
            f"<div style='padding:16px 20px; border-radius:10px; background:#13161c; "
            f"border-left:6px solid #4a90e2;'>"
            f"<div style='font-size:10px; color:#888; text-transform:uppercase; letter-spacing:1.2px;'>"
            f"as of {_brief.get('date_friendly', _brief.get('date', ''))} · "
            f"updated {str(_brief.get('generated_local',''))[11:16]} NZ · refreshes through the day</div>"
            f"<div style='font-size:16px; color:#fff; font-weight:700; margin:5px 0 2px;'>"
            f"{_b2h(_brief.get('summary', ''))}</div>"
            f"{_blines}</div>",
            unsafe_allow_html=True)
    else:
        st.caption("Today's plain-English update will appear here after the next refresh cycle.")
    st.write("")

    # === Four traffic-light cards ===
    def _scard(emoji, title, status, sub, color):
        return (
            f"<div style='padding:14px 16px; border-radius:8px; background:#1a1d24; "
            f"border-left:5px solid {color}; height:150px;'>"
            f"<div style='font-size:24px;'>{emoji}</div>"
            f"<div style='font-size:11px; color:#888; text-transform:uppercase; "
            f"letter-spacing:.5px; margin-top:6px;'>{title}</div>"
            f"<div style='font-size:17px; font-weight:700; color:{color}; margin:4px 0;'>{status}</div>"
            f"<div style='font-size:12px; color:#aaa; line-height:1.4;'>{sub}</div></div>")

    _r1 = st.columns(2)
    with _r1[0]:
        st.markdown(_scard("🪙", "Is Bitcoin cheap?", _btc_status, _btc_sub, _btc_col),
                    unsafe_allow_html=True)
    with _r1[1]:
        _b_sub = "most likely " + _ev_my + (f" (~{_days_to} days)" if _days_to and _days_to > 0 else "")
        st.markdown(_scard("🎯", "Best time to buy (the bottom)", "$52k–$57k", _b_sub, "#f0b90b"),
                    unsafe_allow_html=True)
    st.write("")
    _r2 = st.columns(2)
    with _r2[0]:
        st.markdown(_scard("📉", "How are shares (stocks)?", _stk_status, _stk_sub, _stk_col),
                    unsafe_allow_html=True)
    with _r2[1]:
        st.markdown(_scard("🧭", "What's the plan right now?", _plan_status, _plan_sub, _plan_col),
                    unsafe_allow_html=True)
    st.write("")

    # === Two simple visuals ===
    _v1, _v2 = st.columns(2)
    with _v1:
        st.markdown("**💰 VALUE — how cheap is Bitcoin?**")
        _buy_pct = round((_nb / _nt) * 100) if _nt else 0
        _cg = go.Figure(go.Indicator(
            mode="gauge+number", value=_buy_pct,
            number={"suffix": "%", "font": {"size": 26, "color": "#fff"}},
            title={"text": "of cycle gauges say 'cheap'", "font": {"size": 11, "color": "#888"}},
            gauge={
                "axis": {"range": [0, 100], "tickvals": [0, 50, 100],
                         "ticktext": ["pricey", "fair", "cheap"],
                         "tickfont": {"size": 10, "color": "#888"}},
                "bar": {"color": "rgba(255,255,255,0.9)", "thickness": 0.25},
                "bgcolor": "#0e1117", "borderwidth": 0,
                "steps": [
                    {"range": [0, 33], "color": "rgba(239,68,68,0.55)"},
                    {"range": [33, 66], "color": "rgba(240,185,11,0.40)"},
                    {"range": [66, 100], "color": "rgba(34,197,94,0.55)"}]}))
        _cg.update_layout(paper_bgcolor="#0e1117", height=215,
                          margin=dict(l=20, r=20, t=32, b=8), font=dict(color="#d4d4d4"))
        st.plotly_chart(_cg, width='stretch',
                        config={"displayModeBar": False, "displaylogo": False})
        st.caption("Further right = better value (more of Bitcoin's cycle gauges say 'cheap').")
    with _v2:
        st.markdown("**⏳ TIMING — countdown to the bottom**")
        _prog = 0.5
        try:
            if _evd:
                _tot = max(1, (_evd - CYCLE5_PEAK_DATE).days)
                _elap = (datetime.now(timezone.utc).date() - CYCLE5_PEAK_DATE).days
                _prog = min(1.0, max(0.0, _elap / _tot))
        except Exception:
            _prog = 0.5
        if _days_to and _days_to > 0:
            st.markdown(
                f"<div style='font-size:42px; font-weight:800; color:#f0b90b; margin:6px 0 0;'>"
                f"≈ {_days_to} days</div>"
                f"<div style='color:#aaa; font-size:13px;'>until the most-likely bottom (~{_ev_my})</div>",
                unsafe_allow_html=True)
        else:
            st.markdown("<div style='font-size:30px; font-weight:800; color:#f0b90b;'>"
                        "Bottom window is open</div>", unsafe_allow_html=True)
        st.write("")
        st.markdown(_seg_bar(round(_prog * 12), 12, "#f0b90b", height=12), unsafe_allow_html=True)
        st.caption(f"About {round(_prog * 100)}% of the way from the last peak to the expected bottom.")

    st.divider()

    # === What each tab is, in one line ===
    st.markdown("#### 🗂️ The other tabs, in one line each")
    st.markdown(
        "- **📡 Signals** — the live engine: unified verdict, cockpit gauges and every scorecard.\n"
        "- **🚪 Playbook** — what to actually *do*: rotation trigger, deploy plan, checklist, NZ tax.\n"
        "- **🔬 Research** — the library: backtests, validation math, chart suites and guru feeds.\n")

    # === Glossary ===
    with st.expander("📖 What the words mean (plain-English glossary)"):
        st.markdown(
            "- **The bottom** — the lowest price in Bitcoin's roughly 4-year boom/bust cycle; "
            "historically the best time to buy.\n"
            "- **Accumulation zone** — a cheap stretch where steady buying has paid off before.\n"
            "- **Rotation** — the plan to move money *out of* shares and *into* Bitcoin near the bottom.\n"
            "- **Halving** — a built-in event every ~4 years that cuts new Bitcoin supply; "
            "bottoms have historically come ~18 months after.\n"
            "- **Off its highs / drawdown** — how far something has fallen from its peak.\n"
            "- **Chip stocks (semiconductors)** — they often wobble *before* the wider market, "
            "so they're an early warning.\n"
            "- **Cycle gauges** — a set of tried-and-tested 'is Bitcoin cheap or expensive?' meters.\n")

    st.caption("⚠️ General information from one person's model — not financial advice. "
               "Crypto and shares can fall sharply. Always do your own research.")




# ─────────────────────────────────────────────────────────────────
# CYCLE MATH TAB — halving clock, probability distribution, timeline
# ─────────────────────────────────────────────────────────────────
with tab_cycle:
    # ═══════════════════════════════════════════════════════════════════
    # 📈 CHARTS TAB — cycle timeline (top), then fast-read dials + heatmap,
    # then collapsed detail drawers
    # ═══════════════════════════════════════════════════════════════════
    st.markdown("### 📈 Charts — cycle timeline, fast-read dials + full detail below")

    # ──────────────────────────────────────────────────────────────
    # 📈 CYCLE TIMELINE — prominent headline visual + Olson bottom figures
    # ──────────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>📈 Cycle Timeline — where we are and the bottom ahead</div>",
                unsafe_allow_html=True)

    def _fmt_date(_s, _fmt):
        try:
            return datetime.strptime(_s, "%Y-%m-%d").strftime(_fmt)
        except Exception:
            return _s

    # Jesse Olson "W" double-bottom target + 5-method date convergence (from cache)
    _OLSON_LO, _OLSON_HI = 52_000, 57_000
    _bot_lo_d, _bot_hi_d, _bot_ev_d = "2026-09-15", "2027-01-27", "2026-10-23"
    try:
        _conv = (cached_date_predictions() or {}).get("convergence", {}) or {}
        _bot_lo_d = _conv.get("earliest_estimate") or _bot_lo_d
        _bot_hi_d = _conv.get("latest_estimate") or _bot_hi_d
        _bot_ev_d = _conv.get("ev_date") or _bot_ev_d
    except Exception:
        pass

    _bf = st.columns(3)
    with _bf[0]:
        st.markdown(metric_card("Olson bottom target", "$52–57k",
                                "W-pattern double bottom", C["bull"]),
                    unsafe_allow_html=True)
    with _bf[1]:
        st.markdown(metric_card("Most-likely date", _fmt_date(_bot_ev_d, "%b %d, %Y"),
                                "weighted EV of 5 methods", C["accent"]),
                    unsafe_allow_html=True)
    with _bf[2]:
        st.markdown(metric_card("Rotation buy window",
                                f"{_fmt_date(_bot_lo_d, '%b %Y')} – {_fmt_date(_bot_hi_d, '%b %Y')}",
                                "earliest → latest estimate", C["lth"]),
                    unsafe_allow_html=True)

    _today_tl = datetime.now(timezone.utc).date()
    _h4_tl = datetime(2024, 4, 20).date()
    _h5_tl = datetime(2028, 4, 20).date()
    try:
        _bot_date_tl = datetime.strptime(_bot_ev_d, "%Y-%m-%d").date()
    except Exception:
        _bot_date_tl = _h4_tl + timedelta(days=MEAN_DAYS_TO_BOTTOM)
    _peak6_date_tl = _h5_tl + timedelta(days=MEAN_DAYS_TO_PEAK)
    _c6_target_tl = int(CYCLE5_PEAK_PRICE * 1.6)
    _olson_mid = (_OLSON_LO + _OLSON_HI) // 2

    _tl_pts = [
        ("Halving 4", _h4_tl, 64000, "#f0b90b"),
        ("Cycle 5 PEAK", CYCLE5_PEAK_DATE, CYCLE5_PEAK_PRICE, C["bear"]),
        ("TODAY", _today_tl, btc_price, "#ffffff"),
        ("Cycle 5 BOTTOM<br>$52–57k (Olson)", _bot_date_tl, _olson_mid, C["bull"]),
        ("Halving 5", _h5_tl, 110000, "#f0b90b"),
        ("Cycle 6 PEAK", _peak6_date_tl, _c6_target_tl, C["bear"]),
    ]
    _tlf2 = go.Figure()
    _tlf2.add_hrect(y0=_OLSON_LO, y1=_OLSON_HI, fillcolor="rgba(34,197,94,0.07)", line_width=0)
    try:
        _tlf2.add_vrect(
            x0=datetime.strptime(_bot_lo_d, "%Y-%m-%d").date(),
            x1=datetime.strptime(_bot_hi_d, "%Y-%m-%d").date(),
            fillcolor="rgba(34,197,94,0.12)", line_width=0,
            annotation_text="🎯 rotation buy window", annotation_position="top left",
            annotation_font=dict(size=10, color="#22c55e"))
    except Exception:
        pass
    _tlf2.add_trace(go.Scatter(
        x=[p[1] for p in _tl_pts], y=[p[2] for p in _tl_pts],
        mode="lines+markers+text",
        line=dict(color="#666", width=2, dash="dot"),
        marker=dict(size=[12, 14, 18, 16, 12, 14],
                    color=[p[3] for p in _tl_pts], line=dict(color="white", width=2)),
        text=[f"{p[0]}<br>${p[2]:,.0f}" for p in _tl_pts],
        textposition="top center", textfont=dict(size=11, color=C["text"]),
        hovertemplate="<b>%{text}</b><br>%{x}<extra></extra>", showlegend=False,
    ))
    _tlf2.update_layout(**CHART_LAYOUT, height=430,
                        yaxis=dict(type="log", title="Price (USD, log)", gridcolor="#2a2d34"),
                        xaxis=dict(gridcolor="#2a2d34"))
    st.plotly_chart(_tlf2, width='stretch',
                    config={"displayModeBar": False, "displaylogo": False})
    st.caption(
        "**Bottom expectation (Jesse Olson):** W-pattern double-bottom target **$52–57k**, "
        "lining up with the LTH cost-basis floor (~$53–55k). Five independent date methods "
        f"converge **{_fmt_date(_bot_lo_d, '%b %Y')} → {_fmt_date(_bot_hi_d, '%b %Y')}** "
        f"(weighted EV **{_fmt_date(_bot_ev_d, '%b %d, %Y')}**); halving + 900d lands Oct 7, 2026. "
        f"Today ${btc_price:,.0f}."
    )

    try:
        from core.dashboard_cache import get_cached as _gcd
        _cd = _gcd("cycle_dials")
        if not _cd:
            from core.btc_cycle_dials import all_cycle_dials
            _cd = all_cycle_dials()
        _cdsum = _cd.get("summary", {}) or {}
        _cddials = _cd.get("dials", {}) or {}

        _hl = _cdsum.get("headline", "?")
        _hlc = _cdsum.get("head_color", "#888")
        st.markdown(
            f"<div style='padding:14px 18px; border-radius:10px; background:#13161c; "
            f"border:2px solid {_hlc}; margin-bottom:12px; display:flex; "
            f"justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;'>"
            f"<div><span style='font-size:10px; color:#888; text-transform:uppercase; "
            f"letter-spacing:1.5px;'>Cycle dials — fast read</span><br>"
            f"<span style='font-size:24px; font-weight:800; color:{_hlc};'>{_hl}</span></div>"
            f"<div style='font-size:13px; color:#ccc; text-align:right;'>"
            f"<span style='color:#22c55e; font-weight:700;'>{_cdsum.get('n_buy',0)}</span> buy"
            f"&nbsp;·&nbsp;<span style='color:#f0b90b; font-weight:700;'>{_cdsum.get('n_neutral',0)}</span> neutral"
            f"&nbsp;·&nbsp;<span style='color:#ef4444; font-weight:700;'>{_cdsum.get('n_sell',0)}</span> sell"
            f"<br><span style='font-size:10px; color:#888;'>of {_cdsum.get('n_total',0)} indicators</span>"
            f"</div></div>", unsafe_allow_html=True,
        )

        # Four least-overlapping reads up front; the price-vs-baseline trio
        # (golden ratio / 2-yr MA / log-reg all move together) goes in a drawer.
        _primary = ["risk_index", "mvrv", "nupl", "mayer"]
        _secondary = ["golden_ratio", "two_year_ma", "log_regression"]
        _cfg = {"displayModeBar": False, "scrollZoom": False,
                "doubleClick": False, "displaylogo": False}
        _pres = [k for k in _primary if k in _cddials]
        if _pres:
            _cols = st.columns(len(_pres))
            for _c, _k in zip(_cols, _pres):
                with _c:
                    _fig = _cddials[_k].get("fig")
                    if _fig:
                        st.plotly_chart(_fig, width='stretch', config=_cfg)
        st.caption(
            "Each needle is the current value; coloured bands are the cycle zones "
            "(🟢 accumulation/bottom → 🔴 distribution/top). The headline above counts "
            "all indicators — these four are the least-overlapping reads."
        )
        _pres_sec = [k for k in _secondary if k in _cddials]
        if _pres_sec:
            with st.expander("More cycle gauges — golden ratio · 2-yr MA · log-regression"):
                _scols = st.columns(len(_pres_sec))
                for _c, _k in zip(_scols, _pres_sec):
                    with _c:
                        _fig = _cddials[_k].get("fig")
                        if _fig:
                            st.plotly_chart(_fig, width='stretch', config=_cfg)
    except Exception as _cde:
        st.caption(f"Cycle dials — temporarily unavailable")

    # ═══════════════════════════════════════════════════════════════════
    # 🟩 SIGNAL HEATMAP — every bottom/top criterion as a colored tile.
    # A "sea of green/red" scan: green = bottom/buy signal firing,
    # red = top/sell signal firing, grey = dormant.
    # ═══════════════════════════════════════════════════════════════════
    try:
        from core.dashboard_cache import get_cached as _gch
        _hb = _gch("btc_native_bottom_scorecard") or {}
        _ht = _gch("btc_native_top_scorecard") or {}

        def _tiles(criteria, fired_color, fired_emoji, dormant_label):
            html = ""
            for c in criteria:
                met = bool(c.get("met"))
                label = (c.get("label") or "?")
                # strip leading "NN. " numbering for compactness
                import re as _re
                label = _re.sub(r"^\s*\d+\.\s*", "", label)
                label = label.split("(")[0].strip()[:22]
                bg = fired_color if met else "rgba(255,255,255,0.04)"
                br = fired_color if met else "#2a2d36"
                fg = "#fff" if met else "#777"
                dot = fired_emoji if met else "○"
                html += (
                    f"<div style='flex:1 1 120px; min-width:110px; max-width:170px; "
                    f"padding:8px 10px; border-radius:7px; background:{bg}; "
                    f"border:1px solid {br};'>"
                    f"<div style='font-size:13px;'>{dot} "
                    f"<span style='font-size:10px; color:{fg};'>{label}</span></div>"
                    f"</div>"
                )
            return html

        _bot_crit = _hb.get("criteria", []) or []
        _top_crit = _ht.get("criteria", []) or []
        _bot_n = sum(1 for c in _bot_crit if c.get("met"))
        _top_n = sum(1 for c in _top_crit if c.get("met"))

        if _bot_crit or _top_crit:
            st.markdown(
                f"<div class='section-header' style='margin-top:14px;'>"
                f"🟩 Signal Heatmap — every cycle signal at a glance</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='font-size:11px; color:#888; margin-bottom:8px;'>"
                f"<b style='color:#22c55e;'>{_bot_n}/{len(_bot_crit)}</b> bottom-buy signals firing "
                f"&nbsp;·&nbsp; <b style='color:#ef4444;'>{_top_n}/{len(_top_crit)}</b> top-sell signals firing "
                f"&nbsp;·&nbsp; green = bullish firing, red = bearish firing, grey = dormant</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div style='font-size:10px; color:#22c55e; text-transform:uppercase; "
                "letter-spacing:1px; margin:6px 0 4px 0;'>🔺 BTC Bottom (buy) signals</div>"
                "<div style='display:flex; flex-wrap:wrap; gap:6px;'>"
                + _tiles(_bot_crit, "rgba(34,197,94,0.30)", "🟢", "dormant")
                + "</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div style='font-size:10px; color:#ef4444; text-transform:uppercase; "
                "letter-spacing:1px; margin:12px 0 4px 0;'>🔻 BTC Top (sell) signals</div>"
                "<div style='display:flex; flex-wrap:wrap; gap:6px;'>"
                + _tiles(_top_crit, "rgba(239,68,68,0.30)", "🔴", "dormant")
                + "</div>",
                unsafe_allow_html=True,
            )
    except Exception as _hme:
        st.caption(f"Signal heatmap — temporarily unavailable")

    st.divider()

    st.caption(
        "📂 Full detail below — each section is the numeric backing for the dials "
        "and heatmap above. Expand only what you want to dig into."
    )

    with st.expander("📅 Bottom-date convergence — 4 methods", expanded=False):
        # === BOTTOM DATE CONVERGENCE (4 methods combined) ===
        try:
            _dp = cached_date_predictions()
            _bdc = _dp.get("convergence", {})
            if _bdc and not _bdc.get("error"):
                st.markdown("<div class='section-header'>Bottom Date Convergence — 4 methods combined</div>",
                             unsafe_allow_html=True)
                st.caption(
                    "Combines halving math + cycle 4 analog + indicator extrapolation + "
                    "probability EV. Shows weighted EV date + earliest/latest spread."
                )
                _cv_cols = st.columns([3, 2])
                with _cv_cols[0]:
                    ev_date = _bdc.get("ev_date", "?")
                    earliest = _bdc.get("earliest_estimate", "?")
                    latest = _bdc.get("latest_estimate", "?")
                    spread = _bdc.get("spread_days", 0)
                    st.markdown(
                        f"<div style='padding:14px 18px; background:#1a1d24; "
                        f"border-left:6px solid {C['accent']}; border-radius:6px;'>"
                        f"<div style='font-size:10px; color:#888; text-transform:uppercase;'>"
                        f"Weighted EV bottom date</div>"
                        f"<div style='font-size:24px; font-weight:700; color:#fff; line-height:1;'>"
                        f"{ev_date}</div>"
                        f"<div style='font-size:12px; color:#aaa; margin-top:6px;'>"
                        f"Range: {earliest} → {latest} ({spread} days spread)</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with _cv_cols[1]:
                    # Mini table of each method's estimate
                    method_rows = []
                    for est in _bdc.get("estimates", []):
                        method_rows.append({
                            "Method": est["method"].replace("_", " "),
                            "Date": est["date"],
                            "Weight": f"{est['weight']:.1f}",
                        })
                    st.dataframe(pd.DataFrame(method_rows), width='stretch',
                                  hide_index=True, height=180)
                st.caption(_bdc.get("summary", ""))

            # === INDICATOR EXTRAPOLATION ===
            _ie = _dp.get("extrapolation", {})
            if _ie and _ie.get("indicators"):
                st.markdown("<div class='section-header'>Indicator Extrapolation — when will each bottom signal fire?</div>",
                             unsafe_allow_html=True)
                st.caption(
                    "Linear extrapolation from current 30d trajectory. Tells you "
                    "approximately WHEN each bottom indicator will hit its threshold."
                )
                ie_rows = []
                for name, ind in _ie.get("indicators", {}).items():
                    if isinstance(ind, dict) and not ind.get("error"):
                        ie_rows.append({
                            "Indicator": name.replace("_", " "),
                            "Current": f"{ind.get('current', 0):+.2f}",
                            "Target":  f"{ind.get('target', 0):+.2f}",
                            "Days to fire": ind.get("days_to_target", "n/a"),
                            "Projected date": ind.get("projected_date", "—"),
                        })
                if ie_rows:
                    st.dataframe(pd.DataFrame(ie_rows), width='stretch',
                                  hide_index=True)
                st.caption(_ie.get("summary", ""))

            # === CYCLE 4 ANALOG ===
            _c4 = _dp.get("cycle_4_analog", {})
            if _c4 and not _c4.get("error"):
                st.markdown("<div class='section-header'>Cycle 4 Analog — what was cycle 4 doing at this equivalent day?</div>",
                             unsafe_allow_html=True)
                _c4_cols = st.columns(3)
                with _c4_cols[0]:
                    st.markdown(metric_card(
                        "Today's cycle 4 analog day",
                        _c4.get("cycle_4_analog_date", "?"),
                        f"{_c4.get('days_since_cycle5_peak', 0)}d since cycle 5 peak",
                        C["accent"],
                    ), unsafe_allow_html=True)
                with _c4_cols[1]:
                    proj_date = _c4.get("projected_cycle5_bottom_date", "?")
                    st.markdown(metric_card(
                        "Projected cycle 5 bottom",
                        proj_date,
                        f"{_c4.get('days_to_analog_cycle4_bottom', 0)}d ahead (cycle 4 analog)",
                        C["lth"],
                    ), unsafe_allow_html=True)
                with _c4_cols[2]:
                    implied_price = _c4.get("implied_cycle5_bottom_price", 0)
                    implied_dd = _c4.get("implied_cycle5_bottom_drawdown_pct", 0)
                    st.markdown(metric_card(
                        "Implied bottom price",
                        f"${implied_price:,.0f}",
                        f"{implied_dd:.0f}% from cycle 5 peak (amplitude decay)",
                        C["bull"],
                    ), unsafe_allow_html=True)
                st.caption(_c4.get("summary", ""))

            # === MACRO CALENDAR ===
            _mc = _dp.get("macro_calendar", {})
            if _mc and _mc.get("events"):
                st.markdown("<div class='section-header'>Macro Calendar — next 6 months of BTC-moving events</div>",
                             unsafe_allow_html=True)
                st.caption("FOMC + CPI + NFP dates with BTC sensitivity context")
                mc_rows = []
                for e in _mc.get("events", [])[:20]:
                    sens_color = ("🔴" if e["btc_sensitivity"] == "HIGH"
                                  else "🟠" if "MEDIUM-HIGH" in e["btc_sensitivity"]
                                  else "🟡")
                    mc_rows.append({
                        "Date": e["date"],
                        "Days from now": e["days_from_now"],
                        "Event": e["event"],
                        "BTC sensitivity": f"{sens_color} {e['btc_sensitivity']}",
                        "Context": e["context"][:90],
                    })
                st.dataframe(pd.DataFrame(mc_rows), width='stretch',
                              hide_index=True)
        except Exception as e:
            st.caption(f"Date predictions unavailable: {e}")

    with st.expander("⏳ Halving clock", expanded=False):
        st.markdown("<div class='section-header'>Halving Clock</div>",
                     unsafe_allow_html=True)
        st.caption(
            f"Pattern peak: halving + {MEAN_DAYS_TO_PEAK}d (±{PEAK_STD_DEV}d). "
            f"Pattern bottom: halving + {MEAN_DAYS_TO_BOTTOM}d (±{BOTTOM_STD_DEV}d). "
            "n=3/n=2 cycles — small sample. Treat as probability cloud, not calendar."
        )

        _hc = st.columns(4)
        days_post = pos["days_post_halving"]
        cycle_length = (pos["next_halving"] - pos["current_halving"]).days

        with _hc[0]:
            st.markdown(metric_card(
                "Position in cycle",
                f"{days_post}d",
                f"{pos['pct_through_cycle']:.0f}% of {cycle_length}d cycle",
                verdict_color,
            ), unsafe_allow_html=True)

        with _hc[1]:
            st.markdown(metric_card(
                "Pattern phase",
                phase_info["phase"].replace("_", " "),
                f"Bias: {phase_info['directional_bias']:+.2f}",
                C["accent"],
            ), unsafe_allow_html=True)

        with _hc[2]:
            st.markdown(metric_card(
                "Pattern bottom",
                f"{abs(pos['days_to_pattern_bottom'])}d "
                f"{'ahead' if pos['days_to_pattern_bottom'] > 0 else 'past'}",
                pos["projected_bottom_date"].strftime("%b %d, %Y"),
                C["lth"],
            ), unsafe_allow_html=True)

        with _hc[3]:
            c6_date = ppt["cycle6_peak_date"]
            st.markdown(metric_card(
                "Cycle 6 peak (proj)",
                f"~${ppt['cycle6_peak_mid']/1000:.0f}k",
                f"{c6_date.strftime('%b %Y')} ({ppt['cycle6_peak_chg_pct_mid']:+.0f}%)",
                C["bull"],
            ), unsafe_allow_html=True)

    with st.expander("📊 Bottom probability distribution", expanded=False):
        # === Probability distribution ===
        st.markdown("<div class='section-header'>Bottom Probability Distribution</div>",
                     unsafe_allow_html=True)
        st.caption("Three scenarios with explicit probability weights. Replaces calendar thinking.")
        if pdb and not pdb.get("error"):
            _pf = go.Figure()
            for s in pdb["scenarios"]:
                _pf.add_trace(go.Bar(
                    x=[s["name"]], y=[s["probability"] * 100],
                    marker_color=s["color"],
                    text=[f"{s['probability']*100:.0f}%<br>{s['price_range']}<br>{s['date_range']}"],
                    textposition="inside", textfont=dict(size=11, color="white"),
                    hovertemplate=f"<b>{s['name']}</b><br>{s['description']}<extra></extra>",
                    showlegend=False, width=0.6,
                ))
            _pf.update_layout(
                **CHART_LAYOUT, height=260,
                yaxis=dict(title="Probability (%)", range=[0, 60], gridcolor="#2a2d34"),
                xaxis=dict(gridcolor="#2a2d34"),
            )
            st.plotly_chart(_pf, width='stretch')
            st.caption(
                f"**EV bottom price: ${pdb['expected_value_price']:,.0f}** "
                f"({pdb['expected_value_chg_pct']:+.1f}% from current). "
                "Standard halving-cycle gets 50% weight; the other 50% covers ETF-era distortions."
            )

    with st.expander("🎯 Cycle 6 peak targets", expanded=False):
        # === Cycle 6 peak target table ===
        st.markdown("<div class='section-header'>Cycle 6 Peak Targets (amplitude-decay model)</div>",
                     unsafe_allow_html=True)
        st.caption(
            "Cycles flatten each iteration. Cycle 5 peak was 1.84x cycle 4. "
            "Cycle 6 projected 1.4-1.9x cycle 5 (Woo/Glassnode joint review)."
        )
        targets_df = pd.DataFrame([
            {"Scenario": "Conservative (1.4x)", "Price": f"${ppt['cycle6_peak_conservative']:,.0f}",
             "Change vs now": f"{(ppt['cycle6_peak_conservative']/btc_price-1)*100:+.0f}%"},
            {"Scenario": "Mid (1.6x)", "Price": f"${ppt['cycle6_peak_mid']:,.0f}",
             "Change vs now": f"{ppt['cycle6_peak_chg_pct_mid']:+.0f}%"},
            {"Scenario": "Aggressive (1.9x)", "Price": f"${ppt['cycle6_peak_aggressive']:,.0f}",
             "Change vs now": f"{(ppt['cycle6_peak_aggressive']/btc_price-1)*100:+.0f}%"},
        ])
        st.dataframe(targets_df, width='stretch', hide_index=True)

    # === Historical accuracy expander ===
    with st.expander("Historical halving-clock accuracy"):
        hist_rows = []
        for cyc, d in HISTORICAL.items():
            peak_err = abs(d["days_to_peak"] - MEAN_DAYS_TO_PEAK)
            bot_str = f"{d['days_to_bottom']}d" if d["days_to_bottom"] else "TBD"
            bot_err = abs(d["days_to_bottom"] - MEAN_DAYS_TO_BOTTOM) if d["days_to_bottom"] else None
            hist_rows.append({
                "Cycle": cyc,
                "Halving": d["halving"].strftime("%Y-%m-%d"),
                "Days to peak": f"{d['days_to_peak']}d",
                "Peak error vs avg": f"{peak_err}d",
                "Days to bottom": bot_str,
                "Bottom error vs avg": f"{bot_err}d" if bot_err is not None else "—",
            })
        st.dataframe(pd.DataFrame(hist_rows), width='stretch', hide_index=True)
        st.caption(
            f"Mean across cycles: peak at halving+{MEAN_DAYS_TO_PEAK}d (±{PEAK_STD_DEV}d), "
            f"bottom at halving+{MEAN_DAYS_TO_BOTTOM}d (±{BOTTOM_STD_DEV}d). "
            "Cycle 5 peak predicted to ONE day."
        )


# ─────────────────────────────────────────────────────────────────
# ON-CHAIN TAB — pro signals, cost basis cards, aSOPR counter
# ─────────────────────────────────────────────────────────────────
with tab_onchain:
    st.markdown("### 🔗 On-chain — signal layers, cost-basis cards + native scorecards")
    st.caption(
        "On-chain signal layers first (cost-basis, aSOPR, pro / institutional / "
        "premium-free), then the native top & bottom scorecards below. Open any "
        "drawer for the full criterion list."
    )
    try:
        from core.dashboard_cache import get_cached as _gcv
        _nbv = _gcv("btc_native_bottom_scorecard") or {}
        _ntv = _gcv("btc_native_top_scorecard") or {}
        st.markdown(
            f"<div style='padding:8px 14px; border-radius:6px; background:#13161c; "
            f"border-left:4px solid {C['green']}; margin-bottom:10px; font-size:13px; color:#ccc;'>"
            f"<b style='color:{C['green']};'>Bottom (buy) {_nbv.get('n_met', '?')}/{_nbv.get('n_total', '?')}</b>"
            f" &nbsp;·&nbsp; "
            f"<b style='color:{C['red']};'>Top (sell) {_ntv.get('n_met', '?')}/{_ntv.get('n_total', '?')}</b>"
            f" &nbsp;—&nbsp; the headline read; full breakdown below.</div>",
            unsafe_allow_html=True)
    except Exception:
        pass

    with st.expander("💰 Cost-basis support levels", expanded=False):
        # === Cost-basis triple ===
        st.markdown("<div class='section-header'>Cost-basis support levels</div>",
                     unsafe_allow_html=True)
        _cb = st.columns(3)
        with _cb[0]:
            if rp and not rp.get("error"):
                st.markdown(metric_card(
                    "LTH cost basis (Realized Price)",
                    f"${rp['value']:,.0f}",
                    f"30d: {rp['chg_30d_pct']:+.1f}% — true floor",
                    C["lth"],
                ), unsafe_allow_html=True)
        with _cb[1]:
            if sth and not sth.get("error"):
                st.markdown(metric_card(
                    "STH cost basis (155d MA)",
                    f"${sth['value']:,.0f}",
                    f"Price {sth['price_vs_sth_pct']:+.1f}% vs STH",
                    C["sth"],
                ), unsafe_allow_html=True)
        with _cb[2]:
            if pdb and not pdb.get("error"):
                st.markdown(metric_card(
                    "Probability-weighted EV bottom",
                    f"${pdb['expected_value_price']:,.0f}",
                    f"{pdb['expected_value_chg_pct']:+.1f}% from current",
                    C["bull"],
                ), unsafe_allow_html=True)

    with st.expander("📉 aSOPR — bear-structure counter", expanded=False):
        # === aSOPR rejection counter ===
        st.markdown("<div class='section-header'>aSOPR — bear-structure confirmation</div>",
                     unsafe_allow_html=True)
        st.caption("Bears confirm when aSOPR is rejected at 1.0 line repeatedly. Need 3+ rejections.")
        asopr_sig = _find_signal("asopr")
        if asopr_sig and not asopr_sig.get("error"):
            rejections = asopr_sig.get("rejections_at_1", 0)
            days_below = asopr_sig.get("days_below_1", 0)
            ma7 = asopr_sig.get("ma7", 1.0)
            _ar = st.columns(3)
            with _ar[0]:
                col = C["bear"] if rejections >= 3 else C["neutral"] if rejections >= 1 else C["bull"]
                st.markdown(metric_card("1.0 Rejections", f"{rejections} / 3",
                                          "3+ = bear confirmed", col), unsafe_allow_html=True)
            with _ar[1]:
                st.markdown(metric_card("Days below 1.0 (last 60)", f"{days_below}",
                                          ">30 sustained = bottom forming", C["accent"]),
                              unsafe_allow_html=True)
            with _ar[2]:
                col = C["bear"] if ma7 < 0.97 else C["neutral"] if ma7 < 1.02 else C["bull"]
                st.markdown(metric_card("7d MA", f"{ma7:.3f}",
                                          ">1.0 = profit taking", col), unsafe_allow_html=True)

    with st.expander("🔗 Pro on-chain layer (Woo + Glassnode)", expanded=False):
        # === Pro On-Chain Layer ===
        st.markdown("<div class='section-header'>Pro On-Chain Layer (Woo + Glassnode)</div>",
                     unsafe_allow_html=True)
        st.caption(
            "10 institutional bottom signals. Free-tier proxies for paid Glassnode metrics where needed."
        )
        pro_names = [
            ("realized_cap_drawdown", "Realized Cap drawdown", "Checkmate #1"),
            ("reserve_risk", "Reserve Risk", "Glassnode generational-buy"),
            ("puell_multiple", "Puell Multiple", "Miner stress"),
            ("coinbase_premium_gap", "Coinbase Premium", "US institutional flow"),
            ("difficulty_ribbon", "Difficulty Ribbon", "Woo miner capitulation"),
            ("asopr", "aSOPR (proxy)", "Profit/loss realization"),
            ("lth_sth_supply_ratio", "LTH/STH dynamics", "Cohort positioning"),
            ("cdd_spikes", "CDD spikes (proxy)", "LTH movement"),
            ("dormancy_flow", "Dormancy Flow (proxy)", "LTH vs buyer absorption"),
            ("nvt_signal_woo", "NVT Signal (Woo)", "Network value vs throughput"),
        ]
        pro_rows = []
        pro_scores = []
        for name, display, hint in pro_names:
            sig = _find_signal(name)
            if sig is None or sig.get("error"):
                pro_rows.append({"Signal": display, "Score": "—", "Reading": "(unavailable)"})
                continue
            score = sig.get("score")
            val = sig.get("value")
            pro_scores.append(score) if score is not None else None
            val_str = f"{val:.3f}" if isinstance(val, (int, float)) else str(val)[:14]
            emo = ("●●" if score > 0.5 else "●" if score > 0.2
                   else "○" if score > -0.2 else "▼" if score > -0.5 else "▼▼")
            pro_rows.append({
                "Signal": display,
                "Score": f"{emo} {score:+.2f}" if score is not None else "—",
                "Value": val_str,
                "Note": sig.get("note", hint)[:90],
            })
        _avg_pro = np.mean(pro_scores) if pro_scores else 0
        _bull = sum(1 for s in pro_scores if s > 0.3)
        _bear = sum(1 for s in pro_scores if s < -0.3)
        _neutral = sum(1 for s in pro_scores if -0.3 <= s <= 0.3)
        _summary = st.columns(4)
        with _summary[0]:
            avg_label = ("BULL" if _avg_pro > 0.4 else "mild bull" if _avg_pro > 0.1
                         else "neutral" if _avg_pro > -0.1 else "mild bear" if _avg_pro > -0.4 else "BEAR")
            avg_col = C["bull"] if _avg_pro > 0.1 else C["bear"] if _avg_pro < -0.1 else C["neutral"]
            st.markdown(metric_card("Pro avg", avg_label, f"{_avg_pro:+.2f}", avg_col),
                          unsafe_allow_html=True)
        with _summary[1]:
            st.markdown(metric_card("Bull signals", f"{_bull}", f"of {len(pro_scores)}", C["bull"]),
                          unsafe_allow_html=True)
        with _summary[2]:
            st.markdown(metric_card("Bear signals", f"{_bear}", f"of {len(pro_scores)}", C["bear"]),
                          unsafe_allow_html=True)
        with _summary[3]:
            st.markdown(metric_card("Neutral", f"{_neutral}", f"of {len(pro_scores)}", C["muted"]),
                          unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(pro_rows), width='stretch', hide_index=True)


    with st.expander("🏛️ Clemente + Alden — 15 institutional signals", expanded=False):
        # === CLEMENTE + ALDEN LAYER (15 institutional-grade signals) ===
        st.markdown("<div class='section-header'>Clemente + Alden — institutional bottom signals</div>",
                     unsafe_allow_html=True)
        st.caption(
            "15-signal layer from Will Clemente (Reflexivity Research, called 2024 bottom) "
            "and Lyn Alden (called 2024 bottom + 2025 peak via macro liquidity). "
            "These are the signals that called the last cycle inflection."
        )
        _ca_signals = [
            ("hashrate_drawdown",       "Hashrate drawdown from peak",     "Tier B — miner capitulation"),
            ("cb_premium_streak",       "CB premium negative streak",      "Tier C — 21+ days = bottom"),
            ("aasi",                    "AASI (active address sentiment)", "Tier B — Clemente bottom signal"),
            ("stablecoin_supply_ratio", "SSR (dry powder)",                "Tier A — buying pressure"),
            ("etf_pct_of_supply",       "ETF % of supply",                 "Tier A — institutional ownership"),
            ("btc_dominance",           "BTC dominance",                   "Tier A — capital rotation"),
            ("real_yields_10y",         "10y real yields (TIPS)",          "Tier A — Alden #1 macro"),
            ("difficulty_adjustment",   "Difficulty next adj",             "Tier B — cycle inflection"),
            ("btc_gold_ratio",          "BTC/Gold ratio",                  "Tier B — monetary rotation"),
            ("multi_exch_funding",      "Multi-venue funding agg",         "Tier B — cross-venue leverage"),
            ("rhodl_ratio",             "RHODL (proxy)",                   "Tier C — Glassnode top detector"),
            ("reflexivity_index",       "Reflexivity Index",               "Tier C — Clemente composite"),
            ("urpd_clusters",           "URPD cost-basis clusters",        "Tier C — supply density"),
            ("hodl_waves",              "HODL Waves (>1y supply%)",        "Tier A — cycle composition"),
            ("fiscal_dominance",        "Fiscal Dominance Index",          "Tier C — Alden fiscal lens"),
        ]
        _ca_rows = []
        _ca_scores = []
        for sig_name, display, sub in _ca_signals:
            sig = _find_signal(sig_name)
            if sig is None or sig.get("error"):
                _ca_rows.append({"Signal": display, "Score": "—",
                                  "Reading": "(unavailable)", "Source": sub})
                continue
            score = sig.get("score")
            if score is None:
                _ca_rows.append({"Signal": display, "Score": "—",
                                  "Reading": "(no score)", "Source": sub})
                continue
            _ca_scores.append(score)
            emo = ("●●" if score > 0.5 else "●" if score > 0.2
                   else "○" if score > -0.2 else "▼" if score > -0.5 else "▼▼")
            _ca_rows.append({
                "Signal":  display,
                "Score":   f"{emo} {score:+.2f}",
                "Reading": sig.get("note", sub)[:100],
                "Source":  sub,
            })
        if _ca_scores:
            avg = np.mean(_ca_scores)
            n_bull = sum(1 for s in _ca_scores if s > 0.3)
            n_bear = sum(1 for s in _ca_scores if s < -0.3)
            verdict_col = (C["deep_bull"] if avg > 0.4 else C["bull"] if avg > 0.1
                           else C["neutral"] if avg > -0.1 else C["bear"])
            verdict_txt = ("STRONG BOTTOM SIGNAL" if avg > 0.4
                           else "Bottom forming" if avg > 0.1
                           else "Mixed/neutral" if avg > -0.1
                           else "Top forming")
            st.markdown(
                f"<div style='padding:14px 18px; background:{verdict_col}22; "
                f"border-left:6px solid {verdict_col}; border-radius:6px; margin-bottom:12px;'>"
                f"<div style='font-size:10px; color:#888; text-transform:uppercase;'>"
                f"Layer verdict ({n_bull}🟢 / {n_bear}🔴 of {len(_ca_scores)} valid)</div>"
                f"<div style='font-size:20px; font-weight:700; color:{verdict_col};'>{verdict_txt}</div>"
                f"<div style='font-size:13px; color:#aaa;'>Avg score: {avg:+.2f}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.dataframe(pd.DataFrame(_ca_rows), width='stretch', hide_index=True)

    with st.expander("🆓 Premium-free layer — 18-signal table", expanded=False):
        # === PREMIUM-FREE LAYER (full 18-signal table) ===
        st.markdown("<div class='section-header'>Premium-Free Layer — paid-tier signals from free APIs</div>",
                     unsafe_allow_html=True)
        st.caption(
            "Replicates $500/mo of Glassnode/CryptoQuant/Coinglass/Skew/Bloomberg data. "
            "All free APIs: Farside, DefiLlama, Reddit, GitHub, Deribit, Wikipedia, "
            "mempool.space, Blockchair, FRED, SEC EDGAR."
        )
        _premium_table = [
            ("etf_flows",              "ETF flows (Farside)",        "Tier 1 — institutional accumulation"),
            ("stablecoin_supply",      "Stablecoin supply",          "Tier 1 — fresh liquidity"),
            ("github_activity",        "BTC Core dev activity",      "Tier 1 — project health"),
            ("deribit_greeks",         "Deribit max pain + skew",    "Tier 1 — options-implied pin"),
            ("lth_supply_exact",       "LTH supply (exact)",         "Tier 1 — cohort positioning"),
            ("net_liquidity",          "Net Liquidity (Fed)",        "Tier 2 — Bloomberg's $20K/yr metric"),
            ("miner_holdings",         "Miner SEC filings (MARA)",   "Tier 2 — sell pressure"),
            ("hash_price",             "Hash price",                 "Tier 2 — miner economics"),
            ("mempool_pressure",       "Mempool fees",               "Tier 2 — network demand"),
            ("news_sentiment",         "News sentiment",             "Tier 2 — CryptoPanic RSS"),
            ("exchange_net_flows",     "Whale tx activity",          "Tier 2 — Blockchair"),
            ("wikipedia_views",        "Wikipedia views",            "Tier 3 — retail interest"),
            ("dxy_regime",             "DXY regime",                 "Tier 3 — USD strength"),
            ("energy_prices",          "Energy (oil + gas)",         "Tier 3 — miner cost basis"),
            ("whale_tx_activity",      "Large tx count",             "Tier 3 — whale alert"),
            ("defi_tvl",               "DeFi TVL",                   "Tier 3 — risk appetite"),
            ("stablecoin_chain_flows", "Stables: ETH vs Tron",       "Tier 3 — geographic flows"),
            ("reddit_sentiment",       "r/bitcoin (blocked)",        "Tier 1 — RIP, Reddit OAuth"),
        ]
        _pf_rows = []
        _pf_scores = []
        for name, display, sub in _premium_table:
            sig = _find_signal(name)
            if sig is None or sig.get("error"):
                _pf_rows.append({"Signal": display, "Score": "—",
                                  "Reading": "(unavailable)", "Source": sub})
                continue
            score = sig.get("score")
            if score is None:
                _pf_rows.append({"Signal": display, "Score": "—",
                                  "Reading": "(no score)", "Source": sub})
                continue
            _pf_scores.append(score)
            emo = ("●●" if score > 0.5 else "●" if score > 0.2
                   else "○" if score > -0.2 else "▼" if score > -0.5 else "▼▼")
            _pf_rows.append({
                "Signal":  display,
                "Score":   f"{emo} {score:+.2f}",
                "Reading": sig.get("note", sub)[:90],
                "Source":  sub,
            })
        if _pf_scores:
            avg = np.mean(_pf_scores)
            n_bull = sum(1 for s in _pf_scores if s > 0.3)
            n_bear = sum(1 for s in _pf_scores if s < -0.3)
            st.caption(
                f"Premium-free avg: **{avg:+.2f}** "
                f"({n_bull} bullish / {n_bear} bearish of {len(_pf_scores)} valid). "
                f"Replaces ~$500/mo of paid services."
            )
        st.dataframe(pd.DataFrame(_pf_rows), width='stretch', hide_index=True)


    # ─────────────────────────────────────────────────────────────────
    # TECHNICAL TAB — Olson layer, 3-lens ensemble, key levels
    # ─────────────────────────────────────────────────────────────────
with tab_technical:
    st.markdown("### 🔬 Detail — technical layers, signal breakdown & hit-rates")
    st.caption(
        "Jesse Olson's TA, the 3-lens ensemble, key levels, and the full signal "
        "roster with prediction hit-rates. Open any drawer to dig in."
    )
    try:
        _olv = (cached_olson() or {}).get("verdict_level", "?")
        _ensv = (ensemble or {}).get("consensus", "?")
        _ol_col = {"BULLISH": C["green"], "MILD_BULL": C["green"], "NEUTRAL": C["yellow"],
                   "MILD_BEAR": C["red"], "BEARISH": C["red"]}.get(_olv, C["muted"])
        st.markdown(
            f"<div style='padding:8px 14px; border-radius:6px; background:#13161c; "
            f"border-left:4px solid {_ol_col}; margin-bottom:10px; font-size:13px; color:#ccc;'>"
            f"Olson TA: <b style='color:{_ol_col};'>{str(_olv).replace('_', ' ')}</b> &nbsp;·&nbsp; "
            f"3-lens ensemble: <b>{_ensv}</b> &nbsp;—&nbsp; details in the drawers below.</div>",
            unsafe_allow_html=True)
    except Exception:
        pass

    with st.expander("📐 Jesse Olson technical layer", expanded=False):
        # === Jesse Olson framework ===
        st.markdown("<div class='section-header'>Jesse Olson Technical Layer</div>",
                     unsafe_allow_html=True)
        st.caption(
            "Pure TA on multi-week timeframes. Called cycle 5 top via 3-week MACD bearish cross."
        )
        try:
            olson = cached_olson()
            lvl = olson["verdict_level"]
            olson_color = {"BULLISH": C["deep_bull"], "MILD_BULL": C["bull"],
                           "NEUTRAL": C["neutral"], "MILD_BEAR": C["bear"],
                           "BEARISH": C["deep_bear"]}.get(lvl, C["muted"])
            st.markdown(
                f"<div style='padding:14px 18px; background:{olson_color}22; "
                f"border-left:6px solid {olson_color}; border-radius:6px; margin-bottom:14px;'>"
                f"<div style='font-size:10px; color:#888; text-transform:uppercase;'>"
                f"Olson verdict ({olson['bullish_count']} bull / {olson['bearish_count']} bear of {olson['n_valid']})</div>"
                f"<div style='font-size:20px; font-weight:700; color:{olson_color};'>{olson['verdict']}</div>"
                f"<div style='font-size:13px; color:#aaa;'>Avg score: {olson['avg_score']:+.2f}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            _ol = st.columns(3)
            _sigs = [
                ("three_week_macd", "3-week MACD", "His #1 timing tool"),
                ("weekly_heikin_ashi", "Weekly Heikin Ashi", "Color + wick pattern"),
                ("weekly_rsi_divergence", "Weekly RSI", "Divergence vs price"),
            ]
            for col, (key, name, sub) in zip(_ol, _sigs):
                d = olson["signals"].get(key)
                with col:
                    if d is None or d.get("error"):
                        st.markdown(metric_card(name, "—", sub, C["muted"]),
                                      unsafe_allow_html=True)
                        continue
                    score = d.get("score", 0)
                    phase = d.get("phase", "?").replace("_", " ")
                    acc = (C["deep_bull"] if score > 0.5 else C["bull"] if score > 0.2
                           else C["neutral"] if score > -0.2 else C["bear"] if score > -0.5
                           else C["deep_bear"])
                    note = d.get("note", "")[:120]
                    st.markdown(
                        f"<div style='background:#1a1d24; padding:12px 14px; border-radius:6px; "
                        f"border-left:3px solid {acc}; min-height:140px;'>"
                        f"<div style='font-size:10px; color:#888; text-transform:uppercase;'>{name}</div>"
                        f"<div style='font-size:16px; font-weight:700; color:{acc}; margin:3px 0;'>{phase}</div>"
                        f"<div style='font-size:13px; color:#fff;'>Score: <b>{score:+.2f}</b></div>"
                        f"<div style='font-size:11px; color:#888; margin-top:6px;'>{note}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
        except Exception as e:
            st.caption(f"Olson layer unavailable: {e}")

    with st.expander("🔭 3-Lens ensemble consensus", expanded=True):
        # === 3-Lens Ensemble ===
        st.markdown("<div class='section-header'>3-Lens Ensemble Consensus</div>",
                     unsafe_allow_html=True)
        st.caption("Technical / On-chain / Macro lenses scored independently. Agreement = conviction.")
        if ensemble:
            consensus = ensemble.get("consensus", "?")
            cons_color = (C["deep_bull"] if "UNANIMOUS BULL" in consensus
                           else C["deep_bear"] if "UNANIMOUS BEAR" in consensus
                           else C["neutral"])
            st.markdown(
                f"<div style='padding:12px 16px; background:{cons_color}22; "
                f"border-left:4px solid {cons_color}; border-radius:6px; margin-bottom:12px;'>"
                f"<b style='color:{cons_color};'>Consensus:</b> {consensus}"
                f"</div>",
                unsafe_allow_html=True,
            )
            _ens = st.columns(3)
            for col, (lname, ld) in zip(_ens, ensemble.get("lenses", {}).items()):
                _lscore = ld.get("score", 0) or 0
                _linterp = ld.get("interpretation", "?")
                _ln = ld.get("n_signals", 0)
                _ldisp = lname.replace("_lens", "").title()
                with col:
                    _lg = go.Figure(go.Indicator(
                        mode="gauge+number", value=round(float(_lscore), 2),
                        number={"font": {"size": 22, "color": "#fff"}, "valueformat": "+.2f"},
                        title={"text": f"{_ldisp} lens", "font": {"size": 13, "color": "#d4d4d4"}},
                        gauge={
                            "axis": {"range": [-1, 1], "tickvals": [-1, -0.5, 0, 0.5, 1],
                                     "tickfont": {"size": 8, "color": "#888"}, "tickcolor": "#888"},
                            "bar": {"color": "rgba(255,255,255,0.9)", "thickness": 0.22},
                            "bgcolor": "#0e1117", "borderwidth": 0,
                            "steps": [
                                {"range": [-1, -0.2], "color": "rgba(239,68,68,0.55)"},
                                {"range": [-0.2, 0.2], "color": "rgba(240,185,11,0.40)"},
                                {"range": [0.2, 1], "color": "rgba(34,197,94,0.55)"}],
                            "threshold": {"line": {"color": "#fff", "width": 3},
                                          "thickness": 0.9, "value": float(_lscore)}}))
                    _lg.update_layout(paper_bgcolor="#0e1117", height=185,
                                      margin=dict(l=18, r=18, t=46, b=6),
                                      font=dict(color="#d4d4d4"))
                    st.plotly_chart(_lg, width='stretch',
                                    config={"displayModeBar": False, "displaylogo": False})
                    st.caption(f"**{_linterp}** · {_ln} signals")

    with st.expander("📏 Key levels", expanded=False):
        # === Key Levels ===
        st.markdown("<div class='section-header'>Key Levels</div>",
                     unsafe_allow_html=True)
        try:
            from core.btc_weekly_report import _key_levels
            levels = _key_levels(btc_price)
            _pts = ([(r["price"], r["name"], r["distance_pct"], "res") for r in levels["resistances"]]
                    + [(s["price"], s["name"], s["distance_pct"], "sup") for s in levels["supports"]])
            _klf = go.Figure()
            _klf.add_hline(y=btc_price, line=dict(color="#ffffff", width=1, dash="dot"))
            _klf.add_trace(go.Scatter(
                x=[0] * len(_pts), y=[p[0] for p in _pts], mode="markers+text",
                marker=dict(size=11,
                            color=["#ef4444" if p[3] == "res" else "#22c55e" for p in _pts],
                            line=dict(color="#0e1117", width=1)),
                text=[f"  ${p[0]:,.0f} · {p[1]} ({'+' if p[3] == 'res' else '-'}{p[2]:.1f}%)"
                      for p in _pts],
                textposition="middle right", textfont=dict(size=10, color="#cccccc"),
                hoverinfo="skip", showlegend=False))
            _klf.add_trace(go.Scatter(
                x=[0], y=[btc_price], mode="markers+text",
                marker=dict(size=15, color="#ffffff", symbol="diamond",
                            line=dict(color="#000000", width=1)),
                text=[f"  ● NOW ${btc_price:,.0f}"], textposition="middle right",
                textfont=dict(size=12, color="#ffffff"), hoverinfo="skip", showlegend=False))
            _klf.update_layout(**CHART_LAYOUT, height=470, showlegend=False,
                               yaxis=dict(type="log", title="Price (USD, log)", gridcolor="#2a2d34"),
                               xaxis=dict(visible=False, range=[-0.2, 3.2]))
            st.plotly_chart(_klf, width='stretch',
                            config={"displayModeBar": False, "displaylogo": False})
            st.caption("🔴 resistance above · ⚪ current price · 🟢 support below — distance shown vs spot.")
        except Exception as e:
            st.caption(f"Key levels unavailable: {e}")


    # ─────────────────────────────────────────────────────────────────
    # DETAIL TAB — top signals, all signals expandable, hit rates
    # ─────────────────────────────────────────────────────────────────
with tab_detail:
    with st.expander("🐂 Top 5 bull / bear signals", expanded=True):
        # === Top bull/bear signals ===
        st.markdown("<div class='section-header'>Top 5 Bull / Bear Signals (by score)</div>",
                     unsafe_allow_html=True)
        all_sigs = []
        for cat, cat_sigs in state.get("signals", {}).items():
            if not isinstance(cat_sigs, dict) or cat_sigs.get("error"): continue
            for name, d in cat_sigs.items():
                if not isinstance(d, dict): continue
                s = d.get("score")
                if s is None: continue
                all_sigs.append({"name": name, "category": cat, "score": float(s),
                                 "note": d.get("note", "")[:80]})
        all_sigs.sort(key=lambda x: x["score"])

        _bull5 = list(reversed(all_sigs))[:5]
        _bear5 = all_sigs[:5]
        _seen = {}
        for _s in _bear5 + _bull5:
            _seen[_s["name"]] = _s
        _bars = sorted(_seen.values(), key=lambda x: x["score"])
        if _bars:
            _bnames = [s["name"].replace("_", " ").title() for s in _bars]
            _bscore = [s["score"] for s in _bars]
            _bcol = ["#22c55e" if v >= 0 else "#ef4444" for v in _bscore]
            _bmax = max((abs(v) for v in _bscore), default=1) or 1
            _bbf = go.Figure(go.Bar(
                x=_bscore, y=_bnames, orientation="h",
                marker=dict(color=_bcol, line=dict(color="#0e1117", width=1)),
                text=[f"{v:+.2f}" for v in _bscore], textposition="outside",
                textfont=dict(size=11, color="#cccccc"),
                customdata=[f"{s['category']} · {s['note']}" for s in _bars],
                hovertemplate="<b>%{y}</b><br>score %{x:+.2f}<br>%{customdata}<extra></extra>",
            ))
            _bbf.add_vline(x=0, line=dict(color="#888888", width=1))
            _bbf.update_layout(
                **CHART_LAYOUT, height=max(300, 42 * len(_bars) + 60),
                xaxis=dict(title="bearish ◀ score ▶ bullish",
                           range=[-_bmax * 1.3, _bmax * 1.3],
                           gridcolor="#2a2d34", zeroline=False),
                yaxis=dict(automargin=True), showlegend=False)
            st.plotly_chart(_bbf, width='stretch',
                            config={"displayModeBar": False, "displaylogo": False})
            st.caption("🟢 bullish (score to the right) · 🔴 bearish (to the left) — top signals by absolute score.")
        else:
            st.caption("No scored signals available.")

    # === All signals by category (expandable) ===
    st.markdown(f"<div class='section-header'>All {len(all_sigs)} Signals — by category</div>",
                 unsafe_allow_html=True)
    with st.expander("📋 All signals by category", expanded=False):
        for cat, cat_sigs in state.get("signals", {}).items():
            if not isinstance(cat_sigs, dict) or cat_sigs.get("error"): continue
            scored = [(n, d) for n, d in cat_sigs.items()
                       if isinstance(d, dict) and d.get("score") is not None]
            if not scored: continue
            st.markdown(f"**{cat.upper()}** — {len(scored)} signals")
            rows = []
            for name, d in scored:
                score = d.get("score", 0)
                val = d.get("value", "")
                vs = f"{val:.3f}" if isinstance(val, float) else str(val)[:14]
                emo = "●" if score > 0.3 else "○" if abs(score) <= 0.3 else "▼"
                rows.append({
                    "Signal": name, "Value": vs,
                    "Score": f"{emo} {score:+.2f}",
                    "Note": d.get("note", "")[:80],
                })
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    with st.expander("🎯 Prediction hit-rates + anomalies", expanded=False):
        # === Hit rates + anomalies ===
        hr = state.get("hit_rates_by_horizon", {})
        if hr:
            st.markdown("<div class='section-header'>Prediction Hit Rates (from outcome log)</div>",
                         unsafe_allow_html=True)
            hr_rows = []
            for h, d in hr.items():
                if d.get("n_observations", 0) >= 3:
                    hr_rows.append({
                        "Horizon": h,
                        "Correct": d["n_correct"],
                        "Total": d["n_observations"],
                        "Hit Rate": f"{d['hit_rate']*100:.0f}%",
                    })
            if hr_rows:
                st.dataframe(pd.DataFrame(hr_rows), width='stretch', hide_index=True)

        anom = state.get("signal_anomalies", [])
        if anom:
            st.markdown("<div class='section-header'>Signal Anomalies (regime-change candidates)</div>",
                         unsafe_allow_html=True)
            a_rows = []
            for a in anom[:8]:
                a_rows.append({
                    "Signal": a["signal"], "Z-score": f"{a['z_score']:+.2f}",
                    "Interpretation": a["interpretation"][:60],
                })
            st.dataframe(pd.DataFrame(a_rows), width='stretch', hide_index=True)


    # ═══════════════════════════════════════════════════════════════════════
    # SWIFT RESTRUCTURE: Fresh content blocks for renamed tabs
    # (Reads from cache — no recompute, just clean re-rendering for the new
    # tab organization Phillip Swift recommended)
    # ═══════════════════════════════════════════════════════════════════════

    # ─────────────────────────────────────────────────────────────
    # 📈 CHARTS TAB — Swift charts + LookIntoBitcoin embeds
    # ─────────────────────────────────────────────────────────────
with tab_charts:
    st.caption(
        "Self-hosted cycle charts — generated live from our own data "
        "(they mirror the LookIntoBitcoin / Bitcoin Magazine Pro originals)."
    )

    # === Self-hosted cycle charts (render reliably; the official sites block embedding) ===
    st.markdown("#### 🌈 Cycle charts (live)")
    st.info(
        "📉 **Why these bands look 'muted' vs 2021 / 2017** — each cycle reaches less far into the "
        "extremes (2025 barely tagged the Rainbow's yellow; 2017 hit deep red). Two structural causes: "
        "**diminishing returns** (BTC's multi-trillion cap makes the old price *multiples* unreachable) "
        "and **ETF / institutional smoothing** (steady programmatic flows replace retail blow-offs). "
        "It's **symmetric — bottoms are shallower too** (2022 bottomed near MVRV 0.75, not a 2018-style "
        "washout). The verdict already compensates: the **Cycle-6 detector** (Signals tab) auto-scales "
        "the bottom-buy thresholds ~0.70–0.85× in a muted era, and the scorecards read these metrics by "
        "**percentile-rank within history**, not absolute levels — so it won't wait for a 2018-depth flush "
        "that may never come. **Read the bands relative-to-era, not absolute.**"
    )
    with st.expander("📊 Proof — every metric peaks lower each cycle (verified from our own data)"):
        _cpt = None
        try:
            from core.dashboard_cache import get_cached as _gc2
            _cpt = _gc2("cycle_peak_table")
        except Exception:
            _cpt = None
        if _cpt and _cpt.get("rows"):
            def _cell(x, suf=""):
                return f"{x}{suf}" if x is not None else "—"
            _lines = []
            for _row in _cpt["rows"]:
                _pi = _row.get("pi_cycle")
                _pi_cell = f"{_pi} ✓" if (_pi is not None and _pi >= 1.0) else f"**{_cell(_pi)} ✗**"
                _lines.append(
                    f"| **{_cell(_row.get('cycle'))}** | {_cell(_row.get('price_fmt'))} | {_cell(_row.get('mvrv'))} | "
                    f"{_pi_cell} | {_cell(_row.get('p350'),'×')} | {_cell(_row.get('p730'),'×')} | {_cell(_row.get('puell'))} |"
                )
            _tbl = ("| Cycle top | Price | MVRV | Pi Cycle | Px ÷ 350d-MA | 2-Yr MA mult | Puell |\n"
                    "|---|---|---|---|---|---|---|\n" + "\n".join(_lines))
            st.markdown(
                "Per-cycle **peak** readings — recomputed live each refresh from our CoinMetrics "
                "price/MVRV history (back to 2010) and blockchain.com miner revenue. Every column a "
                "*lower high* than the cycle before:\n\n" + _tbl + "\n\n"
                "✓ = Pi Cycle Top crossed 1.0 (sell signal fired). The amplitude compression — "
                "diminishing returns + ETF/institutional smoothing — is exactly what the **Cycle-6 "
                f"detector** corrects for. _Data through {_cell(_cpt.get('asof'))}._"
            )
        else:
            st.markdown(
"""Per-cycle **peak** readings — from our CoinMetrics price/MVRV history (back to 2010) and blockchain.com miner revenue. Every column is a *lower high* than the cycle before:

| Cycle top | Price | MVRV | Pi Cycle | Px ÷ 350d-MA | 2-Yr MA mult | Puell |
|---|---|---|---|---|---|---|
| **2013** | $1.1k | 5.88 | 1.23 ✓ | 11.9× | 18.0× | 14.4 |
| **2017** | $19.6k | 4.72 | 1.06 ✓ | 5.6× | 10.0× | 7.1 |
| **2021** | $67.5k | 3.96 | 1.00 ✓ | 3.7× | 4.9× | 3.5 |
| **2024–25 (to date)** | $124.8k | 2.78 | **0.74 ✗** | 2.1× | 2.5× | 2.8 |

✓ = Pi Cycle Top crossed 1.0 (sell signal fired). **In 2024–25 it peaked at 0.74 — never crossed, so the classic top signal never fired.** Diminishing returns + ETF smoothing — the compression the **Cycle-6 detector** corrects for.""")
    try:
        from core.dashboard_cache import get_cached as _gc
        _sc = _gc("swift_charts")
        if _sc:
            import plotly.graph_objects as go
            _CFG = {"displayModeBar": False, "scrollZoom": False, "doubleClick": False, "displaylogo": False}
            # One-line explainer under each chart, focused on WHERE THE CYCLE BOTTOM
            # (accumulation) zone is — the rotation trigger this dashboard watches for.
            _EXPLAIN = {
                "rainbow": "**Rainbow** — log-regression price bands. The cool **blue/green bands at the bottom = the historical undervalued / accumulation zone**; warm red-orange at the top = historically overheated.",
                "pi_cycle_top": "**Pi Cycle Top** — 111-day MA divided by (2 × 350-day MA). A cross **up through 1.0 has marked every cycle _top_ within days**; sitting well below 1.0 means we're far from a top.",
                "pi_cycle_bottom": "**Pi Cycle Bottom** — a short MA vs a scaled long MA. **Dips toward / under the line have historically marked major _bottoms_ within weeks.**",
                "golden_ratio": "**Golden Ratio** — 350-day MA times Fibonacci multiples. **Price down near the base 350-day MA line = the historical accumulation zone**; the higher x1.6 to x3 bands have marked tops.",
                "two_year_ma": "**2-Year MA Multiplier** — **price under the green 2-year MA has historically been a deep-value bottom-buy zone**; up at the red band (2-year MA x5) = historically a top.",
                "mvrv_bands": "**MVRV** — market value divided by realized value (average holder cost basis). **Below 1.0 = the average holder is underwater = capitulation/bottom historically**; above ~3.5 = euphoria/top.",
                "puell_bands": "**Puell Multiple** — daily miner revenue vs its 1-year average. **Low / green (below ~0.5) = miner capitulation, a historical bottom**; high / red (above ~4) = top.",
                "hodl_waves": "**HODL Waves** — supply split by coin age. **Bottoms historically show the long-term-holder bands swelling** (old coins quietly accumulating) while short-term-holder supply thins out.",
            }
            def _draw(name):
                fig = _sc.get(name)
                if not fig:
                    return
                try:
                    st.plotly_chart(_swift_fig(fig), width='stretch', config=_CFG, key=f"cyclelive_{name}")
                    _ex = _EXPLAIN.get(name)
                    if _ex:
                        st.caption(_ex)
                except Exception:
                    st.caption(f"- {name.replace('_', ' ')} chart unavailable")
            _draw("rainbow")
            _r1 = st.columns(2)
            with _r1[0]: _draw("pi_cycle_top")
            with _r1[1]: _draw("pi_cycle_bottom")
            _r2 = st.columns(2)
            with _r2[0]: _draw("golden_ratio")
            with _r2[1]: _draw("two_year_ma")
            _r3 = st.columns(2)
            with _r3[0]: _draw("mvrv_bands")
            with _r3[1]: _draw("puell_bands")
            _draw("hodl_waves")
        else:
            st.info("Cycle charts are still warming up \u2014 check back in a few minutes.")
    except Exception as _e:
        st.caption(f"Cycle charts — temporarily unavailable")



# ─────────────────────────────────────────────────────────────
# 📋 SCORECARDS TAB — all multi-criteria scorecards consolidated
# ─────────────────────────────────────────────────────────────
with tab_scorecards:
    st.divider()
    st.markdown("#### 🧮 Native top & bottom scorecards")
    st.caption("The headline verdicts here roll up to the Cockpit on the Signals tab.")

    try:
        from core.dashboard_cache import get_cached as _gc

        # === BTC Native scorecards (top + bottom) ===
        _col1, _col2 = st.columns(2)
        with _col1:
            _ntop = _gc("btc_native_top_scorecard")
            if _ntop:
                _level = _ntop.get("verdict_level", "HOLD")
                _color = _top_color(_level)
                _ntop_n = _ntop.get("n_met") or 0
                _ntop_total = _ntop.get("n_total") or 16
                st.markdown(
                    f"<div style='padding:14px; border-radius:8px; background:#13161c; "
                    f"border-left:4px solid {_color};'>"
                    f"<div style='display:flex; justify-content:space-between; align-items:baseline;'>"
                    f"<span style='font-size:10px; color:#888; text-transform:uppercase;'>🔻 BTC TOP <span style='color:#ef4444;'>(sell BTC?)</span></span>"
                    f"<span>{_age_badge('btc_native_top_scorecard')}</span></div>"
                    f"<div style='font-size:24px; font-weight:700; color:{_color};'>{_level.replace('_',' ')}</div>"
                    f"<div style='font-size:12px; color:#aaa;'>{_ntop_n}/{_ntop_total} criteria firing</div>"
                    f"{_seg_bar(_ntop_n, _ntop_total, _color)}"
                    f"<div style='font-size:11px; color:#ccc; margin-top:6px; line-height:1.4;'>"
                    f"{_dormant_status(_ntop_n, _ntop_total, 'top')}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                with st.expander(f"📋 All 16 BTC-native top criteria"):
                    st.markdown(_crit_tiles(_ntop.get("criteria", []), _color),
                                unsafe_allow_html=True)
        with _col2:
            _nbot = _gc("btc_native_bottom_scorecard")
            if _nbot:
                _level = _nbot.get("verdict_level", "HOLD")
                _color = _bottom_color(_level)
                _nbot_n = _nbot.get("n_met") or 0
                _nbot_total = _nbot.get("n_total") or 16
                st.markdown(
                    f"<div style='padding:14px; border-radius:8px; background:#13161c; "
                    f"border-left:4px solid {_color};'>"
                    f"<div style='display:flex; justify-content:space-between; align-items:baseline;'>"
                    f"<span style='font-size:10px; color:#888; text-transform:uppercase;'>🔺 BTC BOTTOM <span style='color:#22c55e;'>(buy BTC?)</span></span>"
                    f"<span>{_age_badge('btc_native_bottom_scorecard')}</span></div>"
                    f"<div style='font-size:24px; font-weight:700; color:{_color};'>{_level.replace('_',' ')}</div>"
                    f"<div style='font-size:12px; color:#aaa;'>{_nbot_n}/{_nbot_total} guru signals</div>"
                    f"{_seg_bar(_nbot_n, _nbot_total, _color)}"
                    f"<div style='font-size:11px; color:#ccc; margin-top:6px; line-height:1.4;'>"
                    f"{_dormant_status(_nbot_n, _nbot_total, 'bottom')}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                with st.expander(f"📋 All 15 guru-tier bottom criteria"):
                    st.markdown(_crit_tiles(_nbot.get("criteria", []), _color),
                                unsafe_allow_html=True)

        st.divider()

        # === Pattern target zones ===
        st.markdown("#### 🎯 Pattern Target Zones — where price sits vs supply/support")
        _zones = _gc("pattern_zones")
        if _zones and "zones" in _zones:
            _curr_price = _zones.get("price", 0)
            _curr_zone = _zones.get("current_zone", "?")
            st.caption(f"BTC ${_curr_price:,.0f} — currently in: **{_curr_zone}**")
            _zone_cells = []
            for z in _zones["zones"]:
                inside = z["status"] == "INSIDE"
                _dp = z.get("distance_pct", 0)
                if inside:
                    border = "3px solid #ffffff"; bg = "rgba(255,255,255,0.12)"; color = "#fff"
                else:
                    _base = "239,68,68" if _dp > 0 else "34,197,94"  # red above / green below
                    _alpha = max(0.08, min(0.42, 0.42 - abs(_dp) / 100))  # closer zone = stronger
                    border = "1px solid #2a2d36"; bg = f"rgba({_base},{_alpha:.2f})"; color = "#ddd"
                _zone_cells.append(
                    f"<div style='flex:1; padding:8px 6px; border-radius:5px; "
                    f"background:{bg}; border:{border}; text-align:center; min-width:80px;'>"
                    f"<div style='font-size:9px; color:#999;'>{z['label'][:20]}</div>"
                    f"<div style='font-size:11px; color:{color}; font-weight:600;'>"
                    f"${z['low']/1000:.0f}-${z['high']/1000:.0f}k</div>"
                    f"<div style='font-size:9px; color:#aaa;'>{z['distance_pct']:+.1f}%</div>"
                    f"</div>"
                )
            st.markdown(
                "<div style='display:flex; gap:4px; flex-wrap:wrap; margin-bottom:14px;'>"
                + "".join(_zone_cells) + "</div>",
                unsafe_allow_html=True,
            )

        # === ETF flow regime ===
        st.markdown("#### 💰 ETF Flow Regime")
        _etf = _gc("etf_regime")
        if _etf and _etf.get("regime") != "DATA_UNAVAILABLE":
            _regime_name = _etf.get("regime", "?")
            _color = ("#22c55e" if "INFLOW" in _regime_name or "ACCUMULATION" in _regime_name else
                        "#ef4444" if "OUTFLOW" in _regime_name or "CAPITULATION" in _regime_name else
                        "#f0b90b")
            st.markdown(
                f"<div style='padding:12px 16px; border-radius:6px; background:#13161c; "
                f"border-left:3px solid {_color}; margin:8px 0;'>"
                f"<b style='color:{_color}; font-size:16px;'>{_regime_name.replace('_',' ')}</b><br>"
                f"<span style='color:#aaa; font-size:12px;'>"
                f"5d ${_etf.get('flows_5d_M', 0):+,.0f}M | "
                f"30d ${_etf.get('flows_30d_M', 0):+,.0f}M | "
                f"60d ${_etf.get('flows_60d_M', 0):+,.0f}M</span></div>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("ETF flow data unavailable (Farside scraper offline)")

    except Exception as _e:
        st.warning(f"Scorecards data — temporarily unavailable")

    st.caption(
        "📌 The Equity Top Confirmation and Bottom Confirmation scorecards live on the "
        "Signals tab; the ETF-Aware Bottom Trigger is on the Playbook tab."
    )


# ─────────────────────────────────────────────────────────────
# 📊 MACRO TAB — Unified Decision + Rotation + Macro Drivers
# ─────────────────────────────────────────────────────────────
with tab_macro:
    st.markdown("### 📊 Macro — global cycle + rotation + macro drivers")
    st.caption(
        "Macro layer (8 leading indicators) + regime state machine (RISK_ON / "
        "LATE_CYCLE / RECESSIONARY_BEAR) + equity→BTC rotation tracker + ETF flows + "
        "macro driver cards."
    )

    try:
        from core.dashboard_cache import get_cached as _gc

        # === Unified Decision regime + buckets ===
        _ud_macro = _gc("unified_decision")
        if _ud_macro:
            regime = _ud_macro.get("regime", "?")
            regime_color = ("#22c55e" if regime == "RISK_ON" else
                              "#f0b90b" if regime == "LATE_CYCLE" else "#ef4444")
            regime_emoji = {"RISK_ON": "🟢", "LATE_CYCLE": "🟡",
                              "RECESSIONARY_BEAR": "🔴"}.get(regime, "⚪")
            buckets = _ud_macro.get("regime_buckets", {})
            vetoes = _ud_macro.get("vetoes_active", [])

            st.markdown(
                f"<div style='padding:18px 22px; border-radius:10px; margin-bottom:14px; "
                f"background:linear-gradient(135deg, {regime_color}22 0%, #13161c 70%); "
                f"border-left:6px solid {regime_color};'>"
                f"<div style='font-size:10px; color:#888; text-transform:uppercase; letter-spacing:2px;'>"
                f"Macro Regime</div>"
                f"<div style='font-size:30px; font-weight:800; color:{regime_color};'>"
                f"{regime_emoji} {regime}</div>"
                f"<div style='font-size:13px; color:#ccc; margin-top:6px;'>"
                f"Growth: <b>{buckets.get('growth', 0)}/4</b> | "
                f"Plumbing: <b>{buckets.get('plumbing', 0)}/4</b> | "
                f"Credit: <b>{buckets.get('credit', 0)}/3</b> | "
                f"Liquidity z: <b>{_ud_macro.get('liquidity', {}).get('z', 0):+.2f}</b> | "
                f"Vetoes: <b style='color:{'#ef4444' if vetoes else '#ccc'};'>{len(vetoes)}"
                f"{' · ' + ', '.join(str(v) for v in vetoes) if vetoes else ''}</b>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

            # === Allocation summary ===
            t = _ud_macro.get("target_allocation_pct", {})
            n = _ud_macro.get("target_allocation_nzd", {})
            ca = _ud_macro.get("current_allocation_pct", {})
            st.markdown("#### Allocation")
            _aeq = t.get("equity", 0) or 0
            _abt = t.get("btc", 0) or 0
            _asg = t.get("staging", 0) or 0
            _atot = max(1, _aeq + _abt + _asg)
            st.markdown(
                "<div style='display:flex; height:26px; border-radius:6px; overflow:hidden; "
                "margin:2px 0 8px;'>"
                f"<div style='width:{_aeq / _atot * 100:.1f}%; background:#ef4444; display:flex; "
                f"align-items:center; justify-content:center; font-size:11px; color:#fff; "
                f"font-weight:600;'>{_aeq:.0f}% eq</div>"
                f"<div style='width:{_abt / _atot * 100:.1f}%; background:#f0b90b; display:flex; "
                f"align-items:center; justify-content:center; font-size:11px; color:#000; "
                f"font-weight:600;'>{_abt:.0f}% BTC</div>"
                f"<div style='width:{_asg / _atot * 100:.1f}%; background:#22c55e; display:flex; "
                f"align-items:center; justify-content:center; font-size:11px; color:#000; "
                f"font-weight:600;'>{_asg:.0f}% cash</div></div>",
                unsafe_allow_html=True)
            _ac = st.columns(3)
            with _ac[0]:
                st.metric("Equity Target",
                            f"{t.get('equity', 0):.0f}%",
                            (f"{_money(n.get('equity', 0))} (current {ca.get('equity', 0):.0f}%)" if SHOW_PERSONAL else f"current {ca.get('equity', 0):.0f}%"))
            with _ac[1]:
                st.metric("BTC Target",
                            f"{t.get('btc', 0):.0f}%",
                            (f"{_money(n.get('btc', 0))} (current {ca.get('btc', 0):.0f}%)" if SHOW_PERSONAL else f"current {ca.get('btc', 0):.0f}%"))
            with _ac[2]:
                st.metric("Staging (cash) Target",
                            f"{t.get('staging', 0):.0f}%",
                            (f"{_money(n.get('staging', 0))} (current {ca.get('staging', 0):.0f}%)" if SHOW_PERSONAL else f"current {ca.get('staging', 0):.0f}%"))

            sb = _ud_macro.get("staging_basket_pct", {})
            sn = _ud_macro.get("staging_basket_nzd", {})
            if sb:
                st.markdown("#### Staging Basket (BIL / VTIP / GLDM)")
                _bc = st.columns(3)
                with _bc[0]:
                    st.metric("BIL (T-bills)", f"{sb.get('BIL', 0)}%", _money(sn.get('BIL', 0)) if SHOW_PERSONAL else None)
                with _bc[1]:
                    st.metric("VTIP (TIPS)", f"{sb.get('VTIP', 0)}%", _money(sn.get('VTIP', 0)) if SHOW_PERSONAL else None)
                with _bc[2]:
                    st.metric("GLDM (Gold)", f"{sb.get('GLDM', 0)}%", _money(sn.get('GLDM', 0)) if SHOW_PERSONAL else None)
                st.caption(_ud_macro.get("staging_basket_rationale", ""))

        else:
            st.caption("Macro regime + allocation pending — populates on the next precompute.")

        st.divider()

        # === Early Rotation Signal ===
        st.markdown("#### ⚡ Early Rotation Signal (Druckenmiller/PTJ/Zulauf framework)")
        _er = _gc("early_rotation")
        if _er:
            action = _er.get("action", "?")
            color = ("#22c55e" if action == "HOLD" else
                      "#f0b90b" if action in ("WATCH", "REDUCE_TO_CASH") else "#ef4444")
            st.markdown(
                f"<div style='padding:14px 18px; border-radius:8px; "
                f"background:#13161c; border-left:4px solid {color};'>"
                f"<b style='color:{color}; font-size:18px;'>{action}</b> "
                f"<span style='color:#aaa;'>({_er.get('n_firing', 0)}/{_er.get('n_total', 7)} firing)</span><br>"
                f"<span style='color:#aaa; font-size:12px;'>{_er.get('rationale', '')[:140]}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            with st.expander("All 7 leading rotation indicators"):
                inds = _er.get("indicators", {})
                rows = [{"✓": "🔥" if v.get("firing") else "○",
                            "Indicator": k.replace("_", " ").title(),
                            "Status": v.get("status", "?")[:90]}
                          for k, v in inds.items()]
                if rows: st.dataframe(pd.DataFrame(rows),
                                       width='stretch', hide_index=True)

        else:
            st.caption("Early rotation signal pending — populates on the next precompute.")
        st.divider()

        # === Macro Rotation Tracker ===
        st.markdown("#### 🔄 Macro Rotation Tracker (equities → BTC)")
        _rot = _gc("rotation")
        if _rot and not _rot.get("error"):
            phase = _rot.get("phase") or "PRE_ROTATION"
            color = ("#888" if phase == "PRE_ROTATION" else
                      "#f0b90b" if phase == "WATCH" else
                      "#22c55e" if phase in ("ACTIVE", "AGGRESSIVE") else "#22c55e")
            st.markdown(
                f"<div style='padding:14px 18px; border-radius:8px; "
                f"background:#13161c; border-left:4px solid {color};'>"
                f"<b style='color:{color}; font-size:18px;'>{phase.replace('_',' ')}</b><br>"
                f"<span style='color:#aaa; font-size:12px;'>"
                f"SPY DD: {_rot.get('spy_drawdown', 0) or 0:+.1f}% | "
                f"BTC DD: {_rot.get('btc_drawdown', 0) or 0:+.1f}% | "
                f"BTC-SPY corr: {_rot.get('btc_spy_corr', 0) or 0:+.2f} | "
                f"Liquidity: {_rot.get('liquidity_phase', '?')}</span></div>",
                unsafe_allow_html=True,
            )

        else:
            st.caption("Rotation tracker pending — populates on the next precompute.")
        st.divider()

        # === Macro Layer Signals (8 leading indicators) ===
        st.markdown("#### 🌐 Macro Layer — 8 leading indicators")
        _ml = _gc("macro_layer")
        if _ml:
            _signals = _ml.get("signals", {}) if isinstance(_ml, dict) else {}
            if _signals:
                _ml_tiles = []
                for _name, _sig in list(_signals.items())[:8]:
                    if not isinstance(_sig, dict):
                        continue
                    _sval = _sig.get("value", _sig.get("score", "?"))
                    _ssig = str(_sig.get("signal", "?"))
                    _scol = ("#22c55e" if _ssig.upper() in ("BULL", "RISK_ON", "POSITIVE", "GREEN") else
                             "#ef4444" if _ssig.upper() in ("BEAR", "RISK_OFF", "NEGATIVE", "RED") else
                             "#f0b90b")
                    try:
                        _sval_str = f"{float(_sval):.2f}" if isinstance(_sval, (int, float)) else str(_sval)[:12]
                    except Exception:
                        _sval_str = "?"
                    _ml_tiles.append(
                        f"<div style='flex:1 1 calc(25% - 8px); min-width:150px; padding:9px 11px; "
                        f"border-radius:6px; background:{_scol}1f; border:1px solid {_scol}66;'>"
                        f"<div style='font-size:9px; color:#999; text-transform:uppercase; "
                        f"letter-spacing:.3px;'>{_name.replace('_', ' ')}</div>"
                        f"<div style='font-size:15px; color:#fff; font-weight:700;'>{_sval_str}</div>"
                        f"<div style='font-size:10px; color:{_scol}; font-weight:600;'>{_ssig}</div></div>")
                st.markdown(
                    f"<div style='display:flex; flex-wrap:wrap; gap:8px;'>{''.join(_ml_tiles)}</div>",
                    unsafe_allow_html=True)
            else:
                st.caption("Macro layer signals not in expected format.")

        else:
            st.caption("Macro layer pending — populates on the next precompute.")
        st.divider()

        # === ETF Regime ===
        st.markdown("#### 💸 ETF Flow Regime")
        _etfr = _gc("etf_regime")
        if _etfr:
            _eregime = _etfr.get("regime", "?")
            _eflow_60d = _etfr.get("net_flow_60d_b", 0) or 0
            _eflow_z = _etfr.get("flow_z_60d", 0) or 0
            _ecolor = ("#22c55e" if _eregime in ("ACCUMULATION", "INFLOW", "STRONG_INFLOW") else
                        "#ef4444" if _eregime in ("DISTRIBUTION", "OUTFLOW", "STRONG_OUTFLOW") else
                        "#f0b90b")
            _etf_c1, _etf_c2 = st.columns([3, 2])
            with _etf_c1:
                st.markdown(
                    f"<div style='padding:14px 18px; border-radius:8px; "
                    f"background:#13161c; border-left:4px solid {_ecolor}; height:150px;'>"
                    f"<b style='color:{_ecolor}; font-size:20px;'>{_eregime}</b><br>"
                    f"<span style='color:#aaa; font-size:12px;'>60-day net ETF flow: "
                    f"<b style='color:#ccc;'>${_eflow_60d:+,.2f}B</b></span><br>"
                    f"<span style='color:#888; font-size:11px;'>z-score = how unusual that flow is "
                    f"vs history (0 = normal, &gt;0 = heavy buying)</span></div>",
                    unsafe_allow_html=True,
                )
            with _etf_c2:
                _ez = max(-3.0, min(3.0, float(_eflow_z)))
                _ezf = go.Figure(go.Indicator(
                    mode="gauge+number", value=round(float(_eflow_z), 2),
                    number={"font": {"size": 22, "color": "#fff"}, "valueformat": "+.2f"},
                    title={"text": "flow z-score", "font": {"size": 12, "color": "#d4d4d4"}},
                    gauge={
                        "axis": {"range": [-3, 3], "tickvals": [-3, 0, 3],
                                 "tickfont": {"size": 9, "color": "#888"}, "tickcolor": "#888"},
                        "bar": {"color": "rgba(255,255,255,0.9)", "thickness": 0.22},
                        "bgcolor": "#0e1117", "borderwidth": 0,
                        "steps": [
                            {"range": [-3, -0.5], "color": "rgba(239,68,68,0.55)"},
                            {"range": [-0.5, 0.5], "color": "rgba(240,185,11,0.40)"},
                            {"range": [0.5, 3], "color": "rgba(34,197,94,0.55)"}],
                        "threshold": {"line": {"color": "#fff", "width": 3},
                                      "thickness": 0.9, "value": _ez}}))
                _ezf.update_layout(paper_bgcolor="#0e1117", height=150,
                                   margin=dict(l=10, r=10, t=30, b=4), font=dict(color="#d4d4d4"))
                st.plotly_chart(_ezf, width='stretch',
                                config={"displayModeBar": False, "displaylogo": False})

        else:
            st.caption("ETF flow regime pending — populates on the next precompute.")
        st.divider()

        # === Predictor Engine Theme Composites ===
        st.markdown("#### 🧠 Theme Composites (6 themes)")
        _pe_mac = _gc("predictor_engine")
        if _pe_mac:
            _themes = _pe_mac.get("theme_composites", {}) or {}
            if _themes:
                _tnames, _tvals = [], []
                for _tname, _tval in list(_themes.items())[:6]:
                    try:
                        _tvals.append(float(_tval) if _tval is not None else 0.0)
                    except Exception:
                        _tvals.append(0.0)
                    _tnames.append(_tname.replace("_", " ").title())
                _ord = sorted(range(len(_tvals)), key=lambda i: _tvals[i])
                _tnames = [_tnames[i] for i in _ord]
                _tvals = [_tvals[i] for i in _ord]
                _tcols = ["#22c55e" if v >= 0 else "#ef4444" for v in _tvals]
                _tmax = max((abs(v) for v in _tvals), default=1) or 1
                _tcf = go.Figure(go.Bar(
                    x=_tvals, y=_tnames, orientation="h",
                    marker=dict(color=_tcols, line=dict(color="#0e1117", width=1)),
                    text=[f"{v:+.2f}" for v in _tvals], textposition="outside",
                    textfont=dict(size=11, color="#cccccc"), hoverinfo="skip"))
                _tcf.add_vline(x=0, line=dict(color="#888888", width=1))
                _tcf.update_layout(**CHART_LAYOUT, height=max(230, 40 * len(_tvals) + 50),
                                   xaxis=dict(range=[-_tmax * 1.35, _tmax * 1.35],
                                              gridcolor="#2a2d34", zeroline=False),
                                   yaxis=dict(automargin=True), showlegend=False)
                st.plotly_chart(_tcf, width='stretch',
                                config={"displayModeBar": False, "displaylogo": False})
                st.caption("Each theme's tilt — green = bullish, red = bearish (longer bar = stronger).")

        else:
            st.caption("Theme composites pending — populates on the next precompute.")
        st.divider()

        # === BTC Dominance gauge (macro context) ===
        try:
            _dials = _gc("swift_dials")
            if _dials and _dials.get("btc_dominance"):
                st.markdown("#### 📊 BTC Dominance (alt rotation context)")
                st.plotly_chart(_dials["btc_dominance"], width='stretch',
                                  config={"displayModeBar": False, "scrollZoom": False,
                                          "doubleClick": False, "displaylogo": False})
        except Exception: pass

    except Exception as _e:
        st.warning(f"Macro panels — temporarily unavailable")

    st.caption(
        "📌 The Macro Drivers cards and Unified Decision detail panel are on the Signals tab."
    )


# ─────────────────────────────────────────────────────────────
# 🚪 EXIT PLAN TAB — BTC top scale-out ladder (next bull, ~2029)
# Parked at the bottom of Playbook: years away, zero day-to-day relevance.
# Email alerts fire on tier changes regardless of this tab.
# ─────────────────────────────────────────────────────────────
with tab_exit:
    st.markdown("### 🚪 Exit Plan — when and how to scale OUT of BTC")
    st.caption(
        "The exit-side twin of the rotation trigger. Dormant through the bear — "
        "arms automatically when BTC returns to within 15% of its 365-day high "
        "(projected next-bull window ~2029, halving Apr 2028 + ~535d). "
        "**You never need to check this section** — tier escalations are flagged "
        "URGENT in the hourly alert email."
    )
    try:
        from core.dashboard_cache import get_cached as _gc_ex
        _so = _gc_ex("scale_out_trigger")
        if not _so:
            from core.scale_out_trigger import evaluate_scale_out_trigger
            _so = evaluate_scale_out_trigger()

        _so_tier = _so.get("tier", "?")
        _so_color = _so.get("color", "#888")
        _so_action = (_so.get("action") or "")
        _so_pct = _so.get("pct_below_high")
        _so_n = _so.get("top_n_met", 0)
        _so_tot = _so.get("top_n_total", 16)
        _so_th = _so.get("thresholds", {}) or {}
        _so_emoji = {"DORMANT": "💤", "ARMED": "🟢", "TRIM_25": "🟡",
                      "SCALE_OUT_50": "🔴", "EXIT_75": "🚨"}.get(_so_tier, "❓")

        # Hero status
        st.markdown(
            f"<div style='padding:18px 22px; border-radius:10px; background:#13161c; "
            f"border:3px solid {_so_color}; margin-bottom:14px;'>"
            f"<div style='display:flex; justify-content:space-between; align-items:baseline;'>"
            f"<span style='font-size:11px; color:#888; text-transform:uppercase; "
            f"letter-spacing:1.5px; font-weight:700;'>"
            f"🔔 BTC TOP Scale-Out Trigger</span>"
            f"<span>{_age_badge('scale_out_trigger')}</span></div>"
            f"<div style='font-size:34px; font-weight:800; color:{_so_color}; "
            f"line-height:1.1; margin-top:4px;'>{_so_emoji} {_so_tier.replace('_', ' ')}</div>"
            f"<div style='font-size:13px; color:#ccc; margin-top:8px; line-height:1.5;'>"
            f"{_so_action}</div>"
            f"</div>", unsafe_allow_html=True,
        )
        _ath_fire = bool(_so.get("ath_stagnation"))
        _ols_bear = bool(_so.get("olson_bearish"))
        _ec = st.columns(4)
        with _ec[0]:
            st.markdown(metric_card("BTC vs 365d high",
                        f"{_so_pct:+.0f}%" if _so_pct is not None else "?",
                        "how far below the peak", _so_color), unsafe_allow_html=True)
        with _ec[1]:
            st.markdown(metric_card("Top scorecard", f"{_so_n}/{_so_tot}",
                        "sell-signals firing", "#f0b90b"), unsafe_allow_html=True)
        with _ec[2]:
            st.markdown(metric_card("ATH stagnation", "FIRING" if _ath_fire else "quiet",
                        "60d-before-top tell", "#ef4444" if _ath_fire else "#22c55e"),
                        unsafe_allow_html=True)
        with _ec[3]:
            st.markdown(metric_card("Olson 3wk MACD", "BEARISH" if _ols_bear else "not bearish",
                        "called the cycle-5 top", "#ef4444" if _ols_bear else "#22c55e"),
                        unsafe_allow_html=True)

        # === Exit thermometer — current price vs the 'armed' band ===
        _so_price = _so.get("btc_price") or btc_price or 0
        _so_high = _so.get("high_365d") or 0
        if _so_high and _so_price:
            _armed = _so_high * 0.85
            _ylo = min(_so_price, _armed) * 0.88
            _yhi = _so_high * 1.07
            _thf = go.Figure()
            _thf.add_hrect(y0=_ylo, y1=_armed, fillcolor="rgba(90,120,160,0.10)", line_width=0)
            _thf.add_hrect(y0=_armed, y1=_yhi, fillcolor="rgba(240,185,11,0.10)", line_width=0)
            _thf.add_hline(y=_so_high, line=dict(color="#ef4444", width=2, dash="dot"),
                           annotation_text=f"365-day high  ${_so_high:,.0f}",
                           annotation_position="top left",
                           annotation_font=dict(size=11, color="#ef4444"))
            _thf.add_hline(y=_armed, line=dict(color="#f0b90b", width=2, dash="dot"),
                           annotation_text=f"plan arms here — within 15% of high  ${_armed:,.0f}",
                           annotation_position="top left",
                           annotation_font=dict(size=11, color="#f0b90b"))
            _thf.add_trace(go.Scatter(
                x=[0.5], y=[_so_price], mode="markers+text",
                marker=dict(size=20, color="#ffffff", symbol="diamond",
                            line=dict(color="#000000", width=1)),
                text=[(f"  ● BTC NOW ${_so_price:,.0f}  ({_so_pct:+.0f}% vs high)"
                       if _so_pct is not None else f"  ● BTC NOW ${_so_price:,.0f}")],
                textposition="middle right", textfont=dict(size=13, color="#ffffff"),
                hoverinfo="skip", showlegend=False))
            _thf.update_layout(**CHART_LAYOUT, height=360, showlegend=False,
                               yaxis=dict(title="Price (USD)", gridcolor="#2a2d34",
                                          range=[_ylo, _yhi]),
                               xaxis=dict(visible=False, range=[0, 4]))
            st.plotly_chart(_thf, width='stretch',
                            config={"displayModeBar": False, "displaylogo": False})
            _zone_word = ("deep in the DORMANT 'sleep' zone" if _so_price < _armed
                          else "in the ARMED watch zone — the selling rules are live")
            if _so_pct is not None:
                st.caption(
                    f"BTC is **{abs(_so_pct):.0f}% below** its 365-day high — {_zone_word}. "
                    f"The exit plan doesn't even **arm** until BTC climbs back within 15% of the high "
                    f"(~${_armed:,.0f}); until then there's nothing to do here.")
            else:
                st.caption(f"The exit plan arms when BTC climbs within 15% of its 365-day high "
                           f"(~${_armed:,.0f}).")
        st.write("")

        # The ladder
        st.markdown("#### The exit ladder (cycle-scaled thresholds)")
        st.caption(f"Top scorecard is at **{_so_n}/{_so_tot}** — progress toward each sell tier:")
        for _tlbl, _tthr, _tcol in [
            ("🟡 TRIM 25%", _so_th.get("trim", 3), "#f0b90b"),
            ("🔴 SCALE OUT 50%", _so_th.get("scale_out", 4), "#ef4444"),
            ("🚨 EXIT 75%", _so_th.get("exit", 6), "#b91c1c"),
        ]:
            _pc = st.columns([2, 5])
            with _pc[0]:
                st.markdown(
                    f"<div style='font-size:12px; color:#ccc; padding-top:3px;'>{_tlbl} "
                    f"<span style='color:#888;'>({int(_so_n)} of {_tthr})</span></div>",
                    unsafe_allow_html=True)
            with _pc[1]:
                st.markdown(_seg_bar(_so_n, _tthr, _tcol, height=13), unsafe_allow_html=True)
        st.write("")
        _ladder_rows = [
            ("💤 DORMANT", "BTC >15% below 365d high", "Sleep — never trims a bear/recovery", "—"),
            ("🟢 ARMED", "Within 15% of the high", "Ride the bull, watch weekly", "—"),
            ("🟡 TRIM 25", f"Scorecard ≥{_so_th.get('trim', 3)}/16 OR ATH-stagnation fires",
              "Sell 25% of BTC", "First tranche off the table"),
            ("🔴 SCALE OUT 50", f"Scorecard ≥{_so_th.get('scale_out', 4)}/16 AND Olson 3wk MACD bearish",
              "Sell 50% of BTC", "Scorecard + technicals both confirm"),
            ("🚨 EXIT 75", f"Scorecard ≥{_so_th.get('exit', 6)}/16, or stagnation+Olson+scorecard together",
              "Sell 75% of BTC", "Overwhelming cycle-top evidence"),
        ]
        _lrows = [{"Tier": r[0], "Trigger": r[1], "Action": r[2], "Meaning": r[3]}
                   for r in _ladder_rows]
        st.dataframe(pd.DataFrame(_lrows), width='stretch', hide_index=True)

        st.markdown(
            "<div style='padding:10px 14px; margin-top:8px; border-radius:6px; "
            "background:rgba(74,144,226,0.12); border-left:3px solid #4a90e2;'>"
            "<div style='font-size:12px; color:#ccc; line-height:1.5;'>"
            "<b>Why phased (vs the single-shot rotation)?</b> Bottoms are processes "
            "(months long) — one decisive entry works. Tops are events (blow-off weeks) — "
            "tranching out captures the spike without round-tripping it.<br>"
            "<b>Why trust it?</b> This signal stack was backtested against the real "
            "Oct 2025 peak: every classic indicator stayed silent in the muted ETF cycle, "
            "but the ATH-stagnation detector fired <b>60 days before the top</b> — which is "
            "why it has its own express lane into TRIM 25. The 50% tranche requires Olson's "
            "3-week MACD, the signal that called the cycle-5 top correctly. Thresholds "
            "auto-scale with the cycle-era detector, so a muted cycle-6 bull lowers the bar."
            "</div></div>",
            unsafe_allow_html=True,
        )

        # Top scorecard detail
        with st.expander(f"📋 BTC TOP scorecard detail — {_so_n}/{_so_tot} firing", expanded=False):
            _nt_exit = _gc_ex("btc_native_top_scorecard") or {}
            _crit = _nt_exit.get("criteria", [])
            if _crit:
                _erows = [{"✓": "🔥" if c.get("met") else "○",
                            "Criterion": c.get("label", "?"),
                            "Status": (c.get("status", "?") or "")[:90]}
                           for c in _crit]
                st.dataframe(pd.DataFrame(_erows), width='stretch', hide_index=True)
            else:
                st.caption("top scorecard detail pending — refreshes with precompute")
    except Exception as _exe:
        st.caption(f"Exit Plan — temporarily unavailable")


# === Footer ===
st.markdown(
    "<div style='text-align:center; margin-top:30px; font-size:11px; color:#888;'>"
    f"BTC Prediction Engine • Last update: {state.get('as_of', '?')[:19]} • "
    "Cache: 4h disk + 5min Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
