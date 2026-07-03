"""Crypto paper-trading dashboard — exchange-style layout.

Run:  streamlit run dashboard.py --server.port 8510
URL:  http://localhost:8510

Pulls live market data from Binance public endpoints. No auth needed.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from streamlit_autorefresh import st_autorefresh

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from core import (
    attribution, correlation_monitor, data, hmm_regime, monte_carlo,
    portfolio_risk, realized_pnl, regime, stress_test,
)
from ops import circuit_breaker, position_monitor
# Production orchestrator strategies (v2)
from strategies import (
    diverse_mom_ethbtc,
    funding_basis,
    short_term_momentum,
    tsmom,
    vol_breakout,
)
# Research-only strategies (kept on disk, not in orchestrator)
from strategies import (
    btc_dominance,
    long_short_ratio,
    open_interest,
    stablecoin_supply,
    tsmom_v3,
    xs_momentum,
)

# Watchlist = union of orchestrator universe + pro_trend universe so the
# dashboard renders every pair we could have an open position on.
from run import UNIVERSE as _ORCHESTRATOR_UNIVERSE
from strategies.pro_trend import PRO_TREND_PAIRS as _PRO_TREND_UNIVERSE
from strategies.xsmom import XSMOM_UNIVERSE as _XSMOM_UNIVERSE


def _active_position_pairs() -> set[str]:
    """Pairs with open positions across all sleeves (state files + attribution).

    Picks up discretionary force-entries (NEAR), xsmom positions (ATOM, LINK),
    and any orphans not in the standard universe sets.
    """
    pairs: set[str] = set()
    # pro_trend state files
    for f in REPO_ROOT.glob(".pro_trend_state_*.json"):
        try:
            st = json.loads(f.read_text())
        except Exception:
            continue
        if st.get("units"):
            base = f.stem.removeprefix(".pro_trend_state_")
            pairs.add(f"{base}/USDT")
    # pnl_attribution sleeve tags
    attrib_file = REPO_ROOT / ".pnl_attribution.json"
    if attrib_file.exists():
        try:
            attrib = json.loads(attrib_file.read_text())
            for key in attrib:
                base_pair = key.removeprefix("xsmom:").removeprefix("basis:")
                if "/" in base_pair:
                    pairs.add(base_pair)
        except Exception:
            pass
    return pairs


WATCHLIST = sorted(
    set(_ORCHESTRATOR_UNIVERSE)
    | set(_PRO_TREND_UNIVERSE)
    | set(_XSMOM_UNIVERSE)
    | _active_position_pairs()
)
TIMEFRAMES = {"1h": 168, "4h": 168, "1d": 365, "1w": 200}


# ===== Cached data fetches =====
@st.cache_data(ttl=30)
def fetch_watchlist():
    """Fetch tickers for all WATCHLIST pairs. Always falls back to per-pair
    fetches for any pair the bulk endpoint missed (so positions never end up
    with last=0 if the asset is tradeable on Binance)."""
    out: dict = {}
    try:
        bulk = data._EX.fetch_tickers(WATCHLIST)
        out.update({p: t for p, t in bulk.items() if t})
    except Exception:
        pass
    for p in WATCHLIST:
        if p not in out or not out[p].get("last"):
            try:
                out[p] = data._EX.fetch_ticker(p)
            except Exception:
                pass
    return out


@st.cache_data(ttl=10)
def fetch_order_book(pair: str, depth: int = 20):
    return data._EX.fetch_order_book(pair, limit=depth)


@st.cache_data(ttl=15)
def fetch_recent_trades(pair: str, limit: int = 30):
    return data._EX.fetch_trades(pair, limit=limit)


@st.cache_data(ttl=60)
def fetch_ohlcv(pair: str, timeframe: str, limit: int):
    raw = data._EX.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.set_index("ts")


@st.cache_data(ttl=60)
def load_paper_state():
    """2026-05-31 W9 refactor: aggregate across all spot sub-account files.

    Returns a merged dict in the legacy schema: {cash_quote, positions, _sleeves}.
    Sums cash + positions per asset across all spot sub-accounts.
    Adds _sleeves field with per-sleeve breakdown.
    """
    sleeve_files = sorted(REPO_ROOT.glob(".paper_state_*.json"))
    # Exclude backups
    sleeve_files = [f for f in sleeve_files if "legacy_backup" not in f.name]
    if not sleeve_files:
        # Fallback to legacy file if no sub-accounts yet
        f = REPO_ROOT / ".paper_state.json"
        return json.loads(f.read_text()) if f.exists() else None

    merged = {"cash_quote": 0.0, "quote_currency": "USDT", "positions": {}, "_sleeves": {}}
    for sf in sleeve_files:
        try:
            d = json.loads(sf.read_text())
        except Exception:
            continue
        sleeve_name = sf.stem.replace(".paper_state_", "")
        merged["cash_quote"] += float(d.get("cash_quote", 0))
        for asset, qty in d.get("positions", {}).items():
            if abs(qty) > 1e-9:
                merged["positions"][asset] = merged["positions"].get(asset, 0) + qty
        merged["_sleeves"][sleeve_name] = {
            "cash": float(d.get("cash_quote", 0)),
            "positions": {k: v for k, v in d.get("positions", {}).items() if abs(v) > 1e-9},
        }
    return merged


@st.cache_data(ttl=60)
def load_perp_state():
    """W9 refactor: aggregate across all perp sub-account files."""
    sleeve_files = sorted(REPO_ROOT.glob(".paper_perp_state_*.json"))
    sleeve_files = [f for f in sleeve_files if "legacy_backup" not in f.name]
    if not sleeve_files:
        f = REPO_ROOT / ".paper_perp_state.json"
        return json.loads(f.read_text()) if f.exists() else None

    merged = {"cash_quote": 0.0, "quote_currency": "USDT", "positions": {},
              "entry_prices": {}, "_sleeves": {}}
    for sf in sleeve_files:
        try:
            d = json.loads(sf.read_text())
        except Exception:
            continue
        sleeve_name = sf.stem.replace(".paper_perp_state_", "")
        merged["cash_quote"] += float(d.get("cash_quote", 0))
        for asset, qty in d.get("positions", {}).items():
            if abs(qty) > 1e-9:
                merged["positions"][asset] = merged["positions"].get(asset, 0) + qty
        for asset, price in d.get("entry_prices", {}).items():
            if asset not in merged["entry_prices"]:
                merged["entry_prices"][asset] = price
        merged["_sleeves"][sleeve_name] = {
            "cash": float(d.get("cash_quote", 0)),
            "positions": {k: v for k, v in d.get("positions", {}).items() if abs(v) > 1e-9},
        }
    return merged


@st.cache_data(ttl=60)
def load_trades():
    f = REPO_ROOT / "paper_trades.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


@st.cache_data(ttl=60)
def load_daily_status():
    f = REPO_ROOT / "daily_status.csv"
    if not f.exists():
        return pd.DataFrame()
    return pd.read_csv(f, parse_dates=["ts"])


@st.cache_data(ttl=60)
def load_evidence():
    f = REPO_ROOT / ".evidence_ledger.jsonl"
    if not f.exists():
        return []
    out = []
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


@st.cache_data(ttl=300)
def fetch_signals():
    """Return signals from PRODUCTION strategies (in orchestrator) and
    research strategies (off-rotation, just for visibility).
    """
    production = ["diverse_mom_ethbtc", "tsmom", "short_term_momentum",
                  "vol_breakout", "funding_basis"]
    out = {}
    for mod, name in [
        (diverse_mom_ethbtc, "diverse_mom_ethbtc"),
        (tsmom, "tsmom"),
        (short_term_momentum, "short_term_momentum"),
        (vol_breakout, "vol_breakout"),
        (funding_basis, "funding_basis"),
        (tsmom_v3, "tsmom_v3 [research]"),
        (xs_momentum, "xs_momentum [research]"),
        (open_interest, "open_interest [research]"),
        (long_short_ratio, "long_short_ratio [research]"),
        (stablecoin_supply, "stablecoin_supply [research]"),
        (btc_dominance, "btc_dominance [research]"),
    ]:
        try:
            out[name] = float(mod.latest_signal())
        except Exception:
            out[name] = float("nan")
    return out


@st.cache_data(ttl=600)
def fetch_hmm_state(pair: str = "BTC/USDT"):
    try:
        return hmm_regime.fit_hmm_2state(pair, days_back=730)
    except Exception:
        return {"converged": False}


@st.cache_data(ttl=3600)
def fetch_stress_test():
    try:
        from research import signals as res_sig
        sig_fn = lambda p: res_sig.tsmom_multi(p, horizons=(30, 90, 180))
        return stress_test.stress_test_strategy(sig_fn)
    except Exception:
        return None


@st.cache_data(ttl=60)
def fetch_realized_summary():
    try:
        return realized_pnl.realized_summary()
    except Exception:
        return None


@st.cache_data(ttl=300)
def fetch_regime():
    return regime.overall("BTC/USDT")


@st.cache_data(ttl=120)
def fetch_funding_rate(pair: str):
    """Funding rate for the perp version of a spot pair (BTC/USDT -> BTC/USDT:USDT)."""
    perp = f"{pair}:{pair.split('/')[1]}"
    try:
        return data._EX.fetch_funding_rate(perp)
    except Exception:
        return None


def compute_rsi(prices, period: int = 14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def compute_macd(prices, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - sig_line
    return macd_line, sig_line, hist


def compute_bbands(prices, window: int = 20, n_std: float = 2.0):
    mid = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    return upper, mid, lower


def compute_position_summary(
    spot_state: dict,
    perp_state: dict | None,
    trades_df: pd.DataFrame,
    watch_data: dict,
) -> list[dict]:
    """For each open position (spot AND perp), compute entry, current, MTM, P&L,
    risk levels (stop, take-profit, trailing high-water mark), broker tag."""
    out: list[dict] = []

    hwm_file = REPO_ROOT / ".position_hwm.json"
    hwm = json.loads(hwm_file.read_text()) if hwm_file.exists() else {}

    # ----- Spot positions (FIFO entry from trades) -----
    if spot_state:
        quote = spot_state.get("quote_currency", "USDT")
        for asset, qty in spot_state.get("positions", {}).items():
            if abs(qty) < 1e-9:
                continue
            pair = f"{asset}/{quote}"
            last = watch_data.get(pair, {}).get("last") or 0.0
            avg_entry = None
            if not trades_df.empty:
                asset_trades = trades_df[trades_df["pair"].astype(str) == pair].copy()
                if not asset_trades.empty:
                    signed_qty = asset_trades.apply(
                        lambda r: float(r["qty"]) if r["side"] == "buy" else -float(r["qty"]),
                        axis=1,
                    )
                    wsum = (signed_qty.abs() * asset_trades["price"].astype(float)).sum()
                    vsum = signed_qty.abs().sum()
                    if vsum > 0:
                        avg_entry = float(wsum / vsum)
            out.append(_position_row(asset, pair, qty, last, avg_entry, hwm, broker="spot"))

    # ----- Perp positions (entry price from perp_state) -----
    if perp_state:
        quote = perp_state.get("quote_currency", "USDT")
        positions = perp_state.get("positions", {})
        entries = perp_state.get("entry_prices", {})
        funding = perp_state.get("accumulated_funding", {})
        for asset, qty in positions.items():
            if abs(qty) < 1e-9:
                continue
            pair = f"{asset}/{quote}"
            last = watch_data.get(pair, {}).get("last") or 0.0
            avg_entry = entries.get(asset)
            row = _position_row(asset, pair, qty, last, avg_entry, hwm, broker="perp")
            row["accumulated_funding"] = funding.get(asset, 0.0)
            out.append(row)

    return out


def _position_row(asset, pair, qty, last, avg_entry, hwm, broker):
    stop_pct = position_monitor.STOP_LOSS_PCT
    tp_pct = position_monitor.TAKE_PROFIT_PCT
    notional = qty * last if last else 0.0
    direction = 1 if qty > 0 else -1
    if avg_entry and avg_entry > 0 and last and last > 0:
        pnl_quote = qty * (last - avg_entry)
        pnl_pct = (last / avg_entry - 1.0) * direction
        stop_price = avg_entry * (1 - direction * stop_pct) if stop_pct else None
        tp_price = avg_entry * (1 + direction * tp_pct) if tp_pct else None
        dist_to_stop = ((last - stop_price) / last * direction * 100) if stop_price else None
    else:
        pnl_quote = pnl_pct = 0.0
        stop_price = tp_price = dist_to_stop = None
    return {
        "asset": asset,
        "pair": pair,
        "broker": broker,
        "direction": "LONG" if qty > 0 else "SHORT",
        "qty": qty,
        "avg_entry": avg_entry or 0.0,
        "current": last,
        "notional": notional,
        "pnl": pnl_quote,
        "pnl_pct": pnl_pct,
        "stop_price": stop_price,
        "tp_price": tp_price,
        "trail_hwm": hwm.get(asset),
        "dist_to_stop_pct": dist_to_stop,
    }


# ===== Page config =====
st.set_page_config(
    page_title="Crypto Paper Rig (port 8510)",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Auto-refresh every 30 seconds for live tickers + positions
st_autorefresh(interval=5_000, key="dashboard_autorefresh")  # 5s for real-time feel


# ===== REAL-TIME CONTROL ROOM (top of dashboard) =====
def _load_live_prices():
    p = REPO_ROOT / ".live_prices.json"
    if not p.exists():
        return None, None
    try:
        d = json.loads(p.read_text())
        meta = d.pop("_meta", None) if "_meta" in d else None
        flushed_at = None
        if meta:
            flushed_at = datetime.fromisoformat(meta["flushed_at"])
        return d, flushed_at
    except Exception:
        return None, None


_live_prices, _flushed_at = _load_live_prices()
_now_utc = datetime.now(timezone.utc)
_feed_age_sec = (_now_utc - _flushed_at).total_seconds() if _flushed_at else None

# Real-time status bar
_rt_bar_cols = st.columns([2, 2, 2, 2, 2])
with _rt_bar_cols[0]:
    if _feed_age_sec is None:
        st.error("🔴 WS feed: never started")
    elif _feed_age_sec < 5:
        st.success(f"🟢 WS feed: LIVE ({_feed_age_sec:.1f}s old)")
    elif _feed_age_sec < 30:
        st.warning(f"🟡 WS feed: lagging ({_feed_age_sec:.0f}s old)")
    else:
        st.error(f"🔴 WS feed: STALE ({_feed_age_sec:.0f}s old)")

with _rt_bar_cols[1]:
    # Real-time monitor process status
    rt_log = REPO_ROOT / ".ws_feed.log"
    if rt_log.exists():
        _log_age = (_now_utc.timestamp() - rt_log.stat().st_mtime)
        if _log_age < 120:
            st.success(f"🟢 Monitor: alive ({_log_age:.0f}s)")
        else:
            st.warning(f"🟡 Monitor: idle ({_log_age:.0f}s)")
    else:
        st.info("Monitor: not started")

with _rt_bar_cols[2]:
    # Kill switch state
    if (REPO_ROOT / ".kill_switch.json").exists():
        st.error("🚨 KILL SWITCH ACTIVE")
    else:
        st.success("✅ No kill triggered")

with _rt_bar_cols[3]:
    # Walk-forward lock
    wf_lock = REPO_ROOT / ".walk_forward_lock.json"
    if wf_lock.exists():
        try:
            wf = json.loads(wf_lock.read_text())
            end_dt = datetime.fromisoformat(wf["end_date"])
            days_left = max(0, (end_dt - _now_utc).days)
            st.info(f"📊 Walk-fwd: {days_left}d left")
        except Exception:
            st.info("Walk-fwd: ?")
    else:
        st.warning("Walk-fwd: not started")

with _rt_bar_cols[4]:
    # Multi-exchange halt
    if (REPO_ROOT / ".multi_exchange_halt.json").exists():
        st.error("⚠️ Cross-venue divergence")
    else:
        st.success("✅ Venues aligned")


# Real-time live prices panel
if _live_prices:
    with st.expander(f"📈 Real-time prices ({len(_live_prices)} pairs, refreshed every 1s by WS feed)", expanded=True):
        rt_rows = []
        for pair, d in sorted(_live_prices.items()):
            if pair.startswith("_") or not isinstance(d, dict):
                continue
            bid = d.get("bid", 0)
            ask = d.get("ask", 0)
            mid = d.get("mid", (bid + ask) / 2 if bid and ask else 0)
            ts = d.get("ts", 0)
            age = _now_utc.timestamp() - ts if ts else None
            spread_bps = ((ask - bid) / mid * 10000) if mid > 0 else 0
            rt_rows.append({
                "Pair": pair,
                "Bid": f"${bid:,.4f}" if bid < 100 else f"${bid:,.2f}",
                "Ask": f"${ask:,.4f}" if ask < 100 else f"${ask:,.2f}",
                "Spread (bps)": f"{spread_bps:.2f}",
                "Age (s)": f"{age:.1f}" if age is not None else "n/a",
            })
        st.dataframe(pd.DataFrame(rt_rows), use_container_width=True, hide_index=True, height=380)
        st.caption("Source: Binance WebSocket bookTicker streams. Updated by `realtime_monitor.py` background service.")
else:
    st.warning("Real-time price feed not running. Start: `schtasks /Run /TN Crypto_realtime_monitor`")


# ===== P&L OVERVIEW — top of main page (W12) =====
st.divider()
st.subheader("💰 P&L Overview — combined paper account")

# Book Truth — the honest state of the book (effective bets, data sufficiency).
# Reads before any P&L detail: how many INDEPENDENT bets is the book really
# running, and can we even assess edge yet? Cache-first, fail-safe.
try:
    from core.dashboard_cache import get_cached as _bt_get
    _bt = _bt_get("book_truth")
    if _bt is None:
        from core.book_truth import compute as _bt_compute
        _bt = _bt_compute()
    if isinstance(_bt, dict) and _bt.get("status") == "ok":
        _btc = st.columns(4)
        _btc[0].metric("Effective bets", f"{_bt['effective_bets']:.1f}",
                       f"of {_bt['n_sleeves']} sleeves", delta_color="off")
        _btc[1].metric("Sleeves active",
                       f"{_bt['n_sleeves_active']}/{_bt['n_sleeves']}", delta_color="off")
        _btc[2].metric("Portfolio DD", f"{_bt['portfolio_drawdown']*100:+.1f}%",
                       f"net long {_bt['net_long_exposure']*100:.0f}%", delta_color="off")
        _btc[3].metric("Can assess edge?",
                       "yes" if _bt["can_assess_edge"] else "NOT YET", delta_color="off")
        st.caption("⚖️ " + _bt["verdict"])
except Exception:
    pass

def _live_price(asset):
    """Get latest price for asset, prefer WS cache, fall back to watchlist."""
    if _live_prices:
        pair = f"{asset}/USDT"
        d = _live_prices.get(pair)
        if d and isinstance(d, dict):
            return d.get("mid", 0)
    try:
        return fetch_watchlist().get(f"{asset}/USDT", {}).get("last", 0)
    except Exception:
        return 0

_spot_state = load_paper_state() or {}
_perp_state = load_perp_state() or {}

# Spot equity per sleeve
_spot_sleeves = _spot_state.get("_sleeves", {})
_spot_per_sleeve = {}
_spot_total = 0.0
for sname, sdata in _spot_sleeves.items():
    cash = sdata.get("cash", 0)
    pos_mtm = 0
    for asset, qty in sdata.get("positions", {}).items():
        px = _live_price(asset)
        if px > 0:
            pos_mtm += qty * px
    eq = cash + pos_mtm
    _spot_per_sleeve[sname] = {"cash": cash, "pos_mtm": pos_mtm, "equity": eq}
    _spot_total += eq

# Perp equity per sleeve (cash + open PnL relative to entry)
_perp_sleeves_data = {}
_perp_total = 0.0
for psf in sorted(REPO_ROOT.glob(".paper_perp_state_*.json")):
    if "legacy_backup" in psf.name:
        continue
    try:
        d = json.loads(psf.read_text())
    except Exception:
        continue
    sname = psf.stem.replace(".paper_perp_state_", "")
    cash = float(d.get("cash_quote", 0))
    entries = d.get("entry_prices", {})
    open_pnl = 0.0
    for asset, qty in d.get("positions", {}).items():
        if abs(qty) < 1e-9:
            continue
        px = _live_price(asset)
        if px > 0:
            entry = entries.get(asset, px)
            open_pnl += qty * (px - entry)
    eq = cash + open_pnl
    _perp_sleeves_data[sname] = {"cash": cash, "open_pnl": open_pnl, "equity": eq}
    _perp_total += eq

_combined = _spot_total + _perp_total
_started = 200_000.0  # 2 x $100k buckets
_combined_pnl = _combined - _started
_combined_pnl_pct = _combined_pnl / _started

# Big top-line metrics
_pnl_cols = st.columns(4)
with _pnl_cols[0]:
    st.metric("💼 Combined Equity", f"${_combined:,.2f}",
              f"{_combined_pnl:+,.2f} USD ({_combined_pnl_pct*100:+.2f}%)",
              delta_color="normal" if _combined_pnl >= 0 else "inverse")
with _pnl_cols[1]:
    _spot_pnl = _spot_total - 100_000
    st.metric("Spot equity", f"${_spot_total:,.2f}",
              f"{_spot_pnl:+,.2f} ({_spot_pnl/1000:+.2f}%)",
              delta_color="normal" if _spot_pnl >= 0 else "inverse")
with _pnl_cols[2]:
    _perp_pnl = _perp_total - 100_000
    st.metric("Perp equity", f"${_perp_total:,.2f}",
              f"{_perp_pnl:+,.2f} ({_perp_pnl/1000:+.2f}%)",
              delta_color="normal" if _perp_pnl >= 0 else "inverse")
with _pnl_cols[3]:
    n_sleeves = len(_spot_per_sleeve) + len(_perp_sleeves_data)
    st.metric("Active sleeves", f"{n_sleeves}", "isolated sub-accounts", delta_color="off")

# Per-sleeve breakdown table
st.markdown("##### Per-sleeve breakdown (sorted by P&L)")
_pnl_rows = []
for sname, d in _spot_per_sleeve.items():
    # Pick reasonable starting equity from sleeve_circuit_breakers if known
    starting = {"bah_btc": 10_000, "oversold_bounce": 15_000, "orchestrator": 30_000,
                "spot_reserve": 45_000, "grid_trader": 10_000,
                "intraday_momentum": 10_000, "pro_trend": 30_000}.get(sname, 10_000)
    pnl = d["equity"] - starting
    _pnl_rows.append({
        "Venue": "spot",
        "Sleeve": sname,
        "Equity": f"${d['equity']:,.2f}",
        "Cash": f"${d['cash']:,.0f}",
        "Position MTM": f"${d['pos_mtm']:,.0f}",
        "P&L $": f"{pnl:+,.2f}",
        "P&L %": f"{pnl/max(starting, 1)*100:+.2f}%",
        "_pnl_num": pnl,
    })
for sname, d in _perp_sleeves_data.items():
    starting = {"xsmom": 30_000, "pro_trend": 30_000, "overbought_fade": 10_000,
                "basis_arb": 20_000, "perp_reserve": 10_000,
                "intraday_momentum_short": 10_000}.get(sname, 10_000)
    pnl = d["equity"] - starting
    _pnl_rows.append({
        "Venue": "perp",
        "Sleeve": sname,
        "Equity": f"${d['equity']:,.2f}",
        "Cash": f"${d['cash']:,.0f}",
        "Position MTM": f"${d['open_pnl']:,.0f}",
        "P&L $": f"{pnl:+,.2f}",
        "P&L %": f"{pnl/max(starting, 1)*100:+.2f}%",
        "_pnl_num": pnl,
    })
_pnl_df = pd.DataFrame(_pnl_rows).sort_values("_pnl_num", ascending=False).drop(columns=["_pnl_num"])

def _pnl_color(val):
    s = str(val)
    if s.startswith("+") and s.endswith("%"):
        try:
            v = float(s.rstrip("%"))
            return f"color: {'#26a69a' if v > 0 else '#aaaaaa'}; font-weight: bold"
        except Exception:
            return ""
    if s.startswith("-") and s.endswith("%"):
        return "color: #ef5350; font-weight: bold"
    if s.startswith("+") and not s.endswith("%"):
        return "color: #26a69a"
    if s.startswith("-") and not s.endswith("%"):
        return "color: #ef5350"
    return ""

styled_pnl = _pnl_df.style.map(_pnl_color, subset=["P&L $", "P&L %"])
st.dataframe(styled_pnl, use_container_width=True, hide_index=True, height=420)
st.caption(f"Combined: spot ${_spot_total:,.2f} + perp ${_perp_total:,.2f} = ${_combined:,.2f}. Started: $200,000. Live MTM from WebSocket cache.")


# ===== Individual positions with LONG/SHORT direction (W12 enhanced) =====
st.markdown("##### Individual positions with direction")
_indiv_rows = []
# Spot positions are always LONG
for sname, sdata in _spot_sleeves.items():
    for asset, qty in sdata.get("positions", {}).items():
        if abs(qty) < 1e-6:
            continue
        px = _live_price(asset)
        notional = qty * px if px > 0 else 0
        _indiv_rows.append({
            "Sleeve": sname,
            "Venue": "spot",
            "Asset": asset,
            "Direction": "LONG",
            "Qty": f"{qty:+.6f}",
            "Current $": f"${px:,.4f}" if px else "?",
            "Notional": f"${notional:+,.2f}",
            "_dir_sort": 1,
        })
# Perp positions can be LONG or SHORT
for psf in sorted(REPO_ROOT.glob(".paper_perp_state_*.json")):
    if "legacy_backup" in psf.name:
        continue
    try:
        d = json.loads(psf.read_text())
    except Exception:
        continue
    sname = psf.stem.replace(".paper_perp_state_", "")
    entries = d.get("entry_prices", {})
    for asset, qty in d.get("positions", {}).items():
        if abs(qty) < 1e-9:
            continue
        px = _live_price(asset)
        entry = entries.get(asset, px)
        pnl = qty * (px - entry) if px else 0
        notional = qty * px if px else 0
        direction = "LONG " if qty > 0 else "SHORT"
        _indiv_rows.append({
            "Sleeve": sname,
            "Venue": "perp",
            "Asset": asset,
            "Direction": direction,
            "Qty": f"{qty:+.4f}",
            "Current $": f"${px:,.4f}" if px else "?",
            "Notional": f"${notional:+,.2f}",
            "Entry $": f"${entry:,.4f}" if entry else "?",
            "Open P&L": f"${pnl:+,.2f}",
            "_dir_sort": 1 if qty > 0 else -1,
        })

if _indiv_rows:
    _ind_df = pd.DataFrame(_indiv_rows).sort_values(["_dir_sort", "Sleeve"], ascending=[False, True]).drop(columns=["_dir_sort"])

    def _color_direction(val):
        if isinstance(val, str):
            if "LONG" in val:
                return "color: #26a69a; font-weight: bold; background-color: rgba(38,166,154,0.1)"
            if "SHORT" in val:
                return "color: #ef5350; font-weight: bold; background-color: rgba(239,83,80,0.1)"
        return ""

    def _color_pnl_indiv(val):
        s = str(val)
        if s.startswith("+$") or s.startswith("+"):
            return "color: #26a69a; font-weight: bold"
        if s.startswith("-$") or s.startswith("-"):
            return "color: #ef5350; font-weight: bold"
        return ""

    styled_ind = _ind_df.style.map(_color_direction, subset=["Direction"])
    if "Open P&L" in _ind_df.columns:
        styled_ind = styled_ind.map(_color_pnl_indiv, subset=["Open P&L"])
    st.dataframe(styled_ind, use_container_width=True, hide_index=True, height=420)

    _n_long = sum(1 for r in _indiv_rows if "LONG" in r["Direction"])
    _n_short = sum(1 for r in _indiv_rows if "SHORT" in r["Direction"])
    st.caption(f"📈 {_n_long} LONG positions | 📉 {_n_short} SHORT positions | Direct from sub-account state files.")
else:
    st.info("No open positions across any sleeve.")


# ===== Sidebar: account + controls =====
with st.sidebar:
    st.title("🪙 Paper Account")
    state = load_paper_state()
    if state:
        cash = float(state.get("cash_quote", 0))
        positions = state.get("positions", {})
        quote = state.get("quote_currency", "USDT")
        watch = fetch_watchlist()
        equity = cash
        for asset, qty in positions.items():
            pair = f"{asset}/{quote}"
            last = watch.get(pair, {}).get("last")
            if last:
                equity += qty * last
        pnl_pct = (equity - 100_000.0) / 100_000.0
        st.metric(f"Equity ({quote})", f"${equity:,.0f}", f"{pnl_pct:+.2%}")
        st.metric(f"Cash ({quote})", f"${cash:,.0f}")
        if positions:
            st.write("**Positions:**")
            for asset, qty in positions.items():
                if abs(qty) < 1e-9:
                    continue
                pair = f"{asset}/{quote}"
                last = watch.get(pair, {}).get("last", 0)
                value = qty * last
                if qty > 0:
                    st.markdown(
                        f"- **<span style='color:#26a69a'>LONG</span> {asset}**: "
                        f"{qty:.6f} (≈ ${value:,.0f})",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"- **<span style='color:#ef5350'>SHORT</span> {asset}**: "
                        f"{qty:.6f} (exposure ≈ ${value:,.0f})",
                        unsafe_allow_html=True,
                    )
        else:
            st.caption("No open positions")
    else:
        st.warning("No paper state file")

    st.divider()
    st.caption(f"UTC: {datetime.now(timezone.utc).strftime('%H:%M:%S')}")
    if st.button("🔄 Refresh all", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.caption("Mode: **PAPER**")
    st.caption("Broker: Binance public API")
    st.caption("Data: ccxt — no keys required")


# ===== Main: title row =====
st.title("Crypto Markets")
st.caption("Live data from Binance public REST API. Paper-trading rig — no live capital at risk.")


# ===== Circuit breaker status banner =====
kill_file = REPO_ROOT / ".kill_switch.json"
peak_file = REPO_ROOT / ".peak_equity.json"
if kill_file.exists():
    kill_info = json.loads(kill_file.read_text())
    st.error(
        f"🛑 **CIRCUIT BREAKER ACTIVE** — drawdown {kill_info.get('drawdown_pct', 0):.1%} "
        f"triggered liquidation. Trading disabled. Inspect `.kill_switch.json` "
        f"and call `ops.circuit_breaker.reset_kill_switch()` to resume."
    )
elif peak_file.exists():
    peak_info = json.loads(peak_file.read_text())
    peak = peak_info.get("peak", 100_000.0)

# ===== Forward Distribution Envelope (sim vs live) =====
try:
    from ops.dashboard_components import (
        forward_envelope_chart, current_sim_percentile_rank,
    )
    st.subheader("Live equity vs forward Monte Carlo distribution")
    st.caption(
        "Inside the band = normal performance. Below P5 = degradation signal. "
        "Sim = 70/30 portfolio, 50% Sharpe haircut, block bootstrap."
    )
    fig_env = forward_envelope_chart(start_equity=100_000.0)
    if fig_env is not None:
        st.plotly_chart(fig_env, use_container_width=True)
    pct_ranks = current_sim_percentile_rank(start_equity=100_000.0)
    if pct_ranks and "error" not in pct_ranks and "status" not in pct_ranks:
        cols = st.columns(len(pct_ranks))
        for col, (h, info) in zip(cols, pct_ranks.items()):
            verdict = ("STRONG" if info['percentile'] > 75
                       else "OK" if info['percentile'] > 25
                       else "BELOW" if info['percentile'] > 5 else "FAR BELOW")
            col.metric(
                f"{h}-day live percentile",
                f"P{info['percentile']:.0f}",
                f"{info['live_return']:+.1%} vs sim P50 {info['sim_p50']:+.1%}",
            )
    elif pct_ranks.get("status") == "no_equity_log":
        st.info("Equity log empty — comparator activates after first daily snapshot.")
except Exception as _e:
    st.caption(f"Forward-distribution chart unavailable: {_e}")

# ===== Live Positions (top, prominent) =====
st.subheader("Open Positions — live mark-to-market with stops")
trades_for_basis = load_trades()
perp_state = load_perp_state()
positions_summary = compute_position_summary(state, perp_state, trades_for_basis, watch)
if positions_summary:
    rows = []
    for p in positions_summary:
        # Format stop/tp/trail
        stop_str = f"${p['stop_price']:,.2f}" if p["stop_price"] else "-"
        tp_str = f"${p['tp_price']:,.2f}" if p["tp_price"] else "off"
        trail_str = f"${p['trail_hwm']:,.2f}" if p["trail_hwm"] else "-"
        dist_str = f"{p['dist_to_stop_pct']:+.2f}%" if p["dist_to_stop_pct"] is not None else "-"

        rows.append({
            "Asset": p["asset"],
            "Broker": p.get("broker", "spot").upper(),
            "Direction": p["direction"],
            "Qty": f"{p['qty']:+.8f}",
            "Avg Entry": f"${p['avg_entry']:,.4f}" if p["avg_entry"] else "-",
            "Current": f"${p['current']:,.4f}",
            "Stop": stop_str,
            "Notional": f"${p['notional']:+,.2f}",
            "Unrealized P&L": f"${p['pnl']:+,.2f}",
            "P&L %": f"{p['pnl_pct']:+.2%}",
            "_dir": p["direction"],
            "_pnl": p["pnl"],
        })
    pos_df = pd.DataFrame(rows)

    def _color_dir(val):
        if val == "LONG":
            return "color: #26a69a; font-weight: bold"
        if val == "SHORT":
            return "color: #ef5350; font-weight: bold"
        return ""

    def _color_pnl(val):
        if isinstance(val, str) and ("+" in val or "-" in val):
            return ("color: #26a69a; font-weight: bold" if val.startswith("+")
                    else "color: #ef5350; font-weight: bold")
        return ""

    styled_pos = (
        pos_df.drop(columns=["_dir", "_pnl"])
        .style
        .map(_color_dir, subset=["Direction"])
        .map(_color_pnl, subset=["Unrealized P&L", "P&L %"])
    )
    st.dataframe(styled_pos, use_container_width=True, hide_index=True)
    st.caption(
        f"Stop loss at {position_monitor.STOP_LOSS_PCT:.0%} from entry. "
        f"Trailing stop at {position_monitor.TRAILING_STOP_PCT:.0%} from peak (only when in profit). "
        f"Take-profit "
        + (f"at {position_monitor.TAKE_PROFIT_PCT:.0%}" if position_monitor.TAKE_PROFIT_PCT else "OFF")
        + ". Portfolio circuit breaker at "
        f"{circuit_breaker.KILL_DD_PCT:.0%} equity drawdown."
    )

    # Aggregate P&L + drawdown vs peak
    total_pnl = sum(p["pnl"] for p in positions_summary)
    total_notional = sum(abs(p["notional"]) for p in positions_summary)
    cash_now = float(state.get("cash_quote", 0))
    equity_now = cash_now + sum(p["notional"] for p in positions_summary)
    peak_equity = json.loads(peak_file.read_text()).get("peak", 100_000.0) if peak_file.exists() else equity_now
    dd_now = max(0.0, 1 - equity_now / peak_equity) if peak_equity > 0 else 0
    dd_color = "normal" if dd_now < 0.10 else "off"

    a, b, c, d = st.columns(4)
    a.metric("Total Unrealized P&L", f"${total_pnl:+,.0f}", f"{total_pnl/100_000:+.2%} of bankroll")
    b.metric("Gross Exposure", f"${total_notional:,.0f}", f"{total_notional/100_000:.1%} of bankroll")
    c.metric("Equity / Peak", f"${equity_now:,.0f}", f"-{dd_now:.2%} from ${peak_equity:,.0f}", delta_color=dd_color)
    n_long = sum(1 for p in positions_summary if p["direction"] == "LONG")
    n_short = sum(1 for p in positions_summary if p["direction"] == "SHORT")
    d.metric("Position Mix", f"{n_long}L / {n_short}S", f"of {len(positions_summary)} open")
else:
    st.info("No open positions. Run `python run.py` to fire the orchestrator.")

st.divider()

# ===== W15 — Canon (Douglas / Livermore / Lopez de Prado / Hayes) =====
st.subheader("🧠 Canon — Process, confidence, locks, key levels")
_canon_cols = st.columns([1, 1, 1])

with _canon_cols[0]:
    st.markdown("**Sleeve scaling — meta-conf × Kelly × dominance** (composed gates)")
    try:
        from core.meta_confidence import CONFIDENCE_FUNCS, get_meta_confidence
        from core.kelly_sizing import kelly_multiplier
        _mc_rows = []
        for sleeve_name in CONFIDENCE_FUNCS:
            mc = get_meta_confidence(sleeve_name)
            ks = kelly_multiplier(sleeve_name)
            # Decide which dominance scale applies: bah_btc/pro_trend = BTC-regime,
            # the rest are alt-leaning.
            try:
                from core.btc_dominance import alt_regime_scale, btc_regime_scale
                if sleeve_name in ("bah_btc", "pro_trend"):
                    dom = btc_regime_scale()
                else:
                    dom = alt_regime_scale()
            except Exception:
                dom = 1.0
            composed = mc * ks * dom
            if composed >= 1.3:
                emoji = "🟢"; tag = "STRONG"
            elif composed >= 0.9:
                emoji = "⚪"; tag = "normal"
            elif composed >= 0.5:
                emoji = "🟡"; tag = "weak"
            else:
                emoji = "🔴"; tag = "VERY WEAK"
            _mc_rows.append({
                "Sleeve": sleeve_name,
                "Meta": f"{mc:.2f}x",
                "Kelly": f"{ks:.2f}x",
                "BTC.D": f"{dom:.2f}x",
                "Composed": f"{composed:.2f}x",
                "State": f"{emoji} {tag}",
            })
        _mc_df = pd.DataFrame(_mc_rows)
        st.dataframe(_mc_df, use_container_width=True, hide_index=True, height=260)
        st.caption("Composed scale = meta × Kelly × dominance. Combined with min(CB, Sharpe, streak, corr, event, tail, var) inside get_all_gates_scale.")
    except Exception as _e:
        st.error(f"composed gates panel failed: {_e}")

with _canon_cols[1]:
    st.markdown("**Process compliance** (Douglas: 'track process, not P&L')")
    try:
        from ops.process_compliance import compute_daily_score
        comp = compute_daily_score()
        oc = comp.get("overall_compliance")
        if oc is not None:
            st.metric("Overall compliance", f"{oc*100:.0f}%",
                      f"{comp.get('verdict', '?')}", delta_color="off")
        else:
            st.info("No data yet today")
        st.metric("Manual overrides today", f"{comp.get('n_manual_overrides', 0)}",
                  delta_color="off")
        if comp.get("per_sleeve"):
            _pc_rows = []
            for sname, d in comp["per_sleeve"].items():
                _pc_rows.append({
                    "Sleeve": sname,
                    "Signals": d["signals"],
                    "Trades": d["trades"],
                    "Compliance": f"{d['compliance']*100:.0f}%" if d.get("compliance") is not None else "n/a",
                })
            st.dataframe(pd.DataFrame(_pc_rows), use_container_width=True, hide_index=True, height=160)
    except Exception as _e:
        st.error(f"process_compliance failed: {_e}")

with _canon_cols[2]:
    st.markdown("**Loss acceptance locks** (Douglas + Livermore)")
    try:
        from ops.loss_acceptance_lock import status as lock_status
        locks = lock_status()
        active_locks = {k: v for k, v in locks.items() if v.get("locked")}
        if active_locks:
            st.error(f"🔒 {len(active_locks)} sleeve(s) in cooldown")
            for s, d in active_locks.items():
                st.write(f"**{s}** — expires {d['expires_at'][:19]}")
        else:
            st.success("✅ No active cooldowns")
        st.caption("Triggers on -1% loss in 24h. 48h cooldown. Wait before adjusting.")
    except Exception:
        pass

st.markdown("##### BTC key levels + macro regime (Glassnode + Hayes)")
_kl_cols = st.columns(2)
with _kl_cols[0]:
    try:
        from core.btc_key_levels import get_status as btc_status
        bs = btc_status()
        st.metric("BTC Regime", bs.get("regime", "?"),
                  f"${bs.get('price', 0):,.0f}", delta_color="off")
        if bs.get("sth_mvrv_proxy") is not None:
            st.caption(f"STH-MVRV proxy: {bs['sth_mvrv_proxy']:.2f} | regime: {bs.get('sth_mvrv_regime', '?')}")
        for action in bs.get("actions", [])[:3]:
            st.caption(f"→ {action[:120]}")
    except Exception:
        pass

with _kl_cols[1]:
    try:
        from core.macro_correlation import regime_status as macro_status
        ms = macro_status()
        regime = ms.get("regime", "?").upper()
        color_map = {"NORMAL": "🟢", "CAUTION": "🟡", "DE-RISK": "🟠", "FULL_KILL": "🔴"}
        st.metric("Macro Regime (Hayes)", f"{color_map.get(regime, '⚪')} {regime}",
                  f"de_risk_level={ms.get('de_risk_level', 0)}", delta_color="off")
        for flag in ms.get("flags", [])[:3]:
            st.caption(f"! {flag[:120]}")
    except Exception:
        pass

# BTC alpha vs QQQ — "rolling bubbles" regime gauge (Evanss6-inspired, 2026-07).
# The informative variable is BTC's ALPHA vs the Nasdaq/AI trade (went negative
# in 2025-26), NOT its correlation (~0.5, useless). Cache-first (precomputed),
# live-compute fallback, fully fail-safe.
try:
    from core.dashboard_cache import get_cached
    _ar = get_cached("btc_alpha_regime")
    if _ar is None:
        from core.btc_alpha_regime import compute as _ar_compute
        _ar = _ar_compute()
    if isinstance(_ar, dict) and _ar.get("status") == "ok":
        st.markdown("##### BTC alpha vs QQQ — rolling-bubbles regime")
        _ar_emoji = {"CRYPTO STARVED": "🔴", "TRANSITION": "🟡",
                     "CRYPTO LEADING": "🟢"}.get(_ar["regime"], "⚪")
        _arc = st.columns(3)
        _arc[0].metric(f"{_ar_emoji} BTC alpha vs QQQ",
                       f"{_ar['alpha_annual_90d']:+.0%}",
                       f"90d ann · β {_ar['beta_90d']:.2f} · corr {_ar['corr_90d']:.2f}",
                       delta_color="off")
        _arc[1].metric("BTC/QQQ ratio", _ar["btc_qqq_ratio_trend"],
                       f"{_ar['btc_qqq_ratio_vs_1y_avg']:+.0%} vs 1y avg",
                       delta_color="off")
        _arc[2].metric("ETH/BTC (alt bid)", f"{_ar['ethbtc']:.4f}",
                       _ar["ethbtc_trend"], delta_color="off")
        st.caption(_ar["summary"])
except Exception:
    pass

# BTC PREDICTION MACHINE — top-of-dashboard regime banner + forecast panel
try:
    from core.btc_prediction import state_of_btc
    _bp = state_of_btc()
    _regime = _bp.get("regime", "?")
    _short = _bp["horizons"]["short_term"]

    # Top banner with regime
    _regime_emoji = {
        "BULL": "🟢", "RANGE_BULL": "🟢",
        "RANGE": "🟡",
        "RANGE_BEAR": "🟠", "BEAR": "🔴",
    }.get(_regime, "⚪")
    _banner_msg = (
        f"{_regime_emoji} **BTC PREDICTION** — REGIME: {_regime}  |  "
        f"Short-term: {_short['interpretation']} ({_short['direction_score']:+.2f}, {_short['confidence']} conf)"
    )
    if _regime in ("BEAR", "RANGE_BEAR"):
        st.error(_banner_msg)
    elif _regime == "RANGE":
        st.info(_banner_msg)
    else:
        st.success(_banner_msg)

    # Forecast panel
    with st.expander(f"📊 BTC Prediction — multi-horizon forecast ({_bp['btc_price']:,.0f})", expanded=False):
        _fcols = st.columns(4)
        _horizons_meta = [
            ("intraday", "Intraday (1d)"),
            ("short_term", "Short (1-30d)"),
            ("medium_term", "Medium (1-6m)"),
            ("long_term", "Long (6m-2y)"),
        ]
        for col, (h, label) in zip(_fcols, _horizons_meta):
            hd = _bp["horizons"][h]
            t = _bp["price_targets"].get(h, {})
            score_color = "off"
            score_arrow = ""
            if hd["direction_score"] > 0.2:
                score_arrow = "↑"
                score_color = "normal"
            elif hd["direction_score"] < -0.2:
                score_arrow = "↓"
                score_color = "inverse"
            col.metric(
                label,
                f"{hd['interpretation']}",
                f"{score_arrow} {hd['direction_score']:+.2f}  ({hd['confidence']})",
                delta_color=score_color,
            )
            if t:
                col.caption(f"P25 ${t['p25']:,.0f} - P75 ${t['p75']:,.0f}\nmedian ${t['median']:,.0f}")

        # 3-lens ensemble
        _ens = _bp.get("ensemble", {})
        if _ens:
            st.markdown("**3-Lens Ensemble** — independent predictor consensus")
            _ens_cols = st.columns(3)
            _lenses = list(_ens.get("lenses", {}).items())
            for col, (lens_name, ld) in zip(_ens_cols, _lenses):
                label = lens_name.replace("_lens", "").title()
                score = ld["score"]
                if score > 0.2: emoji = "🟢"
                elif score < -0.2: emoji = "🔴"
                else: emoji = "⚪"
                col.metric(f"{emoji} {label}",
                           ld["interpretation"],
                           f"{score:+.2f} ({ld['n_signals']} signals)",
                           delta_color="off")
            _consensus = _ens.get("consensus", "?")
            if "UNANIMOUS" in _consensus and "BULL" in _consensus:
                st.success(f"⚡ Ensemble consensus: {_consensus}")
            elif "UNANIMOUS" in _consensus and "BEAR" in _consensus:
                st.error(f"⚡ Ensemble consensus: {_consensus}")
            else:
                st.info(f"Ensemble consensus: {_consensus}")

        # Category breakdown table
        st.markdown("**Signal categories** (all 11)")
        _bp_rows = []
        for cat in ("technical", "onchain", "sentiment", "derivatives",
                    "macro", "liquidations", "cycle", "flows", "options_adv",
                    "fundamentals", "regime_models"):
            br = _bp["horizons"]["short_term"]["breakdown"].get(cat, {})
            score = br.get("score", 0)
            if score > 0.3: status = "🟢 BULL"
            elif score > 0.05: status = "🟢 mild bull"
            elif score > -0.05: status = "⚪ neutral"
            elif score > -0.3: status = "🔴 mild bear"
            else: status = "🔴 BEAR"
            _bp_rows.append({
                "Category": cat,
                "Score": f"{score:+.2f}",
                "Direction": status,
                "Signals": f"{br.get('n_scored', 0)}/{br.get('n_total', 0)}",
            })
        st.dataframe(pd.DataFrame(_bp_rows), use_container_width=True, hide_index=True)

        # Hit rates (when data available)
        _hr = _bp.get("hit_rates_by_horizon", {})
        if _hr:
            st.markdown("**Hit rates** (rolling — adaptive learning data)")
            _hr_rows = []
            for h, d in _hr.items():
                if d.get("n_observations", 0) >= 3:
                    _hr_rows.append({
                        "Horizon": h,
                        "Correct": d["n_correct"],
                        "Total": d["n_observations"],
                        "Hit Rate": f"{d['hit_rate']*100:.0f}%",
                    })
            if _hr_rows:
                st.dataframe(pd.DataFrame(_hr_rows), use_container_width=True, hide_index=True)

        # Anomalies
        _anom = _bp.get("signal_anomalies", [])
        if _anom:
            st.warning(f"⚠️ {len(_anom)} signal anomaly/anomalies detected — possible regime change")
            for a in _anom[:5]:
                st.caption(f"  {a['signal']}: z={a['z_score']:+.2f} — {a['interpretation']}")

        st.caption("Cached 4h. Run `python btc_predict.py --force` to refresh now. 43+ signals across 11 categories.")
except Exception as _e:
    pass

# Cycle-top percentile detector banner (backtest 94% capture at 2025 peak)
try:
    from core.cycle_top_percentile import cycle_top_score
    _cts = cycle_top_score()
    if not _cts.get("error"):
        _verdict = _cts.get("verdict", "?")
        _score = _cts.get("score", 0)
        if _verdict == "PEAK_ZONE":
            st.error(f"🚨 **CYCLE-TOP DETECTOR: PEAK_ZONE ({_score}/100)** — {_cts.get('action', '')}")
        elif _verdict == "DISTRIBUTION_ZONE":
            st.error(f"⚠️ **Cycle-top detector: DISTRIBUTION ({_score}/100)** — {_cts.get('action', '')}")
        elif _verdict == "ELEVATED":
            st.warning(f"📊 **Cycle-top detector: ELEVATED ({_score}/100)** — {_cts.get('action', '')}")
        elif _verdict == "CAUTION":
            st.info(f"📊 Cycle-top detector: CAUTION ({_score}/100) — {_cts.get('action', '')}")
        elif _verdict == "NOT_NEAR_ATH":
            pass  # don't show in bear regime — too noisy
except Exception:
    pass

# Empirical regime gate banner (2026-06-01 — backtest-calibrated)
try:
    from ops.regime_gate import current_regime, PAUSE_IN_BEAR, should_pause_sleeve
    _r = current_regime()
    _emoji = {"bull": "🟢", "chop": "🟡", "bear": "🔴", "unknown": "⚪"}.get(_r["regime"], "⚪")
    _paused = [s for s in sorted(PAUSE_IN_BEAR) if should_pause_sleeve(s)["should_pause"]]
    _msg = f"{_emoji} **Empirical regime gate** — {_r['label']}"
    if _paused:
        _msg += f"  |  PAUSED: {', '.join(_paused)}"
    else:
        _msg += "  |  All regime-gated sleeves ACTIVE"
    if _r["regime"] == "bear":
        st.error(_msg)
    elif _r["regime"] == "chop":
        st.warning(_msg)
    else:
        st.success(_msg)
except Exception:
    pass

# VaR breach soft-throttle banner (W16.F + plumbing)
_var_breach_file = REPO_ROOT / ".var_kupiec_breach.json"
if _var_breach_file.exists():
    try:
        _vb = json.loads(_var_breach_file.read_text())
        _vb_ts = datetime.fromisoformat(_vb["timestamp"])
        _vb_age_d = (datetime.now(timezone.utc) - _vb_ts).total_seconds() / 86400
        if _vb_age_d < 3:
            st.error(
                f"🟥 **VaR Kupiec breach active** — observed {_vb.get('n_breaches_observed','?')} / "
                f"expected {_vb.get('n_breaches_expected', 0):.1f}. All sleeves throttled to "
                f"**0.5×** for {3 - _vb_age_d:.1f}d. Recalibrate 1% VaR or accept fat-tail empirical VaR."
            )
    except Exception:
        pass

# ===== W16 — Composite signal panel (F&G + BTC.D + tail hedge) =====
st.markdown("##### W16 composite signals — sentiment, dominance, hedge")
_w16_cols = st.columns(3)

with _w16_cols[0]:
    st.markdown("**Fear & Greed Index** (alternative.me)")
    try:
        from core.fear_greed import latest as fg_latest, cycle_composite_score
        fg = fg_latest()
        if fg.get("value") is not None:
            val = fg["value"]
            cls = fg.get("classification", "?")
            chg = fg.get("7d_change", 0)
            color = ("🟢" if val <= 25 else "🟡" if val <= 45 else
                     "⚪" if val <= 55 else "🟠" if val <= 75 else "🔴")
            st.metric(f"{color} F&G", f"{val} / 100", f"{cls} ({chg:+d} 7d)", delta_color="off")
            comp = cycle_composite_score()
            if comp.get("composite_score") is not None:
                st.caption(f"Composite: {comp['composite_score']:.0f}/100 — {comp.get('composite_action', '?')[:90]}")
            else:
                st.caption(fg.get("action", "")[:90])
        else:
            st.info("F&G unavailable")
    except Exception as _e:
        st.error(f"F&G failed: {_e}")

with _w16_cols[1]:
    st.markdown("**BTC dominance** (CoinGecko)")
    try:
        from core.btc_dominance import status as dom_status
        ds = dom_status()
        if not ds.get("error"):
            dot = {
                "BTC_HEGEMONY": "🔴", "BTC_DOMINANT": "🟠", "BALANCED": "⚪",
                "ALT_RECOVERY": "🟢", "ALTSEASON": "🟢",
            }.get(ds.get("regime", "?"), "⚪")
            st.metric(f"{dot} BTC.D", f"{ds['btc_dominance_pct']:.2f}%",
                      ds.get("regime", "?"), delta_color="off")
            st.caption(f"Alt-sleeve gate: {ds.get('alt_scale', 1.0):.2f}x  |  "
                       f"BTC-sleeve gate: {ds.get('btc_scale', 1.0):.2f}x")
            st.caption(ds.get("action", "")[:120])
        else:
            st.info("BTC.D unavailable")
    except Exception as _e:
        st.error(f"BTC.D failed: {_e}")

with _w16_cols[2]:
    st.markdown("**Tail hedge** (Deribit BTC puts)")
    try:
        from core.tail_hedge import compute_hedge_recommendation
        h = compute_hedge_recommendation(bankroll_usd=200_000)
        urgency = h.get("urgency", "?")
        dot = {
            "critical": "🔴", "recommended": "🟠", "optional": "🟡",
            "unnecessary": "🟢",
        }.get(urgency, "⚪")
        st.metric(f"{dot} Urgency", urgency.upper(),
                  f"{h.get('risk_factor_count', 0)}/6 risk factors", delta_color="off")
        if urgency in ("critical", "recommended"):
            st.caption(f"Budget: {h.get('max_premium_pct_of_bankroll', 0)*100:.2f}% of bankroll")
            ss = h.get("suggested_structure")
            if ss:
                st.caption(f"Strike ${ss.get('strike', 0):,.0f} / {ss.get('expiry_days', '?')}d on {ss.get('venue', '?')}")
        else:
            for r in h.get("reasoning", [])[:2]:
                st.caption(f"• {r[:100]}")
    except Exception as _e:
        st.error(f"Tail hedge failed: {_e}")

# Liquidation pressure table
st.markdown("##### Liquidation cascade pressure (W16.A)")
try:
    from core.liquidation_pressure import liquidation_pressure as lp_fn
    _lp_rows = []
    for _p in ["BTC/USDT", "ETH/USDT", "SOL/USDT"]:
        try:
            r = lp_fn(_p)
            _lp_rows.append({
                "Pair": _p,
                "P(long cascade)": f"{r.get('cascade_long_probability', 0)*100:.0f}%",
                "P(short squeeze)": f"{r.get('cascade_short_probability', 0)*100:.0f}%",
                "Funding 8h": f"{r.get('funding_bps_8h', 0):+.2f}bp",
                "Edge": r.get("edge_direction", "?"),
            })
        except Exception:
            continue
    if _lp_rows:
        def _color_edge(val):
            v = str(val)
            if v == "fade_long":
                return "color: #ef5350; font-weight: bold"
            if v == "fade_short":
                return "color: #26a69a; font-weight: bold"
            return "color: #888888"
        _lp_df = pd.DataFrame(_lp_rows)
        st.dataframe(_lp_df.style.map(_color_edge, subset=["Edge"]),
                     use_container_width=True, hide_index=True)
        _actionable = [r for r in _lp_rows if r["Edge"] not in ("no_edge", "?")]
        if _actionable:
            st.warning(f"⚠️ {len(_actionable)} pair(s) showing cascade edge — see overbought_fade/consolidation_breakout")
        else:
            st.caption("No cascade pressure detected. Markets balanced.")
except Exception as _e:
    st.error(f"Liquidation pressure failed: {_e}")

# HRP sleeve allocation (W16.G)
st.markdown("##### HRP sleeve allocation (W16.G, Lopez de Prado AFML 16)")
try:
    from ops.portfolio_allocator import status as hrp_status, BASELINE_WEIGHTS
    hs = hrp_status()
    if hs.get("status") == "ok":
        _hrp_rows = []
        for sname, hw in sorted(hs["weights"].items(), key=lambda x: -x[1]):
            bw = BASELINE_WEIGHTS.get(sname, 0.0)
            scale = hw / bw if bw > 0 else 0.0
            _hrp_rows.append({
                "Sleeve": sname,
                "HRP %": f"{hw*100:.1f}%",
                "Baseline %": f"{bw*100:.1f}%",
                "Scale": f"{scale:.2f}x",
                "Action": "UPSIZE" if scale > 1.25 else ("DOWNSIZE" if scale < 0.8 else "hold"),
            })
        st.dataframe(pd.DataFrame(_hrp_rows), use_container_width=True, hide_index=True)
        st.caption(f"Recomputed at {hs.get('computed_at', '?')[:19]}  ({hs.get('n_observations', '?')} obs)")
    else:
        st.info(f"HRP weights warming up: needs ≥14 days of returns per sleeve. Currently using baseline allocations.")
        # Show baseline so trader still sees the allocation map
        _hrp_rows = []
        for sname, bw in sorted(BASELINE_WEIGHTS.items(), key=lambda x: -x[1]):
            _hrp_rows.append({
                "Sleeve": sname,
                "Baseline %": f"{bw*100:.1f}%",
                "Source": "hardcoded",
            })
        st.dataframe(pd.DataFrame(_hrp_rows), use_container_width=True, hide_index=True, height=260)
except Exception as _e:
    st.error(f"HRP allocator failed: {_e}")

st.divider()

# ===== Cycle indicators (on-chain + IV + funding) =====
st.subheader("Cycle indicators — on-chain, IV, funding skew")
try:
    from core import onchain, options_iv, funding_skew, cvd
    _cyc = onchain.cycle_position()
    _iv = options_iv.get_iv_regime()
    _flows = onchain.get_exchange_flows()
    _aa = onchain.get_active_addresses()

    cyc_col, iv_col, flow_col = st.columns(3)
    with cyc_col:
        st.markdown("**Cycle position**")
        score = _cyc.get("score")
        phase = _cyc.get("phase", "unknown")
        if score is not None:
            phase_color = {
                "DEEP_BEAR": "#26a69a",
                "EARLY_BULL": "#90ee90",
                "MID_BULL": "#f0b90b",
                "LATE_BULL": "#ff8c00",
                "EUPHORIA": "#ef5350",
            }.get(phase, "#aaaaaa")
            st.metric("Score", f"{score:.0f}/100", phase, delta_color="off")
            st.markdown(f"<span style='color:{phase_color}; font-weight:bold'>{phase}</span>",
                        unsafe_allow_html=True)
            st.caption(f"MVRV: {_cyc.get('mvrv', 0):.2f}  |  NUPL: {_cyc.get('nupl', 0):+.2f}")
        else:
            st.info("MVRV unavailable")

    with iv_col:
        st.markdown("**Implied vol regime**")
        dvol = _iv.get("dvol")
        if dvol is not None:
            st.metric("DVOL", f"{dvol:.1f}", _iv.get("regime", "?"), delta_color="off")
            change = _iv.get("dvol_vs_30d_mean", 0)
            st.caption(f"vs 30d mean: {change*100:+.1f}%")
            if _iv.get("skew_label"):
                st.caption(f"Skew: {_iv['skew_label']}")
        else:
            st.info("DVOL unavailable")

    with flow_col:
        st.markdown("**On-chain flow + adoption**")
        nflow = _flows.get("net_flow_usd")
        if nflow is not None:
            arrow = "↑" if nflow > 0 else "↓"
            st.metric("Exchange net flow (24h)", f"${nflow/1e6:+,.1f}M",
                      _flows.get("interpretation", "?"),
                      delta_color="off")
        if _aa.get("active_addresses"):
            st.caption(f"Active addresses: {_aa['active_addresses']:,.0f}  "
                       f"(30d: {_aa['vs_30d_mean']*100:+.1f}%)")

    # CVD divergence per major
    st.markdown("**CVD divergence — recent (30d)**")
    cvd_rows = []
    for _p in ["BTC/USDT", "ETH/USDT", "SOL/USDT", "LINK/USDT", "AVAX/USDT"]:
        try:
            d = cvd.divergence_signal(_p, lookback=30)
            cvd_rows.append({
                "Pair": _p,
                "Price 30d": f"{d.get('price_30d_change_pct', 0):+.1f}%",
                "CVD 30d": f"{d.get('cvd_30d_change', 0):+,.0f}",
                "Signal": d.get("signal", "?"),
            })
        except Exception:
            continue
    if cvd_rows:
        _cvd_df = pd.DataFrame(cvd_rows)
        def _cvd_sig_color(val):
            if "bullish" in str(val).lower():
                return "color: #26a69a; font-weight: bold"
            if "bearish" in str(val).lower():
                return "color: #ef5350; font-weight: bold"
            return ""
        st.dataframe(_cvd_df.style.map(_cvd_sig_color, subset=["Signal"]),
                     use_container_width=True, hide_index=True)

    # Funding skew
    st.markdown("**Multi-exchange funding (8h, annualized %)**")
    fund_rows = []
    for _p in ["BTC/USDT", "ETH/USDT", "SOL/USDT", "LINK/USDT"]:
        try:
            s = funding_skew.skew_analysis(_p)
            if s.get("error"):
                continue
            fund_rows.append({
                "Pair": _p,
                "Mean 8h": f"{s['mean_8h']*10000:+.2f}bp",
                "Annualized": f"{s['mean_annualized_pct']:+.2f}%",
                "Dispersion": f"{s['dispersion_bps']:.2f}bp",
                "Regime": s.get("regime", "?"),
            })
        except Exception:
            continue
    if fund_rows:
        st.dataframe(pd.DataFrame(fund_rows), use_container_width=True, hide_index=True)

except Exception as _e:
    st.error(f"Cycle indicator panel failed to load: {_e}")

st.divider()

# ===== Exit signal monitor =====
st.subheader("Exit Signal Monitor — EMA21 trailing stops")
_exit_state_file = REPO_ROOT / "btc_exit_signal_state.json"
if _exit_state_file.exists():
    try:
        _exit_state = json.loads(_exit_state_file.read_text())
    except Exception:
        _exit_state = {}
else:
    _exit_state = {}

if _exit_state:
    _exit_rows = []
    for _pair, _s in _exit_state.items():
        if not isinstance(_s, dict):
            continue
        _price = _s.get("price", 0) or 0
        _ema = _s.get("ema21_price", 0) or 0
        _dist = _s.get("dist_from_ema21", 0) or 0
        _stop = _s.get("stop_alert", "?")
        _state = _s.get("state", "?")
        _last_fire = _s.get("last_fire", "never") or "never"
        _from_low = _s.get("from_feb_low")
        _rsi = _s.get("rsi", 0) or 0
        _mayer = _s.get("mayer", 0) or 0
        _bb = _s.get("bb_pct", 0) or 0
        _drop_to_stop = (_ema / _price - 1) * 100 if _price > 0 and _s.get("above_ema") else 0.0
        _exit_rows.append({
            "Pair": _pair,
            "Price": f"${_price:,.4f}" if _price < 100 else f"${_price:,.2f}",
            "EMA21": f"${_ema:,.4f}" if _ema < 100 else f"${_ema:,.2f}",
            "Dist": f"{_dist*100:+.1f}%",
            "Drop to stop": f"{_drop_to_stop:+.1f}%",
            "RSI": f"{_rsi:.0f}",
            "Mayer": f"{_mayer:.2f}",
            "BB%": f"{_bb*100:.0f}%",
            "From Feb low": f"{_from_low*100:+.0f}%" if _from_low is not None else "-",
            "Last fire": _last_fire,
            "Stop": _stop,
            "State": _state,
            "_dist_num": _dist,
            "_stop_rank": {"BROKEN": 0, "NEAR": 1, "WATCH": 2, "OK": 3}.get(_stop, 4),
        })
    _exit_df = pd.DataFrame(_exit_rows).sort_values(["_stop_rank", "_dist_num"]).drop(columns=["_dist_num", "_stop_rank"])

    def _stop_style(val):
        if val == "BROKEN":
            return "background-color: #ef5350; color: white; font-weight: bold"
        if val == "NEAR":
            return "background-color: #f0b90b; color: black; font-weight: bold"
        if val == "WATCH":
            return "color: #f0b90b"
        if val == "OK":
            return "color: #26a69a"
        return ""

    def _state_style(val):
        if val in ("EXTREME OB", "PEAK ZONE"):
            return "color: #ef5350; font-weight: bold"
        if val in ("ARMED", "FIRED"):
            return "color: #f0b90b; font-weight: bold"
        if val == "STALE":
            return "color: #888888"
        return ""

    _styled_exit = _exit_df.style.map(_stop_style, subset=["Stop"]).map(_state_style, subset=["State"])
    st.dataframe(_styled_exit, use_container_width=True, hide_index=True)

    _n_broken = sum(1 for r in _exit_rows if r["Stop"] == "BROKEN")
    _n_near = sum(1 for r in _exit_rows if r["Stop"] == "NEAR")
    _n_watch = sum(1 for r in _exit_rows if r["Stop"] == "WATCH")

    if _n_broken:
        st.error(f"{_n_broken} pair(s) BROKEN — EMA21 exit triggered. Sell immediately.")
    if _n_near:
        st.warning(f"{_n_near} pair(s) NEAR EMA21 (within 2%). Stop-loss sell orders should be live.")
    if _n_watch and not (_n_broken or _n_near):
        st.info(f"{_n_watch} pair(s) approaching EMA21 (within 4%).")
    if not (_n_broken or _n_near or _n_watch):
        st.success("All pairs comfortably above EMA21 trailing stop.")

    st.caption(
        "Signal: MACD bear cross + close<EMA21 within 5 days (backtest n=5, "
        "capture 67%, avoided drawdown 126%). State refreshes when `python exit_signal_run.py` runs."
    )
else:
    st.info("Exit signal monitor not yet run. Execute: `python exit_signal_run.py`")

st.divider()

# ===== Watchlist =====
st.subheader("Watchlist — 24h change")
watch = fetch_watchlist()
rows = []
for pair in WATCHLIST:
    t = watch.get(pair, {})
    last = t.get("last") or 0
    pct_24h = t.get("percentage")
    high_24h = t.get("high")
    low_24h = t.get("low")
    vol_24h = t.get("quoteVolume") or t.get("baseVolume", 0)
    rows.append({
        "Pair": pair,
        "Last": f"${last:,.4f}" if last < 10 else f"${last:,.2f}",
        "24h Change": f"{pct_24h:+.2f}%" if pct_24h is not None else "-",
        "24h High": f"${high_24h:,.2f}" if high_24h else "-",
        "24h Low": f"${low_24h:,.2f}" if low_24h else "-",
        "24h Vol (USDT)": f"${vol_24h/1e6:,.1f}M" if vol_24h else "-",
        "_pct": pct_24h or 0.0,
    })
wl_df = pd.DataFrame(rows)


def _color_change(val):
    if isinstance(val, str) and val.endswith("%"):
        try:
            v = float(val.rstrip("%"))
            color = "#26a69a" if v >= 0 else "#ef5350"
            return f"color: {color}; font-weight: bold"
        except Exception:
            return ""
    return ""


styled = wl_df.drop(columns=["_pct"]).style.map(_color_change, subset=["24h Change"])
st.dataframe(styled, use_container_width=True, hide_index=True)


# ===== Pair selector + timeframe =====
ctop1, ctop2, _ = st.columns([2, 2, 6])
selected_pair = ctop1.selectbox("Pair", WATCHLIST, index=0)
selected_tf = ctop2.radio(
    "Timeframe", list(TIMEFRAMES.keys()), index=2, horizontal=True
)
tf_limit = TIMEFRAMES[selected_tf]


# ===== Indicator toggles =====
ind_col1, ind_col2, ind_col3, ind_col4 = st.columns(4)
show_bb = ind_col1.checkbox("Bollinger Bands (20, 2σ)", value=True)
show_sma = ind_col2.checkbox("SMA 50/200", value=True)
show_rsi = ind_col3.checkbox("RSI (14)", value=True)
show_macd = ind_col4.checkbox("MACD (12,26,9)", value=True)

# ===== Candlestick + indicator stack =====
hist = fetch_ohlcv(selected_pair, selected_tf, tf_limit)
if not hist.empty:
    n_extra_rows = sum([show_rsi, show_macd])
    main_height = 0.55 if n_extra_rows else 0.75
    vol_height = 0.20 if n_extra_rows else 0.25
    extra_height = (1 - main_height - vol_height) / max(n_extra_rows, 1) if n_extra_rows else 0
    row_heights = [main_height, vol_height] + [extra_height] * n_extra_rows

    n_rows = 2 + n_extra_rows
    subplot_titles = [""] * n_rows
    subplot_titles[1] = "Volume"
    extra_idx = 2
    if show_rsi:
        subplot_titles[extra_idx] = "RSI"
        rsi_row = extra_idx + 1
        extra_idx += 1
    if show_macd:
        subplot_titles[extra_idx] = "MACD"
        macd_row = extra_idx + 1

    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=row_heights, subplot_titles=subplot_titles,
    )
    fig.add_trace(
        go.Candlestick(
            x=hist.index, open=hist["open"], high=hist["high"], low=hist["low"], close=hist["close"],
            name=selected_pair,
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        ),
        row=1, col=1,
    )
    if show_bb and len(hist) >= 20:
        upper, mid, lower = compute_bbands(hist["close"], window=20, n_std=2.0)
        fig.add_trace(go.Scatter(x=hist.index, y=upper, name="BB upper", line=dict(color="#888", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=mid, name="BB mid", line=dict(color="#aaa", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=lower, name="BB lower", line=dict(color="#888", width=1, dash="dot"), fill="tonexty", fillcolor="rgba(136,136,136,0.05)"), row=1, col=1)

    if show_sma:
        if len(hist) >= 50:
            sma50 = hist["close"].rolling(50).mean()
            fig.add_trace(go.Scatter(x=hist.index, y=sma50, name="SMA 50", line=dict(color="#5e96f7", width=1.5)), row=1, col=1)
        if len(hist) >= 200:
            sma200 = hist["close"].rolling(200).mean()
            fig.add_trace(go.Scatter(x=hist.index, y=sma200, name="SMA 200", line=dict(color="#f0b90b", width=1.5)), row=1, col=1)

    vol_colors = ["#26a69a" if c >= o else "#ef5350" for o, c in zip(hist["open"], hist["close"])]
    fig.add_trace(
        go.Bar(x=hist.index, y=hist["volume"], marker_color=vol_colors, name="Volume", showlegend=False),
        row=2, col=1,
    )

    if show_rsi:
        rsi = compute_rsi(hist["close"], period=14)
        fig.add_trace(go.Scatter(x=hist.index, y=rsi, name="RSI 14", line=dict(color="#bb86fc", width=1.5), showlegend=False), row=rsi_row, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#ef5350", line_width=1, row=rsi_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#26a69a", line_width=1, row=rsi_row, col=1)

    if show_macd:
        macd_line, sig_line, hist_macd = compute_macd(hist["close"])
        macd_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in hist_macd]
        fig.add_trace(go.Bar(x=hist.index, y=hist_macd, marker_color=macd_colors, name="MACD hist", showlegend=False), row=macd_row, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=macd_line, name="MACD", line=dict(color="#5e96f7", width=1.2), showlegend=False), row=macd_row, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=sig_line, name="signal", line=dict(color="#f0b90b", width=1.2), showlegend=False), row=macd_row, col=1)

    fig.update_layout(
        height=600 + n_extra_rows * 120,
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=20, b=0),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
    st.plotly_chart(fig, use_container_width=True)

# ===== Funding rate panel =====
fund = fetch_funding_rate(selected_pair)
if fund:
    funding_pct = fund.get("fundingRate") or 0
    funding_ann_pct = funding_pct * 3 * 365 * 100  # 3 funding periods/day
    next_funding = fund.get("fundingDatetime", "?")
    fcol1, fcol2, fcol3 = st.columns(3)
    color = "#26a69a" if funding_pct >= 0 else "#ef5350"
    fcol1.markdown(f"**Funding rate:** <span style='color:{color}'>{funding_pct*100:+.4f}%</span>", unsafe_allow_html=True)
    fcol2.markdown(f"**Annualized:** <span style='color:{color}'>{funding_ann_pct:+.1f}%</span>", unsafe_allow_html=True)
    fcol3.caption(f"Next funding: {next_funding}")


# ===== Order book + recent trades (side by side) =====
ob_col, tr_col = st.columns(2)

with ob_col:
    st.subheader(f"Order Book — {selected_pair}")
    try:
        ob = fetch_order_book(selected_pair, depth=15)
        bids = ob.get("bids", [])[:15]
        asks = ob.get("asks", [])[:15]

        if bids and asks:
            mid = (bids[0][0] + asks[0][0]) / 2
            spread = asks[0][0] - bids[0][0]
            spread_bps = spread / mid * 10_000
            st.caption(f"Mid: ${mid:,.2f}  |  Spread: ${spread:,.2f} ({spread_bps:.1f} bps)")

            ask_df = pd.DataFrame(asks, columns=["Price", "Size"]).iloc[::-1]
            ask_df["Total"] = ask_df["Size"][::-1].cumsum()[::-1]
            bid_df = pd.DataFrame(bids, columns=["Price", "Size"])
            bid_df["Total"] = bid_df["Size"].cumsum()

            ob_fig = go.Figure()
            ob_fig.add_trace(go.Bar(
                y=ask_df["Price"].astype(str), x=ask_df["Total"], orientation="h",
                marker_color="rgba(239, 83, 80, 0.25)", name="Asks (cum)",
            ))
            ob_fig.add_trace(go.Bar(
                y=bid_df["Price"].astype(str), x=bid_df["Total"], orientation="h",
                marker_color="rgba(38, 166, 154, 0.25)", name="Bids (cum)",
            ))
            ob_fig.update_layout(
                height=500, barmode="overlay", showlegend=False,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="Cumulative size", yaxis_title="",
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(ob_fig, use_container_width=True)

            book_table = pd.concat([
                ask_df.assign(Side="ASK")[::-1],
                bid_df.assign(Side="BID"),
            ])
            book_table["Price"] = book_table["Price"].apply(lambda x: f"${x:,.2f}")
            book_table["Size"] = book_table["Size"].apply(lambda x: f"{x:.4f}")
            book_table["Total"] = book_table["Total"].apply(lambda x: f"{x:.4f}")
    except Exception as e:
        st.error(f"Order book fetch failed: {e}")

with tr_col:
    st.subheader(f"Recent Trades — {selected_pair}")
    try:
        trades = fetch_recent_trades(selected_pair, limit=30)
        rows = []
        for t in reversed(trades):
            ts = pd.Timestamp(t["timestamp"], unit="ms", tz="UTC")
            side = t.get("side", "?")
            price = t.get("price", 0)
            amount = t.get("amount", 0)
            rows.append({
                "Time": ts.strftime("%H:%M:%S"),
                "Side": side.upper(),
                "Price": f"${price:,.2f}" if price > 10 else f"${price:,.6f}",
                "Size": f"{amount:.4f}",
                "_side": side,
            })
        trades_df = pd.DataFrame(rows)

        def _color_side(val):
            if val == "BUY":
                return "color: #26a69a; font-weight: bold"
            if val == "SELL":
                return "color: #ef5350; font-weight: bold"
            return ""

        styled_tr = trades_df.drop(columns=["_side"]).style.map(_color_side, subset=["Side"])
        st.dataframe(styled_tr, use_container_width=True, hide_index=True, height=500)
    except Exception as e:
        st.error(f"Trade tape fetch failed: {e}")


# ===== Strategy signals + regime + concordance =====
st.divider()
st.subheader("Strategy Signals, Regime & Concordance")
sig_col, reg_col, conc_col = st.columns([2, 2, 2])

with sig_col:
    sigs = fetch_signals()

    def _sig_color(val):
        try:
            v = float(val)
            if v > 0.3:
                return "color: #26a69a; font-weight: bold"
            if v < -0.3:
                return "color: #ef5350; font-weight: bold"
            return "color: #aaaaaa"
        except Exception:
            return ""

    sig_rows = []
    for name, val in sigs.items():
        sig_rows.append({"Strategy": name, "Signal": f"{val:+.3f}" if not np.isnan(val) else "n/a"})
    sig_df = pd.DataFrame(sig_rows)
    styled_sig = sig_df.style.map(_sig_color, subset=["Signal"])
    st.dataframe(styled_sig, use_container_width=True, hide_index=True)
    st.caption("Signal in [-1, 1]. Positive = long bias; negative = short; ~0 = flat.")

with reg_col:
    reg = fetch_regime()
    vol = reg["vol"]
    trend = reg["trend"]
    a, b = st.columns(2)
    a.metric("Vol regime", vol["regime"], f"realized vol {vol['realized_vol']:.1%}")
    b.metric("Trend regime", trend["regime"], f"price/SMA200 {trend['price_vs_sma']:.3f}")
    if reg["long_ok"]:
        st.success("✓ Longs allowed by regime gate")
    else:
        st.error("✗ Longs blocked (bear regime)")
    if reg["short_ok"]:
        st.success("✓ Shorts allowed")
    else:
        st.error("✗ Shorts blocked (bull regime)")

with conc_col:
    conc = correlation_monitor.signal_concordance(sigs)
    score = conc["score"]
    color = "#ef5350" if score >= 0.85 else ("#f0b90b" if score >= 0.6 else "#26a69a")
    st.metric("Concordance", f"{score:.0%}",
              f"{conc['n_signaling']} strategies signaling",
              delta_color="off")
    st.markdown(
        f"<span style='color:{color}'>Direction: <b>{conc['direction']}</b></span>"
        f"  ·  +{conc.get('n_pos',0)} long / -{conc.get('n_neg',0)} short",
        unsafe_allow_html=True,
    )
    if score >= 0.85 and conc["n_signaling"] >= 3:
        st.warning("High concordance — possible regime change OR strategy collapse")
    elif score < 0.4:
        st.success("Healthy strategy diversity — independent calls")
    else:
        st.caption("Mixed signals — strategies disagreeing")

# ===== HMM regime panel =====
st.markdown("##### Hidden Markov Regime (BTC, 2-state vol-switching)")
hmm = fetch_hmm_state()
if hmm.get("converged"):
    h_col1, h_col2, h_col3, h_col4 = st.columns(4)
    label = hmm["regime_label"]
    label_color = "#26a69a" if label == "low_vol" else "#ef5350"
    h_col1.markdown(
        f"**Current state:** <span style='color:{label_color}'>{label.upper()}</span>",
        unsafe_allow_html=True,
    )
    h_col2.metric("Regime probability", f"{hmm['regime_prob']:.1%}")
    h_col3.metric("Low-vol regime σ (ann)", f"{hmm['vol_per_regime_ann']['low_vol']:.1%}")
    h_col4.metric("High-vol regime σ (ann)", f"{hmm['vol_per_regime_ann']['high_vol']:.1%}")
    st.caption(
        f"Hamilton (1989) Markov-switching model. Mean daily return: "
        f"low_vol={hmm['mean_per_regime_daily_pct']['low_vol']:+.3f}%, "
        f"high_vol={hmm['mean_per_regime_daily_pct']['high_vol']:+.3f}%. "
        f"Smoothed prob trend (last 10 days): "
        + " → ".join(f"{p:.2f}" for p in hmm["smoothed_probs_low_vol_recent"][-5:])
    )
else:
    st.info("HMM regime fit unavailable.")


# ===== Equity curve + paper trades + evidence (collapsibles) =====
st.divider()
with st.expander("📈 Paper equity curve", expanded=False):
    sdf = load_daily_status()
    if not sdf.empty and "equity_usdt" in sdf.columns:
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=sdf["ts"], y=sdf["equity_usdt"], mode="lines+markers",
            line=dict(color="#f0b90b", width=2), name="Equity",
        ))
        fig_eq.add_hline(y=100_000.0, line_dash="dash", line_color="gray", annotation_text="Start")
        fig_eq.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Equity (USDT)")
        st.plotly_chart(fig_eq, use_container_width=True)
    else:
        st.info("No daily snapshots yet. Run `python ops/daily_log.py` to seed.")

with st.expander("📋 Paper trade history", expanded=False):
    tdf = load_trades()
    if not tdf.empty:
        st.dataframe(tdf.tail(50)[::-1], use_container_width=True, hide_index=True)
        st.caption(f"Total trades: {len(tdf)}")
    else:
        st.info("No paper trades yet.")

with st.expander("🔗 Position correlation matrix — are 'diversified' positions really independent?", expanded=False):
    pos_corr = portfolio_risk.position_correlation()
    if not pos_corr.empty:
        # Color the heatmap
        fig_corr = go.Figure(data=go.Heatmap(
            z=pos_corr.values,
            x=pos_corr.columns,
            y=pos_corr.index,
            colorscale="RdYlGn_r",
            zmid=0.5,
            text=pos_corr.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 11},
            colorbar=dict(title="ρ"),
        ))
        fig_corr.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_corr, use_container_width=True)
        avg_corr = pos_corr.values[~np.eye(len(pos_corr), dtype=bool)].mean()
        st.caption(
            f"Average pairwise correlation: **{avg_corr:.2f}**. "
            f"For real diversification, you want this < 0.4. "
            f"Crypto majors typically correlate 0.7-0.9 — your '10 positions' are roughly 2-3 effective independent bets."
        )
    else:
        st.info("Need ≥2 open positions for correlation matrix.")

with st.expander("🎲 Strategy signal correlation — are signals really independent?", expanded=False):
    strat_corr = portfolio_risk.strategy_signal_correlation(window_days=30)
    if not strat_corr.empty:
        fig_sc = go.Figure(data=go.Heatmap(
            z=strat_corr.values,
            x=strat_corr.columns,
            y=strat_corr.index,
            colorscale="RdYlGn_r",
            zmid=0,
            text=strat_corr.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="ρ"),
        ))
        fig_sc.update_layout(height=500, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_sc, use_container_width=True)
        st.caption(
            "Strategy signal correlation across all (cycle × pair) observations. "
            "High correlation = redundant strategies (they fire together). Low/negative = genuine diversification."
        )
    else:
        st.info("Need ≥3 cycle snapshots. Run `python run.py` a couple more times.")

with st.expander("🎰 Monte Carlo P&L forecast — 30-day distribution", expanded=False):
    try:
        mc = monte_carlo.monte_carlo_forecast(horizon_days=30, n_simulations=10_000)
        if "error" not in mc:
            samples = mc.get("samples", [])
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Expected P&L (30d)", f"${mc['expected_pnl']:+,.0f}",
                       f"vs start ${mc['starting_equity']:,.0f}")
            mc2.metric("Median P&L", f"${mc['median_pnl']:+,.0f}")
            mc3.metric("Prob > 0", f"{mc['prob_positive']:.1%}")
            mc4.metric("Prob > +$10k", f"{mc['prob_gain_10pct']:.1%}")

            if samples:
                hist_fig = go.Figure(data=[go.Histogram(
                    x=samples, nbinsx=60,
                    marker_color="#5e96f7",
                )])
                hist_fig.add_vline(x=0, line_dash="dash", line_color="#aaa")
                hist_fig.add_vline(x=mc["p5"], line_dash="dot", line_color="#ef5350",
                                   annotation_text=f"5%: ${mc['p5']:+,.0f}")
                hist_fig.add_vline(x=mc["p95"], line_dash="dot", line_color="#26a69a",
                                   annotation_text=f"95%: ${mc['p95']:+,.0f}")
                hist_fig.update_layout(
                    height=300, margin=dict(l=0, r=0, t=10, b=0),
                    xaxis_title="30-day P&L (USDT)", yaxis_title="frequency",
                )
                st.plotly_chart(hist_fig, use_container_width=True)

            tail_col1, tail_col2, tail_col3 = st.columns(3)
            tail_col1.metric("Prob loss > $5k", f"{mc['prob_loss_5pct_bankroll']:.1%}")
            tail_col2.metric("Prob loss > $10k", f"{mc['prob_loss_10pct_bankroll']:.1%}")
            tail_col3.metric("Prob loss > $20k", f"{mc['prob_loss_20pct_bankroll']:.1%}")
            st.caption(
                f"Bootstrap {mc['n_simulations']:,} 30-day paths from current portfolio's "
                f"daily returns history. Range: P5=${mc['p5']:+,.0f}, P25=${mc['p25']:+,.0f}, "
                f"P75=${mc['p75']:+,.0f}, P95=${mc['p95']:+,.0f}."
            )
        else:
            st.info(f"Monte Carlo unavailable: {mc['error']}")
    except Exception as e:
        st.info(f"Monte Carlo failed: {e}")

with st.expander("📉 Portfolio VaR — daily downside risk", expanded=False):
    var_info = portfolio_risk.portfolio_var(confidence=0.99)
    if "error" not in var_info:
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("99% Historical VaR (1d)", f"${var_info['historical_var_1d']:+,.0f}",
                  f"{var_info['var_pct_of_exposure']:+.2%} of exposure")
        v2.metric("99% Expected Shortfall (1d)", f"${var_info['historical_es_1d']:+,.0f}",
                  "avg of worst 1% days")
        v3.metric("Daily Vol ($)", f"${var_info['daily_vol_usdt']:,.0f}")
        v4.metric("Sample size", f"{var_info['n_obs']} days")
        st.caption(
            f"On the worst 1% of historical days, our current portfolio would lose "
            f"${abs(var_info['historical_var_1d']):,.0f} (historical VaR) — and "
            f"${abs(var_info['historical_es_1d']):,.0f} on average across those tail days."
        )
    else:
        st.info(f"VaR unavailable: {var_info['error']}")

with st.expander("🌪️ Stress test — replay through historical crashes", expanded=False):
    stress_df = fetch_stress_test()
    if stress_df is not None and not stress_df.empty:
        disp = stress_df.copy()
        for col in ["btc_return_pct", "strategy_return_pct", "strategy_max_dd_pct", "alpha_pct"]:
            if col in disp.columns:
                disp[col] = disp[col].apply(lambda v: f"{v:+.2f}%")
        st.dataframe(disp, use_container_width=True, hide_index=True)
        st.caption(
            "Multi-horizon TSMOM ensemble replayed through major crypto stress events. "
            "Alpha column = strategy return - BTC buy-and-hold over the same window. "
            "Positive alpha during crashes is the regime gate's value proposition."
        )
    else:
        st.info("Stress test unavailable (data fetch failed).")

with st.expander("💰 Realized P&L (FIFO matching)", expanded=False):
    real = fetch_realized_summary()
    if real and real.get("n_closes", 0) > 0:
        rcol1, rcol2, rcol3, rcol4 = st.columns(4)
        rcol1.metric("Total Realized", f"${real['total_realized']:+,.2f}")
        rcol2.metric("Closed Trades", real["n_closes"])
        rcol3.metric("Win Rate", f"{real['win_rate']:.1%}")
        rcol4.metric("Best / Worst", f"${real['best']:+,.2f} / ${real['worst']:+,.2f}")

        closes_df = realized_pnl.compute_realized_pnl()
        if not closes_df.empty:
            disp = closes_df.tail(30).copy()
            disp["realized_pnl"] = disp["realized_pnl"].apply(lambda v: f"${v:+,.2f}")
            disp["open_price"] = disp["open_price"].apply(lambda v: f"${v:,.4f}")
            disp["close_price"] = disp["close_price"].apply(lambda v: f"${v:,.4f}")
            st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        st.info("No closed trades yet — all current positions are still open.")

with st.expander("📊 Per-strategy P&L attribution (last 30 days)", expanded=False):
    try:
        attr_df = attribution.compute_attribution(window_days=30)
        if not attr_df.empty:
            disp = attr_df.copy()
            disp["total_pnl_usdt"] = disp["total_pnl_usdt"].apply(lambda v: f"${v:+,.0f}")
            disp["avg_pnl_per_obs"] = disp["avg_pnl_per_obs"].apply(lambda v: f"${v:+,.2f}")
            disp["win_rate"] = disp["win_rate"].apply(lambda v: f"{v:.1%}")
            disp["best"] = disp["best"].apply(lambda v: f"${v:+,.0f}")
            disp["worst"] = disp["worst"].apply(lambda v: f"${v:+,.0f}")
            st.dataframe(disp, use_container_width=True, hide_index=True)
            st.caption(
                "Hypothetical per-strategy P&L if each strategy traded its signal "
                "independently with the standard per-position cap. Surfaces which "
                "strategies are pulling weight."
            )
        else:
            st.info("Attribution requires ≥2 cycle snapshots. Run `python run.py` a couple more times.")
    except Exception as e:
        st.info(f"Attribution unavailable: {e}")

with st.expander("🧠 Evidence ledger (research claims)", expanded=False):
    ev = load_evidence()
    if ev:
        rows = []
        for e in ev[-20:][::-1]:
            ts = datetime.fromtimestamp(e["ts"], tz=timezone.utc)
            rows.append({
                "When (UTC)": ts.strftime("%Y-%m-%d %H:%M"),
                "Strategy": e.get("strategy", ""),
                "Claim": (e.get("claim", "") or "")[:80],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No evidence entries yet.")

with st.expander("ℹ️ About data sources"):
    st.markdown("""
**All data is free and public — no API keys for paper mode.**

| Section | Source | Auth? |
|---|---|---|
| Watchlist 24h tickers | Binance `/api/v3/ticker/24hr` (bulk) | No |
| Candles | Binance `/api/v3/klines` | No |
| Order book | Binance `/api/v3/depth` | No |
| Recent trades | Binance `/api/v3/trades` | No |
| Funding rates | Binance `/fapi/v1/fundingRate` | No |
| ETF flows | farside.co.uk scrape | No |

Accessed through `ccxt`. Cached locally in `.cache/` with sensible TTLs:
- 24h tickers: 30s | order book: 10s | trades: 15s | OHLCV: 60s
- Signals: 5min | regime: 5min

Your `BINANCE_API_KEY`/`SECRET` are only needed if you flip to live mode.
""")
