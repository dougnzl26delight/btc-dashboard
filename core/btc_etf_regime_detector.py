"""I4: ETF flow regime detector.

Classifies ETF flow regime and detects shifts. Sustained outflows
near peaks signal distribution. Sustained inflows near bottoms signal
accumulation by institutions.

Regimes:
  STRONG_INFLOW       cumulative 60d > +$5B, 30d positive
  ACCUMULATION       30d positive, mild magnitude
  NEUTRAL            mixed flows
  DISTRIBUTION       30d negative, mild magnitude
  HEAVY_OUTFLOW      cumulative 60d < -$3B, 30d negative
  CAPITULATION_FLOW  sustained outflows + price drawdown > -40%

Special alert: HEAVY_OUTFLOW while price within 15% of ATH = top signal
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = REPO_ROOT / ".btc_etf_regime_state.json"

CYCLE5_PEAK_PRICE = 124659


def _load_state() -> dict:
    if not STATE_FILE.exists():
        return {"last_regime": None, "last_check": None}
    try: return json.loads(STATE_FILE.read_text())
    except Exception: return {"last_regime": None, "last_check": None}


def _save_state(s: dict) -> None:
    try: STATE_FILE.write_text(json.dumps(s, indent=2))
    except Exception: pass


def _btc_price_now() -> float:
    try:
        from core import data
        return data.btc_spot()  # region-resilient (Kraken/Coinbase/Binance/Bitstamp)
    except Exception:
        return 0.0


def _etf_flows_history() -> Optional[pd.DataFrame]:
    """Daily US spot BTC ETF net flows ($M) as a 1-col DataFrame.
    Source: Farside Investors public table (same source as
    btc_premium_free.etf_flows, which only returns a summary). The old import
    (`_farside_etf_flows`) never existed, so this panel always read empty."""
    try:
        import io
        from core.btc_premium_free import _http_get
        # 2026-07-07 factual audit: all-data page (full ~640-row history) so the
        # 60d cumulative regime logic actually has 60 days (was 14 -> 5d=30d=60d).
        body = _http_get("https://farside.co.uk/bitcoin-etf-flow-all-data/", ttl=21600)
        if not body:
            return None
        tables = pd.read_html(io.StringIO(body))
        if not tables:
            return None
        df = max(tables, key=len)
        total_col = next((c for c in df.columns if "total" in str(c).lower()), None)
        if total_col is None:
            return None
        flows = pd.to_numeric(
            df[total_col].astype(str).str.replace(r"[^\d\.\-]", "", regex=True),
            errors="coerce",
        ).dropna()
        # Farside appends summary rows (cumulative all-time ~$50B, plus avg/max/
        # min). Real daily net flow is at most a few $B, so drop implausible
        # magnitudes — otherwise the cumulative row poisons the windowed sums.
        flows = flows[flows.abs() <= 3000]
        if flows.empty:
            return None
        return pd.DataFrame({"flow_M": flows.values})
    except Exception:
        return None


def classify_regime() -> dict:
    """Classify current ETF flow regime."""
    df = _etf_flows_history()
    if df is None or df.empty:
        # 2026-07-09 hardening: transient Farside fetch failures kept poisoning
        # the panel cache with zeros ("ETF flat" on the verdict card twice in a
        # week while the live feed was STRONG_INFLOW). Fall back to the LAST
        # GOOD cached read, clearly marked stale, instead of zeroing the card.
        try:
            import pickle as _pkl
            from pathlib import Path as _P
            _pc = _P(__file__).resolve().parent.parent / ".panel_cache" / "etf_regime.pkl"
            if _pc.exists():
                _prev = _pkl.load(open(_pc, "rb"))
                _prev = _prev[1] if isinstance(_prev, tuple) else _prev
                if isinstance(_prev, dict) and _prev.get("regime") not in (None, "DATA_UNAVAILABLE"):
                    _prev = dict(_prev)
                    _prev["stale"] = True
                    _prev["status"] = ("STALE (fetch failed; showing last good read) — "
                                       + str(_prev.get("status", "")))[:160]
                    return _prev
        except Exception:
            pass
        return {
            "regime": "DATA_UNAVAILABLE",
            "status": "ETF flow data not available",
            "flows_5d_M": 0, "flows_30d_M": 0, "flows_60d_M": 0,
        }

    # Calculate flow windows (in millions)
    df = df.sort_index() if df.index.is_monotonic_increasing else df.sort_index()
    flows_5d = float(df.tail(5).sum().iloc[0]) if not df.empty else 0
    flows_30d = float(df.tail(30).sum().iloc[0]) if len(df) >= 30 else flows_5d
    flows_60d = float(df.tail(60).sum().iloc[0]) if len(df) >= 60 else flows_30d

    # Price context
    price = _btc_price_now()
    pct_from_peak = (price / CYCLE5_PEAK_PRICE - 1) * 100 if price > 0 else 0
    near_peak = pct_from_peak > -15  # within 15% of ATH
    deep_drawdown = pct_from_peak < -40

    # Classify
    if flows_60d > 5_000 and flows_30d > 0:
        regime = "STRONG_INFLOW"
    elif flows_30d > 500:
        regime = "ACCUMULATION"
    elif flows_60d < -3_000 and flows_30d < 0:
        if deep_drawdown:
            regime = "CAPITULATION_FLOW"
        else:
            regime = "HEAVY_OUTFLOW"
    elif flows_30d < -500:
        regime = "DISTRIBUTION"
    else:
        regime = "NEUTRAL"

    # Top warning: heavy outflow near peak
    top_warning = (regime in ("HEAVY_OUTFLOW", "DISTRIBUTION")) and near_peak

    # Bottom warning: capitulation flow + deep drawdown
    bottom_warning = regime == "CAPITULATION_FLOW"

    return {
        "regime":           regime,
        "flows_5d_M":       flows_5d,
        "flows_30d_M":      flows_30d,
        "flows_60d_M":      flows_60d,
        "price":            price,
        "pct_from_peak":    pct_from_peak,
        "near_peak":        near_peak,
        "deep_drawdown":    deep_drawdown,
        "top_warning":      top_warning,
        "bottom_warning":   bottom_warning,
        "status":           (f"60d ${flows_60d:+.0f}M, 30d ${flows_30d:+.0f}M, "
                              f"price {pct_from_peak:+.0f}% from peak"),
    }


def check_etf_regime(send_email: bool = True) -> dict:
    r = classify_regime()
    if r.get("regime") == "DATA_UNAVAILABLE":
        return {**r, "alert_sent": False}

    state = _load_state()
    last_regime = state.get("last_regime")
    current_regime = r["regime"]

    state["last_regime"] = current_regime
    state["last_check"] = datetime.now(timezone.utc).isoformat()
    _save_state(state)

    # Email triggers
    regime_changed = last_regime is not None and last_regime != current_regime
    critical_state = r.get("top_warning") or r.get("bottom_warning")

    if not (regime_changed or critical_state):
        return {**r, "alert_sent": False, "message": "no change"}

    if r.get("top_warning"):
        subject = f"!! ETF TOP WARNING: {current_regime} while near peak !!"
        interp = (
            f"HEAVY ETF OUTFLOWS while BTC within 15% of ATH (${CYCLE5_PEAK_PRICE:,}). "
            f"This is the classic ETF-era distribution signal — institutions exiting "
            f"into late-cycle euphoria. Combined with on-chain top signals = high "
            f"confidence cycle top."
        )
    elif r.get("bottom_warning"):
        subject = f"!! ETF BOTTOM WARNING: {current_regime} with deep drawdown !!"
        interp = (
            f"CAPITULATION ETF FLOWS with BTC -{abs(r['pct_from_peak']):.0f}% from peak. "
            f"Final institutional capitulation often marks the cycle low. Watch for "
            f"inflow reversal — the 'first positive flow week' after this pattern has "
            f"historically marked the bottom within 30 days."
        )
    elif regime_changed:
        subject = f"!! ETF REGIME: {last_regime or 'first run'} -> {current_regime} !!"
        interp = f"ETF flow regime shifted from {last_regime} to {current_regime}."
    else:
        subject = ""; interp = ""

    body = f"""BTC ETF FLOW REGIME ALERT

Regime shift:    {last_regime or 'first observation'} -> {current_regime}

================================================================
FLOWS
================================================================
  Last 5 days:    ${r['flows_5d_M']:+,.0f}M
  Last 30 days:   ${r['flows_30d_M']:+,.0f}M
  Last 60 days:   ${r['flows_60d_M']:+,.0f}M

================================================================
PRICE CONTEXT
================================================================
  BTC spot:       ${r['price']:,.0f}
  From peak:      {r['pct_from_peak']:+.1f}%
  Near peak:      {r['near_peak']}
  Deep drawdown:  {r['deep_drawdown']}

================================================================
INTERPRETATION
================================================================
{interp}

================================================================
DASHBOARD
================================================================
http://localhost:8511 (Overview -> Macro Drivers panel)
"""

    if send_email:
        try:
            from ops.alerts import alert
            alert(body, level="critical", subject=subject)
            return {**r, "alert_sent": True, "subject": subject}
        except Exception as e:
            return {**r, "alert_sent": False, "error": str(e)}
    return {**r, "alert_sent": False, "preview": body}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--no-email", action="store_true")
    a = p.parse_args()
    r = check_etf_regime(send_email=not a.no_email)
    print(json.dumps({k: v for k, v in r.items() if k != "preview"}, indent=2, default=str))


if __name__ == "__main__":
    main()
