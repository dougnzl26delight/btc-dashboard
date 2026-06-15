"""Legible regime tag for the operator.

The HMM regime validator is a black box; a single operator making one rotation
needs a label he can REASON about — which playbook is he in. This is a 2x2 of
{net liquidity expanding/contracting} x {BTC weekly trend up/down}, plus the
closest historical cycle analog (so he knows whether to expect a 2022-style slow
grind, a 2020-style V, or a new ETF-era shallow drift). Free data only.
"""
from __future__ import annotations

# archetype drawdown depth + months-of-decline shape (rough, for nearest-match)
ARCHETYPES = {
    "2018 ordinary bear":     {"dd": -0.84, "months": 12},
    "2020 V-crash":           {"dd": -0.63, "months": 1},
    "2022 slow grind bear":   {"dd": -0.77, "months": 12},
    "ETF-era shallow grind":  {"dd": -0.50, "months": 9},
}


def _btc_weekly():
    import yfinance as yf
    return yf.Ticker("BTC-USD").history(period="3y", interval="1wk")["Close"].dropna()


def _liquidity_expanding():
    """Cheap free proxy: DXY falling over 50d = USD liquidity loosening = expanding.
    (Not a substitute for full net-liquidity; it's the legible directional read.)"""
    try:
        import yfinance as yf
        dxy = yf.Ticker("DX-Y.NYB").history(period="6mo")["Close"].dropna()
        if len(dxy) < 50:
            return None, "n/a"
        slope = float(dxy.iloc[-1] / dxy.iloc[-50] - 1)
        return (slope < 0), f"DXY {slope*100:+.1f}%/50d"
    except Exception:
        return None, "n/a"


def regime_tag() -> dict:
    out = {"liquidity": None, "btc_trend": None, "regime": "UNKNOWN",
           "analog": None, "analog_confidence": None, "detail": ""}
    btc_up = None
    try:
        w = _btc_weekly()
        sma20 = w.rolling(20).mean()
        btc_up = bool(w.iloc[-1] > sma20.iloc[-1])
        out["btc_trend"] = "up" if btc_up else "down"
        hi = float(w.max()); last = float(w.iloc[-1]); dd = last / hi - 1
        months = max(1, int((w.index[-1] - w.idxmax()).days / 30))
        best, bestd = None, 1e9
        for name, a in ARCHETYPES.items():
            d = abs(dd - a["dd"]) + abs(months - a["months"]) / 24.0
            if d < bestd:
                bestd, best = d, name
        out["analog"] = best
        out["analog_confidence"] = "low" if bestd > 0.4 else "moderate"
    except Exception:
        pass

    liq, liq_detail = _liquidity_expanding()
    out["liquidity"] = ("expanding" if liq else "contracting") if liq is not None else None

    if liq is not None and btc_up is not None:
        if liq and btc_up:
            out["regime"] = "RISK-ON TAILWIND"
        elif liq and not btc_up:
            out["regime"] = "LIQUIDITY-SUPPORTED DIP (accumulation-friendly)"
        elif (not liq) and btc_up:
            out["regime"] = "LATE-CYCLE / FADING"
        else:
            out["regime"] = "RISK-OFF DOWNTREND"
    out["detail"] = (f"BTC {out['btc_trend'] or '?'} vs 20-wk SMA; "
                     f"liquidity {out['liquidity'] or '?'} ({liq_detail})")
    return out


if __name__ == "__main__":
    r = regime_tag()
    print(f"REGIME: {r['regime']}")
    print(f"  {r['detail']}")
    print(f"  closest analog: {r['analog']} ({r['analog_confidence']} confidence)")
