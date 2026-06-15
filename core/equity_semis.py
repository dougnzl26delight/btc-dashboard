"""Semiconductor leading-indicator tell (SOXX).

Olson's June 2026 breakdown thesis leans on semis leading the move down
(NVDA, AVGO -25%, SMCI -23.6%). Semis historically lead QQQ at turns —
they're the high-beta core of the Nasdaq. This gives an EARLIER warning
than QQQ itself, tightening the lag between Olson's price-action calls
and the rotation trigger.

Design note: semis are highly correlated with QQQ (they ARE a big slice of
it), so this is deliberately a LEADING DISPLAY / pre-warning signal — NOT a
rotation-trigger condition. Adding it to the 2-of-4 trigger would double-count
tech weakness. It flags early; the trigger still fires on its own independent
conditions.

Reuses equity_olson's NaN-resilient fetch (yfinance primary, Yahoo-direct
fallback).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TICKER = "SOXX"   # iShares Semiconductor ETF


def _semis_history(period_y: int = 2):
    """SOXX daily history — reuse equity_olson's resilient fetcher pattern."""
    try:
        from core.equity_olson import _yahoo_direct_qqq  # not QQQ-specific despite name? it is
    except Exception:
        _yahoo_direct_qqq = None

    # Primary: yfinance
    h = None
    try:
        import yfinance as yf
        t = yf.Ticker(TICKER)
        h = t.history(period=f"{period_y}y", interval="1d")
        if h is not None and not h.empty:
            if h["Close"].tail(3).dropna().empty:
                h = None
    except Exception:
        h = None

    # Fallback: Yahoo chart JSON directly (different code path than yfinance scrape)
    if h is None or h.empty:
        try:
            import json as _json
            import urllib.request
            import pandas as pd
            url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{TICKER}"
                   f"?interval=1d&range={min(period_y,10)}y")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as r:
                payload = _json.loads(r.read().decode("utf-8", errors="ignore"))
            res = payload.get("chart", {}).get("result", [None])[0]
            if res:
                ts = res.get("timestamp", [])
                q = res.get("indicators", {}).get("quote", [{}])[0]
                if ts and q.get("close"):
                    h = pd.DataFrame({"Close": q.get("close", [])},
                                     index=pd.to_datetime(ts, unit="s"))
                    h = h.dropna(subset=["Close"]).sort_index()
        except Exception:
            h = None
    return h if (h is not None and not h.empty) else None


def semis_tell() -> dict:
    """SOXX vs 50/200-day MAs + recent momentum. Returns breakdown verdict."""
    h = _semis_history(2)
    if h is None or h.empty:
        return {"error": "no data"}

    closes = h["Close"].dropna()
    if len(closes) < 60:
        return {"error": "insufficient data"}

    last = float(closes.iloc[-1])
    ma50 = float(closes.tail(50).mean())
    ma200 = float(closes.tail(200).mean()) if len(closes) >= 200 else None
    high_252 = float(closes.tail(252).max())

    pct_vs_50 = (last / ma50 - 1) * 100
    pct_vs_200 = (last / ma200 - 1) * 100 if ma200 else None
    pct_off_high = (last / high_252 - 1) * 100

    # 5-day momentum
    mom_5d = (last / float(closes.iloc[-6]) - 1) * 100 if len(closes) >= 6 else 0.0

    below_50 = last < ma50
    below_200 = (ma200 is not None and last < ma200)

    # Tiered verdict — calibrated to the EARLY rollover (off-high + momentum loss),
    # not just MA crosses. In a strong semis uptrend an MA break needs a huge drop,
    # by which point QQQ has already gone — so the lead value is in the off-high read.
    if below_200 or (below_50 and pct_off_high <= -15):
        tier = "BREAKDOWN"
        color = "#ef4444"
        msg = (f"SOXX breaking down: {pct_vs_50:+.1f}% vs 50dMA, {pct_off_high:.1f}% "
               f"off 1y high. Semis in clear distribution — strong lead-warning that "
               f"QQQ follows lower.")
        warn = True
    elif below_50 or pct_off_high <= -10 or mom_5d <= -3:
        tier = "WEAKENING"
        color = "#f0b90b"
        msg = (f"SOXX rolling over: {pct_off_high:.1f}% off high, 5d {mom_5d:+.1f}%, "
               f"{pct_vs_50:+.1f}% vs 50dMA. Early tell ahead of QQQ — watch for "
               f"follow-through.")
        warn = True
    else:
        tier = "INTACT"
        color = "#22c55e"
        msg = (f"SOXX leadership intact: near highs ({pct_off_high:+.1f}%), "
               f"{pct_vs_50:+.1f}% vs 50dMA. No equity-breakdown lead-warning.")
        warn = False

    return {
        "tier":          tier,
        "color":         color,
        "warn":          warn,
        "last":          round(last, 2),
        "ma50":          round(ma50, 2),
        "ma200":         round(ma200, 2) if ma200 else None,
        "pct_vs_50":     round(pct_vs_50, 2),
        "pct_vs_200":    round(pct_vs_200, 2) if pct_vs_200 is not None else None,
        "pct_off_high":  round(pct_off_high, 2),
        "mom_5d":        round(mom_5d, 2),
        "below_50":      below_50,
        "below_200":     below_200,
        "message":       msg,
        "source":        "yahoo(SOXX)",
    }


def main():
    r = semis_tell()
    if r.get("error"):
        print("ERROR:", r["error"]); return
    print(f"SEMIS TELL: {r['tier']}")
    print(f"  SOXX ${r['last']:,.2f}  vs 50dMA {r['pct_vs_50']:+.1f}%  "
          f"vs 200dMA {r['pct_vs_200']}%  off-high {r['pct_off_high']:.1f}%")
    print(f"  {r['message']}")


if __name__ == "__main__":
    main()
