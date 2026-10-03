"""Has the cycle low probably already happened? A real-time, price-only check.

Why (2026-10-04): every "bottom" panel counted down to the timing model's
projected date (~Oct 2026, $52-57k) and the site kept saying "bottom ~1 day
away" while BTC sat ~45% above its 30-Jun-2026 low with a higher low every
month. A countdown can't notice a bottom that came early and shallow; this can.

RULE - fires on the first day ALL hold, using only data up to that day:
  1. a bear happened: the lowest close since the all-time high is >=45% below it
  2. the 200-day SMA has fallen at some point since that ATH
  3. the 200-day SMA is rising now (20-day slope > 0)
  4. the close is >= 1.15 x the 200-day SMA
It FAILS if BTC later closes below the low it fired on, before a new ATH.

Walk-forward test, daily closes 2011-09 -> 2026-10 (Bitstamp until 2014-09, then
Yahoo BTC-USD): it fired once in each earlier bear - Jun 2012, Oct 2015,
May 2019, Jan 2023 - every time AFTER the real low, and price never closed below
that low again before the next ATH. Zero failures at every threshold tried
(drawdown 35-55%, multiple 1.05-1.20), so the result is not a tuned threshold.
n=4 earlier bears: strong history, not a guarantee.

Output is PURE PYTHON so the pickled panel survives numpy/pandas drift on Cloud.
"""
from __future__ import annotations

from datetime import datetime, timezone

DD_MIN = 0.45          # condition 1
SMA_MULT = 1.15        # condition 4
SLOPE_DAYS = 20        # conditions 2-3
MODEL_BAND = (52_000, 57_000)   # the timing model's bottom band shown elsewhere

# From the walk-forward test above (run 2026-10-04). Static history.
HISTORY = [
    {"bear": "2011", "low": 2.24, "low_date": "2011-10-20",
     "fired": "2012-06-13", "gain_at_fire_pct": 162, "note": "data starts Sep 2011"},
    {"bear": "2014-15", "low": 178.10, "low_date": "2015-01-14",
     "fired": "2015-10-26", "gain_at_fire_pct": 60, "note": ""},
    {"bear": "2018", "low": 3236.76, "low_date": "2018-12-15",
     "fired": "2019-05-18", "gain_at_fire_pct": 125, "note": ""},
    {"bear": "2022", "low": 15787.28, "low_date": "2022-11-21",
     "fired": "2023-01-26", "gain_at_fire_pct": 46, "note": ""},
]


def _daily_closes() -> list[tuple[str, float]]:
    """Completed UTC daily closes, oldest first, enough history to hold the ATH.
    Today's in-progress candle is dropped (the rule is on daily closes)."""
    today = datetime.now(timezone.utc).date().isoformat()
    rows: list[tuple[str, float]] = []
    try:
        import yfinance as yf
        s = yf.Ticker("BTC-USD").history(period="max", interval="1d")["Close"].dropna()
        rows = [(str(ix)[:10], float(v)) for ix, v in s.items()]
    except Exception:
        rows = []
    if len(rows) < 400:
        try:
            from core import data
            df = data.ohlcv_extended("BTC/USDT", days_back=1500, timeframe="1d")
            rows = [(str(ix)[:10], float(v)) for ix, v in zip(df.index, df["close"].tolist())]
        except Exception:
            rows = []
    rows = [r for r in rows if r[0] < today]
    return rows


def evaluate(rows: list[tuple[str, float]]) -> dict:
    """Run the rule over the CURRENT ATH-epoch. Pure function (testable)."""
    if len(rows) < 260:
        return {"status": "UNAVAILABLE", "reason": f"only {len(rows)} daily closes"}
    closes = [c for _, c in rows]
    dates = [d for d, _ in rows]
    n = len(closes)
    sma = [None] * n
    run = 0.0
    for i, c in enumerate(closes):
        run += c
        if i >= 200:
            run -= closes[i - 200]
        if i >= 199:
            sma[i] = run / 200.0
    ath_i = max(range(n), key=lambda i: closes[i])
    ath = closes[ath_i]

    low, low_i = ath, ath_i
    sma_fell = False
    fired = None                 # dict while a fire is live
    failures = []                # fires that a lower close later broke
    for i in range(ath_i + 1, n):
        px = closes[i]
        if fired and px < fired["low"]:
            failures.append({**fired, "broken_on": dates[i], "broken_close": round(px, 2)})
            fired = None
        if px < low:
            low, low_i = px, i
        s, s_prev = sma[i], (sma[i - SLOPE_DAYS] if i >= SLOPE_DAYS else None)
        if s is None or s_prev is None:
            continue
        slope = s / s_prev - 1
        if slope < 0:
            sma_fell = True
        if (not fired and low / ath - 1 <= -DD_MIN and sma_fell
                and slope > 0 and px >= SMA_MULT * s):
            fired = {"fire_date": dates[i], "low": round(low, 2),
                     "low_date": dates[low_i], "fire_close": round(px, 2),
                     "fire_ratio": round(px / s, 3)}

    last = closes[-1]
    s_now = sma[-1]
    slope_now = (s_now / sma[-1 - SLOPE_DAYS] - 1) if s_now and sma[-1 - SLOPE_DAYS] else None
    dd = low / ath - 1

    # Monthly lows from the low's month on (higher-lows narrative).
    months: dict[str, float] = {}
    for d, c in zip(dates[low_i:], closes[low_i:]):
        m = d[:7]
        months[m] = min(months.get(m, c), c)
    mlows = [[m, round(v, 2)] for m, v in sorted(months.items())]
    streak = 0
    for a, b in zip(mlows, mlows[1:]):
        streak = streak + 1 if b[1] > a[1] else 0

    if fired:
        status = "LOW_PROBABLY_IN"
    elif failures:
        status = "FAILED"
    elif dd > -DD_MIN:
        status = "NO_BEAR"
    else:
        status = "WATCHING"

    days_since_low = (datetime.fromisoformat(dates[-1]) - datetime.fromisoformat(dates[low_i])).days
    out = {
        "status": status,
        "asof": dates[-1],
        "last_close": round(last, 2),
        "ath": round(ath, 2), "ath_date": dates[ath_i],
        "low": round(low, 2), "low_date": dates[low_i],
        "drawdown_pct": round(dd * 100, 1),
        "pct_above_low": round((last / low - 1) * 100, 1),
        "days_since_low": days_since_low,
        "sma200": round(s_now, 2) if s_now else None,
        "ratio": round(last / s_now, 3) if s_now else None,
        "sma200_rising": bool(slope_now and slope_now > 0),
        "monthly_lows": mlows,
        "higher_lows_months": streak,
        "every_month_higher": streak == len(mlows) - 1 and len(mlows) > 1,
        "fire": fired,
        "failures": failures,
        "fail_below": fired["low"] if fired else None,
        "fall_to_band_pct": [round((MODEL_BAND[1] / last - 1) * 100),
                             round((MODEL_BAND[0] / last - 1) * 100)],
        "history": HISTORY,
        "rule": (f"after a >={int(DD_MIN * 100)}% drop from the ATH, BTC closes >="
                 f"{SMA_MULT:.2f}x a 200-day average that fell and has turned back up"),
    }
    out["low_date_txt"] = _fmt_date(out["low_date"])
    out["fire_date_txt"] = _fmt_date(fired["fire_date"]) if fired else ""
    out.update(_texts(out))
    return out


def _fmt_date(iso: str) -> str:
    try:
        d = datetime.fromisoformat(iso)
        return f"{d.day} {d.strftime('%b %Y')}"
    except Exception:
        return iso


def _texts(r: dict) -> dict:
    """Plain-English strings for the dashboard (one place, so panels agree)."""
    st = r["status"]
    low_s = f"${r['low']:,.0f}"
    low_d = _fmt_date(r["low_date"])
    if st == "LOW_PROBABLY_IN":
        months = ("with a higher low every month since"
                  if r.get("every_month_higher") else "")
        f = r["fire"]
        why = (f"On {_fmt_date(f['fire_date'])} it closed more than {round((SMA_MULT - 1) * 100)}% "
               f"above a 200-day average that had been falling and has turned back up. In every "
               f"earlier bear market in the data (2011, 2014–15, 2018, 2022) that only happened "
               f"after the real low — and price never closed below that low again. 4 for 4: a "
               f"small sample, not a guarantee. It fails if BTC closes below {low_s}.")
        return {
            "headline": f"Cycle low probably in: {low_s} ({low_d})",
            "short": f"probably in: {low_s} on {low_d}",
            "why": why,
            "detail": (f"BTC is {r['pct_above_low']:+.0f}% above that low"
                       f"{', ' + months if months else ''}. " + why),
        }
    if st == "FAILED":
        b = r["failures"][-1]
        return {
            "headline": f"The {_fmt_date(b['low_date'])} low broke",
            "short": f"the {_fmt_date(b['low_date'])} low broke on {_fmt_date(b['broken_on'])}",
            "detail": (f"BTC closed at ${b['broken_close']:,.0f} on {_fmt_date(b['broken_on'])}, below the "
                       f"${b['low']:,.0f} low the 'low is in' check fired on — a first in the data. "
                       f"The bottom may still be ahead."),
        }
    return {"headline": "", "short": "", "detail": ""}


def bottom_status() -> dict:
    try:
        return evaluate(_daily_closes())
    except Exception as e:
        return {"status": "UNAVAILABLE", "reason": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    r = bottom_status()
    print(r.get("status"), "|", r.get("headline"))
    print(r.get("detail"))
    for k in ("asof", "last_close", "low", "low_date", "pct_above_low", "ratio",
              "sma200_rising", "monthly_lows", "fire", "fall_to_band_pct"):
        print(f"  {k}: {r.get(k)}")
