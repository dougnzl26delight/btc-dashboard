"""BTC alpha-vs-QQQ regime gauge  ("Rolling Bubbles" monitor).

Motivation (validated 2026-07 against BTC/QQQ/ETH data):
    BTC's *correlation* to the Nasdaq/AI trade never broke — it sits ~0.4-0.6.
    What changed is its ALPHA. For 2019-2024 BTC was "QQQ with 3-5x the upside"
    (2020 +315% vs +45%; 2023 +152% vs +55%; 2024 +108% vs +27%). In 2025-2026
    that premium didn't shrink, it went NEGATIVE (2026 YTD BTC -32% vs QQQ +16%).
    Correlation intact + negative alpha = the worst risk-asset combination:
    still falls with equities, no longer captures the upside.

    So the informative regime variable is NOT BTC/QQQ correlation (~0.5, useless)
    — it's BTC's rolling ALPHA vs QQQ, plus the BTC/QQQ price ratio trend, plus
    ETH/BTC (has the speculative *alt* bid returned, or is crypto still starved?).

This module computes those three gauges and a regime label. Pure function,
returns a dict; render in the dashboard. Fail-safe: never raises — returns
{"status": "unavailable", ...} on any error so it can't break a hot render.

Dashboard wiring (wrap the fetch in st.cache_data like the other panels):
    from core import btc_alpha_regime
    @st.cache_data(ttl=3600)
    def _alpha_regime():
        return btc_alpha_regime.compute()
    r = _alpha_regime()
    if r["status"] == "ok":
        st.metric("BTC alpha vs QQQ (ann.)", f"{r['alpha_annual_90d']:+.0%}")
        st.caption(r["summary"])
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TICKERS = {"BTC": "BTC-USD", "QQQ": "QQQ", "ETH": "ETH-USD"}


def _fetch_yf(period_days: int = 420) -> dict[str, pd.Series]:
    """Daily close series for BTC-USD, QQQ, ETH-USD via yfinance. Raises on failure."""
    import yfinance as yf
    out: dict[str, pd.Series] = {}
    for key, tkr in TICKERS.items():
        h = yf.Ticker(tkr).history(period=f"{period_days}d", interval="1d")["Close"].dropna()
        # normalise index to tz-naive dates so BTC (7d) aligns with QQQ (5d)
        h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
        out[key] = h
    return out


def _ols_alpha_beta(y: np.ndarray, x: np.ndarray) -> tuple[float, float, float]:
    """Return (daily_alpha, beta, r) for y = alpha + beta*x."""
    if len(x) < 10:
        return (float("nan"),) * 3
    beta, alpha = np.polyfit(x, y, 1)          # slope, intercept
    r = float(np.corrcoef(x, y)[0, 1])
    return float(alpha), float(beta), r


def _trend(series: pd.Series, ma: int) -> str:
    """'rising' if last value > its `ma`-day mean, else 'falling' (or 'n/a')."""
    if len(series) < ma:
        return "n/a"
    return "rising" if series.iloc[-1] > series.tail(ma).mean() else "falling"


def compute(prices: dict[str, pd.Series] | None = None) -> dict:
    """Compute the BTC-alpha / rolling-bubbles regime gauges.

    `prices` (optional): {'BTC':Series,'QQQ':Series,'ETH':Series} of daily closes
    indexed by date — injectable for testing/backtests. Defaults to a live
    yfinance fetch. NEVER raises.
    """
    try:
        px = prices if prices is not None else _fetch_yf()
        btc, qqq, eth = px["BTC"], px["QQQ"], px["ETH"]

        # align BTC & QQQ on common (QQQ) trading days
        df = pd.DataFrame({"btc": btc, "qqq": qqq}).dropna()
        rb = df["btc"].pct_change().dropna()
        rq = df["qqq"].pct_change().dropna()
        j = pd.concat([rb, rq], axis=1, keys=["btc", "qqq"]).dropna()

        def alpha_beta(win: int):
            s = j.tail(win)
            a, b, r = _ols_alpha_beta(s["btc"].to_numpy(), s["qqq"].to_numpy())
            return a * 252.0, b, r          # annualise the daily intercept

        a90, b90, r90 = alpha_beta(90)
        a252, b252, r252 = alpha_beta(252)

        # BTC/QQQ price ratio + trend (is BTC out/under-performing its own trend?)
        ratio = (df["btc"] / df["qqq"]).dropna()
        ratio_trend = _trend(ratio, 200)
        ratio_vs_1y = float(ratio.iloc[-1] / ratio.tail(252).mean() - 1) if len(ratio) >= 252 else float("nan")

        # ETH/BTC — speculative *alt* bid gauge
        eb = pd.DataFrame({"eth": eth, "btc": btc}).dropna()
        ethbtc = (eb["eth"] / eb["btc"]).dropna()
        ethbtc_now = float(ethbtc.iloc[-1])
        ethbtc_trend = _trend(ethbtc, 90)

        # ── regime label ────────────────────────────────────────────────
        alpha_neg = a90 < 0
        if alpha_neg and ratio_trend == "falling":
            regime = "CRYPTO STARVED"
            note = ("Rolling-bubbles regime: BTC correlated to equities but "
                    "alpha-negative — speculative bid has left crypto for the AI trade.")
        elif (not alpha_neg) and ratio_trend == "rising":
            regime = "CRYPTO LEADING"
            note = "Speculative premium back in crypto: BTC outrunning its equity beta."
        else:
            regime = "TRANSITION"
            note = "Mixed: alpha and BTC/QQQ trend disagree — regime in flux."
        alt_note = ("alts bidding (ETH/BTC rising)" if ethbtc_trend == "rising"
                    else "alts starved (ETH/BTC falling)")

        summary = (f"{regime} — BTC 90d alpha vs QQQ {a90:+.0%} ann "
                   f"(beta {b90:.2f}, corr {r90:.2f}); BTC/QQQ ratio {ratio_trend}; "
                   f"{alt_note}. {note}")

        return {
            "status": "ok",
            "regime": regime,
            "alpha_annual_90d": a90, "beta_90d": b90, "corr_90d": r90,
            "alpha_annual_252d": a252, "beta_252d": b252, "corr_252d": r252,
            "btc_qqq_ratio_trend": ratio_trend,
            "btc_qqq_ratio_vs_1y_avg": ratio_vs_1y,
            "ethbtc": ethbtc_now, "ethbtc_trend": ethbtc_trend,
            "asof": str(df.index[-1].date()),
            "summary": summary, "note": note,
        }
    except Exception as e:
        return {"status": "unavailable", "error": f"{type(e).__name__}: {e}",
                "regime": "n/a", "summary": "BTC alpha regime unavailable (data fetch failed)."}


if __name__ == "__main__":
    import json
    print(json.dumps(compute(), indent=2))
